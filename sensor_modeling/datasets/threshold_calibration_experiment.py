"""The threshold-calibration evaluation: the default reference against the calibrated one.

This runs the protocol of :mod:`.threshold_calibration_protocol`, which has
two parts of different standing.

**The test** is on simulated homes reduced to their event sensors.
:func:`run_home` simulates one home in every arm, runs each arm once through
:class:`~sensor_modeling.online.pipeline.BehaviouralSensingPipeline` with the
default reference, and replays the days that run closed under every condition
the protocol declares. :func:`matched` places the calibrated reference's
operating curve against the point where the default ships, and :func:`score`
computes the protocol's estimands and decides its criteria. Every interval
resamples homes, and the matches are placed again in every resample. The
evidence is simulated, and every result is a statement about the simulator.

**The description** replays the same conditions over the TIHM homes, in
:func:`describe_tihm`. The calibrated reference was built from those homes,
so nothing there is a test.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from ..alerts.alert import AlertKind, AlertPolicy
from ..baseline.adaptive import BaselineConfig, ChangeKind
from ..evaluation import ExperimentRecord, InputArtifact, ModelRecord, ReportedInterval
from ..evaluation.resampling import percentile_interval, resample_indices
from ..external.canonical import to_canonical
from ..external.contract import DatasetAdapter, HouseholdData
from ..external.validation import validate_household
from ..fusion import ONLINE, EvidenceSummary
from ..health.monitor import HealthConfig
from ..observations.observation import Observation
from ..observations.registry import SensorRegistry
from ..online.pipeline import BehaviouralSensingPipeline, PipelineConfig, PipelineStep
from ..online.replay import ReplayedStep, replay_days, reproduces
from ..simulation.faults import DegradationConfig, degrade
from ..simulation.household import HouseholdConfig, simulate
from .silent_home_experiment import median_delay, share_estimate
from .silent_home_protocol import event_sensors
from .threshold_calibration_protocol import (
    ARMS,
    BRACKETED,
    CALIBRATED,
    CHANGE,
    CHANGE_ARMS,
    DEFAULT,
    END_OF_THE_GRID,
    EXPERIMENT,
    NOT_REACHED,
    REFERENCES,
    RESULT_SCHEMA,
    RULE_OFF,
    STABLE,
    TAIL_THRESHOLDS,
    TIHM_EXPERIMENT,
    TIHM_PUBLISHED,
    TIHM_SCHEMA,
    TRACKED_FEATURE,
    ThresholdCalibrationProtocol,
    between_the_freeze_and_the_run,
    condition_name,
    equivalent_threshold,
    nominal,
)
from .tihm_protocol import TihmProtocol

DAY = timedelta(days=1)
WEEK = timedelta(days=7)

#: The direction of the alert the injected changes should raise: the resident
#: wakes earlier, so the hours of sleep go down.
EXPECTED_DIRECTION = "decrease"

#: Why a change verdict raised no alert.
NOT_ATTRIBUTABLE = "not_attributable_enough"
OTHERWISE = "graded_too_low_or_withheld"

#: The match, and the roles of the conditions described in full.
SAME_FALSE_ALERTS = "same_false_alerts"
ROLES = ("default", "declared", SAME_FALSE_ALERTS)


# ----------------------------------------------------------------------------
# One arm of one home
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ConditionRun:
    """What one replay of an arm concluded under one condition.

    Attributes
    ----------
    behavioural
        The moment and the subject of every behavioural alert, in order.
    described
        For each behavioural alert, in the same order: the verdict of the
        baseline that raised it, its direction and the alert's severity.
    verdicts
        Change verdicts, by feature and then by kind.
    raised
        Change verdicts that raised a behavioural alert, by kind.
    withheld
        Change verdicts that raised none, by why not.
    bursts
        Notices that alerts were being withheld because too many had been
        raised.
    """

    behavioural: tuple[tuple[datetime, str], ...]
    described: tuple[tuple[str, str, str], ...]
    verdicts: Mapping[str, Mapping[str, int]]
    raised: Mapping[str, int]
    withheld: Mapping[str, int]
    bursts: int

    def first(
        self,
        feature: str,
        begin: datetime,
        end: datetime,
        direction: str | None = None,
    ) -> int | None:
        """The place of the first alert about *feature* in the window, if any."""
        for place, (at, subject) in enumerate(self.behavioural):
            if subject != feature or not begin <= at < end:
                continue
            if direction is None or self.described[place][1] == direction:
                return place
        return None


@dataclass(frozen=True)
class ArmRecord:
    """Everything one arm of one home contributes.

    Attributes
    ----------
    closes
        The moments a day was closed at.
    days
        The days closed, in order.
    usable
        How many of them the baseline was offered.
    values
        For each feature, its hours on each closed day; not a number where
        the day was not offered to the baseline.
    aware
        For each feature, whether the day's reference was weekday-aware.
    deviations
        For each reference and feature, the absolute deviation of each closed
        day; not a number where the baseline gave no verdict. A day's
        deviation does not depend on the threshold it is read against, so one
        replay of each reference gives it for every multiple. A reference
        that was not replayed at its declared thresholds is absent.
    conditions
        The replay under every condition the protocol declares.
    reproduced
        Whether the replay under the default reference at its declared
        thresholds returned the pipeline's own verdicts and alerts.
    checked
        For a home the protocol checks: for each calibrated condition it is
        checked at, whether the replay returned what the pipeline run with
        that reference itself concluded. Empty for the other homes.
    """

    closes: tuple[datetime, ...]
    days: tuple[date, ...]
    usable: int
    values: Mapping[str, np.ndarray]
    aware: Mapping[str, np.ndarray]
    deviations: Mapping[str, Mapping[str, np.ndarray]]
    conditions: Mapping[str, ConditionRun]
    reproduced: bool
    checked: Mapping[str, bool]


@dataclass(frozen=True)
class HomeRecord:
    """One simulated home: its seed and its arms."""

    seed: int
    arms: Mapping[str, ArmRecord]


def condition_run(
    replayed: Sequence[ReplayedStep],
    confidence: Mapping[datetime, float],
    policy: AlertPolicy,
) -> ConditionRun:
    """Reduce a replay to what the estimands read.

    *confidence* gives, for the moment each day was closed at, the product of
    the day's coverage and its attribution, which is what the alert policy
    asks to be at least ``min_confidence`` before a change may raise an alert.
    A change verdict that raised no alert on a day confident enough was
    graded below the minimum score, or was a repeat inside the cooldown, or
    fell in a burst; the three are not told apart.
    """
    behavioural: list[tuple[datetime, str]] = []
    described: list[tuple[str, str, str]] = []
    verdicts: dict[str, Counter[str]] = {}
    raised: Counter[str] = Counter()
    withheld: Counter[str] = Counter()
    bursts = 0
    for step in replayed:
        by_feature = {change.feature: change for change in step.changes}
        alerted = set()
        for alert in step.alerts:
            if alert.kind is AlertKind.BEHAVIOURAL_CHANGE:
                subject = str(alert.subject)
                alerted.add(subject)
                behavioural.append((alert.at, subject))
                verdict = by_feature.get(subject)
                described.append(
                    (
                        verdict.kind.value if verdict is not None else "",
                        str(verdict.direction) if verdict is not None else "",
                        alert.severity.value,
                    )
                )
            elif alert.kind is AlertKind.DATA_QUALITY and alert.subject == "alert_rate":
                bursts += 1
        for change in step.changes:
            if not change.is_change:
                continue
            verdicts.setdefault(change.feature, Counter())[change.kind.value] += 1
            if change.feature in alerted:
                raised[change.kind.value] += 1
            elif confidence.get(step.at, 1.0) < policy.min_confidence:
                withheld[NOT_ATTRIBUTABLE] += 1
            else:
                withheld[OTHERWISE] += 1
    return ConditionRun(
        behavioural=tuple(behavioural),
        described=tuple(described),
        verdicts={feature: dict(counts) for feature, counts in verdicts.items()},
        raised=dict(raised),
        withheld=dict(withheld),
        bursts=bursts,
    )


def _daily(
    replayed: Sequence[ReplayedStep], features: Sequence[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Each feature's value, deviation and kind of reference on each closed day."""
    closed = [step for step in replayed if step.day_closed is not None]
    values = {feature: np.full(len(closed), np.nan) for feature in features}
    deviations = {feature: np.full(len(closed), np.nan) for feature in features}
    aware = {feature: np.zeros(len(closed), dtype=bool) for feature in features}
    for place, step in enumerate(closed):
        for change in step.changes:
            values[change.feature][place] = change.value
            if change.kind is not ChangeKind.INSUFFICIENT_DATA:
                deviations[change.feature][place] = abs(change.deviation)
                aware[change.feature][place] = change.reference.weekday_aware
    return values, deviations, aware


def confidence_of(steps: Iterable[PipelineStep]) -> dict[datetime, float]:
    """Coverage times attribution, for the moment each day was closed at."""
    return {
        step.at: float(step.day_closed.coverage)
        * float(step.context.ambient_attribution())
        for step in steps
        if step.day_closed is not None
    }


def replay_arm(
    steps: Sequence[PipelineStep],
    config: PipelineConfig,
    protocol: ThresholdCalibrationProtocol,
    conditions: Sequence[str],
    *,
    pipeline_under: Callable[[BaselineConfig], Sequence[PipelineStep]] | None = None,
    checked: Sequence[str] = (),
) -> ArmRecord:
    """Replay one run's days under every condition, and check the replay.

    *steps* is the run with the default reference. *pipeline_under* runs the
    same record through the pipeline with another baseline configuration; it
    is called for each condition in *checked*, and the replay under that
    condition must return what that run concluded.
    """
    policy = AlertPolicy()
    confidence = confidence_of(steps)
    features = [f"{state.value}_hours" for state in config.features]
    default = condition_name(DEFAULT, 1.0)
    runs: dict[str, ConditionRun] = {}
    daily: dict[str, tuple[dict[str, np.ndarray], ...]] = {}
    reproduced = False
    verified: dict[str, bool] = {}
    for condition in conditions:
        settings = protocol.baseline(condition)
        replayed = replay_days(steps, config, baseline_config=settings)
        runs[condition] = condition_run(replayed, confidence, policy)
        if condition == default:
            reproduced = reproduces(steps, replayed)
        for reference in REFERENCES:
            if condition == condition_name(reference, 1.0):
                daily[reference] = _daily(replayed, features)
        if condition in checked:
            if pipeline_under is None:
                raise ValueError("a condition cannot be checked without a pipeline")
            verified[condition] = reproduces(pipeline_under(settings), replayed)
    if DEFAULT not in daily:
        raise ValueError("the default reference must be replayed as declared")
    closed = [step for step in steps if step.day_closed is not None]
    return ArmRecord(
        closes=tuple(step.at for step in closed),
        days=tuple(step.day_closed.day for step in closed),  # type: ignore[union-attr]
        usable=len(
            [
                step
                for step in closed
                if step.day_closed.is_usable(  # type: ignore[union-attr]
                    config.min_day_coverage, config.min_day_observed
                )
            ]
        ),
        values=daily[DEFAULT][0],
        aware=daily[DEFAULT][2],
        deviations={reference: found[1] for reference, found in daily.items()},
        conditions=runs,
        reproduced=reproduced,
        checked=verified,
    )


def _run_pipeline(
    registry: SensorRegistry,
    observations: Sequence[Observation],
    config: PipelineConfig,
    end: datetime,
    baseline: BaselineConfig | None = None,
    health: HealthConfig | None = None,
) -> list[PipelineStep]:
    pipeline = BehaviouralSensingPipeline(
        registry, config=config, baseline_config=baseline, health_config=health
    )
    steps = pipeline.run(observations)
    steps.extend(pipeline.close(end))
    return steps


def run_arm(
    seed: int,
    arm: str,
    protocol: ThresholdCalibrationProtocol,
    conditions: Sequence[str] | None = None,
) -> ArmRecord:
    """Simulate one arm of one home, run it once and replay it.

    *conditions* is every condition of the protocol unless given; a caller
    that gives fewer must keep the default reference at its declared scale.
    """
    sensors = event_sensors()
    result = simulate(
        HouseholdConfig(days=protocol.days, seed=seed, shift=protocol.shift(arm))
    )
    delivered, _ = degrade(result.observations_for(sensors), DegradationConfig())
    registry = result.registry.subset(sensors)
    config = PipelineConfig(
        tz=result.config.tz, step=timedelta(minutes=protocol.step_minutes)
    )
    steps = _run_pipeline(registry, delivered, config, result.end)
    chosen = protocol.conditions if conditions is None else tuple(conditions)
    checked = (
        [c for c in protocol.checked_conditions if c in chosen]
        if seed in protocol.checked_seeds()
        else []
    )
    return replay_arm(
        steps,
        config,
        protocol,
        chosen,
        pipeline_under=lambda settings: _run_pipeline(
            registry, delivered, config, result.end, settings
        ),
        checked=checked,
    )


def run_home(
    seed: int,
    protocol: ThresholdCalibrationProtocol,
    conditions: Sequence[str] | None = None,
) -> HomeRecord:
    """Simulate one home and replay every arm under every condition."""
    return HomeRecord(
        seed=seed,
        arms={arm: run_arm(seed, arm, protocol, conditions) for arm in ARMS},
    )


def _checkpoint(directory: Path, seed: int) -> Path:
    return directory / f"home-{seed}.pickle"


def _run_home(
    arguments: tuple[int, ThresholdCalibrationProtocol, Path | None],
) -> HomeRecord:
    """Run one home, or read it back if this run has already finished it."""
    seed, protocol, directory = arguments
    if directory is None:
        return run_home(seed, protocol)
    path = _checkpoint(directory, seed)
    if path.exists():
        with path.open("rb") as handle:
            kept: HomeRecord = pickle.load(handle)  # noqa: S301 - written below
        return kept
    home = run_home(seed, protocol)
    partial = path.with_suffix(".partial")
    with partial.open("wb") as handle:
        pickle.dump(home, handle)
    partial.replace(path)
    return home


def run_homes(
    protocol: ThresholdCalibrationProtocol,
    *,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
    checkpoint: Path | None = None,
) -> list[HomeRecord]:
    """Run every home of the protocol, in the order of its seeds.

    Each home is a function of its seed and the protocol alone, so the result
    does not depend on *jobs*. With *checkpoint*, each home is written to that
    directory as soon as it is finished and read back from it if the run is
    started again. A directory must hold the homes of one protocol run by one
    version of the code; the caller names it accordingly.
    """
    if checkpoint is not None:
        Path(checkpoint).mkdir(parents=True, exist_ok=True)
    tasks = [(seed, protocol, checkpoint) for seed in protocol.study_seeds()]
    homes: list[HomeRecord] = []
    if jobs <= 1:
        for task in tasks:
            homes.append(_run_home(task))
            if progress is not None:
                progress(len(homes), len(tasks))
        return homes
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for home in pool.map(_run_home, tasks):
            homes.append(home)
            if progress is not None:
                progress(len(homes), len(tasks))
    return homes


# ----------------------------------------------------------------------------
# Statistics over homes
# ----------------------------------------------------------------------------
def mean_estimate(
    values: Sequence[float] | np.ndarray, protocol: ThresholdCalibrationProtocol
) -> dict[str, Any]:
    """A mean over homes, its home-bootstrap interval and its Monte Carlo error."""
    data = np.asarray(values, dtype=float)
    result: dict[str, Any] = {
        "estimate": float(data.mean()) if data.size else None,
        "interval": None,
        "mcse": None,
        "homes": int(data.size),
    }
    if data.size < 2:
        return result
    index = resample_indices(data.size, protocol.resamples, protocol.seed)
    result["interval"] = percentile_interval(
        data[index].mean(axis=1), protocol.confidence
    ).to_dict()
    result["mcse"] = float(data.std(ddof=1) / math.sqrt(data.size))
    return result


def ratio_estimate(
    numerator: Sequence[float] | np.ndarray,
    denominator: Sequence[float] | np.ndarray,
    protocol: ThresholdCalibrationProtocol,
) -> dict[str, Any]:
    """A ratio of sums over homes, with its home-bootstrap interval.

    Each home contributes one numerator and one denominator, and each
    resample recomputes both sums. A resample whose denominator is zero has
    no ratio and is left out; how many remained is reported.
    """
    top = np.asarray(numerator, dtype=float)
    bottom = np.asarray(denominator, dtype=float)
    total = float(bottom.sum())
    result: dict[str, Any] = {
        "estimate": float(top.sum()) / total if total > 0 else None,
        "numerator": float(top.sum()),
        "denominator": total,
        "interval": None,
        "homes": int(top.size),
        "resamples_defined": 0,
    }
    if top.size < 2 or total <= 0:
        return result
    index = resample_indices(top.size, protocol.resamples, protocol.seed)
    tops, bottoms = top[index].sum(axis=1), bottom[index].sum(axis=1)
    defined = bottoms > 0
    result["resamples_defined"] = int(defined.sum())
    if defined.any():
        result["interval"] = percentile_interval(
            tops[defined] / bottoms[defined], protocol.confidence
        ).to_dict()
    return result


def _low(estimate: Mapping[str, Any]) -> float:
    """An interval's lower bound; not a number where it has none."""
    interval = estimate["interval"]
    if not interval or interval["low"] is None:
        return float("nan")
    return float(interval["low"])


def _high(estimate: Mapping[str, Any]) -> float:
    """An interval's upper bound; not a number where it has none."""
    interval = estimate["interval"]
    if not interval or interval["high"] is None:
        return float("nan")
    return float(interval["high"])


# ----------------------------------------------------------------------------
# What each home shows
# ----------------------------------------------------------------------------
def false_alerts(homes: Sequence[HomeRecord], condition: str) -> np.ndarray:
    """Each home's behavioural alerts in its stable record."""
    return np.array(
        [len(home.arms[STABLE].conditions[condition].behavioural) for home in homes],
        dtype=float,
    )


def detected(
    homes: Sequence[HomeRecord],
    arm: str,
    condition: str,
    protocol: ThresholdCalibrationProtocol,
    *,
    record: str | None = None,
    direction: str | None = None,
) -> np.ndarray:
    """For each home, whether an alert counts in *arm*'s window.

    The alerts read are those of *record*, which is *arm* itself unless
    given: with the stable record it is the false detection.
    """
    begin, end = protocol.detection_window(arm)
    read = arm if record is None else record
    return np.array(
        [
            home.arms[read]
            .conditions[condition]
            .first(TRACKED_FEATURE, begin, end, direction)
            is not None
            for home in homes
        ]
    )


def excess(
    homes: Sequence[HomeRecord],
    arm: str,
    condition: str,
    protocol: ThresholdCalibrationProtocol,
    *,
    direction: str | None = None,
) -> np.ndarray:
    """Each home's excess detection: detected, minus a false detection."""
    found = detected(homes, arm, condition, protocol, direction=direction)
    false = detected(
        homes, arm, condition, protocol, record=STABLE, direction=direction
    )
    return np.asarray(found.astype(float) - false.astype(float), dtype=float)


def _delays(
    homes: Sequence[HomeRecord],
    arm: str,
    condition: str,
    protocol: ThresholdCalibrationProtocol,
) -> list[float | None]:
    begin, end = protocol.detection_window(arm)
    origin = protocol.begins(arm)
    delays: list[float | None] = []
    for home in homes:
        run = home.arms[arm].conditions[condition]
        place = run.first(TRACKED_FEATURE, begin, end)
        delays.append(
            None if place is None else (run.behavioural[place][0] - origin) / DAY
        )
    return delays


# ----------------------------------------------------------------------------
# The estimands
# ----------------------------------------------------------------------------
def _against_default(
    homes: Sequence[HomeRecord],
    condition: str,
    protocol: ThresholdCalibrationProtocol,
) -> dict[str, Any]:
    """A condition's false alerts and its detection, each minus the default's."""
    default = protocol.default
    alerts = false_alerts(homes, condition) - false_alerts(homes, default)
    result: dict[str, Any] = {
        "false_alerts": {
            **mean_estimate(alerts, protocol),
            "homes_with_more": int((alerts > 0).sum()),
            "homes_with_fewer": int((alerts < 0).sum()),
        }
    }
    for arm in CHANGE_ARMS:
        net = excess(homes, arm, condition, protocol) - excess(
            homes, arm, default, protocol
        )
        found = detected(homes, arm, condition, protocol).astype(float) - detected(
            homes, arm, default, protocol
        ).astype(float)
        false = detected(homes, arm, condition, protocol, record=STABLE).astype(
            float
        ) - detected(homes, arm, default, protocol, record=STABLE).astype(float)
        result[arm] = {
            "excess_detection": mean_estimate(net, protocol),
            "detected": {
                **mean_estimate(found, protocol),
                "only_under_this_condition": int((found > 0).sum()),
                "only_under_the_default": int((found < 0).sum()),
            },
            "false_detections": mean_estimate(false, protocol),
        }
    return result


def _first_alerts(
    homes: Sequence[HomeRecord],
    arm: str,
    condition: str,
    protocol: ThresholdCalibrationProtocol,
) -> dict[str, dict[str, int]]:
    """The verdict and the direction of each home's first alert that counts."""
    begin, end = protocol.detection_window(arm)
    verdicts: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    for home in homes:
        run = home.arms[arm].conditions[condition]
        place = run.first(TRACKED_FEATURE, begin, end)
        if place is not None:
            verdicts[run.described[place][0]] += 1
            directions[run.described[place][1]] += 1
    return {
        "verdict": dict(sorted(verdicts.items())),
        "direction": dict(sorted(directions.items())),
    }


def _by_week(
    homes: Sequence[HomeRecord],
    arm: str,
    condition: str,
    protocol: ThresholdCalibrationProtocol,
) -> dict[str, Any]:
    """Homes first detected, and homes falsely detected, by week of the window."""
    begin, end = protocol.detection_window(arm)
    weeks = math.ceil((end - begin) / WEEK)
    counts = {arm: [0] * weeks, STABLE: [0] * weeks}
    for home in homes:
        for record, row in counts.items():
            run = home.arms[record].conditions[condition]
            place = run.first(TRACKED_FEATURE, begin, end)
            if place is not None:
                row[(run.behavioural[place][0] - begin) // WEEK] += 1
    return {
        "week_of_the_window": list(range(1, weeks + 1)),
        "first_detected": counts[arm],
        "first_false_detection": counts[STABLE],
    }


def _detection(
    homes: Sequence[HomeRecord],
    arm: str,
    condition: str,
    protocol: ThresholdCalibrationProtocol,
) -> dict[str, Any]:
    """What a condition detects in a change arm, raw and beyond chance."""
    found = detected(homes, arm, condition, protocol)
    false = detected(homes, arm, condition, protocol, record=STABLE)
    return {
        "detected": share_estimate(list(found), protocol),  # type: ignore[arg-type]
        "false_detections": share_estimate(list(false), protocol),  # type: ignore[arg-type]
        "excess_detection": mean_estimate(
            excess(homes, arm, condition, protocol), protocol
        ),
        "delay_days": median_delay(
            _delays(homes, arm, condition, protocol), protocol  # type: ignore[arg-type]
        ),
        "first_alert": _first_alerts(homes, arm, condition, protocol),
        "with_the_expected_direction": {
            "direction": EXPECTED_DIRECTION,
            "detected": share_estimate(
                list(
                    detected(
                        homes, arm, condition, protocol, direction=EXPECTED_DIRECTION
                    )
                ),
                protocol,  # type: ignore[arg-type]
            ),
            "false_detections": share_estimate(
                list(
                    detected(
                        homes,
                        arm,
                        condition,
                        protocol,
                        record=STABLE,
                        direction=EXPECTED_DIRECTION,
                    )
                ),
                protocol,  # type: ignore[arg-type]
            ),
            "excess_detection": mean_estimate(
                excess(homes, arm, condition, protocol, direction=EXPECTED_DIRECTION),
                protocol,
            ),
        },
        "by_week": _by_week(homes, arm, condition, protocol),
    }


def _merge(counters: Iterable[Mapping[str, int]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for counter in counters:
        total.update(counter)
    return dict(sorted(total.items()))


def _verdicts(homes: Sequence[HomeRecord], arm: str, condition: str) -> dict[str, Any]:
    """Change verdicts in an arm, and what became of them."""
    runs = [home.arms[arm].conditions[condition] for home in homes]
    by_kind = _merge(counts for run in runs for counts in run.verdicts.values())
    return {
        "change_verdicts": int(sum(by_kind.values())),
        "by_kind": by_kind,
        "by_feature": {
            feature: int(
                sum(sum(run.verdicts.get(feature, {}).values()) for run in runs)
            )
            for feature in sorted({f for run in runs for f in run.verdicts})
        },
        "raised_an_alert": _merge(run.raised for run in runs),
        "raised_none": _merge(run.withheld for run in runs),
        "notices_of_a_burst": int(sum(run.bursts for run in runs)),
    }


def _day_closed(
    arm: ArmRecord, protocol: ThresholdCalibrationProtocol
) -> dict[datetime, int]:
    """For the moment each day was closed at, that day's place in the record.

    A behavioural alert is raised when a day closes, shortly after local
    midnight or at the record's end, so the moment it was raised at names the
    day it is about. Elapsed time does not: the day the clocks go forward is
    an hour short, and a week counted in hours would move by that hour.
    """
    start = protocol.start()
    return {at: (day - start).days for at, day in zip(arm.closes, arm.days)}


def _stable(
    homes: Sequence[HomeRecord],
    condition: str,
    protocol: ThresholdCalibrationProtocol,
) -> dict[str, Any]:
    """False alerts under a condition, in all and by what raised them.

    An alert's week is the week of the day whose close raised it.
    """
    runs = [home.arms[STABLE].conditions[condition] for home in homes]
    alerts = false_alerts(homes, condition)
    by_week = [0] * math.ceil(protocol.days / 7)
    for home in homes:
        arm = home.arms[STABLE]
        closed = _day_closed(arm, protocol)
        for at, _ in arm.conditions[condition].behavioural:
            by_week[closed[at] // 7] += 1
    return {
        "behavioural_alerts": int(alerts.sum()),
        "per_home": mean_estimate(alerts, protocol),
        "per_person_day": mean_estimate(alerts / protocol.days, protocol),
        "homes_with_one": int((alerts > 0).sum()),
        "most_in_one_home": int(alerts.max()) if alerts.size else 0,
        "by_feature": _merge(
            Counter(subject for _, subject in run.behavioural) for run in runs
        ),
        "by_verdict": _merge(
            Counter(verdict for verdict, _, _ in run.described) for run in runs
        ),
        "by_direction": _merge(
            Counter(direction for _, direction, _ in run.described) for run in runs
        ),
        "by_severity": _merge(
            Counter(severity for _, _, severity in run.described) for run in runs
        ),
        "by_week_of_the_day_closed": by_week,
    }


# ----------------------------------------------------------------------------
# E3: what a threshold is passed by
# ----------------------------------------------------------------------------
def _shares(
    homes: Sequence[HomeRecord],
    feature: str,
    reference: str,
    threshold: float,
    protocol: ThresholdCalibrationProtocol,
    phase: str = "all",
) -> dict[str, Any]:
    """The share of a feature's evaluable days in stable homes past a threshold."""
    numerator, denominator = [], []
    for home in homes:
        arm = home.arms[STABLE]
        deviations = arm.deviations[reference][feature]
        judged = ~np.isnan(deviations)
        if phase == "weekday_aware":
            judged &= arm.aware[feature]
        elif phase == "pooled":
            judged &= ~arm.aware[feature]
        numerator.append(float((deviations[judged] >= threshold).sum()))
        denominator.append(float(judged.sum()))
    estimate = ratio_estimate(numerator, denominator, protocol)
    share, interval = estimate["estimate"], estimate["interval"]
    return {
        **estimate,
        "threshold": threshold,
        "stated": nominal(threshold),
        "equivalent_threshold": (
            None if share is None else equivalent_threshold(share)
        ),
        # A larger share is a lower threshold, so the bounds change places.
        "equivalent_threshold_interval": (
            None
            if interval is None
            else {
                "low": equivalent_threshold(interval["high"]),
                "high": equivalent_threshold(interval["low"]),
                "confidence": interval["confidence"],
            }
        ),
    }


def _calibration(
    homes: Sequence[HomeRecord], protocol: ThresholdCalibrationProtocol
) -> dict[str, Any]:
    """E3, and everything described beside it."""
    features = sorted(homes[0].arms[STABLE].values) if homes else []
    replayed = homes[0].arms[STABLE].deviations if homes else {}
    references = [reference for reference in REFERENCES if reference in replayed]
    result: dict[str, Any] = {"feature": TRACKED_FEATURE, "thresholds": {}}
    for threshold in protocol.calibration_thresholds:
        result["thresholds"][f"{threshold:g}"] = {
            reference: {
                **_shares(homes, TRACKED_FEATURE, reference, threshold, protocol),
                "by_phase": {
                    phase: _shares(
                        homes, TRACKED_FEATURE, reference, threshold, protocol, phase
                    )
                    for phase in ("pooled", "weekday_aware")
                },
                "other_features": {
                    feature: _shares(homes, feature, reference, threshold, protocol)
                    for feature in features
                    if feature != TRACKED_FEATURE
                },
            }
            for reference in references
        }
    result["tail"] = {
        reference: {
            f"{threshold:g}": _shares(
                homes, TRACKED_FEATURE, reference, threshold, protocol
            )
            for threshold in TAIL_THRESHOLDS
        }
        for reference in references
    }
    return result


# ----------------------------------------------------------------------------
# E6: the operating curves
# ----------------------------------------------------------------------------
def _curves(
    homes: Sequence[HomeRecord], protocol: ThresholdCalibrationProtocol
) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for condition in protocol.conditions:
        reference, _, scale = condition.partition("@")
        entry: dict[str, Any] = {
            "condition": condition,
            "false_alerts_per_home": mean_estimate(
                false_alerts(homes, condition), protocol
            ),
        }
        for arm in CHANGE_ARMS:
            found = detected(homes, arm, condition, protocol)
            entry[arm] = {
                "detected": share_estimate(list(found), protocol),  # type: ignore[arg-type]
                "excess_detection": mean_estimate(
                    excess(homes, arm, condition, protocol), protocol
                ),
            }
        curves.setdefault(reference, {})[scale] = entry
    return curves


def matched_or_beaten(
    curves: Mapping[str, Any], arm: str, reference: str, other: str
) -> list[str]:
    """The multiples of *reference* that some multiple of *other* is no worse than.

    A multiple of *other* is no worse when it has no more false alerts per
    home and no less excess detection in *arm*. The comparison is of the
    estimates and carries no uncertainty.
    """

    def point(name: str, scale: str) -> tuple[float, float]:
        entry = curves[name][scale]
        return (
            entry["false_alerts_per_home"]["estimate"],
            entry[arm]["excess_detection"]["estimate"],
        )

    found = []
    for scale in curves[reference]:
        false, net = point(reference, scale)
        if any(
            point(other, rival)[0] <= false and point(other, rival)[1] >= net
            for rival in curves[other]
        ):
            found.append(scale)
    return found


def _reading(curves: Mapping[str, Any]) -> dict[str, Any]:
    if not {DEFAULT, CALIBRATED} <= set(curves):
        return {}
    return {
        arm: {
            "default_multiples_a_calibrated_one_is_no_worse_than": matched_or_beaten(
                curves, arm, DEFAULT, CALIBRATED
            ),
            "calibrated_multiples_a_default_one_is_no_worse_than": matched_or_beaten(
                curves, arm, CALIBRATED, DEFAULT
            ),
        }
        for arm in CHANGE_ARMS
    }


# ----------------------------------------------------------------------------
# E1 and E4: the calibrated reference's curve against the default's point
# ----------------------------------------------------------------------------
_STATUSES = (BRACKETED, NOT_REACHED, END_OF_THE_GRID)


def crossing(
    match: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where a curve last has at least a target, along its columns.

    Each row of *match* is one curve and *target* has one value a row. For
    each row this takes the last column whose value is at least the target
    and returns that column, how far towards the next column the line between
    the two meets the target, and a status: 0 when the match is bracketed, 1
    when no column reaches the target and 2 when the last column of all still
    does. Where the match is not bracketed the column is 0 and the share is
    not a number.
    """
    reaches = match >= target[:, None]
    final = match.shape[1] - 1
    last = final - np.argmax(reaches[:, ::-1], axis=1)
    status = np.where(~reaches.any(axis=1), 1, np.where(last == final, 2, 0))
    bracketed = status == 0
    place = np.where(bracketed, last, 0)
    rows = np.arange(match.shape[0])
    here, there = match[rows, place], match[rows, place + 1]
    gap = np.where(bracketed, here - there, 1.0)
    share = np.where(bracketed, (here - target) / gap, np.nan)
    return place, share, status


def read_at(values: np.ndarray, place: np.ndarray, share: np.ndarray) -> np.ndarray:
    """Each row of *values* on the line from a column towards the next."""
    rows = np.arange(values.shape[0])
    here, there = values[rows, place], values[rows, place + 1]
    return np.asarray(here + share * (there - here), dtype=float)


def _samples(units: int, protocol: ThresholdCalibrationProtocol) -> np.ndarray:
    """How many times each home is in the homes as they are and in every resample.

    The first row is the homes as they are, each once. The others are the
    resamples every other interval of the protocol uses. They are counts, so
    that sums over a resample of whole numbers are exact and two multiples
    with the same total are seen to have it.
    """
    index = resample_indices(units, protocol.resamples, protocol.seed)
    offsets = np.arange(index.shape[0])[:, None] * units
    counts = np.bincount((index + offsets).ravel(), minlength=index.size)
    resampled = counts.reshape(index.shape[0], units).astype(np.int64)
    return np.vstack([np.ones((1, units), dtype=np.int64), resampled])


def matched_estimate(
    values: np.ndarray,
    status: np.ndarray,
    protocol: ThresholdCalibrationProtocol,
    homes: int,
) -> dict[str, Any]:
    """An estimand read at a match, with the interval the protocol fixed.

    The first entry of *values* and of *status* is for the homes as they are
    and the others are for the resamples. A resample in which the match is
    not bracketed has no value and counts against whichever claim a bound
    could support: above every value for the upper bound, below every value
    for the lower one. The bounds are resampled values that leave the same
    share of the resamples outside on each side, so a bound is ``None`` when
    more than that share of the resamples had no match.
    """
    bracketed = status[1:] == 0
    resampled = values[1:]
    tail = (1.0 - protocol.confidence) / 2.0
    low = high = None
    outside = math.floor(round(tail * resampled.size, 9))
    if resampled.size > 2 * outside:
        below = np.sort(np.where(bracketed, resampled, -np.inf))[outside]
        above = np.sort(np.where(bracketed, resampled, np.inf))[-1 - outside]
        low = float(below) if np.isfinite(below) else None
        high = float(above) if np.isfinite(above) else None
    kept = resampled[bracketed]
    return {
        "estimate": float(values[0]) if status[0] == 0 else None,
        "interval": {
            "low": low,
            "high": high,
            "confidence": protocol.confidence,
            "method": "percentile, with the match placed again in every resample",
        },
        "mcse": float(kept.std(ddof=1)) if kept.size > 1 else None,
        "homes": homes,
        "status": _STATUSES[int(status[0])],
        "resamples": {
            "all": int(resampled.size),
            **{
                name: int((status[1:] == code).sum())
                for code, name in enumerate(_STATUSES)
            },
        },
    }


def _whole(values: np.ndarray) -> np.ndarray:
    """Counts and differences of flags, as the whole numbers they are."""
    whole = np.rint(values).astype(np.int64)
    if not np.array_equal(whole, values):
        raise ValueError("a count or a difference of flags was not a whole number")
    return whole


def matched(
    homes: Sequence[HomeRecord], protocol: ThresholdCalibrationProtocol
) -> dict[str, Any]:
    """Place the calibrated reference's curve where it has the default's false alerts.

    Returns E1 and E4 with the match they are read at, and the multiple of
    the grid nearest that match. The comparison that places the match is made
    on sums over homes, which are whole numbers; the means are taken
    afterwards.
    """
    count = len(homes)
    scales = protocol.curve_scales
    names = [condition_name(CALIBRATED, scale) for scale in scales]
    default = protocol.default
    samples = _samples(count, protocol)
    false = samples @ _whole(
        np.column_stack([false_alerts(homes, name) for name in names])
    )
    default_false = samples @ _whole(false_alerts(homes, default))

    # The smallest multiple with no more false alerts than the default is the
    # last such when the multiples are walked down from the largest.
    down = np.broadcast_to(np.array(scales[::-1], dtype=float), false.shape)
    place, share, status = crossing(-false[:, ::-1], -default_false)
    found = _STATUSES[int(status[0])]
    multiple = float(read_at(down, place, share)[0]) if found == BRACKETED else None
    estimands: dict[str, Any] = {}
    for arm in CHANGE_ARMS:
        net = samples @ _whole(
            np.column_stack([excess(homes, arm, name, protocol) for name in names])
        )
        default_net = samples @ _whole(excess(homes, arm, default, protocol))
        there = read_at(net[:, ::-1] / count, place, share)
        estimands[arm] = {
            **matched_estimate(there - default_net / count, status, protocol, count),
            "calibrated_at_the_match": float(there[0]) if found == BRACKETED else None,
            "default": float(default_net[0] / count),
        }
    if found == BRACKETED:
        # The match lies a share of the way from one multiple to the next
        # smaller one. Which is nearer is read off that share, which is a
        # ratio of whole numbers, and not off the multiples themselves.
        taken, other = float(down[0, place[0]]), float(down[0, place[0] + 1])
        near = taken if share[0] < 0.5 else other
    else:
        near = min(scales) if found == END_OF_THE_GRID else max(scales)
    return {
        "default": {
            "condition": default,
            "false_alerts_per_home": float(default_false[0] / count),
        },
        "match": {
            "status": found,
            "multiple": multiple,
            "between": (
                sorted((float(down[0, place[0]]), float(down[0, place[0] + 1])))
                if found == BRACKETED
                else None
            ),
            "nearest_multiple": near,
        },
        "excess_detection": estimands,
    }


# ----------------------------------------------------------------------------
# The criteria
# ----------------------------------------------------------------------------
def _three_way(
    estimate: Mapping[str, Any], value: float, below: str, above: str
) -> str:
    """*below* when the interval lies below *value*, *above* when above it."""
    if _high(estimate) < value:
        return below
    if _low(estimate) > value:
        return above
    return "inconclusive"


def _threshold_verdict(
    entry: Mapping[str, Any], protocol: ThresholdCalibrationProtocol
) -> str:
    interval = entry["equivalent_threshold_interval"]
    if interval is None:
        return "inconclusive"
    tolerance, declared = protocol.calibration_tolerance, entry["threshold"]
    low, high = interval["low"], interval["high"]
    if declared - tolerance <= low and high <= declared + tolerance:
        return "holds"
    if high < declared - tolerance or low > declared + tolerance:
        return "does not hold"
    return "inconclusive"


_READINGS = {
    "success": "better",
    "failure": "worse",
    "uninformative": "uninformative",
}


def criteria(
    results: Mapping[str, Any], protocol: ThresholdCalibrationProtocol
) -> dict[str, Any]:
    """Decide the protocol's criteria from the estimands."""
    found = results["matched"]
    default_found = results["detection"][CHANGE][protocol.default]["excess_detection"]
    e1 = found["excess_detection"][CHANGE]
    if default_found["estimate"] < protocol.informative_recall:
        one = "uninformative"
    elif e1["status"] != BRACKETED:
        one = "not bracketed"
    else:
        one = _three_way(e1, 0.0, "failure", "success")
    calibration = results["calibration"]["thresholds"]
    two = {
        threshold: _threshold_verdict(entry[CALIBRATED], protocol)
        for threshold, entry in calibration.items()
        if CALIBRATED in entry
    }
    return {
        "C1_more_is_detected_at_the_same_false_alerts": {
            "verdict": one,
            "E1": e1,
            "match": found["match"],
            "default_excess_detection": default_found,
            "informative_at": protocol.informative_recall,
        },
        "C2_the_threshold_means_what_it_says": {
            "verdicts": two,
            "tolerance": protocol.calibration_tolerance,
            "E3": {
                threshold: entry[CALIBRATED]
                for threshold, entry in calibration.items()
                if CALIBRATED in entry
            },
        },
        "reading": _READINGS.get(one, "not shown"),
    }


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
#: How an alert is written in the record's raw part: the day whose close
#: raised it, then one letter each for its subject, the verdict behind it and
#: its direction.
SUBJECT_LETTERS = {
    "sleeping_hours": "s",
    "kitchen_activity_hours": "k",
    "bathroom_activity_hours": "b",
    "away_hours": "a",
}
VERDICT_LETTERS = {
    "gradual_drift": "G",
    "persistent_change": "P",
    "abrupt_change": "A",
}
DIRECTION_LETTERS = {"decrease": "-", "increase": "+"}


def encode_alerts(
    arm: ArmRecord, condition: str, protocol: ThresholdCalibrationProtocol
) -> str:
    """Write a replay's behavioural alerts in a few characters each.

    ``57sG-`` is an alert about the hours of sleep, raised when day 57 was
    closed, by a gradual-drift verdict of a decrease. A letter the legend
    does not hold is written ``?``.
    """
    run = arm.conditions[condition]
    place = _day_closed(arm, protocol)
    return " ".join(
        f"{place[at]}"
        f"{SUBJECT_LETTERS.get(subject, '?')}"
        f"{VERDICT_LETTERS.get(verdict, '?')}"
        f"{DIRECTION_LETTERS.get(direction, '?')}"
        for (at, subject), (verdict, direction, _) in zip(
            run.behavioural, run.described
        )
    )


def _flags(values: Iterable[Any]) -> str:
    return "".join("1" if value else "0" for value in values)


def _per_home(
    homes: Sequence[HomeRecord],
    protocol: ThresholdCalibrationProtocol,
    conditions: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Each home under the conditions given, in their order.

    A string of ones and zeros has one character for each of the conditions.
    """
    found = {
        arm: np.array(
            [detected(homes, arm, condition, protocol) for condition in conditions]
        )
        for arm in CHANGE_ARMS
    }
    false = {
        arm: np.array(
            [
                detected(homes, arm, condition, protocol, record=STABLE)
                for condition in conditions
            ]
        )
        for arm in CHANGE_ARMS
    }
    rows: dict[str, dict[str, Any]] = {}
    for place, home in enumerate(homes):
        rows[str(home.seed)] = {
            "usable_days": home.arms[STABLE].usable,
            "checked_with_the_calibrated_pipeline": bool(home.arms[STABLE].checked),
            "false_alerts": [
                len(home.arms[STABLE].conditions[condition].behavioural)
                for condition in conditions
            ],
            "detected": {arm: _flags(found[arm][:, place]) for arm in CHANGE_ARMS},
            "false_detection": {
                arm: _flags(false[arm][:, place]) for arm in CHANGE_ARMS
            },
        }
    return rows


def _raw(
    homes: Sequence[HomeRecord],
    protocol: ThresholdCalibrationProtocol,
    conditions: Sequence[str],
) -> dict[str, Any]:
    """Every alert of the conditions given, and every home under the others.

    No estimand reads this. It is kept so that a question asked after the run
    does not need the homes to be run again.
    """
    alerts = {
        str(home.seed): {
            arm: {
                condition: encode_alerts(record, condition, protocol)
                for condition in conditions
            }
            for arm, record in home.arms.items()
        }
        for home in homes
    }
    described = {}
    for condition in protocol.conditions:
        described[condition] = {
            "false_alerts": " ".join(
                str(int(count)) for count in false_alerts(homes, condition)
            ),
            **{
                f"{arm}/detected": _flags(detected(homes, arm, condition, protocol))
                for arm in CHANGE_ARMS
            },
            **{
                f"{arm}/false_detection": _flags(
                    detected(homes, arm, condition, protocol, record=STABLE)
                )
                for arm in CHANGE_ARMS
            },
        }
    return {
        "how_an_alert_is_written": "the day whose close raised it, then a "
        "letter for its subject, a letter for the verdict behind it and a sign "
        "for its direction; alerts are separated by spaces",
        "subjects": {letter: name for name, letter in SUBJECT_LETTERS.items()},
        "verdicts": {letter: name for name, letter in VERDICT_LETTERS.items()},
        "directions": {letter: name for name, letter in DIRECTION_LETTERS.items()},
        "conditions_with_every_alert": list(conditions),
        "alerts": alerts,
        "seeds": [home.seed for home in homes],
        "how_a_condition_is_written": "false alerts are one count a home, and "
        "the others one character a home, in the order of the seeds",
        "conditions": described,
    }


def _features(
    homes: Sequence[HomeRecord],
) -> dict[str, dict[str, Any]]:
    """What the features' daily values look like in the stable homes.

    No estimand reads this. It says how far the days are from what the
    calibrated score assumes.
    """
    described: dict[str, dict[str, Any]] = {}
    features = sorted(homes[0].arms[STABLE].values) if homes else []
    for feature in features:
        spreads, zeros, days, weekend = [], 0, 0, []
        for home in homes:
            arm = home.arms[STABLE]
            values = arm.values[feature]
            used = ~np.isnan(values)
            if used.sum() < 2:
                continue
            kept = values[used]
            spreads.append(float(kept.std(ddof=1)))
            zeros += int((kept < 0.01).sum())
            days += int(kept.size)
            late = np.array([day.weekday() >= 5 for day in arm.days])[used]
            if late.any() and (~late).any():
                weekend.append(float(kept[late].mean() - kept[~late].mean()))
        described[feature] = {
            "homes": len(spreads),
            "days": days,
            "median_standard_deviation_hours": (
                float(np.median(spreads)) if spreads else None
            ),
            "share_of_days_under_a_hundredth_of_an_hour": (
                zeros / days if days else None
            ),
            "median_weekend_minus_weekday_hours": (
                float(np.median(weekend)) if weekend else None
            ),
        }
    return described


def check(
    homes: Sequence[HomeRecord], protocol: ThresholdCalibrationProtocol
) -> dict[str, Any]:
    """Whether every replay returned what the pipeline itself concluded."""
    failed = sorted(
        f"{home.seed}/{arm}"
        for home in homes
        for arm, record in home.arms.items()
        if not record.reproduced
    )
    wanted = set(protocol.checked_seeds())
    verified = {
        f"{home.seed}/{arm}/{condition}": ok
        for home in homes
        if home.seed in wanted
        for arm, record in home.arms.items()
        for condition, ok in record.checked.items()
    }
    present = wanted & {home.seed for home in homes}
    return {
        "default_replays": len(homes) * len(ARMS),
        "default_replays_that_differ": failed,
        "calibrated_runs_checked": len(verified),
        "calibrated_runs_expected": len(present)
        * len(ARMS)
        * len(protocol.checked_conditions),
        "calibrated_runs_that_differ": sorted(
            name for name, ok in verified.items() if not ok
        ),
        "homes_checked": sorted(present),
    }


def in_order(
    homes: Sequence[HomeRecord], protocol: ThresholdCalibrationProtocol
) -> list[HomeRecord]:
    """The protocol's homes, each once, in the order of its seeds."""
    by_seed = {home.seed: home for home in homes}
    if set(by_seed) != set(protocol.study_seeds()) or len(by_seed) != len(homes):
        raise ValueError("the homes are not the protocol's homes, each once")
    return [by_seed[seed] for seed in protocol.study_seeds()]


def score(
    homes: Sequence[HomeRecord], protocol: ThresholdCalibrationProtocol
) -> dict[str, Any]:
    """Compute every estimand on the protocol's homes and decide its criteria.

    Refuses when a replay did not return what the pipeline concluded: every
    condition is a replay, so nothing would mean what it is taken to mean.
    """
    homes = in_order(homes, protocol)
    checked = check(homes, protocol)
    if (
        checked["default_replays_that_differ"]
        or checked["calibrated_runs_that_differ"]
        or checked["calibrated_runs_checked"] != checked["calibrated_runs_expected"]
    ):
        raise ValueError(
            "a replay did not return what the pipeline concluded, so nothing is "
            f"reported: {checked}"
        )
    found = matched(homes, protocol)
    shown = dict(zip(ROLES, protocol.shown(found["match"]["nearest_multiple"])))
    conditions = list(dict.fromkeys(shown.values()))
    results: dict[str, Any] = {
        "result_schema": RESULT_SCHEMA,
        "homes": len(homes),
        "days": protocol.days,
        "person_days": len(homes) * protocol.days,
        "check": checked,
        "matched": found,
        "shown": shown,
        "reported": conditions,
        "stable": {
            condition: _stable(homes, condition, protocol) for condition in conditions
        },
        "detection": {
            arm: {
                condition: _detection(homes, arm, condition, protocol)
                for condition in conditions
            }
            for arm in CHANGE_ARMS
        },
        "against_the_default": {
            condition: _against_default(homes, condition, protocol)
            for condition in conditions
            if condition != protocol.default
        },
        "calibration": _calibration(homes, protocol),
        "verdicts": {
            arm: {
                condition: _verdicts(homes, arm, condition) for condition in conditions
            }
            for arm in ARMS
        },
        "curves": _curves(homes, protocol),
        "features": _features(homes),
        "raw": _raw(homes, protocol, conditions),
    }
    results["curves_reading"] = _reading(results["curves"])
    results["criteria"] = criteria(results, protocol)
    return results


# ----------------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------------
def _interval(
    label: str, estimate: Mapping[str, Any], protocol: ThresholdCalibrationProtocol
) -> ReportedInterval | None:
    interval = estimate.get("interval")
    if not interval or estimate.get("estimate") is None:
        return None
    if interval["low"] is None or interval["high"] is None:
        return None
    return ReportedInterval(
        label=label,
        estimate=estimate["estimate"],
        low=float(interval["low"]),
        high=float(interval["high"]),
        confidence=protocol.confidence,
        method=str(interval["method"]),
        unit="seed",
        n=int(estimate["homes"]),
    )


def _named(
    results: Mapping[str, Any], protocol: ThresholdCalibrationProtocol
) -> dict[str, Mapping[str, Any]]:
    """The differences the estimands name."""
    found = results["matched"]["excess_detection"]
    declared = results["against_the_default"][protocol.declared]
    named = {
        "E1": found[CHANGE],
        "E2_false_alerts": declared["false_alerts"],
        "E2_excess_detection": declared[CHANGE]["excess_detection"],
    }
    for arm in CHANGE_ARMS:
        if arm != CHANGE:
            named[f"E4_{arm}"] = found[arm]
    return named


def _intervals(
    results: Mapping[str, Any], protocol: ThresholdCalibrationProtocol
) -> list[ReportedInterval]:
    labels = {
        "E1": "E1: excess detection, calibrated at the same false alerts "
        "minus default",
        "E2_false_alerts": "E2: false alerts per home, calibrated as declared "
        "minus default",
        "E2_excess_detection": "E2: excess detection, calibrated as declared "
        "minus default",
    }
    wanted = [
        (
            labels.get(
                name,
                f"E4: excess detection in {name[3:]}, calibrated at the same "
                "false alerts minus default",
            ),
            estimate,
        )
        for name, estimate in _named(results, protocol).items()
    ]
    wanted += [
        (f"E3: share of days at or past {threshold}, {reference} reference", entry)
        for threshold, references in results["calibration"]["thresholds"].items()
        for reference, entry in references.items()
    ]
    found = [_interval(label, estimate, protocol) for label, estimate in wanted]
    return [interval for interval in found if interval is not None]


def _mcse(
    results: Mapping[str, Any], protocol: ThresholdCalibrationProtocol
) -> dict[str, float]:
    return {
        name: float(estimate["mcse"])
        for name, estimate in _named(results, protocol).items()
        if estimate.get("mcse") is not None
    }


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    """A run of the evaluation: the record, and where it was written."""

    record: ExperimentRecord
    path: Path | None


def record_of(
    homes: Sequence[HomeRecord],
    protocol: ThresholdCalibrationProtocol,
    *,
    protocol_sha256: str,
    inputs: Sequence[InputArtifact] = (),
    code_changes: Mapping[str, Sequence[str]] | None = None,
    code_note: str = "",
    clean: bool = True,
) -> ExperimentRecord:
    """Score the homes and build the record of the protocol's test.

    *clean* says whether the run was made from a commit with nothing
    modified; the record says so when it was not.
    """
    results = score(homes, protocol)
    between = between_the_freeze_and_the_run()
    changed = {
        name: list((code_changes or {}).get(name, []))
        for name in ("sources", "distributions", "defaults")
    }
    return ExperimentRecord(
        experiment=EXPERIMENT,
        configuration={
            **protocol.to_dict(),
            "protocol_sha256": protocol.sha256(),
            "protocol_file_sha256": protocol_sha256,
            "result_schema": RESULT_SCHEMA,
            "between_the_freeze_and_the_run": list(between),
            "code_changed_since_the_freeze": changed,
            "why_the_code_changed": code_note,
        },
        inference=ONLINE,
        evidence=EvidenceSummary.online(
            moment
            for home in homes
            for arm in home.arms.values()
            for moment in arm.closes
        ),
        seeds=[protocol.seed_root, protocol.seed, *protocol.study_seeds()],
        results=results,
        sensor_subset=list(event_sensors()),
        data_source="simulator",
        metric_definitions={
            name: text
            for name, text in protocol.to_dict()["definitions"].items()
            if name != "order"
        },
        notes=[
            "Pre-specified: the protocol was frozen before any simulated home "
            "was run with the calibrated reference.",
            *(
                []
                if clean
                else [
                    "The run was made from a modified working tree, so the "
                    "record cannot be traced to one commit."
                ]
            ),
            f"{len(homes)} paired simulated homes. The calibrated reference "
            "is compared where its operating curve has the default's false "
            "alerts, and that place is found again in every resample of "
            "homes.",
            "Each arm of each home was "
            "run through the pipeline once, with the default reference, and "
            "every condition is a replay of the days that run closed. The "
            "replay under the default reference returned the pipeline's own "
            "verdicts and alerts in every home and arm, and in the homes the "
            "protocol names it returned those of the pipeline run with the "
            "calibrated reference itself.",
            *(text[0].upper() + text[1:] + "." for text in between),
            (
                "The source files, the libraries' versions and the default "
                "settings the protocol recorded at its freeze are those the "
                "run used."
                if not any(changed.values())
                else "Code the protocol recorded at its freeze had changed by "
                f"the run: {changed}. {code_note}"
            ),
            "A home with a change shares its seed with the same home without "
            "it and not its days, so a detection is set against the same "
            "window of the stable record as a rate, home by home, and not day "
            "by day.",
            "Nothing was fitted. The pipeline's emissions and alert policy are "
            "the declared defaults, and the multiples of the thresholds are "
            "the protocol's grid.",
            "A simulated resident's days come from distributions this project "
            "wrote down. The result says nothing about how a threshold behaves "
            "on a real home.",
        ],
        inputs=list(inputs),
        preprocessing={
            "sensors": list(event_sensors()),
            "delivery": "simulation.faults.degrade, with no loss, lateness, "
            "duplication or fault",
            "step_minutes": protocol.step_minutes,
        },
        models=[
            ModelRecord(
                f"replay_{condition}",
                {
                    "calibrated": protocol.baseline(condition).calibrated,
                    "deviation_threshold": protocol.baseline(
                        condition
                    ).deviation_threshold,
                    "trend_threshold": protocol.baseline(condition).trend_threshold,
                },
                "sensor_modeling.online.replay_days",
            )
            for condition in protocol.conditions
        ],
        household_metrics={
            "simulated_homes": _per_home(homes, protocol, results["reported"])
        },
        intervals=_intervals(results, protocol),
        mcse=_mcse(results, protocol),
    )


def run_threshold_calibration(
    protocol: ThresholdCalibrationProtocol,
    *,
    protocol_sha256: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
    checkpoint: Path | None = None,
    code_changes: Mapping[str, Sequence[str]] | None = None,
    code_note: str = "",
    clean: bool = True,
) -> ThresholdCalibrationResult:
    """Run every home of the protocol and write the record of its test."""
    homes = run_homes(protocol, jobs=jobs, progress=progress, checkpoint=checkpoint)
    record = record_of(
        homes,
        protocol,
        protocol_sha256=protocol_sha256,
        inputs=inputs,
        code_changes=code_changes,
        code_note=code_note,
        clean=clean,
    )
    path = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = record.write(Path(output_dir) / f"{EXPERIMENT}.json")
    return ThresholdCalibrationResult(record, path)


# ----------------------------------------------------------------------------
# The description on TIHM
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TihmRun:
    """One TIHM home under one setting of the silent-home rule.

    Attributes
    ----------
    household
        The dataset's identifier.
    monitored, usable, evaluable
        Days the pipeline closed, days the baseline was offered, and days on
        which some feature received a verdict.
    record
        The replays of every described condition over the run.
    """

    household: str
    monitored: int
    usable: int
    evaluable: int
    record: ArmRecord


def run_tihm_home(
    data: HouseholdData,
    tihm: TihmProtocol,
    protocol: ThresholdCalibrationProtocol,
    hours: float | None,
    check: bool = False,
) -> TihmRun:
    """Run one TIHM home with the default reference and replay every condition.

    With *check*, the home is also run through the pipeline with the
    calibrated reference at the declared thresholds, and the record says
    whether the replay under that condition returned what that run concluded.
    """
    canonical = to_canonical(data, tihm.mapping, source="tihm")
    recording = canonical.recording
    if data.timezone is None:  # pragma: no cover - to_canonical already refused it
        raise ValueError(f"household {data.household!r} has no timezone")
    config = PipelineConfig(
        tz=ZoneInfo(data.timezone), step=timedelta(minutes=tihm.step_minutes)
    )
    health = HealthConfig(
        home_silence_horizon=None if hours is None else timedelta(hours=hours)
    )
    end = recording.observations[-1].timestamp
    steps = _run_pipeline(
        recording.registry, recording.observations, config, end, health=health
    )
    record = replay_arm(
        steps,
        config,
        protocol,
        protocol.conditions,
        pipeline_under=lambda settings: _run_pipeline(
            recording.registry, recording.observations, config, end, settings, health
        ),
        checked=[protocol.declared] if check else [],
    )
    evaluable = np.zeros(len(record.days), dtype=bool)
    for deviations in record.deviations[DEFAULT].values():
        evaluable |= ~np.isnan(deviations)
    return TihmRun(
        household=data.household,
        monitored=len(record.days),
        usable=record.usable,
        evaluable=int(evaluable.sum()),
        record=record,
    )


def _run_tihm(
    arguments: tuple[
        HouseholdData, TihmProtocol, ThresholdCalibrationProtocol, float | None, bool
    ],
) -> TihmRun:
    return run_tihm_home(*arguments)


def tihm_checked(
    households: Iterable[str], protocol: ThresholdCalibrationProtocol
) -> list[str]:
    """The homes the calibrated replay is checked on: the first by identifier."""
    return sorted(households)[: protocol.tihm_checked_homes]


def rule_settings(protocol: ThresholdCalibrationProtocol) -> dict[str, float | None]:
    """The settings of the silent-home rule the description is made under."""
    return {
        RULE_OFF: None,
        f"h{protocol.tihm_horizon_hours:g}": protocol.tihm_horizon_hours,
    }


def run_tihm_homes(
    adapter: DatasetAdapter,
    protocol: ThresholdCalibrationProtocol,
    tihm: TihmProtocol,
    *,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[TihmRun]]:
    """Run every TIHM home that passes the contract, under both settings of the rule."""
    loaded = []
    for name in adapter.households():
        data = adapter.load(name)
        if validate_household(data, tihm.mapping).ok:
            loaded.append(data)
    if not loaded:
        raise ValueError("no household passed the contract, so there is nothing to run")
    settings = rule_settings(protocol)
    checked = set(tihm_checked((data.household for data in loaded), protocol))
    tasks = [
        (data, tihm, protocol, hours, rule == RULE_OFF and data.household in checked)
        for rule, hours in settings.items()
        for data in loaded
    ]
    done: list[TihmRun] = []
    if jobs <= 1:
        for task in tasks:
            done.append(_run_tihm(task))
            if progress is not None:
                progress(len(done), len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for run in pool.map(_run_tihm, tasks):
                done.append(run)
                if progress is not None:
                    progress(len(done), len(tasks))
    size = len(loaded)
    return {
        rule: done[place * size : (place + 1) * size]
        for place, rule in enumerate(settings)
    }


def _tihm_share(
    runs: Sequence[TihmRun],
    feature: str,
    reference: str,
    threshold: float,
) -> dict[str, Any]:
    """The share of a feature's evaluable days past a threshold, over all homes.

    No interval is given. The homes are the ones the option was built from,
    and the share describes them.
    """
    deviating = judged = 0
    for run in runs:
        deviations = run.record.deviations[reference][feature]
        kept = deviations[~np.isnan(deviations)]
        deviating += int((kept >= threshold).sum())
        judged += int(kept.size)
    share = deviating / judged if judged else None
    return {
        "deviating": deviating,
        "evaluable_feature_days": judged,
        "share": share,
        "threshold": threshold,
        "stated": nominal(threshold),
        "equivalent_threshold": (
            None if share is None else equivalent_threshold(share)
        ),
    }


def _tihm_condition(runs: Sequence[TihmRun], condition: str) -> dict[str, Any]:
    replays = [run.record.conditions[condition] for run in runs]
    alerts = [len(replay.behavioural) for replay in replays]
    by_kind = _merge(
        counts for replay in replays for counts in replay.verdicts.values()
    )
    return {
        "behavioural_alerts": {
            "all": int(sum(alerts)),
            "homes_with_one": int(sum(count > 0 for count in alerts)),
            "most_in_one_home": int(max(alerts)) if alerts else 0,
            "by_feature": _merge(
                Counter(subject for _, subject in replay.behavioural)
                for replay in replays
            ),
            "by_verdict": _merge(
                Counter(verdict for verdict, _, _ in replay.described)
                for replay in replays
            ),
        },
        "change_verdicts": {
            "all": int(sum(by_kind.values())),
            "by_kind": by_kind,
            "raised_an_alert": _merge(replay.raised for replay in replays),
            "raised_none": _merge(replay.withheld for replay in replays),
        },
    }


def describe(
    runs: Mapping[str, Sequence[TihmRun]],
    protocol: ThresholdCalibrationProtocol,
    published_homes: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe what each reference does on the TIHM homes.

    Refuses unless every replay under the default reference returned its
    run's verdicts and alerts; the replay under the calibrated reference as
    declared returned, in the homes the protocol names, what the pipeline run
    with that reference concluded; the totals are those of the published
    records under both settings of the rule; and, with *published_homes*,
    the homes are that record's homes, each once, and every one of them has
    with the rule off the monitored days and the behavioural alerts the
    published alert-burden record gives it.
    """
    default = protocol.default
    names = {rule: [run.household for run in homes] for rule, homes in runs.items()}
    if len({tuple(found) for found in names.values()}) != 1:
        raise ValueError("the two settings of the rule were not run on the same homes")
    households = names[RULE_OFF]
    if len(set(households)) != len(households):
        raise ValueError("a home was run more than once")
    checked: dict[str, Any] = {}
    wanted = tihm_checked(households, protocol)
    verified = {
        run.household: run.record.checked.get(protocol.declared)
        for run in runs[RULE_OFF]
        if run.household in wanted
    }
    checked["calibrated_runs"] = {
        "condition": protocol.declared,
        "homes": wanted,
        "that_differ": sorted(name for name, ok in verified.items() if not ok),
    }
    if not wanted or any(ok is not True for ok in verified.values()):
        raise ValueError(
            "the replay under the calibrated reference did not return what the "
            f"pipeline concluded, so nothing is reported: {verified}"
        )
    for rule, homes in runs.items():
        differ = sorted(run.household for run in homes if not run.record.reproduced)
        totals = {
            "monitored_days": int(sum(run.monitored for run in homes)),
            "behavioural_alerts": int(
                sum(len(run.record.conditions[default].behavioural) for run in homes)
            ),
        }
        checked[rule] = {
            **totals,
            "published": dict(TIHM_PUBLISHED[rule]),
            "replays_that_differ": differ,
        }
        if differ or totals != TIHM_PUBLISHED[rule]:
            raise ValueError(
                f"with the rule {rule} the run is not the published one, so "
                f"nothing is reported: {checked[rule]}"
            )
    if published_homes is not None:
        if set(households) != set(published_homes):
            raise ValueError(
                "the homes run are not the published record's homes, so nothing "
                "is reported: "
                f"{sorted(set(households) ^ set(published_homes))}"
            )
        mismatched = sorted(
            run.household
            for run in runs[RULE_OFF]
            if (
                run.monitored,
                len(run.record.conditions[default].behavioural),
            )
            != (
                published_homes.get(run.household, {}).get("monitored_days"),
                published_homes.get(run.household, {}).get("behavioural_alerts"),
            )
        )
        checked["homes_compared_one_by_one"] = len(runs[RULE_OFF])
        checked["homes_that_differ"] = mismatched
        if mismatched:
            raise ValueError(
                "these homes do not have the published record's days and "
                f"alerts, so nothing is reported: {mismatched}"
            )
    features = sorted(runs[RULE_OFF][0].record.values) if runs[RULE_OFF] else []
    thresholds = sorted({*TAIL_THRESHOLDS, *protocol.calibration_thresholds})
    return {
        "result_schema": TIHM_SCHEMA,
        "homes": len(runs[RULE_OFF]),
        "check": checked,
        "rules": {
            rule: {
                "monitored_days": int(sum(run.monitored for run in homes)),
                "usable_days": int(sum(run.usable for run in homes)),
                "evaluable_days": int(sum(run.evaluable for run in homes)),
                "conditions": {
                    condition: _tihm_condition(homes, condition)
                    for condition in protocol.conditions
                },
                "behavioural_alerts_per_home": {
                    condition: {
                        run.household: len(run.record.conditions[condition].behavioural)
                        for run in homes
                    }
                    for condition in (protocol.default, protocol.declared)
                },
                "tail": {
                    reference: {
                        feature: {
                            f"{threshold:g}": _tihm_share(
                                homes, feature, reference, threshold
                            )
                            for threshold in thresholds
                        }
                        for feature in features
                    }
                    for reference in REFERENCES
                },
            }
            for rule, homes in runs.items()
        },
    }


def describe_tihm(
    adapter: DatasetAdapter,
    protocol: ThresholdCalibrationProtocol,
    tihm: TihmProtocol,
    *,
    protocol_sha256: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
    published_homes: Mapping[str, Mapping[str, Any]] | None = None,
    code_changes: Mapping[str, Sequence[str]] | None = None,
    code_note: str = "",
) -> ThresholdCalibrationResult:
    """Replay every condition over the TIHM homes and write the record."""
    if tihm.step_minutes != protocol.tihm_step_minutes:
        raise ValueError("the two protocols do not agree on the step on TIHM")
    runs = run_tihm_homes(adapter, protocol, tihm, jobs=jobs, progress=progress)
    results = describe(runs, protocol, published_homes)
    changed = {
        name: list((code_changes or {}).get(name, []))
        for name in ("sources", "distributions", "defaults")
    }
    record = ExperimentRecord(
        experiment=TIHM_EXPERIMENT,
        configuration={
            **protocol.to_dict(),
            "protocol_sha256": protocol.sha256(),
            "protocol_file_sha256": protocol_sha256,
            "result_schema": TIHM_SCHEMA,
            "alert_burden_protocol_sha256": tihm.sha256(),
            "code_changed_since_the_freeze": changed,
            "why_the_code_changed": code_note,
        },
        inference=ONLINE,
        evidence=EvidenceSummary.online(
            moment
            for homes in runs.values()
            for run in homes
            for moment in run.record.closes
        ),
        seeds=[],
        results=results,
        data_source="tihm",
        metric_definitions={
            name: text
            for name, text in protocol.to_dict()["definitions"].items()
            if name
            in (
                "behavioural_alert",
                "evaluable_feature_day",
                "deviating",
                "stated",
                "equivalent_threshold",
            )
        },
        notes=[
            "A description, not a test. The calibrated reference was built "
            "from what these homes showed, so nothing here is evidence that "
            "it helps.",
            "Each home was run through the pipeline with the default "
            "reference under each setting of the silent-home rule, and "
            "every condition is a replay of the days those runs closed. The "
            "replay under the default reference returned each run's own "
            "verdicts and alerts, the runs are the published ones, and in the "
            "homes the protocol names the replay under the calibrated "
            "reference returned what the pipeline run with it concluded.",
            "Fewer alerts on these homes are not better alerts. No relation "
            "to the dataset's labels is reported.",
            "A share of deviating days is given with no interval: it "
            "describes these homes.",
            (
                "The source files, the libraries' versions and the default "
                "settings the protocol recorded at its freeze are those the "
                "run used."
                if not any(changed.values())
                else "Code the protocol recorded at its freeze had changed by "
                f"the run: {changed}. {code_note}"
            ),
        ],
        inputs=list(inputs),
        preprocessing={
            "contract": "external-dataset contract, strict conversion",
            "mapping_sha256": tihm.mapping.sha256(),
            "step_minutes": tihm.step_minutes,
        },
        models=[
            ModelRecord(
                f"replay_{condition}",
                {
                    "calibrated": protocol.baseline(condition).calibrated,
                    "deviation_threshold": protocol.baseline(
                        condition
                    ).deviation_threshold,
                    "trend_threshold": protocol.baseline(condition).trend_threshold,
                },
                "sensor_modeling.online.replay_days",
            )
            for condition in protocol.conditions
        ],
    )
    path = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = record.write(Path(output_dir) / f"{TIHM_EXPERIMENT}.json")
    return ThresholdCalibrationResult(record, path)
