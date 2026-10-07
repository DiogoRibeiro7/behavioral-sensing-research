"""The silent-home evaluation: what the opt-in rule does to a home that stops reporting.

This runs the protocol of :mod:`.silent_home_protocol`, which has two parts of
different standing.

**The test** is on simulated homes reduced to their event sensors.
:func:`run_home` simulates one home, injects the outages and the change the
protocol declares, and runs every arm through
:class:`~sensor_modeling.online.pipeline.BehaviouralSensingPipeline` with the
rule off and on. :func:`score` computes the protocol's estimands from those
runs and decides its criteria. Every interval resamples homes. The evidence is
simulated, and every result is a statement about the simulator.

**The description** runs the same rule on the TIHM homes, in
:func:`describe_tihm`. The rule was designed from those homes, so nothing
there is a test.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from statistics import NormalDist
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from ..alerts.alert import HOME_SILENCE, AlertKind, AlertPolicy
from ..baseline.adaptive import BaselineConfig
from ..evaluation import ExperimentRecord, InputArtifact, ModelRecord, ReportedInterval
from ..evaluation.resampling import percentile_interval, resample_indices
from ..external.contract import DatasetAdapter, HouseholdData
from ..external.tihm import TIMEZONE as TIHM_TIMEZONE
from ..external.validation import validate_household
from ..fusion import ONLINE, EvidenceSummary
from ..health.fleet import CommonSilence, FleetConfig, common_silences, replay_fleet
from ..health.monitor import HealthConfig
from ..observations.observation import Observation
from ..observations.registry import SensorRegistry
from ..online.pipeline import BehaviouralSensingPipeline, PipelineConfig
from ..simulation.faults import DegradationConfig, degrade, dropout
from ..simulation.household import BehaviourShift, HouseholdConfig, simulate
from .silent_home_protocol import (
    CHANGE,
    CHANGE_AFTER_OUTAGE,
    COMMON,
    EXPERIMENT,
    FLEET_ARMS,
    IN_TIME,
    LATE,
    OFF,
    OUTAGE,
    RESULT_SCHEMA,
    SHORT_OUTAGE,
    STABLE,
    TIHM_EXPERIMENT,
    TIHM_PUBLISHED,
    TIHM_SCHEMA,
    TRACKED_FEATURE,
    SilentHomeProtocol,
    condition_name,
    event_sensors,
)
from .tihm_experiment import HouseholdRun, run_household
from .tihm_protocol import TihmProtocol

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
TIHM_ZONE = ZoneInfo(TIHM_TIMEZONE)
MICROSECOND = timedelta(microseconds=1)
DAY = timedelta(days=1)

#: The arm without the change that each change arm's false detections are
#: counted in.
WITHOUT_THE_CHANGE = {CHANGE: STABLE, CHANGE_AFTER_OUTAGE: OUTAGE}

#: What was run and read between the freezing of the protocol and its run.
#: None of it touched a home of the protocol. It is recorded because one of
#: these homes showed, before the run, how a criterion might come out.
BETWEEN_THE_FREEZE_AND_THE_RUN = (
    "after the protocol was frozen, simulated homes that are not the "
    "protocol's were run with the rule on, to test the scoring code and to "
    "review it: seeds 563265, 679832 and 785210, drawn from root 7, and seeds "
    "126987, 157858 and 244682, drawn from root 2, in records of 26 to 44 "
    "days; and seed 807611, drawn from root 424242, in a record of the "
    "declared shape",
    "the three homes of root 7, whose outage lasted 36 hours, raised no "
    "behavioural alert in an outage arm, so the alerts of the three homes of "
    "root 2 were read, to write tests that do not pass on a cohort without "
    "alerts. Their records are 34 days long with a step of 30 minutes. In two "
    "of the three the outage raised behavioural alerts with the rule off and "
    "none with it on, and in none did the rule change an alert of the stable "
    "home",
    "the alerts of seed 807611 were read during the review of the code. In "
    "that one home the change alone was not detected, with the rule off or "
    "on. With the rule off the change after the outage was detected, by "
    "gradual-drift alerts, and the same home's outage without the change "
    "raised one such alert in the detection window too. With the rule on the "
    "change after the outage was not detected. So it was known before the run "
    "that the detections C5 compares may be alerts the outage itself raises",
    "nothing in the protocol, its estimands, its criteria or their scoring "
    "was changed after that. What was added to the scoring code before the "
    "run is descriptive: the baseline's verdict, its direction and the "
    "severity kept beside each behavioural alert; each home's own values; the "
    "count of all of a home's silence alerts beside the count of those inside "
    "its outage; and the alerts that are neither behavioural nor about "
    "silence, by subject",
)


# ----------------------------------------------------------------------------
# One home through the pipeline
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineRun:
    """What one run of the pipeline on one home delivered.

    Attributes
    ----------
    behavioural
        The moment and the subject of every behavioural alert, in order.
    silence
        The moment of every alert about the whole home's silence.
    other
        The moment, kind and subject of every other alert: about the
        apparatus, or about a burst of alerts.
    closes
        The moments a day was closed at.
    usable
        The days the baseline was offered.
    refused
        The days the baseline refused because part of them fell in a silence
        of the whole home.
    described
        For each behavioural alert, in the same order: the verdict of the
        baseline that raised it, its direction and the alert's severity. No
        estimand reads them. They are kept so that a question asked after the
        run does not need the homes to be run again.
    """

    behavioural: tuple[tuple[datetime, str], ...]
    silence: tuple[datetime, ...]
    other: tuple[tuple[datetime, str, str], ...]
    closes: tuple[datetime, ...]
    usable: int
    refused: tuple[date, ...]
    described: tuple[tuple[str, str, str], ...] = ()

    def count(self, begin: datetime, end: datetime) -> int:
        """Behavioural alerts raised from *begin* and before *end*."""
        return sum(1 for at, _ in self.behavioural if begin <= at < end)

    def first(self, feature: str, begin: datetime, end: datetime) -> datetime | None:
        """The first behavioural alert about *feature* in the window."""
        return next(
            (
                at
                for at, subject in self.behavioural
                if subject == feature and begin <= at < end
            ),
            None,
        )


@dataclass(frozen=True)
class HomeRecord:
    """Everything one simulated home contributes.

    Attributes
    ----------
    seed
        The home's seed.
    runs
        One pipeline run for each arm and condition the protocol declares.
    observed
        For each arm the fleet check reads, the moments of the home's
        delivered observations, in microseconds since the epoch, in order.
    """

    seed: int
    runs: Mapping[tuple[str, str], PipelineRun]
    observed: Mapping[str, np.ndarray]


def change_of(protocol: SilentHomeProtocol) -> BehaviourShift:
    """The behavioural change the protocol injects."""
    return BehaviourShift(
        start_day=protocol.change_day,
        sleep_delta_hours=protocol.change_sleep_delta_hours,
        night_bathroom_extra=protocol.change_night_bathroom_extra,
    )


def deliver(
    observations: Sequence[Observation],
    sensors: Sequence[str],
    outage: tuple[datetime, datetime] | None,
) -> list[Observation]:
    """Deliver a record, with every sensor silent over *outage* if one is given."""
    faults = (
        ()
        if outage is None
        else tuple(dropout(name, outage[0], outage[1] - outage[0]) for name in sensors)
    )
    delivered, _ = degrade(observations, DegradationConfig(faults=faults))
    return delivered


def run_pipeline(
    registry: SensorRegistry,
    observations: Sequence[Observation],
    *,
    horizon: timedelta | None,
    step: timedelta,
    zone: tzinfo,
    end: datetime,
) -> PipelineRun:
    """Run one delivered record through the pipeline, with the rule at *horizon*."""
    pipeline = BehaviouralSensingPipeline(
        registry,
        config=PipelineConfig(tz=zone, step=step),
        health_config=HealthConfig(home_silence_horizon=horizon),
    )
    steps = pipeline.run(observations)
    steps.extend(pipeline.close(end))

    behavioural: list[tuple[datetime, str]] = []
    described: list[tuple[str, str, str]] = []
    silence: list[datetime] = []
    other: list[tuple[datetime, str, str]] = []
    closes: list[datetime] = []
    refused: list[date] = []
    usable = 0
    for moment in steps:
        verdicts = {change.feature: change for change in moment.changes}
        for alert in moment.alerts:
            if alert.kind is AlertKind.BEHAVIOURAL_CHANGE:
                behavioural.append((alert.at, str(alert.subject)))
                verdict = verdicts.get(str(alert.subject))
                described.append(
                    (
                        verdict.kind.value if verdict is not None else "",
                        str(verdict.direction) if verdict is not None else "",
                        alert.severity.value,
                    )
                )
            elif alert.kind is AlertKind.DATA_QUALITY and alert.subject == HOME_SILENCE:
                silence.append(alert.at)
            else:
                other.append((alert.at, alert.kind.value, str(alert.subject)))
        summary = moment.day_closed
        if summary is None:
            continue
        closes.append(moment.at)
        if summary.silent > 0.0:
            refused.append(summary.day)
        elif summary.is_usable(
            pipeline.config.min_day_coverage, pipeline.config.min_day_observed
        ):
            usable += 1
    return PipelineRun(
        behavioural=tuple(behavioural),
        silence=tuple(silence),
        other=tuple(other),
        closes=tuple(closes),
        usable=usable,
        refused=tuple(refused),
        described=tuple(described),
    )


def _moments(observations: Iterable[Observation]) -> np.ndarray:
    return np.array(
        [(o.timestamp - EPOCH) // MICROSECOND for o in observations], dtype=np.int64
    )


@dataclass(frozen=True)
class DeliveredHome:
    """One simulated home's delivered records, before any is run.

    Attributes
    ----------
    registry
        The event sensors' registry.
    zone
        The home's timezone.
    records
        The delivered record of every arm, the fleet check's included.
    ends
        The last instant of simulated time behind each arm.
    """

    registry: SensorRegistry
    zone: tzinfo
    records: Mapping[str, Sequence[Observation]]
    ends: Mapping[str, datetime]


def deliver_home(seed: int, protocol: SilentHomeProtocol) -> DeliveredHome:
    """Simulate one home and deliver the record of every arm the protocol declares.

    The outage arms are the stable home's record with every sensor silent over
    a window, and the change after an outage is the changed home's record with
    the same window silent, so each differs from its pair in the outage alone.
    The stable and the changed home share a seed and not their sensor events:
    the simulator draws the events after it has planned every day.
    """
    sensors = event_sensors()
    stable = simulate(HouseholdConfig(days=protocol.days, seed=seed))
    changed = simulate(
        HouseholdConfig(days=protocol.days, seed=seed, shift=change_of(protocol))
    )
    quiet, moved = stable.observations_for(sensors), changed.observations_for(sensors)
    outage = protocol.outage(seed)
    return DeliveredHome(
        registry=stable.registry.subset(sensors),
        zone=stable.config.tz,
        records={
            STABLE: deliver(quiet, sensors, None),
            OUTAGE: deliver(quiet, sensors, outage),
            SHORT_OUTAGE: deliver(quiet, sensors, protocol.short_outage(seed)),
            CHANGE: deliver(moved, sensors, None),
            CHANGE_AFTER_OUTAGE: deliver(moved, sensors, outage),
            COMMON: deliver(quiet, sensors, protocol.common_outage()),
        },
        ends={
            STABLE: stable.end,
            OUTAGE: stable.end,
            SHORT_OUTAGE: stable.end,
            CHANGE: changed.end,
            CHANGE_AFTER_OUTAGE: changed.end,
            COMMON: stable.end,
        },
    )


def run_home(seed: int, protocol: SilentHomeProtocol) -> HomeRecord:
    """Simulate one home and run every arm and condition the protocol declares."""
    home = deliver_home(seed, protocol)
    step = timedelta(minutes=protocol.step_minutes)
    runs = {
        (arm, condition): run_pipeline(
            home.registry,
            home.records[arm],
            horizon=protocol.horizon(condition),
            step=step,
            zone=home.zone,
            end=home.ends[arm],
        )
        for arm, condition in protocol.runs()
    }
    observed = {arm: _moments(home.records[arm]) for arm in FLEET_ARMS}
    return HomeRecord(seed=seed, runs=runs, observed=observed)


def _checkpoint(directory: Path, seed: int) -> Path:
    return directory / f"home-{seed}.pickle"


def _run_home(arguments: tuple[int, SilentHomeProtocol, Path | None]) -> HomeRecord:
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
    protocol: SilentHomeProtocol,
    *,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
    checkpoint: Path | None = None,
) -> list[HomeRecord]:
    """Run every home of the protocol, in the order of its seeds.

    Each home is a function of its seed and the protocol alone, so the result
    does not depend on *jobs*.

    With *checkpoint*, each home is written to that directory as soon as it
    is finished and read back from it if the run is started again, so that a
    failure late in the run does not cost the homes already done. A directory
    must hold the homes of one protocol run by one version of the code; the
    caller names it accordingly.
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
def wilson(successes: int, trials: int, confidence: float) -> dict[str, Any] | None:
    """The Wilson score interval of a share of independent homes."""
    if trials < 1:
        return None
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    share = successes / trials
    centre = (share + z * z / (2 * trials)) / (1 + z * z / trials)
    half = (
        z
        * math.sqrt(share * (1 - share) / trials + z * z / (4 * trials * trials))
        / (1 + z * z / trials)
    )
    # The bounds are exact at the ends, where the arithmetic leaves a rounding
    # error of the last place.
    return {
        "low": 0.0 if successes == 0 else max(0.0, centre - half),
        "high": 1.0 if successes == trials else min(1.0, centre + half),
        "confidence": confidence,
        "method": "wilson",
    }


def mean_estimate(
    values: Sequence[float] | np.ndarray, protocol: SilentHomeProtocol
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


def share_estimate(
    flags: Sequence[bool], protocol: SilentHomeProtocol
) -> dict[str, Any]:
    """A share of homes with its Wilson interval."""
    count, homes = int(sum(bool(flag) for flag in flags)), len(flags)
    return {
        "estimate": count / homes if homes else None,
        "count": count,
        "homes": homes,
        "interval": wilson(count, homes, protocol.confidence),
    }


def median_delay(
    delays: Sequence[float | None], protocol: SilentHomeProtocol
) -> dict[str, Any]:
    """The pooled median delay over the homes that detected, resampling homes.

    A home that did not detect has no delay and is not given one. A resample
    in which no home detected has no median and is left out; how many
    remained is reported.
    """
    data = np.array([np.nan if d is None else d for d in delays], dtype=float)
    detected = data[~np.isnan(data)]
    result: dict[str, Any] = {
        "estimate": float(np.median(detected)) if detected.size else None,
        "interval": None,
        "detections": int(detected.size),
        "homes": int(data.size),
        "resamples_defined": 0,
        "delays": sorted(float(d) for d in detected),
    }
    if data.size < 2 or not detected.size:
        return result
    index = resample_indices(data.size, protocol.resamples, protocol.seed)
    resampled = data[index]
    defined = ~np.isnan(resampled).all(axis=1)
    result["resamples_defined"] = int(defined.sum())
    if defined.any():
        result["interval"] = percentile_interval(
            np.nanmedian(resampled[defined], axis=1), protocol.confidence
        ).to_dict()
    return result


def _spread(values: Sequence[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    if not data.size:
        return {"homes": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "homes": int(data.size),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def _low(estimate: Mapping[str, Any]) -> float:
    interval = estimate["interval"]
    return float(interval["low"]) if interval else float("nan")


def _high(estimate: Mapping[str, Any]) -> float:
    interval = estimate["interval"]
    return float(interval["high"]) if interval else float("nan")


# ----------------------------------------------------------------------------
# The estimands
# ----------------------------------------------------------------------------
def _excess(
    homes: Sequence[HomeRecord], arm: str, condition: str, protocol: SilentHomeProtocol
) -> np.ndarray:
    """Each home's alerts in its outage window, minus the stable arm's there."""
    values = []
    for home in homes:
        begin, end = protocol.window(home.seed, arm)
        values.append(
            home.runs[(arm, condition)].count(begin, end)
            - home.runs[(STABLE, condition)].count(begin, end)
        )
    return np.asarray(values, dtype=float)


def _changed(homes: Sequence[HomeRecord], arm: str, condition: str) -> list[int]:
    """The homes in which the rule changes a behavioural alert of *arm*."""
    return [
        home.seed
        for home in homes
        if set(home.runs[(arm, condition)].behavioural)
        != set(home.runs[(arm, OFF)].behavioural)
    ]


def _groups(
    homes: Sequence[HomeRecord], protocol: SilentHomeProtocol
) -> dict[str, np.ndarray]:
    group = np.array([protocol.group(home.seed) for home in homes])
    return {name: group == name for name in (IN_TIME, LATE)}


def _outage(
    homes: Sequence[HomeRecord], protocol: SilentHomeProtocol
) -> dict[str, Any]:
    """E1, E2, E3, their groups, and the other horizons' share of S1."""
    groups = _groups(homes, protocol)
    excess = {c: _excess(homes, OUTAGE, c, protocol) for c in protocol.conditions}
    result: dict[str, Any] = {}
    for condition, values in excess.items():
        entry: dict[str, Any] = {
            "excess": mean_estimate(values, protocol),
            "alerts_in_the_window": int(
                sum(
                    home.runs[(OUTAGE, condition)].count(
                        *protocol.window(home.seed, OUTAGE)
                    )
                    for home in homes
                )
            ),
            "stable_alerts_in_the_window": int(
                sum(
                    home.runs[(STABLE, condition)].count(
                        *protocol.window(home.seed, OUTAGE)
                    )
                    for home in homes
                )
            ),
            "homes_with_an_excess": int((values > 0).sum()),
            "by_group": {
                name: mean_estimate(values[mask], protocol)
                for name, mask in groups.items()
            },
        }
        if condition != OFF:
            removed = excess[OFF] - values
            entry["removed"] = mean_estimate(removed, protocol)
            entry["removed_by_group"] = {
                name: mean_estimate(removed[mask], protocol)
                for name, mask in groups.items()
            }
        result[condition] = entry
    return result


def _timeline(
    homes: Sequence[HomeRecord], protocol: SilentHomeProtocol
) -> dict[str, Any]:
    """The alerts behind E1 and E2, by whole days since each home's outage began."""
    length = math.ceil(protocol.outage_hours / 24.0) + protocol.follow_days
    counts: dict[str, dict[str, list[int]]] = {}
    for condition in protocol.conditions:
        by_arm = {OUTAGE: [0] * length, STABLE: [0] * length}
        for home in homes:
            begin, end = protocol.window(home.seed, OUTAGE)
            for arm, row in by_arm.items():
                for at, _ in home.runs[(arm, condition)].behavioural:
                    if begin <= at < end:
                        row[(at - begin) // DAY] += 1
        counts[condition] = by_arm
    return {"days_since_the_outage_began": list(range(length)), "alerts": counts}


def _stable(
    homes: Sequence[HomeRecord], protocol: SilentHomeProtocol
) -> dict[str, Any]:
    """E4, and its counts at every horizon."""
    result: dict[str, Any] = {"person_days": len(homes) * protocol.days}
    for condition in protocol.conditions:
        runs = [home.runs[(STABLE, condition)] for home in homes]
        alerts = [len(run.behavioural) for run in runs]
        silences = [len(run.silence) for run in runs]
        entry: dict[str, Any] = {
            "behavioural_alerts": int(sum(alerts)),
            "per_person_day": mean_estimate(
                [count / protocol.days for count in alerts], protocol
            ),
            "by_feature": dict(
                sorted(
                    Counter(
                        subject for run in runs for _, subject in run.behavioural
                    ).items()
                )
            ),
        }
        if condition != OFF:
            changed = _changed(homes, STABLE, condition)
            entry.update(
                {
                    "homes_in_which_the_rule_changes_an_alert": len(changed),
                    "those_homes": changed,
                    "silence_alerts": int(sum(silences)),
                    "silence_alerts_per_person_day": sum(silences)
                    / (len(homes) * protocol.days),
                    "homes_with_a_silence_alert": int(sum(s > 0 for s in silences)),
                    "days_refused": int(sum(len(run.refused) for run in runs)),
                }
            )
        result[condition] = entry
    return result


def _detection(
    homes: Sequence[HomeRecord], arm: str, protocol: SilentHomeProtocol
) -> dict[str, Any]:
    """E5 or E6: detection of the change with the rule off and on."""
    begin, end = protocol.detection_window()
    primary = condition_name(protocol.primary_hours)
    flags: dict[str, np.ndarray] = {}
    result: dict[str, Any] = {}
    for condition in (OFF, primary):
        first = [
            home.runs[(arm, condition)].first(TRACKED_FEATURE, begin, end)
            for home in homes
        ]
        flags[condition] = np.array([at is not None for at in first])
        false = [
            home.runs[(WITHOUT_THE_CHANGE[arm], condition)].first(
                TRACKED_FEATURE, begin, end
            )
            is not None
            for home in homes
        ]
        result[condition] = {
            "detected": share_estimate(list(flags[condition]), protocol),
            "delay_days": median_delay(
                [
                    None if at is None else (at - protocol.change_begins()) / DAY
                    for at in first
                ],
                protocol,
            ),
            "false_detections": share_estimate(false, protocol),
            "false_detections_in": WITHOUT_THE_CHANGE[arm],
        }
    difference = flags[primary].astype(float) - flags[OFF].astype(float)
    result["on_minus_off"] = {
        **mean_estimate(difference, protocol),
        "detected_only_with_the_rule_on": int((difference > 0).sum()),
        "detected_only_with_the_rule_off": int((difference < 0).sum()),
    }
    return result


def _reporting(
    homes: Sequence[HomeRecord], condition: str, protocol: SilentHomeProtocol
) -> dict[str, Any]:
    """E7: whether a home's outage is reported, how soon and how often.

    The silence alerts per home are counted twice over: those dated inside the
    home's outage, which are the ones that report it, and all of them.
    """
    reported, hours, during, in_all, refused = [], [], [], [], []
    for home in homes:
        begin, end = protocol.outage(home.seed)
        run = home.runs[(OUTAGE, condition)]
        inside = [at for at in run.silence if begin <= at <= end]
        reported.append(bool(inside))
        if inside:
            hours.append((inside[0] - begin) / timedelta(hours=1))
        during.append(len(inside))
        in_all.append(len(run.silence))
        refused.append(len(run.refused))
    return {
        "reported": share_estimate(reported, protocol),
        "hours_to_the_first_silence_alert": _spread(hours),
        "silence_alerts_per_home": _spread(during),
        "silence_alerts_per_home_in_all": _spread(in_all),
        "silence_alerts_outside_the_outage": int(sum(in_all) - sum(during)),
        "days_refused_per_home": _spread(refused),
        "days_refused_distribution": {
            str(days): count for days, count in sorted(Counter(refused).items())
        },
    }


def _datetimes(moments: np.ndarray) -> list[datetime]:
    return [EPOCH + timedelta(microseconds=int(value)) for value in moments]


def fleet_config(protocol: SilentHomeProtocol) -> FleetConfig:
    """The fleet check the protocol declares."""
    return FleetConfig(
        horizon=timedelta(hours=protocol.primary_hours),
        fraction=protocol.fleet_fraction,
        min_homes=protocol.fleet_min_homes,
    )


def _fleet_arm(
    observed: Mapping[str, Sequence[datetime]], protocol: SilentHomeProtocol
) -> tuple[list[CommonSilence], dict[str, Any]]:
    """The stretches of common silence in one fleet, and every assessment's share."""
    config = fleet_config(protocol)
    every = timedelta(hours=protocol.fleet_every_hours)
    stretches = common_silences(observed, config, every=every)
    verdicts = list(replay_fleet(observed, config, every=every))
    shares = [verdict.share for verdict in verdicts]
    largest = max(shares, default=0.0)
    when = next((v.at for v in verdicts if v.share == largest and largest > 0), None)
    return stretches, {
        "stretches": [stretch.to_dict() for stretch in stretches],
        "assessments": len(verdicts),
        "first_assessment": verdicts[0].at.isoformat() if verdicts else None,
        "hours_between_assessments": protocol.fleet_every_hours,
        "largest_share_silent": largest,
        "largest_share_at": when.isoformat() if when is not None else None,
        "share_silent": [round(share, 4) for share in shares],
        "homes_monitored": [len(verdict.monitored) for verdict in verdicts],
    }


def _fleet(homes: Sequence[HomeRecord], protocol: SilentHomeProtocol) -> dict[str, Any]:
    """E8: what the fleet check finds in each arm it reads."""
    begin, end = protocol.common_outage()
    result: dict[str, Any] = {"config": fleet_config(protocol).to_dict()}
    for arm in FLEET_ARMS:
        observed = {str(home.seed): _datetimes(home.observed[arm]) for home in homes}
        stretches, entry = _fleet_arm(observed, protocol)
        if arm == COMMON:
            entry["outage"] = {"begin": begin.isoformat(), "end": end.isoformat()}
            over = [s for s in stretches if s.since <= end and s.until >= begin]
            entry["stretches_overlapping_the_outage"] = len(over)
            entry["hours_to_the_first_assessment_that_calls_it_common"] = (
                (over[0].detected - begin) / timedelta(hours=1) if over else None
            )
        result[arm] = entry
    return result


def _short_outage(
    homes: Sequence[HomeRecord], protocol: SilentHomeProtocol
) -> dict[str, Any]:
    """E9: what an outage shorter than every horizon does, rule off and on."""
    primary = condition_name(protocol.primary_hours)
    changed = _changed(homes, SHORT_OUTAGE, primary)
    silent = [home.seed for home in homes if home.runs[(SHORT_OUTAGE, primary)].silence]
    return {
        OFF: {
            "excess": mean_estimate(
                _excess(homes, SHORT_OUTAGE, OFF, protocol), protocol
            )
        },
        primary: {
            "excess": mean_estimate(
                _excess(homes, SHORT_OUTAGE, primary, protocol), protocol
            ),
            "homes_in_which_the_rule_changes_an_alert": len(changed),
            "those_homes": changed,
            "homes_with_a_silence_alert": len(silent),
            "homes_with_a_silence_alert_listed": silent,
            "days_refused": int(
                sum(len(home.runs[(SHORT_OUTAGE, primary)].refused) for home in homes)
            ),
        },
    }


def _per_home(
    homes: Sequence[HomeRecord], protocol: SilentHomeProtocol
) -> dict[str, dict[str, Any]]:
    """Every home's own values behind the estimands, one flat row each."""
    begin, end = protocol.detection_window()
    starts = protocol.outage_starts()
    table: dict[str, dict[str, Any]] = {}
    for home in homes:
        day, hour = starts[home.seed]
        outage = protocol.outage(home.seed)
        first_day = protocol.start() + timedelta(days=day)
        windows = {
            OUTAGE: protocol.window(home.seed, OUTAGE),
            SHORT_OUTAGE: protocol.window(home.seed, SHORT_OUTAGE),
        }
        row: dict[str, Any] = {
            "outage_day": day,
            "outage_hour": hour,
            "group": protocol.group(home.seed),
        }
        for (arm, condition), run in home.runs.items():
            name = f"{arm}/{condition}"
            row[f"{name}/behavioural_alerts"] = len(run.behavioural)
            row[f"{name}/silence_alerts"] = len(run.silence)
            row[f"{name}/days_refused"] = len(run.refused)
            row[f"{name}/other_alerts"] = len(run.other)
            if arm in (STABLE, OUTAGE):
                row[f"{name}/alerts_in_the_outage_window"] = run.count(*windows[OUTAGE])
            if arm in (STABLE, SHORT_OUTAGE) and (SHORT_OUTAGE, condition) in home.runs:
                row[f"{name}/alerts_in_the_short_outage_window"] = run.count(
                    *windows[SHORT_OUTAGE]
                )
            if (
                arm in (CHANGE, CHANGE_AFTER_OUTAGE)
                or (CHANGE, condition) in home.runs
                and arm in WITHOUT_THE_CHANGE.values()
            ):
                first = run.first(TRACKED_FEATURE, begin, end)
                row[f"{name}/meets_the_detection_definition"] = first is not None
                row[f"{name}/delay_days"] = (
                    None if first is None else (first - protocol.change_begins()) / DAY
                )
            if arm == OUTAGE and condition != OFF:
                inside = [at for at in run.silence if outage[0] <= at <= outage[1]]
                row[f"{name}/reported"] = bool(inside)
                row[f"{name}/hours_to_the_first_silence_alert"] = (
                    (inside[0] - outage[0]) / timedelta(hours=1) if inside else None
                )
                row[f"{name}/silence_alerts_inside_the_outage"] = len(inside)
                row[f"{name}/first_outage_day_refused"] = first_day in run.refused
        table[str(home.seed)] = row
    return table


def _raw(homes: Sequence[HomeRecord]) -> dict[str, dict[str, Any]]:
    """Every alert and every refused day of every run, as they were delivered.

    No estimand reads the verdict, the direction or the severity kept beside
    each behavioural alert.
    """
    table: dict[str, dict[str, Any]] = {}
    for home in homes:
        runs: dict[str, Any] = {}
        for (arm, condition), run in home.runs.items():
            described = run.described or (("", "", ""),) * len(run.behavioural)
            runs[f"{arm}/{condition}"] = {
                "behavioural_alerts": [
                    [at.isoformat(), subject, *about]
                    for (at, subject), about in zip(run.behavioural, described)
                ],
                "silence_alerts": [at.isoformat() for at in run.silence],
                "other_alerts": [
                    [at.isoformat(), kind, subject] for at, kind, subject in run.other
                ],
                "days_refused": [day.isoformat() for day in run.refused],
            }
        table[str(home.seed)] = runs
    return table


def _totals(
    homes: Sequence[HomeRecord], protocol: SilentHomeProtocol
) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for arm, condition in protocol.runs():
        runs = [home.runs[(arm, condition)] for home in homes]
        totals[f"{arm}/{condition}"] = {
            "days_closed": int(sum(len(run.closes) for run in runs)),
            "usable_days": int(sum(run.usable for run in runs)),
            "days_refused_for_silence": int(sum(len(run.refused) for run in runs)),
            "behavioural_alerts": int(sum(len(run.behavioural) for run in runs)),
            "silence_alerts": int(sum(len(run.silence) for run in runs)),
            "other_alerts": int(sum(len(run.other) for run in runs)),
            "other_alerts_by_subject": dict(
                sorted(
                    Counter(
                        subject for run in runs for _, _, subject in run.other
                    ).items()
                )
            ),
        }
    return totals


# ----------------------------------------------------------------------------
# The criteria
# ----------------------------------------------------------------------------
def _non_inferiority(
    detection: Mapping[str, Any], protocol: SilentHomeProtocol
) -> dict[str, Any]:
    off = detection[OFF]["detected"]["estimate"]
    difference = detection["on_minus_off"]
    low, high = _low(difference), _high(difference)
    if off < protocol.informative_recall:
        verdict = "uninformative"
    elif low > -protocol.margin_recall:
        verdict = "non-inferior"
    elif high < -protocol.margin_recall:
        verdict = "inferior"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "share_detected_with_the_rule_off": off,
        "difference": difference["estimate"],
        "interval": difference["interval"],
        "margin": -protocol.margin_recall,
    }


def criteria(
    results: Mapping[str, Any], protocol: SilentHomeProtocol
) -> dict[str, dict[str, Any]]:
    """Decide the protocol's criteria from its estimands, as it declares them."""
    primary = condition_name(protocol.primary_hours)
    e1 = results["outage"][OFF]["excess"]
    e2 = results["outage"][primary]["excess"]
    e3 = results["outage"][primary]["removed"]
    reproduced = _low(e1) > 0.0
    if not reproduced:
        removes = "not testable"
    elif _low(e3) > 0.0 and _high(e2) < protocol.margin_alerts:
        removes = "success"
    elif not _low(e3) > 0.0:
        removes = "failure"
    else:
        removes = "partial"

    stable = results["stable"][primary]
    harmless = (
        stable["homes_in_which_the_rule_changes_an_alert"] == 0
        and stable["silence_alerts"] == 0
    )
    reported = results["reporting"][primary]["reported"]
    fleet = results["fleet"]
    told = (
        len(fleet[COMMON]["stretches"]) == 1
        and fleet[COMMON]["stretches_overlapping_the_outage"] == 1
        and not fleet[STABLE]["stretches"]
        and not fleet[OUTAGE]["stretches"]
    )
    short = results["short_outage"][primary]
    unseen = (
        short["homes_in_which_the_rule_changes_an_alert"] == 0
        and short["homes_with_a_silence_alert"] == 0
    )
    return {
        "C1": {
            "verdict": "reproduced" if reproduced else "not reproduced",
            "E1": e1["estimate"],
            "interval": e1["interval"],
        },
        "C2": {
            "verdict": removes,
            "E3": e3["estimate"],
            "E3_interval": e3["interval"],
            "E2": e2["estimate"],
            "E2_interval": e2["interval"],
            "margin": protocol.margin_alerts,
        },
        "C3": {
            "verdict": "success" if harmless else "failure",
            "homes_in_which_the_rule_changes_an_alert": stable[
                "homes_in_which_the_rule_changes_an_alert"
            ],
            "silence_alerts": stable["silence_alerts"],
        },
        "C4": _non_inferiority(results["detection"][CHANGE], protocol),
        "C5": _non_inferiority(results["detection"][CHANGE_AFTER_OUTAGE], protocol),
        "C6": {
            "verdict": (
                "success"
                if reported["estimate"] >= protocol.reported_share
                else "failure"
            ),
            "share_reported": reported["estimate"],
            "needed": protocol.reported_share,
        },
        "C7": {
            "verdict": "success" if told else "failure",
            "stretches": {arm: len(fleet[arm]["stretches"]) for arm in FLEET_ARMS},
            "overlapping_the_outage": fleet[COMMON]["stretches_overlapping_the_outage"],
        },
        "C8": {
            "verdict": "confirmed" if unseen else "not confirmed",
            "homes_in_which_the_rule_changes_an_alert": short[
                "homes_in_which_the_rule_changes_an_alert"
            ],
            "homes_with_a_silence_alert": short["homes_with_a_silence_alert"],
        },
    }


def score(homes: Sequence[HomeRecord], protocol: SilentHomeProtocol) -> dict[str, Any]:
    """Every estimand and criterion of the protocol, from the homes' runs."""
    on = [c for c in protocol.conditions if c != OFF]
    results: dict[str, Any] = {
        "result_schema": RESULT_SCHEMA,
        "homes": len(homes),
        "days": protocol.days,
        "runs": _totals(homes, protocol),
        "outage": _outage(homes, protocol),
        "timeline": _timeline(homes, protocol),
        "stable": _stable(homes, protocol),
        "detection": {
            arm: _detection(homes, arm, protocol)
            for arm in (CHANGE, CHANGE_AFTER_OUTAGE)
        },
        "reporting": {c: _reporting(homes, c, protocol) for c in on},
        "fleet": _fleet(homes, protocol),
        "short_outage": _short_outage(homes, protocol),
        "per_home": _per_home(homes, protocol),
        "raw": {
            "behavioural_alert_fields": [
                "moment",
                "subject",
                "verdict",
                "direction",
                "severity",
            ],
            "other_alert_fields": ["moment", "kind", "subject"],
            "homes": _raw(homes),
        },
    }
    results["criteria"] = criteria(results, protocol)
    return results


# ----------------------------------------------------------------------------
# The record of the test
# ----------------------------------------------------------------------------
def _interval(
    label: str, estimate: Mapping[str, Any], protocol: SilentHomeProtocol
) -> ReportedInterval | None:
    interval = estimate.get("interval")
    if not interval:
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


def _intervals(
    results: Mapping[str, Any], protocol: SilentHomeProtocol
) -> list[ReportedInterval]:
    primary = condition_name(protocol.primary_hours)
    outage, detection = results["outage"], results["detection"]
    wanted = [
        ("E1: excess alerts per home, outage, rule off", outage[OFF]["excess"]),
        ("E2: excess alerts per home, outage, rule on", outage[primary]["excess"]),
        ("E3: excess alerts the rule removes per home", outage[primary]["removed"]),
        (
            "E4: behavioural alerts per person-day, stable, rule off",
            results["stable"][OFF]["per_person_day"],
        ),
        (
            "E4: behavioural alerts per person-day, stable, rule on",
            results["stable"][primary]["per_person_day"],
        ),
        *[
            (
                f"{name}: share detected, {arm}, rule {state}",
                detection[arm][c]["detected"],
            )
            for name, arm in (("E5", CHANGE), ("E6", CHANGE_AFTER_OUTAGE))
            for c, state in ((OFF, "off"), (primary, "on"))
        ],
        *[
            (
                f"{name}: share detected, {arm}, on minus off",
                detection[arm]["on_minus_off"],
            )
            for name, arm in (("E5", CHANGE), ("E6", CHANGE_AFTER_OUTAGE))
        ],
        (
            "E7: share of homes whose outage is reported",
            results["reporting"][primary]["reported"],
        ),
        (
            "E9: excess alerts per home, short outage, rule off",
            results["short_outage"][OFF]["excess"],
        ),
        (
            "E9: excess alerts per home, short outage, rule on",
            results["short_outage"][primary]["excess"],
        ),
    ]
    found = [_interval(label, estimate, protocol) for label, estimate in wanted]
    return [interval for interval in found if interval is not None]


def _mcse(results: Mapping[str, Any], protocol: SilentHomeProtocol) -> dict[str, float]:
    primary = condition_name(protocol.primary_hours)
    outage, detection = results["outage"], results["detection"]
    named = {
        "E1": outage[OFF]["excess"],
        "E2": outage[primary]["excess"],
        "E3": outage[primary]["removed"],
        "E5_on_minus_off": detection[CHANGE]["on_minus_off"],
        "E6_on_minus_off": detection[CHANGE_AFTER_OUTAGE]["on_minus_off"],
        "E9_off": results["short_outage"][OFF]["excess"],
        "E9_on": results["short_outage"][primary]["excess"],
    }
    for condition in protocol.conditions:
        if condition not in (OFF, primary):
            named[f"S1_E2_{condition}"] = outage[condition]["excess"]
            named[f"S1_E3_{condition}"] = outage[condition]["removed"]
    return {
        name: float(estimate["mcse"])
        for name, estimate in named.items()
        if estimate.get("mcse") is not None
    }


@dataclass(frozen=True)
class SilentHomeResult:
    """A run of the silent-home evaluation: the record, and where it was written."""

    record: ExperimentRecord
    path: Path | None


def record_of(
    homes: Sequence[HomeRecord],
    protocol: SilentHomeProtocol,
    *,
    protocol_sha256: str,
    inputs: Sequence[InputArtifact] = (),
) -> ExperimentRecord:
    """Score the homes and build the record of the protocol's test."""
    results = score(homes, protocol)
    baseline, policy, pipeline = BaselineConfig(), AlertPolicy(), PipelineConfig()
    return ExperimentRecord(
        experiment=EXPERIMENT,
        configuration={
            **protocol.to_dict(),
            "protocol_sha256": protocol.sha256(),
            "protocol_file_sha256": protocol_sha256,
            "result_schema": RESULT_SCHEMA,
            "between_the_freeze_and_the_run": list(BETWEEN_THE_FREEZE_AND_THE_RUN),
        },
        inference=ONLINE,
        evidence=EvidenceSummary.online(
            moment
            for home in homes
            for run in home.runs.values()
            for moment in run.closes
        ),
        seeds=[protocol.seed_root, protocol.seed, *protocol.study_seeds()],
        results=results,
        sensor_subset=list(event_sensors()),
        data_source="simulator",
        metric_definitions=dict(protocol.to_dict()["definitions"]),
        notes=[
            "Pre-specified: the protocol was frozen and pushed before any "
            "simulated home was run with the rule on.",
            f"{len(homes)} paired simulated homes, each run in every arm and "
            "condition. No pilot informed the protocol: no simulated home had "
            "been run with the rule on when it was frozen. The count is the "
            "floor docs/SIMULATION_PROTOCOLS.md sets, and the Monte Carlo "
            "error achieved is reported for every paired difference.",
            *(
                text[0].upper() + text[1:] + "."
                for text in BETWEEN_THE_FREEZE_AND_THE_RUN
            ),
            "The outage arms are the stable home's record with every sensor "
            "silent over a window, and the change after an outage is the "
            "changed home's record with the same window silent, so those "
            "pairs differ in the outage alone. The stable and the changed "
            "home share a seed and not their sensor events before the change, "
            "because the simulator draws the events after planning every day. "
            "No paired estimand compares the two.",
            "Nothing was fitted. The pipeline's emissions, baseline and alert "
            "policy are the declared defaults, and the horizons were declared, "
            "not estimated.",
            "A simulated home is silent for long only when a fault makes it "
            "so. The result says nothing about how often a real home is silent "
            "for a harmless reason, or about which horizon a real home needs.",
        ],
        inputs=list(inputs),
        preprocessing={
            "sensors": list(event_sensors()),
            "delivery": "simulation.faults.degrade, with no loss, lateness or "
            "duplication; an outage is a dropout fault on every sensor",
            "step_minutes": protocol.step_minutes,
        },
        models=[
            ModelRecord(
                f"online_pipeline_{condition}",
                {
                    "home_silence_horizon_hours": (
                        None
                        if protocol.horizon(condition) is None
                        else protocol.horizon(condition) / timedelta(hours=1)  # type: ignore[operator]
                    ),
                    "features": [f"{s.value}_hours" for s in pipeline.features],
                    "baseline_min_samples": baseline.min_samples,
                    "deviation_threshold": baseline.deviation_threshold,
                    "persistence_days": baseline.persistence_days,
                    "cooldown_hours": policy.cooldown / timedelta(hours=1),
                },
                "BehaviouralSensingPipeline",
            )
            for condition in protocol.conditions
        ],
        household_metrics={"simulated_homes": results["per_home"]},
        intervals=_intervals(results, protocol),
        mcse=_mcse(results, protocol),
    )


def run_silent_home(
    protocol: SilentHomeProtocol,
    *,
    protocol_sha256: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
    checkpoint: Path | None = None,
) -> SilentHomeResult:
    """Run every home of the protocol and write the record of its test."""
    homes = run_homes(protocol, jobs=jobs, progress=progress, checkpoint=checkpoint)
    record = record_of(homes, protocol, protocol_sha256=protocol_sha256, inputs=inputs)
    path = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = record.write(Path(output_dir) / f"{EXPERIMENT}.json")
    return SilentHomeResult(record, path)


# ----------------------------------------------------------------------------
# The description on TIHM
# ----------------------------------------------------------------------------
def _run_tihm_home(
    arguments: tuple[HouseholdData, TihmProtocol, float | None],
) -> HouseholdRun:
    data, tihm, hours = arguments
    horizon = None if hours is None else timedelta(hours=hours)
    return run_household(data, tihm, HealthConfig(home_silence_horizon=horizon))


def run_tihm_homes(
    adapter: DatasetAdapter,
    protocol: SilentHomeProtocol,
    tihm: TihmProtocol,
    *,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[HouseholdRun]]:
    """Run every TIHM home that passes the contract, under every condition."""
    loaded = []
    for name in adapter.households():
        data = adapter.load(name)
        if validate_household(data, tihm.mapping).ok:
            loaded.append(data)
    if not loaded:
        raise ValueError("no household passed the contract, so there is nothing to run")
    hours: dict[str, float | None] = {
        condition: (
            None
            if protocol.horizon(condition) is None
            else protocol.horizon(condition) / timedelta(hours=1)  # type: ignore[operator]
        )
        for condition in protocol.conditions
    }
    tasks = [(data, tihm, hours[c]) for c in protocol.conditions for data in loaded]
    done: list[HouseholdRun] = []
    if jobs <= 1:
        for task in tasks:
            done.append(_run_tihm_home(task))
            if progress is not None:
                progress(len(done), len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for run in pool.map(_run_tihm_home, tasks):
                done.append(run)
                if progress is not None:
                    progress(len(done), len(tasks))
    size = len(loaded)
    return {
        condition: done[k * size : (k + 1) * size]
        for k, condition in enumerate(protocol.conditions)
    }


def _tihm_alerts(runs: Sequence[HouseholdRun]) -> set[tuple[str, date, str]]:
    return {
        (run.household, day.day, subject)
        for run in runs
        for day in run.days
        for subject, _ in day.alerts
    }


def _tihm_condition(
    runs: Sequence[HouseholdRun],
    off: Sequence[HouseholdRun],
    stretches: Sequence[CommonSilence],
    zone: tzinfo,
) -> dict[str, Any]:
    days = [day for run in runs for day in run.days]
    by_day: Counter[date] = Counter()
    by_date: Counter[date] = Counter()
    by_feature: Counter[str] = Counter()
    for run in runs:
        # A day is closed at the first step of the next one, so an alert is
        # raised on the local date after the day whose summary raised it.
        for day, close in zip(run.days, run.closes):
            for subject, _ in day.alerts:
                by_day[day.day] += 1
                by_date[close.astimezone(zone).date()] += 1
                by_feature[subject] += 1
    refused = {
        run.household: sum(1 for day in run.days if day.silent > 0.0) for run in runs
    }
    silences = {run.household: len(run.silence_alerts) for run in runs}
    moments = [at for run in runs for at in run.silence_alerts]
    alerts, before = _tihm_alerts(runs), _tihm_alerts(off)
    return {
        "monitored_days": len(days),
        "usable_days": sum(day.usable for day in days),
        "evaluable_days": sum(day.evaluable for day in days),
        "days_refused_because_of_the_rule": {
            "all": int(sum(refused.values())),
            "homes_with_one": sum(1 for count in refused.values() if count),
            "per_home": refused,
        },
        "behavioural_alerts": {
            "all": int(sum(by_day.values())),
            "by_feature": dict(sorted(by_feature.items())),
            "by_day_summarised": {
                day.isoformat(): n for day, n in sorted(by_day.items())
            },
            "by_date_raised": {
                day.isoformat(): n for day, n in sorted(by_date.items())
            },
            "homes_with_one": sum(
                1 for run in runs if any(day.alerts for day in run.days)
            ),
            "also_raised_with_the_rule_off": len(alerts & before),
            "raised_only_with_the_rule_off": len(before - alerts),
            "raised_only_under_this_condition": len(alerts - before),
        },
        "silence_alerts": {
            "all": len(moments),
            "homes_with_one": sum(1 for count in silences.values() if count),
            "per_home": silences,
            "per_monitored_day": len(moments) / len(days) if days else None,
            "inside_a_stretch_of_common_silence": sum(
                1 for at in moments if any(s.covers(at) for s in stretches)
            ),
        },
    }


def describe(
    runs: Mapping[str, Sequence[HouseholdRun]],
    protocol: SilentHomeProtocol,
    *,
    published: Mapping[str, int] = TIHM_PUBLISHED,
    published_homes: Mapping[str, Mapping[str, Any]] | None = None,
    zone: tzinfo = TIHM_ZONE,
) -> dict[str, Any]:
    """What the rule does on TIHM, under each condition. A description.

    Nothing is reported unless the run with the rule off gives the monitored
    days and the behavioural alerts of the *published* record: a run that
    does not reproduce it is not the run the rule is being added to. With
    *published_homes*, the published record's table of households, every
    home's two counts are compared as well as their totals.
    """
    off = runs[OFF]
    monitored = sum(len(run.days) for run in off)
    alerts = sum(len(day.alerts) for run in off for day in run.days)
    if (monitored, alerts) != (
        published["monitored_days"],
        published["behavioural_alerts"],
    ):
        raise ValueError(
            f"with the rule off the run gives {monitored:,} monitored days and "
            f"{alerts} behavioural alerts, not the published record's "
            f"{published['monitored_days']:,} and "
            f"{published['behavioural_alerts']}; nothing is reported"
        )
    if published_homes is not None:
        here = {
            run.household: (len(run.days), sum(len(day.alerts) for day in run.days))
            for run in off
        }
        there = {
            name: (int(row["monitored_days"]), int(row["behavioural_alerts"]))
            for name, row in published_homes.items()
        }
        if here != there:
            differing = sorted(
                name
                for name in set(here) | set(there)
                if here.get(name) != there.get(name)
            )
            raise ValueError(
                "with the rule off the run does not give the published record's "
                f"monitored days and behavioural alerts in {len(differing)} "
                f"home(s), the first being {differing[0]}; nothing is reported"
            )
    observed = {run.household: list(run.observed) for run in off}
    stretches, fleet = _fleet_arm(observed, protocol)
    return {
        "result_schema": TIHM_SCHEMA,
        "homes": len(off),
        "check": {
            "monitored_days": monitored,
            "behavioural_alerts": alerts,
            "published": dict(published),
            "homes_compared_one_by_one": (
                len(published_homes) if published_homes is not None else 0
            ),
        },
        "fleet": {"config": fleet_config(protocol).to_dict(), **fleet},
        "conditions": {
            condition: _tihm_condition(runs[condition], off, stretches, zone)
            for condition in protocol.conditions
        },
    }


def describe_tihm(
    adapter: DatasetAdapter,
    protocol: SilentHomeProtocol,
    tihm: TihmProtocol,
    *,
    protocol_sha256: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
    published: Mapping[str, int] = TIHM_PUBLISHED,
    published_homes: Mapping[str, Mapping[str, Any]] | None = None,
) -> SilentHomeResult:
    """Run the rule on the TIHM homes and write the record of the description."""
    runs = run_tihm_homes(adapter, protocol, tihm, jobs=jobs, progress=progress)
    results = describe(
        runs, protocol, published=published, published_homes=published_homes
    )
    record = ExperimentRecord(
        experiment=TIHM_EXPERIMENT,
        configuration={
            "silent_home_protocol": protocol.to_dict()["tihm"],
            "the_rule": protocol.to_dict()["the_rule"],
            "fleet": protocol.to_dict()["fleet"],
            "protocol_sha256": protocol.sha256(),
            "protocol_file_sha256": protocol_sha256,
            "tihm_protocol_sha256": tihm.sha256(),
            "result_schema": TIHM_SCHEMA,
        },
        inference=ONLINE,
        evidence=EvidenceSummary.online(
            moment for group in runs.values() for run in group for moment in run.closes
        ),
        seeds=[],
        results=results,
        data_source="tihm",
        metric_definitions={
            "behavioural_alert": "an alert of kind behavioural_change. It is "
            "counted on the day whose summary raised it, which is the "
            "alert-burden protocol's alert day, and also on the local date it "
            "was raised on, which is the day after",
            "silence_alert": protocol.to_dict()["definitions"]["silence_alert"],
            "day_refused_because_of_the_rule": "a monitored day part of which "
            "fell in a silence of the whole home, which the baseline is "
            "therefore not offered",
        },
        notes=[
            "A description, not a test. The rule was designed from these homes, "
            "so what it does to them is not evidence that it works.",
            "No relation to the dataset's labels is reported.",
            "Nothing was fitted on TIHM. Apart from the rule, the run is the "
            "alert-burden protocol's.",
        ],
        inputs=list(inputs),
        preprocessing={
            "contract": "external-dataset contract, strict conversion",
            "mapping_sha256": tihm.mapping.sha256(),
            "step_minutes": tihm.step_minutes,
        },
        models=[
            ModelRecord(
                f"online_pipeline_{condition}",
                {
                    "home_silence_horizon_hours": (
                        None
                        if protocol.horizon(condition) is None
                        else protocol.horizon(condition) / timedelta(hours=1)  # type: ignore[operator]
                    )
                },
                "BehaviouralSensingPipeline",
            )
            for condition in protocol.conditions
        ],
    )
    path = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = record.write(Path(output_dir) / f"{TIHM_EXPERIMENT}.json")
    return SilentHomeResult(record, path)
