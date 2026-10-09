"""The matched-sensor study: each home with the simulator's sensors and with TIHM's.

This runs the protocol of :mod:`.matched_sensors_protocol`. :func:`run_home`
simulates both arms of one of the threshold-calibration study's homes and runs
each arm through :class:`~sensor_modeling.online.pipeline.BehaviouralSensingPipeline`
under every profile the home has: the simulator's own event record, and the
event record drawn again from the same plan under the matched profile, and on
the first homes under the sensitivity profile too. :func:`score` computes the
estimands and decides the criteria. Every interval resamples homes. The
evidence is simulated, and every result is a statement about the simulator.
"""

from __future__ import annotations

import bisect
import json
import pickle
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ..alerts.alert import AlertKind, AlertPolicy
from ..evaluation import ExperimentRecord, InputArtifact, ReportedInterval
from ..evaluation.resampling import percentile_interval, resample_indices
from ..fusion import ONLINE, EvidenceSummary
from ..observations.observation import Observation
from ..observations.registry import SensorRegistry
from ..observations.types import Modality
from ..online.pipeline import BehaviouralSensingPipeline, PipelineConfig, PipelineStep
from ..simulation.faults import DegradationConfig, degrade
from ..simulation.household import HouseholdConfig, simulate
from ..simulation.sensor_profile import profile_observations, profile_registry
from ..states.ontology import BehaviouralState
from .matched_sensors_planning import cohort_moments, home_moments, simulated_records
from .matched_sensors_protocol import (
    ARMS,
    C1,
    C2,
    C3,
    DOES_NOT_FOLLOW,
    DOES_NOT_SURVIVE,
    EXPERIMENT,
    FEATURES,
    FOLLOWS,
    INCONCLUSIVE,
    MATCHED,
    NOT_REPRODUCED,
    PROFILES,
    PUBLISHED_CONDITION,
    REPRODUCED,
    RESULT_SCHEMA,
    SENSITIVITY,
    STANDARD,
    STATE_HOURS,
    SURVIVES,
    TRACKED_FEATURE,
    MatchedSensorsProtocol,
    between_the_freeze_and_the_run,
)
from .silent_home_experiment import wilson
from .silent_home_protocol import event_sensors
from .threshold_calibration_experiment import (
    DIRECTION_LETTERS,
    SUBJECT_LETTERS,
    VERDICT_LETTERS,
)
from .threshold_calibration_protocol import CHANGE, STABLE

DAY = timedelta(days=1)
DECREASE = "decrease"

#: The motion sensors whose sole activations E8 reads the belief after.
SOLE_ROOMS = {"kitchen": "kitchen_motion", "bathroom": "bathroom_motion"}

#: Why a change verdict raised no alert.
NOT_ATTRIBUTABLE = "not_attributable_enough"
OTHERWISE = "graded_too_low_or_withheld"


def state_of(feature: str) -> str:
    """The state a feature sums: ``sleeping_hours`` is ``sleeping``."""
    return feature.removesuffix("_hours")


# ----------------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RunRecord:
    """What one run of one arm of one home under one profile contributes.

    Attributes
    ----------
    behavioural
        Each behavioural alert: the moment it was raised at, its subject, the
        verdict behind it, its direction and the place in the record of the
        day whose close raised it.
    closes
        The moments a day was closed at.
    usable
        The days the baseline was offered.
    days
        The matched days of the stable arm, in order; empty for the changed arm.
    hours, truth
        For each state, the pipeline's value and the truth on each matched day.
    moments
        The planning's moments of the record; empty for the changed arm.
    verdicts
        Change verdicts, those that raised an alert, those held back and why,
        and the notices of a burst.
    attribution
        The resident's ambient attribution at each day's close.
    beliefs
        For each room of :data:`SOLE_ROOMS`: how many steps had a window with
        motion activations of that room's sensor alone, and the sum of the
        belief vectors at their ends, in the ontology's order of states.
    states
        The ontology's states, in the order of the belief vectors.
    """

    behavioural: tuple[tuple[datetime, str, str, str, int], ...]
    closes: tuple[datetime, ...]
    usable: int
    days: tuple[date, ...] = ()
    hours: Mapping[str, tuple[float, ...]] | None = None
    truth: Mapping[str, tuple[float, ...]] | None = None
    moments: Mapping[str, float | None] | None = None
    verdicts: Mapping[str, int] | None = None
    attribution: tuple[float, ...] = ()
    beliefs: Mapping[str, tuple[int, tuple[float, ...]]] | None = None
    states: tuple[str, ...] = ()

    def encoded(self) -> str:
        """The alerts as the published record writes them: ``57sG-``."""
        return " ".join(
            f"{place}"
            f"{SUBJECT_LETTERS.get(subject, '?')}"
            f"{VERDICT_LETTERS.get(verdict, '?')}"
            f"{DIRECTION_LETTERS.get(direction, '?')}"
            for _, subject, verdict, direction, place in self.behavioural
        )

    def first(
        self,
        feature: str,
        begin: datetime,
        end: datetime,
        direction: str | None = None,
    ) -> datetime | None:
        """The moment of the first alert about *feature* in the window."""
        for at, subject, _, way, _ in self.behavioural:
            if subject != feature or not begin <= at < end:
                continue
            if direction is None or way == direction:
                return at
        return None


@dataclass(frozen=True)
class HomeRun:
    """One home: its seed and its runs, by profile and arm."""

    seed: int
    runs: Mapping[str, Mapping[str, RunRecord]]


def _alerts(
    steps: Sequence[PipelineStep], start: date
) -> tuple[tuple[datetime, str, str, str, int], ...]:
    found: list[tuple[datetime, str, str, str, int]] = []
    for step in steps:
        behavioural = [a for a in step.alerts if a.kind is AlertKind.BEHAVIOURAL_CHANGE]
        if step.day_closed is None:
            if behavioural:
                raise ValueError("a behavioural alert was raised with no day closed")
            continue
        place = (step.day_closed.day - start).days
        by_feature = {change.feature: change for change in step.changes}
        for alert in behavioural:
            subject = str(alert.subject)
            verdict = by_feature.get(subject)
            found.append(
                (
                    alert.at,
                    subject,
                    verdict.kind.value if verdict is not None else "",
                    str(verdict.direction) if verdict is not None else "",
                    place,
                )
            )
    return tuple(found)


def _verdicts(
    steps: Sequence[PipelineStep],
) -> tuple[dict[str, int], tuple[float, ...]]:
    """What became of the change verdicts, and the attribution at each close."""
    policy = AlertPolicy()
    counts: Counter[str] = Counter()
    attribution: list[float] = []
    for step in steps:
        alerted = {
            str(a.subject)
            for a in step.alerts
            if a.kind is AlertKind.BEHAVIOURAL_CHANGE
        }
        counts["bursts"] += sum(
            1
            for a in step.alerts
            if a.kind is AlertKind.DATA_QUALITY and a.subject == "alert_rate"
        )
        if step.day_closed is None:
            continue
        share = float(step.context.ambient_attribution())
        attribution.append(share)
        confidence = float(step.day_closed.coverage) * share
        for change in step.changes:
            if not change.is_change:
                continue
            counts["change_verdicts"] += 1
            if change.feature in alerted:
                counts["raised"] += 1
            elif confidence < policy.min_confidence:
                counts[NOT_ATTRIBUTABLE] += 1
            else:
                counts[OTHERWISE] += 1
    return dict(counts), tuple(attribution)


def _beliefs(
    steps: Sequence[PipelineStep], delivered: Sequence[Observation]
) -> dict[str, tuple[int, tuple[float, ...]]]:
    """The belief at the end of the steps whose window held one room's sensor alone."""
    motion = sorted(
        (o.timestamp, o.sensor_id) for o in delivered if o.modality is Modality.MOTION
    )
    moments = [m for m, _ in motion]
    size = len(steps[0].state.belief) if steps else 0
    sums = {room: np.zeros(size) for room in SOLE_ROOMS}
    counts = dict.fromkeys(SOLE_ROOMS, 0)
    for before, step in zip(steps, steps[1:]):
        low = bisect.bisect_right(moments, before.at)
        high = bisect.bisect_right(moments, step.at)
        seen = {sensor for _, sensor in motion[low:high]}
        for room, sensor in SOLE_ROOMS.items():
            if seen == {sensor}:
                counts[room] += 1
                sums[room] += np.asarray(step.state.belief, dtype=float)
    return {
        room: (counts[room], tuple(float(v) for v in sums[room])) for room in SOLE_ROOMS
    }


def run_once(
    registry: SensorRegistry,
    observations: Sequence[Observation],
    config: PipelineConfig,
    end: datetime,
) -> list[PipelineStep]:
    """One pass of the pipeline over a delivered record."""
    pipeline = BehaviouralSensingPipeline(registry, config=config)
    steps = pipeline.run(observations)
    steps.extend(pipeline.close(end))
    return steps


def profiles_of(seed: int, protocol: MatchedSensorsProtocol) -> tuple[str, ...]:
    """The profiles a home is run under."""
    if seed in protocol.sensitivity_seeds():
        return (*PROFILES, SENSITIVITY)
    return PROFILES


def run_home(
    seed: int,
    protocol: MatchedSensorsProtocol,
    days: int | None = None,
    profiles: Sequence[str] | None = None,
) -> HomeRun:
    """Simulate both arms of a home and run each under its profiles.

    *days* shortens the records and *profiles* chooses them, for a test; the
    study passes neither.
    """
    study = protocol.study
    length = study.days if days is None else days
    chosen = tuple(profiles_of(seed, protocol) if profiles is None else profiles)
    sensors = event_sensors()
    start = study.start()
    first, last = start, start + timedelta(days=length - 1)
    runs: dict[str, dict[str, RunRecord]] = {profile: {} for profile in chosen}
    for arm in ARMS:
        result = simulate(
            HouseholdConfig(days=length, seed=seed, shift=study.shift(arm))
        )
        base = result.registry.subset(sensors)
        config = PipelineConfig(
            tz=result.config.tz, step=timedelta(minutes=study.step_minutes)
        )
        states = {state.value: state for state in BehaviouralState}
        for profile in chosen:
            if profile == STANDARD:
                registry = base
                record: Sequence[Observation] = result.observations_for(sensors)
            else:
                settings = protocol.profile(profile)
                registry = profile_registry(base, settings)
                record = profile_observations(
                    result.truth, seed, settings, stream=protocol.stream(arm)
                )
            delivered, _ = degrade(record, DegradationConfig())
            steps = run_once(registry, delivered, config, result.end)
            alerts = _alerts(steps, start)
            closes = tuple(s.at for s in steps if s.day_closed is not None)
            summaries = [s.day_closed for s in steps if s.day_closed is not None]
            usable = [
                s
                for s in summaries
                if s.is_usable(config.min_day_coverage, config.min_day_observed)
            ]
            if arm != STABLE:
                runs[profile][arm] = RunRecord(alerts, closes, len(usable))
                continue
            kept = [s for s in usable if first < s.day < last]
            hours: dict[str, tuple[float, ...]] = {}
            truth: dict[str, tuple[float, ...]] = {}
            for name in STATE_HOURS:
                daily = result.truth.daily_hours(states[name])
                hours[name] = tuple(float(s.hours_in(states[name])) for s in kept)
                truth[name] = tuple(float(daily.get(s.day, 0.0)) for s in kept)
            verdicts, attribution = _verdicts(steps)
            runs[profile][arm] = RunRecord(
                behavioural=alerts,
                closes=closes,
                usable=len(usable),
                days=tuple(s.day for s in kept),
                hours=hours,
                truth=truth,
                moments=home_moments(simulated_records(delivered), result.config.tz),
                verdicts=verdicts,
                attribution=attribution,
                beliefs=_beliefs(steps, delivered),
                states=(
                    tuple(s.value for s in steps[0].state.ontology.states)
                    if steps
                    else ()
                ),
            )
    return HomeRun(seed=seed, runs=runs)


def _checkpoint(directory: Path, seed: int) -> Path:
    return directory / f"home-{seed}.pickle"


def _run_home(arguments: tuple[int, MatchedSensorsProtocol, Path | None]) -> HomeRun:
    """Run one home, or read it back if this run has already finished it."""
    seed, protocol, directory = arguments
    if directory is None:
        return run_home(seed, protocol)
    path = _checkpoint(directory, seed)
    if path.exists():
        with path.open("rb") as handle:
            kept: HomeRun = pickle.load(handle)  # noqa: S301 - written below
        return kept
    home = run_home(seed, protocol)
    partial = path.with_suffix(".partial")
    with partial.open("wb") as handle:
        pickle.dump(home, handle)
    partial.replace(path)
    return home


def run_homes(
    protocol: MatchedSensorsProtocol,
    seeds: Sequence[int] | None = None,
    *,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
    checkpoint: Path | None = None,
) -> list[HomeRun]:
    """Run every home of the study, in the order of its seeds.

    Each home is a function of its seed and the protocol alone, so the result
    does not depend on *jobs*. With *checkpoint*, each home is written to a
    directory named for the protocol's digest inside it when it is finished,
    and read back from there if the run is started again.
    """
    chosen = tuple(protocol.study.study_seeds() if seeds is None else seeds)
    directory = None
    if checkpoint is not None:
        directory = Path(checkpoint) / protocol.sha256()[:16]
        directory.mkdir(parents=True, exist_ok=True)
    tasks = [(seed, protocol, directory) for seed in chosen]
    homes: list[HomeRun] = []
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
# The check against the published record
# ----------------------------------------------------------------------------
def published_alerts(record: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Each home's published alerts under the default, by arm."""
    alerts = record["results"]["raw"]["alerts"]
    return {
        seed: {arm: arms[arm][PUBLISHED_CONDITION] for arm in ARMS}
        for seed, arms in alerts.items()
    }


def check(
    homes: Sequence[HomeRun], published: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    """Whether every standard run raised the published alerts, home by home."""
    differ = [
        f"{home.seed}/{arm}"
        for home in homes
        for arm in ARMS
        if home.runs[STANDARD][arm].encoded()
        != published.get(str(home.seed), {}).get(arm)
    ]
    return {
        "homes": len(homes),
        "runs_compared": len(homes) * len(ARMS),
        "runs_that_differ": differ,
        "reproduced": not differ,
    }


# ----------------------------------------------------------------------------
# Statistics over homes
# ----------------------------------------------------------------------------
def mean_estimate(
    values: Sequence[float] | np.ndarray, protocol: MatchedSensorsProtocol
) -> dict[str, Any]:
    """A mean over homes and its home-bootstrap interval."""
    data = np.asarray(values, dtype=float)
    result: dict[str, Any] = {
        "estimate": float(data.mean()) if data.size else None,
        "interval": None,
        "homes": int(data.size),
    }
    if data.size < 2:
        return result
    index = resample_indices(data.size, protocol.resamples, protocol.seed)
    result["interval"] = percentile_interval(
        data[index].mean(axis=1), protocol.confidence
    ).to_dict()
    return result


def ratio_estimate(
    numerator: Sequence[float],
    denominator: Sequence[float],
    protocol: MatchedSensorsProtocol,
) -> dict[str, Any]:
    """A ratio of sums over homes, with its home-bootstrap interval."""
    top = np.asarray(numerator, dtype=float)
    bottom = np.asarray(denominator, dtype=float)
    total = float(bottom.sum())
    result: dict[str, Any] = {
        "estimate": float(top.sum()) / total if total > 0 else None,
        "numerator": float(top.sum()),
        "denominator": total,
        "interval": None,
        "homes": int(top.size),
    }
    if top.size < 2 or total <= 0:
        return result
    index = resample_indices(top.size, protocol.resamples, protocol.seed)
    result["interval"] = percentile_interval(
        top[index].sum(axis=1) / bottom[index].sum(axis=1), protocol.confidence
    ).to_dict()
    return result


def pooled_median_estimate(
    homes: Sequence[Sequence[float]], protocol: MatchedSensorsProtocol
) -> dict[str, Any]:
    """The median of every day of every home, with a home-bootstrap interval."""
    pooled = [value for home in homes for value in home]
    result: dict[str, Any] = {
        "estimate": float(np.median(pooled)) if pooled else None,
        "interval": None,
        "homes": len(homes),
        "days": len(pooled),
    }
    if len(homes) < 2 or not pooled:
        return result
    arrays = [np.asarray(home, dtype=float) for home in homes]
    index = resample_indices(len(arrays), protocol.resamples, protocol.seed)
    medians = []
    for row in index:
        drawn = np.concatenate([arrays[k] for k in row])
        if drawn.size:
            medians.append(float(np.median(drawn)))
    result["interval"] = percentile_interval(
        np.asarray(medians), protocol.confidence
    ).to_dict()
    return result


def share_estimate(
    flags: Sequence[bool], protocol: MatchedSensorsProtocol
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
    delays: Sequence[float | None], protocol: MatchedSensorsProtocol
) -> dict[str, Any]:
    """The pooled median delay over the homes that detected, resampling homes."""
    data = np.array([np.nan if d is None else d for d in delays], dtype=float)
    found = data[~np.isnan(data)]
    result: dict[str, Any] = {
        "estimate": float(np.median(found)) if found.size else None,
        "interval": None,
        "detections": int(found.size),
        "homes": int(data.size),
    }
    if data.size < 2 or not found.size:
        return result
    index = resample_indices(data.size, protocol.resamples, protocol.seed)
    resampled = data[index]
    defined = ~np.isnan(resampled).all(axis=1)
    if defined.any():
        result["interval"] = percentile_interval(
            np.nanmedian(resampled[defined], axis=1), protocol.confidence
        ).to_dict()
    return result


def spearman(x: Sequence[float], y: Sequence[float], tolerance: float = 1e-9) -> float:
    """The rank correlation; 0 where either side spans no more than *tolerance*."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.size < 3 or np.ptp(a) <= tolerance or np.ptp(b) <= tolerance:
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


def _low(estimate: Mapping[str, Any]) -> float:
    interval = estimate.get("interval")
    return float(interval["low"]) if interval else float("nan")


def _high(estimate: Mapping[str, Any]) -> float:
    interval = estimate.get("interval")
    return float(interval["high"]) if interval else float("nan")


def survival_reading(estimate: Mapping[str, Any], margin: float) -> str:
    """C1: survives, does not survive or inconclusive."""
    if _low(estimate) > -margin:
        return SURVIVES
    if _high(estimate) < -margin:
        return DOES_NOT_SURVIVE
    return INCONCLUSIVE


def tracking_reading(estimate: Mapping[str, Any], margin: float) -> str:
    """C2: follows, does not follow or inconclusive."""
    if _low(estimate) > margin:
        return FOLLOWS
    if _high(estimate) < margin:
        return DOES_NOT_FOLLOW
    return INCONCLUSIVE


def collapse_reading(
    kitchen: float | None, bathroom: float | None, protocol: MatchedSensorsProtocol
) -> str:
    """C3: reproduced or not reproduced."""
    if kitchen is None or bathroom is None:
        return NOT_REPRODUCED
    if (
        kitchen < protocol.kitchen_ceiling_hours
        and bathroom < protocol.bathroom_ceiling_hours
    ):
        return REPRODUCED
    return NOT_REPRODUCED


# ----------------------------------------------------------------------------
# What each home shows
# ----------------------------------------------------------------------------
def detected(
    homes: Sequence[HomeRun],
    profile: str,
    arm: str,
    protocol: MatchedSensorsProtocol,
    direction: str | None = None,
) -> np.ndarray:
    """For each home, whether an alert about the tracked feature falls in the window."""
    begin, end = protocol.study.detection_window(CHANGE)
    return np.array(
        [
            home.runs[profile][arm].first(TRACKED_FEATURE, begin, end, direction)
            is not None
            for home in homes
        ]
    )


def excess(
    homes: Sequence[HomeRun],
    profile: str,
    protocol: MatchedSensorsProtocol,
    direction: str | None = None,
) -> np.ndarray:
    """Each home's detection minus its false detection."""
    found = detected(homes, profile, CHANGE, protocol, direction).astype(float)
    false = detected(homes, profile, STABLE, protocol, direction).astype(float)
    return np.asarray(found - false, dtype=float)


def delays(
    homes: Sequence[HomeRun], profile: str, protocol: MatchedSensorsProtocol
) -> list[float | None]:
    """Each home's delay in days from the start of the first changed day."""
    begin, end = protocol.study.detection_window(CHANGE)
    origin = protocol.study.begins(CHANGE)
    found: list[float | None] = []
    for home in homes:
        at = home.runs[profile][CHANGE].first(TRACKED_FEATURE, begin, end)
        found.append(None if at is None else (at - origin) / DAY)
    return found


def correlations(
    homes: Sequence[HomeRun],
    profile: str,
    feature: str,
    protocol: MatchedSensorsProtocol,
) -> list[float | None]:
    """Each home's correlation; ``None`` for a home with too few matched days."""
    state = state_of(feature)
    rows: list[float | None] = []
    for home in homes:
        run = home.runs[profile][STABLE]
        if len(run.days) < protocol.min_matched_days or run.hours is None:
            rows.append(None)
            continue
        assert run.truth is not None
        rows.append(
            spearman(run.hours[state], run.truth[state], protocol.constant_tolerance)
        )
    return rows


# ----------------------------------------------------------------------------
# The estimands
# ----------------------------------------------------------------------------
def _detection(
    homes: Sequence[HomeRun], profile: str, protocol: MatchedSensorsProtocol
) -> dict[str, Any]:
    def shares(direction: str | None) -> dict[str, Any]:
        return {
            "detected": share_estimate(
                list(detected(homes, profile, CHANGE, protocol, direction)), protocol
            ),
            "false_detections": share_estimate(
                list(detected(homes, profile, STABLE, protocol, direction)), protocol
            ),
            "excess_detection": mean_estimate(
                excess(homes, profile, protocol, direction), protocol
            ),
        }

    return {
        **shares(None),
        "delay_days": median_delay(delays(homes, profile, protocol), protocol),
        "decrease_only": shares(DECREASE),
    }


def _difference(
    homes: Sequence[HomeRun], profile: str, protocol: MatchedSensorsProtocol
) -> dict[str, Any]:
    """A profile's detection against the standard profile's, home by home."""
    return {
        "excess_detection": mean_estimate(
            excess(homes, profile, protocol) - excess(homes, STANDARD, protocol),
            protocol,
        ),
        **{
            name: mean_estimate(
                detected(homes, profile, arm, protocol).astype(float)
                - detected(homes, STANDARD, arm, protocol).astype(float),
                protocol,
            )
            for name, arm in (("detected", CHANGE), ("false_detections", STABLE))
        },
    }


def _alerts_by_feature(
    homes: Sequence[HomeRun], profile: str, protocol: MatchedSensorsProtocol
) -> dict[str, Any]:
    days = protocol.study.days
    counts = [
        Counter(
            subject for _, subject, _, _, _ in home.runs[profile][STABLE].behavioural
        )
        for home in homes
    ]
    features = sorted({feature for row in counts for feature in row})
    return {
        "false_alerts": mean_estimate([sum(row.values()) for row in counts], protocol),
        "person_days": days * len(homes),
        "per_person_day": {
            feature: ratio_estimate(
                [row[feature] for row in counts], [days] * len(homes), protocol
            )
            for feature in features
        },
        "by_feature": {
            feature: int(sum(row[feature] for row in counts)) for feature in features
        },
    }


def _tracking(
    homes: Sequence[HomeRun], protocol: MatchedSensorsProtocol, profiles: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for feature in FEATURES:
        rows = {p: correlations(homes, p, feature, protocol) for p in profiles}
        entry: dict[str, Any] = {
            p: mean_estimate([r for r in rows[p] if r is not None], protocol)
            for p in profiles
        }
        for p in profiles:
            if p == STANDARD:
                continue
            both = [
                (m, s)
                for m, s in zip(rows[p], rows[STANDARD])
                if m is not None and s is not None
            ]
            entry[f"{p}_minus_standard"] = mean_estimate(
                [m - s for m, s in both], protocol
            )
        result[feature] = entry
    return result


def _level(
    homes: Sequence[HomeRun], protocol: MatchedSensorsProtocol
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in STATE_HOURS:
        result[name] = {}
        for profile in PROFILES:
            runs = [home.runs[profile][STABLE] for home in homes]
            pipeline = [list((run.hours or {}).get(name, ())) for run in runs]
            truth = [v for run in runs for v in (run.truth or {}).get(name, ())]
            pooled = [v for home in pipeline for v in home]
            entry: dict[str, Any] = {
                "pooled_day_median_pipeline": (
                    float(np.median(pooled)) if pooled else None
                ),
                "pooled_day_median_truth": float(np.median(truth)) if truth else None,
                "days": len(pooled),
            }
            if f"{name}_hours" in FEATURES:
                entry["mean_difference"] = mean_estimate(
                    [
                        float(
                            np.mean(
                                np.subtract(
                                    (run.hours or {})[name], (run.truth or {})[name]
                                )
                            )
                        )
                        for run in runs
                        if run.days
                    ],
                    protocol,
                )
            if name in ("kitchen_activity", "bathroom_activity"):
                entry["pooled_day_median_interval"] = pooled_median_estimate(
                    pipeline, protocol
                )
            result[name][profile] = entry
    return result


def _withheld(
    homes: Sequence[HomeRun], protocol: MatchedSensorsProtocol
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for profile in PROFILES:
        runs = [home.runs[profile][STABLE] for home in homes]
        totals: Counter[str] = Counter()
        for run in runs:
            totals.update(run.verdicts or {})
        attribution = [
            float(np.mean(run.attribution)) for run in runs if run.attribution
        ]
        result[profile] = {
            "change_verdicts": totals["change_verdicts"],
            "raised_an_alert": totals["raised"],
            "raised_none": {
                NOT_ATTRIBUTABLE: totals[NOT_ATTRIBUTABLE],
                OTHERWISE: totals[OTHERWISE],
            },
            "notices_of_a_burst": totals["bursts"],
            "attribution_at_close": mean_estimate(attribution, protocol),
        }
    return result


def _sole_beliefs(homes: Sequence[HomeRun]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for profile in PROFILES:
        runs = [home.runs[profile][STABLE] for home in homes]
        states = next((run.states for run in runs if run.states), ())
        entry: dict[str, Any] = {}
        for room in SOLE_ROOMS:
            count = 0
            total = np.zeros(len(states))
            for run in runs:
                found = (run.beliefs or {}).get(room)
                if found is None or not found[0]:
                    continue
                count += found[0]
                total += np.asarray(found[1], dtype=float)
            entry[room] = {
                "steps": count,
                "mean_belief": {
                    state: float(total[k] / count) if count else None
                    for k, state in enumerate(states)
                },
            }
        result[profile] = entry
    return result


def _sensitivity(
    homes: Sequence[HomeRun], protocol: MatchedSensorsProtocol
) -> dict[str, Any]:
    chosen = [home for home in homes if SENSITIVITY in home.runs]
    if not chosen:
        return {"homes": 0}
    return {
        "homes": len(chosen),
        "profile": protocol.profile(SENSITIVITY).to_dict(),
        "E1": _difference(chosen, SENSITIVITY, protocol)["excess_detection"],
        "E2": {
            "by_profile": {
                p: _detection(chosen, p, protocol) for p in (STANDARD, SENSITIVITY)
            },
            "sensitivity_minus_standard": _difference(chosen, SENSITIVITY, protocol),
        },
        "E4": _tracking(chosen, protocol, (STANDARD, SENSITIVITY)),
    }


def score(
    homes: Sequence[HomeRun],
    protocol: MatchedSensorsProtocol,
    planning: Mapping[str, Any],
    published: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """The estimands, the criteria and the check."""
    checked = check(homes, published)
    if not checked["reproduced"]:
        return {"check": checked}
    detection = {profile: _detection(homes, profile, protocol) for profile in PROFILES}
    difference = _difference(homes, MATCHED, protocol)
    e1 = difference["excess_detection"]
    tracking = _tracking(homes, protocol, PROFILES)
    level = _level(homes, protocol)
    moments = {
        profile: cohort_moments(
            home.runs[profile][STABLE].moments or {} for home in homes
        )
        for profile in PROFILES
    }
    kitchen = level["kitchen_activity"][MATCHED]["pooled_day_median_pipeline"]
    bathroom = level["bathroom_activity"][MATCHED]["pooled_day_median_pipeline"]
    criteria = {
        C1: {
            "estimand": "E1",
            "estimate": e1,
            "margin": protocol.survival_margin,
            "reading": survival_reading(e1, protocol.survival_margin),
        },
        C2: {
            "estimand": "E4",
            "estimate": tracking[TRACKED_FEATURE][MATCHED],
            "margin": protocol.tracking_margin,
            "reading": tracking_reading(
                tracking[TRACKED_FEATURE][MATCHED], protocol.tracking_margin
            ),
        },
        C3: {
            "estimand": "E5",
            "kitchen_pooled_day_median": kitchen,
            "bathroom_pooled_day_median": bathroom,
            "ceilings": [
                protocol.kitchen_ceiling_hours,
                protocol.bathroom_ceiling_hours,
            ],
            "reading": collapse_reading(kitchen, bathroom, protocol),
        },
    }
    return {
        "check": checked,
        "criteria": criteria,
        "E1": e1,
        "E2": {
            "by_profile": detection,
            "matched_minus_standard": {
                k: v for k, v in difference.items() if k != "excess_detection"
            },
        },
        "E3": {
            profile: _alerts_by_feature(homes, profile, protocol)
            for profile in PROFILES
        },
        "E4": tracking,
        "E5": level,
        "E6": {**moments, "tihm": planning["results"]["tihm"]["moments"]},
        "E7": _withheld(homes, protocol),
        "E8": _sole_beliefs(homes),
        "E9": _sensitivity(homes, protocol),
        "homes": len(homes),
    }


def per_home(
    homes: Sequence[HomeRun], protocol: MatchedSensorsProtocol
) -> dict[str, Any]:
    """Each home's alerts, detections and correlations, by profile."""
    rows: dict[str, Any] = {}
    for home in homes:
        row: dict[str, Any] = {}
        for profile, arms in home.runs.items():
            stable = arms[STABLE]
            begin, end = protocol.study.detection_window(CHANGE)
            row[profile] = {
                "alerts": {arm: arms[arm].encoded() for arm in ARMS},
                "detected": arms[CHANGE].first(TRACKED_FEATURE, begin, end) is not None,
                "false_detection": stable.first(TRACKED_FEATURE, begin, end)
                is not None,
                "matched_days": len(stable.days),
                "spearman": {
                    feature: (
                        spearman(
                            (stable.hours or {})[state_of(feature)],
                            (stable.truth or {})[state_of(feature)],
                            protocol.constant_tolerance,
                        )
                        if len(stable.days) >= protocol.min_matched_days
                        else None
                    )
                    for feature in FEATURES
                },
            }
        rows[str(home.seed)] = row
    return rows


def _interval(
    label: str, estimate: Mapping[str, Any], protocol: MatchedSensorsProtocol
) -> ReportedInterval | None:
    interval = estimate.get("interval")
    if not interval or estimate.get("estimate") is None:
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
    results: Mapping[str, Any], protocol: MatchedSensorsProtocol
) -> list[ReportedInterval]:
    wanted = [("E1: excess detection, matched minus standard", results["E1"])]
    for profile in PROFILES:
        for feature in FEATURES:
            wanted.append(
                (
                    f"E4: within-home correlation of {feature} with the truth, "
                    f"{profile}",
                    results["E4"][feature][profile],
                )
            )
    found = [_interval(label, estimate, protocol) for label, estimate in wanted]
    return [interval for interval in found if interval is not None]


def _closes(homes: Iterable[HomeRun]) -> Iterable[datetime]:
    for home in homes:
        for arms in home.runs.values():
            for run in arms.values():
                yield from run.closes


def record_of(
    homes: Sequence[HomeRun],
    protocol: MatchedSensorsProtocol,
    *,
    planning: Mapping[str, Any],
    published: Mapping[str, Mapping[str, str]],
    protocol_sha256: str,
    inputs: Sequence[InputArtifact] = (),
    code_changes: Mapping[str, Sequence[str]] | None = None,
    clean: bool = True,
) -> ExperimentRecord:
    """Score the homes and build the record.

    When the check fails the record holds the check and nothing else: no
    estimand, criterion or home's outcome.
    """
    results = score(homes, protocol, planning, published)
    reproduced = bool(results["check"]["reproduced"])
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
        },
        inference=ONLINE,
        evidence=EvidenceSummary.online(_closes(homes)),
        seeds=[protocol.study.seed_root, protocol.seed, *[h.seed for h in homes]],
        results=results,
        sensor_subset=list(event_sensors()),
        data_source="simulator",
        notes=[
            "Pre-specified: the protocol was frozen before any home of the study "
            "was run with the matched profile.",
            *(
                []
                if clean
                else [
                    "The run was made from a modified working tree, so the record "
                    "cannot be traced to one commit."
                ]
            ),
            f"{len(homes)} simulated homes, each run in both arms under the "
            "standard and the matched profiles, and the first "
            f"{protocol.sensitivity_homes} under the sensitivity profile as well. "
            "The profiles of a home and arm share the plan.",
            (
                "The standard profile's runs raised, home by home and arm by arm, "
                "the alerts the published threshold-calibration record gives under "
                "the default reference."
                if reproduced
                else "The standard profile's runs did not reproduce the published "
                "record, so nothing is reported."
            ),
            *(text[0].upper() + text[1:] + "." for text in between),
            "Nothing was fitted. The pipeline's emissions and alert policy are the "
            "declared defaults.",
            "The residents' days come from distributions this project wrote down, "
            "and the matched profile is matched to TIHM on eight moments of the "
            "sensor records. The result says nothing about how the pipeline does "
            "on a real home.",
        ],
        inputs=list(inputs),
        preprocessing={
            "sensors": {
                STANDARD: list(event_sensors()),
                MATCHED: [*event_sensors(), "hall_motion"],
            },
            "delivery": "simulation.faults.degrade, with no loss, lateness, "
            "duplication or fault",
            "step_minutes": protocol.study.step_minutes,
        },
        household_metrics=(
            {"simulated_homes": per_home(homes, protocol)} if reproduced else {}
        ),
        intervals=_intervals(results, protocol) if reproduced else [],
    )


def read_json(path: Path) -> dict[str, Any]:
    """A JSON file."""
    loaded: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return loaded
