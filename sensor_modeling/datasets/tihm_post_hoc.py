"""Descriptions of the TIHM run, made after its results had been read.

Nothing here is part of the protocol of :mod:`.tihm_protocol`. That protocol's
run found that what the pipeline flags on the TIHM homes does not relate to
the labels a clinical team verified. This module describes the same run to
say why, and every description in it was chosen with the run's alerts in
view. The descriptions are written to a record of their own, so the protocol's
record stays what the protocol declared.

- **What the filter infers.** The hours each day's summary gives every
  behavioural state.
- **Silent days.** Monitored days on which no sensor of the home reported an
  event, whether the pipeline used them, and what it read into them.
- **The calendar.** For each date, the households monitored, those silent,
  the cohort's sensor events and the behavioural alerts raised.
- **Replays.** The adaptive baseline run again over each household's recorded
  feature values, over the same values with the silent days left out, over
  stationary Gaussian values with each household's own centre and spread, and
  over the recorded days in shuffled order. The first must reproduce the run,
  feature-day by feature-day; the others say how much of what was raised
  appears with no change over time at all.
- **Reference size.** How often a day reaches the deviation threshold, by the
  number of days its reference was estimated from, beside the rate a Gaussian
  value reaches it against a median and MAD of that many others, and against
  a mean and standard deviation of that many others.
- **Direction.** Which feature drives a deviating day and which way, and the
  day's sensor event count, on label days and on other days.

A replay skips a silent day the way the pipeline skips a day it did not
observe. That is a diagnosis of the run, not a corrected result: the rule was
written after the alerts had been read.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ..baseline.adaptive import (
    MAD_TO_SIGMA,
    AdaptiveBaseline,
    BaselineConfig,
    ChangeKind,
)
from ..evaluation import ExperimentRecord, InputArtifact, ModelRecord
from ..external.contract import DatasetAdapter
from ..external.validation import validate_household
from ..fusion import ONLINE, EvidenceSummary
from ..online.pipeline import PipelineConfig
from .tihm_experiment import (
    DayRecord,
    HouseholdRun,
    expanding_z,
    run_household,
    score,
)
from .tihm_protocol import PRIMARY_LABEL, RESULT_SCHEMA, TihmProtocol

POST_HOC_EXPERIMENT = "tihm-alert-burden-post-hoc"
POST_HOC_SCHEMA = "tihm-alert-burden-post-hoc/1"

STATUS = (
    "post hoc: every description here was chosen after the protocol's results "
    "had been read; none is part of the protocol, enters an estimand or "
    "changes a result"
)

#: The verdicts the pipeline can raise a behavioural alert about.
CHANGE_KINDS = (
    ChangeKind.ABRUPT_CHANGE.value,
    ChangeKind.PERSISTENT_CHANGE.value,
    ChangeKind.GRADUAL_DRIFT.value,
)

#: Draws simulated at once when estimating a Gaussian exceedance rate.
_CHUNK = 50_000

#: The relative difference two runs' results may have and still be the same.
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PostHocConfig:
    """What the descriptions simulate.

    Attributes
    ----------
    replicates
        Stationary and shuffled replays of the baseline.
    null_draws
        Gaussian draws behind each reference size's exceedance rate.
    seed
        Seed of every simulation.
    confidence
        The central share of replicates a range covers.
    """

    replicates: int = 200
    null_draws: int = 2_000_000
    seed: int = 0
    confidence: float = 0.95

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.replicates < 1 or self.null_draws < 1:
            raise ValueError("replicates and null_draws must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """A serialisable form of the configuration."""
        return {
            "replicates": self.replicates,
            "null_draws": self.null_draws,
            "seed": self.seed,
            "confidence": self.confidence,
        }


def is_silent(day: DayRecord) -> bool:
    """Whether no sensor of the home reported an event on the day."""
    return day.events == 0


def features() -> tuple[str, ...]:
    """The baseline's features, in the pipeline's order."""
    return tuple(f"{state.value}_hours" for state in PipelineConfig().features)


def _judged(day: DayRecord, feature: str) -> bool:
    kind = day.kinds.get(feature, ChangeKind.INSUFFICIENT_DATA.value)
    return kind != ChangeKind.INSUFFICIENT_DATA.value


def _quantile(values: Sequence[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else None


def _share(part: int | float, whole: int | float) -> float | None:
    return float(part / whole) if whole else None


# ----------------------------------------------------------------------------
# What the filter infers
# ----------------------------------------------------------------------------
def _state_hours(runs: Sequence[HouseholdRun]) -> dict[str, Any]:
    used = [day for run in runs for day in run.days if day.usable]
    states = sorted({state for day in used for state in day.hours})
    per_household = [
        float(np.median([d.hours.get("sleeping", 0.0) for d in run.days if d.usable]))
        for run in runs
        if any(d.usable for d in run.days)
    ]
    return {
        "usable_days": len(used),
        "hours": {
            state: {
                "median": _quantile([d.hours.get(state, 0.0) for d in used], 0.5),
                "tenth": _quantile([d.hours.get(state, 0.0) for d in used], 0.1),
                "ninetieth": _quantile([d.hours.get(state, 0.0) for d in used], 0.9),
            }
            for state in states
        },
        "household_median_sleeping_hours": {
            "households": len(per_household),
            "smallest": _quantile(per_household, 0.0),
            "quarter": _quantile(per_household, 0.25),
            "median": _quantile(per_household, 0.5),
            "three_quarters": _quantile(per_household, 0.75),
            "largest": _quantile(per_household, 1.0),
        },
    }


# ----------------------------------------------------------------------------
# Silent days
# ----------------------------------------------------------------------------
def run_lengths(flags: Sequence[bool]) -> list[int]:
    """The lengths of the runs of consecutive true values."""
    lengths: list[int] = []
    length = 0
    for flag in flags:
        if flag:
            length += 1
            continue
        if length:
            lengths.append(length)
        length = 0
    if length:
        lengths.append(length)
    return lengths


def _silent_days(runs: Sequence[HouseholdRun], threshold: float) -> dict[str, Any]:
    days = [day for run in runs for day in run.days]
    silent = [day for day in days if is_silent(day)]
    lengths: Counter[int] = Counter(
        length
        for run in runs
        for length in run_lengths([is_silent(d) for d in run.days])
    )
    on_silent: Counter[str] = Counter()
    alerts = 0
    for day in days:
        for subject, _ in day.alerts:
            alerts += 1
            if is_silent(day):
                on_silent[f"{subject}: {day.kinds[subject]}"] += 1
    evaluable = [day for day in days if day.evaluable]
    groups = {
        "alert_days": [d for d in evaluable if d.alerts],
        "deviating_days": [d for d in evaluable if d.deviating(threshold)],
        "label_days": [d for d in evaluable if PRIMARY_LABEL in d.labels],
        "all_evaluable_days": evaluable,
    }
    return {
        "definition": "a monitored day on which no sensor of the home reported "
        "an event",
        "monitored_days": len(days),
        "silent_days": len(silent),
        "usable": sum(d.usable for d in silent),
        "evaluable": sum(d.evaluable for d in silent),
        "households": sum(1 for run in runs if any(is_silent(d) for d in run.days)),
        "runs_by_length": {str(k): lengths[k] for k in sorted(lengths)},
        "longest_run": max(lengths) if lengths else 0,
        "median_sleeping_hours": {
            "silent_days": _quantile(
                [d.hours.get("sleeping", 0.0) for d in silent if d.usable], 0.5
            ),
            "other_days": _quantile(
                [
                    d.hours.get("sleeping", 0.0)
                    for d in days
                    if d.usable and not is_silent(d)
                ],
                0.5,
            ),
        },
        "change_verdicts": sum(
            kind in CHANGE_KINDS for day in days for kind in day.kinds.values()
        ),
        "change_verdicts_on_silent_days": sum(
            kind in CHANGE_KINDS for day in silent for kind in day.kinds.values()
        ),
        "behavioural_alerts": alerts,
        "alerts_on_silent_days": int(sum(on_silent.values())),
        "alerts_on_silent_days_by_feature_and_verdict": dict(sorted(on_silent.items())),
        "share_silent": {
            name: {
                "days": len(group),
                "silent": sum(is_silent(d) for d in group),
                "share": _share(sum(is_silent(d) for d in group), len(group)),
            }
            for name, group in groups.items()
        },
    }


# ----------------------------------------------------------------------------
# The calendar
# ----------------------------------------------------------------------------
def _calendar(runs: Sequence[HouseholdRun]) -> dict[str, Any]:
    """Every date of the cohort, and the date on which most households are silent."""
    dates: dict[date, dict[str, int]] = {}
    for run in runs:
        for day in run.days:
            row = dates.setdefault(
                day.day,
                {
                    "monitored_households": 0,
                    "silent_households": 0,
                    "events": 0,
                    "behavioural_alerts": 0,
                },
            )
            row["monitored_households"] += 1
            row["silent_households"] += int(is_silent(day))
            row["events"] += day.events
            row["behavioural_alerts"] += len(day.alerts)
    daily = [{"date": day.isoformat(), **dates[day]} for day in sorted(dates)]
    peak = max(daily, key=lambda row: row["silent_households"], default=None)
    if peak is not None and peak["silent_households"] == 0:
        peak = None
    after = None
    if peak is not None:
        following = (date.fromisoformat(peak["date"]) + timedelta(days=1)).isoformat()
        after = next((row for row in daily if row["date"] == following), None)
    alerts = sum(row["behavioural_alerts"] for row in daily)
    return {
        "daily": daily,
        "most_silent_date": peak,
        "day_after_the_most_silent_date": after,
        "share_of_alerts_on_the_day_after": (
            _share(after["behavioural_alerts"], alerts) if after is not None else None
        ),
    }


# ----------------------------------------------------------------------------
# Replays of the baseline
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ReplayCount:
    """What the baseline concluded over one set of feature values.

    Attributes
    ----------
    evaluable_days, deviating_days
        Days with a verdict on any feature, and those among them on which a
        feature's deviation reached the threshold.
    verdicts
        Feature-days by verdict, insufficient data left out.
    label_days, label_days_deviating
        Evaluable days with the primary label, and those among them that
        deviate.
    """

    evaluable_days: int
    deviating_days: int
    verdicts: Mapping[str, int]
    label_days: int
    label_days_deviating: int

    @property
    def change_verdicts(self) -> int:
        """Verdicts the pipeline can raise a behavioural alert about."""
        return sum(self.verdicts.get(kind, 0) for kind in CHANGE_KINDS)

    def to_dict(self) -> dict[str, Any]:
        """A serialisable form of the counts."""
        other = self.evaluable_days - self.label_days
        return {
            "evaluable_days": self.evaluable_days,
            "deviating_days": self.deviating_days,
            "deviating_share": _share(self.deviating_days, self.evaluable_days),
            "verdicts": dict(sorted(self.verdicts.items())),
            "change_verdicts": self.change_verdicts,
            "change_verdicts_per_evaluable_day": _share(
                self.change_verdicts, self.evaluable_days
            ),
            "label_days": self.label_days,
            "label_days_deviating": self.label_days_deviating,
            "share_of_label_days_deviating": _share(
                self.label_days_deviating, self.label_days
            ),
            "share_of_other_days_deviating": _share(
                self.deviating_days - self.label_days_deviating, other
            ),
        }


def recorded_values(
    runs: Sequence[HouseholdRun],
) -> dict[tuple[str, str], np.ndarray]:
    """Each household's recorded values of each feature; not a number if unused."""
    return {
        (run.household, feature): np.asarray(
            [day.values.get(feature, math.nan) for day in run.days], dtype=float
        )
        for run in runs
        for feature in features()
    }


def replay(
    runs: Sequence[HouseholdRun],
    baseline: BaselineConfig,
    values: Mapping[tuple[str, str], np.ndarray],
    *,
    skip_silent: bool = False,
) -> ReplayCount:
    """Run the adaptive baseline over *values*, day by day, as the pipeline does.

    A day whose value is not a number is left out of the history, which is
    what the pipeline does with a day it did not observe well enough.
    ``skip_silent`` leaves a silent day out in the same way.
    """
    verdicts: Counter[str] = Counter()
    evaluable = deviating = label = label_deviating = 0
    for run in runs:
        judged = np.zeros(len(run.days), dtype=bool)
        largest = np.zeros(len(run.days), dtype=float)
        for feature in features():
            model = AdaptiveBaseline(feature, baseline)
            series = values[(run.household, feature)]
            for k, day in enumerate(run.days):
                value = float(series[k])
                if not math.isfinite(value) or (skip_silent and is_silent(day)):
                    continue
                change = model.observe(day.day, value)
                if change.kind is ChangeKind.INSUFFICIENT_DATA:
                    continue
                verdicts[change.kind.value] += 1
                judged[k] = True
                largest[k] = max(largest[k], abs(change.deviation))
        flagged = judged & (largest >= baseline.deviation_threshold)
        labelled = np.asarray([PRIMARY_LABEL in d.labels for d in run.days], dtype=bool)
        evaluable += int(judged.sum())
        deviating += int(flagged.sum())
        label += int((judged & labelled).sum())
        label_deviating += int((flagged & labelled).sum())
    return ReplayCount(evaluable, deviating, dict(verdicts), label, label_deviating)


def stationary_values(
    values: Mapping[tuple[str, str], np.ndarray], rng: np.random.Generator
) -> dict[tuple[str, str], np.ndarray]:
    """Independent Gaussian values with each series' own median and robust spread."""
    drawn: dict[tuple[str, str], np.ndarray] = {}
    for key in sorted(values):
        series = values[key].copy()
        used = np.isfinite(series)
        if used.any():
            centre = float(np.median(series[used]))
            spread = MAD_TO_SIGMA * float(np.median(np.abs(series[used] - centre)))
            series[used] = rng.normal(centre, spread, int(used.sum()))
        drawn[key] = series
    return drawn


def shuffled_values(
    values: Mapping[tuple[str, str], np.ndarray], rng: np.random.Generator
) -> dict[tuple[str, str], np.ndarray]:
    """Each household's days in a random order, a day's features kept together.

    Only the days on which every feature has a value are moved, so the days the
    pipeline did not use stay where they are.
    """
    drawn: dict[tuple[str, str], np.ndarray] = {}
    for household in sorted({name for name, _ in values}):
        keys = sorted(key for key in values if key[0] == household)
        used = np.all([np.isfinite(values[key]) for key in keys], axis=0)
        order = rng.permutation(np.flatnonzero(used))
        for key in keys:
            series = values[key].copy()
            series[used] = values[key][order]
            drawn[key] = series
    return drawn


def replay_differences(
    runs: Sequence[HouseholdRun], baseline: BaselineConfig
) -> dict[str, int]:
    """Feature-days on which the baseline, run again, differs from the run.

    A feature-day differs when the verdict is another one, or the deviation is
    another number. A day the pipeline did not use must be recorded without a
    verdict.
    """
    compared = differing = 0
    for run in runs:
        for feature in features():
            model = AdaptiveBaseline(feature, baseline)
            for day in run.days:
                value = float(day.values.get(feature, math.nan))
                compared += 1
                if not math.isfinite(value):
                    differing += int(_judged(day, feature))
                    continue
                change = model.observe(day.day, value)
                same = change.kind.value == day.kinds.get(feature) and math.isclose(
                    change.deviation,
                    day.deviations.get(feature, math.nan),
                    rel_tol=_TOLERANCE,
                    abs_tol=_TOLERANCE,
                )
                differing += int(not same)
    return {"feature_days_compared": compared, "feature_days_differing": differing}


def _spread(values: Sequence[float], confidence: float) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "low": float(np.quantile(array, tail)),
        "high": float(np.quantile(array, 1.0 - tail)),
    }


def _replicates(counts: Sequence[ReplayCount], confidence: float) -> dict[str, Any]:
    rows = [count.to_dict() for count in counts]
    return {
        "replicates": len(rows),
        "deviating_share": _spread([r["deviating_share"] for r in rows], confidence),
        "change_verdicts": _spread([r["change_verdicts"] for r in rows], confidence),
        "change_verdicts_per_evaluable_day": _spread(
            [r["change_verdicts_per_evaluable_day"] for r in rows], confidence
        ),
        "verdicts": {
            kind: _spread([r["verdicts"].get(kind, 0) for r in rows], confidence)
            for kind in (*CHANGE_KINDS, ChangeKind.TEMPORARY_DISTURBANCE.value)
        },
    }


def _mcse(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0


def _replays(
    runs: Sequence[HouseholdRun], baseline: BaselineConfig, config: PostHocConfig
) -> tuple[dict[str, Any], dict[str, float]]:
    values = recorded_values(runs)
    recorded = replay(runs, baseline, values)
    differences = replay_differences(runs, baseline)
    gaussian = np.random.default_rng([config.seed, 1])
    order = np.random.default_rng([config.seed, 2])
    stationary = [
        replay(runs, baseline, stationary_values(values, gaussian))
        for _ in range(config.replicates)
    ]
    shuffled = [
        replay(runs, baseline, shuffled_values(values, order))
        for _ in range(config.replicates)
    ]
    result = {
        "recorded": {
            **recorded.to_dict(),
            **differences,
            "reproduces_the_run": differences["feature_days_differing"] == 0,
        },
        "silent_days_skipped": replay(
            runs, baseline, values, skip_silent=True
        ).to_dict(),
        "stationary_gaussian": _replicates(stationary, config.confidence),
        "shuffled": _replicates(shuffled, config.confidence),
        "what_each_is": {
            "recorded": "the baseline over each household's recorded feature "
            "values; it must give the run's own verdicts",
            "silent_days_skipped": "the same, with every silent day left out of "
            "the history as an unobserved day is",
            "stationary_gaussian": "independent Gaussian values with each "
            "household's own median and MAD-based spread of each feature, on "
            "the same days",
            "shuffled": "each household's recorded days in a random order, a "
            "day's features kept together, which keeps the values and what "
            "occurs on the same day and removes their order in time, weekly "
            "rhythm included",
        },
    }
    mcse = {
        f"{name}: mean {quantity} over replicates": _mcse(
            [getattr(count, attribute) for count in counts]
        )
        for name, counts in (
            ("stationary_gaussian", stationary),
            ("shuffled", shuffled),
        )
        for quantity, attribute in (
            ("deviating days", "deviating_days"),
            ("change verdicts", "change_verdicts"),
        )
    }
    return result, mcse


# ----------------------------------------------------------------------------
# Reference size
# ----------------------------------------------------------------------------
def null_exceedance(
    size: int, threshold: float, draws: int, rng: np.random.Generator
) -> float:
    """How often a Gaussian value lies *threshold* robust SDs from *size* others.

    The reference is the median of the others and their MAD scaled to a
    standard deviation, as the baseline computes it. No floor is applied to
    the scale.
    """
    if size < 1 or draws < 1:
        raise ValueError("size and draws must be positive")
    hits = 0
    done = 0
    while done < draws:
        n = min(_CHUNK, draws - done)
        others = rng.standard_normal((n, size))
        value = rng.standard_normal(n)
        centre = np.median(others, axis=1)
        scale = MAD_TO_SIGMA * np.median(np.abs(others - centre[:, None]), axis=1)
        hits += int((np.abs(value - centre) >= threshold * scale).sum())
        done += n
    return hits / draws


def nominal_exceedance(threshold: float) -> float:
    """How often a Gaussian value lies *threshold* known SDs from a known mean."""
    return math.erfc(threshold / math.sqrt(2.0))


def estimated_exceedance(size: int, threshold: float) -> float:
    """How often a Gaussian value lies *threshold* SDs from the mean of *size* others.

    The mean and the standard deviation are estimated from the others, so the
    standardised value follows Student's t with ``size - 1`` degrees of
    freedom after scaling by ``sqrt(1 + 1 / size)``. This is exact.
    """
    if size < 2:
        raise ValueError("a standard deviation needs at least two values")
    return float(2.0 * stats.t.sf(threshold / math.sqrt(1.0 + 1.0 / size), df=size - 1))


def _reference_size(
    runs: Sequence[HouseholdRun], baseline: BaselineConfig, config: PostHocConfig
) -> dict[str, Any]:
    threshold = baseline.deviation_threshold
    rows: dict[str, list[tuple[bool, int, float, bool]]] = {f: [] for f in features()}
    for run in runs:
        for day in run.days:
            for feature in features():
                if not _judged(day, feature):
                    continue
                reference = day.references[feature]
                aware = bool(reference["weekday_aware"])
                size = int(
                    reference["weekday_samples"] if aware else reference["samples"]
                )
                rows[feature].append(
                    (
                        aware,
                        size,
                        float(day.deviations[feature]),
                        float(reference["scale"]) <= baseline.min_scale,
                    )
                )
    sizes = sorted({size for entries in rows.values() for _, size, _, _ in entries})
    rng = np.random.default_rng([config.seed, 3])
    null = {
        size: null_exceedance(size, threshold, config.null_draws, rng) for size in sizes
    }

    def group(entries: Sequence[tuple[bool, int, float, bool]]) -> dict[str, Any]:
        return {
            "days": len(entries),
            "share_at_threshold": _share(
                sum(abs(z) >= threshold for _, _, z, _ in entries), len(entries)
            ),
            "gaussian_expectation": (
                float(np.mean([null[size] for _, size, _, _ in entries]))
                if entries
                else None
            ),
            "smallest_reference": min((s for _, s, _, _ in entries), default=None),
            "largest_reference": max((s for _, s, _, _ in entries), default=None),
        }

    per_feature: dict[str, Any] = {}
    for feature, entries in rows.items():
        aware = [e for e in entries if e[0]]
        per_feature[feature] = {
            **group(entries),
            "above": _share(
                sum(z >= threshold for _, _, z, _ in entries), len(entries)
            ),
            "below": _share(
                sum(z <= -threshold for _, _, z, _ in entries), len(entries)
            ),
            "references_at_the_scale_floor": _share(
                sum(floor for _, _, _, floor in entries), len(entries)
            ),
            "pooled": group([e for e in entries if not e[0]]),
            "weekday_aware": group(aware),
            "weekday_aware_by_size": {
                str(size): group([e for e in aware if e[1] == size])
                for size in sorted({e[1] for e in aware})
            },
        }
    return {
        "threshold": threshold,
        "nominal": nominal_exceedance(threshold),
        "draws": config.null_draws,
        "weekday_min_samples": baseline.weekday_min_samples,
        "min_samples": baseline.min_samples,
        "scale_floor": baseline.min_scale,
        "gaussian_null": {str(size): null[size] for size in sizes},
        "gaussian_null_mcse": {
            str(size): math.sqrt(null[size] * (1.0 - null[size]) / config.null_draws)
            for size in sizes
        },
        "mean_and_sd_estimated": {
            str(size): estimated_exceedance(size, threshold)
            for size in sizes
            if size >= 2
        },
        "features": per_feature,
        "note": "the Gaussian rate applies no floor to the scale, so it "
        "overstates a feature whose references sit at the floor",
    }


# ----------------------------------------------------------------------------
# Direction
# ----------------------------------------------------------------------------
def _direction(
    runs: Sequence[HouseholdRun], protocol: TihmProtocol, threshold: float
) -> dict[str, Any]:
    drivers: dict[str, Counter[str]] = {
        "label_days": Counter(),
        "other_days": Counter(),
    }
    deviation: dict[str, dict[str, list[float]]] = {
        feature: {"label_days": [], "other_days": []} for feature in features()
    }
    event_z: dict[str, list[float]] = {
        "alert_days": [],
        "deviating_days": [],
        "label_days": [],
        "all_evaluable_days": [],
    }
    for run in runs:
        z = expanding_z([d.events for d in run.days], protocol.min_previous_days)
        for k, day in enumerate(run.days):
            if not day.evaluable:
                continue
            which = "label_days" if PRIMARY_LABEL in day.labels else "other_days"
            judged = [f for f in features() if _judged(day, f)]
            for feature in judged:
                deviation[feature][which].append(float(day.deviations[feature]))
            if day.deviating(threshold):
                driver = max(judged, key=lambda f: abs(day.deviations[f]))
                side = "above" if day.deviations[driver] > 0 else "below"
                drivers[which][f"{driver}: {side}"] += 1
                event_z["deviating_days"].append(float(z[k]))
            if day.alerts:
                event_z["alert_days"].append(float(z[k]))
            if PRIMARY_LABEL in day.labels:
                event_z["label_days"].append(float(z[k]))
            event_z["all_evaluable_days"].append(float(z[k]))
    return {
        "deviating_day_drivers": {
            which: {"days": int(sum(counts.values())), **dict(sorted(counts.items()))}
            for which, counts in drivers.items()
        },
        "feature_deviation": {
            feature: {
                which: {
                    "days": len(values),
                    "median": _quantile(values, 0.5),
                    "share_at_threshold": _share(
                        sum(abs(v) >= threshold for v in values), len(values)
                    ),
                }
                for which, values in groups.items()
            }
            for feature, groups in deviation.items()
        },
        "event_count_z": {
            name: {
                "days": len(values),
                "mean": float(np.mean(values)) if values else None,
                "median": _quantile(values, 0.5),
            }
            for name, values in event_z.items()
        },
        "event_count_z_is": "the day's sensor event count against the "
        "household's previous monitored days, as the protocol's event-count "
        "reference computes it",
    }


def _alerts(runs: Sequence[HouseholdRun]) -> dict[str, int]:
    counts: Counter[str] = Counter(
        f"{subject}: {day.kinds[subject]}"
        for run in runs
        for day in run.days
        for subject, _ in day.alerts
    )
    return dict(sorted(counts.items()))


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------
def describe(
    runs: Sequence[HouseholdRun], protocol: TihmProtocol, config: PostHocConfig
) -> tuple[dict[str, Any], dict[str, float]]:
    """Every description, and the Monte Carlo errors of the replicate means."""
    baseline = BaselineConfig()
    replays, mcse = _replays(runs, baseline, config)
    size = _reference_size(runs, baseline, config)
    mcse.update(
        {
            f"gaussian value at the threshold, reference of {count}": error
            for count, error in size["gaussian_null_mcse"].items()
        }
    )
    results = {
        "result_schema": POST_HOC_SCHEMA,
        "status": STATUS,
        "state_hours": _state_hours(runs),
        "silent_days": _silent_days(runs, baseline.deviation_threshold),
        "calendar": _calendar(runs),
        "replays": replays,
        "reference_size": size,
        "direction": _direction(runs, protocol, baseline.deviation_threshold),
        "alerts_by_feature_and_verdict": _alerts(runs),
    }
    return results, mcse


def same_results(first: Any, second: Any) -> bool:
    """Whether two result trees agree, numbers to a relative ``1e-9``."""
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return set(first) == set(second) and all(
            same_results(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(
            same_results(a, b) for a, b in zip(first, second)
        )
    if isinstance(first, bool) or isinstance(second, bool):
        return first is second
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(first, second, rel_tol=_TOLERANCE, abs_tol=_TOLERANCE)
    return bool(first == second)


@dataclass(frozen=True)
class PostHocResult:
    """The record of the descriptions and where it was written."""

    record: ExperimentRecord
    path: Path | None


def run_post_hoc(
    adapter: DatasetAdapter,
    physiology: Sequence[tuple[str, datetime, str, float]],
    protocol: TihmProtocol,
    published: Mapping[str, Any],
    *,
    config: PostHocConfig | None = None,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> PostHocResult:
    """Run every household again, check it is the published run, and describe it.

    *published* is the record of the protocol's run. The descriptions are
    refused unless scoring these runs gives that record's results, so what is
    described is the run that was published and not another.
    """
    config = config or PostHocConfig()
    expected = published["results"]
    if expected.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    if published["configuration"].get("protocol_sha256") != protocol.sha256():
        raise ValueError("the published record was made under another protocol")
    names = list(adapter.households())
    runs: list[HouseholdRun] = []
    refused: dict[str, list[str]] = {}
    for name in names:
        data = adapter.load(name)
        report = validate_household(data, protocol.mapping)
        if not report.ok:
            refused[name] = sorted({issue.code for issue in report.errors})
            continue
        runs.append(run_household(data, protocol))
    again = score(runs, physiology, protocol, households=len(names), refused=refused)
    if not same_results(again, expected):
        raise ValueError(
            "these runs do not reproduce the published record's results, so "
            "they are not described"
        )
    results, mcse = describe(runs, protocol, config)
    if not results["replays"]["recorded"]["reproduces_the_run"]:
        raise ValueError(
            "the baseline run again over the recorded values does not give the "
            "run's verdicts, so the replays are not reported"
        )
    results["describes"] = {
        "experiment": published["experiment"],
        "recorded_at": published["recorded_at"],
        "git_commit": published["environment"]["git_commit"],
        "protocol_sha256": published["configuration"]["protocol_sha256"],
        "results_reproduced": True,
    }
    baseline = BaselineConfig()
    record = ExperimentRecord(
        experiment=POST_HOC_EXPERIMENT,
        configuration={
            **config.to_dict(),
            "status": STATUS,
            "result_schema": POST_HOC_SCHEMA,
            "protocol_sha256": protocol.sha256(),
            "step_minutes": protocol.step_minutes,
            "min_previous_days": protocol.min_previous_days,
        },
        inference=ONLINE,
        evidence=EvidenceSummary.online(
            moment for run in runs for moment in run.closes
        ),
        seeds=[config.seed],
        results=results,
        data_source="tihm",
        metric_definitions={
            "silent day": results["silent_days"]["definition"],
            "change verdict": "a verdict of abrupt change, persistent change or "
            "gradual drift, which the pipeline can raise a behavioural alert "
            "about",
            "deviating day": protocol.to_dict()["definitions"]["deviating_day"],
            "replicate range": "the central share of replicates given by the "
            "configuration's confidence; not an interval for these homes",
        },
        notes=[
            "Post hoc. Every description was chosen after the protocol's "
            "results had been read. None is part of the protocol.",
            "A replay that leaves silent days out is a diagnosis of the run, "
            "not a corrected result.",
            "The dataset holds sensor activations and no signal that a sensor "
            "or its gateway was working. A silent day may be an empty home or "
            "an apparatus that was not reporting; the data cannot say which. "
            "The dataset paper attributes a large drop in the activity data in "
            "mid-June 2019 to a technical failure in the data collection "
            "server.",
            "The stationary and shuffled replays re-run the baseline alone. "
            "They count verdicts, not alerts: the alert policy is not replayed.",
        ],
        inputs=list(inputs),
        preprocessing={
            "contract": "external-dataset contract, strict conversion",
            "mapping_sha256": protocol.mapping.sha256(),
            "step_minutes": protocol.step_minutes,
        },
        models=[
            ModelRecord(
                "adaptive_baseline_defaults",
                {
                    "min_samples": baseline.min_samples,
                    "weekday_min_samples": baseline.weekday_min_samples,
                    "deviation_threshold": baseline.deviation_threshold,
                    "persistence_days": baseline.persistence_days,
                    "trend_window": baseline.trend_window,
                    "trend_threshold": baseline.trend_threshold,
                    "min_scale": baseline.min_scale,
                },
                "AdaptiveBaseline",
            )
        ],
        mcse=mcse,
    )
    path = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = record.write(Path(output_dir) / f"{POST_HOC_EXPERIMENT}.json")
    return PostHocResult(record, path)
