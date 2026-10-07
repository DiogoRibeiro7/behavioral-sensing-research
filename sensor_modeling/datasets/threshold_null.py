"""What the baseline's thresholds mean on days with nothing in them.

The TIHM run found that the personal baseline's deviation threshold is passed
far more often than three standard deviations would be, because a
weekday-aware reference rests on four or five days.
``BaselineConfig.calibrated`` was built from that finding. This module
measures both references where the answer is known: on synthetic daily values
that are independent draws from one Gaussian, given to
:class:`~sensor_modeling.baseline.adaptive.AdaptiveBaseline` directly. No home
is simulated and no state is inferred.

It measures four things.

**How often a day passes the threshold**, by how many days the reference
rests on, for the default reference and the calibrated one.

**What threshold the default reference would need** for a day to pass it as
often as three standard deviations states, at each sample size.

**What each reference finds** when a step or a slow ramp is added to the
same series, and how often it reports a change when nothing was added, over a
grid of thresholds. This is where the two references are compared at the same
rate of false reports, which is the only fair comparison of two rules that do
not fire equally often.

**Where the calibrated reference matches the default**: the multiple of its
thresholds at which it finds a step as often, and the one at which it reports
falsely as often, by the rule the threshold-calibration protocol applies to
its tuning homes.

The values are synthetic. A real day is not an independent Gaussian draw, so
nothing here says how either reference behaves in a home. It says what each
does when its own assumptions hold, and that is the least a threshold should
get right.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from ..baseline.adaptive import (
    DOF_PER_RESIDUAL,
    MAD_TO_SIGMA,
    AdaptiveBaseline,
    BaselineConfig,
    ChangeKind,
)
from ..evaluation import ExperimentRecord, ModelRecord
from ..fusion import ONLINE, NotEnumerated

EXPERIMENT = "threshold-null"
RESULT_SCHEMA = "threshold-null/1"

#: The two references.
DEFAULT = "default"
CALIBRATED = "calibrated"
REFERENCES = (DEFAULT, CALIBRATED)

#: What is added to the Gaussian days.
NONE = "none"
WEEKEND = "weekend"
STEP2 = "step2"
STEP3 = "step3"
RAMP2 = "ramp2"
WEEKEND_STEP2 = "weekend_step2"
NULL_SHAPES = (NONE, WEEKEND)
SHAPES = (NONE, WEEKEND, STEP2, STEP3, RAMP2, WEEKEND_STEP2)

#: The two settings the references are matched in: what is added when nothing
#: changed, and what is added when a step of two standard deviations did.
SETTINGS = {"plain": (NONE, STEP2), "weekly_rhythm": (WEEKEND, WEEKEND_STEP2)}

#: The verdicts the pipeline can raise a behavioural alert about.
CHANGE_KINDS = (
    ChangeKind.PERSISTENT_CHANGE,
    ChangeKind.ABRUPT_CHANGE,
    ChangeKind.GRADUAL_DRIFT,
)
KINDS = tuple(ChangeKind)

#: What each stream of random numbers is for.
_RATES, _OPERATING, _QUANTILES = 0, 1, 2
_BLOCK = 100
_CHUNK = 200_000


@dataclass(frozen=True)
class ThresholdNull:
    """Every setting of the measurement.

    Attributes
    ----------
    seed_root
        The root every random number is derived from.
    rate_series, series
        Series behind the rates of days past a threshold, and behind the
        operating points.
    days, start
        Length of a series and its first day, a Monday.
    level
        The mean of the days. Nothing depends on it.
    rate_thresholds
        The thresholds a day's deviation is counted against.
    weekend
        What the weekend shape adds to Saturdays and to Sundays, in standard
        deviations.
    step_day, ramp_day, ramp_days, window_days
        Where a step begins, where a ramp begins and how long it takes, and
        the days, from the step's first, in which a change verdict counts.
    scales
        The multiples of the declared thresholds each reference is run at.
        Both thresholds are multiplied together.
    sizes, quantile_draws
        The sample sizes the needed threshold is given for, and the draws
        behind each.
    """

    seed_root: int = 20261010
    rate_series: int = 10_000
    series: int = 2_000
    days: int = 120
    start: date = date(2024, 3, 4)
    level: float = 8.0
    rate_thresholds: tuple[float, ...] = (1.5, 1.8, 2.0, 2.5, 3.0)
    weekend: tuple[float, float] = (1.5, 2.0)
    step_day: int = 56
    ramp_day: int = 42
    ramp_days: int = 28
    window_days: int = 21
    scales: tuple[float, ...] = (
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.8,
        0.9,
        1.0,
        1.25,
        1.5,
        2.0,
    )
    sizes: tuple[int, ...] = (4, 5, 6, 8, 10, 12, 14, 17, 27)
    quantile_draws: int = 2_000_000

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.rate_series < 2 or self.series < 2:
            raise ValueError("at least two series are needed for a standard error")
        if self.start.weekday() != 0:
            raise ValueError("the first day must be a Monday")
        if not 0 < self.ramp_day < self.step_day < self.days:
            raise ValueError("the ramp, the step and the end are not in order")
        if self.step_day + self.window_days > self.days:
            raise ValueError("the window runs past the end of the series")
        if 1.0 not in self.scales or any(s <= 0 for s in self.scales):
            raise ValueError("scales must be positive and include 1")
        if len(set(self.scales)) != len(self.scales):
            raise ValueError("scales must be distinct")
        if any(size < 2 for size in self.sizes) or self.quantile_draws < 1000:
            raise ValueError("sizes must be at least 2 and draws at least 1,000")

    def window(self) -> range:
        """The days in which a change verdict counts as finding the change."""
        return range(self.step_day, self.step_day + self.window_days)

    def offsets(self, shape: str) -> np.ndarray:
        """What a shape adds to each day, in standard deviations."""
        index = np.arange(self.days)
        if shape == NONE:
            return np.zeros(self.days)
        if shape == WEEKEND:
            weekday = (index + self.start.weekday()) % 7
            return np.where(
                weekday == 5,
                self.weekend[0],
                np.where(weekday == 6, self.weekend[1], 0.0),
            )
        if shape in (STEP2, STEP3):
            size = 2.0 if shape == STEP2 else 3.0
            return np.where(index >= self.step_day, -size, 0.0)
        if shape == WEEKEND_STEP2:
            return self.offsets(WEEKEND) + self.offsets(STEP2)
        if shape == RAMP2:
            return -2.0 * np.clip((index - self.ramp_day) / self.ramp_days, 0.0, 1.0)
        raise KeyError(f"no shape '{shape}'")

    def noise(self, purpose: int, block: int, size: int) -> np.ndarray:
        """Independent standard Gaussian days for one block of series."""
        rng = np.random.default_rng([self.seed_root, purpose, block])
        return rng.standard_normal((size, self.days))

    def to_dict(self) -> dict[str, Any]:
        """Return the settings and what they mean."""
        baseline = BaselineConfig()
        window = self.window()
        return {
            "schema": "threshold-null-design/1",
            "name": EXPERIMENT,
            "status": "a measurement on synthetic values, made to design and "
            "to check the calibrated reference. It tests nothing about a home",
            "values": "independent draws from one Gaussian with standard "
            f"deviation 1 around {self.level:g}, one a day, given to "
            "`AdaptiveBaseline` directly. No home is simulated and no state is "
            "inferred",
            "seed_root": self.seed_root,
            "seeds": "the series of block b for purpose p come from "
            "`numpy.random.default_rng([seed_root, p, b])`, in blocks of "
            f"{_BLOCK}; p is {_RATES} for the rates, {_OPERATING} for the "
            f"operating points and {_QUANTILES} for the needed thresholds",
            "rate_series": self.rate_series,
            "series": self.series,
            "days": self.days,
            "start": self.start.isoformat(),
            "pairing": "every reference, shape and threshold is run on the "
            "same series, so two of them differ in the reference, the shape or "
            "the threshold alone",
            "references": {
                DEFAULT: "the default: the centre and the scale are the median "
                "and the MAD of the days of the same weekday once "
                f"{baseline.weekday_min_samples} exist, and of all the days "
                "before that",
                CALIBRATED: "`BaselineConfig.calibrated`: the same centre, a "
                "scale pooled over every retained day, and the deviation "
                "mapped through a Student t on "
                f"{DOF_PER_RESIDUAL:g} degrees of freedom for each day behind "
                "the scale",
            },
            "baseline": {
                "min_samples": baseline.min_samples,
                "weekday_min_samples": baseline.weekday_min_samples,
                "deviation_threshold": baseline.deviation_threshold,
                "persistence_days": baseline.persistence_days,
                "trend_window": baseline.trend_window,
                "trend_threshold": baseline.trend_threshold,
                "min_scale": baseline.min_scale,
                "min_scale_in_standard_deviations": baseline.min_scale,
                "history_days": baseline.history_days,
            },
            "rate_thresholds": list(self.rate_thresholds),
            "level": self.level,
            "weekend": list(self.weekend),
            "step_day": self.step_day,
            "ramp_day": self.ramp_day,
            "ramp_days": self.ramp_days,
            "window_days": self.window_days,
            "shapes": {
                NONE: "nothing is added",
                WEEKEND: f"{self.weekend[0]:g} standard deviations are added to "
                f"every Saturday and {self.weekend[1]:g} to every Sunday: a "
                "weekly rhythm and no change",
                STEP2: f"2 standard deviations are taken off from day "
                f"{self.step_day}",
                STEP3: f"3 standard deviations are taken off from day "
                f"{self.step_day}",
                RAMP2: f"2 standard deviations are taken off gradually, from day "
                f"{self.ramp_day} over {self.ramp_days} days",
                WEEKEND_STEP2: "the weekly rhythm, and 2 standard deviations "
                f"taken off from day {self.step_day}",
            },
            "scales": list(self.scales),
            "thresholds": "at scale s the deviation threshold is "
            f"{baseline.deviation_threshold:g} s and the trend threshold "
            f"{baseline.trend_threshold:g} s",
            "matching": {
                "same_detection": "the largest multiple at which the "
                "calibrated reference finds the step of two standard "
                "deviations in at least as many series as the default does at "
                "its declared thresholds; the smallest multiple if there is "
                "none",
                "same_false_reports": "the smallest multiple at which the "
                "calibrated reference reports a change, with no step added, "
                "in no more series than the default does at its declared "
                "thresholds; the largest multiple if there is none",
                "settings": {
                    name: {"nothing_changed": quiet, "a_step": moved}
                    for name, (quiet, moved) in SETTINGS.items()
                },
                "why": "this is the rule the threshold-calibration protocol "
                "applies to its tuning homes, so that what it finds there can "
                "be set against what Gaussian days give",
            },
            "definitions": {
                "evaluable_day": "a day the baseline gives a verdict on other "
                f"than insufficient data: from the day {baseline.min_samples} "
                "others are retained",
                "deviating_day": "an evaluable day whose deviation is at or "
                "past the threshold in absolute value",
                "nominal": "the share of Gaussian values at or past the "
                "threshold in absolute value, which is what the threshold "
                "states",
                "change_verdict": "a verdict of persistent change, abrupt "
                "change or gradual drift, which the pipeline can raise a "
                "behavioural alert about. The alert policy is not applied",
                "found": "a series with a change verdict on a day from "
                f"{window.start} to {window.stop - 1}. Where nothing was added "
                "this is a false report, and where a ramp was added it began "
                f"{self.step_day - self.ramp_day} days before the window opens",
                "needed_threshold": "the value a Gaussian day's deviation from "
                "the median of n other days, in units of 1.4826 times their "
                "MAD, exceeds in absolute value as often as the nominal rate "
                "of the declared threshold. The floor on the scale is not "
                "applied",
            },
            "sizes": list(self.sizes),
            "quantile_draws": self.quantile_draws,
            "standard_errors": "a share of days is a ratio over series, and "
            "its standard error treats the series as the unit, since the days "
            "of one series share a history. A share of series has the "
            "binomial standard error, and a paired difference the standard "
            "deviation of the differences over the root of their number",
            "what_was_seen_before": [
                "the calibrated reference was designed on series of this kind, "
                f"drawn from other seeds. The {DOF_PER_RESIDUAL:g} degrees of "
                "freedom for each residual were settled in exploratory runs "
                "that are not recorded",
                "this measurement was made once before in full, from another "
                "root and with fewer multiples, when the calibrated reference "
                "fitted its trend to the days as they are. With a weekly "
                "rhythm added it then reported a gradual drift more often, as "
                "the default does. Its trend was changed because of that, to "
                "be fitted to each day's distance from its own weekday, and a "
                "shape with both a rhythm and a step was added. That first "
                "record is not published",
                "none of the series here was used to choose anything",
            ],
        }


def baseline_config(reference: str, scale: float = 1.0) -> BaselineConfig:
    """The baseline's configuration for a reference at a multiple of its thresholds."""
    if reference not in REFERENCES:
        raise KeyError(f"no reference '{reference}'")
    declared = BaselineConfig()
    return BaselineConfig(
        calibrated=reference == CALIBRATED,
        deviation_threshold=declared.deviation_threshold * scale,
        trend_threshold=declared.trend_threshold * scale,
    )


def judge(
    values: Sequence[float], config: BaselineConfig, start: date
) -> tuple[np.ndarray, np.ndarray]:
    """Give a series to a fresh baseline; return each day's deviation and verdict.

    The deviation is absolute, and not a number on a day without a verdict.
    The verdict is its place in :data:`KINDS`.
    """
    baseline = AdaptiveBaseline("value", config)
    deviations = np.full(len(values), np.nan)
    kinds = np.zeros(len(values), dtype=np.int8)
    for index, value in enumerate(values):
        verdict = baseline.observe(start + timedelta(days=index), float(value))
        kinds[index] = KINDS.index(verdict.kind)
        if verdict.kind is not ChangeKind.INSUFFICIENT_DATA:
            deviations[index] = abs(verdict.deviation)
    return deviations, kinds


def layout(config: ThresholdNull) -> list[tuple[bool, int]]:
    """For each day, whether its reference is weekday-aware and its sample size.

    The size is the number of days behind the centre: those of the weekday
    when the reference is weekday-aware, and all the retained days when not.
    It depends on the day alone, since no day of a series is skipped.
    """
    baseline = AdaptiveBaseline("value", BaselineConfig())
    places: list[tuple[bool, int]] = []
    for index in range(config.days):
        verdict = baseline.observe(config.start + timedelta(days=index), 0.0)
        reference = verdict.reference
        places.append(
            (
                reference.weekday_aware,
                (
                    reference.weekday_samples
                    if reference.weekday_aware
                    else reference.samples
                ),
            )
        )
    return places


# ----------------------------------------------------------------------------
# Blocks of series
# ----------------------------------------------------------------------------
Rates = dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]
Operating = dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray, np.ndarray]]


def _rates_block(arguments: tuple[ThresholdNull, int, int]) -> Rates:
    """Deviations and verdicts of one block, for each null shape and reference."""
    config, block, size = arguments
    noise = config.noise(_RATES, block, size)
    found: Rates = {}
    for shape in NULL_SHAPES:
        values = config.level + noise + config.offsets(shape)
        for reference in REFERENCES:
            judged = [
                judge(row, baseline_config(reference), config.start) for row in values
            ]
            found[shape, reference] = (
                np.array([deviations for deviations, _ in judged]),
                np.array([kinds for _, kinds in judged]),
            )
    return found


def _operating_block(arguments: tuple[ThresholdNull, int, int]) -> Operating:
    """What one block shows at every shape, reference and scale.

    For each series: whether a change verdict fell in the window, how many
    days had one, and how many days were evaluable.
    """
    config, block, size = arguments
    noise = config.noise(_OPERATING, block, size)
    window = config.window()
    change = [KINDS.index(kind) for kind in CHANGE_KINDS]
    insufficient = KINDS.index(ChangeKind.INSUFFICIENT_DATA)
    found: Operating = {}
    for shape in SHAPES:
        values = config.level + noise + config.offsets(shape)
        for reference in REFERENCES:
            for scale in config.scales:
                settings = baseline_config(reference, scale)
                kinds = np.array(
                    [judge(row, settings, config.start)[1] for row in values]
                )
                verdicts = np.isin(kinds, change)
                found[shape, reference, scale] = (
                    verdicts[:, window.start : window.stop].any(axis=1),
                    verdicts.sum(axis=1),
                    (kinds != insufficient).sum(axis=1),
                )
    return found


def _blocks(total: int) -> list[tuple[int, int]]:
    return [
        (block, min(_BLOCK, total - block * _BLOCK))
        for block in range(math.ceil(total / _BLOCK))
    ]


def _gather(
    worker: Callable[[tuple[ThresholdNull, int, int]], dict[Any, tuple[Any, ...]]],
    config: ThresholdNull,
    total: int,
    jobs: int,
    progress: Callable[[int, int], None] | None,
) -> dict[Any, tuple[np.ndarray, ...]]:
    """Run every block and join the blocks in order, whatever *jobs* is."""
    tasks = [(config, block, size) for block, size in _blocks(total)]
    parts: list[dict[Any, tuple[Any, ...]]] = []
    if jobs <= 1:
        for task in tasks:
            parts.append(worker(task))
            if progress is not None:
                progress(len(parts), len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for part in pool.map(worker, tasks):
                parts.append(part)
                if progress is not None:
                    progress(len(parts), len(tasks))
    return {
        key: tuple(
            np.concatenate([part[key][place] for part in parts])
            for place in range(len(parts[0][key]))
        )
        for key in parts[0]
    }


# ----------------------------------------------------------------------------
# Rates
# ----------------------------------------------------------------------------
def nominal(threshold: float) -> float:
    """The share of Gaussian values at or past *threshold* in absolute value."""
    return 2.0 * (1.0 - NormalDist().cdf(threshold))


def clustered_rate(hits: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    """A share of days, with a standard error whose unit is the series.

    *hits* and *counts* hold one entry for each series. The share is the
    ratio of their sums, and its standard error is that of a ratio of means
    over independent series.
    """
    hits, counts = np.asarray(hits, dtype=float), np.asarray(counts, dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return {"rate": None, "se": None, "days": 0}
    rate = float(hits.sum()) / total
    series = hits.size
    error = None
    if series > 1:
        residual = hits - rate * counts
        error = math.sqrt(series / (series - 1) * float((residual**2).sum())) / total
    return {"rate": rate, "se": error, "days": int(total)}


def share(flags: np.ndarray) -> dict[str, Any]:
    """A share of independent series, with its binomial standard error."""
    flags = np.asarray(flags, dtype=bool)
    value = float(flags.mean())
    return {
        "share": value,
        "se": math.sqrt(value * (1.0 - value) / flags.size),
        "count": int(flags.sum()),
        "series": int(flags.size),
    }


def paired(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """The mean of *first* minus *second* over the same series."""
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return {
        "estimate": float(difference.mean()),
        "se": float(difference.std(ddof=1) / math.sqrt(difference.size)),
        "series": int(difference.size),
    }


def _exceedance(
    deviations: np.ndarray, days: Sequence[int], thresholds: Sequence[float]
) -> dict[str, Any]:
    """The share of the given days at or past each threshold."""
    chosen = deviations[:, list(days)]
    judged = ~np.isnan(chosen)
    counts = judged.sum(axis=1)
    return {
        f"{threshold:g}": clustered_rate(
            (np.where(judged, chosen, 0.0) >= threshold).sum(axis=1), counts
        )
        for threshold in thresholds
    }


def _groups(config: ThresholdNull) -> dict[str, dict[str, list[int]]]:
    """The days of a series, grouped by what their reference rests on."""
    places = layout(config)
    evaluable = BaselineConfig().min_samples
    days = range(evaluable, config.days)
    pooled = [day for day in days if not places[day][0]]
    aware = [day for day in days if places[day][0]]
    by_size: dict[str, list[int]] = {}
    for day in aware:
        by_size.setdefault(str(places[day][1]), []).append(day)
    by_week: dict[str, list[int]] = {}
    for day in days:
        by_week.setdefault(str(day // 7 + 1), []).append(day)
    return {
        "phase": {"all": list(days), "pooled": pooled, "weekday_aware": aware},
        "by_weekday_size": by_size,
        "by_week": by_week,
    }


def _rates(found: Rates, config: ThresholdNull) -> dict[str, Any]:
    groups = _groups(config)
    change = [KINDS.index(kind) for kind in CHANGE_KINDS]
    results: dict[str, Any] = {}
    for shape in NULL_SHAPES:
        results[shape] = {}
        for reference in REFERENCES:
            deviations, kinds = found[shape, reference]
            judged = (~np.isnan(deviations)).sum(axis=1)
            results[shape][reference] = {
                "evaluable_days": int(judged.sum()),
                **{
                    name: {
                        label: _exceedance(deviations, days, config.rate_thresholds)
                        for label, days in members.items()
                    }
                    for name, members in groups.items()
                },
                "verdicts": {
                    kind.value: clustered_rate(
                        (kinds == KINDS.index(kind)).sum(axis=1), judged
                    )
                    for kind in KINDS
                    if kind is not ChangeKind.INSUFFICIENT_DATA
                },
                "change_verdicts": clustered_rate(
                    np.isin(kinds, change).sum(axis=1), judged
                ),
            }
    return results


# ----------------------------------------------------------------------------
# The threshold a reference of a few days would need
# ----------------------------------------------------------------------------
def reference_deviations(size: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Absolute deviations of Gaussian days from a median and MAD of *size* others."""
    if size < 2 or draws < 1:
        raise ValueError("size must be at least 2 and draws positive")
    parts = []
    done = 0
    while done < draws:
        count = min(_CHUNK, draws - done)
        others = rng.standard_normal((count, size))
        centre = np.median(others, axis=1)
        spread = MAD_TO_SIGMA * np.median(np.abs(others - centre[:, None]), axis=1)
        parts.append(np.abs(rng.standard_normal(count) - centre) / spread)
        done += count
    return np.concatenate(parts)


def _needed(config: ThresholdNull) -> dict[str, Any]:
    declared = BaselineConfig().deviation_threshold
    rate = nominal(declared)
    z = NormalDist().inv_cdf(0.975)
    results: dict[str, Any] = {}
    for size in config.sizes:
        rng = np.random.default_rng([config.seed_root, _QUANTILES, size])
        deviations = np.sort(reference_deviations(size, config.quantile_draws, rng))
        draws = deviations.size
        passed = float((deviations >= declared).mean())
        # The quantile's interval is read from the order statistics whose
        # ranks bound the binomial count of draws past it.
        half = z * math.sqrt(draws * rate * (1.0 - rate))
        place = draws * (1.0 - rate)
        low = deviations[max(int(math.floor(place - half)), 0)]
        high = deviations[min(int(math.ceil(place + half)), draws - 1)]
        results[str(size)] = {
            "passes_the_declared_threshold": {
                "rate": passed,
                "se": math.sqrt(passed * (1.0 - passed) / draws),
            },
            "needed_threshold": {
                "value": float(np.quantile(deviations, 1.0 - rate)),
                "low": float(low),
                "high": float(high),
                "confidence": 0.95,
            },
        }
    return {
        "declared_threshold": declared,
        "nominal": rate,
        "draws": config.quantile_draws,
        "by_size": results,
    }


# ----------------------------------------------------------------------------
# Operating points
# ----------------------------------------------------------------------------
def _operating(found: Operating, config: ThresholdNull) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for shape in SHAPES:
        results[shape] = {}
        for reference in REFERENCES:
            results[shape][reference] = {}
            for scale in config.scales:
                hit, verdict_days, evaluable = found[shape, reference, scale]
                results[shape][reference][f"{scale:g}"] = {
                    "found": share(hit),
                    "change_verdicts_per_day": clustered_rate(verdict_days, evaluable),
                }
    return results


def matching_multiples(
    false: Mapping[float, float],
    detection: Mapping[float, float],
    default_false: float,
    default_detection: float,
) -> dict[str, Any]:
    """The multiples at which one reference matches another's operating point.

    *false* and *detection* give, for each multiple of the reference being
    matched, how often it reports falsely and how often it finds the change.
    The same rule is applied to the simulated homes of the
    threshold-calibration protocol.
    """
    enough = [scale for scale in detection if detection[scale] >= default_detection]
    quiet = [scale for scale in false if false[scale] <= default_false]
    return {
        "same_detection": max(enough) if enough else min(detection),
        "same_detection_was_reached": bool(enough),
        "same_false_reports": min(quiet) if quiet else max(false),
        "same_false_reports_was_reached": bool(quiet),
    }


def _matching(found: Operating, config: ThresholdNull) -> dict[str, Any]:
    """Where the calibrated reference matches the default, in each setting."""
    results: dict[str, Any] = {}
    for name, (quiet, moved) in SETTINGS.items():
        matched = matching_multiples(
            {s: float(found[quiet, CALIBRATED, s][0].mean()) for s in config.scales},
            {s: float(found[moved, CALIBRATED, s][0].mean()) for s in config.scales},
            float(found[quiet, DEFAULT, 1.0][0].mean()),
            float(found[moved, DEFAULT, 1.0][0].mean()),
        )
        results[name] = {
            **matched,
            "nothing_changed": quiet,
            "a_step": moved,
            "calibrated_minus_default": {
                f"{scale:g}": {
                    shape: paired(
                        found[shape, CALIBRATED, scale][0],
                        found[shape, DEFAULT, 1.0][0],
                    )
                    for shape in SHAPES
                }
                for scale in dict.fromkeys(
                    (1.0, matched["same_detection"], matched["same_false_reports"])
                )
            },
        }
    return results


def measure(
    config: ThresholdNull,
    *,
    jobs: int = 1,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Run the whole measurement and return its results."""

    def stage(name: str) -> Callable[[int, int], None] | None:
        if progress is None:
            return None
        return lambda done, total: progress(name, done, total)

    rates = _gather(_rates_block, config, config.rate_series, jobs, stage("rates"))
    operating = _gather(
        _operating_block, config, config.series, jobs, stage("operating")
    )
    return {
        "result_schema": RESULT_SCHEMA,
        "nominal": {f"{t:g}": nominal(t) for t in config.rate_thresholds},
        "rates": _rates(rates, config),  # type: ignore[arg-type]
        "needed_threshold": _needed(config),
        "operating": _operating(operating, config),  # type: ignore[arg-type]
        "matching": _matching(operating, config),  # type: ignore[arg-type]
    }


def _mcse(results: Mapping[str, Any]) -> dict[str, float]:
    named: dict[str, float] = {}
    for reference in REFERENCES:
        cell = results["rates"][NONE][reference]["phase"]["all"]
        for threshold, entry in cell.items():
            named[f"deviating_share_{reference}_{threshold}"] = float(entry["se"])
    for scale, shapes in results["matching"]["plain"][
        "calibrated_minus_default"
    ].items():
        for shape, entry in shapes.items():
            named[f"found_calibrated_{scale}_minus_default_{shape}"] = float(
                entry["se"]
            )
    return named


@dataclass(frozen=True)
class ThresholdNullResult:
    """A run of the measurement: the record, and where it was written."""

    record: ExperimentRecord
    path: Path | None


def run_threshold_null(
    config: ThresholdNull | None = None,
    *,
    output_dir: Path | None = None,
    jobs: int = 1,
    progress: Callable[[str, int, int], None] | None = None,
) -> ThresholdNullResult:
    """Run the measurement and write its record."""
    config = config or ThresholdNull()
    results = measure(config, jobs=jobs, progress=progress)
    declared = BaselineConfig()
    record = ExperimentRecord(
        experiment=EXPERIMENT,
        configuration={**config.to_dict(), "result_schema": RESULT_SCHEMA},
        inference=ONLINE,
        evidence=NotEnumerated(
            "no state is inferred: synthetic daily values are given to the "
            "baseline directly, one a day, and each is judged before it joins "
            "the history"
        ),
        seeds=[config.seed_root],
        results=results,
        data_source="synthetic",
        metric_definitions=dict(config.to_dict()["definitions"]),
        notes=[
            "The values are synthetic: independent Gaussian days, with a step, "
            "a ramp or a weekly rhythm added where a shape says so. No home is "
            "simulated and no state is inferred, so nothing here is evidence "
            "about a home.",
            "A measurement, not a test. It was made to check the calibrated "
            "reference where the answer is known. The series here were not "
            "used to choose anything.",
            "Change verdicts are counted, not alerts: the alert policy is not "
            "applied.",
            f"{config.rate_series:,} series are behind the rates and "
            f"{config.series:,} behind the operating points, all derived from "
            "one recorded root. docs/SIMULATION_PROTOCOLS.md sets a floor of "
            "100, and every share is given with its standard error.",
        ],
        models=[
            ModelRecord(
                f"adaptive_baseline_{reference}",
                {
                    "calibrated": reference == CALIBRATED,
                    "min_samples": declared.min_samples,
                    "weekday_min_samples": declared.weekday_min_samples,
                    "deviation_threshold": declared.deviation_threshold,
                    "persistence_days": declared.persistence_days,
                    "trend_window": declared.trend_window,
                    "trend_threshold": declared.trend_threshold,
                    "min_scale": declared.min_scale,
                    "scales": list(config.scales),
                },
                "AdaptiveBaseline",
            )
            for reference in REFERENCES
        ],
        mcse=_mcse(results),
    )
    path = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = record.write(Path(output_dir) / f"{EXPERIMENT}.json")
    return ThresholdNullResult(record, path)
