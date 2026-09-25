"""Household-level statistics: each household is one unit, however long.

Three layers, kept apart
------------------------
1. **Timestamp level.** :func:`~sensor_modeling.evaluation.prediction_metrics`
   scores the timestamps of *one* household. Nothing in this module does that.
2. **Household level.** :func:`score_households` applies it to each household
   separately, :func:`household_values` takes one metric per household, and
   :func:`summarise_households` describes those values with each household
   counted once.
3. **Across households.** :func:`compare_households` pairs two models' household
   values and estimates their difference by resampling households.

Why households, not timestamps
------------------------------
Timestamps within one home are strongly dependent. They share a resident, a
routine, a sensor layout and an annotator, and consecutive states persist for
many steps. A household with 50,000 scored timestamps is still one home.
Treating its timestamps as independent would make intervals too narrow by
roughly the square root of the design effect ``1 + (m - 1) rho``, where ``m`` is
the timestamps per home and ``rho`` their within-home correlation. With ``m``
in the thousands, even a small ``rho`` shrinks an interval by an order of
magnitude.

The question these comparisons answer is also about homes: would a model do
better in another household like these? Households are therefore the unit that
is scored, summarised and resampled. Every household carries the same weight,
however many timestamps it contributed.

Effect estimates and intervals are the output. No null-hypothesis test is
reported.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np

from ..states.ontology import DEFAULT_STATES, BehaviouralState
from .metrics import PredictionMetrics, prediction_metrics
from .resampling import (
    Interval,
    bca_interval,
    check_settings,
    jackknife,
    percentile_interval,
    resample_indices,
)

#: Interval methods :func:`compare_households` accepts.
INTERVAL_METHODS = ("percentile", "bca")

HouseholdValues = Mapping[str, float | None]
MetricGetter = Callable[[PredictionMetrics], float | None]

_SCALAR_METRICS = frozenset(
    field.name
    for field in fields(PredictionMetrics)
    if field.name not in {"n", "per_class_recall", "confusion"}
)


# ----------------------------------------------------------------------------
# Household level
# ----------------------------------------------------------------------------
def score_households(
    truth: Mapping[str, Sequence[BehaviouralState | None]],
    predicted: Mapping[str, Sequence[BehaviouralState]],
    probabilities: Mapping[str, np.ndarray] | None = None,
    *,
    states: Sequence[BehaviouralState] = DEFAULT_STATES,
) -> dict[str, PredictionMetrics | None]:
    """Score each household on its own timestamps, never pooled with others.

    A household with no labelled timestamp gets ``None``. It has no metric,
    which is not the same as a metric of zero.
    """
    if set(truth) != set(predicted) or (
        probabilities is not None and set(probabilities) != set(truth)
    ):
        raise ValueError(
            "truth, predictions and probabilities must name the same households"
        )
    scores: dict[str, PredictionMetrics | None] = {}
    for household in sorted(truth):
        labels = truth[household]
        if all(label is None for label in labels):
            scores[household] = None
            continue
        scores[household] = prediction_metrics(
            labels,
            predicted[household],
            probabilities[household] if probabilities is not None else None,
            states=states,
        )
    return scores


def recall_of(state: BehaviouralState) -> MetricGetter:
    """Metric getter for one state's recall.

    The value is missing in a household where the state never occurred.
    """
    return lambda metrics: metrics.per_class_recall.get(state)


def household_values(
    scores: Mapping[str, PredictionMetrics | None], metric: str | MetricGetter
) -> dict[str, float | None]:
    """One value of *metric* per household, ``None`` where it does not exist.

    A value is missing when the household was not scored, when the model gave
    no probabilities for a probability-based metric, or, for :func:`recall_of`,
    when the state never occurred there.
    """
    getter: MetricGetter
    if isinstance(metric, str):
        if metric not in _SCALAR_METRICS:
            raise ValueError(
                f"unknown household metric {metric!r}; choose from "
                f"{sorted(_SCALAR_METRICS)} or pass a getter such as recall_of()"
            )
        name = metric
        getter = lambda metrics: getattr(metrics, name)  # noqa: E731
    else:
        getter = metric
    values: dict[str, float | None] = {}
    for household in sorted(scores):
        score = scores[household]
        value = None if score is None else getter(score)
        values[household] = None if value is None or math.isnan(value) else float(value)
    return values


@dataclass(frozen=True)
class HouseholdSummary:
    """A metric across households, each counted once.

    Attributes
    ----------
    households
        Households with a value.
    missing
        Households without one. They are excluded, not counted as zero.
    """

    households: tuple[str, ...]
    missing: tuple[str, ...]
    mean: float | None
    median: float | None
    sd: float | None
    minimum: float | None
    maximum: float | None

    @property
    def n(self) -> int:
        """Number of households with a value."""
        return len(self.households)

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form."""
        return {
            "n": self.n,
            "missing": list(self.missing),
            "mean": self.mean,
            "median": self.median,
            "sd": self.sd,
            "min": self.minimum,
            "max": self.maximum,
        }


def summarise_households(values: HouseholdValues) -> HouseholdSummary:
    """Describe one metric across households, each household weighted equally.

    A household with 20 timestamps and one with 20,000 count the same. The
    standard deviation needs at least two households and is ``None`` otherwise.
    """
    present, missing = _split_present(values)
    array = np.array(list(present.values()), dtype=float)
    return HouseholdSummary(
        households=tuple(present),
        missing=missing,
        mean=float(array.mean()) if array.size else None,
        median=float(np.median(array)) if array.size else None,
        sd=float(array.std(ddof=1)) if array.size > 1 else None,
        minimum=float(array.min()) if array.size else None,
        maximum=float(array.max()) if array.size else None,
    )


# ----------------------------------------------------------------------------
# Across households
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Estimate:
    """One statistic of the household differences, with its uncertainty.

    ``standard_error`` is the standard deviation of the statistic across the
    household resamples. It and ``interval`` are ``None`` when there is only one
    household, since no between-household variability can be estimated.
    """

    value: float
    standard_error: float | None
    interval: Interval | None

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form."""
        return {
            "estimate": self.value,
            "standard_error": self.standard_error,
            "interval": self.interval.to_dict() if self.interval else None,
        }


@dataclass(frozen=True)
class HouseholdComparison:
    """A model against a reference, paired by household.

    Every difference is oriented so that a positive value favours the model.

    Attributes
    ----------
    households
        Households with a value from both models, sorted.
    excluded
        Households missing a value from either model.
    differences
        The oriented difference in each paired household, in household order.
    mean, median
        The mean and median household difference.
    favours_model, favours_reference, tied
        How many households favour each side, or neither.
    effect_size
        Cohen's dz: the mean difference over its standard deviation. ``None``
        with one household, or when every household differs by the same
        amount.
    note
        Anything a reader must know to interpret the intervals.
    """

    households: tuple[str, ...]
    excluded: tuple[str, ...]
    differences: tuple[float, ...]
    mean: Estimate
    median: Estimate
    favours_model: int
    favours_reference: int
    tied: int
    effect_size: float | None
    resamples: int
    seed: int
    note: str | None = None

    @property
    def n(self) -> int:
        """Number of paired households."""
        return len(self.households)

    def proportion(self, side: str) -> float:
        """Share of paired households that favour ``"model"``, ``"reference"`` or ``"tied"``."""
        counts = {
            "model": self.favours_model,
            "reference": self.favours_reference,
            "tied": self.tied,
        }
        if side not in counts:
            raise ValueError(f"side must be one of {sorted(counts)}")
        return counts[side] / self.n

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form."""
        return {
            "n": self.n,
            "households": list(self.households),
            "excluded": list(self.excluded),
            "differences": dict(zip(self.households, self.differences)),
            "mean": self.mean.to_dict(),
            "median": self.median.to_dict(),
            "favours_model": self.favours_model,
            "favours_reference": self.favours_reference,
            "tied": self.tied,
            "proportion_favouring_model": self.proportion("model"),
            "proportion_favouring_reference": self.proportion("reference"),
            "proportion_tied": self.proportion("tied"),
            "effect_size": self.effect_size,
            "resamples": self.resamples,
            "seed": self.seed,
            "note": self.note,
        }


def _split_present(
    values: HouseholdValues,
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Separate households with a finite value, in sorted order, from the rest."""
    present: dict[str, float] = {}
    missing: list[str] = []
    for household in sorted(values):
        value = values[household]
        if value is None or math.isnan(value):
            missing.append(household)
        elif math.isinf(value):
            raise ValueError(f"household {household!r} has an infinite value")
        else:
            present[household] = float(value)
    return present, tuple(missing)


def _estimate(
    differences: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    replicates: np.ndarray | None,
    method: str,
    confidence: float,
) -> tuple[Estimate, bool]:
    """Point estimate, standard error and interval; flags a BCa fallback."""
    value = statistic(differences)
    if replicates is None:
        return Estimate(value, None, None), False
    interval: Interval | None = None
    fell_back = False
    if method == "bca":
        interval = bca_interval(
            replicates, value, jackknife(differences, statistic), confidence
        )
        fell_back = interval is None
    if interval is None:
        interval = percentile_interval(replicates, confidence)
    return Estimate(value, float(replicates.std(ddof=1)), interval), fell_back


def compare_households(
    model: HouseholdValues,
    reference: HouseholdValues,
    *,
    higher_is_better: bool = True,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
    interval: str = "percentile",
) -> HouseholdComparison:
    """Estimate how much a model differs from a reference across households.

    Parameters
    ----------
    model, reference
        One metric value per household, as from :func:`household_values`. Only
        households with a finite value from both are paired. The rest are
        reported in ``excluded``, never imputed.
    higher_is_better
        Orients the differences so that a positive value favours the model:
        model minus reference when true, reference minus model otherwise.
    confidence, resamples, seed
        Bootstrap settings. Households, not timestamps, are resampled, and the
        same resamples are used for the mean and the median.
    interval
        ``"percentile"`` or ``"bca"``. Where a BCa interval is undefined, the
        percentile interval is reported instead and ``note`` says so.

    Raises
    ------
    ValueError
        If no household has a value from both, or a setting is invalid.
    """
    check_settings(confidence, resamples)
    if interval not in INTERVAL_METHODS:
        raise ValueError(f"interval must be one of {INTERVAL_METHODS}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    ours, _ = _split_present(model)
    theirs, _ = _split_present(reference)
    paired = tuple(sorted(set(ours) & set(theirs)))
    excluded = tuple(sorted((set(model) | set(reference)) - set(paired)))
    if not paired:
        raise ValueError("no household has a value from both models")

    sign = 1.0 if higher_is_better else -1.0
    differences = np.array([sign * (ours[h] - theirs[h]) for h in paired])

    notes: list[str] = []
    samples = None
    if differences.size == 1:
        notes.append(
            "one household: between-household variability cannot be estimated, "
            "so no standard error or interval is reported"
        )
    else:
        samples = differences[resample_indices(differences.size, resamples, seed)]

    def mean(values: np.ndarray) -> float:
        return float(np.mean(values))

    def median(values: np.ndarray) -> float:
        return float(np.median(values))

    estimates = []
    for name, statistic, replicate in (
        ("mean", mean, samples.mean(axis=1) if samples is not None else None),
        ("median", median, np.median(samples, axis=1) if samples is not None else None),
    ):
        estimate, fell_back = _estimate(
            differences, statistic, replicate, interval, confidence
        )
        if fell_back:
            notes.append(f"BCa undefined for the {name}; percentile interval reported")
        estimates.append(estimate)

    spread = float(differences.std(ddof=1)) if differences.size > 1 else 0.0
    if differences.size > 1 and spread == 0.0:
        notes.append(
            "every household differs by the same amount; intervals have zero width"
        )
    return HouseholdComparison(
        households=paired,
        excluded=excluded,
        differences=tuple(float(d) for d in differences),
        mean=estimates[0],
        median=estimates[1],
        favours_model=int((differences > 0).sum()),
        favours_reference=int((differences < 0).sum()),
        tied=int((differences == 0).sum()),
        effect_size=float(differences.mean() / spread) if spread > 0 else None,
        resamples=resamples,
        seed=seed,
        note="; ".join(notes) if notes else None,
    )
