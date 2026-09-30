"""Selective prediction: how well a risk signal orders predictions for rejection.

Phase 4 asks which signal, if any, should decide when to abstain. This module
evaluates any candidate: confidence, entropy, model or evidence-channel
disagreement, predictive mismatch, expected loss, or a later diagnostic. It
selects no threshold and fixes no abstention rule. Each signal is evaluated
over a grid of coverage levels, and the curves are the output.

Direction
---------
A signal's direction is always explicit. :data:`HIGHER_IS_RISKIER` rejects the
highest values first, entropy for example. :data:`HIGHER_IS_SAFER` rejects the
lowest first, confidence for example. Nothing assumes either.

Selection
---------
At coverage ``c``, the fraction ``c`` of windows the signal ranks safest is
retained. Windows tied at the boundary are retained in equal part, each with
the fraction of weight that makes the coverage exact. That is the expectation
of breaking ties at random, so a signal carries no information within its ties.
A constant signal therefore performs exactly as random rejection does.

- **Pooled.** The panel curve selects over every household's windows together,
  as one threshold on the signal would. Each household's share of the retained
  windows then follows from the signal.
- **Per household.** Each household's curve retains the fraction ``c`` of its
  own windows.

Metrics are aggregated over the timestamps of the selection.

Uncertainty
-----------
Timestamps within a household are dependent, so they are never resampled.
Households are. Each bootstrap resample draws households with replacement,
selects again at every coverage level over the resampled panel, and recomputes
every metric (``docs/EVALUATION_DESIGN.md``). With one household, no
between-household uncertainty can be estimated, and no interval is reported.

References
----------
- **Random rejection.** Retaining a uniformly random fraction ``c`` keeps, in
  expectation, the fraction ``c`` of every count. Every ratio metric therefore
  stays at its full-coverage value: exactly so for the risk, and to first order
  for the others. The random reference is that value at every coverage.
- **The oracle.** It rejects the windows with the greatest loss first: the best
  any signal could do on these predictions.

Curve summaries
---------------
- **AURC.** The area under the risk-coverage curve, by the trapezoid rule over
  the grid, divided by the grid's span: the mean selective risk over the
  coverage levels. Random rejection's is the full-coverage risk.
- **Excess AURC.** The AURC minus the oracle's: what a perfect ordering would
  still remove.
- **Gain.** ``(AURC_random − AURC) / (AURC_random − AURC_oracle)``: 1 for the
  oracle, 0 for random rejection, negative for a signal worse than random.
  It is undefined when the predictions have no loss to remove.

The AURC averages over coverage levels no deployment would use together. It
summarises how well a signal orders predictions; it is not a deployed risk.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .resampling import check_settings, percentile_interval, resample_indices

#: The layout of a report.
REPORT_FORMAT = "selective-prediction/1"

#: A signal whose higher values mark riskier predictions, rejected first.
HIGHER_IS_RISKIER = "higher_is_riskier"
#: A signal whose higher values mark safer predictions, retained first.
HIGHER_IS_SAFER = "higher_is_safer"
DIRECTIONS = (HIGHER_IS_RISKIER, HIGHER_IS_SAFER)

#: What to do with windows whose signal is missing: refuse the signal, or
#: rank them before every other window for rejection or for retention.
MISSING_POLICIES = ("refuse", "reject_first", "retain_first")

#: The default coverage grid: 5% to 100% in steps of 5%.
DEFAULT_COVERAGES = tuple(round(0.05 * k, 2) for k in range(1, 21))

#: Equal-width confidence bins of the retained predictions' calibration error.
CALIBRATION_BINS = 10

#: Scalar metrics with household-bootstrap intervals on the pooled curve.
INTERVAL_METRICS = ("risk", "error", "balanced_accuracy", "excess_risk")

#: Resamples bootstrapped at once, which bounds memory.
_CHUNK = 64

#: Relative weight below which a retained count is empty.
_EMPTY = 1e-12


# ----------------------------------------------------------------------------
# The scored predictions
# ----------------------------------------------------------------------------
class _Layout:
    """Where each per-window statistic sits in the statistics matrix."""

    def __init__(self, states: int, bins: int) -> None:
        self.count, self.error, self.loss = 0, 1, 2
        self.truth = slice(3, 3 + states)
        self.hits = slice(3 + states, 3 + 2 * states)
        start = 3 + 2 * states
        self.bin_count = slice(start, start + bins)
        self.bin_correct = slice(start + bins, start + 2 * bins)
        self.bin_confidence = slice(start + 2 * bins, start + 3 * bins)
        self.size = start + 3 * bins


@dataclass(frozen=True)
class SelectiveData:
    """The scored predictions every signal is evaluated on.

    Attributes
    ----------
    households
        ``(windows,)``: each window's household.
    truth, predicted
        ``(windows,)``: indices into ``states``.
    states
        The state labels.
    confidence
        ``(windows,)``: the probability the model stated for its predicted
        state, in ``[0, 1]``; or ``None``, and then no calibration is reported.
    loss
        ``(states, states)``: the loss of predicting the column when the row is
        true, non-negative. Defaults to 0-1 loss, and the risk is then the
        error rate.
    """

    households: np.ndarray
    truth: np.ndarray
    predicted: np.ndarray
    states: tuple[str, ...]
    confidence: np.ndarray | None = None
    loss: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate the predictions and freeze them."""
        states = tuple(self.states)
        if not states or len(set(states)) != len(states):
            raise ValueError("states must be distinct and non-empty")
        households = np.asarray(self.households, dtype=str)
        truth = np.asarray(self.truth, dtype=int)
        predicted = np.asarray(self.predicted, dtype=int)
        n = households.size
        if households.ndim != 1 or not n:
            raise ValueError("at least one scored window is required")
        if truth.shape != (n,) or predicted.shape != (n,):
            raise ValueError("truth and predictions need one value per window")
        for name, values in (("truth", truth), ("predicted", predicted)):
            if values.min() < 0 or values.max() >= len(states):
                raise ValueError(f"{name} must index the states")
        confidence = None
        if self.confidence is not None:
            confidence = np.asarray(self.confidence, dtype=float)
            if confidence.shape != (n,) or not np.all(
                (confidence >= 0.0) & (confidence <= 1.0)
            ):
                raise ValueError("confidence needs one probability per window")
        loss = (
            1.0 - np.eye(len(states))
            if self.loss is None
            else np.asarray(self.loss, dtype=float)
        )
        if loss.shape != (len(states), len(states)) or not np.all(
            np.isfinite(loss) & (loss >= 0.0)
        ):
            raise ValueError(
                "loss must be a finite, non-negative (states, states) matrix"
            )
        arrays: tuple[tuple[str, np.ndarray | None], ...] = (
            ("households", households),
            ("truth", truth),
            ("predicted", predicted),
            ("confidence", confidence),
            ("loss", loss),
        )
        for name, array in arrays:
            if array is not None:
                array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "states", states)

    @classmethod
    def from_households(
        cls,
        states: Sequence[str],
        truth: Mapping[str, ArrayLike],
        predicted: Mapping[str, ArrayLike],
        confidence: Mapping[str, ArrayLike] | None = None,
        loss: np.ndarray | None = None,
    ) -> SelectiveData:
        """Predictions given per household; windows are ordered by household."""
        if set(truth) != set(predicted) or (
            confidence is not None and set(confidence) != set(truth)
        ):
            raise ValueError("every household needs truth, predictions and confidence")
        homes = sorted(truth)
        return cls(
            households=np.concatenate(
                [np.full(np.asarray(truth[h]).size, h, dtype=object) for h in homes]
            ).astype(str),
            truth=np.concatenate([np.asarray(truth[h], dtype=int) for h in homes]),
            predicted=np.concatenate(
                [np.asarray(predicted[h], dtype=int) for h in homes]
            ),
            states=tuple(states),
            confidence=(
                np.concatenate([np.asarray(confidence[h], dtype=float) for h in homes])
                if confidence is not None
                else None
            ),
            loss=loss,
        )

    def align(self, values: Mapping[str, ArrayLike]) -> np.ndarray:
        """A signal given per household, in this data's window order."""
        out = np.empty(self.households.size)
        for home in self.names:
            rows = self.households == home
            column = np.asarray(values[home], dtype=float)
            if column.shape != (int(rows.sum()),):
                raise ValueError(f"the signal of {home!r} needs one value per window")
            out[rows] = column
        return out

    @property
    def names(self) -> tuple[str, ...]:
        """The households, sorted."""
        return tuple(sorted(set(self.households.tolist())))

    @property
    def household_index(self) -> np.ndarray:
        """``(windows,)``: each window's position in :attr:`names`."""
        index: np.ndarray = np.searchsorted(np.array(self.names), self.households)
        return index

    @property
    def window_loss(self) -> np.ndarray:
        """``(windows,)``: each prediction's loss."""
        assert self.loss is not None
        values: np.ndarray = self.loss[self.truth, self.predicted]
        return values

    def statistics(self, bins: int = CALIBRATION_BINS) -> tuple[_Layout, np.ndarray]:
        """Each window's additive statistics, from which every metric is a ratio."""
        n, size = self.truth.size, len(self.states)
        layout = _Layout(size, bins)
        x = np.zeros((n, layout.size))
        rows = np.arange(n)
        correct = self.truth == self.predicted
        x[:, layout.count] = 1.0
        x[:, layout.error] = ~correct
        x[:, layout.loss] = self.window_loss
        x[rows, layout.truth.start + self.truth] = 1.0
        x[rows, layout.hits.start + self.truth] = correct
        if self.confidence is not None:
            # Bins are (low, high], as the calibration error elsewhere uses.
            b = np.clip(np.ceil(self.confidence * bins).astype(int) - 1, 0, bins - 1)
            x[rows, layout.bin_count.start + b] = 1.0
            x[rows, layout.bin_correct.start + b] = correct
            x[rows, layout.bin_confidence.start + b] = self.confidence
        return layout, x


@dataclass(frozen=True)
class Signal:
    """A candidate risk signal, with its direction stated.

    Attributes
    ----------
    name
        What the signal is.
    values
        ``(windows,)``, in the data's window order; NaN where missing.
    direction
        :data:`HIGHER_IS_RISKIER` or :data:`HIGHER_IS_SAFER`. Required.
    missing
        One of :data:`MISSING_POLICIES`. A missing value is refused unless a
        policy says where it ranks.
    """

    name: str
    values: np.ndarray
    direction: str
    missing: str = "refuse"

    def __post_init__(self) -> None:
        """Validate the declaration."""
        if not str(self.name).strip():
            raise ValueError("a signal needs a name")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if self.missing not in MISSING_POLICIES:
            raise ValueError(f"missing must be one of {MISSING_POLICIES}")
        values = np.array(self.values, dtype=float)
        if values.ndim != 1:
            raise ValueError("a signal has one value per window")
        if np.isinf(values).any():
            raise ValueError("signal values must be finite or NaN")
        if np.isnan(values).any() and self.missing == "refuse":
            raise ValueError(
                f"the signal {self.name!r} is missing for {int(np.isnan(values).sum())} "
                "windows; state a missing policy"
            )
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def risk(self) -> np.ndarray:
        """Values oriented so that higher is riskier, missing ones placed by the policy."""
        oriented = self.values if self.direction == HIGHER_IS_RISKIER else -self.values
        finite = oriented[~np.isnan(oriented)]
        top = float(finite.max()) + 1.0 if finite.size else 1.0
        bottom = float(finite.min()) - 1.0 if finite.size else -1.0
        placed = top if self.missing == "reject_first" else bottom
        values: np.ndarray = np.where(np.isnan(oriented), placed, oriented)
        return values


# ----------------------------------------------------------------------------
# Selection with exact ties, for any household weights
# ----------------------------------------------------------------------------
class _Ranking:
    """Windows ranked by a risk, safest first, grouped into ties.

    Selecting at any coverage, for any weights of the households, needs only
    each household's running sums of its statistics in this order.
    """

    def __init__(
        self, risk: np.ndarray, household: np.ndarray, x: np.ndarray, homes: int
    ):
        order = np.argsort(risk, kind="stable")
        ranked = risk[order]
        breaks = np.flatnonzero(ranked[1:] != ranked[:-1]) + 1
        starts = np.concatenate([[0], breaks])
        ends = np.concatenate([breaks, [ranked.size]])
        owner = household[order]
        ordered = x[order]
        self.cnt_start = np.zeros((starts.size, homes), dtype=np.int64)
        self.cnt_end = np.zeros((starts.size, homes), dtype=np.int64)
        self.prefix: list[np.ndarray] = []
        for h in range(homes):
            positions = np.flatnonzero(owner == h)
            self.cnt_start[:, h] = np.searchsorted(positions, starts)
            self.cnt_end[:, h] = np.searchsorted(positions, ends)
            running = np.cumsum(ordered[positions], axis=0)
            self.prefix.append(np.vstack([np.zeros((1, x.shape[1])), running]))
        self.sizes = self.cnt_end[-1].astype(float)
        self.totals = np.stack([p[-1] for p in self.prefix])

    def contributions(self, weights: np.ndarray, coverages: np.ndarray) -> np.ndarray:
        """``(resamples, coverages, households, statistics)`` retained sums.

        *weights* is ``(resamples, households)``: how many times each household
        is drawn. The selection is over the weighted panel, ties split exactly.
        """
        weights = np.asarray(weights, dtype=float)
        target = coverages[None, :] * (weights @ self.sizes)[:, None]
        lo = np.zeros(target.shape, dtype=np.int64)
        hi = np.full(target.shape, self.cnt_end.shape[0] - 1, dtype=np.int64)
        while np.any(lo < hi):
            mid = (lo + hi) // 2
            reached = np.einsum("blh,bh->bl", self.cnt_end[mid], weights)
            left = reached >= target
            hi = np.where(left, mid, hi)
            lo = np.where(left, lo, mid + 1)
        start = np.einsum("blh,bh->bl", self.cnt_start[lo], weights)
        end = np.einsum("blh,bh->bl", self.cnt_end[lo], weights)
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.clip(
                np.where(end > start, (target - start) / (end - start), 0.0), 0.0, 1.0
            )
        out = np.zeros(target.shape + (len(self.prefix), self.totals.shape[1]))
        for h, prefix in enumerate(self.prefix):
            below = prefix[self.cnt_start[lo][..., h]]
            tied = prefix[self.cnt_end[lo][..., h]] - below
            out[:, :, h, :] = weights[:, h, None, None] * (
                below + fraction[..., None] * tied
            )
        return out


# ----------------------------------------------------------------------------
# Metrics of a selection
# ----------------------------------------------------------------------------
def _ratio(numerator: np.ndarray, denominator: np.ndarray, scale: Any) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        values: np.ndarray = np.where(
            denominator > _EMPTY * np.maximum(scale, 1.0),
            numerator / denominator,
            np.nan,
        )
    return values


def _metrics(
    layout: _Layout, kept: np.ndarray, total: np.ndarray
) -> dict[str, np.ndarray]:
    """Every metric of retained sums *kept* out of *total*, over the leading axes."""
    n = total[..., layout.count]
    count = kept[..., layout.count]
    truth_kept, truth_all = kept[..., layout.truth], total[..., layout.truth]
    rejected = n - count
    recall = _ratio(kept[..., layout.hits], truth_kept, n[..., None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        balanced = np.nanmean(recall, axis=-1)
    bin_count = kept[..., layout.bin_count]
    bin_gap = np.abs(
        _ratio(kept[..., layout.bin_correct], bin_count, n[..., None])
        - _ratio(kept[..., layout.bin_confidence], bin_count, n[..., None])
    )
    has_confidence = total[..., layout.bin_count].sum(axis=-1) > 0
    shares = _ratio(bin_count, count[..., None], n[..., None])
    ece = np.where(
        has_confidence,
        np.nansum(np.nan_to_num(shares) * np.nan_to_num(bin_gap), -1),
        np.nan,
    )
    mean_confidence = np.where(
        has_confidence,
        _ratio(kept[..., layout.bin_confidence].sum(axis=-1), count, n),
        np.nan,
    )
    error = _ratio(kept[..., layout.error], count, n)
    return {
        "coverage": _ratio(count, n, n),
        "risk": _ratio(kept[..., layout.loss], count, n),
        "error": error,
        "balanced_accuracy": balanced,
        "states_scored": (truth_kept > _EMPTY * np.maximum(n[..., None], 1.0)).sum(-1),
        "state_coverage": _ratio(truth_kept, truth_all, n[..., None]),
        "rejected_composition": _ratio(
            truth_all - truth_kept, rejected[..., None], n[..., None]
        ),
        "rejected_error": _ratio(
            total[..., layout.error] - kept[..., layout.error], rejected, n
        ),
        "calibration_error": ece,
        "mean_confidence": mean_confidence,
        "calibration_gap": mean_confidence - (1.0 - error),
    }


def _area(risk: np.ndarray, coverages: np.ndarray) -> np.ndarray:
    """The trapezoidal mean of a curve over the coverage grid, on the last axis."""
    if coverages.size == 1:
        single: np.ndarray = risk[..., 0]
        return single
    widths = np.diff(coverages)
    area = ((risk[..., 1:] + risk[..., :-1]) / 2.0 * widths).sum(axis=-1)
    values: np.ndarray = area / (coverages[-1] - coverages[0])
    return values


def _gain(area: np.ndarray, random: np.ndarray, oracle: np.ndarray) -> np.ndarray:
    room = random - oracle
    with np.errstate(invalid="ignore", divide="ignore"):
        values: np.ndarray = np.where(room > 1e-12, (random - area) / room, np.nan)
    return values


def check_coverages(coverages: Sequence[float]) -> np.ndarray:
    """A coverage grid: strictly increasing, within ``(0, 1]``, ending at 1."""
    grid = np.asarray(coverages, dtype=float)
    if grid.ndim != 1 or not grid.size:
        raise ValueError("the coverage grid needs at least one level")
    if grid[0] <= 0.0 or grid[-1] != 1.0 or np.any(np.diff(grid) <= 0.0):
        raise ValueError(
            "coverages must increase strictly within (0, 1] and end at 1, where "
            "the random reference is read"
        )
    return grid


# ----------------------------------------------------------------------------
# Evaluating signals
# ----------------------------------------------------------------------------
def _nullable(values: np.ndarray) -> Any:
    array = np.asarray(values, dtype=float)
    out = array.astype(object)
    out[~np.isfinite(array)] = None
    return out.tolist()


def _scalar(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _curve(
    metrics: Mapping[str, np.ndarray], states: Sequence[str], at: Any = ...
) -> dict[str, Any]:
    """A curve's serialisable form: one list per metric, per-state ones by state."""
    out: dict[str, Any] = {}
    for key, values in metrics.items():
        chosen = values[at]
        if key in ("state_coverage", "rejected_composition"):
            out[key] = {s: _nullable(chosen[..., k]) for k, s in enumerate(states)}
        elif key == "states_scored":
            out[key] = [int(v) for v in chosen]
        else:
            out[key] = _nullable(chosen)
    return out


def _interval(
    replicates: np.ndarray, confidence: float
) -> tuple[float | None, float | None]:
    finite = replicates[np.isfinite(replicates)]
    if finite.size < 2:
        return None, None
    interval = percentile_interval(finite, confidence)
    return interval.low, interval.high


def _band(
    estimate: np.ndarray, replicates: np.ndarray | None, confidence: float
) -> dict[str, Any]:
    """An estimate per coverage level with its household-bootstrap interval."""
    if replicates is None:
        return {"estimate": _nullable(estimate), "low": None, "high": None}
    bounds = [_interval(replicates[:, k], confidence) for k in range(estimate.size)]
    return {
        "estimate": _nullable(estimate),
        "low": [b[0] for b in bounds],
        "high": [b[1] for b in bounds],
    }


def _summary_band(
    estimate: float, replicates: np.ndarray | None, confidence: float
) -> dict[str, Any]:
    low, high = (
        _interval(replicates, confidence) if replicates is not None else (None, None)
    )
    return {"estimate": _scalar(estimate), "low": low, "high": high}


@dataclass(frozen=True)
class _Setting:
    coverages: np.ndarray
    confidence: float
    resamples: int
    seed: int
    bins: int


class _Panel:
    """The data's statistics and the references every signal is compared with."""

    def __init__(self, data: SelectiveData, setting: _Setting) -> None:
        self.data, self.setting = data, setting
        self.layout, self.x = data.statistics(setting.bins)
        self.index = data.household_index
        self.homes = len(data.names)
        self.oracle = _Ranking(data.window_loss, self.index, self.x, self.homes)
        self.weights: np.ndarray | None = None
        self._oracle_replicates: np.ndarray | None = None
        if self.homes > 1:
            draws = resample_indices(self.homes, setting.resamples, setting.seed)
            self.weights = np.stack(
                [np.bincount(row, minlength=self.homes) for row in draws]
            ).astype(float)

    def pooled(self, ranking: _Ranking, weights: np.ndarray) -> dict[str, np.ndarray]:
        """Metrics of the pooled selection for each row of household weights."""
        kept = ranking.contributions(weights, self.setting.coverages).sum(axis=2)
        total = weights @ ranking.totals
        return _metrics(self.layout, kept, total[:, None, :])

    def per_household(self, ranking: _Ranking) -> dict[str, np.ndarray]:
        """Metrics of each household's own selection: ``(households, coverages)``."""
        identity = np.eye(self.homes)
        kept = ranking.contributions(identity, self.setting.coverages).sum(axis=2)
        return _metrics(self.layout, kept, ranking.totals[:, None, :])

    def bootstrap(
        self, ranking: _Ranking
    ) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
        """Every pooled metric per resample, and the oracle's risk curve per resample."""
        if self.weights is None:
            return None
        chunks = [
            self.weights[begin : begin + _CHUNK]
            for begin in range(0, self.weights.shape[0], _CHUNK)
        ]
        parts = [self.pooled(ranking, chunk) for chunk in chunks]
        merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        if self._oracle_replicates is None:
            self._oracle_replicates = np.concatenate(
                [self.pooled(self.oracle, chunk)["risk"] for chunk in chunks]
            )
        return merged, self._oracle_replicates


@dataclass(frozen=True)
class SignalEvaluation:
    """One signal's curves, pooled and per household, with their summaries.

    Every array is over the coverage grid, whose last level is full coverage.
    """

    name: str
    direction: str
    missing: str
    missing_windows: int
    coverages: np.ndarray
    states: tuple[str, ...]
    households: tuple[str, ...]
    pooled: Mapping[str, np.ndarray]
    household_share: np.ndarray
    per_household: Mapping[str, np.ndarray]
    replicates: Mapping[str, np.ndarray] | None
    oracle_risk: np.ndarray
    oracle_replicates: np.ndarray | None
    household_oracle_risk: np.ndarray
    confidence: float
    resamples: int

    # ------------------------------------------------------------------
    @property
    def aurc(self) -> float:
        """The pooled curve's area under the risk-coverage curve, per unit coverage."""
        return float(_area(self.pooled["risk"], self.coverages))

    @property
    def oracle_aurc(self) -> float:
        """The oracle's AURC on the same predictions."""
        return float(_area(self.oracle_risk, self.coverages))

    @property
    def random_aurc(self) -> float:
        """Random rejection's AURC: the full-coverage risk."""
        return float(self.pooled["risk"][-1])

    @property
    def gain(self) -> float | None:
        """``(AURC_random − AURC) / (AURC_random − AURC_oracle)``, or ``None``."""
        return _scalar(
            _gain(
                np.array(self.aurc),
                np.array(self.random_aurc),
                np.array(self.oracle_aurc),
            )
        )

    def household_summaries(self) -> dict[str, np.ndarray]:
        """Each household's AURC, excess AURC and gain, in household order."""
        risk = self.per_household["risk"]
        area = _area(risk, self.coverages)
        oracle = _area(self.household_oracle_risk, self.coverages)
        return {
            "aurc": area,
            "excess_aurc": area - oracle,
            "gain": _gain(area, risk[:, -1], oracle),
        }

    def _across(self, values: np.ndarray, seed: int) -> dict[str, Any]:
        """Household values described with equal weight, and their mean's interval."""
        finite = values[np.isfinite(values)]
        out: dict[str, Any] = {
            "n": int(finite.size),
            "mean": _scalar(finite.mean()) if finite.size else None,
            "median": _scalar(np.median(finite)) if finite.size else None,
            "min": _scalar(finite.min()) if finite.size else None,
            "max": _scalar(finite.max()) if finite.size else None,
            "low": None,
            "high": None,
        }
        if finite.size > 1:
            draws = resample_indices(finite.size, self.resamples, seed)
            out["low"], out["high"] = _interval(
                finite[draws].mean(axis=1), self.confidence
            )
        return out

    def to_dict(self, seed: int = 0) -> dict[str, Any]:
        """Return the serialisable form a report carries."""
        replicates = self.replicates
        pooled = _curve(self.pooled, self.states)
        excess = self.pooled["risk"] - self.pooled["risk"][-1]
        excess_replicates = (
            replicates["risk"] - replicates["risk"][:, -1:]
            if replicates is not None
            else None
        )
        bands = {
            key: _band(
                self.pooled[key],
                replicates[key] if replicates is not None else None,
                self.confidence,
            )
            for key in ("risk", "error", "balanced_accuracy")
        }
        bands["excess_risk"] = _band(excess, excess_replicates, self.confidence)
        share = self.household_share
        summaries = self.household_summaries()
        area_replicates = oracle_replicates = gain_replicates = None
        if replicates is not None and self.oracle_replicates is not None:
            area_replicates = _area(replicates["risk"], self.coverages)
            oracle_replicates = _area(self.oracle_replicates, self.coverages)
            gain_replicates = _gain(
                area_replicates, replicates["risk"][:, -1], oracle_replicates
            )
        per_household = self.per_household
        return {
            "direction": self.direction,
            "missing": self.missing,
            "missing_windows": self.missing_windows,
            "pooled": {
                **pooled,
                "intervals": bands,
                "household_coverage": {
                    "min": _nullable(share.min(axis=0)),
                    "median": _nullable(np.median(share, axis=0)),
                    "max": _nullable(share.max(axis=0)),
                },
            },
            "summary": {
                "aurc": _summary_band(self.aurc, area_replicates, self.confidence),
                "excess_aurc": _summary_band(
                    self.aurc - self.oracle_aurc,
                    (
                        area_replicates - oracle_replicates
                        if area_replicates is not None and oracle_replicates is not None
                        else None
                    ),
                    self.confidence,
                ),
                "gain": _summary_band(
                    self.gain if self.gain is not None else math.nan,
                    gain_replicates,
                    self.confidence,
                ),
                "random_aurc": _scalar(self.random_aurc),
                "oracle_aurc": _scalar(self.oracle_aurc),
                "undefined_gain_resamples": (
                    int((~np.isfinite(gain_replicates)).sum())
                    if gain_replicates is not None
                    else None
                ),
            },
            "households": {
                home: {
                    **_curve(per_household, self.states, (h, ...)),
                    "pooled_coverage": _nullable(share[h]),
                    "aurc": _scalar(summaries["aurc"][h]),
                    "excess_aurc": _scalar(summaries["excess_aurc"][h]),
                    "gain": _scalar(summaries["gain"][h]),
                }
                for h, home in enumerate(self.households)
            },
            "across_households": {
                **{
                    key: [
                        self._across(per_household[key][:, k], seed + k)
                        for k in range(self.coverages.size)
                    ]
                    for key in ("risk", "balanced_accuracy")
                },
                "excess_risk": [
                    self._across(
                        per_household["risk"][:, k] - per_household["risk"][:, -1],
                        seed + k,
                    )
                    for k in range(self.coverages.size)
                ],
                **{
                    key: self._across(values, seed) for key, values in summaries.items()
                },
            },
        }


def evaluate_signal(
    data: SelectiveData,
    signal: Signal,
    *,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
    bins: int = CALIBRATION_BINS,
) -> SignalEvaluation:
    """Evaluate one signal's selective prediction on *data*.

    Parameters
    ----------
    data
        The scored predictions.
    signal
        The candidate, with its direction and missing policy.
    coverages
        The grid, strictly increasing within ``(0, 1]`` and ending at 1.
    confidence, resamples, seed
        The household bootstrap. The same seed draws the same households for
        every signal, so their intervals are comparable.
    bins
        Confidence bins of the retained predictions' calibration error.
    """
    grid = check_coverages(coverages)
    check_settings(confidence, resamples)
    return _evaluate(
        _Panel(data, _Setting(grid, confidence, resamples, seed, bins)), signal
    )


def _evaluate(panel: _Panel, signal: Signal) -> SignalEvaluation:
    data, grid = panel.data, panel.setting.coverages
    if signal.values.shape != data.truth.shape:
        raise ValueError(f"the signal {signal.name!r} needs one value per window")
    ranking = _Ranking(signal.risk(), panel.index, panel.x, panel.homes)
    ones = np.ones((1, panel.homes))
    contributions = ranking.contributions(ones, grid)[0]
    pooled = _metrics(
        panel.layout, contributions.sum(axis=1), ranking.totals.sum(axis=0)
    )
    household_share = (
        contributions[:, :, panel.layout.count] / ranking.sizes[None, :]
    ).T
    bootstrap = panel.bootstrap(ranking)
    oracle_pooled = panel.pooled(panel.oracle, ones)
    oracle_household = panel.per_household(panel.oracle)
    return SignalEvaluation(
        name=signal.name,
        direction=signal.direction,
        missing=signal.missing,
        missing_windows=int(np.isnan(signal.values).sum()),
        coverages=grid,
        states=data.states,
        households=data.names,
        pooled=pooled,
        household_share=household_share,
        per_household=panel.per_household(ranking),
        replicates=bootstrap[0] if bootstrap is not None else None,
        oracle_risk=oracle_pooled["risk"][0],
        oracle_replicates=bootstrap[1] if bootstrap is not None else None,
        household_oracle_risk=oracle_household["risk"],
        confidence=panel.setting.confidence,
        resamples=panel.setting.resamples,
    )


def evaluate_signals(
    data: SelectiveData,
    signals: Sequence[Signal],
    *,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
    bins: int = CALIBRATION_BINS,
) -> SelectiveReport:
    """Evaluate several signals on the same predictions, with shared resamples."""
    names = [s.name for s in signals]
    if not signals or len(set(names)) != len(names):
        raise ValueError("signals must be at least one, with distinct names")
    grid = check_coverages(coverages)
    check_settings(confidence, resamples)
    panel = _Panel(data, _Setting(grid, confidence, resamples, seed, bins))
    evaluations = [_evaluate(panel, signal) for signal in signals]
    return SelectiveReport(_payload(data, panel, evaluations, seed))


def _payload(
    data: SelectiveData,
    panel: _Panel,
    evaluations: Sequence[SignalEvaluation],
    seed: int,
) -> dict[str, Any]:
    setting = panel.setting
    full = _metrics(
        panel.layout, panel.oracle.totals.sum(axis=0), panel.oracle.totals.sum(axis=0)
    )
    oracle = panel.pooled(panel.oracle, np.ones((1, panel.homes)))
    counts = np.bincount(panel.index, minlength=panel.homes)
    return {
        "format": REPORT_FORMAT,
        "unit": "household",
        "coverages": setting.coverages.tolist(),
        "states": list(data.states),
        "loss": np.asarray(data.loss).tolist(),
        "calibration_bins": setting.bins,
        "bootstrap": {
            "resamples": setting.resamples if panel.homes > 1 else 0,
            "confidence": setting.confidence,
            "seed": setting.seed,
            "method": "percentile",
            "resampled": "households",
        },
        "households": {
            home: {
                "windows": int(counts[h]),
                "states": {
                    s: int(((panel.index == h) & (data.truth == k)).sum())
                    for k, s in enumerate(data.states)
                },
            }
            for h, home in enumerate(data.names)
        },
        "reference": {
            "random": {
                **{
                    key: _scalar(full[key])
                    for key in (
                        "risk",
                        "error",
                        "balanced_accuracy",
                        "calibration_error",
                        "mean_confidence",
                        "calibration_gap",
                    )
                },
                "states_scored": int(full["states_scored"]),
            },
            "oracle": {
                "risk": _nullable(oracle["risk"][0]),
                "aurc": _scalar(_area(oracle["risk"][0], setting.coverages)),
            },
        },
        "signals": {e.name: e.to_dict(seed) for e in evaluations},
    }


# ----------------------------------------------------------------------------
# In an experiment record
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SelectiveReport:
    """What an experiment record keeps of a selective-prediction evaluation.

    It holds the serialised report, validated on construction.
    """

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate the payload."""
        problems = validate_selective(self.payload)
        if problems:
            raise ValueError("; ".join(problems))

    @property
    def signals(self) -> tuple[str, ...]:
        """The evaluated signals' names."""
        return tuple(self.payload["signals"])

    def to_dict(self) -> dict[str, Any]:
        """Return the serialisable form an experiment record carries."""
        return dict(self.payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SelectiveReport:
        """Rebuild a report written by :meth:`to_dict`."""
        return cls(dict(payload))


def _in_unit(values: Any) -> bool:
    return all(v is None or 0.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in values)


def _check_band(band: Any, length: int, where: str, problems: list[str]) -> None:
    if not isinstance(band, Mapping) or set(band) != {"estimate", "low", "high"}:
        problems.append(f"{where} must have estimate, low and high")
        return
    if len(band["estimate"]) != length:
        problems.append(f"{where}.estimate needs one value per coverage level")
    if band["low"] is None or band["high"] is None:
        return
    for low, high in zip(band["low"], band["high"]):
        if low is not None and high is not None and low > high + 1e-12:
            problems.append(f"{where} has an interval whose low exceeds its high")
            break


def validate_selective(payload: Mapping[str, Any]) -> list[str]:
    """Every problem with a record's selective-prediction section."""
    expected = {
        "format",
        "unit",
        "coverages",
        "states",
        "loss",
        "calibration_bins",
        "bootstrap",
        "households",
        "reference",
        "signals",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        return [f"selective_prediction must have exactly {sorted(expected)}"]
    problems: list[str] = []
    if payload["format"] != REPORT_FORMAT:
        problems.append(f"selective_prediction.format must be {REPORT_FORMAT!r}")
    if payload["unit"] != "household":
        problems.append("selective_prediction.unit must be 'household'")
    try:
        grid = check_coverages(payload["coverages"])
    except (ValueError, TypeError) as exc:
        return problems + [f"selective_prediction.coverages: {exc}"]
    length = grid.size
    states = payload["states"]
    if not isinstance(states, list) or not states or len(set(states)) != len(states):
        problems.append("selective_prediction.states must be distinct and non-empty")
        return problems
    bootstrap = payload["bootstrap"]
    if not isinstance(bootstrap, Mapping) or bootstrap.get("resampled") != "households":
        problems.append("selective_prediction.bootstrap must resample households")
    if not isinstance(payload["households"], Mapping) or not payload["households"]:
        problems.append("selective_prediction.households must not be empty")
    signals = payload["signals"]
    if not isinstance(signals, Mapping) or not signals:
        problems.append("selective_prediction.signals must not be empty")
        return problems
    for name, entry in signals.items():
        where = f"selective_prediction.signals.{name}"
        if not isinstance(entry, Mapping):
            problems.append(f"{where} must be an object")
            continue
        if entry.get("direction") not in DIRECTIONS:
            problems.append(f"{where}.direction must be one of {list(DIRECTIONS)}")
        if entry.get("missing") not in MISSING_POLICIES:
            problems.append(f"{where}.missing must be one of {list(MISSING_POLICIES)}")
        pooled = entry.get("pooled")
        if not isinstance(pooled, Mapping):
            problems.append(f"{where}.pooled must be an object")
            continue
        for key in ("coverage", "error", "balanced_accuracy", "rejected_error"):
            values = pooled.get(key)
            if not isinstance(values, list) or len(values) != length:
                problems.append(
                    f"{where}.pooled.{key} needs one value per coverage level"
                )
            elif not _in_unit(values):
                problems.append(f"{where}.pooled.{key} must lie in [0, 1]")
        for key in ("state_coverage", "rejected_composition"):
            table = pooled.get(key)
            if not isinstance(table, Mapping) or set(table) != set(states):
                problems.append(f"{where}.pooled.{key} must cover every state")
            elif not all(_in_unit(v) for v in table.values()):
                problems.append(f"{where}.pooled.{key} must lie in [0, 1]")
        intervals = pooled.get("intervals")
        if not isinstance(intervals, Mapping) or set(intervals) != set(
            INTERVAL_METRICS
        ):
            problems.append(
                f"{where}.pooled.intervals must cover {list(INTERVAL_METRICS)}"
            )
        else:
            for key, band in intervals.items():
                _check_band(band, length, f"{where}.pooled.intervals.{key}", problems)
        households = entry.get("households")
        if not isinstance(households, Mapping) or set(households) != set(
            payload["households"]
        ):
            problems.append(f"{where}.households must cover every household")
    return problems
