"""Posterior predictive checks of the fitted hurdle channel observation model.

The fitted hurdle model (Phase 3.3 follow-up) improved calibration, but its
active-window counts are over-dispersed and ``home_active`` recall fell. Before
another distribution is introduced, this diagnostic locates where the model is
misspecified. It compares observed channel counts with the model's predictive
distribution for each household, state and channel with enough data. It
changes no model and no inference.

Every setting and the rule that reads the result are declared in
:class:`PredictiveProtocol` and frozen in a committed file before any household
is examined.

Two references
--------------
Each cell, one household's labelled windows of one state on one channel, is
checked against two hurdle models:

``own``
    The hurdle fitted to the cell itself: ``π`` is its share of silent
    windows and ``μ`` solves the zero-truncated Poisson mean for its active
    windows. It reproduces the cell's silence and mean exactly, so those are
    not judged. Anything else it misses is a failure of the family itself,
    whatever the household.
``population``
    The fold's population hurdle, fitted on the training households as in the
    Phase 3.3 follow-up and used unchanged by inference. Its misses combine
    the family's with differences between households.

The statistics
--------------
For each cell and reference, the observed value of each statistic is compared
with its predictive distribution, from replicate data of the same windows
drawn from the reference:

- ``silence``: the share of silent windows;
- ``mean`` and ``variance``: of the count over every window;
- ``active_mean``: the mean count of an active window;
- ``active_dispersion``: the variance of an active window's count over the
  zero-truncated Poisson's;
- ``q90`` and ``q99``: upper quantiles of an active window's count;
- ``tail_excess``: active windows above the predicted 99th percentile;
- ``quiet_runs``: windows in runs of at least ``quiet_run_windows`` silent
  windows;
- ``bursts``: windows in runs of at least ``burst_windows`` active windows.

Runs are counted within stretches of consecutive labelled windows of the
state. The model draws each window independently given the state, so an excess
of either run is dependence in time that no count distribution can produce.

For ``own``, each replicate is refitted before its statistics are computed, a
parametric bootstrap. A correctly specified model then puts the observed value
inside the replicates' central band as often as the band's coverage.

Households, never windows, are the unit across households. Each household's
value for a group is the mean of its cells' log ratios, and the summaries
resample households.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy.stats import poisson

from ..evaluation.households import compare_households, summarise_households
from ..evaluation.provenance import ExperimentRecord, InputArtifact, ModelRecord
from ..fusion.regime import ONLINE, NotEnumerated
from ..states.ontology import BehaviouralState, StateOntology
from .casas import CasasRecording
from .channel_models import (
    HURDLE,
    FittedChannels,
    HurdleChannel,
    combine_statistics,
    home_statistics,
    household_channel_counts,
    truncated_poisson_rate,
)
from .information_sets import EvidenceResolution
from .matched_evaluation import HouseholdSplit, _finite
from .recoverable_gap import FrozenSplits

PROTOCOL_SCHEMA = "predictive-checks-protocol/1"
RESULT_SCHEMA = "predictive-checks-results/1"

S = BehaviouralState

OWN = "own"
POPULATION = "population"
REFERENCES = (OWN, POPULATION)

STATISTICS = (
    "silence",
    "mean",
    "variance",
    "active_mean",
    "active_dispersion",
    "q90",
    "q99",
    "tail_excess",
    "quiet_runs",
    "bursts",
)

#: Statistics of an active window's count, which need enough active windows.
ACTIVE_STATISTICS = frozenset(
    {"active_mean", "active_dispersion", "q90", "q99", "tail_excess"}
)

#: Statistics counting runs, which need enough runs to compare.
RUN_STATISTICS = frozenset({"quiet_runs", "bursts"})

#: Statistics the household's own hurdle reproduces exactly, so not judged by it.
FITTED_BY_OWN = frozenset({"silence", "mean", "active_mean"})

#: The quantile each quantile statistic reads.
QUANTILES = {"q90": 0.9, "q99": 0.99}

#: The states the conclusions read: those that are common in every home.
COMMON_STATES = (S.AWAY, S.HOME_ACTIVE, S.HOME_INACTIVE, S.SLEEPING)

#: Lower edges of the count bins the distribution comparison uses; the last is open.
BIN_EDGES = (0, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64)

#: What each conclusion is about.
CONCLUSION_STATISTICS = {
    "active_dispersion": "active_dispersion",
    "quiet_runs": "quiet_runs",
    "bursts": "bursts",
}

#: The next model family each pattern points to, as declared before any household was examined.
ROUTES = {
    "temporal": "within-state temporal dependence: activity sub-states or a "
    "Markov-modulated emission within each state; no marginal count family "
    "produces runs beyond independence given the state, and a latent intensity "
    "that varies over time also over-disperses the counts",
    "dispersion": "an over-dispersed zero-truncated count model for the active "
    "part, such as the zero-truncated negative binomial, keeping the hurdle's "
    "silence parameter",
}

CRITERIA: dict[str, str] = {
    "cell": "consistent: the observed log ratio lies within the central band of "
    "the replicates' log ratios; above or below: outside it, by at least the "
    "minimal ratio in that direction; small: outside it by less",
    "across": "positive: the household mean log ratio is at least log of the "
    "minimal ratio and its lower bound above 0; negative: at most minus that, "
    "upper bound below 0; negligible: the whole interval within plus or minus "
    "it; uncertain: anything else; insufficient: fewer than min_households "
    "households",
    "systematic": "positive in at least three of the four common states and "
    "negative in none, own reference",
    "partial": "positive in one or two common states, own reference",
    "absent": "negligible or negative in every common state, own reference",
    "inconclusive": "anything else",
    "routing": "a systematic excess of quiet runs or bursts points to "
    "within-state temporal dependence, which would also over-disperse the "
    "counts; otherwise systematic active dispersion points to an over-dispersed "
    "zero-truncated count model; otherwise no new family is indicated",
    "roadmap": "the roadmap names a next model family only if the routing "
    "indicates one",
}

PREDICTIVE_METRIC_DEFINITIONS: dict[str, str] = {
    "cell": "One household's labelled windows of one state on one instrumented "
    "channel, in time order.",
    "stretch": "A maximal run of consecutive labelled windows of the cell's "
    "state; runs are counted within stretches.",
    "silence": "Share of the cell's windows with no activation; compared as a log "
    "odds ratio with half a window added to each side.",
    "mean": "Mean count over the cell's windows.",
    "variance": "Variance of the count over the cell's windows, with one degree of "
    "freedom removed; the hurdle's is (1 - pi) (v + m^2) - ((1 - pi) m)^2 for the "
    "zero-truncated Poisson's mean m and variance v.",
    "active_mean": "Mean count of the cell's active windows; the model's is "
    "mu / (1 - exp(-mu)).",
    "active_dispersion": "Variance of the active windows' counts over the "
    "zero-truncated Poisson's, m (1 + mu - m); a value above one means the "
    "model is under-dispersed.",
    "q90": "The 90th percentile of the active windows' counts, the smallest count "
    "whose empirical share at or below it reaches 0.9, over the model's.",
    "q99": "The same at the 99th percentile.",
    "tail_excess": "Active windows above the model's 99th percentile, observed "
    "over expected, with half a window added to each.",
    "quiet_runs": "Windows in maximal runs of at least quiet_run_windows silent "
    "windows within stretches, observed over the replicates' mean, with half a "
    "window added to each.",
    "bursts": "Windows in maximal runs of at least burst_windows active windows "
    "within stretches, observed over the replicates' mean, with half a window "
    "added to each.",
    "log_ratio": "Log of the observed statistic over the reference's.",
    "band": "The central interval of the replicates' log ratios, at the declared "
    "coverage.",
    "household_value": "For a group of cells, such as a state, the mean of one "
    "household's cell log ratios in the group.",
    "variance_mean_slope": "Least-squares slope of log active variance on log "
    "active mean across one household's cells with enough active windows: "
    "about 1 for Poisson-like counts, approaching 2 when the excess variance "
    "grows with the square of the mean.",
}


# ----------------------------------------------------------------------------
# The zero-truncated Poisson and the hurdle
# ----------------------------------------------------------------------------
def ztp_moments(rate: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of the zero-truncated Poisson with *rate*."""
    mu = np.asarray(rate, dtype=float)
    mean = mu / -np.expm1(-mu)
    return mean, mean * (1.0 + mu - mean)


def ztp_quantile(rate: float, q: float) -> int:
    """The smallest count whose zero-truncated Poisson distribution reaches *q*."""
    floor = math.exp(-rate)
    return max(1, int(poisson.ppf(floor + q * (1.0 - floor), rate)))


def ztp_tail(rate: float, count: int) -> float:
    """``P(X > count)`` under the zero-truncated Poisson with *rate*."""
    return float(poisson.sf(count, rate) / -math.expm1(-rate))


def hurdle_bins(silence: float, rate: float, edges: Sequence[int]) -> list[float]:
    """The hurdle's probability of each count bin; the last bin is open."""
    floor = -math.expm1(-rate)
    probabilities = [silence]
    for low, high in zip(edges[1:], [*edges[2:], None], strict=True):
        upper = 1.0 if high is None else poisson.cdf(high - 1, rate)
        lower = poisson.cdf(low - 1, rate)
        probabilities.append((1.0 - silence) * float(upper - lower) / floor)
    return probabilities


def observed_bins(counts: np.ndarray, edges: Sequence[int]) -> list[int]:
    """How many windows fall in each count bin; the last bin is open."""
    upper = [*edges[1:], np.inf]
    return [
        int(np.sum((counts >= low) & (counts < high)))
        for low, high in zip(edges, upper, strict=True)
    ]


def sample_hurdle(
    rng: np.random.Generator, silence: float, rate: float, shape: tuple[int, ...]
) -> np.ndarray:
    """Counts drawn independently from the hurdle model.

    An active window's count is the Poisson inverse distribution function of a
    uniform draw above ``P(0)``, which is exactly the zero-truncated Poisson.
    The distribution function is tabulated once, to where its tail is below
    ``1e-15``.
    """
    active = rng.random(shape) >= silence
    floor = math.exp(-rate)
    top = int(rate + 40.0 * math.sqrt(rate) + 60.0)
    table = np.cumsum(poisson.pmf(np.arange(top + 1), rate))
    uniforms = rng.uniform(floor, 1.0, size=int(active.sum()))
    draws = np.searchsorted(table, uniforms, side="left")
    counts = np.zeros(shape, dtype=float)
    counts[active] = np.maximum(draws, 1)
    return counts


def run_windows(flags: np.ndarray, stretch: np.ndarray, length: int) -> np.ndarray:
    """Windows in maximal runs of at least *length* true windows, per row.

    *flags* is ``(rows, windows)``, or one row. *stretch* labels each window's
    stretch, and a run never continues across a change of stretch. Counting
    windows rather than runs keeps the statistic monotone in clustering: runs
    that merge into longer ones are fewer, but hold more windows.
    """
    flags = np.atleast_2d(np.asarray(flags, dtype=bool))
    rows = flags.shape[0]
    stretch = np.asarray(stretch)
    new = np.concatenate([[True], stretch[1:] != stretch[:-1]])
    before = np.concatenate([np.zeros((rows, 1), dtype=bool), flags[:, :-1]], axis=1)
    starts = flags & (~before | new)
    ids = np.where(flags, np.cumsum(starts.ravel()).reshape(flags.shape), 0)
    lengths = np.bincount(ids.ravel())
    lengths[0] = 0
    found: np.ndarray = ((lengths[ids] >= length) & flags).sum(axis=1)
    return found


def stretches_of(rows: np.ndarray) -> np.ndarray:
    """A stretch label for each row: it changes wherever rows are not consecutive."""
    rows = np.asarray(rows)
    if rows.size == 0:
        return np.zeros(0, dtype=int)
    labels: np.ndarray = np.concatenate([[0], np.cumsum(np.diff(rows) != 1)])
    return labels


# ----------------------------------------------------------------------------
# The protocol
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictiveProtocol:
    """Everything the diagnostic fixes before any household is examined."""

    folds: tuple[HouseholdSplit, ...]
    splits_sha256: str
    resolution: EvidenceResolution = field(default_factory=EvidenceResolution)
    pseudo_windows: float = 12.0
    min_windows: int = 50
    min_active: int = 30
    min_runs: float = 5.0
    quiet_run_windows: int = 12
    burst_windows: int = 3
    replicates: int = 200
    band: float = 0.95
    minimal_ratio: float = 1.25
    common_states: tuple[BehaviouralState, ...] = COMMON_STATES
    min_households: int = 5
    resamples: int = 10_000
    confidence: float = 0.95
    seed: int = 0
    bin_edges: tuple[int, ...] = BIN_EDGES
    name: str = "phase3-hurdle-predictive-checks"

    def __post_init__(self) -> None:
        """Validate the declaration."""
        folds = tuple(self.folds)
        if not folds:
            raise ValueError("at least one fold is required")
        tested = [home for fold in folds for home in fold.test]
        if len(tested) != len(set(tested)) or len(tested) < 2:
            raise ValueError("each household must be held out exactly once")
        if not isinstance(self.splits_sha256, str) or len(self.splits_sha256) != 64:
            raise ValueError("splits_sha256 must be a SHA-256 hex digest")
        for name in (
            "min_windows",
            "min_active",
            "quiet_run_windows",
            "burst_windows",
            "replicates",
            "min_households",
            "resamples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.replicates < 40:
            raise ValueError("replicates must be at least 40 to place a band")
        for name in ("pseudo_windows", "min_runs"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.minimal_ratio) or self.minimal_ratio <= 1.0:
            raise ValueError("minimal_ratio must exceed 1")
        for name in ("band", "confidence"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
        common = tuple(BehaviouralState(s) for s in self.common_states)
        if not common or len(set(common)) != len(common):
            raise ValueError("common_states must be distinct and non-empty")
        if any(s not in StateOntology().states for s in common):
            raise ValueError("every common state must be an ontology state")
        edges = tuple(int(e) for e in self.bin_edges)
        if edges[:2] != (0, 1) or list(edges) != sorted(set(edges)):
            raise ValueError("bin_edges must start 0, 1 and increase")
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "common_states", common)
        object.__setattr__(self, "bin_edges", edges)

    @property
    def homes(self) -> tuple[str, ...]:
        """Every household, each held out once, sorted."""
        return tuple(sorted(home for fold in self.folds for home in fold.test))

    @property
    def minimal_log_ratio(self) -> float:
        """The minimal log ratio a verdict reads."""
        return math.log(self.minimal_ratio)

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        return {
            "schema": PROTOCOL_SCHEMA,
            "name": self.name,
            "households": {
                "splits_file": "artifacts/phase1/household_splits.json",
                "splits_sha256": self.splits_sha256,
                "folds": [
                    {**fold.to_dict(), "sha256": fold.sha256()} for fold in self.folds
                ],
                "use": "the development panel only; each household is examined "
                "once, against its own hurdle fit and against the population "
                "fitted on its fold's training households",
            },
            "data": {
                "counts": "each instrumented channel's activations in each "
                f"{self.resolution.step.total_seconds() / 60.0:g}-minute window, "
                "as the fitted channel models read them",
                "labels": "truth_series of each recording's annotations; "
                "unlabelled windows are not used",
                "external": "the frozen external cohort is not touched",
            },
            "model": {
                "family": "hurdle: silence with probability pi, otherwise a "
                "zero-truncated Poisson count with rate mu, per channel and state",
                "inference": "unchanged: no model is refitted for inference and no "
                "state is estimated",
            },
            "references": {
                OWN: "the hurdle fitted to the cell alone: pi its share of silent "
                "windows, mu solved from its active windows' mean; it reproduces "
                "the cell's silence, mean and active mean exactly, so those are "
                "not judged against it",
                POPULATION: "the fold's population hurdle fitted on its training "
                f"households with {self.pseudo_windows:g} pseudo-windows, as in "
                "the Phase 3.3 follow-up, applied unchanged",
            },
            "pseudo_windows": self.pseudo_windows,
            "statistics": {
                name: PREDICTIVE_METRIC_DEFINITIONS[name] for name in STATISTICS
            },
            "quantiles": dict(QUANTILES),
            "runs": {
                "quiet_run_windows": self.quiet_run_windows,
                "burst_windows": self.burst_windows,
                "within": PREDICTIVE_METRIC_DEFINITIONS["stretch"],
            },
            "replicates": {
                "count": self.replicates,
                "own": "each replicate drawn from the cell's own fit and refitted "
                "before its statistics are computed: a parametric bootstrap",
                POPULATION: "each replicate drawn from the population hurdle, not "
                "refitted: the predictive distribution for a new household",
                "band": self.band,
                "seed": "numpy default_rng of (seed, household, state, channel, "
                "reference) indices",
            },
            "sufficiency": {
                "min_windows": self.min_windows,
                "min_active": self.min_active,
                "min_runs": self.min_runs,
                "rule": "a cell enters with at least min_windows labelled windows; "
                "active-window statistics need min_active active windows; a run "
                "statistic needs its observed or predicted windows in runs to "
                "reach min_runs times the run length; the own reference needs a "
                "silent and an active window",
            },
            "distribution": {
                "bin_edges": list(self.bin_edges),
                "rule": "observed windows in each count bin against the "
                "reference's expected windows; the last bin is open",
            },
            "groups": {
                "state": "each state",
                "channel": "each channel",
                "room": "each room, pooling a room's channels within a household",
                "modality": "motion or door",
                "state_channel": "each state and channel: descriptive, no interval",
                "household": "every household's cells, listed",
            },
            "variance_mean": PREDICTIVE_METRIC_DEFINITIONS["variance_mean_slope"],
            "common_states": [s.value for s in self.common_states],
            "minimal_ratio": self.minimal_ratio,
            "min_households": self.min_households,
            "criteria": dict(CRITERIA),
            "routes": dict(ROUTES),
            "bootstrap": {
                "unit": "household",
                "statistics": "mean and median household value",
                "resamples": self.resamples,
                "confidence": self.confidence,
                "interval": "percentile",
            },
            "seeds": {"replicates_and_bootstrap": self.seed},
            "tuning": "none: every threshold is declared; nothing is fitted on the "
            "statistics it is judged by",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_protocol(splits: FrozenSplits) -> PredictiveProtocol:
    """The protocol as declared for the development panel's frozen folds."""
    return PredictiveProtocol(splits.folds, splits.sha256)


def check_frozen_protocol(protocol: PredictiveProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after the diagnostic begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


# ----------------------------------------------------------------------------
# One cell
# ----------------------------------------------------------------------------
def own_parameters(counts: np.ndarray) -> tuple[float, float] | None:
    """The hurdle fitted to *counts* alone, or ``None`` without a silent and an active window."""
    silent = float(np.mean(counts == 0))
    active = counts[counts > 0]
    if active.size == 0 or silent == 0.0:
        return None
    return silent, truncated_poisson_rate(float(active.mean()))


def _batch(
    counts: np.ndarray,
    stretch: np.ndarray,
    silence: np.ndarray,
    rate: np.ndarray,
    protocol: PredictiveProtocol,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each statistic's observed and predicted value, per row of *counts*.

    *silence* and *rate* are each row's reference parameters. A value that does
    not exist, such as a quantile with too few active windows, is NaN. A run
    statistic's prediction is left NaN: it comes from the replicates.
    """
    counts = np.atleast_2d(counts)
    rows, size = counts.shape
    zero = counts == 0
    active = size - zero.sum(axis=1)
    total = counts.sum(axis=1)
    squares = (counts * counts).sum(axis=1)
    enough = active >= protocol.min_active
    with np.errstate(divide="ignore", invalid="ignore"):
        active_mean = total / active
        active_variance = (squares - total * total / active) / (active - 1)
    m, v = ztp_moments(rate)
    predicted_mean = (1.0 - silence) * m
    predicted_variance = (1.0 - silence) * (v + m * m) - predicted_mean**2
    floor = np.exp(-rate)
    predicted_quantile = {
        name: np.maximum(1.0, poisson.ppf(floor + q * (1.0 - floor), rate))
        for name, q in QUANTILES.items()
    }
    top = predicted_quantile["q99"]
    tail = poisson.sf(top, rate) / -np.expm1(-rate)
    observed_quantile = {name: np.full(rows, np.nan) for name in QUANTILES}
    above = np.full(rows, np.nan)
    for row in np.flatnonzero(enough):
        values = counts[row][counts[row] > 0]
        for name, q in QUANTILES.items():
            observed_quantile[name][row] = np.quantile(values, q, method="inverted_cdf")
        above[row] = np.sum(values > top[row])

    def only(values: np.ndarray) -> np.ndarray:
        kept: np.ndarray = np.where(enough, values, np.nan)
        return kept

    nothing = np.full(rows, np.nan)
    return {
        "silence": ((zero.sum(axis=1) + 0.5) / (size + 1.0), silence),
        "mean": (total / size, predicted_mean),
        "variance": (counts.var(axis=1, ddof=1), predicted_variance),
        "active_mean": (only(active_mean), only(m)),
        "active_dispersion": (only(active_variance), only(v)),
        "q90": (observed_quantile["q90"], only(predicted_quantile["q90"])),
        "q99": (observed_quantile["q99"], only(predicted_quantile["q99"])),
        "tail_excess": (above, only(active * tail)),
        "quiet_runs": (
            run_windows(zero, stretch, protocol.quiet_run_windows).astype(float),
            nothing,
        ),
        "bursts": (
            run_windows(~zero, stretch, protocol.burst_windows).astype(float),
            nothing,
        ),
    }


def log_ratios(name: str, observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Log of observed over predicted, on each statistic's scale; NaN where undefined."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "silence":
            ratio = np.log(observed / (1.0 - observed)) - np.log(
                predicted / (1.0 - predicted)
            )
        elif name in ("tail_excess", *RUN_STATISTICS):
            ratio = np.log((observed + 0.5) / (predicted + 0.5))
        else:
            ratio = np.where(
                (observed > 0.0) & (predicted > 0.0),
                np.log(observed / predicted),
                np.nan,
            )
    finite: np.ndarray = np.where(np.isfinite(ratio), ratio, np.nan)
    return finite


def _number(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def cell_verdict(
    log_ratio: float | None, band: Sequence[float] | None, minimal: float
) -> str:
    """Consistent, above, below or small; ``not estimable`` without a value."""
    if log_ratio is None or band is None:
        return "not estimable"
    low, high = band
    if low <= log_ratio <= high:
        return "consistent"
    if abs(log_ratio) < minimal:
        return "small"
    return "above" if log_ratio > 0.0 else "below"


def check_cell(
    counts: np.ndarray,
    stretch: np.ndarray,
    reference: str,
    parameters: tuple[float, float] | None,
    protocol: PredictiveProtocol,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Every statistic of one cell against one reference's predictive distribution.

    For ``own``, *parameters* are ignored: the cell's own hurdle is fitted and
    each replicate refitted. For ``population``, they are the population's
    ``(silence, rate)`` for this state and channel.
    """
    counts = np.asarray(counts, dtype=float)
    if reference == OWN:
        parameters = own_parameters(counts)
    elif reference != POPULATION:
        raise ValueError(f"reference must be one of {REFERENCES}")
    if parameters is None:
        return {"parameters": None, "statistics": {}}
    silence, rate = parameters
    observed = _batch(
        counts[None, :], stretch, np.array([silence]), np.array([rate]), protocol
    )

    replicates = sample_hurdle(rng, silence, rate, (protocol.replicates, counts.size))
    if reference == OWN:
        fits = [own_parameters(row) for row in replicates]
        replicates = replicates[np.array([f is not None for f in fits], dtype=bool)]
        rep_silence = np.array([f[0] for f in fits if f is not None])
        rep_rate = np.array([f[1] for f in fits if f is not None])
    else:
        rep_silence = np.full(replicates.shape[0], silence)
        rep_rate = np.full(replicates.shape[0], rate)
    simulated = _batch(replicates, stretch, rep_silence, rep_rate, protocol)

    tail = (1.0 - protocol.band) / 2.0
    statistics: dict[str, Any] = {}
    for name in STATISTICS:
        value = float(observed[name][0][0])
        predicted = float(observed[name][1][0])
        rep_observed, rep_predicted = simulated[name]
        if name in RUN_STATISTICS:
            predicted = float(np.mean(rep_observed))
            rep_predicted = np.full(rep_observed.shape, predicted)
            length = (
                protocol.quiet_run_windows
                if name == "quiet_runs"
                else protocol.burst_windows
            )
            if max(value, predicted) < protocol.min_runs * length:
                statistics[name] = {
                    "observed": value,
                    "predicted": predicted,
                    "verdict": "not estimable",
                }
                continue
        if not (math.isfinite(value) and math.isfinite(predicted)):
            statistics[name] = {"verdict": "not estimable"}
            continue
        ratio = _number(
            float(log_ratios(name, np.array([value]), np.array([predicted]))[0])
        )
        spread = log_ratios(name, rep_observed, rep_predicted)
        spread = spread[np.isfinite(spread)]
        band = (
            [float(np.quantile(spread, tail)), float(np.quantile(spread, 1.0 - tail))]
            if spread.size >= protocol.replicates // 2
            else None
        )
        entry: dict[str, Any] = {
            "observed": value,
            "predicted": predicted,
            "log_ratio": ratio,
            "band": band,
        }
        if reference == OWN and name in FITTED_BY_OWN:
            entry["verdict"] = "fitted"
        else:
            entry["verdict"] = cell_verdict(ratio, band, protocol.minimal_log_ratio)
        statistics[name] = entry
    return {
        "parameters": {"silence": silence, "rate": rate},
        "replicates": int(replicates.shape[0]),
        "expected_bins": [
            p * counts.size for p in hurdle_bins(silence, rate, protocol.bin_edges)
        ],
        "statistics": statistics,
    }


def household_cells(
    recording: CasasRecording,
    protocol: PredictiveProtocol,
    population: FittedChannels,
    *,
    household: str,
    index: int,
    ontology: StateOntology,
) -> list[dict[str, Any]]:
    """Every sufficient cell of one household, checked against both references."""
    space = tuple(ontology.states)
    counts, rows, labels, _ = household_channel_counts(
        recording, protocol.resolution, ontology, household=household
    )
    models = population.models(
        recording.registry, protocol.resolution, HURDLE, ontology
    )
    cells: list[dict[str, Any]] = []
    for s, state in enumerate(space):
        chosen = rows[labels == s]
        if chosen.size < protocol.min_windows:
            continue
        stretch = stretches_of(chosen)
        for channel in sorted(counts):
            c = protocol.resolution.channels.index(channel)
            values = counts[channel][chosen]
            model = models[channel]
            if not isinstance(model, HurdleChannel):  # pragma: no cover - by family
                raise TypeError("the population family must be the hurdle")
            entry: dict[str, Any] = {
                "state": state.value,
                "channel": channel.name,
                "room": channel.room,
                "modality": channel.modality.value,
                "windows": int(values.size),
                "silent": int(np.sum(values == 0)),
                "active": int(np.sum(values > 0)),
                "stretches": int(stretch[-1] + 1),
                "observed_bins": observed_bins(values, protocol.bin_edges),
                "references": {},
            }
            for r, reference in enumerate(REFERENCES):
                rng = np.random.default_rng([protocol.seed, index, s, c, r])
                parameters = (
                    None
                    if reference == OWN
                    else (float(model.silence[s]), float(model.rate[s]))
                )
                entry["references"][reference] = check_cell(
                    values, stretch, reference, parameters, protocol, rng
                )
            cells.append(entry)
    return cells


# ----------------------------------------------------------------------------
# Across households
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictiveResult:
    """A completed diagnostic: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def across_households(
    values: Mapping[str, float | None], protocol: PredictiveProtocol
) -> dict[str, Any]:
    """One value per household: mean and median with household intervals, and a verdict."""
    present = {h: v for h, v in values.items() if v is not None and math.isfinite(v)}
    if len(present) < protocol.min_households:
        return {
            "n": len(present),
            "values": dict(sorted(present.items())),
            "verdict": "insufficient",
        }
    comparison = compare_households(
        present,
        dict.fromkeys(present, 0.0),
        higher_is_better=True,
        confidence=protocol.confidence,
        resamples=protocol.resamples,
        seed=protocol.seed,
    ).to_dict()
    mean: dict[str, Any] = comparison["mean"]  # type: ignore[assignment]
    minimal = protocol.minimal_log_ratio
    low, high = mean["interval"]["low"], mean["interval"]["high"]
    if mean["estimate"] >= minimal and low > 0.0:
        verdict = "positive"
    elif mean["estimate"] <= -minimal and high < 0.0:
        verdict = "negative"
    elif -minimal < low and high < minimal:
        verdict = "negligible"
    else:
        verdict = "uncertain"
    return {
        "n": comparison["n"],
        "values": dict(sorted(present.items())),
        "mean": mean,
        "median": comparison["median"],
        "above": comparison["favours_model"],
        "below": comparison["favours_reference"],
        "sd": summarise_households(present).sd,
        "verdict": verdict,
    }


def _group_key(cell: Mapping[str, Any], group: str) -> str:
    return str(cell[group])


def household_group_values(
    households: Mapping[str, Mapping[str, Any]],
    reference: str,
    statistic: str,
    group: str,
) -> dict[str, dict[str, float | None]]:
    """For each group, each household's mean cell log ratio in it."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for home, entry in households.items():
        for cell in entry["cells"]:
            stat = cell["references"][reference]["statistics"].get(statistic, {})
            ratio = stat.get("log_ratio")
            if ratio is None or stat.get("verdict") in ("not estimable", "fitted"):
                continue
            grouped.setdefault(_group_key(cell, group), {}).setdefault(home, []).append(
                float(ratio)
            )
    return {
        key: {home: float(np.mean(v)) for home, v in sorted(per_home.items())}
        for key, per_home in sorted(grouped.items())
    }


def verdict_counts(
    cells: Sequence[Mapping[str, Any]], reference: str, statistic: str
) -> dict[str, int]:
    """How many of *cells* each verdict covers, for one reference and statistic."""
    tally: dict[str, int] = {}
    for cell in cells:
        verdict = (
            cell["references"][reference]["statistics"]
            .get(statistic, {})
            .get("verdict", "not estimable")
        )
        tally[verdict] = tally.get(verdict, 0) + 1
    return dict(sorted(tally.items()))


def variance_mean_slope(cells: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """One household's slope of log active variance on log active mean, observed and predicted."""
    points = []
    for cell in cells:
        stat = cell["references"][OWN]["statistics"].get("active_dispersion", {})
        mean = cell["references"][OWN]["statistics"].get("active_mean", {})
        if "observed" not in stat or "observed" not in mean:
            continue
        observed, predicted = stat["observed"], stat["predicted"]
        if observed > 0.0 and predicted > 0.0 and mean["observed"] > 1.0:
            points.append(
                (math.log(mean["observed"]), math.log(observed), math.log(predicted))
            )
    if len(points) < 4:
        return {"cells": len(points), "observed": None, "predicted": None}
    x = np.array([p[0] for p in points])
    if np.ptp(x) == 0.0:
        return {"cells": len(points), "observed": None, "predicted": None}
    observed_slope = float(np.polyfit(x, [p[1] for p in points], 1)[0])
    predicted_slope = float(np.polyfit(x, [p[2] for p in points], 1)[0])
    return {
        "cells": len(points),
        "observed": observed_slope,
        "predicted": predicted_slope,
    }


def pattern_conclusion(verdicts: Mapping[str, str]) -> str:
    """Systematic, partial, absent or inconclusive, from the common states' verdicts."""
    positive = sum(v == "positive" for v in verdicts.values())
    negative = sum(v == "negative" for v in verdicts.values())
    if positive >= 3 and negative == 0:
        return "systematic"
    if positive >= 1:
        return "partial"
    if verdicts and all(v in ("negligible", "negative") for v in verdicts.values()):
        return "absent"
    return "inconclusive"


def route(conclusions: Mapping[str, str]) -> dict[str, Any]:
    """The next model family the conclusions point to, by the declared rule."""
    if "systematic" in (conclusions["quiet_runs"], conclusions["bursts"]):
        family: str | None = "temporal"
    elif conclusions["active_dispersion"] == "systematic":
        family = "dispersion"
    else:
        family = None
    return {
        "family": family,
        "description": ROUTES[family] if family is not None else None,
        "roadmap": "names the family" if family is not None else "unchanged",
    }


def run_predictive_checks(
    recordings: Mapping[str, CasasRecording],
    protocol: PredictiveProtocol,
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> PredictiveResult:
    """Run the declared diagnostic and build its record."""
    homes = sorted({h for fold in protocol.folds for h in (*fold.train, *fold.test)})
    missing = sorted(set(homes) - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    ontology = StateOntology()
    space = tuple(ontology.states)
    statistics = {
        home: home_statistics(
            recordings[home], protocol.resolution, ontology, household=home
        )
        for home in homes
    }
    order = {home: i for i, home in enumerate(protocol.homes)}
    fitted: dict[str, FittedChannels] = {}
    households: dict[str, dict[str, Any]] = {}
    for fold in protocol.folds:
        population = combine_statistics(
            {h: statistics[h] for h in fold.train},
            states=space,
            pseudo_windows=protocol.pseudo_windows,
        )
        fitted[fold.name] = population
        for home in sorted(fold.test):
            cells = household_cells(
                recordings[home],
                protocol,
                population,
                household=home,
                index=order[home],
                ontology=ontology,
            )
            households[home] = {"fold": fold.name, "cells": cells}
    for entry in households.values():
        entry["verdicts"] = {
            reference: {
                statistic: verdict_counts(entry["cells"], reference, statistic)
                for statistic in STATISTICS
            }
            for reference in REFERENCES
        }
        entry["variance_mean"] = variance_mean_slope(entry["cells"])

    groups: dict[str, Any] = {}
    for group in ("state", "channel", "room", "modality"):
        groups[group] = {
            reference: {
                statistic: {
                    key: across_households(values, protocol)
                    for key, values in household_group_values(
                        households, reference, statistic, group
                    ).items()
                }
                for statistic in STATISTICS
            }
            for reference in REFERENCES
        }
    state_channel: dict[str, Any] = {}
    for reference in REFERENCES:
        state_channel[reference] = {}
        for statistic in STATISTICS:
            table: dict[str, dict[str, Any]] = {}
            for home, entry in households.items():
                for cell in entry["cells"]:
                    stat = cell["references"][reference]["statistics"].get(
                        statistic, {}
                    )
                    ratio = stat.get("log_ratio")
                    if ratio is None or stat.get("verdict") in (
                        "not estimable",
                        "fitted",
                    ):
                        continue
                    slot = table.setdefault(cell["state"], {}).setdefault(
                        cell["channel"], {"values": {}}
                    )
                    slot["values"][home] = ratio
            # Each household's value is in its cells; the table keeps only the
            # across-household description.
            for per_channel in table.values():
                for channel, slot in per_channel.items():
                    values = list(slot["values"].values())
                    per_channel[channel] = {
                        "n": len(values),
                        "median": float(median(values)),
                        "above": sum(v > 0.0 for v in values),
                        "below": sum(v < 0.0 for v in values),
                    }
            state_channel[reference][statistic] = table

    slopes = {
        key: across_households(
            {h: e["variance_mean"][key] for h, e in households.items()}, protocol
        )
        for key in ("observed", "predicted")
    }
    slopes["difference"] = across_households(
        {
            h: (
                e["variance_mean"]["observed"] - e["variance_mean"]["predicted"]
                if e["variance_mean"]["observed"] is not None
                else None
            )
            for h, e in households.items()
        },
        protocol,
    )

    common = [s.value for s in protocol.common_states]
    conclusions: dict[str, dict[str, Any]] = {}
    for key, statistic in CONCLUSION_STATISTICS.items():
        by_state = groups["state"][OWN][statistic]
        verdicts = {
            state: by_state.get(state, {"verdict": "insufficient"})["verdict"]
            for state in common
        }
        conclusions[key] = {
            "verdicts": verdicts,
            "conclusion": pattern_conclusion(verdicts),
        }
    routing = route({key: value["conclusion"] for key, value in conclusions.items()})

    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified diagnostic; development panel, which earlier work "
        "has inspected",
        "states": [state.value for state in space],
        "common_states": common,
        "channels": [
            {"name": c.name, "room": c.room, "modality": c.modality.value}
            for c in protocol.resolution.channels
        ],
        "households": dict(sorted(households.items())),
        "groups": groups,
        "state_channel": state_channel,
        "variance_mean": slopes,
        "conclusions": conclusions,
        "routing": routing,
        "fitted": {
            fold: {**population.to_dict(), "sha256": population.sha256()}
            for fold, population in fitted.items()
        },
    }
    record = ExperimentRecord(
        experiment=protocol.name,
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        inference=ONLINE,
        evidence=NotEnumerated(
            "no state is estimated: the diagnostic compares observed channel counts "
            "in labelled windows with the observation model's predictive "
            "distribution"
        ),
        seeds=[protocol.seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions=PREDICTIVE_METRIC_DEFINITIONS,
        notes=[
            "Every setting and the rule that reads the result were declared in the "
            "protocol before any household was examined.",
            "Diagnostic only: no model is changed and no inference is run.",
            "Households are the unit across households: each household's value is "
            "the mean of its cells' log ratios, and summaries resample households.",
            "The own reference is fitted to each cell, so its silence and means "
            "are exact by construction and are not judged.",
        ],
        inputs=list(inputs),
        split={
            "folds": [{**f.to_dict(), "sha256": f.sha256()} for f in protocol.folds]
        },
        information_set={
            "name": "current",
            "evidence": "each instrumented channel's count in each labelled window",
            "step_seconds": protocol.resolution.step.total_seconds(),
        },
        preprocessing={
            "moments": "one per step from each recording's first observation to "
            "its last",
            "labels": "truth_series of each recording's annotations; unlabelled "
            "windows are not used",
            "step_seconds": protocol.resolution.step.total_seconds(),
        },
        models=[
            ModelRecord(
                OWN, {"family": "hurdle", "fitted": "the cell alone"}, "own_parameters"
            ),
            ModelRecord(
                POPULATION,
                {"family": "hurdle", "pseudo_windows": protocol.pseudo_windows},
                "combine_statistics",
            ),
        ],
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return PredictiveResult(record=record, path=path)
