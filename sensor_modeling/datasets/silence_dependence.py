"""Phase 3.3 diagnostic: does independent Poisson silence over-concentrate the posterior?

The generative model multiplies one Poisson likelihood per sensor. A channel
that records nothing in a window contributes ``−λ_c(s) Δ`` to the log-evidence
for state ``s``, and a quiet window's evidence is the sum of those terms over
every silent channel. If silences on different channels co-occur more often,
given the state, than independence implies, the sum overstates the evidence,
and the posterior concentrates more than the data justify. This is the
roadmap's correlated-silence hypothesis (3.3).

This module measures that on real households. It is diagnostic only: the
emission model, the filter and every default are unchanged. Every setting is
declared in :class:`SilenceProtocol` and frozen in a committed file before any
household is examined.

Streams
-------
A stream is an evidence channel of the information sets: one room and one
modality, pooling that room's event sensors, counted in the same 5-minute step
windows as every Phase 1 and Phase 3 experiment. For a silent window the
filter's likelihood is identical whether a channel's sensors are pooled or
multiplied one by one, so pooling loses nothing for the silence terms.

What is measured
----------------
Per household, from its labelled windows, conditional on the annotated state:

- **Joint silence.** The observed probability that every instrumented channel
  is silent, against three independence baselines:
  - the product of the channels' observed silence probabilities, which tests
    dependence alone;
  - independent Poisson streams with each channel's observed mean count, which
    adds each channel's departure from a Poisson count;
  - the filter's declared rates, which is what the filter assumes.
- **Pairwise dependence.** For every pair of channels: the log odds ratio of
  silence, the log ratio of joint to independent silence, and the correlation
  and covariance of counts.
- **Dispersion.** The variance of the number of silent channels, against its
  variance if channels fell silent independently.
- **Silence-evidence inflation.** A constant excess of joint silence in every
  state shifts each state's evidence equally and cannot move a posterior. Only
  its variation across states can. The inflation factor is the spread across
  states of the independent joint-silence log-evidence over the spread of the
  observed one. It is computed with the channels' observed silence
  probabilities, for dependence alone, and with the declared terms the filter
  uses.
- **Concentration.** The filter's generative model, fed every window of the
  recording, gives each labelled window a posterior. Within each state, the
  diagnostic measures the slope of overconfidence, confidence minus
  correctness, on the number of silent channels. It also measures the slope
  along consecutive fully silent windows.
- **Representative quiet periods.** For each household and quiet state, one
  pre-specified quiet run is decomposed step by step: the prediction from the
  transition, each silent channel's contribution, the posterior, the number
  of silent channels and the confidence.

Households, not windows, are the unit of replication. Every across-household
statistic uses one value per household, and intervals resample households.
Windows within a household are autocorrelated, so no within-household interval
is reported.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from ..evaluation.households import compare_households, summarise_households
from ..evaluation.provenance import (
    ExperimentRecord,
    InputArtifact,
    ModelRecord,
    ReportedInterval,
)
from ..evaluation.resampling import percentile_interval, resample_indices
from ..fusion.regime import ONLINE
from ..states.ontology import BehaviouralState, StateOntology
from .casas import CasasRecording
from .information_sets import (
    EvidenceChannel,
    EvidenceResolution,
    InformationComponent,
    InformationSet,
    build_feature_table,
    evidence_column,
)
from .matched_evaluation import _finite, _Household, _regular_moments
from .recoverable_gap import GENERATIVE_CONFIGURATION, FrozenSplits
from .restricted_filter import ChannelLikelihood, channel_likelihoods
from .time_prior_experiment import verdict

PROTOCOL_SCHEMA = "silence-protocol/1"
RESULT_SCHEMA = "silence-results/1"

S = BehaviouralState

#: The states in which the published quiet-period saturation was observed.
QUIET_STATES: tuple[BehaviouralState, ...] = (S.SLEEPING, S.AWAY, S.HOME_INACTIVE)

#: The smallest effect on each scale that counts as material.
MINIMAL_EFFECTS: dict[str, float] = {
    "log_ratio": math.log(1.25),
    "correlation": 0.1,
    "slope": 0.02,
}

#: The information the diagnostic reads: each channel's current window.
MODEL_CONFIGURATION: dict[str, object] = {
    **GENERATIVE_CONFIGURATION,
    "model": "the generative filter's model fed every window of the recording",
    "prior": "stationary distribution of the default ontology, one step before "
    "each recording's first window",
    "windows": "every step window from each recording's first observation to its "
    "last, in order, each channel's pooled count",
    "prediction": "most probable state of the posterior; no abstention",
}

#: Pre-specified rules, stated in words for the record.
CRITERIA: dict[str, str] = {
    "verdicts": "positive: mean >= delta and lower bound > 0; negative: mean <= "
    "-delta and upper bound < 0; negligible: the whole interval within (-delta, "
    "delta); uncertain: anything else",
    "supported": "D1 and C1 are both positive: independence materially inflates "
    "silence evidence, and overconfidence grows with the number of silent channels",
    "weakened": "D1 or C1 is negligible or negative",
    "inconclusive": "neither supported nor weakened",
    "roadmap": "ROADMAP.md is updated only if the conclusion is supported or weakened",
    "why_inflation": "a constant excess of joint silence in every state shifts "
    "every state's evidence equally and cannot move a posterior; only its "
    "variation across states can, which is what D1 measures",
    "log_ratio": "log 1.25: at an inflation of 1.25, a posterior of 0.95 reached "
    "on silence alone would be 0.91 with the evidence corrected, which leaves the "
    "0.95 to 1.00 confidence band of the uncertainty study",
    "slope": "0.02 per silent channel or per hour, the calibration-error minimal "
    "difference of Phases 3.1 and 3.2",
    "correlation": "0.1, the conventional smallest correlation of note",
}

_VERDICTS = {
    "favours model": "positive",
    "favours reference": "negative",
    "negligible": "negligible",
    "uncertain": "uncertain",
}


@dataclass(frozen=True)
class Estimand:
    """One pre-specified household-level quantity, tested against independence."""

    key: str
    role: str
    quantity: str
    scale: str
    question: str

    def to_dict(self) -> dict[str, str]:
        """Return a serialisable form."""
        return {
            "key": self.key,
            "role": self.role,
            "quantity": self.quantity,
            "scale": self.scale,
            "question": self.question,
        }


ESTIMANDS: tuple[Estimand, ...] = (
    Estimand(
        "D1",
        "primary",
        "log_inflation",
        "log_ratio",
        "how much independence overstates the spread of joint-silence "
        "log-evidence across states, with the channels' observed silence "
        "probabilities: dependence alone",
    ),
    Estimand(
        "D2",
        "primary",
        "log_joint_silence_quiet",
        "log_ratio",
        "how much more often every channel is silent than independence implies, "
        "in the quiet states",
    ),
    Estimand(
        "C1",
        "primary",
        "gap_slope",
        "slope",
        "how much overconfidence grows per additional silent channel, within a state",
    ),
    Estimand(
        "S1",
        "secondary",
        "log_inflation_declared",
        "log_ratio",
        "how much the filter's declared silence terms overstate the spread of "
        "joint-silence log-evidence across states",
    ),
    Estimand(
        "S2",
        "secondary",
        "log_marginal_factor",
        "log_ratio",
        "how much of S1 comes from the declared rates rather than dependence: "
        "S1 minus D1",
    ),
    Estimand(
        "S3",
        "secondary",
        "log_odds_silence_quiet",
        "log_ratio",
        "the median pairwise log odds ratio of silence, in the quiet states",
    ),
    Estimand(
        "S4",
        "secondary",
        "count_correlation_quiet",
        "correlation",
        "the mean pairwise correlation of counts, in the quiet states",
    ),
    Estimand(
        "S5",
        "secondary",
        "log_dispersion_quiet",
        "log_ratio",
        "the variance of the number of silent channels against independence, "
        "in the quiet states",
    ),
    Estimand(
        "S6",
        "secondary",
        "confidence_slope",
        "slope",
        "how much confidence grows per additional silent channel, within a state",
    ),
    Estimand(
        "S7",
        "secondary",
        "accuracy_slope",
        "slope",
        "how much accuracy grows per additional silent channel, within a state",
    ),
    Estimand(
        "S8",
        "secondary",
        "accumulation_slope",
        "slope",
        "how much overconfidence grows per hour of consecutive fully silent "
        "windows, within a state",
    ),
)

#: Per-state quantities, each reported across households for every state.
PER_STATE: dict[str, str] = {
    "log_joint_silence": "log of observed joint silence over the product of the "
    "channels' observed silence probabilities",
    "log_joint_silence_poisson": "log of observed joint silence over independent "
    "Poisson streams with each channel's observed mean count",
    "log_joint_silence_declared": "log of observed joint silence over the filter's "
    "declared rates",
    "joint_silence_difference": "observed joint silence minus the product of the "
    "channels' observed silence probabilities",
    "log_odds_silence": "median over channel pairs of the log odds ratio of silence",
    "log_pair_silence": "median over channel pairs of the log ratio of joint to "
    "independent silence",
    "count_correlation": "mean over channel pairs of the correlation of counts",
    "log_dispersion": "log of the variance of the number of silent channels over "
    "its variance under independence",
}

#: How each quantity is computed, for the record.
DEFINITIONS: dict[str, str] = {
    "silent": "a channel is silent in a window when it records no activation",
    "silence_probability": "(silent windows + a) / (windows + 2a), a = the "
    "declared smoothing",
    "joint_silence": "the same, for windows in which every instrumented channel "
    "is silent",
    "pair_silence": "the same, for windows in which both channels of a pair are silent",
    "log_odds_silence": "log((n00 + 0.5)(n11 + 0.5) / ((n01 + 0.5)(n10 + 0.5))) "
    "over the pair's silent (0) and active (1) windows",
    "count_correlation": "Pearson correlation of the two channels' counts; "
    "undefined where either count is constant",
    "count_covariance": "covariance of the two channels' counts, divided by the "
    "number of windows",
    "log_dispersion": "variance of the number of silent channels over the sum of "
    "p(1 - p) over channels, p unsmoothed; undefined where either is zero",
    "log_inflation": "log of SD_s[sum_c log p_c(s)] / SD_s[log p_joint(s)] over "
    "the household's eligible states whose joint silence is estimable, "
    "population SDs",
    "log_inflation_declared": "the same with sum_c of the filter's declared "
    "silence log-likelihood in place of sum_c log p_c(s)",
    "log_marginal_factor": "log_inflation_declared minus log_inflation",
    "quiet": "the mean over the household's eligible quiet states",
    "gap": "confidence (largest posterior probability) minus 1 if its state is "
    "the annotated state, else 0",
    "gap_slope": "least-squares slope of the gap on the number of silent "
    "channels, after removing each predicted state's mean from both, over every "
    "labelled window; zero in expectation for a calibrated model",
    "confidence_slope": "the same for confidence",
    "accuracy_slope": "the same for correctness",
    "accumulation_slope": "the same for the gap on the position in a run of "
    "consecutive fully silent windows, in hours, over labelled fully silent "
    "windows at positions up to the declared limit",
    "association": "Spearman correlation across households of log_inflation and "
    "gap_slope, with a household bootstrap interval",
}


# ----------------------------------------------------------------------------
# The protocol
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SilenceProtocol:
    """Everything the diagnostic fixes before any household is examined."""

    households: tuple[str, ...]
    splits_sha256: str
    resolution: EvidenceResolution = field(default_factory=EvidenceResolution)
    min_windows: int = 50
    smoothing: float = 0.5
    min_expected: float = 5.0
    quiet_states: tuple[BehaviouralState, ...] = QUIET_STATES
    case_min_steps: int = 12
    case_steps: int = 12
    accumulation_steps: int = 36
    resamples: int = 10_000
    confidence: float = 0.95
    seed: int = 0
    minimal_effects: Mapping[str, float] = field(
        default_factory=lambda: dict(MINIMAL_EFFECTS)
    )
    name: str = "phase3-correlated-silence"

    def __post_init__(self) -> None:
        """Validate the declaration."""
        homes = tuple(sorted(self.households))
        if len(homes) < 2 or len(set(homes)) != len(homes):
            raise ValueError("at least two distinct households are required")
        if not isinstance(self.splits_sha256, str) or len(self.splits_sha256) != 64:
            raise ValueError("splits_sha256 must be a SHA-256 hex digest")
        for name in ("min_windows", "case_min_steps", "case_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.case_steps > self.case_min_steps:
            raise ValueError("case_steps cannot exceed case_min_steps")
        if self.accumulation_steps < 2:
            raise ValueError("accumulation_steps must be at least 2")
        if not math.isfinite(self.smoothing) or self.smoothing <= 0.0:
            raise ValueError("smoothing must be positive")
        if not math.isfinite(self.min_expected) or self.min_expected <= 0.0:
            raise ValueError("min_expected must be positive")
        if not self.quiet_states or any(
            s not in StateOntology().states for s in self.quiet_states
        ):
            raise ValueError("quiet_states must be ontology states")
        if not 0.0 < self.confidence < 1.0 or self.resamples < 100:
            raise ValueError("confidence must lie in (0, 1), resamples be >= 100")
        missing = sorted(set(MINIMAL_EFFECTS) - set(self.minimal_effects))
        if missing or any(
            not math.isfinite(v) or v <= 0.0 for v in self.minimal_effects.values()
        ):
            raise ValueError(
                f"every scale needs a positive minimal effect; missing {missing}"
            )
        object.__setattr__(self, "households", homes)
        object.__setattr__(self, "quiet_states", tuple(self.quiet_states))
        object.__setattr__(self, "minimal_effects", dict(self.minimal_effects))

    @property
    def information_set(self) -> InformationSet:
        """The current windows of every channel, and nothing else."""
        return InformationSet(
            "current",
            frozenset({InformationComponent.CURRENT_EVIDENCE}),
            self.resolution,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        info = self.information_set
        return {
            "schema": PROTOCOL_SCHEMA,
            "name": self.name,
            "households": {
                "splits_file": "artifacts/phase1/household_splits.json",
                "splits_sha256": self.splits_sha256,
                "households": list(self.households),
                "use": "every development household once; nothing is fitted, so "
                "the folds are not used",
            },
            "streams": {
                "information_set": {**info.to_dict(), "sha256": info.sha256()},
                "definition": "each instrumented evidence channel's activations "
                "in each step window; uninstrumented channels are left out",
            },
            "model": dict(MODEL_CONFIGURATION),
            "inference": "unchanged: the diagnostic reads the filter's own "
            "emission terms and transition, and changes neither",
            "states": {
                "quiet": [state.value for state in self.quiet_states],
                "eligibility": f"a state enters a household's statistics with at "
                f"least {self.min_windows} labelled windows there",
            },
            "smoothing": self.smoothing,
            "min_expected": {
                "value": self.min_expected,
                "rule": "a joint-silence statistic is computed only where at "
                "least this many windows would be jointly silent if the channels "
                "fell silent independently; a log odds ratio needs every cell of "
                "its table to reach it; the classical expected-count rule for "
                "contingency tables",
            },
            "definitions": dict(DEFINITIONS),
            "estimands": [estimand.to_dict() for estimand in ESTIMANDS],
            "per_state": dict(PER_STATE),
            "per_pair": "log odds ratio of silence and log pair silence ratio per "
            "channel pair, each the mean over the household's eligible quiet "
            "states where it is estimable",
            "cases": {
                "rule": "for each household and quiet state, the maximal runs of "
                "consecutive windows that are fully silent and labelled with that "
                f"state, at least {self.case_min_steps} windows long; the case "
                "is the run of median length (the lower median, the earliest "
                f"among equal lengths); its first {self.case_steps} windows are "
                "recorded",
                "channel_order": "each silent channel is added in the declared "
                "channel order to show the cumulative confidence",
            },
            "accumulation_steps": self.accumulation_steps,
            "minimal_effects": dict(self.minimal_effects),
            "criteria": dict(CRITERIA),
            "bootstrap": {
                "unit": "household",
                "statistics": "mean and median of the household values, against "
                "zero, the value under independence",
                "resamples": self.resamples,
                "confidence": self.confidence,
                "interval": "percentile",
            },
            "seeds": {"bootstrap": self.seed},
            "tuning": "none: every setting is declared here, before any household "
            "is examined",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_protocol(splits: FrozenSplits) -> SilenceProtocol:
    """The protocol as declared for the development panel."""
    return SilenceProtocol(tuple(splits.homes), splits.sha256)


def check_frozen_protocol(protocol: SilenceProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after the diagnostic has run"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


# ----------------------------------------------------------------------------
# Dependence, from counts and labels
# ----------------------------------------------------------------------------
def _smoothed(hits: float, total: int, smoothing: float) -> float:
    return (hits + smoothing) / (total + 2.0 * smoothing)


def _log_odds_silence(first: np.ndarray, second: np.ndarray) -> float:
    both = float(np.sum(first & second))
    neither = float(np.sum(~first & ~second))
    only_first = float(np.sum(first & ~second))
    only_second = float(np.sum(~first & second))
    return math.log(
        (both + 0.5) * (neither + 0.5) / ((only_first + 0.5) * (only_second + 0.5))
    )


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.std() == 0.0 or second.std() == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def state_dependence(
    counts: np.ndarray,
    channels: Sequence[str],
    expected: np.ndarray,
    *,
    smoothing: float,
    min_expected: float,
) -> dict[str, Any]:
    """Every dependence statistic for one household's windows in one state.

    A joint-silence statistic is left undefined where it cannot be estimated:
    where fewer than *min_expected* windows would be jointly silent if the
    channels fell silent independently. Below that, the smoothed estimate sits
    at its floor, and the ratio to independence would measure the floor. The
    rule depends only on the channels' own silence, so it does not select on
    dependence. A log odds ratio needs every cell of its table to reach the
    same expected count.

    Parameters
    ----------
    counts
        ``(windows, channels)`` activation counts, all observed.
    channels
        Channel names, in the column order of *counts*.
    expected
        Each channel's expected activations per window under the declared
        rates, in the same order.
    smoothing
        Pseudo-count added to silent and active windows alike.
    min_expected
        Smallest expected count, under independence, for a joint statistic.
    """
    counts = np.asarray(counts, dtype=float)
    windows = counts.shape[0]
    silent = counts == 0
    marginal = np.array(
        [_smoothed(float(column.sum()), windows, smoothing) for column in silent.T]
    )
    joint = _smoothed(float(silent.all(axis=1).sum()), windows, smoothing)
    independent = float(np.prod(marginal))
    means = counts.mean(axis=0)
    estimable = windows * independent >= min_expected
    pairs: dict[str, dict[str, float | None]] = {}
    for i, j in combinations(range(len(channels)), 2):
        pair = _smoothed(float(np.sum(silent[:, i] & silent[:, j])), windows, smoothing)
        cells = windows * np.array(
            [
                marginal[i] * marginal[j],
                marginal[i] * (1.0 - marginal[j]),
                (1.0 - marginal[i]) * marginal[j],
                (1.0 - marginal[i]) * (1.0 - marginal[j]),
            ]
        )
        pairs[f"{channels[i]}|{channels[j]}"] = {
            "log_odds_silence": (
                _log_odds_silence(silent[:, i], silent[:, j])
                if cells.min() >= min_expected
                else None
            ),
            "log_pair_silence": (
                math.log(pair / (marginal[i] * marginal[j]))
                if cells[0] >= min_expected
                else None
            ),
            "count_correlation": _correlation(counts[:, i], counts[:, j]),
            "count_covariance": float(
                np.mean((counts[:, i] - means[i]) * (counts[:, j] - means[j]))
            ),
        }
    raw = silent.mean(axis=0)
    independent_variance = float(np.sum(raw * (1.0 - raw)))
    observed_variance = float(silent.sum(axis=1).var())
    dispersion = (
        math.log(observed_variance / independent_variance)
        if observed_variance > 0.0 and independent_variance > 0.0
        else None
    )

    def median_of(key: str) -> float | None:
        values: list[float] = [v for p in pairs.values() if (v := p[key]) is not None]
        return float(np.median(values)) if values else None

    correlations = [
        p["count_correlation"]
        for p in pairs.values()
        if p["count_correlation"] is not None
    ]
    return {
        "windows": windows,
        "silence": dict(zip(channels, marginal.tolist())),
        "poisson_silence": dict(zip(channels, np.exp(-means).tolist())),
        "declared_silence": dict(zip(channels, np.exp(-expected).tolist())),
        "joint_silence": joint,
        "independent_joint_silence": independent,
        "poisson_joint_silence": float(np.exp(-means.sum())),
        "declared_joint_silence": float(np.exp(-np.sum(expected))),
        "joint_estimable": bool(estimable),
        "log_joint_silence": math.log(joint / independent) if estimable else None,
        "log_joint_silence_poisson": (
            math.log(joint) + float(means.sum()) if estimable else None
        ),
        "log_joint_silence_declared": (
            math.log(joint) + float(np.sum(expected)) if estimable else None
        ),
        "joint_silence_difference": joint - independent if estimable else None,
        "pairs": pairs,
        "log_odds_silence": median_of("log_odds_silence"),
        "log_pair_silence": median_of("log_pair_silence"),
        "count_correlation": float(np.mean(correlations)) if correlations else None,
        "log_dispersion": dispersion,
    }


def silence_inflation(
    by_state: Mapping[str, Mapping[str, Any]], declared: Mapping[str, float]
) -> dict[str, float | None]:
    """How much independence overstates the spread of joint-silence evidence.

    Parameters
    ----------
    by_state
        :func:`state_dependence` for each eligible state. Only states whose
        joint silence is estimable enter.
    declared
        The filter's joint-silence log-likelihood in each of those states: the
        sum over channels of its declared silence terms. Constants shared by
        every state do not matter.
    """
    states = sorted(s for s in by_state if by_state[s]["joint_estimable"])
    if len(states) < 2:
        return {"log_inflation": None, "log_inflation_declared": None}
    observed = np.array([math.log(by_state[s]["joint_silence"]) for s in states])
    independent = np.array(
        [sum(math.log(p) for p in by_state[s]["silence"].values()) for s in states]
    )
    model = np.array([declared[s] for s in states])
    spread = float(observed.std())
    if spread == 0.0:
        return {"log_inflation": None, "log_inflation_declared": None}
    inflation = float(independent.std()) / spread
    declared_inflation = float(model.std()) / spread
    return {
        "log_inflation": math.log(inflation) if inflation > 0.0 else None,
        "log_inflation_declared": (
            math.log(declared_inflation) if declared_inflation > 0.0 else None
        ),
    }


# ----------------------------------------------------------------------------
# The filter's posterior, window by window
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Trajectory:
    """The filter's recursion over consecutive windows, decomposed.

    Attributes
    ----------
    predicted
        ``(windows, states)`` belief after the transition, before the window.
    channel_loglik
        ``(windows, channels, states)`` each channel's log-likelihood, centred
        so its largest value is zero.
    posterior
        ``(windows, states)`` belief after the window.
    """

    predicted: np.ndarray
    channel_loglik: np.ndarray
    posterior: np.ndarray


def filter_trajectory(
    counts: np.ndarray,
    per_activation: np.ndarray,
    per_window: np.ndarray,
    transition: np.ndarray,
    prior: np.ndarray,
) -> Trajectory:
    """Run the filter's recursion over consecutive windows of pooled counts.

    Starting from *prior* one step before the first window, each window is a
    transition then every channel's Poisson log-likelihood: its count times
    ``per_activation`` plus ``per_window``. That is what the production filter
    reports when fed the same windows (the restricted-filter tests check the
    per-window terms, and this module's tests the recursion).

    Parameters
    ----------
    counts
        ``(windows, channels)`` pooled counts, all observed.
    per_activation, per_window
        ``(channels, states)`` terms, as in
        :class:`~sensor_modeling.datasets.restricted_filter.ChannelLikelihood`.
    transition
        ``(states, states)`` one-step transition matrix.
    prior
        Belief one step before the first window.
    """
    counts = np.asarray(counts, dtype=float)
    if np.isnan(counts).any():
        raise ValueError("every channel's count must be observed in every window")
    loglik = counts[:, :, None] * per_activation[None, :, :] + per_window[None, :, :]
    loglik = loglik - loglik.max(axis=2, keepdims=True)
    total = loglik.sum(axis=1)
    predicted = np.empty((counts.shape[0], prior.size))
    posterior = np.empty_like(predicted)
    belief = np.asarray(prior, dtype=float)
    for row in range(counts.shape[0]):
        belief = belief @ transition
        predicted[row] = belief
        weights = belief * np.exp(total[row] - total[row].max())
        belief = weights / weights.sum()
        posterior[row] = belief
    return Trajectory(predicted, loglik, posterior)


def _within_state_slope(
    y: np.ndarray, x: np.ndarray, groups: np.ndarray
) -> float | None:
    """Least-squares slope of *y* on *x* after removing each group's means."""
    y_c = y.astype(float).copy()
    x_c = x.astype(float).copy()
    for group in np.unique(groups):
        rows = groups == group
        y_c[rows] -= y_c[rows].mean()
        x_c[rows] -= x_c[rows].mean()
    denominator = float(np.sum(x_c * x_c))
    if denominator <= 1e-12:
        return None
    return float(np.sum(x_c * y_c) / denominator)


def quiet_positions(silent_count: np.ndarray, channels: int) -> np.ndarray:
    """Each window's position in its run of consecutive fully silent windows.

    Zero for a window in which some channel was active.
    """
    positions = np.zeros(silent_count.size, dtype=int)
    run = 0
    for row, count in enumerate(silent_count):
        run = run + 1 if count == channels else 0
        positions[row] = run
    return positions


def concentration(
    posterior: np.ndarray,
    rows: np.ndarray,
    labels: np.ndarray,
    silent_count: np.ndarray,
    channels: int,
    *,
    accumulation_steps: int,
    steps_per_hour: float,
) -> dict[str, Any]:
    """How overconfidence varies with silent channels and quiet-run position.

    Slopes are taken within the *predicted* state. A calibrated model has an
    expected gap of zero given anything computed from the data, such as its
    prediction and the number of silent channels, so every slope is zero in
    expectation. Grouping by the annotated state would condition on the truth,
    and even a calibrated model then shows a slope. Every labelled window
    enters, for the same reason: selecting windows by their annotation would
    break the identity.

    Parameters
    ----------
    posterior
        ``(windows, states)`` for every window of the recording.
    rows, labels
        Positions of the labelled windows and their state indices.
    silent_count
        Number of silent channels in every window.
    channels
        Number of instrumented channels.
    """
    chosen = posterior[rows]
    confidence = chosen.max(axis=1)
    predicted = chosen.argmax(axis=1)
    correct = (predicted == labels).astype(float)
    gap = confidence - correct
    silent = silent_count[rows]
    positions = quiet_positions(silent_count, channels)[rows]
    quiet = (positions >= 1) & (positions <= accumulation_steps)
    by_count = {
        str(k): {
            "windows": int(np.sum(silent == k)),
            "confidence": float(confidence[silent == k].mean()),
            "accuracy": float(correct[silent == k].mean()),
        }
        for k in range(channels + 1)
        if np.any(silent == k)
    }
    by_position = {
        str(j): {
            "windows": int(np.sum(positions == j)),
            "confidence": float(confidence[positions == j].mean()),
            "accuracy": float(correct[positions == j].mean()),
        }
        for j in range(1, accumulation_steps + 1)
        if np.any(positions == j)
    }
    return {
        "windows": int(rows.size),
        "gap": float(gap.mean()) if gap.size else None,
        "gap_slope": _within_state_slope(gap, silent, predicted),
        "confidence_slope": _within_state_slope(confidence, silent, predicted),
        "accuracy_slope": _within_state_slope(correct, silent, predicted),
        "accumulation_slope": (
            _within_state_slope(
                gap[quiet], positions[quiet] / steps_per_hour, predicted[quiet]
            )
            if quiet.any()
            else None
        ),
        "by_silent_channels": by_count,
        "by_quiet_position": by_position,
    }


def representative_runs(
    fully_silent: np.ndarray,
    truth: Sequence[int | None],
    state: int,
    *,
    min_steps: int,
) -> tuple[int, int] | None:
    """The pre-specified quiet run for one state: ``(start, length)`` or ``None``.

    Runs are maximal stretches of consecutive windows that are fully silent and
    labelled *state*, at least *min_steps* long. The case is the run of median
    length, the lower median, the earliest among equal lengths.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for row, (silent, label) in enumerate(zip(fully_silent, truth, strict=True)):
        inside = bool(silent) and label == state
        if inside and start is None:
            start = row
        if not inside and start is not None:
            runs.append((start, row - start))
            start = None
    if start is not None:
        runs.append((start, len(truth) - start))
    long_enough = [(begin, length) for begin, length in runs if length >= min_steps]
    if not long_enough:
        return None
    lengths = sorted(length for _, length in long_enough)
    target = lengths[(len(lengths) - 1) // 2]
    begin = min(begin for begin, length in long_enough if length == target)
    return begin, target


def decompose_case(
    trajectory: Trajectory,
    start: int,
    steps: int,
    channels: Sequence[str],
    states: Sequence[BehaviouralState],
    moments: Sequence[Any],
    counts: np.ndarray,
) -> list[dict[str, Any]]:
    """Each step of a quiet run: prediction, each silent channel, posterior."""
    out: list[dict[str, Any]] = []
    for offset in range(steps):
        row = start + offset
        predicted = trajectory.predicted[row]
        silent = [c for c in range(len(channels)) if counts[row, c] == 0]
        belief = predicted.copy()
        contributions = []
        for c in silent:
            loglik = trajectory.channel_loglik[row, c]
            belief = belief * np.exp(loglik)
            belief = belief / belief.sum()
            contributions.append(
                {
                    "channel": channels[c],
                    "log_likelihood": loglik.tolist(),
                    "cumulative_confidence": float(belief.max()),
                }
            )
        posterior = trajectory.posterior[row]
        out.append(
            {
                "step": offset + 1,
                "moment": moments[row].isoformat(),
                "predicted": predicted.tolist(),
                "predicted_confidence": float(predicted.max()),
                "silent_channels": len(silent),
                "channels": contributions,
                "posterior": posterior.tolist(),
                "confidence": float(posterior.max()),
                "state": states[int(posterior.argmax())].value,
            }
        )
    return out


# ----------------------------------------------------------------------------
# One household
# ----------------------------------------------------------------------------
def household_diagnostic(
    recording: CasasRecording,
    protocol: SilenceProtocol,
    *,
    household: str,
    ontology: StateOntology | None = None,
) -> dict[str, Any]:
    """Every per-household quantity of the diagnostic."""
    ontology = ontology or StateOntology()
    states = tuple(ontology.states)
    resolution = protocol.resolution
    info = protocol.information_set
    moments = _regular_moments(recording, resolution.step)
    table = build_feature_table(recording, info, moments, household=household)
    built = _Household.build(household, "diagnostic", table, recording)
    terms, ignored = channel_likelihoods(recording.registry, resolution, ontology)
    instrumented: list[EvidenceChannel] = sorted(terms)
    names = [channel.name for channel in instrumented]
    counts = np.column_stack(
        [table.column(evidence_column(channel.name, 0)) for channel in instrumented]
    )
    likelihoods: list[ChannelLikelihood] = [terms[c] for c in instrumented]
    per_activation = np.stack([t.per_activation for t in likelihoods])
    per_window = np.stack([t.per_window for t in likelihoods])
    expected = np.stack([t.expected for t in likelihoods])

    labels = np.array([states.index(label) for label in built.labels], dtype=int)
    rows = built.labelled
    windows = {
        state.value: int(np.sum(labels == index)) for index, state in enumerate(states)
    }
    eligible = [
        index
        for index, state in enumerate(states)
        if windows[state.value] >= protocol.min_windows
    ]
    by_state = {
        states[index].value: state_dependence(
            counts[rows[labels == index]],
            names,
            expected[:, index],
            smoothing=protocol.smoothing,
            min_expected=protocol.min_expected,
        )
        for index in eligible
    }
    declared = {
        states[index].value: float(per_window[:, index].sum()) for index in eligible
    }
    inflation = silence_inflation(by_state, declared)

    trajectory = filter_trajectory(
        counts,
        per_activation,
        per_window,
        ontology.transition(resolution.step),
        ontology.stationary(),
    )
    silent_count = (counts == 0).sum(axis=1)
    steps_per_hour = 3600.0 / resolution.step.total_seconds()
    concentrated = concentration(
        trajectory.posterior,
        rows,
        labels,
        silent_count,
        len(names),
        accumulation_steps=protocol.accumulation_steps,
        steps_per_hour=steps_per_hour,
    )

    truth: list[int | None] = [None] * len(moments)
    for row, label in zip(rows, labels, strict=True):
        truth[int(row)] = int(label)
    fully_silent = silent_count == len(names)
    cases: dict[str, Any] = {}
    for state in protocol.quiet_states:
        index = states.index(state)
        chosen = representative_runs(
            fully_silent, truth, index, min_steps=protocol.case_min_steps
        )
        if chosen is None:
            continue
        start, length = chosen
        cases[state.value] = {
            "start": moments[start].isoformat(),
            "run_windows": length,
            "steps": decompose_case(
                trajectory,
                start,
                protocol.case_steps,
                names,
                states,
                moments,
                counts,
            ),
        }

    quiet = [s.value for s in protocol.quiet_states if s.value in by_state]

    def quiet_mean(key: str) -> float | None:
        values = [by_state[s][key] for s in quiet if by_state[s][key] is not None]
        return float(np.mean(values)) if values else None

    pairs: dict[str, dict[str, float | None]] = {}
    for pair in by_state[quiet[0]]["pairs"] if quiet else ():
        pairs[pair] = {}
        for key in ("log_odds_silence", "log_pair_silence"):
            present = [
                by_state[s]["pairs"][pair][key]
                for s in quiet
                if by_state[s]["pairs"][pair][key] is not None
            ]
            pairs[pair][key] = float(np.mean(present)) if present else None

    declared_minus = (
        inflation["log_inflation_declared"] - inflation["log_inflation"]
        if inflation["log_inflation"] is not None
        and inflation["log_inflation_declared"] is not None
        else None
    )
    values = {
        **inflation,
        "log_marginal_factor": declared_minus,
        "log_joint_silence_quiet": quiet_mean("log_joint_silence"),
        "log_odds_silence_quiet": quiet_mean("log_odds_silence"),
        "count_correlation_quiet": quiet_mean("count_correlation"),
        "log_dispersion_quiet": quiet_mean("log_dispersion"),
        "gap_slope": concentrated["gap_slope"],
        "confidence_slope": concentrated["confidence_slope"],
        "accuracy_slope": concentrated["accuracy_slope"],
        "accumulation_slope": concentrated["accumulation_slope"],
    }
    return {
        "channels": names,
        "ignored_sensors": list(ignored),
        "moments": len(moments),
        "labelled": int(rows.size),
        "windows": windows,
        "eligible_states": [states[index].value for index in eligible],
        "quiet_states": quiet,
        "fully_silent_share": {
            s: float(
                np.mean(counts[rows[labels == states.index(S(s))]].sum(axis=1) == 0)
            )
            for s in by_state
        },
        "values": values,
        "states": by_state,
        "declared_joint_loglik": declared,
        "pairs_quiet": pairs,
        "concentration": concentrated,
        "cases": cases,
    }


# ----------------------------------------------------------------------------
# Across households
# ----------------------------------------------------------------------------
def against_independence(
    values: Mapping[str, float | None], protocol: SilenceProtocol
) -> dict[str, Any]:
    """Mean and median across households against zero, with household intervals."""
    present = {h: v for h, v in values.items() if v is not None and math.isfinite(v)}
    if not present:
        return {"n": 0, "mean": None, "median": None}
    comparison = compare_households(
        present,
        dict.fromkeys(present, 0.0),
        higher_is_better=True,
        confidence=protocol.confidence,
        resamples=protocol.resamples,
        seed=protocol.seed,
    ).to_dict()
    spread = summarise_households(present).to_dict()
    return {
        "n": comparison["n"],
        "values": dict(sorted(present.items())),
        "mean": comparison["mean"],
        "median": comparison["median"],
        "above": comparison["favours_model"],
        "below": comparison["favours_reference"],
        "sd": spread["sd"],
        "min": spread["min"],
        "max": spread["max"],
    }


def read_verdict(summary: Mapping[str, Any], minimal: float) -> str:
    """The pre-specified reading of one across-household summary."""
    if not summary.get("n") or summary["mean"] is None:
        return "uncertain"
    return _VERDICTS[verdict(summary, minimal)]


def conclusion(verdicts: Mapping[str, str]) -> str:
    """Supported, weakened or inconclusive, from the D1 and C1 verdicts."""
    if verdicts["D1"] == "positive" and verdicts["C1"] == "positive":
        return "supported"
    if any(verdicts[key] in ("negligible", "negative") for key in ("D1", "C1")):
        return "weakened"
    return "inconclusive"


def association(
    first: Mapping[str, float | None],
    second: Mapping[str, float | None],
    protocol: SilenceProtocol,
) -> dict[str, Any]:
    """Spearman correlation across households, with a household bootstrap interval."""
    homes = sorted(
        h
        for h in set(first) & set(second)
        if first[h] is not None and second[h] is not None
    )
    if len(homes) < 3:
        return {"n": len(homes), "spearman": None, "interval": None}
    x = np.array([first[h] for h in homes], dtype=float)
    y = np.array([second[h] for h in homes], dtype=float)
    rho = float(spearmanr(x, y).statistic)
    replicates = []
    for sample in resample_indices(len(homes), protocol.resamples, protocol.seed):
        if np.unique(x[sample]).size < 2 or np.unique(y[sample]).size < 2:
            continue
        replicates.append(float(spearmanr(x[sample], y[sample]).statistic))
    interval = (
        percentile_interval(np.array(replicates), protocol.confidence).to_dict()
        if replicates
        else None
    )
    return {"n": len(homes), "spearman": rho, "interval": interval}


@dataclass(frozen=True)
class SilenceResult:
    """A completed diagnostic: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def run_silence_diagnostic(
    recordings: Mapping[str, CasasRecording],
    protocol: SilenceProtocol,
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> SilenceResult:
    """Run the pre-specified diagnostic and build its record."""
    missing = sorted(set(protocol.households) - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    ontology = StateOntology()
    states = [state.value for state in ontology.states]
    households = {
        home: household_diagnostic(
            recordings[home], protocol, household=home, ontology=ontology
        )
        for home in protocol.households
    }

    estimands: list[dict[str, Any]] = []
    for estimand in ESTIMANDS:
        summary = against_independence(
            {h: d["values"][estimand.quantity] for h, d in households.items()},
            protocol,
        )
        estimands.append(
            {
                **estimand.to_dict(),
                "summary": summary,
                "verdict": read_verdict(
                    summary, protocol.minimal_effects[estimand.scale]
                ),
            }
        )
    verdicts = {e["key"]: e["verdict"] for e in estimands}

    scales = {
        "log_joint_silence": "log_ratio",
        "log_joint_silence_poisson": "log_ratio",
        "log_joint_silence_declared": "log_ratio",
        "joint_silence_difference": None,
        "log_odds_silence": "log_ratio",
        "log_pair_silence": "log_ratio",
        "count_correlation": "correlation",
        "log_dispersion": "log_ratio",
    }
    per_state: dict[str, dict[str, Any]] = {}
    for state in states:
        entry: dict[str, Any] = {}
        for quantity, scale in scales.items():
            summary = against_independence(
                {
                    h: d["states"][state][quantity]
                    for h, d in households.items()
                    if state in d["states"]
                },
                protocol,
            )
            entry[quantity] = {
                "summary": summary,
                "verdict": (
                    read_verdict(summary, protocol.minimal_effects[scale])
                    if scale is not None
                    else None
                ),
            }
        per_state[state] = entry

    pair_names = sorted({p for d in households.values() for p in d["pairs_quiet"]})
    per_pair = {
        pair: {
            key: against_independence(
                {
                    h: d["pairs_quiet"][pair][key]
                    for h, d in households.items()
                    if pair in d["pairs_quiet"]
                },
                protocol,
            )
            for key in ("log_odds_silence", "log_pair_silence")
        }
        for pair in pair_names
    }
    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified diagnostic; development panel, which earlier "
        "work has inspected",
        "states": states,
        "quiet_states": [s.value for s in protocol.quiet_states],
        "estimands": estimands,
        "conclusion": conclusion(verdicts),
        "per_state": per_state,
        "per_pair": per_pair,
        "association": association(
            {h: d["values"]["log_inflation"] for h, d in households.items()},
            {h: d["values"]["gap_slope"] for h, d in households.items()},
            protocol,
        ),
        "households": households,
    }
    record = ExperimentRecord(
        experiment=protocol.name,
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        inference=ONLINE,
        seeds=[protocol.seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions=dict(DEFINITIONS),
        notes=[
            "Diagnostic only: no inference, emission model or default was changed.",
            "Every setting, estimand and criterion was declared in the protocol "
            "before any household was examined.",
            "Households are the unit of replication; intervals resample "
            "households, and no within-household interval is reported because "
            "windows are autocorrelated.",
            "Verdicts compare effect sizes and household bootstrap intervals with "
            "declared minimal effects; they are not significance tests.",
        ],
        inputs=list(inputs),
        split={"households": list(protocol.households)},
        information_set={
            **protocol.information_set.to_dict(),
            "sha256": protocol.information_set.sha256(),
        },
        preprocessing={
            "moments": "one per step from each recording's first observation to "
            "its last",
            "labels": "truth_series of each recording's annotations; unlabelled "
            "windows enter the filter's recursion but no statistic",
            "step_seconds": protocol.resolution.step.total_seconds(),
        },
        models=[
            ModelRecord("filter_model", dict(MODEL_CONFIGURATION), "filter_trajectory")
        ],
        household_metrics={
            "diagnostic": {
                home: dict(_finite(d["values"])) for home, d in households.items()
            }
        },
        intervals=_intervals(estimands),
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return SilenceResult(record=record, path=path)


def _intervals(estimands: Sequence[Mapping[str, Any]]) -> list[ReportedInterval]:
    reported: list[ReportedInterval] = []
    for entry in estimands:
        summary = entry["summary"]
        for statistic in ("mean", "median"):
            estimate = summary.get(statistic)
            if not estimate or estimate["interval"] is None:
                continue
            reported.append(
                ReportedInterval(
                    label=f"{entry['key']}: {statistic} {entry['quantity']}",
                    estimate=estimate["estimate"],
                    low=estimate["interval"]["low"],
                    high=estimate["interval"]["high"],
                    confidence=estimate["interval"]["confidence"],
                    method=estimate["interval"]["method"],
                    unit="household",
                    n=summary["n"],
                )
            )
    return reported
