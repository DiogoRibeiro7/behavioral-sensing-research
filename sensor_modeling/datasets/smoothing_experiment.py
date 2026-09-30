"""Phase 3.5: how much fixed-lag smoothing recovers, kept apart from online inference.

A fixed-lag smoother revises each window's estimate with up to ``lag`` later
windows. It reads evidence a live system does not yet have, and its estimate
for a window exists only ``lag`` windows after it. This experiment measures
what that buys over the online filter, on households none of the parameters
were fitted on. Every difference it reports is a smoothing gain, available
only after the smoother's delay, never an improvement of the online filter
(:mod:`~sensor_modeling.fusion.regime`).

Every setting, comparison and criterion is declared in
:class:`SmoothingProtocol` and frozen in a committed file before any household
is scored.

The formulation
---------------
Smoothing needs a recursion over consecutive windows. The one on ``develop``
with a pre-specified success in the recursion is the Phase 3.3 follow-up's
``filter_hurdle``: population hurdle channel models, fitted per fold on the
training households, in the filter's recursion over every window of the
recording. Partial pooling in the recursion was inconclusive (Phase 3.4,
P2), and the Phase 3.1 time prior was evaluated only on declared information
sets, never as a recursion. The online filter and every smoother share one
filter pass. Each smoother only revises it, with the declared lag.

What is scored
--------------
- **Windows.** Each held-out household's labelled windows, except the last
  ``max(lags)`` windows of its recording. Every regime is scored on exactly
  those windows, and every smoothed estimate reads its full lag.
- **Metrics.** Balanced accuracy, per-state recall, log loss, Brier score and
  calibration error, per household.
- **Operational metrics.** Relative to the online filter on the same windows:
  the share of reported states changed, the share of the filter's errors
  corrected, and the share of its correct states made wrong. Each regime's
  reporting delay is its lag.
- **Transitions.** Accuracy near true state changes, and each regime's
  decision delay: how long after a true change the regime first reports the
  new state, including its reporting delay.
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

from ..evaluation.households import (
    compare_households,
    household_values,
    recall_of,
    summarise_households,
)
from ..evaluation.metrics import PredictionMetrics
from ..evaluation.provenance import (
    ExperimentRecord,
    InputArtifact,
    ModelRecord,
    ReportedInterval,
)
from ..evaluation.regime_results import (
    SMOOTHING_GAIN,
    RegimeComparison,
    RegimeResult,
    compare_results,
)
from ..fusion.regime import (
    ONLINE,
    EvidenceSummary,
    InferenceRegime,
    regime_beliefs,
)
from ..states.ontology import BehaviouralState, StateOntology
from .casas import CasasRecording
from .channel_models import (
    HURDLE,
    ChannelStatistics,
    FittedChannels,
    combine_statistics,
    filter_recursion,
    home_statistics,
    household_channel_counts,
    total_loglik,
)
from .information_sets import EvidenceChannel, EvidenceResolution
from .matched_evaluation import (
    COMPARABLE_METRICS,
    MATCHED_METRIC_DEFINITIONS,
    HouseholdSplit,
    _finite,
)
from .recoverable_gap import (
    DEFAULT_METRICS,
    GENERATIVE_CONFIGURATION,
    FrozenSplits,
    GapProtocol,
    _cell,
    _score,
)
from .time_prior_experiment import MINIMAL_DIFFERENCES, verdict

PROTOCOL_SCHEMA = "smoothing-protocol/1"
RESULT_SCHEMA = "smoothing-results/1"

S = BehaviouralState

#: The online filter's key; each smoother's is :func:`regime_key` of its lag.
ONLINE_KEY = "online"

#: The Phase 3.3 follow-up cell the online filter reproduces on every labelled window.
REPRODUCED = "filter_hurdle@R"

#: The group of every transition, beside one group per focus state.
ALL_TRANSITIONS = "all"

#: What each declared lag stands for, operationally.
LAG_ROLES: dict[int, str] = {
    1: "short: one window, for a display that may run one window behind",
    6: "medium: half an hour, for a non-urgent check within the hour, such as "
    "whether the resident has got up",
    12: "long: one hour, for retrospective hourly or daily reporting; the "
    "longest delay treated as plausible for a decision made the same day",
}

#: Minimal important differences beyond the Phase 3 metrics.
TRANSITION_MINIMAL_DIFFERENCES: dict[str, float] = {
    "boundary_accuracy": 0.02,
    "detection_rate": 0.05,
    "decision_delay_minutes": 5.0,
}

#: Whether a higher value of each compared quantity is better.
HIGHER_IS_BETTER: dict[str, bool] = {
    **COMPARABLE_METRICS,
    "boundary_accuracy": True,
    "detection_rate": True,
    "decision_delay_minutes": False,
}

CRITERIA: dict[str, str] = {
    "verdicts": "favours model: mean >= delta and lower bound > 0; favours "
    "reference: mean <= -delta and upper bound < 0; negligible: the whole "
    "interval within (-delta, delta); uncertain: anything else",
    "gain": "balanced accuracy favours the smoother, and neither log loss nor "
    "calibration error favours the online filter",
    "trade-off": "balanced accuracy favours the smoother, and log loss or "
    "calibration error favours the online filter",
    "probability gain": "balanced accuracy does not favour the smoother, log loss "
    "does, and neither balanced accuracy nor calibration error favours the "
    "online filter",
    "no gain": "balanced accuracy is negligible or favours the online filter, and "
    "log loss is negligible or favours the online filter",
    "inconclusive": "anything else",
    "primary": "each smoother against the online filter on the same windows; "
    "every estimand is primary and read by the same rule",
    "not_online": "every difference is a smoothing gain: the smoother reads up to "
    "its lag after each window and reports that long after it, so no difference "
    "is an improvement of the online filter",
}

SMOOTHING_METRIC_DEFINITIONS: dict[str, str] = {
    **MATCHED_METRIC_DEFINITIONS,
    "reporting_delay_minutes": "How long after its window a regime's estimate "
    "can first be reported: zero online, the lag for a smoother.",
    "changed": "Share of a household's scored windows whose most probable state "
    "under the smoother differs from the online filter's.",
    "corrected": "Share of a household's scored windows where the online filter's "
    "most probable state is wrong that the smoother gets right.",
    "broken": "Share of a household's scored windows where the online filter's "
    "most probable state is right that the smoother gets wrong.",
    "corrections_among_changes": "Share of a household's changed windows where "
    "the smoother is right and the online filter wrong.",
    "transition": "A scored window whose labelled state differs from the "
    "labelled state of the window before it. Its episode runs while the "
    "following windows keep that labelled state, up to the last scored window; "
    "an unlabelled window ends it.",
    "boundary_accuracy": "Share of a household's scored windows within "
    "boundary_windows of a transition, before or after it, whose most probable "
    "state is right.",
    "interior_accuracy": "The same, for the scored windows further from every "
    "transition.",
    "detection_rate": "Share of a household's transitions after which the regime "
    "reports the new state at some window of the new episode.",
    "decision_delay_minutes": "The median over a household's detected transitions "
    "of the time from the transition to the moment the regime can first report "
    "the new state: the windows until its most probable state first equals the "
    "new state, plus its lag, times the step. Lower is better.",
}


def regime_key(lag: int) -> str:
    """The key of the regime with *lag* windows of lag: ``online`` for zero."""
    return ONLINE_KEY if lag == 0 else f"lag_{lag}"


def smoothing_conclusion(verdicts: Mapping[str, str]) -> str:
    """Gain, trade-off, probability gain, no gain or inconclusive, for one smoother."""
    accuracy = verdicts["balanced_accuracy"]
    loss = verdicts["log_loss"]
    calibration = verdicts["calibration_error"]
    against = "favours reference"
    if accuracy == "favours model":
        return "trade-off" if against in (loss, calibration) else "gain"
    if loss == "favours model" and against not in (accuracy, calibration):
        return "probability gain"
    if accuracy in ("negligible", against) and loss in ("negligible", against):
        return "no gain"
    return "inconclusive"


@dataclass(frozen=True)
class SmoothingProtocol:
    """Everything the experiment fixes before any household is scored."""

    folds: tuple[HouseholdSplit, ...]
    splits_sha256: str
    resolution: EvidenceResolution = field(default_factory=EvidenceResolution)
    pseudo_windows: float = 12.0
    lags: tuple[int, ...] = (1, 6, 12)
    boundary_windows: int = 6
    focus_states: tuple[BehaviouralState, ...] = (S.AWAY, S.HOME_ACTIVE, S.SLEEPING)
    seed: int = 0
    metrics: tuple[str, ...] = DEFAULT_METRICS
    resamples: int = 10_000
    confidence: float = 0.95
    minimal_differences: Mapping[str, float] = field(
        default_factory=lambda: {
            **MINIMAL_DIFFERENCES,
            **TRANSITION_MINIMAL_DIFFERENCES,
        }
    )
    rare_share: float = 0.05
    name: str = "phase3-fixed-lag-smoothing"

    def __post_init__(self) -> None:
        """Validate through the recoverable-gap protocol, then the rest."""
        base = self.as_gap_protocol()
        if not isinstance(self.splits_sha256, str) or len(self.splits_sha256) != 64:
            raise ValueError("splits_sha256 must be a SHA-256 hex digest")
        if not math.isfinite(self.pseudo_windows) or self.pseudo_windows <= 0.0:
            raise ValueError("pseudo_windows must be positive")
        lags = tuple(self.lags)
        if (
            not lags
            or any(isinstance(g, bool) or not isinstance(g, int) or g < 1 for g in lags)
            or list(lags) != sorted(set(lags))
        ):
            raise ValueError("lags must be distinct increasing positive integers")
        missing_roles = sorted(set(lags) - set(LAG_ROLES))
        if missing_roles:
            raise ValueError(f"lags {missing_roles} have no declared operational role")
        window = self.boundary_windows
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("boundary_windows must be a positive integer")
        focus = tuple(BehaviouralState(s) for s in self.focus_states)
        if not focus or len(set(focus)) != len(focus):
            raise ValueError("focus_states must be distinct and non-empty")
        if any(s not in StateOntology().states for s in focus):
            raise ValueError("every focus state must be an ontology state")
        needed = {*base.metrics, "recall", *TRANSITION_MINIMAL_DIFFERENCES}
        missing = sorted(needed - set(self.minimal_differences))
        if missing or any(
            not math.isfinite(v) or v <= 0.0 for v in self.minimal_differences.values()
        ):
            raise ValueError(
                f"every compared quantity needs a positive minimal difference; "
                f"missing {missing}"
            )
        if not 0.0 < self.rare_share < 1.0:
            raise ValueError("rare_share must lie in (0, 1)")
        object.__setattr__(self, "folds", base.folds)
        object.__setattr__(self, "metrics", base.metrics)
        object.__setattr__(self, "lags", lags)
        object.__setattr__(self, "focus_states", focus)
        object.__setattr__(self, "minimal_differences", dict(self.minimal_differences))

    def as_gap_protocol(self) -> GapProtocol:
        """The equivalent recoverable-gap protocol, for validation and helpers."""
        return GapProtocol(
            tuple(self.folds),
            seed=self.seed,
            metrics=tuple(self.metrics),
            resamples=self.resamples,
            confidence=self.confidence,
        )

    @property
    def homes(self) -> tuple[str, ...]:
        """Every held-out household, sorted."""
        return tuple(sorted(home for fold in self.folds for home in fold.test))

    @property
    def max_lag(self) -> int:
        """The longest declared lag, in windows."""
        return self.lags[-1]

    @property
    def regimes(self) -> dict[str, InferenceRegime]:
        """The online filter, then each smoother, by key."""
        step = self.resolution.step
        return {
            ONLINE_KEY: ONLINE,
            **{
                regime_key(lag): InferenceRegime.smoother(lag, step)
                for lag in self.lags
            },
        }

    def estimand_key(self, lag: int) -> str:
        """The key of the estimand comparing the smoother with *lag* to the filter."""
        return f"G{lag}"

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        step_minutes = self.resolution.step.total_seconds() / 60.0
        return {
            "schema": PROTOCOL_SCHEMA,
            "name": self.name,
            "households": {
                "splits_file": "artifacts/phase1/household_splits.json",
                "splits_sha256": self.splits_sha256,
                "folds": [
                    {**fold.to_dict(), "sha256": fold.sha256()} for fold in self.folds
                ],
                "use": "each household is held out once; its fold's training "
                "households fit the channel models",
            },
            "formulation": {
                **GENERATIVE_CONFIGURATION,
                "model": "the Phase 3.3 follow-up's filter_hurdle recursion",
                "channels": "hurdle, fitted per fold on the training households as "
                "in the Phase 3.3 follow-up",
                "pseudo_windows": self.pseudo_windows,
                "prior": "stationary distribution of the default ontology, one step "
                "before the first window of the recording",
                "windows": "every window of the recording, each channel's current "
                "count",
                "fitted": "the population hurdle parameters, per fold, on the "
                "fold's training households' labelled windows only",
                "why": "it is the recursion over every window with a pre-specified "
                "success on develop (Phase 3.3 follow-up, FR); partial pooling in "
                "the recursion was inconclusive (Phase 3.4, P2), and the Phase 3.1 "
                "time prior was evaluated only on declared information sets, never "
                "as a recursion",
            },
            "regimes": {
                key: {
                    **regime.to_dict(),
                    "role": (
                        "reference: what a live system reports at each window"
                        if regime.is_online
                        else LAG_ROLES[regime.lag_steps]
                    ),
                }
                for key, regime in self.regimes.items()
            },
            "lags": {
                "windows": list(self.lags),
                "step_minutes": step_minutes,
                "why": "a small set declared before scoring and never searched: "
                "short, medium and long delays with an operational use each; lags "
                "beyond an hour serve only offline analysis and are not scored",
                "smoother": "smooth_beliefs through regime_beliefs: each window's "
                "filtered belief revised with the filtered beliefs of up to lag "
                "later windows and the transition, starting every correction from "
                "the filtered belief at its horizon",
            },
            "scored_windows": "each held-out household's labelled windows except "
            f"the last {self.max_lag} of its recording, identical for every "
            "regime, so every smoothed estimate reads its full lag",
            "reproduction": "the online filter is also scored on every labelled "
            f"window, as {REPRODUCED}, to reproduce the Phase 3.3 follow-up",
            "metrics": list(self.metrics),
            "calibration": "expected calibration error over 10 confidence bins",
            "per_state_recall": {
                "states": "every state, for every estimand",
                "focus_states": [s.value for s in self.focus_states],
                "rare_rule": f"share of labelled time below {self.rare_share} in "
                "either fold's training households",
            },
            "operational": {
                "reporting_delay": "the regime's lag times the step",
                "changed": SMOOTHING_METRIC_DEFINITIONS["changed"],
                "corrected": SMOOTHING_METRIC_DEFINITIONS["corrected"],
                "broken": SMOOTHING_METRIC_DEFINITIONS["broken"],
                "corrections_among_changes": SMOOTHING_METRIC_DEFINITIONS[
                    "corrections_among_changes"
                ],
                "by_state": "the same shares within the scored windows whose "
                "labelled state is each focus state",
                "summary": "one value per household; the mean and the median "
                "across households with household bootstrap intervals; no verdict",
            },
            "transitions": {
                "transition": SMOOTHING_METRIC_DEFINITIONS["transition"],
                "boundary_windows": self.boundary_windows,
                "boundary_accuracy": SMOOTHING_METRIC_DEFINITIONS["boundary_accuracy"],
                "interior_accuracy": SMOOTHING_METRIC_DEFINITIONS["interior_accuracy"],
                "detection_rate": SMOOTHING_METRIC_DEFINITIONS["detection_rate"],
                "decision_delay_minutes": SMOOTHING_METRIC_DEFINITIONS[
                    "decision_delay_minutes"
                ],
                "groups": [ALL_TRANSITIONS, *(s.value for s in self.focus_states)],
                "group_rule": "all transitions, and the transitions into each focus "
                "state",
                "comparisons": "boundary accuracy, and each group's detection rate "
                "and decision delay, as paired household differences against the "
                "online filter, read with the verdicts",
            },
            "estimands": [
                {
                    "key": self.estimand_key(lag),
                    "role": "primary",
                    "question": f"how much does the fixed-lag smoother with lag {lag} "
                    "recover over the online filter",
                    "model": regime_key(lag),
                    "reference": ONLINE_KEY,
                    "kind": SMOOTHING_GAIN,
                }
                for lag in self.lags
            ],
            "minimal_differences": dict(self.minimal_differences),
            "criteria": dict(CRITERIA),
            "bootstrap": {
                "unit": "household",
                "statistics": "mean and median paired differences",
                "resamples": self.resamples,
                "confidence": self.confidence,
                "interval": "percentile",
            },
            "seeds": {"bootstrap": self.seed},
            "inference": "the online filter reads no window after the scored one; "
            "each smoother reads up to its lag after it and reports that long "
            "after it; the record states the longest lag, which bounds every "
            "estimate in it, and labels every cell and comparison with its own "
            "regime",
            "tuning": "none: the lags, the boundary and the focus states are "
            "declared; nothing is fitted on held-out households",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_protocol(splits: FrozenSplits) -> SmoothingProtocol:
    """The protocol as declared for the development panel's frozen folds."""
    return SmoothingProtocol(splits.folds, splits.sha256)


def check_frozen_protocol(protocol: SmoothingProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after scoring begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


# ----------------------------------------------------------------------------
# One household
# ----------------------------------------------------------------------------
def regime_posteriors(
    loglik: np.ndarray,
    transition: np.ndarray,
    prior: np.ndarray,
    regimes: Mapping[str, InferenceRegime],
) -> dict[str, np.ndarray]:
    """Each regime's posterior at every window, from one pass of the filter.

    The online filter reports the filtered posterior. Each smoother revises it
    with the declared lag, through :func:`~sensor_modeling.fusion.regime_beliefs`.
    """
    _, filtered = filter_recursion(loglik, transition, prior)
    return {
        key: regime_beliefs(filtered, transition, regime)
        for key, regime in regimes.items()
    }


def state_changes(
    truth: np.ndarray, online: np.ndarray, smoothed: np.ndarray
) -> dict[str, int]:
    """How the smoother's reported states differ from the online filter's.

    All three arrays hold state indices at the same scored windows.
    """
    online_right = online == truth
    smoothed_right = smoothed == truth
    changed = online != smoothed
    return {
        "windows": int(truth.size),
        "changed": int(changed.sum()),
        "online_right": int(online_right.sum()),
        "online_wrong": int((~online_right).sum()),
        "corrected": int((~online_right & smoothed_right).sum()),
        "broken": int((online_right & ~smoothed_right).sum()),
    }


def change_shares(counts: Mapping[str, int]) -> dict[str, float | None]:
    """The operational shares of :func:`state_changes` counts; ``None`` if undefined."""

    def share(part: str, whole: str) -> float | None:
        return counts[part] / counts[whole] if counts[whole] else None

    return {
        "changed": share("changed", "windows"),
        "corrected": share("corrected", "online_wrong"),
        "broken": share("broken", "online_right"),
        "corrections_among_changes": share("corrected", "changed"),
    }


@dataclass(frozen=True)
class Transition:
    """One true change of state between consecutive labelled windows.

    Attributes
    ----------
    row
        The first window of the new state.
    source, target
        The state indices before and after.
    end
        The last window of the new episode, at most the last scored window.
    """

    row: int
    source: int
    target: int
    end: int


def find_transitions(truth: np.ndarray, last: int) -> list[Transition]:
    """Every transition at or before window *last*.

    *truth* holds each window's labelled state index, and ``-1`` where the
    window is unlabelled. A change is counted only between two consecutive
    labelled windows.
    """
    found: list[Transition] = []
    for row in range(1, min(last, truth.size - 1) + 1):
        before, now = int(truth[row - 1]), int(truth[row])
        if before < 0 or now < 0 or before == now:
            continue
        end = row
        while end + 1 <= last and int(truth[end + 1]) == now:
            end += 1
        found.append(Transition(row, before, now, end))
    return found


def boundary_mask(
    size: int, transitions: Sequence[Transition], window: int
) -> np.ndarray:
    """Windows within *window* of a transition: the *window* before it and its first *window*."""
    mask = np.zeros(size, dtype=bool)
    for item in transitions:
        mask[max(0, item.row - window) : min(size, item.row + window)] = True
    return mask


def decision_delays(
    predicted: np.ndarray, transitions: Sequence[Transition], lag: int
) -> list[int | None]:
    """Windows from each transition until the regime can report its new state.

    That is the windows until the regime's most probable state first equals
    the new state within the new episode, plus its lag. ``None`` where it never
    does.
    """
    delays: list[int | None] = []
    for item in transitions:
        hits = np.flatnonzero(predicted[item.row : item.end + 1] == item.target)
        delays.append(None if hits.size == 0 else int(hits[0]) + lag)
    return delays


def _transition_summary(
    delays: Sequence[int | None], step_minutes: float
) -> dict[str, Any]:
    detected = [d for d in delays if d is not None]
    return {
        "transitions": len(delays),
        "detected": len(detected),
        "detection_rate": len(detected) / len(delays) if delays else None,
        "decision_delay_minutes": (
            float(median(detected)) * step_minutes if detected else None
        ),
    }


def _accuracy(predicted: np.ndarray, truth: np.ndarray) -> float | None:
    return float(np.mean(predicted == truth)) if truth.size else None


def household_scores(
    recording: CasasRecording,
    protocol: SmoothingProtocol,
    population: FittedChannels,
    *,
    household: str,
    ontology: StateOntology,
) -> dict[str, Any]:
    """Every regime's scores and operational values for one held-out household."""
    space = tuple(ontology.states)
    resolution = protocol.resolution
    step_minutes = resolution.step.total_seconds() / 60.0
    regimes = protocol.regimes
    counts, rows, labels, moments = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    size = len(moments)
    last = size - 1 - protocol.max_lag
    models = population.models(recording.registry, resolution, HURDLE, ontology)
    transition = ontology.transition(resolution.step)
    beliefs = regime_posteriors(
        total_loglik(models, counts), transition, ontology.stationary(), regimes
    )

    truth = np.full(size, -1, dtype=int)
    truth[rows] = labels
    scored = rows[rows <= last]
    scored_truth = truth[scored]
    predicted = {key: np.argmax(b, axis=1) for key, b in beliefs.items()}

    scores: dict[str, PredictionMetrics] = {
        key: _score([space[i] for i in scored_truth], beliefs[key][scored], space)
        for key in regimes
    }
    reproduction = _score([space[i] for i in labels], beliefs[ONLINE_KEY][rows], space)

    evidence = {
        key: (
            EvidenceSummary.of(
                (moments[int(r)], moments[int(r) + regime.lag_steps]) for r in scored
            )
            if scored.size
            else None
        )
        for key, regime in regimes.items()
    }

    changes: dict[str, Any] = {}
    online_scored = predicted[ONLINE_KEY][scored]
    for lag in protocol.lags:
        key = regime_key(lag)
        smoothed = predicted[key][scored]
        entry: dict[str, Any] = state_changes(scored_truth, online_scored, smoothed)
        entry["by_state"] = {}
        for state in protocol.focus_states:
            within = scored_truth == space.index(state)
            entry["by_state"][state.value] = state_changes(
                scored_truth[within], online_scored[within], smoothed[within]
            )
        changes[key] = entry

    found = find_transitions(truth, last)
    near = boundary_mask(size, found, protocol.boundary_windows)[scored]
    groups: dict[str, list[int]] = {
        ALL_TRANSITIONS: list(range(len(found))),
        **{
            state.value: [
                i for i, t in enumerate(found) if t.target == space.index(state)
            ]
            for state in protocol.focus_states
        },
    }
    by_regime: dict[str, Any] = {}
    for key, regime in regimes.items():
        guessed = predicted[key][scored]
        delays = decision_delays(predicted[key], found, regime.lag_steps)
        by_regime[key] = {
            "reporting_delay_minutes": regime.delay.total_seconds() / 60.0,
            "boundary_accuracy": _accuracy(guessed[near], scored_truth[near]),
            "interior_accuracy": _accuracy(guessed[~near], scored_truth[~near]),
            "transitions": {
                group: _transition_summary([delays[i] for i in members], step_minutes)
                for group, members in groups.items()
            },
        }

    return {
        "scores": scores,
        "reproduction": reproduction,
        "evidence": evidence,
        "windows": size,
        "labelled": int(rows.size),
        "scored": int(scored.size),
        "boundary_windows_scored": int(near.sum()),
        "changes": changes,
        "regimes": by_regime,
    }


# ----------------------------------------------------------------------------
# Across households
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SmoothingResult:
    """A completed experiment: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def _shares(
    fold: HouseholdSplit,
    space: tuple[BehaviouralState, ...],
    statistics: Mapping[str, Mapping[EvidenceChannel, ChannelStatistics]],
) -> dict[str, float]:
    """Each state's share of labelled windows in the fold's training households."""
    totals = np.zeros(len(space))
    for home in fold.train:
        parts = statistics[home]
        if parts:
            totals += next(iter(parts.values())).windows
    return {state.value: float(v) for state, v in zip(space, totals / totals.sum())}


def _across(
    values: Mapping[str, float | None], protocol: SmoothingProtocol
) -> dict[str, Any]:
    """One value per household: mean and median with household intervals, no verdict."""
    present = {h: v for h, v in values.items() if v is not None and math.isfinite(v)}
    if not present:
        return {"n": 0, "values": {}, "mean": None, "median": None}
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
        "min": spread["min"],
        "max": spread["max"],
    }


def _change_share(
    households: Mapping[str, Mapping[str, Any]],
    key: str,
    name: str,
    state: str | None = None,
) -> dict[str, float | None]:
    """One operational share per household, overall or within one focus state."""
    values: dict[str, float | None] = {}
    for home, entry in households.items():
        counts = entry["changes"][key]
        if state is not None:
            counts = counts["by_state"][state]
        values[home] = change_shares(counts)[name]
    return values


def run_smoothing(
    recordings: Mapping[str, CasasRecording],
    protocol: SmoothingProtocol,
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> SmoothingResult:
    """Run the pre-specified experiment and build its record."""
    homes = sorted({h for fold in protocol.folds for h in (*fold.train, *fold.test)})
    missing = sorted(set(homes) - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    ontology = StateOntology()
    space = tuple(ontology.states)
    gap = protocol.as_gap_protocol()
    regimes = protocol.regimes
    statistics = {
        home: home_statistics(
            recordings[home], protocol.resolution, ontology, household=home
        )
        for home in homes
    }
    fits: dict[str, dict[str, Any]] = {}
    for fold in protocol.folds:
        fits[fold.name] = {
            "population": combine_statistics(
                {h: statistics[h] for h in fold.train},
                states=space,
                pseudo_windows=protocol.pseudo_windows,
            ),
            "shares": _shares(fold, space, statistics),
        }
    rare = sorted(
        s.value
        for s in space
        if any(f["shares"][s.value] < protocol.rare_share for f in fits.values())
    )

    scores: dict[str, dict[str, PredictionMetrics]] = {}
    reproduction: dict[str, PredictionMetrics] = {}
    evidence_parts: dict[str, list[EvidenceSummary]] = {key: [] for key in regimes}
    households: dict[str, dict[str, Any]] = {}
    for fold in protocol.folds:
        for home in sorted(fold.test):
            scored = household_scores(
                recordings[home],
                protocol,
                fits[fold.name]["population"],
                household=home,
                ontology=ontology,
            )
            for key, metrics in scored.pop("scores").items():
                scores.setdefault(key, {})[home] = metrics
            reproduction[home] = scored.pop("reproduction")
            for key, summary in scored.pop("evidence").items():
                if summary is not None:
                    evidence_parts[key].append(summary)
            households[home] = {"fold": fold.name, **scored}
    if len(households) < 2:
        raise ValueError("paired household comparisons need at least two households")
    evidence = {
        key: EvidenceSummary.combine(parts) for key, parts in evidence_parts.items()
    }

    def labelled(key: str, values: Mapping[str, float | None]) -> RegimeResult:
        return RegimeResult(regimes[key], values, evidence[key])

    def gain(
        key: str,
        model: Mapping[str, float | None],
        online: Mapping[str, float | None],
        quantity: str,
    ) -> RegimeComparison | None:
        if not any(
            model.get(h) is not None and online.get(h) is not None for h in model
        ):
            return None
        return compare_results(
            labelled(key, model),
            labelled(ONLINE_KEY, online),
            smoothing_gain=True,
            higher_is_better=HIGHER_IS_BETTER[quantity],
            confidence=protocol.confidence,
            resamples=protocol.resamples,
            seed=protocol.seed,
        )

    def judged(comparison: RegimeComparison | None, quantity: str) -> dict[str, Any]:
        if comparison is None:
            return {"comparison": None, "verdict": "uncertain"}
        payload = comparison.comparison.to_dict()
        return {
            "comparison": payload,
            "verdict": verdict(payload, protocol.minimal_differences[quantity]),
        }

    def metric_values(key: str, metric: Any) -> dict[str, float | None]:
        return household_values(scores[key], metric)

    def per_home(key: str, *path: str) -> dict[str, float | None]:
        values: dict[str, float | None] = {}
        for home, entry in households.items():
            node: Any = entry["regimes"][key]
            for part in path:
                node = node[part]
            values[home] = node
        return values

    cells = [
        {
            **_cell(key, "R", scores[key], space, gap),
            "regime": key,
            "inference": regime.to_dict(),
        }
        for key, regime in regimes.items()
    ]

    estimands: list[dict[str, Any]] = []
    for lag in protocol.lags:
        key = regime_key(lag)
        compared = {
            metric: gain(
                key,
                metric_values(key, metric),
                metric_values(ONLINE_KEY, metric),
                metric,
            )
            for metric in protocol.metrics
        }
        labels = next(c for c in compared.values() if c is not None).to_dict()
        if labels["kind"] != SMOOTHING_GAIN:  # pragma: no cover - by construction
            raise ValueError("every estimand must be a smoothing gain")
        metric_entries = {
            metric: judged(comparison, metric)
            for metric, comparison in compared.items()
        }
        verdicts = {
            metric: entry["verdict"] for metric, entry in metric_entries.items()
        }
        recall = {}
        for state in space:
            entry = judged(
                gain(
                    key,
                    metric_values(key, recall_of(state)),
                    metric_values(ONLINE_KEY, recall_of(state)),
                    "balanced_accuracy",
                ),
                "recall",
            )
            recall[state.value] = {
                "rare": state.value in rare,
                "focus": state in protocol.focus_states,
                **entry,
            }

        operational = {
            "reporting_delay_minutes": regimes[key].delay.total_seconds() / 60.0,
            **{
                name: _across(_change_share(households, key, name), protocol)
                for name in (
                    "changed",
                    "corrected",
                    "broken",
                    "corrections_among_changes",
                )
            },
            "by_state": {
                state.value: {
                    name: _across(
                        _change_share(households, key, name, state.value), protocol
                    )
                    for name in ("changed", "corrected", "broken")
                }
                for state in protocol.focus_states
            },
        }

        transitions: dict[str, Any] = {
            "boundary_accuracy": judged(
                gain(
                    key,
                    per_home(key, "boundary_accuracy"),
                    per_home(ONLINE_KEY, "boundary_accuracy"),
                    "boundary_accuracy",
                ),
                "boundary_accuracy",
            ),
            "groups": {},
        }
        for group in (ALL_TRANSITIONS, *(s.value for s in protocol.focus_states)):
            transitions["groups"][group] = {
                quantity: judged(
                    gain(
                        key,
                        per_home(key, "transitions", group, quantity),
                        per_home(ONLINE_KEY, "transitions", group, quantity),
                        quantity,
                    ),
                    quantity,
                )
                for quantity in ("detection_rate", "decision_delay_minutes")
            }

        estimands.append(
            {
                "key": protocol.estimand_key(lag),
                "role": "primary",
                "lag": lag,
                "model": key,
                "reference": ONLINE_KEY,
                "kind": labels["kind"],
                "label": labels["label"],
                "model_inference": labels["model_inference"],
                "reference_inference": labels["reference_inference"],
                "comparisons": {
                    metric: entry["comparison"]
                    for metric, entry in metric_entries.items()
                },
                "verdicts": verdicts,
                "conclusion": smoothing_conclusion(verdicts),
                "per_state_recall": recall,
                "operational": operational,
                "transitions": transitions,
            }
        )

    levels = {
        key: {
            "boundary_accuracy": _across(per_home(key, "boundary_accuracy"), protocol),
            "interior_accuracy": _across(per_home(key, "interior_accuracy"), protocol),
            "transitions": {
                group: {
                    quantity: _across(
                        per_home(key, "transitions", group, quantity), protocol
                    )
                    for quantity in ("detection_rate", "decision_delay_minutes")
                }
                for group in (
                    ALL_TRANSITIONS,
                    *(s.value for s in protocol.focus_states),
                )
            },
        }
        for key in regimes
    }

    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified; development panel, which earlier work has inspected",
        "states": [state.value for state in space],
        "focus_states": [state.value for state in protocol.focus_states],
        "rare_states": rare,
        "regimes": {
            key: {
                **regime.to_dict(),
                "evidence": evidence[key].to_dict(),
            }
            for key, regime in regimes.items()
        },
        "households": dict(sorted(households.items())),
        "cells": cells,
        "transition_levels": levels,
        "estimands": estimands,
        "conclusions": {e["key"]: e["conclusion"] for e in estimands},
        "fitted": {
            fold: {
                "population": {
                    **fit["population"].to_dict(),
                    "sha256": fit["population"].sha256(),
                },
                "training_shares": fit["shares"],
            }
            for fold, fit in fits.items()
        },
    }
    bound = regimes[regime_key(protocol.max_lag)]
    record = ExperimentRecord(
        experiment=protocol.name,
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        inference=bound,
        evidence=EvidenceSummary.combine(evidence.values()),
        seeds=[protocol.seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions=SMOOTHING_METRIC_DEFINITIONS,
        notes=[
            "Every setting, estimand and criterion was declared in the protocol "
            "before any household was scored.",
            "The record compares inference regimes. Its inference field states the "
            "longest lag, which bounds every estimate in it; every cell and "
            "comparison in the results carries its own regime.",
            "Every difference is a smoothing gain, available only after the "
            "smoother's lag, never an improvement of the online filter.",
            "Every regime is scored on the same labelled windows; the last "
            f"{protocol.max_lag} windows of each recording are not scored, so "
            "every smoothed estimate reads its full lag.",
            "Verdicts compare effect sizes and household bootstrap intervals with "
            "declared minimal differences; they are not significance tests.",
        ],
        inputs=list(inputs),
        split={
            "folds": [{**f.to_dict(), "sha256": f.sha256()} for f in protocol.folds]
        },
        information_set={
            "name": "recursion",
            "evidence": "each instrumented channel's current-window count at every "
            "window of the recording, read up to each regime's horizon",
            "step_seconds": protocol.resolution.step.total_seconds(),
        },
        preprocessing={
            "moments": "one per step from each recording's first observation to "
            "its last",
            "labels": "truth_series of each recording's annotations; unlabelled "
            "windows are not scored",
            "scored": f"labelled windows except the last {protocol.max_lag} of "
            "each recording",
            "step_seconds": protocol.resolution.step.total_seconds(),
        },
        models=[
            ModelRecord(
                key,
                {"regime": regime.to_dict(), "channels": "hurdle, population"},
                "filter_recursion and regime_beliefs",
            )
            for key, regime in regimes.items()
        ],
        household_metrics={
            **{
                key: {h: m.to_dict() for h, m in sorted(per_home_.items())}
                for key, per_home_ in scores.items()
            },
            REPRODUCED: {h: m.to_dict() for h, m in sorted(reproduction.items())},
        },
        intervals=_intervals(estimands),
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return SmoothingResult(record=record, path=path)


def _intervals(estimands: Sequence[Mapping[str, Any]]) -> list[ReportedInterval]:
    reported: list[ReportedInterval] = []
    for entry in estimands:
        for metric, comparison in entry["comparisons"].items():
            if comparison is None:
                continue
            for statistic in ("mean", "median"):
                estimate = comparison[statistic]
                if estimate["interval"] is None:
                    continue
                reported.append(
                    ReportedInterval(
                        label=f"{entry['key']}: {statistic} {metric} smoothing gain",
                        estimate=estimate["estimate"],
                        low=estimate["interval"]["low"],
                        high=estimate["interval"]["high"],
                        confidence=estimate["interval"]["confidence"],
                        method=estimate["interval"]["method"],
                        unit="household",
                        n=comparison["n"],
                    )
                )
    return reported
