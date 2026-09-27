"""Phase 3.3 follow-up: fitted silence and activity rates, pre-specified.

The correlated-silence diagnostic (Phase 3.3) concluded *weakened*: dependence
between channels is real, but the declared rates overstate the evidence of
silence more than it does, and overconfidence builds along quiet runs. By the
routing rule declared before this experiment, that result selects this route:
fit the channels' observation models on training households, keep the channels
conditionally independent, and compare with the declared rates on identical
information.

Every setting, comparison and criterion is declared in
:class:`FittedRatesProtocol` and frozen in a committed file before any
household is scored.

Models
------
Every model has the same prior, transition and channels. Only the channel
observation model differs (:mod:`~sensor_modeling.datasets.channel_models`):

| Model | Channels | Scored in |
| --- | --- | --- |
| ``generative`` | declared Poisson | ``I0``, ``I2`` |
| ``generative_hurdle`` | fitted hurdle | ``I0``, ``I2`` |
| ``generative_fitted_poisson`` | fitted Poisson | ``I0`` |
| ``generative_periodic`` | declared, with the Phase 3.1 time prior | ``I1``, ``I3`` |
| ``generative_periodic_hurdle`` | fitted hurdle, with the time prior | ``I1``, ``I3`` |
| ``filter_declared``, ``filter_hurdle``, ``filter_fitted_poisson`` | each family | ``R`` |

``R`` is the filter's recursion fed every window of the recording, the
setting in which the diagnostic measured quiet-run accumulation. Both sides of
every comparison in ``R`` receive the same windows. It belongs to no Phase 1
information set, so no comparison crosses ``R`` and a set.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.households import household_values, recall_of
from ..evaluation.metrics import PredictionMetrics
from ..evaluation.provenance import (
    ExperimentRecord,
    InputArtifact,
    ModelRecord,
    ReportedInterval,
)
from ..states.ontology import BehaviouralState, StateOntology
from .casas import CasasRecording
from .channel_models import (
    DECLARED,
    HURDLE,
    POISSON,
    FittedChannels,
    HurdleChannel,
    filter_recursion,
    fit_channel_models,
    household_channel_counts,
    total_loglik,
)
from .history_experiment import Estimand
from .information_sets import (
    InformationSet,
    build_feature_table,
    nested_information_sets,
)
from .matched_evaluation import (
    MATCHED_METRIC_DEFINITIONS,
    HouseholdSplit,
    _finite,
    _Household,
    _regular_moments,
)
from .periodic_prior import (
    PeriodicPriorConfig,
    PeriodicStatePrior,
    fit_periodic_prior,
    hour_state_counts,
)
from .recoverable_gap import (
    DEFAULT_METRICS,
    GENERATIVE_CONFIGURATION,
    GENERATIVE_PERIODIC_CONFIGURATION,
    LABELS,
    FrozenSplits,
    GapProtocol,
    _cell,
    _paired,
    _score,
)
from .restricted_filter import ChannelLikelihood, restricted_posteriors
from .silence_dependence import (
    MINIMAL_EFFECTS,
    QUIET_STATES,
    SilenceProtocol,
    against_independence,
    concentration,
    read_verdict,
    representative_runs,
    silence_inflation,
    state_dependence,
)
from .silence_dependence import Estimand as Mechanism
from .time_prior_experiment import MINIMAL_DIFFERENCES, verdict

PROTOCOL_SCHEMA = "fitted-rates-protocol/1"
RESULT_SCHEMA = "fitted-rates-results/1"

#: The filter's recursion over every window, in order.
RECURSION = "R"

DECLARED_MODEL = "generative"
HURDLE_MODEL = "generative_hurdle"
POISSON_MODEL = "generative_fitted_poisson"
DECLARED_PERIODIC = "generative_periodic"
HURDLE_PERIODIC = "generative_periodic_hurdle"
FILTER_DECLARED = "filter_declared"
FILTER_HURDLE = "filter_hurdle"
FILTER_POISSON = "filter_fitted_poisson"

#: Where each model is scored.
CELLS: dict[str, tuple[str, ...]] = {
    DECLARED_MODEL: ("I0", "I2"),
    HURDLE_MODEL: ("I0", "I2"),
    POISSON_MODEL: ("I0",),
    DECLARED_PERIODIC: ("I1", "I3"),
    HURDLE_PERIODIC: ("I1", "I3"),
    FILTER_DECLARED: (RECURSION,),
    FILTER_HURDLE: (RECURSION,),
    FILTER_POISSON: (RECURSION,),
}

#: Each model's channel family.
FAMILY: dict[str, str] = {
    DECLARED_MODEL: DECLARED,
    HURDLE_MODEL: HURDLE,
    POISSON_MODEL: POISSON,
    DECLARED_PERIODIC: DECLARED,
    HURDLE_PERIODIC: HURDLE,
    FILTER_DECLARED: DECLARED,
    FILTER_HURDLE: HURDLE,
    FILTER_POISSON: POISSON,
}

#: Models with the Phase 3.1 time prior.
PERIODIC_MODELS = frozenset({DECLARED_PERIODIC, HURDLE_PERIODIC})

S_SLEEPING = BehaviouralState.SLEEPING.value

#: Filter models, by family.
FILTERS: dict[str, str] = {
    DECLARED: FILTER_DECLARED,
    HURDLE: FILTER_HURDLE,
    POISSON: FILTER_POISSON,
}


def _formulation(
    key: str, role: str, question: str, model: str, reference: str, label: str
) -> Estimand:
    return Estimand(key, role, "formulation", question, model, label, reference, label)


ESTIMANDS: tuple[Estimand, ...] = (
    _formulation(
        "F0",
        "primary",
        "fitted hurdle channels against the declared rates, current windows",
        HURDLE_MODEL,
        DECLARED_MODEL,
        "I0",
    ),
    _formulation(
        "FR",
        "primary",
        "fitted hurdle channels against the declared rates, the filter's "
        "recursion over every window",
        FILTER_HURDLE,
        FILTER_DECLARED,
        RECURSION,
    ),
    _formulation(
        "F1",
        "secondary",
        "the same, with the Phase 3.1 time prior",
        HURDLE_PERIODIC,
        DECLARED_PERIODIC,
        "I1",
    ),
    _formulation(
        "F2",
        "secondary",
        "the same, with three lagged windows",
        HURDLE_MODEL,
        DECLARED_MODEL,
        "I2",
    ),
    _formulation(
        "F3",
        "secondary",
        "the same, with the time prior and three lagged windows",
        HURDLE_PERIODIC,
        DECLARED_PERIODIC,
        "I3",
    ),
    _formulation(
        "P0",
        "secondary",
        "fitted Poisson channels against the declared rates, current windows",
        POISSON_MODEL,
        DECLARED_MODEL,
        "I0",
    ),
    _formulation(
        "PR",
        "secondary",
        "fitted Poisson channels against the declared rates, the recursion",
        FILTER_POISSON,
        FILTER_DECLARED,
        RECURSION,
    ),
)

#: Estimands whose per-state recall is reported.
RECALL_ESTIMANDS = ("F0", "FR")

#: Mechanism checks, one value per household, tested against zero.
MECHANISMS: tuple[Mechanism, ...] = (
    Mechanism(
        "M1",
        "mechanism",
        "inflation_reduction_hurdle",
        "log_ratio",
        "how much the fitted hurdle channels reduce the overstatement of "
        "silence evidence: log inflation, declared minus hurdle",
    ),
    Mechanism(
        "M1P",
        "mechanism",
        "inflation_reduction_poisson",
        "log_ratio",
        "the same for the fitted Poisson channels",
    ),
    Mechanism(
        "M2",
        "mechanism",
        "accumulation_reduction_hurdle",
        "slope",
        "how much the fitted hurdle channels slow the growth of overconfidence "
        "along quiet runs: slope per hour, declared minus hurdle",
    ),
    Mechanism(
        "M2P",
        "mechanism",
        "accumulation_reduction_poisson",
        "slope",
        "the same for the fitted Poisson channels",
    ),
)

CRITERIA: dict[str, str] = {
    "verdicts": "favours model: mean >= delta and lower bound > 0; favours "
    "reference: mean <= -delta and upper bound < 0; negligible: the whole interval "
    "within (-delta, delta); uncertain: anything else",
    "success": "calibration error favours the fitted model, and neither log "
    "loss nor balanced accuracy favours the declared rates",
    "trade-off": "calibration error favours the fitted model and balanced "
    "accuracy favours the declared rates",
    "failure": "calibration error is negligible or favours the declared rates",
    "inconclusive": "anything else",
    "primary": "F0 and FR are each judged by the rule; the other estimands are "
    "read by it",
    "why_calibration": "the diagnostic found the declared rates overstating "
    "evidence, which shows as overconfidence; balanced accuracy is a guard, "
    "not the target",
    "mechanisms": "M1 and M2 get verdicts on the log-ratio and slope scales of "
    "Phase 3.3: positive, negative, negligible or uncertain",
}

ROUTING = (
    "Phase 3.3 concluded 'weakened' under its frozen rule, which selects this "
    "route: fit the channels' marginal observation models on training "
    "households, keeping them conditionally independent. The two-stage "
    "count-and-allocation model is the route for 'supported' and is not built."
)

HURDLE_CONFIGURATION: dict[str, object] = {
    **GENERATIVE_CONFIGURATION,
    "model": "restricted generative filter with fitted hurdle channels",
    "emissions": "per channel and state: P(silent) = pi, and a zero-truncated "
    "Poisson with rate mu for an active window's count",
    "fitted": "pi and mu per fold, per channel and state, on the fold's "
    "training households' labelled windows only, shrunk toward the declared "
    "model by the declared pseudo-windows",
}

POISSON_CONFIGURATION: dict[str, object] = {
    **GENERATIVE_CONFIGURATION,
    "model": "restricted generative filter with fitted Poisson channels",
    "emissions": "per channel and state: a Poisson with the fitted mean count",
    "fitted": "the mean count per fold, per channel and state, on the fold's "
    "training households' labelled windows only, shrunk toward the declared "
    "model by the declared pseudo-windows",
}


def conclusion(verdicts: Mapping[str, str]) -> str:
    """Success, trade-off, failure or inconclusive, from one estimand's verdicts."""
    calibration = verdicts["calibration_error"]
    if calibration in ("negligible", "favours reference"):
        return "failure"
    if calibration == "favours model":
        if verdicts["balanced_accuracy"] == "favours reference":
            return "trade-off"
        if verdicts["log_loss"] != "favours reference":
            return "success"
    return "inconclusive"


@dataclass(frozen=True)
class FittedRatesProtocol:
    """Everything the experiment fixes before any household is scored."""

    folds: tuple[HouseholdSplit, ...]
    splits_sha256: str
    information_sets: tuple[InformationSet, ...] = field(
        default_factory=nested_information_sets
    )
    periodic_prior: PeriodicPriorConfig = field(default_factory=PeriodicPriorConfig)
    pseudo_windows: float = 12.0
    seed: int = 0
    metrics: tuple[str, ...] = DEFAULT_METRICS
    resamples: int = 10_000
    confidence: float = 0.95
    minimal_differences: Mapping[str, float] = field(
        default_factory=lambda: dict(MINIMAL_DIFFERENCES)
    )
    minimal_effects: Mapping[str, float] = field(
        default_factory=lambda: dict(MINIMAL_EFFECTS)
    )
    rare_share: float = 0.05
    name: str = "phase3-fitted-rates"

    def __post_init__(self) -> None:
        """Validate through the recoverable-gap protocol, then the rest."""
        base = self.as_gap_protocol()
        if not isinstance(self.splits_sha256, str) or len(self.splits_sha256) != 64:
            raise ValueError("splits_sha256 must be a SHA-256 hex digest")
        if not math.isfinite(self.pseudo_windows) or self.pseudo_windows <= 0.0:
            raise ValueError("pseudo_windows must be positive")
        missing = sorted({*base.metrics, "recall"} - set(self.minimal_differences))
        if missing:
            raise ValueError(
                f"every metric needs a positive minimal difference; missing {missing}"
            )
        if sorted(self.minimal_effects) != sorted(MINIMAL_EFFECTS):
            raise ValueError("minimal_effects must declare every Phase 3.3 scale")
        values = [*self.minimal_differences.values(), *self.minimal_effects.values()]
        if any(not math.isfinite(v) or v <= 0.0 for v in values):
            raise ValueError("every minimal difference and effect must be positive")
        if not 0.0 < self.rare_share < 1.0:
            raise ValueError("rare_share must lie in (0, 1)")
        object.__setattr__(self, "folds", base.folds)
        object.__setattr__(self, "information_sets", base.information_sets)
        object.__setattr__(self, "metrics", base.metrics)
        object.__setattr__(self, "minimal_differences", dict(self.minimal_differences))
        object.__setattr__(self, "minimal_effects", dict(self.minimal_effects))

    def as_gap_protocol(self) -> GapProtocol:
        """The equivalent recoverable-gap protocol, for validation and helpers."""
        return GapProtocol(
            tuple(self.folds),
            information_sets=tuple(self.information_sets),
            seed=self.seed,
            metrics=tuple(self.metrics),
            resamples=self.resamples,
            confidence=self.confidence,
            periodic_prior=self.periodic_prior,
        )

    @property
    def homes(self) -> tuple[str, ...]:
        """Every household, sorted."""
        return tuple(sorted(home for fold in self.folds for home in fold.test))

    @property
    def silence(self) -> SilenceProtocol:
        """The Phase 3.3 settings, with this protocol's bootstrap."""
        return SilenceProtocol(
            self.homes,
            self.splits_sha256,
            resamples=self.resamples,
            confidence=self.confidence,
            seed=self.seed,
            minimal_effects=self.minimal_effects,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        silence = self.silence
        return {
            "schema": PROTOCOL_SCHEMA,
            "name": self.name,
            "routing": ROUTING,
            "households": {
                "splits_file": "artifacts/phase1/household_splits.json",
                "splits_sha256": self.splits_sha256,
                "folds": [
                    {**fold.to_dict(), "sha256": fold.sha256()} for fold in self.folds
                ],
                "development_households": "none: no setting is selected on data",
            },
            "information_sets": {
                **{
                    label: {**info.to_dict(), "sha256": info.sha256()}
                    for label, info in zip(LABELS, self.information_sets)
                },
                RECURSION: "the filter's recursion fed every window of the "
                "recording from the stationary distribution; both sides of a "
                "comparison receive the same windows",
            },
            "models": {
                DECLARED_MODEL: GENERATIVE_CONFIGURATION,
                HURDLE_MODEL: HURDLE_CONFIGURATION,
                POISSON_MODEL: POISSON_CONFIGURATION,
                DECLARED_PERIODIC: {
                    **GENERATIVE_PERIODIC_CONFIGURATION,
                    "periodic_prior": self.periodic_prior.to_dict(),
                },
                HURDLE_PERIODIC: {
                    **HURDLE_CONFIGURATION,
                    "prior": GENERATIVE_PERIODIC_CONFIGURATION["prior"],
                    "transition": GENERATIVE_PERIODIC_CONFIGURATION["transition"],
                    "periodic_prior": self.periodic_prior.to_dict(),
                },
                FILTER_DECLARED: "declared Poisson channels, the recursion",
                FILTER_HURDLE: "fitted hurdle channels, the recursion",
                FILTER_POISSON: "fitted Poisson channels, the recursion",
            },
            "cells": {model: list(labels) for model, labels in CELLS.items()},
            "fitting": {
                "unit": "one parameter set per fold, per channel and state, pooled "
                "over the fold's training households' labelled windows",
                "hurdle": "pi = (Z + k pi0) / (W + k); m = (S + k m0) / (P + k); "
                "mu solves mu / (1 - exp(-mu)) = m",
                "poisson": "lambda = (S + k lambda0) / (W + k)",
                "prior": "lambda0 the declared expected count, pi0 = exp(-lambda0), "
                "m0 = lambda0 / (1 - exp(-lambda0)), each averaged over the training "
                "windows; a channel with no training window in a state takes the "
                "held-out household's declared value",
                "pseudo_windows": self.pseudo_windows,
                "periodic_prior": "fitted per fold on the training households, "
                "the population prior for every held-out household",
            },
            "inclusion": {
                HURDLE_MODEL: "primary: the diagnostic found silence far more "
                "frequent than a Poisson with the observed mean allows, so "
                "silence needs its own parameter",
                POISSON_MODEL: "secondary: the declared family with fitted values, "
                "to test whether silence needs its own parameter",
                HURDLE_PERIODIC: "secondary: Phase 3.1 met its success rule, so the "
                "time prior is the current best generative model",
            },
            "metrics": list(self.metrics),
            "per_state_recall": {
                "estimands": list(RECALL_ESTIMANDS),
                "rare_rule": f"share of labelled time below {self.rare_share} in "
                "either fold's training households",
            },
            "calibration": "expected calibration error over 10 confidence bins",
            "estimands": [estimand.to_dict() for estimand in ESTIMANDS],
            "mechanisms": [mechanism.to_dict() for mechanism in MECHANISMS],
            "silence_settings": {
                "min_windows": silence.min_windows,
                "smoothing": silence.smoothing,
                "min_expected": silence.min_expected,
                "quiet_states": [s.value for s in QUIET_STATES],
                "case_min_steps": silence.case_min_steps,
                "case_steps": silence.case_steps,
                "accumulation_steps": silence.accumulation_steps,
                "inflation": "Phase 3.3's log inflation with each family's joint "
                "silence log-likelihood: declared sum of per-window terms, hurdle "
                "sum of log pi, Poisson sum of -lambda",
            },
            "descriptive": {
                "drift": "in each held-out household's Phase 3.3 representative "
                "quiet runs, the state each recursion reports after the recorded "
                "windows, and its confidence",
                "dispersion": "the variance of an active window's count against "
                "its mean, per fold, channel and state, from the training "
                "households",
            },
            "minimal_differences": dict(self.minimal_differences),
            "minimal_effects": dict(self.minimal_effects),
            "criteria": dict(CRITERIA),
            "bootstrap": {
                "unit": "household",
                "statistics": "mean and median paired differences",
                "resamples": self.resamples,
                "confidence": self.confidence,
                "interval": "percentile",
            },
            "seeds": {"bootstrap": self.seed},
            "tuning": "none: every setting is declared here, before scoring",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_protocol(splits: FrozenSplits) -> FittedRatesProtocol:
    """The protocol as declared for the development panel's frozen folds."""
    return FittedRatesProtocol(splits.folds, splits.sha256)


def check_frozen_protocol(protocol: FittedRatesProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after scoring begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


@dataclass(frozen=True)
class FoldFit:
    """What one fold fits on its training households."""

    channels: FittedChannels
    periodic: PeriodicStatePrior
    shares: Mapping[str, float]


def fold_fits(
    recordings: Mapping[str, CasasRecording],
    protocol: FittedRatesProtocol,
    ontology: StateOntology,
) -> dict[str, FoldFit]:
    """Each fold's channel models, time prior and state shares, from training homes."""
    resolution = protocol.information_sets[0].resolution
    states = tuple(ontology.states)
    fits: dict[str, FoldFit] = {}
    for fold in protocol.folds:
        training = {home: recordings[home] for home in fold.train}
        counts = {
            home: hour_state_counts(
                recording,
                _regular_moments(recording, resolution.step),
                states=states,
                step=resolution.step,
            )
            for home, recording in training.items()
        }
        pooled = sum(counts.values(), np.zeros((24, len(states))))
        fits[fold.name] = FoldFit(
            fit_channel_models(
                training,
                resolution=resolution,
                pseudo_windows=protocol.pseudo_windows,
                ontology=ontology,
            ),
            fit_periodic_prior(counts, states=states, config=protocol.periodic_prior),
            {
                state.value: float(share)
                for state, share in zip(states, pooled.sum(axis=0) / pooled.sum())
            },
        )
    return fits


@dataclass(frozen=True)
class FittedRatesResult:
    """A completed experiment: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def _silence_sums(models: Mapping[Any, Any], index: int) -> float:
    """A family's joint-silence log-likelihood in one state, up to a constant."""
    total = 0.0
    for model in models.values():
        if isinstance(model, HurdleChannel):
            total += float(model.zero_term[index])
        elif isinstance(model, ChannelLikelihood):
            total += float(model.per_window[index])
        else:  # pragma: no cover - every family is one of the two
            raise TypeError(f"unknown channel model {type(model).__name__}")
    return total


def household_scores(
    recording: CasasRecording,
    protocol: FittedRatesProtocol,
    fit: FoldFit,
    *,
    household: str,
    ontology: StateOntology,
) -> dict[str, Any]:
    """Every model's scores and every mechanism value for one held-out household."""
    states = tuple(ontology.states)
    sets = dict(zip(LABELS, protocol.information_sets))
    resolution = protocol.information_sets[0].resolution
    silence = protocol.silence
    moments = _regular_moments(recording, resolution.step)
    families = {
        family: fit.channels.models(recording.registry, resolution, family, ontology)
        for family in (DECLARED, HURDLE, POISSON)
    }
    scores: dict[str, dict[str, PredictionMetrics]] = {}
    labels_of: dict[str, Any] = {}
    for label, info in sets.items():
        table = build_feature_table(recording, info, moments, household=household)
        built = _Household.build(household, "test", table, recording)
        labels_of[label] = built.moments_sha256
        rows = built.labelled
        for model, cells in CELLS.items():
            if label not in cells:
                continue
            options: dict[str, Any] = (
                {"periodic_prior": fit.periodic} if model in PERIODIC_MODELS else {}
            )
            beliefs = restricted_posteriors(
                table, families[FAMILY[model]], ontology, **options
            )[rows]
            scores.setdefault(model, {})[label] = _score(built.labels, beliefs, states)

    counts, rows, labels, moments = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    transition = ontology.transition(resolution.step)
    silent_count = np.sum([counts[c] == 0 for c in counts], axis=0)
    truth: list[int | None] = [None] * len(moments)
    for row, label in zip(rows, labels, strict=True):
        truth[int(row)] = int(label)
    fully_silent = silent_count == len(counts)
    runs = {
        state.value: representative_runs(
            fully_silent, truth, states.index(state), min_steps=silence.case_min_steps
        )
        for state in QUIET_STATES
    }
    accumulation: dict[str, float | None] = {}
    drift: dict[str, dict[str, Any]] = {}
    for family, models in families.items():
        _, posterior = filter_recursion(
            total_loglik(models, counts), transition, ontology.stationary()
        )
        model = FILTERS[family]
        scores.setdefault(model, {})[RECURSION] = _score(
            [states[i] for i in labels], posterior[rows], states
        )
        accumulation[family] = concentration(
            posterior,
            rows,
            labels,
            silent_count,
            len(counts),
            accumulation_steps=silence.accumulation_steps,
            steps_per_hour=3600.0 / resolution.step.total_seconds(),
        )["accumulation_slope"]
        drift[family] = {}
        for state, run in runs.items():
            if run is None:
                continue
            last = run[0] + silence.case_steps - 1
            drift[family][state] = {
                "state": states[int(posterior[last].argmax())].value,
                "confidence": float(posterior[last].max()),
            }

    windows = {s.value: int(np.sum(labels == i)) for i, s in enumerate(states)}
    eligible = [
        i for i, s in enumerate(states) if windows[s.value] >= silence.min_windows
    ]
    channels = sorted(counts)
    names = [c.name for c in channels]
    matrix = np.column_stack([counts[c] for c in channels])
    declared = families[DECLARED]
    expected = np.stack(
        [
            terms.expected
            for c in channels
            if isinstance(terms := declared[c], ChannelLikelihood)
        ]
    )
    by_state = {
        states[i].value: state_dependence(
            matrix[rows[labels == i]],
            names,
            expected[:, i],
            smoothing=silence.smoothing,
            min_expected=silence.min_expected,
        )
        for i in eligible
    }
    inflation = {
        family: silence_inflation(
            by_state,
            {states[i].value: _silence_sums(models, i) for i in eligible},
        )["log_inflation_declared"]
        for family, models in families.items()
    }

    def reduction(values: Mapping[str, float | None], family: str) -> float | None:
        declared, fitted = values[DECLARED], values[family]
        if declared is None or fitted is None:
            return None
        return declared - fitted

    return {
        "scores": scores,
        "moments_sha256": labels_of,
        "labelled": int(rows.size),
        "channels": names,
        "untrained_channels": sorted(
            c.name for c in counts if c not in fit.channels.statistics
        ),
        "log_inflation": inflation,
        "accumulation_slope": accumulation,
        "mechanisms": {
            "inflation_reduction_hurdle": reduction(inflation, HURDLE),
            "inflation_reduction_poisson": reduction(inflation, POISSON),
            "accumulation_reduction_hurdle": reduction(accumulation, HURDLE),
            "accumulation_reduction_poisson": reduction(accumulation, POISSON),
        },
        "drift": drift,
    }


def run_fitted_rates(
    recordings: Mapping[str, CasasRecording],
    protocol: FittedRatesProtocol,
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> FittedRatesResult:
    """Run the pre-specified experiment and build its record."""
    missing = sorted(set(protocol.homes) - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    ontology = StateOntology()
    space = tuple(ontology.states)
    gap = protocol.as_gap_protocol()
    fits = fold_fits(recordings, protocol, ontology)
    rare = sorted(
        s.value
        for s in space
        if any(f.shares[s.value] < protocol.rare_share for f in fits.values())
    )

    scores: dict[str, dict[str, dict[str, PredictionMetrics]]] = {
        model: {label: {} for label in labels} for model, labels in CELLS.items()
    }
    households: dict[str, dict[str, Any]] = {}
    for fold in protocol.folds:
        for home in sorted(fold.test):
            scored = household_scores(
                recordings[home],
                protocol,
                fits[fold.name],
                household=home,
                ontology=ontology,
            )
            for model, by_label in scored.pop("scores").items():
                for label, metrics in by_label.items():
                    scores[model][label][home] = metrics
            households[home] = {"fold": fold.name, **scored}
    households = dict(sorted(households.items()))

    def values(model: str, label: str, metric: Any) -> dict[str, float | None]:
        return household_values(scores[model][label], metric)

    cells = [
        _cell(model, label, scores[model][label], space, gap)
        for model, labels in CELLS.items()
        for label in labels
    ]
    estimands: list[dict[str, Any]] = []
    for estimand in ESTIMANDS:
        comparisons = {
            metric: _paired(
                values(estimand.model, estimand.model_set, metric),
                values(estimand.reference, estimand.reference_set, metric),
                metric,
                gap,
            )
            for metric in protocol.metrics
        }
        verdicts = {
            metric: verdict(comparison, protocol.minimal_differences[metric])
            for metric, comparison in comparisons.items()
        }
        entry: dict[str, Any] = {
            **estimand.to_dict(),
            "comparisons": comparisons,
            "verdicts": verdicts,
            "conclusion": conclusion(verdicts),
        }
        if estimand.key in RECALL_ESTIMANDS:
            entry["per_state_recall"] = {}
            for state in space:
                comparison = _paired(
                    values(estimand.model, estimand.model_set, recall_of(state)),
                    values(
                        estimand.reference, estimand.reference_set, recall_of(state)
                    ),
                    "balanced_accuracy",
                    gap,
                )
                entry["per_state_recall"][state.value] = {
                    "rare": state.value in rare,
                    "comparison": comparison,
                    "verdict": verdict(
                        comparison, protocol.minimal_differences["recall"]
                    ),
                }
        estimands.append(entry)

    silence = protocol.silence
    mechanisms = []
    for mechanism in MECHANISMS:
        summary = against_independence(
            {h: d["mechanisms"][mechanism.quantity] for h, d in households.items()},
            silence,
        )
        mechanisms.append(
            {
                **mechanism.to_dict(),
                "summary": summary,
                "verdict": read_verdict(
                    summary, protocol.minimal_effects[mechanism.scale]
                ),
            }
        )

    drift = {
        family: {
            state.value: {
                "runs": sum(
                    1 for d in households.values() if state.value in d["drift"][family]
                ),
                "ending_sleeping": sum(
                    1
                    for d in households.values()
                    if d["drift"][family].get(state.value, {}).get("state")
                    == S_SLEEPING
                ),
                "median_confidence": _median(
                    d["drift"][family][state.value]["confidence"]
                    for d in households.values()
                    if state.value in d["drift"][family]
                ),
            }
            for state in QUIET_STATES
        }
        for family in (DECLARED, HURDLE, POISSON)
    }

    by_key = {e["key"]: e for e in estimands}
    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified; development panel, which earlier work has inspected",
        "routing": ROUTING,
        "states": [state.value for state in space],
        "rare_states": rare,
        "households": households,
        "cells": cells,
        "estimands": estimands,
        "mechanisms": mechanisms,
        "drift": drift,
        "conclusions": {
            "current_windows": by_key["F0"]["conclusion"],
            "recursion": by_key["FR"]["conclusion"],
        },
        "fitted": {
            fold: {
                "channels": {**f.channels.to_dict(), "sha256": f.channels.sha256()},
                "periodic_prior": {
                    **f.periodic.to_dict(),
                    "sha256": f.periodic.sha256(),
                },
                "training_shares": dict(f.shares),
            }
            for fold, f in fits.items()
        },
    }
    record = ExperimentRecord(
        experiment=protocol.name,
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        seeds=[protocol.seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions=MATCHED_METRIC_DEFINITIONS,
        notes=[
            "Every setting, estimand and criterion was declared in the protocol "
            "before any household was scored.",
            "Every comparison is between two channel observation models on "
            "identical information: the same set, or the same recursion.",
            "Channel parameters and the time prior are fitted per fold on "
            "training households only.",
            "Verdicts compare effect sizes and household bootstrap intervals with "
            "declared minimal differences; they are not significance tests.",
        ],
        inputs=list(inputs),
        split={
            "folds": [{**f.to_dict(), "sha256": f.sha256()} for f in protocol.folds]
        },
        information_set={
            "sets": [
                {"label": label, **info.to_dict(), "sha256": info.sha256()}
                for label, info in zip(LABELS, protocol.information_sets)
            ]
        },
        preprocessing={
            "moments": "one per step from each recording's first observation to "
            "its last, identical for every model",
            "labels": "truth_series of each recording's annotations; unlabelled "
            "moments are not scored",
            "step_seconds": protocol.information_sets[
                0
            ].resolution.step.total_seconds(),
        },
        models=[
            ModelRecord(
                DECLARED_MODEL, GENERATIVE_CONFIGURATION, "restricted_posteriors"
            ),
            ModelRecord(HURDLE_MODEL, HURDLE_CONFIGURATION, "restricted_posteriors"),
            ModelRecord(POISSON_MODEL, POISSON_CONFIGURATION, "restricted_posteriors"),
            ModelRecord(
                DECLARED_PERIODIC,
                {
                    **GENERATIVE_PERIODIC_CONFIGURATION,
                    "periodic_prior": protocol.periodic_prior.to_dict(),
                },
                "restricted_posteriors(periodic_prior=...)",
            ),
            ModelRecord(
                HURDLE_PERIODIC,
                {
                    **HURDLE_CONFIGURATION,
                    "periodic_prior": protocol.periodic_prior.to_dict(),
                },
                "restricted_posteriors(periodic_prior=...)",
            ),
            *(
                ModelRecord(FILTERS[family], {"channels": family}, "filter_recursion")
                for family in (DECLARED, HURDLE, POISSON)
            ),
        ],
        household_metrics={
            f"{model}@{label}": {h: m.to_dict() for h, m in per_home.items()}
            for model, by_label in scores.items()
            for label, per_home in by_label.items()
        },
        intervals=_intervals(estimands, mechanisms),
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return FittedRatesResult(record=record, path=path)


def _median(values: Any) -> float | None:
    collected = sorted(values)
    return float(np.median(collected)) if collected else None


def _intervals(
    estimands: Sequence[Mapping[str, Any]], mechanisms: Sequence[Mapping[str, Any]]
) -> list[ReportedInterval]:
    reported: list[ReportedInterval] = []
    for entry in estimands:
        for metric, comparison in entry["comparisons"].items():
            for statistic in ("mean", "median"):
                estimate = comparison[statistic]
                if estimate["interval"] is None:
                    continue
                reported.append(
                    ReportedInterval(
                        label=f"{entry['key']}: {statistic} {metric} difference",
                        estimate=estimate["estimate"],
                        low=estimate["interval"]["low"],
                        high=estimate["interval"]["high"],
                        confidence=estimate["interval"]["confidence"],
                        method=estimate["interval"]["method"],
                        unit="household",
                        n=comparison["n"],
                    )
                )
    for entry in mechanisms:
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
