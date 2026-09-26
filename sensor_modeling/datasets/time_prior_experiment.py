"""Phase 3.1: the pre-specified experiment for the periodic state prior.

The question is whether the generative model can use time of day once it has
an explicit, hierarchical time-of-day prior, and at what cost in probability
quality. Every setting, comparison and criterion below is declared in
:class:`TimePriorProtocol` and frozen in a committed protocol file before any
household is scored.

Models
------
- ``generative``: the original generative model, restricted to the set, as in
  the recoverable-information-gap experiment. It is scored in ``I0`` and ``I2``.
- ``generative_untimed``: the hierarchical model with time disabled. Its prior
  is fitted exactly as the periodic prior is, but on counts pooled over hours,
  so every periodic coefficient is zero and ``π_h`` is the same at every hour.
  The same prior with no hour dependence is a time-homogeneous ontology whose
  dwell times are rescaled by ``c_s = π̄(s) / π(s)``. That consumes no hour, so
  it is scored in ``I0`` and ``I2``.
- ``generative_periodic``: the hierarchical model with the hour, scored in
  ``I1`` and ``I3``.
- ``diagnostic``, ``logistic`` and ``state_frequency``: the matched runner's
  models, scored in ``I1`` and ``I3``.

Each fold's priors are fitted on that fold's training households only. Held-out
households receive the population prior. The adaptation arm is the one
exception, and a separate analysis: there, a held-out household's deviation is
fitted from its own labels in an initial adaptation window, and scored only
after it.

Estimands and criteria
----------------------
Every comparison is a mean paired household difference with a household
bootstrap interval. It is oriented so that a positive value favours the first
model. Against a declared minimal important difference ``δ``, its verdict is:

- ``favours model``: mean ≥ ``δ`` and the interval's lower bound > 0;
- ``favours reference``: mean ≤ ``-δ`` and the interval's upper bound < 0;
- ``negligible``: the whole interval lies within ``(-δ, δ)``;
- ``uncertain``: anything else.

These are reading aids for effect sizes and their uncertainty, not significance
tests. The protocol states the overall success and failure rules.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
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
from .information_sets import (
    InformationSet,
    build_feature_table,
    nested_information_sets,
)
from .matched_evaluation import (
    MATCHED_METRIC_DEFINITIONS,
    HouseholdSplit,
    MatchedEvaluation,
    ModelSpec,
    _finite,
    _Household,
    _regular_moments,
    held_out_metrics,
    run_matched_evaluation,
)
from .periodic_prior import (
    PeriodicPriorConfig,
    PeriodicStatePrior,
    fit_periodic_prior,
    hour_state_counts,
)
from .recoverable_gap import (
    DEFAULT_METRICS,
    DIAGNOSTIC,
    GENERATIVE_CONFIGURATION,
    GENERATIVE_PERIODIC_CONFIGURATION,
    LABELS,
    LOGISTIC,
    REFERENCE,
    FrozenSplits,
    GapProtocol,
    _cell,
    _paired,
    _score,
    gap_models,
)
from .restricted_filter import channel_likelihoods, restricted_posteriors

#: Schema of a time-prior protocol and of its results.
PROTOCOL_SCHEMA = "time-prior-protocol/1"
RESULT_SCHEMA = "time-prior-results/1"

ORIGINAL = "generative"
UNTIMED = "generative_untimed"
PERIODIC = "generative_periodic"

#: Where each model is scored.
CELLS: dict[str, tuple[str, ...]] = {
    ORIGINAL: ("I0", "I2"),
    UNTIMED: ("I0", "I2"),
    PERIODIC: ("I1", "I3"),
    DIAGNOSTIC: ("I1", "I3"),
    LOGISTIC: ("I1", "I3"),
    REFERENCE: ("I1", "I3"),
}

#: Minimal important differences, declared before scoring. Balanced accuracy's
#: is the smallest information gain Phase 1 reported as real (history to
#: logistic regression, +0.021).
MINIMAL_DIFFERENCES: dict[str, float] = {
    "balanced_accuracy": 0.02,
    "log_loss": 0.05,
    "brier": 0.01,
    "calibration_error": 0.02,
    "recall": 0.05,
}

UNTIMED_CONFIGURATION: dict[str, object] = {
    **GENERATIVE_CONFIGURATION,
    "model": "restricted generative filter with a time-disabled periodic prior",
    "prior": "the periodic prior fitted on counts pooled over hours, so every "
    "periodic coefficient is zero; its constant pi_bar is the starting belief",
    "transition": "default StateOntology with dwell times scaled by "
    "pi_bar(s) / pi(s), the time-homogeneous form of the same prior",
    "fitted": "the population prior, per fold, on the fold's training "
    "households' labels only",
}


@dataclass(frozen=True)
class Estimand:
    """One pre-specified paired comparison: *model* in one set against *reference*."""

    key: str
    role: str
    question: str
    model: str
    model_set: str
    reference: str
    reference_set: str

    def to_dict(self) -> dict[str, str]:
        """Return a serialisable form."""
        return {
            "key": self.key,
            "role": self.role,
            "question": self.question,
            "model": f"{self.model}@{self.model_set}",
            "reference": f"{self.reference}@{self.reference_set}",
        }


ESTIMANDS: tuple[Estimand, ...] = (
    Estimand(
        "P1",
        "primary",
        "what the hour is worth to the hierarchical model",
        PERIODIC,
        "I1",
        UNTIMED,
        "I0",
    ),
    Estimand(
        "P2",
        "primary",
        "the hierarchical model with the hour against the original model without it",
        PERIODIC,
        "I1",
        ORIGINAL,
        "I0",
    ),
    Estimand(
        "P3",
        "primary",
        "formulation gap at I1: the diagnostic against the hierarchical model",
        DIAGNOSTIC,
        "I1",
        PERIODIC,
        "I1",
    ),
    Estimand(
        "P4",
        "primary",
        "formulation gap at I1: logistic regression against the hierarchical model",
        LOGISTIC,
        "I1",
        PERIODIC,
        "I1",
    ),
    Estimand(
        "S1",
        "secondary",
        "the fitted prior against the declared one, both without the hour",
        UNTIMED,
        "I0",
        ORIGINAL,
        "I0",
    ),
    Estimand(
        "S2",
        "secondary",
        "what the hour is worth to the hierarchical model with recent history",
        PERIODIC,
        "I3",
        UNTIMED,
        "I2",
    ),
    Estimand(
        "S3",
        "secondary",
        "the hierarchical model with the hour and history "
        "against the original model with history",
        PERIODIC,
        "I3",
        ORIGINAL,
        "I2",
    ),
    Estimand(
        "S4",
        "secondary",
        "formulation gap at I3: the diagnostic against the hierarchical model",
        DIAGNOSTIC,
        "I3",
        PERIODIC,
        "I3",
    ),
    Estimand(
        "S5",
        "secondary",
        "formulation gap at I3: logistic regression against the hierarchical model",
        LOGISTIC,
        "I3",
        PERIODIC,
        "I3",
    ),
    Estimand(
        "S6",
        "secondary",
        "what recent history is worth to the hierarchical model",
        PERIODIC,
        "I3",
        PERIODIC,
        "I1",
    ),
)

#: Estimands whose per-state recall differences are reported.
RECALL_ESTIMANDS = ("P1", "P2")

#: The pre-specified success and failure rules.
CRITERIA: dict[str, str] = {
    "verdicts": "favours model: mean >= delta and lower bound > 0; favours "
    "reference: mean <= -delta and upper bound < 0; negligible: the whole "
    "interval within (-delta, delta); uncertain: anything else",
    "success": "P1 balanced accuracy favours the model, and P1 log loss and P1 "
    "calibration error do not favour the reference",
    "failure": "P1 balanced accuracy is negligible or favours the reference",
    "inconclusive": "neither success nor failure",
    "practical": "P2 is judged by the same rule, as the improvement over the "
    "original model",
    "hierarchy": "household adaptation helps if A-I1 balanced accuracy favours "
    "the adapted prior and A-I1 log loss does not favour the population prior",
}


def verdict(comparison: Mapping[str, Any], minimal: float) -> str:
    """The pre-specified reading of one oriented paired comparison."""
    mean = comparison["mean"]["estimate"]
    interval = comparison["mean"]["interval"]
    if interval is None or mean is None:
        return "uncertain"
    low, high = interval["low"], interval["high"]
    if mean >= minimal and low > 0.0:
        return "favours model"
    if mean <= -minimal and high < 0.0:
        return "favours reference"
    if -minimal < low and high < minimal:
        return "negligible"
    return "uncertain"


def conclusion(verdicts: Mapping[str, str]) -> str:
    """Success, failure or inconclusive, from one estimand's metric verdicts."""
    accuracy = verdicts["balanced_accuracy"]
    if accuracy in ("negligible", "favours reference"):
        return "failure"
    if accuracy == "favours model" and all(
        verdicts.get(metric) != "favours reference"
        for metric in ("log_loss", "calibration_error")
    ):
        return "success"
    return "inconclusive"


def hierarchy_conclusion(verdicts: Mapping[str, str]) -> str:
    """Whether household adaptation helps, from the declared arm's verdicts in I1."""
    accuracy = verdicts["balanced_accuracy"]
    if accuracy == "favours model" and verdicts["log_loss"] != "favours reference":
        return "helps"
    if accuracy in ("negligible", "favours reference"):
        return "does not help"
    return "inconclusive"


def hour_collapsed(counts: np.ndarray) -> np.ndarray:
    """Counts spread evenly over the hours, removing every trace of the hour."""
    values = np.asarray(counts, dtype=float)
    collapsed: np.ndarray = np.tile(
        values.sum(axis=0) / values.shape[0], (values.shape[0], 1)
    )
    return collapsed


def untimed_ontology(prior: PeriodicStatePrior, base: StateOntology) -> StateOntology:
    """The time-homogeneous ontology equal to *prior* when it ignores the hour.

    Its dwell times are the base's scaled by ``π̄(s) / π(s)``, which is the
    prior's circadian generator with the same stickiness at every hour. Its
    stationary distribution is ``π̄``.
    """
    hourly = prior.hourly()
    if np.abs(hourly - hourly[0]).max() > 1e-9:
        raise ValueError("the prior depends on the hour, so it is not time-disabled")
    ratio = hourly[0] / base.stationary()
    return replace(
        base,
        dwell={
            state: base.dwell[state] * float(ratio[index])
            for index, state in enumerate(base.states)
        },
    )


@dataclass(frozen=True)
class TimePriorProtocol:
    """Everything the experiment fixes before any household is scored.

    Attributes
    ----------
    folds
        The frozen cross-fitted folds.
    splits_sha256
        Digest of the frozen split file the folds came from.
    information_sets
        The four nested Phase 1 sets.
    periodic_prior
        Settings of every fitted prior, including the pooling strength.
    models
        The matched runner's models: reference, diagnostic, logistic.
    seed, metrics, resamples, confidence
        As in the recoverable-information-gap protocol.
    minimal_differences
        Declared minimal important differences, by metric, and ``recall``.
    rare_share
        A state is rare if its share of labelled time in either fold's
        training households is below this.
    adaptation_days
        Length of each held-out household's adaptation window, from its first
        observation. Only moments after it are scored in the adaptation arm.
    sensitivity_precisions
        Further pooling strengths reported in the adaptation arm, beside the
        declared one. None is selected.
    name
        Name of the record.
    """

    folds: tuple[HouseholdSplit, ...]
    splits_sha256: str
    information_sets: tuple[InformationSet, ...] = field(
        default_factory=nested_information_sets
    )
    periodic_prior: PeriodicPriorConfig = field(default_factory=PeriodicPriorConfig)
    models: tuple[ModelSpec, ...] = field(default_factory=gap_models)
    seed: int = 0
    metrics: tuple[str, ...] = DEFAULT_METRICS
    resamples: int = 10_000
    confidence: float = 0.95
    minimal_differences: Mapping[str, float] = field(
        default_factory=lambda: dict(MINIMAL_DIFFERENCES)
    )
    rare_share: float = 0.05
    adaptation_days: float = 7.0
    sensitivity_precisions: tuple[float, ...] = (2.4, 240.0)
    name: str = "phase3-hierarchical-time-prior"

    def __post_init__(self) -> None:
        """Validate through the recoverable-gap protocol, then the rest."""
        base = GapProtocol(
            tuple(self.folds),
            information_sets=tuple(self.information_sets),
            seed=self.seed,
            metrics=tuple(self.metrics),
            resamples=self.resamples,
            confidence=self.confidence,
            models=tuple(self.models),
            periodic_prior=self.periodic_prior,
        )
        if not isinstance(self.splits_sha256, str) or len(self.splits_sha256) != 64:
            raise ValueError("splits_sha256 must be a SHA-256 hex digest")
        missing = sorted({*base.metrics, "recall"} - set(self.minimal_differences))
        if missing or any(
            not math.isfinite(v) or v <= 0.0 for v in self.minimal_differences.values()
        ):
            raise ValueError(
                f"every metric needs a positive minimal difference; missing {missing}"
            )
        if not 0.0 < self.rare_share < 1.0:
            raise ValueError("rare_share must lie in (0, 1)")
        if not math.isfinite(self.adaptation_days) or self.adaptation_days <= 0.0:
            raise ValueError("adaptation_days must be positive")
        if any(not math.isfinite(p) or p <= 0.0 for p in self.sensitivity_precisions):
            raise ValueError("sensitivity precisions must be positive and finite")
        object.__setattr__(self, "folds", base.folds)
        object.__setattr__(self, "information_sets", base.information_sets)
        object.__setattr__(self, "metrics", base.metrics)
        object.__setattr__(self, "models", base.models)
        object.__setattr__(self, "minimal_differences", dict(self.minimal_differences))
        object.__setattr__(
            self,
            "sensitivity_precisions",
            tuple(float(p) for p in self.sensitivity_precisions),
        )

    @property
    def homes(self) -> tuple[str, ...]:
        """Every household, sorted."""
        return tuple(sorted(home for fold in self.folds for home in fold.test))

    @property
    def precisions(self) -> tuple[float, ...]:
        """The declared pooling strength, then the sensitivity ones."""
        return (self.periodic_prior.household_precision, *self.sensitivity_precisions)

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
                "development_households": "none: no setting is selected on data",
            },
            "information_sets": {
                label: {**info.to_dict(), "sha256": info.sha256()}
                for label, info in zip(LABELS, self.information_sets)
            },
            "models": {
                **{spec.name: dict(spec.configuration) for spec in self.models},
                ORIGINAL: GENERATIVE_CONFIGURATION,
                UNTIMED: {
                    **UNTIMED_CONFIGURATION,
                    "periodic_prior": self.periodic_prior.to_dict(),
                },
                PERIODIC: {
                    **GENERATIVE_PERIODIC_CONFIGURATION,
                    "periodic_prior": self.periodic_prior.to_dict(),
                },
            },
            "cells": {model: list(labels) for model, labels in CELLS.items()},
            "pooling": {
                "household_precision": self.periodic_prior.household_precision,
                "selection": "declared before scoring, not selected on any data",
                "population_arm": "held-out households get the population prior",
                "adaptation_arm": {
                    "window_days": self.adaptation_days,
                    "window": "from each held-out household's first observation; "
                    "its labels fit the household deviation",
                    "scored": "only that household's labelled moments after the window",
                    "precisions": list(self.precisions),
                    "declared_precision": self.periodic_prior.household_precision,
                    "comparison": "adapted prior against the population prior, same "
                    "model and moments, in I1 and I3",
                },
            },
            "metrics": list(self.metrics),
            "per_state_recall": {
                "estimands": list(RECALL_ESTIMANDS),
                "rare_rule": f"share of labelled time below {self.rare_share} in "
                "either fold's training households",
            },
            "calibration": "expected calibration error over 10 confidence bins",
            "estimands": [estimand.to_dict() for estimand in ESTIMANDS],
            "minimal_differences": dict(self.minimal_differences),
            "criteria": CRITERIA,
            "bootstrap": {
                "unit": "household",
                "statistic": "mean paired difference, with the median also recorded",
                "resamples": self.resamples,
                "confidence": self.confidence,
                "interval": "percentile",
            },
            "seeds": {
                "models": self.seed,
                "bootstrap": self.seed,
                "row_order": self.seed,
            },
            "tuning": "none: every setting is declared here, before scoring",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()

    def as_gap_protocol(self) -> GapProtocol:
        """The equivalent recoverable-gap protocol, for its shared helpers."""
        return GapProtocol(
            self.folds,
            information_sets=self.information_sets,
            seed=self.seed,
            metrics=self.metrics,
            resamples=self.resamples,
            confidence=self.confidence,
            models=self.models,
            periodic_prior=self.periodic_prior,
        )


def declared_protocol(splits: FrozenSplits) -> TimePriorProtocol:
    """The protocol as declared for the development panel's frozen folds."""
    return TimePriorProtocol(splits.folds, splits.sha256)


def check_frozen_protocol(protocol: TimePriorProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*.

    Returns the frozen file's digest, with line endings normalised, for the
    record's inputs.
    """
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    declared = {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}
    if frozen != declared:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after scoring begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


@dataclass(frozen=True)
class TimePriorResult:
    """A completed experiment: its record, the runner's runs, and where it was written."""

    record: ExperimentRecord
    runs: Mapping[str, tuple[MatchedEvaluation, ...]]
    path: Path | None = None


def _counts(
    recordings: Mapping[str, CasasRecording],
    homes: Sequence[str],
    states: Sequence[BehaviouralState],
    step: timedelta,
) -> dict[str, np.ndarray]:
    return {
        home: hour_state_counts(
            recordings[home],
            _regular_moments(recordings[home], step),
            states=states,
            step=step,
        )
        for home in homes
    }


def fold_models(
    recordings: Mapping[str, CasasRecording],
    protocol: TimePriorProtocol,
    states: Sequence[BehaviouralState],
) -> dict[str, dict[str, Any]]:
    """Each fold's periodic prior, time-disabled prior and rare states.

    Everything is fitted on the fold's training households only.
    """
    step = protocol.information_sets[0].resolution.step
    fitted: dict[str, dict[str, Any]] = {}
    for fold in protocol.folds:
        counts = _counts(recordings, fold.train, states, step)
        pooled = sum(counts.values(), np.zeros((24, len(states))))
        shares = pooled.sum(axis=0) / pooled.sum()
        fitted[fold.name] = {
            "periodic": fit_periodic_prior(
                counts, states=states, config=protocol.periodic_prior
            ),
            "untimed": fit_periodic_prior(
                {home: hour_collapsed(c) for home, c in counts.items()},
                states=states,
                config=protocol.periodic_prior,
            ),
            "shares": {state.value: float(s) for state, s in zip(states, shares)},
        }
    return fitted


def run_time_prior_experiment(
    recordings: Mapping[str, CasasRecording],
    protocol: TimePriorProtocol,
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> TimePriorResult:
    """Run the pre-specified experiment and build its record."""
    missing = sorted(set(protocol.homes) - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    sets = dict(zip(LABELS, protocol.information_sets))
    resolution = protocol.information_sets[0].resolution
    step = resolution.step

    runs: dict[str, tuple[MatchedEvaluation, ...]] = {}
    scores: dict[str, dict[str, Mapping[str, PredictionMetrics]]] = {}
    for label in ("I1", "I3"):
        runs[label] = tuple(
            run_matched_evaluation(
                recordings,
                split=fold,
                information_set=sets[label],
                models=protocol.models,
                seed=protocol.seed,
                data_source=data_source,
                metrics=protocol.metrics,
                resamples=protocol.resamples,
                confidence=protocol.confidence,
                inputs=inputs,
                experiment=f"{protocol.name}-{label}-{fold.name}",
            )
            for fold in protocol.folds
        )
        for spec in protocol.models:
            scores.setdefault(spec.name, {})[label] = held_out_metrics(
                runs[label], spec.name
            )
    space = runs["I1"][0].states
    base = StateOntology()
    if tuple(base.states) != space:
        raise ValueError("the generative model's states differ from the label space")
    digests = {
        label: {home: d for run in folded for home, d in run.moments.items()}
        for label, folded in runs.items()
    }
    fitted = fold_models(recordings, protocol, space)
    fold_of = {home: fold.name for fold in protocol.folds for home in fold.test}
    rare = sorted(
        state
        for state in (s.value for s in space)
        if any(f["shares"][state] < protocol.rare_share for f in fitted.values())
    )

    generative: dict[str, dict[str, dict[str, PredictionMetrics]]] = {
        model: {label: {} for label in CELLS[model]}
        for model in (ORIGINAL, UNTIMED, PERIODIC)
    }
    variants = ("population", *(f"adapted@{p:g}" for p in protocol.precisions))
    adaptation: dict[str, dict[str, dict[str, PredictionMetrics]]] = {
        variant: {"I1": {}, "I3": {}} for variant in variants
    }
    households: dict[str, dict[str, Any]] = {}
    for home in sorted(digests["I1"]):
        recording = recordings[home]
        fold = fold_of[home]
        moments = _regular_moments(recording, step)
        terms, _ = channel_likelihoods(recording.registry, resolution, base)
        start = min(o.timestamp for o in recording.observations)
        cutoff = start + timedelta(days=protocol.adaptation_days)
        own = hour_state_counts(
            recording, moments, states=space, step=step, until=cutoff
        )
        periodic = fitted[fold]["periodic"]
        untimed = untimed_ontology(fitted[fold]["untimed"], base)
        priors = {
            "population": periodic,
            **{
                f"adapted@{p:g}": replace(
                    periodic, config=replace(periodic.config, household_precision=p)
                ).adapt(home, own)
                for p in protocol.precisions
            },
        }
        labelled: int | None = None
        after: list[int] = []
        for label, info in sets.items():
            table = build_feature_table(recording, info, moments, household=home)
            built = _Household.build(home, "test", table, recording)
            if built.moments_sha256 != digests["I1"][home]:
                raise ValueError(
                    f"rows for {home!r} in {label} differ from the runner's"
                )
            if labelled is None:
                labelled = int(built.labelled.size)
                after = [
                    position
                    for position, row in enumerate(built.labelled)
                    if table.moments[row] > cutoff
                ]
            elif built.labelled.size != labelled:
                raise ValueError(f"labelled rows for {home!r} differ between sets")
            rows = built.labelled
            if label in CELLS[ORIGINAL]:
                beliefs = restricted_posteriors(table, terms, base)[rows]
                generative[ORIGINAL][label][home] = _score(built.labels, beliefs, space)
                beliefs = restricted_posteriors(table, terms, untimed)[rows]
                generative[UNTIMED][label][home] = _score(built.labels, beliefs, space)
            if label in CELLS[PERIODIC]:
                labels_after = [built.labels[i] for i in after]
                for variant, prior in priors.items():
                    beliefs = restricted_posteriors(
                        table, terms, base, periodic_prior=prior, household=home
                    )[rows]
                    if variant == "population":
                        generative[PERIODIC][label][home] = _score(
                            built.labels, beliefs, space
                        )
                    if after:
                        adaptation[variant][label][home] = _score(
                            labels_after, beliefs[after], space
                        )
        households[home] = {
            "fold": fold,
            "labelled": labelled,
            "moments_sha256": digests["I1"][home],
            "adaptation_cutoff": cutoff.isoformat(),
            "adaptation_labelled_hours": float(own.sum()),
            "scored_after_adaptation": len(after),
        }
    for model, by_label in generative.items():
        scores[model] = dict(by_label.items())

    def values(model: str, label: str, metric: Any) -> dict[str, float | None]:
        return household_values(scores[model][label], metric)

    cells = [
        _cell(model, label, scores[model][label], space, protocol.as_gap_protocol())
        for model, labels in CELLS.items()
        for label in labels
    ]

    gap = protocol.as_gap_protocol()
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

    declared_variant = f"adapted@{protocol.periodic_prior.household_precision:g}"
    arm: list[dict[str, Any]] = []
    for variant in variants[1:]:
        for label in ("I1", "I3"):
            comparisons = {
                metric: _paired(
                    household_values(adaptation[variant][label], metric),
                    household_values(adaptation["population"][label], metric),
                    metric,
                    gap,
                )
                for metric in protocol.metrics
            }
            verdicts = {
                metric: verdict(comparison, protocol.minimal_differences[metric])
                for metric, comparison in comparisons.items()
            }
            arm.append(
                {
                    "key": (
                        f"A-{label}"
                        if variant == declared_variant
                        else f"A-{label} ({variant})"
                    ),
                    "variant": variant,
                    "declared": variant == declared_variant,
                    "information_set": label,
                    "comparisons": comparisons,
                    "verdicts": verdicts,
                }
            )
    declared = next(a for a in arm if a["declared"] and a["information_set"] == "I1")
    hierarchy = hierarchy_conclusion(declared["verdicts"])

    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified; development panel, which earlier work has inspected",
        "states": [state.value for state in space],
        "rare_states": rare,
        "households": households,
        "cells": cells,
        "estimands": estimands,
        "adaptation": arm,
        "conclusions": {
            "time_of_day": next(e for e in estimands if e["key"] == "P1")["conclusion"],
            "against_original": next(e for e in estimands if e["key"] == "P2")[
                "conclusion"
            ],
            "household_adaptation": hierarchy,
        },
        "priors": {
            fold: {
                "periodic": {
                    **f["periodic"].to_dict(),
                    "sha256": f["periodic"].sha256(),
                },
                "untimed": {**f["untimed"].to_dict(), "sha256": f["untimed"].sha256()},
                "training_shares": f["shares"],
            }
            for fold, f in fitted.items()
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
            "Verdicts compare effect sizes and their household bootstrap intervals "
            "with declared minimal important differences; they are not "
            "significance tests.",
            "Only the adaptation arm uses held-out households' own labels, from "
            "before the moments it scores.",
        ],
        inputs=list(inputs),
        split={
            "folds": [{**f.to_dict(), "sha256": f.sha256()} for f in protocol.folds]
        },
        information_set={
            "sets": [
                {"label": label, **info.to_dict(), "sha256": info.sha256()}
                for label, info in sets.items()
            ]
        },
        preprocessing={
            "moments": "one per step from each recording's first observation to "
            "its last, identical for every model",
            "labels": "truth_series of each recording's annotations; unlabelled "
            "moments are not scored",
            "step_seconds": step.total_seconds(),
        },
        models=[
            *(
                ModelRecord(spec.name, spec.configuration, str(spec.to_dict()["build"]))
                for spec in protocol.models
            ),
            ModelRecord(ORIGINAL, GENERATIVE_CONFIGURATION, "restricted_posteriors"),
            ModelRecord(
                UNTIMED,
                {
                    **UNTIMED_CONFIGURATION,
                    "periodic_prior": protocol.periodic_prior.to_dict(),
                },
                "restricted_posteriors(untimed_ontology(...))",
            ),
            ModelRecord(
                PERIODIC,
                {
                    **GENERATIVE_PERIODIC_CONFIGURATION,
                    "periodic_prior": protocol.periodic_prior.to_dict(),
                },
                "restricted_posteriors(periodic_prior=...)",
            ),
        ],
        household_metrics={
            **{
                f"{model}@{label}": {h: m.to_dict() for h, m in per_home.items()}
                for model, by_label in scores.items()
                for label, per_home in by_label.items()
            },
            **{
                f"adaptation/{variant}@{label}": {
                    h: m.to_dict() for h, m in per_home.items()
                }
                for variant, by_label in adaptation.items()
                for label, per_home in by_label.items()
            },
        },
        intervals=_intervals(estimands, arm),
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return TimePriorResult(record=record, runs=runs, path=path)


def _intervals(
    estimands: Sequence[Mapping[str, Any]], arm: Sequence[Mapping[str, Any]]
) -> list[ReportedInterval]:
    reported: list[ReportedInterval] = []
    for entry in (*estimands, *arm):
        for metric, comparison in entry["comparisons"].items():
            mean = comparison["mean"]
            if mean["interval"] is None:
                continue
            reported.append(
                ReportedInterval(
                    label=f"{entry['key']}: mean {metric} difference",
                    estimate=mean["estimate"],
                    low=mean["interval"]["low"],
                    high=mean["interval"]["high"],
                    confidence=mean["interval"]["confidence"],
                    method=mean["interval"]["method"],
                    unit="household",
                    n=comparison["n"],
                )
            )
    return reported
