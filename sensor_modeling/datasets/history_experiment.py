"""Phase 3.2: the pre-specified evaluation of the explicit-history generative model.

The question is whether the generative model recovers materially more from
recent history once it has an explicit history state
(:mod:`~sensor_modeling.fusion.history`) than the +0.021 the original model
recovered in Phase 1. Every setting, comparison and criterion is declared in
:class:`HistoryProtocol` and frozen in a committed file before any household
is scored.

Models and where they are scored
--------------------------------
Every comparison between two models is made on identical information: the
same set, households and moments.

| Model | Scored in |
| --- | --- |
| ``generative``, the original model | ``I0``, ``I2`` |
| ``generative_history``, with the explicit history state | ``I2`` |
| ``generative_periodic``, with the hierarchical time prior | ``I1``, ``I3`` |
| ``generative_periodic_history``, with both, declared here | ``I3`` |
| ``diagnostic``, ``logistic``, ``state_frequency`` | ``I0`` to ``I3`` |

- **Why ``generative_history`` is not scored in ``I0``.** In ``I0`` the
  history is missing, which is exactly neutral, so it would be the original
  model. Its gain from recent history is therefore ``generative_history`` in
  ``I2`` against ``generative`` in ``I0``.
- **Fitting.** Each fold's periodic prior and history coefficients are fitted
  on that fold's training households only. Held-out households get the
  population prior and the fitted coefficients, with no adaptation.

Estimand kinds
--------------
- **Formulation.** Two models, one set.
- **Information.** One model family, a smaller set against a larger one.
- **Interaction.** How much more recent history is worth to a family when
  the hour is known than when it is not.

A comparison of models with different information is never reported as a
modelling improvement.
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
from ..fusion.history import HistoryConfig, HistoryModel
from ..states.ontology import BehaviouralState, StateOntology
from .casas import CasasRecording
from .history_fit import fit_history_model
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
    _oriented_gap,
    _paired,
    _score,
    gap_models,
)
from .restricted_filter import channel_likelihoods, restricted_posteriors
from .time_prior_experiment import CRITERIA as TIME_PRIOR_CRITERIA
from .time_prior_experiment import MINIMAL_DIFFERENCES, conclusion, verdict

PROTOCOL_SCHEMA = "history-protocol/1"
RESULT_SCHEMA = "history-results/1"

ORIGINAL = "generative"
PERIODIC = "generative_periodic"
HISTORY = "generative_history"
COMBINED = "generative_periodic_history"

#: Where each model is scored.
CELLS: dict[str, tuple[str, ...]] = {
    ORIGINAL: ("I0", "I2"),
    HISTORY: ("I2",),
    PERIODIC: ("I1", "I3"),
    COMBINED: ("I3",),
    DIAGNOSTIC: LABELS,
    LOGISTIC: LABELS,
    REFERENCE: LABELS,
}

HISTORY_CONFIGURATION: dict[str, object] = {
    **GENERATIVE_CONFIGURATION,
    "model": "restricted generative filter with the explicit history state",
    "history": "the current window's rates conditioned on each channel's "
    "activations over the set's three lagged windows",
    "fitted": "the history coefficients, per fold, on the fold's training "
    "households' labels only",
}

COMBINED_CONFIGURATION: dict[str, object] = {
    **GENERATIVE_PERIODIC_CONFIGURATION,
    "model": "restricted generative filter with the periodic prior and the "
    "explicit history state",
    "history": HISTORY_CONFIGURATION["history"],
    "fitted": "the periodic prior and the history coefficients, per fold, on "
    "the fold's training households' labels only",
}


@dataclass(frozen=True)
class Estimand:
    """One pre-specified paired comparison, and what kind of comparison it is."""

    key: str
    role: str
    kind: str
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
            "kind": self.kind,
            "question": self.question,
            "model": f"{self.model}@{self.model_set}",
            "reference": f"{self.reference}@{self.reference_set}",
        }


ESTIMANDS: tuple[Estimand, ...] = (
    Estimand(
        "H1",
        "primary",
        "formulation",
        "what the explicit history state adds on identical information",
        HISTORY,
        "I2",
        ORIGINAL,
        "I2",
    ),
    Estimand(
        "H2",
        "primary",
        "information",
        "what recent history is worth to the explicit-history model",
        HISTORY,
        "I2",
        ORIGINAL,
        "I0",
    ),
    Estimand(
        "H3",
        "primary",
        "formulation",
        "the diagnostic against the explicit-history model",
        DIAGNOSTIC,
        "I2",
        HISTORY,
        "I2",
    ),
    Estimand(
        "H4",
        "primary",
        "formulation",
        "logistic regression against the explicit-history model",
        LOGISTIC,
        "I2",
        HISTORY,
        "I2",
    ),
    Estimand(
        "R1",
        "reference",
        "information",
        "what recent history is worth to the original model (+0.021 in Phase 1)",
        ORIGINAL,
        "I2",
        ORIGINAL,
        "I0",
    ),
    Estimand(
        "S1",
        "secondary",
        "formulation",
        "what the explicit history state adds when the hour is known",
        COMBINED,
        "I3",
        PERIODIC,
        "I3",
    ),
    Estimand(
        "S2",
        "secondary",
        "information",
        "what recent history is worth to the model with the history state and the hour",
        COMBINED,
        "I3",
        PERIODIC,
        "I1",
    ),
    Estimand(
        "S3",
        "secondary",
        "formulation",
        "the diagnostic against the model with both",
        DIAGNOSTIC,
        "I3",
        COMBINED,
        "I3",
    ),
    Estimand(
        "S4",
        "secondary",
        "formulation",
        "logistic regression against the model with both",
        LOGISTIC,
        "I3",
        COMBINED,
        "I3",
    ),
    Estimand(
        "R2",
        "reference",
        "information",
        "what recent history is worth to "
        "the periodic model without the history state (+0.002 in Phase 3.1)",
        PERIODIC,
        "I3",
        PERIODIC,
        "I1",
    ),
)

#: Estimands whose per-state recall differences are reported.
RECALL_ESTIMANDS = ("H1", "H2", "S1")

#: Model families whose time-by-history interaction is reported, member by set.
FAMILIES: dict[str, dict[str, str]] = {
    "generative with the history state": {
        "I0": ORIGINAL,
        "I1": PERIODIC,
        "I2": HISTORY,
        "I3": COMBINED,
    },
    "generative without the history state": {
        "I0": ORIGINAL,
        "I1": PERIODIC,
        "I2": ORIGINAL,
        "I3": PERIODIC,
    },
    "diagnostic": {label: DIAGNOSTIC for label in LABELS},
    "logistic": {label: LOGISTIC for label in LABELS},
}

CRITERIA: dict[str, str] = {
    "verdicts": TIME_PRIOR_CRITERIA["verdicts"],
    "success": "H1 balanced accuracy favours the explicit-history model, and H1 "
    "log loss and H1 calibration error do not favour the original model",
    "failure": "H1 balanced accuracy is negligible or favours the original model",
    "inconclusive": "neither success nor failure",
    "with_the_hour": "S1 is judged by the same rule",
    "materially_more": "H1 equals the explicit-history model's history gain (H2) "
    "minus the original model's (R1) on the same homes. Materially more means H1 "
    "favours the model: at least the declared minimal difference, with its "
    "interval above zero. The earlier +0.021 is re-measured as R1 and is not a "
    "threshold",
    "interaction": "for each family, the history gain with the hour (I1 to I3) "
    "minus the gain without it (I0 to I2), per household; positive means time "
    "and history are complementary",
}


@dataclass(frozen=True)
class HistoryProtocol:
    """Everything the experiment fixes before any household is scored."""

    folds: tuple[HouseholdSplit, ...]
    splits_sha256: str
    information_sets: tuple[InformationSet, ...] = field(
        default_factory=nested_information_sets
    )
    periodic_prior: PeriodicPriorConfig = field(default_factory=PeriodicPriorConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    models: tuple[ModelSpec, ...] = field(default_factory=gap_models)
    seed: int = 0
    metrics: tuple[str, ...] = DEFAULT_METRICS
    resamples: int = 10_000
    confidence: float = 0.95
    minimal_differences: Mapping[str, float] = field(
        default_factory=lambda: dict(MINIMAL_DIFFERENCES)
    )
    rare_share: float = 0.05
    name: str = "phase3-explicit-history"

    def __post_init__(self) -> None:
        """Validate through the recoverable-gap protocol, then the rest."""
        base = self.as_gap_protocol()
        if not isinstance(self.splits_sha256, str) or len(self.splits_sha256) != 64:
            raise ValueError("splits_sha256 must be a SHA-256 hex digest")
        if (
            self.history.window_steps
            > base.information_sets[0].resolution.history_steps
        ):
            raise ValueError(
                "the history window must fit inside the sets' lagged windows"
            )
        missing = sorted({*base.metrics, "recall"} - set(self.minimal_differences))
        if missing or any(
            not math.isfinite(v) or v <= 0.0 for v in self.minimal_differences.values()
        ):
            raise ValueError(
                f"every metric needs a positive minimal difference; missing {missing}"
            )
        if not 0.0 < self.rare_share < 1.0:
            raise ValueError("rare_share must lie in (0, 1)")
        object.__setattr__(self, "folds", base.folds)
        object.__setattr__(self, "information_sets", base.information_sets)
        object.__setattr__(self, "metrics", base.metrics)
        object.__setattr__(self, "models", base.models)
        object.__setattr__(self, "minimal_differences", dict(self.minimal_differences))

    def as_gap_protocol(self) -> GapProtocol:
        """The equivalent recoverable-gap protocol, for validation and helpers."""
        return GapProtocol(
            tuple(self.folds),
            information_sets=tuple(self.information_sets),
            seed=self.seed,
            metrics=tuple(self.metrics),
            resamples=self.resamples,
            confidence=self.confidence,
            models=tuple(self.models),
            periodic_prior=self.periodic_prior,
        )

    @property
    def homes(self) -> tuple[str, ...]:
        """Every household, sorted."""
        return tuple(sorted(home for fold in self.folds for home in fold.test))

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        history = self.history.to_dict()
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
                HISTORY: {**HISTORY_CONFIGURATION, "history_state": history},
                PERIODIC: {
                    **GENERATIVE_PERIODIC_CONFIGURATION,
                    "periodic_prior": self.periodic_prior.to_dict(),
                },
                COMBINED: {
                    **COMBINED_CONFIGURATION,
                    "periodic_prior": self.periodic_prior.to_dict(),
                    "history_state": history,
                },
            },
            "inclusion": {
                PERIODIC: "included because Phase 3.1 met its pre-specified "
                "success rule on the development panel",
                COMBINED: "declared here, before any result of the history "
                "state was observed",
            },
            "cells": {model: list(labels) for model, labels in CELLS.items()},
            "metrics": list(self.metrics),
            "per_state_recall": {
                "estimands": list(RECALL_ESTIMANDS),
                "rare_rule": f"share of labelled time below {self.rare_share} in "
                "either fold's training households",
            },
            "calibration": "expected calibration error over 10 confidence bins",
            "estimands": [estimand.to_dict() for estimand in ESTIMANDS],
            "interactions": {
                family: dict(members) for family, members in FAMILIES.items()
            },
            "minimal_differences": dict(self.minimal_differences),
            "criteria": CRITERIA,
            "bootstrap": {
                "unit": "household",
                "statistics": "mean and median paired differences",
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


def declared_protocol(splits: FrozenSplits) -> HistoryProtocol:
    """The protocol as declared for the development panel's frozen folds."""
    return HistoryProtocol(splits.folds, splits.sha256)


def check_frozen_protocol(protocol: HistoryProtocol, path: Path) -> str:
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
class HistoryResult:
    """A completed experiment: its record, the runner's runs, and where it was written."""

    record: ExperimentRecord
    runs: Mapping[str, tuple[MatchedEvaluation, ...]]
    path: Path | None = None


def fold_models(
    recordings: Mapping[str, CasasRecording],
    protocol: HistoryProtocol,
    states: Sequence[BehaviouralState],
) -> dict[str, dict[str, Any]]:
    """Each fold's periodic prior, history coefficients and training shares.

    Everything is fitted on the fold's training households only.
    """
    resolution = protocol.information_sets[0].resolution
    fitted: dict[str, dict[str, Any]] = {}
    for fold in protocol.folds:
        counts = {
            home: hour_state_counts(
                recordings[home],
                _regular_moments(recordings[home], resolution.step),
                states=states,
                step=resolution.step,
            )
            for home in fold.train
        }
        pooled = sum(counts.values(), np.zeros((24, len(states))))
        fitted[fold.name] = {
            "periodic": fit_periodic_prior(
                counts, states=states, config=protocol.periodic_prior
            ),
            "history": fit_history_model(
                {home: recordings[home] for home in fold.train},
                config=protocol.history,
                resolution=resolution,
            ),
            "shares": {
                state.value: float(share)
                for state, share in zip(states, pooled.sum(axis=0) / pooled.sum())
            },
        }
    return fitted


def run_history_experiment(
    recordings: Mapping[str, CasasRecording],
    protocol: HistoryProtocol,
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> HistoryResult:
    """Run the pre-specified experiment and build its record."""
    missing = sorted(set(protocol.homes) - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    sets = dict(zip(LABELS, protocol.information_sets))
    resolution = protocol.information_sets[0].resolution
    step = resolution.step
    gap = protocol.as_gap_protocol()

    runs: dict[str, tuple[MatchedEvaluation, ...]] = {}
    scores: dict[str, dict[str, Mapping[str, PredictionMetrics]]] = {}
    for label, info in sets.items():
        runs[label] = tuple(
            run_matched_evaluation(
                recordings,
                split=fold,
                information_set=info,
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
    space = runs["I0"][0].states
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
        state.value
        for state in space
        if any(f["shares"][state.value] < protocol.rare_share for f in fitted.values())
    )

    generative: dict[str, dict[str, dict[str, PredictionMetrics]]] = {
        model: {label: {} for label in CELLS[model]}
        for model in (ORIGINAL, PERIODIC, HISTORY, COMBINED)
    }
    households: dict[str, dict[str, Any]] = {}
    for home in sorted(digests["I0"]):
        recording = recordings[home]
        fold = fold_of[home]
        prior: PeriodicStatePrior = fitted[fold]["periodic"]
        history: HistoryModel = fitted[fold]["history"]
        moments = _regular_moments(recording, step)
        terms, _ = channel_likelihoods(recording.registry, resolution, base)
        for label, info in sets.items():
            table = build_feature_table(recording, info, moments, household=home)
            built = _Household.build(home, "test", table, recording)
            if built.moments_sha256 != digests[label][home]:
                raise ValueError(
                    f"rows for {home!r} in {label} differ from the runner's"
                )
            rows = built.labelled
            variants: dict[str, Any] = {}
            if label in CELLS[ORIGINAL]:
                variants[ORIGINAL] = {}
            if label in CELLS[HISTORY]:
                variants[HISTORY] = {"history_model": history}
            if label in CELLS[PERIODIC]:
                variants[PERIODIC] = {"periodic_prior": prior}
            if label in CELLS[COMBINED]:
                variants[COMBINED] = {"periodic_prior": prior, "history_model": history}
            for model, options in variants.items():
                beliefs = restricted_posteriors(table, terms, base, **options)[rows]
                generative[model][label][home] = _score(built.labels, beliefs, space)
        households[home] = {
            "fold": fold,
            "labelled": int(built.labelled.size),
            "moments_sha256": digests["I0"][home],
        }
    for model, by_label in generative.items():
        scores[model] = dict(by_label.items())

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

    interactions: list[dict[str, Any]] = []
    for family, members in FAMILIES.items():
        comparisons = {
            metric: _paired(
                _oriented_gap(
                    values(members["I3"], "I3", metric),
                    values(members["I1"], "I1", metric),
                    metric,
                ),
                _oriented_gap(
                    values(members["I2"], "I2", metric),
                    values(members["I0"], "I0", metric),
                    metric,
                ),
                metric,
                gap,
                higher_is_better=True,
            )
            for metric in protocol.metrics
        }
        interactions.append(
            {
                "family": family,
                "members": dict(members),
                "comparisons": comparisons,
                "verdicts": {
                    metric: verdict(comparison, protocol.minimal_differences[metric])
                    for metric, comparison in comparisons.items()
                },
            }
        )

    by_key = {e["key"]: e for e in estimands}
    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified; development panel, which earlier work has inspected",
        "states": [state.value for state in space],
        "rare_states": rare,
        "households": households,
        "cells": cells,
        "estimands": estimands,
        "interactions": interactions,
        "conclusions": {
            "explicit_history": by_key["H1"]["conclusion"],
            "with_the_hour": by_key["S1"]["conclusion"],
        },
        "fitted": {
            fold: {
                "periodic_prior": {
                    **f["periodic"].to_dict(),
                    "sha256": f["periodic"].sha256(),
                },
                "history_model": {
                    **f["history"].to_dict(),
                    "sha256": f["history"].sha256(),
                },
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
            "Formulation estimands compare two models on one set; information "
            "estimands compare one model family across nested sets.",
            "Verdicts compare effect sizes and household bootstrap intervals with "
            "declared minimal important differences; they are not significance "
            "tests.",
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
                HISTORY,
                {**HISTORY_CONFIGURATION, "history_state": protocol.history.to_dict()},
                "restricted_posteriors(history_model=...)",
            ),
            ModelRecord(
                PERIODIC,
                {
                    **GENERATIVE_PERIODIC_CONFIGURATION,
                    "periodic_prior": protocol.periodic_prior.to_dict(),
                },
                "restricted_posteriors(periodic_prior=...)",
            ),
            ModelRecord(
                COMBINED,
                {
                    **COMBINED_CONFIGURATION,
                    "periodic_prior": protocol.periodic_prior.to_dict(),
                    "history_state": protocol.history.to_dict(),
                },
                "restricted_posteriors(periodic_prior=..., history_model=...)",
            ),
        ],
        household_metrics={
            f"{model}@{label}": {h: m.to_dict() for h, m in per_home.items()}
            for model, by_label in scores.items()
            for label, per_home in by_label.items()
        },
        intervals=_intervals(estimands, interactions),
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return HistoryResult(record=record, runs=runs, path=path)


def _intervals(
    estimands: Sequence[Mapping[str, Any]], interactions: Sequence[Mapping[str, Any]]
) -> list[ReportedInterval]:
    reported: list[ReportedInterval] = []
    labelled = [
        *((entry["key"], entry) for entry in estimands),
        *((f"interaction, {entry['family']}", entry) for entry in interactions),
    ]
    for label, entry in labelled:
        for metric, comparison in entry["comparisons"].items():
            for statistic in ("mean", "median"):
                estimate = comparison[statistic]
                if estimate["interval"] is None:
                    continue
                reported.append(
                    ReportedInterval(
                        label=f"{label}: {statistic} {metric} difference",
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
