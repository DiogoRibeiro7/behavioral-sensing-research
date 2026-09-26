"""Phase 1: the recoverable-information gap under matched information sets.

The roadmap asks how much of the real-data gap comes from what the sensors
support and how much from the current probabilistic formulation. This module
runs the comparison that separates the two:

- **Information.** One model, fixed, is scored on a smaller and a larger nested
  set. The difference is what the added information is worth to that model.
- **Formulation.** Two models are scored on the same set, households and
  moments. The difference is what the formulation is worth given that
  information.

The sets are the four Phase 1 sets:

- ``I0``: current evidence;
- ``I1``: plus time of day;
- ``I2``: plus recent room-resolved history;
- ``I3``: plus both.

The models:

- **Diagnostic.** :class:`GradientBoostingBaseline`, the supervised diagnostic.
- **Logistic.** :class:`LogisticBaseline`, regularised multinomial logistic
  regression.
- **Generative.** The generative filter's model restricted to the set
  (:mod:`~sensor_modeling.datasets.restricted_filter`). It is scored only where
  it can consume the set exactly: ``I0`` and ``I2``. ``I1`` and ``I3`` are
  recorded as unsupported, with the reason.
- **State frequency.** A no-information reference.

Two further runs are recorded but are not matched to any set:

- **The production filter.** It runs as deployed, and its belief conditions on
  every earlier window.
- **The circadian candidate.** It is listed as unsupported, because its
  profile was fitted on every development home.

The two differences do not add up to a decomposition. What the added
information is worth can depend on the formulation. The record therefore
reports the interaction: the difference between what the same added
information is worth to two models.

Fitting follows the matched runner. Each model is fitted on its fold's training
households and scored on the held-out ones, under frozen folds. No setting is
selected on any data, so no household is used for tuning.
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

from ..evaluation.households import (
    compare_households,
    household_values,
    recall_of,
    summarise_households,
)
from ..evaluation.metrics import PredictionMetrics, prediction_metrics
from ..evaluation.provenance import (
    ExperimentRecord,
    InputArtifact,
    ModelRecord,
    ReportedInterval,
)
from ..evaluation.resampling import (
    check_settings,
    percentile_interval,
    resample_indices,
)
from ..states.ontology import BehaviouralState, StateOntology
from .baseline_models import (
    GradientBoostingBaseline,
    LogisticBaseline,
    StateFrequencyBaseline,
)
from .casas import CasasRecording
from .information_sets import (
    InformationComponent,
    InformationSet,
    build_feature_table,
    nested_information_sets,
)
from .matched_evaluation import (
    COMPARABLE_METRICS,
    MATCHED_METRIC_DEFINITIONS,
    HouseholdSplit,
    MatchedEvaluation,
    ModelSpec,
    _finite,
    _Household,
    _regular_moments,
    compare_information_sets,
    held_out_metrics,
    run_matched_evaluation,
)
from .restricted_filter import (
    channel_likelihoods,
    filter_posteriors,
    restricted_posteriors,
    unsupported_reason,
)

#: Schema of the results section of a recoverable-gap record.
RESULT_SCHEMA = "recoverable-gap/1"

#: Schema of a frozen household-split file.
SPLITS_SCHEMA = "household-splits/1"

#: Short labels of the four Phase 1 sets, in nesting order.
LABELS = ("I0", "I1", "I2", "I3")

REFERENCE = "state_frequency"
DIAGNOSTIC = "diagnostic"
LOGISTIC = "logistic"
GENERATIVE = "generative"
FILTER = "filter"

#: Models compared within each set, in reporting order.
CANDIDATES = (DIAGNOSTIC, LOGISTIC, GENERATIVE)

#: Nested pairs of sets, smaller first, compared within each model.
NESTED_PAIRS = (("I0", "I1"), ("I0", "I2"), ("I1", "I3"), ("I2", "I3"), ("I0", "I3"))

#: Pairs of models compared within each set; positive favours the first.
FORMULATION_PAIRS = (
    (DIAGNOSTIC, LOGISTIC),
    (DIAGNOSTIC, GENERATIVE),
    (LOGISTIC, GENERATIVE),
)

#: Metrics every recoverable-gap protocol reports.
DEFAULT_METRICS = ("balanced_accuracy", "log_loss", "brier", "calibration_error")

#: Why the production filter belongs to no declared set.
FILTER_UNMATCHED = (
    "the production filter is recursive: its belief conditions on every earlier "
    "window and on health and attribution layers built from the whole history, "
    "so it belongs to no declared set; its information strictly contains I0 and "
    "I2 and neither contains nor is contained in I1 or I3"
)

#: Why the circadian candidate is not scored on this panel.
CIRCADIAN_UNSCORED = (
    "the v0.3 circadian candidate's profile was fitted on every development "
    "home, so no household of this panel is held out from it"
)

#: How the restricted generative model is specified. Nothing is fitted.
GENERATIVE_CONFIGURATION: dict[str, object] = {
    "model": "restricted generative filter",
    "prior": "stationary distribution of the default ontology, one step before "
    "the first declared window",
    "transition": "default StateOntology, time-homogeneous, one step per window",
    "emissions": "default_emissions of the household registry, pooled per channel",
    "reliability": 1.0,
    "attribution": 1.0,
    "windows": "the set's current window, plus its lagged windows with recent history",
    "prediction": "most probable state of the posterior; no abstention",
    "fitted": "nothing: every parameter is the declared default",
}

#: How the production filter is run. Nothing is fitted.
FILTER_CONFIGURATION: dict[str, object] = {
    "model": "BehaviouralSensingPipeline",
    "configuration": "PipelineConfig defaults at the information-set step",
    "information": "every earlier observation, through the recursive posterior "
    "and the health and attribution layers",
    "prediction": "most probable state of the posterior; the pipeline's "
    "abstentions are not scored",
    "fitted": "nothing",
}


def _build_reference(seed: int) -> StateFrequencyBaseline:
    return StateFrequencyBaseline()


def _build_diagnostic(seed: int) -> GradientBoostingBaseline:
    return GradientBoostingBaseline(seed=seed)


def _build_logistic(seed: int) -> LogisticBaseline:
    return LogisticBaseline(seed=seed)


def gap_models() -> tuple[ModelSpec, ...]:
    """The runner models of the protocol, reference first, every setting fixed."""
    seeded = {"random_state": "run seed"}
    return (
        ModelSpec(
            REFERENCE, _build_reference, StateFrequencyBaseline().configuration()
        ),
        ModelSpec(
            DIAGNOSTIC,
            _build_diagnostic,
            {**GradientBoostingBaseline().configuration(), **seeded},
        ),
        ModelSpec(
            LOGISTIC, _build_logistic, {**LogisticBaseline().configuration(), **seeded}
        ),
    )


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class FrozenSplits:
    """Household folds read from a frozen split file.

    Attributes
    ----------
    folds
        The cross-fitted folds, in file order.
    homes
        Each household's recording file name and SHA-256.
    sha256
        SHA-256 of the file's bytes, recorded as input provenance.
    """

    folds: tuple[HouseholdSplit, ...]
    homes: Mapping[str, Mapping[str, str]]
    sha256: str


def load_frozen_splits(path: Path) -> FrozenSplits:
    """Read and check a frozen household-split file.

    Raises
    ------
    ValueError
        If the file has another schema, or its folds and homes disagree.
    """
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema") != SPLITS_SCHEMA:
        raise ValueError(f"{path} is not a {SPLITS_SCHEMA} file")
    folds = tuple(
        HouseholdSplit(
            fold["name"], train=tuple(fold["train"]), test=tuple(fold["test"])
        )
        for fold in payload["folds"]
    )
    homes = {home: dict(entry) for home, entry in payload["homes"].items()}
    named = {home for fold in folds for home in fold.households}
    if named != set(homes):
        raise ValueError("the folds and the listed homes name different households")
    return FrozenSplits(folds, homes, hashlib.sha256(raw).hexdigest())


@dataclass(frozen=True)
class GapProtocol:
    """Everything the experiment fixes before any household is scored.

    Attributes
    ----------
    folds
        Cross-fitted folds. Their held-out sets must partition the households,
        each fold must train on every household it does not hold out, and none
        may declare development households, since nothing is tuned.
    information_sets
        The four nested Phase 1 sets at one resolution.
    seed
        Seeds model construction, row order and every household bootstrap.
    metrics
        Metrics summarised and compared; they must include balanced accuracy.
    resamples, confidence
        Household bootstrap settings. Intervals are percentile intervals.
    models
        The runner models: the reference, the diagnostic and the logistic
        model, in that order. Defaults to :func:`gap_models`, the declared
        settings. Their configurations are recorded with the result.
    name
        Name of the record.
    """

    folds: tuple[HouseholdSplit, ...]
    information_sets: tuple[InformationSet, ...] = field(
        default_factory=nested_information_sets
    )
    seed: int = 0
    metrics: tuple[str, ...] = DEFAULT_METRICS
    resamples: int = 10_000
    confidence: float = 0.95
    models: tuple[ModelSpec, ...] = field(default_factory=gap_models)
    name: str = "phase1-recoverable-information-gap"

    def __post_init__(self) -> None:
        """Refuse a protocol that could score a household more than once or tune."""
        folds = tuple(self.folds)
        if len(folds) < 2:
            raise ValueError("at least two folds are required")
        if len({fold.name for fold in folds}) != len(folds):
            raise ValueError("fold names must be unique")
        held_out = [home for fold in folds for home in fold.test]
        if len(held_out) != len(set(held_out)):
            raise ValueError("a household is held out in more than one fold")
        everyone = set(held_out)
        for fold in folds:
            if fold.development:
                raise ValueError(
                    f"fold {fold.name!r} declares development households, but "
                    "nothing in this protocol is tuned"
                )
            if set(fold.train) != everyone - set(fold.test):
                raise ValueError(
                    f"fold {fold.name!r} must train on exactly the households "
                    "it does not hold out"
                )
        sets = tuple(self.information_sets)
        if not sets or sets != nested_information_sets(sets[0].resolution):
            raise ValueError(
                "the protocol compares exactly the four nested Phase 1 sets at "
                "one resolution"
            )
        metrics = tuple(self.metrics)
        if len(set(metrics)) != len(metrics) or set(metrics) - set(COMPARABLE_METRICS):
            raise ValueError(
                f"metrics must be distinct names from {sorted(COMPARABLE_METRICS)}"
            )
        if "balanced_accuracy" not in metrics:
            raise ValueError("the metrics must include balanced_accuracy")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        check_settings(self.confidence, self.resamples)
        models = tuple(self.models)
        if tuple(spec.name for spec in models) != (REFERENCE, DIAGNOSTIC, LOGISTIC):
            raise ValueError(
                f"the runner models must be {REFERENCE!r}, {DIAGNOSTIC!r} and "
                f"{LOGISTIC!r}, in that order"
            )
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "information_sets", sets)
        object.__setattr__(self, "metrics", metrics)

    @property
    def homes(self) -> tuple[str, ...]:
        """Every household in the protocol, sorted."""
        return tuple(sorted(home for fold in self.folds for home in fold.test))

    @property
    def labels(self) -> dict[str, str]:
        """Short label of each set, by set name."""
        return {
            information_set.name: label
            for label, information_set in zip(LABELS, self.information_sets)
        }

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        return {
            "result_schema": RESULT_SCHEMA,
            "name": self.name,
            "folds": [
                {**fold.to_dict(), "sha256": fold.sha256()} for fold in self.folds
            ],
            "information_sets": {
                label: {**info.to_dict(), "sha256": info.sha256()}
                for label, info in zip(LABELS, self.information_sets)
            },
            "models": {
                **{spec.name: dict(spec.configuration) for spec in self.models},
                GENERATIVE: {
                    **GENERATIVE_CONFIGURATION,
                    "unsupported": {
                        label: unsupported_reason(info)
                        for label, info in zip(LABELS, self.information_sets)
                        if unsupported_reason(info) is not None
                    },
                },
                FILTER: FILTER_CONFIGURATION,
            },
            "reference": REFERENCE,
            "metrics": list(self.metrics),
            "seed": self.seed,
            "bootstrap": {
                "unit": "household",
                "resamples": self.resamples,
                "confidence": self.confidence,
                "interval": "percentile",
            },
            "tuning": "none: every setting is fixed here and no fold declares "
            "development households",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return _digest(self.to_dict())


@dataclass(frozen=True)
class GapResult:
    """A completed experiment.

    Attributes
    ----------
    record
        The artifact, as written if an output directory was given.
    runs
        The matched runner's runs, by set label, one per fold.
    path
        Where the record was written, if anywhere.
    """

    record: ExperimentRecord
    runs: Mapping[str, tuple[MatchedEvaluation, ...]]
    path: Path | None = None


def _score(
    labels: Sequence[BehaviouralState],
    probabilities: np.ndarray,
    space: tuple[BehaviouralState, ...],
) -> PredictionMetrics:
    """Score a posterior by its most probable state, as the runner scores models."""
    predicted = [space[int(i)] for i in np.argmax(probabilities, axis=1)]
    return prediction_metrics(labels, predicted, probabilities, states=space)


def _level(values: Mapping[str, float | None], protocol: GapProtocol) -> dict[str, Any]:
    """Summarise one metric across households, with an interval for the mean."""
    summary: dict[str, Any] = summarise_households(values).to_dict()
    present = np.array(
        [v for v in values.values() if v is not None and math.isfinite(v)]
    )
    summary["mean_interval"] = (
        percentile_interval(
            present[
                resample_indices(present.size, protocol.resamples, protocol.seed)
            ].mean(axis=1),
            protocol.confidence,
        ).to_dict()
        if present.size > 1
        else None
    )
    return summary


def _paired(
    model: Mapping[str, float | None],
    reference: Mapping[str, float | None],
    metric: str,
    protocol: GapProtocol,
    *,
    higher_is_better: bool | None = None,
) -> dict[str, Any]:
    oriented = (
        COMPARABLE_METRICS[metric] if higher_is_better is None else higher_is_better
    )
    return compare_households(
        model,
        reference,
        higher_is_better=oriented,
        confidence=protocol.confidence,
        resamples=protocol.resamples,
        seed=protocol.seed,
    ).to_dict()


def _oriented_gap(
    first: Mapping[str, float | None],
    second: Mapping[str, float | None],
    metric: str,
) -> dict[str, float | None]:
    """Per household, how much *first* beats *second*; positive favours *first*."""
    sign = 1.0 if COMPARABLE_METRICS[metric] else -1.0
    return {
        home: (
            None
            if first.get(home) is None or second.get(home) is None
            else sign * (first[home] - second[home])  # type: ignore[operator]
        )
        for home in sorted(set(first) | set(second))
    }


def run_recoverable_gap(
    recordings: Mapping[str, CasasRecording],
    protocol: GapProtocol,
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> GapResult:
    """Run the recoverable-information-gap experiment.

    Parameters
    ----------
    recordings
        Every household the protocol names, by identifier.
    protocol
        The frozen protocol.
    data_source
        Where the recordings came from, recorded as provenance.
    inputs
        Files read, with their digests, recorded as provenance.
    output_dir
        If given, the record is written to ``<output_dir>/<protocol.name>.json``.

    Raises
    ------
    ValueError
        If a household has no recording, or the generative model's label space
        or rows differ from the runner's.
    """
    missing = sorted(set(protocol.homes) - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    labels = protocol.labels
    specs = protocol.models
    sets = {labels[info.name]: info for info in protocol.information_sets}
    resolution = protocol.information_sets[0].resolution

    runs: dict[str, tuple[MatchedEvaluation, ...]] = {}
    scores: dict[str, dict[str, Mapping[str, PredictionMetrics]]] = {}
    for label, info in sets.items():
        runs[label] = tuple(
            run_matched_evaluation(
                recordings,
                split=fold,
                information_set=info,
                models=specs,
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
        for spec in specs:
            scores.setdefault(spec.name, {})[label] = held_out_metrics(
                runs[label], spec.name
            )

    space = runs[LABELS[0]][0].states
    ontology = StateOntology()
    if tuple(ontology.states) != space:
        raise ValueError("the generative model's states differ from the label space")
    digests = {
        label: {home: d for run in folded for home, d in run.moments.items()}
        for label, folded in runs.items()
    }
    fold_of = {home: fold.name for fold in protocol.folds for home in fold.test}

    generative: dict[str, dict[str, PredictionMetrics]] = {
        label: {} for label, info in sets.items() if unsupported_reason(info) is None
    }
    production: dict[str, PredictionMetrics] = {}
    households: dict[str, dict[str, Any]] = {}
    for home in sorted(digests[LABELS[0]]):
        recording = recordings[home]
        moments = _regular_moments(recording, resolution.step)
        terms, ignored = channel_likelihoods(recording.registry, resolution, ontology)
        built: _Household | None = None
        for label, info in sets.items():
            if label not in generative:
                continue
            table = build_feature_table(recording, info, moments, household=home)
            built = _Household.build(home, "test", table, recording)
            if built.moments_sha256 != digests[label][home]:
                raise ValueError(
                    f"generative rows for {home!r} in {label} differ from the runner's"
                )
            posterior = restricted_posteriors(table, terms, ontology)
            generative[label][home] = _score(
                built.labels, posterior[built.labelled], space
            )
        if built is None:  # pragma: no cover - I0 is always supported
            raise ValueError("the generative model supports none of the sets")
        beliefs, states = filter_posteriors(recording, moments, step=resolution.step)
        if states != space:
            raise ValueError(
                "the production filter's states differ from the label space"
            )
        production[home] = _score(built.labels, beliefs[built.labelled], space)
        households[home] = {
            "fold": fold_of[home],
            "labelled": int(built.labelled.size),
            "moments_sha256": built.moments_sha256,
            "generative_ignored_sensors": list(ignored),
        }
    scores[GENERATIVE] = {label: per_home for label, per_home in generative.items()}

    def values(model: str, label: str, metric: Any) -> dict[str, float | None]:
        return household_values(scores[model][label], metric)

    def supported(model: str, label: str) -> bool:
        return label in scores[model]

    unsupported = [
        {
            "model": GENERATIVE,
            "information_set": label,
            "reason": unsupported_reason(sets[label]),
        }
        for label in LABELS
        if not supported(GENERATIVE, label)
    ]
    unsupported += [
        {"model": FILTER, "information_set": label, "reason": FILTER_UNMATCHED}
        for label in LABELS
    ]
    unsupported.append(
        {
            "model": "circadian filter",
            "information_set": "any",
            "reason": CIRCADIAN_UNSCORED,
        }
    )

    cells: list[dict[str, Any]] = []
    for model in (*CANDIDATES, REFERENCE):
        for label in LABELS:
            if not supported(model, label):
                cells.append(
                    {
                        "model": model,
                        "information_set": label,
                        "status": "unsupported",
                        "reason": unsupported_reason(sets[label]),
                    }
                )
                continue
            cells.append(_cell(model, label, scores[model][label], space, protocol))
    cells.append(_cell(FILTER, "unbounded", production, space, protocol))

    information: list[dict[str, Any]] = []
    for model in CANDIDATES:
        for small, large in NESTED_PAIRS:
            if not (supported(model, small) and supported(model, large)):
                continue
            for metric in protocol.metrics:
                comparison = (
                    _paired(
                        values(model, large, metric),
                        values(model, small, metric),
                        metric,
                        protocol,
                    )
                    if model == GENERATIVE
                    else compare_information_sets(
                        runs[small],
                        runs[large],
                        model=model,
                        metric=metric,
                        confidence=protocol.confidence,
                        resamples=protocol.resamples,
                        seed=protocol.seed,
                    ).to_dict()
                )
                information.append(
                    {
                        "model": model,
                        "from": small,
                        "to": large,
                        "metric": metric,
                        "comparison": comparison,
                    }
                )

    formulation: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for first, second in FORMULATION_PAIRS:
        for label in LABELS:
            if not (supported(first, label) and supported(second, label)):
                continue
            for metric in protocol.metrics:
                formulation.append(
                    {
                        "information_set": label,
                        "model": first,
                        "reference": second,
                        "metric": metric,
                        "comparison": _paired(
                            values(first, label, metric),
                            values(second, label, metric),
                            metric,
                            protocol,
                        ),
                    }
                )
        for small, large in NESTED_PAIRS:
            if not all(
                supported(model, label)
                for model in (first, second)
                for label in (small, large)
            ):
                continue
            for metric in protocol.metrics:
                interactions.append(
                    {
                        "model": first,
                        "reference": second,
                        "from": small,
                        "to": large,
                        "metric": metric,
                        "comparison": _paired(
                            _oriented_gap(
                                values(first, large, metric),
                                values(second, large, metric),
                                metric,
                            ),
                            _oriented_gap(
                                values(first, small, metric),
                                values(second, small, metric),
                                metric,
                            ),
                            metric,
                            protocol,
                            higher_is_better=True,
                        ),
                    }
                )

    against_filter: list[dict[str, Any]] = []
    for model in CANDIDATES:
        for label in LABELS:
            if not supported(model, label):
                continue
            relation = (
                "neither set contains the other"
                if InformationComponent.TIME_OF_DAY in sets[label].components
                else "the filter's information strictly contains the set"
            )
            for metric in protocol.metrics:
                against_filter.append(
                    {
                        "model": model,
                        "information_set": label,
                        "relation": relation,
                        "metric": metric,
                        "comparison": _paired(
                            values(model, label, metric),
                            household_values(production, metric),
                            metric,
                            protocol,
                        ),
                    }
                )

    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "exploratory: the development homes have been inspected in "
        "earlier work",
        "information_sets": {
            label: {"name": info.name, "sha256": info.sha256()}
            for label, info in sets.items()
        },
        "states": [state.value for state in space],
        "households": households,
        "cells": cells,
        "information_gains": information,
        "formulation_gaps": formulation,
        "interactions": interactions,
        "against_filter": against_filter,
        "unsupported": unsupported,
    }
    record = ExperimentRecord(
        experiment=protocol.name,
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        seeds=[protocol.seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions=MATCHED_METRIC_DEFINITIONS,
        notes=[
            "Information gains hold the model fixed and add information; "
            "formulation gaps hold the information fixed and change the model. "
            "They are not additive; see interactions.",
            "The production filter is a reference outside every declared set.",
        ],
        inputs=list(inputs),
        split={
            "folds": [
                {**fold.to_dict(), "sha256": fold.sha256()} for fold in protocol.folds
            ]
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
            "step_seconds": resolution.step.total_seconds(),
            "timezone": str(
                recordings[protocol.homes[0]].observations[0].timestamp.tzinfo
            ),
        },
        models=[
            *(
                ModelRecord(spec.name, spec.configuration, str(spec.to_dict()["build"]))
                for spec in specs
            ),
            ModelRecord(GENERATIVE, GENERATIVE_CONFIGURATION, "restricted_posteriors"),
            ModelRecord(FILTER, FILTER_CONFIGURATION, "filter_posteriors"),
        ],
        household_metrics={
            **{
                f"{model}@{label}": {
                    home: metrics.to_dict() for home, metrics in per_home.items()
                }
                for model, by_label in scores.items()
                for label, per_home in by_label.items()
            },
            FILTER: {home: metrics.to_dict() for home, metrics in production.items()},
        },
        intervals=_intervals(information, formulation, interactions, against_filter),
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return GapResult(record=record, runs=runs, path=path)


def _cell(
    model: str,
    label: str,
    per_home: Mapping[str, PredictionMetrics],
    space: tuple[BehaviouralState, ...],
    protocol: GapProtocol,
) -> dict[str, Any]:
    status = {REFERENCE: "reference", FILTER: "reference"}.get(model, "matched")
    return {
        "model": model,
        "information_set": label,
        "status": status,
        "households": len(per_home),
        "metrics": {
            metric: _level(household_values(per_home, metric), protocol)
            for metric in protocol.metrics
        },
        "per_state_recall": {
            state.value: _level(household_values(per_home, recall_of(state)), protocol)
            for state in space
        },
    }


def _intervals(
    information: Sequence[Mapping[str, Any]],
    formulation: Sequence[Mapping[str, Any]],
    interactions: Sequence[Mapping[str, Any]],
    against_filter: Sequence[Mapping[str, Any]],
) -> list[ReportedInterval]:
    """Every mean paired difference with its interval, labelled for quoting."""
    labelled = [
        *(
            (f"information gain: {e['model']}, {e['from']} to {e['to']}", e)
            for e in information
        ),
        *(
            (
                f"formulation gap: {e['model']} vs {e['reference']}, "
                f"{e['information_set']}",
                e,
            )
            for e in formulation
        ),
        *(
            (
                f"interaction: {e['model']} vs {e['reference']}, "
                f"{e['from']} to {e['to']}",
                e,
            )
            for e in interactions
        ),
        *(
            (f"against the filter: {e['model']}, {e['information_set']}", e)
            for e in against_filter
        ),
    ]
    reported: list[ReportedInterval] = []
    for label, entry in labelled:
        mean = entry["comparison"]["mean"]
        if mean["interval"] is None:
            continue
        reported.append(
            ReportedInterval(
                label=f"{label}: mean {entry['metric']} difference",
                estimate=mean["estimate"],
                low=mean["interval"]["low"],
                high=mean["interval"]["high"],
                confidence=mean["interval"]["confidence"],
                method=mean["interval"]["method"],
                unit="household",
                n=entry["comparison"]["n"],
            )
        )
    return reported
