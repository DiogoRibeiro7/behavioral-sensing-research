"""Comparing models that all receive exactly the same information.

A difference between two models is evidence about modelling only if both saw
the same information. This runner makes that the default instead of something
each experiment has to remember.

What the runner enforces
------------------------
- **One information set per run.** The runner builds every feature table itself
  from a single :class:`~sensor_modeling.datasets.InformationSet`. Models never
  receive a recording, a timestamp or a household identifier at prediction
  time, only the permitted columns.
- **Fitting never sees a held-out household.** Models are fitted on the
  training households, may use the development households for their own
  selection, and receive held-out rows only to predict, without labels.
- **No history from row order.** Rows reach a model in seeded random order, so
  adjacent rows cannot be chained into history the information set does not
  include.
- **Each prediction depends on its own row.** The held-out rows are predicted a
  second time as a random subset in a different order. A model whose prediction
  for any row changes is refused, because it is using information from other
  rows, such as statistics over a household's future.
- **Statistics are household-level.** Each held-out household is scored
  separately. Models are compared by paired differences across households
  scored on identical observations, never by pooling observations into one
  sample.

The generative filter is not a runner model. It is recursive and reads raw
observations, so it cannot be restricted to a declared information set; see
``docs/INFORMATION_SETS.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..evaluation.metrics import (
    ConfusionMatrix,
    PairedDifference,
    PredictionMetrics,
    _label_space,
    paired_difference,
    prediction_metrics,
    summarise,
)
from ..evaluation.provenance import METRIC_DEFINITIONS, ExperimentRecord
from ..states.ontology import DEFAULT_STATES, BehaviouralState
from .casas import CasasRecording, truth_series
from .information_sets import FeatureTable, InformationSet, build_feature_table

#: Identifier of the ``results`` layout. Bump when a field changes meaning.
RESULT_SCHEMA = "matched-evaluation/1"

#: Metrics the runner can summarise and compare, and whether higher is better.
COMPARABLE_METRICS: Mapping[str, bool] = {
    "balanced_accuracy": True,
    "accuracy": True,
    "macro_f1": True,
    "log_loss": False,
    "brier": False,
    "calibration_error": False,
}

#: Metrics that exist only for models reporting probabilities.
PROBABILISTIC_METRICS = frozenset({"log_loss", "brier", "calibration_error"})

DEFAULT_METRICS: tuple[str, ...] = ("balanced_accuracy", "log_loss", "brier")

#: Definitions written into every matched-evaluation record.
MATCHED_METRIC_DEFINITIONS: dict[str, str] = {
    **{
        name: METRIC_DEFINITIONS[name]
        for name in (
            "accuracy",
            "selective_accuracy",
            "abstention_rate",
            "balanced_accuracy",
            "macro_f1",
            "log_loss",
            "brier",
            "calibration_error",
            "per_class_recall",
            "confusion",
        )
    },
    "household_summary": (
        "Median, mean, standard deviation, minimum and maximum of a metric "
        "across held-out households, each household counted once. "
        "Observations are never pooled across households."
    ),
    "confusion_summed": (
        "Sum of the per-household confusion matrices. Descriptive only; no "
        "comparison is computed from it."
    ),
    "improvement": (
        "Per-household paired difference oriented so that a positive value "
        "favours the model over the reference: model minus reference where "
        "higher is better, reference minus model where lower is better."
    ),
    "mean_difference": (
        "Mean improvement across held-out households, every model scored on "
        "identical observations."
    ),
    "ci_low, ci_high": (
        "Percentile bootstrap interval on mean_difference, resampling "
        "households rather than observations."
    ),
    "effect_size": (
        "Cohen's dz of the household improvements; null when every household "
        "improved by the same non-zero amount."
    ),
    "wins, losses": (
        "Households where the model did better, or worse, than the reference."
    ),
    "mcse": "Standard error of mean_difference across households.",
}


@dataclass(frozen=True)
class HouseholdSplit:
    """Which households train, which inform selection, and which are held out.

    Attributes
    ----------
    name
        Label recorded with every result, such as ``"development-11-11"``.
    train
        Households whose labelled rows models are fitted on.
    test
        Held-out households. Models only ever predict on them, and only these
        are scored.
    development
        Optional households whose labelled rows models may use for their own
        selection, such as early stopping or choosing a hyperparameter. Never
        scored.

    Each role is stored sorted, and no household may hold two roles.
    """

    name: str
    train: tuple[str, ...]
    test: tuple[str, ...]
    development: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate roles and store them in canonical order."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a household split needs a name")
        seen: dict[str, str] = {}
        for role in ("train", "development", "test"):
            households = tuple(getattr(self, role))
            for household in households:
                if not isinstance(household, str) or not household.strip():
                    raise ValueError("household identifiers must be non-empty strings")
                if household in seen:
                    raise ValueError(
                        f"household {household!r} is in both {seen[household]} "
                        f"and {role}"
                    )
                seen[household] = role
            object.__setattr__(self, role, tuple(sorted(households)))
        if not self.train:
            raise ValueError("a split needs at least one training household")
        if not self.test:
            raise ValueError("a split needs at least one held-out household")

    @property
    def households(self) -> tuple[str, ...]:
        """Every household in the split, sorted."""
        return tuple(sorted((*self.train, *self.development, *self.test)))

    def role_of(self, household: str) -> str:
        """Return ``"train"``, ``"development"`` or ``"test"``."""
        for role in ("train", "development", "test"):
            if household in getattr(self, role):
                return role
        raise KeyError(household)

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serialisable declaration."""
        return {
            "name": self.name,
            "train": list(self.train),
            "development": list(self.development),
            "test": list(self.test),
        }

    def sha256(self) -> str:
        """Return the SHA-256 of the canonical serialised declaration."""
        return _digest(self.to_dict())


@dataclass(frozen=True, eq=False)
class FeatureRows:
    """Rows a model may predict from: permitted features only.

    Attributes
    ----------
    columns
        Feature names, from the run's information set.
    values
        Read-only ``(rows, columns)`` array. NaN means the information does
        not exist; see ``docs/INFORMATION_SETS.md``.
    states
        The label space. Probabilities must be reported in this column order.
    """

    columns: tuple[str, ...]
    values: np.ndarray
    states: tuple[BehaviouralState, ...]

    def __post_init__(self) -> None:
        """Freeze a copy of the values and check they match the columns."""
        values = np.array(self.values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.columns):
            raise ValueError(
                f"values have shape {values.shape}; expected {len(self.columns)} "
                "columns"
            )
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def __len__(self) -> int:
        """Number of rows."""
        return int(self.values.shape[0])


@dataclass(frozen=True, eq=False)
class LabelledRows(FeatureRows):
    """Rows a model may be fitted on: features, labels and household of origin.

    The household of each row is provided so that a model can select
    hyperparameters by household-grouped validation instead of leaking one
    household across folds.
    """

    labels: tuple[BehaviouralState, ...]
    households: tuple[str, ...]

    def __post_init__(self) -> None:
        """Check labels and households align with the rows."""
        super().__post_init__()
        if len(self.labels) != len(self) or len(self.households) != len(self):
            raise ValueError("labels and households must have one entry per row")


@dataclass(frozen=True, eq=False)
class StatePredictions:
    """What a model reports for each row it was given.

    Attributes
    ----------
    labels
        Reported state per row. ``UNKNOWN`` is an abstention and is never
        correct.
    probabilities
        Optional ``(rows, states)`` array with columns in the order of
        :attr:`FeatureRows.states`. Log loss, Brier score and calibration are
        reported only when every prediction carries one.
    """

    labels: tuple[BehaviouralState, ...]
    probabilities: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Normalise the labels and probabilities."""
        object.__setattr__(
            self, "labels", tuple(BehaviouralState(label) for label in self.labels)
        )
        if self.probabilities is not None:
            object.__setattr__(
                self, "probabilities", np.asarray(self.probabilities, dtype=float)
            )


@runtime_checkable
class StateModel(Protocol):
    """What the runner needs from a model.

    ``predict`` must be a deterministic function of each row alone. The runner
    checks this and refuses a model that violates it.
    """

    def fit(self, training: LabelledRows, development: LabelledRows | None) -> None:
        """Fit on training rows, optionally using development rows for selection."""

    def predict(self, rows: FeatureRows) -> StatePredictions:
        """Report a state, and optionally probabilities, for every row."""


@dataclass(frozen=True)
class ModelSpec:
    """A named model, how to build it, and the configuration to record.

    Attributes
    ----------
    name
        Unique name within a run.
    build
        Returns a fresh, unfitted model given the run's seed. The runner builds
        each model itself, so no model arrives already fitted on something else.
    configuration
        Every setting that affects the model, recorded as provenance. Must be
        JSON-serialisable.
    """

    name: str
    build: Callable[[int], StateModel]
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the name and snapshot the configuration."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a model needs a name")
        try:
            snapshot = json.loads(
                json.dumps(dict(self.configuration), sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"configuration of model {self.name!r} must be JSON-serialisable"
            ) from exc
        object.__setattr__(self, "configuration", snapshot)

    def to_dict(self) -> dict[str, object]:
        """Return the model's provenance."""
        # A functools.partial or callable instance has no qualified name.
        source = self.build if hasattr(self.build, "__qualname__") else type(self.build)
        return {
            "name": self.name,
            "build": f"{source.__module__}.{source.__qualname__}",
            "configuration": dict(self.configuration),
        }


@dataclass(frozen=True)
class PairedComparison:
    """One model against a reference on one metric, paired by household.

    ``difference`` is ``None`` when the comparison could not be made, and
    ``skipped`` then says why.
    """

    model: str
    reference: str
    metric: str
    households: tuple[str, ...]
    difference: PairedDifference | None
    skipped: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form."""
        higher = COMPARABLE_METRICS[self.metric]
        payload: dict[str, object] = {
            "model": self.model,
            "reference": self.reference,
            "metric": self.metric,
            "higher_is_better": higher,
            "improvement": ("model - reference" if higher else "reference - model"),
            "households": list(self.households),
        }
        if self.difference is None:
            payload["skipped"] = self.skipped
        else:
            payload.update(self.difference.to_dict())
        return payload


@dataclass(frozen=True)
class MatchedEvaluation:
    """Everything a matched run produced.

    Attributes
    ----------
    record
        The full provenance record, as written to disk.
    household_metrics
        Metrics per model, then per scored held-out household.
    comparisons
        Paired household-level comparisons between every pair of models.
    path
        Where the record was written, if an output directory was given.
    """

    record: ExperimentRecord
    household_metrics: Mapping[str, Mapping[str, PredictionMetrics]]
    comparisons: tuple[PairedComparison, ...]
    path: Path | None = None


@dataclass(frozen=True)
class _Household:
    """One household's table, labels and role, before any model sees it."""

    name: str
    role: str
    table: FeatureTable
    labelled: np.ndarray
    labels: tuple[BehaviouralState, ...]

    @classmethod
    def build(
        cls, name: str, role: str, table: FeatureTable, recording: CasasRecording
    ) -> _Household:
        """Label the table's moments and keep only the positions that have one."""
        truth = truth_series(recording.activities, list(table.moments))
        rows = [row for row, label in enumerate(truth) if label is not None]
        labels = tuple(label for label in truth if label is not None)
        return cls(name, role, table, np.array(rows, dtype=int), labels)

    def summary(self) -> dict[str, object]:
        """Return what was built for this household."""
        return {
            "role": self.role,
            "moments": len(self.table.moments),
            "labelled": int(self.labelled.size),
            "moments_sha256": hashlib.sha256(
                "\n".join(m.isoformat() for m in self.table.moments).encode("utf-8")
            ).hexdigest(),
            "uninstrumented": [c.name for c in self.table.uninstrumented],
            "excluded_sensors": list(self.table.excluded_sensors),
        }


def _digest(payload: object) -> str:
    """Return the SHA-256 of canonical JSON."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _regular_moments(recording: CasasRecording, step: timedelta) -> list[datetime]:
    """One moment per step from the first observation to the last.

    The anchor matches the online pipeline, which starts its grid at the first
    observation.
    """
    stamps = [observation.timestamp for observation in recording.observations]
    if not stamps:
        return []
    first, last = min(stamps), max(stamps)
    return [first + step * k for k in range((last - first) // step + 1)]


def _labelled_rows(
    households: Sequence[_Household],
    columns: tuple[str, ...],
    states: tuple[BehaviouralState, ...],
    rng: np.random.Generator,
) -> LabelledRows:
    """Pool labelled rows from *households* and present them in seeded order."""
    values: list[np.ndarray] = []
    labels: list[BehaviouralState] = []
    origin: list[str] = []
    for household in households:
        values.append(household.table.values[household.labelled])
        labels.extend(household.labels)
        origin.extend(household.name for _ in household.labels)
    stacked = np.vstack(values) if values else np.empty((0, len(columns)))
    order = rng.permutation(len(labels))
    return LabelledRows(
        columns=columns,
        values=stacked[order],
        states=states,
        labels=tuple(labels[i] for i in order),
        households=tuple(origin[i] for i in order),
    )


def _checked(
    predictions: object, name: str, rows: int, states: tuple[BehaviouralState, ...]
) -> StatePredictions:
    """Validate a model's output, naming the model in any error."""
    if not isinstance(predictions, StatePredictions):
        raise TypeError(f"model {name!r} must return StatePredictions")
    if len(predictions.labels) != rows:
        raise ValueError(
            f"model {name!r} returned {len(predictions.labels)} labels for {rows} rows"
        )
    allowed = {*states, BehaviouralState.UNKNOWN}
    if any(label not in allowed for label in predictions.labels):
        raise ValueError(f"model {name!r} reported a state outside the label space")
    probabilities = predictions.probabilities
    if probabilities is not None:
        if probabilities.shape != (rows, len(states)):
            raise ValueError(
                f"model {name!r} returned probabilities of shape "
                f"{probabilities.shape}; expected {(rows, len(states))}"
            )
        if not np.all(np.isfinite(probabilities)) or (probabilities < 0).any():
            raise ValueError(f"model {name!r} returned invalid probabilities")
        if rows and not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"model {name!r} returned probabilities not summing to 1")
    return predictions


def _predict(
    model: StateModel, name: str, rows: FeatureRows, check: np.ndarray
) -> StatePredictions:
    """Predict every row, then confirm each prediction depends on its row alone."""
    full = _checked(model.predict(rows), name, len(rows), rows.states)
    subset = FeatureRows(rows.columns, rows.values[check], rows.states)
    partial = _checked(model.predict(subset), name, len(subset), rows.states)

    same_labels = all(
        full.labels[row] == partial.labels[position]
        for position, row in enumerate(check)
    )
    if full.probabilities is None or partial.probabilities is None:
        same_probabilities = full.probabilities is partial.probabilities
    else:
        same_probabilities = bool(
            np.allclose(
                full.probabilities[check], partial.probabilities, rtol=1e-7, atol=1e-9
            )
        )
    if not (same_labels and same_probabilities):
        raise ValueError(
            f"model {name!r} changed its prediction for a row when other rows "
            "were removed or reordered; predictions must depend only on the "
            "row's own permitted features"
        )
    return full


def _finite(value: Any) -> Any:
    """Replace non-finite floats with ``None`` so the record is strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _compare(
    model: str,
    reference: str,
    metric: str,
    scores: Mapping[str, Mapping[str, PredictionMetrics]],
    households: tuple[str, ...],
    *,
    seed: int,
    resamples: int,
    confidence: float,
) -> PairedComparison:
    """Pair two models on one metric across the scored held-out households."""
    if metric in PROBABILISTIC_METRICS and not all(
        scores[name][household].probabilistic
        for name in (model, reference)
        for household in households
    ):
        return PairedComparison(
            model,
            reference,
            metric,
            households,
            None,
            "a model reported no probabilities",
        )
    if len(households) < 2:
        return PairedComparison(
            model,
            reference,
            metric,
            households,
            None,
            "fewer than two held-out households",
        )
    ours = [float(getattr(scores[model][h], metric)) for h in households]
    theirs = [float(getattr(scores[reference][h], metric)) for h in households]
    treatment, control = (
        (ours, theirs) if COMPARABLE_METRICS[metric] else (theirs, ours)
    )
    return PairedComparison(
        model,
        reference,
        metric,
        households,
        paired_difference(
            treatment, control, confidence=confidence, resamples=resamples, seed=seed
        ),
    )


def _summary(
    metrics: Sequence[str], scores: Mapping[str, PredictionMetrics]
) -> dict[str, object]:
    """Summarise household metrics, each household counted once."""
    summary: dict[str, object] = {}
    for metric in metrics:
        values = [
            float(value)
            for household in sorted(scores)
            if (value := getattr(scores[household], metric)) is not None
        ]
        if len(values) < len(scores):
            summary[metric] = None
            continue
        summary[metric] = {
            "median": float(np.median(values)),
            **summarise({metric: values})[metric],
        }
    return summary


def run_matched_evaluation(
    recordings: Mapping[str, CasasRecording],
    *,
    split: HouseholdSplit,
    information_set: InformationSet,
    models: Sequence[ModelSpec],
    seed: int,
    data_source: str,
    output_dir: Path | str | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    moments: Mapping[str, Sequence[datetime]] | None = None,
    experiment: str = "matched_evaluation",
    states: Sequence[BehaviouralState] = DEFAULT_STATES,
    resamples: int = 2000,
    confidence: float = 0.95,
) -> MatchedEvaluation:
    """Fit and score several models under one information set.

    Parameters
    ----------
    recordings
        Recordings by household. Only households named in *split* are read.
    split
        Training, development and held-out households.
    information_set
        The only information any model receives.
    models
        Models to compare, each built fresh with *seed*.
    seed
        Seeds model construction, row order, the row-independence check and
        the household bootstrap.
    data_source
        Where the recordings came from, such as ``"casas-hh"`` or
        ``"simulator"``. Recorded as provenance.
    output_dir
        If given, the record is written to ``<output_dir>/<experiment>.json``.
    metrics
        Metrics to summarise and compare; see :data:`COMPARABLE_METRICS`. Every
        per-household metric, including the confusion matrix and per-state
        recall, is always recorded.
    moments
        Prediction moments by household. Defaults to one per information-set
        step from each recording's first observation to its last. Pass the
        filter's scoring moments to align with it.
    experiment
        Name of the record.
    states
        The label space. Defaults to the ontology's states.
    resamples, confidence
        Household bootstrap settings for the paired comparisons.

    Raises
    ------
    ValueError
        If the specification is inconsistent, a model's output is malformed or
        depends on rows other than its own, or nothing held out is labelled.
    """
    space = _label_space(states)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not models:
        raise ValueError("at least one model is required")
    names = [spec.name for spec in models]
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique")
    metrics = tuple(metrics)
    unknown = sorted(set(metrics) - set(COMPARABLE_METRICS))
    if unknown or not metrics or len(set(metrics)) != len(metrics):
        raise ValueError(
            f"metrics must be distinct names from {sorted(COMPARABLE_METRICS)}; "
            f"got {list(metrics)}"
        )
    missing = sorted(set(split.households) - set(recordings))
    if missing:
        raise ValueError(f"no recording for split households {missing}")
    if moments is not None and set(split.households) - set(moments):
        raise ValueError(
            f"no moments for split households "
            f"{sorted(set(split.households) - set(moments))}"
        )

    households: dict[str, _Household] = {}
    for name in split.households:
        recording = recordings[name]
        grid = (
            moments[name]
            if moments is not None
            else _regular_moments(recording, information_set.resolution.step)
        )
        table = build_feature_table(recording, information_set, grid, household=name)
        household = _Household.build(name, split.role_of(name), table, recording)
        outside = set(household.labels) - set(space)
        if outside:
            raise ValueError(
                f"household {name!r} carries labels outside the label space: "
                f"{sorted(label.value for label in outside)}"
            )
        households[name] = household

    columns = information_set.columns
    rng = np.random.default_rng(seed)
    training = _labelled_rows(
        [households[name] for name in split.train], columns, space, rng
    )
    if not len(training):
        raise ValueError("no training household has a labelled moment")
    development: LabelledRows | None = None
    if split.development:
        development = _labelled_rows(
            [households[name] for name in split.development], columns, space, rng
        )
        if not len(development):
            raise ValueError("no development household has a labelled moment")

    scored = tuple(name for name in split.test if households[name].labelled.size)
    if not scored:
        raise ValueError("no held-out household has a labelled moment")
    blocks = [
        households[name].table.values[households[name].labelled] for name in scored
    ]
    sizes = [block.shape[0] for block in blocks]
    order = rng.permutation(sum(sizes))
    held_out = FeatureRows(columns, np.vstack(blocks)[order], space)
    check = rng.permutation(len(held_out))[: max(1, len(held_out) // 2)]
    restore = np.argsort(order)

    household_metrics: dict[str, dict[str, PredictionMetrics]] = {}
    for spec in models:
        model = spec.build(seed)
        if not isinstance(model, StateModel):
            raise TypeError(f"model {spec.name!r} must provide fit and predict")
        model.fit(training, development)
        predicted = _predict(model, spec.name, held_out, check)
        labels = [predicted.labels[i] for i in restore]
        probabilities = (
            predicted.probabilities[restore]
            if predicted.probabilities is not None
            else None
        )
        per_household: dict[str, PredictionMetrics] = {}
        start = 0
        for name, size in zip(scored, sizes):
            household = households[name]
            per_household[name] = prediction_metrics(
                household.labels,
                labels[start : start + size],
                (
                    probabilities[start : start + size]
                    if probabilities is not None
                    else None
                ),
                states=space,
            )
            start += size
        household_metrics[spec.name] = per_household

    comparisons = tuple(
        _compare(
            names[later],
            names[earlier],
            metric,
            household_metrics,
            scored,
            seed=seed,
            resamples=resamples,
            confidence=confidence,
        )
        for earlier in range(len(names))
        for later in range(earlier + 1, len(names))
        for metric in metrics
    )

    configuration = {
        "result_schema": RESULT_SCHEMA,
        "information_set": {
            **information_set.to_dict(),
            "sha256": information_set.sha256(),
        },
        "split": {**split.to_dict(), "sha256": split.sha256()},
        "models": [spec.to_dict() for spec in models],
        "states": [state.value for state in space],
        "metrics": list(metrics),
        "moments": (
            "supplied by caller"
            if moments is not None
            else "one per step from each recording's first observation"
        ),
        "row_order": "seeded random permutation; no household or time at predict",
        "row_independence_check": (
            "held-out rows predicted again as a random half in a new order"
        ),
        "bootstrap": {
            "unit": "household",
            "statistic": "mean paired improvement",
            "resamples": resamples,
            "confidence": confidence,
        },
    }
    results = {
        "households": {
            name: household.summary() for name, household in households.items()
        },
        "unscored_households": sorted(set(split.test) - set(scored)),
        "models": {
            spec.name: {
                "households": {
                    name: scores.to_dict()
                    for name, scores in household_metrics[spec.name].items()
                },
                "summary": _summary(metrics, household_metrics[spec.name]),
                "confusion_summed": ConfusionMatrix.total(
                    [m.confusion for m in household_metrics[spec.name].values()]
                ).to_dict(),
            }
            for spec in models
        },
        "comparisons": [comparison.to_dict() for comparison in comparisons],
    }
    record = ExperimentRecord(
        experiment=experiment,
        configuration=configuration,
        seeds=[seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions=MATCHED_METRIC_DEFINITIONS,
        notes=[
            "Every model received the same feature rows built from one "
            "information set; comparisons are paired by held-out household.",
        ],
    )
    path = (
        record.write(Path(output_dir) / f"{experiment}.json")
        if output_dir is not None
        else None
    )
    return MatchedEvaluation(
        record=record,
        household_metrics=household_metrics,
        comparisons=comparisons,
        path=path,
    )
