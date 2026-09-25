"""Pre-declared reference models for the Phase 2 baseline benchmark.

Phase 2 of the roadmap sets up baselines before any new model is proposed.
These four are the simplest defensible members of that benchmark. Each one
consumes exactly the rows the matched evaluation runner builds from a declared
information set, so it is scored on the same information as every other model.
None of them is proposed as a new model.

- :class:`StateFrequencyBaseline` uses the training labels only and reports
  the training state frequencies for every row.
- :class:`PersistenceBaseline` uses the previous window's evidence and reports
  its inner model's probabilities for that window.
- :class:`LogisticBaseline` is regularised multinomial logistic regression on
  every permitted column.
- :class:`TreeBaseline` is one depth-limited decision tree on every permitted
  column; its probabilities are leaf frequencies.

Shared conventions
------------------
- Fitting uses the training rows only. No baseline reads the development rows.
  Every setting is declared in advance, so there is nothing to select.
- Probabilities cover every state in the label space, in its order. A state
  with no training rows gets probability zero and is never predicted.
- When two states are equally probable, the one earlier in the label space is
  reported.
- Missing evidence is encoded, not imputed away. Each evidence column
  contributes its value, with NaN replaced by zero, and a separate indicator
  that the value was missing. An absent sensor therefore never looks like a
  silent one. The encoding uses no statistics from any data.
- A fitted model refuses rows whose columns differ from those it was fitted on,
  so a model fitted under one information set cannot be scored under another.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ..states.ontology import BehaviouralState
from .information_sets import (
    HOUR_OF_DAY,
    InformationComponent,
    InformationSet,
    evidence_column,
    parse_evidence_column,
)
from .matched_evaluation import FeatureRows, LabelledRows, ModelSpec, StatePredictions

#: Levels of the hour-of-day column.
HOURS = 24


def encode_rows(rows: FeatureRows, *, one_hot_hour: bool) -> np.ndarray:
    """Turn permitted columns into a complete numeric matrix, row by row.

    Each evidence column becomes two: its value with NaN replaced by zero, and
    an indicator that the value was missing. ``hour_of_day`` becomes 24
    indicators when *one_hot_hour* is set, and stays a single ordinal column
    otherwise. Both encodings carry exactly the information in the row.
    """
    blocks: list[np.ndarray] = []
    for position, name in enumerate(rows.columns):
        column = rows.values[:, position]
        if name == HOUR_OF_DAY:
            blocks.append(
                (column[:, None] == np.arange(HOURS)[None, :])
                if one_hot_hour
                else column[:, None]
            )
            continue
        missing = np.isnan(column)
        blocks.append(np.where(missing, 0.0, column)[:, None])
        blocks.append(missing[:, None])
    if not blocks:
        return np.empty((len(rows), 0))
    return np.hstack([block.astype(float) for block in blocks])


def _label_codes(training: LabelledRows) -> np.ndarray:
    """Position of each training label in the label space."""
    index = {state: position for position, state in enumerate(training.states)}
    return np.array([index[label] for label in training.labels], dtype=int)


def _expand(classes: np.ndarray, scores: np.ndarray, size: int) -> np.ndarray:
    """Place a classifier's probabilities in label-space columns."""
    full = np.zeros((scores.shape[0], size))
    full[:, np.asarray(classes, dtype=int)] = scores
    return full


class Baseline(ABC):
    """The prediction interface shared by every baseline.

    A baseline reports a probability for every state in the label space for
    every row, and derives its reported state from those probabilities. It
    implements the runner's :class:`~sensor_modeling.datasets.StateModel`
    protocol.
    """

    def __init__(self) -> None:
        self._states: tuple[BehaviouralState, ...] | None = None
        self._columns: tuple[str, ...] | None = None

    def fit(self, training: LabelledRows, development: LabelledRows | None) -> None:
        """Fit on *training*. *development* is accepted and never read."""
        if not len(training):
            raise ValueError(f"{type(self).__name__} needs at least one training row")
        self._states = training.states
        self._columns = training.columns
        self._fit(training)

    def predict_proba(self, rows: FeatureRows) -> np.ndarray:
        """Return ``(rows, states)`` probabilities in label-space order."""
        if self._states is None or self._columns is None:
            raise RuntimeError(f"{type(self).__name__} must be fitted before use")
        if rows.states != self._states:
            raise ValueError("rows use a different label space from the one fitted")
        if rows.columns != self._columns:
            raise ValueError(
                f"{type(self).__name__} was fitted on columns {list(self._columns)} "
                f"and refuses rows with columns {list(rows.columns)}; a model "
                "fitted under one information set cannot be scored under another"
            )
        probabilities = np.asarray(self._probabilities(rows), dtype=float)
        if probabilities.shape != (len(rows), len(rows.states)):
            raise RuntimeError("baseline produced probabilities of the wrong shape")
        if len(rows) and not (
            np.all(np.isfinite(probabilities))
            and (probabilities >= 0.0).all()
            and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
        ):
            raise RuntimeError(
                "baseline produced probabilities that are not normalised"
            )
        return probabilities

    def predict(self, rows: FeatureRows) -> StatePredictions:
        """Report the most probable state and the full probability vector."""
        probabilities = self.predict_proba(rows)
        labels = tuple(rows.states[int(i)] for i in np.argmax(probabilities, axis=1))
        return StatePredictions(labels, probabilities)

    @abstractmethod
    def configuration(self) -> dict[str, object]:
        """Every setting except the seed, which the runner records itself."""

    @abstractmethod
    def _fit(self, training: LabelledRows) -> None:
        """Fit the model."""

    @abstractmethod
    def _probabilities(self, rows: FeatureRows) -> np.ndarray:
        """Return probabilities for rows already checked against the fit."""


class StateFrequencyBaseline(Baseline):
    """The training state frequencies, reported for every row.

    Its reported state is the majority training state. Its probabilities are
    the reference a model must improve on to show that it extracted any
    information from the rows at all.
    """

    def _fit(self, training: LabelledRows) -> None:
        counts = np.bincount(_label_codes(training), minlength=len(training.states))
        self._frequencies = counts / counts.sum()

    def _probabilities(self, rows: FeatureRows) -> np.ndarray:
        return np.tile(self._frequencies, (len(rows), 1))

    def configuration(self) -> dict[str, object]:
        """Every setting except the seed."""
        return {"model": "StateFrequencyBaseline", "smoothing": None}


class LogisticBaseline(Baseline):
    """L2-regularised multinomial logistic regression on standardised features.

    Parameters
    ----------
    C
        Inverse regularisation strength. Features are standardised on the
        training rows first, so the penalty weighs every feature alike.
    max_iter
        Iteration limit for the L-BFGS solver.
    seed
        Random state passed to the estimator.
    """

    def __init__(self, *, C: float = 1.0, max_iter: int = 1000, seed: int = 0) -> None:
        super().__init__()
        if not math.isfinite(C) or C <= 0.0:
            raise ValueError("C must be positive and finite")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.seed = int(seed)

    def _fit(self, training: LabelledRows) -> None:
        codes = _label_codes(training)
        if np.unique(codes).size < 2:
            raise ValueError(
                "logistic regression needs at least two states in the training rows"
            )
        self._model: Pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.C, max_iter=self.max_iter, random_state=self.seed
            ),
        )
        self._model.fit(encode_rows(training, one_hot_hour=True), codes)

    def _probabilities(self, rows: FeatureRows) -> np.ndarray:
        scores = self._model.predict_proba(encode_rows(rows, one_hot_hour=True))
        return _expand(self._model.classes_, scores, len(rows.states))

    def configuration(self) -> dict[str, object]:
        """Every setting except the seed."""
        return {
            "model": "LogisticBaseline",
            "penalty": "l2",
            "C": self.C,
            "solver": "lbfgs",
            "max_iter": self.max_iter,
            "standardised": True,
            "hour_of_day": "one-hot",
            "missing": "zero plus indicator",
        }


class TreeBaseline(Baseline):
    """One depth-limited decision tree; probabilities are leaf frequencies.

    Parameters
    ----------
    max_depth, min_samples_leaf
        Fixed complexity limits. They are declared, not searched.
    seed
        Random state passed to the estimator, fixing how ties between equally
        good splits are broken.
    """

    def __init__(
        self, *, max_depth: int = 6, min_samples_leaf: int = 20, seed: int = 0
    ) -> None:
        super().__init__()
        if max_depth < 1 or min_samples_leaf < 1:
            raise ValueError("max_depth and min_samples_leaf must be at least 1")
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.seed = int(seed)

    def _fit(self, training: LabelledRows) -> None:
        self._model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
        )
        self._model.fit(
            encode_rows(training, one_hot_hour=False), _label_codes(training)
        )

    def _probabilities(self, rows: FeatureRows) -> np.ndarray:
        scores = self._model.predict_proba(encode_rows(rows, one_hot_hour=False))
        return _expand(self._model.classes_, scores, len(rows.states))

    def configuration(self) -> dict[str, object]:
        """Every setting except the seed."""
        return {
            "model": "TreeBaseline",
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "hour_of_day": "ordinal",
            "missing": "zero plus indicator",
        }


class PersistenceBaseline(Baseline):
    """The state inferred for the previous window, carried forward.

    The protocol never shows a model the true state of a held-out household,
    so the only previous state available is one inferred from the previous
    window's evidence. The inner model is fitted to map one window's
    per-channel evidence to its state, using the current window of every
    training row. At prediction it is applied to the previous window instead,
    and its probabilities are reported for the current moment. The one
    assumption this adds is that the state has not changed since the previous
    window. Time of day and older history are not used.

    At the first timestamp of a recording the previous window precedes the
    recording, so every previous-window column is NaN and no previous state
    exists. There, and in any other row with no previous-window evidence at
    all, the training state frequencies are reported.

    Parameters
    ----------
    inner
        Maps one window's evidence to state probabilities. Defaults to
        :class:`LogisticBaseline` with its default settings.

    Raises
    ------
    ValueError
        At fit, if the rows carry no previous-window evidence, which means the
        information set does not include recent history.
    """

    def __init__(self, inner: Baseline | None = None) -> None:
        super().__init__()
        self.inner = inner if inner is not None else LogisticBaseline()

    def _fit(self, training: LabelledRows) -> None:
        lags: dict[str, dict[int, int]] = {}
        for position, name in enumerate(training.columns):
            parsed = parse_evidence_column(name)
            if parsed is not None:
                lags.setdefault(parsed[0], {})[parsed[1]] = position
        channels = [c for c in lags if 0 in lags[c] and 1 in lags[c]]
        if not channels:
            raise ValueError(
                "persistence needs the previous window's evidence; use an "
                "information set that includes recent history"
            )
        self._window = tuple(evidence_column(channel, 0) for channel in channels)
        self._current = [lags[channel][0] for channel in channels]
        self._previous = [lags[channel][1] for channel in channels]

        self._prior = StateFrequencyBaseline()
        self._prior.fit(training, None)
        self.inner.fit(
            LabelledRows(
                columns=self._window,
                values=training.values[:, self._current],
                states=training.states,
                labels=training.labels,
                households=training.households,
            ),
            None,
        )

    def _probabilities(self, rows: FeatureRows) -> np.ndarray:
        previous = rows.values[:, self._previous]
        probabilities = self.inner.predict_proba(
            FeatureRows(self._window, previous, rows.states)
        ).copy()
        unobserved = np.isnan(previous).all(axis=1)
        if unobserved.any():
            probabilities[unobserved] = self._prior.predict_proba(rows)[unobserved]
        return probabilities

    def configuration(self) -> dict[str, object]:
        """Every setting except the seed."""
        return {
            "model": "PersistenceBaseline",
            "carried_from": "previous window (lag 1)",
            "without_previous_window": "training state frequencies",
            "inner": self.inner.configuration(),
        }


def _build_state_frequency(seed: int) -> StateFrequencyBaseline:
    return StateFrequencyBaseline()


def _build_persistence(seed: int) -> PersistenceBaseline:
    return PersistenceBaseline(LogisticBaseline(seed=seed))


def _build_logistic(seed: int) -> LogisticBaseline:
    return LogisticBaseline(seed=seed)


def _build_tree(seed: int) -> TreeBaseline:
    return TreeBaseline(seed=seed)


def baseline_suite(information_set: InformationSet) -> tuple[ModelSpec, ...]:
    """Return the pre-declared baselines applicable to *information_set*.

    The state-frequency baseline comes first, so the runner compares every
    other baseline against it. Persistence is included only when the set
    includes recent history, since it has nothing to carry forward otherwise.
    Every setting is fixed here, before any household is scored, and none is
    tuned.
    """
    seeded = {"random_state": "run seed"}
    specs = [
        ModelSpec(
            "state_frequency",
            _build_state_frequency,
            StateFrequencyBaseline().configuration(),
        )
    ]
    if InformationComponent.RECENT_HISTORY in information_set.components:
        specs.append(
            ModelSpec(
                "persistence",
                _build_persistence,
                {**PersistenceBaseline().configuration(), **seeded},
            )
        )
    specs.append(
        ModelSpec(
            "logistic",
            _build_logistic,
            {**LogisticBaseline().configuration(), **seeded},
        )
    )
    specs.append(
        ModelSpec("tree", _build_tree, {**TreeBaseline().configuration(), **seeded})
    )
    return tuple(specs)
