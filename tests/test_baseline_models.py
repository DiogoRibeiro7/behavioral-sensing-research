"""Tests for the Phase 2 baseline models.

The label space used here is deliberately not in alphabetical order, so a
baseline that reported probabilities in its classifier's own class order would
put them in the wrong columns.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import (
    ActivityInterval,
    Baseline,
    CasasRecording,
    FeatureRows,
    HouseholdSplit,
    LabelledRows,
    LogisticBaseline,
    ModelSpec,
    PersistenceBaseline,
    StateFrequencyBaseline,
    StateModel,
    StatePredictions,
    TreeBaseline,
    baseline_suite,
    nested_information_sets,
    run_matched_evaluation,
)
from sensor_modeling.datasets.baseline_models import encode_rows
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S

STATES = (S.SLEEPING, S.AWAY, S.KITCHEN_ACTIVITY, S.HOME_ACTIVE, S.BED_AWAKE)
COLUMNS = (
    "events_bedroom_motion_lag0",
    "events_kitchen_motion_lag0",
    "hour_of_day",
    "events_bedroom_motion_lag1",
    "events_kitchen_motion_lag1",
)
#: States present in the synthetic training rows. AWAY and BED_AWAKE are not.
SEEN = (S.SLEEPING, S.KITCHEN_ACTIVITY, S.HOME_ACTIVE)

FACTORIES: dict[str, Callable[[], Baseline]] = {
    "state_frequency": StateFrequencyBaseline,
    "persistence": PersistenceBaseline,
    "logistic": LogisticBaseline,
    "tree": lambda: TreeBaseline(min_samples_leaf=5),
}


def synthetic(
    n: int = 300, *, seed: int = 0, states: Sequence[S] = SEEN
) -> tuple[np.ndarray, tuple[S, ...]]:
    """Rows whose bedroom and kitchen counts, now and one window back, reveal the state."""
    rng = np.random.default_rng(seed)
    labels = tuple(states[i] for i in rng.integers(0, len(states), n))
    bedroom = {S.SLEEPING: 1.0, S.KITCHEN_ACTIVITY: 0.0, S.HOME_ACTIVE: 2.0}
    kitchen = {S.SLEEPING: 0.0, S.KITCHEN_ACTIVITY: 8.0, S.HOME_ACTIVE: 1.0}
    values = np.empty((n, len(COLUMNS)))
    for row, label in enumerate(labels):
        b, k = bedroom.get(label, 3.0), kitchen.get(label, 3.0)
        values[row] = np.array(
            [
                rng.poisson(b),
                rng.poisson(k),
                rng.integers(0, 24),
                rng.poisson(b),
                rng.poisson(k),
            ],
            dtype=float,
        )
    return values, labels


def labelled(
    values: np.ndarray,
    labels: Sequence[S],
    *,
    states: tuple[S, ...] = STATES,
    columns: tuple[str, ...] = COLUMNS,
) -> LabelledRows:
    return LabelledRows(
        columns=columns,
        values=values,
        states=states,
        labels=tuple(labels),
        households=tuple("train" for _ in labels),
    )


def rows(
    values: np.ndarray,
    *,
    states: tuple[S, ...] = STATES,
    columns: tuple[str, ...] = COLUMNS,
) -> FeatureRows:
    return FeatureRows(columns=columns, values=values, states=states)


def fitted(name: str, training: LabelledRows | None = None) -> Baseline:
    model = FACTORIES[name]()
    model.fit(training if training is not None else labelled(*synthetic()), None)
    return model


ALL = pytest.mark.parametrize("name", sorted(FACTORIES))


class TestInterface:
    @ALL
    def test_every_baseline_is_a_state_model(self, name: str) -> None:
        model = fitted(name)
        assert isinstance(model, StateModel)
        values, _ = synthetic(20, seed=1)
        predictions = model.predict(rows(values))
        assert isinstance(predictions, StatePredictions)
        assert predictions.probabilities is not None
        assert predictions.labels == tuple(
            STATES[i] for i in predictions.probabilities.argmax(axis=1)
        )

    @ALL
    def test_every_configuration_is_serialisable(self, name: str) -> None:
        assert json.loads(json.dumps(FACTORIES[name]().configuration()))

    @ALL
    def test_prediction_before_fitting_is_refused(self, name: str) -> None:
        with pytest.raises(RuntimeError, match="fitted"):
            FACTORIES[name]().predict(rows(synthetic(5)[0]))

    def test_ties_go_to_the_earlier_state_in_the_label_space(self) -> None:
        model = StateFrequencyBaseline()
        model.fit(labelled(np.zeros((2, 5)), [S.HOME_ACTIVE, S.SLEEPING]), None)
        assert model.predict(rows(np.zeros((1, 5)))).labels == (S.SLEEPING,)


class TestNoLeakage:
    @ALL
    def test_development_rows_are_never_read(self, name: str) -> None:
        training = labelled(*synthetic(seed=2))
        values, labels = synthetic(seed=3, states=(S.BED_AWAKE,))
        development = labelled(values, labels)
        alone, with_development = FACTORIES[name](), FACTORIES[name]()
        alone.fit(training, None)
        with_development.fit(training, development)
        probe = rows(synthetic(40, seed=4)[0])
        np.testing.assert_array_equal(
            alone.predict_proba(probe), with_development.predict_proba(probe)
        )

    def test_held_out_labels_cannot_change_any_prediction(self) -> None:
        homes = {f"sim{s}": simulated(s) for s in range(1, 5)}
        relabelled = {
            name: (
                recording
                if name in ("sim1", "sim2")
                else CasasRecording(
                    recording.registry,
                    recording.observations,
                    tuple(
                        ActivityInterval(i.label, i.start, i.end, S.SLEEPING)
                        for i in recording.activities
                    ),
                )
            )
            for name, recording in homes.items()
        }
        split = HouseholdSplit("s", train=("sim1", "sim2"), test=("sim3", "sim4"))
        information_set = nested_information_sets()[-1]
        seen: list[list[np.ndarray]] = [[], []]
        for log, recordings in zip(seen, (homes, relabelled)):
            specs = [
                spy_spec(spec.name, spec.build, log)
                for spec in baseline_suite(information_set)
            ]
            run_matched_evaluation(
                recordings,
                split=split,
                information_set=information_set,
                models=specs,
                seed=0,
                data_source="simulator",
            )
        assert len(seen[0]) == len(seen[1]) > 0
        for original, changed in zip(*seen):
            np.testing.assert_array_equal(original, changed)


class TestDeterminism:
    @ALL
    def test_the_same_fit_gives_the_same_probabilities(self, name: str) -> None:
        probe = rows(synthetic(50, seed=5)[0])
        np.testing.assert_array_equal(
            fitted(name).predict_proba(probe), fitted(name).predict_proba(probe)
        )

    @ALL
    def test_training_row_order_does_not_matter(self, name: str) -> None:
        values, labels = synthetic(seed=6)
        order = np.random.default_rng(0).permutation(len(labels))
        shuffled = labelled(values[order], [labels[i] for i in order])
        probe = rows(synthetic(50, seed=7)[0])
        np.testing.assert_allclose(
            fitted(name, labelled(values, labels)).predict_proba(probe),
            fitted(name, shuffled).predict_proba(probe),
            atol=1e-6,
        )


class TestProbabilities:
    @ALL
    def test_rows_are_normalised_even_with_missing_evidence(self, name: str) -> None:
        values = synthetic(60, seed=8)[0]
        values[::3, 0] = np.nan
        values[::5, 3:] = np.nan
        probabilities = fitted(name).predict_proba(rows(values))
        assert np.all(np.isfinite(probabilities)) and (probabilities >= 0).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)

    @ALL
    def test_columns_follow_the_label_space_not_the_classifier(self, name: str) -> None:
        values, labels = synthetic(seed=9)
        reverse = tuple(reversed(STATES))
        forward = fitted(name, labelled(values, labels))
        backward = fitted(name, labelled(values, labels, states=reverse))
        probe = synthetic(40, seed=10)[0]
        np.testing.assert_allclose(
            forward.predict_proba(rows(probe)),
            backward.predict_proba(rows(probe, states=reverse))[:, ::-1],
            atol=1e-3,
        )

    def test_frequencies_land_in_their_own_columns(self) -> None:
        model = StateFrequencyBaseline()
        model.fit(
            labelled(
                np.zeros((4, 5)),
                [S.HOME_ACTIVE, S.HOME_ACTIVE, S.HOME_ACTIVE, S.SLEEPING],
            ),
            None,
        )
        # Add-one over five states: (count + 1) / (4 + 5).
        expected = {S.HOME_ACTIVE: 4 / 9, S.SLEEPING: 2 / 9}
        probabilities = model.predict_proba(rows(np.zeros((1, 5))))[0]
        for column, state in enumerate(STATES):
            assert probabilities[column] == pytest.approx(expected.get(state, 1 / 9))

    @ALL
    def test_no_probability_is_ever_zero(self, name: str) -> None:
        values = synthetic(120, seed=20)[0]
        values[::4, 3:] = np.nan
        assert (fitted(name).predict_proba(rows(values)) > 0).all()


class TestUnseenAndRareStates:
    @ALL
    def test_states_without_training_rows_are_never_reported_or_ruled_out(
        self, name: str
    ) -> None:
        predictions = fitted(name).predict(rows(synthetic(80, seed=11)[0]))
        assert predictions.probabilities is not None
        for state in (S.AWAY, S.BED_AWAKE):
            assert (predictions.probabilities[:, STATES.index(state)] > 0).all()
            assert state not in predictions.labels

    @pytest.mark.parametrize("name", ["state_frequency", "persistence", "logistic"])
    def test_states_without_training_rows_get_the_add_one_probability(
        self, name: str
    ) -> None:
        values = synthetic(80, seed=21)[0]
        values[:5, 3:] = np.nan  # first timestamps, for persistence
        probabilities = fitted(name).predict_proba(rows(values))
        add_one = 1 / (300 + len(STATES))
        for state in (S.AWAY, S.BED_AWAKE):
            np.testing.assert_allclose(probabilities[:, STATES.index(state)], add_one)

    def test_logistic_keeps_its_odds_among_the_states_it_represents(self) -> None:
        values, labels = synthetic(seed=22)
        full = fitted("logistic", labelled(values, labels))
        seen = fitted("logistic", labelled(values, labels, states=SEEN))
        probe = synthetic(40, seed=23)[0]
        represented = [STATES.index(state) for state in SEEN]
        np.testing.assert_allclose(
            full.predict_proba(rows(probe))[:, represented] / (1 - 2 / 305),
            seen.predict_proba(rows(probe, states=SEEN)),
            atol=1e-9,
        )

    def test_tree_leaves_are_laplace_corrected(self) -> None:
        values = np.zeros((20, 5))
        values[:10, 1] = 8.0
        labels = [S.KITCHEN_ACTIVITY] * 10 + [S.SLEEPING] * 10
        tree = TreeBaseline(max_depth=1, min_samples_leaf=1)
        tree.fit(labelled(values, labels), None)
        probabilities = tree.predict_proba(rows(values[[0, 19]]))
        # Two pure leaves of ten rows: (10 + 1) / (10 + 5) and 1 / 15 elsewhere.
        kitchen, sleeping = STATES.index(S.KITCHEN_ACTIVITY), STATES.index(S.SLEEPING)
        assert probabilities[0, kitchen] == pytest.approx(11 / 15)
        assert probabilities[1, sleeping] == pytest.approx(11 / 15)
        assert probabilities[0, sleeping] == pytest.approx(1 / 15)

    @ALL
    def test_a_state_with_a_single_training_row_is_handled(self, name: str) -> None:
        values, labels = synthetic(seed=12)
        values = np.vstack([values, [[3, 3, 12, 3, 3]]])
        training = labelled(values, [*labels, S.BED_AWAKE])
        probabilities = fitted(name, training).predict_proba(rows(values))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)

    def test_logistic_regression_needs_two_training_states(self) -> None:
        values, _ = synthetic(10)
        with pytest.raises(ValueError, match="two states"):
            LogisticBaseline().fit(labelled(values, [S.SLEEPING] * 10), None)

    @pytest.mark.parametrize("name", ["state_frequency", "tree"])
    def test_one_training_state_is_reported_but_not_with_certainty(
        self, name: str
    ) -> None:
        values, _ = synthetic(10)
        model = fitted(name, labelled(values, [S.SLEEPING] * 10))
        predictions = model.predict(rows(values))
        assert set(predictions.labels) == {S.SLEEPING}
        assert predictions.probabilities is not None
        np.testing.assert_allclose(
            predictions.probabilities[:, STATES.index(S.SLEEPING)], 11 / 15
        )


class TestMissingFeatures:
    def test_missing_is_encoded_apart_from_zero(self) -> None:
        silent = rows(np.array([[0.0, 0.0, 3.0, 0.0, 0.0]]))
        absent = rows(np.array([[np.nan, 0.0, 3.0, 0.0, 0.0]]))
        assert not np.array_equal(
            encode_rows(silent, one_hot_hour=True),
            encode_rows(absent, one_hot_hour=True),
        )

    def test_hour_is_one_hot_for_the_linear_model_and_ordinal_for_the_tree(
        self,
    ) -> None:
        sample = rows(np.array([[1.0, 2.0, 23.0, 0.0, 0.0]]))
        one_hot = encode_rows(sample, one_hot_hour=True)
        ordinal = encode_rows(sample, one_hot_hour=False)
        assert one_hot.shape == (1, 4 * 2 + 24) and one_hot[0, 4 + 23] == 1.0
        assert ordinal.shape == (1, 4 * 2 + 1) and ordinal[0, 4] == 23.0

    @ALL
    def test_a_channel_missing_throughout_training_is_tolerated(
        self, name: str
    ) -> None:
        values, labels = synthetic(seed=13)
        values[:, [0, 3]] = np.nan
        model = fitted(name, labelled(values, labels))
        probe = synthetic(20, seed=14)[0]
        probabilities = model.predict_proba(rows(probe))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


class TestPersistence:
    def test_the_first_timestamp_reports_the_training_frequencies(self) -> None:
        training = labelled(*synthetic(seed=15))
        model = fitted("persistence", training)
        prior = fitted("state_frequency", training)
        first = synthetic(4, seed=16)[0]
        first[:, 3:] = np.nan
        np.testing.assert_array_equal(
            model.predict_proba(rows(first)), prior.predict_proba(rows(first))
        )

    def test_it_carries_the_state_inferred_for_the_previous_window(self) -> None:
        values, labels = synthetic(seed=17)
        model = fitted("persistence", labelled(values, labels))
        window = ("events_bedroom_motion_lag0", "events_kitchen_motion_lag0")
        inner = LogisticBaseline()
        inner.fit(labelled(values[:, :2], labels, columns=window), None)
        probe = synthetic(30, seed=18)[0]
        np.testing.assert_allclose(
            model.predict_proba(rows(probe)),
            inner.predict_proba(rows(probe[:, 3:], columns=window)),
        )

    def test_it_ignores_the_current_window_and_the_hour(self) -> None:
        model = fitted("persistence")
        probe = synthetic(30, seed=19)[0]
        altered = probe.copy()
        altered[:, :3] = np.random.default_rng(0).integers(0, 20, (30, 3))
        np.testing.assert_array_equal(
            model.predict_proba(rows(probe)), model.predict_proba(rows(altered))
        )

    def test_it_needs_recent_history(self) -> None:
        values, labels = synthetic()
        with pytest.raises(ValueError, match="recent history"):
            PersistenceBaseline().fit(
                labelled(values[:, :3], labels, columns=COLUMNS[:3]), None
            )


class TestInformationSetConsumption:
    @ALL
    def test_rows_from_another_information_set_are_refused(self, name: str) -> None:
        model = fitted(name)
        with pytest.raises(ValueError, match="information set"):
            model.predict(rows(synthetic(5)[0][:, :3], columns=COLUMNS[:3]))
        with pytest.raises(ValueError, match="label space"):
            model.predict(rows(synthetic(5)[0], states=tuple(reversed(STATES))))

    @pytest.mark.parametrize("index", range(4))
    def test_the_suite_consumes_exactly_the_declared_set(self, index: int) -> None:
        information_set = nested_information_sets()[index]
        seen: list[Any] = []
        specs = [
            spy_spec(spec.name, spec.build, seen, columns=True)
            for spec in baseline_suite(information_set)
        ]
        homes = {f"sim{s}": simulated(s) for s in range(1, 4)}
        run_matched_evaluation(
            homes,
            split=HouseholdSplit("s", train=("sim1", "sim2"), test=("sim3",)),
            information_set=information_set,
            models=specs,
            seed=0,
            data_source="simulator",
        )
        assert seen and all(columns == information_set.columns for columns in seen)
        names = [spec.name for spec in specs]
        assert names[0] == "state_frequency"
        assert ("persistence" in names) == (index >= 2)


@dataclass
class Spy:
    """Records what a baseline was given, or what it predicted."""

    inner: Baseline
    log: list[Any]
    columns: bool = False

    def fit(self, training: LabelledRows, development: LabelledRows | None) -> None:
        if self.columns:
            self.log.append(training.columns)
        self.inner.fit(training, development)

    def predict(self, rows: FeatureRows) -> StatePredictions:
        predictions = self.inner.predict(rows)
        if self.columns:
            self.log.append(rows.columns)
        else:
            self.log.append(predictions.probabilities)
        return predictions


def spy_spec(
    name: str, build: Callable[[int], Any], log: list[Any], **options: Any
) -> ModelSpec:
    return ModelSpec(name, lambda seed: Spy(build(seed), log, **options))


def simulated(seed: int) -> CasasRecording:
    sim = simulate(HouseholdConfig(days=2, seed=seed))
    return CasasRecording(
        sim.registry,
        sim.observations,
        tuple(
            ActivityInterval(e.state.value, e.start, e.end, e.state)
            for e in sim.truth.episodes
        ),
    )


def test_the_documented_example_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The example in docs/BASELINES.md must stay executable."""
    page = Path(__file__).resolve().parents[1] / "docs" / "BASELINES.md"
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), re.S)
    assert len(blocks) == 1
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, Any] = {}
    exec(compile(blocks[0], str(page), "exec"), namespace)  # noqa: S102
    assert namespace["result"].comparisons
