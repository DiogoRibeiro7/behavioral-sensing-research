"""Tests for the matched-information evaluation runner.

The runner's value is in what it refuses to let happen: models seeing different
information, a model fitted on a held-out household, a prediction that uses
other rows, and statistics that treat pooled observations as independent. The
stub models here exist only to exercise those guarantees; none is a proposal.
"""

from __future__ import annotations

import functools
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    EvidenceChannel,
    EvidenceResolution,
    FeatureRows,
    FeatureTable,
    HouseholdSplit,
    InformationComponent,
    InformationSet,
    LabelledRows,
    MatchedEvaluation,
    ModelSpec,
    StatePredictions,
    build_feature_table,
    run_matched_evaluation,
    truth_series,
)
from sensor_modeling.evaluation import load_record, prediction_metrics
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
)
from sensor_modeling.states import BehaviouralState as S

UTC = timezone.utc
T0 = datetime(2024, 3, 1, tzinfo=UTC)
SPECS = (
    SensorSpec("Bedroom", Modality.MOTION, room="bedroom"),
    SensorSpec("Kitchen", Modality.MOTION, room="kitchen"),
)
RESOLUTION = EvidenceResolution(
    channels=(
        EvidenceChannel("bedroom", Modality.MOTION),
        EvidenceChannel("kitchen", Modality.MOTION),
    ),
    history_steps=1,
)
CURRENT = InformationSet(
    "current", frozenset({InformationComponent.CURRENT_EVIDENCE}), RESOLUTION
)
KITCHEN = "events_kitchen_motion_lag0"


def home(seed: int, *, blocks: int = 8, labelled: bool = True) -> CasasRecording:
    """Alternating sleep and kitchen blocks whose state the kitchen count reveals."""
    rng = random.Random(seed)
    events: list[tuple[str, datetime]] = []
    activities: list[ActivityInterval] = []
    start = T0
    for block in range(blocks):
        state = S.KITCHEN_ACTIVITY if (block + seed) % 2 else S.SLEEPING
        length = timedelta(minutes=rng.choice([20, 30, 40]))
        sensor, gap = ("Kitchen", 1) if state is S.KITCHEN_ACTIVITY else ("Bedroom", 9)
        moment = start
        while moment < start + length:
            events.append((sensor, moment + timedelta(seconds=rng.uniform(0, 30))))
            moment += timedelta(minutes=gap)
        activities.append(ActivityInterval(state.value, start, start + length, state))
        start += length
    observations = tuple(
        sorted(
            (
                Observation(at, sensor, Modality.MOTION, ObservationKind.EVENT, 1.0)
                for sensor, at in events
            ),
            key=lambda observation: observation.timestamp,
        )
    )
    return CasasRecording(
        registry=SensorRegistry.from_specs(SPECS),
        observations=observations,
        activities=tuple(activities) if labelled else (),
    )


RECORDINGS = {f"h{i}": home(i) for i in range(1, 7)}
SPLIT = HouseholdSplit(
    "unit", train=("h1", "h2"), development=("h3",), test=("h4", "h5", "h6")
)


class KitchenRule:
    """Deterministic and row-wise: kitchen motion now means kitchen activity."""

    def fit(self, training: LabelledRows, development: LabelledRows | None) -> None:
        return None

    def predict(self, rows: FeatureRows) -> StatePredictions:
        kitchen = np.nan_to_num(rows.values[:, rows.columns.index(KITCHEN)]) > 0
        probabilities = np.zeros((len(rows), len(rows.states)))
        probabilities[:, rows.states.index(S.KITCHEN_ACTIVITY)] = np.where(
            kitchen, 0.9, 0.1
        )
        probabilities[:, rows.states.index(S.SLEEPING)] = np.where(kitchen, 0.1, 0.9)
        labels = tuple(S.KITCHEN_ACTIVITY if k else S.SLEEPING for k in kitchen)
        return StatePredictions(labels, probabilities)


class Prior:
    """Training state frequencies at every row."""

    def fit(self, training: LabelledRows, development: LabelledRows | None) -> None:
        counts = np.array([training.labels.count(state) for state in training.states])
        self.frequencies = counts / counts.sum()

    def predict(self, rows: FeatureRows) -> StatePredictions:
        label = rows.states[int(np.argmax(self.frequencies))]
        return StatePredictions(
            tuple(label for _ in range(len(rows))),
            np.tile(self.frequencies, (len(rows), 1)),
        )


class LabelsOnly(KitchenRule):
    """The rule, reporting no probabilities."""

    def predict(self, rows: FeatureRows) -> StatePredictions:
        return StatePredictions(super().predict(rows).labels)


@dataclass
class Spy:
    """Wraps a model and records exactly what the runner handed it."""

    inner: Any
    log: dict[str, Any] = field(default_factory=dict)

    def fit(self, training: LabelledRows, development: LabelledRows | None) -> None:
        self.log["training"] = training
        self.log["development"] = development
        self.inner.fit(training, development)

    def predict(self, rows: FeatureRows) -> StatePredictions:
        self.log.setdefault("predict", []).append(rows)
        result: StatePredictions = self.inner.predict(rows)
        return result


class OrderDependent(KitchenRule):
    """Uses row position, which is how adjacency would leak history."""

    def predict(self, rows: FeatureRows) -> StatePredictions:
        states = (S.SLEEPING, S.KITCHEN_ACTIVITY)
        return StatePredictions(tuple(states[i % 2] for i in range(len(rows))))


class Transductive(KitchenRule):
    """Centres on the batch mean, so each prediction depends on other rows."""

    def predict(self, rows: FeatureRows) -> StatePredictions:
        kitchen = np.nan_to_num(rows.values[:, rows.columns.index(KITCHEN)])
        centred = 1.0 / (1.0 + np.exp(-(kitchen - kitchen.mean())))
        probabilities = np.zeros((len(rows), len(rows.states)))
        probabilities[:, rows.states.index(S.KITCHEN_ACTIVITY)] = centred
        probabilities[:, rows.states.index(S.SLEEPING)] = 1.0 - centred
        labels = tuple(S.KITCHEN_ACTIVITY if p > 0.5 else S.SLEEPING for p in centred)
        return StatePredictions(labels, probabilities)


def spec(name: str, factory: Any, **configuration: Any) -> ModelSpec:
    return ModelSpec(name, lambda seed: factory(), configuration)


def run(**overrides: Any) -> MatchedEvaluation:
    arguments: dict[str, Any] = {
        "split": SPLIT,
        "information_set": CURRENT,
        "models": [spec("prior", Prior), spec("rule", KitchenRule)],
        "seed": 7,
        "data_source": "synthetic-test",
    }
    arguments.update(overrides)
    return run_matched_evaluation(arguments.pop("recordings", RECORDINGS), **arguments)


class TestSplit:
    def test_roles_are_canonical_and_digested(self) -> None:
        split = HouseholdSplit(
            "s", train=("b", "a"), test=("d", "c"), development=("e",)
        )
        assert split.train == ("a", "b") and split.test == ("c", "d")
        assert split.households == ("a", "b", "c", "d", "e")
        assert split.role_of("e") == "development"
        same = HouseholdSplit(
            "s", train=("a", "b"), test=("c", "d"), development=("e",)
        )
        assert split.sha256() == same.sha256()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"train": ("a",), "test": ("a",)},
            {"train": ("a",), "test": ("b",), "development": ("a",)},
            {"train": (), "test": ("b",)},
            {"train": ("a",), "test": ()},
            {"train": ("",), "test": ("b",)},
        ],
    )
    def test_inconsistent_splits_are_rejected(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            HouseholdSplit("s", **kwargs)


class TestModelSpec:
    def test_configuration_must_be_serialisable_and_is_snapshotted(self) -> None:
        configuration: dict[str, Any] = {"depth": 3}
        model = ModelSpec("m", lambda seed: KitchenRule(), configuration)
        configuration["depth"] = 99
        assert model.to_dict()["configuration"] == {"depth": 3}
        with pytest.raises(ValueError, match="JSON"):
            ModelSpec("m", lambda seed: KitchenRule(), {"bad": object()})
        with pytest.raises(ValueError, match="JSON"):
            ModelSpec("m", lambda seed: KitchenRule(), {"bad": float("nan")})

    def test_the_builder_is_recorded_even_without_a_qualified_name(self) -> None:
        def make(seed: int, depth: int) -> KitchenRule:
            return KitchenRule()

        partial = ModelSpec("p", functools.partial(make, depth=2))
        assert partial.to_dict()["build"] == "functools.partial"


class TestMatchedInformation:
    def test_every_model_receives_identical_rows(self) -> None:
        logs: list[dict[str, Any]] = [{}, {}]
        run(
            models=[
                ModelSpec("a", lambda seed: Spy(Prior(), logs[0])),
                ModelSpec("b", lambda seed: Spy(KitchenRule(), logs[1])),
            ]
        )
        first, second = logs
        for key in ("training", "development"):
            np.testing.assert_array_equal(first[key].values, second[key].values)
            assert first[key].labels == second[key].labels
            assert first[key].households == second[key].households
        for one, two in zip(first["predict"], second["predict"], strict=True):
            np.testing.assert_array_equal(one.values, two.values)

    def test_models_see_only_the_declared_columns(self) -> None:
        log: dict[str, Any] = {}
        run(models=[ModelSpec("a", lambda seed: Spy(KitchenRule(), log))])
        assert log["training"].columns == CURRENT.columns
        assert "hour_of_day" not in log["training"].columns
        rows = log["predict"][0]
        assert type(rows) is FeatureRows
        assert not hasattr(rows, "labels") and not hasattr(rows, "households")

    def test_fitting_never_sees_a_held_out_household(self) -> None:
        log: dict[str, Any] = {}
        run(models=[ModelSpec("a", lambda seed: Spy(KitchenRule(), log))])
        assert set(log["training"].households) == {"h1", "h2"}
        assert set(log["development"].households) == {"h3"}
        held_out = sum(
            label is not None
            for h in SPLIT.test
            for label in truth_series(RECORDINGS[h].activities, list(build(h).moments))
        )
        assert len(log["predict"][0]) == held_out

    def test_rows_arrive_in_seeded_random_order(self) -> None:
        log: dict[str, Any] = {}
        run(models=[ModelSpec("a", lambda seed: Spy(KitchenRule(), log))])
        households = list(log["training"].households)
        assert households != sorted(households)

    def test_a_model_using_row_order_is_refused(self) -> None:
        with pytest.raises(ValueError, match="changed its prediction"):
            run(models=[spec("ordered", OrderDependent)])

    def test_a_model_using_other_rows_is_refused(self) -> None:
        with pytest.raises(ValueError, match="changed its prediction"):
            run(models=[spec("transductive", Transductive)])


def build(household: str) -> FeatureTable:
    recording = RECORDINGS[household]
    stamps = [o.timestamp for o in recording.observations]
    step = RESOLUTION.step
    grid = [stamps[0] + step * k for k in range((stamps[-1] - stamps[0]) // step + 1)]
    return build_feature_table(recording, CURRENT, grid, household=household)


class TestScoring:
    def test_household_metrics_match_scoring_each_household_directly(self) -> None:
        result = run()
        rule = KitchenRule()
        for household in SPLIT.test:
            table = build(household)
            truth = truth_series(RECORDINGS[household].activities, list(table.moments))
            keep = [i for i, label in enumerate(truth) if label is not None]
            rows = FeatureRows(
                table.columns,
                table.values[keep],
                tuple(state for state in S if state is not S.UNKNOWN),
            )
            direct = rule.predict(rows)
            expected = prediction_metrics(
                [truth[i] for i in keep],
                direct.labels,
                direct.probabilities,
                states=rows.states,
            )
            assert result.household_metrics["rule"][household].to_dict() == (
                expected.to_dict()
            )

    def test_comparisons_are_paired_by_household(self) -> None:
        result = run()
        by_metric = {c.metric: c for c in result.comparisons}
        assert set(by_metric) == {"balanced_accuracy", "log_loss", "brier"}
        for metric, comparison in by_metric.items():
            assert comparison.model == "rule" and comparison.reference == "prior"
            assert comparison.households == SPLIT.test
            assert comparison.difference is not None
            assert comparison.difference.n == len(SPLIT.test)
            rule = [
                getattr(result.household_metrics["rule"][h], metric) for h in SPLIT.test
            ]
            prior = [
                getattr(result.household_metrics["prior"][h], metric)
                for h in SPLIT.test
            ]
            improvement = (
                np.subtract(rule, prior)
                if metric == "balanced_accuracy"
                else np.subtract(prior, rule)
            )
            difference = comparison.difference
            assert difference.differences == pytest.approx(tuple(improvement))
            assert difference.mean.value == pytest.approx(improvement.mean())
            assert difference.median.value == pytest.approx(np.median(improvement))
            assert difference.favours_model == int((improvement > 0).sum())
            assert difference.mean.value > 0
            assert difference.mean.interval is not None

    def test_probabilistic_metrics_need_probabilities_from_both_models(self) -> None:
        result = run(models=[spec("prior", Prior), spec("labels", LabelsOnly)])
        by_metric = {c.metric: c for c in result.comparisons}
        assert by_metric["balanced_accuracy"].difference is not None
        for metric in ("log_loss", "brier"):
            assert by_metric[metric].difference is None
            assert by_metric[metric].skipped == "a model reported no probabilities"
        summary = result.record.results["models"]["labels"]["summary"]
        assert summary["log_loss"] is None
        assert summary["balanced_accuracy"]["n"] == len(SPLIT.test)

    def test_one_held_out_household_gets_an_estimate_without_an_interval(
        self,
    ) -> None:
        split = HouseholdSplit("one", train=("h1",), test=("h2",))
        result = run(split=split)
        for comparison in result.comparisons:
            assert comparison.difference is not None
            assert comparison.difference.n == 1
            assert comparison.difference.mean.interval is None
            assert "one household" in (comparison.difference.note or "")
        assert set(result.household_metrics["rule"]) == {"h2"}

    def test_bca_intervals_can_be_requested(self) -> None:
        comparison = run(interval="bca").comparisons[0].difference
        assert comparison is not None and comparison.mean.interval is not None
        assert comparison.mean.interval.method in {"bca", "percentile"}
        with pytest.raises(ValueError, match="interval"):
            run(interval="studentised")

    def test_an_unlabelled_held_out_household_is_recorded_not_scored(self) -> None:
        recordings = {**RECORDINGS, "blank": home(9, labelled=False)}
        split = HouseholdSplit("s", train=("h1", "h2"), test=("blank", "h4", "h5"))
        result = run(recordings=recordings, split=split)
        assert result.record.results["unscored_households"] == ["blank"]
        assert all(c.households == ("h4", "h5") for c in result.comparisons)
        assert "blank" not in result.household_metrics["rule"]

    def test_supplied_moments_are_used(self) -> None:
        moments = {h: list(build(h).moments)[::2] for h in SPLIT.households}
        result = run(moments=moments)
        households = result.record.results["households"]
        assert households["h4"]["moments"] == len(moments["h4"])
        assert result.record.configuration["moments"] == "supplied by caller"

    def test_scores_do_not_depend_on_the_seed_for_row_wise_models(self) -> None:
        first, second = run(seed=1), run(seed=2)
        for model in ("prior", "rule"):
            for household in SPLIT.test:
                assert (
                    first.household_metrics[model][household].to_dict()
                    == second.household_metrics[model][household].to_dict()
                )


class TestRecord:
    def test_the_record_is_written_with_its_provenance(self, tmp_path: Path) -> None:
        result = run(output_dir=tmp_path, experiment="unit-run")
        assert result.path == tmp_path / "unit-run.json"
        payload = load_record(result.path)

        def strict(token: str) -> None:
            raise ValueError(f"non-standard JSON constant {token}")

        json.loads(result.path.read_text(encoding="utf-8"), parse_constant=strict)

        assert payload["seeds"] == [7]
        assert payload["data_source"] == "synthetic-test"
        assert payload["recorded_at"]
        assert {"git_commit", "sensor_modeling"} <= set(payload["environment"])
        configuration = payload["configuration"]
        assert configuration["result_schema"] == "matched-evaluation/2"
        assert configuration["bootstrap"]["unit"] == "household"
        assert configuration["information_set"]["sha256"] == CURRENT.sha256()
        assert configuration["information_set"]["columns"] == list(CURRENT.columns)
        assert configuration["split"]["sha256"] == SPLIT.sha256()
        assert [m["name"] for m in configuration["models"]] == ["prior", "rule"]
        definitions = payload["metric_definitions"]
        assert {"balanced_accuracy", "confusion", "improvement", "brier"} <= set(
            definitions
        )
        assert not any("simulator" in note for note in payload["notes"])

        rule = payload["results"]["models"]["rule"]
        household = rule["households"]["h4"]
        assert {"balanced_accuracy", "per_class_recall", "confusion", "brier"} <= set(
            household
        )
        assert rule["confusion_summed"]["predicted"][-1] == "unknown"
        assert set(rule["summary"]["balanced_accuracy"]) >= {"median", "mean", "n"}
        assert payload["results"]["households"]["h1"]["role"] == "train"

    def test_the_same_seed_reproduces_the_record(self) -> None:
        first = run().record.to_dict()
        second = run().record.to_dict()
        for key in ("configuration", "results", "seeds", "metric_definitions"):
            assert first[key] == second[key]


class TestValidation:
    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"seed": -1}, "seed"),
            ({"seed": True}, "seed"),
            ({"models": []}, "at least one model"),
            ({"models": [spec("a", Prior), spec("a", KitchenRule)]}, "unique"),
            ({"metrics": ("balanced_accuracy", "auc")}, "metrics"),
            ({"metrics": ()}, "metrics"),
            ({"recordings": {"h1": RECORDINGS["h1"]}}, "no recording"),
            ({"moments": {"h1": []}}, "no moments"),
            ({"states": (S.SLEEPING,)}, "outside the label space"),
        ],
    )
    def test_inconsistent_runs_are_rejected(
        self, overrides: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            run(**overrides)

    def test_splits_with_nothing_labelled_to_learn_from_are_rejected(self) -> None:
        recordings = {**RECORDINGS, "blank": home(9, labelled=False)}
        with pytest.raises(ValueError, match="training"):
            run(
                recordings=recordings,
                split=HouseholdSplit("s", train=("blank",), test=("h4",)),
            )
        with pytest.raises(ValueError, match="development"):
            run(
                recordings=recordings,
                split=HouseholdSplit(
                    "s", train=("h1",), development=("blank",), test=("h4",)
                ),
            )
        with pytest.raises(ValueError, match="held-out"):
            run(
                recordings=recordings,
                split=HouseholdSplit("s", train=("h1",), test=("blank",)),
            )

    def test_an_object_without_fit_and_predict_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="fit and predict"):
            run(models=[ModelSpec("x", lambda seed: object())])  # type: ignore[arg-type,return-value]

    @pytest.mark.parametrize(
        "predictions",
        [
            lambda rows: StatePredictions((S.SLEEPING,)),
            lambda rows: StatePredictions(tuple(S.AWAY for _ in range(len(rows)))),
            lambda rows: StatePredictions(
                tuple(S.SLEEPING for _ in range(len(rows))),
                np.full((len(rows), len(rows.states)), 0.7),
            ),
            lambda rows: [S.SLEEPING] * len(rows),
        ],
    )
    def test_malformed_predictions_are_rejected(self, predictions: Any) -> None:
        class Broken(KitchenRule):
            def predict(self, rows: FeatureRows) -> StatePredictions:
                result: StatePredictions = predictions(rows)
                return result

        with pytest.raises((ValueError, TypeError), match="'broken'"):
            run(
                models=[spec("broken", Broken)], states=(S.SLEEPING, S.KITCHEN_ACTIVITY)
            )


def test_the_documented_example_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The example in docs/MATCHED_EVALUATION.md must stay executable."""
    page = Path(__file__).resolve().parents[1] / "docs" / "MATCHED_EVALUATION.md"
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), re.S)
    assert len(blocks) == 1
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, Any] = {}
    exec(compile(blocks[0], str(page), "exec"), namespace)  # noqa: S102
    written = tmp_path / "results" / "matched_evaluation.json"
    assert written.exists()
    assert namespace["result"].comparisons
