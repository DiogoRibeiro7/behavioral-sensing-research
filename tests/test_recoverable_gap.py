"""Tests for the recoverable-information-gap experiment.

They guard what makes its comparisons interpretable:
- the folds are the frozen Phase 1 folds and nothing is tuned;
- the protocol is fixed and digested;
- the restricted generative model is the production filter's model and reads
  nothing outside its declared windows;
- every model is scored on identical rows;
- unsupported comparisons are recorded rather than forced;
- the result and its Markdown summary are reproducible.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    EvidenceResolution,
    GradientBoostingBaseline,
    HouseholdSplit,
    InformationComponent,
    InformationSet,
    ModelSpec,
    build_feature_table,
    nested_information_sets,
    truth_series,
)
from sensor_modeling.datasets.gap_summary import render_summary
from sensor_modeling.datasets.matched_evaluation import _regular_moments
from sensor_modeling.datasets.recoverable_gap import (
    CIRCADIAN_UNSCORED,
    DEFAULT_METRICS,
    FILTER_UNMATCHED,
    GapProtocol,
    GapResult,
    gap_models,
    load_frozen_splits,
    run_recoverable_gap,
)
from sensor_modeling.datasets.restricted_filter import (
    TIME_OF_DAY_UNSUPPORTED,
    channel_likelihoods,
    filter_posteriors,
    restricted_posteriors,
    unsupported_reason,
)
from sensor_modeling.evaluation import load_record, prediction_metrics
from sensor_modeling.fusion.defaults import default_emissions
from sensor_modeling.fusion.filter import MultimodalBayesFilter
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
)
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states.ontology import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
PROFILE = ROOT / "artifacts" / "v03" / "v03_circadian_profile.json"
PUBLISHED = ROOT / "artifacts" / "phase1" / "phase1-recoverable-information-gap.json"
DOC = ROOT / "docs" / "PHASE1_RECOVERABLE_GAP.md"
I0, I1, I2, I3 = nested_information_sets()
STEP = I0.resolution.step
UTC = timezone.utc
T0 = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)


def simulated(
    seed: int, *, unlabelled: timedelta = timedelta(0), days: int = 2
) -> CasasRecording:
    """A simulated household, optionally without labels for its first *unlabelled*."""
    sim = simulate(HouseholdConfig(days=days, seed=seed))
    start = min(o.timestamp for o in sim.observations) + unlabelled
    return CasasRecording(
        sim.registry,
        sim.observations,
        tuple(
            ActivityInterval(e.state.value, e.start, e.end, e.state)
            for e in sim.truth.episodes
            if e.start >= start
        ),
    )


FOLDS = (
    HouseholdSplit("a", train=("sim2", "sim4"), test=("sim1", "sim3")),
    HouseholdSplit("b", train=("sim1", "sim3"), test=("sim2", "sim4")),
)


def light_models() -> tuple[ModelSpec, ...]:
    """The declared models with a cheaper diagnostic, to keep the suite fast."""
    reference, _, logistic = gap_models()
    return (
        reference,
        ModelSpec(
            "diagnostic",
            lambda seed: GradientBoostingBaseline(max_iter=10, seed=seed),
            GradientBoostingBaseline(max_iter=10).configuration(),
        ),
        logistic,
    )


def protocol(**changes: Any) -> GapProtocol:
    settings: dict[str, Any] = {
        "folds": FOLDS,
        "resamples": 100,
        "models": light_models(),
    }
    settings.update(changes)
    return GapProtocol(**settings)


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    """Four households; the first has no labels for half a day."""
    return {
        f"sim{s}": simulated(
            s, unlabelled=timedelta(hours=12) if s == 1 else timedelta(0)
        )
        for s in range(1, 5)
    }


@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[GapResult]:
    with threadpool_limits(1):
        yield run_recoverable_gap(
            homes,
            protocol(),
            data_source="simulator",
            output_dir=tmp_path_factory.mktemp("gap"),
        )


def results_of(gap: GapResult) -> dict[str, Any]:
    return dict(gap.record.to_dict()["results"])


class TestFrozenSplits:
    def test_the_folds_are_the_phase1_folds(self) -> None:
        splits = load_frozen_splits(SPLITS)
        profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        homes = sorted(profile["fitting_home_ids"])
        assert splits.folds == (
            HouseholdSplit(
                "phase1-fold-a", train=tuple(homes[1::2]), test=tuple(homes[0::2])
            ),
            HouseholdSplit(
                "phase1-fold-b", train=tuple(homes[0::2]), test=tuple(homes[1::2])
            ),
        )

    def test_the_homes_are_the_frozen_v03_panel(self) -> None:
        splits = load_frozen_splits(SPLITS)
        profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        panel = {home["id"]: home for home in profile["development_panel_homes"]}
        assert set(splits.homes) == set(profile["fitting_home_ids"])
        excluded = {item["id"] for item in profile["fitting_exclusions"]}
        assert not excluded & set(splits.homes)
        for home, entry in splits.homes.items():
            assert entry == {
                "filename": panel[home]["filename"],
                "sha256": panel[home]["sha256"],
            }

    def test_every_home_is_held_out_exactly_once(self) -> None:
        declared = GapProtocol(load_frozen_splits(SPLITS).folds)
        assert len(declared.homes) == 20
        held_out = [home for fold in declared.folds for home in fold.test]
        assert sorted(held_out) == list(declared.homes)

    def test_the_file_digest_is_recorded(self) -> None:
        import hashlib

        assert (
            load_frozen_splits(SPLITS).sha256
            == hashlib.sha256(SPLITS.read_bytes()).hexdigest()
        )

    def test_malformed_files_are_refused(self, tmp_path: Path) -> None:
        payload = json.loads(SPLITS.read_text(encoding="utf-8"))
        other = tmp_path / "other.json"
        other.write_text(json.dumps({**payload, "schema": "x"}), encoding="utf-8")
        with pytest.raises(ValueError, match="household-splits"):
            load_frozen_splits(other)
        missing = dict(payload["homes"])
        missing.pop("hh101")
        short = tmp_path / "short.json"
        short.write_text(json.dumps({**payload, "homes": missing}), encoding="utf-8")
        with pytest.raises(ValueError, match="different households"):
            load_frozen_splits(short)


class TestProtocol:
    def test_the_declared_defaults(self) -> None:
        declared = GapProtocol(FOLDS)
        assert declared.information_sets == nested_information_sets()
        assert declared.metrics == DEFAULT_METRICS
        assert set(DEFAULT_METRICS) == {
            "balanced_accuracy",
            "brier",
            "log_loss",
            "calibration_error",
        }
        assert (declared.seed, declared.resamples, declared.confidence) == (
            0,
            10_000,
            0.95,
        )
        names = [spec.name for spec in declared.models]
        assert names == ["state_frequency", "diagnostic", "logistic"]
        diagnostic = dict(declared.models[1].configuration)
        assert diagnostic["max_iter"] == 100 and diagnostic["early_stopping"] is False
        assert declared.models[2].configuration["C"] == 1.0
        assert declared.labels == {
            "current": "I0",
            "current+time_of_day": "I1",
            "current+history": "I2",
            "current+time_of_day+history": "I3",
        }

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"folds": FOLDS[:1]}, "two folds"),
            (
                {
                    "folds": (
                        FOLDS[0],
                        HouseholdSplit("b", train=("sim2",), test=("sim1", "sim4")),
                    )
                },
                "more than one fold",
            ),
            (
                {
                    "folds": (
                        FOLDS[0],
                        HouseholdSplit("b", train=("sim1",), test=("sim2", "sim4")),
                    )
                },
                "exactly the households",
            ),
            (
                {
                    "folds": (
                        HouseholdSplit(
                            "a",
                            train=("sim2",),
                            test=("sim1", "sim3"),
                            development=("sim4",),
                        ),
                        FOLDS[1],
                    )
                },
                "development",
            ),
            ({"folds": (FOLDS[0], replace(FOLDS[1], name="a"))}, "unique"),
            ({"information_sets": nested_information_sets()[::-1]}, "four nested"),
            (
                {
                    "information_sets": nested_information_sets(
                        EvidenceResolution(history_steps=2)
                    )[:3]
                },
                "four nested",
            ),
            ({"metrics": ("log_loss", "brier")}, "balanced_accuracy"),
            ({"metrics": ("balanced_accuracy", "auc")}, "distinct names"),
            ({"metrics": ("balanced_accuracy",) * 2}, "distinct names"),
            ({"seed": -1}, "seed"),
            ({"resamples": 10}, "resamples"),
            ({"models": gap_models()[::-1]}, "in that order"),
        ],
    )
    def test_invalid_protocols_are_refused(
        self, changes: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            protocol(**changes)

    def test_the_declaration_is_serialisable_and_digested(self) -> None:
        declared = GapProtocol(FOLDS)
        payload = json.loads(json.dumps(declared.to_dict(), allow_nan=False))
        assert payload["tuning"].startswith("none")
        assert payload["models"]["generative"]["unsupported"] == {
            "I1": TIME_OF_DAY_UNSUPPORTED,
            "I3": TIME_OF_DAY_UNSUPPORTED,
        }
        assert declared.sha256() == GapProtocol(FOLDS).sha256()
        for changed in (
            GapProtocol(FOLDS, seed=1),
            GapProtocol(FOLDS, resamples=1000),
            GapProtocol(FOLDS, models=light_models()),
            GapProtocol(FOLDS[::-1]),
        ):
            assert changed.sha256() != declared.sha256()


def registry_of(*specs: SensorSpec) -> SensorRegistry:
    return SensorRegistry.from_specs(specs)


def recording_of(
    registry: SensorRegistry, events: Sequence[tuple[str, datetime]]
) -> CasasRecording:
    kinds = {spec.sensor_id: spec for spec in registry}
    return CasasRecording(
        registry,
        tuple(
            sorted(
                (
                    Observation(
                        at, sensor, kinds[sensor].modality, ObservationKind.EVENT, 1.0
                    )
                    for sensor, at in events
                ),
                key=lambda observation: observation.timestamp,
            )
        ),
        (),
    )


KITCHEN = SensorSpec("K1", Modality.MOTION, kind=ObservationKind.EVENT, room="kitchen")
KITCHEN_2 = SensorSpec(
    "K2", Modality.MOTION, kind=ObservationKind.EVENT, room="kitchen"
)
BEDROOM = SensorSpec("B1", Modality.MOTION, kind=ObservationKind.EVENT, room="bedroom")
DOOR = SensorSpec("D1", Modality.DOOR, kind=ObservationKind.EVENT, room="hall")


def posterior_at(
    recording: CasasRecording, moment: datetime, information_set: Any
) -> np.ndarray:
    table = build_feature_table(recording, information_set, [moment], household="h")
    terms, _ = channel_likelihoods(recording.registry, information_set.resolution)
    posterior: np.ndarray = restricted_posteriors(table, terms)[0]
    return posterior


class TestRestrictedFilter:
    def test_only_current_evidence_and_history_are_supported(self) -> None:
        assert unsupported_reason(I0) is None and unsupported_reason(I2) is None
        assert unsupported_reason(I1) == TIME_OF_DAY_UNSUPPORTED
        assert unsupported_reason(I3) == TIME_OF_DAY_UNSUPPORTED
        summaries = InformationSet(
            "current+history_summary",
            I0.components | {InformationComponent.HISTORY_SUMMARY},
        )
        assert unsupported_reason(summaries) == (
            "the generative model has no input for history_summary"
        )
        table = build_feature_table(
            recording_of(registry_of(KITCHEN), [("K1", T0)]), I1, [T0], household="h"
        )
        with pytest.raises(ValueError, match="time-of-day"):
            restricted_posteriors(table, {})

    @pytest.mark.parametrize("information_set", [I0, I2], ids=["I0", "I2"])
    def test_it_is_the_production_filter_fed_only_the_declared_windows(
        self, information_set: Any
    ) -> None:
        source = simulated(5)
        moments = _regular_moments(source, STEP)
        table = build_feature_table(source, information_set, moments, household="h")
        terms, ignored = channel_likelihoods(
            source.registry, information_set.resolution
        )
        assert ignored  # the simulator also has sensors outside the channels
        posteriors = restricted_posteriors(table, terms)

        ontology = StateOntology()
        routed = {sensor for term in terms.values() for sensor in term.sensors}
        emissions = [
            e
            for e in default_emissions(source.registry, ontology)
            if e.sensor_id in routed
        ]
        depth = 3 if information_set is I2 else 0
        observations = sorted(source.observations, key=lambda o: o.timestamp)
        for row in range(depth + 1, len(moments), 41):
            moment = moments[row]
            production = MultimodalBayesFilter(ontology, emissions, source.registry)
            production.update(moment - STEP * (depth + 1))
            for lag in range(depth, -1, -1):
                end = moment - STEP * lag
                estimate = production.update(
                    end, [o for o in observations if end - STEP < o.timestamp <= end]
                )
            np.testing.assert_allclose(estimate.belief, posteriors[row], atol=1e-12)

    def test_nothing_outside_the_declared_windows_is_read(self) -> None:
        registry = registry_of(KITCHEN, BEDROOM, DOOR)
        moment = T0 + timedelta(hours=2)
        base = [("B1", T0), ("K1", moment - timedelta(minutes=3)), ("D1", moment)]
        reference = recording_of(registry, base)
        outside = recording_of(
            registry,
            [
                *base,
                ("K1", moment - STEP * 4),  # closes the window before lag 3
                ("D1", moment - timedelta(minutes=45)),
                ("B1", moment + timedelta(seconds=1)),
            ],
        )
        inside = recording_of(
            registry, [*base, ("K1", moment - STEP * 4 + timedelta(seconds=1))]
        )
        np.testing.assert_array_equal(
            posterior_at(reference, moment, I2), posterior_at(outside, moment, I2)
        )
        assert not np.allclose(
            posterior_at(reference, moment, I2), posterior_at(inside, moment, I2)
        )
        earlier = recording_of(registry, [*base, ("K1", moment - STEP)])
        np.testing.assert_array_equal(
            posterior_at(reference, moment, I0), posterior_at(earlier, moment, I0)
        )

    def test_a_pooled_channel_count_gives_the_sensors_likelihood(self) -> None:
        registry = registry_of(KITCHEN, KITCHEN_2, BEDROOM)
        moment = T0 + timedelta(hours=1)
        at = moment - timedelta(minutes=2)
        one_sensor = recording_of(registry, [("B1", T0), ("K1", at), ("K1", at)])
        both = recording_of(registry, [("B1", T0), ("K1", at), ("K2", at)])
        np.testing.assert_allclose(
            posterior_at(one_sensor, moment, I0), posterior_at(both, moment, I0)
        )
        fewer = recording_of(
            registry_of(KITCHEN, BEDROOM), [("B1", T0), ("K1", at), ("K1", at)]
        )
        assert not np.allclose(
            posterior_at(one_sensor, moment, I0), posterior_at(fewer, moment, I0)
        )

    def test_channels_whose_pooling_would_not_be_exact_are_refused(self) -> None:
        grouped = replace(KITCHEN_2, redundancy_group="kitchen")
        grouped_too = SensorSpec(
            "K3",
            Modality.MOTION,
            kind=ObservationKind.EVENT,
            room="kitchen",
            redundancy_group="kitchen",
        )
        with pytest.raises(ValueError, match="different rates"):
            channel_likelihoods(
                registry_of(KITCHEN, grouped, grouped_too), I0.resolution
            )

    def test_uninstrumented_channels_contribute_nothing(self) -> None:
        source = recording_of(registry_of(KITCHEN), [("K1", T0)])
        table = build_feature_table(
            source, I2, [T0 + timedelta(hours=1)], household="h"
        )
        terms, _ = channel_likelihoods(source.registry, I2.resolution)
        assert set(terms) == set(I2.resolution.channels) - set(table.uninstrumented)
        assert np.isfinite(restricted_posteriors(table, terms)).all()
        with pytest.raises(ValueError, match="instrumented"):
            restricted_posteriors(table, {})

    def test_a_circadian_ontology_is_refused(self) -> None:
        source = recording_of(registry_of(KITCHEN), [("K1", T0)])
        table = build_feature_table(source, I0, [T0], household="h")
        terms, _ = channel_likelihoods(source.registry, I0.resolution)
        circadian = StateOntology(circadian={S.SLEEPING: [1.0] * 24})
        with pytest.raises(ValueError, match="time-of-day"):
            restricted_posteriors(table, terms, circadian)

    def test_the_production_filter_is_read_at_the_grid(self) -> None:
        source = simulated(6)
        moments = _regular_moments(source, STEP)
        beliefs, states = filter_posteriors(source, moments, step=STEP)
        assert beliefs.shape == (len(moments), len(states))
        np.testing.assert_allclose(beliefs.sum(axis=1), 1.0)
        with pytest.raises(ValueError, match="no step"):
            filter_posteriors(source, [moments[3] + timedelta(minutes=1)], step=STEP)


class TestExperiment:
    def test_the_record_validates_and_is_written(self, result: GapResult) -> None:
        assert result.path is not None
        payload = load_record(result.path)
        assert payload["experiment"] == "phase1-recoverable-information-gap"
        assert payload["configuration"]["protocol_sha256"] == protocol().sha256()
        assert payload["results"]["result_schema"] == "recoverable-gap/1"

    def test_every_model_is_scored_on_the_runners_rows(self, result: GapResult) -> None:
        households = results_of(result)["households"]
        assert sorted(households) == ["sim1", "sim2", "sim3", "sim4"]
        for label, runs in result.runs.items():
            for run in runs:
                for home, digest in run.moments.items():
                    assert households[home]["moments_sha256"] == digest, label
        metrics = result.record.household_metrics
        for cell in ("generative@I0", "generative@I2", "filter", "diagnostic@I3"):
            assert sorted(metrics[cell]) == sorted(households)
            for home, scores in metrics[cell].items():
                assert scores["n"] == households[home]["labelled"]

    @pytest.mark.parametrize("cell", ["generative@I2", "filter"])
    def test_generative_cells_score_exactly_the_labelled_rows(
        self, homes: dict[str, CasasRecording], result: GapResult, cell: str
    ) -> None:
        recording = homes["sim1"]
        moments = _regular_moments(recording, STEP)
        truth = truth_series(recording.activities, moments)
        labelled = [row for row, label in enumerate(truth) if label is not None]
        assert 0 < len(labelled) < len(moments)
        if cell == "filter":
            beliefs, _ = filter_posteriors(recording, moments, step=STEP)
        else:
            table = build_feature_table(recording, I2, moments, household="sim1")
            terms, _ = channel_likelihoods(recording.registry, I2.resolution)
            beliefs = restricted_posteriors(table, terms)
        space = tuple(StateOntology().states)
        chosen = beliefs[labelled]
        expected = prediction_metrics(
            [truth[row] for row in labelled],
            [space[int(i)] for i in chosen.argmax(axis=1)],
            chosen,
            states=space,
        ).to_dict()
        recorded = result.record.household_metrics[cell]["sim1"]
        for metric in (
            "n",
            "balanced_accuracy",
            "log_loss",
            "brier",
            "calibration_error",
        ):
            assert recorded[metric] == pytest.approx(expected[metric]), metric

    def test_rows_that_differ_from_the_runners_are_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sensor_modeling.datasets.recoverable_gap as gap

        def shifted(recording: CasasRecording, step: timedelta) -> list[datetime]:
            return _regular_moments(recording, step)[1:]

        monkeypatch.setattr(gap, "_regular_moments", shifted)
        small = {f"sim{s}": simulated(s, days=1) for s in range(1, 5)}
        with (
            threadpool_limits(1),
            pytest.raises(ValueError, match="differ from the runner"),
        ):
            run_recoverable_gap(small, protocol(), data_source="simulator")

    def test_unsupported_comparisons_are_recorded_not_forced(
        self, result: GapResult
    ) -> None:
        results = results_of(result)
        unsupported = {
            (entry["model"], entry["information_set"]): entry["reason"]
            for entry in results["unsupported"]
        }
        assert unsupported[("generative", "I1")] == TIME_OF_DAY_UNSUPPORTED
        assert unsupported[("generative", "I3")] == TIME_OF_DAY_UNSUPPORTED
        assert all(
            unsupported[("filter", label)] == FILTER_UNMATCHED
            for label in ("I0", "I1", "I2", "I3")
        )
        assert unsupported[("circadian filter", "any")] == CIRCADIAN_UNSCORED
        assert "generative@I1" not in result.record.household_metrics
        for group in ("information_gains", "formulation_gaps", "interactions"):
            for entry in results[group]:
                touched = {
                    entry.get("information_set"),
                    entry.get("from"),
                    entry.get("to"),
                }
                if "generative" in (entry["model"], entry.get("reference")):
                    assert not touched & {"I1", "I3"}

    def test_cells_report_every_required_metric(self, result: GapResult) -> None:
        cells = {
            (c["model"], c["information_set"]): c for c in results_of(result)["cells"]
        }
        assert cells[("generative", "I1")]["status"] == "unsupported"
        assert cells[("filter", "unbounded")]["status"] == "reference"
        for key in (("diagnostic", "I3"), ("logistic", "I0"), ("generative", "I2")):
            cell = cells[key]
            assert cell["status"] == "matched"
            assert set(cell["metrics"]) == set(DEFAULT_METRICS)
            assert cell["metrics"]["balanced_accuracy"]["mean_interval"] is not None
            assert list(cell["per_state_recall"]) == results_of(result)["states"]

    def test_information_gains_hold_the_model_fixed(self, result: GapResult) -> None:
        gains = results_of(result)["information_gains"]
        pairs = {(g["model"], g["from"], g["to"]) for g in gains}
        assert ("generative", "I0", "I2") in pairs
        assert {(m, f, t) for m, f, t in pairs if m == "generative"} == {
            ("generative", "I0", "I2")
        }
        assert len({(m, f, t) for m, f, t in pairs if m == "diagnostic"}) == 5
        metrics = result.record.household_metrics
        entry = next(
            g
            for g in gains
            if (g["model"], g["from"], g["to"], g["metric"])
            == ("generative", "I0", "I2", "balanced_accuracy")
        )
        for home, difference in entry["comparison"]["differences"].items():
            expected = (
                metrics["generative@I2"][home]["balanced_accuracy"]
                - metrics["generative@I0"][home]["balanced_accuracy"]
            )
            assert difference == pytest.approx(expected)

    def test_interactions_are_differences_of_gains(self, result: GapResult) -> None:
        metrics = result.record.household_metrics
        entry = next(
            i
            for i in results_of(result)["interactions"]
            if (i["model"], i["reference"], i["from"], i["to"], i["metric"])
            == ("diagnostic", "generative", "I0", "I2", "log_loss")
        )

        def loss(cell: str, home: str) -> float:
            return float(metrics[cell][home]["log_loss"])

        for home, difference in entry["comparison"]["differences"].items():
            gain_first = loss("diagnostic@I0", home) - loss("diagnostic@I2", home)
            gain_second = loss("generative@I0", home) - loss("generative@I2", home)
            assert difference == pytest.approx(gain_first - gain_second)

    def test_filter_comparisons_state_their_relation(self, result: GapResult) -> None:
        relations = {
            (e["model"], e["information_set"]): e["relation"]
            for e in results_of(result)["against_filter"]
        }
        assert relations[("generative", "I2")].startswith("the filter's information")
        assert relations[("diagnostic", "I3")] == "neither set contains the other"

    def test_every_interval_is_quotable(self, result: GapResult) -> None:
        labels = [interval.label for interval in result.record.intervals]
        assert len(labels) == len(set(labels))
        assert any(
            label.startswith("formulation gap: diagnostic vs generative, I2")
            for label in labels
        )
        assert all(
            interval.low <= interval.high and interval.unit == "household"
            for interval in result.record.intervals
        )


class TestReproducibility:
    def test_a_second_run_gives_the_same_record(
        self, homes: dict[str, CasasRecording], result: GapResult
    ) -> None:
        with threadpool_limits(1):
            again = run_recoverable_gap(homes, protocol(), data_source="simulator")
        first, second = result.record.to_dict(), again.record.to_dict()
        for payload in (first, second):
            payload.pop("recorded_at")
        assert first == second

    def test_the_summary_is_generated_from_the_written_record(
        self, result: GapResult
    ) -> None:
        assert result.path is not None
        written = load_record(result.path)
        summary = render_summary(written)
        assert summary == render_summary(
            json.loads(json.dumps(result.record.to_dict()))
        )
        assert summary == render_summary(load_record(result.path))
        for heading in (
            "## What each model receives",
            "## Balanced accuracy",
            "## Probability quality",
            "## What added information is worth",
            "## What the formulation is worth",
            "## Interactions",
            "## Against the production filter",
            "## Per-state recall",
            "## Not compared",
        ):
            assert heading in summary
        assert "do not add up" in summary
        assert f"`{written['configuration']['protocol_sha256'][:12]}`" in summary
        assert render_summary(written, level=2).startswith("## ")

    def test_the_summary_refuses_other_records(self, result: GapResult) -> None:
        payload = result.record.to_dict()
        payload["results"] = {"result_schema": "matched-evaluation/3"}
        with pytest.raises(ValueError, match="recoverable-gap"):
            render_summary(payload)

    def test_missing_households_are_refused(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        partial = {name: homes[name] for name in ("sim1", "sim2", "sim3")}
        with pytest.raises(ValueError, match="sim4"):
            run_recoverable_gap(partial, protocol(), data_source="simulator")


class TestPublishedResult:
    """The published development-panel result, and the page that reports it."""

    def test_it_ran_the_declared_protocol_on_a_clean_tree(self) -> None:
        payload = load_record(PUBLISHED)
        splits = load_frozen_splits(SPLITS)
        assert payload["configuration"]["protocol_sha256"] == (
            GapProtocol(splits.folds).sha256()
        )
        assert payload["environment"]["git_dirty"] == "false"
        inputs = {item["name"]: item["sha256"] for item in payload["inputs"]}
        assert inputs.pop(SPLITS.name) == splits.sha256
        assert inputs == {
            entry["filename"]: entry["sha256"] for entry in splits.homes.values()
        }
        assert sorted(payload["results"]["households"]) == sorted(splits.homes)

    def test_the_page_carries_the_summary_generated_from_it(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        block = text.split("<!-- generated-summary:start -->")[1]
        block = block.split("<!-- generated-summary:end -->")[0]
        assert block.strip() == render_summary(load_record(PUBLISHED), level=3).strip()


def test_no_value_in_the_record_is_non_finite(result: GapResult) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(result.record.to_dict())
