"""Tests for the pre-specified partial-pooling experiment.

They guard what makes its result interpretable:
- the protocol that ran is the one frozen before scoring;
- within an arm, every model is scored on the same households and windows;
- the population and the selected strength see only training households;
- a household's adaptation reads nothing after its cut-off;
- conclusions follow the declared rules, and the record and its summary are
  reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from sensor_modeling.datasets import ActivityInterval, CasasRecording, HouseholdSplit
from sensor_modeling.datasets.channel_models import (
    combine_statistics,
    fit_channel_models,
    home_statistics,
)
from sensor_modeling.datasets.information_sets import EvidenceResolution
from sensor_modeling.datasets.pooling_experiment import (
    ARM_MODELS,
    CRITERIA,
    DAY,
    ESTIMANDS,
    WEEK,
    PoolingProtocol,
    PoolingResult,
    cell_key,
    check_frozen_protocol,
    declared_protocol,
    overfitting_conclusion,
    pooling_conclusion,
    run_pooling,
    select_strength,
    strata,
)
from sensor_modeling.datasets.pooling_summary import render_summary
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.evaluation import load_record
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
PROTOCOL = ROOT / "artifacts" / "phase3" / "pooling_protocol.json"
ONTOLOGY = StateOntology()
FOLDS = (
    HouseholdSplit("a", train=("sim2", "sim4", "sim6"), test=("sim1", "sim3", "sim5")),
    HouseholdSplit("b", train=("sim1", "sim3", "sim5"), test=("sim2", "sim4", "sim6")),
)


def simulated(seed: int) -> CasasRecording:
    sim = simulate(HouseholdConfig(days=4, seed=seed))
    return CasasRecording(
        sim.registry,
        sim.observations,
        tuple(
            ActivityInterval(e.state.value, e.start, e.end, e.state)
            for e in sim.truth.episodes
        ),
    )


def relabelled_after(recording: CasasRecording, moment: Any) -> CasasRecording:
    """Every annotation that starts after *moment* made ``sleeping``."""
    return CasasRecording(
        recording.registry,
        recording.observations,
        tuple(
            (
                a
                if a.start <= moment
                else ActivityInterval("Sleep", a.start, a.end, S.SLEEPING)
            )
            for a in recording.activities
        ),
    )


def small_protocol(**changes: Any) -> PoolingProtocol:
    """The declared protocol, with arms short enough for four-day recordings."""
    settings: dict[str, Any] = {
        "folds": FOLDS,
        "splits_sha256": "0" * 64,
        "resamples": 200,
        "arm_days": {WEEK: 1.0, DAY: 0.25},
    }
    settings.update(changes)
    return PoolingProtocol(**settings)


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {f"sim{s}": simulated(s) for s in range(1, 7)}


@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[PoolingResult]:
    yield run_pooling(
        homes,
        small_protocol(),
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("pooling"),
    )


def results_of(outcome: PoolingResult) -> dict[str, Any]:
    return dict(outcome.record.to_dict()["results"])


# ----------------------------------------------------------------------------
# The protocol and its rules
# ----------------------------------------------------------------------------
class TestFrozenProtocol:
    def test_the_committed_protocol_is_the_declared_one(self) -> None:
        splits = load_frozen_splits(SPLITS)
        declared = declared_protocol(splits)
        check_frozen_protocol(declared, PROTOCOL)
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert frozen["protocol_sha256"] == declared.sha256()

    def test_it_freezes_everything_the_experiment_depends_on(self) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert frozen["arm_days"] == {"week": 7.0, "day": 1.0}
        assert frozen["pooling"]["strength"] == 288.0
        assert frozen["models"]["unconstrained"]["strength"] == 0.5
        assert frozen["selection"]["grid"] == [24.0, 72.0, 288.0, 1152.0, 4608.0]
        assert "never used" in frozen["selection"]["held_out"]
        assert frozen["eligibility"]["min_adaptation_windows"] == 12
        assert frozen["eligibility"]["min_scored_windows"] == 288
        assert [e["key"] for e in frozen["estimands"] if e["role"] == "primary"] == [
            "P1",
            "P2",
            "O1",
        ]
        assert frozen["criteria"] == CRITERIA
        assert frozen["bootstrap"]["resamples"] == 10_000

    @pytest.mark.parametrize(
        "change", [{"seed": 1}, {"strength": 144.0}, {"min_scored_windows": 100}]
    )
    def test_any_change_to_the_protocol_is_refused(
        self, change: dict[str, Any]
    ) -> None:
        changed = replace(declared_protocol(load_frozen_splits(SPLITS)), **change)
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(changed, PROTOCOL)

    def test_a_tampered_protocol_file_is_refused(self, tmp_path: Path) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        frozen["selection"]["grid"] = [288.0]
        tampered = tmp_path / "protocol.json"
        tampered.write_text(json.dumps(frozen), encoding="utf-8")
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(
                declared_protocol(load_frozen_splits(SPLITS)), tampered
            )

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"splits_sha256": "abc"}, "SHA-256"),
            ({"strength_grid": (288.0, 24.0)}, "increasing"),
            ({"strength_grid": (288.0,)}, "increasing"),
            ({"unconstrained_strength": 300.0}, "smallest"),
            ({"arm_days": {WEEK: 7.0}}, "week and day"),
            ({"min_scored_windows": 0}, "positive integer"),
            ({"strength": 0.0}, "positive"),
        ],
    )
    def test_invalid_protocols_are_refused(
        self, change: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            small_protocol(**change)


class TestRules:
    @pytest.mark.parametrize(
        ("log_loss", "accuracy", "calibration", "expected"),
        [
            ("favours model", "negligible", "negligible", "success"),
            ("favours model", "favours reference", "negligible", "trade-off"),
            ("favours model", "uncertain", "favours reference", "inconclusive"),
            ("negligible", "favours model", "favours model", "failure"),
            ("favours reference", "favours model", "favours model", "failure"),
            ("uncertain", "favours model", "favours model", "inconclusive"),
        ],
    )
    def test_the_pooling_rule(
        self, log_loss: str, accuracy: str, calibration: str, expected: str
    ) -> None:
        verdicts = {
            "log_loss": log_loss,
            "balanced_accuracy": accuracy,
            "calibration_error": calibration,
        }
        assert pooling_conclusion(verdicts) == expected

    @pytest.mark.parametrize(
        ("log_loss", "expected"),
        [
            ("favours model", "overfits"),
            ("negligible", "does not overfit"),
            ("favours reference", "does not overfit"),
            ("uncertain", "inconclusive"),
        ],
    )
    def test_the_overfitting_rule(self, log_loss: str, expected: str) -> None:
        assert overfitting_conclusion({"log_loss": log_loss}) == expected

    def test_strata_split_by_adaptation_data_then_name(self) -> None:
        split = strata({"b": 10, "a": 10, "c": 5, "d": 40, "e": 7})
        assert split == {"less data": ["c", "e"], "more data": ["a", "b", "d"]}

    def test_comparisons_stay_within_one_setting_and_arm(self) -> None:
        for estimand in ESTIMANDS:
            assert estimand.model in ARM_MODELS[estimand.arm]
            assert estimand.reference in ARM_MODELS[estimand.arm]
            assert estimand.to_dict()["model"].split("@")[1] == (
                estimand.to_dict()["reference"].split("@")[1]
            )


# ----------------------------------------------------------------------------
# No leakage
# ----------------------------------------------------------------------------
class TestNoLeakage:
    def test_combining_statistics_is_fitting(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        resolution = EvidenceResolution()
        training = {h: homes[h] for h in ("sim2", "sim4")}
        combined = combine_statistics(
            {
                h: home_statistics(r, resolution, ONTOLOGY, household=h)
                for h, r in training.items()
            },
            states=tuple(ONTOLOGY.states),
            pseudo_windows=12.0,
        )
        fitted = fit_channel_models(
            training, resolution=resolution, pseudo_windows=12.0
        )
        assert combined.sha256() == fitted.sha256()

    def test_the_strength_is_selected_on_training_households_only(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        protocol = small_protocol()
        resolution = protocol.resolution

        def selection(recordings: dict[str, CasasRecording]) -> dict[str, Any]:
            statistics = {
                h: home_statistics(r, resolution, ONTOLOGY, household=h)
                for h, r in recordings.items()
            }
            return select_strength(recordings, FOLDS[0], statistics, protocol, ONTOLOGY)

        first = selection(homes)
        assert set(first["households"]) <= set(FOLDS[0].train)
        start = min(o.timestamp for o in homes["sim1"].observations)
        held_out_changed = {**homes, "sim1": relabelled_after(homes["sim1"], start)}
        assert selection(held_out_changed) == first
        training_changed = {**homes, "sim2": relabelled_after(homes["sim2"], start)}
        assert selection(training_changed)["mean_log_loss"] != first["mean_log_loss"]

    def test_no_household_is_scored_with_a_population_fitted_on_it(
        self, result: PoolingResult
    ) -> None:
        results = results_of(result)
        for home, entry in results["households"].items():
            assert (
                home not in results["fitted"][entry["fold"]]["population"]["fitted_on"]
            )
            assert home not in results["selection"][entry["fold"]]["households"]
            for arm in entry["arms"].values():
                assert home not in arm["pooled"]["population"]["fitted_on"]

    def test_the_adaptation_reads_nothing_after_its_cutoff(
        self, homes: dict[str, CasasRecording], result: PoolingResult
    ) -> None:
        results = results_of(result)
        cutoff = results["households"]["sim1"]["arms"][WEEK]["cutoff"]
        from datetime import datetime

        changed = {
            **homes,
            "sim1": relabelled_after(homes["sim1"], datetime.fromisoformat(cutoff)),
        }
        again = results_of(
            run_pooling(changed, small_protocol(), data_source="simulator")
        )
        assert (
            again["households"]["sim1"]["arms"][WEEK]["pooled"]
            == results["households"]["sim1"]["arms"][WEEK]["pooled"]
        )


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------
class TestExperiment:
    def test_the_record_validates(self, result: PoolingResult) -> None:
        assert result.path is not None
        payload = load_record(result.path)
        assert payload["results"]["result_schema"] == "pooling-results/1"
        assert payload["configuration"]["protocol_sha256"] == small_protocol().sha256()
        assert payload["inference"]["mode"] == "online_filter"

    def test_the_record_lists_the_scored_windows(self, result: PoolingResult) -> None:
        # The arms' scored windows are nested, so each household contributes
        # those of its largest eligible arm, and each reads up to itself.
        households = results_of(result)["households"].values()
        scored = sum(
            max(
                (a["scored_windows"] for a in e["arms"].values() if a["eligible"]),
                default=0,
            )
            for e in households
        )
        evidence = load_record(result.path)["inference"]["evidence"]
        assert evidence["enumerated"] is True
        assert evidence["predictions"] == scored
        assert evidence["max_lead_seconds"] == 0.0
        cutoffs = [
            datetime.fromisoformat(a["cutoff"])
            for e in households
            for a in e["arms"].values()
            if a["eligible"]
        ]
        assert datetime.fromisoformat(evidence["first_prediction"]) > min(cutoffs)

    def test_models_in_an_arm_are_scored_on_the_same_windows(
        self, result: PoolingResult
    ) -> None:
        metrics = result.record.household_metrics
        results = results_of(result)
        for arm, models in ARM_MODELS.items():
            eligible = sorted(
                h
                for h, e in results["households"].items()
                if e["arms"][arm]["eligible"]
            )
            for setting in ("I0", "R"):
                keys = [cell_key(model, setting, arm) for model in models]
                assert all(sorted(metrics[key]) == eligible for key in keys)
                for home in eligible:
                    sizes = {metrics[key][home]["n"] for key in keys}
                    assert sizes == {
                        results["households"][home]["arms"][arm]["scored_windows"]
                    }

    def test_every_household_reports_its_pooling(self, result: PoolingResult) -> None:
        for entry in results_of(result)["households"].values():
            for arm in (WEEK, DAY):
                pooled = entry["arms"][arm]["pooled"]
                assert pooled["pooling"] == {"strength": 288.0}
                for channel in pooled["channels"]:
                    for parameter in ("silence", "active_mean"):
                        estimate = channel[parameter]
                        for key in (
                            "raw",
                            "population",
                            "pooled",
                            "shrinkage",
                            "count",
                        ):
                            assert len(estimate[key]) == ONTOLOGY.size

    def test_conclusions_follow_the_rules(self, result: PoolingResult) -> None:
        results = results_of(result)
        rules = {"pooling": pooling_conclusion, "overfitting": overfitting_conclusion}
        for entry in results["estimands"]:
            assert entry["conclusion"] == rules[entry["rule"]](entry["verdicts"])
            members = [h for s in entry["strata"].values() for h in s["households"]]
            arm = entry["model"].split("/")[1]
            assert sorted(members) == sorted(
                h
                for h, e in results["households"].items()
                if e["arms"][arm]["eligible"]
            )
        by_key = {e["key"]: e["conclusion"] for e in results["estimands"]}
        assert results["conclusions"] == {
            "pooling_current_windows": by_key["P1"],
            "pooling_recursion": by_key["P2"],
            "overfitting_small_homes": by_key["O1"],
        }

    def test_an_arm_without_two_eligible_households_is_refused(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        with pytest.raises(ValueError, match="eligible"):
            run_pooling(
                homes,
                small_protocol(min_adaptation_windows=10_000),
                data_source="simulator",
            )

    def test_a_second_run_gives_the_same_record(
        self, homes: dict[str, CasasRecording], result: PoolingResult
    ) -> None:
        again = run_pooling(homes, small_protocol(), data_source="simulator")
        first, second = result.record.to_dict(), again.record.to_dict()
        first.pop("recorded_at")
        second.pop("recorded_at")
        assert first == second

    def test_the_summary_is_generated_from_the_written_record(
        self, result: PoolingResult
    ) -> None:
        assert result.path is not None
        summary = render_summary(load_record(result.path))
        assert summary == render_summary(
            json.loads(json.dumps(result.record.to_dict()))
        )
        for heading in (
            "## Pre-specified conclusions",
            "## Strength selection",
            "## Every cell",
            "## Estimands: log loss",
            "## Estimands: the guards",
            "## By amount of adaptation data",
            "## Per-state recall",
            "## Pooled silence by state",
            "## Households",
        ):
            assert heading in summary
        assert "- Inference regime: online filter" in summary
        payload = result.record.to_dict()
        payload["results"] = {"result_schema": "fitted-rates-results/1"}
        with pytest.raises(ValueError, match="pooling-results"):
            render_summary(payload)


# ----------------------------------------------------------------------------
# The published result
# ----------------------------------------------------------------------------
PUBLISHED = ROOT / "artifacts" / "phase3" / "phase3-partial-pooling.json"
DOC = ROOT / "docs" / "PHASE3_PARTIAL_POOLING.md"


class TestPublishedResult:
    """The published development-panel result, and the page that reports it."""

    def test_it_ran_the_frozen_protocol_on_a_clean_tree(self) -> None:
        payload = load_record(PUBLISHED)
        splits = load_frozen_splits(SPLITS)
        declared = declared_protocol(splits)
        assert payload["configuration"]["protocol_sha256"] == declared.sha256()
        assert payload["environment"]["git_dirty"] == "false"
        assert payload["inference"]["mode"] == "online_filter"
        inputs = {item["name"]: item["sha256"] for item in payload["inputs"]}
        assert inputs.pop(PROTOCOL.name) == check_frozen_protocol(declared, PROTOCOL)
        assert inputs.pop(SPLITS.name) == splits.sha256
        assert inputs == {
            entry["filename"]: entry["sha256"] for entry in splits.homes.values()
        }
        assert sorted(payload["results"]["households"]) == sorted(splits.homes)

    def test_the_population_is_the_phase_3_3_follow_up_fit(self) -> None:
        fitted = load_record(ROOT / "artifacts" / "phase3" / "phase3-fitted-rates.json")
        earlier = fitted["results"]["fitted"]
        for fold, entry in load_record(PUBLISHED)["results"]["fitted"].items():
            assert entry["population"]["sha256"] == earlier[fold]["channels"]["sha256"]

    def test_the_page_carries_the_summary_generated_from_it(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        block = text.split("<!-- generated-summary:start -->")[1]
        block = block.split("<!-- generated-summary:end -->")[0]
        assert block.strip() == render_summary(load_record(PUBLISHED), level=3).strip()
