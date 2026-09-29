"""Tests for the pre-specified evaluation of the hurdle negative binomial.

They guard what makes its result interpretable:
- the protocol that ran is the one frozen before scoring;
- the three observation models share every other part of inference, and are
  scored on the same windows;
- the questions and the decision follow the declared rules, which never accept
  the extra parameter because one metric improves;
- the dispersion estimates are recorded and classified as declared;
- the record and its summary are reproducible from the artifact.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import ActivityInterval, CasasRecording, HouseholdSplit
from sensor_modeling.datasets.channel_models import (
    HURDLE,
    HURDLE_NB,
    combine_statistics,
    home_statistics,
)
from sensor_modeling.datasets.dispersion_experiment import (
    CRITERIA,
    MODELS,
    DispersionProtocol,
    DispersionResult,
    accuracy_answer,
    calibration_answer,
    check_frozen_protocol,
    decision,
    declared_protocol,
    dispersion_class,
    extreme_reading,
    log_loss_answer,
    overall_decision,
    recall_answer,
    run_dispersion,
)
from sensor_modeling.datasets.dispersion_summary import render_summary
from sensor_modeling.datasets.information_sets import build_feature_table
from sensor_modeling.datasets.matched_evaluation import _regular_moments
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.restricted_filter import restricted_posteriors
from sensor_modeling.evaluation import load_record
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
PROTOCOL = ROOT / "artifacts" / "phase3" / "dispersion_protocol.json"
ONTOLOGY = StateOntology()
FOLDS = (
    HouseholdSplit("a", train=("sim2", "sim4", "sim6"), test=("sim1", "sim3", "sim5")),
    HouseholdSplit("b", train=("sim1", "sim3", "sim5"), test=("sim2", "sim4", "sim6")),
)


def small_protocol(**changes: Any) -> DispersionProtocol:
    settings: dict[str, Any] = {
        "folds": FOLDS,
        "splits_sha256": "0" * 64,
        "resamples": 200,
    }
    settings.update(changes)
    return DispersionProtocol(**settings)


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


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {f"sim{s}": simulated(s) for s in range(1, 7)}


@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> DispersionResult:
    return run_dispersion(
        homes,
        small_protocol(),
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("dispersion"),
    )


def results_of(outcome: DispersionResult) -> dict[str, Any]:
    results: dict[str, Any] = outcome.record.to_dict()["results"]
    return results


# ----------------------------------------------------------------------------
# The protocol
# ----------------------------------------------------------------------------
class TestProtocol:
    def test_the_committed_protocol_is_the_declared_one(self) -> None:
        declared = declared_protocol(load_frozen_splits(SPLITS))
        check_frozen_protocol(declared, PROTOCOL)
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert frozen["protocol_sha256"] == declared.sha256()

    def test_it_freezes_everything_the_experiment_depends_on(self) -> None:
        declared: dict[str, Any] = declared_protocol(
            load_frozen_splits(SPLITS)
        ).to_dict()
        assert list(declared["models"]) == ["declared", "hurdle", "hurdle_nb"]
        assert set(declared["settings"]) == {"I0", "R"}
        assert [e["key"] for e in declared["estimands"]] == [
            "N0",
            "NR",
            "D0",
            "DR",
            "H0",
            "HR",
        ]
        assert declared["fitting"]["pseudo_windows"] == 12.0
        assert declared["fitting"]["max_dispersion"] == 100.0
        assert declared["per_state_recall"]["focus_state"] == "home_active"
        assert declared["minimal_differences"]["accumulation_slope"] == 0.02
        assert declared["criteria"] == CRITERIA
        assert declared["bootstrap"]["resamples"] == 10_000

    @pytest.mark.parametrize(
        "change",
        [
            {"pseudo_windows": 24.0},
            {"extreme_dispersion": 5.0},
            {"min_active_windows": 50},
            {"accumulation_steps": 24},
            {"resamples": 5_000},
        ],
    )
    def test_any_change_to_the_protocol_is_refused(
        self, change: dict[str, Any]
    ) -> None:
        declared = declared_protocol(load_frozen_splits(SPLITS))
        with pytest.raises(ValueError, match="cannot change after scoring"):
            check_frozen_protocol(replace(declared, **change), PROTOCOL)

    def test_a_tampered_protocol_file_is_refused(self, tmp_path: Path) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        frozen["criteria"]["adopt"] = "log loss favours the negative binomial"
        tampered = tmp_path / "protocol.json"
        tampered.write_text(json.dumps(frozen), encoding="utf-8")
        declared = declared_protocol(load_frozen_splits(SPLITS))
        with pytest.raises(ValueError, match="cannot change after scoring"):
            check_frozen_protocol(declared, tampered)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"splits_sha256": "0"}, "splits_sha256"),
            ({"pseudo_windows": 0.0}, "pseudo_windows"),
            ({"max_dispersion": 50.0}, "MAX_DISPERSION"),
            ({"extreme_dispersion": 100.0}, "below max_dispersion"),
            ({"accumulation_steps": 1}, "accumulation_steps"),
            ({"min_active_windows": 1}, "min_active_windows"),
            ({"minimal_differences": {"balanced_accuracy": 0.02}}, "missing"),
            ({"rare_share": 0.0}, "rare_share"),
        ],
    )
    def test_invalid_protocols_are_refused(
        self, change: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            small_protocol(**change)


# ----------------------------------------------------------------------------
# The rules
# ----------------------------------------------------------------------------
def verdicts(loss: str, accuracy: str = "negligible") -> dict[str, str]:
    return {
        "log_loss": loss,
        "balanced_accuracy": accuracy,
        "calibration_error": "negligible",
        "brier": "negligible",
    }


class TestRules:
    @pytest.mark.parametrize(
        ("verdict", "answer"),
        [
            ("favours model", "recovers"),
            ("negligible", "no change"),
            ("favours reference", "loses more"),
            ("uncertain", "uncertain"),
        ],
    )
    def test_question_one(self, verdict: str, answer: str) -> None:
        assert recall_answer(verdict) == answer

    @pytest.mark.parametrize(
        ("primary", "against_declared", "answer"),
        [
            ("negligible", "favours model", "preserved"),
            ("favours model", "favours model", "preserved"),
            ("uncertain", "favours model", "preserved"),
            ("favours reference", "favours model", "lost"),
            ("negligible", "negligible", "lost"),
            ("negligible", "favours reference", "lost"),
            ("negligible", "uncertain", "uncertain"),
        ],
    )
    def test_question_two(
        self, primary: str, against_declared: str, answer: str
    ) -> None:
        assert calibration_answer(primary, against_declared) == answer

    def test_questions_three_and_four(self) -> None:
        assert log_loss_answer("favours model") == "improves"
        assert log_loss_answer("favours reference") == "worsens"
        assert log_loss_answer("negligible") == "no change"
        assert accuracy_answer("negligible") == "not degraded"
        assert accuracy_answer("favours model") == "not degraded"
        assert accuracy_answer("favours reference") == "degraded"
        assert accuracy_answer("uncertain") == "uncertain"

    @pytest.mark.parametrize(
        ("loss", "accuracy", "recall", "calibration", "expected"),
        [
            ("favours model", "negligible", "negligible", "preserved", "adopt"),
            ("favours model", "favours model", "favours model", "preserved", "adopt"),
            ("favours model", "negligible", "negligible", "lost", "trade-off"),
            (
                "favours model",
                "favours reference",
                "negligible",
                "preserved",
                "trade-off",
            ),
            (
                "favours model",
                "negligible",
                "favours reference",
                "preserved",
                "trade-off",
            ),
            ("favours model", "uncertain", "negligible", "preserved", "inconclusive"),
            ("favours model", "negligible", "negligible", "uncertain", "inconclusive"),
            ("negligible", "favours model", "favours model", "preserved", "reject"),
            (
                "favours reference",
                "favours model",
                "favours model",
                "preserved",
                "reject",
            ),
            ("uncertain", "negligible", "negligible", "preserved", "inconclusive"),
        ],
    )
    def test_the_decision(
        self, loss: str, accuracy: str, recall: str, calibration: str, expected: str
    ) -> None:
        assert decision(verdicts(loss, accuracy), recall, calibration) == expected

    def test_one_improving_metric_is_not_enough(self) -> None:
        # Balanced accuracy and home_active recall both improve, but without a
        # log-loss gain the extra parameter is rejected.
        assert (
            decision(
                verdicts("negligible", "favours model"), "favours model", "preserved"
            )
            == "reject"
        )

    @pytest.mark.parametrize(
        ("decisions", "expected"),
        [
            (("adopt", "adopt"), "adopt"),
            (("reject", "reject"), "reject"),
            (("adopt", "trade-off"), "not adopted"),
            (("adopt", "reject"), "not adopted"),
        ],
    )
    def test_the_overall_decision(
        self, decisions: tuple[str, str], expected: str
    ) -> None:
        assert overall_decision(dict(zip(("I0", "R"), decisions))) == expected

    @pytest.mark.parametrize(
        ("alpha", "active", "flatness", "expected"),
        [
            (0.0, 500, None, "poisson"),
            (3.0, 500, 20.0, "moderate"),
            (50.0, 40, 5.0, "extreme, low data"),
            (50.0, 400, 0.5, "extreme, flat"),
            (100.0, 4000, 30.0, "extreme, supported"),
        ],
    )
    def test_dispersion_classes(
        self, alpha: float, active: int, flatness: float | None, expected: str
    ) -> None:
        assert dispersion_class(alpha, active, flatness, small_protocol()) == expected

    def test_the_reading_of_extreme_estimates(self) -> None:
        rows = [
            {"class": "extreme, low data"},
            {"class": "extreme, low data"},
            {"class": "extreme, supported"},
            {"class": "moderate"},
        ]
        assert extreme_reading(rows)["answer"] == "insufficient data"
        rows.append({"class": "extreme, flat"})
        assert extreme_reading(rows)["answer"] == "not a data shortage"
        assert extreme_reading([{"class": "poisson"}])["answer"] == (
            "no extreme estimates"
        )


# ----------------------------------------------------------------------------
# A run on simulated households
# ----------------------------------------------------------------------------
class TestExperiment:
    def test_the_record_validates(self, result: DispersionResult) -> None:
        assert result.path is not None
        payload = load_record(result.path)
        assert payload["results"]["result_schema"] == "dispersion-results/1"
        assert payload["configuration"]["protocol_sha256"] == small_protocol().sha256()
        assert payload["inference"]["mode"] == "online_filter"

    def test_every_model_is_scored_on_the_same_windows(
        self, result: DispersionResult
    ) -> None:
        metrics = result.record.household_metrics
        results = results_of(result)
        for setting in ("I0", "R"):
            for home, entry in results["households"].items():
                sizes = {metrics[f"{model}@{setting}"][home]["n"] for model in MODELS}
                assert sizes == {entry["labelled"]}

    def test_a_silent_window_scores_alike_under_both_hurdle_families(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        protocol = small_protocol()
        population = combine_statistics(
            {
                h: home_statistics(homes[h], protocol.resolution, ONTOLOGY, household=h)
                for h in FOLDS[0].train
            },
            states=tuple(ONTOLOGY.states),
            pseudo_windows=protocol.pseudo_windows,
        )
        recording = homes["sim1"]
        table = build_feature_table(
            recording,
            protocol.information_set,
            _regular_moments(recording, protocol.resolution.step),
            household="sim1",
        )
        posteriors = {
            family: restricted_posteriors(
                table,
                population.models(
                    recording.registry, protocol.resolution, family, ONTOLOGY
                ),
                ONTOLOGY,
            )
            for family in (HURDLE, HURDLE_NB)
        }
        counts = np.column_stack(
            [
                table.column(name)
                for name in table.columns
                if name.startswith("events_") and name.endswith("_lag0")
            ]
        )
        # Uninstrumented channels have no count; every instrumented one is 0.
        silent = np.all((counts == 0) | np.isnan(counts), axis=1)
        assert silent.any() and not silent.all()
        np.testing.assert_allclose(
            posteriors[HURDLE][silent], posteriors[HURDLE_NB][silent], atol=1e-12
        )
        assert not np.allclose(
            posteriors[HURDLE][~silent], posteriors[HURDLE_NB][~silent]
        )

    def test_the_questions_follow_the_rules(self, result: DispersionResult) -> None:
        results = results_of(result)
        by_key = {e["key"]: e for e in results["estimands"]}
        for setting, (primary, against) in (("I0", ("N0", "D0")), ("R", ("NR", "DR"))):
            answered = results["questions"][setting]
            primary_verdicts = by_key[primary]["verdicts"]
            recall = by_key[primary]["per_state_recall"]["home_active"]["verdict"]
            calibration = calibration_answer(
                primary_verdicts["calibration_error"],
                by_key[against]["verdicts"]["calibration_error"],
            )
            assert answered["home_active_recall"] == recall_answer(recall)
            assert answered["calibration"] == calibration
            assert answered["log_loss"] == log_loss_answer(primary_verdicts["log_loss"])
            assert answered["decision"] == decision(
                primary_verdicts, recall, calibration
            )
        assert results["conclusion"] == overall_decision(
            {s: q["decision"] for s, q in results["questions"].items()}
        )

    def test_quiet_runs_are_compared_in_the_recursion(
        self, result: DispersionResult
    ) -> None:
        results = results_of(result)
        for entry in results["estimands"]:
            assert ("quiet_runs" in entry) == (entry["setting"] == "R")
        assert set(results["quiet_runs"]) == set(MODELS)
        for household in results["households"].values():
            assert set(household["accumulation_slope"]) == set(MODELS)

    def test_the_dispersion_estimates_are_the_fits(
        self, homes: dict[str, CasasRecording], result: DispersionResult
    ) -> None:
        protocol = small_protocol()
        estimates = results_of(result)["dispersion"]["estimates"]
        for fold in FOLDS:
            population = combine_statistics(
                {
                    h: home_statistics(
                        homes[h], protocol.resolution, ONTOLOGY, household=h
                    )
                    for h in fold.train
                },
                states=tuple(ONTOLOGY.states),
                pseudo_windows=protocol.pseudo_windows,
            )
            for row in estimates[fold.name]:
                channel = next(
                    c for c in population.statistics if c.name == row["channel"]
                )
                i = [s.value for s in ONTOLOGY.states].index(row["state"])
                assert row["dispersion"] == pytest.approx(
                    float(population.dispersion(channel)[i])
                )
                assert row["class"] == dispersion_class(
                    row["dispersion"], row["active_windows"], row["flatness"], protocol
                )

    def test_a_second_run_gives_the_same_record(
        self, homes: dict[str, CasasRecording], result: DispersionResult
    ) -> None:
        again = run_dispersion(homes, small_protocol(), data_source="simulator")
        first, second = result.record.to_dict(), again.record.to_dict()
        first.pop("recorded_at")
        second.pop("recorded_at")
        assert first == second

    def test_missing_recordings_are_refused(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        partial = {h: r for h, r in homes.items() if h != "sim2"}
        with pytest.raises(ValueError, match="sim2"):
            run_dispersion(partial, small_protocol(), data_source="simulator")

    def test_the_summary_is_generated_from_the_written_record(
        self, result: DispersionResult
    ) -> None:
        assert result.path is not None
        summary = render_summary(load_record(result.path))
        assert summary == render_summary(
            json.loads(json.dumps(result.record.to_dict()))
        )
        for heading in (
            "## Pre-specified questions",
            "## Every cell",
            "## Estimands",
            "## Per-state recall",
            "## Overconfidence along quiet runs",
            "## Estimated dispersion",
            "## Households",
        ):
            assert heading in summary
        payload = result.record.to_dict()
        payload["results"] = {"result_schema": "fitted-rates-results/1"}
        with pytest.raises(ValueError, match="dispersion-results"):
            render_summary(payload)
