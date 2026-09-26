"""Tests for the pre-specified explicit-history evaluation.

They guard what makes its result interpretable:
- the protocol that ran is the one frozen before scoring;
- every comparison is either two models on one set, or one family across
  nested sets, never a change of model and information at once;
- H1 is exactly the difference between the two history gains;
- priors and coefficients see only training households;
- the record and its summary are reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    GradientBoostingBaseline,
    HouseholdSplit,
    InformationComponent,
    ModelSpec,
    nested_information_sets,
)
from sensor_modeling.datasets.history_experiment import (
    CELLS,
    COMBINED,
    CRITERIA,
    ESTIMANDS,
    FAMILIES,
    HISTORY,
    ORIGINAL,
    PERIODIC,
    HistoryProtocol,
    HistoryResult,
    check_frozen_protocol,
    declared_protocol,
    fold_models,
    run_history_experiment,
)
from sensor_modeling.datasets.history_summary import render_summary
from sensor_modeling.datasets.recoverable_gap import gap_models, load_frozen_splits
from sensor_modeling.datasets.time_prior_experiment import conclusion
from sensor_modeling.evaluation import load_record
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
PROTOCOL = ROOT / "artifacts" / "phase3" / "history_protocol.json"
PUBLISHED = ROOT / "artifacts" / "phase3" / "phase3-explicit-history.json"
DOC = ROOT / "docs" / "PHASE3_HISTORY_STATE.md"
SETS = dict(zip(("I0", "I1", "I2", "I3"), nested_information_sets()))
FOLDS = (
    HouseholdSplit("a", train=("sim2", "sim4"), test=("sim1", "sim3")),
    HouseholdSplit("b", train=("sim1", "sim3"), test=("sim2", "sim4")),
)
#: Each model paired with the same model without the history state.
WITHOUT_HISTORY = {
    HISTORY: ORIGINAL,
    COMBINED: PERIODIC,
    ORIGINAL: ORIGINAL,
    PERIODIC: PERIODIC,
}


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


def protocol(**changes: Any) -> HistoryProtocol:
    """The declared protocol with a cheaper diagnostic and fewer resamples."""
    reference, _, logistic = gap_models()
    settings: dict[str, Any] = {
        "folds": FOLDS,
        "splits_sha256": "0" * 64,
        "models": (
            reference,
            ModelSpec(
                "diagnostic",
                lambda seed: GradientBoostingBaseline(max_iter=10, seed=seed),
                GradientBoostingBaseline(max_iter=10).configuration(),
            ),
            logistic,
        ),
        "resamples": 100,
    }
    settings.update(changes)
    return HistoryProtocol(**settings)


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {f"sim{s}": simulated(s) for s in range(1, 5)}


@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[HistoryResult]:
    with threadpool_limits(1):
        yield run_history_experiment(
            homes,
            protocol(),
            data_source="simulator",
            output_dir=tmp_path_factory.mktemp("history"),
        )


def results_of(outcome: HistoryResult) -> dict[str, Any]:
    return dict(outcome.record.to_dict()["results"])


def relabelled(recording: CasasRecording) -> CasasRecording:
    return CasasRecording(
        recording.registry,
        recording.observations,
        tuple(
            ActivityInterval("Sleep", a.start, a.end, S.SLEEPING)
            for a in recording.activities
        ),
    )


class TestFrozenProtocol:
    def test_the_committed_protocol_is_the_declared_one(self) -> None:
        splits = load_frozen_splits(SPLITS)
        declared = declared_protocol(splits)
        check_frozen_protocol(declared, PROTOCOL)
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert frozen["protocol_sha256"] == declared.sha256()
        assert frozen["households"]["splits_sha256"] == splits.sha256

    def test_it_freezes_everything_the_experiment_depends_on(self) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert set(frozen["models"]) == {
            "state_frequency",
            "diagnostic",
            "logistic",
            ORIGINAL,
            HISTORY,
            PERIODIC,
            COMBINED,
        }
        assert frozen["models"][HISTORY]["history_state"]["window_steps"] == 3
        assert (
            frozen["models"][COMBINED]["periodic_prior"]["household_precision"] == 24.0
        )
        assert "declared here" in frozen["inclusion"][COMBINED]
        assert frozen["bootstrap"]["statistics"] == "mean and median paired differences"
        assert frozen["bootstrap"]["resamples"] == 10_000
        assert frozen["seeds"] == {"models": 0, "bootstrap": 0, "row_order": 0}
        assert frozen["minimal_differences"]["balanced_accuracy"] == 0.02
        assert [e["key"] for e in frozen["estimands"] if e["role"] == "primary"] == [
            "H1",
            "H2",
            "H3",
            "H4",
        ]
        assert set(frozen["interactions"]) == set(FAMILIES)

    def test_the_earlier_gain_is_not_a_threshold(self) -> None:
        assert "0.021" not in CRITERIA["success"]
        assert "not a threshold" in CRITERIA["materially_more"]

    @pytest.mark.parametrize(
        "change", [{"seed": 1}, {"rare_share": 0.1}, {"resamples": 5000}]
    )
    def test_any_change_to_the_protocol_is_refused(
        self, change: dict[str, Any]
    ) -> None:
        changed = replace(declared_protocol(load_frozen_splits(SPLITS)), **change)
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(changed, PROTOCOL)

    def test_a_tampered_protocol_file_is_refused(self, tmp_path: Path) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        frozen["criteria"]["success"] = "anything"
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
            (
                {"minimal_differences": {"balanced_accuracy": 0.02}},
                "minimal difference",
            ),
            ({"rare_share": 1.0}, "rare_share"),
            ({"folds": FOLDS[:1]}, "two folds"),
        ],
    )
    def test_invalid_protocols_are_refused(
        self, change: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            protocol(**change)

    def test_the_history_window_must_fit_the_sets(self) -> None:
        from sensor_modeling.fusion import HistoryConfig

        with pytest.raises(ValueError, match="history window"):
            protocol(history=HistoryConfig(window_steps=4))


class TestComparisons:
    def test_no_comparison_changes_model_and_information_at_once(self) -> None:
        for estimand in ESTIMANDS:
            if estimand.kind == "formulation":
                assert estimand.model_set == estimand.reference_set, estimand.key
                continue
            assert estimand.kind == "information", estimand.key
            assert WITHOUT_HISTORY[estimand.model] == estimand.reference, estimand.key
            added = (
                SETS[estimand.model_set].components
                - SETS[estimand.reference_set].components
            )
            assert added == {InformationComponent.RECENT_HISTORY}, estimand.key

    def test_every_estimand_cell_is_scored(self) -> None:
        for estimand in ESTIMANDS:
            assert estimand.model_set in CELLS[estimand.model]
            assert estimand.reference_set in CELLS[estimand.reference]


class TestNoLeakage:
    def test_fold_models_see_only_training_households(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        states = tuple(StateOntology().states)
        fitted = fold_models(homes, protocol(), states)
        for fold in FOLDS:
            assert fitted[fold.name]["periodic"].fitted_on == tuple(sorted(fold.train))
            assert fitted[fold.name]["history"].fitted_on == tuple(sorted(fold.train))
        changed = {
            home: relabelled(r) if home in FOLDS[0].test else r
            for home, r in homes.items()
        }
        again = fold_models(changed, protocol(), states)
        for kind in ("periodic", "history"):
            assert again["a"][kind].sha256() == fitted["a"][kind].sha256()
        assert again["b"]["history"].sha256() != fitted["b"]["history"].sha256()

    def test_no_household_is_scored_with_models_fitted_on_it(
        self, result: HistoryResult
    ) -> None:
        results = results_of(result)
        for home, entry in results["households"].items():
            fitted = results["fitted"][entry["fold"]]
            assert home not in fitted["periodic_prior"]["fitted_on"]
            assert home not in fitted["history_model"]["fitted_on"]


class TestExperiment:
    def test_the_record_validates(self, result: HistoryResult) -> None:
        assert result.path is not None
        payload = load_record(result.path)
        assert payload["results"]["result_schema"] == "history-results/1"
        assert payload["configuration"]["protocol_sha256"] == protocol().sha256()

    def test_every_declared_cell_is_scored(self, result: HistoryResult) -> None:
        cells = {
            (c["model"], c["information_set"]) for c in results_of(result)["cells"]
        }
        assert cells == {(m, s) for m, sets in CELLS.items() for s in sets}

    def test_h1_is_exactly_the_difference_of_the_two_history_gains(
        self, result: HistoryResult
    ) -> None:
        estimands = {e["key"]: e for e in results_of(result)["estimands"]}
        for metric in ("balanced_accuracy", "log_loss"):
            h1 = estimands["H1"]["comparisons"][metric]["differences"]
            h2 = estimands["H2"]["comparisons"][metric]["differences"]
            r1 = estimands["R1"]["comparisons"][metric]["differences"]
            for home in h1:
                assert h1[home] == pytest.approx(h2[home] - r1[home]), (metric, home)

    def test_every_estimand_reports_mean_and_median(
        self, result: HistoryResult
    ) -> None:
        for entry in results_of(result)["estimands"]:
            for comparison in entry["comparisons"].values():
                assert comparison["mean"]["interval"] is not None
                assert comparison["median"]["interval"] is not None
                assert (
                    comparison["favours_model"]
                    + comparison["favours_reference"]
                    + (comparison["tied"])
                    == comparison["n"]
                )

    def test_interactions_are_differences_of_history_gains(
        self, result: HistoryResult
    ) -> None:
        metrics = result.record.household_metrics
        entry = next(
            i
            for i in results_of(result)["interactions"]
            if i["family"] == "generative with the history state"
        )

        def score(model: str, label: str, home: str) -> float:
            return float(metrics[f"{model}@{label}"][home]["balanced_accuracy"])

        for home, difference in entry["comparisons"]["balanced_accuracy"][
            "differences"
        ].items():
            with_hour = score(COMBINED, "I3", home) - score(PERIODIC, "I1", home)
            without = score(HISTORY, "I2", home) - score(ORIGINAL, "I0", home)
            assert difference == pytest.approx(with_hour - without)

    def test_conclusions_follow_the_rules(self, result: HistoryResult) -> None:
        results = results_of(result)
        estimands = {e["key"]: e for e in results["estimands"]}
        assert results["conclusions"]["explicit_history"] == conclusion(
            estimands["H1"]["verdicts"]
        )
        assert results["conclusions"]["with_the_hour"] == conclusion(
            estimands["S1"]["verdicts"]
        )
        recall = {e["key"] for e in results["estimands"] if "per_state_recall" in e}
        assert recall == {"H1", "H2", "S1"}

    def test_a_second_run_gives_the_same_record(
        self, homes: dict[str, CasasRecording], result: HistoryResult
    ) -> None:
        with threadpool_limits(1):
            again = run_history_experiment(homes, protocol(), data_source="simulator")
        first, second = result.record.to_dict(), again.record.to_dict()
        first.pop("recorded_at")
        second.pop("recorded_at")
        assert first == second

    def test_the_summary_is_generated_from_the_written_record(
        self, result: HistoryResult
    ) -> None:
        assert result.path is not None
        written = load_record(result.path)
        summary = render_summary(written)
        assert summary == render_summary(
            json.loads(json.dumps(result.record.to_dict()))
        )
        for heading in (
            "## Pre-specified conclusions",
            "## Every cell",
            "## Estimands: balanced accuracy",
            "## Estimands: probability quality",
            "## Per-state recall",
            "## Time and history interaction",
        ):
            assert heading in summary
        payload = result.record.to_dict()
        payload["results"] = {"result_schema": "time-prior-results/1"}
        with pytest.raises(ValueError, match="history-results"):
            render_summary(payload)
