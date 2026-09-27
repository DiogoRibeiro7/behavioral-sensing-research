"""Tests for the pre-specified hierarchical time-of-day prior experiment.

They guard what makes its result trustworthy:
- the protocol that ran is the one frozen before scoring;
- the time-disabled model is exactly the hierarchical model without the hour;
- priors see only training households, and the adaptation arm only labels
  from before the moments it scores;
- the verdicts follow the declared rules;
- the record and its summary are reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    GradientBoostingBaseline,
    HouseholdSplit,
    ModelSpec,
    build_feature_table,
    fit_periodic_prior,
    hour_state_counts,
    nested_information_sets,
)
from sensor_modeling.datasets.matched_evaluation import _regular_moments
from sensor_modeling.datasets.recoverable_gap import gap_models, load_frozen_splits
from sensor_modeling.datasets.restricted_filter import (
    channel_likelihoods,
    restricted_posteriors,
)
from sensor_modeling.datasets.time_prior_experiment import (
    CELLS,
    ESTIMANDS,
    TimePriorProtocol,
    TimePriorResult,
    check_frozen_protocol,
    conclusion,
    declared_protocol,
    fold_models,
    hierarchy_conclusion,
    hour_collapsed,
    run_time_prior_experiment,
    untimed_ontology,
    verdict,
)
from sensor_modeling.datasets.time_prior_summary import render_summary
from sensor_modeling.evaluation import load_record
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
PROTOCOL = ROOT / "artifacts" / "phase3" / "time_prior_protocol.json"
PUBLISHED = ROOT / "artifacts" / "phase3" / "phase3-hierarchical-time-prior.json"
DOC = ROOT / "docs" / "PHASE3_TIME_PRIOR.md"
I0, I1, I2, I3 = nested_information_sets()
STEP = timedelta(minutes=5)
DIGEST = "0" * 64
FOLDS = (
    HouseholdSplit("a", train=("sim2", "sim4"), test=("sim1", "sim3")),
    HouseholdSplit("b", train=("sim1", "sim3"), test=("sim2", "sim4")),
)


def simulated(seed: int, days: int = 3) -> CasasRecording:
    sim = simulate(HouseholdConfig(days=days, seed=seed))
    return CasasRecording(
        sim.registry,
        sim.observations,
        tuple(
            ActivityInterval(e.state.value, e.start, e.end, e.state)
            for e in sim.truth.episodes
        ),
    )


def relabelled(recording: CasasRecording) -> CasasRecording:
    return CasasRecording(
        recording.registry,
        recording.observations,
        tuple(
            ActivityInterval("Sleep", a.start, a.end, S.SLEEPING)
            for a in recording.activities
        ),
    )


def protocol(**changes: Any) -> TimePriorProtocol:
    """The declared protocol with a cheaper diagnostic and a short adaptation window."""
    reference, _, logistic = gap_models()
    settings: dict[str, Any] = {
        "folds": FOLDS,
        "splits_sha256": DIGEST,
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
        "adaptation_days": 1.0,
    }
    settings.update(changes)
    return TimePriorProtocol(**settings)


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {f"sim{s}": simulated(s) for s in range(1, 5)}


@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[TimePriorResult]:
    with threadpool_limits(1):
        yield run_time_prior_experiment(
            homes,
            protocol(),
            data_source="simulator",
            output_dir=tmp_path_factory.mktemp("time-prior"),
        )


def results_of(outcome: TimePriorResult) -> dict[str, Any]:
    return dict(outcome.record.to_dict()["results"])


def comparison(mean: float, low: float | None, high: float | None) -> dict[str, Any]:
    interval = None if low is None else {"low": low, "high": high}
    return {"mean": {"estimate": mean, "interval": interval}}


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
        assert [f["name"] for f in frozen["households"]["folds"]] == [
            "phase1-fold-a",
            "phase1-fold-b",
        ]
        assert set(frozen["information_sets"]) == {"I0", "I1", "I2", "I3"}
        models = frozen["models"]
        assert set(models) == {
            "state_frequency",
            "diagnostic",
            "logistic",
            "generative",
            "generative_untimed",
            "generative_periodic",
        }
        assert models["diagnostic"]["max_iter"] == 100
        assert models["logistic"]["C"] == 1.0
        prior = models["generative_periodic"]["periodic_prior"]
        assert (prior["harmonics"], prior["household_precision"]) == (2, 24.0)
        assert frozen["pooling"]["household_precision"] == 24.0
        assert "not selected" in frozen["pooling"]["selection"]
        assert frozen["metrics"] == [
            "balanced_accuracy",
            "log_loss",
            "brier",
            "calibration_error",
        ]
        assert frozen["bootstrap"]["resamples"] == 10_000
        assert frozen["seeds"] == {"models": 0, "bootstrap": 0, "row_order": 0}
        assert [e["key"] for e in frozen["estimands"] if e["role"] == "primary"] == [
            "P1",
            "P2",
            "P3",
            "P4",
        ]
        assert {"success", "failure", "inconclusive", "verdicts"} <= set(
            frozen["criteria"]
        )
        assert frozen["minimal_differences"]["balanced_accuracy"] == 0.02
        assert "p <" not in json.dumps(frozen["criteria"]).lower()

    @pytest.mark.parametrize(
        "change",
        [{"seed": 1}, {"adaptation_days": 14.0}, {"sensitivity_precisions": (2.4,)}],
    )
    def test_any_change_to_the_protocol_is_refused(
        self, change: dict[str, Any]
    ) -> None:
        splits = load_frozen_splits(SPLITS)
        changed = replace(declared_protocol(splits), **change)
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(changed, PROTOCOL)

    def test_a_tampered_protocol_file_is_refused(self, tmp_path: Path) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        frozen["minimal_differences"]["balanced_accuracy"] = 0.001
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
            ({"rare_share": 0.0}, "rare_share"),
            ({"adaptation_days": 0.0}, "adaptation_days"),
            ({"sensitivity_precisions": (0.0,)}, "sensitivity"),
            ({"folds": FOLDS[:1]}, "two folds"),
        ],
    )
    def test_invalid_protocols_are_refused(
        self, change: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            protocol(**change)


class TestVerdicts:
    @pytest.mark.parametrize(
        ("mean", "low", "high", "expected"),
        [
            (0.05, 0.01, 0.09, "favours model"),
            (0.02, 0.001, 0.04, "favours model"),
            (0.03, -0.01, 0.07, "uncertain"),
            (0.01, 0.002, 0.018, "negligible"),
            (-0.001, -0.015, 0.012, "negligible"),
            (-0.05, -0.08, -0.02, "favours reference"),
            (-0.019, -0.03, -0.005, "uncertain"),
            (0.05, None, None, "uncertain"),
        ],
    )
    def test_the_declared_rule(
        self, mean: float, low: float | None, high: float | None, expected: str
    ) -> None:
        assert verdict(comparison(mean, low, high), 0.02) == expected

    def test_the_success_and_failure_rules(self) -> None:
        good = {"balanced_accuracy": "favours model", "log_loss": "negligible"}
        assert conclusion({**good, "calibration_error": "uncertain"}) == "success"
        assert (
            conclusion({**good, "calibration_error": "favours reference"})
            == "inconclusive"
        )
        assert conclusion({"balanced_accuracy": "negligible"}) == "failure"
        assert conclusion({"balanced_accuracy": "favours reference"}) == "failure"
        assert conclusion({"balanced_accuracy": "uncertain"}) == "inconclusive"
        assert hierarchy_conclusion(good) == "helps"
        assert (
            hierarchy_conclusion({"balanced_accuracy": "negligible"}) == "does not help"
        )
        assert (
            hierarchy_conclusion(
                {"balanced_accuracy": "favours model", "log_loss": "favours reference"}
            )
            == "inconclusive"
        )


class TestTimeDisabledModel:
    def test_pooling_over_hours_removes_the_hour(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        counts = hour_state_counts(
            homes["sim1"], _regular_moments(homes["sim1"], STEP), step=STEP
        )
        flat = hour_collapsed(counts)
        np.testing.assert_allclose(flat.sum(axis=0), counts.sum(axis=0))
        np.testing.assert_allclose(flat, flat[0][None, :].repeat(24, axis=0))
        prior = fit_periodic_prior({"a": flat})
        assert np.abs(prior.population[:, 1:]).max() < 1e-10

    def test_it_is_the_hierarchical_model_without_the_hour(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        counts = {
            h: hour_collapsed(
                hour_state_counts(r, _regular_moments(r, STEP), step=STEP)
            )
            for h, r in homes.items()
            if h != "sim1"
        }
        flat = fit_periodic_prior(counts)
        base = StateOntology()
        untimed = untimed_ontology(flat, base)
        np.testing.assert_allclose(untimed.stationary(), flat.hourly()[0], atol=1e-9)
        recording = homes["sim1"]
        moments = _regular_moments(recording, STEP)
        terms, _ = channel_likelihoods(recording.registry, I1.resolution)
        without_hour = restricted_posteriors(
            build_feature_table(recording, I0, moments, household="h"), terms, untimed
        )
        with_flat_hour = restricted_posteriors(
            build_feature_table(recording, I1, moments, household="h"),
            terms,
            base,
            periodic_prior=flat,
        )
        np.testing.assert_allclose(without_hour, with_flat_hour, atol=1e-8)

    def test_an_hour_dependent_prior_is_refused(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        recording = homes["sim2"]
        timed = fit_periodic_prior(
            {
                "a": hour_state_counts(
                    recording, _regular_moments(recording, STEP), step=STEP
                )
            }
        )
        with pytest.raises(ValueError, match="depends on the hour"):
            untimed_ontology(timed, StateOntology())


class TestNoLeakage:
    def test_fold_priors_see_only_training_households(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        states = tuple(StateOntology().states)
        fitted = fold_models(homes, protocol(), states)
        for fold in FOLDS:
            for kind in ("periodic", "untimed"):
                assert fitted[fold.name][kind].fitted_on == tuple(sorted(fold.train))
        changed = {
            home: relabelled(r) if home in FOLDS[0].test else r
            for home, r in homes.items()
        }
        again = fold_models(changed, protocol(), states)
        for kind in ("periodic", "untimed"):
            assert again["a"][kind].sha256() == fitted["a"][kind].sha256()
        assert again["a"]["shares"] == fitted["a"]["shares"]

    def test_no_household_is_scored_with_a_prior_fitted_on_it(
        self, result: TimePriorResult
    ) -> None:
        results = results_of(result)
        for home, entry in results["households"].items():
            for kind in ("periodic", "untimed"):
                assert home not in results["priors"][entry["fold"]][kind]["fitted_on"]

    def test_the_adaptation_arm_scores_only_after_its_window(
        self, homes: dict[str, CasasRecording], result: TimePriorResult
    ) -> None:
        results = results_of(result)
        metrics = result.record.household_metrics
        for home, entry in results["households"].items():
            recording = homes[home]
            moments = _regular_moments(recording, STEP)
            cutoff = min(o.timestamp for o in recording.observations) + timedelta(
                days=1
            )
            assert entry["adaptation_cutoff"] == cutoff.isoformat()
            own = hour_state_counts(recording, moments, step=STEP, until=cutoff)
            assert entry["adaptation_labelled_hours"] == pytest.approx(own.sum())
            for variant in ("population", "adapted@24", "adapted@2.4", "adapted@240"):
                assert (
                    metrics[f"adaptation/{variant}@I1"][home]["n"]
                    == entry["scored_after_adaptation"]
                )
            assert entry["scored_after_adaptation"] < entry["labelled"]


class TestExperiment:
    def test_the_record_validates(self, result: TimePriorResult) -> None:
        assert result.path is not None
        payload = load_record(result.path)
        assert payload["results"]["result_schema"] == "time-prior-results/1"
        assert payload["configuration"]["protocol_sha256"] == protocol().sha256()

    def test_every_declared_cell_is_scored(self, result: TimePriorResult) -> None:
        cells = {
            (c["model"], c["information_set"]) for c in results_of(result)["cells"]
        }
        assert cells == {(m, s) for m, sets in CELLS.items() for s in sets}
        households = results_of(result)["households"]
        for key, per_home in result.record.household_metrics.items():
            if key.startswith("adaptation/"):
                continue
            assert set(per_home) == set(households), key

    def test_every_estimand_has_its_comparisons_and_verdicts(
        self, result: TimePriorResult
    ) -> None:
        estimands = results_of(result)["estimands"]
        assert [e["key"] for e in estimands] == [e.key for e in ESTIMANDS]
        for entry in estimands:
            assert set(entry["comparisons"]) == {
                "balanced_accuracy",
                "log_loss",
                "brier",
                "calibration_error",
            }
            assert set(entry["verdicts"]) == set(entry["comparisons"])
        recall = {e["key"]: e for e in estimands if "per_state_recall" in e}
        assert set(recall) == {"P1", "P2"}
        states = results_of(result)["states"]
        assert set(recall["P1"]["per_state_recall"]) == set(states)

    def test_p1_is_the_per_household_difference(self, result: TimePriorResult) -> None:
        metrics = result.record.household_metrics
        p1 = next(e for e in results_of(result)["estimands"] if e["key"] == "P1")
        for home, difference in p1["comparisons"]["balanced_accuracy"][
            "differences"
        ].items():
            expected = (
                metrics["generative_periodic@I1"][home]["balanced_accuracy"]
                - metrics["generative_untimed@I0"][home]["balanced_accuracy"]
            )
            assert difference == pytest.approx(expected)

    def test_conclusions_follow_the_rules(self, result: TimePriorResult) -> None:
        results = results_of(result)
        estimands = {e["key"]: e for e in results["estimands"]}
        assert results["conclusions"]["time_of_day"] == conclusion(
            estimands["P1"]["verdicts"]
        )
        declared = next(
            a
            for a in results["adaptation"]
            if a["declared"] and a["information_set"] == "I1"
        )
        assert results["conclusions"]["household_adaptation"] == hierarchy_conclusion(
            declared["verdicts"]
        )

    def test_a_second_run_gives_the_same_record(
        self, homes: dict[str, CasasRecording], result: TimePriorResult
    ) -> None:
        with threadpool_limits(1):
            again = run_time_prior_experiment(
                homes, protocol(), data_source="simulator"
            )
        first, second = result.record.to_dict(), again.record.to_dict()
        first.pop("recorded_at")
        second.pop("recorded_at")
        assert first == second

    def test_the_summary_is_generated_from_the_written_record(
        self, result: TimePriorResult
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
            "## Household adaptation",
        ):
            assert heading in summary
        assert render_summary(written, level=3).startswith("### ")
        payload = result.record.to_dict()
        payload["results"] = {"result_schema": "recoverable-gap/1"}
        with pytest.raises(ValueError, match="time-prior-results"):
            render_summary(payload)


class TestPublishedResult:
    """The published development-panel result, and the page that reports it."""

    def test_it_ran_the_frozen_protocol_on_a_clean_tree(self) -> None:
        payload = load_record(PUBLISHED)
        splits = load_frozen_splits(SPLITS)
        declared = declared_protocol(splits)
        assert payload["configuration"]["protocol_sha256"] == declared.sha256()
        assert payload["environment"]["git_dirty"] == "false"
        inputs = {item["name"]: item["sha256"] for item in payload["inputs"]}
        assert inputs.pop(PROTOCOL.name) == check_frozen_protocol(declared, PROTOCOL)
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
