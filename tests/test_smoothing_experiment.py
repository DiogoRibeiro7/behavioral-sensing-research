"""Tests for the pre-specified fixed-lag smoothing experiment (Phase 3.5).

They guard what makes its result interpretable:
- the protocol that ran is the one frozen before scoring;
- every regime is scored on the same windows, and each smoother reads exactly
  its lag;
- every comparison is a labelled smoothing gain, never an online one, and the
  record states the longest lag, which bounds every estimate in it;
- the operational and transition measures count what they say;
- conclusions follow the declared rule, and the record and its summary are
  reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import ActivityInterval, CasasRecording, HouseholdSplit
from sensor_modeling.datasets.channel_models import filter_recursion
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.smoothing_experiment import (
    CRITERIA,
    ONLINE_KEY,
    REPRODUCED,
    SmoothingProtocol,
    SmoothingResult,
    Transition,
    boundary_mask,
    change_shares,
    check_frozen_protocol,
    decision_delays,
    declared_protocol,
    find_transitions,
    regime_key,
    regime_posteriors,
    run_smoothing,
    smoothing_conclusion,
    state_changes,
)
from sensor_modeling.datasets.smoothing_summary import render_summary
from sensor_modeling.evaluation import load_record
from sensor_modeling.fusion import (
    ONLINE,
    EvidenceLeakageError,
    InferenceRegime,
    assert_respects_horizon,
    regime_beliefs,
)
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
PROTOCOL = ROOT / "artifacts" / "phase3" / "smoothing_protocol.json"
ONTOLOGY = StateOntology()
STEP_SECONDS = 300.0
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


def small_protocol(**changes: Any) -> SmoothingProtocol:
    """The declared protocol on simulated folds, with fewer resamples."""
    settings: dict[str, Any] = {
        "folds": FOLDS,
        "splits_sha256": "0" * 64,
        "resamples": 200,
    }
    settings.update(changes)
    return SmoothingProtocol(**settings)


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {f"sim{s}": simulated(s) for s in range(1, 7)}


@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[SmoothingResult]:
    yield run_smoothing(
        homes,
        small_protocol(),
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("smoothing"),
    )


def results_of(outcome: SmoothingResult) -> dict[str, Any]:
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
        assert declared["formulation"]["model"] == (
            "the Phase 3.3 follow-up's filter_hurdle recursion"
        )
        assert declared["formulation"]["pseudo_windows"] == 12.0
        assert declared["lags"]["windows"] == [1, 6, 12]
        regimes = declared["regimes"]
        assert list(regimes) == ["online", "lag_1", "lag_6", "lag_12"]
        assert regimes["online"]["causal"] is True
        assert (
            regimes["lag_12"]["label"] == "fixed-lag smoother, lag 12 windows (60 min)"
        )
        assert regimes["lag_12"]["delay_seconds"] == 3600.0
        assert all(not regimes[k]["causal"] for k in ("lag_1", "lag_6", "lag_12"))
        assert [e["key"] for e in declared["estimands"]] == ["G1", "G6", "G12"]
        assert {e["kind"] for e in declared["estimands"]} == {"smoothing gain"}
        assert declared["transitions"]["boundary_windows"] == 6
        assert declared["per_state_recall"]["focus_states"] == [
            "away",
            "home_active",
            "sleeping",
        ]
        assert declared["bootstrap"]["resamples"] == 10_000
        assert declared["criteria"] == CRITERIA
        assert set(declared["minimal_differences"]) == {
            "balanced_accuracy",
            "log_loss",
            "brier",
            "calibration_error",
            "recall",
            "boundary_accuracy",
            "detection_rate",
            "decision_delay_minutes",
        }

    @pytest.mark.parametrize(
        "change",
        [
            {"lags": (1, 6)},
            {"lags": (1, 6, 12), "boundary_windows": 3},
            {"pseudo_windows": 24.0},
            {"resamples": 5_000},
            {"focus_states": (S.AWAY, S.SLEEPING)},
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
        frozen["lags"]["windows"] = [1, 3, 12]
        tampered = tmp_path / "protocol.json"
        tampered.write_text(json.dumps(frozen), encoding="utf-8")
        declared = declared_protocol(load_frozen_splits(SPLITS))
        with pytest.raises(ValueError, match="cannot change after scoring"):
            check_frozen_protocol(declared, tampered)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"lags": ()}, "distinct increasing"),
            ({"lags": (0, 1)}, "distinct increasing"),
            ({"lags": (6, 1)}, "distinct increasing"),
            ({"lags": (1, 1)}, "distinct increasing"),
            ({"lags": (True,)}, "distinct increasing"),
            ({"lags": (2,)}, "no declared operational role"),
            ({"boundary_windows": 0}, "boundary_windows"),
            ({"focus_states": ()}, "focus_states"),
            ({"focus_states": (S.AWAY, S.AWAY)}, "focus_states"),
            ({"pseudo_windows": 0.0}, "pseudo_windows"),
            ({"minimal_differences": {"balanced_accuracy": 0.02}}, "missing"),
            ({"rare_share": 1.0}, "rare_share"),
            ({"splits_sha256": "0"}, "splits_sha256"),
        ],
    )
    def test_invalid_protocols_are_refused(
        self, change: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            small_protocol(**change)

    def test_the_regimes_are_the_filter_and_one_smoother_per_lag(self) -> None:
        protocol = small_protocol()
        assert protocol.regimes == {
            "online": ONLINE,
            "lag_1": InferenceRegime.smoother(1, protocol.resolution.step),
            "lag_6": InferenceRegime.smoother(6, protocol.resolution.step),
            "lag_12": InferenceRegime.smoother(12, protocol.resolution.step),
        }
        assert regime_key(0) == ONLINE_KEY and regime_key(6) == "lag_6"


# ----------------------------------------------------------------------------
# The rule
# ----------------------------------------------------------------------------
class TestRule:
    @pytest.mark.parametrize(
        ("accuracy", "loss", "calibration", "expected"),
        [
            ("favours model", "favours model", "favours model", "gain"),
            ("favours model", "uncertain", "negligible", "gain"),
            ("favours model", "favours reference", "favours model", "trade-off"),
            ("favours model", "favours model", "favours reference", "trade-off"),
            ("negligible", "favours model", "negligible", "probability gain"),
            ("uncertain", "favours model", "uncertain", "probability gain"),
            ("favours reference", "favours model", "negligible", "inconclusive"),
            ("negligible", "favours model", "favours reference", "inconclusive"),
            ("negligible", "negligible", "favours model", "no gain"),
            ("favours reference", "favours reference", "uncertain", "no gain"),
            ("uncertain", "uncertain", "uncertain", "inconclusive"),
            ("negligible", "uncertain", "negligible", "inconclusive"),
        ],
    )
    def test_the_conclusion_rule(
        self, accuracy: str, loss: str, calibration: str, expected: str
    ) -> None:
        verdicts = {
            "balanced_accuracy": accuracy,
            "log_loss": loss,
            "calibration_error": calibration,
            "brier": "uncertain",
        }
        assert smoothing_conclusion(verdicts) == expected


# ----------------------------------------------------------------------------
# What is counted
# ----------------------------------------------------------------------------
class TestCounting:
    def test_state_changes_count_corrections_and_breakages(self) -> None:
        truth = np.array([0, 0, 1, 1, 2, 2])
        online = np.array([0, 1, 1, 0, 2, 0])
        smoothed = np.array([0, 0, 0, 1, 2, 1])
        counts = state_changes(truth, online, smoothed)
        assert counts == {
            "windows": 6,
            "changed": 4,
            "online_right": 3,
            "online_wrong": 3,
            "corrected": 2,
            "broken": 1,
        }
        assert change_shares(counts) == {
            "changed": 4 / 6,
            "corrected": 2 / 3,
            "broken": 1 / 3,
            "corrections_among_changes": 2 / 4,
        }

    def test_shares_without_a_denominator_are_missing(self) -> None:
        empty = np.array([], dtype=int)
        assert set(change_shares(state_changes(empty, empty, empty)).values()) == {None}

    def test_transitions_need_two_consecutive_labelled_windows(self) -> None:
        truth = np.array([0, 0, 1, 1, 1, -1, 2, 2, 0, 0, 0, 0])
        found = find_transitions(truth, last=9)
        assert found == [
            Transition(row=2, source=0, target=1, end=4),
            Transition(row=8, source=2, target=0, end=9),
        ]
        assert find_transitions(truth, last=7) == [Transition(2, 0, 1, 4)]

    def test_the_boundary_is_the_windows_either_side(self) -> None:
        mask = boundary_mask(12, [Transition(5, 0, 1, 9)], 2)
        assert np.flatnonzero(mask).tolist() == [3, 4, 5, 6]
        assert boundary_mask(4, [Transition(1, 0, 1, 3)], 3).all()

    def test_decision_delays_add_the_lag_and_miss_what_never_arrives(self) -> None:
        predicted = np.array([0, 0, 0, 1, 1, 1, 0, 0, 2, 2])
        found = [Transition(2, 0, 1, 5), Transition(6, 1, 2, 7)]
        assert decision_delays(predicted, found, 0) == [1, None]
        assert decision_delays(predicted, found, 3) == [4, None]
        assert decision_delays(predicted, [Transition(3, 0, 1, 5)], 1) == [1]


# ----------------------------------------------------------------------------
# The regimes
# ----------------------------------------------------------------------------
TRANSITION = np.array([[0.95, 0.05], [0.05, 0.95]])
PRIOR = np.array([0.5, 0.5])


def loglik() -> np.ndarray:
    """B, an ambiguous turn, a decisive A, then 23 windows that keep changing.

    Long enough that even the longest lag leaves several later windows to
    perturb.
    """
    rows = [np.log([0.3, 0.7])] * 5 + [np.zeros(2), np.log([0.99, 0.01])]
    rows += [np.log([0.7, 0.3] if k % 4 < 2 else [0.2, 0.8]) for k in range(23)]
    return np.array(rows)


class TestRegimePosteriors:
    def test_one_filter_pass_serves_every_regime(self) -> None:
        protocol = small_protocol()
        found = regime_posteriors(loglik(), TRANSITION, PRIOR, protocol.regimes)
        filtered = filter_recursion(loglik(), TRANSITION, PRIOR)[1]
        np.testing.assert_array_equal(found[ONLINE_KEY], filtered)
        for key, regime in protocol.regimes.items():
            np.testing.assert_array_equal(
                found[key], regime_beliefs(filtered, TRANSITION, regime)
            )

    @pytest.mark.parametrize("lag", [0, 1, 6, 12])
    def test_each_regime_reads_exactly_its_lag(self, lag: int) -> None:
        protocol = small_protocol()
        key = regime_key(lag)
        regime = ONLINE if lag == 0 else protocol.regimes[key]

        def estimate(values: np.ndarray) -> np.ndarray:
            return regime_posteriors(values, TRANSITION, PRIOR, {key: regime})[key]

        assert_respects_horizon(estimate, loglik(), regime)
        if lag:
            shorter = InferenceRegime.smoother(lag - 1, protocol.resolution.step)
            with pytest.raises(EvidenceLeakageError):
                assert_respects_horizon(
                    estimate, loglik(), shorter if lag > 1 else ONLINE
                )


# ----------------------------------------------------------------------------
# A run on simulated households
# ----------------------------------------------------------------------------
class TestExperiment:
    def test_the_record_validates_and_is_bounded_by_the_longest_lag(
        self, result: SmoothingResult
    ) -> None:
        assert result.path is not None
        payload = load_record(result.path)
        assert payload["results"]["result_schema"] == "smoothing-results/1"
        assert payload["configuration"]["protocol_sha256"] == small_protocol().sha256()
        inference = payload["inference"]
        assert inference["label"] == "fixed-lag smoother, lag 12 windows (60 min)"
        assert inference["causal"] is False
        assert inference["evidence"]["max_lead_seconds"] == 3600.0

    def test_each_regime_lists_the_evidence_it_read(
        self, result: SmoothingResult
    ) -> None:
        results = results_of(result)
        scored = sum(h["scored"] for h in results["households"].values())
        for key, regime in results["regimes"].items():
            evidence = regime["evidence"]
            assert evidence["predictions"] == scored
            assert evidence["max_lead_seconds"] == regime["lag_steps"] * STEP_SECONDS
            assert regime["causal"] is (key == ONLINE_KEY)
        total = result.record.to_dict()["inference"]["evidence"]["predictions"]
        assert total == scored * len(results["regimes"])

    def test_every_regime_is_scored_on_the_same_windows(
        self, result: SmoothingResult
    ) -> None:
        metrics = result.record.household_metrics
        results = results_of(result)
        for home, entry in results["households"].items():
            sizes = {metrics[key][home]["n"] for key in results["regimes"]}
            assert sizes == {entry["scored"]}
            assert metrics[REPRODUCED][home]["n"] == entry["labelled"]
            assert entry["scored"] <= entry["labelled"]
            assert entry["labelled"] - entry["scored"] <= 12

    def test_every_estimand_is_a_labelled_smoothing_gain(
        self, result: SmoothingResult
    ) -> None:
        for entry in results_of(result)["estimands"]:
            lag = entry["lag"]
            assert entry["kind"] == "smoothing gain"
            assert entry["reference"] == ONLINE_KEY
            assert entry["reference_inference"]["mode"] == "online_filter"
            assert entry["model_inference"]["causal"] is False
            assert entry["model_inference"]["delay_seconds"] == lag * STEP_SECONDS
            minutes = f"{lag * STEP_SECONDS / 60:g} min"
            assert entry["label"].startswith("smoothing gain: the fixed-lag smoother")
            assert f"reported {minutes} after their moments" in entry["label"]
            assert entry["operational"]["reporting_delay_minutes"] == lag * 5

    def test_conclusions_follow_the_rule(self, result: SmoothingResult) -> None:
        results = results_of(result)
        for entry in results["estimands"]:
            assert entry["conclusion"] == smoothing_conclusion(entry["verdicts"])
            assert results["conclusions"][entry["key"]] == entry["conclusion"]

    def test_the_operational_shares_match_the_accuracy_change(
        self, result: SmoothingResult
    ) -> None:
        metrics = result.record.household_metrics
        results = results_of(result)
        for entry in results["estimands"]:
            key = entry["model"]
            for home, household in results["households"].items():
                counts = household["changes"][key]
                change = (counts["corrected"] - counts["broken"]) / counts["windows"]
                accuracy = metrics[key][home]["accuracy"]
                online = metrics[ONLINE_KEY][home]["accuracy"]
                assert change == pytest.approx(accuracy - online, abs=1e-12)
                shares = change_shares(counts)
                assert entry["operational"]["changed"]["values"][home] == pytest.approx(
                    shares["changed"]
                )

    def test_the_transition_measures_are_reported_per_regime(
        self, result: SmoothingResult
    ) -> None:
        results = results_of(result)
        groups = {"all", "away", "home_active", "sleeping"}
        for household in results["households"].values():
            for key, regime in household["regimes"].items():
                assert set(regime["transitions"]) == groups
                assert regime["reporting_delay_minutes"] == (
                    0.0 if key == ONLINE_KEY else int(key.split("_")[1]) * 5.0
                )
                every = regime["transitions"]["all"]
                assert every["detected"] <= every["transitions"]
                if every["decision_delay_minutes"] is not None:
                    assert (
                        every["decision_delay_minutes"]
                        >= regime["reporting_delay_minutes"]
                    )
        for entry in results["estimands"]:
            assert set(entry["transitions"]["groups"]) == groups

    def test_per_state_recall_flags_the_focus_states(
        self, result: SmoothingResult
    ) -> None:
        for entry in results_of(result)["estimands"]:
            recall = entry["per_state_recall"]
            assert set(recall) == {s.value for s in ONTOLOGY.states}
            assert {s for s, v in recall.items() if v["focus"]} == {
                "away",
                "home_active",
                "sleeping",
            }

    def test_a_second_run_gives_the_same_record(
        self, homes: dict[str, CasasRecording], result: SmoothingResult
    ) -> None:
        again = run_smoothing(homes, small_protocol(), data_source="simulator")
        first, second = result.record.to_dict(), again.record.to_dict()
        first.pop("recorded_at")
        second.pop("recorded_at")
        assert first == second

    def test_missing_recordings_are_refused(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        partial = {h: r for h, r in homes.items() if h != "sim6"}
        with pytest.raises(ValueError, match="sim6"):
            run_smoothing(partial, small_protocol(), data_source="simulator")

    def test_the_summary_is_generated_from_the_written_record(
        self, result: SmoothingResult
    ) -> None:
        assert result.path is not None
        summary = render_summary(load_record(result.path))
        assert summary == render_summary(
            json.loads(json.dumps(result.record.to_dict()))
        )
        for heading in (
            "## Pre-specified conclusions",
            "## Every regime",
            "## Smoothing gains",
            "## Per-state recall",
            "## What smoothing changes",
            "## Transitions",
            "## Evidence read",
            "## Households",
        ):
            assert heading in summary
        assert "**Every difference below is a smoothing gain.**" in summary
        assert "No difference is an improvement of the online filter." in summary
        assert "online gain" not in summary
        for label in (
            "online filter",
            "fixed-lag smoother, lag 1 window (5 min)",
            "fixed-lag smoother, lag 12 windows (60 min)",
        ):
            assert label in summary
        payload = result.record.to_dict()
        payload["results"] = {"result_schema": "pooling-results/1"}
        with pytest.raises(ValueError, match="smoothing-results"):
            render_summary(payload)


# ----------------------------------------------------------------------------
# The published result
# ----------------------------------------------------------------------------
PUBLISHED = ROOT / "artifacts" / "phase3" / "phase3-fixed-lag-smoothing.json"
FITTED_RATES = ROOT / "artifacts" / "phase3" / "phase3-fitted-rates.json"
DOC = ROOT / "docs" / "PHASE3_SMOOTHING.md"


class TestPublishedResult:
    """The published development-panel result, and the page that reports it."""

    def test_it_ran_the_frozen_protocol_on_a_clean_tree(self) -> None:
        payload = load_record(PUBLISHED)
        splits = load_frozen_splits(SPLITS)
        declared = declared_protocol(splits)
        assert payload["configuration"]["protocol_sha256"] == declared.sha256()
        assert payload["environment"]["git_dirty"] == "false"
        assert payload["inference"]["label"] == (
            "fixed-lag smoother, lag 12 windows (60 min)"
        )
        inputs = {item["name"]: item["sha256"] for item in payload["inputs"]}
        assert inputs.pop(PROTOCOL.name) == check_frozen_protocol(declared, PROTOCOL)
        assert inputs.pop(SPLITS.name) == splits.sha256
        assert inputs == {
            entry["filename"]: entry["sha256"] for entry in splits.homes.values()
        }
        assert sorted(payload["results"]["households"]) == sorted(splits.homes)

    def test_the_population_is_the_phase_3_3_follow_up_fit(self) -> None:
        earlier = load_record(FITTED_RATES)["results"]["fitted"]
        for fold, entry in load_record(PUBLISHED)["results"]["fitted"].items():
            assert entry["population"]["sha256"] == earlier[fold]["channels"]["sha256"]

    def test_the_online_filter_reproduces_the_phase_3_3_follow_up(self) -> None:
        published = load_record(PUBLISHED)["household_metrics"][REPRODUCED]
        earlier = load_record(FITTED_RATES)["household_metrics"][REPRODUCED]
        assert published == earlier

    def test_every_published_difference_is_a_labelled_smoothing_gain(self) -> None:
        results = load_record(PUBLISHED)["results"]
        for entry in results["estimands"]:
            assert entry["kind"] == "smoothing gain"
            assert entry["reference_inference"]["mode"] == "online_filter"
            assert entry["model_inference"]["causal"] is False
            assert entry["conclusion"] == smoothing_conclusion(entry["verdicts"])
        for key, regime in results["regimes"].items():
            assert regime["evidence"]["max_lead_seconds"] == regime["delay_seconds"]
            assert regime["causal"] is (key == ONLINE_KEY)

    def test_the_page_carries_the_summary_generated_from_it(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        block = text.split("<!-- generated-summary:start -->")[1]
        block = block.split("<!-- generated-summary:end -->")[0]
        assert block.strip() == render_summary(load_record(PUBLISHED), level=3).strip()
