"""Tests for the Phase 4 comparison of richer uncertainty diagnostics.

The protocol is frozen before scoring, every signal ranks the same predictions,
the decisions follow the declared rules, and the documentation page and figures
are generated from the record alone.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import ActivityInterval, CasasRecording
from sensor_modeling.datasets.matched_evaluation import HouseholdSplit
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.uncertainty_experiment import (
    COMPARATOR,
    DECISIONS,
    ENSEMBLE,
    SIGNALS,
    STRUCTURAL_SIGNALS,
    UncertaintyProtocol,
    check_frozen_protocol,
    decide,
    declared_protocol,
    difficult_minority_states,
    normalised_entropy,
    run_uncertainty,
)
from sensor_modeling.datasets.uncertainty_figures import (
    FIGURES,
    data_sha256,
    draw_figures,
    figure_data,
)
from sensor_modeling.datasets.uncertainty_summary import render_page
from sensor_modeling.evaluation import load_record
from sensor_modeling.simulation import HouseholdConfig, simulate

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts" / "phase4" / "uncertainty_protocol.json"
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
FOLDS = (
    HouseholdSplit("a", train=("sim2", "sim4", "sim6"), test=("sim1", "sim3", "sim5")),
    HouseholdSplit("b", train=("sim1", "sim3", "sim5"), test=("sim2", "sim4", "sim6")),
)


def protocol(**changes: Any) -> UncertaintyProtocol:
    settings: dict[str, Any] = {
        "folds": FOLDS,
        "splits_sha256": "0" * 64,
        "curve_resamples": 200,
        "resamples": 1000,
    }
    settings.update(changes)
    return UncertaintyProtocol(**settings)


# ----------------------------------------------------------------------------
class TestProtocol:
    """The declaration, frozen before scoring."""

    def test_the_frozen_file_is_the_declared_protocol(self) -> None:
        declared = declared_protocol(load_frozen_splits(SPLITS))
        check_frozen_protocol(declared, PROTOCOL)
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert frozen["protocol_sha256"] == declared.sha256()
        assert list(frozen["signals"]) == sorted(SIGNALS)
        assert frozen["comparator"] == COMPARATOR
        assert frozen["structural_signals"] == list(STRUCTURAL_SIGNALS)
        assert [s["name"] for s in frozen["ensemble"]["specifications"]] == [
            s.name for s in ENSEMBLE
        ]
        assert all(s.parameters == "population" for s in ENSEMBLE)
        assert "none selected" in frozen["thresholds"]

    def test_a_changed_protocol_is_refused(self, tmp_path: Path) -> None:
        declared = declared_protocol(load_frozen_splits(SPLITS))
        payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        payload["minimal_differences"]["aurc"] = 0.02
        changed = tmp_path / "protocol.json"
        changed.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="cannot change"):
            check_frozen_protocol(declared, changed)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"inspection": (0.57,)}, "on the grid"),
            ({"inspection": (1.0,)}, "below 1"),
            ({"minority_share": 1.5}, "minority_share"),
            ({"minimal_differences": {"aurc": 0.01}}, "minimal differences"),
            ({"splits_sha256": "x"}, "SHA-256"),
            ({"coverages": (0.5, 0.9)}, "end at 1"),
        ],
    )
    def test_invalid_declarations(self, changes: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            protocol(**changes)


# ----------------------------------------------------------------------------
class TestRules:
    """The declared rules, on their own."""

    def test_the_decision_rule(self) -> None:
        assert decide("favours signal", True) == "materially better"
        assert decide("favours signal", False) == (
            "better but rejects difficult minority states"
        )
        assert decide("negligible", True) == "negligible difference"
        assert decide("favours confidence", True) == "worse"
        assert decide("uncertain", True) == "uncertain"
        assert set(DECISIONS) == {
            decide(v, g)
            for v in ("favours signal", "negligible", "favours confidence", "uncertain")
            for g in (True, False)
        }

    def test_difficult_minority_states(self) -> None:
        truth = np.array([0] * 50 + [1] * 45 + [2] * 5)
        predicted = truth.copy()
        predicted[95:98] = 0  # state 2: 2 of 5 right
        predicted[50:55] = 0  # state 1: 40 of 45 right
        table = difficult_minority_states(truth, predicted, ["a", "b", "c", "d"], 0.10)
        assert table["c"]["difficult_minority"] is True
        assert table["c"]["share"] == pytest.approx(0.05)
        assert table["c"]["recall"] == pytest.approx(0.4)
        assert table["b"]["difficult_minority"] is False  # common
        assert table["d"] == {"share": 0.0, "recall": None, "difficult_minority": False}

    def test_normalised_entropy(self) -> None:
        values = normalised_entropy(
            np.array([[1.0, 0, 0, 0], [0.25] * 4, [0.5, 0.5, 0, 0]])
        )
        np.testing.assert_allclose(values, [0.0, 1.0, 0.5])


# ----------------------------------------------------------------------------
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


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    homes = {f"sim{s}": simulated(s) for s in range(1, 7)}
    out = tmp_path_factory.mktemp("uncertainty")
    result = run_uncertainty(homes, protocol(), data_source="simulator", output_dir=out)
    assert result.path is not None
    return {"payload": load_record(result.path), "path": result.path, "dir": out}


class TestOnSimulatedHouseholds:
    """The whole experiment on simulated households."""

    def test_every_signal_ranks_the_same_predictions(
        self, written: dict[str, Any]
    ) -> None:
        payload = written["payload"]
        section = payload["selective_prediction"]
        assert set(section["signals"]) == set(SIGNALS)
        for name in SIGNALS:
            entry = section["signals"][name]
            assert entry["pooled"]["error"][-1] == pytest.approx(
                section["reference"]["random"]["error"]
            )
        results = payload["results"]
        windows = sum(h["windows"] for h in results["households"].values())
        assert windows == sum(h["windows"] for h in section["households"].values())

    def test_decisions_follow_the_declared_rules(self, written: dict[str, Any]) -> None:
        results = written["payload"]["results"]
        difficult = set(results["difficult_minority_states"])
        for name, entry in results["comparisons"].items():
            failures = [
                (state, level)
                for level, states in entry["state_coverage"].items()
                for state, item in states.items()
                if state in difficult and item["verdict"] == "favours confidence"
            ]
            assert entry["guard"]["holds"] is (not failures)
            assert entry["decision"] == decide(
                entry["aurc"]["verdict"], entry["guard"]["holds"]
            )
            assert results["decisions"][name] == entry["decision"]
        better = [
            s
            for s in STRUCTURAL_SIGNALS
            if results["decisions"][s] == "materially better"
        ]
        assert results["key_question"]["answer"] == ("yes" if better else "no")
        assert set(results["comparisons"]) == set(SIGNALS) - {COMPARATOR}

    def test_verdicts_follow_their_intervals(self, written: dict[str, Any]) -> None:
        results = written["payload"]["results"]
        for entry in results["comparisons"].values():
            item = entry["aurc"]
            mean = item["comparison"]["mean"]
            low, high = mean["interval"]["low"], mean["interval"]["high"]
            minimal = item["minimal"]
            if item["verdict"] == "favours signal":
                assert mean["estimate"] >= minimal and low > 0
            elif item["verdict"] == "favours confidence":
                assert mean["estimate"] <= -minimal and high < 0
            elif item["verdict"] == "negligible":
                assert -minimal < low and high < minimal

    def test_the_rejection_bias_is_the_pooled_state_coverage(
        self, written: dict[str, Any]
    ) -> None:
        payload = written["payload"]
        bias = payload["results"]["rejection_bias"]
        grid = payload["selective_prediction"]["coverages"]
        for name in SIGNALS:
            table = payload["selective_prediction"]["signals"][name]["pooled"]
            k = grid.index(0.5)
            for state, values in table["state_coverage"].items():
                entry = bias[name]["0.5"][state]
                assert entry["retained"] == values[k]
                if values[k] is not None:
                    assert entry["ratio"] == pytest.approx(values[k] / 0.5)

    def test_the_reproduction_needs_the_published_record(
        self, written: dict[str, Any]
    ) -> None:
        reproduction = written["payload"]["results"]["reproduction"]
        assert reproduction["published"] is None
        for values in reproduction["households"].values():
            assert 0.0 <= values["balanced_accuracy"] <= 1.0

    def test_the_page_is_generated_from_the_record(
        self, written: dict[str, Any]
    ) -> None:
        payload = written["payload"]
        page = render_page(payload)
        assert page == render_page(load_record(written["path"]))
        results = payload["results"]
        assert f"**{results['key_question']['answer'].capitalize()}.**" in page
        for name, decision in results["decisions"].items():
            assert re.search(rf"`{name}`.*\*\*{decision}\*\*", page)
        assert payload["configuration"]["protocol_sha256"] in page
        for figure in FIGURES:
            assert f"figures/phase4-uncertainty-{figure}.svg" in page
        # Table cells never break on a pipe in a definition.
        for line in page.splitlines():
            if line.startswith("| `predictive_mismatch` | `higher"):
                assert "p(x \\| s)" in line

    def test_figures_are_deterministic_and_carry_their_data(
        self, written: dict[str, Any], tmp_path: Path
    ) -> None:
        payload = written["payload"]
        first = draw_figures(payload, tmp_path / "first")
        second = draw_figures(payload, tmp_path / "second")
        data = figure_data(payload)
        for name in FIGURES:
            assert first[name].read_bytes() == second[name].read_bytes()
            assert f"data sha256 {data_sha256(data[name])}" in first[name].read_text(
                encoding="utf-8"
            )
        assert set(data["retention"]["states"]) == set(
            payload["results"]["difficult_minority_states"]
        )

    def test_the_record_is_strict_json(self, written: dict[str, Any]) -> None:
        text = written["path"].read_text(encoding="utf-8")
        json.loads(text, parse_constant=lambda token: pytest.fail(token))
        results = written["payload"]["results"]
        assert results["signals"]["evidence_disagreement"]["missing"] == "reject_first"
        assert all(
            math.isfinite(e["summary"]["aurc"]["estimate"])
            for e in results["signals"].values()
        )
