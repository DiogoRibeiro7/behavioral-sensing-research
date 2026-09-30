"""Tests for the selective-prediction evaluation framework.

A risk signal is judged by how well it orders predictions for rejection. The
framework must reproduce the exact curves of the perfect and reversed
orderings, leave a random or constant signal at random rejection's level,
require every signal's direction, and quantify uncertainty across households,
never across timestamps.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.evaluation import (
    HIGHER_IS_RISKIER,
    HIGHER_IS_SAFER,
    ArtifactError,
    ExperimentRecord,
    SelectiveData,
    SelectiveReport,
    Signal,
    evaluate_signal,
    evaluate_signals,
    load_record,
)
from sensor_modeling.evaluation.metrics import _expected_calibration_error
from sensor_modeling.evaluation.selective import (
    DEFAULT_COVERAGES,
    INTERVAL_METRICS,
    check_coverages,
    validate_selective,
)
from sensor_modeling.fusion import ONLINE, NotEnumerated

STATES = ("away", "active", "asleep")
GRID = np.array(DEFAULT_COVERAGES)


def households(
    sizes: dict[str, int], every: dict[str, int], seed: int = 0
) -> SelectiveData:
    """Households whose every ``every[h]``-th window is wrong; truth cycles the states."""
    rng = np.random.default_rng(seed)
    truth, predicted, confidence = {}, {}, {}
    for home, size in sizes.items():
        t = np.arange(size) % len(STATES)
        wrong = (
            (np.arange(size) % every[home]) == 0
            if every[home]
            else np.zeros(size, bool)
        )
        truth[home] = t
        predicted[home] = np.where(wrong, (t + 1) % len(STATES), t)
        confidence[home] = np.clip(
            np.where(wrong, 0.55, 0.85) + rng.normal(0, 0.05, size), 0.34, 1.0
        )
    return SelectiveData.from_households(STATES, truth, predicted, confidence)


def pooled(data: SelectiveData, signal: Signal, **kwargs: Any) -> dict[str, Any]:
    """The pooled curve and summary of one signal, serialised."""
    kwargs.setdefault("resamples", 200)
    entry: dict[str, Any] = evaluate_signals(data, [signal], **kwargs).to_dict()[
        "signals"
    ][signal.name]
    return entry


@pytest.fixture(scope="module")
def panel() -> SelectiveData:
    return households(
        {"h1": 600, "h2": 450, "h3": 900, "h4": 300},
        {"h1": 3, "h2": 4, "h3": 5, "h4": 2},
    )


# ----------------------------------------------------------------------------
class TestDirection:
    """The direction is never assumed."""

    def test_the_direction_is_required(self) -> None:
        with pytest.raises(TypeError):
            Signal("x", np.zeros(3))  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="direction"):
            Signal("x", np.zeros(3), "higher_is_better")

    def test_flipping_the_direction_is_negating_the_values(
        self, panel: SelectiveData
    ) -> None:
        values = panel.confidence
        assert values is not None
        safer = evaluate_signal(
            panel, Signal("c", values, HIGHER_IS_SAFER), resamples=100
        )
        riskier = evaluate_signal(
            panel, Signal("c", -values, HIGHER_IS_RISKIER), resamples=100
        )
        for key in ("risk", "balanced_accuracy", "state_coverage"):
            np.testing.assert_array_equal(safer.pooled[key], riskier.pooled[key])

    def test_missing_values_need_a_policy(self, panel: SelectiveData) -> None:
        values = np.arange(panel.truth.size, dtype=float)
        values[:30] = np.nan
        with pytest.raises(ValueError, match="missing policy"):
            Signal("x", values, HIGHER_IS_RISKIER)
        with pytest.raises(ValueError, match="missing"):
            Signal("x", values, HIGHER_IS_RISKIER, missing="ignore")
        with pytest.raises(ValueError, match="finite"):
            Signal("x", np.full(3, np.inf), HIGHER_IS_RISKIER)
        first = Signal("x", values, HIGHER_IS_RISKIER, missing="reject_first")
        last = Signal("x", values, HIGHER_IS_RISKIER, missing="retain_first")
        n = panel.truth.size
        keep = (n - 30) / n
        found = evaluate_signal(panel, first, coverages=[keep, 1.0], resamples=100)
        # Rejecting the missing windows first rejects exactly them.
        missing_states = np.bincount(panel.truth[:30], minlength=3) / 30
        np.testing.assert_allclose(
            found.pooled["rejected_composition"][0], missing_states, atol=1e-12
        )
        assert found.missing_windows == 30
        retained = evaluate_signal(panel, last, coverages=[30 / n, 1.0], resamples=100)
        np.testing.assert_allclose(
            retained.pooled["state_coverage"][0]
            * np.bincount(panel.truth, minlength=3),
            np.bincount(panel.truth[:30], minlength=3),
            atol=1e-9,
        )


# ----------------------------------------------------------------------------
class TestRankings:
    """Oracle, reversed, random and constant signals."""

    def test_perfect_oracle_ranking(self, panel: SelectiveData) -> None:
        loss = panel.window_loss
        entry = pooled(panel, Signal("oracle", loss, HIGHER_IS_RISKIER))
        n, errors = loss.size, loss.sum()
        expected = np.maximum(0.0, errors - (1.0 - GRID) * n) / (GRID * n)
        np.testing.assert_allclose(entry["pooled"]["risk"], expected, atol=1e-12)
        summary = entry["summary"]
        assert summary["gain"]["estimate"] == pytest.approx(1.0)
        assert summary["excess_aurc"]["estimate"] == pytest.approx(0.0, abs=1e-12)
        assert summary["aurc"]["estimate"] == pytest.approx(summary["oracle_aurc"])
        # Every rejected window was an error, while errors remained.
        rejected = np.array(entry["pooled"]["rejected_error"][:-1], dtype=float)
        assert np.all(rejected[GRID[:-1] >= 1.0 - errors / n] == pytest.approx(1.0))

    def test_reversed_ranking(self, panel: SelectiveData) -> None:
        loss = panel.window_loss
        entry = pooled(panel, Signal("reversed", loss, HIGHER_IS_SAFER))
        n, errors = loss.size, loss.sum()
        expected = np.minimum(1.0, errors / (GRID * n))
        np.testing.assert_allclose(entry["pooled"]["risk"], expected, atol=1e-12)
        summary = entry["summary"]
        base = errors / n
        area = summary["aurc"]["estimate"]
        assert summary["gain"]["estimate"] == pytest.approx(
            (base - area) / (base - summary["oracle_aurc"])
        )
        assert summary["gain"]["estimate"] < -1.0
        assert summary["gain"]["high"] < 0.0

    def test_random_ranking(self) -> None:
        data = households(
            {f"h{i}": 3000 for i in range(6)}, {f"h{i}": 3 + i % 3 for i in range(6)}
        )
        rng = np.random.default_rng(5)
        entry = pooled(
            data, Signal("random", rng.random(data.truth.size), HIGHER_IS_RISKIER)
        )
        base = data.window_loss.mean()
        risk = np.array(entry["pooled"]["risk"])
        assert np.all(np.abs(risk[GRID >= 0.2] - base) < 0.03)
        assert abs(entry["summary"]["gain"]["estimate"]) < 0.05
        low, high = entry["summary"]["gain"]["low"], entry["summary"]["gain"]["high"]
        assert low < 0.05 and high > -0.05
        for state in STATES:
            coverage = np.array(entry["pooled"]["state_coverage"][state])
            assert np.all(np.abs(coverage - GRID) < 0.05)

    def test_constant_score_is_random_rejection_exactly(
        self, panel: SelectiveData
    ) -> None:
        report = evaluate_signals(
            panel,
            [Signal("constant", np.zeros(panel.truth.size), HIGHER_IS_RISKIER)],
            resamples=200,
        ).to_dict()
        entry, random = report["signals"]["constant"], report["reference"]["random"]
        curve = entry["pooled"]
        np.testing.assert_allclose(curve["risk"], random["risk"], atol=1e-12)
        np.testing.assert_allclose(
            curve["balanced_accuracy"], random["balanced_accuracy"], atol=1e-12
        )
        np.testing.assert_allclose(
            curve["calibration_error"], random["calibration_error"], atol=1e-12
        )
        np.testing.assert_allclose(curve["coverage"], GRID, atol=1e-12)
        overall = np.bincount(panel.truth, minlength=3) / panel.truth.size
        for k, state in enumerate(STATES):
            np.testing.assert_allclose(curve["state_coverage"][state], GRID, atol=1e-12)
            np.testing.assert_allclose(
                curve["rejected_composition"][state][:-1], overall[k], atol=1e-12
            )
        assert entry["summary"]["gain"]["estimate"] == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(
            entry["pooled"]["intervals"]["excess_risk"]["estimate"], 0.0, atol=1e-12
        )

    def test_ties_are_split_exactly(self, panel: SelectiveData) -> None:
        values = (np.arange(panel.truth.size) % 2).astype(float)
        found = evaluate_signal(
            panel, Signal("two", values, HIGHER_IS_RISKIER), resamples=100
        )
        np.testing.assert_allclose(found.pooled["coverage"], GRID, atol=1e-12)


# ----------------------------------------------------------------------------
class TestHouseholds:
    """Uncertainty across households, and households of very different sizes."""

    def imbalanced(self, small_copies: int = 1) -> SelectiveData:
        truth = {
            "big": np.arange(10_000) % 3,
            "small": np.tile(np.arange(100) % 3, small_copies),
        }
        predicted = {"big": truth["big"], "small": (truth["small"] + 1) % 3}
        return SelectiveData.from_households(STATES, truth, predicted)

    def test_uncertainty_is_across_households(self) -> None:
        data = self.imbalanced()
        entry = pooled(
            data,
            Signal("c", np.zeros(data.truth.size), HIGHER_IS_RISKIER),
            resamples=500,
        )
        band = entry["pooled"]["intervals"]["risk"]
        # The timestamp-level estimate is dominated by the big household...
        assert band["estimate"][-1] == pytest.approx(100 / 10_100)
        # ...but households are resampled: drawing the small one twice gives 1.
        assert band["low"][-1] == pytest.approx(0.0)
        assert band["high"][-1] == pytest.approx(1.0)
        # Each household counts once across households.
        across = entry["across_households"]["risk"][-1]
        assert across["mean"] == pytest.approx(0.5)
        assert across["n"] == 2

    def test_household_curves_ignore_their_size(self) -> None:
        once, ten = self.imbalanced(1), self.imbalanced(10)
        values = {"once": np.zeros(once.truth.size), "ten": np.zeros(ten.truth.size)}
        first = pooled(once, Signal("c", values["once"], HIGHER_IS_RISKIER))
        second = pooled(ten, Signal("c", values["ten"], HIGHER_IS_RISKIER))
        assert (
            first["households"]["small"]["risk"]
            == second["households"]["small"]["risk"]
        )
        assert (
            first["across_households"]["risk"][-1]["mean"]
            == second["across_households"]["risk"][-1]["mean"]
        )
        assert first["pooled"]["risk"][-1] != second["pooled"]["risk"][-1]

    def test_one_household_has_no_interval(self) -> None:
        data = households({"only": 300}, {"only": 3})
        report = evaluate_signals(
            data, [Signal("c", data.window_loss, HIGHER_IS_RISKIER)], resamples=100
        ).to_dict()
        band = report["signals"]["c"]["pooled"]["intervals"]["risk"]
        assert band["low"] is None and band["high"] is None
        assert report["bootstrap"]["resamples"] == 0

    def test_a_pooled_threshold_covers_households_unequally(self) -> None:
        data = self.imbalanced()
        # Every small-household window is riskier than every big-household one.
        values = (data.households == "small").astype(float)
        entry = pooled(data, Signal("home", values, HIGHER_IS_RISKIER))
        at = list(GRID).index(0.95)
        assert entry["households"]["small"]["pooled_coverage"][at] == pytest.approx(0.0)
        assert entry["households"]["big"]["pooled_coverage"][at] == pytest.approx(
            0.95 * 10_100 / 10_000
        )
        coverage = entry["pooled"]["household_coverage"]
        assert coverage["min"][at] == pytest.approx(0.0)
        # Each household's own curve still retains its own fraction.
        assert entry["households"]["small"]["coverage"][at] == pytest.approx(0.95)


# ----------------------------------------------------------------------------
class TestMetrics:
    """The retained predictions' metrics, against direct computation."""

    def test_an_exact_boundary_matches_direct_scoring(self) -> None:
        data = households({"h": 400}, {"h": 3}, seed=2)
        rng = np.random.default_rng(3)
        values = rng.permutation(400).astype(float)
        found = evaluate_signal(
            data,
            Signal("s", values, HIGHER_IS_RISKIER),
            coverages=[0.5, 1.0],
            resamples=100,
        )
        kept = values < 200
        truth, predicted = data.truth[kept], data.predicted[kept]
        correct = truth == predicted
        assert found.pooled["error"][0] == pytest.approx(1.0 - correct.mean())
        recalls = [correct[truth == k].mean() for k in range(3)]
        assert found.pooled["balanced_accuracy"][0] == pytest.approx(np.mean(recalls))
        assert data.confidence is not None
        assert found.pooled["calibration_error"][0] == pytest.approx(
            _expected_calibration_error(data.confidence[kept], correct.astype(float))
        )
        assert found.pooled["mean_confidence"][0] == pytest.approx(
            data.confidence[kept].mean()
        )
        rejected = data.truth[~kept]
        np.testing.assert_allclose(
            found.pooled["rejected_composition"][0],
            np.bincount(rejected, minlength=3) / rejected.size,
        )

    def test_state_coverage_adds_up(self, panel: SelectiveData) -> None:
        assert panel.confidence is not None
        found = evaluate_signal(
            panel, Signal("c", panel.confidence, HIGHER_IS_SAFER), resamples=100
        )
        sizes = np.bincount(panel.truth, minlength=3)
        np.testing.assert_allclose(
            found.pooled["state_coverage"] @ sizes,
            found.pooled["coverage"] * sizes.sum(),
        )
        composition = found.pooled["rejected_composition"][:-1]
        np.testing.assert_allclose(composition.sum(axis=1), 1.0)

    def test_a_loss_matrix(self) -> None:
        truth = {"h": np.array([0, 0, 1, 1, 2, 2] * 50)}
        predicted = {"h": np.array([0, 1, 1, 2, 2, 0] * 50)}
        loss = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 5.0], [2.0, 1.0, 0.0]])
        data = SelectiveData.from_households(STATES, truth, predicted, loss=loss)
        oracle = evaluate_signal(
            data, Signal("o", data.window_loss, HIGHER_IS_RISKIER), resamples=100
        )
        assert oracle.pooled["risk"][-1] == pytest.approx((1 + 5 + 2) / 6)
        assert oracle.pooled["error"][-1] == pytest.approx(0.5)
        # The costliest errors go first: at 80%, all 50 fives and 10 of the twos.
        assert oracle.pooled["risk"][GRID.tolist().index(0.8)] == pytest.approx(
            (50 * 1 + 40 * 2) / 240
        )
        assert oracle.gain == pytest.approx(1.0)

    def test_without_confidence_no_calibration(self) -> None:
        data = SelectiveData.from_households(STATES, {"h": [0, 1, 2]}, {"h": [0, 1, 1]})
        found = evaluate_signal(
            data, Signal("s", np.arange(3.0), HIGHER_IS_RISKIER), resamples=100
        )
        assert np.all(np.isnan(found.pooled["calibration_error"]))

    @pytest.mark.parametrize(
        ("grid", "message"),
        [
            ([0.5], "end at 1"),
            ([0.0, 1.0], "within"),
            ([0.6, 0.5, 1.0], "increase"),
            ([], "at least one"),
        ],
    )
    def test_invalid_grids(self, grid: list[float], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            check_coverages(grid)

    def test_invalid_data(self) -> None:
        with pytest.raises(ValueError, match="index the states"):
            SelectiveData(np.array(["h"]), np.array([3]), np.array([0]), STATES)
        with pytest.raises(ValueError, match="loss"):
            SelectiveData(
                np.array(["h"]),
                np.array([0]),
                np.array([0]),
                STATES,
                loss=-np.ones((3, 3)),
            )
        with pytest.raises(ValueError, match="confidence"):
            SelectiveData(
                np.array(["h"]),
                np.array([0]),
                np.array([0]),
                STATES,
                confidence=np.array([2.0]),
            )


# ----------------------------------------------------------------------------
class TestReport:
    """The report, and the record section at schema 1.6."""

    def report(self, panel: SelectiveData) -> SelectiveReport:
        assert panel.confidence is not None
        return evaluate_signals(
            panel,
            [
                Signal("confidence", panel.confidence, HIGHER_IS_SAFER),
                Signal("oracle", panel.window_loss, HIGHER_IS_RISKIER),
            ],
            resamples=200,
        )

    def test_deterministic_and_json_safe(self, panel: SelectiveData) -> None:
        first = json.dumps(
            self.report(panel).to_dict(), sort_keys=True, allow_nan=False
        )
        assert json.dumps(self.report(panel).to_dict(), sort_keys=True) == first
        payload = json.loads(first)
        assert payload["bootstrap"]["resampled"] == "households"
        assert set(payload["signals"]["confidence"]["pooled"]["intervals"]) == set(
            INTERVAL_METRICS
        )
        assert payload["households"]["h1"]["windows"] == 600

    def test_one_signal_alone_matches_the_report(self, panel: SelectiveData) -> None:
        assert panel.confidence is not None
        alone = evaluate_signal(
            panel,
            Signal("confidence", panel.confidence, HIGHER_IS_SAFER),
            resamples=200,
        )
        together = self.report(panel).to_dict()["signals"]["confidence"]
        np.testing.assert_allclose(together["pooled"]["risk"], alone.pooled["risk"])
        assert together["summary"]["aurc"]["estimate"] == pytest.approx(alone.aurc)

    def test_invalid_sections_are_refused(self, panel: SelectiveData) -> None:
        good = self.report(panel).to_dict()
        assert validate_selective(good) == []
        for change, message in (
            (lambda p: p["signals"]["oracle"].update(direction="up"), "direction"),
            (lambda p: p.update(unit="timestamp"), "unit"),
            (lambda p: p["bootstrap"].update(resampled="timestamps"), "households"),
            (lambda p: p.update(coverages=[0.5, 0.4]), "coverages"),
            (
                lambda p: p["signals"]["oracle"]["pooled"]["intervals"]["risk"].update(
                    low=[1.0] * GRID.size, high=[0.0] * GRID.size
                ),
                "low exceeds",
            ),
            (
                lambda p: p["signals"]["oracle"]["pooled"].update(
                    error=[2.0] * GRID.size
                ),
                "[0, 1]",
            ),
        ):
            payload = json.loads(json.dumps(good))
            change(payload)
            assert any(message in p for p in validate_selective(payload)), message
        with pytest.raises(ValueError):
            SelectiveReport({"format": "x"})
        with pytest.raises(ValueError, match="distinct"):
            evaluate_signals(panel, [])

    def test_the_record_section(self, panel: SelectiveData, tmp_path: Path) -> None:
        def record(section: SelectiveReport | None) -> ExperimentRecord:
            return ExperimentRecord(
                experiment="selective",
                configuration={},
                inference=ONLINE,
                evidence=NotEnumerated("a unit-test record"),
                data_source="synthetic-test",
                selective_prediction=section,
            )

        section = self.report(panel)
        payload = load_record(record(section).write(tmp_path / "r.json"))
        assert payload["schema_version"] == "1.6"
        assert payload["selective_prediction"]["signals"]["oracle"]["summary"]["gain"][
            "estimate"
        ] == pytest.approx(1.0)
        rebuilt = ExperimentRecord.from_dict(payload).selective_prediction
        assert rebuilt is not None and rebuilt.signals == ("confidence", "oracle")
        old = record(None).to_dict()
        del old["selective_prediction"]
        old["schema_version"] = "1.5"
        path = tmp_path / "old.json"
        path.write_text(json.dumps(old), encoding="utf-8")
        migrated = load_record(path)
        assert migrated["migrated_from"] == "1.5"
        assert migrated["selective_prediction"] is None
        broken = record(section).to_dict()
        broken["selective_prediction"]["unit"] = "timestamp"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ArtifactError, match="unit"):
            load_record(path)
        with pytest.raises(TypeError, match="SelectiveReport"):
            record({"not": "a report"})  # type: ignore[arg-type]

    def test_gain_is_undefined_without_errors(self) -> None:
        data = SelectiveData.from_households(
            STATES,
            {"a": [0, 1, 2] * 10, "b": [2, 1, 0] * 10},
            {"a": [0, 1, 2] * 10, "b": [2, 1, 0] * 10},
        )
        entry = pooled(data, Signal("s", np.arange(60.0), HIGHER_IS_RISKIER))
        assert entry["summary"]["gain"]["estimate"] is None
        assert math.isclose(entry["summary"]["aurc"]["estimate"], 0.0)
