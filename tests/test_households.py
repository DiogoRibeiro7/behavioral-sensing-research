"""Tests for household-level statistics.

The synthetic households here are built so that every paired difference is
known exactly, which makes each estimate checkable by hand.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pytest

from sensor_modeling.evaluation import (
    compare_households,
    household_values,
    monte_carlo_standard_error,
    paired_difference,
    recall_of,
    score_households,
    summarise_households,
)
from sensor_modeling.evaluation import households as household_module
from sensor_modeling.evaluation.resampling import bca_interval
from sensor_modeling.states import BehaviouralState as S

#: The paired difference, model minus reference, in each household.
KNOWN = {"h01": 0.10, "h02": 0.20, "h03": -0.05, "h04": 0.0, "h05": 0.30}
REFERENCE = {h: 0.5 + 0.01 * i for i, h in enumerate(KNOWN)}
MODEL = {h: REFERENCE[h] + d for h, d in KNOWN.items()}


def panel(differences: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    """Model and reference values whose paired differences are *differences*."""
    reference = {h: 0.4 for h in differences}
    return {h: 0.4 + d for h, d in differences.items()}, reference


class TestKnownDifferences:
    def test_estimates_and_counts_match_the_constructed_differences(self) -> None:
        result = compare_households(MODEL, REFERENCE)
        assert result.households == tuple(sorted(KNOWN))
        assert result.differences == pytest.approx(tuple(KNOWN.values()))
        assert result.mean.value == pytest.approx(np.mean(list(KNOWN.values())))
        assert result.median.value == pytest.approx(0.10)
        assert (result.favours_model, result.favours_reference, result.tied) == (
            3,
            1,
            1,
        )
        assert result.proportion("model") == pytest.approx(0.6)
        assert result.proportion("tied") == pytest.approx(0.2)
        lowest, highest = min(result.differences), max(result.differences)
        for estimate in (result.mean, result.median):
            assert estimate.interval is not None and estimate.standard_error
            assert lowest <= estimate.interval.low <= estimate.value
            assert estimate.value <= estimate.interval.high <= highest

    def test_lower_is_better_reverses_the_orientation(self) -> None:
        higher = compare_households(MODEL, REFERENCE)
        lower = compare_households(MODEL, REFERENCE, higher_is_better=False)
        assert lower.mean.value == pytest.approx(-higher.mean.value)
        assert (lower.favours_model, lower.favours_reference) == (1, 3)

    def test_swapping_the_models_mirrors_the_result(self) -> None:
        forward = compare_households(MODEL, REFERENCE, seed=3)
        backward = compare_households(REFERENCE, MODEL, seed=3)
        assert forward.mean.interval is not None
        assert backward.mean.interval is not None
        assert backward.mean.value == pytest.approx(-forward.mean.value)
        assert backward.mean.interval.low == pytest.approx(-forward.mean.interval.high)
        assert backward.mean.interval.high == pytest.approx(-forward.mean.interval.low)

    @pytest.mark.parametrize("method", ["percentile", "bca"])
    def test_a_constant_shift_gives_a_zero_width_interval(self, method: str) -> None:
        result = compare_households(
            *panel({f"h{i}": 0.05 for i in range(6)}), interval=method
        )
        for estimate in (result.mean, result.median):
            assert estimate.value == pytest.approx(0.05)
            assert estimate.standard_error == pytest.approx(0.0, abs=1e-15)
            assert estimate.interval is not None
            assert estimate.interval.low == pytest.approx(0.05)
            assert estimate.interval.high == pytest.approx(0.05)
            assert estimate.interval.method == method
        assert result.effect_size is None
        assert "same amount" in (result.note or "")

    def test_household_intervals_cover_the_truth_where_timestamp_pooling_does_not(
        self,
    ) -> None:
        """Why the household is the unit, measured rather than asserted.

        Each household has its own true effect, and its timestamps scatter
        around that effect. An interval that treats every timestamp as
        independent covers the population effect far less often than its
        nominal rate, while resampling households stays close to nominal.
        """
        rng = np.random.default_rng(20260925)
        truth, households, timestamps, panels = 0.02, 12, 200, 300
        household_hits = pooled_hits = 0
        for panel_seed in range(panels):
            effects = rng.normal(truth, 0.03, households)
            stamps = effects[:, None] + rng.normal(0.0, 0.05, (households, timestamps))
            values = {f"h{i:02d}": float(v) for i, v in enumerate(stamps.mean(axis=1))}
            result = compare_households(
                values,
                {h: 0.0 for h in values},
                confidence=0.9,
                resamples=1000,
                seed=panel_seed,
            )
            assert result.mean.interval is not None
            household_hits += int(
                result.mean.interval.low <= truth <= result.mean.interval.high
            )
            pooled = stamps.ravel()
            half = 1.645 * pooled.std(ddof=1) / math.sqrt(pooled.size)
            pooled_hits += int(abs(pooled.mean() - truth) <= half)
        assert household_hits / panels >= 0.8
        assert pooled_hits / panels <= 0.5


class TestHouseholdLevel:
    def test_households_count_once_however_long_they_are(self) -> None:
        truth = {
            "short": [S.SLEEPING] * 4,
            "long": [S.SLEEPING] * 1000 + [S.AWAY] * 3000,
        }
        predicted = {
            "short": [S.SLEEPING] * 4,
            "long": [S.SLEEPING] * 4000,
        }
        values = household_values(score_households(truth, predicted), "accuracy")
        assert values == {"long": 0.25, "short": 1.0}
        summary = summarise_households(values)
        pooled = (4 + 1000) / 4004
        assert summary.mean == pytest.approx(0.625)
        assert summary.mean != pytest.approx(pooled)

    def test_an_unlabelled_household_has_no_score_rather_than_a_zero(self) -> None:
        scores = score_households(
            {"a": [S.SLEEPING, S.AWAY], "b": [None, None]},
            {"a": [S.SLEEPING, S.SLEEPING], "b": [S.AWAY, S.AWAY]},
        )
        assert scores["b"] is None
        summary = summarise_households(household_values(scores, "balanced_accuracy"))
        assert summary.households == ("a",) and summary.missing == ("b",)

    def test_metrics_that_do_not_exist_in_a_household_are_missing(self) -> None:
        scores = score_households(
            {"with": [S.AWAY, S.SLEEPING], "without": [S.SLEEPING, S.SLEEPING]},
            {"with": [S.AWAY, S.AWAY], "without": [S.SLEEPING, S.AWAY]},
        )
        assert household_values(scores, recall_of(S.AWAY)) == {
            "with": 1.0,
            "without": None,
        }
        assert household_values(scores, "log_loss") == {"with": None, "without": None}

    def test_probabilities_are_scored_per_household(self) -> None:
        certain = np.zeros((2, 7))
        certain[:, 3] = 1.0  # sleeping, in the default label space
        scores = score_households(
            {"a": [S.SLEEPING, S.SLEEPING]},
            {"a": [S.SLEEPING, S.SLEEPING]},
            {"a": certain},
        )
        assert household_values(scores, "brier") == {"a": 0.0}

    def test_summaries_of_one_or_no_household(self) -> None:
        one = summarise_households({"a": 0.4})
        assert (one.n, one.mean, one.sd) == (1, 0.4, None)
        none = summarise_households({"a": None, "b": float("nan")})
        assert none.n == 0 and none.mean is None and none.missing == ("a", "b")

    def test_inconsistent_inputs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same households"):
            score_households({"a": [S.AWAY]}, {"b": [S.AWAY]})
        with pytest.raises(ValueError, match="unknown household metric"):
            household_values({}, "auc")


class TestMissingAndEdgeCases:
    def test_missing_values_are_excluded_not_imputed(self) -> None:
        model = {**MODEL, "h02": None, "h03": float("nan")}
        reference = {h: v for h, v in REFERENCE.items() if h != "h05"}
        result = compare_households(model, reference)
        assert result.households == ("h01", "h04")
        assert result.excluded == ("h02", "h03", "h05")
        assert result.mean.value == pytest.approx((0.10 + 0.0) / 2)

    def test_one_household_gets_an_estimate_and_no_interval(self) -> None:
        result = compare_households({"a": 0.7}, {"a": 0.5, "b": 0.1})
        assert result.n == 1 and result.excluded == ("b",)
        for estimate in (result.mean, result.median):
            assert estimate.value == pytest.approx(0.2)
            assert estimate.interval is None and estimate.standard_error is None
        assert result.effect_size is None
        assert result.proportion("model") == 1.0
        assert "one household" in (result.note or "")
        assert result.to_dict()["mean"] == {
            "estimate": pytest.approx(0.2),
            "standard_error": None,
            "interval": None,
        }

    def test_nothing_to_pair_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no household"):
            compare_households({"a": 0.1}, {"b": 0.2})
        with pytest.raises(ValueError, match="no household"):
            compare_households({"a": None}, {"a": 0.2})

    def test_an_infinite_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="infinite"):
            compare_households({"a": math.inf, "b": 0.1}, {"a": 0.1, "b": 0.1})

    @pytest.mark.parametrize(
        "settings",
        [
            {"confidence": 0.0},
            {"confidence": 1.0},
            {"resamples": 50},
            {"interval": "basic"},
            {"seed": -1},
            {"seed": True},
        ],
    )
    def test_invalid_settings_are_rejected(self, settings: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            compare_households(MODEL, REFERENCE, **settings)


class TestDeterminism:
    def test_the_same_seed_gives_the_same_result(self) -> None:
        first = compare_households(MODEL, REFERENCE, seed=11, interval="bca")
        second = compare_households(MODEL, REFERENCE, seed=11, interval="bca")
        assert first.to_dict() == second.to_dict()

    def test_different_seeds_draw_different_resamples(self) -> None:
        first = compare_households(MODEL, REFERENCE, seed=1).mean.interval
        second = compare_households(MODEL, REFERENCE, seed=2).mean.interval
        assert first != second

    def test_household_order_does_not_matter(self) -> None:
        reversed_model = dict(reversed(list(MODEL.items())))
        assert (
            compare_households(reversed_model, REFERENCE, seed=4).to_dict()
            == compare_households(MODEL, REFERENCE, seed=4).to_dict()
        )


class TestBCa:
    def test_it_agrees_with_scipy_on_a_skewed_sample(self) -> None:
        from scipy.stats import bootstrap

        differences = np.random.default_rng(5).lognormal(0.0, 0.8, 30) - 1.0
        model, reference = panel(
            {f"h{i:02d}": float(d) for i, d in enumerate(differences)}
        )
        ours = compare_households(
            model, reference, interval="bca", resamples=20000, confidence=0.9
        ).mean.interval
        seeding = (
            "rng"
            if "rng" in inspect.signature(bootstrap).parameters
            else "random_state"
        )
        theirs = bootstrap(
            (differences,),
            np.mean,
            confidence_level=0.9,
            n_resamples=20000,
            method="BCa",
            **{seeding: np.random.default_rng(6)},
        ).confidence_interval
        assert ours is not None and ours.method == "bca"
        width = theirs.high - theirs.low
        assert ours.low == pytest.approx(theirs.low, abs=0.05 * width)
        assert ours.high == pytest.approx(theirs.high, abs=0.05 * width)

    def test_it_is_undefined_when_every_replicate_lies_on_one_side(self) -> None:
        replicates = np.linspace(1.0, 2.0, 200)
        assert bca_interval(replicates, 0.5, np.array([0.4, 0.6]), 0.95) is None

    def test_an_undefined_bca_falls_back_to_percentile_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(household_module, "bca_interval", lambda *args: None)
        result = compare_households(MODEL, REFERENCE, interval="bca")
        assert result.mean.interval is not None
        assert result.mean.interval.method == "percentile"
        assert "BCa undefined for the mean" in (result.note or "")


class TestSimulationSummaries:
    def test_paired_difference_draws_the_same_household_resamples(self) -> None:
        rng = np.random.default_rng(9)
        treatment = rng.normal(0.5, 0.1, 25)
        control = treatment - rng.normal(0.01, 0.02, 25)
        legacy = paired_difference(treatment.tolist(), control.tolist(), seed=7)
        names = [f"s{i:02d}" for i in range(25)]
        current = compare_households(
            dict(zip(names, treatment.tolist())),
            dict(zip(names, control.tolist())),
            seed=7,
        ).mean
        assert current.interval is not None
        assert legacy.ci_low == current.interval.low
        assert legacy.ci_high == current.interval.high
        assert legacy.mean_difference == pytest.approx(current.value)

    def test_monte_carlo_standard_error_of_replicates(self) -> None:
        replicates = [0.10, 0.12, 0.08, 0.11, 0.09]
        expected = np.std(replicates, ddof=1) / math.sqrt(5)
        assert monte_carlo_standard_error(replicates) == pytest.approx(expected)
        assert monte_carlo_standard_error([0.3, 0.3, 0.3]) == 0.0
        legacy = paired_difference(replicates, [0.0] * 5)
        assert legacy.mcse == pytest.approx(expected)
        with pytest.raises(ValueError, match="two replicates"):
            monte_carlo_standard_error([0.1])
        with pytest.raises(ValueError, match="finite"):
            monte_carlo_standard_error([0.1, math.nan])
