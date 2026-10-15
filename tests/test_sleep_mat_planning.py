"""The sleep-mat planning trials: their pieces, on cases worked by hand."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from sensor_modeling.datasets.sleep_mat_planning import (
    PERCENTILE,
    STUDENT,
    Planning,
    intervals,
    level_verdict,
    plan,
    spearman_rows,
    tracking_verdict,
)


def small() -> Planning:
    return Planning(
        replications=100,
        resamples=400,
        truth_homes=300,
        centres=(0.5,),
        spreads=(0.3,),
        contamination=(0.0, 0.03),
        biases=(0.0, 3.0),
        bias_spreads=(1.0,),
    )


class TestPieces:
    def test_spearman_rows_is_scipys_row_by_row(self) -> None:
        rng = np.random.default_rng(3)
        x, y = rng.standard_normal((5, 30)), rng.standard_normal((5, 30))
        y[:, :3] = x[:, :3]
        x[0, :4] = 1.0  # ties
        expected = [stats.spearmanr(a, b)[0] for a, b in zip(x, y)]
        assert spearman_rows(x, y) == pytest.approx(expected)

    def test_the_t_interval_is_the_textbook_one(self) -> None:
        values = np.array([0.1, 0.4, 0.5, 0.7, 0.3])
        low, high = intervals(values, np.random.default_rng(0), small())[STUDENT]
        half = stats.t.ppf(0.975, 4) * values.std(ddof=1) / np.sqrt(5)
        assert (low, high) == pytest.approx(
            (values.mean() - half, values.mean() + half)
        )

    def test_the_bootstrap_leaves_the_stated_share_outside(self) -> None:
        planning = small()
        values = np.arange(10, dtype=float)
        low, high = intervals(values, np.random.default_rng(0), planning)[PERCENTILE]
        rng = np.random.default_rng(0)
        means = np.sort(values[rng.integers(0, 10, (400, 10))].mean(axis=1))
        assert (low, high) == (means[10], means[389])

    @pytest.mark.parametrize(
        ("low", "high", "reading"),
        [
            (0.51, 0.9, "follows"),
            (0.1, 0.49, "does_not_follow"),
            (0.5, 0.9, "inconclusive"),
            (0.1, 0.5, "inconclusive"),
        ],
    )
    def test_tracking_readings(self, low: float, high: float, reading: str) -> None:
        assert tracking_verdict(low, high, 0.5) == reading

    @pytest.mark.parametrize(
        ("low", "high", "reading"),
        [
            (-0.9, 0.9, "agrees"),
            (1.1, 3.0, "does_not_agree"),
            (-3.0, -1.1, "does_not_agree"),
            (-1.0, 0.5, "inconclusive"),
            (0.5, 1.5, "inconclusive"),
        ],
    )
    def test_level_readings(self, low: float, high: float, reading: str) -> None:
        assert level_verdict(low, high, 1.0) == reading


class TestTrials:
    def test_the_trials_are_repeatable_and_complete(self) -> None:
        first, second = plan(small()), plan(small())
        assert first == second
        assert len(first["tracking"]) == 4 and len(first["level"]) == 4
        assert first["chosen"] in (PERCENTILE, STUDENT)
        for cell in first["tracking"] + first["level"]:
            for method in (PERCENTILE, STUDENT):
                assert sum(cell[method]["verdicts"].values()) == pytest.approx(1.0)
                assert 0.0 <= cell[method]["coverage"] <= 1.0

    def test_a_large_bias_does_not_agree(self) -> None:
        large = plan(small())["level"][1]
        assert large["bias"] == 3.0
        assert large[STUDENT]["verdicts"].get("does_not_agree", 0.0) > 0.9

    def test_the_outlying_homes_raise_the_true_bias(self) -> None:
        planning = small()
        level = plan(planning)["level"]
        assert [cell["outliers"] for cell in level] == [False, False, True, True]
        extra = 4.0 * 3 / len(planning.days)
        assert level[2]["truth"] == pytest.approx(0.0 + extra)
        assert level[3]["truth"] == pytest.approx(3.0 + extra)

    def test_the_outlying_homes_lower_the_true_correlation(self) -> None:
        tracking = plan(small())["tracking"]
        plain = [c for c in tracking if not c["outliers"] and not c["contamination"]]
        outlying = [c for c in tracking if c["outliers"] and not c["contamination"]]
        assert outlying[0]["truth"] < plain[0]["truth"] - 0.05

    def test_too_few_replications_are_refused(self) -> None:
        with pytest.raises(ValueError, match="100 replications"):
            Planning(replications=99)
