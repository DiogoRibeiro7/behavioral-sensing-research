"""Tests for Granger causality utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sensor_modeling.analysis.granger_causality import GrangerCausalityTest


def test_granger_test_returns_finite_aligned_statistics():
    x = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y = np.array([0, 0, 1, 0, 1, 0, 1, 0])

    result = GrangerCausalityTest(max_lags=2).test(x, y, lags=2)

    assert np.isfinite(result["test_statistic"])
    assert np.isfinite(result["p_value"])
    assert result["lags_used"] == 2
    assert 0.0 <= result["p_value"] <= 1.0


def test_granger_test_validates_inputs():
    tester = GrangerCausalityTest()

    with pytest.raises(ValueError, match="same length"):
        tester.test(np.array([0, 1, 0]), np.array([0, 1]))
    with pytest.raises(ValueError, match="binary"):
        tester.test(np.array([0, 2, 0]), np.array([0, 1, 0]))
    with pytest.raises(ValueError, match="lags must be at least 1"):
        tester.test(np.array([0, 1, 0]), np.array([0, 1, 0]), lags=0)


def test_test_all_pairs_reports_failures_without_stopping():
    data = pd.DataFrame(
        {
            "sensor_0": [0, 1, 0, 1],
            "sensor_1": [0, 2, 0, 1],
        }
    )

    results = GrangerCausalityTest().test_all_pairs(data)

    assert len(results) == 2
    assert results["causality_detected"].eq(False).all()
    assert results["p_value"].isna().all()


def test_causality_summary_handles_empty_results():
    summary = GrangerCausalityTest().create_causality_summary(
        pd.DataFrame(
            columns=[
                "cause",
                "effect",
                "test_statistic",
                "p_value",
                "causality_detected",
                "lags_used",
            ]
        )
    )

    assert summary["total_pairs_tested"] == 0
    assert summary["causality_rate"] == 0
    assert summary["bidirectional_relationships"] == []
