"""Tests for model validation and plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from sensor_modeling.utils import plotting
from sensor_modeling.utils.data_io import SensorDataset
from sensor_modeling.utils.validation import (
    create_model_comparison_report,
    validate_model_predictions,
)


class _DummyIntervalModel:
    def predict_probabilities(self, data: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(data))

    def compute_quantile_intervals(
        self, probabilities: np.ndarray, confidence: float
    ) -> dict[str, np.ndarray]:
        return {
            "lower_quantiles": np.zeros(96),
            "upper_quantiles": np.full(96, 2),
            "means": np.ones(96),
        }


def _two_day_sensor_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=192, freq="15min")
    values = np.tile([0, 1], 96)
    return pd.DataFrame({"sensor_0": values}, index=index)


def test_validate_model_predictions_can_run_without_plotting():
    dataset = SensorDataset(_two_day_sensor_frame())

    result = validate_model_predictions(
        _DummyIntervalModel(),
        dataset,
        "sensor_0",
        confidence=0.95,
        plot=False,
    )

    assert result["validation_successful"] is True
    assert result["outside_interval_count"] == 0
    assert result["is_well_calibrated"] is False


def test_validate_model_predictions_rejects_invalid_inputs():
    model = _DummyIntervalModel()
    df = _two_day_sensor_frame()

    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        validate_model_predictions(model, df, "sensor_0", confidence=1.0, plot=False)

    with pytest.raises(KeyError, match="Sensor 'missing' not found"):
        validate_model_predictions(model, df, "missing", plot=False)


def test_create_model_comparison_report_identifies_best_bic():
    report = create_model_comparison_report(
        {
            "baseline_model": {"total_bic": 10.0},
            "candidate_model": {"total_bic": 5.0},
            "improvements": {"candidate_vs_baseline": 5.0},
        }
    )

    assert "Candidate Model: 5.00" in report
    assert "Candidate Vs Baseline: 5.00 (better)" in report
    assert "Best performing approach: Candidate Model" in report


def test_plotting_helpers_return_figures_without_showing():
    actual_counts = np.zeros(96)
    quantiles = {
        "lower_quantiles": np.zeros(96),
        "upper_quantiles": np.ones(96),
        "means": np.full(96, 0.5),
    }
    hourly = pd.DataFrame(
        {"sensor_0": [0, 1, 0]},
        index=pd.date_range("2024-01-01", periods=3, freq="1h"),
    )

    figures = [
        plotting.plot_quantile_intervals(actual_counts, quantiles, show=False),
        plotting.plot_sensor_activity_patterns(hourly, show=False),
        plotting.plot_change_points(np.array([0, 1, 0]), np.array([1]), show=False),
        plotting.plot_benchmark_results({"pelt": 0.1}, show=False),
    ]

    assert all(isinstance(fig, Figure) for fig in figures)
    for fig in figures:
        plt.close(fig)
