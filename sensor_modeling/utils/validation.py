"""Validation utilities for sensor models."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Protocol

import numpy as np
import pandas as pd

from .data_io import SensorDataset
from .plotting import plot_quantile_intervals

logger = logging.getLogger(__name__)


class SupportsPredictionIntervals(Protocol):
    """Model interface required by prediction validation."""

    def predict_probabilities(self, data: pd.DataFrame) -> np.ndarray:
        """Return predicted event probabilities."""
        ...

    def compute_quantile_intervals(
        self, probabilities: np.ndarray, confidence: float
    ) -> Mapping[str, np.ndarray]:
        """Return lower, upper, and mean prediction intervals."""
        ...


def validate_model_predictions(
    model: SupportsPredictionIntervals,
    test_data: pd.DataFrame | SensorDataset,
    sensor_name: str,
    confidence: float = 0.95,
    *,
    plot: bool = True,
) -> dict[str, object]:
    """Validate model predictions using quantile coverage metrics."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    df = test_data.to_dataframe() if isinstance(test_data, SensorDataset) else test_data
    if sensor_name not in df.columns:
        raise KeyError(f"Sensor '{sensor_name}' not found in test data")

    probabilities = model.predict_probabilities(df)
    quantile_info = model.compute_quantile_intervals(probabilities, confidence)
    if not quantile_info:
        return {
            "validation_successful": False,
            "reason": "Insufficient data for validation",
        }

    test_array = df[sensor_name].values
    n_days = len(test_array) // 96
    if n_days == 0:
        return {"validation_successful": False, "reason": "Insufficient test data"}
    test_reshaped = test_array[: n_days * 96].reshape(n_days, 96)
    actual_counts = np.sum(test_reshaped, axis=0)

    outside_count = np.sum(
        (actual_counts < quantile_info["lower_quantiles"])
        | (actual_counts > quantile_info["upper_quantiles"])
    )
    outside_percentage = (outside_count / 96) * 100
    expected_outside = (1 - confidence) * 100
    mse = np.mean((actual_counts - quantile_info["means"]) ** 2)
    coverage_accuracy = 100 - abs(outside_percentage - expected_outside)

    if plot:
        plot_quantile_intervals(actual_counts, quantile_info, sensor_name=sensor_name)

    return {
        "validation_successful": True,
        "outside_interval_count": int(outside_count),
        "outside_interval_percentage": outside_percentage,
        "expected_outside_percentage": expected_outside,
        "coverage_accuracy": coverage_accuracy,
        "mean_squared_error": mse,
        "actual_counts": actual_counts,
        "quantile_info": quantile_info,
        "is_well_calibrated": bool(abs(outside_percentage - expected_outside) < 2.5),
    }


def create_model_comparison_report(results: Mapping[str, object]) -> str:
    """Create a formatted comparison report for multiple modeling approaches."""
    report = ["MODEL COMPARISON REPORT", "=" * 40, "\nBIC COMPARISON:"]
    bic_scores: dict[str, float] = {}
    for approach, data in results.items():
        if isinstance(data, Mapping) and isinstance(data.get("total_bic"), int | float):
            total_bic = float(data["total_bic"])
            report.append(f"{approach.replace('_', ' ').title()}: {total_bic:.2f}")
            bic_scores[approach] = total_bic

    improvements = results.get("improvements")
    if isinstance(improvements, Mapping):
        report.append("\nIMPROVEMENTS (Lower BIC is better):")
        for comparison, improvement in improvements.items():
            if not isinstance(improvement, int | float):
                continue
            improvement_value = float(improvement)
            direction = "better" if improvement_value > 0 else "worse"
            report.append(
                f"{comparison.replace('_', ' ').title()}: {improvement_value:.2f} ({direction})"
            )

    if bic_scores:
        best_approach = min(bic_scores, key=bic_scores.get)
        report.append(
            f"\nBest performing approach: {best_approach.replace('_', ' ').title()}"
        )
        report.append(f"BIC score: {bic_scores[best_approach]:.2f}")

    return "\n".join(report)
