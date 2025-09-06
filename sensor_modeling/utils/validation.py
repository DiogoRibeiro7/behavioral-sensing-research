"""Validation utilities for sensor models."""

from __future__ import annotations

from typing import Dict
import logging

from .data_io import SensorDataset
import numpy as np
import pandas as pd

from .plotting import plot_quantile_intervals

logger = logging.getLogger(__name__)


def validate_model_predictions(model, test_data: pd.DataFrame | SensorDataset,
                             sensor_name: str, confidence: float = 0.95) -> Dict:
    """Validate model predictions using quantile coverage metrics."""
    df = test_data.to_dataframe() if isinstance(test_data, SensorDataset) else test_data
    probabilities = model.predict_probabilities(df)
    quantile_info = model.compute_quantile_intervals(probabilities, confidence)
    if not quantile_info:
        return {'validation_successful': False, 'reason': 'Insufficient data for validation'}

    test_array = df[sensor_name].values
    n_days = len(test_array) // 96
    if n_days == 0:
        return {'validation_successful': False, 'reason': 'Insufficient test data'}
    test_reshaped = test_array[:n_days * 96].reshape(n_days, 96)
    actual_counts = np.sum(test_reshaped, axis=0)

    outside_count = np.sum((actual_counts < quantile_info['lower_quantiles']) |
                           (actual_counts > quantile_info['upper_quantiles']))
    outside_percentage = (outside_count / 96) * 100
    expected_outside = (1 - confidence) * 100
    mse = np.mean((actual_counts - quantile_info['means']) ** 2)
    coverage_accuracy = 100 - abs(outside_percentage - expected_outside)

    plot_quantile_intervals(actual_counts, quantile_info, sensor_name=sensor_name)

    return {
        'validation_successful': True,
        'outside_interval_count': int(outside_count),
        'outside_interval_percentage': outside_percentage,
        'expected_outside_percentage': expected_outside,
        'coverage_accuracy': coverage_accuracy,
        'mean_squared_error': mse,
        'actual_counts': actual_counts,
        'quantile_info': quantile_info,
        'is_well_calibrated': abs(outside_percentage - expected_outside) < 2.5,
    }


def create_model_comparison_report(results: Dict) -> str:
    """Create a formatted comparison report for multiple modeling approaches."""
    report = ["MODEL COMPARISON REPORT", "=" * 40, "\nBIC COMPARISON:"]
    bic_scores = {}
    for approach, data in results.items():
        if 'total_bic' in data:
            report.append(f"{approach.replace('_', ' ').title()}: {data['total_bic']:.2f}")
            bic_scores[approach] = data['total_bic']

    if 'improvements' in results:
        report.append("\nIMPROVEMENTS (Lower BIC is better):")
        for comparison, improvement in results['improvements'].items():
            direction = "better" if improvement > 0 else "worse"
            report.append(f"{comparison.replace('_', ' ').title()}: {improvement:.2f} ({direction})")

    if bic_scores:
        best_approach = min(bic_scores, key=bic_scores.get)
        report.append(f"\nBest performing approach: {best_approach.replace('_', ' ').title()}")
        report.append(f"BIC score: {bic_scores[best_approach]:.2f}")

    return "\n".join(report)
