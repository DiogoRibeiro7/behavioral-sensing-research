"""Behavioral metrics calculated from sensor datasets."""

from __future__ import annotations

import logging

import pandas as pd

from ._frame import prepare_sensor_frame

logger = logging.getLogger(__name__)


def calculate_behavioral_metrics(data: pd.DataFrame) -> dict[str, object]:
    """Calculate basic behavioral pattern metrics from sensor data."""
    sensor_data = prepare_sensor_frame(data, context="calculate_behavioral_metrics")
    metrics: dict[str, object] = {}

    total_activations = sensor_data.sum().sum()
    total_possible = len(sensor_data) * len(sensor_data.columns)
    metrics["overall_activity_rate"] = float(total_activations / total_possible)
    metrics["total_activations"] = int(total_activations)

    hourly_activity = sensor_data.groupby(sensor_data.index.hour).sum()
    metrics["peak_activity_hour"] = int(hourly_activity.sum(axis=1).idxmax())
    metrics["quietest_hour"] = int(hourly_activity.sum(axis=1).idxmin())

    sensor_metrics: dict[str, dict[str, float | int]] = {}
    for sensor in sensor_data.columns:
        series = sensor_data[sensor]
        daily_variance = series.groupby(series.index.date).sum().var()
        sensor_metrics[sensor] = {
            "activation_rate": float(series.mean()),
            "total_activations": int(series.sum()),
            "longest_inactive_period": _find_longest_streak(series, 0),
            "longest_active_period": _find_longest_streak(series, 1),
            "daily_variance": 0.0 if pd.isna(daily_variance) else float(daily_variance),
        }
    metrics["sensor_metrics"] = sensor_metrics

    dow_activity = sensor_data.groupby(sensor_data.index.dayofweek).sum()
    weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    metrics["most_active_day"] = weekday_names[dow_activity.sum(axis=1).idxmax()]
    metrics["least_active_day"] = weekday_names[dow_activity.sum(axis=1).idxmin()]

    return metrics


def _find_longest_streak(series: pd.Series, value: int) -> int:
    max_streak = 0
    current_streak = 0
    for val in series:
        if val == value:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak
