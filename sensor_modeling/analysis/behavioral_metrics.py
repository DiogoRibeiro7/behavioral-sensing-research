"""Behavioral metrics calculated from sensor datasets."""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_behavioral_metrics(data: pd.DataFrame) -> Dict:
    """Calculate basic behavioral pattern metrics from sensor data."""
    metrics: Dict = {}

    total_activations = data.sum().sum()
    total_possible = len(data) * len(data.columns)
    metrics["overall_activity_rate"] = total_activations / total_possible
    metrics["total_activations"] = int(total_activations)

    data_with_hour = data.copy()
    data_with_hour["hour"] = data.index.hour
    hourly_activity = data_with_hour.groupby("hour")[data.columns].sum()
    metrics["peak_activity_hour"] = int(hourly_activity.sum(axis=1).idxmax())
    metrics["quietest_hour"] = int(hourly_activity.sum(axis=1).idxmin())

    sensor_metrics = {}
    for sensor in data.columns:
        sensor_data = data[sensor]
        sensor_metrics[sensor] = {
            "activation_rate": float(sensor_data.mean()),
            "total_activations": int(sensor_data.sum()),
            "longest_inactive_period": _find_longest_streak(sensor_data, 0),
            "longest_active_period": _find_longest_streak(sensor_data, 1),
            "daily_variance": float(
                sensor_data.groupby(sensor_data.index.date).sum().var()
            ),
        }
    metrics["sensor_metrics"] = sensor_metrics

    data_with_dow = data.copy()
    data_with_dow["day_of_week"] = data.index.dayofweek
    dow_activity = data_with_dow.groupby("day_of_week")[data.columns].sum()
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
