"""High-level behavioral analysis utilities."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._frame import prepare_sensor_frame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def recognize_activity_patterns(data: pd.DataFrame) -> dict[str, int]:
    """Identify peak and quiet hours of activity."""
    sensor_data = prepare_sensor_frame(data, context="recognize_activity_patterns")
    hourly = sensor_data.groupby(sensor_data.index.hour).sum()
    totals = hourly.sum(axis=1)
    return {
        "peak_hours": int(totals.idxmax()),
        "quiet_hours": int(totals.idxmin()),
    }


# ---------------------------------------------------------------------------
def score_anomalies(data: pd.DataFrame) -> pd.Series:
    """Simple z-score based anomaly metric for each timestamp."""
    sensor_data = prepare_sensor_frame(data, context="score_anomalies")
    std = sensor_data.std(ddof=0).replace(0, np.nan)
    zscores = (sensor_data - sensor_data.mean()) / std
    return zscores.fillna(0.0).abs().sum(axis=1)


# ---------------------------------------------------------------------------
def detect_trends(data: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Rolling mean trend indicator."""
    if window < 1:
        raise ValueError("window must be at least 1")
    sensor_data = prepare_sensor_frame(data, context="detect_trends")
    return sensor_data.rolling(window=window, min_periods=1).mean()


# ---------------------------------------------------------------------------
def health_indicators(data: pd.DataFrame) -> dict[str, float]:
    """Basic health status indicators derived from activity levels."""
    sensor_data = prepare_sensor_frame(data, context="health_indicators")
    activity = sensor_data.sum(axis=1)
    overall = float(activity.mean())
    variability = float(activity.std(ddof=0))
    sedentary_ratio = float((activity == 0).mean())
    return {
        "overall_activity": overall,
        "activity_variability": variability,
        "sedentary_ratio": sedentary_ratio,
    }
