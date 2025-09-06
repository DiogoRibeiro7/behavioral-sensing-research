"""High-level behavioral analysis utilities."""

from __future__ import annotations

from typing import Dict
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def recognize_activity_patterns(data: pd.DataFrame) -> Dict[str, int]:
    """Identify peak and quiet hours of activity."""
    data = data.copy()
    data["hour"] = data.index.hour
    hourly = data.groupby("hour")[data.columns].sum()
    totals = hourly.sum(axis=1)
    return {
        "peak_hours": int(totals.idxmax()),
        "quiet_hours": int(totals.idxmin()),
    }


# ---------------------------------------------------------------------------
def score_anomalies(data: pd.DataFrame) -> pd.Series:
    """Simple z-score based anomaly metric for each timestamp."""
    zscores = (data - data.mean()) / data.std(ddof=0)
    return zscores.abs().sum(axis=1)


# ---------------------------------------------------------------------------
def detect_trends(data: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Rolling mean trend indicator."""
    return data.rolling(window=window, min_periods=1).mean()


# ---------------------------------------------------------------------------
def health_indicators(data: pd.DataFrame) -> Dict[str, float]:
    """Basic health status indicators derived from activity levels."""
    activity = data.sum(axis=1)
    overall = float(activity.mean())
    variability = float(activity.std(ddof=0))
    sedentary_ratio = float((activity == 0).mean())
    return {
        "overall_activity": overall,
        "activity_variability": variability,
        "sedentary_ratio": sedentary_ratio,
    }
