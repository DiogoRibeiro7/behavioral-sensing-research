"""Healthcare focused visualizations and summaries."""

from __future__ import annotations

from collections.abc import Collection
from typing import Dict

import pandas as pd
import plotly.express as px


def _require_columns(
    data: pd.DataFrame, required: Collection[str], context: str
) -> None:
    """Raise a clear error when required visualization columns are missing."""
    missing = sorted(set(required) - set(data.columns))
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"{context} requires columns: {names}")


def activity_summary(data: pd.DataFrame) -> px.bar:
    """Return a patient-friendly bar chart summarizing activity levels."""
    _require_columns(data, {"activity"}, "activity_summary")
    summary = data.groupby("activity").size().reset_index(name="count")
    return px.bar(summary, x="activity", y="count")


def clinical_alerts(
    data: pd.DataFrame, thresholds: Dict[str, float]
) -> Dict[str, bool]:
    """Flag sensors that exceed clinical thresholds."""
    _require_columns(data, {"sensor", "value"}, "clinical_alerts")
    alerts = {}
    for sensor, thresh in thresholds.items():
        alerts[sensor] = bool((data[data["sensor"] == sensor]["value"] > thresh).any())
    return alerts


def trend_monitor(data: pd.DataFrame, window: int = 7) -> px.line:
    """Plot rolling averages over weeks or months."""
    _require_columns(data, {"sensor", "timestamp", "value"}, "trend_monitor")
    if window < 1:
        raise ValueError("window must be at least 1")

    rolled = (
        data.set_index("timestamp")
        .groupby("sensor")["value"]
        .rolling(window)
        .mean()
        .reset_index()
    )
    return px.line(rolled, x="timestamp", y="value", color="sensor")


def compare_norms(data: pd.DataFrame, normative: pd.DataFrame) -> px.line:
    """Compare patient data against normative statistics."""
    required = {"sensor", "timestamp", "value"}
    _require_columns(data, required, "compare_norms patient data")
    _require_columns(normative, required, "compare_norms normative data")
    merged = pd.merge(
        data, normative, on=["timestamp", "sensor"], suffixes=("_patient", "_norm")
    )
    fig = px.line(
        merged, x="timestamp", y=["value_patient", "value_norm"], color="sensor"
    )
    return fig
