"""Healthcare focused visualizations and summaries."""
from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd
import plotly.express as px


def activity_summary(data: pd.DataFrame) -> px.bar:
    """Return a patient-friendly bar chart summarizing activity levels."""
    summary = data.groupby("activity").size().reset_index(name="count")
    return px.bar(summary, x="activity", y="count")


def clinical_alerts(
    data: pd.DataFrame, thresholds: Dict[str, float]
) -> Dict[str, bool]:
    """Flag sensors that exceed clinical thresholds."""
    alerts = {}
    for sensor, thresh in thresholds.items():
        alerts[sensor] = bool((data[data["sensor"] == sensor]["value"] > thresh).any())
    return alerts


def trend_monitor(data: pd.DataFrame, window: int = 7) -> px.line:
    """Plot rolling averages over weeks or months."""
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
    merged = pd.merge(
        data, normative, on=["timestamp", "sensor"], suffixes=("_patient", "_norm")
    )
    fig = px.line(
        merged, x="timestamp", y=["value_patient", "value_norm"], color="sensor"
    )
    return fig
