"""Interactive dashboards using Plotly and Bokeh.

This module exposes helpers that make it easy to explore sensor data in
real time, tune model parameters, drill into detected changes, and export
figures for presentations. The functions are intentionally lightweight so
that they can run in environments without a full web server.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import plotly.express as px
from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import Slider
from bokeh.plotting import figure
from bokeh.resources import CDN


def real_time_display(data: pd.DataFrame) -> px.line:
    """Create a Plotly line figure for streaming sensor data.

    Parameters
    ----------
    data:
        DataFrame with at least ``timestamp``, ``sensor`` and ``value`` columns.
    """
    return px.line(data, x="timestamp", y="value", color="sensor")


def parameter_tuning(model: Any, param: str, values: Iterable[float]) -> column:
    """Return a Bokeh layout with a slider bound to ``model.param``.

    Updating the slider will update the attribute on ``model`` in real time.
    """
    slider = Slider(
        start=min(values),
        end=max(values),
        step=(max(values) - min(values)) / max(len(values) - 1, 1),
        value=getattr(model, param, min(values)),
        title=param,
    )

    def _update(attr, old, new):  # pragma: no cover - callback
        setattr(model, param, slider.value)

    slider.on_change("value", _update)
    return column(slider, figure(height=0))  # placeholder plot for layout


def drill_down(changes: pd.DataFrame) -> px.scatter:
    """Plot detected changes with interactive hover information."""
    return px.scatter(changes, x="time", y="score")


def export(fig: Any, path: str) -> None:
    """Export a Plotly or Bokeh figure to an HTML file."""
    if hasattr(fig, "write_html"):
        fig.write_html(path)
    elif fig.__class__.__module__.startswith("bokeh"):
        html = file_html(fig, CDN, "export")
        with open(
            path, "w", encoding="utf-8"
        ) as fh:  # pragma: no cover - simple file write
            fh.write(html)
    else:  # pragma: no cover - defensive fallback
        raise TypeError("Unsupported figure type")
