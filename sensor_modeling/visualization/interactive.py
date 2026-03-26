"""Interactive dashboards using Plotly and Bokeh.

This module exposes helpers that make it easy to explore sensor data in
real time, tune model parameters, drill into detected changes, and export
figures for presentations. The functions are intentionally lightweight so
that they can run in environments without a full web server.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd
import plotly.express as px
from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, Div, HoverTool, Slider
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


def _score_parameter(model: Any, param: str, value: float) -> float:
    """Return a diagnostic score for ``param=value`` using model hooks."""
    if hasattr(model, "score_parameter"):
        scorer = getattr(model, "score_parameter")
        try:
            return float(scorer(param, value))
        except TypeError:
            return float(scorer(value))
    if hasattr(model, "evaluate_parameter"):
        evaluator = getattr(model, "evaluate_parameter")
        try:
            return float(evaluator(param, value))
        except TypeError:
            return float(evaluator(value))
    raise ValueError(
        "parameter_tuning requires model.score_parameter(...) or "
        "model.evaluate_parameter(...) to compute diagnostics"
    )


def _build_parameter_sweep(
    model: Any, param: str, values: Sequence[float]
) -> tuple[list[float], list[float]]:
    """Evaluate the model over candidate parameter values."""
    ordered_values = sorted({float(v) for v in values})
    if not ordered_values:
        raise ValueError("values must contain at least one candidate")
    scores = [_score_parameter(model, param, value) for value in ordered_values]
    return ordered_values, scores


def parameter_tuning(model: Any, param: str, values: Iterable[float]):
    """Return a Bokeh layout for parameter diagnostics.

    The model must expose either ``score_parameter(param, value)`` or
    ``evaluate_parameter(param, value)``. The returned view shows a score curve
    across candidate values and updates the highlighted selection plus summary
    text when the slider moves. In Python-backed Bokeh sessions, the callback
    also writes the chosen value back to ``model.<param>``.
    """

    param_values, scores = _build_parameter_sweep(model, param, list(values))
    current_value = float(getattr(model, param, param_values[0]))
    if current_value not in param_values:
        current_value = param_values[0]
    current_idx = param_values.index(current_value)
    best_idx = max(range(len(scores)), key=lambda idx: scores[idx])

    curve_source = ColumnDataSource(
        {
            "x": param_values,
            "y": scores,
            "label": [f"{value:.4g}" for value in param_values],
        }
    )
    selected_source = ColumnDataSource(
        {"x": [param_values[current_idx]], "y": [scores[current_idx]]}
    )

    plot = figure(
        height=320,
        sizing_mode="stretch_width",
        title=f"{param} diagnostic sweep",
        x_axis_label=param,
        y_axis_label="score",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    plot.line("x", "y", source=curve_source, line_width=2, color="#1f77b4")
    plot.scatter(
        "x", "y", source=curve_source, size=8, color="#1f77b4", alpha=0.85
    )
    plot.scatter(
        "x",
        "y",
        source=selected_source,
        size=14,
        color="#d62728",
        line_color="white",
        line_width=2,
    )
    plot.add_tools(
        HoverTool(
            tooltips=[(param, "@label"), ("score", "@y{0.000}")],
            renderers=plot.renderers[:2],
        )
    )

    summary = Div(
        text=(
            f"<b>{param}</b>: {param_values[current_idx]:.4g} | "
            f"<b>score</b>: {scores[current_idx]:.4f} | "
            f"<b>best</b>: {param_values[best_idx]:.4g} "
            f"({scores[best_idx]:.4f})"
        ),
        sizing_mode="stretch_width",
    )

    slider = Slider(
        start=0,
        end=len(param_values) - 1,
        step=1,
        value=current_idx,
        title=f"{param}: {param_values[current_idx]:.4g}",
    )

    def _update(attr: str, old: int, new: int) -> None:  # pragma: no cover
        value = param_values[int(new)]
        setattr(model, param, value)

    slider.on_change("value", _update)
    slider.js_on_change(
        "value",
        CustomJS(
            args=dict(
                slider=slider,
                selected=selected_source,
                summary=summary,
                values=param_values,
                scores=scores,
                param=param,
                best_value=param_values[best_idx],
                best_score=scores[best_idx],
            ),
            code="""
const idx = slider.value;
const value = values[idx];
const score = scores[idx];
selected.data = {x: [value], y: [score]};
selected.change.emit();
slider.title = `${param}: ${value.toPrecision(4)}`;
summary.text =
  `<b>${param}</b>: ${value.toPrecision(4)} | <b>score</b>: ${score.toFixed(4)} | ` +
  `<b>best</b>: ${best_value.toPrecision(4)} (${best_score.toFixed(4)})`;
""",
        ),
    )
    return column(slider, summary, plot, sizing_mode="stretch_width")


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
