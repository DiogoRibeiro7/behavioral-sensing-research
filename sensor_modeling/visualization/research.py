"""Research oriented plotting utilities."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns


def _require_columns(
    data: pd.DataFrame, required: Collection[str], context: str
) -> None:
    """Raise a clear error when required plotting columns are missing."""
    missing = sorted(set(required) - set(data.columns))
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"{context} requires columns: {names}")


def _finite_residuals(residuals: Sequence[float]) -> list[float]:
    """Return residuals as finite floats for diagnostic plotting."""
    values = np.asarray(list(residuals), dtype=float)
    if values.size == 0:
        raise ValueError("residuals must contain at least one value")
    if not np.isfinite(values).all():
        raise ValueError("residuals must contain only finite values")
    return values.tolist()


def publication_figure(data: pd.DataFrame, x: str, y: str, hue: str | None = None):
    """Generate a publication quality figure using seaborn."""
    required = {x, y}
    if hue is not None:
        required.add(hue)
    _require_columns(data, required, "publication_figure")
    ax = sns.lineplot(data=data, x=x, y=y, hue=hue)
    ax.set_title("Publication Quality Figure")
    return ax.get_figure()


def model_diagnostics(residuals: Sequence[float]):
    """Return a Plotly histogram of model residuals."""
    return px.histogram(
        x=_finite_residuals(residuals), nbins=30, title="Model Residuals"
    )


def performance_comparison(scores: pd.DataFrame) -> px.bar:
    """Bar chart comparing performance metrics across models."""
    _require_columns(scores, {"metric", "model", "score"}, "performance_comparison")
    return px.bar(scores, x="model", y="score", color="metric")


def statistical_tests(results: pd.DataFrame) -> px.scatter:
    """Scatter plot visualizing statistical test outcomes."""
    _require_columns(results, {"model", "pvalue", "test"}, "statistical_tests")
    return px.scatter(results, x="test", y="pvalue", color="model")
