"""Research oriented plotting utilities."""
from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt


def publication_figure(data: pd.DataFrame, x: str, y: str, hue: str | None = None):
    """Generate a publication quality figure using seaborn."""
    ax = sns.lineplot(data=data, x=x, y=y, hue=hue)
    ax.set_title("Publication Quality Figure")
    return ax.get_figure()


def model_diagnostics(residuals: Sequence[float]):
    """Return a Plotly histogram of model residuals."""
    return px.histogram(x=list(residuals), nbins=30, title="Model Residuals")


def performance_comparison(scores: pd.DataFrame) -> px.bar:
    """Bar chart comparing performance metrics across models."""
    return px.bar(scores, x="model", y="score", color="metric")


def statistical_tests(results: pd.DataFrame) -> px.scatter:
    """Scatter plot visualizing statistical test outcomes."""
    return px.scatter(results, x="test", y="pvalue", color="model")
