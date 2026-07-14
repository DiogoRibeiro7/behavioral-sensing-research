"""Utilities for comparing different sensor models."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import TimeSeriesSplit

from ..utils.data_io import SensorDataset

logger = logging.getLogger(__name__)

ModelScorer = Callable[[Any, pd.DataFrame, pd.DataFrame], float]
FoldError = (AttributeError, FloatingPointError, RuntimeError, TypeError, ValueError)


def time_series_splits(
    data: SensorDataset | pd.DataFrame, n_splits: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return chronological train/test splits for sensor time series."""
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    dataset = data if isinstance(data, SensorDataset) else SensorDataset(data)
    df = dataset.to_dataframe()
    if len(df) < 2:
        raise ValueError("At least two observations are required for cross-validation")
    split_count = min(n_splits, len(df) - 1)
    splitter = TimeSeriesSplit(n_splits=split_count)
    return list(splitter.split(df))


def _fit_and_score_model(
    name: str,
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scorers: dict[str, ModelScorer],
) -> float:
    """Fit one model on a fold and return its score."""
    model.fit(train_df.values if name == "hmm" else train_df)
    if name in scorers:
        return float(scorers[name](model, train_df, test_df))
    return float(model.score(test_df.values))


def _mean_score(values: list[float]) -> float:
    """Return the mean of valid fold scores, or NaN when every fold failed."""
    finite_scores = [score for score in values if not np.isnan(score)]
    if not finite_scores:
        return float("nan")
    return float(np.mean(finite_scores))


# ---------------------------------------------------------------------------
def cross_validate(
    models: dict[str, Any],
    data: SensorDataset | pd.DataFrame,
    n_splits: int = 3,
    scorers: dict[str, ModelScorer] | None = None,
) -> dict[str, float]:
    """Perform chronological cross-validation across *models*.

    Parameters
    ----------
    models : dict[str, Any]
        Mapping from model name to model instance. Each model must implement a
        :py:meth:`fit` method and either a :py:meth:`score` method or a scorer
        supplied via ``scorers``.
    data : SensorDataset | pd.DataFrame
        Input dataset.
    n_splits : int
        Number of chronological folds.
    scorers : dict[str, ModelScorer] | None
        Optional per-model scoring functions accepting
        ``(model, train_df, test_df)``.
    """
    dataset = data if isinstance(data, SensorDataset) else SensorDataset(data)
    df = dataset.to_dataframe()
    scorers = scorers or {}
    unsupported = [
        name
        for name, model in models.items()
        if name not in scorers and not hasattr(model, "score")
    ]
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise TypeError(
            "Cross-validation requires each model to define score() or an "
            f"explicit scorer. Missing scorer for: {names}"
        )

    scores: dict[str, list[float]] = {name: [] for name in models}
    for train_idx, test_idx in time_series_splits(dataset, n_splits=n_splits):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        for name, model in models.items():
            try:
                score = _fit_and_score_model(name, model, train_df, test_df, scorers)
                scores[name].append(score)
            except FoldError as exc:
                logger.error("Failed fold for %s: %s", name, exc)
                scores[name].append(float("nan"))
    return {name: _mean_score(vals) for name, vals in scores.items()}


# ---------------------------------------------------------------------------
def significance_test(scores_a: list[float], scores_b: list[float]) -> float:
    """Paired t-test returning the p-value."""
    _, pvalue = ttest_rel(scores_a, scores_b, nan_policy="omit")
    return float(pvalue)


# ---------------------------------------------------------------------------
def standardize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Scale metric values to the [0,1] range."""
    vals = np.array(list(metrics.values()), dtype=float)
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    rng = vmax - vmin if vmax != vmin else 1.0
    return {k: (v - vmin) / rng for k, v in metrics.items()}


# ---------------------------------------------------------------------------
def visualize_comparison(metrics: dict[str, float], ax=None):
    """Visualize model comparison scores as a bar chart."""
    if ax is None:
        _, ax = plt.subplots()
    names = list(metrics.keys())
    vals = [metrics[n] for n in names]
    ax.bar(names, vals)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    return ax
