"""Utilities for comparing different sensor models."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import TimeSeriesSplit

from ..utils.data_io import SensorDataset

logger = logging.getLogger(__name__)


def time_series_splits(
    data: SensorDataset | pd.DataFrame, n_splits: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return chronological train/test splits for sensor time series."""
    dataset = data if isinstance(data, SensorDataset) else SensorDataset(data)
    df = dataset.to_dataframe()
    if len(df) < 2:
        raise ValueError("At least two observations are required for cross-validation")
    split_count = min(n_splits, len(df) - 1)
    splitter = TimeSeriesSplit(n_splits=split_count)
    return list(splitter.split(df))


# ---------------------------------------------------------------------------
def cross_validate(
    models: dict[str, Any], data: SensorDataset | pd.DataFrame, n_splits: int = 3
) -> dict[str, float]:
    """Perform chronological cross-validation across *models*.

    Parameters
    ----------
    models : dict[str, Any]
        Mapping from model name to model instance. Each model must implement a
        :py:meth:`fit` method and optionally :py:meth:`score` or
        :py:meth:`predict`.
    data : SensorDataset | pd.DataFrame
        Input dataset.
    n_splits : int
        Number of chronological folds.
    """
    dataset = data if isinstance(data, SensorDataset) else SensorDataset(data)
    df = dataset.to_dataframe()
    scores: dict[str, list[float]] = {name: [] for name in models}
    for train_idx, test_idx in time_series_splits(dataset, n_splits=n_splits):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        X_test = test_df.values
        for name, model in models.items():
            try:
                model.fit(train_df.values if name == "hmm" else train_df)
                if hasattr(model, "score"):
                    score = float(model.score(X_test))
                elif hasattr(model, "predict"):
                    preds = model.predict(X_test)
                    if preds.shape[0] == X_test.shape[0]:
                        score = float(np.mean(preds == X_test[:, 0]))
                    else:
                        score = float("nan")
                else:
                    score = float("nan")
                scores[name].append(score)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed fold for %s: %s", name, exc)
                scores[name].append(float("nan"))
    return {name: float(np.nanmean(vals)) for name, vals in scores.items()}


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
