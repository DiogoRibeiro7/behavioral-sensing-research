"""Plotting helpers for sensor modeling."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_quantile_intervals(
    actual_counts: np.ndarray,
    quantile_info: dict,
    title: str = "Model Validation",
    sensor_name: str = "",
    show: bool = True,
) -> Figure:
    """Plot observed counts against predicted quantile intervals."""
    time_of_day = np.arange(len(actual_counts)) * 15 / 60.0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(
        time_of_day,
        quantile_info["lower_quantiles"],
        quantile_info["upper_quantiles"],
        alpha=0.3,
        color="blue",
        label="95% Prediction Interval",
    )
    ax.plot(time_of_day, actual_counts, "r-", linewidth=2, label="Observed Events")
    ax.plot(time_of_day, quantile_info["means"], "k--", label="Predicted Mean")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Event Count")
    ax.set_title(title or f"Model Validation for {sensor_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    logger.info("Displayed validation plot for sensor %s", sensor_name)
    if show:
        plt.show()
    return fig


def plot_sensor_activity_patterns(data: pd.DataFrame, show: bool = True) -> Figure:
    """Plot average sensor activation patterns across the day."""
    df = data.copy()
    df["hour"] = df.index.hour
    means = df.groupby("hour")[data.columns].mean()
    ax = means.plot(figsize=(10, 6))
    fig = ax.figure
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Activation Probability")
    ax.set_title("Sensor Daily Activity Patterns")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    logger.info("Plotted sensor activity patterns")
    if show:
        plt.show()
    return fig


def plot_change_points(
    series: np.ndarray,
    change_points: np.ndarray,
    title: str = "Change Points",
    show: bool = True,
) -> Figure:
    """Plot time series with vertical lines at detected change points."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series, label="series")
    for cp in change_points:
        ax.axvline(cp, color="red", linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    logger.info("Plotted %d change points", len(change_points))
    if show:
        plt.show()
    return fig


def plot_benchmark_results(results: dict[str, float], show: bool = True) -> Figure:
    """Bar chart of benchmark times for different algorithms."""
    names = list(results.keys())
    times = list(results.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, times, color="skyblue")
    ax.set_ylabel("Seconds")
    ax.set_title("CPD Benchmark Runtime")
    fig.tight_layout()
    logger.info("Plotted benchmark results for %s", names)
    if show:
        plt.show()
    return fig
