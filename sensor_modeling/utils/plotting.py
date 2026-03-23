"""Plotting helpers for sensor modeling."""

from __future__ import annotations

import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_quantile_intervals(
    actual_counts: np.ndarray,
    quantile_info: Dict,
    title: str = "Model Validation",
    sensor_name: str = "",
):
    """Plot observed counts against predicted quantile intervals."""
    time_of_day = np.arange(len(actual_counts)) * 15 / 60.0

    plt.figure(figsize=(12, 6))
    plt.fill_between(
        time_of_day,
        quantile_info["lower_quantiles"],
        quantile_info["upper_quantiles"],
        alpha=0.3,
        color="blue",
        label="95% Prediction Interval",
    )
    plt.plot(time_of_day, actual_counts, "r-", linewidth=2, label="Observed Events")
    plt.plot(time_of_day, quantile_info["means"], "k--", label="Predicted Mean")
    plt.xlabel("Hour of Day")
    plt.ylabel("Event Count")
    plt.title(title or f"Model Validation for {sensor_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    logger.info("Displayed validation plot for sensor %s", sensor_name)
    plt.show()


def plot_sensor_activity_patterns(data: pd.DataFrame) -> None:
    """Plot average sensor activation patterns across the day."""
    df = data.copy()
    df["hour"] = df.index.hour
    means = df.groupby("hour")[data.columns].mean()
    means.plot(figsize=(10, 6))
    plt.xlabel("Hour of Day")
    plt.ylabel("Activation Probability")
    plt.title("Sensor Daily Activity Patterns")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    logger.info("Plotted sensor activity patterns")
    plt.show()


def plot_change_points(
    series: np.ndarray, change_points: np.ndarray, title: str = "Change Points"
) -> None:
    """Plot time series with vertical lines at detected change points."""
    plt.figure(figsize=(10, 4))
    plt.plot(series, label="series")
    for cp in change_points:
        plt.axvline(cp, color="red", linestyle="--", alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    logger.info("Plotted %d change points", len(change_points))
    plt.show()


def plot_benchmark_results(results: Dict[str, float]) -> None:
    """Bar chart of benchmark times for different algorithms."""
    names = list(results.keys())
    times = list(results.values())
    plt.figure(figsize=(8, 4))
    plt.bar(names, times, color="skyblue")
    plt.ylabel("Seconds")
    plt.title("CPD Benchmark Runtime")
    plt.tight_layout()
    logger.info("Plotted benchmark results for %s", names)
    plt.show()
