"""Data loading and simulation utilities for sensor modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SensorDataset:
    """Unified in-memory representation of sensor time series data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame indexed by timestamps with one column per sensor.
    """

    data: pd.DataFrame

    @classmethod
    def from_csv(cls, path: str) -> "SensorDataset":
        """Load sensor data from a CSV file."""
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        logger.info("Loaded %d rows from %s", len(df), path)
        return cls(df)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self.data

    def to_event_sequences(self, sensor: str) -> List[np.ndarray]:
        """Convert binary activations for *sensor* into per-day event times."""
        if sensor not in self.data.columns:
            raise KeyError(f"Sensor '{sensor}' not found in dataset")
        df = self.data[self.data[sensor] > 0]
        grouped = df.groupby(df.index.date)
        events: List[np.ndarray] = []
        for _, day_df in grouped:
            times = day_df.index
            events.append(
                np.array([t.hour + t.minute / 60.0 for t in times], dtype=float)
            )
        return events


def simulate_sensor_data(
    n_days: int = 60,
    n_sensors: int = 4,
    seed: int = 42,
    interaction_strength: float = 0.3,
) -> SensorDataset:
    """Simulate realistic binary sensor activations."""
    np.random.seed(seed)

    n_intervals = n_days * 96
    time_index = pd.date_range("2024-01-01", periods=n_intervals, freq="15min")
    sensor_names = [f"sensor_{i}" for i in range(n_sensors)]

    data = pd.DataFrame(index=time_index, columns=sensor_names)
    sensor_data = np.zeros((n_intervals, n_sensors))

    activity_patterns = []
    for i in range(n_sensors):
        morning_peak = 24 + i * 4
        evening_peak = 72 + i * 2
        activity_patterns.append(
            {
                "morning_start": morning_peak,
                "morning_end": morning_peak + 12,
                "morning_prob": 0.4 - i * 0.05,
                "evening_start": evening_peak,
                "evening_end": evening_peak + 16,
                "evening_prob": 0.5 - i * 0.06,
                "baseline_prob": 0.02 + i * 0.01,
            }
        )

    for day in range(n_days):
        day_start = day * 96
        for sensor_idx, pattern in enumerate(activity_patterns):
            day_probs = np.full(96, pattern["baseline_prob"])
            day_probs[pattern["morning_start"] : pattern["morning_end"]] = pattern[
                "morning_prob"
            ]
            evening_end = min(pattern["evening_start"] + 16, 96)
            day_probs[pattern["evening_start"] : evening_end] = pattern["evening_prob"]
            for t in range(96):
                if day_start + t < n_intervals:
                    base_activation = np.random.binomial(1, day_probs[t])
                    sensor_data[day_start + t, sensor_idx] = base_activation

    if interaction_strength > 0:
        for day in range(n_days):
            day_start = day * 96
            for t in range(1, 96):
                global_t = day_start + t
                if global_t >= n_intervals:
                    break
                for target_sensor in range(n_sensors):
                    if sensor_data[global_t, target_sensor] == 0:
                        influence_prob = 0.0
                        for source_sensor in range(n_sensors):
                            if source_sensor == target_sensor:
                                continue
                            for lag in range(1, 4):
                                if (
                                    global_t - lag >= 0
                                    and sensor_data[global_t - lag, source_sensor] == 1
                                ):
                                    if abs(source_sensor - target_sensor) == 1:
                                        influence_prob += (
                                            interaction_strength
                                            * 0.3
                                            * (0.8 ** (lag - 1))
                                        )
                                    else:
                                        influence_prob += (
                                            interaction_strength
                                            * 0.1
                                            * (0.8 ** (lag - 1))
                                        )
                        if influence_prob > np.random.random():
                            sensor_data[global_t, target_sensor] = 1

    for i, sensor in enumerate(sensor_names):
        data[sensor] = sensor_data[:, i]

    return SensorDataset(data.astype(int))


def export_analysis_results(
    results: Dict, filename: str = "sensor_analysis_results"
) -> None:
    """Export analysis results to JSON/CSV files."""
    try:
        import json

        json_filename = f"{filename}.json"
        with open(json_filename, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results exported to %s", json_filename)
        if "causality_results" in results and isinstance(
            results["causality_results"], pd.DataFrame
        ):
            csv_filename = f"{filename}_causality.csv"
            results["causality_results"].to_csv(csv_filename, index=False)
            logger.info("Causality results exported to %s", csv_filename)
    except ImportError:
        logger.warning("JSON export not available")
    except Exception as e:
        logger.error("Export failed: %s", e)
