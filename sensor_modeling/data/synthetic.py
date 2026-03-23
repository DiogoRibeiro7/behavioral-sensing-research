"""Synthetic data generation utilities for benchmarking."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

from sensor_modeling.utils.data_io import SensorDataset

logger = logging.getLogger(__name__)


@dataclass
class SyntheticConfig:
    n_steps: int = 1000
    n_sensors: int = 3
    change_points: List[int] = None
    failure_rate: float = 0.0
    seed: int = 0


def generate(config: SyntheticConfig) -> Tuple[SensorDataset, Dict[str, List[int]]]:
    """Generate synthetic sensor data with ground truth change points."""
    rng = np.random.default_rng(config.seed)
    cps = config.change_points or [config.n_steps // 2]
    probs = np.zeros((config.n_steps, config.n_sensors)) + 0.1
    for cp in cps:
        probs[cp:] += 0.5  # behavioral change after change point
    data = rng.binomial(1, probs)
    # introduce sensor failures
    for s in range(config.n_sensors):
        if rng.random() < config.failure_rate:
            fail_start = rng.integers(0, config.n_steps // 2)
            data[fail_start:, s] = 0
            logger.warning(
                "Injected failure in sensor %s starting at %s", s, fail_start
            )
    index = pd.date_range("2024-01-01", periods=config.n_steps, freq="1min")
    df = pd.DataFrame(
        data, index=index, columns=[f"sensor_{i}" for i in range(config.n_sensors)]
    )
    return SensorDataset(df), {"change_points": cps}


def export(dataset: SensorDataset, metadata: Dict, path: str, fmt: str = "csv") -> None:
    """Export synthetic dataset and metadata in multiple formats."""
    df = dataset.to_dataframe()
    try:
        if fmt == "csv":
            df.to_csv(path)
            with open(f"{path}.meta.json", "w") as f:
                json.dump(metadata, f)
        elif fmt == "json":
            records = (
                df.reset_index()
                .rename(columns={df.index.name or "index": "timestamp"})
                .to_dict(orient="records")
            )
            with open(path, "w") as f:
                json.dump({"data": records, "meta": metadata}, f)
        elif fmt == "hdf5":
            if h5py is None:
                raise ImportError("h5py is required for HDF5 export")
            with h5py.File(path, "w") as h5:
                dset = h5.create_dataset("data", data=df.values)
                dset.attrs["timestamp"] = df.index.astype(str).to_list()
                h5.create_dataset("meta", data=json.dumps(metadata).encode("utf-8"))
        else:
            raise ValueError(f"Unsupported export format: {fmt}")
        logger.info("Exported synthetic dataset to %s (format=%s)", path, fmt)
    except Exception as exc:
        logger.error("Failed to export dataset: %s", exc)
        raise
