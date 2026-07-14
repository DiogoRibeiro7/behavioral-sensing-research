"""Synthetic data generation utilities for benchmarking."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd

from sensor_modeling.utils.data_io import SensorDataset

logger = logging.getLogger(__name__)


@dataclass
class SyntheticConfig:
    n_steps: int = 1000
    n_sensors: int = 3
    change_points: list[int] | None = None
    failure_rate: float = 0.0
    seed: int = 0


def _validate_config(config: SyntheticConfig) -> list[int]:
    """Validate a synthetic generation config and return change points."""
    if config.n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if config.n_sensors < 1:
        raise ValueError("n_sensors must be at least 1")
    if not 0 <= config.failure_rate <= 1:
        raise ValueError("failure_rate must be between 0 and 1")

    change_points = config.change_points or [config.n_steps // 2]
    invalid = [
        cp
        for cp in change_points
        if not isinstance(cp, Integral)
        or isinstance(cp, bool)
        or cp < 0
        or cp >= config.n_steps
    ]
    if invalid:
        raise ValueError(
            "change_points must be integer offsets between 0 and n_steps - 1"
        )
    return change_points


def generate(config: SyntheticConfig) -> tuple[SensorDataset, dict[str, list[int]]]:
    """Generate synthetic sensor data with ground truth change points."""
    cps = _validate_config(config)
    rng = np.random.default_rng(config.seed)
    probs = np.zeros((config.n_steps, config.n_sensors)) + 0.1
    for cp in cps:
        probs[cp:] += 0.5  # behavioral change after change point
    probs = np.clip(probs, 0.0, 1.0)
    data = rng.binomial(1, probs)
    # introduce sensor failures
    for s in range(config.n_sensors):
        if rng.random() < config.failure_rate:
            fail_start = rng.integers(0, max(config.n_steps // 2, 1))
            data[fail_start:, s] = 0
            logger.warning(
                "Injected failure in sensor %s starting at %s", s, fail_start
            )
    index = pd.date_range("2024-01-01", periods=config.n_steps, freq="1min")
    df = pd.DataFrame(
        data, index=index, columns=[f"sensor_{i}" for i in range(config.n_sensors)]
    )
    return SensorDataset(df), {"change_points": cps}


def export(
    dataset: SensorDataset,
    metadata: Mapping[str, object],
    path: str | PathLike[str],
    fmt: str = "csv",
) -> dict[str, Path]:
    """Export synthetic dataset and metadata in multiple formats."""
    if fmt not in {"csv", "json", "hdf5"}:
        raise ValueError(f"Unsupported export format: {fmt}")

    df = dataset.to_dataframe()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(output_path)
        metadata_path = Path(f"{output_path}.meta.json")
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f)
        output_paths = {"data": output_path, "metadata": metadata_path}
    elif fmt == "json":
        records_df = df.reset_index().rename(
            columns={df.index.name or "index": "timestamp"}
        )
        records_df["timestamp"] = records_df["timestamp"].astype(str)
        records = records_df.to_dict(orient="records")
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"data": records, "meta": metadata}, f)
        output_paths = {"data": output_path}
    else:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - dependency is installed in CI
            raise ImportError("h5py is required for HDF5 export") from exc

        with h5py.File(output_path, "w") as h5:
            dset = h5.create_dataset("data", data=df.values)
            dset.attrs["timestamp"] = df.index.astype(str).to_list()
            h5.create_dataset("meta", data=json.dumps(metadata).encode("utf-8"))
        output_paths = {"data": output_path}

    logger.info("Exported synthetic dataset to %s (format=%s)", output_path, fmt)
    return output_paths
