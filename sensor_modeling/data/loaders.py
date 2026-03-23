"""Flexible data loaders for multiple sensor data formats."""
from __future__ import annotations

import json
import logging
from collections.abc import Generator, Iterable
from typing import Dict

import pandas as pd

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    h5py = None  # type: ignore

from sensor_modeling.utils.data_io import SensorDataset

logger = logging.getLogger(__name__)


def load_csv(path: str, timestamp_col: str = "timestamp", **kwargs) -> SensorDataset:
    """Load sensor readings from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    timestamp_col : str, default="timestamp"
        Column containing timestamp information.

    Returns
    -------
    SensorDataset
        Dataset containing sensor readings indexed by timestamps.
    """
    try:
        df = pd.read_csv(path, parse_dates=[timestamp_col], **kwargs)
    except Exception as exc:
        logger.error("Failed to read CSV %s: %s", path, exc)
        raise ValueError(f"Unable to read CSV file: {path}") from exc
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' missing from CSV")
    df = df.set_index(timestamp_col).sort_index()
    logger.info("Loaded CSV with shape %s from %s", df.shape, path)
    return SensorDataset(df)


def load_json(path: str, timestamp_field: str = "timestamp") -> SensorDataset:
    """Load sensor event log data from a JSON file."""
    try:
        with open(path) as f:
            records = json.load(f)
    except Exception as exc:
        logger.error("Failed to read JSON %s: %s", path, exc)
        raise ValueError(f"Unable to read JSON file: {path}") from exc
    if not isinstance(records, list):
        raise ValueError("JSON file must contain a list of records")
    try:
        df = pd.DataFrame(records)
    except Exception as exc:
        logger.error("JSON structure invalid: %s", exc)
        raise ValueError("Invalid JSON structure for tabular data") from exc
    if timestamp_field not in df.columns:
        raise ValueError(
            f"Timestamp field '{timestamp_field}' missing from JSON records"
        )
    df[timestamp_field] = pd.to_datetime(df[timestamp_field])
    df = df.set_index(timestamp_field).sort_index()
    logger.info("Loaded JSON with shape %s from %s", df.shape, path)
    return SensorDataset(df)


def load_hdf5(path: str, key: str = "data") -> SensorDataset:
    """Load sensor data from an HDF5 file."""
    if h5py is None:
        raise ImportError("h5py is required for HDF5 support")
    try:
        with h5py.File(path, "r") as h5:
            if key not in h5:
                raise ValueError(f"Dataset '{key}' not found in HDF5 file")
            data = pd.DataFrame(h5[key][:])
            if "timestamp" in h5[key].attrs:
                ts = pd.to_datetime(h5[key].attrs["timestamp"])
                data.index = ts
    except Exception as exc:
        logger.error("Failed to read HDF5 %s: %s", path, exc)
        raise ValueError(f"Unable to read HDF5 file: {path}") from exc
    logger.info("Loaded HDF5 dataset '%s' with shape %s from %s", key, data.shape, path)
    return SensorDataset(data)


def stream_data(source: Iterable[Dict]) -> Generator[SensorDataset, None, None]:
    """Yield datasets from a real-time streaming source.

    Parameters
    ----------
    source : Iterable[Dict]
        Iterable producing dictionaries with sensor readings and timestamps.
    """
    for item in source:
        try:
            timestamp = item.get("timestamp")
            if timestamp is None:
                raise ValueError("Streaming item missing 'timestamp' field")
            df = pd.DataFrame([item]).set_index(pd.to_datetime([timestamp]))
            yield SensorDataset(df.drop(columns=["timestamp"]))
        except Exception as exc:
            logger.warning("Skipping malformed streaming item %s: %s", item, exc)
            continue
