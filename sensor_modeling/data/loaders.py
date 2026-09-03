"""Flexible data loaders for multiple sensor data formats."""

from __future__ import annotations

import json
import logging
from collections.abc import Generator, Iterable, Mapping
from os import PathLike

import pandas as pd
from dataexcept import (
    DataExceptError,
    DataFormatError,
    DataLoadingError,
    DataValidationError,
    DependencyError,
    MissingDataError,
)

from sensor_modeling.utils.data_io import SensorDataset, read_sensor_csv

logger = logging.getLogger(__name__)


def _parse_timestamps(values: object, field_name: str) -> pd.Series:
    """Parse timestamp values and reject missing or invalid entries."""
    timestamps = pd.to_datetime(values, errors="coerce")
    if pd.isna(timestamps).any():
        raise DataValidationError(
            field_name,
            "<invalid timestamp>",
            f"Timestamp field '{field_name}' contains invalid timestamps",
        )
    return timestamps


def load_csv(
    path: str | PathLike[str], timestamp_col: str = "timestamp", **kwargs
) -> SensorDataset:
    """Load sensor readings from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    timestamp_col : str, default="timestamp"
        Preferred timestamp column. If absent, an unnamed saved index is parsed
        as datetimes when possible; otherwise the CSV is loaded as a plain
        tabular sensor matrix.

    Returns
    -------
    SensorDataset
        Dataset containing sensor readings indexed by timestamps.
    """
    try:
        df = read_sensor_csv(path, timestamp_col=timestamp_col, **kwargs)
    except DataExceptError as exc:
        logger.error("Failed to read CSV %s: %s", path, exc)
        raise
    logger.info("Loaded CSV with shape %s from %s", df.shape, path)
    return SensorDataset(df)


def load_json(
    path: str | PathLike[str], timestamp_field: str = "timestamp"
) -> SensorDataset:
    """Load sensor event log data from a JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.error("Failed to read JSON %s: %s", path, exc)
        raise DataLoadingError(str(path), exc) from exc

    df = _records_to_frame(records, timestamp_field=timestamp_field)
    logger.info("Loaded JSON with shape %s from %s", df.shape, path)
    return SensorDataset(df)


def _records_to_frame(records: object, timestamp_field: str) -> pd.DataFrame:
    """Convert JSON records into a timestamp-indexed DataFrame."""
    if not isinstance(records, list):
        raise DataFormatError(["JSON list of records"], type(records).__name__)
    if any(not isinstance(record, Mapping) for record in records):
        raise DataFormatError(
            ["JSON list of object records"],
            "list containing non-object entries",
        )

    try:
        df = pd.DataFrame(records)
    except (TypeError, ValueError) as exc:
        logger.error("JSON structure invalid: %s", exc)
        raise DataFormatError(
            ["JSON list of object records"],
            f"invalid tabular structure ({exc})",
        ) from exc

    if timestamp_field not in df.columns:
        raise MissingDataError(
            timestamp_field,
            f"Timestamp field '{timestamp_field}' missing from JSON records",
        )

    df[timestamp_field] = _parse_timestamps(df[timestamp_field], timestamp_field)
    return df.set_index(timestamp_field).sort_index()


def load_hdf5(path: str | PathLike[str], key: str = "data") -> SensorDataset:
    """Load sensor data from an HDF5 file."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency is installed in CI
        raise DependencyError("h5py", "h5py is required for HDF5 support") from exc

    try:
        with h5py.File(path, "r") as h5:
            if key not in h5:
                raise MissingDataError(
                    key,
                    f"Dataset '{key}' not found in HDF5 file",
                )
            data = pd.DataFrame(h5[key][:])
            if "timestamp" in h5[key].attrs:
                data.index = _parse_timestamps(
                    h5[key].attrs["timestamp"],
                    "timestamp",
                )
    except OSError as exc:
        logger.error("Failed to read HDF5 %s: %s", path, exc)
        raise DataLoadingError(str(path), exc) from exc
    logger.info("Loaded HDF5 dataset '%s' with shape %s from %s", key, data.shape, path)
    return SensorDataset(data)


def stream_data(source: Iterable[object]) -> Generator[SensorDataset, None, None]:
    """Yield datasets from a real-time streaming source.

    Parameters
    ----------
    source : Iterable[Dict]
        Iterable producing dictionaries with sensor readings and timestamps.
    """
    for item in source:
        try:
            yield _stream_item_to_dataset(item)
        except DataExceptError as exc:
            logger.warning("Skipping malformed streaming item %s: %s", item, exc)
            continue


def _stream_item_to_dataset(item: object) -> SensorDataset:
    """Convert one streaming record into a single-row dataset."""
    if not isinstance(item, Mapping):
        raise DataFormatError(["mapping"], type(item).__name__)

    timestamp = item.get("timestamp")
    if timestamp is None:
        raise MissingDataError(
            "timestamp",
            "Streaming item missing 'timestamp' field",
        )

    timestamp_index = _parse_timestamps([timestamp], "timestamp")
    df = pd.DataFrame([item]).set_index(timestamp_index)
    return SensorDataset(df.drop(columns=["timestamp"]))
