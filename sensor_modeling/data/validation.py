"""Data validation utilities."""

from __future__ import annotations

import logging

import pandas as pd

from sensor_modeling.utils.data_io import SensorDataset

logger = logging.getLogger(__name__)


def check_temporal_consistency(dataset: SensorDataset) -> bool:
    """Verify that timestamps are monotonic and evenly spaced."""
    idx = dataset.to_dataframe().index
    if not idx.is_monotonic_increasing:
        logger.error("Timestamps are not sorted")
        return False
    diffs = idx.to_series().diff().dropna().value_counts()
    if len(diffs) > 1:
        logger.warning("Irregular sampling detected: %s", diffs.to_dict())
        return False
    return True


def validate_sensor_ranges(
    dataset: SensorDataset, min_val: float = 0.0, max_val: float = 1.0
) -> bool:
    """Ensure sensor readings fall within expected bounds."""
    df = dataset.to_dataframe()
    valid = df.apply(lambda c: c.between(min_val, max_val).all()).all()
    if not valid:
        logger.error("Sensor readings outside [%s, %s]", min_val, max_val)
    return bool(valid)


def detect_sensor_failures(
    dataset: SensorDataset, window: int = 100
) -> dict[str, bool]:
    """Detect potential sensor failures using long constant stretches."""
    df = dataset.to_dataframe()
    failures: dict[str, bool] = {}
    for col in df.columns:
        series = df[col]
        rolling = series.rolling(window=window, min_periods=window)
        constant_windows = rolling.apply(
            lambda x: x.nunique(dropna=False) <= 1,
            raw=False,
        )
        failures[col] = bool(constant_windows.fillna(0).astype(bool).any())
        if failures[col]:
            logger.warning("Possible failure detected in sensor '%s'", col)
    return failures


def cross_sensor_correlation(dataset: SensorDataset) -> pd.DataFrame:
    """Compute correlation matrix across sensors."""
    corr = dataset.to_dataframe().corr()
    logger.debug("Sensor correlation matrix:\n%s", corr)
    return corr
