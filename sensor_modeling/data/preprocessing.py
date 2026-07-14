"""Data cleaning and preparation routines for sensor data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sensor_modeling.utils.data_io import SensorDataset
from sensor_modeling.utils.missing import handle_missing_data

logger = logging.getLogger(__name__)


def detect_missing(dataset: SensorDataset) -> pd.Series:
    """Return the fraction of missing values per sensor."""
    df = dataset.to_dataframe()
    miss = df.isna().mean()
    logger.debug("Missing value ratios: %s", miss.to_dict())
    return miss


def impute_missing(dataset: SensorDataset, strategy: str = "ffill") -> SensorDataset:
    """Impute missing values using the specified strategy."""
    df = dataset.to_dataframe().copy()
    if strategy == "ffill":
        df = handle_missing_data(df, strategy="gap_aware").data
    elif strategy == "interpolate":
        df = handle_missing_data(df, strategy="interpolate").data
    elif strategy == "mean":
        numeric_means = df.select_dtypes(include="number").mean()
        df = df.fillna(numeric_means)
    else:
        raise ValueError(f"Unsupported imputation strategy: {strategy}")
    logger.info("Imputed missing values using %s strategy", strategy)
    return SensorDataset(df)


def detect_outliers(dataset: SensorDataset, z_thresh: float = 3.0) -> pd.DataFrame:
    """Identify outlier readings using a z-score threshold."""
    if z_thresh <= 0:
        raise ValueError("z_thresh must be positive")

    df = dataset.to_dataframe()
    outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        std = numeric.std(ddof=0).replace(0, np.nan)
        z = (numeric - numeric.mean()) / std
        outliers.loc[:, numeric.columns] = (np.abs(z) > z_thresh).fillna(False)
    logger.debug("Outlier counts per sensor: %s", outliers.sum().to_dict())
    return outliers


def align_sensors(
    datasets: list[SensorDataset], freq: str = "1min"
) -> list[SensorDataset]:
    """Temporal alignment across multiple sensors/datasets."""
    if not datasets:
        raise ValueError("No datasets provided for alignment")
    target_index = pd.date_range(
        start=min(ds.to_dataframe().index.min() for ds in datasets),
        end=max(ds.to_dataframe().index.max() for ds in datasets),
        freq=freq,
    )
    aligned = []
    for ds in datasets:
        df = ds.to_dataframe().reindex(target_index).interpolate()
        aligned.append(SensorDataset(df))
    logger.info("Aligned %d datasets to frequency %s", len(datasets), freq)
    return aligned


def data_quality_report(dataset: SensorDataset) -> dict[str, float]:
    """Compute simple data quality metrics."""
    df = dataset.to_dataframe()
    report = {
        "missing_ratio": 0.0 if df.empty else float(df.isna().mean().mean()),
        "outlier_ratio": float(detect_outliers(dataset).mean().mean()),
    }
    logger.info("Data quality report: %s", report)
    return report
