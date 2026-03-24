"""Data cleaning and preparation routines for sensor data."""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from sensor_modeling.utils.data_io import SensorDataset

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
        df = df.ffill().bfill()
    elif strategy == "mean":
        df = df.fillna(df.mean())
    else:
        raise ValueError(f"Unsupported imputation strategy: {strategy}")
    logger.info("Imputed missing values using %s strategy", strategy)
    return SensorDataset(df)


def detect_outliers(dataset: SensorDataset, z_thresh: float = 3.0) -> pd.DataFrame:
    """Identify outlier readings using a z-score threshold."""
    df = dataset.to_dataframe()
    z = (df - df.mean()) / df.std(ddof=0)
    outliers = np.abs(z) > z_thresh
    logger.debug("Outlier counts per sensor: %s", outliers.sum().to_dict())
    return outliers


def align_sensors(
    datasets: List[SensorDataset], freq: str = "1min"
) -> List[SensorDataset]:
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


def data_quality_report(dataset: SensorDataset) -> Dict[str, float]:
    """Compute simple data quality metrics."""
    df = dataset.to_dataframe()
    report = {
        "missing_ratio": df.isna().mean().mean(),
        "outlier_ratio": detect_outliers(dataset).mean().mean(),
    }
    logger.info("Data quality report: %s", report)
    return report
