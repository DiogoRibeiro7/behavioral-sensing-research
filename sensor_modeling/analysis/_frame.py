"""Shared DataFrame validation helpers for analysis routines."""

from __future__ import annotations

import pandas as pd


def prepare_sensor_frame(data: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Return a numeric sensor frame with a datetime index."""
    if data.empty:
        raise ValueError(f"{context} requires at least one row")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(f"{context} requires a pandas DatetimeIndex")

    sensor_data = data.select_dtypes(include="number")
    if sensor_data.empty:
        raise ValueError(f"{context} requires at least one numeric sensor column")

    return sensor_data.copy()
