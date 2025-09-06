"""Utilities for handling missing sensor data."""
from __future__ import annotations

import pandas as pd


def forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values in a DataFrame."""
    return df.ffill()


def interpolate_linear(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate missing values in a DataFrame."""
    return df.interpolate(method="linear")
