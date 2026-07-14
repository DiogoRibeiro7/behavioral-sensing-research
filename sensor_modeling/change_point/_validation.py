"""Validation helpers for lightweight change-point detectors."""

from __future__ import annotations

import numpy as np


def validate_positive_int(value: int, name: str) -> None:
    """Require a positive integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_series(series: np.ndarray) -> np.ndarray:
    """Return a one-dimensional finite numeric series."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if values.size == 0:
        raise ValueError("series must contain at least one observation")
    if not np.isfinite(values).all():
        raise ValueError("series must contain only finite values")
    return values


def validate_positive_threshold(threshold: float) -> float:
    """Return a positive finite detection threshold."""
    value = float(threshold)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("threshold must be positive and finite")
    return value
