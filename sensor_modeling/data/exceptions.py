"""Sensor-ingestion exceptions with DataExcept taxonomy and legacy compatibility."""

from __future__ import annotations

from dataexcept import (
    DataFormatError,
    DataLoadingError,
    DataValidationError,
    DependencyError,
    MissingDataError,
)


class SensorDataLoadingError(DataLoadingError, ValueError):
    """Data loading failure that remains catchable as the legacy ``ValueError``."""


class SensorDataFormatError(DataFormatError, ValueError):
    """Malformed sensor payload that remains catchable as ``ValueError``."""


class SensorDataValidationError(DataValidationError, ValueError):
    """Invalid sensor value that remains catchable as ``ValueError``."""


class SensorMissingDataError(MissingDataError, ValueError):
    """Missing required sensor field that remains catchable as ``ValueError``."""


class SensorDependencyError(DependencyError, ImportError):
    """Missing ingestion dependency that remains catchable as ``ImportError``."""


__all__ = [
    "SensorDataFormatError",
    "SensorDataLoadingError",
    "SensorDataValidationError",
    "SensorDependencyError",
    "SensorMissingDataError",
]
