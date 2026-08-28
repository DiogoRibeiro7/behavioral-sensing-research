"""Sensor health states and the reliability weights they imply.

Sensor reliability is part of the analytical model, not an operational
afterthought. The central rule this module exists to enforce is that a failed
sensor must never be read as reduced human activity: when a sensor stops
reporting, the correct conclusion is that evidence is missing, not that
nothing happened.
"""

from __future__ import annotations

from enum import Enum


class SensorStatus(str, Enum):
    """Assessed operating condition of a sensor."""

    HEALTHY = "healthy"
    """Reporting as declared, with plausible values."""

    DEGRADED = "degraded"
    """Still reporting, but with reduced measurement quality."""

    MISSING = "missing"
    """Silent for far longer than its declared reporting interval."""

    DROPOUT = "dropout"
    """Silent long enough to suspect a fault, but not yet written off."""

    STUCK = "stuck"
    """Returning an unchanging value where variation is expected."""

    DRIFTING = "drifting"
    """Calibration appears to have shifted relative to its own history."""

    OUT_OF_RANGE = "out_of_range"
    """Reporting values outside the physically plausible declared range."""

    UNKNOWN = "unknown"
    """Not enough evidence to judge. The honest default, not a failure."""


#: Multiplicative reliability attached to each status.
#:
#: ``MISSING`` is deliberately zero: a sensor that is not reporting supplies no
#: evidence at all, and the fusion layer must fall back on other modalities
#: rather than treating its silence as an observation of inactivity.
#: ``UNKNOWN`` sits at an intermediate value because absence of a verdict is
#: not the same as a verdict of failure.
STATUS_RELIABILITY: dict[SensorStatus, float] = {
    SensorStatus.HEALTHY: 1.0,
    SensorStatus.DEGRADED: 0.6,
    SensorStatus.DRIFTING: 0.5,
    SensorStatus.UNKNOWN: 0.5,
    SensorStatus.OUT_OF_RANGE: 0.2,
    SensorStatus.STUCK: 0.1,
    SensorStatus.DROPOUT: 0.05,
    SensorStatus.MISSING: 0.0,
}

#: Statuses that indicate the sensor cannot currently be trusted as evidence.
FAULTY_STATUSES = frozenset(
    {
        SensorStatus.MISSING,
        SensorStatus.DROPOUT,
        SensorStatus.STUCK,
        SensorStatus.OUT_OF_RANGE,
    }
)


def status_reliability(status: SensorStatus) -> float:
    """Return the multiplicative reliability weight implied by *status*."""
    return STATUS_RELIABILITY[SensorStatus(status)]
