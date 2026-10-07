"""Sensor health and reliability as part of the analytical model.

Sensor failures are treated as a first-class inference problem rather than an
operations concern. The output of this package is an evidence weight per
sensor that the fusion layer applies, which is what prevents a broken sensor
from being read as a resident who has stopped moving.
"""

from .fleet import (
    CommonSilence,
    FleetConfig,
    FleetSilence,
    assess_fleet,
    common_silences,
)
from .monitor import (
    HealthConfig,
    SensorHealthMonitor,
    SensorHealthReport,
    SystemHealthReport,
)
from .status import (
    FAULTY_STATUSES,
    STATUS_RELIABILITY,
    SensorStatus,
    status_reliability,
)

__all__ = [
    "FAULTY_STATUSES",
    "STATUS_RELIABILITY",
    "CommonSilence",
    "FleetConfig",
    "FleetSilence",
    "HealthConfig",
    "SensorHealthMonitor",
    "SensorHealthReport",
    "SensorStatus",
    "SystemHealthReport",
    "assess_fleet",
    "common_silences",
    "status_reliability",
]
