"""Occupancy, visitors, and uncertainty-aware attribution of activity.

Ambient sensors observe a home, not a person. This package estimates who is
present and converts that into the probability that the monitored resident --
rather than a visitor or carer -- generated what each sensor saw. It uses
only anonymous evidence, and deliberately never attempts biometric identity.
"""

from .occupancy import (
    CONTEXTS,
    DEFAULT_BEACON_PRESENCE,
    DEFAULT_CONTEXT_DWELL,
    DEFAULT_RESIDENT_SHARE,
    DEFAULT_TRACK_COUNTS,
    ContextConfig,
    ContextEstimate,
    OccupancyContext,
    ResidentContextEstimator,
    rooms_active_at,
)

__all__ = [
    "CONTEXTS",
    "DEFAULT_BEACON_PRESENCE",
    "DEFAULT_CONTEXT_DWELL",
    "DEFAULT_RESIDENT_SHARE",
    "DEFAULT_TRACK_COUNTS",
    "ContextConfig",
    "ContextEstimate",
    "OccupancyContext",
    "ResidentContextEstimator",
    "rooms_active_at",
]
