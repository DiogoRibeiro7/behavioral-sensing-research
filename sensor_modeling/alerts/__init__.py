"""Restrained, explainable alerting.

An unusual observation is not an alert, and neither is every behavioural
change. This package applies the last filter -- magnitude, duration,
observation quality, attribution, deduplication and rate limiting -- and
keeps alerts about the sensing apparatus strictly separate from alerts about
the resident.
"""

from .alert import (
    Alert,
    AlertEngine,
    AlertKind,
    AlertPolicy,
    AlertSeverity,
    unresolved_kinds,
)

__all__ = [
    "Alert",
    "AlertEngine",
    "AlertKind",
    "AlertPolicy",
    "AlertSeverity",
    "unresolved_kinds",
]
