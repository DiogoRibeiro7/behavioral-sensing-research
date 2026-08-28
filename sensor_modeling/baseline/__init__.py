"""Adaptive personal baselines over non-stationary behaviour.

Normal behaviour is modelled as a slowly moving distribution rather than a
fixed calibration window, and the reasons a day can look unusual -- ordinary
variability, weekly rhythm, a temporary disturbance, a persistent change, a
gradual drift, or simply a day the sensors did not watch -- are distinguished
rather than collapsed into a single anomaly score.
"""

from .adaptive import (
    MAD_TO_SIGMA,
    AdaptiveBaseline,
    BaselineConfig,
    BaselineReference,
    BehaviouralChange,
    ChangeKind,
)
from .features import DailySummary, feature_series, summarise_days

__all__ = [
    "MAD_TO_SIGMA",
    "AdaptiveBaseline",
    "BaselineConfig",
    "BaselineReference",
    "BehaviouralChange",
    "ChangeKind",
    "DailySummary",
    "feature_series",
    "summarise_days",
]
