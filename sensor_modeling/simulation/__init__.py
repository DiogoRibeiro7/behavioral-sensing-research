"""Synthetic households with controlled ground truth.

Behaviour is generated from a stochastic daily *schedule*, deliberately not
from the continuous-time chain the inference layer uses. Sharing a generative
model between simulator and estimator would make good results prove only that
the code can invert its own assumptions.

Sensor faults are injected separately from behaviour, so a robustness study
can hold the resident fixed and vary only what went wrong with the apparatus.
"""

from .faults import (
    DegradationConfig,
    Fault,
    FaultKind,
    degrade,
    dropout,
    not_worn,
    stuck,
)
from .household import (
    ACTIVE_RATE,
    IDLE_RATE,
    WEARABLE_LEVEL,
    BehaviourShift,
    Episode,
    GroundTruth,
    HouseholdConfig,
    SimulationResult,
    VisitorPeriod,
    build_registry,
    simulate,
)

__all__ = [
    "ACTIVE_RATE",
    "IDLE_RATE",
    "WEARABLE_LEVEL",
    "BehaviourShift",
    "DegradationConfig",
    "Episode",
    "Fault",
    "FaultKind",
    "GroundTruth",
    "HouseholdConfig",
    "SimulationResult",
    "VisitorPeriod",
    "build_registry",
    "degrade",
    "dropout",
    "not_worn",
    "simulate",
    "stuck",
]
