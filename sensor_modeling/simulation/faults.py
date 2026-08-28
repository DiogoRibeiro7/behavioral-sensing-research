"""Degrading a clean sensor record into a realistic one.

Fault injection is kept separate from behaviour generation so that a
robustness study can hold the resident's behaviour fixed and vary only what
went wrong with the apparatus. That separation is what makes a paired
comparison possible: the same simulated fortnight, once with a working
wearable and once without, differs in exactly one thing.

The faults available correspond to what actually happens in deployments:

.. code-block:: text

    dropout            a sensor stops reporting for a while
    stuck              a sensor keeps reporting its last value
    missing            records are lost in transit at random
    non_adherence      a wearable is not worn
    late_arrival       records reach the system out of order
    duplication        the same record is delivered twice
    clock_drift        one gateway's clock runs fast or slow
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum

import numpy as np

from ..observations.observation import Observation

logger = logging.getLogger(__name__)


class FaultKind(str, Enum):
    """A way in which a sensor record can be degraded."""

    DROPOUT = "dropout"
    """The sensor stops reporting entirely for a window."""

    STUCK = "stuck"
    """The sensor keeps reporting the value it had when the fault began."""

    NON_ADHERENCE = "non_adherence"
    """A wearable is not worn, so it reports nothing about the resident."""


@dataclass(frozen=True)
class Fault:
    """One injected sensor fault over a known window."""

    sensor_id: str
    kind: FaultKind
    start: datetime
    end: datetime

    def covers(self, observation: Observation) -> bool:
        """Whether *observation* falls inside this fault's window."""
        return (
            observation.sensor_id == self.sensor_id
            and self.start <= observation.timestamp < self.end
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the fault."""
        return {
            "sensor_id": self.sensor_id,
            "kind": self.kind.value,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


@dataclass
class DegradationConfig:
    """How a clean record should be degraded.

    Parameters
    ----------
    missing_rate
        Probability that any individual record is lost in transit.
    duplication_rate
        Probability that a record is delivered twice.
    late_rate
        Probability that a record arrives materially after its timestamp.
    late_delay
        How late such a record arrives.
    clock_drift
        Per-source timestamp offset to apply, simulating a mis-set gateway
        clock. Keys are source names as they appear on observations.
    faults
        Explicit sensor faults to inject.
    seed
        Random seed, so degradation is reproducible independently of the
        behaviour it is applied to.
    """

    missing_rate: float = 0.0
    duplication_rate: float = 0.0
    late_rate: float = 0.0
    late_delay: timedelta = timedelta(minutes=15)
    clock_drift: dict[str, timedelta] = field(default_factory=dict)
    faults: tuple[Fault, ...] = ()
    seed: int = 7

    def __post_init__(self) -> None:
        """Validate the degradation configuration."""
        for name in ("missing_rate", "duplication_rate", "late_rate"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.late_delay < timedelta(0):
            raise ValueError("late_delay must be non-negative")
        for fault in self.faults:
            if fault.end <= fault.start:
                raise ValueError("fault windows must have positive duration")


def degrade(
    observations: Iterable[Observation], config: DegradationConfig
) -> tuple[list[Observation], list[Observation]]:
    """Apply degradation to a clean record.

    Returns
    -------
    tuple[list[Observation], list[Observation]]
        The degraded records in arrival order, and the records that were
        dropped. The dropped list exists so that an evaluation can report how
        much evidence was actually withheld rather than having to infer it.
    """
    rng = np.random.default_rng(config.seed)
    kept: list[Observation] = []
    dropped: list[Observation] = []
    stuck_values: dict[str, float] = {}

    for observation in observations:
        fault = next((f for f in config.faults if f.covers(observation)), None)
        if fault is not None:
            if fault.kind in (FaultKind.DROPOUT, FaultKind.NON_ADHERENCE):
                dropped.append(observation)
                continue
            if fault.kind is FaultKind.STUCK:
                held = stuck_values.setdefault(observation.sensor_id, observation.value)
                observation = replace(observation, value=held)
        else:
            stuck_values.pop(observation.sensor_id, None)

        if config.missing_rate and rng.random() < config.missing_rate:
            dropped.append(observation)
            continue

        drift = config.clock_drift.get(observation.source)
        if drift:
            observation = observation.shifted(drift, flag=False)

        received = observation.timestamp
        if config.late_rate and rng.random() < config.late_rate:
            received = received + config.late_delay
        observation = replace(observation, received_at=received)

        kept.append(observation)
        if config.duplication_rate and rng.random() < config.duplication_rate:
            kept.append(observation)

    # Sort by arrival rather than by event time: that is the order a live
    # system genuinely sees, and it is what exercises out-of-order handling.
    kept.sort(key=lambda obs: (obs.received_at or obs.timestamp))
    logger.info("Degraded record: %d delivered, %d withheld", len(kept), len(dropped))
    return kept, dropped


def dropout(sensor_id: str, start: datetime, duration: timedelta) -> Fault:
    """Build a dropout fault of *duration* starting at *start*."""
    return Fault(sensor_id, FaultKind.DROPOUT, start, start + duration)


def stuck(sensor_id: str, start: datetime, duration: timedelta) -> Fault:
    """Build a stuck-value fault of *duration* starting at *start*."""
    return Fault(sensor_id, FaultKind.STUCK, start, start + duration)


def not_worn(
    sensor_ids: Sequence[str], start: datetime, duration: timedelta
) -> tuple[Fault, ...]:
    """Build non-adherence faults for every device on the resident's person."""
    return tuple(
        Fault(sensor_id, FaultKind.NON_ADHERENCE, start, start + duration)
        for sensor_id in sensor_ids
    )
