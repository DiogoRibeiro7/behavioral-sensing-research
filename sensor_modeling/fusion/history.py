"""Bounded, non-overlapping event evidence accumulation for v0.3.

The current filter consumes each pipeline step independently.  Real-home
feature ablations show that recent room-resolved event history contains signal
that the instantaneous evidence does not.  Re-feeding already-used events on
every step would be statistically wrong because the recursive posterior would
count the same observation repeatedly.

``EventEvidenceWindow`` therefore accumulates event observations until a fixed
window closes and releases each event exactly once.  State and sample evidence
are intentionally not accumulated here because their temporal semantics are
different.  The buffer is serialisable so an edge restart cannot silently drop
unconsumed evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..observations.observation import Observation, require_aware


@dataclass
class EventEvidenceWindow:
    """Accumulate event observations and release them exactly once per window."""

    width: timedelta
    _opened_at: datetime | None = None
    _events: list[Observation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.width <= timedelta(0):
            raise ValueError("evidence window width must be positive")

    @property
    def opened_at(self) -> datetime | None:
        """Start of the current window, if one has begun."""
        return self._opened_at

    @property
    def pending(self) -> tuple[Observation, ...]:
        """Unconsumed event evidence currently held in the window."""
        return tuple(self._events)

    def add(self, observation: Observation) -> None:
        """Add one event observation to the current window."""
        if not observation.is_event:
            raise ValueError("EventEvidenceWindow accepts event observations only")
        if self._opened_at is None:
            self._opened_at = observation.timestamp
        if observation.timestamp < self._opened_at:
            raise ValueError("event precedes the current evidence window")
        self._events.append(observation)
        self._events.sort(key=lambda item: item.timestamp)

    def ready(self, now: datetime) -> bool:
        """Whether the current window is due to be consumed at *now*."""
        moment = require_aware(now, "now")
        return self._opened_at is not None and moment >= self._opened_at + self.width

    def release(self, now: datetime, *, force: bool = False) -> tuple[Observation, ...]:
        """Release pending events once and begin a new window at *now*.

        Without ``force``, calling before the configured width has elapsed
        returns an empty tuple and leaves the buffer unchanged.
        """
        moment = require_aware(now, "now")
        if self._opened_at is None:
            self._opened_at = moment
            return ()
        if not force and not self.ready(moment):
            return ()
        released = tuple(self._events)
        self._events = []
        self._opened_at = moment
        return released

    def reset(self) -> None:
        """Discard pending evidence and clear the window clock."""
        self._opened_at = None
        self._events = []

    def snapshot(self) -> dict[str, object]:
        """Return JSON-serialisable restart state."""
        return {
            "width_seconds": self.width.total_seconds(),
            "opened_at": self._opened_at.isoformat() if self._opened_at else None,
            "events": [event.to_dict() for event in self._events],
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, object]) -> "EventEvidenceWindow":
        """Restore a window produced by :meth:`snapshot`."""
        window = cls(timedelta(seconds=float(payload["width_seconds"])))
        opened = payload.get("opened_at")
        window._opened_at = require_aware(opened, "opened_at") if opened else None
        raw_events = payload.get("events") or []
        if not isinstance(raw_events, list):
            raise TypeError("events must be a list")
        window._events = [Observation.from_dict(item) for item in raw_events]
        return window
