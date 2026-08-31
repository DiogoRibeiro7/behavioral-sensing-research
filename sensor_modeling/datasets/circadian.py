"""Fit circadian state-dynamics profiles from annotated development homes.

The optional circadian prior in :class:`sensor_modeling.states.StateOntology`
uses 24 positive stickiness multipliers per state.  This module turns labelled
CASAS intervals into those multipliers in a deterministic, auditable way.

For state ``z`` and local hour ``h`` the raw multiplier is the occupancy lift

    P(Z=z | H=h) / P(Z=z).

A small equivalent-duration prior shrinks sparse hour/state cells back toward
one, and the final multipliers are clipped to a declared positive range.  The
fit uses annotations only; it never evaluates the behavioural inference model
and therefore cannot inspect external-test outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import math

from ..states import BehaviouralState, StateOntology
from .casas import CasasRecording


@dataclass(frozen=True)
class CircadianProfileFit:
    """A fitted per-state 24-hour stickiness profile and its provenance."""

    profile: Mapping[BehaviouralState, tuple[float, ...]]
    labelled_seconds: float
    recordings: int
    shrinkage_hours: float
    minimum_multiplier: float
    maximum_multiplier: float

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serialisable representation."""
        return {
            "recordings": self.recordings,
            "labelled_seconds": self.labelled_seconds,
            "shrinkage_hours": self.shrinkage_hours,
            "minimum_multiplier": self.minimum_multiplier,
            "maximum_multiplier": self.maximum_multiplier,
            "profile": {
                state.value: [float(value) for value in values]
                for state, values in sorted(
                    self.profile.items(), key=lambda item: item[0].value
                )
            },
        }

    def sha256(self) -> str:
        """Return the SHA-256 of the canonical serialised fit."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _split_hourly(start: datetime, end: datetime) -> list[tuple[int, float]]:
    """Split ``[start, end)`` into local-hour pieces.

    Returns ``(hour, seconds)`` pairs.  The timestamps are expected to already
    be timezone-aware local wall-clock instants, as required by the CASAS
    readers.  Splitting by the next wall-clock hour preserves the intended
    circadian semantics across arbitrary interval lengths.
    """
    if end <= start:
        return []
    pieces: list[tuple[int, float]] = []
    cursor = start
    while cursor < end:
        boundary = cursor.replace(minute=0, second=0, microsecond=0) + timedelta(
            hours=1
        )
        stop = min(boundary, end)
        seconds = (stop - cursor).total_seconds()
        if seconds > 0.0:
            pieces.append((cursor.hour, seconds))
        cursor = stop
    return pieces


def fit_circadian_profile(
    recordings: Sequence[CasasRecording],
    *,
    ontology: StateOntology | None = None,
    shrinkage_hours: float = 2.0,
    minimum_multiplier: float = 0.25,
    maximum_multiplier: float = 4.0,
) -> CircadianProfileFit:
    """Fit 24-hour state stickiness multipliers from annotated recordings.

    Parameters
    ----------
    recordings
        Development-only annotated recordings.  Test homes must never be
        passed here.
    ontology
        State set to fit.  Defaults to the standard seven-state ontology.
    shrinkage_hours
        Equivalent labelled time per hour used to shrink the conditional state
        share toward the overall state share.  A value of zero disables
        shrinkage.
    minimum_multiplier, maximum_multiplier
        Positive clipping bounds.  They prevent a sparse cell from making a
        state effectively impossible to leave or implausibly sticky.

    Notes
    -----
    The fit uses only mapped annotation durations.  Sensor observations and
    inference predictions are not consulted.  A state absent from the
    development annotations receives an all-ones profile rather than an
    invented circadian pattern.
    """
    if not recordings:
        raise ValueError("at least one recording is required")
    if not math.isfinite(shrinkage_hours) or shrinkage_hours < 0.0:
        raise ValueError("shrinkage_hours must be finite and non-negative")
    if (
        not math.isfinite(minimum_multiplier)
        or not math.isfinite(maximum_multiplier)
        or minimum_multiplier <= 0.0
        or maximum_multiplier < minimum_multiplier
    ):
        raise ValueError("multiplier bounds must be positive and ordered")

    states = ontology or StateOntology()
    per_state_hour = {
        state: [0.0 for _ in range(24)] for state in states.states
    }
    per_hour = [0.0 for _ in range(24)]
    per_state = {state: 0.0 for state in states.states}
    labelled_total = 0.0

    for recording in recordings:
        for interval in recording.activities:
            state = interval.state
            if state is None or state not in per_state_hour:
                continue
            for hour, seconds in _split_hourly(interval.start, interval.end):
                per_state_hour[state][hour] += seconds
                per_hour[hour] += seconds
                per_state[state] += seconds
                labelled_total += seconds

    if labelled_total <= 0.0:
        raise ValueError("recordings contain no mapped labelled duration")

    shrinkage_seconds = shrinkage_hours * 3600.0
    profile: dict[BehaviouralState, tuple[float, ...]] = {}
    for state in states.states:
        overall_share = per_state[state] / labelled_total
        if overall_share <= 0.0:
            profile[state] = tuple(1.0 for _ in range(24))
            continue

        values: list[float] = []
        for hour in range(24):
            hour_total = per_hour[hour]
            if hour_total <= 0.0:
                lift = 1.0
            else:
                conditional_share = (
                    per_state_hour[state][hour]
                    + shrinkage_seconds * overall_share
                ) / (hour_total + shrinkage_seconds)
                lift = conditional_share / overall_share
            values.append(
                min(max(float(lift), minimum_multiplier), maximum_multiplier)
            )
        profile[state] = tuple(values)

    return CircadianProfileFit(
        profile=profile,
        labelled_seconds=labelled_total,
        recordings=len(recordings),
        shrinkage_hours=shrinkage_hours,
        minimum_multiplier=minimum_multiplier,
        maximum_multiplier=maximum_multiplier,
    )
