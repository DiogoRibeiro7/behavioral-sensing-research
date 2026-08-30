"""Measuring the event rates a real recording actually produces.

The emission defaults in :mod:`sensor_modeling.fusion.defaults` are *declared*:
chosen by reasoning about what a sensor should do in each state, not fitted to
anything. That is a defensible starting point and it is stated as such, but it
leaves an obvious question unanswered — are the declared numbers anywhere near
what real homes emit?

This module answers it for an annotated recording. For every labelled interval
it counts activations per sensor and divides by the interval's duration, giving
an empirical activations-per-hour for each ``(state, sensor)`` pair. Those can
be compared directly against ``EmissionDefaults.active_rate``, ``idle_rate`` and
``away_rate``.

It measures. It does not fit: nothing here writes back into the emission model,
because choosing rates from data needs held-out validation that a single
recording cannot provide.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import median

from ..states import BehaviouralState, StateOntology
from .casas import CasasRecording

logger = logging.getLogger(__name__)

#: Intervals shorter than this are dropped when measuring a rate.
#:
#: A rate is a count divided by a duration, so a very short interval turns one
#: stray activation into an enormous apparent rate. Three minutes is long
#: enough that a single event cannot dominate.
MIN_INTERVAL_HOURS = 0.05


@dataclass(frozen=True)
class RateSample:
    """Observed activation rates for one state, split by sensor placement.

    Attributes
    ----------
    state
        The annotated behavioural state.
    in_room
        Rates from sensors in the room the ontology associates with this state.
        Empty for states the ontology does not place in a room.
    elsewhere
        Rates from sensors in any other room.
    overall
        Total activations per hour across every sensor. This is the only
        measurement available for states the ontology does not place in a room
        -- ``HOME_INACTIVE`` and ``AWAY`` among them -- and those are precisely
        the states whose declared rates are hardest to reason about.
    intervals
        How many annotated intervals contributed.
    """

    state: BehaviouralState
    in_room: tuple[float, ...] = ()
    elsewhere: tuple[float, ...] = ()
    overall: tuple[float, ...] = ()
    intervals: int = 0

    @property
    def median_in_room(self) -> float | None:
        """Median activations per hour from a sensor in the state's room."""
        return median(self.in_room) if self.in_room else None

    @property
    def median_elsewhere(self) -> float | None:
        """Median activations per hour from a sensor in another room."""
        return median(self.elsewhere) if self.elsewhere else None

    @property
    def median_overall(self) -> float | None:
        """Median total activations per hour across the whole deployment."""
        return median(self.overall) if self.overall else None

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable summary."""
        return {
            "state": self.state.value,
            "intervals": self.intervals,
            "median_in_room": self.median_in_room,
            "median_elsewhere": self.median_elsewhere,
            "median_overall": self.median_overall,
            "samples_in_room": len(self.in_room),
            "samples_elsewhere": len(self.elsewhere),
        }


@dataclass(frozen=True)
class RateReport:
    """Empirical event rates for a recording, per behavioural state."""

    samples: Mapping[BehaviouralState, RateSample] = field(default_factory=dict)

    def for_state(self, state: BehaviouralState) -> RateSample | None:
        """Return the sample for one state, if it was annotated at all."""
        return self.samples.get(state)

    def compare(self, active_rate: float, idle_rate: float) -> dict[str, object]:
        """Contrast the measured rates with two declared constants.

        The ratio is the interesting number. A declared rate that is out by an
        order of magnitude does not merely blur the posterior: in a Poisson
        likelihood it changes which state a given count favours.
        """
        rows = []
        for state, sample in sorted(self.samples.items(), key=lambda kv: kv[0].value):
            observed = sample.median_in_room
            rows.append(
                {
                    "state": state.value,
                    "declared_in_room": active_rate,
                    "observed_in_room": observed,
                    "ratio": (
                        (observed / active_rate) if observed and active_rate else None
                    ),
                    "declared_elsewhere": idle_rate,
                    "observed_elsewhere": sample.median_elsewhere,
                    "observed_overall": sample.median_overall,
                    "intervals": sample.intervals,
                }
            )
        return {"active_rate": active_rate, "idle_rate": idle_rate, "states": rows}

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable summary."""
        return {state.value: sample.to_dict() for state, sample in self.samples.items()}


def measure_event_rates(
    recording: CasasRecording,
    *,
    ontology: StateOntology | None = None,
    min_interval_hours: float = MIN_INTERVAL_HOURS,
) -> RateReport:
    """Measure activations per hour per state from an annotated recording.

    Parameters
    ----------
    recording
        A parsed recording carrying both observations and annotations.
    ontology
        Supplies the room associated with each state, which decides whether a
        sensor counts as in-room or elsewhere. Defaults to the standard one.
    min_interval_hours
        Intervals shorter than this are skipped; see :data:`MIN_INTERVAL_HOURS`.

    Notes
    -----
    Only intervals whose label mapped to a state are used. Unmapped annotation
    is skipped rather than pooled, since its state is unknown by construction.
    """
    states = ontology or StateOntology()
    observations = recording.observations
    if not observations:
        raise ValueError("recording contains no observations to measure")

    rooms = {spec.sensor_id: spec.room for spec in recording.registry}
    timestamps = [observation.timestamp for observation in observations]

    in_room: dict[BehaviouralState, list[float]] = {}
    elsewhere: dict[BehaviouralState, list[float]] = {}
    overall: dict[BehaviouralState, list[float]] = {}
    counts: dict[BehaviouralState, int] = {}

    for interval in recording.activities:
        if interval.state is None:
            continue
        hours = (interval.end - interval.start).total_seconds() / 3600.0
        if hours < min_interval_hours:
            continue

        low = bisect.bisect_left(timestamps, interval.start)
        high = bisect.bisect_left(timestamps, interval.end)
        window = observations[low:high]

        seen: dict[str, int] = {}
        for observation in window:
            seen[observation.sensor_id] = seen.get(observation.sensor_id, 0) + 1

        target = states.room_of(interval.state)
        counts[interval.state] = counts.get(interval.state, 0) + 1
        overall.setdefault(interval.state, []).append(len(window) / hours)
        for sensor_id, room in rooms.items():
            rate = seen.get(sensor_id, 0) / hours
            if target is not None and room == target:
                in_room.setdefault(interval.state, []).append(rate)
            elif target is not None:
                elsewhere.setdefault(interval.state, []).append(rate)

    samples = {
        state: RateSample(
            state=state,
            in_room=tuple(in_room.get(state, ())),
            elsewhere=tuple(elsewhere.get(state, ())),
            overall=tuple(overall.get(state, ())),
            intervals=intervals,
        )
        for state, intervals in counts.items()
    }
    if not samples:
        raise ValueError(
            "no annotated interval was long enough to measure a rate from; "
            "check the activity mapping and min_interval_hours"
        )
    return RateReport(samples=samples)


def pooled_rate_report(
    recordings: Sequence[CasasRecording],
    *,
    ontology: StateOntology | None = None,
) -> RateReport:
    """Measure rates across several recordings, pooling the samples.

    Pooling homes is appropriate for describing what this instrumentation
    produces in general. It is *not* a fitting procedure: choosing emission
    rates from data requires holding homes out, which this does not do.
    """
    if not recordings:
        raise ValueError("at least one recording is required")

    merged_in: dict[BehaviouralState, list[float]] = {}
    merged_out: dict[BehaviouralState, list[float]] = {}
    merged_all: dict[BehaviouralState, list[float]] = {}
    merged_n: dict[BehaviouralState, int] = {}

    for recording in recordings:
        report = measure_event_rates(recording, ontology=ontology)
        for state, sample in report.samples.items():
            merged_in.setdefault(state, []).extend(sample.in_room)
            merged_out.setdefault(state, []).extend(sample.elsewhere)
            merged_all.setdefault(state, []).extend(sample.overall)
            merged_n[state] = merged_n.get(state, 0) + sample.intervals

    return RateReport(
        samples={
            state: RateSample(
                state=state,
                in_room=tuple(merged_in.get(state, ())),
                elsewhere=tuple(merged_out.get(state, ())),
                overall=tuple(merged_all.get(state, ())),
                intervals=intervals,
            )
            for state, intervals in merged_n.items()
        }
    )
