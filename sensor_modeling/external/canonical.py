"""Converting a validated external household into the repository's canonical form.

The output is the recording type the rest of the repository consumes: a sensor
registry, canonical observations in time order, and labelled intervals whose
state is ``None`` wherever nothing may be scored. Nothing is discarded without
being counted. :attr:`CanonicalHousehold.dispositions` says, by reason, how many
events, sensors and annotated seconds did not make it through.

Rules
-----
- **Refusal.** A household with a validation error is refused, unless
  ``strict=False``. Then its unknown sensors' events are counted and left out.
- **Timestamps.** Naive timestamps are placed in the household's declared
  timezone. In a daylight-saving fold, the earlier reading is used.
- **Order and duplicates.** Events are put in time order, and exact duplicates
  are kept once and counted.
- **Sensors.** A sensor whose type does not map exactly or approximately is
  left out, and so are its events. A sensor's room is its location's target,
  or ``None``.
- **Values.** An event sensor emits ``1`` for an activation token, counts
  deactivations and does not emit them. A state sensor emits ``1`` and ``0``.
  A numeric value is emitted as is. A value outside the declared semantics is
  counted and left out.
- **Annotations.** Interval annotations are split at every boundary.
  - **One state.** A segment gets a state when its usable annotations,
    whether exact or approximate, agree on one.
  - **No state.** It stays unscored, ``None``, when they conflict, when none is
    usable, or when more than one resident is present.
  - **Points.** Point annotations are counted and carry no span.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..datasets.casas import ActivityInterval, CasasRecording
from ..observations import Observation, ObservationKind, SensorRegistry, SensorSpec
from ..states import BehaviouralState
from .contract import AnnotationSemantics, ContractError, HouseholdData, RawEvent
from .mapping import OntologyMapping, Outcome, SensorSemantics
from .validation import localise, valid_interval, validate_household, zone_of

#: Why an annotated segment carries no state.
CONFLICT = "conflict"
MULTI_RESIDENT = "multi_resident"


@dataclass(frozen=True)
class CanonicalHousehold:
    """One external household in canonical form, with what was left behind.

    Attributes
    ----------
    household
        Which household.
    recording
        The registry, observations and labelled intervals the repository
        consumes. Unscored intervals carry ``state=None``.
    dispositions
        Counts by reason of everything not carried: events, sensors, values,
        and annotated seconds by outcome.
    """

    household: str
    recording: CasasRecording
    dispositions: Mapping[str, Any]


def _value(event: RawEvent, semantics: SensorSemantics) -> float | str | None:
    """The canonical value, ``"deactivation"`` to count, or ``None`` if it does not fit."""
    raw = event.value
    if isinstance(raw, float):
        return raw if semantics.numeric else None
    if raw in semantics.activation:
        return 1.0
    if raw in semantics.deactivation:
        return 0.0 if semantics.kind is ObservationKind.STATE else "deactivation"
    return None


def _segments(
    data: HouseholdData,
    mapping: OntologyMapping,
    zone: Any,
) -> tuple[list[ActivityInterval], dict[str, float], Counter[str]]:
    """Non-overlapping labelled segments, with seconds and labels by disposition."""
    intervals = [
        (localise(a.start, zone), localise(a.end, zone), a)  # type: ignore[arg-type]
        for a in data.annotations
        if valid_interval(a)
    ]
    crowded = [
        (localise(p.start, zone), localise(p.end, zone))
        for p in data.occupancy
        if p.residents > 1 and p.end > p.start
    ]
    seconds: dict[str, float] = defaultdict(float)
    unscored_labels: Counter[str] = Counter()
    for _, _, a in intervals:
        resolution = mapping.label(a.label)
        if not resolution.usable:
            unscored_labels[a.label] += 1
    # Split at multi-resident periods too, so only their overlap goes unscored.
    boundaries = sorted(
        {m for s, e, _ in intervals for m in (s, e)}
        | {m for s, e in crowded for m in (s, e)}
    )
    segments: list[ActivityInterval] = []
    for start, end in zip(boundaries, boundaries[1:]):
        active = [a for s, e, a in intervals if s <= start and e >= end]
        if not active:
            continue
        length = (end - start).total_seconds()
        residents = {a.resident for a in active if a.resident is not None}
        resolutions = [mapping.label(a.label) for a in active]
        states = {r.target for r in resolutions if r.usable}
        label = "+".join(sorted({a.label for a in active}))
        state: BehaviouralState | None = None
        if len(residents) > 1 or any(s < end and e > start for s, e in crowded):
            reason = MULTI_RESIDENT
        elif len(states) > 1:
            reason = CONFLICT
        elif len(states) == 1:
            state = next(iter(states))
            approximate = any(
                r.status == Outcome.APPROXIMATE.value for r in resolutions if r.usable
            )
            reason = Outcome.APPROXIMATE.value if approximate else Outcome.EXACT.value
        else:
            statuses = {r.status for r in resolutions}
            reason = sorted(statuses)[0] if len(statuses) == 1 else "unusable"
        seconds[reason] += length
        if (
            segments
            and segments[-1].end == start
            and segments[-1].label == label
            and (segments[-1].state == state)
        ):
            previous = segments.pop()
            segments.append(ActivityInterval(label, previous.start, end, state))
        else:
            segments.append(ActivityInterval(label, start, end, state))
    return segments, seconds, unscored_labels


def to_canonical(
    data: HouseholdData,
    mapping: OntologyMapping,
    *,
    strict: bool = True,
    source: str = "",
) -> CanonicalHousehold:
    """Convert one household, refusing it on any validation error when *strict*.

    Parameters
    ----------
    data
        The household, in the dataset's terms.
    mapping
        The declared correspondence to the ontology.
    strict
        Refuse a household with any validation error. Without it, conversion
        proceeds where it can, and reports what it leaves out.
    source
        Recorded on every observation, such as the dataset's name.
    """
    report = validate_household(data, mapping)
    if strict and not report.ok:
        raise ContractError(
            f"household {data.household!r} fails the contract: "
            + ", ".join(sorted({i.code for i in report.errors}))
        )
    zone = zone_of(data)
    if zone is None:
        raise ContractError(
            f"household {data.household!r} has no usable timezone; local times "
            "cannot be placed"
        )
    specs: list[SensorSpec] = []
    semantics: dict[str, SensorSemantics] = {}
    excluded: dict[str, str] = {}
    for sensor in data.sensors:
        kind = mapping.sensor_type(sensor.sensor_type)
        if not kind.usable:
            excluded[sensor.sensor_id] = kind.status
            continue
        place = mapping.location(sensor.location)
        target: SensorSemantics = kind.target
        semantics[sensor.sensor_id] = target
        specs.append(
            SensorSpec(
                sensor.sensor_id,
                target.modality,
                kind=target.kind,
                unit=target.unit,
                room=place.target if place.usable else None,
                description=f"{sensor.sensor_type} at {sensor.location or 'no location'}",
            )
        )
    events: Counter[str] = Counter()
    seen: set[tuple[datetime, str, Any]] = set()
    observations: list[Observation] = []
    for event in data.events:
        key = (event.timestamp, event.sensor_id, event.value)
        if key in seen:
            events["duplicate"] += 1
            continue
        seen.add(key)
        if event.sensor_id in excluded:
            events[f"sensor_{excluded[event.sensor_id]}"] += 1
            continue
        declared = semantics.get(event.sensor_id)
        if declared is None:
            events["unknown_sensor"] += 1
            continue
        value = _value(event, declared)
        if value is None:
            events["unfit_value"] += 1
            continue
        if value == "deactivation":
            events["deactivation"] += 1
            continue
        observations.append(
            Observation(
                localise(event.timestamp, zone),
                event.sensor_id,
                declared.modality,
                declared.kind,
                float(value),
                declared.unit,
                source=source,
            )
        )
        events["emitted"] += 1
    observations.sort(key=lambda o: o.timestamp)
    segments, seconds, unscored = _segments(data, mapping, zone)
    points = sum(
        1 for a in data.annotations if a.semantics is AnnotationSemantics.POINT
    )
    impossible = sum(
        1
        for a in data.annotations
        if a.semantics is AnnotationSemantics.INTERVAL and not valid_interval(a)
    )
    recording = CasasRecording(
        registry=SensorRegistry.from_specs(specs),
        observations=tuple(observations),
        activities=tuple(segments),
        unmapped_sensors=frozenset(excluded),
        unmapped_activities=dict(sorted(unscored.items())),
        deactivations=events["deactivation"],
    )
    return CanonicalHousehold(
        household=data.household,
        recording=recording,
        dispositions={
            "events": dict(sorted(events.items())),
            "sensors_excluded": dict(sorted(excluded.items())),
            "annotated_seconds": dict(sorted(seconds.items())),
            "point_annotations": points,
            "impossible_intervals": impossible,
            "unscored_labels": dict(sorted(unscored.items())),
        },
    )
