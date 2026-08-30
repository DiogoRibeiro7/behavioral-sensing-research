"""Reading CASAS smart-home recordings into canonical observations.

The CASAS project at Washington State University publishes annotated recordings
from instrumented single-resident apartments. Each line of a raw recording is
one sensor report::

    2010-11-04 00:03:50.209589	M003	ON
    2010-11-04 00:15:08.984841	T002	21.5
    2010-11-04 05:40:51.303739	M004	ON	Sleeping	end

The trailing columns appear only where an annotator marked an activity starting
or ending, so labels arrive as sparse boundary markers rather than as a value on
every row. Long stretches carry no label at all, and this module leaves them
unlabelled instead of filling them in.

Obtaining the data
------------------
Recordings are available from the CASAS project at
``https://casas.wsu.edu/datasets/``, under the terms stated there. Nothing is
redistributed with this package. Point :func:`read_casas` at a downloaded file.

What this adapter will not do
-----------------------------
Three things a convenience reader might do silently, and why this one refuses.

**It will not guess a timezone.** CASAS timestamps are local wall-clock with no
offset. Reading them as UTC displaces every event by hours and quietly ruins any
daily-rhythm analysis, and the pipeline requires aware instants regardless. The
caller must say which zone the apartment was in.

**It will not invent labels for unlabelled time.** Much of a typical recording
carries no annotation. :func:`truth_series` returns ``None`` there, and the
evaluation metrics skip those positions rather than scoring a guess.

**It will not pretend the ontology matches.** CASAS activity vocabularies differ
between apartments and do not correspond one-to-one with this package's
behavioural states. Unmapped labels are reported on the recording rather than
collapsed into ``UNKNOWN``, so a reader can see how much annotation was
discarded.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from pathlib import Path

from ..observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
    Unit,
)
from ..states import BehaviouralState

logger = logging.getLogger(__name__)


class CasasReadError(ValueError):
    """A recording could not be read as CASAS data."""


#: How a sensor identifier's prefix maps onto the canonical model.
#:
#: CASAS names sensors by role: ``M`` motion, ``D`` door, ``T`` temperature,
#: ``I`` item, ``L`` light, ``P`` power, ``AD`` analogue. Only prefixes with an
#: unambiguous canonical meaning are mapped. The rest are reported as unmapped
#: rather than forced into ``Modality.OTHER``, where they would contribute
#: evidence nobody had reasoned about.
SENSOR_PREFIXES: Mapping[str, tuple[Modality, ObservationKind, Unit]] = {
    "M": (Modality.MOTION, ObservationKind.EVENT, Unit.NONE),
    "D": (Modality.DOOR, ObservationKind.EVENT, Unit.NONE),
    "T": (Modality.ENVIRONMENTAL, ObservationKind.SAMPLE, Unit.CELSIUS),
}

#: Sensor values that count as an activation for an event-kind sensor.
ACTIVATION_VALUES = frozenset({"ON", "OPEN", "PRESENT", "START"})

#: Values that close an activation. Counted, not emitted.
DEACTIVATION_VALUES = frozenset({"OFF", "CLOSE", "ABSENT", "END"})

#: CASAS activity labels mapped onto this package's behavioural states.
#:
#: The correspondence is imperfect, and the imperfections matter:
#:
#: - ``Eating`` maps to ``HOME_ACTIVE`` rather than ``KITCHEN_ACTIVITY``. The
#:   ontology's kitchen state means being active in the kitchen, and meals are
#:   frequently eaten elsewhere; mapping it to the kitchen would manufacture
#:   agreement with an inference that keys on kitchen sensors.
#: - ``Bed_to_Toilet`` maps to ``BATHROOM_ACTIVITY`` for the destination, losing
#:   the fact that it is a night-time transition out of bed.
#: - ``Respirate`` has no counterpart and is deliberately absent, so it is
#:   reported unmapped rather than scored.
#:
#: Pass a different mapping to :func:`read_casas` to make other choices. The
#: point is that the choice is visible rather than buried.
CASAS_ACTIVITY_STATES: Mapping[str, BehaviouralState] = {
    "Sleeping": BehaviouralState.SLEEPING,
    "Bed_to_Toilet": BehaviouralState.BATHROOM_ACTIVITY,
    "Meal_Preparation": BehaviouralState.KITCHEN_ACTIVITY,
    "Wash_Dishes": BehaviouralState.KITCHEN_ACTIVITY,
    "Housekeeping": BehaviouralState.HOME_ACTIVE,
    "Work": BehaviouralState.HOME_ACTIVE,
    "Eating": BehaviouralState.HOME_ACTIVE,
    "Enter_Home": BehaviouralState.HOME_ACTIVE,
    "Relax": BehaviouralState.HOME_INACTIVE,
    "Leave_Home": BehaviouralState.AWAY,
}


@dataclass(frozen=True)
class ActivityInterval:
    """One annotated activity, from its begin marker to its end marker."""

    label: str
    start: datetime
    end: datetime
    state: BehaviouralState | None

    def __post_init__(self) -> None:
        """Reject an interval that ends before it starts."""
        if self.end < self.start:
            raise CasasReadError(
                f"activity {self.label!r} ends at {self.end} before it starts "
                f"at {self.start}"
            )

    def contains(self, moment: datetime) -> bool:
        """Whether *moment* falls inside the interval, end-exclusive."""
        return self.start <= moment < self.end


@dataclass(frozen=True)
class CasasRecording:
    """A parsed recording, together with what could not be represented.

    Attributes
    ----------
    registry
        Sensors seen in the file, described in canonical terms.
    observations
        Every reading that mapped onto a canonical sensor, ordered in time.
    activities
        Annotated intervals, including those whose label did not map.
    unmapped_sensors
        Sensor identifiers whose prefix has no canonical meaning. Their readings
        do not appear in *observations*.
    unmapped_activities
        Activity labels with no counterpart in the state ontology, and how often
        each appeared. Those intervals are present but carry ``state=None``.
    deactivations
        Count of OFF-style readings, which are not emitted. See
        :func:`read_casas`.
    unparsed_lines
        Lines that were not readable as sensor reports.
    """

    registry: SensorRegistry
    observations: tuple[Observation, ...]
    activities: tuple[ActivityInterval, ...]
    unmapped_sensors: frozenset[str] = frozenset()
    unmapped_activities: Mapping[str, int] = field(default_factory=dict)
    deactivations: int = 0
    unparsed_lines: int = 0

    @property
    def labelled_fraction(self) -> float:
        """Fraction of the recorded span covered by a mapped activity.

        The number to check before believing any accuracy computed against this
        recording, because it says how much of the time was actually scored.
        """
        if not self.observations:
            return 0.0
        span = (
            self.observations[-1].timestamp - self.observations[0].timestamp
        ).total_seconds()
        if span <= 0:
            return 0.0
        covered = sum(
            (interval.end - interval.start).total_seconds()
            for interval in self.activities
            if interval.state is not None
        )
        return min(covered / span, 1.0)

    def summary(self) -> dict[str, object]:
        """Return what was read and what was discarded."""
        return {
            "observations": len(self.observations),
            "sensors": len(self.registry.sensor_ids()),
            "activities": len(self.activities),
            "labelled_fraction": self.labelled_fraction,
            "unmapped_sensors": sorted(self.unmapped_sensors),
            "unmapped_activities": dict(self.unmapped_activities),
            "deactivations_ignored": self.deactivations,
            "unparsed_lines": self.unparsed_lines,
        }


def _sensor_prefix(sensor_id: str) -> str:
    """Return the leading alphabetic run of a CASAS sensor identifier."""
    prefix = ""
    for character in sensor_id:
        if not character.isalpha():
            break
        prefix += character
    return prefix.upper()


def casas_sensor_specs(
    sensor_ids: Iterable[str],
    *,
    rooms: Mapping[str, str] | None = None,
) -> tuple[list[SensorSpec], frozenset[str]]:
    """Describe CASAS sensors canonically, and report those that cannot be.

    The raw files carry no machine-readable sensor-to-room map, so *rooms* is
    optional and unset by default. Room drives the occupancy layer, and
    inferring it from sensor numbering would be fabrication.
    """
    room_map = dict(rooms or {})
    specs: list[SensorSpec] = []
    unmapped: set[str] = set()

    for sensor_id in sorted(set(sensor_ids)):
        mapping = SENSOR_PREFIXES.get(_sensor_prefix(sensor_id))
        if mapping is None:
            unmapped.add(sensor_id)
            continue
        modality, kind, unit = mapping
        specs.append(
            SensorSpec(
                sensor_id=sensor_id,
                modality=modality,
                kind=kind,
                unit=unit,
                room=room_map.get(sensor_id),
                attributable=False,
                description=f"CASAS sensor {sensor_id}",
            )
        )
    return specs, frozenset(unmapped)


ParsedLine = tuple[datetime, str, str, "str | None", "str | None"]


def _parse_line(line: str) -> ParsedLine | None:
    """Split one raw line, or return ``None`` if it is not a sensor report."""
    parts = line.split()
    if len(parts) < 4:
        return None

    stamp = f"{parts[0]} {parts[1]}"
    for pattern in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            moment = datetime.strptime(stamp, pattern)
            break
        except ValueError:
            continue
    else:
        return None

    label = parts[4] if len(parts) > 4 else None
    marker = parts[5].lower() if len(parts) > 5 else None
    return moment, parts[2], parts[3], label, marker


def read_casas(
    source: Path | str | Iterable[str],
    *,
    timezone: tzinfo,
    activity_states: Mapping[str, BehaviouralState] | None = None,
    rooms: Mapping[str, str] | None = None,
) -> CasasRecording:
    """Read a CASAS recording into canonical observations and annotations.

    Parameters
    ----------
    source
        Path to a raw recording, or an iterable of its lines.
    timezone
        The zone the apartment was in. Required, because CASAS timestamps are
        naive local wall-clock and there is no correct default.
    activity_states
        Overrides :data:`CASAS_ACTIVITY_STATES`.
    rooms
        Optional sensor-to-room assignment, if one is known for the apartment.

    Notes
    -----
    Motion and door sensors report ON/OFF and OPEN/CLOSE pairs. Only the
    activating half becomes an observation, because the pipeline models these as
    event-kind sensors whose *rate* carries the information, and admitting the
    closing half would double every count. The discarded readings are counted in
    :attr:`CasasRecording.deactivations` rather than dropped silently, because
    treating them as state-kind evidence instead is a legitimate alternative
    this adapter does not take.
    """
    if timezone is None:  # pragma: no cover - defensive
        raise CasasReadError("a timezone is required; CASAS timestamps are naive")

    lines: Iterator[str]
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise CasasReadError(f"no CASAS recording at {path}")
        lines = iter(path.read_text(encoding="utf-8", errors="replace").splitlines())
    else:
        lines = iter(source)

    states = dict(
        activity_states if activity_states is not None else CASAS_ACTIVITY_STATES
    )

    parsed: list[ParsedLine] = []
    unparsed = 0
    for raw in lines:
        if not raw.strip():
            continue
        record = _parse_line(raw)
        if record is None:
            unparsed += 1
            continue
        parsed.append(record)

    if not parsed:
        raise CasasReadError("no readable sensor reports found")

    specs, unmapped_sensors = casas_sensor_specs(
        (sensor_id for _, sensor_id, _, _, _ in parsed), rooms=rooms
    )
    registry = SensorRegistry.from_specs(specs)
    by_id = {spec.sensor_id: spec for spec in specs}

    observations: list[Observation] = []
    deactivations = 0
    open_activities: dict[str, datetime] = {}
    activities: list[ActivityInterval] = []
    unmapped_activities: dict[str, int] = {}

    for moment, sensor_id, value, label, marker in parsed:
        aware = moment.replace(tzinfo=timezone)

        if label is not None and marker in {"begin", "end"}:
            if marker == "begin":
                open_activities[label] = aware
            else:
                start = open_activities.pop(label, None)
                if start is None:
                    # An end without a begin has no duration to record.
                    logger.debug("activity %r ended without beginning", label)
                else:
                    state = states.get(label)
                    if state is None:
                        unmapped_activities[label] = (
                            unmapped_activities.get(label, 0) + 1
                        )
                    activities.append(
                        ActivityInterval(
                            label=label, start=start, end=aware, state=state
                        )
                    )

        spec = by_id.get(sensor_id)
        if spec is None:
            continue

        upper = value.upper()
        if spec.kind is ObservationKind.EVENT:
            if upper in DEACTIVATION_VALUES:
                deactivations += 1
                continue
            if upper not in ACTIVATION_VALUES:
                continue
            numeric = 1.0
        else:
            try:
                numeric = float(value)
            except ValueError:
                continue

        observations.append(
            Observation(
                timestamp=aware,
                sensor_id=sensor_id,
                modality=spec.modality,
                kind=spec.kind,
                value=numeric,
                unit=spec.unit,
                source="casas",
            )
        )

    observations.sort(key=lambda observation: observation.timestamp)
    activities.sort(key=lambda interval: interval.start)

    if open_activities:
        logger.warning(
            "%d activity annotation(s) never ended and were dropped: %s",
            len(open_activities),
            sorted(open_activities),
        )

    return CasasRecording(
        registry=registry,
        observations=tuple(observations),
        activities=tuple(activities),
        unmapped_sensors=unmapped_sensors,
        unmapped_activities=unmapped_activities,
        deactivations=deactivations,
        unparsed_lines=unparsed,
    )


def truth_series(
    activities: Sequence[ActivityInterval],
    moments: Sequence[datetime],
) -> list[BehaviouralState | None]:
    """Return the annotated state at each moment, ``None`` where unlabelled.

    ``None`` is the honest answer for the large unannotated stretches of a real
    recording, and the evaluation metrics skip those positions rather than
    scoring a guess against them. Where annotations overlap, the one that
    started most recently wins.
    """
    ordered = sorted(activities, key=lambda interval: interval.start)
    series: list[BehaviouralState | None] = []
    for moment in moments:
        found: BehaviouralState | None = None
        for interval in ordered:
            if interval.start > moment:
                break
            if interval.contains(moment):
                found = interval.state
        series.append(found)
    return series
