"""Reading the CASAS ``hh`` CSV export into canonical observations.

CASAS publishes its recordings in more than one shape. :mod:`.casas` reads the
classic space-separated form, where the third field is a sensor identifier such
as ``M004``. The archives currently distributed on Zenodo use a different
export, and the difference is not cosmetic::

    2011-06-15,01:03:39.14962,Bedroom,ON,Sleep="begin"
    2011-06-15,01:03:40.284526,Bedroom,OFF

Three things change. Fields are comma-separated with date and time split apart.
The third field is a **location**, not a sensor: individual sensors have already
been collapsed into the room they observe. And activity markers are written
``Name="begin"`` rather than as two trailing words.

The vocabulary changes too. This export labels ``Sleep``, ``Cook_Dinner``,
``Toilet`` and ``Watch_TV`` where the classic files label ``Sleeping``,
``Meal_Preparation`` and ``Relax``, so a mapping written for one is nearly
useless on the other. :data:`HH_ACTIVITY_STATES` is separate for that reason
rather than shared.

That the third field is a location is a genuine convenience: the room
assignment the occupancy and fusion layers need is present in the data instead
of having to be reconstructed from a floor plan. It also means each room
contributes exactly one aggregate sensor, so redundancy between sensors within a
room is not observable here.

Obtaining the data
------------------
``https://zenodo.org/records/15708568`` (CC-BY-4.0). Nothing is redistributed
with this package.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
from .casas import (
    ACTIVATION_VALUES,
    DEACTIVATION_VALUES,
    ActivityInterval,
    CasasReadError,
    CasasRecording,
)

logger = logging.getLogger(__name__)

#: Location names mapped to a canonical room and modality.
#:
#: The ontology has no office, so ``WorkArea`` is treated as living space: it is
#: somewhere the resident is awake and occupied, which is what the behavioural
#: states distinguish. ``LoungeChair`` is a seat rather than a room, and is
#: mapped to the living room it stands in.
HH_LOCATIONS: Mapping[str, tuple[str, Modality]] = {
    "Kitchen": ("kitchen", Modality.MOTION),
    "Bathroom": ("bathroom", Modality.MOTION),
    "Bedroom": ("bedroom", Modality.MOTION),
    "LivingRoom": ("living", Modality.MOTION),
    "LoungeChair": ("living", Modality.MOTION),
    "WorkArea": ("living", Modality.MOTION),
    "Office": ("living", Modality.MOTION),
    # A dining room is mapped to living space rather than to the kitchen. The
    # kitchen state means being active in the kitchen, and eating is mapped to
    # HOME_ACTIVE for the same reason; routing dining motion into kitchen
    # evidence would manufacture agreement between the two.
    "DiningRoom": ("living", Modality.MOTION),
    "Hall": ("hall", Modality.MOTION),
    "Entry": ("hall", Modality.MOTION),
    "OutsideDoor": ("hall", Modality.DOOR),
    "FrontDoor": ("hall", Modality.DOOR),
    "BackDoor": ("hall", Modality.DOOR),
}

#: ``hh`` activity labels mapped onto this package's behavioural states.
#:
#: Deliberate choices worth knowing about:
#:
#: - Every ``Eat_*`` maps to ``HOME_ACTIVE`` rather than ``KITCHEN_ACTIVITY``,
#:   for the same reason as in the classic mapping: the kitchen state means
#:   being active in the kitchen, and meals are often eaten elsewhere.
#: - ``Wake_Up`` and ``Go_To_Sleep`` map to ``BED_AWAKE``, which is exactly what
#:   that state exists to capture: in bed, but not asleep.
#: - ``Step_Out`` maps to ``AWAY`` alongside ``Leave_Home``, since the resident
#:   is out of sensor range either way.
#: - ``Other_Activity`` is absent on purpose. It is the dataset's own label for
#:   "something happened that we did not categorise", and turning that into a
#:   behavioural state would invent a claim the annotator declined to make.
HH_ACTIVITY_STATES: Mapping[str, BehaviouralState] = {
    "Sleep": BehaviouralState.SLEEPING,
    "Sleep_Out_Of_Bed": BehaviouralState.HOME_INACTIVE,
    "Wake_Up": BehaviouralState.BED_AWAKE,
    "Go_To_Sleep": BehaviouralState.BED_AWAKE,
    "Bed_Toilet_Transition": BehaviouralState.BATHROOM_ACTIVITY,
    "Toilet": BehaviouralState.BATHROOM_ACTIVITY,
    "Personal_Hygiene": BehaviouralState.BATHROOM_ACTIVITY,
    "Bathe": BehaviouralState.BATHROOM_ACTIVITY,
    "Groom": BehaviouralState.BATHROOM_ACTIVITY,
    "Cook": BehaviouralState.KITCHEN_ACTIVITY,
    "Cook_Breakfast": BehaviouralState.KITCHEN_ACTIVITY,
    "Cook_Lunch": BehaviouralState.KITCHEN_ACTIVITY,
    "Cook_Dinner": BehaviouralState.KITCHEN_ACTIVITY,
    "Wash_Dishes": BehaviouralState.KITCHEN_ACTIVITY,
    "Wash_Breakfast_Dishes": BehaviouralState.KITCHEN_ACTIVITY,
    "Wash_Lunch_Dishes": BehaviouralState.KITCHEN_ACTIVITY,
    "Wash_Dinner_Dishes": BehaviouralState.KITCHEN_ACTIVITY,
    "Eat": BehaviouralState.HOME_ACTIVE,
    "Eat_Breakfast": BehaviouralState.HOME_ACTIVE,
    "Eat_Lunch": BehaviouralState.HOME_ACTIVE,
    "Eat_Dinner": BehaviouralState.HOME_ACTIVE,
    "Dress": BehaviouralState.HOME_ACTIVE,
    "Work": BehaviouralState.HOME_ACTIVE,
    "Work_At_Table": BehaviouralState.HOME_ACTIVE,
    "Work_On_Computer": BehaviouralState.HOME_ACTIVE,
    "Housekeeping": BehaviouralState.HOME_ACTIVE,
    "Laundry": BehaviouralState.HOME_ACTIVE,
    # Leave_Home and Enter_Home mark the act of crossing the threshold, with a
    # median duration of about twelve seconds. They are moments of activity in
    # the home, not the state of being out of it; the away period is the gap
    # between them, which carries no label of its own. See derive_away.
    "Enter_Home": BehaviouralState.HOME_ACTIVE,
    "Watch_TV": BehaviouralState.HOME_INACTIVE,
    "Relax": BehaviouralState.HOME_INACTIVE,
    "Read": BehaviouralState.HOME_INACTIVE,
    "Phone": BehaviouralState.HOME_INACTIVE,
    "Entertain_Guests": BehaviouralState.HOME_ACTIVE,
    "Drink": BehaviouralState.HOME_ACTIVE,
    "Take_Medicine": BehaviouralState.HOME_ACTIVE,
    "Morning_Meds": BehaviouralState.HOME_ACTIVE,
    "Evening_Meds": BehaviouralState.HOME_ACTIVE,
    "Work_At_Desk": BehaviouralState.HOME_ACTIVE,
    "Exercise": BehaviouralState.HOME_ACTIVE,
    "Leave_Home": BehaviouralState.HOME_ACTIVE,
    "Step_Out": BehaviouralState.AWAY,
}

#: Fixture and coverage suffixes CASAS appends to a room name.
#:
#: The published description explains the convention: a trailing letter
#: distinguishes rooms of the same type (``BedroomA``, ``BedroomB``), ``Area``
#: marks an unconstrained sensor covering a whole room, and the remaining
#: suffixes name the fixture a narrow-lensed sensor points at.
_FIXTURES = (
    "Area",
    "Bed",
    "Chair",
    "DiningChair",
    "Door",
    "Refrigerator",
    "Sink",
    "Stove",
    "Toilet",
    "Temperature",
    "Entryway",
    "Shower",
    "Cabinet",
    "Closet",
    "Desk",
    "Couch",
    "Table",
    "Microwave",
    "Washer",
    "Dryer",
)

#: Room words appearing in the extended vocabulary, mapped to canonical rooms.
#:
#: Only rooms actually present in the CASAS recordings are listed. Adding a
#: speculative entry would silently start admitting a sensor nobody has looked
#: at, which is the opposite of what the unmapped report is for. Rooms the
#: ontology does not model -- an office, a sewing room, a laundry -- map to
#: living space, since what the states distinguish is being awake and occupied
#: rather than the specific room.
_ROOM_WORDS: Mapping[str, str] = {
    "Bathroom": "bathroom",
    "Bedroom": "bedroom",
    "Kitchen": "kitchen",
    "LivingRoom": "living",
    "DiningRoom": "living",
    "Office": "living",
    "MOffice": "living",
    "SewingRoom": "living",
    "LaundryRoom": "living",
    "WorkArea": "living",
    "Hallway": "hall",
    "Entryway": "hall",
    "Main": "hall",
}

_TRAILING_LETTER = re.compile(r"^(?P<room>.*?)(?P<instance>[A-Z])$")


def normalise_location(location: str) -> tuple[str, Modality] | None:
    """Resolve an extended CASAS location name to a room and modality.

    Later releases name sensors ``<Room>[Instance][Fixture]`` -- ``BedroomABed``,
    ``KitchenARefrigerator``, ``HallwayA`` -- where the flat vocabulary used a
    bare room. Returns ``None`` for a name this cannot resolve, so an
    unrecognised sensor is still excluded and reported rather than guessed at.

    Every sensor here is a PIR motion detector or a magnetic door contact,
    per the dataset description. A sensor aimed at a bed is therefore *motion
    at the bed*, not an occupancy measurement, and is mapped as motion.
    Treating it as presence would invent a measurement the deployment does not
    make.
    """
    if location in HH_LOCATIONS:
        return HH_LOCATIONS[location]

    stem = location
    modality = Modality.MOTION
    for fixture in sorted(_FIXTURES, key=len, reverse=True):
        if stem.endswith(fixture) and stem != fixture:
            if fixture == "Temperature":
                modality = Modality.ENVIRONMENTAL
            elif fixture == "Door" and stem.startswith("Main"):
                modality = Modality.DOOR
            stem = stem[: -len(fixture)]
            break

    match = _TRAILING_LETTER.match(stem)
    if match and match.group("room") in _ROOM_WORDS:
        stem = match.group("room")

    room = _ROOM_WORDS.get(stem)
    if room is None:
        return None
    return room, modality


_MARKER = re.compile(r'^\s*([A-Za-z0-9_]+)\s*=\s*"?(begin|end)"?\s*$')
_BARE_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def hh_sensor_specs(
    locations: Iterable[str],
) -> tuple[list[SensorSpec], frozenset[str]]:
    """Describe ``hh`` locations as canonical sensors.

    Each location becomes one sensor carrying that room, because the export has
    already aggregated the room's physical sensors into a single stream.
    """
    specs: list[SensorSpec] = []
    unmapped: set[str] = set()

    for location in sorted(set(locations)):
        mapping = normalise_location(location)
        if mapping is None:
            unmapped.add(location)
            continue
        room, modality = mapping
        environmental = modality is Modality.ENVIRONMENTAL
        specs.append(
            SensorSpec(
                sensor_id=location,
                modality=modality,
                kind=(
                    ObservationKind.SAMPLE if environmental else ObservationKind.EVENT
                ),
                unit=Unit.CELSIUS if environmental else Unit.NONE,
                room=room,
                attributable=False,
                description=f"CASAS hh location {location}",
            )
        )
    return specs, frozenset(unmapped)


def _derive_away_intervals(
    activities: Sequence[ActivityInterval],
) -> list[ActivityInterval]:
    """Infer the away periods that lie between leaving and returning.

    ``Leave_Home`` and ``Enter_Home`` are instants, not durations: their median
    length is around twelve seconds, because they annotate the act of crossing
    the threshold. The resident is inside and moving while they are marked, so
    scoring them as ``AWAY`` penalises an inference that correctly reports
    activity, and the hours actually spent out carry no label at all.

    The away period is the gap from the end of a departure to the start of the
    next arrival. Deriving it turns unlabelled time into the only annotation of
    absence the dataset supports.
    """
    ordered = sorted(activities, key=lambda interval: interval.start)
    derived: list[ActivityInterval] = []
    left_at = None
    for interval in ordered:
        if interval.label == "Leave_Home":
            left_at = interval.end
        elif interval.label == "Enter_Home" and left_at is not None:
            if interval.start > left_at:
                derived.append(
                    ActivityInterval(
                        label="Away",
                        start=left_at,
                        end=interval.start,
                        state=BehaviouralState.AWAY,
                    )
                )
            left_at = None
    return derived


def read_casas_hh(
    source: Path | str | Iterable[str],
    *,
    timezone: tzinfo,
    activity_states: Mapping[str, BehaviouralState] | None = None,
    locations: Mapping[str, tuple[str, Modality]] | None = None,
    derive_away: bool = True,
) -> CasasRecording:
    """Read a CASAS ``hh`` CSV recording.

    Parameters
    ----------
    source
        Path to a ``.csv`` recording, or an iterable of its lines.
    timezone
        The zone the home was in. Required for the same reason as in
        :func:`~sensor_modeling.datasets.read_casas`: the timestamps are naive
        local wall-clock and reading them as UTC displaces every event.
    activity_states
        Overrides :data:`HH_ACTIVITY_STATES`.
    locations
        Overrides :data:`HH_LOCATIONS`.
    derive_away
        Add an ``AWAY`` interval spanning each gap between a ``Leave_Home`` and
        the next ``Enter_Home``. Those two labels mark instants of crossing the
        threshold, so without this the time actually spent out is unlabelled and
        the only annotation scored as absence is a few seconds of the resident
        walking to the door. See :func:`_derive_away_intervals`.

    Notes
    -----
    As in the classic reader, only the activating half of each ON/OFF and
    OPEN/CLOSE pair becomes an observation, and the closing half is counted in
    :attr:`CasasRecording.deactivations`. These sensors are modelled as
    event-kind, where the rate carries the information, so admitting both halves
    would double every count.
    """
    lines: Iterator[str]
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise CasasReadError(f"no CASAS recording at {path}")
        lines = iter(path.read_text(encoding="utf-8", errors="replace").splitlines())
    else:
        lines = iter(source)

    states = dict(
        activity_states if activity_states is not None else HH_ACTIVITY_STATES
    )
    places = dict(locations if locations is not None else HH_LOCATIONS)

    rows: list[tuple[datetime, str, str, str | None]] = []
    unparsed = 0
    for raw in lines:
        if not raw.strip():
            continue
        parts = raw.rstrip("\n").split(",")
        if len(parts) < 4:
            unparsed += 1
            continue
        stamp = f"{parts[0]} {parts[1]}"
        moment = None
        for pattern in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                moment = datetime.strptime(stamp, pattern)
                break
            except ValueError:
                continue
        if moment is None:
            unparsed += 1
            continue
        marker = parts[4] if len(parts) > 4 and parts[4].strip() else None
        rows.append((moment, parts[2].strip(), parts[3].strip(), marker))

    if not rows:
        raise CasasReadError("no readable sensor reports found")

    specs, unmapped_locations = hh_sensor_specs(
        (location for _, location, _, _ in rows)
    )
    # A caller-supplied location map may name places the default does not.
    if locations is not None:
        specs = [spec for spec in specs if spec.sensor_id in places]
    registry = SensorRegistry.from_specs(specs)
    by_id = {spec.sensor_id: spec for spec in specs}

    observations: list[Observation] = []
    deactivations = 0
    open_activities: dict[str, datetime] = {}
    activities: list[ActivityInterval] = []
    unmapped_activities: dict[str, int] = {}

    # A bare label in the fifth column annotates that one event rather than
    # opening an interval. Consecutive events carrying the same bare label are
    # coalesced into the span they cover, which is the most the annotation
    # actually supports: it says the resident was doing this at these instants,
    # not that they did it continuously between them.
    run_label: str | None = None
    run_start: datetime | None = None
    run_last: datetime | None = None

    def close_run() -> None:
        nonlocal run_label, run_start, run_last
        if run_label is None or run_start is None or run_last is None:
            return
        if run_last > run_start:
            state = states.get(run_label)
            if state is None:
                unmapped_activities[run_label] = (
                    unmapped_activities.get(run_label, 0) + 1
                )
            activities.append(
                ActivityInterval(
                    label=run_label, start=run_start, end=run_last, state=state
                )
            )
        run_label = run_start = run_last = None

    for moment, location, value, marker in rows:
        aware = moment.replace(tzinfo=timezone)

        if marker is not None:
            match = _MARKER.match(marker)
            if match is None:
                bare = marker.strip().strip('"')
                if _BARE_LABEL.match(bare):
                    if bare != run_label:
                        close_run()
                        run_label, run_start = bare, aware
                    run_last = aware
                else:
                    unparsed += 1
            else:
                close_run()
                label, edge = match.group(1), match.group(2)
                if edge == "begin":
                    open_activities[label] = aware
                else:
                    start = open_activities.pop(label, None)
                    if start is None:
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
        else:
            close_run()

        spec = by_id.get(location)
        if spec is None:
            continue

        if spec.kind is ObservationKind.SAMPLE:
            try:
                numeric = float(value)
            except ValueError:
                continue
        else:
            upper = value.upper()
            if upper in DEACTIVATION_VALUES:
                deactivations += 1
                continue
            if upper not in ACTIVATION_VALUES:
                continue
            numeric = 1.0

        observations.append(
            Observation(
                timestamp=aware,
                sensor_id=spec.sensor_id,
                modality=spec.modality,
                kind=spec.kind,
                value=numeric,
                unit=spec.unit,
                source="casas-hh",
            )
        )

    close_run()

    if derive_away:
        activities.extend(_derive_away_intervals(activities))

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
        unmapped_sensors=unmapped_locations,
        unmapped_activities=unmapped_activities,
        deactivations=deactivations,
        unparsed_lines=unparsed,
    )
