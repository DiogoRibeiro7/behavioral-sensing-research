"""The ContextAct@A4H dataset: one resident of the Amiqual4Home apartment.

Lago, Lang, Roncancio, Jiménez-Guarín, Mateescu and Bonnefond collected it in
the Amiqual4Home smart apartment of Inria, in France. A woman of 28 lived there
for 7 days in July 2016 and 21 days in November 2016, and annotated eight
activities herself, each with a start and an end, through interfaces in the
apartment. It is independent of CASAS. The apartment has two floors: the ground
floor is one open space holding the kitchen, the dining room and the living
room; upstairs are the bathroom, the bedroom and a study.

Source and terms
----------------
Mendeley Data, dataset ``fcj2hmz5kb`` version 3, DOI
``10.17632/fcj2hmz5kb.3``, under CC BY 4.0. The README asks that publications
cite Lago, P.; Lang, F.; Roncancio, C.; Jiménez-Guarín, C.; Mateescu, R.;
Bonnefond, N. "The ContextAct@A4H Real-Life Dataset of Daily-Living
Activities", CONTEXT 2017, Lecture Notes in Computer Science 10257. Nothing is
redistributed with this package.

Format
------
- **Measurement logs.** One line per record, ``"timestamp";"sensor id";"value"``,
  with timestamps such as ``2016-07-25 16:53:31,482``. They are not UTF-8, and
  are read as Latin-1. Each period has an
  ``All_Update`` log, every update recorded, and an ``All_Change`` log, a line
  only when a value changed. The adapter reads the ``All_Change`` logs: for an
  event sensor a change is an activation or its end, and the update logs only
  add repeated readings of continuous variables.
- **The description file.** ``Variable Description.xlsx`` gives each variable's
  type, what it measures and where it is.
- **Annotations.** Three variables are the resident's own annotations, not
  sensors. ``current_activity`` records ``<label>_Start`` and ``<label>_End``.
  ``current_people`` records how many people were present. ``current_position``
  records where she was, and is not read.

In the contract's terms:

- **Households.** The two recording periods, :data:`HOUSEHOLDS`. They are one
  resident in one apartment, months apart, and not independent homes.
- **Sensors.** Every variable the description file lists, with its type and
  location exactly as written there. A contact sensor's native type adds what it
  measures, ``Contact Sensor/Opening Main Entrance Door``, because that decides
  whether it is the home's entrance door. A variable the file does not list has
  the native type :data:`UNDOCUMENTED` and no location.
- **Events.** Each line, with its value as recorded. A value that is a plain
  number becomes a number; any other value, such as ``ON``, stays a token.
- **Annotations.** Each ``_Start`` is paired with the next ``_End`` or ``_Stop``
  of the same label, as an interval; the README names only ``_End``. A start that is never ended, or started again first,
  becomes an interval without an end; an end without a start becomes an interval
  ending where it starts. Validation reports both, and neither is scored.
- **Occupancy.** Each number of people holds from its line until the next. A
  count of zero, the home empty, is not an occupancy period the contract can
  express, and is left out: it says nothing about a second resident.
- **Timezone.** The dataset gives none. The apartment is in France, and
  ``Europe/Paris`` is declared. Neither period contains a daylight-saving
  change.

The mapping
-----------
:data:`CONTEXTACT_MAPPING` was written from the README, the description file
and the dataset paper, before any model was run on the data. From the change
logs were read: the value tokens of the motion, presence, door and bed sensors,
the time span of each log and the sensor identifiers present; and, of the
annotations, only the distinct activity labels and markers, to declare each.
No annotation's time, duration or count was read.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from ..datasets.information_sets import HH_EVIDENCE_CHANNELS, EvidenceChannel
from ..observations import Modality, ObservationKind
from ..states import BehaviouralState
from .contract import (
    Annotation,
    AnnotationSemantics,
    ContractError,
    DatasetProvenance,
    EventValue,
    HouseholdData,
    OccupancyPeriod,
    RawEvent,
    SensorDescription,
)
from .mapping import (
    OntologyMapping,
    SensorSemantics,
    ambiguous,
    approximate,
    exact,
    unmappable,
)
from .validation import Severity, ValidationReport

S = BehaviouralState

#: The two recording periods, as households.
HOUSEHOLDS = ("A4H-2016-07", "A4H-2016-11")

#: Each period's change log, as published.
LOGS: Mapping[str, str] = {
    "A4H-2016-07": "sensor measurements/July_All_Change.csv.log",
    "A4H-2016-11": "sensor measurements/November_All_Change.csv.log",
}

#: The description file, as published.
DESCRIPTION = "Variable Description.xlsx"

#: The timezone declared for both periods.
TIMEZONE = "Europe/Paris"

#: The Mendeley Data record.
RECORD = "https://data.mendeley.com/datasets/fcj2hmz5kb/3"

#: Each published file's SHA-256, as downloaded on 2026-10-12. Mendeley Data
#: publishes the same digests.
FILES: dict[str, str] = {
    "README.txt": "a0b91ca101a9f61628d897c7cf80c89abcac37d279c801cd84ec6c4411b25163",
    "Variable Description.xlsx": (
        "66e5e89fac90ae5398f09086a3d9e4cb3388c857aebfa20c9afafc32af82b282"
    ),
    "apartment plan.pdf": (
        "4e3ebcff4df06329127e6fcab2dea58a846973ee5ec33f388962a6b5af3a5be6"
    ),
    "indoor location measurements/bsp-info.txt": (
        "518d2b7c7a77b7744fa41b1b42e5d0128f979c1b7ea5843f5da491fc6639a165"
    ),
    "indoor location measurements/bsp.csv.log": (
        "c22735593e21ee968a2e9cec6a84224445b62871c87147011efabc75eb6f8c55"
    ),
    "sensor measurements/July_All_Change.csv.log": (
        "328022acc248259c71dd432324829519ba6ce2914b62d94902577d0410c23bad"
    ),
    "sensor measurements/July_All_Update.csv.log": (
        "ee7bef1619f5adb25d2ac1d016579a531266ec093c371b7050d02880e0e549ec"
    ),
    "sensor measurements/November_All_Change.csv.log": (
        "e985c6254ad7ecc2dd367acfbff80b5039342543eac49209c1b3986874636afb"
    ),
    "sensor measurements/November_All_Update.csv.log": (
        "8628ab9f90ca51014d91a712b9058b906119144a2b81d7985bd199461a8b90f3"
    ),
}

PROVENANCE = DatasetProvenance(
    name="ContextAct@A4H",
    version="Mendeley Data version 3, published 2019-02-19",
    source=f"https://doi.org/10.17632/fcj2hmz5kb.3; {RECORD}",
    licence="CC BY 4.0 (Mendeley Data); the README asks for the citation below",
    citation="Lago P., Lang F., Roncancio C., Jiménez-Guarín C., Mateescu R., "
    "Bonnefond N. (2017) The ContextAct@A4H Real-Life Dataset of Daily-Living "
    "Activities. In: Modeling and Using Context, CONTEXT 2017. Lecture Notes in "
    "Computer Science, vol 10257. Springer, Cham",
    retrieved="2026-10-12",
    files=FILES,
    notes=(
        "one resident in one apartment, recorded in two periods, read as two "
        "households that are not independent",
        f"no timezone is given; the apartment is in France, and {TIMEZONE} is "
        "declared",
        "the README dates the first period July 2016 and the record's description "
        "June; the files are named July, and their records begin on 2016-07-25",
        "the change logs are read; the update logs add repeated readings only",
        "the logs hold 14 variables the description file does not list",
        "the radiators' presence variables R1_Presence to R7_Presence record a "
        "single OFF in each change log and nothing else",
    ),
)

#: The frozen mapping, written by ``scripts/freeze_contextact_mapping.py``.
MAPPING_FILE = "artifacts/contextact/contextact_mapping.json"

#: Validation errors that leave their items unscored and reported. Any other
#: error excludes the period: unmatched annotations and labels the mapping does
#: not declare carry no state, and nothing else is expected to fail.
TOLERATED = frozenset({"impossible_interval", "undeclared_label"})

#: The annotation variables, which are not sensors.
ACTIVITY = "current_activity"
PEOPLE = "current_people"
POSITION = "current_position"
ANNOTATION_VARIABLES = frozenset({ACTIVITY, PEOPLE, POSITION})

#: The native type of a variable the description file does not list.
UNDOCUMENTED = "undocumented"

#: The native type of a listed variable whose type the file leaves empty.
UNSPECIFIED = "unspecified"

#: The activity labels the README lists, in French as recorded.
LABELS = (
    "Cuisine",
    "Dormir",
    "Douche",
    "Manger",
    "Sortir",
    "Toilettes",
    "Travail",
    "Vaisselle",
)

#: Labels the logs hold beyond the README's eight. They are options of the
#: annotation interface (the paper's Fig. 2), read from the logs' activity
#: tokens.
EXTRA_LABELS = ("Loisir", "Musique", "TV")

ON, OFF = "ON", "OFF"
OPEN, CLOSED = "OPEN", "CLOSED"
START = "Start"

#: The markers that end an activity: the README's ``_End``, and ``_Stop``, which
#: the logs also use.
ENDS = frozenset({"End", "Stop"})


# ----------------------------------------------------------------------------
# The mapping
# ----------------------------------------------------------------------------
def _events(modality: Modality, on: str, off: str) -> SensorSemantics:
    return SensorSemantics(
        modality, ObservationKind.EVENT, frozenset({on}), frozenset({off})
    )


def _ambient() -> SensorSemantics:
    return SensorSemantics(Modality.ENVIRONMENTAL, ObservationKind.SAMPLE, numeric=True)


_CONTACT = _events(Modality.CONTACT, OPEN, CLOSED)
_INTERIOR = (
    "Bathroom Door Opening",
    "Bedroom Door Opening",
    "Dining Room Window Opening",
    "Drawer 1 Opening",
    "Drawer 2 Opening",
    "Dressing Door Opening",
    "Fridge Door Opening",
    "Kitchen Gabinet Door Opening",
    "Living Room Window Opening",
    "Shower Door Opening",
    "Study Room Door Opening",
    "Study Room Window Opening",
    "bedroom window Opening",
)
_AMBIENT = (
    "a room's ambient level; the description file gives no unit for most of "
    "these, so values are carried without one"
)
_NO_MODALITY: dict[str, str] = {
    "Base mode for radiator": "a radiator's heating mode",
    "Battery Level": "a controller's battery level",
    "Controller": "a controller's state",
    "Dimmer": "a lamp's dimming level",
    "Effective Mode for Radiator": "a radiator's heating mode",
    "Energy Sensor": "appliance or circuit energy use",
    "Hue Bulb Color": "a lamp's colour",
    "Hue Bulb Dimming": "a lamp's dimming level",
    "Hue Bulb Toggleing": "a lamp's on-off state",
    "Music": "music player metadata",
    "Music  Play": "music player metadata",
    "Music Artist": "music player metadata",
    "Music File": "music player metadata",
    "Music Genre": "music player metadata",
    "Music Play": "music player metadata",
    "Music PlayList": "music player metadata",
    "Music Power": "music player metadata",
    "Music Title": "music player metadata",
    "Music Volume": "music player metadata",
    "Power Factor": "the circuit's power factor",
    "Power Sensor": "appliance or circuit power",
    "Remote Controller": "a remote control's state",
    "Rollershutter": "a curtain's opening",
    "Smart electrical outlet": "an outlet's on-off state",
    "Speed": "a fan's speed",
    "Switch": "a wall switch or appliance on-off state",
    "Temperature": "the kettle's water temperature, not the room's",
    "TemperatureController": "a radiator's set-point",
    "Tension Sensor": "a circuit's voltage",
    "Voltage": "the supply voltage",
    "Water Meter": "water use at a tap or fixture",
    "Weather Condition": "outdoor weather from a web service",
    "Window for temperature": "a radiator's window-open setting",
    "electric intensity": "a circuit's current",
}

CONTEXTACT_MAPPING = OntologyMapping(
    dataset="contextact-a4h",
    version="3",
    description=(
        "Written from the README, the description file and the dataset paper, "
        "with the value tokens of the motion, presence, door and bed sensors and "
        "the distinct activity labels and markers read from the change logs, "
        "before any model was run on the data. No annotation's time, duration "
        "or count was read."
    ),
    labels=(
        exact("Dormir", S.SLEEPING),
        approximate(
            "Sortir",
            S.AWAY,
            "going out, annotated from leaving to returning: the paper's Fig. 3 "
            "gives time out of home as 34% of the November period and 5% of July's",
        ),
        approximate("Douche", S.BATHROOM_ACTIVITY, "showering, one bathroom activity"),
        approximate(
            "Toilettes", S.BATHROOM_ACTIVITY, "using the toilet, one bathroom activity"
        ),
        approximate("Cuisine", S.KITCHEN_ACTIVITY, "cooking, one kitchen activity"),
        approximate(
            "Vaisselle", S.KITCHEN_ACTIVITY, "washing the dishes, one kitchen activity"
        ),
        approximate(
            "Manger",
            S.HOME_ACTIVE,
            "eating; as in the CASAS labels, a meal is activity at home rather than "
            "kitchen activity, and the dining table is in the open living space",
        ),
        approximate(
            "Travail",
            S.HOME_ACTIVE,
            "working at home, annotated in July only; CASAS's work labels are "
            "activity at home",
        ),
        approximate(
            "TV",
            S.HOME_INACTIVE,
            "watching television, an option of the annotation interface (the "
            "paper's Fig. 2) that the README does not list; as CASAS's Watch_TV",
        ),
        ambiguous(
            "Musique",
            (S.HOME_ACTIVE, S.HOME_INACTIVE),
            "listening to music, an option of the annotation interface that the "
            "README does not list; at rest or while doing something else",
        ),
        ambiguous(
            "Loisir",
            (S.HOME_ACTIVE, S.HOME_INACTIVE),
            "leisure, an option of the annotation interface that the README does "
            "not list; active or not",
        ),
    ),
    sensor_types=(
        exact("Infrared Movement Sensor", _events(Modality.MOTION, ON, OFF)),
        exact(
            "Contact Sensor/Opening Main Entrance Door",
            _events(Modality.DOOR, OPEN, CLOSED),
        ),
        approximate(
            "Contact Sensor/Garden Door Opening",
            _events(Modality.DOOR, OPEN, CLOSED),
            "the door from the ground floor to the terrace and garden, an outside "
            "door that is not the entrance",
        ),
        *(exact(f"Contact Sensor/{measured}", _CONTACT) for measured in _INTERIOR),
        exact(
            "Pressure Sensor",
            SensorSemantics(
                Modality.BED_PRESSURE, ObservationKind.SAMPLE, numeric=True
            ),
        ),
        approximate("Carbon Dioxide Meter", _ambient(), _AMBIENT),
        approximate("Humidity Sensor", _ambient(), _AMBIENT),
        approximate("Temperature Sensor", _ambient(), _AMBIENT),
        approximate("Brightness Sensor", _ambient(), _AMBIENT),
        approximate("Noise Level Sensor", _ambient(), _AMBIENT),
        unmappable(
            "Presence Sensor",
            "a radiator controller's presence variable: the documentation gives "
            "neither how it detects presence nor its values, and each change log "
            "records it once, as OFF",
        ),
        *(
            unmappable(kind, f"{what}, which has no modality in the observation model")
            for kind, what in _NO_MODALITY.items()
        ),
        unmappable(UNSPECIFIED, "the description file gives the variable no type"),
        unmappable(UNDOCUMENTED, "a variable the description file does not list"),
    ),
    locations=(
        exact("Bathroom", "bathroom"),
        exact("Bedroom", "bedroom"),
        exact("Kitchen", "kitchen"),
        exact("Living Room", "living"),
        approximate(
            "Dining Room",
            "living",
            "the dining area of the ground floor's open space; CASAS's dining rooms "
            "are mapped to living space too",
        ),
        approximate(
            "Study Room",
            "living",
            "a study upstairs; the ontology has no office, and CASAS's offices are "
            "mapped to living space",
        ),
        approximate("Main Entrance", "hall", "the entrance, where the main door is"),
        approximate("Hallway", "hall", "a hallway"),
        approximate("Hallway First Floor", "hall", "the ground floor's hallway"),
        approximate("Hallway, First Floor", "hall", "the ground floor's hallway"),
        approximate("Hallway Second Floor", "hall", "the upstairs hallway"),
        approximate("Hallway, Second Floor", "hall", "the upstairs hallway"),
        approximate("Stairway", "hall", "the stairs between the two floors"),
        approximate("Stove", "kitchen", "the stove, in the kitchen"),
        ambiguous(
            "First Floor",
            ("kitchen", "living", "hall"),
            "the ground floor as a whole, one open space holding the kitchen, the "
            "dining room and the living room",
        ),
        unmappable("OpenWeather API", "a web service outside the home"),
    ),
)

_LINE = re.compile(r'^"([^"]*)";"([^"]*)";"(.*)"$')
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_SHEET = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_HEADER = {
    "id": "Sensor Id in Dataset",
    "type": "Sensor Type",
    "property": "Measured Property",
    "location": "Sensor Location",
}


# ----------------------------------------------------------------------------
# The description file
# ----------------------------------------------------------------------------
def _column(reference: str) -> int:
    """The zero-based column of a cell reference such as ``C12``."""
    match = re.match(r"[A-Z]+", reference)
    if match is None:
        raise ContractError(f"cell reference {reference!r} has no column")
    index = 0
    for letter in match.group():
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _sheet_rows(path: Path) -> list[dict[int, str]]:
    """Every row of the workbook's first sheet, as column index to text."""
    with zipfile.ZipFile(path) as workbook:
        names = workbook.namelist()
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared = [
                "".join(t.text or "" for t in item.iter(f"{_SHEET}t"))
                for item in root.findall(f"{_SHEET}si")
            ]
        if "xl/worksheets/sheet1.xml" not in names:
            raise ContractError(f"{path.name} has no first sheet")
        sheet = ElementTree.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter(f"{_SHEET}row"):
        cells: dict[int, str] = {}
        for cell in row.findall(f"{_SHEET}c"):
            kind = cell.get("t")
            if kind == "inlineStr":
                text = "".join(t.text or "" for t in cell.iter(f"{_SHEET}t"))
            else:
                value = cell.find(f"{_SHEET}v")
                text = "" if value is None or value.text is None else value.text
                if kind == "s" and text:
                    text = shared[int(text)]
            if text.strip():
                cells[_column(cell.get("r", ""))] = text.strip()
        rows.append(cells)
    return rows


def native_type(sensor_type: str, measured: str) -> str:
    """A variable's native type: its type, and for a contact what it measures."""
    kind = sensor_type or UNSPECIFIED
    return f"{kind}/{measured}" if kind == "Contact Sensor" else kind


def read_description(path: Path) -> dict[str, SensorDescription]:
    """Every variable the description file lists, except the annotation variables."""
    rows = _sheet_rows(Path(path))
    if not rows:
        raise ContractError(f"{Path(path).name} is empty")
    header = {text: column for column, text in rows[0].items()}
    missing = [name for name in _HEADER.values() if name not in header]
    if missing:
        raise ContractError(f"{Path(path).name} has no column {missing}")
    columns = {key: header[name] for key, name in _HEADER.items()}
    described: dict[str, SensorDescription] = {}
    for row in rows[1:]:
        sensor = row.get(columns["id"], "")
        if not sensor or sensor in ANNOTATION_VARIABLES:
            continue
        if sensor in described:
            raise ContractError(f"{Path(path).name} lists {sensor!r} twice")
        described[sensor] = SensorDescription(
            sensor,
            native_type(row.get(columns["type"], ""), row.get(columns["property"], "")),
            row.get(columns["location"]) or None,
            {"measured": row.get(columns["property"], "")},
        )
    return described


# ----------------------------------------------------------------------------
# The logs
# ----------------------------------------------------------------------------
def _time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S,%f")


def _value(text: str) -> EventValue:
    return float(text) if _NUMBER.match(text) else text


def read_log(path: Path) -> Iterator[tuple[datetime, str, str]]:
    """Each record of a measurement log: its time, variable and raw value."""
    # The logs are not UTF-8. Latin-1 decodes every byte to one character, so no
    # record is lost; only free text such as music titles may read differently.
    with Path(path).open(encoding="latin-1") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            match = _LINE.match(line)
            if match is None:
                raise ContractError(f"{Path(path).name} line {number} is not a record")
            stamp, sensor, value = match.groups()
            try:
                moment = _time(stamp)
            except ValueError as error:
                raise ContractError(
                    f"{Path(path).name} line {number} has no valid timestamp"
                ) from error
            yield moment, sensor, value


def pair_annotations(records: Sequence[tuple[datetime, str]]) -> tuple[Annotation, ...]:
    """The activity records as intervals, each start paired with the next end."""
    opened: dict[str, datetime] = {}
    annotations: list[Annotation] = []
    interval = AnnotationSemantics.INTERVAL
    for moment, value in records:
        label, _, mark = value.rpartition("_")
        if not label or (mark != START and mark not in ENDS):
            raise ContractError(
                f"activity record {value!r} is neither <label>_Start nor "
                "<label>_End or _Stop"
            )
        if mark == START:
            if label in opened:
                annotations.append(Annotation(label, opened[label], None, interval))
            opened[label] = moment
        else:
            start = opened.pop(label, moment)
            annotations.append(Annotation(label, start, moment, interval))
    annotations.extend(
        Annotation(label, start, None, interval) for label, start in opened.items()
    )
    return tuple(sorted(annotations, key=lambda a: (a.start, a.label)))


def occupancy(
    records: Sequence[tuple[datetime, str]], end: datetime
) -> tuple[OccupancyPeriod, ...]:
    """The self-annotated numbers of people, each held until the next, or *end*."""
    periods = []
    stops = [moment for moment, _ in records[1:]] + [end]
    for (moment, value), stop in zip(records, stops, strict=True):
        if not _NUMBER.match(value):
            raise ContractError(f"number of people {value!r} is not a number")
        people = int(float(value))
        if people >= 1 and stop > moment:
            periods.append(
                OccupancyPeriod(moment, stop, people, "self-annotated number of people")
            )
    return tuple(periods)


def read_household(directory: Path, household: str) -> HouseholdData:
    """One recording period in the contract's terms, exactly as recorded."""
    if household not in HOUSEHOLDS:
        raise KeyError(f"no household {household!r} in the dataset")
    directory = Path(directory)
    described = read_description(directory / DESCRIPTION)
    events: list[RawEvent] = []
    activities: list[tuple[datetime, str]] = []
    people: list[tuple[datetime, str]] = []
    extra: dict[str, SensorDescription] = {}
    for moment, sensor, value in read_log(directory / LOGS[household]):
        if sensor == ACTIVITY:
            activities.append((moment, value))
        elif sensor == PEOPLE:
            people.append((moment, value))
        elif sensor == POSITION:
            continue
        else:
            if sensor not in described:
                extra.setdefault(sensor, SensorDescription(sensor, UNDOCUMENTED))
            events.append(RawEvent(moment, sensor, _value(value)))
    if not events:
        raise ContractError(f"{LOGS[household]} has no sensor record")
    last = max(event.timestamp for event in events)
    return HouseholdData(
        household=household,
        timezone=TIMEZONE,
        sensors=(*described.values(), *extra.values()),
        events=tuple(events),
        annotations=pair_annotations(activities),
        occupancy=occupancy(people, last),
        residents=1,
    )


@dataclass(frozen=True)
class ContextActAdapter:
    """The contract's adapter for the downloaded dataset."""

    directory: Path

    @property
    def provenance(self) -> DatasetProvenance:
        """The dataset's provenance, with every file's frozen digest."""
        return PROVENANCE

    def households(self) -> Sequence[str]:
        """The two recording periods."""
        return HOUSEHOLDS

    def load(self, household: str) -> HouseholdData:
        """One period's data."""
        return read_household(self.directory, household)


# ----------------------------------------------------------------------------
# What the CASAS-trained model can read, and which periods are eligible
# ----------------------------------------------------------------------------
def model_channel(
    sensor: SensorDescription, mapping: OntologyMapping = CONTEXTACT_MAPPING
) -> tuple[str | None, str]:
    """The CASAS-trained model's channel a sensor feeds, or ``None`` with the reason."""
    semantics = mapping.sensor_type(sensor.sensor_type)
    room = mapping.location(sensor.location)
    if not semantics.usable:
        return None, f"its type is {semantics.status}: {semantics.rationale}"
    target = semantics.target
    if target.kind is not ObservationKind.EVENT:
        return None, f"a {target.kind.value} sensor; the channels count events"
    if not room.usable:
        reason = f": {room.rationale}" if room.rationale else ""
        return None, f"its location is {room.status}{reason}"
    channel = EvidenceChannel(room.target, target.modality)
    if channel not in HH_EVIDENCE_CHANNELS:
        return None, f"no {channel.name} channel in the CASAS-trained model"
    return channel.name, "routed"


def sensor_table(
    data: HouseholdData, mapping: OntologyMapping = CONTEXTACT_MAPPING
) -> list[dict[str, Any]]:
    """Every sensor of a period: its mapping and the model channel it feeds, if any."""
    rows = []
    for sensor in sorted(data.sensors, key=lambda s: s.sensor_id):
        semantics = mapping.sensor_type(sensor.sensor_type)
        room = mapping.location(sensor.location)
        channel, reason = model_channel(sensor, mapping)
        rows.append(
            {
                "sensor": sensor.sensor_id,
                "native_type": sensor.sensor_type,
                "location": sensor.location,
                "type_status": semantics.status,
                "modality": (
                    semantics.target.modality.value if semantics.usable else None
                ),
                "room": room.target if room.usable else None,
                "room_status": room.status,
                "model_channel": channel,
                "unsupported_because": None if channel else reason,
            }
        )
    return rows


def eligibility(report: ValidationReport) -> tuple[bool, list[str]]:
    """Whether a validated period is eligible, and the errors that exclude it."""
    blocking = sorted(
        {i.code for i in report.issues if i.severity is Severity.ERROR} - TOLERATED
    )
    return not blocking, blocking
