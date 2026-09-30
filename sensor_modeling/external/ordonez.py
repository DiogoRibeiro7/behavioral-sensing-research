"""The UCI "Activities of Daily Living Recognition Using Binary Sensors" dataset.

Ordóñez, de Toledo and Sanchis collected it in two single-resident homes in
Spain: home A for 14 labelled days, home B for 21. It is independent of CASAS,
and its sensing is materially different: fixture-level PIR sensors, magnetic
switches on doors and cupboards, a toilet flush sensor, pressure mats on a seat
and a bed, and electric sensors on appliances. It is the Phase 5 external
dataset.

Source and terms
----------------
UCI Machine Learning Repository, dataset 271, DOI ``10.24432/C5J02M``, under
CC BY 4.0. The dataset's README asks that publications cite Ordóñez, F.J.; de
Toledo, P.; Sanchis, A. "Activity Recognition Using Hybrid
Generative/Discriminative Models on Home Environments Using Binary Sensors",
*Sensors* 2013, 13, 5460-5477, and prohibits commercial use. Nothing is
redistributed with this package.

Format
------
Each home has three files. ``<home>_Description.txt`` documents it.
``<home>_Sensors.txt`` has one row per sensor activation, with start, end,
``Location`` (the object: ``Shower``, ``Maindoor``, ``Fridge``...), ``Type``
(``PIR``, ``Magnetic``, ``Flush``, ``Pressure``, ``Electric``) and ``Place``
(the room). ``<home>_ADLs.txt`` has one row per annotated activity, with start,
end and label. Columns are tab-separated with varying padding, after a
two-line header.

In the contract's terms:

- **Sensors.** A sensor's identifier is ``<Location>.<Type>.<Place>``. Its
  native type is ``<Type>/<Location>``, because the object decides its meaning:
  a magnetic switch on the main door is a door, and on the fridge a contact.
  Its location is ``Place``.
- **Events.** Each activation row becomes an ``ON`` event at its start and an
  ``OFF`` event at its end.
- **Annotations.** Each annotation row is an interval.
- **Timezone.** Timestamps are local and carry no timezone. The dataset gives
  none; the homes were in Spain, and ``Europe/Madrid`` is declared.

The mapping
-----------
:data:`ORDONEZ_MAPPING` was written from the dataset's documentation: the
README and the two description files, which list every label and sensor. The
sensors' rooms and the meaning of ``Leaving`` were read from the first seven
days of each home, the protocol's adaptation period, and from nothing later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from ..observations import Modality, ObservationKind
from ..states import BehaviouralState
from .contract import (
    Annotation,
    AnnotationSemantics,
    ContractError,
    DatasetProvenance,
    HouseholdData,
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

S = BehaviouralState

#: The two homes, as the dataset names them.
HOUSEHOLDS = ("OrdonezA", "OrdonezB")

#: The timezone declared for both homes.
TIMEZONE = "Europe/Madrid"

#: The folder the archive extracts to.
FOLDER = "UCI ADL Binary Dataset"

#: Each file's SHA-256, as downloaded from the UCI archive on 2026-09-30.
FILES: dict[str, str] = {
    "OrdonezA_ADLs.txt": "b018601885e1c62a9769f6b2ac7b47fcd51497d8880e272c1b3ff678eb63e15d",
    "OrdonezA_Description.txt": "4ce4b8e4dbbaa9208069ac184e5686c2b8fdcf97a446a8357112f3ea2bc7b3dc",
    "OrdonezA_Sensors.txt": "3ec41f2b1d15c90aec0753d02a21c9e7add0b6f761ef76f3d8d428fb2dd0fa9d",
    "OrdonezB_ADLs.txt": "d2c59c9ada5da5dbebe26c6f02f02e331fc61a5d4ec425e19ef5a9de98852d58",
    "OrdonezB_Description.txt": "727f35c52afca460dee311eff8f3897065dfd75bc755b3dbb9de6b616df4d43d",
    "OrdonezB_Sensors.txt": "b0c77812b48f3b7d81afa446c80faba35b33bd99075350e929fa34a463c49a00",
    "README.txt": "c80e86f00bbc279d09a7154ae2d0a1f7d3d72a84ec20d4d694ed4641a3341014",
}

#: The downloaded archive's SHA-256.
ARCHIVE_SHA256 = "a03060857f2e9f9de2d0c7489e63bfd89b83c7d092e56e414b31f00c209964f5"

#: Where the archive is downloaded from.
ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/271/"
    "activities+of+daily+living+adls+recognition+using+binary+sensors.zip"
)

PROVENANCE = DatasetProvenance(
    name="UCI Activities of Daily Living (ADLs) Recognition Using Binary Sensors",
    version="1.0 (README, November 2013); UCI dataset 271, donated 2013-10-27",
    source=f"https://doi.org/10.24432/C5J02M; archive {ARCHIVE_URL}",
    licence="CC BY 4.0 (UCI Machine Learning Repository); the README also asks "
    "for the citation below and prohibits commercial use",
    citation="Ordóñez, F.J.; de Toledo, P.; Sanchis, A. Activity Recognition Using "
    "Hybrid Generative/Discriminative Models on Home Environments Using Binary "
    "Sensors. Sensors 2013, 13, 5460-5477",
    retrieved="2026-09-30",
    files=FILES,
    notes=(
        f"archive sha256 {ARCHIVE_SHA256}",
        "two single-resident homes in Spain; no timezone is given, and "
        f"{TIMEZONE} is declared",
        "OrdonezB_Description.txt lists door PIR sensors in the kitchen, "
        "bathroom and bedroom; the recorded sensors place them in the kitchen, "
        "living room and bedroom, and the recorded rooms are used",
    ),
)

#: Every sensor, as ``(Location, Type, Place)``: the description files' sensors
#: with the rooms the recordings give them in the adaptation period.
SENSORS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "OrdonezA": (
        ("Basin", "PIR", "Bathroom"),
        ("Bed", "Pressure", "Bedroom"),
        ("Cabinet", "Magnetic", "Bathroom"),
        ("Cooktop", "PIR", "Kitchen"),
        ("Cupboard", "Magnetic", "Kitchen"),
        ("Fridge", "Magnetic", "Kitchen"),
        ("Maindoor", "Magnetic", "Entrance"),
        ("Microwave", "Electric", "Kitchen"),
        ("Seat", "Pressure", "Living"),
        ("Shower", "PIR", "Bathroom"),
        ("Toaster", "Electric", "Kitchen"),
        ("Toilet", "Flush", "Bathroom"),
    ),
    "OrdonezB": (
        ("Basin", "PIR", "Bathroom"),
        ("Bed", "Pressure", "Bedroom"),
        ("Cupboard", "Magnetic", "Kitchen"),
        ("Door", "PIR", "Bedroom"),
        ("Door", "PIR", "Kitchen"),
        ("Door", "PIR", "Living"),
        ("Fridge", "Magnetic", "Kitchen"),
        ("Maindoor", "Magnetic", "Entrance"),
        ("Microwave", "Electric", "Kitchen"),
        ("Seat", "Pressure", "Living"),
        ("Shower", "PIR", "Bathroom"),
        ("Toilet", "Flush", "Bathroom"),
    ),
}

#: The labels the description files list for both homes.
LABELS = (
    "Breakfast",
    "Dinner",
    "Grooming",
    "Leaving",
    "Lunch",
    "Showering",
    "Sleeping",
    "Snack",
    "Spare_Time/TV",
    "Toileting",
)

ON, OFF = "ON", "OFF"


def sensor_id(location: str, kind: str, place: str) -> str:
    """The contract identifier of a sensor row."""
    return f"{location}.{kind}.{place}"


def native_type(location: str, kind: str) -> str:
    """The native sensor type: the dataset's type and the object it is on."""
    return f"{kind}/{location}"


def _events(
    kind: Modality, state: ObservationKind = ObservationKind.EVENT
) -> SensorSemantics:
    return SensorSemantics(kind, state, frozenset({ON}), frozenset({OFF}))


_FIXTURE_PIR = (
    "a PIR aimed at a fixture, detecting movement near it rather than across the "
    "room, as the CASAS area sensors do"
)
_MEAL = (
    "the label covers preparing and eating the meal, and does not say which, or "
    "where it was eaten"
)

ORDONEZ_MAPPING = OntologyMapping(
    dataset="uci-adl-binary-ordonez",
    version="1.0",
    description=(
        "Written from the README and the two description files, which list every "
        "label and sensor; the rooms and the meaning of Leaving were checked on "
        "the first seven days of each home only. Frozen before any scoring."
    ),
    labels=(
        exact("Sleeping", S.SLEEPING),
        approximate(
            "Leaving",
            S.AWAY,
            "named for the act, but in this annotation format it spans the "
            "absence: from 3 minutes to over 4 hours in the adaptation period",
        ),
        approximate("Toileting", S.BATHROOM_ACTIVITY, "one kind of bathroom activity"),
        approximate("Showering", S.BATHROOM_ACTIVITY, "one kind of bathroom activity"),
        approximate(
            "Grooming",
            S.BATHROOM_ACTIVITY,
            "grooming at the bathroom basin; one kind of bathroom activity",
        ),
        approximate(
            "Spare_Time/TV",
            S.HOME_INACTIVE,
            "leisure at home, mostly watching television and assumed seated; "
            "some spare time may be active",
        ),
        ambiguous("Breakfast", (S.KITCHEN_ACTIVITY, S.HOME_ACTIVE), _MEAL),
        ambiguous("Lunch", (S.KITCHEN_ACTIVITY, S.HOME_ACTIVE), _MEAL),
        ambiguous("Dinner", (S.KITCHEN_ACTIVITY, S.HOME_ACTIVE), _MEAL),
        ambiguous(
            "Snack",
            (S.KITCHEN_ACTIVITY, S.HOME_ACTIVE, S.HOME_INACTIVE),
            "a snack may be prepared in the kitchen or eaten anywhere",
        ),
    ),
    sensor_types=(
        approximate("PIR/Shower", _events(Modality.MOTION), _FIXTURE_PIR),
        approximate("PIR/Basin", _events(Modality.MOTION), _FIXTURE_PIR),
        approximate("PIR/Cooktop", _events(Modality.MOTION), _FIXTURE_PIR),
        approximate(
            "PIR/Door",
            _events(Modality.MOTION),
            "a PIR at a room's doorway, detecting passage rather than presence "
            "across the room",
        ),
        exact("Magnetic/Maindoor", _events(Modality.DOOR)),
        exact("Magnetic/Fridge", _events(Modality.CONTACT)),
        exact("Magnetic/Cupboard", _events(Modality.CONTACT)),
        exact("Magnetic/Cabinet", _events(Modality.CONTACT)),
        approximate(
            "Flush/Toilet",
            _events(Modality.CONTACT),
            "a toilet flush is the use of a fixture, recorded like a contact",
        ),
        exact("Pressure/Bed", _events(Modality.BED_PRESSURE, ObservationKind.STATE)),
        exact(
            "Pressure/Seat",
            _events(Modality.BED_PRESSURE, ObservationKind.STATE),
        ),
        unmappable(
            "Electric/Microwave",
            "appliance power use has no modality in the observation model",
        ),
        unmappable(
            "Electric/Toaster",
            "appliance power use has no modality in the observation model",
        ),
    ),
    locations=(
        exact("Bathroom", "bathroom"),
        exact("Bedroom", "bedroom"),
        exact("Kitchen", "kitchen"),
        exact("Living", "living"),
        approximate("Entrance", "hall", "the entrance, where the main door is"),
    ),
)


def _fields(line: str) -> list[str]:
    return [field.strip() for field in line.split("\t") if field.strip()]


def _time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def _rows(path: Path, width: int) -> list[list[str]]:
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    if len(lines) < 2 or not set(lines[1].replace("\t", "")) <= {"-", " "}:
        raise ContractError(f"{path.name} does not have the dataset's two-line header")
    rows = []
    for number, line in enumerate(lines[2:], start=3):
        if not line.strip():
            continue
        fields = _fields(line)
        if len(fields) != width:
            raise ContractError(
                f"{path.name} line {number} has {len(fields)} fields, not {width}"
            )
        rows.append(fields)
    return rows


def read_household(directory: Path, household: str) -> HouseholdData:
    """One home's files in the contract's terms, exactly as recorded."""
    if household not in HOUSEHOLDS:
        raise KeyError(f"no household {household!r} in the dataset")
    directory = Path(directory)
    described = {
        sensor_id(*triple): SensorDescription(
            sensor_id(*triple), native_type(triple[0], triple[1]), triple[2]
        )
        for triple in SENSORS[household]
    }
    events: list[RawEvent] = []
    extra: dict[str, SensorDescription] = {}
    for start, end, location, kind, place in _rows(
        directory / f"{household}_Sensors.txt", 5
    ):
        identifier = sensor_id(location, kind, place)
        if identifier not in described:
            extra.setdefault(
                identifier,
                SensorDescription(identifier, native_type(location, kind), place),
            )
        events.append(RawEvent(_time(start), identifier, ON))
        events.append(RawEvent(_time(end), identifier, OFF))
    annotations = tuple(
        Annotation(label, _time(start), _time(end), AnnotationSemantics.INTERVAL)
        for start, end, label in _rows(directory / f"{household}_ADLs.txt", 3)
    )
    return HouseholdData(
        household=household,
        timezone=TIMEZONE,
        sensors=(*described.values(), *extra.values()),
        events=tuple(events),
        annotations=annotations,
        residents=1,
    )


@dataclass(frozen=True)
class OrdonezAdapter:
    """The contract's adapter for the extracted archive."""

    directory: Path

    @property
    def provenance(self) -> DatasetProvenance:
        """The dataset's provenance, with every file's frozen digest."""
        return PROVENANCE

    def households(self) -> Sequence[str]:
        """The two homes."""
        return HOUSEHOLDS

    def load(self, household: str) -> HouseholdData:
        """One home's data."""
        return read_household(self.directory, household)


def first_day(data: HouseholdData) -> date:
    """The local date of the household's first annotation."""
    return min(a.start for a in data.annotations).date()


def period_start(data: HouseholdData, days: int) -> datetime:
    """Local midnight *days* days after the household's first annotated day."""
    return datetime.combine(first_day(data), datetime.min.time()) + timedelta(days=days)
