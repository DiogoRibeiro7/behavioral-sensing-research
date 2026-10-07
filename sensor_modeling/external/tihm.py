"""The TIHM dataset: in-home monitoring of people living with dementia.

Palermo and colleagues released the records of 56 people living with dementia,
collected between April and June 2019 by the TIHM (Technology Integrated Health
Management) study: motion sensors in or at rooms, sensors on the entrance
doors and the fridge door, daily physiological measurements, a sleep mat in
some homes, and a table of alerts that a clinical monitoring team verified. It
is independent of CASAS and of the UCI ADL Binary dataset, and it differs from
both in kind: its annotations are verified alerts, not activities.

Source and terms
----------------
Zenodo record 7622128, DOI ``10.5281/zenodo.7622128``, version 1.0, under
CC BY 4.0. The dataset's repository asks that Surrey and Borders Partnership
NHS Foundation Trust and Howz be acknowledged in any publication or use, and
that the dataset paper be cited: Palermo, F. et al. "TIHM: An open dataset for
remote healthcare monitoring in dementia", *Scientific Data* 10, 606 (2023).
Nothing is redistributed with this package.

Format
------
The archive extracts to a ``Dataset`` folder of five CSV files, cross-referenced
by ``patient_id``:

- ``Activity.csv``: one row per sensor activation, with ``patient_id``,
  ``location_name`` and ``date``, a timestamp to the second;
- ``Labels.csv``: one row per verified alert, with ``patient_id``, ``date`` and
  ``type``;
- ``Demographics.csv``: one row per participant, with an age band and sex;
- ``Physiology.csv`` and ``Sleep.csv``: daily measurements and per-minute sleep
  mat records.

What this adapter exposes
-------------------------
The adapter reads ``Activity.csv``, ``Labels.csv`` and ``Demographics.csv``. It
does not read ``Physiology.csv`` or ``Sleep.csv``: a spot measurement taken
once a day has no modality in the observation model, and the sleep mat reports
a sleep stage computed by the device, in 17 of the 56 homes. Both files are
pinned by digest so a result can say exactly which release it did not read.

In the contract's terms:

- **Households.** One per ``patient_id`` in ``Demographics.csv``. The dataset
  does not say how many people live in a home, so ``residents`` is ``None``.
- **Sensors.** The table names a location and nothing else. A household's
  sensors are the locations that appear in its rows, and a sensor's identifier
  is its ``location_name``. The table names no sensor type, and the dataset
  paper is not of one voice about it. Its methods place passive infrared
  sensors in the hallway and living room, movement sensors on the kitchen,
  bedroom and bathroom doors, and a door sensor on the main entrance; its
  first figure's caption says passive infrared and door sensors are included
  in each room. The native types :data:`PIR`, :data:`DOOR` and
  :data:`FRIDGE_DOOR` are named from that description, and the frozen mapping
  takes the five room locations as motion in that room, with the sensor type
  declared approximate. A room sensor's location is its room. The dataset
  does not say which room a door or the fridge is in, so their location is
  ``None``.
- **Events.** Each row is one activation. The table records no value, so every
  event carries the token :data:`ACTIVE`.
- **Annotations.** Each label row is a point annotation at its timestamp. The
  timestamp is when the alert was recorded, not when anything began or ended.
- **Timezone.** Timestamps carry no timezone and the dataset gives none. The
  study was run with an NHS trust in England, and ``Europe/London`` is
  declared.

The mapping
-----------
:data:`TIHM_MAPPING` was written from the dataset paper, the repository README
and the files' column values. Every label is unmappable: an alert that a team
verified is a judgement about a person, and the ontology's states describe
what a resident is doing. No annotated time can therefore be scored against a
state, and the contract says so instead of guessing a correspondence.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from ..observations import Modality, ObservationKind
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
    approximate,
    exact,
    unmappable,
)

#: The timezone declared for every home.
TIMEZONE = "Europe/London"

#: The folder the archive extracts to.
FOLDER = "Dataset"

#: Each file's SHA-256, as downloaded from Zenodo record 7622128 on 2026-10-07.
FILES: dict[str, str] = {
    "Activity.csv": "ead7b1e3c2a91fbd679a909bff3a309a6465ae1ad18528dae57a4cf8599cafca",
    "Demographics.csv": "15b13a680da6ee3790dfadc25d3a6102070f6e8fe1a47ea6da8f64b6c4b757d4",
    "Labels.csv": "39495a08fdf1ff8f9b9b88bc98483a565c4433f58713ba3e4d0846e4449b846b",
    "Physiology.csv": "377b3170b4b650795648e18258f30f9b985c214937823e9c9ea405c66b4ba696",
    "Sleep.csv": "f1d5a6980263a55a8b69841e13bb349ce6e79152c15bdde0beffeea15293b9fe",
}

#: The files the adapter reads. The others are pinned and not read.
READ_FILES = ("Activity.csv", "Demographics.csv", "Labels.csv")

#: The downloaded archive's SHA-256, and the MD5 the Zenodo record publishes.
ARCHIVE_SHA256 = "368d642b4cdc680d0706abfd3c8b2e5387824d9737ad33f50531bafa6a4bf5a1"
ARCHIVE_MD5 = "ccf913f38a7808bc366bf0e799b409e4"

#: Where the archive is downloaded from.
ARCHIVE_URL = "https://zenodo.org/records/7622128/files/TIHM_Dataset.zip"

PROVENANCE = DatasetProvenance(
    name="TIHM: An Open Dataset for Remote Healthcare Monitoring in Dementia",
    version="1.0 (Zenodo record 7622128, published 2023-02-10)",
    source=f"https://doi.org/10.5281/zenodo.7622128; archive {ARCHIVE_URL}",
    licence="CC BY 4.0; the dataset's repository asks that Surrey and Borders "
    "Partnership NHS Foundation Trust and Howz be acknowledged in any "
    "publication or use",
    citation="Palermo, F.; Chen, Y.; Capstick, A.; Fletcher-Lloyd, N.; Walsh, C.; "
    "Kouchaki, S.; True, J.; Balazikova, O.; Soreq, E.; Scott, G.; Rostill, H.; "
    "Nilforooshan, R.; Barnaghi, P. TIHM: An open dataset for remote healthcare "
    "monitoring in dementia. Scientific Data 10, 606 (2023). "
    "https://doi.org/10.1038/s41597-023-02519-y",
    retrieved="2026-10-07",
    files=FILES,
    notes=(
        f"archive sha256 {ARCHIVE_SHA256}; md5 {ARCHIVE_MD5}, as Zenodo publishes",
        "56 people living with dementia, monitored between 2019-04-01 and "
        f"2019-06-30; no timezone is given, and {TIMEZONE} is declared",
        "Physiology.csv and Sleep.csv are pinned and not read by the adapter",
        "labels are alerts the study's monitoring team verified; seven "
        "participants have none and do not appear in Labels.csv",
    ),
)

#: The native sensor types, named from the dataset paper's description.
PIR, DOOR, FRIDGE_DOOR = "pir", "door", "fridge_door"

#: A location the adapter was not written for. The mapping does not declare it,
#: so validation reports it.
UNKNOWN_TYPE = "unknown"

#: Every ``location_name`` in the release, and the native type it is given.
LOCATIONS: dict[str, str] = {
    "Back Door": DOOR,
    "Bathroom": PIR,
    "Bedroom": PIR,
    "Fridge Door": FRIDGE_DOOR,
    "Front Door": DOOR,
    "Hallway": PIR,
    "Kitchen": PIR,
    "Lounge": PIR,
}

#: Every label ``type`` in the release.
LABELS = (
    "Agitation",
    "Blood pressure",
    "Body temperature",
    "Body water",
    "Pulse",
    "Weight",
)

#: The value every event carries: the table records an activation as a row.
ACTIVE = "ACTIVE"

_NO_TYPE = (
    "the Activity table names a location and no sensor type; the dataset paper "
    "describes {what}, and each row is taken as one activation"
)
_ALERT = (
    "an alert the study's monitoring team verified, recorded at a moment: a "
    "judgement about a person, not a behavioural state the resident is in"
)


def _event(modality: Modality) -> SensorSemantics:
    return SensorSemantics(modality, ObservationKind.EVENT, frozenset({ACTIVE}))


TIHM_MAPPING = OntologyMapping(
    dataset="tihm",
    version="1.0",
    description=(
        "Written from the dataset paper, the repository README and the column "
        "values of the release. Sensor types are named from the paper's "
        "description, because the Activity table gives a location only. Every "
        "label is a verified alert and is unmappable to a behavioural state."
    ),
    labels=tuple(unmappable(label, _ALERT) for label in LABELS),
    sensor_types=(
        approximate(
            PIR,
            _event(Modality.MOTION),
            _NO_TYPE.format(what="passive infrared sensors in rooms"),
        ),
        approximate(
            DOOR,
            _event(Modality.DOOR),
            _NO_TYPE.format(what="door sensors on the front and back doors")
            + ", with no open or closed value",
        ),
        approximate(
            FRIDGE_DOOR,
            _event(Modality.CONTACT),
            _NO_TYPE.format(what="a door sensor on the fridge")
            + ", with no open or closed value",
        ),
    ),
    locations=(
        exact("Bathroom", "bathroom"),
        exact("Bedroom", "bedroom"),
        exact("Hallway", "hall"),
        exact("Kitchen", "kitchen"),
        exact("Lounge", "living"),
    ),
)


def sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_files(directory: Path) -> dict[str, str]:
    """Files that are missing or do not match their pinned digest, with why."""
    problems: dict[str, str] = {}
    for name, digest in FILES.items():
        path = Path(directory) / name
        if not path.is_file():
            problems[name] = "missing"
        elif sha256(path) != digest:
            problems[name] = "does not match its pinned digest"
    return problems


def _rows(path: Path, columns: Sequence[str]) -> list[list[str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != list(columns):
            raise ContractError(
                f"{Path(path).name} has columns {header}, not {list(columns)}"
            )
        return [row for row in reader if row]


def _time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def describe(location: str) -> SensorDescription:
    """One sensor, from the only thing the table says about it: its location name."""
    kind = LOCATIONS.get(location, UNKNOWN_TYPE)
    return SensorDescription(location, kind, location if kind == PIR else None)


class TihmAdapter:
    """The contract's adapter for the extracted archive.

    Parameters
    ----------
    directory
        The extracted ``Dataset`` folder.

    The activity table has over a million rows, so it is read once and held by
    household.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._events: dict[str, list[RawEvent]] | None = None
        self._annotations: dict[str, list[Annotation]] | None = None
        self._households: tuple[str, ...] | None = None

    @property
    def provenance(self) -> DatasetProvenance:
        """The dataset's provenance, with every file's pinned digest."""
        return PROVENANCE

    def households(self) -> Sequence[str]:
        """Every participant in the demographics table, sorted."""
        if self._households is None:
            rows = _rows(
                self.directory / "Demographics.csv", ("patient_id", "age", "sex")
            )
            self._households = tuple(sorted(row[0] for row in rows))
        return self._households

    def demographics(self) -> dict[str, dict[str, str]]:
        """Each participant's age band and sex, as the dataset gives them."""
        rows = _rows(self.directory / "Demographics.csv", ("patient_id", "age", "sex"))
        return {row[0]: {"age": row[1], "sex": row[2]} for row in rows}

    def _read(self) -> None:
        events: dict[str, list[RawEvent]] = defaultdict(list)
        for household, location, moment in _rows(
            self.directory / "Activity.csv", ("patient_id", "location_name", "date")
        ):
            events[household].append(RawEvent(_time(moment), location, ACTIVE))
        annotations: dict[str, list[Annotation]] = defaultdict(list)
        for household, moment, label in _rows(
            self.directory / "Labels.csv", ("patient_id", "date", "type")
        ):
            annotations[household].append(
                Annotation(label, _time(moment), None, AnnotationSemantics.POINT)
            )
        self._events, self._annotations = dict(events), dict(annotations)

    def load(self, household: str) -> HouseholdData:
        """One participant's home, in the dataset's own terms."""
        if household not in self.households():
            raise KeyError(f"no household {household!r} in the dataset")
        if self._events is None or self._annotations is None:
            self._read()
        assert self._events is not None and self._annotations is not None
        events = self._events.get(household, [])
        return HouseholdData(
            household=household,
            timezone=TIMEZONE,
            sensors=tuple(
                describe(location) for location in sorted({e.sensor_id for e in events})
            ),
            events=tuple(events),
            annotations=tuple(self._annotations.get(household, [])),
            residents=None,
        )
