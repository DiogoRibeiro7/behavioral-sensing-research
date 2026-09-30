"""The external-dataset contract: what an adapter must expose, in the dataset's own terms.

Phase 5 tests whether results survive outside the CASAS ecosystem, on an
independently collected, annotated smart-home dataset. Such a dataset names its
sensors, types, rooms and activities its own way. This module fixes what an
adapter must deliver, before anything is interpreted:

============================  =============================================
Exposed                       Where
============================  =============================================
household identifier          :attr:`HouseholdData.household`
timestamp                     :attr:`RawEvent.timestamp`
sensor identifier             :attr:`RawEvent.sensor_id`,
                              :attr:`SensorDescription.sensor_id`
sensor type                   :attr:`SensorDescription.sensor_type`
location or room              :attr:`SensorDescription.location`
raw event or value            :attr:`RawEvent.value`, exactly as recorded
behavioural annotation        :attr:`Annotation.label`, the dataset's own
annotation semantics          :attr:`Annotation.semantics`: interval or point
timezone metadata             :attr:`HouseholdData.timezone`, an IANA name
dataset provenance            :class:`DatasetProvenance`
============================  =============================================

Everything here is the dataset's native vocabulary. Nothing assumes CASAS
sensor prefixes, begin and end markers, room names or activity labels.
Interpretation happens in two separate, explicit steps:

- :mod:`.mapping` declares how native labels, sensor types and locations
  correspond to this repository's ontology, entry by entry;
- :mod:`.validation` checks a household against the contract and the mapping,
  and :mod:`.canonical` converts it, reporting everything it cannot carry.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

EventValue = str | float


class ContractError(ValueError):
    """Data that cannot be expressed in the contract at all."""


@dataclass(frozen=True)
class DatasetProvenance:
    """Where a dataset comes from, and on what terms.

    Attributes
    ----------
    name, version
        The dataset and the release read.
    source
        Where it was obtained: a DOI, a URL or an archive identifier.
    licence
        The terms it is used under.
    citation
        How to cite it.
    retrieved
        When it was obtained, ISO 8601.
    files
        Each input file's name and SHA-256.
    notes
        Anything a reader needs, such as known collection quirks.
    """

    name: str
    version: str
    source: str
    licence: str
    citation: str = ""
    retrieved: str | None = None
    files: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require the fields a result's provenance cannot do without."""
        for name in ("name", "version", "source", "licence"):
            if not str(getattr(self, name)).strip():
                raise ContractError(f"dataset provenance needs a {name}")
        for filename, digest in self.files.items():
            if not isinstance(digest, str) or len(digest) != 64:
                raise ContractError(f"{filename!r} needs a SHA-256 hex digest")
        object.__setattr__(self, "files", dict(sorted(self.files.items())))
        object.__setattr__(self, "notes", tuple(self.notes))

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form."""
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "licence": self.licence,
            "citation": self.citation,
            "retrieved": self.retrieved,
            "files": dict(self.files),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SensorDescription:
    """One sensor, as the dataset describes it.

    Attributes
    ----------
    sensor_id
        The dataset's identifier.
    sensor_type
        The dataset's own type, such as ``"pir"`` or ``"reed_switch"``.
    location
        The dataset's own location or room, or ``None`` where it gives none.
    attributes
        Any further native metadata, kept verbatim.
    """

    sensor_id: str
    sensor_type: str
    location: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Require an identifier and a type."""
        if not str(self.sensor_id).strip():
            raise ContractError("a sensor needs an identifier")
        if not str(self.sensor_type).strip():
            raise ContractError(f"sensor {self.sensor_id!r} needs a type")


@dataclass(frozen=True)
class RawEvent:
    """One sensor report, exactly as recorded.

    Attributes
    ----------
    timestamp
        When it happened. Naive timestamps are allowed here, and interpreted
        only through the household's declared timezone.
    sensor_id
        Which sensor reported.
    value
        The raw value: a native token such as ``"ON"``, or a number.
    """

    timestamp: datetime
    sensor_id: str
    value: EventValue


class AnnotationSemantics(str, Enum):
    """Whether an annotation covers a span of time or marks a moment."""

    INTERVAL = "interval"
    """The label holds from ``start`` until ``end``, end-exclusive."""

    POINT = "point"
    """The label marks the moment ``start``; it says nothing about a span."""


@dataclass(frozen=True)
class Annotation:
    """One behavioural annotation, in the dataset's own label.

    Attributes
    ----------
    label
        The dataset's label.
    start, end
        The span for an interval; ``end`` is ``None`` for a point.
    semantics
        :attr:`AnnotationSemantics.INTERVAL` or :attr:`AnnotationSemantics.POINT`.
    resident
        Whom the annotation describes, where the dataset says.
    """

    label: str
    start: datetime
    end: datetime | None
    semantics: AnnotationSemantics
    resident: str | None = None

    def __post_init__(self) -> None:
        """Normalise the semantics; impossible spans are reported by validation."""
        object.__setattr__(self, "semantics", AnnotationSemantics(self.semantics))
        if not str(self.label).strip():
            raise ContractError("an annotation needs a label")


@dataclass(frozen=True)
class OccupancyPeriod:
    """How many people were in the home over a period.

    Attributes
    ----------
    start, end
        The period, end-exclusive.
    residents
        People present or monitored. More than one is a multi-resident period,
        which the repository's single-resident ontology does not support.
    note
        Why the dataset records it, such as a visitor or a second tenant.
    """

    start: datetime
    end: datetime
    residents: int
    note: str = ""


@dataclass(frozen=True)
class HouseholdData:
    """Everything the dataset holds about one household, in its own terms.

    Attributes
    ----------
    household
        The dataset's household identifier.
    timezone
        The IANA timezone the household's local times are in, or ``None`` where
        the dataset does not say; validation reports it.
    sensors, events, annotations
        As the dataset describes them.
    occupancy
        Periods with a known number of residents.
    residents
        The household's nominal number of residents, where the dataset says.
    """

    household: str
    timezone: str | None
    sensors: tuple[SensorDescription, ...]
    events: tuple[RawEvent, ...]
    annotations: tuple[Annotation, ...]
    occupancy: tuple[OccupancyPeriod, ...] = ()
    residents: int | None = None

    def __post_init__(self) -> None:
        """Require an identifier and freeze the collections."""
        if not str(self.household).strip():
            raise ContractError("a household needs an identifier")
        for name in ("sensors", "events", "annotations", "occupancy"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        ids = [s.sensor_id for s in self.sensors]
        if len(set(ids)) != len(ids):
            raise ContractError(
                f"household {self.household!r} describes a sensor more than once"
            )


class DatasetAdapter(Protocol):
    """What every external dataset adapter provides."""

    @property
    def provenance(self) -> DatasetProvenance:
        """Where the dataset comes from."""

    def households(self) -> Sequence[str]:
        """The dataset's household identifiers, sorted."""

    def load(self, household: str) -> HouseholdData:
        """One household's data, in the dataset's own terms."""


@dataclass(frozen=True)
class InMemoryAdapter:
    """An adapter over data already held in memory: the reference implementation."""

    provenance: DatasetProvenance
    data: Mapping[str, HouseholdData]

    def __post_init__(self) -> None:
        """Check that every entry is filed under its own identifier."""
        for key, household in self.data.items():
            if key != household.household:
                raise ContractError(
                    f"household {household.household!r} is filed under {key!r}"
                )

    def households(self) -> Sequence[str]:
        """The household identifiers, sorted."""
        return tuple(sorted(self.data))

    def load(self, household: str) -> HouseholdData:
        """One household's data."""
        if household not in self.data:
            raise KeyError(f"no household {household!r} in {self.provenance.name}")
        return self.data[household]


# ----------------------------------------------------------------------------
# A long-format CSV adapter
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class CsvLayout:
    """Which columns of a long-format dataset hold which contract field.

    A dataset in this layout has four files: one row per event, per
    annotation, per sensor, and per household. Column names are the dataset's
    and are declared here rather than guessed.
    """

    events: str = "events.csv"
    annotations: str = "annotations.csv"
    sensors: str = "sensors.csv"
    households: str = "households.csv"
    household: str = "household"
    timestamp: str = "timestamp"
    sensor_id: str = "sensor_id"
    value: str = "value"
    sensor_type: str = "sensor_type"
    location: str = "location"
    label: str = "label"
    start: str = "start"
    end: str = "end"
    semantics: str = "semantics"
    resident: str = "resident"
    timezone: str = "timezone"
    residents: str = "residents"
    timestamp_format: str | None = None
    """``strptime`` format; ISO 8601 when ``None``."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _optional(value: str | None) -> str | None:
    return value if value not in (None, "") else None


def _raw_value(text: str) -> EventValue:
    try:
        return float(text)
    except ValueError:
        return text


class CsvAdapter:
    """An adapter over a long-format CSV dataset, with its columns declared.

    Parameters
    ----------
    directory
        Where the four files are.
    provenance
        The dataset's provenance, without files: each file's digest is added.
    layout
        Which columns hold which field.
    """

    def __init__(
        self,
        directory: Path,
        provenance: DatasetProvenance,
        layout: CsvLayout | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.layout = layout or CsvLayout()
        names = (
            self.layout.events,
            self.layout.annotations,
            self.layout.sensors,
            self.layout.households,
        )
        files = {name: _sha256(self.directory / name) for name in names}
        self._provenance = DatasetProvenance(
            **{**provenance.to_dict(), "files": files, "notes": tuple(provenance.notes)}
        )
        self._households = {
            row[self.layout.household]: row
            for row in _rows(self.directory / self.layout.households)
        }

    @property
    def provenance(self) -> DatasetProvenance:
        """The provenance, with every file's digest."""
        return self._provenance

    def households(self) -> Sequence[str]:
        """The household identifiers, sorted."""
        return tuple(sorted(self._households))

    def _time(self, text: str) -> datetime:
        if self.layout.timestamp_format is None:
            return datetime.fromisoformat(text)
        return datetime.strptime(text, self.layout.timestamp_format)

    def _of(self, name: str, household: str) -> Iterable[dict[str, str]]:
        return (
            row
            for row in _rows(self.directory / name)
            if row[self.layout.household] == household
        )

    def load(self, household: str) -> HouseholdData:
        """One household's rows, in file order."""
        if household not in self._households:
            raise KeyError(f"no household {household!r}")
        layout = self.layout
        meta = self._households[household]
        residents = _optional(meta.get(layout.residents))
        return HouseholdData(
            household=household,
            timezone=_optional(meta.get(layout.timezone)),
            sensors=tuple(
                SensorDescription(
                    row[layout.sensor_id],
                    row[layout.sensor_type],
                    _optional(row.get(layout.location)),
                )
                for row in self._of(layout.sensors, household)
            ),
            events=tuple(
                RawEvent(
                    self._time(row[layout.timestamp]),
                    row[layout.sensor_id],
                    _raw_value(row[layout.value]),
                )
                for row in self._of(layout.events, household)
            ),
            annotations=tuple(
                Annotation(
                    row[layout.label],
                    self._time(row[layout.start]),
                    (
                        self._time(row[layout.end])
                        if _optional(row.get(layout.end))
                        else None
                    ),
                    AnnotationSemantics(row[layout.semantics]),
                    _optional(row.get(layout.resident)),
                )
                for row in self._of(layout.annotations, household)
            ),
            residents=int(residents) if residents is not None else None,
        )
