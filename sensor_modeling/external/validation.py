"""Validating an external household against the contract, before anything is scored.

Every check reports an :class:`Issue` with a code, a severity, a count and a few
examples. Nothing is repaired here. :mod:`.canonical` decides what each issue
means for conversion and reports what it discards.

========================  ========  ==================================================
Code                      Severity  Meaning
========================  ========  ==================================================
``missing_timezone``      error     no timezone, so local times cannot be placed
``invalid_timezone``      error     a timezone name no IANA database knows
``mixed_awareness``       error     some timestamps carry an offset and some do not
``dst_transition``        warning   naive local times in a daylight-saving gap or fold
``empty_household``       error     a household with no events at all
``timestamp_order``       warning   events not in time order as delivered
``duplicate_event``       warning   the same sensor, time and value more than once
``unknown_sensor``        error     events from a sensor the dataset does not describe
``unused_sensor``         info      a described sensor that never reports
``impossible_interval``   error     an interval that ends at or before it starts,
                                    a point with an end, an interval without one,
                                    or an occupancy period that cannot exist
``outside_recording``     warning   an annotation outside the span of the events
``overlapping_labels``    warning   annotations that overlap in time
``multi_resident``        warning   periods with more than one resident, or
                                    overlapping annotations of different residents:
                                    unsupported, and excluded from scoring
``multi_resident_home``   error     a household declared multi-resident throughout
``undeclared_label``      error     a label the mapping does not declare
``undeclared_sensor``     error     a sensor type or location the mapping does not
                                    declare
``sensor_semantics``      warning   a sensor whose type or location maps other than
                                    exactly, or whose values do not fit its
                                    declared semantics
``point_annotations``     info      annotations of a moment, not a span
========================  ========  ==================================================
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .contract import (
    Annotation,
    AnnotationSemantics,
    DatasetAdapter,
    EventValue,
    HouseholdData,
)
from .mapping import (
    MISSING,
    UNDECLARED,
    OntologyMapping,
    Outcome,
    Resolution,
    SensorSemantics,
)

#: How many examples an issue lists.
EXAMPLES = 5


class Severity(str, Enum):
    """How much an issue matters."""

    ERROR = "error"
    """The household cannot be converted as it stands."""

    WARNING = "warning"
    """It can be, and conversion reports what it discards or repairs."""

    INFO = "info"
    """Worth knowing; nothing is lost."""


@dataclass(frozen=True)
class Issue:
    """One problem found in one household."""

    code: str
    severity: Severity
    household: str
    message: str
    count: int = 1
    examples: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "household": self.household,
            "message": self.message,
            "count": self.count,
            "examples": list(self.examples),
        }


def _examples(items: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in list(items)[:EXAMPLES])


def zone_of(data: HouseholdData) -> ZoneInfo | None:
    """The household's timezone, or ``None`` if it has none or an unknown one."""
    if data.timezone is None:
        return None
    try:
        return ZoneInfo(data.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def localise(moment: datetime, zone: ZoneInfo) -> datetime:
    """An aware instant: a naive local time placed in *zone*, an aware one kept."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=zone)


def _dst_problem(moment: datetime, zone: ZoneInfo) -> str | None:
    """``"gap"`` or ``"fold"`` for a naive local time the zone makes non-unique."""
    early = moment.replace(tzinfo=zone, fold=0)
    late = moment.replace(tzinfo=zone, fold=1)
    if early.utcoffset() == late.utcoffset():
        return None
    back = early.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
    return "gap" if back != moment else "fold"


def _seconds(annotation: Annotation) -> float:
    if annotation.semantics is AnnotationSemantics.POINT or annotation.end is None:
        return 0.0
    return max((annotation.end - annotation.start).total_seconds(), 0.0)


def valid_interval(annotation: Annotation) -> bool:
    """Whether an annotation is an interval with a positive length."""
    return (
        annotation.semantics is AnnotationSemantics.INTERVAL
        and annotation.end is not None
        and annotation.end > annotation.start
    )


@dataclass(frozen=True)
class ValidationReport:
    """Everything validation found in one household.

    Attributes
    ----------
    household
        Which household.
    issues
        Every issue, in code order.
    labels
        Per native label: its mapping status, annotations and annotated seconds.
    sensors
        Per sensor: its type's and location's mapping status.
    coverage
        Annotated seconds and annotations by mapping status.
    """

    household: str
    issues: tuple[Issue, ...]
    labels: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    sensors: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    coverage: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @property
    def errors(self) -> tuple[Issue, ...]:
        """The issues that block conversion."""
        return tuple(i for i in self.issues if i.severity is Severity.ERROR)

    @property
    def ok(self) -> bool:
        """Whether the household can be converted."""
        return not self.errors

    def codes(self) -> set[str]:
        """Every issue code found."""
        return {i.code for i in self.issues}

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form."""
        return {
            "household": self.household,
            "ok": self.ok,
            "issues": [i.to_dict() for i in self.issues],
            "labels": {k: dict(v) for k, v in sorted(self.labels.items())},
            "sensors": {k: dict(v) for k, v in sorted(self.sensors.items())},
            "coverage": {k: dict(v) for k, v in self.coverage.items()},
        }


class _Collector:
    def __init__(self, household: str) -> None:
        self.household = household
        self.issues: list[Issue] = []

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        items: Sequence[Any] = (),
        count: int | None = None,
    ) -> None:
        self.issues.append(
            Issue(
                code,
                severity,
                self.household,
                message,
                count if count is not None else max(len(items), 1),
                _examples(items),
            )
        )


def _check_time(data: HouseholdData, found: _Collector) -> None:
    moments = [e.timestamp for e in data.events] + [
        m for a in data.annotations for m in (a.start, a.end) if m is not None
    ]
    naive = [m for m in moments if m.tzinfo is None]
    if naive and len(naive) != len(moments):
        found.add(
            "mixed_awareness",
            Severity.ERROR,
            "some timestamps carry an offset and some do not",
            count=len(naive),
        )
    if data.timezone is None:
        found.add(
            "missing_timezone",
            Severity.ERROR,
            "the household declares no timezone, so its local times and "
            "time-of-day features cannot be placed",
        )
        return
    zone = zone_of(data)
    if zone is None:
        found.add(
            "invalid_timezone",
            Severity.ERROR,
            f"{data.timezone!r} is not an IANA timezone",
            [data.timezone],
        )
        return
    problems = [(m, _dst_problem(m, zone)) for m in naive]
    flagged = [f"{m.isoformat()} ({kind})" for m, kind in problems if kind is not None]
    if flagged:
        found.add(
            "dst_transition",
            Severity.WARNING,
            "naive local times fall in a daylight-saving gap or fold, where "
            "they do not name one instant; the earlier reading is used",
            flagged,
        )


def _check_events(data: HouseholdData, found: _Collector) -> None:
    events = data.events
    if not events:
        found.add("empty_household", Severity.ERROR, "the household has no events")
        return
    try:
        backwards = [
            f"{b.timestamp.isoformat()} after {a.timestamp.isoformat()}"
            for a, b in zip(events, events[1:])
            if b.timestamp < a.timestamp
        ]
    except TypeError:
        backwards = []  # naive against aware: reported as mixed awareness
    if backwards:
        found.add(
            "timestamp_order",
            Severity.WARNING,
            "events are not in time order as delivered; conversion sorts them",
            backwards,
        )
    seen: Counter[tuple[datetime, str, EventValue]] = Counter(
        (e.timestamp, e.sensor_id, e.value) for e in events
    )
    duplicates = [key for key, n in seen.items() if n > 1]
    if duplicates:
        found.add(
            "duplicate_event",
            Severity.WARNING,
            "the same sensor, time and value appear more than once; conversion "
            "keeps one",
            [f"{s} {v!r} at {t.isoformat()}" for t, s, v in duplicates],
            count=sum(seen[k] - 1 for k in duplicates),
        )
    described = {s.sensor_id for s in data.sensors}
    unknown = Counter(e.sensor_id for e in events if e.sensor_id not in described)
    if unknown:
        found.add(
            "unknown_sensor",
            Severity.ERROR,
            "events come from sensors the dataset does not describe",
            [f"{s} ({n} events)" for s, n in sorted(unknown.items())],
            count=sum(unknown.values()),
        )
    reporting = {e.sensor_id for e in events}
    silent = sorted(described - reporting)
    if silent:
        found.add(
            "unused_sensor",
            Severity.INFO,
            "described sensors that never report",
            silent,
        )


def _check_intervals(data: HouseholdData, found: _Collector) -> None:
    impossible: list[str] = []
    for a in data.annotations:
        if a.semantics is AnnotationSemantics.INTERVAL:
            if a.end is None:
                impossible.append(f"{a.label} at {a.start.isoformat()}: no end")
            elif a.end <= a.start:
                impossible.append(
                    f"{a.label}: {a.start.isoformat()} to {a.end.isoformat()}"
                )
        elif a.end is not None and a.end != a.start:
            impossible.append(
                f"{a.label} at {a.start.isoformat()}: a point with an end"
            )
    for period in data.occupancy:
        if period.end <= period.start or period.residents < 1:
            impossible.append(
                f"occupancy {period.start.isoformat()} to {period.end.isoformat()} "
                f"with {period.residents} residents"
            )
    if impossible:
        found.add(
            "impossible_interval",
            Severity.ERROR,
            "intervals that cannot exist",
            impossible,
        )
    points = [a for a in data.annotations if a.semantics is AnnotationSemantics.POINT]
    if points:
        found.add(
            "point_annotations",
            Severity.INFO,
            "annotations of a moment carry no state span; conversion does not "
            "turn them into one",
            [f"{a.label} at {a.start.isoformat()}" for a in points],
        )
    intervals = sorted(
        (a for a in data.annotations if valid_interval(a)), key=lambda a: a.start
    )
    if data.events and intervals:
        try:
            first = min(e.timestamp for e in data.events)
            last = max(e.timestamp for e in data.events)
            outside = [
                f"{a.label}: {a.start.isoformat()} to {a.end.isoformat()}"  # type: ignore[union-attr]
                for a in intervals
                if a.start < first or a.end > last  # type: ignore[operator]
            ]
        except TypeError:
            outside = []
        if outside:
            found.add(
                "outside_recording",
                Severity.WARNING,
                "annotations extend beyond the span of the events",
                outside,
            )
    overlaps: list[str] = []
    residents: list[str] = []
    for i, a in enumerate(intervals):
        for b in intervals[i + 1 :]:
            if b.start >= a.end:  # type: ignore[operator]
                break
            text = f"{a.label} and {b.label} from {b.start.isoformat()}"
            if (
                a.resident is not None
                and b.resident is not None
                and a.resident != b.resident
            ):
                residents.append(f"{text}: residents {a.resident} and {b.resident}")
            else:
                overlaps.append(text)
    if overlaps:
        found.add(
            "overlapping_labels",
            Severity.WARNING,
            "annotations overlap; where their states differ, conversion leaves "
            "the overlap unscored",
            overlaps,
        )
    crowded = [p for p in data.occupancy if p.residents > 1 and p.end > p.start]
    if crowded or residents:
        found.add(
            "multi_resident",
            Severity.WARNING,
            "periods with more than one resident are unsupported by the "
            "single-resident ontology; conversion leaves them unscored",
            [
                f"{p.start.isoformat()} to {p.end.isoformat()}: {p.residents} "
                f"residents{' (' + p.note + ')' if p.note else ''}"
                for p in crowded
            ]
            + residents,
        )
    if data.residents is not None and data.residents > 1:
        found.add(
            "multi_resident_home",
            Severity.ERROR,
            f"the household is declared to have {data.residents} residents "
            "throughout: unsupported",
            [str(data.residents)],
        )


def _value_fits(value: EventValue, resolution: Resolution) -> bool:
    semantics: SensorSemantics = resolution.target
    if isinstance(value, float):
        return semantics.numeric
    return value in semantics.activation or value in semantics.deactivation


def _check_mapping(
    data: HouseholdData,
    mapping: OntologyMapping,
    found: _Collector,
) -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, float]]
]:
    labels: dict[str, dict[str, Any]] = {}
    for a in data.annotations:
        resolution = mapping.label(a.label)
        entry = labels.setdefault(
            a.label,
            {
                "status": resolution.status,
                "targets": [getattr(t, "value", t) for t in resolution.targets],
                "annotations": 0,
                "seconds": 0.0,
            },
        )
        entry["annotations"] += 1
        entry["seconds"] += _seconds(a)
    undeclared = sorted(k for k, v in labels.items() if v["status"] == UNDECLARED)
    if undeclared:
        found.add(
            "undeclared_label",
            Severity.ERROR,
            "labels the mapping does not declare; declare each as exact, "
            "approximate, unmappable or ambiguous",
            undeclared,
        )
    coverage: dict[str, dict[str, float]] = defaultdict(
        lambda: {"annotations": 0.0, "seconds": 0.0}
    )
    for entry in labels.values():
        coverage[entry["status"]]["annotations"] += entry["annotations"]
        coverage[entry["status"]]["seconds"] += entry["seconds"]

    sensors: dict[str, dict[str, Any]] = {}
    by_sensor: dict[str, list[EventValue]] = defaultdict(list)
    for e in data.events:
        by_sensor[e.sensor_id].append(e.value)
    undeclared_sensors: list[str] = []
    mismatched: list[str] = []
    for s in data.sensors:
        kind = mapping.sensor_type(s.sensor_type)
        place = mapping.location(s.location)
        sensors[s.sensor_id] = {
            "sensor_type": s.sensor_type,
            "type_status": kind.status,
            "location": s.location,
            "location_status": place.status,
            "room": place.target if place.usable else None,
        }
        if kind.status == UNDECLARED:
            undeclared_sensors.append(f"{s.sensor_id}: type {s.sensor_type!r}")
        if place.status == UNDECLARED:
            undeclared_sensors.append(f"{s.sensor_id}: location {s.location!r}")
        reasons = []
        if kind.status not in (Outcome.EXACT.value, UNDECLARED):
            reasons.append(f"type {s.sensor_type!r} is {kind.status}")
        if place.status not in (Outcome.EXACT.value, UNDECLARED):
            reasons.append(
                "no location"
                if place.status == MISSING
                else f"location {s.location!r} is {place.status}"
            )
        if kind.usable:
            values = by_sensor.get(s.sensor_id, [])
            bad = Counter(v for v in values if not _value_fits(v, kind))
            if bad:
                reasons.append(
                    f"{sum(bad.values())} values outside its semantics, such as "
                    f"{', '.join(repr(v) for v, _ in bad.most_common(3))}"
                )
                sensors[s.sensor_id]["unfit_values"] = sum(bad.values())
        if reasons:
            mismatched.append(f"{s.sensor_id}: " + "; ".join(reasons))
    if undeclared_sensors:
        found.add(
            "undeclared_sensor",
            Severity.ERROR,
            "sensor types or locations the mapping does not declare",
            undeclared_sensors,
        )
    if mismatched:
        found.add(
            "sensor_semantics",
            Severity.WARNING,
            "sensors whose semantics do not match the ontology exactly",
            mismatched,
        )
    return labels, sensors, dict(coverage)


def validate_household(
    data: HouseholdData, mapping: OntologyMapping | None = None
) -> ValidationReport:
    """Check one household against the contract, and against *mapping* if given."""
    found = _Collector(data.household)
    _check_time(data, found)
    _check_events(data, found)
    _check_intervals(data, found)
    labels: dict[str, dict[str, Any]] = {}
    sensors: dict[str, dict[str, Any]] = {}
    coverage: dict[str, dict[str, float]] = {}
    if mapping is not None:
        labels, sensors, coverage = _check_mapping(data, mapping, found)
    order = {s: k for k, s in enumerate(Severity)}
    issues = sorted(found.issues, key=lambda i: (order[i.severity], i.code))
    return ValidationReport(data.household, tuple(issues), labels, sensors, coverage)


@dataclass(frozen=True)
class DatasetReport:
    """Validation of every household of a dataset, against one mapping."""

    provenance: Mapping[str, Any]
    mapping_sha256: str | None
    households: tuple[ValidationReport, ...]

    @property
    def ok(self) -> bool:
        """Whether every household can be converted."""
        return all(h.ok for h in self.households)

    def totals(self) -> dict[str, Any]:
        """Issues by code, and annotated seconds by mapping status, over the dataset."""
        issues: Counter[str] = Counter()
        coverage: dict[str, dict[str, float]] = defaultdict(
            lambda: {"annotations": 0.0, "seconds": 0.0}
        )
        for household in self.households:
            for issue in household.issues:
                issues[issue.code] += issue.count
            for status, values in household.coverage.items():
                for key, value in values.items():
                    coverage[status][key] += value
        return {
            "households": len(self.households),
            "convertible": sum(1 for h in self.households if h.ok),
            "issues": dict(sorted(issues.items())),
            "coverage": {k: dict(v) for k, v in sorted(coverage.items())},
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form."""
        return {
            "provenance": dict(self.provenance),
            "mapping_sha256": self.mapping_sha256,
            "ok": self.ok,
            "totals": self.totals(),
            "households": [h.to_dict() for h in self.households],
        }


def validate_dataset(
    adapter: DatasetAdapter, mapping: OntologyMapping | None = None
) -> DatasetReport:
    """Validate every household an adapter provides."""
    return DatasetReport(
        adapter.provenance.to_dict(),
        mapping.sha256() if mapping is not None else None,
        tuple(
            validate_household(adapter.load(h), mapping) for h in adapter.households()
        ),
    )
