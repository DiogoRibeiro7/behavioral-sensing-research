"""Tests for the Phase 5 external-dataset contract, on synthetic external-style data.

The synthetic "Aurora Living Lab" names its sensors, rooms and activities its
own way. The contract must expose it without CASAS assumptions, map it only
through declared entries, report every problem, and never drop what it cannot
map without counting it.
"""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from sensor_modeling.datasets.casas import truth_series
from sensor_modeling.datasets.information_sets import HH_EVIDENCE_CHANNELS
from sensor_modeling.external import (
    CANONICAL_ROOMS,
    MISSING,
    UNDECLARED,
    Annotation,
    AnnotationSemantics,
    ContractError,
    CsvAdapter,
    DatasetProvenance,
    HouseholdData,
    InMemoryAdapter,
    MappingEntry,
    MappingError,
    OccupancyPeriod,
    OntologyMapping,
    Outcome,
    RawEvent,
    SensorDescription,
    SensorSemantics,
    Severity,
    ambiguous,
    approximate,
    exact,
    to_canonical,
    unmappable,
    validate_dataset,
    validate_household,
)
from sensor_modeling.observations import Modality, ObservationKind, Unit
from sensor_modeling.states import BehaviouralState
from sensor_modeling.states.ontology import DEFAULT_ROOMS

S = BehaviouralState
ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2024, 5, 6, 8, 0)
INTERVAL = AnnotationSemantics.INTERVAL
POINT = AnnotationSemantics.POINT

PROVENANCE = DatasetProvenance(
    name="aurora-living-lab",
    version="2.1",
    source="https://example.org/aurora",
    licence="CC-BY-4.0",
    citation="Aurora Living Lab, synthetic test fixture",
)

PIR = SensorSemantics(
    Modality.MOTION, ObservationKind.EVENT, frozenset({"ACTIVE"}), frozenset({"CLEAR"})
)
MAPPING = OntologyMapping(
    dataset="aurora-living-lab",
    version="2.1",
    description="written for the tests, before any result",
    labels=(
        exact("sleep", S.SLEEPING),
        exact("cooking", S.KITCHEN_ACTIVITY),
        exact("out", S.AWAY),
        approximate(
            "toilet", S.BATHROOM_ACTIVITY, "toileting is one bathroom activity"
        ),
        approximate("tv", S.HOME_INACTIVE, "watching television, assumed seated"),
        approximate("chores", S.HOME_ACTIVE, "housework anywhere in the home"),
        ambiguous("nap", (S.SLEEPING, S.HOME_INACTIVE), "a nap in bed or on the sofa"),
        unmappable("medication", "a brief action, not a behavioural state"),
    ),
    sensor_types=(
        exact("pir", PIR),
        exact(
            "reed",
            SensorSemantics(
                Modality.DOOR,
                ObservationKind.EVENT,
                frozenset({"OPEN"}),
                frozenset({"CLOSED"}),
            ),
        ),
        exact(
            "bed_mat",
            SensorSemantics(
                Modality.BED_PRESSURE,
                ObservationKind.STATE,
                frozenset({"OCCUPIED"}),
                frozenset({"EMPTY"}),
            ),
        ),
        approximate(
            "temp",
            SensorSemantics(
                Modality.ENVIRONMENTAL,
                ObservationKind.SAMPLE,
                numeric=True,
                unit=Unit.CELSIUS,
            ),
            "reported every 10 minutes rather than on change",
        ),
        unmappable("smart_plug", "appliance power has no counterpart in the ontology"),
        ambiguous(
            "presence",
            (
                PIR,
                SensorSemantics(
                    Modality.ROOM_OCCUPANCY,
                    ObservationKind.STATE,
                    frozenset({"ACTIVE"}),
                    frozenset({"CLEAR"}),
                ),
            ),
            "the vendor does not say whether it reports motion or occupancy",
        ),
    ),
    locations=(
        exact("Kitchen", "kitchen"),
        exact("Bath", "bathroom"),
        exact("Bedroom 1", "bedroom"),
        exact("Front door", "hall"),
        approximate("Lounge", "living", "an open-plan lounge-diner"),
        unmappable("Garage", "outside the monitored home"),
        ambiguous("Stairs", ("hall", "living"), "the stairs open onto both"),
    ),
)

SENSORS = (
    SensorDescription("k1", "pir", "Kitchen"),
    SensorDescription("b1", "pir", "Bath"),
    SensorDescription("l1", "pir", "Lounge"),
    SensorDescription("d1", "reed", "Front door"),
    SensorDescription("m1", "bed_mat", "Bedroom 1"),
    SensorDescription("t1", "temp", "Kitchen"),
    SensorDescription("p1", "smart_plug", "Kitchen"),
)


def minutes(m: float) -> datetime:
    return T0 + timedelta(minutes=m)


def clean() -> HouseholdData:
    """A household with nothing wrong."""
    events = [
        RawEvent(minutes(0), "m1", "OCCUPIED"),
        RawEvent(minutes(10), "m1", "EMPTY"),
        RawEvent(minutes(11), "b1", "ACTIVE"),
        RawEvent(minutes(12), "b1", "CLEAR"),
        RawEvent(minutes(20), "k1", "ACTIVE"),
        RawEvent(minutes(21), "t1", 21.5),
        RawEvent(minutes(22), "p1", 850.0),
        RawEvent(minutes(40), "l1", "ACTIVE"),
        RawEvent(minutes(60), "d1", "OPEN"),
        RawEvent(minutes(61), "d1", "CLOSED"),
        RawEvent(minutes(120), "d1", "OPEN"),
    ]
    annotations = [
        Annotation("sleep", minutes(0), minutes(10), INTERVAL),
        Annotation("toilet", minutes(10), minutes(15), INTERVAL),
        Annotation("cooking", minutes(18), minutes(35), INTERVAL),
        Annotation("tv", minutes(38), minutes(58), INTERVAL),
        Annotation("out", minutes(61), minutes(119), INTERVAL),
        Annotation("medication", minutes(30), None, POINT),
    ]
    return HouseholdData(
        "A01", "Europe/Lisbon", SENSORS, tuple(events), tuple(annotations)
    )


def codes(
    data: HouseholdData, mapping: OntologyMapping | None = MAPPING
) -> dict[str, Any]:
    return {i.code: i for i in validate_household(data, mapping).issues}


# ----------------------------------------------------------------------------
class TestContract:
    """What an adapter must expose."""

    def test_provenance_needs_its_fields(self) -> None:
        with pytest.raises(ContractError, match="licence"):
            DatasetProvenance("x", "1", "doi:x", "")
        with pytest.raises(ContractError, match="SHA-256"):
            DatasetProvenance("x", "1", "doi:x", "CC0", files={"a.csv": "abc"})
        assert PROVENANCE.to_dict()["licence"] == "CC-BY-4.0"

    def test_every_field_is_exposed_in_native_terms(self) -> None:
        data = clean()
        event = data.events[5]
        assert (data.household, data.timezone) == ("A01", "Europe/Lisbon")
        assert (event.timestamp, event.sensor_id, event.value) == (
            minutes(21),
            "t1",
            21.5,
        )
        sensor = data.sensors[2]
        assert (sensor.sensor_id, sensor.sensor_type, sensor.location) == (
            "l1",
            "pir",
            "Lounge",
        )
        note = data.annotations[-1]
        assert (note.label, note.semantics, note.end) == ("medication", POINT, None)

    def test_a_sensor_described_twice_is_refused(self) -> None:
        with pytest.raises(ContractError, match="more than once"):
            replace(clean(), sensors=(*SENSORS, SENSORS[0]))

    def test_the_generic_interface_holds_no_casas_assumption(self) -> None:
        for name in ("contract", "mapping", "validation"):
            source = (ROOT / "sensor_modeling" / "external" / f"{name}.py").read_text(
                encoding="utf-8"
            )
            imported = {
                node.module
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.ImportFrom) and node.module
            }
            assert not any("datasets" in m or "casas" in m for m in imported), name


# ----------------------------------------------------------------------------
class TestMapping:
    """Declared correspondences, and nothing silently resolved."""

    def test_the_four_outcomes_and_undeclared(self) -> None:
        assert MAPPING.label("sleep").status == "exact"
        assert MAPPING.label("sleep").target is S.SLEEPING
        assert MAPPING.label("toilet").status == "approximate"
        assert MAPPING.label("toilet").usable
        nap = MAPPING.label("nap")
        assert nap.status == "ambiguous" and not nap.usable
        assert set(nap.targets) == {S.SLEEPING, S.HOME_INACTIVE}
        assert MAPPING.label("medication").status == "unmappable"
        assert MAPPING.label("yoga").status == UNDECLARED
        assert MAPPING.location(None).status == MISSING
        with pytest.raises(MappingError, match="no single target"):
            _ = nap.target

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            (lambda: MappingEntry("x", Outcome.EXACT, ()), "one target"),
            (lambda: MappingEntry("x", Outcome.AMBIGUOUS, (S.AWAY,), "r"), "two"),
            (
                lambda: MappingEntry("x", Outcome.AMBIGUOUS, (S.AWAY, S.AWAY), "r"),
                "two",
            ),
            (lambda: MappingEntry("x", Outcome.UNMAPPABLE, (S.AWAY,), "r"), "none"),
            (lambda: approximate("x", S.AWAY, ""), "rationale"),
            (lambda: unmappable("x", " "), "rationale"),
        ],
    )
    def test_invalid_entries(self, entry: Any, message: str) -> None:
        with pytest.raises(MappingError, match=message):
            entry()

    def test_invalid_targets(self) -> None:
        base: dict[str, Any] = {"dataset": "d", "version": "1", "sensor_types": ()}
        with pytest.raises(MappingError, match="latent states"):
            OntologyMapping(labels=(exact("x", S.UNKNOWN),), **base)
        with pytest.raises(MappingError, match="must map to one of"):
            OntologyMapping(labels=(), locations=(exact("x", "attic"),), **base)
        with pytest.raises(MappingError, match="SensorSemantics"):
            OntologyMapping(labels=(), **{**base, "sensor_types": (exact("x", "pir"),)})
        with pytest.raises(MappingError, match="twice"):
            OntologyMapping(labels=(exact("x", S.AWAY), exact("x", S.SLEEPING)), **base)
        with pytest.raises(MappingError, match="numeric"):
            SensorSemantics(Modality.ENVIRONMENTAL, ObservationKind.SAMPLE)
        with pytest.raises(MappingError, match="activation"):
            SensorSemantics(Modality.MOTION, ObservationKind.EVENT)

    def test_a_mapping_is_frozen_with_its_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "mapping.json"
        digest = MAPPING.write(path)
        assert digest == MAPPING.sha256()
        back = OntologyMapping.read(path, digest)
        assert back.to_dict() == MAPPING.to_dict()
        assert back.sha256() == digest
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["labels"][0]["outcome"] = "exact"
        payload["labels"][0]["targets"] = ["away"]
        payload["labels"][0]["rationale"] = ""
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(MappingError, match="digest"):
            OntologyMapping.read(path)
        with pytest.raises(MappingError, match="is not the mapping"):
            OntologyMapping.read(_written(tmp_path), "0" * 64)

    def test_canonical_rooms_cover_the_ontology(self) -> None:
        rooms = {r for r in DEFAULT_ROOMS.values() if r is not None}
        channels = {c.room for c in HH_EVIDENCE_CHANNELS}
        assert rooms | channels == set(CANONICAL_ROOMS)


def _written(tmp_path: Path) -> Path:
    path = tmp_path / "again.json"
    MAPPING.write(path)
    return path


# ----------------------------------------------------------------------------
class TestValidation:
    """Every problem is reported, with its severity."""

    def test_a_clean_household(self) -> None:
        report = validate_household(clean(), MAPPING)
        assert report.ok
        assert not report.errors
        assert report.codes() <= {"sensor_semantics", "point_annotations"}
        assert report.labels["sleep"]["status"] == "exact"
        assert report.coverage["exact"]["seconds"] == pytest.approx((10 + 17 + 58) * 60)
        assert report.coverage["approximate"]["seconds"] == pytest.approx((5 + 20) * 60)
        assert report.coverage["unmappable"]["annotations"] == 1

    def test_missing_and_invalid_timezones(self) -> None:
        missing = codes(replace(clean(), timezone=None))
        assert missing["missing_timezone"].severity is Severity.ERROR
        invalid = codes(replace(clean(), timezone="Mars/Olympus_Mons"))
        assert invalid["invalid_timezone"].severity is Severity.ERROR
        data = clean()
        aware = RawEvent(minutes(5).replace(tzinfo=timezone.utc), "k1", "ACTIVE")
        mixed = codes(replace(data, events=(*data.events, aware)))
        assert mixed["mixed_awareness"].severity is Severity.ERROR

    def test_daylight_saving_gaps_and_folds(self) -> None:
        data = clean()
        gap = RawEvent(datetime(2024, 3, 31, 1, 30), "k1", "ACTIVE")
        fold = RawEvent(datetime(2024, 10, 27, 1, 30), "k1", "ACTIVE")
        issue = codes(replace(data, events=(*data.events, gap, fold)))["dst_transition"]
        assert issue.severity is Severity.WARNING
        assert issue.count == 2
        assert any("gap" in e for e in issue.examples)
        assert any("fold" in e for e in issue.examples)

    def test_timestamp_order_and_duplicates(self) -> None:
        data = clean()
        events = list(data.events)
        events[3], events[4] = events[4], events[3]
        events.append(events[0])
        events.append(events[0])
        found = codes(replace(data, events=tuple(events)))
        assert found["timestamp_order"].severity is Severity.WARNING
        assert found["duplicate_event"].count == 2

    def test_unknown_and_unused_sensors(self) -> None:
        data = clean()
        stray = RawEvent(minutes(50), "x9", "ACTIVE")
        found = codes(replace(data, events=(*data.events, stray, stray)))
        assert found["unknown_sensor"].severity is Severity.ERROR
        assert found["unknown_sensor"].count == 2
        spare = replace(
            data, sensors=(*SENSORS, SensorDescription("s9", "pir", "Bath"))
        )
        assert codes(spare)["unused_sensor"].severity is Severity.INFO

    @pytest.mark.parametrize(
        "annotation",
        [
            Annotation("sleep", minutes(10), minutes(5), INTERVAL),
            Annotation("sleep", minutes(10), minutes(10), INTERVAL),
            Annotation("sleep", minutes(10), None, INTERVAL),
            Annotation("medication", minutes(10), minutes(12), POINT),
        ],
    )
    def test_impossible_intervals(self, annotation: Annotation) -> None:
        data = clean()
        found = codes(replace(data, annotations=(*data.annotations, annotation)))
        assert found["impossible_interval"].severity is Severity.ERROR

    def test_impossible_occupancy(self) -> None:
        period = OccupancyPeriod(minutes(20), minutes(10), 2)
        assert "impossible_interval" in codes(replace(clean(), occupancy=(period,)))

    def test_overlapping_labels(self) -> None:
        data = clean()
        overlap = Annotation("chores", minutes(30), minutes(40), INTERVAL)
        found = codes(replace(data, annotations=(*data.annotations, overlap)))
        assert found["overlapping_labels"].severity is Severity.WARNING
        assert found["overlapping_labels"].count == 2

    def test_unsupported_multi_resident_periods(self) -> None:
        data = clean()
        guest = OccupancyPeriod(minutes(38), minutes(58), 2, "visitor")
        found = codes(replace(data, occupancy=(guest,)))
        assert found["multi_resident"].severity is Severity.WARNING
        assert "visitor" in found["multi_resident"].examples[0]
        two = (
            Annotation("tv", minutes(40), minutes(50), INTERVAL, resident="r1"),
            Annotation("cooking", minutes(45), minutes(55), INTERVAL, resident="r2"),
        )
        found = codes(replace(data, annotations=two))
        assert "multi_resident" in found and "overlapping_labels" not in found
        assert codes(replace(data, residents=2))["multi_resident_home"].severity is (
            Severity.ERROR
        )

    def test_annotations_outside_the_recording(self) -> None:
        data = clean()
        late = Annotation("out", minutes(119), minutes(300), INTERVAL)
        assert "outside_recording" in codes(replace(data, annotations=(late,)))

    def test_undeclared_labels_are_errors_not_drops(self) -> None:
        data = clean()
        yoga = Annotation("yoga", minutes(40), minutes(45), INTERVAL)
        report = validate_household(
            replace(data, annotations=(*data.annotations, yoga)), MAPPING
        )
        issue = {i.code: i for i in report.issues}["undeclared_label"]
        assert issue.severity is Severity.ERROR and issue.examples == ("yoga",)
        assert report.labels["yoga"]["status"] == UNDECLARED
        assert report.coverage[UNDECLARED]["seconds"] == pytest.approx(300.0)

    def test_sensor_semantic_mismatches(self) -> None:
        data = clean()
        sensors = (
            *SENSORS,
            SensorDescription("g1", "pir", "Garage"),
            SensorDescription("s1", "pir", "Stairs"),
            SensorDescription("n1", "pir", None),
            SensorDescription("v1", "presence", "Kitchen"),
            SensorDescription("q1", "doorbell", "Front door"),
            SensorDescription("a1", "pir", "Attic"),
        )
        events = (
            *data.events,
            RawEvent(minutes(41), "l1", "MAYBE"),
            RawEvent(minutes(42), "k1", 3.0),
        )
        report = validate_household(
            replace(data, sensors=sensors, events=events), MAPPING
        )
        found = {i.code: i for i in report.issues}
        # The issue lists a few examples; the sensor table holds every detail.
        statuses = {
            sensor: (entry["type_status"], entry["location_status"])
            for sensor, entry in report.sensors.items()
        }
        assert statuses["g1"] == ("exact", "unmappable")
        assert statuses["s1"] == ("exact", "ambiguous")
        assert statuses["n1"] == ("exact", MISSING)
        assert statuses["v1"] == ("ambiguous", "exact")
        assert statuses["p1"] == ("unmappable", "exact")
        assert statuses["t1"] == ("approximate", "exact")
        assert statuses["l1"] == ("exact", "approximate")
        assert statuses["q1"] == (UNDECLARED, "exact")
        assert statuses["a1"] == ("exact", UNDECLARED)
        assert report.sensors["l1"]["unfit_values"] == 1
        assert report.sensors["k1"]["unfit_values"] == 1
        mismatch = found["sensor_semantics"]
        assert mismatch.severity is Severity.WARNING
        assert mismatch.count == 8  # t1, p1, l1, k1, g1, s1, n1, v1
        undeclared = found["undeclared_sensor"]
        assert undeclared.severity is Severity.ERROR
        assert undeclared.count == 2
        assert any("doorbell" in e for e in undeclared.examples)
        assert any("Attic" in e for e in undeclared.examples)

    def test_without_a_mapping_only_the_contract_is_checked(self) -> None:
        report = validate_household(clean())
        assert report.ok and not report.labels and not report.sensors


# ----------------------------------------------------------------------------
class TestCanonical:
    """Conversion into the repository's form, with nothing lost uncounted."""

    def test_a_household_with_errors_is_refused(self) -> None:
        with pytest.raises(ContractError, match="missing_timezone"):
            to_canonical(replace(clean(), timezone=None), MAPPING)

    def test_events_become_canonical_observations(self) -> None:
        converted = to_canonical(clean(), MAPPING, source="aurora")
        recording = converted.recording
        zone = ZoneInfo("Europe/Lisbon")
        assert all(o.timestamp.tzinfo is zone for o in recording.observations)
        assert [o.timestamp for o in recording.observations] == sorted(
            o.timestamp for o in recording.observations
        )
        by_sensor: dict[str, list[float]] = {
            o.sensor_id: [] for o in recording.observations
        }
        for o in recording.observations:
            by_sensor[o.sensor_id].append(o.value)
        assert by_sensor["m1"] == [1.0, 0.0]  # a state emits both levels
        assert by_sensor["b1"] == [1.0]  # an event's deactivation is counted
        assert by_sensor["t1"] == [21.5]
        assert "p1" not in by_sensor
        assert recording.unmapped_sensors == frozenset({"p1"})
        spec = recording.registry.get("l1")
        assert spec is not None and spec.room == "living"
        assert spec.modality is Modality.MOTION
        assert recording.registry.get("d1").room == "hall"  # type: ignore[union-attr]
        events = converted.dispositions["events"]
        assert sum(events.values()) == len(clean().events)
        assert events == {
            "deactivation": 2,
            "emitted": 8,
            "sensor_unmappable": 1,
        }

    def test_annotations_become_non_overlapping_segments(self) -> None:
        data = clean()
        extra = (
            Annotation("chores", minutes(30), minutes(40), INTERVAL),  # conflicts
            Annotation(
                "nap", minutes(0), minutes(5), INTERVAL
            ),  # ambiguous, under sleep
        )
        converted = to_canonical(
            replace(data, annotations=(*data.annotations, *extra)), MAPPING
        )
        segments = converted.recording.activities
        for a, b in zip(segments, segments[1:]):
            assert a.end <= b.start
        states = {(s.start, s.label): s.state for s in segments}
        zone = ZoneInfo("Europe/Lisbon")
        at = {
            m: minutes(m).replace(tzinfo=zone) for m in (0, 5, 10, 18, 30, 35, 38, 40)
        }
        assert states[(at[0], "nap+sleep")] is S.SLEEPING  # the usable label decides
        assert states[(at[10], "toilet")] is S.BATHROOM_ACTIVITY
        assert states[(at[30], "chores+cooking")] is None  # conflicting states
        assert states[(at[38], "chores+tv")] is None
        seconds = converted.dispositions["annotated_seconds"]
        assert seconds["conflict"] == pytest.approx((5 + 2) * 60)
        assert converted.dispositions["point_annotations"] == 1
        # Unscored interval labels are listed; the point is counted separately.
        assert converted.recording.unmapped_activities == {"nap": 1}
        grid = [minutes(m).replace(tzinfo=zone) for m in (2, 12, 32, 45, 70)]
        assert truth_series(segments, grid) == [
            S.SLEEPING,
            S.BATHROOM_ACTIVITY,
            None,
            S.HOME_INACTIVE,
            S.AWAY,
        ]

    def test_multi_resident_periods_are_left_unscored(self) -> None:
        data = replace(
            clean(), occupancy=(OccupancyPeriod(minutes(40), minutes(50), 2),)
        )
        converted = to_canonical(data, MAPPING)
        seconds = converted.dispositions["annotated_seconds"]
        assert seconds["multi_resident"] == pytest.approx(10 * 60)
        zone = ZoneInfo("Europe/Lisbon")
        grid = [minutes(m).replace(tzinfo=zone) for m in (39, 45, 55)]
        assert truth_series(converted.recording.activities, grid) == [
            S.HOME_INACTIVE,
            None,
            S.HOME_INACTIVE,
        ]

    def test_every_annotated_second_is_accounted_for(self) -> None:
        converted = to_canonical(clean(), MAPPING)
        seconds = converted.dispositions["annotated_seconds"]
        covered = sum(
            (s.end - s.start).total_seconds() for s in converted.recording.activities
        )
        assert sum(seconds.values()) == pytest.approx(covered)
        assert seconds == {"approximate": 25 * 60, "exact": 85 * 60}

    def test_lenient_conversion_counts_what_it_leaves_out(self) -> None:
        data = clean()
        stray = RawEvent(minutes(50), "x9", "ACTIVE")
        odd = RawEvent(minutes(51), "k1", "MAYBE")
        dup = data.events[0]
        converted = to_canonical(
            replace(data, events=(*data.events, stray, odd, dup)), MAPPING, strict=False
        )
        events = converted.dispositions["events"]
        assert events["unknown_sensor"] == 1
        assert events["unfit_value"] == 1
        assert events["duplicate"] == 1
        assert sum(events.values()) == len(data.events) + 3

    def test_the_canonical_recording_feeds_the_pipeline_types(self) -> None:
        recording = to_canonical(clean(), MAPPING).recording
        summary = recording.summary()
        assert summary["observations"] == 8
        assert summary["unmapped_sensors"] == ["p1"]
        assert 0.0 < recording.labelled_fraction <= 1.0


# ----------------------------------------------------------------------------
class TestAdapters:
    """The reference adapters and the dataset report."""

    def test_in_memory_adapter_and_dataset_report(self) -> None:
        second = replace(clean(), household="A02", timezone=None)
        adapter = InMemoryAdapter(PROVENANCE, {"A01": clean(), "A02": second})
        report = validate_dataset(adapter, MAPPING)
        assert [h.household for h in report.households] == ["A01", "A02"]
        assert not report.ok
        totals = report.totals()
        assert totals["convertible"] == 1
        assert totals["issues"]["missing_timezone"] == 1
        assert totals["coverage"]["exact"]["annotations"] == 6
        payload = report.to_dict()
        assert payload["mapping_sha256"] == MAPPING.sha256()
        json.dumps(payload, allow_nan=False)
        with pytest.raises(ContractError, match="filed under"):
            InMemoryAdapter(PROVENANCE, {"B": clean()})
        with pytest.raises(KeyError):
            adapter.load("Z")

    def test_csv_adapter(self, tmp_path: Path) -> None:
        data = clean()

        def write(name: str, header: list[str], rows: list[list[Any]]) -> None:
            with (tmp_path / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(rows)

        write(
            "households.csv",
            ["household", "timezone", "residents"],
            [["A01", "Europe/Lisbon", "1"]],
        )
        write(
            "sensors.csv",
            ["household", "sensor_id", "sensor_type", "location"],
            [
                ["A01", s.sensor_id, s.sensor_type, s.location or ""]
                for s in data.sensors
            ],
        )
        write(
            "events.csv",
            ["household", "timestamp", "sensor_id", "value"],
            [
                ["A01", e.timestamp.isoformat(), e.sensor_id, e.value]
                for e in data.events
            ],
        )
        write(
            "annotations.csv",
            ["household", "label", "start", "end", "semantics", "resident"],
            [
                [
                    "A01",
                    a.label,
                    a.start.isoformat(),
                    a.end.isoformat() if a.end else "",
                    a.semantics.value,
                    "",
                ]
                for a in data.annotations
            ],
        )
        adapter = CsvAdapter(tmp_path, PROVENANCE)
        assert adapter.households() == ("A01",)
        assert set(adapter.provenance.files) == {
            "annotations.csv",
            "events.csv",
            "households.csv",
            "sensors.csv",
        }
        loaded = adapter.load("A01")
        assert loaded.events == data.events
        assert loaded.annotations == data.annotations
        assert loaded.sensors == data.sensors
        assert loaded.residents == 1
        assert validate_dataset(adapter, MAPPING).ok
