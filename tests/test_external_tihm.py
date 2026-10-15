"""Tests for the TIHM adapter, on synthetic files in the dataset's format.

The real dataset is not redistributed. These tests write small files with the
release's column layout and check that the adapter exposes them in the
contract's terms, that its mapping declares every native value, and that the
contract refuses to score anything against a state.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from sensor_modeling.external import (
    AnnotationSemantics,
    ContractError,
    OntologyMapping,
    Outcome,
    to_canonical,
    validate_dataset,
    validate_household,
)
from sensor_modeling.external.tihm import (
    ACTIVE,
    DOOR,
    FILES,
    FRIDGE_DOOR,
    LABELS,
    LOCATIONS,
    PIR,
    PROVENANCE,
    READ_FILES,
    TIHM_MAPPING,
    TIMEZONE,
    UNKNOWN_TYPE,
    TihmAdapter,
    check_files,
    describe,
)
from sensor_modeling.observations import Modality, ObservationKind

ROOT = Path(__file__).resolve().parents[1]
START = datetime(2019, 4, 1, 7, 0, 0)


def stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def write_dataset(
    directory: Path,
    activity: list[tuple[str, str, datetime]],
    labels: list[tuple[str, datetime, str]],
    participants: list[str],
) -> Path:
    """Write the three files the adapter reads, with the release's columns."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "Activity.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(["patient_id", "location_name", "date"])
        writer.writerows([p, loc, stamp(t)] for p, loc, t in activity)
    with (directory / "Labels.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(["patient_id", "date", "type"])
        writer.writerows([p, stamp(t), kind] for p, t, kind in labels)
    with (directory / "Demographics.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(["patient_id", "age", "sex"])
        writer.writerows([p, "(80, 90]", "Female"] for p in participants)
    return directory


@pytest.fixture()
def small(tmp_path: Path) -> Path:
    activity = [
        ("aaaaa", "Kitchen", START),
        ("aaaaa", "Fridge Door", START + timedelta(minutes=1)),
        ("aaaaa", "Front Door", START + timedelta(hours=2)),
        ("aaaaa", "Lounge", START + timedelta(hours=3)),
        ("bbbbb", "Bedroom", START + timedelta(minutes=5)),
        ("bbbbb", "Bathroom", START + timedelta(minutes=9)),
    ]
    labels = [
        ("aaaaa", START + timedelta(hours=5, minutes=1), "Agitation"),
        ("aaaaa", START + timedelta(hours=1), "Blood pressure"),
    ]
    return write_dataset(tmp_path / "Dataset", activity, labels, ["aaaaa", "bbbbb"])


class TestAdapter:
    def test_households_come_from_the_demographics_table(self, small: Path) -> None:
        adapter = TihmAdapter(small)
        assert adapter.households() == ("aaaaa", "bbbbb")
        assert adapter.demographics()["aaaaa"] == {"age": "(80, 90]", "sex": "Female"}

    def test_a_household_is_exposed_in_the_datasets_terms(self, small: Path) -> None:
        data = TihmAdapter(small).load("aaaaa")
        assert data.timezone == TIMEZONE
        assert data.residents is None
        assert [s.sensor_id for s in data.sensors] == [
            "Fridge Door",
            "Front Door",
            "Kitchen",
            "Lounge",
        ]
        assert {e.value for e in data.events} == {ACTIVE}
        assert len(data.events) == 4
        assert all(e.timestamp.tzinfo is None for e in data.events)

    def test_a_room_sensor_has_its_room_and_a_door_has_none(self) -> None:
        assert describe("Kitchen").sensor_type == PIR
        assert describe("Kitchen").location == "Kitchen"
        assert describe("Front Door").sensor_type == DOOR
        assert describe("Front Door").location is None
        assert describe("Fridge Door").sensor_type == FRIDGE_DOOR
        assert describe("Fridge Door").location is None

    def test_labels_are_point_annotations(self, small: Path) -> None:
        annotations = TihmAdapter(small).load("aaaaa").annotations
        assert [a.label for a in annotations] == ["Agitation", "Blood pressure"]
        assert all(a.semantics is AnnotationSemantics.POINT for a in annotations)
        assert all(a.end is None for a in annotations)

    def test_a_participant_without_labels_has_none(self, small: Path) -> None:
        assert TihmAdapter(small).load("bbbbb").annotations == ()

    def test_an_unknown_household_is_refused(self, small: Path) -> None:
        with pytest.raises(KeyError, match="no household"):
            TihmAdapter(small).load("zzzzz")

    def test_other_columns_are_refused(self, tmp_path: Path) -> None:
        directory = write_dataset(tmp_path / "Dataset", [], [], ["aaaaa"])
        (directory / "Activity.csv").write_text(
            "patient,location,time\n", encoding="utf-8"
        )
        with pytest.raises(ContractError, match="Activity.csv has columns"):
            TihmAdapter(directory).load("aaaaa")

    def test_an_unlisted_location_is_reported_not_guessed(self, tmp_path: Path) -> None:
        directory = write_dataset(
            tmp_path / "Dataset", [("aaaaa", "Garage", START)], [], ["aaaaa"]
        )
        data = TihmAdapter(directory).load("aaaaa")
        assert data.sensors[0].sensor_type == UNKNOWN_TYPE
        report = validate_household(data, TIHM_MAPPING)
        assert not report.ok
        assert "undeclared_sensor" in report.codes()

    def test_missing_and_altered_files_are_named(self, small: Path) -> None:
        problems = check_files(small)
        assert problems["Physiology.csv"] == "missing"
        assert problems["Activity.csv"] == "does not match its pinned digest"


class TestMapping:
    def test_every_native_value_is_declared(self) -> None:
        for label in LABELS:
            assert TIHM_MAPPING.label(label).status == Outcome.UNMAPPABLE.value
        for kind in set(LOCATIONS.values()):
            assert TIHM_MAPPING.sensor_type(kind).usable
        for location, kind in LOCATIONS.items():
            if kind == PIR:
                assert TIHM_MAPPING.location(location).status == Outcome.EXACT.value

    def test_sensor_types_map_to_event_modalities(self) -> None:
        expected = {
            PIR: Modality.MOTION,
            DOOR: Modality.DOOR,
            FRIDGE_DOOR: Modality.CONTACT,
        }
        for kind, modality in expected.items():
            semantics = TIHM_MAPPING.sensor_type(kind).target
            assert semantics.modality is modality
            assert semantics.kind is ObservationKind.EVENT
            assert semantics.activation == frozenset({ACTIVE})

    def test_no_label_maps_to_a_state(self) -> None:
        assert all(not TIHM_MAPPING.label(label).usable for label in LABELS)

    def test_the_frozen_mapping_file_is_the_declared_mapping(self) -> None:
        path = ROOT / "artifacts" / "tihm" / "tihm_mapping.json"
        frozen = OntologyMapping.read(path, TIHM_MAPPING.sha256())
        assert frozen.to_dict() == TIHM_MAPPING.to_dict()


class TestContract:
    def test_the_dataset_converts_and_scores_nothing(self, small: Path) -> None:
        adapter = TihmAdapter(small)
        report = validate_dataset(adapter, TIHM_MAPPING)
        assert report.ok
        totals = report.totals()
        assert totals["convertible"] == 2
        assert set(totals["issues"]) == {"point_annotations", "sensor_semantics"}
        canonical = to_canonical(adapter.load("aaaaa"), TIHM_MAPPING, source="tihm")
        assert canonical.dispositions["events"] == {"emitted": 4}
        assert canonical.dispositions["point_annotations"] == 2
        assert canonical.recording.activities == ()
        assert sum(canonical.dispositions["annotated_seconds"].values()) == 0

    def test_rooms_resolve_and_doors_have_none(self, small: Path) -> None:
        recording = to_canonical(
            TihmAdapter(small).load("aaaaa"), TIHM_MAPPING
        ).recording
        rooms = {spec.sensor_id: spec.room for spec in recording.registry}
        assert rooms == {
            "Fridge Door": None,
            "Front Door": None,
            "Kitchen": "kitchen",
            "Lounge": "living",
        }

    def test_observations_are_placed_in_the_declared_timezone(
        self, small: Path
    ) -> None:
        recording = to_canonical(
            TihmAdapter(small).load("aaaaa"), TIHM_MAPPING
        ).recording
        first = recording.observations[0].timestamp
        assert first.utcoffset() == timedelta(hours=1)
        assert first.replace(tzinfo=None) == START


class TestProvenance:
    def test_every_file_is_pinned_and_the_read_ones_are_named(self) -> None:
        assert set(PROVENANCE.files) == set(FILES)
        assert set(READ_FILES) <= set(FILES)
        assert "Physiology.csv" not in READ_FILES
        assert "Sleep.csv" not in READ_FILES

    def test_the_terms_ask_for_the_acknowledgement(self) -> None:
        assert "CC BY 4.0" in PROVENANCE.licence
        assert (
            "Surrey and Borders Partnership NHS Foundation Trust" in PROVENANCE.licence
        )
        assert "Howz" in PROVENANCE.licence
