"""Tests for the versioned experiment-artefact schema.

An artefact has to stay readable and exact long after the code that wrote it
has moved on. These tests pin the three properties that make that true: every
file is strict, deterministic JSON; a malformed or foreign file is refused with
a reason; and a file from an older schema version is migrated, not guessed at.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.evaluation.provenance import (
    SCHEMA_VERSION,
    SIMULATOR_NOTE,
    ArtifactError,
    ExperimentRecord,
    InputArtifact,
    ModelRecord,
    ReportedInterval,
    json_safe,
    load_record,
    validate_record,
)

DIGEST = "a" * 64


def full_record(**overrides: Any) -> ExperimentRecord:
    """A record using every section of the schema."""
    fields: dict[str, Any] = {
        "experiment": "unit",
        "configuration": {"step_minutes": 5, "metrics": ["balanced_accuracy"]},
        "seeds": [0, 1],
        "results": {"summary": {"balanced_accuracy": {"median": 0.5}}},
        "data_source": "casas-hh",
        "inputs": [InputArtifact("hh101.csv", DIGEST, "recording", "zenodo:15708568")],
        "split": {"name": "fold-a", "train": ["hh101"], "test": ["hh102"]},
        "information_set": {"name": "current", "history_steps": 3},
        "preprocessing": {"timezone": "America/Los_Angeles", "step_seconds": 300},
        "models": [ModelRecord("tree", {"max_depth": 6}, "module.build_tree")],
        "household_metrics": {"tree": {"hh102": {"balanced_accuracy": 0.5}}},
        "intervals": [
            ReportedInterval(
                "tree vs prior", 0.1, 0.05, 0.15, 0.95, "percentile", "household", 20
            )
        ],
        "mcse": {"mean paired difference": 0.002},
        "recorded_at": "2026-09-26T08:00:00+00:00",
        "environment": {
            "git_commit": "0" * 40,
            "git_dirty": "false",
            "python": "3.13.5",
            "sensor_modeling": "0.6.0",
        },
        "resolved_defaults": {"pipeline": {"step": "0:05:00"}},
    }
    fields.update(overrides)
    return ExperimentRecord(**fields)


def payload(**overrides: Any) -> dict[str, Any]:
    data = full_record().to_dict()
    data.update(overrides)
    return data


def write_json(path: Path, data: Any, *, allow_nan: bool = False) -> Path:
    path.write_text(json.dumps(data, allow_nan=allow_nan), encoding="utf-8")
    return path


#: A record exactly as the 1.0 writer produced it, including the NaN token the
#: old writer allowed and no data_source, which 1.0 lacked before 0.6.0.
LEGACY_1_0 = """{
  "experiment": "ablation_study",
  "schema_version": "1.0",
  "recorded_at": "2026-08-29T10:00:00+00:00",
  "environment": {"python": "3.12.0", "platform": "Linux", "numpy": "2.1",
                  "sensor_modeling": "0.3.0", "git_commit": "abc", "git_dirty": "false"},
  "configuration": {"days": 14, "step_minutes": 10},
  "resolved_defaults": {"pipeline": {"step": "0:05:00"}},
  "seeds": [11, 22],
  "sensor_subset": null,
  "metric_definitions": {"balanced_accuracy": "Unweighted mean of per-class recall."},
  "results": {"summary": {"median_delay_days": NaN, "balanced_accuracy": 0.81}},
  "notes": ["Generated from the bundled simulator. Not validated against real sensor data; see docs/limitations.md."]
}"""


class TestRoundTrip:
    def test_every_section_survives_a_write_and_a_load(self, tmp_path: Path) -> None:
        record = full_record()
        loaded = ExperimentRecord.load(record.write(tmp_path / "run.json"))
        assert loaded.to_dict() == record.to_dict()
        assert loaded.inputs == record.inputs
        assert loaded.models == record.models
        assert loaded.intervals == record.intervals

    def test_rewriting_a_loaded_record_gives_the_same_bytes(
        self, tmp_path: Path
    ) -> None:
        first = full_record().write(tmp_path / "a.json")
        second = ExperimentRecord.load(first).write(tmp_path / "b.json")
        assert first.read_bytes() == second.read_bytes()

    def test_the_simulator_note_is_not_duplicated_on_reload(
        self, tmp_path: Path
    ) -> None:
        record = full_record(data_source="simulator")
        loaded = ExperimentRecord.load(record.write(tmp_path / "sim.json"))
        assert loaded.notes.count(SIMULATOR_NOTE) == 1


class TestDeterminism:
    def test_the_same_record_writes_the_same_bytes(self, tmp_path: Path) -> None:
        record = full_record()
        assert record.write(tmp_path / "a.json").read_bytes() == (
            record.write(tmp_path / "b.json").read_bytes()
        )

    def test_key_order_does_not_change_the_file(self) -> None:
        forward = full_record(configuration={"a": 1, "b": 2})
        backward = full_record(configuration={"b": 2, "a": 1})
        assert forward.to_json() == backward.to_json()

    def test_files_are_strict_json_with_lf_endings(self, tmp_path: Path) -> None:
        raw = full_record().write(tmp_path / "run.json").read_bytes()
        assert b"\r\n" not in raw and raw.endswith(b"}\n")

    def test_time_and_environment_are_fixed_when_the_record_is_made(self) -> None:
        record = ExperimentRecord(experiment="x", configuration={})
        assert record.to_dict()["recorded_at"] == record.to_dict()["recorded_at"]
        assert record.environment["git_commit"]


class TestNonFiniteNumbers:
    def test_nan_and_infinity_are_written_as_null(self, tmp_path: Path) -> None:
        record = full_record(
            results={"nan": math.nan, "inf": math.inf, "neg": -math.inf, "ok": 1.5},
            mcse={"a": 0.1},
            intervals=[
                ReportedInterval("x", math.nan, 0.0, 1.0, 0.9, "percentile", "seed")
            ],
        )
        path = record.write(tmp_path / "run.json")

        def refuse(token: str) -> None:
            raise AssertionError(f"wrote non-standard JSON token {token}")

        written = json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)
        assert written["results"] == {"nan": None, "inf": None, "neg": None, "ok": 1.5}
        assert written["intervals"][0]["estimate"] is None

    def test_numpy_values_become_plain_json(self) -> None:
        converted = json_safe(
            {"f": np.float64(np.nan), "i": np.int64(3), "a": np.array([1.0, np.inf])}
        )
        assert converted == {"f": None, "i": 3, "a": [1.0, None]}
        assert isinstance(converted["i"], int)

    def test_a_value_json_cannot_hold_is_refused_with_its_location(self) -> None:
        with pytest.raises(TypeError, match=r"\$\.results\.bad"):
            json_safe({"results": {"bad": object()}})

    def test_a_current_file_containing_nan_is_refused(self, tmp_path: Path) -> None:
        path = write_json(
            tmp_path / "nan.json", payload(results={"x": math.nan}), allow_nan=True
        )
        with pytest.raises(ArtifactError, match="strict JSON"):
            load_record(path)


class TestValidation:
    @pytest.mark.parametrize(
        "key",
        [
            "schema_version",
            "experiment",
            "recorded_at",
            "environment",
            "data_source",
            "configuration",
            "seeds",
            "metric_definitions",
            "results",
            "notes",
            "household_metrics",
            "intervals",
        ],
    )
    def test_a_missing_required_field_is_named(self, key: str) -> None:
        data = payload()
        del data[key]
        with pytest.raises(ArtifactError, match=key):
            validate_record(data)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"surprise": 1}, "unknown field 'surprise'"),
            ({"seeds": ["0"]}, "seeds must be integers"),
            ({"experiment": " "}, "experiment is empty"),
            ({"recorded_at": "2026-09-26T08:00:00"}, "no time zone"),
            ({"recorded_at": "yesterday"}, "not an ISO 8601"),
            ({"environment": {"python": "3.13"}}, "environment lacks git_commit"),
            (
                {"inputs": [{"name": "a", "sha256": "xyz", "role": "r", "source": ""}]},
                "not a lower-case SHA-256",
            ),
            (
                {
                    "intervals": [
                        ReportedInterval("x", 0.0, 1.0, 0.0, 0.9, "p", "seed").to_dict()
                    ]
                },
                "low above high",
            ),
            (
                {
                    "intervals": [
                        ReportedInterval("x", 0.0, 0.0, 1.0, 1.5, "p", "seed").to_dict()
                    ]
                },
                r"confidence must lie in \(0, 1\)",
            ),
            ({"mcse": {"a": -0.1}}, "non-negative"),
            (
                {
                    "models": [
                        ModelRecord("m", {}).to_dict(),
                        ModelRecord("m", {}).to_dict(),
                    ]
                },
                "model names must be unique",
            ),
            (
                {"household_metrics": {"m": {"h": 0.5}}},
                "must map households to metrics",
            ),
            ({"results": {"x": math.inf}}, "not a finite number"),
        ],
    )
    def test_an_invalid_field_is_refused_with_a_reason(
        self, change: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ArtifactError, match=message):
            validate_record(payload(**change))

    def test_every_problem_is_reported_at_once(self) -> None:
        with pytest.raises(ArtifactError) as error:
            validate_record(payload(surprise=1, seeds=["0"], experiment=" "))
        text = str(error.value)
        assert "surprise" in text and "seeds" in text and "experiment" in text

    def test_an_invalid_record_is_never_written(self, tmp_path: Path) -> None:
        bad = full_record(
            intervals=[ReportedInterval("x", 0.0, 1.0, 0.0, 0.9, "p", "seed")]
        )
        with pytest.raises(ArtifactError, match="low above high"):
            bad.write(tmp_path / "bad.json")
        assert not (tmp_path / "bad.json").exists()

    def test_non_json_and_non_object_files_are_refused(self, tmp_path: Path) -> None:
        text = tmp_path / "text.json"
        text.write_text("not json", encoding="utf-8")
        with pytest.raises(ArtifactError, match="not JSON"):
            load_record(text)
        with pytest.raises(ArtifactError, match="not a JSON object"):
            load_record(write_json(tmp_path / "list.json", [1, 2]))


class TestSchemaVersions:
    def test_the_writer_emits_the_current_version(self) -> None:
        assert full_record().to_dict()["schema_version"] == SCHEMA_VERSION == "1.1"

    def test_a_1_0_record_is_migrated_not_rewritten(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.json"
        path.write_text(LEGACY_1_0, encoding="utf-8")
        migrated = load_record(path)
        assert migrated["schema_version"] == SCHEMA_VERSION
        assert migrated["migrated_from"] == "1.0"
        assert migrated["data_source"] == "simulator"
        assert migrated["results"] == {
            "summary": {"median_delay_days": None, "balanced_accuracy": 0.81}
        }
        assert migrated["inputs"] == [] and migrated["household_metrics"] == {}
        assert ExperimentRecord.from_dict(migrated).migrated_from == "1.0"

    def test_a_1_0_record_without_the_simulator_note_has_an_unknown_source(
        self, tmp_path: Path
    ) -> None:
        legacy = json.loads(LEGACY_1_0.replace("NaN", "null"))
        legacy["notes"] = []
        assert load_record(write_json(tmp_path / "old.json", legacy))[
            "data_source"
        ] == ("unknown")

    @pytest.mark.parametrize(
        ("version", "message"),
        [
            ("1.2", "newer than 1.1"),
            ("2.0", "reads major version 1 only"),
            ("0.9", "reads major version 1 only"),
            ("one", "not MAJOR.MINOR"),
            (1.1, "not MAJOR.MINOR"),
        ],
    )
    def test_unreadable_versions_are_refused(
        self, tmp_path: Path, version: Any, message: str
    ) -> None:
        path = write_json(tmp_path / "v.json", payload(schema_version=version))
        with pytest.raises(ArtifactError, match=message):
            load_record(path)

    def test_a_file_without_a_version_is_refused(self, tmp_path: Path) -> None:
        data = payload()
        del data["schema_version"]
        with pytest.raises(ArtifactError, match="no schema_version"):
            load_record(write_json(tmp_path / "none.json", data))

    def test_only_current_payloads_can_be_rebuilt_directly(self) -> None:
        with pytest.raises(ArtifactError, match="expected '1.1'"):
            ExperimentRecord.from_dict(payload(schema_version="1.0"))
