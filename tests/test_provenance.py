"""Tests that experiment artefacts are interpretable without the code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sensor_modeling.evaluation import (
    METRIC_DEFINITIONS,
    ExperimentRecord,
    environment,
    load_record,
)


def record(**overrides: object) -> ExperimentRecord:
    """Build a record with selective overrides."""
    payload: dict[str, object] = {
        "experiment": "example",
        "configuration": {"days": 10, "step_minutes": 15},
        "seeds": [1, 2, 3],
        "results": {"balanced_accuracy": 0.81},
    }
    payload.update(overrides)
    return ExperimentRecord(**payload)  # type: ignore[arg-type]


class TestSelfDescription:
    def test_an_artefact_carries_everything_needed_to_repeat_it(self) -> None:
        payload = record().to_dict()
        assert payload["configuration"] == {"days": 10, "step_minutes": 15}
        assert payload["seeds"] == [1, 2, 3]
        assert payload["environment"]["python"]
        assert payload["recorded_at"]

    def test_every_reported_metric_has_a_written_definition(self) -> None:
        """A bare number is nearly worthless six months later."""
        payload = record().to_dict()
        definitions = payload["metric_definitions"]
        assert "balanced_accuracy" in definitions  # type: ignore[operator]
        assert "abstention" in definitions["balanced_accuracy"] or len(  # type: ignore[index]
            definitions["balanced_accuracy"]  # type: ignore[index]
        )

    def test_the_abstention_convention_is_stated_not_implied(self) -> None:
        assert "UNKNOWN is never correct" in METRIC_DEFINITIONS["accuracy"]

    def test_the_pairing_caveat_travels_with_the_effect_size(self) -> None:
        assert "unpaired" in METRIC_DEFINITIONS["effect_size"]

    def test_every_artefact_records_that_it_came_from_a_simulator(self) -> None:
        notes = record().to_dict()["notes"]
        assert any("simulator" in note for note in notes)  # type: ignore[union-attr]

    def test_library_versions_are_captured(self) -> None:
        captured = environment()
        assert set(captured) >= {"python", "platform", "numpy", "sensor_modeling"}

    def test_a_record_needs_a_name(self) -> None:
        with pytest.raises(ValueError, match="needs a name"):
            record(experiment="  ")

    def test_a_sensor_subset_is_recorded_when_it_varies(self) -> None:
        payload = record(sensor_subset=["door", "bed"]).to_dict()
        assert payload["sensor_subset"] == ["door", "bed"]

    def test_no_subset_is_recorded_as_such(self) -> None:
        assert record().to_dict()["sensor_subset"] is None


class TestWriteAndLoad:
    def test_writing_creates_the_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "run.json"
        assert record().write(target) == target
        assert target.exists()

    def test_a_written_artefact_round_trips(self, tmp_path: Path) -> None:
        target = record().write(tmp_path / "run.json")
        loaded = load_record(target)
        assert loaded["experiment"] == "example"
        assert loaded["results"]["balanced_accuracy"] == 0.81

    def test_the_results_are_stable_across_writes(self, tmp_path: Path) -> None:
        """The timestamp differs; the findings must not."""
        first = load_record(record().write(tmp_path / "a.json"))
        second = load_record(record().write(tmp_path / "b.json"))
        assert first["results"] == second["results"]
        assert first["configuration"] == second["configuration"]

    def test_an_artefact_without_provenance_is_refused(self, tmp_path: Path) -> None:
        bare = tmp_path / "bare.json"
        bare.write_text(json.dumps({"balanced_accuracy": 0.81}), encoding="utf-8")
        with pytest.raises(ValueError, match="missing provenance fields"):
            load_record(bare)

    def test_the_artefact_is_json_serialisable(self, tmp_path: Path) -> None:
        target = record().write(tmp_path / "run.json")
        assert json.loads(target.read_text(encoding="utf-8"))
