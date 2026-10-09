"""The matched-sensor declaration: what it pins, and that it is what was frozen."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from sensor_modeling.datasets.matched_sensors_protocol import (
    HOLD_OFF_SECONDS,
    MATCHED,
    MATCHED_PRESENCE_SCALE,
    MATCHED_SPILL_RATE,
    PLANNING_RECORD,
    SENSITIVITY,
    STANDARD,
    THRESHOLD_PROTOCOL,
    THRESHOLD_PROTOCOL_SHA256,
    THRESHOLD_RECORD,
    THRESHOLD_RECORD_SHA256,
    TIHM_DAY_MEDIANS,
    TIHM_POST_HOC_RECORD,
    TIHM_POST_HOC_RECORD_SHA256,
    MatchedSensorsProtocol,
    check_frozen_protocol,
    code_changes,
    declared_protocol,
    file_sha256,
    pinned_sources,
    planned_choice,
)
from sensor_modeling.datasets.matched_sensors_summary import (
    PROTOCOL_PAGE,
    RESULTS_PAGE,
    render_page,
    render_protocol,
)
from sensor_modeling.simulation.sensor_profile import STANDARD_PROFILE

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/matched_sensors/matched_sensors_protocol.json"
RECORD = ROOT / "artifacts/matched_sensors/matched-sensors.json"
PLANNING = ROOT / PLANNING_RECORD


class TestWhatItPins:
    @pytest.mark.parametrize(
        ("path", "digest"),
        [
            (THRESHOLD_PROTOCOL, THRESHOLD_PROTOCOL_SHA256),
            (THRESHOLD_RECORD, THRESHOLD_RECORD_SHA256),
            (TIHM_POST_HOC_RECORD, TIHM_POST_HOC_RECORD_SHA256),
        ],
    )
    def test_each_pinned_record_is_the_file_in_the_repository(
        self, path: str, digest: str
    ) -> None:
        assert file_sha256(ROOT / path) == digest

    def test_the_tihm_medians_are_the_post_hoc_records(self) -> None:
        hours = json.loads((ROOT / TIHM_POST_HOC_RECORD).read_text("utf-8"))["results"][
            "state_hours"
        ]["hours"]
        for feature, quoted in TIHM_DAY_MEDIANS.items():
            state = feature.removesuffix("_hours")
            assert round(hours[state]["median"], 4) == quoted

    def test_the_scripts_that_freeze_and_run_it_are_pinned(self) -> None:
        sources = pinned_sources(ROOT)
        assert "scripts/freeze_matched_sensors.py" in sources
        assert "scripts/run_matched_sensors.py" in sources
        assert "sensor_modeling/simulation/sensor_profile.py" in sources


@pytest.mark.skipif(not PLANNING.exists(), reason="the planning has not been run")
class TestThePlanning:
    def planning(self) -> dict:
        return json.loads(PLANNING.read_text(encoding="utf-8"))

    def test_the_planning_record_is_the_pinned_file(self) -> None:
        assert file_sha256(PLANNING) == declared_protocol().planning_record_sha256

    def test_the_profiles_are_the_planning_records_choice(self) -> None:
        planning = self.planning()
        chosen = planning["results"]["chosen"]
        assert chosen["presence_scale"] == MATCHED_PRESENCE_SCALE
        assert chosen["spill_rate"] == MATCHED_SPILL_RATE
        assert chosen["hold_off_seconds"] == HOLD_OFF_SECONDS
        assert not chosen["at_the_edge_of_the_grid"]
        protocol = declared_protocol()
        for name, point in planned_choice(planning).items():
            profile = protocol.profile(name)
            assert (profile.presence_scale, profile.spill_rate) == point

    def test_the_planning_was_made_from_a_clean_commit(self) -> None:
        assert self.planning()["environment"]["git_dirty"] == "false"


class TestTheDeclaration:
    def test_it_serialises_without_a_missing_number(self) -> None:
        protocol = declared_protocol()
        json.dumps(protocol.to_dict(), allow_nan=False)
        assert protocol.sha256() == declared_protocol().sha256()

    def test_the_profiles(self) -> None:
        protocol = declared_protocol()
        assert protocol.profile(STANDARD) == STANDARD_PROFILE
        matched = protocol.profile(MATCHED)
        assert matched.hallway and matched.paired_contacts
        assert matched.hold_off_seconds == HOLD_OFF_SECONDS
        assert protocol.profile(SENSITIVITY) != matched
        assert protocol.sensitivity_seeds() == protocol.study.study_seeds()[:100]
        assert protocol.stream("stable") != protocol.stream("change")
        with pytest.raises(KeyError):
            protocol.profile("other")

    @pytest.mark.parametrize(
        "change",
        [
            {"survival_margin": 0.0},
            {"tracking_margin": 1.0},
            {"kitchen_ceiling_hours": 0.0},
            {"min_matched_days": 2},
            {"confidence": 1.0},
        ],
    )
    def test_a_declaration_that_cannot_be_run_is_refused(self, change: dict) -> None:
        with pytest.raises(ValueError):
            dataclasses.replace(MatchedSensorsProtocol(), **change)


@pytest.mark.skipif(not FROZEN.exists(), reason="the protocol is not frozen yet")
class TestTheFrozenProtocol:
    def test_the_frozen_file_is_what_the_code_declares(self) -> None:
        check_frozen_protocol(declared_protocol(), FROZEN)

    def test_a_changed_declaration_is_refused(self) -> None:
        changed = dataclasses.replace(declared_protocol(), survival_margin=0.2)
        with pytest.raises(ValueError, match="cannot change"):
            check_frozen_protocol(changed, FROZEN)

    def test_the_page_is_the_frozen_files(self) -> None:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        page = (ROOT / "docs" / PROTOCOL_PAGE).read_text(encoding="utf-8")
        assert page == render_protocol(frozen)

    def test_the_freeze_recorded_the_code(self) -> None:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        assert set(frozen["at_freeze"]) == {"sources", "distributions", "defaults"}
        assert set(code_changes(FROZEN, ROOT)) == {
            "sources",
            "distributions",
            "defaults",
        }


@pytest.mark.skipif(not RECORD.exists(), reason="the protocol has not been run yet")
class TestThePublishedRun:
    def record(self) -> dict:
        return json.loads(RECORD.read_text(encoding="utf-8"))

    def test_the_record_comes_from_the_frozen_protocol_unchanged(self) -> None:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        record = self.record()
        assert record["environment"]["git_dirty"] == "false"
        assert record["configuration"]["protocol_sha256"] == frozen["protocol_sha256"]
        assert not any(
            record["configuration"]["code_changed_since_the_freeze"].values()
        )

    def test_the_standard_profile_reproduced_the_published_run(self) -> None:
        check = self.record()["results"]["check"]
        assert check["reproduced"]
        assert check["runs_compared"] == 2 * declared_protocol().study.homes

    def test_the_page_is_the_records(self) -> None:
        page = (ROOT / "docs" / RESULTS_PAGE).read_text(encoding="utf-8")
        assert page == render_page(self.record())
