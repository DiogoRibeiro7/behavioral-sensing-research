"""Tests for the silent-home protocol's declaration.

The protocol is frozen before any simulated home is run with the rule on, so
these tests check the declaration alone: that the frozen file is what the code
declares, that its page is generated from it, and that the homes, the outages
and the windows it fixes are reproducible and fit the record.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from sensor_modeling.datasets.silent_home_protocol import (
    CHANGE,
    CHANGE_AFTER_OUTAGE,
    COMMON,
    FLEET_ARMS,
    IN_TIME,
    LATE,
    OFF,
    OUTAGE,
    PIPELINE_ARMS,
    SHORT_OUTAGE,
    STABLE,
    SilentHomeProtocol,
    check_frozen_protocol,
    condition_name,
    declared_protocol,
    event_sensors,
    write_protocol,
)
from sensor_modeling.datasets.silent_home_summary import (
    PROTOCOL_FILE,
    PROTOCOL_PAGE,
    render_protocol,
)
from sensor_modeling.simulation.household import build_registry

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / PROTOCOL_FILE
LISBON = ZoneInfo("Europe/Lisbon")


class TestFrozenProtocol:
    def test_the_frozen_file_is_the_declared_protocol(self) -> None:
        digest = check_frozen_protocol(declared_protocol(), FROZEN)
        assert len(digest) == 64

    def test_the_protocol_page_is_generated_from_the_frozen_file(self) -> None:
        payload = json.loads(FROZEN.read_text(encoding="utf-8"))
        page = ROOT / "docs" / PROTOCOL_PAGE
        assert page.read_text(encoding="utf-8") == render_protocol(payload)

    @pytest.mark.parametrize(
        "changed",
        [
            {"horizons_hours": (18.0, 24.0)},
            {"margin_alerts": 0.5},
            {"homes": 120},
            {"follow_days": 14},
        ],
    )
    def test_a_changed_protocol_is_refused(self, changed: dict) -> None:
        with pytest.raises(ValueError, match="differs from the one this code declares"):
            check_frozen_protocol(SilentHomeProtocol(**changed), FROZEN)

    def test_writing_it_round_trips(self, tmp_path: Path) -> None:
        protocol = SilentHomeProtocol(resamples=300)
        path = tmp_path / "protocol.json"
        digest = write_protocol(protocol, path)
        assert digest == protocol.sha256()
        assert json.loads(path.read_text(encoding="utf-8"))["protocol_sha256"] == digest
        check_frozen_protocol(protocol, path)
        with pytest.raises(ValueError):
            check_frozen_protocol(declared_protocol(), path)

    def test_it_says_what_it_is_and_what_had_been_seen(self) -> None:
        declared = declared_protocol().to_dict()
        assert declared["status"].startswith("pre-specified on simulated homes")
        assert "Nothing on TIHM is a test" in declared["status"]
        assert any(
            "No simulated household had been run with the rule on" in item
            for item in declared["inspected_before"]
        )
        assert declared["simulator"]["evidence"].startswith("simulated")
        assert declared["tihm"]["standing"].startswith("a description, not a test")
        assert declared["the_rule"]["default"] == "off"
        assert declared["bootstrap"]["unit"] == "homes"
        assert len(declared["what_the_simulator_cannot_show"]) == 3

    def test_every_criterion_names_an_estimand_that_is_declared(self) -> None:
        declared = declared_protocol().to_dict()
        assert list(declared["estimands"]) == [
            "E1",
            "E2",
            "E3",
            "E2_by_group",
            "E4",
            "E5",
            "E6",
            "E7",
            "E8",
            "E9",
            "S1",
        ]
        assert [name.split("_")[0] for name in declared["criteria"]] == [
            f"C{k}" for k in range(1, 9)
        ]


class TestHomes:
    def test_the_seeds_come_from_the_root_and_are_distinct(self) -> None:
        protocol = declared_protocol()
        seeds = protocol.study_seeds()
        assert len(seeds) == len(set(seeds)) == protocol.homes == 100
        assert seeds == tuple(sorted(seeds))
        assert seeds == declared_protocol().study_seeds()
        assert SilentHomeProtocol(seed_root=1).study_seeds() != seeds
        # The household the run time was measured on is not one of them.
        assert 999 not in seeds

    def test_the_homes_keep_only_sensors_that_promise_nothing(self) -> None:
        registry = build_registry()
        kept = event_sensors()
        assert kept
        assert all(registry.get(name).expected_interval is None for name in kept)
        left_out = {spec.sensor_id for spec in registry} - set(kept)
        assert left_out == set(
            declared_protocol().to_dict()["simulator"]["sensors_left_out"]
        )
        assert all(
            registry.get(name).expected_interval is not None for name in left_out
        )

    def test_every_home_has_an_outage_of_its_own(self) -> None:
        protocol = declared_protocol()
        starts = protocol.outage_starts()
        assert set(starts) == set(protocol.study_seeds())
        assert starts == declared_protocol().outage_starts()
        for day, hour in starts.values():
            assert protocol.outage_first_day <= day <= protocol.outage_last_day
            assert 0 <= hour <= 23
        assert len(set(starts.values())) > protocol.homes // 2
        listed = protocol.to_dict()["simulator"]["outages"]
        assert [(home["seed"], home["day"], home["hour"]) for home in listed] == [
            (seed, *starts[seed]) for seed in protocol.study_seeds()
        ]

    def test_both_groups_are_in_the_homes(self) -> None:
        protocol = declared_protocol()
        groups = [protocol.group(seed) for seed in protocol.study_seeds()]
        assert groups.count(IN_TIME) + groups.count(LATE) == protocol.homes
        assert groups.count(IN_TIME) >= 30
        assert groups.count(LATE) >= 30
        for seed in protocol.study_seeds():
            _, hour = protocol.outage_starts()[seed]
            in_time = hour + protocol.primary_hours < 24
            assert protocol.group(seed) == (IN_TIME if in_time else LATE)


class TestWindows:
    def test_an_outage_begins_at_its_local_hour(self) -> None:
        protocol = declared_protocol()
        for seed in protocol.study_seeds()[:10]:
            day, hour = protocol.outage_starts()[seed]
            begin, end = protocol.outage(seed)
            local = begin.astimezone(LISBON)
            assert local.date() == protocol.start() + timedelta(days=day)
            assert (local.hour, local.minute) == (hour, 0)
            assert end - begin == timedelta(hours=protocol.outage_hours)
            short_begin, short_end = protocol.short_outage(seed)
            assert short_begin == begin
            assert short_end - short_begin == timedelta(
                hours=protocol.short_outage_hours
            )

    def test_the_alerts_of_an_outage_are_counted_for_four_weeks_after_it(self) -> None:
        protocol = declared_protocol()
        seed = protocol.study_seeds()[0]
        for arm, (begin, end) in (
            (OUTAGE, protocol.outage(seed)),
            (SHORT_OUTAGE, protocol.short_outage(seed)),
        ):
            assert protocol.window(seed, arm) == (
                begin,
                end + timedelta(days=protocol.follow_days),
            )
        with pytest.raises(KeyError):
            protocol.window(seed, STABLE)

    def test_every_window_fits_the_record(self) -> None:
        protocol = declared_protocol()
        record_ends = protocol.local(protocol.days, 0.0)
        for seed in protocol.study_seeds():
            assert protocol.window(seed, OUTAGE)[1] <= record_ends
            # The change follows every outage by at least a week.
            assert protocol.change_begins() - protocol.outage(seed)[1] >= timedelta(
                days=7
            )
        assert protocol.common_outage()[1] <= record_ends
        assert protocol.detection_window()[1] <= record_ends

    def test_no_outage_falls_on_the_day_the_clocks_change(self) -> None:
        protocol = declared_protocol()
        first = protocol.local(protocol.outage_first_day, 0.0).astimezone(LISBON)
        last = protocol.local(protocol.days, 0.0).astimezone(LISBON)
        assert first.utcoffset() == last.utcoffset() == timedelta(hours=1)

    def test_a_detection_needs_a_day_of_the_change(self) -> None:
        protocol = declared_protocol()
        begin, end = protocol.detection_window()
        # The change day closes a day after it begins; the alert raised at the
        # close of the day before the change is outside the window.
        assert begin - protocol.change_begins() == timedelta(days=1)
        assert end - begin == timedelta(days=protocol.max_delay_days)

    def test_the_common_outage_is_one_window_for_every_home(self) -> None:
        protocol = declared_protocol()
        begin, end = protocol.common_outage()
        local = begin.astimezone(LISBON)
        assert local.date() == protocol.start() + timedelta(days=protocol.common_day)
        assert local.hour == protocol.common_hour
        assert end - begin == timedelta(hours=protocol.outage_hours)


class TestRuns:
    def test_every_arm_is_run_off_and_at_the_primary_horizon(self) -> None:
        protocol = declared_protocol()
        primary = condition_name(protocol.primary_hours)
        assert primary == "h12"
        runs = protocol.runs()
        assert len(runs) == len(set(runs)) == 12
        for arm in PIPELINE_ARMS:
            assert (arm, OFF) in runs
            assert (arm, primary) in runs
        others = {(arm, c) for arm, c in runs if c not in (OFF, primary)}
        assert others == {(STABLE, "h24"), (OUTAGE, "h24")}

    def test_the_common_outage_is_read_by_the_fleet_check_alone(self) -> None:
        protocol = declared_protocol()
        assert COMMON not in PIPELINE_ARMS
        assert COMMON in FLEET_ARMS
        assert all(arm != COMMON for arm, _ in protocol.runs())
        assert {CHANGE, CHANGE_AFTER_OUTAGE}.isdisjoint(FLEET_ARMS)

    def test_a_condition_gives_its_horizon(self) -> None:
        protocol = declared_protocol()
        assert protocol.conditions == (OFF, "h12", "h24")
        assert protocol.horizon(OFF) is None
        assert protocol.horizon("h12") == timedelta(hours=12)
        assert protocol.horizon("h24") == timedelta(hours=24)
        with pytest.raises(KeyError):
            protocol.horizon("h6")


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"homes": 1},
            {"horizons_hours": ()},
            {"horizons_hours": (12.0, 12.0)},
            {"horizons_hours": (-1.0,)},
            {"short_outage_hours": 12.0},
            {"outage_hours": 24.0},
            {"outage_first_day": 50, "outage_last_day": 40},
            {"common_hour": 24},
            {"change_day": 47},
            {"days": 70},
        ],
    )
    def test_a_declaration_that_cannot_be_run_is_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            SilentHomeProtocol(**kwargs)
