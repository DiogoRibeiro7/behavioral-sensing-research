"""The matched simulator's sensor profile and the moments it is chosen on."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from sensor_modeling.datasets.matched_sensors_planning import (
    CONTACT,
    MOMENTS,
    Planning,
    cohort_moments,
    distance,
    home_moments,
    sensor_facts,
)
from sensor_modeling.datasets.silent_home_protocol import event_sensors
from sensor_modeling.datasets.threshold_calibration_protocol import (
    declared_protocol as threshold_protocol,
)
from sensor_modeling.observations.types import Modality
from sensor_modeling.simulation.household import HouseholdConfig, simulate
from sensor_modeling.simulation.sensor_profile import (
    FRIDGE,
    FRONT_DOOR,
    HALLWAY,
    STANDARD_PROFILE,
    SensorProfile,
    hold_off,
    profile_observations,
    profile_registry,
)

UTC = timezone.utc
T0 = datetime(2024, 3, 4, tzinfo=UTC)


@pytest.fixture(scope="module")
def home():
    return simulate(HouseholdConfig(days=5, seed=4242))


MATCHED = SensorProfile(
    hold_off_seconds=61.0,
    presence_scale=1.0,
    spill_rate=3.0,
    hallway=True,
    paired_contacts=True,
)


class TestTheProfile:
    def test_no_motion_sensor_reports_inside_its_hold_off(self, home) -> None:
        records = profile_observations(home.truth, 4242, MATCHED)
        last: dict[str, datetime] = {}
        for obs in records:
            if obs.modality is not Modality.MOTION:
                continue
            if obs.sensor_id in last:
                gap = (obs.timestamp - last[obs.sensor_id]).total_seconds()
                assert gap >= 61.0
            last[obs.sensor_id] = obs.timestamp

    def test_the_hold_off_restarts_only_at_a_reported_activation(self) -> None:
        moments = [T0 + timedelta(seconds=s) for s in (0, 40, 70, 100, 135)]
        kept = hold_off([(m, "k", "resident") for m in moments], 61.0)
        assert [m for m, _, _ in kept] == [moments[0], moments[2], moments[4]]

    def test_paired_contacts_record_each_activation_twice(self, home) -> None:
        single = SensorProfile(
            hold_off_seconds=61.0, presence_scale=1.0, spill_rate=3.0, hallway=True
        )
        once = [
            o
            for o in profile_observations(home.truth, 4242, single)
            if o.sensor_id in (FRIDGE, FRONT_DOOR)
        ]
        twice = [
            o
            for o in profile_observations(home.truth, 4242, MATCHED)
            if o.sensor_id in (FRIDGE, FRONT_DOOR)
        ]
        assert len(twice) == 2 * len(once)

    def test_the_hallway_reports_at_every_change_of_room(self, home) -> None:
        quiet = SensorProfile(spill_rate=0.0, hallway=True)
        records = profile_observations(home.truth, 4242, quiet)
        hall = {o.timestamp for o in records if o.sensor_id == HALLWAY}
        rooms = []
        for episode in home.truth.episodes:
            where = episode.room if episode.room is not None else None
            if episode.state.value == "away":
                where = None
            rooms.append((episode.start, where))
        changes = {
            start
            for (_, before), (start, after) in zip(rooms, rooms[1:])
            if before != after
        }
        assert changes <= hall

    def test_the_hallway_does_not_report_at_the_first_instant(self, home) -> None:
        quiet = SensorProfile(spill_rate=0.0, hallway=True)
        records = profile_observations(home.truth, 4242, quiet)
        first = home.truth.episodes[0].start
        assert not any(o.sensor_id == HALLWAY and o.timestamp == first for o in records)

    def test_two_streams_share_no_sensor_noise(self, home) -> None:
        one = profile_observations(home.truth, 4242, MATCHED, stream=0)
        two = profile_observations(home.truth, 4242, MATCHED, stream=1)
        moments = {o.timestamp for o in one if o.modality is Modality.MOTION}
        shared = [
            o for o in two if o.modality is Modality.MOTION and o.timestamp in moments
        ]
        hall_changes = [o for o in shared if o.sensor_id == HALLWAY]
        assert len(shared) == len(hall_changes)

    def test_the_standard_profile_has_no_hallway(self, home) -> None:
        records = profile_observations(home.truth, 4242, STANDARD_PROFILE)
        assert HALLWAY not in {o.sensor_id for o in records}
        registry = home.registry.subset(event_sensors())
        assert HALLWAY not in profile_registry(registry, STANDARD_PROFILE)

    def test_the_registry_gains_the_hallway_once(self, home) -> None:
        registry = home.registry.subset(event_sensors())
        once = profile_registry(registry, MATCHED)
        assert HALLWAY in once
        assert len(profile_registry(once, MATCHED)) == len(once)

    def test_a_record_is_a_function_of_plan_seed_and_profile(self, home) -> None:
        first = profile_observations(home.truth, 4242, MATCHED)
        second = profile_observations(home.truth, 4242, MATCHED)
        other = profile_observations(home.truth, 4243, MATCHED)
        assert first == second
        assert first != other

    @pytest.mark.parametrize(
        "change",
        [
            {"hold_off_seconds": -1.0},
            {"presence_scale": 0.0},
            {"spill_rate": -0.5},
        ],
    )
    def test_an_impossible_profile_is_refused(self, change: dict) -> None:
        with pytest.raises(ValueError):
            SensorProfile(**change)


def _record(spec: list[tuple[float, str, str]]):
    return [(T0 + timedelta(hours=h), sensor, role) for h, sensor, role in spec]


class TestTheMoments:
    def test_the_first_and_last_days_are_left_out(self) -> None:
        records = _record(
            [
                (1, "k", "kitchen"),
                (25, "k", "kitchen"),
                (25.5, "b", "bathroom"),
                (49, "k", "kitchen"),
            ]
        )
        moments = home_moments(records, UTC)
        assert moments["kitchen_per_day"] == 1.0
        assert moments["bathroom_per_day"] == 1.0
        assert moments["bedroom_per_day"] is None

    def test_switches_nights_and_retriggers(self) -> None:
        records = _record(
            [
                (0.5, "x", CONTACT),
                (25.0, "bed", "bedroom"),
                (25.0 + 100 / 3600, "bed", "bedroom"),
                (25.0 + 400 / 3600, "k", "kitchen"),
                (25.0 + 2000 / 3600, "k", "kitchen"),
                (26.0, "bed", "bedroom"),
                (49.0, "x", CONTACT),
            ]
        )
        moments = home_moments(records, UTC)
        assert moments["switch_share"] == pytest.approx(2 / 4)
        assert moments["night_bedroom"] == 3.0
        assert moments["retrigger_gap_seconds"] == 100.0

    def test_cohort_medians_leave_out_homes_without_the_sensor(self) -> None:
        rows = [dict.fromkeys(MOMENTS, None) for _ in range(3)]
        rows[0]["hall_per_day"] = 10.0
        rows[1]["hall_per_day"] = 30.0
        summary = cohort_moments(rows)
        assert summary["hall_per_day"] == {"median": 20.0, "homes": 2}
        assert summary["kitchen_per_day"] == {"median": None, "homes": 0}

    def test_the_distance_is_zero_at_the_target_and_finite_at_zero(self) -> None:
        target = {name: {"median": 2.0, "homes": 1} for name in MOMENTS}
        assert distance(target, target) == 0.0
        empty = {name: {"median": 0.0, "homes": 1} for name in MOMENTS}
        assert math.isfinite(distance(empty, target))
        assert distance(empty, target) > 0.0


class TestTheFacts:
    def test_the_hold_off_is_the_shortest_motion_gap(self) -> None:
        homes = {
            "a": _record(
                [
                    (0, "Kitchen", "kitchen"),
                    (0 + 61 / 3600, "Kitchen", "kitchen"),
                    (1, "Fridge Door", CONTACT),
                    (1 + 9 / 3600, "Fridge Door", CONTACT),
                    (2, "Fridge Door", CONTACT),
                ]
            ),
            "b": _record(
                [(0, "Kitchen", "kitchen"), (0 + 90 / 3600, "Kitchen", "kitchen")]
            ),
        }
        facts = sensor_facts(homes)
        assert facts["hold_off_seconds"] == 61.0
        fridge = facts["contacts"]["Fridge Door"]
        assert fridge["share_under_a_minute"] == 0.5
        assert fridge["median_gap_under_a_minute_seconds"] == 9.0


class TestThePlanning:
    def test_the_seeds_are_distinct_and_belong_to_no_study(self) -> None:
        seeds = Planning().seeds()
        assert len(seeds) == len(set(seeds)) == Planning().homes
        assert not set(seeds) & set(threshold_protocol().study_seeds())
