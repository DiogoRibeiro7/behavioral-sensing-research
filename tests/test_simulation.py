"""Tests for the synthetic household and its fault injection."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta, timezone

import pytest

from sensor_modeling.context import OccupancyContext
from sensor_modeling.observations import ObservationFlag, ObservationStream
from sensor_modeling.simulation import (
    BehaviourShift,
    DegradationConfig,
    HouseholdConfig,
    build_registry,
    degrade,
    dropout,
    not_worn,
    simulate,
    stuck,
)
from sensor_modeling.states import BehaviouralState as S


def small(**overrides: object) -> HouseholdConfig:
    """A short household simulation for fast tests."""
    settings: dict[str, object] = {"days": 8, "seed": 4242}
    settings.update(overrides)
    return HouseholdConfig(**settings)  # type: ignore[arg-type]


class TestGroundTruth:
    def test_simulation_is_reproducible_from_the_seed(self) -> None:
        first = simulate(small())
        second = simulate(small())
        assert [obs.to_dict() for obs in first.observations] == [
            obs.to_dict() for obs in second.observations
        ]

    def test_a_different_seed_gives_a_different_household(self) -> None:
        assert len(simulate(small(seed=1)).observations) != len(
            simulate(small(seed=2)).observations
        )

    def test_episodes_tile_the_simulated_span_without_gaps(self) -> None:
        truth = simulate(small()).truth
        for previous, current in zip(truth.episodes, truth.episodes[1:]):
            assert current.start == previous.end

    def test_every_episode_has_positive_duration(self) -> None:
        truth = simulate(small()).truth
        assert all(episode.duration > timedelta(0) for episode in truth.episodes)

    def test_states_at_matches_the_scanning_lookup(self) -> None:
        truth = simulate(small(days=3)).truth
        moments = [
            truth.episodes[0].start + timedelta(minutes=17 * step)
            for step in range(200)
        ]
        assert truth.states_at(moments) == [truth.state_at(m) for m in moments]

    def test_the_resident_sleeps_a_plausible_amount(self) -> None:
        hours = simulate(small(days=14)).truth.daily_hours(S.SLEEPING)
        # Exclude the truncated final day, which ends at simulation end.
        interior = [value for _, value in sorted(hours.items())][1:-1]
        assert all(4.0 < value < 12.0 for value in interior)

    def test_away_periods_have_no_room(self) -> None:
        truth = simulate(small()).truth
        for episode in truth.episodes:
            if episode.state is S.AWAY:
                assert episode.room is None
            else:
                assert episode.room is not None

    def test_occupancy_context_reflects_resident_and_visitors(self) -> None:
        truth = simulate(small()).truth
        visitor = truth.visitors[0]
        midpoint = visitor.start + (visitor.end - visitor.start) / 2
        context = truth.context_at(midpoint)
        assert context in {
            OccupancyContext.RESIDENT_WITH_VISITOR,
            OccupancyContext.VISITOR_ONLY,
        }

    def test_a_behaviour_shift_changes_the_ground_truth(self) -> None:
        shifted = simulate(
            small(
                days=24,
                seed=99,
                shift=BehaviourShift(
                    start_day=12, sleep_delta_hours=1.5, night_bathroom_extra=2.0
                ),
            )
        )
        sleep = shifted.truth.daily_hours(S.SLEEPING)
        days = sorted(sleep)
        before = sum(sleep[day] for day in days[2:12]) / 10
        after = sum(sleep[day] for day in days[14:24]) / 10
        assert after < before - 0.5

    def test_a_shift_before_its_start_day_has_no_effect(self) -> None:
        without = simulate(small(days=6, seed=7))
        with_late_shift = simulate(
            small(
                days=6, seed=7, shift=BehaviourShift(start_day=99, sleep_delta_hours=4)
            )
        )
        assert without.truth.daily_hours(
            S.SLEEPING
        ) == with_late_shift.truth.daily_hours(S.SLEEPING)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"days": 0},
            {"wake_hour": 25.0},
            {"wake_hour": 20.0, "sleep_hour": 8.0},
            {"timing_jitter_minutes": -1.0},
            {"night_bathroom_rate": -1.0},
            {"outing_probability": 1.5},
            {"visitor_probability": -0.1},
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            HouseholdConfig(**kwargs)

    def test_a_negative_shift_start_day_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="start_day"):
            BehaviourShift(start_day=-1)


class TestSensorRecord:
    def test_every_observation_belongs_to_a_registered_sensor(self) -> None:
        result = simulate(small())
        registry = result.registry
        assert all(obs.sensor_id in registry for obs in result.observations)

    def test_every_observation_satisfies_its_declared_contract(self) -> None:
        result = simulate(small())
        for observation in result.observations:
            normalised = result.registry.normalise(observation)
            assert result.registry.in_range(normalised)

    def test_all_declared_sensors_actually_report(self) -> None:
        result = simulate(small())
        reporting = {obs.sensor_id for obs in result.observations}
        assert reporting == set(result.registry.sensor_ids())

    def test_observations_are_timestamp_ordered_and_timezone_aware(self) -> None:
        result = simulate(small())
        for previous, current in zip(result.observations, result.observations[1:]):
            assert previous.timestamp <= current.timestamp
            assert current.timestamp.tzinfo is not None

    def test_visitors_trip_ambient_sensors(self) -> None:
        """The contamination that makes attribution necessary must exist."""
        result = simulate(small())
        sources = Counter(
            obs.context.get("generated_by") for obs in result.observations
        )
        assert sources["visitor"] > 0
        assert sources["resident"] > sources["visitor"]

    def test_the_wearable_is_never_attributed_to_a_visitor(self) -> None:
        result = simulate(small())
        for observation in result.observations:
            if observation.sensor_id in {"wearable_motion", "resident_beacon"}:
                assert observation.context.get("generated_by") == "resident"

    def test_the_beacon_reports_out_of_range_while_away(self) -> None:
        result = simulate(small())
        away = [
            obs
            for obs in result.observations
            if obs.sensor_id == "resident_beacon"
            and result.truth.state_at(obs.timestamp) is S.AWAY
        ]
        assert away
        assert all(obs.value == 0.0 for obs in away)

    def test_the_record_contains_false_activations(self) -> None:
        """A perfectly clean sensor record would not be a useful test bed."""
        result = simulate(small())
        stray = [
            obs
            for obs in result.observations
            if obs.sensor_id == "kitchen_motion"
            and result.truth.room_at(obs.timestamp) != "kitchen"
            and obs.context.get("generated_by") == "resident"
        ]
        assert stray

    def test_the_radar_reports_with_reduced_confidence(self) -> None:
        result = simulate(small())
        radar = [o for o in result.observations if o.sensor_id == "living_radar"]
        assert radar
        assert all(obs.confidence < 1.0 for obs in radar)

    def test_the_registry_documents_what_each_sensor_measures(self) -> None:
        for spec in build_registry():
            assert spec.description


class TestDegradation:
    def test_a_clean_configuration_changes_nothing(self) -> None:
        result = simulate(small(days=3))
        kept, dropped = degrade(result.observations, DegradationConfig())
        assert len(kept) == len(result.observations)
        assert dropped == []

    def test_missing_records_are_reported_rather_than_inferred(self) -> None:
        result = simulate(small(days=3))
        kept, dropped = degrade(
            result.observations, DegradationConfig(missing_rate=0.2, seed=3)
        )
        assert len(dropped) > 0
        assert len(kept) + len(dropped) == len(result.observations)

    def test_a_dropout_removes_exactly_the_faulted_window(self) -> None:
        result = simulate(small(days=3))
        start = result.start + timedelta(days=1)
        fault = dropout("bed_pressure", start, timedelta(hours=6))
        kept, dropped = degrade(result.observations, DegradationConfig(faults=(fault,)))
        assert all(obs.sensor_id == "bed_pressure" for obs in dropped)
        assert not [
            obs for obs in kept if obs.sensor_id == "bed_pressure" and fault.covers(obs)
        ]

    def test_a_stuck_sensor_repeats_one_value(self) -> None:
        result = simulate(small(days=3))
        start = result.start + timedelta(days=1)
        fault = stuck("bed_pressure", start, timedelta(hours=8))
        kept, _ = degrade(result.observations, DegradationConfig(faults=(fault,)))
        inside = [
            obs.value
            for obs in kept
            if obs.sensor_id == "bed_pressure" and fault.covers(obs)
        ]
        assert len(set(inside)) == 1

    def test_non_adherence_silences_every_worn_device(self) -> None:
        result = simulate(small(days=3))
        start = result.start + timedelta(days=1)
        faults = not_worn(
            ["wearable_motion", "resident_beacon"], start, timedelta(hours=12)
        )
        kept, dropped = degrade(result.observations, DegradationConfig(faults=faults))
        assert {obs.sensor_id for obs in dropped} == {
            "wearable_motion",
            "resident_beacon",
        }
        assert all(not any(f.covers(obs) for f in faults) for obs in kept)

    def test_late_records_arrive_out_of_order(self) -> None:
        result = simulate(small(days=2))
        kept, _ = degrade(
            result.observations,
            DegradationConfig(late_rate=0.1, late_delay=timedelta(minutes=30), seed=5),
        )
        out_of_order = [
            current
            for previous, current in zip(kept, kept[1:])
            if current.timestamp < previous.timestamp
        ]
        assert out_of_order

    def test_duplicates_are_delivered_and_then_collapsed_by_the_stream(self) -> None:
        result = simulate(small(days=2))
        kept, _ = degrade(
            result.observations, DegradationConfig(duplication_rate=0.15, seed=11)
        )
        assert len(kept) > len(result.observations)
        stream = ObservationStream.from_observations(kept)
        assert len(stream) == len(result.observations)

    def test_clock_drift_shifts_only_the_affected_source(self) -> None:
        result = simulate(small(days=2))
        offset = timedelta(minutes=4)
        kept, _ = degrade(
            result.observations,
            DegradationConfig(clock_drift={"sim-radar": offset}),
        )
        original = {
            source: sorted(
                obs.timestamp for obs in result.observations if obs.source == source
            )
            for source in ("sim-radar", "sim-hub")
        }
        degraded = {
            source: sorted(obs.timestamp for obs in kept if obs.source == source)
            for source in ("sim-radar", "sim-hub")
        }
        assert degraded["sim-radar"] == [t + offset for t in original["sim-radar"]]
        assert degraded["sim-hub"] == original["sim-hub"]

    def test_degradation_is_reproducible_from_its_own_seed(self) -> None:
        result = simulate(small(days=2))
        config = DegradationConfig(missing_rate=0.1, seed=17)
        first, _ = degrade(result.observations, config)
        second, _ = degrade(
            result.observations, DegradationConfig(missing_rate=0.1, seed=17)
        )
        assert [o.to_dict() for o in first] == [o.to_dict() for o in second]

    def test_late_arrival_is_visible_to_the_ingestor(self) -> None:
        result = simulate(small(days=2))
        kept, _ = degrade(
            result.observations,
            DegradationConfig(late_rate=1.0, late_delay=timedelta(hours=1), seed=2),
        )
        assert all(
            obs.received_at is not None and obs.latency == timedelta(hours=1)
            for obs in kept
        )
        assert ObservationFlag.LATE_ARRIVAL not in kept[0].flags

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"missing_rate": 1.5},
            {"duplication_rate": -0.1},
            {"late_rate": 2.0},
            {"late_delay": timedelta(seconds=-1)},
        ],
    )
    def test_invalid_degradation_configuration_is_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            DegradationConfig(**kwargs)

    def test_a_zero_length_fault_window_is_rejected(self) -> None:
        result = simulate(small(days=2))
        with pytest.raises(ValueError, match="positive duration"):
            DegradationConfig(
                faults=(dropout("bed_pressure", result.start, timedelta(0)),)
            )


class TestLocalClock:
    def test_the_household_follows_its_local_timezone(self) -> None:
        result = simulate(small(days=3, start=date(2024, 6, 3)))
        assert result.truth.tz is not timezone.utc
        first_night = result.truth.episodes[0]
        assert first_night.state is S.SLEEPING

    def test_a_dst_transition_does_not_break_the_schedule(self) -> None:
        """Lisbon springs forward on 2024-03-31; the routine must survive it."""
        result = simulate(small(days=4, start=date(2024, 3, 29), seed=31))
        truth = result.truth
        for previous, current in zip(truth.episodes, truth.episodes[1:]):
            assert current.start == previous.end
        assert any(
            episode.state is S.SLEEPING
            and episode.start.astimezone(truth.tz).date() == date(2024, 3, 31)
            for episode in truth.episodes
        )
