"""Tests for occupancy context estimation and activity attribution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sensor_modeling.context import (
    CONTEXTS,
    ContextConfig,
    ContextEstimate,
    OccupancyContext,
    ResidentContextEstimator,
    rooms_active_at,
)
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
    Unit,
)

T0 = datetime(2024, 5, 1, 9, 0, tzinfo=timezone.utc)


def household() -> SensorRegistry:
    """A household with a door, two room sensors, radar, and a beacon."""
    return SensorRegistry.from_specs(
        [
            SensorSpec("front_door", Modality.DOOR, room="hall"),
            SensorSpec("kitchen_motion", Modality.MOTION, room="kitchen"),
            SensorSpec("bathroom_motion", Modality.MOTION, room="bathroom"),
            SensorSpec(
                "living_radar",
                Modality.RADAR,
                kind=ObservationKind.SAMPLE,
                unit=Unit.COUNT,
                room="living",
                expected_interval=timedelta(seconds=30),
            ),
            SensorSpec(
                "resident_beacon",
                Modality.PROXIMITY,
                kind=ObservationKind.STATE,
                attributable=True,
                expected_interval=timedelta(minutes=1),
            ),
        ]
    )


def door(at: datetime) -> Observation:
    """A front-door crossing."""
    return Observation(at, "front_door", Modality.DOOR, ObservationKind.EVENT, 1.0)


def beacon(at: datetime, in_range: float) -> Observation:
    """A personal presence beacon report."""
    return Observation(
        at, "resident_beacon", Modality.PROXIMITY, ObservationKind.STATE, in_range
    )


def radar(at: datetime, tracks: float) -> Observation:
    """A radar track count."""
    return Observation(
        at,
        "living_radar",
        Modality.RADAR,
        ObservationKind.SAMPLE,
        tracks,
        unit=Unit.COUNT,
    )


def motion(at: datetime, sensor_id: str) -> Observation:
    """A room motion activation."""
    return Observation(at, sensor_id, Modality.MOTION, ObservationKind.EVENT, 1.0)


def settle(
    estimator: ResidentContextEstimator,
    start: datetime,
    build: object,
    minutes: int = 40,
) -> tuple[ContextEstimate, datetime]:
    """Feed *minutes* of repeated evidence and return the final estimate."""
    now = start
    estimate = None
    for step in range(minutes):
        now = start + timedelta(minutes=step)
        estimate = estimator.update(now, build(now))  # type: ignore[operator]
    assert estimate is not None
    return estimate, now


class TestOccupancyInference:
    def test_beacon_and_single_track_indicate_the_resident_alone(self) -> None:
        estimator = ResidentContextEstimator(household())
        estimate, _ = settle(
            estimator, T0, lambda now: [beacon(now, 1.0), radar(now, 1.0)]
        )
        assert estimate.most_likely is OccupancyContext.RESIDENT_ALONE
        assert estimate.resident_home > 0.9
        assert estimate.visitor_present < 0.1
        assert estimate.ambient_attribution() > 0.9

    def test_two_tracks_indicate_a_visitor(self) -> None:
        """Two tracked people outweigh a prior that favours living alone.

        The posterior stays deliberately short of certainty. Successive radar
        samples are discounted because they are correlated, and the declared
        dynamics say visits are short, so sustained two-track evidence shifts
        the belief without ever asserting a visitor beyond doubt.
        """
        estimator = ResidentContextEstimator(household())
        estimate, _ = settle(
            estimator, T0, lambda now: [beacon(now, 1.0), radar(now, 2.0)]
        )
        assert estimate.most_likely is OccupancyContext.RESIDENT_WITH_VISITOR
        assert estimate.multiple_people > 0.5
        assert estimate.ambient_attribution() < 0.8

    def test_more_tracks_give_stronger_multi_person_evidence(self) -> None:
        def multiple_people_given(tracks: float) -> float:
            estimator = ResidentContextEstimator(household())
            estimate, _ = settle(
                estimator, T0, lambda now: [beacon(now, 1.0), radar(now, tracks)]
            )
            return estimate.multiple_people

        assert (
            multiple_people_given(1.0)
            < multiple_people_given(2.0)
            < multiple_people_given(3.0)
        )

    def test_activity_without_the_beacon_indicates_someone_else(self) -> None:
        """A carer's round must not be credited to the resident."""
        estimator = ResidentContextEstimator(household())
        estimate, _ = settle(
            estimator,
            T0,
            lambda now: [
                beacon(now, 0.0),
                radar(now, 1.0),
                motion(now, "kitchen_motion"),
            ],
        )
        assert estimate.most_likely is OccupancyContext.VISITOR_ONLY
        assert estimate.resident_home < 0.15
        assert estimate.ambient_attribution() < 0.15

    def test_an_empty_home_is_recognised(self) -> None:
        estimator = ResidentContextEstimator(household())
        estimate, _ = settle(
            estimator, T0, lambda now: [beacon(now, 0.0), radar(now, 0.0)]
        )
        assert estimate.most_likely is OccupancyContext.EMPTY
        assert estimate.ambient_attribution() < 0.1

    def test_simultaneous_activity_in_two_rooms_indicates_two_people(self) -> None:
        """One person cannot be in the kitchen and the bathroom at once."""
        estimator = ResidentContextEstimator(household())
        estimate, _ = settle(
            estimator,
            T0,
            lambda now: [
                beacon(now, 1.0),
                motion(now, "kitchen_motion"),
                motion(now + timedelta(seconds=10), "bathroom_motion"),
            ],
        )
        assert estimate.concurrency == 2
        assert estimate.multiple_people > 0.5

    def test_sequential_room_activity_does_not_imply_two_people(self) -> None:
        estimator = ResidentContextEstimator(
            household(), ContextConfig(concurrency_window=timedelta(seconds=30))
        )
        estimate = estimator.update(
            T0,
            [
                motion(T0, "kitchen_motion"),
                motion(T0 + timedelta(minutes=5), "bathroom_motion"),
            ],
        )
        assert estimate.concurrency == 1

    def test_a_door_crossing_loosens_a_confident_belief(self) -> None:
        """Whoever was home an hour ago may have just left."""
        estimator = ResidentContextEstimator(household())
        confident, now = settle(
            estimator, T0, lambda t: [beacon(t, 1.0), radar(t, 1.0)]
        )
        after = estimator.update(now + timedelta(minutes=1), [door(now)])
        assert after.door_events == 1
        assert after.resident_home < confident.resident_home

    def test_door_mixing_can_be_disabled(self) -> None:
        estimator = ResidentContextEstimator(
            household(), ContextConfig(door_mixing=0.0)
        )
        confident, now = settle(
            estimator, T0, lambda t: [beacon(t, 1.0), radar(t, 1.0)]
        )
        after = estimator.update(now + timedelta(seconds=1), [door(now)])
        assert after.resident_home == pytest.approx(confident.resident_home, abs=1e-3)


class TestAttribution:
    def test_attributable_sensors_are_always_the_residents(self) -> None:
        estimator = ResidentContextEstimator(household())
        estimate, _ = settle(
            estimator, T0, lambda now: [beacon(now, 1.0), radar(now, 2.0)]
        )
        weights = estimator.attribution(estimate)
        assert weights["resident_beacon"] == 1.0
        assert weights["kitchen_motion"] < 1.0

    def test_ambient_sensors_share_one_attribution_weight(self) -> None:
        estimator = ResidentContextEstimator(household())
        estimate, _ = settle(
            estimator, T0, lambda now: [beacon(now, 1.0), radar(now, 2.0)]
        )
        weights = estimator.attribution(estimate)
        ambient = {
            weights[sensor]
            for sensor in ("front_door", "kitchen_motion", "bathroom_motion")
        }
        assert len(ambient) == 1

    def test_attribution_collapses_when_the_resident_is_out(self) -> None:
        estimator = ResidentContextEstimator(household())
        estimate, _ = settle(
            estimator,
            T0,
            lambda now: [
                beacon(now, 0.0),
                radar(now, 1.0),
                motion(now, "kitchen_motion"),
            ],
        )
        assert estimator.attribution(estimate)["kitchen_motion"] < 0.15

    def test_rooms_active_at_ignores_attributable_sensors(self) -> None:
        registry = household()
        rooms = rooms_active_at(
            registry,
            [motion(T0, "kitchen_motion"), beacon(T0, 1.0)],
        )
        assert rooms == {"kitchen"}


class TestEstimatorMechanics:
    def test_unreliable_sensors_are_excluded_from_the_evidence(self) -> None:
        estimator = ResidentContextEstimator(household())
        with_radar, _ = settle(
            estimator, T0, lambda now: [beacon(now, 1.0), radar(now, 2.0)]
        )
        blind = ResidentContextEstimator(household())
        now = T0
        for step in range(40):
            now = T0 + timedelta(minutes=step)
            without = blind.update(
                now,
                [beacon(now, 1.0), radar(now, 2.0)],
                reliabilities={"living_radar": 0.0},
            )
        assert without.multiple_people < with_radar.multiple_people

    def test_unregistered_sensors_are_ignored(self) -> None:
        estimator = ResidentContextEstimator(household())
        stray = Observation(T0, "ghost", Modality.MOTION, ObservationKind.EVENT, 1.0)
        before = estimator.belief
        estimator.update(T0, [stray])
        assert np.allclose(estimator.belief, before)

    def test_updates_must_not_move_backwards_in_time(self) -> None:
        estimator = ResidentContextEstimator(household())
        estimator.update(T0 + timedelta(hours=1), [])
        with pytest.raises(ValueError, match="non-decreasing"):
            estimator.update(T0, [])

    def test_snapshot_and_restore_reproduce_the_belief(self) -> None:
        estimator = ResidentContextEstimator(household())
        settle(estimator, T0, lambda now: [beacon(now, 1.0), radar(now, 2.0)])
        state = estimator.snapshot()

        restarted = ResidentContextEstimator(household())
        restarted.restore(state)
        assert np.allclose(restarted.belief, estimator.belief)

    def test_restore_rejects_a_foreign_context_set(self) -> None:
        estimator = ResidentContextEstimator(household())
        estimator.update(T0, [])
        state = estimator.snapshot()
        state["contexts"] = ["empty", "occupied"]
        with pytest.raises(ValueError, match="different occupancy contexts"):
            estimator.restore(state)

    def test_reset_returns_to_the_prior(self) -> None:
        estimator = ResidentContextEstimator(household())
        prior = estimator.belief
        settle(estimator, T0, lambda now: [beacon(now, 1.0), radar(now, 2.0)])
        estimator.reset()
        assert np.allclose(estimator.belief, prior)

    def test_context_probabilities_form_a_distribution(self) -> None:
        estimator = ResidentContextEstimator(household())
        estimate = estimator.update(T0, [beacon(T0, 1.0)])
        assert sum(estimate.probabilities.values()) == pytest.approx(1.0)
        assert set(estimate.probabilities) == set(CONTEXTS)
        assert estimate.to_dict()["most_likely"] == estimate.most_likely.value

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"door_mixing": 1.5},
            {"concurrency_window": timedelta(0)},
            {"concurrency_odds": 0.5},
            {"sample_weight": 0.0},
            {"resident_share": {c: 2.0 for c in CONTEXTS}},
            {"dwell": {OccupancyContext.EMPTY: timedelta(hours=1)}},
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            ContextConfig(**kwargs)

    def test_invalid_prior_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="prior"):
            ResidentContextEstimator(household(), prior=np.zeros(len(CONTEXTS)))
