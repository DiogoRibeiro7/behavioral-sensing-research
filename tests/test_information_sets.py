"""Tests for matched information sets.

These guard the properties a Phase 1 comparison depends on: that nested sets
really are nested, that no feature can see past its prediction moment or into
another household, and that information which does not exist is reported as
missing rather than as a count of zero.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from sensor_modeling.datasets import (
    HH_EVIDENCE_CHANNELS,
    ActivityInterval,
    CasasRecording,
    EvidenceChannel,
    EvidenceResolution,
    FeatureTable,
    InformationComponent,
    InformationSet,
    build_feature_table,
    build_panel_features,
    hh_sensor_specs,
    nested_information_sets,
)
from sensor_modeling.datasets.casas_hh import _ROOM_WORDS, HH_LOCATIONS
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
    Unit,
)
from sensor_modeling.online import BehaviouralSensingPipeline, PipelineConfig
from sensor_modeling.online.pipeline import scoring_steps
from sensor_modeling.states import BehaviouralState

UTC = timezone.utc
STEP = timedelta(minutes=5)
T0 = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

C = InformationComponent.CURRENT_EVIDENCE
H = InformationComponent.TIME_OF_DAY
R = InformationComponent.RECENT_HISTORY

SPECS = (
    SensorSpec("Bathroom", Modality.MOTION, room="bathroom"),
    SensorSpec("Bedroom", Modality.MOTION, room="bedroom"),
    SensorSpec("FrontDoor", Modality.DOOR, room="hall"),
    SensorSpec("Hall", Modality.MOTION, room="hall"),
    SensorSpec("Kitchen", Modality.MOTION, room="kitchen"),
    SensorSpec("LivingRoom", Modality.MOTION, room="living"),
)


def recording(
    events: Iterable[tuple[str, datetime]],
    specs: Sequence[SensorSpec] = SPECS,
    activities: tuple[ActivityInterval, ...] = (),
) -> CasasRecording:
    """A recording of activations from *specs*, in time order."""
    by_id = {spec.sensor_id: spec for spec in specs}
    observations = tuple(
        sorted(
            (
                Observation(
                    timestamp=at,
                    sensor_id=sensor_id,
                    modality=by_id[sensor_id].modality,
                    kind=by_id[sensor_id].kind or ObservationKind.EVENT,
                    value=1.0,
                )
                for sensor_id, at in events
            ),
            key=lambda observation: observation.timestamp,
        )
    )
    return CasasRecording(
        registry=SensorRegistry.from_specs(specs),
        observations=observations,
        activities=activities,
    )


def grid(count: int, start: datetime = T0) -> list[datetime]:
    return [start + STEP * k for k in range(count)]


def random_events(seed: int, *, start: datetime = T0) -> list[tuple[str, datetime]]:
    """Seeded activations over three hours, one of them at the start."""
    rng = random.Random(seed)
    events = [("Kitchen", start)]
    for _ in range(200):
        sensor = rng.choice([spec.sensor_id for spec in SPECS])
        events.append((sensor, start + timedelta(seconds=rng.uniform(0, 3 * 3600))))
    return events


def full_set() -> InformationSet:
    return nested_information_sets()[-1]


def build(
    source: CasasRecording,
    moments: Sequence[datetime],
    information_set: InformationSet | None = None,
    household: str = "hh001",
) -> FeatureTable:
    return build_feature_table(
        source, information_set or full_set(), moments, household=household
    )


class TestDeclaration:
    def test_the_four_phase_one_sets_share_one_resolution(self) -> None:
        sets = nested_information_sets()
        assert [s.components for s in sets] == [
            frozenset({C}),
            frozenset({C, H}),
            frozenset({C, R}),
            frozenset({C, H, R}),
        ]
        assert len({s.resolution for s in sets}) == 1
        assert len({s.name for s in sets}) == 4

    def test_nesting_is_a_lattice_not_a_chain(self) -> None:
        current, timed, history, both = nested_information_sets()
        for smaller, larger in [
            (current, timed),
            (current, history),
            (timed, both),
            (history, both),
            (current, both),
        ]:
            assert smaller.is_nested_in(larger)
            assert not larger.is_nested_in(smaller)
        assert not timed.is_nested_in(history)
        assert not history.is_nested_in(timed)
        assert all(s.is_nested_in(s) for s in (current, timed, history, both))

    def test_sets_at_different_resolutions_are_not_nested(self) -> None:
        coarse = nested_information_sets(EvidenceResolution(step=timedelta(minutes=10)))
        fine = nested_information_sets()
        assert not coarse[0].is_nested_in(fine[-1])
        deeper = nested_information_sets(EvidenceResolution(history_steps=6))
        assert not fine[0].is_nested_in(deeper[-1])

    def test_default_columns(self) -> None:
        channels = [
            "bathroom_motion",
            "bedroom_motion",
            "hall_door",
            "hall_motion",
            "kitchen_motion",
            "living_motion",
        ]
        current = [f"events_{name}_lag0" for name in channels]
        history = [f"events_{name}_lag{lag}" for lag in (1, 2, 3) for name in channels]
        sets = nested_information_sets()
        assert list(sets[0].columns) == current
        assert list(sets[1].columns) == [*current, "hour_of_day"]
        assert list(sets[2].columns) == [*current, *history]
        assert list(sets[3].columns) == [*current, "hour_of_day", *history]

    def test_default_channels_cover_every_hh_event_channel(self) -> None:
        specs, _ = hh_sensor_specs(HH_LOCATIONS)
        produced = {
            EvidenceChannel(spec.room, spec.modality)
            for spec in specs
            if spec.kind is ObservationKind.EVENT and spec.room
        }
        produced |= {
            EvidenceChannel(room, Modality.MOTION) for room in _ROOM_WORDS.values()
        }
        assert produced == set(HH_EVIDENCE_CHANNELS)

    def test_channel_order_does_not_change_the_declaration(self) -> None:
        shuffled = EvidenceResolution(channels=tuple(reversed(HH_EVIDENCE_CHANNELS)))
        assert shuffled == EvidenceResolution()
        assert InformationSet("x", frozenset({C}), shuffled).sha256() == (
            InformationSet("x", frozenset({C})).sha256()
        )

    def test_declaration_is_serialisable_and_its_digest_stable(self) -> None:
        first, second = full_set(), nested_information_sets()[-1]
        payload = json.loads(json.dumps(first.to_dict()))
        assert payload["step_seconds"] == 300.0
        assert payload["history_steps"] == 3
        assert payload["columns"] == list(first.columns)
        assert first.sha256() == second.sha256()
        other = nested_information_sets(EvidenceResolution(step=timedelta(minutes=10)))
        assert other[-1].sha256() != first.sha256()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"step": timedelta(0)},
            {"step": timedelta(minutes=-5)},
            {"channels": ()},
            {"channels": HH_EVIDENCE_CHANNELS + HH_EVIDENCE_CHANNELS[:1]},
            {
                "channels": (
                    EvidenceChannel("x", Modality.WEARABLE_MOTION),
                    EvidenceChannel("x_wearable", Modality.MOTION),
                )
            },
            {"history_steps": -1},
            {"history_steps": True},
        ],
    )
    def test_invalid_resolutions_are_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            EvidenceResolution(**kwargs)  # type: ignore[arg-type]

    def test_invalid_sets_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="name"):
            InformationSet(" ", frozenset({C}))
        with pytest.raises(ValueError, match="no components"):
            InformationSet("empty", frozenset())
        with pytest.raises(ValueError, match="history"):
            InformationSet("h", frozenset({C, R}), EvidenceResolution(history_steps=0))
        with pytest.raises(ValueError, match="room"):
            EvidenceChannel("", Modality.MOTION)


class TestNestedFeatures:
    def test_smaller_sets_are_exact_projections_of_the_largest(self) -> None:
        source = recording(random_events(1))
        moments = grid(36)
        largest = build(source, moments)
        for information_set in nested_information_sets():
            table = build(source, moments, information_set)
            indices = [largest.columns.index(name) for name in table.columns]
            assert indices == sorted(indices)
            np.testing.assert_array_equal(table.values, largest.values[:, indices])


class TestNoFutureLeakage:
    def test_each_row_depends_only_on_observations_up_to_its_moment(self) -> None:
        source = recording(random_events(2))
        moments = grid(36)
        full = build(source, moments)
        for row, moment in enumerate(moments):
            truncated = CasasRecording(
                registry=source.registry,
                observations=tuple(
                    observation
                    for observation in source.observations
                    if observation.timestamp <= moment
                ),
                activities=(),
            )
            alone = build(truncated, [moment])
            np.testing.assert_array_equal(alone.values[0], full.values[row])

    def test_windows_are_right_closed_at_the_moment(self) -> None:
        t = T0 + STEP * 4
        source = recording(
            [
                ("Kitchen", T0),
                ("Bedroom", t),
                ("Bathroom", t - STEP),
                ("LivingRoom", t + timedelta(microseconds=1)),
            ]
        )
        row = build(source, [t])
        assert row.column("events_bedroom_motion_lag0")[0] == 1
        assert row.column("events_bathroom_motion_lag0")[0] == 0
        assert row.column("events_bathroom_motion_lag1")[0] == 1
        living = [c for c in row.columns if c.startswith("events_living")]
        assert all(row.column(name)[0] == 0 for name in living)

    def test_annotations_are_never_read(self) -> None:
        events = random_events(3)
        label = ActivityInterval(
            "Sleep", T0, T0 + timedelta(hours=3), BehaviouralState.SLEEPING
        )
        plain = build(recording(events), grid(36))
        labelled = build(recording(events, activities=(label,)), grid(36))
        np.testing.assert_array_equal(plain.values, labelled.values)


class TestHouseholdIsolation:
    def test_panel_tables_equal_single_household_builds(self) -> None:
        homes = {
            "hh002": recording(random_events(4)),
            "hh001": recording(random_events(5)),
        }
        moments = {home: grid(36) for home in homes}
        panel = build_panel_features(homes, full_set(), moments)
        assert list(panel) == ["hh001", "hh002"]
        for home, table in panel.items():
            assert table.household == home
            alone = build(homes[home], moments[home], household=home)
            np.testing.assert_array_equal(table.values, alone.values)

    def test_history_never_reaches_into_another_household(self) -> None:
        later = T0 + timedelta(hours=1)
        early = recording([("Kitchen", later - STEP * k) for k in range(1, 5)])
        late = recording([("Kitchen", later)])
        panel = build_panel_features(
            {"early": early, "late": late},
            full_set(),
            {"early": [later], "late": [later]},
        )
        history = [
            c for c in full_set().columns if c.endswith(("lag1", "lag2", "lag3"))
        ]
        assert all(np.isnan(panel["late"].column(name)[0]) for name in history)
        assert panel["early"].column("events_kitchen_motion_lag1")[0] == 1

    def test_households_must_match_between_recordings_and_moments(self) -> None:
        with pytest.raises(ValueError, match="same households"):
            build_panel_features(
                {"hh001": recording([("Kitchen", T0)])}, full_set(), {"hh002": [T0]}
            )


class TestDeterminism:
    def test_repeated_builds_are_identical(self) -> None:
        source = recording(random_events(6))
        first, second = build(source, grid(36)), build(source, grid(36))
        assert first.columns == second.columns
        assert first.moments == second.moments
        np.testing.assert_array_equal(first.values, second.values)

    def test_observation_order_does_not_matter(self) -> None:
        source = recording(random_events(7))
        shuffled = list(source.observations)
        random.Random(0).shuffle(shuffled)
        reordered = CasasRecording(
            registry=source.registry, observations=tuple(shuffled), activities=()
        )
        np.testing.assert_array_equal(
            build(source, grid(36)).values, build(reordered, grid(36)).values
        )

    def test_values_are_read_only(self) -> None:
        table = build(recording([("Kitchen", T0)]), [T0])
        with pytest.raises(ValueError):
            table.values[0, 0] = 99.0

    @pytest.mark.parametrize(
        "moments",
        [[T0, T0], [T0 + STEP, T0], [datetime(2024, 3, 1, 12)]],
    )
    def test_invalid_moments_are_rejected(self, moments: list[datetime]) -> None:
        with pytest.raises(ValueError):
            build(recording([("Kitchen", T0)]), moments)


class TestShortHistory:
    def test_history_before_the_recording_is_missing_not_zero(self) -> None:
        table = build(recording([("Kitchen", T0)]), grid(4))
        lagged = {
            lag: table.column(f"events_kitchen_motion_lag{lag}") for lag in (1, 2, 3)
        }
        assert all(np.isnan(lagged[lag][0]) for lag in (1, 2, 3))
        assert lagged[1][1] == 1 and np.isnan(lagged[2][1]) and np.isnan(lagged[3][1])
        assert lagged[1][3] == 0 and lagged[2][3] == 0 and lagged[3][3] == 1
        assert not np.isnan(table.column("events_kitchen_motion_lag0")).any()

    def test_quiet_history_inside_the_recording_is_zero(self) -> None:
        table = build(recording([("Kitchen", T0)]), [T0 + STEP * 10])
        events = [c for c in table.columns if c.startswith("events_")]
        assert all(table.column(name)[0] == 0 for name in events)

    def test_windows_after_the_last_observation_are_zero(self) -> None:
        table = build(recording([("Kitchen", T0)]), [T0 + timedelta(days=1)])
        events = [c for c in table.columns if c.startswith("events_")]
        assert all(table.column(name)[0] == 0 for name in events)

    def test_no_moments_gives_an_empty_table(self) -> None:
        table = build(recording([("Kitchen", T0)]), [])
        assert table.values.shape == (0, len(full_set().columns))

    def test_a_recording_without_observations_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no observations"):
            build(recording([]), [T0])


class TestMissingSensors:
    def test_an_uninstrumented_channel_is_missing_at_every_lag(self) -> None:
        specs = [spec for spec in SPECS if spec.room != "bathroom"]
        table = build(recording([("Kitchen", T0)], specs), grid(8))
        bathroom = [c for c in table.columns if "bathroom" in c]
        assert len(bathroom) == 4
        assert all(np.isnan(table.column(name)).all() for name in bathroom)
        assert table.uninstrumented == (EvidenceChannel("bathroom", Modality.MOTION),)
        assert not np.isnan(table.column("events_kitchen_motion_lag0")).any()

    def test_an_installed_but_silent_sensor_reports_zero(self) -> None:
        table = build(recording([("Kitchen", T0)]), grid(8))
        assert (table.column("events_bathroom_motion_lag0") == 0).all()
        assert table.uninstrumented == ()

    def test_sensors_that_reach_no_channel_are_excluded_and_reported(self) -> None:
        specs = [
            *SPECS,
            SensorSpec(
                "KitchenTemperature",
                Modality.ENVIRONMENTAL,
                kind=ObservationKind.SAMPLE,
                unit=Unit.CELSIUS,
                room="kitchen",
            ),
            SensorSpec("M099", Modality.MOTION),
            SensorSpec("Garage", Modality.MOTION, room="garage"),
        ]
        base = recording([("Kitchen", T0)], specs)
        extra = (
            Observation(
                T0,
                "KitchenTemperature",
                Modality.ENVIRONMENTAL,
                ObservationKind.SAMPLE,
                21.5,
                unit=Unit.CELSIUS,
            ),
            Observation(T0, "M099", Modality.MOTION, ObservationKind.EVENT, 1.0),
            Observation(T0, "Garage", Modality.MOTION, ObservationKind.EVENT, 1.0),
        )
        source = CasasRecording(
            registry=base.registry,
            observations=base.observations + extra,
            activities=(),
        )
        table = build(source, [T0])
        assert table.excluded_sensors == ("Garage", "KitchenTemperature", "M099")
        current = [c for c in table.columns if c.endswith("lag0")]
        assert sum(table.column(name)[0] for name in current) == 1

    def test_same_channel_sensors_pool_and_modalities_stay_apart(self) -> None:
        specs = [*SPECS, SensorSpec("Entry", Modality.MOTION, room="hall")]
        events = [("Hall", T0), ("Entry", T0), ("FrontDoor", T0)]
        table = build(recording(events, specs), [T0])
        assert table.column("events_hall_motion_lag0")[0] == 2
        assert table.column("events_hall_door_lag0")[0] == 1

    def test_zero_valued_events_are_not_activations(self) -> None:
        base = recording([("Kitchen", T0)])
        off = Observation(T0, "Kitchen", Modality.MOTION, ObservationKind.EVENT, 0.0)
        source = CasasRecording(
            registry=base.registry,
            observations=base.observations + (off,),
            activities=(),
        )
        assert build(source, [T0]).column("events_kitchen_motion_lag0")[0] == 1


class TestTimeOfDay:
    @pytest.mark.parametrize(
        ("moment", "hour"),
        [
            (datetime(2024, 3, 1, 23, 59, 59, 999999, tzinfo=UTC), 23),
            (datetime(2024, 3, 2, 0, 0, tzinfo=UTC), 0),
            (datetime(2024, 3, 2, 6, 59, 59, tzinfo=UTC), 6),
            (datetime(2024, 3, 2, 7, 0, tzinfo=UTC), 7),
        ],
    )
    def test_hour_boundaries(self, moment: datetime, hour: int) -> None:
        source = recording([("Kitchen", datetime(2024, 3, 1, tzinfo=UTC))])
        assert build(source, [moment]).column("hour_of_day")[0] == hour

    def test_hour_is_local_to_the_recording(self) -> None:
        zone = ZoneInfo("America/Los_Angeles")
        source = recording([("Kitchen", datetime(2024, 7, 1, 7, 0, tzinfo=zone))])
        utc_moment = datetime(2024, 7, 1, 15, 0, tzinfo=UTC)
        table = build(source, [utc_moment])
        assert table.column("hour_of_day")[0] == 8
        assert table.moments[0].tzinfo is zone
        assert table.moments[0] == utc_moment

    def test_a_window_spanning_midnight_counts_across_the_date_change(self) -> None:
        start = datetime(2024, 3, 1, 23, 50, tzinfo=UTC)
        source = recording(
            [("Kitchen", start), ("Bedroom", start + timedelta(minutes=8))]
        )
        midnight = datetime(2024, 3, 2, 0, 2, tzinfo=UTC)
        table = build(source, [midnight])
        assert table.column("hour_of_day")[0] == 0
        assert table.column("events_bedroom_motion_lag0")[0] == 1
        assert table.column("events_kitchen_motion_lag2")[0] == 1


class TestPipelineAlignment:
    def test_current_evidence_is_the_batch_the_pipeline_scores(self) -> None:
        source = recording(random_events(8))
        pipeline = BehaviouralSensingPipeline(
            source.registry, config=PipelineConfig(tz=UTC, step=STEP)
        )
        steps = pipeline.run(source.observations)
        steps.extend(pipeline.close(source.observations[-1].timestamp))
        steps = scoring_steps(steps)
        table = build(source, [step.at for step in steps], nested_information_sets()[1])

        channel_of = {
            spec.sensor_id: EvidenceChannel(str(spec.room), spec.modality)
            for spec in SPECS
        }
        for row, step in enumerate(steps):
            batch: dict[EvidenceChannel, int] = {}
            for contribution in step.state.evidence:
                channel = channel_of[contribution.sensor_id]
                batch[channel] = batch.get(channel, 0) + contribution.observations
            for channel, count in batch.items():
                assert table.column(f"events_{channel.name}_lag0")[row] == count
            assert table.column("hour_of_day")[row] == step.at.hour
        assert sum(sum(c.observations for c in s.state.evidence) for s in steps) == len(
            source.observations
        )
