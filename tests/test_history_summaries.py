"""Tests for the history-summary information component.

The summaries must stay inside the information the filter receives, so the
central check recomputes every summary from the per-step lagged counts of the
same recording, with a separate, deliberately plain implementation. The others
guard what a matched comparison relies on: no feature sees past its moment or
into another household, missing information is NaN and never zero, and the
declaration is stable.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    EvidenceChannel,
    EvidenceResolution,
    FeatureTable,
    HouseholdSplit,
    InformationComponent,
    InformationSet,
    baseline_suite,
    build_feature_table,
    build_panel_features,
    compare_information_sets,
    nested_information_sets,
    run_matched_evaluation,
)
from sensor_modeling.datasets.information_sets import (
    SUMMARY_FEATURES,
    parse_evidence_column,
    parse_summary_column,
    summary_column,
)
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
    Unit,
)
from sensor_modeling.simulation import HouseholdConfig, simulate

UTC = timezone.utc
STEP = timedelta(minutes=5)
T0 = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

C = InformationComponent.CURRENT_EVIDENCE
H = InformationComponent.TIME_OF_DAY
R = InformationComponent.RECENT_HISTORY
S = InformationComponent.HISTORY_SUMMARY

SPECS = (
    SensorSpec("Bathroom", Modality.MOTION, room="bathroom"),
    SensorSpec("Bedroom", Modality.MOTION, room="bedroom"),
    SensorSpec("FrontDoor", Modality.DOOR, room="hall"),
    SensorSpec("Hall", Modality.MOTION, room="hall"),
    SensorSpec("Kitchen", Modality.MOTION, room="kitchen"),
    SensorSpec("LivingRoom", Modality.MOTION, room="living"),
)
#: No bathroom sensor and no hall motion: one room and one channel uninstrumented.
PARTIAL = tuple(spec for spec in SPECS if spec.sensor_id not in {"Bathroom", "Hall"})

#: The pre-declared default lookbacks: one hour and three hours.
WINDOWS = (60, 180)
DEFAULT_SUMMARIES = (
    [f"history_count_{c.name}_60m" for c in EvidenceResolution().channels]
    + ["history_room_changes_60m"]
    + [f"history_count_{c.name}_180m" for c in EvidenceResolution().channels]
    + ["history_room_changes_180m"]
    + [f"history_quiet_minutes_{c.name}_180m" for c in EvidenceResolution().channels]
    + [
        f"history_last_room_{room}_180m"
        for room in ("bathroom", "bedroom", "hall", "kitchen", "living")
    ]
)


def recording(
    events: Iterable[tuple[str, datetime]], specs: Sequence[SensorSpec] = SPECS
) -> CasasRecording:
    """A recording of activations from *specs*, in time order."""
    by_id = {spec.sensor_id: spec for spec in specs}
    return CasasRecording(
        registry=SensorRegistry.from_specs(specs),
        observations=tuple(
            sorted(
                (
                    Observation(
                        at,
                        sensor_id,
                        by_id[sensor_id].modality,
                        by_id[sensor_id].kind or ObservationKind.EVENT,
                        1.0,
                    )
                    for sensor_id, at in events
                ),
                key=lambda observation: observation.timestamp,
            )
        ),
        activities=(),
    )


def grid(count: int, start: datetime = T0, step: timedelta = STEP) -> list[datetime]:
    return [start + step * k for k in range(count)]


def bursty_events(
    seed: int, specs: Sequence[SensorSpec] = SPECS, hours: float = 6.0
) -> list[tuple[str, datetime]]:
    """Seeded bursts of activity separated by quiet spells, starting at T0.

    Quiet spells of up to four hours exercise censoring at the horizon, and
    bursts that hop between rooms exercise room changes.
    """
    rng = random.Random(seed)
    sensors = [spec.sensor_id for spec in specs]
    events = [(sensors[0], T0)]
    at = T0
    end = T0 + timedelta(hours=hours)
    while at < end:
        at += timedelta(minutes=rng.choice([1, 3, 8, 20, 70, 200]))
        for _ in range(rng.randint(1, 12)):
            at += timedelta(seconds=rng.uniform(0, 90))
            events.append((rng.choice(sensors), at))
    return events


def summary_set(resolution: EvidenceResolution | None = None) -> InformationSet:
    return InformationSet(
        "current+history_summary",
        frozenset({C, S}),
        resolution or EvidenceResolution(),
    )


def build(
    source: CasasRecording,
    moments: Sequence[datetime],
    information_set: InformationSet | None = None,
    household: str = "hh001",
) -> FeatureTable:
    return build_feature_table(
        source, information_set or summary_set(), moments, household=household
    )


def value(table: FeatureTable, name: str, row: int = 0) -> float:
    return float(table.column(name)[row])


def summaries_of(table: FeatureTable) -> dict[str, np.ndarray]:
    return {
        name: table.column(name)
        for name in table.columns
        if parse_summary_column(name) is not None
    }


def reference_summaries(
    source: CasasRecording,
    moments: Sequence[datetime],
    resolution: EvidenceResolution,
) -> dict[str, list[float]]:
    """Every summary recomputed from lagged per-step counts, one row at a time.

    The lagged counts come from the recent-history component, whose windows are
    the filter's step windows. Anything recomputed from them alone uses no
    timing or ordering inside a step.
    """
    step_minutes = resolution.step / timedelta(minutes=1)
    depth = resolution.horizon // resolution.step
    lagged = build(
        source,
        moments,
        InformationSet(
            "lags",
            frozenset({C, R}),
            EvidenceResolution(
                step=resolution.step,
                channels=resolution.channels,
                history_steps=depth - 1,
            ),
        ),
    )
    uninstrumented = set(lagged.uninstrumented)
    instrumented_rooms = {
        channel.room for channel in resolution.channels if channel not in uninstrumented
    }
    out: dict[str, list[float]] = {}

    def put(name: str, number: float) -> None:
        out.setdefault(name, []).append(number)

    for row in range(len(moments)):
        counts = {
            channel: [
                value(lagged, f"events_{channel.name}_lag{lag}", row)
                for lag in range(depth)
            ]
            for channel in resolution.channels
        }

        def complete(steps: int) -> bool:
            return all(
                not math.isnan(counts[channel][steps - 1])
                for channel in resolution.channels
                if channel not in uninstrumented
            )

        active_rooms = [
            {
                channel.room
                for channel in resolution.channels
                if counts[channel][lag] > 0
            }
            for lag in range(depth)
        ]
        for window in resolution.summary_windows:
            steps = window // resolution.step
            minutes = window // timedelta(minutes=1)
            for channel in resolution.channels:
                run = counts[channel][:steps]
                put(
                    f"history_count_{channel.name}_{minutes}m",
                    math.nan if any(math.isnan(c) for c in run) else sum(run),
                )
            sequence = [rooms for rooms in reversed(active_rooms[:steps]) if rooms]
            changes = sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)
            put(
                f"history_room_changes_{minutes}m",
                changes if instrumented_rooms and complete(steps) else math.nan,
            )
        horizon = resolution.horizon // timedelta(minutes=1)
        for channel in resolution.channels:
            run = counts[channel]
            active = [lag for lag, count in enumerate(run) if count > 0]
            if channel in uninstrumented:
                quiet = math.nan
            elif active:
                quiet = active[0] * step_minutes
            elif complete(depth):
                quiet = depth * step_minutes
            else:
                quiet = math.nan
            put(f"history_quiet_minutes_{channel.name}_{horizon}m", quiet)
        latest = next((rooms for rooms in active_rooms if rooms), None)
        for room in resolution.rooms:
            if room not in instrumented_rooms:
                indicator = math.nan
            elif latest is not None:
                indicator = float(room in latest)
            elif complete(depth):
                indicator = 0.0
            else:
                indicator = math.nan
            put(f"history_last_room_{room}_{horizon}m", indicator)
    return out


class TestDeclaration:
    def test_default_summary_columns(self) -> None:
        full = InformationSet("all", frozenset({C, H, R, S}))
        assert list(full.columns[: 6 + 1 + 18]) == list(
            nested_information_sets()[-1].columns
        )
        assert list(full.columns[25:]) == DEFAULT_SUMMARIES
        assert len(summary_set().columns) == 6 + 25

    def test_default_windows_are_one_and_three_hours(self) -> None:
        resolution = EvidenceResolution()
        assert resolution.summary_windows == tuple(
            timedelta(minutes=minutes) for minutes in WINDOWS
        )
        assert resolution.horizon == timedelta(hours=3)
        assert resolution.rooms == ("bathroom", "bedroom", "hall", "kitchen", "living")

    def test_names_are_machine_readable_and_round_trip(self) -> None:
        for name in DEFAULT_SUMMARIES:
            parsed = parse_summary_column(name)
            assert parsed is not None and parsed[0] in SUMMARY_FEATURES
            feature, subject, minutes = parsed
            assert summary_column(feature, timedelta(minutes=minutes), subject) == name
            assert parse_evidence_column(name) is None
        assert parse_summary_column("history_quiet_minutes_hall_door_180m") == (
            "quiet_minutes",
            "hall_door",
            180,
        )
        assert parse_summary_column("history_room_changes_60m") == (
            "room_changes",
            None,
            60,
        )
        for other in nested_information_sets()[-1].columns:
            assert parse_summary_column(other) is None
        for malformed in ("history_count_60m", "history_room_changes_x_60m"):
            assert parse_summary_column(malformed) is None

    def test_column_names_are_validated(self) -> None:
        with pytest.raises(ValueError, match="one of"):
            summary_column("mean", timedelta(minutes=60), "kitchen_motion")
        with pytest.raises(ValueError, match="subject"):
            summary_column("count", timedelta(minutes=60))
        with pytest.raises(ValueError, match="subject"):
            summary_column("room_changes", timedelta(minutes=60), "kitchen")

    def test_windows_are_stored_in_increasing_order(self) -> None:
        shuffled = EvidenceResolution(
            summary_windows=(timedelta(minutes=180), timedelta(minutes=60))
        )
        assert shuffled == EvidenceResolution()

    @pytest.mark.parametrize(
        "windows",
        [
            (),
            (timedelta(0),),
            (timedelta(minutes=-60),),
            (timedelta(seconds=90),),
            (timedelta(minutes=60), timedelta(minutes=60)),
            (60,),
        ],
    )
    def test_invalid_windows_are_rejected(self, windows: tuple[object, ...]) -> None:
        with pytest.raises(ValueError, match="summary window"):
            EvidenceResolution(summary_windows=windows)  # type: ignore[arg-type]

    def test_summaries_need_windows_of_whole_steps(self) -> None:
        odd = EvidenceResolution(step=timedelta(minutes=7))
        with pytest.raises(ValueError, match="whole numbers of steps"):
            summary_set(odd)
        assert InformationSet("current", frozenset({C}), odd).columns

    def test_the_phase_one_declarations_are_unchanged(self) -> None:
        """Digests recorded before this component existed must still match."""
        assert [s.sha256() for s in nested_information_sets()] == [
            "c4ef81951d5c35bca43ed458926c4a14dbb48c1af3e7496721f2a139c0649bdb",
            "cb70da40bb03ea3a2b7ac30b6087a988e99df599cfc222291a638ac31d67ea76",
            "eeaff036f46399dc343344718f430a98c145f7d511d249c2c51959b650c3970e",
            "02e25a6acc1f5a31076bd3374e520f7afe53b0afb49f2ea28223de3ee3ed931d",
        ]
        assert all(
            "summary_windows_seconds" not in s.to_dict()
            for s in nested_information_sets()
        )

    def test_the_windows_are_part_of_the_declaration(self) -> None:
        payload = summary_set().to_dict()
        assert payload["summary_windows_seconds"] == [3600.0, 10800.0]
        assert "history_summary" in payload["components"]  # type: ignore[operator]
        other = summary_set(
            EvidenceResolution(summary_windows=(timedelta(minutes=120),))
        )
        assert other.sha256() != summary_set().sha256()
        assert summary_set().sha256() == summary_set().sha256()

    def test_nesting(self) -> None:
        with_summaries = InformationSet("all", frozenset({C, H, R, S}))
        for smaller in nested_information_sets():
            assert smaller.is_nested_in(with_summaries)
            assert not with_summaries.is_nested_in(smaller)
        assert summary_set().is_nested_in(with_summaries)
        assert not summary_set().is_nested_in(nested_information_sets()[-1])
        other = EvidenceResolution(summary_windows=(timedelta(minutes=120),))
        assert not summary_set(other).is_nested_in(with_summaries)


class TestMatchedToTheFilterSteps:
    """Every summary is a function of the per-step counts the filter sees."""

    @pytest.mark.parametrize("seed", [1, 2, 3])
    @pytest.mark.parametrize("specs", [SPECS, PARTIAL], ids=["full", "partial"])
    def test_defaults_equal_a_recomputation_from_lagged_counts(
        self, seed: int, specs: Sequence[SensorSpec]
    ) -> None:
        source = recording(bursty_events(seed, specs), specs)
        moments = grid(100, T0 - timedelta(minutes=20))
        table = build(source, moments)
        expected = reference_summaries(source, moments, EvidenceResolution())
        actual = summaries_of(table)
        assert list(actual) == DEFAULT_SUMMARIES
        for name, column in actual.items():
            np.testing.assert_array_equal(column, expected[name], err_msg=name)

    def test_another_resolution_and_irregular_moments(self) -> None:
        resolution = EvidenceResolution(
            step=timedelta(minutes=10),
            summary_windows=(timedelta(minutes=20), timedelta(minutes=50)),
        )
        source = recording(bursty_events(4))
        moments = grid(60, T0 - timedelta(minutes=3), timedelta(minutes=7))
        table = build(source, moments, summary_set(resolution))
        expected = reference_summaries(source, moments, resolution)
        for name, column in summaries_of(table).items():
            assert name.endswith(("_20m", "_50m"))
            np.testing.assert_array_equal(column, expected[name], err_msg=name)

    def test_order_inside_a_step_is_never_used(self) -> None:
        t = T0 + timedelta(hours=1)
        inside = t - timedelta(minutes=12)
        first = recording(
            [
                ("Kitchen", T0),
                ("Kitchen", inside),
                ("LivingRoom", inside + timedelta(minutes=1)),
            ]
        )
        second = recording(
            [
                ("Kitchen", T0),
                ("LivingRoom", inside),
                ("Kitchen", inside + timedelta(minutes=1)),
            ]
        )
        np.testing.assert_array_equal(
            build(first, [t]).values, build(second, [t]).values
        )


class TestNoFutureLeakage:
    def test_each_row_depends_only_on_observations_up_to_its_moment(self) -> None:
        source = recording(bursty_events(5))
        moments = grid(90)
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
            np.testing.assert_array_equal(
                build(truncated, [moment]).values[0], full.values[row]
            )

    def test_future_events_change_no_earlier_row(self) -> None:
        base = bursty_events(6)
        moments = grid(90)
        original = build(recording(base), moments)
        cut = 45
        future = [
            (sensor, moments[cut] + timedelta(seconds=s))
            for s in range(1, 3600, 37)
            for sensor in ("Bathroom", "FrontDoor", "Kitchen")
        ]
        extended = build(recording(base + future), moments)
        np.testing.assert_array_equal(
            extended.values[: cut + 1], original.values[: cut + 1]
        )
        assert not np.array_equal(
            extended.values[cut + 1 :], original.values[cut + 1 :]
        )

    def test_windows_are_right_closed(self) -> None:
        t = T0 + timedelta(hours=4)
        hour = timedelta(minutes=60)
        source = recording(
            [
                ("Bedroom", T0),
                ("Kitchen", t - hour),
                ("Bathroom", t - hour + timedelta(microseconds=1)),
                ("LivingRoom", t),
                ("Hall", t + timedelta(microseconds=1)),
            ]
        )
        table = build(source, [t])
        assert value(table, "history_count_kitchen_motion_60m") == 0
        assert value(table, "history_count_kitchen_motion_180m") == 1
        assert value(table, "history_count_bathroom_motion_60m") == 1
        assert value(table, "history_count_living_motion_60m") == 1
        assert value(table, "history_quiet_minutes_living_motion_180m") == 0
        assert value(table, "history_count_hall_motion_60m") == 0
        assert value(table, "history_quiet_minutes_hall_motion_180m") == 180


class TestHouseholdIsolation:
    def test_panel_tables_equal_single_household_builds(self) -> None:
        homes = {
            "hh002": recording(bursty_events(7)),
            "hh001": recording(bursty_events(8)),
        }
        moments = {home: grid(80) for home in homes}
        panel = build_panel_features(homes, summary_set(), moments)
        for home, table in panel.items():
            alone = build(homes[home], moments[home], household=home)
            np.testing.assert_array_equal(table.values, alone.values)

    def test_a_lookback_never_reaches_into_another_household(self) -> None:
        later = T0 + timedelta(hours=1)
        early = recording(
            [("Kitchen", T0 + timedelta(minutes=k)) for k in range(1, 60, 4)]
        )
        late = recording([("Bedroom", later)])
        panel = build_panel_features(
            {"early": early, "late": late},
            summary_set(),
            {"early": [later], "late": [later]},
        )
        assert value(panel["early"], "history_count_kitchen_motion_60m") == 15
        late_row = panel["late"]
        assert math.isnan(value(late_row, "history_count_kitchen_motion_60m"))
        assert math.isnan(value(late_row, "history_quiet_minutes_kitchen_motion_180m"))
        assert value(late_row, "history_last_room_bedroom_180m") == 1
        assert value(late_row, "history_last_room_kitchen_180m") == 0


class TestDeterminism:
    def test_repeated_builds_are_identical(self) -> None:
        source = recording(bursty_events(9))
        np.testing.assert_array_equal(
            build(source, grid(80)).values, build(source, grid(80)).values
        )

    def test_observation_order_does_not_matter(self) -> None:
        source = recording(bursty_events(10))
        shuffled = list(source.observations)
        random.Random(0).shuffle(shuffled)
        reordered = CasasRecording(source.registry, tuple(shuffled), ())
        np.testing.assert_array_equal(
            build(source, grid(80)).values, build(reordered, grid(80)).values
        )

    def test_smaller_sets_are_exact_projections(self) -> None:
        source = recording(bursty_events(11))
        moments = grid(80)
        largest = build(source, moments, InformationSet("all", frozenset({C, H, R, S})))
        for information_set in (*nested_information_sets(), summary_set()):
            table = build(source, moments, information_set)
            indices = [largest.columns.index(name) for name in table.columns]
            assert indices == sorted(indices)
            np.testing.assert_array_equal(table.values, largest.values[:, indices])


class TestSparseAndMissing:
    def test_a_quiet_lookback_inside_the_recording_is_explicit(self) -> None:
        t = T0 + timedelta(hours=5)
        table = build(recording([("Kitchen", T0)]), [t])
        for name in DEFAULT_SUMMARIES:
            feature = name.split("_")[1]
            expected = {"count": 0, "room": 0, "quiet": 180, "last": 0}[feature]
            assert value(table, name) == expected, name

    def test_after_the_last_observation_nothing_is_missing(self) -> None:
        table = build(recording([("Kitchen", T0)]), grid(3, T0 + timedelta(hours=4)))
        assert not np.isnan(table.values).any()

    def test_a_lookback_before_the_recording_is_missing(self) -> None:
        t = T0 + timedelta(minutes=30)
        table = build(recording([("Kitchen", T0), ("Bedroom", t)]), [t])
        for window in WINDOWS:
            assert math.isnan(value(table, f"history_count_kitchen_motion_{window}m"))
            assert math.isnan(value(table, f"history_room_changes_{window}m"))
        assert value(table, "history_quiet_minutes_kitchen_motion_180m") == 30
        assert value(table, "history_quiet_minutes_bedroom_motion_180m") == 0
        assert math.isnan(value(table, "history_quiet_minutes_living_motion_180m"))
        assert value(table, "history_last_room_bedroom_180m") == 1
        assert value(table, "history_last_room_kitchen_180m") == 0

    def test_a_lookback_becomes_observed_once_its_oldest_step_is(self) -> None:
        source = recording([("Kitchen", T0)])
        before, at = T0 + timedelta(minutes=54), T0 + timedelta(minutes=55)
        table = build(source, [before, at])
        column = table.column("history_count_kitchen_motion_60m")
        assert math.isnan(column[0]) and column[1] == 1

    def test_uninstrumented_channels_and_rooms_are_missing(self) -> None:
        source = recording(bursty_events(12, PARTIAL), PARTIAL)
        table = build(source, grid(10, T0 + timedelta(hours=4)))
        assert set(table.uninstrumented) == {
            EvidenceChannel("bathroom", Modality.MOTION),
            EvidenceChannel("hall", Modality.MOTION),
        }
        for name in (
            "history_count_bathroom_motion_60m",
            "history_count_hall_motion_180m",
            "history_quiet_minutes_bathroom_motion_180m",
            "history_quiet_minutes_hall_motion_180m",
            "history_last_room_bathroom_180m",
        ):
            assert np.isnan(table.column(name)).all(), name
        for name in (
            "history_count_hall_door_60m",
            "history_last_room_hall_180m",
            "history_room_changes_60m",
        ):
            assert not np.isnan(table.column(name)).any(), name

    def test_a_silent_installed_sensor_is_censored_not_missing(self) -> None:
        table = build(recording([("Kitchen", T0)]), [T0 + timedelta(hours=4)])
        assert value(table, "history_count_bathroom_motion_180m") == 0
        assert value(table, "history_quiet_minutes_bathroom_motion_180m") == 180

    def test_a_household_with_no_event_channel_has_no_summaries(self) -> None:
        thermometer = SensorSpec(
            "T1", Modality.ENVIRONMENTAL, unit=Unit.CELSIUS, kind=ObservationKind.SAMPLE
        )
        source = CasasRecording(
            SensorRegistry.from_specs([thermometer]),
            (
                Observation(
                    T0, "T1", Modality.ENVIRONMENTAL, ObservationKind.SAMPLE, 20.0
                ),
            ),
            (),
        )
        table = build(source, [T0 + timedelta(hours=4)])
        assert np.isnan(table.values).all()


class TestMeaning:
    def test_quiet_minutes_count_whole_quiet_steps(self) -> None:
        t = T0 + timedelta(hours=4)
        cases = {
            timedelta(0): 0,
            timedelta(minutes=4, seconds=59): 0,
            timedelta(minutes=5): 5,
            timedelta(minutes=7): 5,
            timedelta(minutes=179): 175,
            timedelta(minutes=180): 180,
        }
        for ago, quiet in cases.items():
            table = build(recording([("Bedroom", T0), ("Kitchen", t - ago)]), [t])
            assert (
                value(table, "history_quiet_minutes_kitchen_motion_180m") == quiet
            ), ago

    def test_last_room_marks_every_room_active_in_the_latest_active_step(self) -> None:
        t = T0 + timedelta(hours=4)
        source = recording(
            [
                ("Bedroom", T0),
                ("Kitchen", t - timedelta(minutes=40)),
                ("Bathroom", t - timedelta(minutes=12)),
                ("LivingRoom", t - timedelta(minutes=11)),
                ("FrontDoor", t - timedelta(minutes=30)),
            ]
        )
        table = build(source, [t])
        marked = {
            room
            for room in EvidenceResolution().rooms
            if value(table, f"history_last_room_{room}_180m") == 1
        }
        assert marked == {"bathroom", "living"}

    def test_a_door_is_its_room(self) -> None:
        t = T0 + timedelta(hours=4)
        table = build(
            recording([("Bedroom", T0), ("FrontDoor", t - timedelta(minutes=50))]), [t]
        )
        assert value(table, "history_last_room_hall_180m") == 1
        assert value(table, "history_quiet_minutes_hall_door_180m") == 50
        assert value(table, "history_quiet_minutes_hall_motion_180m") == 180

    def test_zero_valued_events_are_not_activations(self) -> None:
        t = T0 + timedelta(hours=4)
        opened = recording([("Bedroom", T0)])
        closed = Observation(
            t - timedelta(minutes=1),
            "FrontDoor",
            Modality.DOOR,
            ObservationKind.EVENT,
            0.0,
        )
        source = CasasRecording(opened.registry, (*opened.observations, closed), ())
        table = build(source, [t])
        assert value(table, "history_count_hall_door_60m") == 0
        assert value(table, "history_quiet_minutes_hall_door_180m") == 180
        assert value(table, "history_last_room_hall_180m") == 0

    def test_room_changes_between_active_steps(self) -> None:
        t = T0 + timedelta(hours=4)
        minutes = timedelta(minutes=1)
        source = recording(
            [
                ("Bedroom", T0),
                ("Bedroom", t - 100 * minutes),  # outside the hour
                ("Kitchen", t - 52 * minutes),
                ("LivingRoom", t - 27 * minutes),  # after 20 quiet minutes
                ("LivingRoom", t - 22 * minutes),
                ("FrontDoor", t - 17 * minutes),
                ("Hall", t - 16 * minutes),
                ("Hall", t - 12 * minutes),  # door and motion: same room
                ("Bathroom", t - 3 * minutes),
                ("Bedroom", t - 2 * minutes),  # two rooms in one step
            ]
        )
        table = build(source, [t])
        # kitchen > living > living > hall > hall > {bathroom, bedroom}
        assert value(table, "history_room_changes_60m") == 3
        # plus bedroom at t - 100 > kitchen; T0 is outside both windows
        assert value(table, "history_room_changes_180m") == 4


def simulated(seed: int) -> CasasRecording:
    sim = simulate(HouseholdConfig(days=2, seed=seed))
    return CasasRecording(
        sim.registry,
        sim.observations,
        tuple(
            ActivityInterval(e.state.value, e.start, e.end, e.state)
            for e in sim.truth.episodes
        ),
    )


class TestMatchedRunner:
    def test_the_gap_to_history_summaries_is_a_nested_comparison(self) -> None:
        diagnostic = nested_information_sets()[-1]
        extended = InformationSet(
            "current+time_of_day+history+history_summary",
            frozenset({C, H, R, S}),
            diagnostic.resolution,
        )
        homes = {f"sim{s}": simulated(s) for s in range(1, 4)}
        split = HouseholdSplit("s", train=("sim1", "sim2"), test=("sim3",))
        runs = [
            run_matched_evaluation(
                homes,
                split=split,
                information_set=information_set,
                models=baseline_suite(information_set),
                seed=0,
                data_source="simulator",
            )
            for information_set in (diagnostic, extended)
        ]
        assert [spec.name for spec in baseline_suite(extended)] == [
            "state_frequency",
            "persistence",
            "logistic",
            "tree",
        ]
        assert runs[1].information_set.to_dict()["summary_windows_seconds"] == [
            3600.0,
            10800.0,
        ]
        comparison = compare_information_sets(
            runs[0],
            runs[1],
            model="logistic",
            metric="balanced_accuracy",
            resamples=100,
        )
        assert comparison.households == ("sim3",)
