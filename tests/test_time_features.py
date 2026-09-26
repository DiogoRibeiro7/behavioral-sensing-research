"""Tests for time-of-day features: clock semantics, encodings and interpretation.

The DST cases use America/Los_Angeles on 2011-03-13 (clocks jump from 02:00 to
03:00) and 2011-11-06 (clocks fall back from 02:00 to 01:00), dates inside the
CASAS recording period.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    HouseholdSplit,
    LogisticBaseline,
    ModelSpec,
    build_feature_table,
    nested_information_sets,
    run_matched_evaluation,
)
from sensor_modeling.datasets.baseline_models import encode_rows
from sensor_modeling.datasets.matched_evaluation import FeatureRows
from sensor_modeling.datasets.time_features import (
    MAX_HARMONICS,
    cyclic_hour_columns,
    cyclic_hour_features,
    hour_angle,
    local_hour,
    peak_hour,
)
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
)
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S

LA = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc
HOURS = np.arange(24)


class TestLocalHour:
    @pytest.mark.parametrize(
        ("moment", "hour"),
        [
            (datetime(2024, 3, 1, 23, 59, 59, 999999, tzinfo=LA), 23),
            (datetime(2024, 3, 2, 0, 0, tzinfo=LA), 0),
            (datetime(2024, 3, 2, 0, 0, 1, tzinfo=LA), 0),
        ],
    )
    def test_midnight_boundaries(self, moment: datetime, hour: int) -> None:
        assert local_hour(moment, LA) == hour

    def test_an_instant_in_another_zone_is_read_on_the_local_clock(self) -> None:
        assert local_hour(datetime(2024, 7, 1, 15, 0, tzinfo=UTC), LA) == 8
        assert local_hour(datetime(2024, 7, 1, 15, 0, tzinfo=UTC), UTC) == 15

    def test_a_naive_moment_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            local_hour(datetime(2024, 7, 1, 15, 0), LA)

    def test_spring_forward_instants_never_read_hour_two(self) -> None:
        instants = [
            datetime(2011, 3, 13, 9, 0, tzinfo=UTC) + timedelta(minutes=m)
            for m in range(0, 180, 5)
        ]
        hours = {local_hour(instant, LA) for instant in instants}
        assert 2 not in hours and {1, 3} <= hours

    def test_a_grid_moment_naming_the_skipped_hour_is_read_as_written(self) -> None:
        """The same reading the circadian prior makes with ``at.hour``."""
        grid = datetime(2011, 3, 13, 1, 30, tzinfo=LA) + timedelta(minutes=60)
        assert (grid.hour, grid.minute) == (2, 30)
        assert local_hour(grid, LA) == grid.hour == 2

    def test_both_fall_back_occurrences_read_the_same_clock(self) -> None:
        first = datetime(2011, 11, 6, 8, 30, tzinfo=UTC).astimezone(LA)
        second = datetime(2011, 11, 6, 9, 30, tzinfo=UTC).astimezone(LA)
        assert (first.fold, second.fold) == (0, 1)
        assert local_hour(first, LA) == local_hour(second, LA) == 1
        np.testing.assert_array_equal(
            cyclic_hour_features([local_hour(first, LA)], 2),
            cyclic_hour_features([local_hour(second, LA)], 2),
        )

    def test_the_feature_table_uses_the_same_definition(self) -> None:
        spec = SensorSpec("Kitchen", Modality.MOTION, room="kitchen")
        start = datetime(2011, 11, 6, 0, 0, tzinfo=LA)
        recording = CasasRecording(
            SensorRegistry.from_specs([spec]),
            (
                Observation(
                    start, "Kitchen", Modality.MOTION, ObservationKind.EVENT, 1.0
                ),
            ),
            (),
        )
        moments = [
            datetime(2011, 11, 6, 8, 0, tzinfo=UTC) + timedelta(minutes=30) * k
            for k in range(6)
        ]
        table = build_feature_table(
            recording, nested_information_sets()[1], moments, household="h"
        )
        assert list(table.column("hour_of_day")) == [local_hour(m, LA) for m in moments]


class TestCyclicEncoding:
    def test_columns_are_named_and_ordered(self) -> None:
        assert cyclic_hour_columns(2) == (
            "hour_of_day_sin1",
            "hour_of_day_cos1",
            "hour_of_day_sin2",
            "hour_of_day_cos2",
        )
        assert cyclic_hour_features(HOURS, 2).shape == (24, 4)

    def test_every_adjacent_pair_is_equally_close_including_23_and_0(self) -> None:
        points = cyclic_hour_features(HOURS)
        gaps = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        np.testing.assert_allclose(gaps, gaps[0])
        wrap = np.linalg.norm(points[23] - points[0])
        assert wrap == pytest.approx(np.linalg.norm(points[0] - points[1]))
        assert np.linalg.norm(points[0] - points[12]) == pytest.approx(2.0)

    def test_hours_are_the_midpoints_on_the_circle(self) -> None:
        np.testing.assert_allclose(
            hour_angle([0, 6, 23]), [np.pi / 24, 13 * np.pi / 24, 47 * np.pi / 24]
        )
        points = cyclic_hour_features(HOURS, 3)
        for k in range(3):
            np.testing.assert_allclose(
                points[:, 2 * k] ** 2 + points[:, 2 * k + 1] ** 2, 1.0
            )

    def test_the_kth_harmonic_repeats_k_times_a_day(self) -> None:
        points = cyclic_hour_features(HOURS, 3)
        second, third = points[:, 2:4], points[:, 4:6]
        np.testing.assert_allclose(second[:12], second[12:], atol=1e-12)
        np.testing.assert_allclose(third[:8], third[8:16], atol=1e-12)
        assert not np.allclose(points[:12, 0:2], points[12:, 0:2])

    def test_one_harmonic_keeps_every_hour_distinct(self) -> None:
        points = np.round(cyclic_hour_features(HOURS), 12)
        assert len({tuple(point) for point in points}) == 24

    def test_the_encoding_is_deterministic(self) -> None:
        np.testing.assert_array_equal(
            cyclic_hour_features([3, 17], 4), cyclic_hour_features([3, 17], 4)
        )

    @pytest.mark.parametrize("harmonics", [0, MAX_HARMONICS + 1, True, 1.0])
    def test_invalid_harmonics_are_refused(self, harmonics: object) -> None:
        with pytest.raises(ValueError, match="harmonics"):
            cyclic_hour_features(HOURS, harmonics)  # type: ignore[arg-type]

    @pytest.mark.parametrize("hours", [[24], [-1], [1.5], [math.nan], [[1, 2]]])
    def test_invalid_hours_are_refused(self, hours: list[object]) -> None:
        with pytest.raises(ValueError):
            cyclic_hour_features(hours)  # type: ignore[arg-type]


class TestInterpretation:
    @pytest.mark.parametrize("hour", range(24))
    def test_a_score_aligned_with_an_hour_peaks_at_that_hour(self, hour: int) -> None:
        sin_1, cos_1 = cyclic_hour_features([hour])[0]
        assert peak_hour(3.0 * sin_1, 3.0 * cos_1) == pytest.approx(hour)

    def test_the_peak_is_where_the_score_is_largest(self) -> None:
        beta, gamma = 0.8, -1.3
        scores = cyclic_hour_features(np.arange(24)) @ np.array([beta, gamma])
        assert abs(peak_hour(beta, gamma) - int(np.argmax(scores))) <= 0.5

    def test_a_flat_score_has_no_peak(self) -> None:
        with pytest.raises(ValueError, match="no peak"):
            peak_hour(0.0, 0.0)


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


class TestLogisticOption:
    def test_the_default_is_unchanged(self) -> None:
        assert LogisticBaseline().configuration()["hour_of_day"] == "one-hot"
        assert "hour_harmonics" not in LogisticBaseline().configuration()

    def test_cyclic_hours_replace_the_hour_column_only(self) -> None:
        rows = FeatureRows(
            ("events_kitchen_motion_lag0", "hour_of_day"),
            np.array([[1.0, 23.0], [0.0, 0.0]]),
            (S.SLEEPING, S.AWAY),
        )
        design = encode_rows(rows, hour="cyclic", harmonics=2)
        assert design.shape == (2, 2 + 4)
        np.testing.assert_array_equal(design[:, 2:], cyclic_hour_features([23, 0], 2))
        with pytest.raises(ValueError, match="hour encoding"):
            encode_rows(rows, hour="circular")

    @pytest.mark.parametrize(
        "settings", [{"hour_encoding": "ordinal"}, {"harmonics": 0}, {"harmonics": 12}]
    )
    def test_invalid_settings_are_refused(self, settings: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            LogisticBaseline(**settings)  # type: ignore[arg-type]

    def test_it_runs_under_the_matched_runner(self) -> None:
        information_set = nested_information_sets()[1]
        result = run_matched_evaluation(
            {f"sim{s}": simulated(s) for s in range(1, 4)},
            split=HouseholdSplit("s", train=("sim1", "sim2"), test=("sim3",)),
            information_set=information_set,
            models=[
                ModelSpec(
                    "cyclic",
                    lambda seed: LogisticBaseline(
                        seed=seed, hour_encoding="cyclic", harmonics=2
                    ),
                    LogisticBaseline(
                        hour_encoding="cyclic", harmonics=2
                    ).configuration(),
                )
            ],
            seed=0,
            data_source="simulator",
        )
        assert set(result.household_metrics["cyclic"]) == {"sim3"}
        assert result.models[0].configuration["hour_harmonics"] == 2
