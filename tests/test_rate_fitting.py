"""Tests for measuring and fitting emission rates from annotated recordings.

The package ships declared rates, reasoned about rather than measured. These
cover the machinery that checks them against reality, and the guard rails that
stop a measurement being mistaken for a validated fit.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sensor_modeling.datasets import (
    fit_emission_defaults,
    measure_event_rates,
    pooled_rate_report,
    read_casas_hh,
)
from sensor_modeling.states import BehaviouralState

UTC = timezone.utc


def recording(kitchen_events: int, sleep_events: int) -> object:
    """A day with a cooking burst and a quiet night, at controlled densities."""
    start = datetime(2011, 6, 15, 0, 0)
    lines: list[str] = []

    def stamp(minute: float) -> str:
        # date and time are separate comma-separated fields in this export
        return (start + timedelta(minutes=minute)).strftime("%Y-%m-%d,%H:%M:%S.%f")

    # Sleeping 00:00-06:00, six hours.
    lines.append(f'{stamp(0)},Bedroom,ON,Sleep="begin"')
    for i in range(sleep_events):
        lines.append(f"{stamp(10 + i * 5)},Bedroom,ON")
    lines.append(f'{stamp(360)},Bedroom,OFF,Sleep="end"')

    # Cooking 07:00-08:00, one hour.
    lines.append(f'{stamp(420)},Kitchen,ON,Cook_Breakfast="begin"')
    for i in range(kitchen_events):
        lines.append(f"{stamp(421 + i * 0.5)},Kitchen,ON")
    lines.append(f'{stamp(480)},Kitchen,OFF,Cook_Breakfast="end"')

    return read_casas_hh(sorted(lines), timezone=UTC)


class TestMeasurement:
    def test_a_denser_burst_measures_a_higher_rate(self) -> None:
        low = measure_event_rates(recording(kitchen_events=10, sleep_events=2))
        high = measure_event_rates(recording(kitchen_events=60, sleep_events=2))

        kitchen = BehaviouralState.KITCHEN_ACTIVITY
        assert (
            high.for_state(kitchen).median_in_room
            > low.for_state(kitchen).median_in_room
        )

    def test_rates_are_per_hour_not_per_interval(self) -> None:
        """A count divided by a duration, so a longer interval dilutes it."""
        report = measure_event_rates(recording(kitchen_events=60, sleep_events=6))

        # Sixty events in one hour of cooking against six in six hours of sleep.
        kitchen = report.for_state(BehaviouralState.KITCHEN_ACTIVITY)
        sleeping = report.for_state(BehaviouralState.SLEEPING)
        assert kitchen.median_in_room > 10 * sleeping.median_in_room

    def test_states_without_a_room_still_get_a_deployment_rate(self) -> None:
        """`HOME_INACTIVE` has no room, and is the state most in question."""
        report = measure_event_rates(recording(kitchen_events=20, sleep_events=4))
        sleeping = report.for_state(BehaviouralState.SLEEPING)

        assert sleeping.median_overall is not None

    def test_a_recording_with_no_observations_is_refused(self) -> None:
        empty = recording(kitchen_events=1, sleep_events=1)
        object.__setattr__(empty, "observations", ())

        with pytest.raises(ValueError, match="no observations"):
            measure_event_rates(empty)

    def test_comparison_reports_the_ratio_against_a_declared_rate(self) -> None:
        """The ratio is the point: an order of magnitude changes which state wins."""
        report = measure_event_rates(recording(kitchen_events=60, sleep_events=2))
        comparison = report.compare(active_rate=40.0, idle_rate=0.15)

        kitchen = next(
            row for row in comparison["states"] if row["state"] == "kitchen_activity"
        )
        assert kitchen["ratio"] is not None
        assert kitchen["ratio"] > 1.0


class TestFitting:
    def test_fitted_levels_are_relative_to_the_busiest_state(self) -> None:
        defaults = fit_emission_defaults([recording(60, 2)])

        levels = defaults.activity_levels
        assert levels[BehaviouralState.KITCHEN_ACTIVITY] == pytest.approx(1.0)
        assert (
            levels[BehaviouralState.SLEEPING]
            < levels[BehaviouralState.KITCHEN_ACTIVITY]
        )

    def test_away_is_fitted_to_zero(self) -> None:
        """Absence emits nothing, and that is not something to measure."""
        defaults = fit_emission_defaults([recording(30, 2)])
        assert defaults.activity_levels[BehaviouralState.AWAY] == 0.0

    def test_the_active_rate_follows_the_measurement(self) -> None:
        sparse = fit_emission_defaults([recording(10, 2)])
        dense = fit_emission_defaults([recording(120, 2)])

        assert dense.active_rate > sparse.active_rate

    def test_no_recordings_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one recording"):
            fit_emission_defaults([])

    def test_pooling_needs_at_least_one_recording(self) -> None:
        with pytest.raises(ValueError, match="at least one recording"):
            pooled_rate_report([])

    def test_pooling_combines_samples_across_recordings(self) -> None:
        one = measure_event_rates(recording(30, 2))
        both = pooled_rate_report([recording(30, 2), recording(30, 2)])

        kitchen = BehaviouralState.KITCHEN_ACTIVITY
        assert both.for_state(kitchen).intervals == 2 * one.for_state(kitchen).intervals
