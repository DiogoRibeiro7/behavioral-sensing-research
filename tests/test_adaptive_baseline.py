"""Tests for daily behavioural features and the adaptive personal baseline."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from sensor_modeling.baseline import (
    AdaptiveBaseline,
    BaselineConfig,
    ChangeKind,
    feature_series,
    summarise_days,
)
from sensor_modeling.fusion.estimate import StateEstimate
from sensor_modeling.states import BehaviouralState, StateOntology

LISBON = ZoneInfo("Europe/Lisbon")
S = BehaviouralState
DAY_ONE = date(2024, 1, 1)


def estimate(
    at: datetime,
    state: BehaviouralState,
    *,
    confidence: float = 0.95,
    completeness: float = 1.0,
    min_completeness: float = 0.25,
) -> StateEstimate:
    """Build a state estimate concentrated on *state*."""
    ontology = StateOntology()
    belief = np.full(ontology.size, (1.0 - confidence) / (ontology.size - 1))
    belief[ontology.index(state)] = confidence
    return StateEstimate(
        at=at,
        ontology=ontology,
        belief=belief,
        evidence=(),
        completeness=completeness,
        min_confidence=0.35,
        min_completeness=min_completeness,
    )


def constant_day(
    day_start: datetime, state: BehaviouralState, *, completeness: float = 1.0
) -> list[StateEstimate]:
    """A full day of quarter-hourly estimates in one state."""
    return [
        estimate(
            day_start + timedelta(minutes=15 * step), state, completeness=completeness
        )
        for step in range(97)
    ]


class TestDailyFeatures:
    def test_a_full_day_in_one_state_accumulates_its_hours(self) -> None:
        start = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
        summaries = summarise_days(constant_day(start, S.SLEEPING))
        assert len(summaries) == 1
        # The posterior is 0.95 on sleeping, so expected hours fall short of 24.
        assert summaries[0].hours_in(S.SLEEPING) == pytest.approx(24 * 0.95, abs=0.1)
        assert summaries[0].observed == pytest.approx(1.0, abs=0.01)

    def test_expected_hours_use_the_posterior_not_the_argmax(self) -> None:
        """A day of hesitant guesses must not look like a day of certainties."""
        start = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
        confident = summarise_days(
            [
                estimate(start + timedelta(minutes=15 * i), S.SLEEPING, confidence=0.99)
                for i in range(97)
            ]
        )
        hesitant = summarise_days(
            [
                estimate(start + timedelta(minutes=15 * i), S.SLEEPING, confidence=0.40)
                for i in range(97)
            ]
        )
        assert hesitant[0].hours_in(S.SLEEPING) < confident[0].hours_in(S.SLEEPING)

    def test_days_are_bounded_by_the_local_clock(self) -> None:
        start = datetime(2024, 5, 1, 22, 0, tzinfo=LISBON)
        estimates = [
            estimate(start + timedelta(minutes=30 * step), S.SLEEPING)
            for step in range(10)
        ]
        summaries = summarise_days(estimates, tz=LISBON)
        assert [s.day for s in summaries] == [date(2024, 5, 1), date(2024, 5, 2)]

    def test_a_dst_day_is_not_assumed_to_be_24_hours(self) -> None:
        """Lisbon's spring-forward day is 23 hours long, so a fully observed
        day reports more than 24 hours of coverage relative to a nominal day."""
        start = datetime(2024, 3, 31, 0, 0, tzinfo=LISBON)
        estimates = [
            estimate(
                (
                    start.astimezone(timezone.utc) + timedelta(minutes=15 * step)
                ).astimezone(LISBON),
                S.SLEEPING,
            )
            for step in range(93)
        ]
        summaries = summarise_days(estimates, tz=LISBON)
        short_day = next(s for s in summaries if s.day == date(2024, 3, 31))
        assert short_day.observed == pytest.approx(23 / 24, abs=0.02)

    def test_long_gaps_are_not_attributed_to_the_last_known_state(self) -> None:
        start = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
        estimates = [
            estimate(start, S.SLEEPING),
            estimate(start + timedelta(hours=10), S.SLEEPING),
        ]
        summaries = summarise_days(estimates, max_interval=timedelta(hours=1))
        assert summaries == []

    def test_coverage_and_abstention_are_recorded(self) -> None:
        start = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
        summaries = summarise_days(constant_day(start, S.SLEEPING, completeness=0.4))
        assert summaries[0].coverage == pytest.approx(0.4, abs=0.01)
        assert summaries[0].abstention == pytest.approx(0.0, abs=0.01)
        assert not summaries[0].is_usable()

    def test_abstained_time_is_counted(self) -> None:
        start = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
        estimates = [
            estimate(
                start + timedelta(minutes=15 * step),
                S.SLEEPING,
                completeness=0.1,
                min_completeness=0.5,
            )
            for step in range(97)
        ]
        summaries = summarise_days(estimates)
        assert summaries[0].abstention == pytest.approx(1.0, abs=0.01)

    def test_out_of_order_estimates_are_rejected(self) -> None:
        start = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="non-decreasing"):
            summarise_days(
                [
                    estimate(start + timedelta(hours=1), S.SLEEPING),
                    estimate(start, S.SLEEPING),
                ]
            )

    def test_empty_input_yields_no_summaries(self) -> None:
        assert summarise_days([]) == []

    def test_feature_series_drops_poorly_observed_days(self) -> None:
        start = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
        good = constant_day(start, S.SLEEPING)
        bad = constant_day(start + timedelta(days=1), S.SLEEPING, completeness=0.1)
        days, values = feature_series(summarise_days(good + bad), S.SLEEPING)
        assert days == [date(2024, 5, 1)]
        assert len(values) == 1


def feed(
    baseline: AdaptiveBaseline, values: list[float], start: date = DAY_ONE
) -> list:
    """Feed a run of daily values and return the verdicts."""
    return [
        baseline.observe(start + timedelta(days=offset), value)
        for offset, value in enumerate(values)
    ]


class TestAdaptiveBaseline:
    def test_early_days_are_reported_as_insufficient_data(self) -> None:
        baseline = AdaptiveBaseline("sleep_hours", BaselineConfig(min_samples=10))
        verdicts = feed(baseline, [8.0] * 5)
        assert all(v.kind is ChangeKind.INSUFFICIENT_DATA for v in verdicts)

    def test_a_stable_routine_stays_ordinary(self) -> None:
        baseline = AdaptiveBaseline(
            "sleep_hours", BaselineConfig(min_samples=10, weekday_min_samples=99)
        )
        wobble = [0.0, 0.3, -0.2, 0.1, -0.3, 0.2, -0.1]
        verdicts = feed(baseline, [8.0 + wobble[i % 7] for i in range(40)])
        assert all(v.kind is ChangeKind.ORDINARY for v in verdicts[10:])

    def test_a_single_unusual_day_is_a_temporary_disturbance(self) -> None:
        config = BaselineConfig(
            min_samples=10, weekday_min_samples=99, persistence_days=3, min_scale=0.1
        )
        baseline = AdaptiveBaseline("sleep_hours", config)
        wobble = [0.0, 0.1, -0.1, 0.2, -0.2]
        feed(baseline, [8.0 + wobble[i % 5] for i in range(20)])
        verdict = baseline.observe(DAY_ONE + timedelta(days=20), 2.0)
        assert verdict.kind is ChangeKind.TEMPORARY_DISTURBANCE
        assert verdict.direction == "decrease"

    def test_a_sustained_shift_becomes_a_persistent_change(self) -> None:
        config = BaselineConfig(
            min_samples=10, weekday_min_samples=99, persistence_days=3, min_scale=0.1
        )
        baseline = AdaptiveBaseline("sleep_hours", config)
        wobble = [0.0, 0.1, -0.1, 0.2, -0.2]
        feed(baseline, [8.0 + wobble[i % 5] for i in range(20)])
        verdicts = [
            baseline.observe(DAY_ONE + timedelta(days=20 + offset), 3.0)
            for offset in range(5)
        ]
        assert verdicts[0].kind is ChangeKind.TEMPORARY_DISTURBANCE
        assert verdicts[-1].kind in {
            ChangeKind.PERSISTENT_CHANGE,
            ChangeKind.ABRUPT_CHANGE,
        }
        assert verdicts[-1].is_change
        assert verdicts[-1].duration_days >= 3

    def test_a_persistent_shift_is_located_as_an_abrupt_change(self) -> None:
        config = BaselineConfig(
            min_samples=10,
            weekday_min_samples=99,
            persistence_days=3,
            min_scale=0.1,
            change_point_penalty=2.0,
        )
        baseline = AdaptiveBaseline("sleep_hours", config)
        wobble = [0.0, 0.1, -0.1, 0.2, -0.2]
        feed(baseline, [8.0 + wobble[i % 5] for i in range(25)])
        verdicts = [
            baseline.observe(DAY_ONE + timedelta(days=25 + offset), 3.0)
            for offset in range(6)
        ]
        located = [v for v in verdicts if v.kind is ChangeKind.ABRUPT_CHANGE]
        assert located
        assert located[-1].change_point is not None
        assert located[-1].change_point >= DAY_ONE + timedelta(days=20)

    def test_a_slow_trend_is_reported_as_gradual_drift(self) -> None:
        config = BaselineConfig(
            min_samples=10,
            weekday_min_samples=99,
            trend_window=28,
            trend_threshold=1.5,
            deviation_threshold=6.0,
            min_scale=0.1,
        )
        baseline = AdaptiveBaseline("sleep_hours", config)
        verdicts = feed(baseline, [8.0 - 0.06 * day for day in range(40)])
        assert any(v.kind is ChangeKind.GRADUAL_DRIFT for v in verdicts[-10:])

    def test_weekly_rhythm_is_not_reported_as_change(self) -> None:
        """Sundays are compared against Sundays, not against the working week."""
        config = BaselineConfig(
            min_samples=10, weekday_min_samples=4, persistence_days=2, min_scale=0.1
        )
        baseline = AdaptiveBaseline("kitchen_hours", config)
        # Monday-start series where weekends are reliably much quieter.
        start = date(2024, 1, 1)  # a Monday
        values = [
            1.0 if (start + timedelta(days=day)).weekday() < 5 else 4.0
            for day in range(70)
        ]
        verdicts = feed(baseline, values, start)
        settled = verdicts[35:]
        assert not any(v.is_change for v in settled)

    def test_the_weekday_reference_activates_once_there_is_enough_history(self) -> None:
        config = BaselineConfig(min_samples=5, weekday_min_samples=3)
        baseline = AdaptiveBaseline("sleep_hours", config)
        feed(baseline, [8.0] * 30)
        reference = baseline.reference(DAY_ONE + timedelta(days=30))
        assert reference.weekday_aware
        assert reference.weekday_samples >= 3

    def test_a_skipped_day_never_enters_the_history(self) -> None:
        """A sensor outage must not redefine what normal looks like."""
        config = BaselineConfig(min_samples=5, weekday_min_samples=99, min_scale=0.1)
        baseline = AdaptiveBaseline("sleep_hours", config)
        feed(baseline, [8.0] * 20)
        before = baseline.reference(DAY_ONE + timedelta(days=20))

        verdict = baseline.skip(DAY_ONE + timedelta(days=20))
        after = baseline.reference(DAY_ONE + timedelta(days=21))

        assert verdict.kind is ChangeKind.INSUFFICIENT_DATA
        assert math.isnan(verdict.value)
        assert baseline.samples == 20
        assert after.centre == before.centre

    def test_the_baseline_adapts_to_a_genuinely_new_normal(self) -> None:
        config = BaselineConfig(
            min_samples=10, weekday_min_samples=99, history_days=30, min_scale=0.1
        )
        baseline = AdaptiveBaseline("sleep_hours", config)
        feed(baseline, [8.0] * 30)
        assert baseline.reference().centre == pytest.approx(8.0)
        feed(baseline, [5.0] * 40, DAY_ONE + timedelta(days=30))
        assert baseline.reference().centre == pytest.approx(5.0)

    def test_history_is_bounded(self) -> None:
        baseline = AdaptiveBaseline("sleep_hours", BaselineConfig(history_days=20))
        feed(baseline, [8.0] * 100)
        assert baseline.samples == 20

    def test_days_must_be_strictly_increasing(self) -> None:
        baseline = AdaptiveBaseline("sleep_hours")
        baseline.observe(DAY_ONE, 8.0)
        with pytest.raises(ValueError, match="strictly increasing"):
            baseline.observe(DAY_ONE, 8.0)

    def test_non_finite_values_are_rejected(self) -> None:
        baseline = AdaptiveBaseline("sleep_hours")
        with pytest.raises(ValueError, match="finite"):
            baseline.observe(DAY_ONE, float("nan"))

    def test_snapshot_and_restore_preserve_the_reference(self) -> None:
        baseline = AdaptiveBaseline("sleep_hours", BaselineConfig(min_samples=5))
        feed(baseline, [8.0, 8.5, 7.5, 8.2, 7.8, 8.1])
        state = baseline.snapshot()

        restarted = AdaptiveBaseline("sleep_hours", BaselineConfig(min_samples=5))
        restarted.restore(state)
        assert restarted.samples == baseline.samples
        assert restarted.reference().centre == baseline.reference().centre

    def test_restore_rejects_mismatched_payloads(self) -> None:
        baseline = AdaptiveBaseline("sleep_hours")
        with pytest.raises(ValueError, match="same length"):
            baseline.restore({"days": ["2024-01-01"], "values": []})

    def test_verdicts_serialise(self) -> None:
        baseline = AdaptiveBaseline("sleep_hours", BaselineConfig(min_samples=2))
        payload = feed(baseline, [8.0, 8.1, 8.2])[-1].to_dict()
        assert payload["feature"] == "sleep_hours"
        assert payload["kind"] in {kind.value for kind in ChangeKind}
        assert "reference" in payload

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"history_days": 1},
            {"min_samples": 0},
            {"weekday_min_samples": 0},
            {"deviation_threshold": 0.0},
            {"persistence_days": 0},
            {"trend_window": 2},
            {"trend_threshold": 0.0},
            {"change_point_penalty": 0.0},
            {"min_scale": 0.0},
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            BaselineConfig(**kwargs)
