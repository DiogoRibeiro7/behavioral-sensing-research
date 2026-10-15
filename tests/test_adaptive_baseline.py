"""Tests for daily behavioural features and the adaptive personal baseline."""

from __future__ import annotations

import math
import statistics
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from scipy import stats

from sensor_modeling.baseline import (
    MAD_TO_SIGMA,
    AdaptiveBaseline,
    BaselineConfig,
    ChangeKind,
    feature_series,
    summarise_days,
)
from sensor_modeling.baseline.adaptive import (
    DOF_PER_RESIDUAL,
    MAX_EQUIVALENT_DEVIATION,
    medians_without_each,
    normal_equivalent,
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

    def test_a_drift_reports_the_direction_of_its_trend(self) -> None:
        """Regression: direction was read off the day, not the trend.

        A single day inside a slow decline can sit above the reference, so
        deriving the direction from the day let a verdict announce a decrease
        while reporting a rising slope.
        """
        config = BaselineConfig(
            min_samples=10,
            weekday_min_samples=99,
            trend_window=28,
            trend_threshold=1.5,
            deviation_threshold=8.0,
            min_scale=0.1,
        )

        def drift_verdicts(slope: float) -> list:
            baseline = AdaptiveBaseline("sleep_hours", config)
            verdicts = feed(baseline, [8.0 + slope * day for day in range(40)])
            return [v for v in verdicts if v.kind is ChangeKind.GRADUAL_DRIFT]

        falling = drift_verdicts(-0.06)
        rising = drift_verdicts(+0.06)
        assert falling and rising
        assert falling[-1].slope_per_day < 0
        assert falling[-1].direction == "decrease"
        assert rising[-1].slope_per_day > 0
        assert rising[-1].direction == "increase"

    def test_a_drift_carries_the_movement_that_identified_it(self) -> None:
        config = BaselineConfig(
            min_samples=10,
            weekday_min_samples=99,
            trend_window=28,
            trend_threshold=1.5,
            deviation_threshold=8.0,
            min_scale=0.1,
        )
        baseline = AdaptiveBaseline("sleep_hours", config)
        verdicts = feed(baseline, [8.0 - 0.06 * day for day in range(40)])
        drifts = [v for v in verdicts if v.kind is ChangeKind.GRADUAL_DRIFT]
        assert drifts
        assert drifts[-1].trend_strength >= config.trend_threshold

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


CALIBRATED = BaselineConfig(calibrated=True)

#: Four weeks in which the Mondays happen to agree to within a few minutes and
#: the other days do not. Day one is a Monday.
TIGHT_MONDAYS = [
    8.0 + (0.02 * (k // 7) if k % 7 == 0 else ((k * 5) % 14 - 6.5) / 7.0)
    for k in range(28)
]


class TestNormalEquivalent:
    def test_it_is_the_gaussian_value_with_the_same_tail(self) -> None:
        for dof in (1.5, 5.2, 44.0):
            for score in (-12.0, -3.0, -0.2, 0.4, 2.0, 3.0, 8.0):
                tail = stats.t.sf(abs(score), dof)
                expected = math.copysign(stats.norm.isf(tail), score)
                assert normal_equivalent(score, dof) == pytest.approx(expected)

    def test_a_scale_from_a_few_days_makes_a_score_less_unusual(self) -> None:
        assert normal_equivalent(3.0, 5.0) < normal_equivalent(3.0, 20.0) < 3.0
        assert normal_equivalent(3.0, 5.0) == pytest.approx(2.169, abs=1e-3)

    def test_a_well_determined_scale_changes_nothing(self) -> None:
        assert normal_equivalent(3.0, 1e7) == pytest.approx(3.0, abs=1e-5)

    def test_it_keeps_the_sign_and_the_order(self) -> None:
        scores = [-6.0, -2.5, -0.1, 0.0, 0.1, 2.5, 6.0]
        mapped = [normal_equivalent(score, 4.0) for score in scores]
        assert mapped == sorted(mapped)
        assert mapped[3] == 0.0 and math.copysign(1.0, mapped[3]) == 1.0
        assert mapped[0] == pytest.approx(-mapped[-1])

    def test_a_score_beyond_what_a_double_holds_is_capped(self) -> None:
        assert normal_equivalent(1e6, 200.0) == MAX_EQUIVALENT_DEVIATION
        assert normal_equivalent(float("inf"), 5.0) == MAX_EQUIVALENT_DEVIATION
        assert normal_equivalent(float("-inf"), 5.0) == -MAX_EQUIVALENT_DEVIATION

    def test_it_refuses_what_is_not_a_score(self) -> None:
        with pytest.raises(ValueError, match="dof"):
            normal_equivalent(1.0, 0.0)
        with pytest.raises(ValueError, match="number"):
            normal_equivalent(float("nan"), 5.0)


class TestMediansWithoutEach:
    @pytest.mark.parametrize("size", [2, 3, 4, 5, 8, 9, 17])
    def test_it_is_the_median_of_the_others(self, size: int) -> None:
        rng = np.random.default_rng(size)
        for tied in (False, True):
            values = (
                [float(v) for v in rng.integers(0, 4, size=size)]
                if tied
                else [float(v) for v in rng.standard_normal(size)]
            )
            expected = [
                statistics.median(values[:k] + values[k + 1 :]) for k in range(size)
            ]
            assert medians_without_each(values) == pytest.approx(expected)

    def test_one_value_has_no_others(self) -> None:
        assert medians_without_each([]) == []
        assert medians_without_each([3.5]) == [3.5]


class TestCalibratedReference:
    def test_it_is_off_unless_asked_for(self) -> None:
        assert BaselineConfig().calibrated is False
        baseline = AdaptiveBaseline("sleep_hours")
        feed(baseline, TIGHT_MONDAYS)
        reference = baseline.reference(DAY_ONE + timedelta(days=28))
        assert reference.calibrated is False
        assert reference.dof is None and reference.scale_samples == 0
        assert reference.deviation(8.9) == pytest.approx(
            (8.9 - reference.centre) / reference.scale
        )

    def test_the_centre_stays_weekday_aware_and_the_scale_is_pooled(self) -> None:
        plain, calibrated = AdaptiveBaseline("x"), AdaptiveBaseline("x", CALIBRATED)
        feed(plain, TIGHT_MONDAYS)
        feed(calibrated, TIGHT_MONDAYS)
        monday = DAY_ONE + timedelta(days=28)
        before, after = plain.reference(monday), calibrated.reference(monday)
        assert after.weekday_aware and after.weekday_samples == 4
        assert after.centre == before.centre
        assert after.scale_samples == after.samples == 28
        assert after.dof == pytest.approx(DOF_PER_RESIDUAL * 28)
        assert after.calibrated

    def test_four_days_that_agree_no_longer_make_the_fifth_a_deviation(self) -> None:
        """The case the option exists for.

        Against four Mondays that agree to within a few minutes, a Monday 52
        minutes longer is 3.5 of their robust standard deviations out. The
        other days of the month say that an hour either way is ordinary.
        """
        plain, calibrated = AdaptiveBaseline("x"), AdaptiveBaseline("x", CALIBRATED)
        feed(plain, TIGHT_MONDAYS)
        feed(calibrated, TIGHT_MONDAYS)
        monday = DAY_ONE + timedelta(days=28)
        assert plain.observe(monday, 8.9).kind is ChangeKind.TEMPORARY_DISTURBANCE
        verdict = calibrated.observe(monday, 8.9)
        assert verdict.kind is ChangeKind.ORDINARY
        assert abs(verdict.deviation) < 1.0

    def test_the_scale_is_the_spread_of_the_left_out_residuals(self) -> None:
        """Worked by hand, with a weekday reference from two days.

        Each of the fourteen days is compared with the other day of its
        weekday, so its residual is their difference, and the two days of a
        weekday share it up to sign.
        """
        config = BaselineConfig(calibrated=True, weekday_min_samples=2, min_scale=0.01)
        first = [8.0, 7.0, 9.0, 8.5, 6.0, 7.5, 8.0]
        gaps = [0.1, 0.4, -0.2, 0.6, -0.3, 0.5, 0.0]
        baseline = AdaptiveBaseline("x", config)
        feed(baseline, first + [a + g for a, g in zip(first, gaps)])
        reference = baseline.reference(DAY_ONE + timedelta(days=14))
        assert reference.scale == pytest.approx(
            MAD_TO_SIGMA * statistics.median(abs(g) for g in gaps + gaps)
        )
        assert reference.centre == pytest.approx(8.05)

    def test_a_weekday_with_too_few_days_is_compared_with_all_the_others(self) -> None:
        config = BaselineConfig(calibrated=True, min_scale=0.01)
        values = [8.0, 7.0, 9.0, 8.5, 6.0, 7.5, 8.2, 7.9, 7.1]
        baseline = AdaptiveBaseline("x", config)
        feed(baseline, values)
        residuals = [
            value - statistics.median(values[:k] + values[k + 1 :])
            for k, value in enumerate(values)
        ]
        reference = baseline.reference(DAY_ONE + timedelta(days=9))
        assert not reference.weekday_aware
        assert reference.scale == pytest.approx(
            MAD_TO_SIGMA * statistics.median(abs(r) for r in residuals)
        )

    def test_a_weekly_rhythm_does_not_widen_the_scale(self) -> None:
        rng = np.random.default_rng(4)
        noise = [float(v) for v in rng.normal(0.0, 0.5, size=56)]
        rhythm = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0]
        flat, weekly = AdaptiveBaseline("x", CALIBRATED), AdaptiveBaseline(
            "x", CALIBRATED
        )
        feed(flat, [8.0 + n for n in noise])
        feed(weekly, [8.0 + n + rhythm[k % 7] for k, n in enumerate(noise)])
        sunday = DAY_ONE + timedelta(days=62)
        assert weekly.reference(sunday).scale == pytest.approx(
            flat.reference(sunday).scale
        )
        assert weekly.reference(sunday).centre == pytest.approx(
            flat.reference(sunday).centre + 3.0
        )

    def test_a_perfectly_regular_routine_keeps_the_floor(self) -> None:
        baseline = AdaptiveBaseline("x", CALIBRATED)
        feed(baseline, [8.0] * 30)
        reference = baseline.reference(DAY_ONE + timedelta(days=30))
        assert reference.scale == CALIBRATED.min_scale

    def test_one_retained_day_gives_a_reference(self) -> None:
        baseline = AdaptiveBaseline("x", CALIBRATED)
        feed(baseline, [8.0])
        reference = baseline.reference(DAY_ONE + timedelta(days=1))
        assert reference.scale == CALIBRATED.min_scale
        assert reference.scale_samples == 1
        assert math.isfinite(reference.deviation(9.0))

    def test_a_stationary_day_passes_the_threshold_about_as_often_as_stated(
        self,
    ) -> None:
        """A threshold of two states 4.6% of days. Seeded Gaussian days, 8,400."""

        def share(config: BaselineConfig) -> float:
            rng = np.random.default_rng(11)
            beyond = judged = 0
            for _ in range(150):
                baseline = AdaptiveBaseline("x", config)
                for verdict in feed(baseline, list(8.0 + rng.standard_normal(70))):
                    if verdict.kind is not ChangeKind.INSUFFICIENT_DATA:
                        judged += 1
                        beyond += abs(verdict.deviation) >= config.deviation_threshold
            return beyond / judged

        assert share(BaselineConfig(deviation_threshold=2.0)) > 0.15
        calibrated = share(BaselineConfig(deviation_threshold=2.0, calibrated=True))
        assert 0.03 < calibrated < 0.06

    def test_a_sustained_shift_is_still_a_change(self) -> None:
        rng = np.random.default_rng(8)
        baseline = AdaptiveBaseline("x", CALIBRATED)
        feed(baseline, list(8.0 + rng.normal(0.0, 0.4, size=42)))
        shifted = feed(
            baseline,
            list(4.0 + rng.normal(0.0, 0.4, size=5)),
            DAY_ONE + timedelta(days=42),
        )
        assert shifted[2].kind in {
            ChangeKind.PERSISTENT_CHANGE,
            ChangeKind.ABRUPT_CHANGE,
        }
        assert shifted[2].direction == "decrease"
        assert "normal-equivalent SD" in shifted[2].detail

    def test_a_trend_is_measured_against_the_pooled_scale(self) -> None:
        config = BaselineConfig(
            calibrated=True,
            min_samples=10,
            weekday_min_samples=99,
            trend_threshold=1.5,
            deviation_threshold=6.0,
            min_scale=0.1,
        )
        baseline = AdaptiveBaseline("x", config)
        verdicts = feed(baseline, [8.0 - 0.06 * day for day in range(40)])
        drift = next(v for v in verdicts if v.kind is ChangeKind.GRADUAL_DRIFT)
        assert drift.reference.calibrated
        assert drift.trend_strength == pytest.approx(
            abs(drift.slope_per_day) * config.trend_window / drift.reference.scale
        )
        assert drift.direction == "decrease"

    def test_a_weekly_rhythm_is_no_part_of_the_trend(self) -> None:
        """The trend is fitted to each day's distance from its own weekday.

        Past the fifth week every weekday has its own centre, and from then
        on a calibrated baseline gives the same verdicts on days with a
        weekly rhythm as on the same days without one. The default does not:
        its trend is fitted to the days as they are.
        """
        rng = np.random.default_rng(21)
        noise = [float(v) for v in rng.normal(0.0, 0.5, size=84)]
        rhythm = [0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 2.0]

        def verdicts(config: BaselineConfig, weekly: bool) -> list:
            values = [
                8.0 + n + (rhythm[k % 7] if weekly else 0.0)
                for k, n in enumerate(noise)
            ]
            return feed(AdaptiveBaseline("x", config), values)[35:]

        flat, weekly = verdicts(CALIBRATED, False), verdicts(CALIBRATED, True)
        assert [v.kind for v in weekly] == [v.kind for v in flat]
        for with_rhythm, without in zip(weekly, flat):
            assert with_rhythm.slope_per_day == pytest.approx(without.slope_per_day)
            assert with_rhythm.trend_strength == pytest.approx(without.trend_strength)
            assert with_rhythm.deviation == pytest.approx(without.deviation)
        plain = BaselineConfig()
        slopes = [v.slope_per_day for v in verdicts(plain, True)]
        assert slopes != pytest.approx(
            [v.slope_per_day for v in verdicts(plain, False)]
        )

    def test_the_verdict_says_what_scale_it_is_on(self) -> None:
        baseline = AdaptiveBaseline("x", CALIBRATED)
        payload = feed(baseline, TIGHT_MONDAYS)[-1].to_dict()["reference"]
        assert payload["scale_samples"] == 27
        assert payload["dof"] == pytest.approx(DOF_PER_RESIDUAL * 27)
        plain = feed(AdaptiveBaseline("x"), TIGHT_MONDAYS)[-1].to_dict()["reference"]
        assert plain["scale_samples"] == 0 and plain["dof"] is None

    def test_snapshot_and_restore_preserve_the_calibrated_reference(self) -> None:
        baseline = AdaptiveBaseline("x", CALIBRATED)
        feed(baseline, TIGHT_MONDAYS)
        restarted = AdaptiveBaseline("x", CALIBRATED)
        restarted.restore(baseline.snapshot())
        monday = DAY_ONE + timedelta(days=28)
        assert restarted.reference(monday) == baseline.reference(monday)
