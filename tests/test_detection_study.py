"""Tests for change-detection delay, alert burden and robustness."""

from __future__ import annotations

import math
from datetime import timedelta

import pytest

from sensor_modeling.evaluation import ChangeArm, run_detection_study, standard_arms
from sensor_modeling.evaluation.detection import ArmOutcome, DetectionStudy
from sensor_modeling.evaluation.metrics import DetectionMetrics
from sensor_modeling.simulation import BehaviourShift, HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S

DAYS = 40
CHANGE_DAY = 22
SEEDS = (11, 22)
STEP = timedelta(minutes=30)


def study(arms: list[ChangeArm]) -> object:
    """Run a short detection study over the shared settings."""
    return run_detection_study(
        arms,
        seeds=SEEDS,
        days=DAYS,
        change_day=CHANGE_DAY,
        step=STEP,
        max_delay_days=18.0,
    )


def arm_named(name: str) -> ChangeArm:
    """Pick one arm out of the standard set."""
    return next(a for a in standard_arms(DAYS, CHANGE_DAY) if a.name == name)


class TestRampedChange:
    def test_a_step_shift_takes_full_effect_immediately(self) -> None:
        shift = BehaviourShift(start_day=10, sleep_delta_hours=1.5)
        assert shift.strength_on(9) == 0.0
        assert shift.strength_on(10) == 1.0
        assert shift.strength_on(30) == 1.0

    def test_a_ramped_shift_phases_in(self) -> None:
        shift = BehaviourShift(start_day=10, sleep_delta_hours=1.5, ramp_days=10)
        assert shift.strength_on(9) == 0.0
        assert shift.strength_on(15) == pytest.approx(0.5)
        assert shift.strength_on(20) == 1.0
        assert shift.strength_on(40) == 1.0

    def test_a_negative_ramp_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ramp_days"):
            BehaviourShift(start_day=1, ramp_days=-1)

    def test_a_ramp_produces_a_slower_decline_than_a_step(self) -> None:
        """The ground truth itself must differ, or the arms are the same test."""

        def sleep_by_day(shift: BehaviourShift) -> list[float]:
            result = simulate(HouseholdConfig(days=30, seed=77, shift=shift))
            hours = result.truth.daily_hours(S.SLEEPING)
            return [hours[day] for day in sorted(hours)]

        step = sleep_by_day(BehaviourShift(start_day=10, sleep_delta_hours=2.0))
        ramp = sleep_by_day(
            BehaviourShift(start_day=10, sleep_delta_hours=2.0, ramp_days=15)
        )
        # Just after the change begins, the step has already dropped further.
        assert sum(step[11:16]) < sum(ramp[11:16])


class TestDetection:
    def test_a_real_change_is_detected(self) -> None:
        summary = study([arm_named("abrupt_change")]).summary("abrupt_change")  # type: ignore[attr-defined]
        assert summary["recall"] > 0.0
        assert not math.isnan(summary["median_delay_days"])

    def test_a_stable_record_produces_few_alerts(self) -> None:
        """Alert burden on an unchanged record is what decides usability."""
        summary = study([arm_named("stable")]).summary("stable")  # type: ignore[attr-defined]
        assert summary["recall"] == 0.0
        assert summary["false_positives_per_person_day"] < 0.1

    def test_missingness_does_not_manufacture_findings(self) -> None:
        """The safety property: losing data must not invent a change."""
        results = study([arm_named("stable"), arm_named("stable_degraded")])
        stable = results.summary("stable")  # type: ignore[attr-defined]
        degraded = results.summary("stable_degraded")  # type: ignore[attr-defined]
        assert degraded["false_positives_per_person_day"] <= (
            stable["false_positives_per_person_day"] + 0.02
        )

    def test_detection_survives_a_degraded_record(self) -> None:
        results = study(
            [arm_named("abrupt_change"), arm_named("abrupt_change_degraded")]
        )
        clean = results.summary("abrupt_change")  # type: ignore[attr-defined]
        degraded = results.summary("abrupt_change_degraded")  # type: ignore[attr-defined]
        assert degraded["recall"] > 0.0
        # Detection is allowed to slow down, but not to disappear.
        assert degraded["recall"] >= clean["recall"] - 0.5

    def test_alerts_carry_a_confidence(self) -> None:
        summary = study([arm_named("abrupt_change")]).summary("abrupt_change")  # type: ignore[attr-defined]
        assert 0.0 < summary["mean_alert_confidence"] <= 1.0


class TestStudyMechanics:
    def test_every_arm_runs_on_every_seed(self) -> None:
        results = study([arm_named("stable"), arm_named("abrupt_change")])
        assert results.arms() == ["stable", "abrupt_change"]  # type: ignore[attr-defined]
        assert len(results.for_arm("stable")) == len(SEEDS)  # type: ignore[attr-defined]

    def test_outcomes_are_ordered_by_seed(self) -> None:
        outcomes = study([arm_named("stable")]).for_arm("stable")  # type: ignore[attr-defined]
        assert [o.seed for o in outcomes] == sorted(SEEDS)

    def test_the_standard_arms_pair_changed_with_unchanged(self) -> None:
        """Burden attributable to a change must be separable from baseline."""
        arms = standard_arms()
        names = {a.name for a in arms}
        assert {"stable", "abrupt_change"} <= names
        assert {"stable_degraded", "abrupt_change_degraded"} <= names
        assert any(a.has_change for a in arms)
        assert any(not a.has_change for a in arms)

    def test_every_arm_documents_what_it_isolates(self) -> None:
        assert all(a.description for a in standard_arms())

    def test_a_study_serialises(self) -> None:
        payload = study([arm_named("stable")]).to_dict()  # type: ignore[attr-defined]
        assert "arms" in payload and "outcomes" in payload
        assert payload["arms"]["stable"]["seeds"] == float(len(SEEDS))

    def test_an_unknown_arm_raises(self) -> None:
        with pytest.raises(KeyError, match="no outcomes for arm"):
            study([arm_named("stable")]).summary("nonexistent")  # type: ignore[attr-defined]

    def test_an_empty_study_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one arm"):
            run_detection_study([], seeds=SEEDS)


class TestDelayAggregation:
    """A mean of per-seed medians is not a median.

    The provenance record defines ``median_delay_days`` as the median days
    between a change occurring and being reported. Averaging each seed's own
    median silently reports something else, and weights a seed that detected
    one change as heavily as a seed that detected twenty.
    """

    @staticmethod
    def _outcome(arm: str, seed: int, delays: tuple[float, ...]) -> ArmOutcome:
        return ArmOutcome(
            arm=arm,
            seed=seed,
            metrics=DetectionMetrics(
                true_changes=len(delays),
                detected=len(delays),
                false_positives=0,
                person_days=30.0,
                delays_days=delays,
            ),
            behavioural_alerts=len(delays),
            person_days=30.0,
            mean_alert_confidence=0.5,
        )

    def test_delays_are_pooled_rather_than_averaged_per_seed(self) -> None:
        """One seed detecting many changes must not be outvoted by one that did not.

        Seed A detects a single very late change; seed B detects five prompt
        ones. Pooling gives the median of all six delays, which is prompt. A
        mean of the two seed medians would sit halfway to the outlier and
        report a delay no detection actually had.
        """
        study = DetectionStudy(
            outcomes=[
                self._outcome("changed", 1, (40.0,)),
                self._outcome("changed", 2, (2.0, 2.0, 3.0, 3.0, 4.0)),
            ]
        )

        summary = study.summary("changed")

        assert summary["median_delay_days"] == pytest.approx(3.0)
        assert summary["mean_seed_median_delay_days"] == pytest.approx(21.5)
        assert summary["detected_changes"] == pytest.approx(6.0)

    def test_the_sample_size_behind_the_median_is_reported(self) -> None:
        """An interval-free median still needs its n to be interpretable."""
        study = DetectionStudy(
            outcomes=[
                self._outcome("changed", 1, (5.0, 7.0)),
                self._outcome("changed", 2, (6.0,)),
            ]
        )

        assert study.summary("changed")["detected_changes"] == pytest.approx(3.0)

    def test_an_arm_that_detected_nothing_reports_no_delay(self) -> None:
        study = DetectionStudy(outcomes=[self._outcome("stable", 1, ())])
        summary = study.summary("stable")

        assert math.isnan(summary["median_delay_days"])
        assert summary["detected_changes"] == pytest.approx(0.0)
