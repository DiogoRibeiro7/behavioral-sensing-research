"""Tests for the scoring of the threshold-calibration protocol.

The homes in the first part are built by hand: an alert is put where a test
wants it, and no home is simulated. The second part replays a hand-built home
of three event sensors.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

import numpy as np
import pytest

from sensor_modeling.alerts import Alert as RaisedAlert
from sensor_modeling.alerts import AlertKind, AlertPolicy, AlertSeverity
from sensor_modeling.baseline import BaselineConfig, ChangeKind
from sensor_modeling.datasets import threshold_calibration_experiment as experiment
from sensor_modeling.datasets.threshold_calibration_experiment import (
    EXPECTED_DIRECTION,
    NOT_ATTRIBUTABLE,
    OTHERWISE,
    ArmRecord,
    ConditionRun,
    HomeRecord,
    TihmRun,
    check,
    condition_run,
    confidence_of,
    criteria,
    crossing,
    describe,
    detected,
    excess,
    false_alerts,
    in_order,
    matched,
    matched_estimate,
    matched_or_beaten,
    mean_estimate,
    ratio_estimate,
    read_at,
    record_of,
    replay_arm,
    score,
)
from sensor_modeling.datasets.threshold_calibration_figures import (
    figure_data,
    tihm_figure_data,
)
from sensor_modeling.datasets.threshold_calibration_protocol import (
    ARMS,
    BRACKETED,
    CALIBRATED,
    CHANGE,
    CHANGE_ARMS,
    DEFAULT,
    END_OF_THE_GRID,
    GRADUAL_CHANGE,
    NOT_REACHED,
    REFERENCES,
    RESULT_SCHEMA,
    RULE_OFF,
    SMALL_CHANGE,
    STABLE,
    TIHM_PUBLISHED,
    TIHM_SCHEMA,
    TRACKED_FEATURE,
    ThresholdCalibrationProtocol,
    equivalent_threshold,
    nominal,
)
from sensor_modeling.datasets.threshold_calibration_summary import render_page
from sensor_modeling.evaluation.provenance import json_safe
from sensor_modeling.evaluation.resampling import resample_indices
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
)
from sensor_modeling.online import (
    BehaviouralSensingPipeline,
    PipelineConfig,
    PipelineStep,
    replay_days,
)

FEATURES = (
    "sleeping_hours",
    "kitchen_activity_hours",
    "bathroom_activity_hours",
    "away_hours",
)
D1, C1, C6, C5 = "default@1", "calibrated@1", "calibrated@0.6", "calibrated@0.5"
C55 = "calibrated@0.55"
C1_NAME = "C1_more_is_detected_at_the_same_false_alerts"
C2_NAME = "C2_the_threshold_means_what_it_says"
Alert = tuple[int, str, str, str]
Entry = dict[str, dict[str, list[Alert]]]


#: A coarser grid than the protocol's, which the hand-built homes are written
#: for; the scoring does not depend on which grid it is given.
GRID = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0)


def scoring_protocol(**changes: object) -> ThresholdCalibrationProtocol:
    """A protocol of twelve homes on a coarse grid, for homes built by hand."""
    settings: dict[str, object] = {
        "seed_root": 5,
        "homes": 12,
        "checked_homes": 2,
        "resamples": 400,
        "curve_scales": GRID,
    }
    settings.update(changes)
    return ThresholdCalibrationProtocol(**settings)  # type: ignore[arg-type]


def a_run(protocol: ThresholdCalibrationProtocol, alerts: list[Alert]) -> ConditionRun:
    """A replay whose alerts are raised at the close of the days given.

    Each alert is the day it closes, its subject, the verdict behind it and
    its direction.
    """
    ordered = sorted(alerts)
    return ConditionRun(
        behavioural=tuple(
            (protocol.local(day + 1), subject) for day, subject, _, _ in ordered
        ),
        described=tuple(
            (verdict, direction, "information") for _, _, verdict, direction in ordered
        ),
        verdicts={},
        raised={},
        withheld={},
        bursts=0,
    )


def an_arm(
    protocol: ThresholdCalibrationProtocol,
    alerts: dict[str, list[Alert]] | None = None,
    deviations: dict[str, np.ndarray] | None = None,
    checked: dict[str, bool] | None = None,
    reproduced: bool = True,
) -> ArmRecord:
    """An arm whose replays raised the alerts given, by condition."""
    days = protocol.days
    start = protocol.start()
    quiet = {reference: np.full(days, np.nan) for reference in REFERENCES}
    aware = np.arange(days) >= 28
    return ArmRecord(
        closes=tuple(protocol.local(day + 1) for day in range(days)),
        days=tuple(start + timedelta(days=day) for day in range(days)),
        usable=days,
        values={feature: np.full(days, 8.0) for feature in FEATURES},
        aware={feature: aware for feature in FEATURES},
        deviations={
            reference: {
                feature: (
                    (deviations or quiet)[reference]
                    if feature == TRACKED_FEATURE
                    else np.full(days, np.nan)
                )
                for feature in FEATURES
            }
            for reference in REFERENCES
        },
        conditions={
            condition: a_run(protocol, (alerts or {}).get(condition, []))
            for condition in protocol.conditions
        },
        reproduced=reproduced,
        checked=checked or {},
    )


def checks(protocol: ThresholdCalibrationProtocol, seed: int) -> dict[str, bool]:
    """What a home the protocol checks carries, when every check passed."""
    if seed not in protocol.checked_seeds():
        return {}
    return dict.fromkeys(protocol.checked_conditions, True)


def a_home(
    protocol: ThresholdCalibrationProtocol,
    seed: int,
    deviations: dict[str, np.ndarray] | None = None,
    **arms: dict[str, list[Alert]],
) -> HomeRecord:
    """A home whose arms raised the alerts given; the others raised none."""
    return HomeRecord(
        seed=seed,
        arms={
            arm: an_arm(
                protocol,
                arms.get(arm),
                deviations=deviations,
                checked=checks(protocol, seed),
            )
            for arm in ARMS
        },
    )


def sleep(
    day: int, direction: str = "decrease", verdict: str = "gradual_drift"
) -> Alert:
    return (day, TRACKED_FEATURE, verdict, direction)


def away(day: int) -> Alert:
    """An alert that no detection counts, since it is not about sleep."""
    return (day, "away_hours", "gradual_drift", "increase")


def some_homes(
    protocol: ThresholdCalibrationProtocol, homes: list[Entry]
) -> list[HomeRecord]:
    """One home for each entry, on the protocol's first seeds."""
    return [
        a_home(protocol, seed, **arms)
        for seed, arms in zip(protocol.study_seeds(), homes)
    ]


def cohort(
    protocol: ThresholdCalibrationProtocol, homes: list[Entry]
) -> list[HomeRecord]:
    """Every home of the protocol, one for each entry."""
    assert len(homes) == protocol.homes
    return some_homes(protocol, homes)


def quiet(count: int) -> list[Entry]:
    return [{} for _ in range(count)]


def scored(protocol: ThresholdCalibrationProtocol, homes: list[Entry]) -> dict:
    return score(cohort(protocol, homes), protocol)


def found_day(protocol: ThresholdCalibrationProtocol, arm: str) -> int:
    """A day whose close is inside an arm's window."""
    return protocol.first_changed_day(arm) + 4


def entry(
    protocol: ThresholdCalibrationProtocol,
    *,
    default_false: int = 0,
    default_found: bool = False,
    false: Callable[[float], int] = lambda scale: 0,
    found: Callable[[float], bool] = lambda scale: False,
    arm: str = CHANGE,
) -> Entry:
    """A home by what it does at each multiple of the calibrated reference.

    *false* gives its false alerts at a multiple and *found* whether it finds
    the change there. The false alerts are about another feature, so none of
    them is a false detection.
    """
    day = found_day(protocol, arm)
    stable: dict[str, list[Alert]] = {
        protocol.default: [away(14 + k) for k in range(default_false)]
    }
    change: dict[str, list[Alert]] = {
        protocol.default: [sleep(day)] if default_found else []
    }
    for scale in protocol.curve_scales:
        name = f"calibrated@{scale:g}"
        stable[name] = [away(14 + k) for k in range(false(scale))]
        change[name] = [sleep(day)] if found(scale) else []
    return {STABLE: stable, arm: change}


def step_down(
    below: int, from_scale: float, at_and_above: int
) -> Callable[[float], int]:
    """False alerts that fall from one count to another at a multiple."""
    return lambda scale: at_and_above if scale >= from_scale else below


def better(protocol: ThresholdCalibrationProtocol) -> list[Entry]:
    """Homes in which the calibrated reference finds more at the same false alerts.

    The default raises two false alerts a home and finds the step in three
    homes of twelve. The calibrated reference raises three below a multiple of
    0.6 and one from there on, so it has the default's two halfway between
    0.55 and 0.6. It finds the step in every home at 0.55 and in six at 0.6:
    three quarters of the homes at the match.
    """
    return [
        entry(
            protocol,
            default_false=2,
            default_found=k < 3,
            false=step_down(3, 0.6, 1),
            found=lambda scale, k=k: scale <= (0.6 if k < 6 else 0.55),
        )
        for k in range(protocol.homes)
    ]


class TestStatistics:
    def test_a_mean_carries_its_interval_and_its_monte_carlo_error(self) -> None:
        protocol = scoring_protocol()
        values = [1.0, 2.0, 0.0, 3.0, 1.0, 1.0, 2.0, 2.0]
        found = mean_estimate(values, protocol)
        assert found["estimate"] == pytest.approx(1.5)
        assert found["mcse"] == pytest.approx(np.std(values, ddof=1) / math.sqrt(8))
        assert found["interval"]["low"] < 1.5 < found["interval"]["high"]
        assert found["homes"] == 8
        assert mean_estimate([2.0], protocol)["interval"] is None

    def test_a_ratio_resamples_both_sums(self) -> None:
        protocol = scoring_protocol()
        top, bottom = [1.0, 0.0, 3.0, 2.0], [10.0, 10.0, 20.0, 10.0]
        found = ratio_estimate(top, bottom, protocol)
        assert found["estimate"] == pytest.approx(6 / 50)
        assert found["numerator"] == 6 and found["denominator"] == 50
        assert found["interval"]["low"] <= 6 / 50 <= found["interval"]["high"]
        assert found["resamples_defined"] == protocol.resamples
        # Not the mean of the homes' own ratios, which is 0.1125.
        assert found["estimate"] != pytest.approx(np.mean(np.divide(top, bottom)))

    def test_a_ratio_s_interval_is_that_of_the_resampled_ratios(self) -> None:
        protocol = scoring_protocol()
        # One home has nearly all the days, so resampling the numerator alone
        # would give another interval.
        top, bottom = [1.0, 0.0, 40.0, 2.0], [10.0, 10.0, 400.0, 10.0]
        found = ratio_estimate(top, bottom, protocol)
        index = resample_indices(4, protocol.resamples, protocol.seed)
        ratios = np.array(top)[index].sum(axis=1) / np.array(bottom)[index].sum(axis=1)
        assert found["interval"]["low"] == pytest.approx(np.quantile(ratios, 0.025))
        assert found["interval"]["high"] == pytest.approx(np.quantile(ratios, 0.975))

    def test_a_ratio_over_nothing_is_not_given(self) -> None:
        found = ratio_estimate([0.0, 0.0], [0.0, 0.0], scoring_protocol())
        assert found["estimate"] is None and found["interval"] is None

    def test_the_equivalent_threshold_undoes_what_a_threshold_states(self) -> None:
        for threshold in (1.5, 1.8, 3.0):
            assert equivalent_threshold(nominal(threshold)) == pytest.approx(threshold)
        assert equivalent_threshold(0.0) == math.inf
        assert equivalent_threshold(1.0) == 0.0
        assert equivalent_threshold(0.5) < equivalent_threshold(0.05)
        with pytest.raises(ValueError):
            equivalent_threshold(1.5)


class TestFalseAlerts:
    def test_every_alert_of_the_stable_record_is_a_false_one(self) -> None:
        protocol = scoring_protocol()
        homes = some_homes(
            protocol,
            [
                {STABLE: {D1: [sleep(30), away(40)]}},
                {STABLE: {D1: [sleep(83)], C6: [sleep(20)]}, CHANGE: {D1: [sleep(60)]}},
            ],
        )
        assert list(false_alerts(homes, D1)) == [2.0, 1.0]
        assert list(false_alerts(homes, C6)) == [0.0, 1.0]
        assert list(false_alerts(homes, C5)) == [0.0, 0.0]


class TestDetection:
    def test_the_window_opens_when_the_first_changed_day_closes(self) -> None:
        protocol = scoring_protocol()
        day = protocol.change_day
        homes = some_homes(
            protocol,
            [
                {CHANGE: {D1: [sleep(day - 1)]}},  # closes before any change
                {CHANGE: {D1: [sleep(day)]}},  # the first changed day closes
                {CHANGE: {D1: [sleep(day + 20)]}},  # the last day of the window
                {CHANGE: {D1: [sleep(day + 21)]}},  # the window is open on the right
            ],
        )
        assert list(detected(homes, CHANGE, D1, protocol)) == [False, True, True, False]

    def test_a_gradual_change_begins_the_day_after_it_is_declared_from(self) -> None:
        protocol = scoring_protocol()
        declared = protocol.gradual_day
        assert protocol.first_changed_day(GRADUAL_CHANGE) == declared + 1
        assert protocol.first_changed_day(CHANGE) == protocol.change_day
        homes = some_homes(
            protocol,
            [
                {GRADUAL_CHANGE: {D1: [sleep(declared)]}},  # nothing has changed yet
                {GRADUAL_CHANGE: {D1: [sleep(declared + 1)]}},
                {GRADUAL_CHANGE: {D1: [sleep(declared + 42)]}},
                {GRADUAL_CHANGE: {D1: [sleep(declared + 43)]}},
            ],
        )
        assert list(detected(homes, GRADUAL_CHANGE, D1, protocol)) == [
            False,
            True,
            True,
            False,
        ]

    def test_only_an_alert_about_sleep_counts(self) -> None:
        protocol = scoring_protocol()
        day = protocol.change_day + 3
        homes = some_homes(
            protocol,
            [{CHANGE: {D1: [away(day)]}}, {CHANGE: {D1: [sleep(day)]}}],
        )
        assert list(detected(homes, CHANGE, D1, protocol)) == [False, True]

    def test_a_false_detection_is_read_in_the_stable_record(self) -> None:
        protocol = scoring_protocol()
        day = protocol.change_day + 3
        homes = some_homes(
            protocol,
            [
                {CHANGE: {D1: [sleep(day)]}},
                {CHANGE: {D1: [sleep(day)]}, STABLE: {D1: [sleep(day + 1)]}},
                {STABLE: {D1: [sleep(day)]}},
                {STABLE: {D1: [sleep(protocol.change_day - 5)]}},
            ],
        )
        false = detected(homes, CHANGE, D1, protocol, record=STABLE)
        assert list(false) == [False, True, True, False]
        assert list(excess(homes, CHANGE, D1, protocol)) == [1.0, 0.0, -1.0, 0.0]

    def test_the_direction_is_asked_only_when_it_is_asked(self) -> None:
        protocol = scoring_protocol()
        day = protocol.change_day + 3
        homes = some_homes(
            protocol,
            [
                {CHANGE: {D1: [sleep(day, "increase")]}},
                {CHANGE: {D1: [sleep(day, "increase"), sleep(day + 2, "decrease")]}},
                {STABLE: {D1: [sleep(day, "increase")]}},
            ],
        )
        assert list(detected(homes, CHANGE, D1, protocol)) == [True, True, False]
        asked = detected(homes, CHANGE, D1, protocol, direction=EXPECTED_DIRECTION)
        assert list(asked) == [False, True, False]
        # An alert of an increase in the stable record is a false detection,
        # and not one when the direction is asked.
        assert list(excess(homes, CHANGE, D1, protocol)) == [1.0, 1.0, -1.0]
        assert list(
            excess(homes, CHANGE, D1, protocol, direction=EXPECTED_DIRECTION)
        ) == [0.0, 1.0, 0.0]

    def test_each_arm_is_read_in_its_own_window(self) -> None:
        protocol = scoring_protocol()
        early = protocol.gradual_day + 5
        homes = some_homes(
            protocol,
            [{arm: {D1: [sleep(early)]} for arm in CHANGE_ARMS}, {}],
        )
        assert list(detected(homes, GRADUAL_CHANGE, D1, protocol)) == [True, False]
        assert list(detected(homes, CHANGE, D1, protocol)) == [False, False]
        assert list(detected(homes, SMALL_CHANGE, D1, protocol)) == [False, False]


class TestCrossing:
    def test_the_last_column_that_reaches_the_target_is_taken(self) -> None:
        match = np.array([[5, 3, 1], [5, 3, 1], [5, 3, 1], [4, 2, 1]])
        place, share, status = crossing(match, np.array([2, 0, 6, 2]))
        # Between 3 and 1, halfway; the last column still reaches 0; no column
        # reaches 6; and a column that equals the target is the match itself.
        assert list(status) == [0, 2, 1, 0]
        assert list(place) == [1, 0, 0, 1]
        assert share[0] == pytest.approx(0.5)
        assert np.isnan(share[1]) and np.isnan(share[2])
        assert share[3] == 0.0

    def test_a_curve_that_turns_back_is_read_at_its_last_crossing(self) -> None:
        place, share, status = crossing(np.array([[1, 4, 2, 5, 0]]), np.array([3]))
        assert (int(status[0]), int(place[0])) == (0, 3)
        assert share[0] == pytest.approx(0.4)

    def test_a_value_is_read_on_the_line_towards_the_next_column(self) -> None:
        values = np.array([[10.0, 20.0, 40.0], [1.0, 2.0, 3.0]])
        read = read_at(values, np.array([1, 0]), np.array([0.5, 0.25]))
        assert list(read) == [pytest.approx(30.0), pytest.approx(1.25)]

    def test_the_first_sample_is_the_homes_as_they_are(self) -> None:
        protocol = scoring_protocol()
        samples = experiment._samples(5, protocol)
        assert samples.shape == (protocol.resamples + 1, 5)
        assert list(samples[0]) == [1, 1, 1, 1, 1]
        assert (samples.sum(axis=1) == 5).all()
        index = resample_indices(5, protocol.resamples, protocol.seed)
        assert list(samples[1]) == list(np.bincount(index[0], minlength=5))
        assert list(samples[-1]) == list(np.bincount(index[-1], minlength=5))

    def test_only_whole_numbers_are_summed(self) -> None:
        assert list(experiment._whole(np.array([2.0, -1.0, 0.0]))) == [2, -1, 0]
        with pytest.raises(ValueError, match="whole number"):
            experiment._whole(np.array([0.5]))


def resampled(values: list[float], unbracketed: list[int]) -> dict:
    """An estimand whose resamples have the values given, in their order.

    The places listed had no match: one not reached, then one at the end of
    the grid, and so on in turn.
    """
    protocol = scoring_protocol()
    status = np.zeros(len(values) + 1, dtype=int)
    for turn, place in enumerate(unbracketed):
        status[place + 1] = 1 + turn % 2
    return matched_estimate(np.array([0.5, *values]), status, protocol, 12)


class TestMatchedEstimate:
    def test_the_bounds_are_resampled_values_at_the_two_quantiles(self) -> None:
        found = resampled([float(value) for value in range(1, 401)], [])
        # Of 400 values in order, ten are left outside on each side.
        assert found["estimate"] == 0.5
        assert found["interval"]["low"] == 11.0
        assert found["interval"]["high"] == 390.0
        assert found["interval"]["confidence"] == 0.95
        assert found["status"] == BRACKETED and found["homes"] == 12
        assert found["mcse"] == pytest.approx(np.std(np.arange(1, 401), ddof=1))
        assert found["resamples"] == {
            "all": 400,
            BRACKETED: 400,
            NOT_REACHED: 0,
            END_OF_THE_GRID: 0,
        }

    def test_a_resample_with_no_match_counts_against_either_claim(self) -> None:
        values = [float(value) for value in range(1, 401)]
        # The five largest values had no match. Below every value, they move
        # the lower bound down by five; above every value, they are where they
        # were, and the upper bound stays.
        found = resampled(values, [395, 396, 397, 398, 399])
        assert found["interval"]["low"] == 6.0
        assert found["interval"]["high"] == 390.0
        assert found["resamples"] == {
            "all": 400,
            BRACKETED: 395,
            NOT_REACHED: 3,
            END_OF_THE_GRID: 2,
        }
        assert found["mcse"] == pytest.approx(np.std(np.arange(1, 396), ddof=1))
        # The five smallest instead: below everything they are where they
        # were, and the lower bound stays; above everything, they move the
        # upper bound up by five.
        found = resampled(values, [0, 1, 2, 3, 4])
        assert found["interval"]["low"] == 11.0
        assert found["interval"]["high"] == 395.0

    def test_a_bound_among_resamples_with_no_match_does_not_exist(self) -> None:
        values = [float(value) for value in range(1, 401)]
        eleven = resampled(values, list(range(100, 111)))
        assert eleven["interval"]["low"] is None
        assert eleven["interval"]["high"] is None
        # Ten in 400 are the ones left outside, and both bounds are the
        # outermost of the values that had a match.
        ten = resampled(values, list(range(100, 110)))
        assert ten["interval"]["low"] == 1.0
        assert ten["interval"]["high"] == 400.0

    def test_no_match_on_the_homes_as_they_are_is_no_estimate(self) -> None:
        protocol = scoring_protocol()
        status = np.zeros(401, dtype=int)
        status[0] = 2
        found = matched_estimate(np.full(401, 1.0), status, protocol, 12)
        assert found["estimate"] is None
        assert found["status"] == END_OF_THE_GRID


def by_hand(
    protocol: ThresholdCalibrationProtocol,
    homes: list[HomeRecord],
    arm: str = CHANGE,
) -> list[float | str]:
    """E1 or E4 on the homes as they are and on every resample, with plain loops."""
    scales = protocol.curve_scales
    names = [f"calibrated@{scale:g}" for scale in scales]
    false = [[int(count) for count in false_alerts(homes, name)] for name in names]
    net = [[int(v) for v in excess(homes, arm, name, protocol)] for name in names]
    default_false = [int(count) for count in false_alerts(homes, protocol.default)]
    default_net = [int(v) for v in excess(homes, arm, protocol.default, protocol)]
    index = resample_indices(len(homes), protocol.resamples, protocol.seed)
    values: list[float | str] = []
    for rows in [list(range(len(homes))), *index.tolist()]:
        target = sum(default_false[row] for row in rows)
        total = [
            sum(false[column][row] for row in rows) for column in range(len(scales))
        ]
        few = [column for column in range(len(scales)) if total[column] <= target]
        if not few:
            values.append(NOT_REACHED)
            continue
        taken = min(few)
        if taken == 0:
            values.append(END_OF_THE_GRID)
            continue
        share = Fraction(target - total[taken], total[taken - 1] - total[taken])
        here = sum(net[taken][row] for row in rows)
        there = sum(net[taken - 1][row] for row in rows)
        read = here + share * (there - here)
        values.append(float((read - sum(default_net[row] for row in rows)) / len(rows)))
    return values


def bounds_by_hand(values: list[float | str]) -> tuple[float | None, float | None]:
    """The protocol's bounds from resampled values, some of them with no match."""
    outside = len(values) // 40
    low = sorted(-math.inf if isinstance(v, str) else v for v in values)[outside]
    high = sorted(math.inf if isinstance(v, str) else v for v in values)[
        len(values) - 1 - outside
    ]
    return (
        None if math.isinf(low) else low,
        None if math.isinf(high) else high,
    )


class TestMatch:
    def test_the_match_lies_where_the_curve_has_the_defaults_false_alerts(
        self,
    ) -> None:
        protocol = scoring_protocol()
        found = matched(cohort(protocol, better(protocol)), protocol)
        assert found["default"] == {"condition": D1, "false_alerts_per_home": 2.0}
        match = found["match"]
        assert match["status"] == BRACKETED
        assert match["between"] == [0.55, 0.6]
        assert match["multiple"] == pytest.approx(0.575)
        # Halfway, and of two equally near the smaller is described in full.
        assert match["nearest_multiple"] == 0.55
        e1 = found["excess_detection"][CHANGE]
        assert e1["default"] == pytest.approx(0.25)
        assert e1["calibrated_at_the_match"] == pytest.approx(0.75)
        assert e1["estimate"] == pytest.approx(0.5)
        assert e1["interval"]["low"] > 0.0
        assert e1["resamples"][BRACKETED] == protocol.resamples
        assert e1["homes"] == 12

    def test_the_protocol_s_own_grid_is_matched_the_same_way(self) -> None:
        protocol = scoring_protocol(
            curve_scales=ThresholdCalibrationProtocol().curve_scales
        )
        found = matched(cohort(protocol, better(protocol)), protocol)
        # Three false alerts below 0.6 and one from there on: the neighbour
        # below 0.6 is now 0.575.
        assert found["match"]["between"] == [0.575, 0.6]
        assert found["match"]["multiple"] == pytest.approx(0.5875)
        assert found["match"]["nearest_multiple"] == 0.575
        # Half the homes find the step at both 0.575 and 0.6, against a
        # quarter under the default.
        assert found["excess_detection"][CHANGE]["estimate"] == pytest.approx(0.25)
        assert len(protocol.conditions) == 84

    @pytest.mark.parametrize(
        ("default_false", "multiple", "near"),
        [(2, 0.5875, 0.6), (4, 0.5625, 0.55), (1, 0.6, 0.6)],
    )
    def test_the_multiple_described_in_full_is_the_nearest_of_the_grid(
        self, default_false: int, multiple: float, near: float
    ) -> None:
        protocol = scoring_protocol()
        # Five false alerts below 0.6 and one from there on.
        homes = [
            entry(protocol, default_false=default_false, false=step_down(5, 0.6, 1))
            for _ in range(protocol.homes)
        ]
        match = matched(cohort(protocol, homes), protocol)["match"]
        assert match["multiple"] == pytest.approx(multiple)
        assert match["between"] == [0.55, 0.6]
        assert match["nearest_multiple"] == near

    def test_the_smallest_multiple_with_few_enough_false_alerts_is_taken(self) -> None:
        protocol = scoring_protocol()
        # Three false alerts at 0.4, one at 0.45, three at 0.5 and one above:
        # the default's two are met twice, and the match is the lower one.
        homes = [
            entry(
                protocol,
                default_false=2,
                false=lambda scale: 3 if scale in (0.4, 0.5) else 1,
                found=lambda scale: scale <= 0.4,
            )
            for _ in range(protocol.homes)
        ]
        found = matched(cohort(protocol, homes), protocol)
        assert found["match"]["between"] == [0.4, 0.45]
        assert found["match"]["multiple"] == pytest.approx(0.425)
        # Found at 0.4 in every home and at 0.45 in none: half, at the match.
        e1 = found["excess_detection"][CHANGE]
        assert e1["calibrated_at_the_match"] == pytest.approx(0.5)

    def test_a_curve_that_never_has_so_few_false_alerts_is_not_reached(self) -> None:
        protocol = scoring_protocol()
        homes = [
            entry(protocol, default_false=2, default_found=True, false=lambda scale: 3)
            for _ in range(protocol.homes)
        ]
        found = matched(cohort(protocol, homes), protocol)
        assert found["match"] == {
            "status": NOT_REACHED,
            "multiple": None,
            "between": None,
            "nearest_multiple": 2.0,
        }
        e1 = found["excess_detection"][CHANGE]
        assert e1["estimate"] is None and e1["calibrated_at_the_match"] is None
        assert e1["interval"]["low"] is None and e1["interval"]["high"] is None
        assert e1["resamples"][NOT_REACHED] == protocol.resamples
        assert e1["default"] == 1.0

    def test_a_curve_with_so_few_at_its_smallest_multiple_is_at_the_end(self) -> None:
        protocol = scoring_protocol()
        homes = [
            entry(protocol, default_false=2, false=lambda scale: 1)
            for _ in range(protocol.homes)
        ]
        found = matched(cohort(protocol, homes), protocol)
        assert found["match"]["status"] == END_OF_THE_GRID
        assert found["match"]["nearest_multiple"] == 0.4
        e1 = found["excess_detection"][CHANGE]
        assert e1["resamples"][END_OF_THE_GRID] == protocol.resamples

    def test_what_is_read_is_the_excess_and_not_the_share_detected(self) -> None:
        protocol = scoring_protocol()
        day = found_day(protocol, CHANGE)
        homes = better(protocol)
        # In half the homes the calibrated reference alerts in the same window
        # of the stable record, at both multiples around the match.
        for k in range(0, protocol.homes, 2):
            for name in (C55, C6):
                homes[k][STABLE][name] = [*homes[k][STABLE][name], sleep(day)]
            homes[k][STABLE][D1] = [*homes[k][STABLE][D1], away(60)]
        found = matched(cohort(protocol, homes), protocol)
        # The extra alert is in both references' stable records, so the match
        # stays halfway between 0.55 and 0.6.
        assert found["match"]["multiple"] == pytest.approx(0.575)
        e1 = found["excess_detection"][CHANGE]
        # At 0.55: 12 found less 6 falsely; at 0.6: 6 found less 6 falsely.
        assert e1["calibrated_at_the_match"] == pytest.approx(0.25)
        assert e1["estimate"] == pytest.approx(0.0)

    def test_the_other_changes_are_read_at_the_same_place(self) -> None:
        protocol = scoring_protocol()
        homes = better(protocol)
        for k, home in enumerate(homes):
            for arm, wide, narrow in ((SMALL_CHANGE, 4, 2), (GRADUAL_CHANGE, 12, 9)):
                other = entry(
                    protocol,
                    default_found=k < narrow,
                    found=lambda scale, k=k, wide=wide: scale <= 0.55
                    or (scale <= 0.6 and k < wide),
                    arm=arm,
                )
                home[arm] = other[arm]
        found = matched(cohort(protocol, homes), protocol)["excess_detection"]
        # The smaller step: every home at 0.55 and four at 0.6, against two.
        small = found[SMALL_CHANGE]
        assert small["calibrated_at_the_match"] == pytest.approx((1.0 + 4 / 12) / 2)
        assert small["estimate"] == pytest.approx((1.0 + 4 / 12) / 2 - 2 / 12)
        # The gradual change: every home at both, against nine.
        gradual = found[GRADUAL_CHANGE]
        assert gradual["calibrated_at_the_match"] == pytest.approx(1.0)
        assert gradual["estimate"] == pytest.approx(0.25)
        assert found[CHANGE]["estimate"] == pytest.approx(0.5)

    @pytest.mark.parametrize("arm", CHANGE_ARMS)
    def test_every_resample_is_matched_as_the_homes_are(self, arm: str) -> None:
        protocol = scoring_protocol()
        # Homes that differ in everything: the default's false alerts, where
        # the calibrated reference's fall, and what each finds.
        homes = []
        for k in range(protocol.homes):
            home = entry(
                protocol,
                default_false=(0, 0, 4)[k % 3],
                default_found=k % 4 != 0,
                false=lambda scale, k=k: max(0, 9 - k % 5 - int(scale * 10))
                + (k % 4 == 0),
                found=lambda scale, k=k: scale <= (0.5, 0.65, 0.9, 1.5)[k % 4],
                arm=arm,
            )
            if k % 6 == 1:
                home[STABLE][C6] = [*home[STABLE][C6], sleep(found_day(protocol, arm))]
            homes.append(home)
        built = cohort(protocol, homes)
        expected = by_hand(protocol, built, arm)
        found = matched(built, protocol)["excess_detection"][arm]
        assert not isinstance(expected[0], str)
        assert found["estimate"] == pytest.approx(expected[0])
        low, high = bounds_by_hand(expected[1:])
        assert found["interval"]["low"] == (None if low is None else pytest.approx(low))
        assert found["interval"]["high"] == (
            None if high is None else pytest.approx(high)
        )
        counted = {
            name: sum(1 for value in expected[1:] if value == name)
            for name in (NOT_REACHED, END_OF_THE_GRID)
        }
        assert found["resamples"][NOT_REACHED] == counted[NOT_REACHED]
        assert found["resamples"][END_OF_THE_GRID] == counted[END_OF_THE_GRID]
        assert found["resamples"][BRACKETED] == protocol.resamples - sum(
            counted.values()
        )
        # The homes were built so that some resamples have no match.
        assert 0 < sum(counted.values()) < protocol.resamples


class TestCriteria:
    def test_more_found_at_the_same_false_alerts_is_better(self) -> None:
        protocol = scoring_protocol()
        results = scored(protocol, better(protocol))
        found = results["criteria"]
        one = found[C1_NAME]
        assert one["verdict"] == "success"
        assert one["E1"]["estimate"] == pytest.approx(0.5)
        assert one["match"]["status"] == BRACKETED
        assert one["default_excess_detection"]["estimate"] == pytest.approx(0.25)
        assert one["informative_at"] == 0.2
        assert found["reading"] == "better"
        assert results["shown"] == {
            "default": D1,
            "declared": C1,
            "same_false_alerts": C55,
        }
        assert results["reported"] == [D1, C1, C55]

    def test_less_found_at_the_same_false_alerts_is_worse(self) -> None:
        protocol = scoring_protocol()
        homes = [
            entry(
                protocol,
                default_false=2,
                default_found=True,
                false=step_down(3, 0.6, 1),
                found=lambda scale, k=k: k < 3,
            )
            for k in range(protocol.homes)
        ]
        found = scored(protocol, homes)["criteria"]
        assert found[C1_NAME]["verdict"] == "failure"
        assert found[C1_NAME]["E1"]["estimate"] == pytest.approx(-0.75)
        assert found[C1_NAME]["E1"]["interval"]["high"] < 0.0
        assert found["reading"] == "worse"

    def test_an_interval_that_reaches_zero_shows_nothing(self) -> None:
        protocol = scoring_protocol()
        homes = [
            entry(
                protocol,
                default_false=2,
                default_found=k < 6,
                false=step_down(3, 0.6, 1),
                found=lambda scale, k=k: k >= 6,
            )
            for k in range(protocol.homes)
        ]
        found = scored(protocol, homes)["criteria"]
        one = found[C1_NAME]
        assert one["E1"]["estimate"] == pytest.approx(0.0)
        assert one["E1"]["interval"]["low"] < 0.0 < one["E1"]["interval"]["high"]
        assert one["verdict"] == "inconclusive"
        assert found["reading"] == "not shown"

    def test_a_default_that_finds_too_little_decides_nothing(self) -> None:
        protocol = scoring_protocol()
        homes = better(protocol)
        for home in homes[2:]:
            home[CHANGE][D1] = []
        results = scored(protocol, homes)
        default = results["detection"][CHANGE][D1]["excess_detection"]
        assert default["estimate"] == pytest.approx(2 / 12)
        found = results["criteria"]
        assert found[C1_NAME]["verdict"] == "uninformative"
        assert found[C1_NAME]["E1"]["interval"]["low"] > 0.0
        assert found["reading"] == "uninformative"

    def test_a_default_that_finds_nothing_beyond_chance_decides_nothing(self) -> None:
        protocol = scoring_protocol()
        day = found_day(protocol, CHANGE)
        homes = better(protocol)
        # Detected in every home, and falsely detected in every home.
        for home in homes:
            home[CHANGE][D1] = [sleep(day)]
            home[STABLE][D1] = [away(14), sleep(day)]
        results = scored(protocol, homes)
        default = results["detection"][CHANGE][D1]
        assert default["detected"]["estimate"] == 1.0
        assert default["false_detections"]["estimate"] == 1.0
        assert default["excess_detection"]["estimate"] == 0.0
        assert results["criteria"][C1_NAME]["verdict"] == "uninformative"

    def test_a_fifth_of_the_homes_is_enough_to_be_informative(self) -> None:
        protocol = scoring_protocol(homes=10)
        homes = [
            entry(
                protocol,
                default_false=2,
                default_found=k < 2,
                false=step_down(3, 0.6, 1),
                found=lambda scale: scale <= 0.6,
            )
            for k in range(protocol.homes)
        ]
        results = scored(protocol, homes)
        default = results["detection"][CHANGE][D1]["excess_detection"]
        assert default["estimate"] == protocol.informative_recall == 0.2
        assert results["criteria"][C1_NAME]["verdict"] == "success"

    def test_a_match_that_is_not_bracketed_claims_nothing(self) -> None:
        protocol = scoring_protocol()
        homes = [
            entry(
                protocol,
                default_false=2,
                default_found=k < 3,
                false=lambda scale: 3,
                found=lambda scale: True,
            )
            for k in range(protocol.homes)
        ]
        results = scored(protocol, homes)
        found = results["criteria"]
        assert found[C1_NAME]["verdict"] == "not bracketed"
        assert found[C1_NAME]["match"]["status"] == NOT_REACHED
        assert found["reading"] == "not shown"
        # The largest multiple is described in its place.
        assert results["shown"]["same_false_alerts"] == "calibrated@2"

    @pytest.mark.parametrize(
        ("low", "high", "verdict"),
        [
            (0.01, 0.2, "success"),
            (-0.2, -0.01, "failure"),
            (0.0, 0.2, "inconclusive"),
            (-0.2, 0.0, "inconclusive"),
            (None, -0.01, "failure"),
            (0.01, None, "success"),
            (None, 0.2, "inconclusive"),
            (-0.2, None, "inconclusive"),
            (None, None, "inconclusive"),
        ],
    )
    def test_an_interval_lies_above_or_below_zero_only_strictly(
        self, low: float | None, high: float | None, verdict: str
    ) -> None:
        estimate = {"interval": {"low": low, "high": high}}
        assert experiment._three_way(estimate, 0.0, "failure", "success") == verdict

    def test_the_criteria_are_decided_from_the_results_they_are_given(self) -> None:
        protocol = scoring_protocol()
        results = scored(protocol, better(protocol))
        results["matched"]["excess_detection"][CHANGE]["interval"] = {
            "low": -0.1,
            "high": 0.9,
        }
        assert criteria(results, protocol)[C1_NAME]["verdict"] == "inconclusive"


def deviating(days: int, share: float, threshold: float) -> np.ndarray:
    """Days of which a share lie just past a threshold; none judged before day 14."""
    values = np.full(days, 0.1)
    values[:14] = np.nan
    judged = days - 14
    values[14 : 14 + round(share * judged)] = threshold + 0.01
    return values


def with_deviations(
    protocol: ThresholdCalibrationProtocol,
    calibrated: list[np.ndarray] | np.ndarray,
    change: np.ndarray | None = None,
) -> list[HomeRecord]:
    """Every home of the protocol with the deviations given, and no alert.

    One array is given to every home; a list gives one to each home. With
    *change*, the change arm of every home has those deviations under the
    calibrated reference instead.
    """
    default = deviating(protocol.days, 0.3, 3.0)
    each = (
        list(calibrated)
        if isinstance(calibrated, list)
        else [calibrated] * protocol.homes
    )
    homes = []
    for seed, own in zip(protocol.study_seeds(), each):
        home = a_home(protocol, seed, deviations={DEFAULT: default, CALIBRATED: own})
        if change is not None:
            home.arms[CHANGE] = an_arm(  # type: ignore[index]
                protocol,
                deviations={DEFAULT: default, CALIBRATED: change},
                checked=checks(protocol, seed),
            )
        homes.append(home)
    return homes


class TestThresholdMeansWhatItSays:
    def test_the_share_is_a_ratio_of_sums_over_the_homes(self) -> None:
        protocol = scoring_protocol()
        results = score(
            with_deviations(protocol, deviating(protocol.days, 0.1, 1.5)), protocol
        )
        thresholds = results["calibration"]["thresholds"]
        assert list(thresholds) == ["1.5", "2", "3"]
        found = thresholds["1.5"][CALIBRATED]
        # Twelve homes of seventy judged days.
        assert found["denominator"] == 12 * 70
        assert found["numerator"] == 12 * 7
        assert found["estimate"] == pytest.approx(0.1)
        assert found["threshold"] == 1.5
        assert found["stated"] == pytest.approx(nominal(1.5))
        assert found["equivalent_threshold"] == pytest.approx(equivalent_threshold(0.1))
        # The same days are read against each threshold.
        assert thresholds["2"][CALIBRATED]["numerator"] == 0
        assert thresholds["3"][DEFAULT]["estimate"] == pytest.approx(0.3)
        assert thresholds["1.5"][DEFAULT]["estimate"] == pytest.approx(0.3)

    def test_the_share_is_read_in_the_stable_records(self) -> None:
        protocol = scoring_protocol()
        homes = with_deviations(
            protocol,
            deviating(protocol.days, 0.1, 1.5),
            change=deviating(protocol.days, 0.9, 1.5),
        )
        found = score(homes, protocol)["calibration"]["thresholds"]["1.5"][CALIBRATED]
        assert found["estimate"] == pytest.approx(0.1)

    def test_a_day_at_the_threshold_is_past_it(self) -> None:
        protocol = scoring_protocol()
        days = np.full(protocol.days, 0.1)
        days[:14] = np.nan
        days[14:21] = 1.5
        found = score(with_deviations(protocol, days), protocol)["calibration"]
        assert found["thresholds"]["1.5"][CALIBRATED]["numerator"] == 12 * 7

    @pytest.mark.parametrize(
        ("share", "verdict"),
        [
            (nominal(1.5), "holds"),
            (nominal(1.5 - 0.2), "holds"),
            (nominal(1.5 + 0.2), "holds"),
            (nominal(1.5 - 0.5), "does not hold"),
            (nominal(1.5 + 0.5), "does not hold"),
        ],
    )
    def test_the_verdict_is_read_on_the_scale_of_the_threshold(
        self, share: float, verdict: str
    ) -> None:
        protocol = scoring_protocol()
        results = score(
            with_deviations(protocol, deviating(protocol.days, share, 1.5)), protocol
        )
        two = results["criteria"][C2_NAME]
        assert set(two["verdicts"]) == {"1.5", "2", "3"}
        assert two["verdicts"]["1.5"] == verdict
        assert two["E3"]["1.5"]["threshold"] == 1.5
        assert two["tolerance"] == 0.25

    @pytest.mark.parametrize(
        ("low", "high", "verdict"),
        [
            (1.25, 1.75, "holds"),
            (1.3, 1.7, "holds"),
            (1.24, 1.7, "inconclusive"),
            (1.3, 1.76, "inconclusive"),
            (1.4, 1.9, "inconclusive"),
            (1.76, 1.9, "does not hold"),
            (1.0, 1.24, "does not hold"),
            (1.75, 1.9, "inconclusive"),
            (1.0, 1.25, "inconclusive"),
        ],
    )
    def test_the_band_holds_its_ends(
        self, low: float, high: float, verdict: str
    ) -> None:
        protocol = scoring_protocol()
        found = {
            "threshold": 1.5,
            "equivalent_threshold_interval": {"low": low, "high": high},
        }
        assert experiment._threshold_verdict(found, protocol) == verdict

    def test_a_share_with_no_interval_is_inconclusive(self) -> None:
        found = {"threshold": 1.5, "equivalent_threshold_interval": None}
        assert (
            experiment._threshold_verdict(found, scoring_protocol()) == "inconclusive"
        )

    def test_the_bounds_change_places_on_the_way_to_a_threshold(self) -> None:
        protocol = scoring_protocol()
        mixed = [deviating(protocol.days, 0.05 + 0.02 * k, 1.5) for k in range(12)]
        results = score(with_deviations(protocol, mixed), protocol)
        found = results["calibration"]["thresholds"]["1.5"][CALIBRATED]
        share, threshold = found["interval"], found["equivalent_threshold_interval"]
        assert share["low"] < share["high"]
        assert threshold["low"] == pytest.approx(equivalent_threshold(share["high"]))
        assert threshold["high"] == pytest.approx(equivalent_threshold(share["low"]))
        assert threshold["low"] < found["equivalent_threshold"] < threshold["high"]

    def test_the_days_are_split_by_the_kind_of_reference(self) -> None:
        protocol = scoring_protocol()
        # Seven deviating days, all before the reference is weekday-aware.
        results = score(
            with_deviations(protocol, deviating(protocol.days, 0.1, 1.5)), protocol
        )
        phases = results["calibration"]["thresholds"]["1.5"][CALIBRATED]["by_phase"]
        assert phases["pooled"]["denominator"] == 12 * 14
        assert phases["pooled"]["numerator"] == 12 * 7
        assert phases["weekday_aware"]["denominator"] == 12 * 56
        assert phases["weekday_aware"]["numerator"] == 0
        tail = results["calibration"]["tail"]
        assert set(tail) == set(REFERENCES)
        assert set(tail[CALIBRATED]) == {"1.5", "1.8", "2", "2.5", "3"}
        assert tail[CALIBRATED]["1.5"]["estimate"] == pytest.approx(0.1)
        assert tail[DEFAULT]["2.5"]["estimate"] == pytest.approx(0.3)


def curve(points: dict[str, tuple[float, float]]) -> dict:
    return {
        scale: {
            "false_alerts_per_home": {"estimate": false},
            CHANGE: {"excess_detection": {"estimate": net}},
        }
        for scale, (false, net) in points.items()
    }


class TestCurves:
    def test_a_multiple_is_matched_by_one_no_worse_on_both_counts(self) -> None:
        curves = {
            DEFAULT: curve({"1": (1.0, 0.6), "2": (0.1, 0.2)}),
            CALIBRATED: curve({"0.6": (0.3, 0.6), "1": (0.0, 0.1)}),
        }
        assert matched_or_beaten(curves, CHANGE, DEFAULT, CALIBRATED) == ["1"]
        assert matched_or_beaten(curves, CHANGE, CALIBRATED, DEFAULT) == []
        assert experiment._reading({DEFAULT: curves[DEFAULT]}) == {}

    def test_fewer_false_alerts_alone_do_not_match_a_multiple(self) -> None:
        curves = {
            DEFAULT: curve({"1": (1.0, 0.6)}),
            CALIBRATED: curve({"1": (0.0, 0.59)}),
        }
        assert matched_or_beaten(curves, CHANGE, DEFAULT, CALIBRATED) == []

    def test_the_same_on_both_counts_is_a_match(self) -> None:
        curves = {
            DEFAULT: curve({"1": (1.0, 0.6)}),
            CALIBRATED: curve({"0.6": (1.0, 0.6)}),
        }
        assert matched_or_beaten(curves, CHANGE, DEFAULT, CALIBRATED) == ["1"]
        assert matched_or_beaten(curves, CHANGE, CALIBRATED, DEFAULT) == ["0.6"]

    def test_every_described_condition_is_on_a_curve(self) -> None:
        protocol = scoring_protocol()
        day = found_day(protocol, CHANGE)
        homes = better(protocol)
        # A false detection under one condition, and alerts in the change arm
        # that are no false alerts.
        homes[0][STABLE][C6] = [*homes[0][STABLE][C6], sleep(day)]
        homes[1][CHANGE][C6] = [*homes[1][CHANGE][C6], away(20), away(21)]
        results = scored(protocol, homes)
        curves = results["curves"]
        assert set(curves) == set(REFERENCES)
        for reference in REFERENCES:
            assert list(curves[reference]) == [f"{s:g}" for s in protocol.curve_scales]
        point = curves[CALIBRATED]["0.6"]
        assert point["condition"] == C6
        assert point[CHANGE]["detected"]["estimate"] == pytest.approx(0.5)
        assert point[CHANGE]["excess_detection"]["estimate"] == pytest.approx(5 / 12)
        assert point["false_alerts_per_home"]["estimate"] == pytest.approx(13 / 12)
        assert curves[CALIBRATED]["0.55"][CHANGE]["detected"]["estimate"] == 1.0
        assert curves[CALIBRATED]["0.7"][CHANGE]["detected"]["estimate"] == 0.0
        assert curves[DEFAULT]["1"]["false_alerts_per_home"]["estimate"] == 2.0
        assert curves[DEFAULT]["1"][CHANGE]["detected"]["count"] == 3

    def test_the_curves_are_read_both_ways(self) -> None:
        protocol = scoring_protocol()
        results = scored(protocol, better(protocol))
        reading = results["curves_reading"]
        assert set(reading) == set(CHANGE_ARMS)
        step = reading[CHANGE]
        # The default at its declared thresholds, with two false alerts and a
        # quarter found, is no better than the calibrated reference at 0.6,
        # with one and a half. No multiple of the default has as few false
        # alerts as the calibrated reference with as much found.
        assert "1" in step["default_multiples_a_calibrated_one_is_no_worse_than"]
        assert "0.6" not in step["calibrated_multiples_a_default_one_is_no_worse_than"]
        assert "0.55" not in step["calibrated_multiples_a_default_one_is_no_worse_than"]


class TestDescriptions:
    def test_what_raised_the_false_alerts_and_when(self) -> None:
        protocol = scoring_protocol()
        homes = [
            {
                STABLE: {
                    D1: [
                        sleep(30),
                        sleep(31, "increase", "persistent_change"),
                        (36, "away_hours", "gradual_drift", "increase"),
                    ]
                }
            },
            *quiet(11),
        ]
        stable = scored(protocol, homes)["stable"][D1]
        assert stable["behavioural_alerts"] == 3
        assert stable["per_home"]["estimate"] == pytest.approx(3 / 12)
        assert stable["per_person_day"]["estimate"] == pytest.approx(3 / (12 * 84))
        assert stable["homes_with_one"] == 1 and stable["most_in_one_home"] == 3
        assert stable["by_feature"] == {"away_hours": 1, "sleeping_hours": 2}
        assert stable["by_verdict"] == {"gradual_drift": 2, "persistent_change": 1}
        assert stable["by_direction"] == {"decrease": 1, "increase": 2}
        assert stable["by_severity"] == {"information": 3}
        # Raised at the close of days 30, 31 and 36: weeks five and six.
        weeks = stable["by_week_of_the_day_closed"]
        assert len(weeks) == 12
        assert weeks[4] == 2 and weeks[5] == 1 and sum(weeks) == 3

    def test_a_week_is_the_week_of_the_day_closed(self) -> None:
        protocol = scoring_protocol()
        # Sundays and Mondays on both sides of day 27, when the clocks go
        # forward, and the last day of the record.
        days = [6, 7, 20, 21, 27, 28, 34, 35, 83]
        homes = [{STABLE: {D1: [sleep(day) for day in days]}}, *quiet(11)]
        weeks = scored(protocol, homes)["stable"][D1]["by_week_of_the_day_closed"]
        assert weeks == [1, 1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 1]

    def test_the_first_alert_that_counts_is_described(self) -> None:
        protocol = scoring_protocol()
        day = protocol.change_day
        homes = [
            {
                CHANGE: {
                    D1: [sleep(day + 2, "decrease", "abrupt_change"), sleep(day + 9)]
                }
            },
            {CHANGE: {D1: [sleep(day + 9)]}, STABLE: {D1: [sleep(day + 15)]}},
            {CHANGE: {D1: [sleep(day + 7, "increase")]}},
            *quiet(9),
        ]
        found = scored(protocol, homes)["detection"][CHANGE][D1]
        assert found["first_alert"]["verdict"] == {
            "abrupt_change": 1,
            "gradual_drift": 2,
        }
        assert found["first_alert"]["direction"] == {"decrease": 2, "increase": 1}
        assert found["delay_days"]["delays"] == [3.0, 8.0, 10.0]
        assert found["delay_days"]["estimate"] == pytest.approx(8.0)
        # Detected in three homes and falsely detected in one: the two shares
        # differ, and so does each from the excess.
        assert found["detected"]["count"] == 3
        assert found["false_detections"]["count"] == 1
        assert found["excess_detection"]["estimate"] == pytest.approx(2 / 12)
        asked = found["with_the_expected_direction"]
        assert asked["direction"] == "decrease"
        assert asked["detected"]["count"] == 2
        assert asked["false_detections"]["count"] == 1
        assert asked["excess_detection"]["estimate"] == pytest.approx(1 / 12)

    def test_detections_are_counted_by_week_of_the_window(self) -> None:
        protocol = scoring_protocol()
        day = protocol.change_day
        # The window opens when day 56 closes. Its first week holds the closes
        # of days 56 to 62, and its second begins with day 63.
        homes = [
            {CHANGE: {D1: [sleep(day)]}},
            {CHANGE: {D1: [sleep(day + 6)]}},
            {CHANGE: {D1: [sleep(day + 7)]}},
            {CHANGE: {D1: [sleep(day + 20)]}, STABLE: {D1: [sleep(day + 14)]}},
            *quiet(8),
        ]
        weeks = scored(protocol, homes)["detection"][CHANGE][D1]["by_week"]
        assert weeks["week_of_the_window"] == [1, 2, 3]
        assert weeks["first_detected"] == [2, 1, 1]
        assert weeks["first_false_detection"] == [0, 0, 1]

    def test_a_condition_is_set_against_the_default_home_by_home(self) -> None:
        protocol = scoring_protocol()
        day = protocol.change_day + 2
        early = found_day(protocol, GRADUAL_CHANGE)
        homes = [
            {CHANGE: {D1: [sleep(day)]}, STABLE: {D1: [sleep(20)], C6: [sleep(20)]}},
            {CHANGE: {C6: [sleep(day)]}, STABLE: {C6: [sleep(day)]}},
            {CHANGE: {D1: [sleep(day)]}, GRADUAL_CHANGE: {C6: [sleep(early)]}},
            {SMALL_CHANGE: {D1: [sleep(day)]}},
            *quiet(8),
        ]
        results = scored(protocol, homes)
        # The condition is not one the match gives here, so it is asked for.
        against = experiment._against_default(cohort(protocol, homes), C6, protocol)
        assert results["against_the_default"][C1]["false_alerts"]["estimate"] == (
            pytest.approx(-1 / 12)
        )
        assert against["false_alerts"]["estimate"] == pytest.approx(1 / 12)
        assert against["false_alerts"]["homes_with_more"] == 1
        assert against["false_alerts"]["homes_with_fewer"] == 0
        change = against[CHANGE]
        # Found under the default alone in two homes and under the condition
        # alone in one.
        assert change["detected"]["estimate"] == pytest.approx(-1 / 12)
        assert change["detected"]["only_under_this_condition"] == 1
        assert change["detected"]["only_under_the_default"] == 2
        assert change["false_detections"]["estimate"] == pytest.approx(1 / 12)
        # Excess: 1, 0 and 1 under the default, 0, 0 and 0 under the condition.
        assert change["excess_detection"]["estimate"] == pytest.approx(-2 / 12)
        # Each arm is set against the default in that arm. An alert in the
        # step's window of the stable record is in the gradual change's too.
        gradual = against[GRADUAL_CHANGE]
        assert gradual["detected"]["estimate"] == pytest.approx(1 / 12)
        assert gradual["false_detections"]["estimate"] == pytest.approx(1 / 12)
        assert gradual["excess_detection"]["estimate"] == pytest.approx(0.0)
        # The smaller step is found under the default in one home, and the
        # condition's false detection counts against it there as well.
        small = against[SMALL_CHANGE]
        assert small["detected"]["estimate"] == pytest.approx(-1 / 12)
        assert small["excess_detection"]["estimate"] == pytest.approx(-2 / 12)

    def test_what_became_of_the_verdicts_is_counted(self) -> None:
        protocol = scoring_protocol()
        homes = cohort(protocol, quiet(12))
        run = ConditionRun(
            behavioural=(),
            described=(),
            verdicts={
                TRACKED_FEATURE: {"gradual_drift": 3, "abrupt_change": 1},
                "away_hours": {"gradual_drift": 2},
            },
            raised={"gradual_drift": 4},
            withheld={NOT_ATTRIBUTABLE: 1, OTHERWISE: 1},
            bursts=2,
        )
        first = homes[0]
        arm = first.arms[STABLE]
        homes[0] = HomeRecord(
            seed=first.seed,
            arms={
                **first.arms,
                STABLE: ArmRecord(
                    closes=arm.closes,
                    days=arm.days,
                    usable=arm.usable,
                    values=arm.values,
                    aware=arm.aware,
                    deviations=arm.deviations,
                    conditions={**arm.conditions, D1: run},
                    reproduced=True,
                    checked=arm.checked,
                ),
            },
        )
        found = experiment._verdicts(homes, STABLE, D1)
        assert found == {
            "change_verdicts": 6,
            "by_kind": {"abrupt_change": 1, "gradual_drift": 5},
            "by_feature": {"away_hours": 2, TRACKED_FEATURE: 4},
            "raised_an_alert": {"gradual_drift": 4},
            "raised_none": {NOT_ATTRIBUTABLE: 1, OTHERWISE: 1},
            "notices_of_a_burst": 2,
        }
        assert experiment._verdicts(homes, CHANGE, D1)["change_verdicts"] == 0

    def test_every_home_and_every_alert_is_kept(self) -> None:
        protocol = scoring_protocol()
        day = protocol.change_day + 2
        entries = better(protocol)
        entries[4] = {
            CHANGE: {
                D1: [sleep(day), (day + 1, "away_hours", "abrupt_change", "increase")]
            },
            STABLE: {
                "default@2": [sleep(30)],
                C55: [sleep(day), sleep(40)],
                D1: [away(14), away(15)],
                C6: [away(14)],
            },
        }
        homes = cohort(protocol, entries)
        results = score(homes, protocol)
        seeds = protocol.study_seeds()
        first, fifth = str(seeds[0]), str(seeds[4])
        raw = results["raw"]
        assert raw["conditions_with_every_alert"] == [D1, C1, C55]
        assert raw["alerts"][fifth][CHANGE][D1] == f"{day}sG- {day + 1}aA+"
        assert raw["alerts"][fifth][STABLE][C55] == f"40sG- {day}sG-"
        assert raw["alerts"][first][STABLE][D1] == "14aG+ 15aG+"
        assert raw["alerts"][fifth][CHANGE][C1] == ""
        assert raw["subjects"]["s"] == "sleeping_hours"
        assert raw["verdicts"]["G"] == "gradual_drift"
        assert raw["directions"]["-"] == "decrease"
        # Every home is there under every condition.
        assert raw["seeds"] == list(seeds)
        assert raw["conditions"]["default@2"]["false_alerts"] == (
            "0 0 0 0 1 0 0 0 0 0 0 0"
        )
        assert raw["conditions"][D1][f"{CHANGE}/detected"] == "111010000000"
        assert raw["conditions"][C55][f"{CHANGE}/false_detection"] == "000010000000"
        assert results["result_schema"] == RESULT_SCHEMA
        assert (results["homes"], results["person_days"]) == (12, 12 * 84)

        rows = record_of(homes, protocol, protocol_sha256="0" * 64).to_dict()[
            "household_metrics"
        ]["simulated_homes"]
        assert set(rows) == {str(seed) for seed in seeds}
        assert rows[fifth]["false_alerts"] == [2, 0, 2]
        assert rows[fifth]["detected"][CHANGE] == "100"
        assert rows[fifth]["false_detection"][CHANGE] == "001"
        assert rows[fifth]["detected"][GRADUAL_CHANGE] == "000"
        assert rows[first]["false_alerts"] == [2, 1, 3]
        assert rows[first]["detected"][CHANGE] == "101"
        assert rows[first]["checked_with_the_calibrated_pipeline"] is True
        assert rows[fifth]["checked_with_the_calibrated_pipeline"] is False
        assert rows[fifth]["usable_days"] == 84

    def test_what_the_days_look_like_is_described(self) -> None:
        protocol = scoring_protocol()
        homes = cohort(protocol, quiet(12))
        arm = homes[0].arms[STABLE]
        # A weekend an hour longer, in one home; day 0 is a Monday.
        arm.values[TRACKED_FEATURE][:] = [  # type: ignore[index]
            9.0 if day % 7 >= 5 else 8.0 for day in range(protocol.days)
        ]
        arm.values["away_hours"][:10] = 0.0  # type: ignore[index]
        found = experiment._features(homes)
        assert list(found) == sorted(FEATURES)
        assert found[TRACKED_FEATURE]["homes"] == 12
        assert found[TRACKED_FEATURE]["days"] == 12 * 84
        assert found[TRACKED_FEATURE]["median_weekend_minus_weekday_hours"] == 0.0
        only = experiment._features(homes[:1])
        assert only[TRACKED_FEATURE]["median_weekend_minus_weekday_hours"] == (
            pytest.approx(1.0)
        )
        assert only["away_hours"]["share_of_days_under_a_hundredth_of_an_hour"] == (
            pytest.approx(10 / 84)
        )


class TestCheck:
    def test_the_homes_must_be_the_protocols(self) -> None:
        protocol = scoring_protocol()
        homes = cohort(protocol, quiet(12))
        with pytest.raises(ValueError, match="not the protocol's homes"):
            in_order(homes[:-1], protocol)
        with pytest.raises(ValueError, match="not the protocol's homes"):
            score([*homes[:-1], homes[0]], protocol)
        shuffled = [*homes[6:], *homes[:6]]
        assert in_order(shuffled, protocol) == homes
        assert score(shuffled, protocol)["raw"]["seeds"] == list(protocol.study_seeds())

    def test_a_replay_that_is_not_the_run_stops_everything(self) -> None:
        protocol = scoring_protocol()
        homes = cohort(protocol, quiet(12))
        broken = HomeRecord(
            seed=homes[-1].seed,
            arms={**homes[-1].arms, CHANGE: an_arm(protocol, reproduced=False)},
        )
        found = check([*homes[:-1], broken], protocol)
        assert found["default_replays"] == 48
        assert found["default_replays_that_differ"] == [f"{broken.seed}/{CHANGE}"]
        with pytest.raises(ValueError, match="did not return what the pipeline"):
            score([*homes[:-1], broken], protocol)

    def test_the_homes_checked_are_the_first_by_seed(self) -> None:
        protocol = scoring_protocol()
        assert protocol.checked_seeds() == protocol.study_seeds()[:2]
        found = check(cohort(protocol, quiet(12)), protocol)
        assert found["homes_checked"] == sorted(protocol.checked_seeds())
        assert found["calibrated_runs_checked"] == 24
        assert found["calibrated_runs_expected"] == 24
        assert found["calibrated_runs_that_differ"] == []

    def test_a_checked_home_whose_calibrated_run_differs_stops_everything(
        self,
    ) -> None:
        protocol = scoring_protocol()
        homes = cohort(protocol, quiet(12))
        seed = protocol.checked_seeds()[1]
        differs = {C1: True, C6: False, C5: True}
        homes[1] = HomeRecord(
            seed=seed, arms={arm: an_arm(protocol, checked=differs) for arm in ARMS}
        )
        found = check(homes, protocol)
        assert found["calibrated_runs_checked"] == 24
        assert found["calibrated_runs_that_differ"] == [
            f"{seed}/{arm}/{C6}" for arm in sorted(ARMS)
        ]
        with pytest.raises(ValueError, match="did not return what the pipeline"):
            score(homes, protocol)

    def test_a_checked_home_that_was_not_checked_stops_everything(self) -> None:
        protocol = scoring_protocol()
        homes = cohort(protocol, quiet(12))
        seed = protocol.checked_seeds()[0]
        homes[0] = HomeRecord(seed=seed, arms={arm: an_arm(protocol) for arm in ARMS})
        found = check(homes, protocol)
        assert found["calibrated_runs_checked"] == 12
        assert found["calibrated_runs_expected"] == 24
        with pytest.raises(ValueError, match="did not return what the pipeline"):
            score(homes, protocol)


class TestRecord:
    def test_the_record_says_the_evidence_is_simulated(self) -> None:
        protocol = scoring_protocol()
        homes = cohort(protocol, better(protocol))
        record = record_of(homes, protocol, protocol_sha256="0" * 64)
        payload = record.to_dict()
        assert payload["data_source"] == "simulator"
        assert payload["experiment"] == "threshold-calibration"
        assert payload["configuration"]["protocol_sha256"] == protocol.sha256()
        assert payload["configuration"]["code_changed_since_the_freeze"] == {
            "sources": [],
            "distributions": [],
            "defaults": [],
        }
        assert any("Pre-specified" in note for note in payload["notes"])
        assert not any("modified working tree" in note for note in payload["notes"])
        assert len(payload["models"]) == len(protocol.conditions) == 26
        assert payload["seeds"][2:] == list(protocol.study_seeds())
        record.to_json()

    def test_the_named_estimands_are_the_ones_reported(self) -> None:
        protocol = scoring_protocol()
        homes = cohort(protocol, better(protocol))
        payload = record_of(homes, protocol, protocol_sha256="0" * 64).to_dict()
        results = payload["results"]
        e1 = results["matched"]["excess_detection"][CHANGE]
        declared = results["against_the_default"][C1]
        assert set(payload["mcse"]) == {
            "E1",
            "E2_false_alerts",
            "E2_excess_detection",
            f"E4_{SMALL_CHANGE}",
            f"E4_{GRADUAL_CHANGE}",
        }
        assert payload["mcse"]["E1"] == pytest.approx(e1["mcse"])
        assert payload["mcse"]["E2_false_alerts"] == pytest.approx(
            declared["false_alerts"]["mcse"]
        )
        assert payload["mcse"][f"E4_{SMALL_CHANGE}"] == pytest.approx(
            results["matched"]["excess_detection"][SMALL_CHANGE]["mcse"]
        )
        intervals = {interval["label"]: interval for interval in payload["intervals"]}
        first = next(value for label, value in intervals.items() if label[:3] == "E1:")
        assert first["estimate"] == pytest.approx(0.5)
        assert first["low"] == pytest.approx(e1["interval"]["low"])
        assert first["high"] == pytest.approx(e1["interval"]["high"])
        assert first["n"] == 12
        second = next(
            value
            for label, value in intervals.items()
            if label.startswith("E2: false alerts")
        )
        # As declared the calibrated reference raises one false alert a home,
        # against the default's two.
        assert second["estimate"] == pytest.approx(-1.0)

    def test_the_shares_of_deviating_days_are_reported_with_their_intervals(
        self,
    ) -> None:
        protocol = scoring_protocol()
        mixed = [deviating(protocol.days, 0.05 + 0.01 * k, 1.5) for k in range(12)]
        payload = record_of(
            with_deviations(protocol, mixed), protocol, protocol_sha256="0" * 64
        ).to_dict()
        intervals = {interval["label"]: interval for interval in payload["intervals"]}
        share = intervals["E3: share of days at or past 1.5, calibrated reference"]
        found = payload["results"]["calibration"]["thresholds"]["1.5"][CALIBRATED]
        assert share["estimate"] == pytest.approx(found["estimate"])
        assert share["low"] == pytest.approx(found["interval"]["low"])
        assert "E3: share of days at or past 3, default reference" in intervals

    def test_an_estimand_with_no_bound_has_no_reported_interval(self) -> None:
        protocol = scoring_protocol()
        homes = [
            entry(protocol, default_false=2, default_found=True, false=lambda s: 3)
            for _ in range(protocol.homes)
        ]
        payload = record_of(
            cohort(protocol, homes), protocol, protocol_sha256="0" * 64
        ).to_dict()
        assert not any(i["label"].startswith("E1:") for i in payload["intervals"])
        assert "E1" not in payload["mcse"]

    def test_code_that_changed_since_the_freeze_is_said(self) -> None:
        protocol = scoring_protocol()
        homes = cohort(protocol, quiet(12))
        record = record_of(
            homes,
            protocol,
            protocol_sha256="0" * 64,
            code_changes={
                "sources": ["sensor_modeling/alerts/alert.py"],
                "distributions": ["numpy"],
                "defaults": [],
            },
            code_note="A comment was corrected.",
            clean=False,
        )
        payload = record.to_dict()
        changed = payload["configuration"]["code_changed_since_the_freeze"]
        assert changed["sources"] == ["sensor_modeling/alerts/alert.py"]
        assert changed["distributions"] == ["numpy"]
        assert any("A comment was corrected." in note for note in payload["notes"])
        assert any("modified working tree" in note for note in payload["notes"])


# ----------------------------------------------------------------------------
# A hand-built home of three event sensors, replayed
# ----------------------------------------------------------------------------
START = datetime(2024, 3, 4, tzinfo=timezone.utc)
DAYS = 44
CHANGE_DAY = 26
END = START + timedelta(days=DAYS - 1, hours=20)
CONFIG = PipelineConfig(tz=timezone.utc, step=timedelta(minutes=30))


def registry() -> SensorRegistry:
    return SensorRegistry.from_specs(
        [
            SensorSpec("front_door", Modality.DOOR, room="hall"),
            SensorSpec("living_motion", Modality.MOTION, room="living"),
            SensorSpec("kitchen_motion", Modality.MOTION, room="kitchen"),
        ]
    )


def observations() -> list[Observation]:
    """Motion from 07:00 and then, after the change, from 11:00."""
    records = []
    for day in range(DAYS):
        first = (11 * 60 if day >= CHANGE_DAY else 7 * 60) + 20 * ((day * 3) % 4)
        for minute in range(first, 23 * 60, 20):
            moment = START + timedelta(days=day, minutes=minute)
            if moment >= END:
                continue
            records.append(
                Observation(
                    timestamp=moment,
                    sensor_id=(
                        "kitchen_motion" if minute % 240 == 0 else "living_motion"
                    ),
                    modality=Modality.MOTION,
                    kind=ObservationKind.EVENT,
                    value=1.0,
                )
            )
    return records


def run_pipeline(baseline: BaselineConfig | None = None) -> list[PipelineStep]:
    pipeline = BehaviouralSensingPipeline(
        registry(), config=CONFIG, baseline_config=baseline
    )
    steps = pipeline.run(observations())
    steps.extend(pipeline.close(END))
    return steps


@pytest.fixture(scope="module")
def steps() -> list[PipelineStep]:
    return run_pipeline()


@pytest.fixture(scope="module")
def replayed(steps: list[PipelineStep]) -> ArmRecord:
    protocol = scoring_protocol()
    return replay_arm(
        steps,
        CONFIG,
        protocol,
        protocol.conditions,
        pipeline_under=run_pipeline,
        checked=protocol.checked_conditions,
    )


class TestReplayedArm:
    def test_every_condition_is_replayed_and_the_replay_is_checked(
        self, replayed: ArmRecord
    ) -> None:
        protocol = scoring_protocol()
        assert set(replayed.conditions) == set(protocol.conditions)
        assert replayed.reproduced is True
        assert replayed.checked == {C1: True, C6: True, C5: True}

    def test_the_home_raises_alerts_under_the_default(
        self, replayed: ArmRecord, steps: list[PipelineStep]
    ) -> None:
        """Otherwise the checks above would compare empty lists."""
        run = replayed.conditions[D1]
        assert TRACKED_FEATURE in {subject for _, subject in run.behavioural}
        assert len(run.described) == len(run.behavioural)
        assert sum(run.raised.values()) == len(run.behavioural)
        assert sum(
            count for kinds in run.verdicts.values() for count in kinds.values()
        ) == sum(run.raised.values()) + sum(run.withheld.values())

    def test_the_days_and_their_values_are_kept(
        self, replayed: ArmRecord, steps: list[PipelineStep]
    ) -> None:
        closed = [step for step in steps if step.day_closed is not None]
        assert len(replayed.days) == len(closed) == DAYS
        assert replayed.days[0] == date(2024, 3, 4)
        assert replayed.closes == tuple(step.at for step in closed)
        assert set(replayed.values) == set(FEATURES)
        hours = replayed.values[TRACKED_FEATURE]
        assert np.nanmean(hours[CHANGE_DAY + 1 : -1]) > np.nanmean(hours[1:CHANGE_DAY])
        assert replayed.usable == int((~np.isnan(hours)).sum())

    def test_a_deviation_is_kept_for_each_reference(self, replayed: ArmRecord) -> None:
        assert set(replayed.deviations) == set(REFERENCES)
        default = replayed.deviations[DEFAULT][TRACKED_FEATURE]
        calibrated = replayed.deviations[CALIBRATED][TRACKED_FEATURE]
        judged = ~np.isnan(default)
        assert judged.sum() > 20
        assert np.array_equal(judged, ~np.isnan(calibrated))
        assert not np.allclose(default[judged], calibrated[judged])
        assert replayed.aware[TRACKED_FEATURE][-1]
        assert not replayed.aware[TRACKED_FEATURE][14]

    def test_a_deviation_does_not_depend_on_the_threshold(
        self, steps: list[PipelineStep]
    ) -> None:
        protocol = scoring_protocol()
        for reference in REFERENCES:
            found = []
            for scale in (1.0, 0.5):
                replay = replay_days(
                    steps,
                    CONFIG,
                    baseline_config=protocol.baseline(f"{reference}@{scale:g}"),
                )
                found.append(
                    [
                        change.deviation
                        for step in replay
                        for change in step.changes
                        if change.feature == TRACKED_FEATURE
                    ]
                )
            assert found[0] == found[1]

    def test_a_check_without_a_pipeline_is_refused(
        self, steps: list[PipelineStep]
    ) -> None:
        protocol = scoring_protocol()
        with pytest.raises(ValueError, match="cannot be checked"):
            replay_arm(steps, CONFIG, protocol, [D1, C6], checked=[C6])
        with pytest.raises(ValueError, match="default reference must be replayed"):
            replay_arm(steps, CONFIG, protocol, [C6])


class TestConditionRun:
    def test_a_verdict_on_a_day_not_attributable_enough_is_told_apart(
        self, steps: list[PipelineStep]
    ) -> None:
        protocol = scoring_protocol()
        replay = replay_days(steps, CONFIG, baseline_config=protocol.baseline(D1))
        confident = confidence_of(steps)
        assert set(confident) == {
            step.at for step in steps if step.day_closed is not None
        }
        assert all(0.0 <= value <= 1.0 for value in confident.values())
        usual = condition_run(replay, confident, AlertPolicy())
        doubted = condition_run(replay, dict.fromkeys(confident, 0.0), AlertPolicy())
        # The alerts are the replay's; only what is said of the verdicts that
        # raised none depends on the confidence given.
        assert doubted.behavioural == usual.behavioural
        assert doubted.raised == usual.raised
        silent = sum(usual.withheld.values())
        assert doubted.withheld == ({NOT_ATTRIBUTABLE: silent} if silent else {})
        assert set(usual.withheld) <= {NOT_ATTRIBUTABLE, OTHERWISE}

    def test_the_first_alert_in_a_window(self) -> None:
        protocol = scoring_protocol()
        run = a_run(
            protocol,
            [
                sleep(10, "increase"),
                (12, "away_hours", "gradual_drift", "decrease"),
                sleep(15),
            ],
        )
        begin, end = protocol.local(11), protocol.local(40)
        assert run.first(TRACKED_FEATURE, begin, end) == 0
        assert run.first(TRACKED_FEATURE, begin, end, "decrease") == 2
        assert run.first("away_hours", begin, end) == 1
        assert run.first(TRACKED_FEATURE, protocol.local(17), end) is None


class TestReplayChecks:
    def test_a_pipeline_that_ran_something_else_is_not_reproduced(
        self, steps: list[PipelineStep]
    ) -> None:
        protocol = scoring_protocol()
        # The pipeline is asked for the calibrated reference and runs the
        # default: the replay under the calibrated reference is not that run.
        deaf = replay_arm(
            steps,
            CONFIG,
            protocol,
            [D1, C6],
            pipeline_under=lambda settings: run_pipeline(),
            checked=[C6],
        )
        assert deaf.reproduced is True
        assert deaf.checked == {C6: False}

    def test_a_run_under_other_thresholds_is_not_the_default_replayed(self) -> None:
        protocol = scoring_protocol()
        other = run_pipeline(
            BaselineConfig(deviation_threshold=1.0, trend_threshold=1.0)
        )
        found = replay_arm(other, CONFIG, protocol, [D1])
        assert found.reproduced is False

    def test_an_alert_about_the_apparatus_is_not_a_behavioural_one(
        self, steps: list[PipelineStep]
    ) -> None:
        protocol = scoring_protocol()
        replay = replay_days(steps, CONFIG, baseline_config=protocol.baseline(D1))
        confident = confidence_of(steps)
        usual = condition_run(replay, confident, AlertPolicy())
        place = next(k for k, step in enumerate(replay) if step.day_closed is not None)
        about = replay[place]
        apparatus = [
            RaisedAlert(
                at=about.at,
                kind=kind,
                severity=AlertSeverity.ATTENTION,
                subject=subject,
                summary="not about the resident",
                score=1.0,
                confidence=1.0,
            )
            for kind, subject in (
                (AlertKind.SYSTEM_HEALTH, TRACKED_FEATURE),
                (AlertKind.DATA_QUALITY, "coverage"),
            )
        ]
        with_them = list(replay)
        with_them[place] = dataclasses.replace(
            about, alerts=(*about.alerts, *apparatus)
        )
        found = condition_run(with_them, confident, AlertPolicy())
        assert found.behavioural == usual.behavioural
        assert found.raised == usual.raised
        assert found.bursts == usual.bursts
        burst = dataclasses.replace(apparatus[1], subject="alert_rate")
        with_them[place] = dataclasses.replace(about, alerts=(*about.alerts, burst))
        assert condition_run(with_them, confident, AlertPolicy()).bursts == (
            usual.bursts + 1
        )

    def test_a_deviation_is_kept_without_its_sign(
        self, steps: list[PipelineStep], replayed: ArmRecord
    ) -> None:
        protocol = scoring_protocol()
        replay = replay_days(steps, CONFIG, baseline_config=protocol.baseline(D1))
        signed = [
            change.deviation
            for step in replay
            if step.day_closed is not None
            for change in step.changes
            if change.feature == TRACKED_FEATURE
            and change.kind is not ChangeKind.INSUFFICIENT_DATA
        ]
        kept = replayed.deviations[DEFAULT][TRACKED_FEATURE]
        assert min(signed) < 0.0 < max(signed)
        assert (kept[~np.isnan(kept)] >= 0.0).all()
        assert np.nanmax(kept) == pytest.approx(max(abs(value) for value in signed))

    def test_a_verdict_on_a_confident_day_was_not_withheld_for_attribution(
        self, steps: list[PipelineStep]
    ) -> None:
        protocol = scoring_protocol()
        replay = list(replay_days(steps, CONFIG, baseline_config=protocol.baseline(D1)))
        # A day with a change verdict, with the alert it raised taken away.
        place = next(
            k
            for k, step in enumerate(replay)
            if any(change.is_change for change in step.changes) and step.alerts
        )
        replay[place] = dataclasses.replace(replay[place], alerts=())
        silent = sum(1 for change in replay[place].changes if change.is_change)
        moments = confidence_of(steps)
        edge = AlertPolicy().min_confidence
        for confidence, reason in (
            (1.0, OTHERWISE),
            (edge, OTHERWISE),
            (edge - 0.01, NOT_ATTRIBUTABLE),
            (0.0, NOT_ATTRIBUTABLE),
        ):
            found = condition_run(
                replay, dict.fromkeys(moments, confidence), AlertPolicy()
            )
            assert found.withheld == {reason: silent}
        # Every verdict either raised an alert or is counted as having raised
        # none.
        usual = condition_run(replay, moments, AlertPolicy())
        total = sum(n for counts in usual.verdicts.values() for n in counts.values())
        assert total == sum(usual.raised.values()) + silent


# ----------------------------------------------------------------------------
# The description on TIHM, on runs built by hand
# ----------------------------------------------------------------------------
HOMES = {"a01": (1000, 100, 30), "b02": (1000, 80, 19), "c03": (850, 3, 0)}
PUBLISHED = {
    name: {"monitored_days": days, "behavioural_alerts": off}
    for name, (days, off, _) in HOMES.items()
}


def tihm_runs(
    protocol: ThresholdCalibrationProtocol, **changes: dict
) -> dict[str, list[TihmRun]]:
    """Three homes whose totals are the published ones, under both settings.

    A home's entry in *changes* replaces fields of its run with the rule off.
    """
    runs: dict[str, list[TihmRun]] = {}
    declared = protocol.declared
    for place, rule in enumerate((RULE_OFF, "h12")):
        runs[rule] = []
        for name, counts in HOMES.items():
            checked = (
                {declared: True} if rule == RULE_OFF and name in ("a01", "b02") else {}
            )
            arm = an_arm(
                protocol,
                alerts={
                    D1: [away(k % protocol.days) for k in range(counts[1 + place])],
                    C1: [sleep(40)] if name == "a01" else [],
                },
                deviations={
                    DEFAULT: deviating(protocol.days, 0.3, 3.0),
                    CALIBRATED: deviating(protocol.days, 0.1, 1.5),
                },
                checked=checked,
            )
            fields = {
                "household": name,
                "monitored": counts[0],
                "usable": counts[0] - 10,
                "evaluable": 70,
                "record": arm,
            }
            if rule == RULE_OFF:
                fields.update(changes.get(name, {}))
            runs[rule].append(TihmRun(**fields))
    return runs


def tihm_protocol() -> ThresholdCalibrationProtocol:
    return scoring_protocol(tihm_checked_homes=2)


class TestTihm:
    def test_the_published_totals_are_what_the_hand_built_runs_give(self) -> None:
        assert sum(days for days, _, _ in HOMES.values()) == (
            TIHM_PUBLISHED[RULE_OFF]["monitored_days"]
        )
        assert sum(off for _, off, _ in HOMES.values()) == (
            TIHM_PUBLISHED[RULE_OFF]["behavioural_alerts"]
        )
        assert sum(on for _, _, on in HOMES.values()) == (
            TIHM_PUBLISHED["h12"]["behavioural_alerts"]
        )

    def test_the_references_are_described_on_the_homes(self) -> None:
        protocol = tihm_protocol()
        found = describe(tihm_runs(protocol), protocol, PUBLISHED)
        assert found["result_schema"] == TIHM_SCHEMA
        assert found["homes"] == 3
        check_ = found["check"]
        assert check_[RULE_OFF]["behavioural_alerts"] == 183
        assert check_["h12"]["behavioural_alerts"] == 49
        assert check_["calibrated_runs"] == {
            "condition": C1,
            "homes": ["a01", "b02"],
            "that_differ": [],
        }
        assert check_["homes_compared_one_by_one"] == 3
        assert check_["homes_that_differ"] == []
        off = found["rules"][RULE_OFF]
        assert off["monitored_days"] == 2850 and off["usable_days"] == 2820
        assert off["evaluable_days"] == 210
        assert off["conditions"][D1]["behavioural_alerts"]["all"] == 183
        assert off["conditions"][D1]["behavioural_alerts"]["homes_with_one"] == 3
        assert off["conditions"][D1]["behavioural_alerts"]["most_in_one_home"] == 100
        assert off["conditions"][C1]["behavioural_alerts"]["all"] == 1
        assert off["conditions"][C1]["behavioural_alerts"]["by_feature"] == {
            TRACKED_FEATURE: 1
        }
        assert off["behavioural_alerts_per_home"][D1] == {
            "a01": 100,
            "b02": 80,
            "c03": 3,
        }
        assert (
            found["rules"]["h12"]["conditions"][D1]["behavioural_alerts"]["all"] == 49
        )

    def test_each_reference_s_days_are_read_against_the_thresholds(self) -> None:
        protocol = tihm_protocol()
        tail = describe(tihm_runs(protocol), protocol, PUBLISHED)["rules"][RULE_OFF][
            "tail"
        ]
        assert set(tail) == set(REFERENCES)
        default, calibrated = tail[DEFAULT], tail[CALIBRATED]
        assert set(default[TRACKED_FEATURE]) == {"1.5", "1.8", "2", "2.5", "3"}
        # Three tenths of the days past three under the default reference,
        # and a tenth past one and a half under the calibrated one.
        assert default[TRACKED_FEATURE]["3"]["share"] == pytest.approx(0.3)
        assert default[TRACKED_FEATURE]["3"]["evaluable_feature_days"] == 210
        assert calibrated[TRACKED_FEATURE]["1.5"]["share"] == pytest.approx(0.1)
        assert calibrated[TRACKED_FEATURE]["1.8"]["share"] == 0.0
        assert calibrated[TRACKED_FEATURE]["1.8"]["equivalent_threshold"] == math.inf
        assert calibrated[TRACKED_FEATURE]["1.5"]["stated"] == pytest.approx(
            nominal(1.5)
        )
        assert default["away_hours"]["3"]["share"] is None

    def test_totals_that_are_not_the_published_ones_stop_everything(self) -> None:
        protocol = tihm_protocol()
        runs = tihm_runs(protocol, c03={"monitored": 851})
        with pytest.raises(ValueError, match="is not the published one"):
            describe(runs, protocol)
        with pytest.raises(ValueError, match="is not the published one"):
            describe(runs, protocol, PUBLISHED)

    def test_a_home_that_differs_from_the_published_one_stops_everything(
        self,
    ) -> None:
        protocol = tihm_protocol()
        # The totals are kept: a day moves from one home to another.
        runs = tihm_runs(protocol, a01={"monitored": 999}, b02={"monitored": 1001})
        assert describe(runs, protocol)["check"][RULE_OFF]["monitored_days"] == 2850
        with pytest.raises(ValueError, match="do not have the published record's"):
            describe(runs, protocol, PUBLISHED)

    def test_the_homes_must_be_the_published_ones(self) -> None:
        protocol = tihm_protocol()
        runs = tihm_runs(protocol)
        fewer = {name: PUBLISHED[name] for name in ("a01", "b02")}
        with pytest.raises(ValueError, match="not the published record's homes"):
            describe(runs, protocol, fewer)
        more = {**PUBLISHED, "d04": {"monitored_days": 0, "behavioural_alerts": 0}}
        with pytest.raises(ValueError, match="not the published record's homes"):
            describe(runs, protocol, more)
        twice = {rule: [*homes, homes[0]] for rule, homes in runs.items()}
        with pytest.raises(ValueError, match="more than once"):
            describe(twice, protocol)
        with pytest.raises(ValueError, match="not run on the same homes"):
            describe({**runs, "h12": runs["h12"][:2]}, protocol)

    def test_a_replay_that_is_not_the_run_stops_everything(self) -> None:
        protocol = tihm_protocol()
        broken = dataclasses.replace(
            tihm_runs(protocol)[RULE_OFF][2].record, reproduced=False
        )
        runs = tihm_runs(protocol, c03={"record": broken})
        with pytest.raises(ValueError, match="is not the published one"):
            describe(runs, protocol, PUBLISHED)

    @pytest.mark.parametrize("checked", [{}, {C1: False}])
    def test_a_calibrated_run_that_was_not_reproduced_stops_everything(
        self, checked: dict
    ) -> None:
        protocol = tihm_protocol()
        unchecked = dataclasses.replace(
            tihm_runs(protocol)[RULE_OFF][1].record, checked=checked
        )
        runs = tihm_runs(protocol, b02={"record": unchecked})
        with pytest.raises(ValueError, match="under the calibrated reference"):
            describe(runs, protocol, PUBLISHED)


# ----------------------------------------------------------------------------
# The page and the figures, from a record of homes built by hand
# ----------------------------------------------------------------------------
def a_record(protocol: ThresholdCalibrationProtocol) -> dict:
    """The record of twelve homes with alerts and deviating days, as written."""
    homes = cohort(protocol, better(protocol))
    days = deviating(protocol.days, 0.1, 1.5)
    default = deviating(protocol.days, 0.3, 3.0)
    for home in homes:
        for reference, values in ((DEFAULT, default), (CALIBRATED, days)):
            home.arms[STABLE].deviations[reference][TRACKED_FEATURE][:] = values
    record = record_of(homes, protocol, protocol_sha256="0" * 64)
    return json.loads(record.to_json())


def a_tihm_record(protocol: ThresholdCalibrationProtocol) -> dict:
    return {
        "experiment": "threshold-calibration-tihm",
        "recorded_at": "2026-10-08T00:00:00+00:00",
        "environment": {"git_commit": "abcdef0123456789"},
        "configuration": {
            **protocol.to_dict(),
            "protocol_sha256": protocol.sha256(),
            "code_changed_since_the_freeze": {
                "sources": [],
                "distributions": [],
                "defaults": [],
            },
            "why_the_code_changed": "",
        },
        "results": json_safe(describe(tihm_runs(protocol), protocol, PUBLISHED)),
    }


class TestPages:
    def test_the_page_says_what_the_record_holds(self) -> None:
        protocol = tihm_protocol()
        payload = a_record(protocol)
        page = render_page(payload)
        assert page.startswith("# Threshold calibration: results\n")
        assert "**The evidence is simulated.** 12 paired simulated homes" in page
        assert (
            "| C1 | More is detected at the same false alerts | **success** |" in page
        )
        assert "E1 is +0.50 [" in page
        assert "the calibrated reference is: better.**" in page
        assert "it is at 0.575 times the declared thresholds, between the " in page
        assert "multiples 0.55 and 0.6 of the grid" in page
        assert "`calibrated@0.55`, at the multiple of the grid" in page
        assert "| the step change | 0.25 | 0.75 | +0.50 [" in page
        assert "400 of 400 resamples bracketed" in page
        assert "| C2, at 1.5 | The threshold means what it says | **" in page
        assert "## Between the freeze and the run" not in page
        assert "## TIHM" not in page
        assert "tuning" not in page and "tuned" not in page

    def test_the_page_does_not_depend_on_the_order_of_the_records_keys(self) -> None:
        protocol = tihm_protocol()
        payload = a_record(protocol)
        shuffled = json.loads(json.dumps(payload, sort_keys=True))
        backwards = json.loads(
            json.dumps(payload), object_pairs_hook=lambda pairs: dict(pairs[::-1])
        )
        assert render_page(shuffled) == render_page(payload)
        assert render_page(backwards) == render_page(payload)

    def test_a_match_that_is_not_bracketed_is_said_on_the_page(self) -> None:
        protocol = tihm_protocol()
        homes = [
            entry(protocol, default_false=2, default_found=True, false=lambda s: 3)
            for _ in range(protocol.homes)
        ]
        record = record_of(cohort(protocol, homes), protocol, protocol_sha256="0" * 64)
        page = render_page(json.loads(record.to_json()))
        assert "**not bracketed**" in page
        assert "not reached: at every multiple of the grid" in page
        assert "the calibrated reference is: not shown.**" in page
        assert "0 of 400 resamples bracketed, 400 not reached" in page

    def test_code_that_changed_is_said_on_the_page(self) -> None:
        protocol = tihm_protocol()
        payload = a_record(protocol)
        assert "recorded at the freeze are those the run used" in render_page(payload)
        payload["configuration"]["code_changed_since_the_freeze"] = {
            "sources": ["sensor_modeling/alerts/alert.py"],
            "distributions": ["numpy"],
            "defaults": [],
        }
        payload["configuration"]["why_the_code_changed"] = "A comment was corrected."
        page = render_page(payload)
        assert (
            "had changed by the run: `sensor_modeling/alerts/alert.py`, `numpy`. "
            "A comment was corrected." in page
        )

    def test_what_happened_between_the_freeze_and_the_run_is_listed(self) -> None:
        protocol = tihm_protocol()
        payload = a_record(protocol)
        payload["configuration"]["between_the_freeze_and_the_run"] = [
            "a cohort that is not the protocol's was run"
        ]
        page = render_page(payload)
        assert "## Between the freeze and the run" in page
        assert "- A cohort that is not the protocol's was run." in page

    def test_the_part_on_tihm_is_from_its_own_record(self) -> None:
        protocol = tihm_protocol()
        payload, tihm = a_record(protocol), a_tihm_record(protocol)
        page = render_page(payload, tihm)
        assert "## TIHM: what the references do on the homes" in page
        assert "Commit `abcdef0`" in page
        assert (
            "2,850 monitored days and 183 behavioural alerts with the rule `off`"
            in (page)
        )
        assert "In 2 homes, the first by identifier" in page
        assert "| 1 | 183 (3) | 1 (1) |" in page
        other = scoring_protocol(tihm_checked_homes=2, resamples=300)
        with pytest.raises(ValueError, match="same protocol"):
            render_page(a_record(other), tihm)
        with pytest.raises(ValueError, match="not a threshold-calibration/1 record"):
            render_page(tihm)

    def test_the_figures_plot_what_the_record_holds(self) -> None:
        protocol = tihm_protocol()
        payload = a_record(protocol)
        data = figure_data(payload)
        assert data["weeks"]["homes"] == 12
        assert data["weeks"]["conditions"] == [D1, C1, C55]
        # Two false alerts a home under the default, raised when days 14 and
        # 15 closed: the third week.
        assert data["weeks"]["alerts"][D1] == [0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert data["weeks"]["alerts"][C55] == [0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        curves = data["curves"]
        assert curves["arms"] == list(CHANGE_ARMS)
        assert curves["marked"] == {DEFAULT: ["1"], CALIBRATED: ["0.55"]}
        assert curves["scales"][CALIBRATED] == [f"{s:g}" for s in protocol.curve_scales]
        place = curves["scales"][CALIBRATED].index("0.6")
        assert curves["false_alerts"][CALIBRATED][place] == pytest.approx(1.0)
        assert curves["excess_detection"][CHANGE][CALIBRATED][place] == (
            pytest.approx(0.5)
        )
        assert (
            curves["false_alerts"][DEFAULT][curves["scales"][DEFAULT].index("1")] == 2
        )
        assert curves["match"]["false_alerts"] == pytest.approx(2.0)
        assert curves["match"]["excess_detection"][CHANGE] == pytest.approx(0.75)
        tail = data["tail"]
        assert tail["thresholds"] == [1.5, 1.8, 2.0, 2.5, 3.0]
        assert tail["share"][CALIBRATED][0] == pytest.approx(0.1)
        assert tail["share"][DEFAULT][-1] == pytest.approx(0.3)
        assert tail["stated"][0] == pytest.approx(nominal(1.5))
        assert figure_data(json.loads(json.dumps(payload, sort_keys=True))) == data
        with pytest.raises(ValueError, match="not a threshold-calibration/1 record"):
            figure_data(a_tihm_record(protocol))

    def test_the_figure_of_tihm_plots_both_settings_of_the_rule(self) -> None:
        protocol = tihm_protocol()
        data = tihm_figure_data(a_tihm_record(protocol))["tihm"]
        assert data["homes"] == 3 and data["feature"] == TRACKED_FEATURE
        assert data["thresholds"] == [1.5, 1.8, 2.0, 2.5, 3.0]
        assert set(data["share"]) == {RULE_OFF, "h12"}
        assert data["share"][RULE_OFF][DEFAULT][-1] == pytest.approx(0.3)
        assert data["share"]["h12"][CALIBRATED][0] == pytest.approx(0.1)
        with pytest.raises(ValueError, match="not a threshold-calibration-tihm/1"):
            tihm_figure_data(a_record(protocol))
