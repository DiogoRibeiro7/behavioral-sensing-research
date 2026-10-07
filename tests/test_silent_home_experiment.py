"""Tests for the silent-home evaluation.

The scoring is tested on hand-built runs, where every count is known, so that
each estimand and each criterion can be checked against the protocol's words
without running a home. One small simulated cohort, on seeds that are not the
protocol's, checks that homes are run as the protocol says. A synthetic TIHM
cohort checks the description.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from sensor_modeling.alerts import HOME_SILENCE
from sensor_modeling.datasets.silent_home_experiment import (
    BETWEEN_THE_FREEZE_AND_THE_RUN,
    EPOCH,
    MICROSECOND,
    HomeRecord,
    PipelineRun,
    change_of,
    criteria,
    deliver_home,
    describe,
    describe_tihm,
    mean_estimate,
    median_delay,
    record_of,
    run_home,
    run_homes,
    run_tihm_homes,
    score,
    share_estimate,
    wilson,
)
from sensor_modeling.datasets.silent_home_protocol import (
    CHANGE,
    CHANGE_AFTER_OUTAGE,
    COMMON,
    EXPERIMENT,
    IN_TIME,
    LATE,
    OFF,
    OUTAGE,
    RESULT_SCHEMA,
    SHORT_OUTAGE,
    SILENCE_SUBJECT,
    STABLE,
    TIHM_EXPERIMENT,
    TIHM_SCHEMA,
    SilentHomeProtocol,
    declared_protocol,
)
from sensor_modeling.datasets.tihm_experiment import DayRecord, HouseholdRun
from sensor_modeling.datasets.tihm_protocol import TihmProtocol
from sensor_modeling.evaluation import load_record
from sensor_modeling.evaluation.provenance import SIMULATOR_NOTE
from sensor_modeling.evaluation.resampling import resample_indices
from sensor_modeling.external.tihm import TihmAdapter

ON = "h12"
SLEEP = "sleeping_hours"
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)


def scoring_protocol(**changes: object) -> SilentHomeProtocol:
    """A protocol with few homes, to score hand-built runs under."""
    settings: dict = {
        "seed_root": 7,
        "homes": 8,
        "days": 60,
        "outage_first_day": 20,
        "outage_last_day": 26,
        "change_day": 40,
        "max_delay_days": 14.0,
        "follow_days": 20,
        "common_day": 23,
        "resamples": 400,
    }
    settings.update(changes)
    return SilentHomeProtocol(**settings)


def a_run(
    alerts: tuple[tuple[datetime, str], ...] = (),
    silence: tuple[datetime, ...] = (),
    refused: tuple[date, ...] = (),
) -> PipelineRun:
    return PipelineRun(
        behavioural=tuple(sorted(alerts)),
        silence=tuple(sorted(silence)),
        other=(),
        closes=(),
        usable=0,
        refused=refused,
    )


def hourly(
    protocol: SilentHomeProtocol, gap: tuple[datetime, datetime] | None = None
) -> np.ndarray:
    """One observation an hour over the record, without those inside *gap*."""
    begin = protocol.local(0, 0.0)
    moments = [begin + k * HOUR for k in range(protocol.days * 24)]
    return np.array(
        [
            (m - EPOCH) // MICROSECOND
            for m in moments
            if gap is None or not gap[0] <= m < gap[1]
        ],
        dtype=np.int64,
    )


def a_home(
    seed: int,
    protocol: SilentHomeProtocol,
    runs: dict[tuple[str, str], PipelineRun] | None = None,
) -> HomeRecord:
    """A home whose runs are empty unless given, with an hourly record."""
    filled = {pair: a_run() for pair in protocol.runs()}
    filled.update(runs or {})
    return HomeRecord(
        seed=seed,
        runs=filled,
        observed={
            STABLE: hourly(protocol),
            OUTAGE: hourly(protocol, protocol.outage(seed)),
            COMMON: hourly(protocol, protocol.common_outage()),
        },
    )


def in_window(
    seed: int, protocol: SilentHomeProtocol, count: int
) -> tuple[tuple[datetime, str], ...]:
    """*count* alerts inside a home's outage window, a day apart."""
    begin, _ = protocol.window(seed, OUTAGE)
    return tuple((begin + (k + 1) * DAY, SLEEP) for k in range(count))


def cohort(
    protocol: SilentHomeProtocol,
    *,
    off: int = 2,
    on: int = 0,
    silence: bool = True,
) -> list[HomeRecord]:
    """Homes whose outage raises *off* alerts with the rule off and *on* with it on."""
    homes = []
    for seed in protocol.study_seeds():
        begin, end = protocol.outage(seed)
        alerts = (begin + 13 * HOUR, begin + 33 * HOUR) if silence else ()
        homes.append(
            a_home(
                seed,
                protocol,
                {
                    (OUTAGE, OFF): a_run(in_window(seed, protocol, off)),
                    (OUTAGE, ON): a_run(
                        in_window(seed, protocol, on),
                        silence=alerts,
                        refused=(begin.date(), end.date()),
                    ),
                    (OUTAGE, "h24"): a_run(
                        in_window(seed, protocol, on), silence=alerts[1:]
                    ),
                },
            )
        )
    return homes


# ----------------------------------------------------------------------------
# Statistics over homes
# ----------------------------------------------------------------------------
class TestStatistics:
    def test_the_wilson_interval_matches_known_values(self) -> None:
        interval = wilson(5, 10, 0.95)
        assert interval is not None
        assert interval["low"] == pytest.approx(0.2366, abs=1e-4)
        assert interval["high"] == pytest.approx(0.7634, abs=1e-4)
        none, every = wilson(0, 10, 0.95), wilson(10, 10, 0.95)
        assert none is not None and every is not None
        assert none["low"] == 0.0 and none["high"] == pytest.approx(0.2775, abs=1e-4)
        assert every["high"] == 1.0 and every["low"] == pytest.approx(0.7225, abs=1e-4)
        assert wilson(0, 0, 0.95) is None

    def test_a_mean_carries_its_interval_and_its_monte_carlo_error(self) -> None:
        protocol = scoring_protocol()
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        estimate = mean_estimate(values, protocol)
        assert estimate["estimate"] == 2.5
        assert estimate["homes"] == 6
        assert estimate["mcse"] == pytest.approx(np.std(values, ddof=1) / math.sqrt(6))
        assert estimate["interval"]["low"] < 2.5 < estimate["interval"]["high"]
        assert estimate["interval"]["method"] == "percentile"

    def test_the_interval_is_the_declared_bootstrap_over_homes(self) -> None:
        protocol = scoring_protocol(resamples=700, seed=3, confidence=0.9)
        values = np.array([0.0, 0.0, 1.0, 4.0, 2.0, 0.0, 7.0, 1.0])
        index = resample_indices(values.size, 700, 3)
        means = values[index].mean(axis=1)
        interval = mean_estimate(values, protocol)["interval"]
        assert interval == {
            "low": float(np.quantile(means, 0.05)),
            "high": float(np.quantile(means, 0.95)),
            "confidence": 0.9,
            "method": "percentile",
        }
        # Another seed, or fewer resamples, gives another interval.
        assert (
            mean_estimate(values, scoring_protocol(resamples=700, seed=4))["interval"]
            != mean_estimate(values, scoring_protocol(resamples=700, seed=3))[
                "interval"
            ]
        )
        declared = declared_protocol()
        assert (declared.resamples, declared.seed, declared.confidence) == (
            5000,
            0,
            0.95,
        )

    def test_a_mean_of_one_home_has_no_interval(self) -> None:
        protocol = scoring_protocol()
        assert mean_estimate([3.0], protocol) == {
            "estimate": 3.0,
            "interval": None,
            "mcse": None,
            "homes": 1,
        }
        assert mean_estimate([], protocol)["estimate"] is None

    def test_a_share_counts_homes(self) -> None:
        estimate = share_estimate([True, False, True, True], scoring_protocol())
        assert (estimate["estimate"], estimate["count"], estimate["homes"]) == (
            0.75,
            3,
            4,
        )
        assert estimate["interval"]["method"] == "wilson"

    def test_the_median_delay_pools_the_homes_that_detected(self) -> None:
        protocol = scoring_protocol()
        estimate = median_delay([2.0, None, 4.0, 9.0, None], protocol)
        assert estimate["estimate"] == 4.0
        assert estimate["detections"] == 3
        assert estimate["homes"] == 5
        assert estimate["delays"] == [2.0, 4.0, 9.0]
        assert 2.0 <= estimate["interval"]["low"] <= estimate["interval"]["high"] <= 9.0
        # A resample in which no home detected has no median.
        assert 0 < estimate["resamples_defined"] <= protocol.resamples

    def test_no_detection_gives_no_delay(self) -> None:
        estimate = median_delay([None, None, None], scoring_protocol())
        assert estimate["estimate"] is None
        assert estimate["interval"] is None
        assert estimate["detections"] == 0


# ----------------------------------------------------------------------------
# The estimands, on runs whose counts are known
# ----------------------------------------------------------------------------
class TestExcessAlerts:
    def test_excess_is_the_outage_arm_minus_the_stable_arm_in_the_window(self) -> None:
        protocol = scoring_protocol()
        homes = []
        for seed in protocol.study_seeds():
            begin, end = protocol.window(seed, OUTAGE)
            homes.append(
                a_home(
                    seed,
                    protocol,
                    {
                        (OUTAGE, OFF): a_run(
                            (
                                (begin, SLEEP),
                                (begin + DAY, SLEEP),
                                (begin + 2 * DAY, "away_hours"),
                                # Before the outage and at the window's end:
                                # outside it.
                                (begin - HOUR, SLEEP),
                                (end, SLEEP),
                            )
                        ),
                        (STABLE, OFF): a_run(((begin + DAY, SLEEP), (end, SLEEP))),
                    },
                )
            )
        results = score(homes, protocol)
        off = results["outage"][OFF]
        assert off["excess"]["estimate"] == 2.0
        assert off["alerts_in_the_window"] == 3 * protocol.homes
        assert off["stable_alerts_in_the_window"] == protocol.homes
        assert off["homes_with_an_excess"] == protocol.homes
        assert results["outage"][ON]["excess"]["estimate"] == 0.0
        assert results["outage"][ON]["removed"]["estimate"] == 2.0

    def test_the_groups_split_the_homes_by_when_the_outage_begins(self) -> None:
        protocol = scoring_protocol(homes=24)
        homes = []
        for seed in protocol.study_seeds():
            late = protocol.group(seed) == LATE
            homes.append(
                a_home(
                    seed,
                    protocol,
                    {
                        (OUTAGE, OFF): a_run(in_window(seed, protocol, 2)),
                        (OUTAGE, ON): a_run(
                            in_window(seed, protocol, 1 if late else 0)
                        ),
                    },
                )
            )
        results = score(homes, protocol)
        on = results["outage"][ON]
        groups = [protocol.group(seed) for seed in protocol.study_seeds()]
        assert on["by_group"][IN_TIME]["homes"] == groups.count(IN_TIME) > 0
        assert on["by_group"][LATE]["homes"] == groups.count(LATE) > 0
        assert on["by_group"][IN_TIME]["estimate"] == 0.0
        assert on["by_group"][LATE]["estimate"] == 1.0
        assert on["removed_by_group"][IN_TIME]["estimate"] == 2.0
        assert on["removed_by_group"][LATE]["estimate"] == 1.0
        assert on["excess"]["estimate"] == pytest.approx(
            groups.count(LATE) / protocol.homes
        )

    def test_the_timeline_holds_the_alerts_behind_the_estimand(self) -> None:
        protocol = scoring_protocol()
        results = score(cohort(protocol, off=2, on=0), protocol)
        timeline = results["timeline"]
        days = timeline["days_since_the_outage_began"]
        assert days == list(range(math.ceil(60 / 24) + protocol.follow_days))
        off = timeline["alerts"][OFF][OUTAGE]
        assert sum(off) == results["outage"][OFF]["alerts_in_the_window"]
        assert off[1] == off[2] == protocol.homes
        assert sum(timeline["alerts"][ON][OUTAGE]) == 0
        assert sum(timeline["alerts"][OFF][STABLE]) == 0

    def test_a_short_outage_is_scored_in_its_own_window(self) -> None:
        protocol = scoring_protocol()
        homes = []
        for k, seed in enumerate(protocol.study_seeds()):
            begin, end = protocol.window(seed, SHORT_OUTAGE)
            # The second alert is after the short outage's window has closed
            # and inside the long outage's, which ends two days and four hours
            # later: it is no part of this estimand.
            assert end + HOUR < protocol.window(seed, OUTAGE)[1]
            both = ((begin + DAY, SLEEP), (end + HOUR, SLEEP))
            runs = {(SHORT_OUTAGE, OFF): a_run(both)}
            if k == 0:
                # In one home the outage joined a quiet stretch and was seen.
                runs[(SHORT_OUTAGE, ON)] = a_run(both[1:], silence=(begin + 12 * HOUR,))
            else:
                runs[(SHORT_OUTAGE, ON)] = a_run(both)
            homes.append(a_home(seed, protocol, runs))
        results = score(homes, protocol)
        short = results["short_outage"]
        assert short[OFF]["excess"]["estimate"] == 1.0
        assert short[ON]["excess"]["estimate"] == pytest.approx(7 / 8)
        assert short[ON]["homes_in_which_the_rule_changes_an_alert"] == 1
        assert short[ON]["those_homes"] == [protocol.study_seeds()[0]]
        assert short[ON]["homes_with_a_silence_alert"] == 1
        assert results["criteria"]["C8"]["verdict"] == "not confirmed"


class TestStableHomes:
    def test_alerts_are_counted_per_person_day(self) -> None:
        protocol = scoring_protocol()
        start = protocol.local(30, 0.0)
        homes = [
            a_home(
                seed,
                protocol,
                {
                    (STABLE, c): a_run(((start, SLEEP), (start + DAY, "away_hours")))
                    for c in protocol.conditions
                },
            )
            for seed in protocol.study_seeds()
        ]
        stable = score(homes, protocol)["stable"]
        assert stable["person_days"] == protocol.homes * protocol.days
        for condition in protocol.conditions:
            assert stable[condition]["behavioural_alerts"] == 2 * protocol.homes
            assert stable[condition]["per_person_day"]["estimate"] == pytest.approx(
                2 / protocol.days
            )
            assert stable[condition]["by_feature"] == {
                "away_hours": protocol.homes,
                SLEEP: protocol.homes,
            }
        assert stable[ON]["homes_in_which_the_rule_changes_an_alert"] == 0
        assert stable[ON]["silence_alerts"] == 0

    def test_a_rule_that_moves_an_alert_has_changed_it(self) -> None:
        protocol = scoring_protocol()
        start = protocol.local(30, 0.0)
        homes = [a_home(seed, protocol) for seed in protocol.study_seeds()]
        first = protocol.study_seeds()[0]
        homes[0] = a_home(
            first,
            protocol,
            {
                (STABLE, OFF): a_run(((start, SLEEP),)),
                # The same number of alerts, a day later: not the same alert.
                (STABLE, ON): a_run(((start + DAY, SLEEP),), refused=(start.date(),)),
            },
        )
        results = score(homes, protocol)
        stable = results["stable"][ON]
        assert stable["homes_in_which_the_rule_changes_an_alert"] == 1
        assert stable["those_homes"] == [first]
        assert stable["days_refused"] == 1
        assert results["criteria"]["C3"]["verdict"] == "failure"

    def test_a_silence_alert_in_a_stable_home_is_harm(self) -> None:
        protocol = scoring_protocol()
        homes = [a_home(seed, protocol) for seed in protocol.study_seeds()]
        homes[3] = a_home(
            protocol.study_seeds()[3],
            protocol,
            {(STABLE, ON): a_run(silence=(protocol.local(30, 9.0),))},
        )
        results = score(homes, protocol)
        assert results["stable"][ON]["silence_alerts"] == 1
        assert results["stable"][ON]["homes_with_a_silence_alert"] == 1
        assert results["stable"][ON]["silence_alerts_per_person_day"] == pytest.approx(
            1 / (protocol.homes * protocol.days)
        )
        assert results["criteria"]["C3"]["verdict"] == "failure"


class TestDetection:
    def homes(
        self,
        protocol: SilentHomeProtocol,
        off: int,
        on: int,
        arm: str = CHANGE,
        delay_days: float = 3.0,
    ) -> list[HomeRecord]:
        """The first *off* homes detect with the rule off, the first *on* with it on."""
        alert = ((protocol.change_begins() + timedelta(days=delay_days), SLEEP),)
        return [
            a_home(
                seed,
                protocol,
                {
                    (arm, OFF): a_run(alert if k < off else ()),
                    (arm, ON): a_run(alert if k < on else ()),
                },
            )
            for k, seed in enumerate(protocol.study_seeds())
        ]

    def test_the_share_detected_and_its_paired_difference(self) -> None:
        protocol = scoring_protocol()
        detection = score(self.homes(protocol, off=6, on=4), protocol)["detection"]
        change = detection[CHANGE]
        assert change[OFF]["detected"]["estimate"] == 6 / 8
        assert change[ON]["detected"]["estimate"] == 4 / 8
        assert change["on_minus_off"]["estimate"] == pytest.approx(-0.25)
        assert change["on_minus_off"]["detected_only_with_the_rule_off"] == 2
        assert change["on_minus_off"]["detected_only_with_the_rule_on"] == 0
        assert change[OFF]["delay_days"]["estimate"] == 3.0
        assert change[OFF]["delay_days"]["detections"] == 6
        assert detection[CHANGE_AFTER_OUTAGE][OFF]["detected"]["estimate"] == 0.0

    def test_only_an_alert_about_sleep_inside_the_window_is_a_detection(self) -> None:
        protocol = scoring_protocol()
        begin, end = protocol.detection_window()
        seeds = protocol.study_seeds()
        cases = [
            ((begin, SLEEP),),  # the close of the change day: the first that counts
            ((end - HOUR, SLEEP),),
            ((begin - HOUR, SLEEP),),  # the close of the day before the change
            ((end, SLEEP),),  # the window is open on the right
            ((begin + DAY, "away_hours"),),  # another feature
            ((protocol.change_begins() + 5 * HOUR, SLEEP),),  # the change day itself
        ]
        homes = [
            a_home(
                seed,
                protocol,
                {(CHANGE, OFF): a_run(cases[k]) if k < len(cases) else a_run()},
            )
            for k, seed in enumerate(seeds)
        ]
        results = score(homes, protocol)
        assert results["detection"][CHANGE][OFF]["detected"]["count"] == 2
        assert [
            results["per_home"][str(seed)][
                f"{CHANGE}/{OFF}/meets_the_detection_definition"
            ]
            for seed in seeds
        ] == [True, True, False, False, False, False, False, False]
        first = results["per_home"][str(seeds[0])]
        assert first[f"{CHANGE}/{OFF}/delay_days"] == 1.0
        assert results["per_home"][str(seeds[2])][f"{CHANGE}/{OFF}/delay_days"] is None
        # The arms without the change carry the same flag, as false detections.
        assert first[f"{STABLE}/{OFF}/meets_the_detection_definition"] is False
        assert first[f"{OUTAGE}/{ON}/meets_the_detection_definition"] is False
        assert f"{OUTAGE}/h24/meets_the_detection_definition" not in first
        assert results["detection"][CHANGE][OFF]["delay_days"]["delays"] == [
            1.0,
            pytest.approx(1.0 + protocol.max_delay_days - 1 / 24),
        ]

    def test_a_false_detection_is_counted_in_the_arm_without_the_change(self) -> None:
        protocol = scoring_protocol()
        alert = ((protocol.detection_window()[0] + DAY, SLEEP),)
        seeds = protocol.study_seeds()
        homes = [a_home(seed, protocol) for seed in seeds]
        homes[0] = a_home(seeds[0], protocol, {(STABLE, OFF): a_run(alert)})
        homes[1] = a_home(seeds[1], protocol, {(OUTAGE, OFF): a_run(alert)})
        homes[2] = a_home(seeds[2], protocol, {(OUTAGE, ON): a_run(alert)})
        detection = score(homes, protocol)["detection"]
        assert detection[CHANGE][OFF]["false_detections_in"] == STABLE
        assert detection[CHANGE][OFF]["false_detections"]["count"] == 1
        assert detection[CHANGE][ON]["false_detections"]["count"] == 0
        assert detection[CHANGE_AFTER_OUTAGE][OFF]["false_detections_in"] == OUTAGE
        assert detection[CHANGE_AFTER_OUTAGE][OFF]["false_detections"]["count"] == 1
        assert detection[CHANGE_AFTER_OUTAGE][ON]["false_detections"]["count"] == 1

    @pytest.mark.parametrize(
        ("homes", "off", "on", "verdict"),
        [
            (40, 30, 30, "non-inferior"),
            (40, 30, 32, "non-inferior"),
            (40, 36, 4, "inferior"),
            (40, 30, 26, "inconclusive"),
            # Two homes in forty lose the detection: -0.05, inside the margin,
            # with an interval that is not.
            (40, 30, 28, "inconclusive"),
            (40, 4, 0, "uninformative"),
            # The share detected with the rule off is informative from 0.2 up.
            (40, 7, 7, "uninformative"),
            (40, 8, 8, "non-inferior"),
        ],
    )
    def test_detection_is_judged_against_the_margin(
        self, homes: int, off: int, on: int, verdict: str
    ) -> None:
        protocol = scoring_protocol(homes=homes)
        for arm, criterion in ((CHANGE, "C4"), (CHANGE_AFTER_OUTAGE, "C5")):
            results = score(self.homes(protocol, off, on, arm), protocol)
            judged = results["criteria"][criterion]
            assert judged["verdict"] == verdict
            assert judged["share_detected_with_the_rule_off"] == off / homes
            assert judged["difference"] == pytest.approx((on - off) / homes)
            assert judged["margin"] == -protocol.margin_recall


class TestReporting:
    def test_an_outage_is_reported_by_a_silence_alert_inside_it(self) -> None:
        protocol = scoring_protocol()
        homes = cohort(protocol)
        seeds = protocol.study_seeds()
        begin, end = protocol.outage(seeds[0])
        # One home's only silence alert comes after its outage has ended.
        homes[0] = a_home(
            seeds[0], protocol, {(OUTAGE, ON): a_run(silence=(end + HOUR,))}
        )
        results = score(homes, protocol)
        reporting = results["reporting"][ON]
        assert reporting["reported"]["count"] == protocol.homes - 1
        assert reporting["silence_alerts_outside_the_outage"] == 1
        assert reporting["hours_to_the_first_silence_alert"] == {
            "homes": protocol.homes - 1,
            "mean": 13.0,
            "median": 13.0,
            "min": 13.0,
            "max": 13.0,
        }
        assert reporting["silence_alerts_per_home"]["max"] == 2.0
        assert reporting["silence_alerts_per_home"]["min"] == 0.0
        assert reporting["days_refused_distribution"] == {
            "0": 1,
            "2": protocol.homes - 1,
        }
        assert results["criteria"]["C6"] == {
            "verdict": "failure",
            "share_reported": (protocol.homes - 1) / protocol.homes,
            "needed": protocol.reported_share,
        }
        # The other horizon is reported too, and by its own alerts.
        assert results["reporting"]["h24"]["reported"]["count"] == protocol.homes - 1
        assert (
            results["reporting"]["h24"]["hours_to_the_first_silence_alert"]["median"]
            == 33.0
        )

    def test_the_outage_is_closed_at_both_ends(self) -> None:
        protocol = scoring_protocol()
        seeds = protocol.study_seeds()
        homes = cohort(protocol, silence=False)
        edges = {}
        for k, shift in enumerate((timedelta(0), MICROSECOND, -MICROSECOND)):
            begin, end = protocol.outage(seeds[2 * k])
            # At the beginning, a moment inside, a moment before it.
            edges[seeds[2 * k]] = begin + shift
            # At the end, a moment after it, a moment inside.
            edges[seeds[2 * k + 1]] = protocol.outage(seeds[2 * k + 1])[1] + shift
            homes[2 * k] = a_home(
                seeds[2 * k], protocol, {(OUTAGE, ON): a_run(silence=(begin + shift,))}
            )
            homes[2 * k + 1] = a_home(
                seeds[2 * k + 1],
                protocol,
                {
                    (OUTAGE, ON): a_run(
                        silence=(protocol.outage(seeds[2 * k + 1])[1] + shift,)
                    )
                },
            )
        results = score(homes, protocol)
        row = results["per_home"]
        assert [row[str(seed)][f"{OUTAGE}/{ON}/reported"] for seed in seeds[:6]] == [
            True,  # at the beginning
            True,  # at the end
            True,  # a moment after the beginning
            False,  # a moment after the end
            False,  # a moment before the beginning
            True,  # a moment before the end
        ]
        reporting = results["reporting"][ON]
        assert reporting["reported"]["count"] == 4
        assert reporting["silence_alerts_outside_the_outage"] == 2
        assert reporting["silence_alerts_per_home"]["mean"] == 4 / 8
        assert reporting["silence_alerts_per_home_in_all"]["mean"] == 6 / 8
        assert (
            row[str(seeds[0])][f"{OUTAGE}/{ON}/hours_to_the_first_silence_alert"] == 0.0
        )
        assert (
            row[str(seeds[3])][f"{OUTAGE}/{ON}/hours_to_the_first_silence_alert"]
            is None
        )

    @pytest.mark.parametrize(
        ("reported", "verdict"), [(20, "success"), (19, "success"), (18, "failure")]
    )
    def test_the_share_reported_is_judged_at_the_declared_share(
        self, reported: int, verdict: str
    ) -> None:
        protocol = scoring_protocol(homes=20)
        homes = cohort(protocol)
        for k, seed in enumerate(protocol.study_seeds()):
            if k >= reported:
                homes[k] = a_home(seed, protocol)
        judged = score(homes, protocol)["criteria"]["C6"]
        assert judged["share_reported"] == reported / 20
        assert judged["verdict"] == verdict

    def test_every_home_reported_meets_the_criterion(self) -> None:
        protocol = scoring_protocol()
        results = score(cohort(protocol), protocol)
        assert results["reporting"][ON]["reported"]["estimate"] == 1.0
        assert results["criteria"]["C6"]["verdict"] == "success"


class TestFleet:
    def test_a_shared_outage_is_one_stretch_and_scattered_ones_are_none(self) -> None:
        protocol = scoring_protocol(outage_first_day=10, outage_last_day=30, days=70)
        results = score(cohort(protocol), protocol)
        fleet = results["fleet"]
        begin, end = protocol.common_outage()
        assert fleet["config"] == {
            "horizon_hours": 12.0,
            "fraction": protocol.fleet_fraction,
            "min_homes": protocol.fleet_min_homes,
        }
        assert fleet[STABLE]["stretches"] == []
        assert fleet[STABLE]["largest_share_silent"] == 0.0
        (stretch,) = fleet[COMMON]["stretches"]
        assert stretch["homes"] == stretch["monitored"] == protocol.homes
        assert fleet[COMMON]["stretches_overlapping_the_outage"] == 1
        assert fleet[COMMON]["largest_share_silent"] == 1.0
        # Hourly records: last heard an hour before the outage, so the horizon
        # is reached eleven hours into it.
        assert (
            fleet[COMMON]["hours_to_the_first_assessment_that_calls_it_common"] == 11.0
        )
        assert datetime.fromisoformat(stretch["since"]) == begin - HOUR
        assert datetime.fromisoformat(stretch["until"]) < end
        assert fleet[COMMON]["assessments"] == protocol.days * 24
        # The homes' own outages overlap, and never enough of them at once.
        assert 0.0 < fleet[OUTAGE]["largest_share_silent"] < protocol.fleet_fraction
        assert fleet[OUTAGE]["stretches"] == []
        assert results["criteria"]["C7"] == {
            "verdict": "success",
            "stretches": {STABLE: 0, OUTAGE: 0, COMMON: 1},
            "overlapping_the_outage": 1,
        }

    def test_outages_that_coincide_are_called_common_and_fail_the_criterion(
        self,
    ) -> None:
        # Every home's own outage begins on the same day, so they coincide.
        protocol = scoring_protocol(outage_first_day=22, outage_last_day=22)
        results = score(cohort(protocol), protocol)
        assert len(results["fleet"][OUTAGE]["stretches"]) >= 1
        assert results["criteria"]["C7"]["verdict"] == "failure"

    def test_no_stretch_where_there_was_an_outage_fails_the_criterion(self) -> None:
        # More homes are asked for than there are, so nothing is called common.
        protocol = scoring_protocol(
            outage_first_day=10, outage_last_day=30, days=70, fleet_min_homes=9
        )
        results = score(cohort(protocol), protocol)
        assert results["fleet"][COMMON]["stretches"] == []
        assert results["fleet"][COMMON]["largest_share_silent"] == 1.0
        assert (
            results["fleet"][COMMON][
                "hours_to_the_first_assessment_that_calls_it_common"
            ]
            is None
        )
        assert results["criteria"]["C7"] == {
            "verdict": "failure",
            "stretches": {STABLE: 0, OUTAGE: 0, COMMON: 0},
            "overlapping_the_outage": 0,
        }

    def test_a_stretch_that_is_not_the_outage_fails_the_criterion(self) -> None:
        protocol = scoring_protocol(outage_first_day=10, outage_last_day=30, days=70)
        begin, end = protocol.common_outage()
        elsewhere = (begin + 20 * DAY, end + 20 * DAY)
        homes = [
            HomeRecord(
                home.seed,
                home.runs,
                {**home.observed, COMMON: hourly(protocol, elsewhere)},
            )
            for home in cohort(protocol)
        ]
        results = score(homes, protocol)
        assert len(results["fleet"][COMMON]["stretches"]) == 1
        assert results["fleet"][COMMON]["stretches_overlapping_the_outage"] == 0
        assert (
            results["fleet"][COMMON][
                "hours_to_the_first_assessment_that_calls_it_common"
            ]
            is None
        )
        assert results["criteria"]["C7"]["verdict"] == "failure"

    def test_two_stretches_fail_the_criterion(self) -> None:
        protocol = scoring_protocol(outage_first_day=10, outage_last_day=30, days=70)
        begin, end = protocol.common_outage()

        def twice(home: HomeRecord) -> np.ndarray:
            moments = hourly(protocol, (begin, end))
            again = (
                (begin + 20 * DAY - EPOCH) // MICROSECOND,
                (end + 20 * DAY - EPOCH) // MICROSECOND,
            )
            return moments[(moments < again[0]) | (moments >= again[1])]

        homes = [
            HomeRecord(home.seed, home.runs, {**home.observed, COMMON: twice(home)})
            for home in cohort(protocol)
        ]
        results = score(homes, protocol)
        assert len(results["fleet"][COMMON]["stretches"]) == 2
        assert results["fleet"][COMMON]["stretches_overlapping_the_outage"] == 1
        # The hours are those of the stretch that is the outage.
        assert (
            results["fleet"][COMMON][
                "hours_to_the_first_assessment_that_calls_it_common"
            ]
            == 11.0
        )
        assert results["criteria"]["C7"]["verdict"] == "failure"


class TestCriteria:
    @pytest.mark.parametrize(
        ("off", "on", "first", "second"),
        [
            (2, 0, "reproduced", "success"),
            (3, 1, "reproduced", "partial"),
            (2, 2, "reproduced", "failure"),
            (2, 3, "reproduced", "failure"),
            (0, 0, "not reproduced", "not testable"),
        ],
    )
    def test_the_outage_and_what_the_rule_removes(
        self, off: int, on: int, first: str, second: str
    ) -> None:
        protocol = scoring_protocol()
        judged = score(cohort(protocol, off=off, on=on), protocol)["criteria"]
        assert judged["C1"]["verdict"] == first
        assert judged["C1"]["E1"] == float(off)
        assert judged["C2"]["verdict"] == second
        assert judged["C2"]["E2"] == float(on)
        assert judged["C2"]["E3"] == float(off - on)
        assert judged["C2"]["margin"] == protocol.margin_alerts

    def test_an_excess_whose_interval_reaches_zero_is_not_reproduced(self) -> None:
        protocol = scoring_protocol()
        homes = cohort(protocol, off=0, on=0)
        seed = protocol.study_seeds()[0]
        # One home in eight has an excess: the mean is above zero and the
        # interval is not.
        homes[0] = a_home(
            seed, protocol, {(OUTAGE, OFF): a_run(in_window(seed, protocol, 1))}
        )
        judged = score(homes, protocol)["criteria"]
        assert judged["C1"]["E1"] == 1 / 8
        assert judged["C1"]["interval"]["low"] == 0.0
        assert judged["C1"]["verdict"] == "not reproduced"
        assert judged["C2"]["verdict"] == "not testable"

    def test_a_removal_whose_interval_reaches_zero_is_a_failure(self) -> None:
        protocol = scoring_protocol()
        homes = cohort(protocol, off=2, on=2)
        seed = protocol.study_seeds()[0]
        # The rule removes one alert in one home of eight.
        homes[0] = a_home(
            seed,
            protocol,
            {
                (OUTAGE, OFF): a_run(in_window(seed, protocol, 2)),
                (OUTAGE, ON): a_run(in_window(seed, protocol, 1)),
            },
        )
        judged = score(homes, protocol)["criteria"]
        assert judged["C1"]["verdict"] == "reproduced"
        assert judged["C2"]["E3"] == 1 / 8
        assert judged["C2"]["E3_interval"]["low"] == 0.0
        assert judged["C2"]["verdict"] == "failure"

    def test_a_residue_whose_interval_reaches_the_margin_is_partial(self) -> None:
        protocol = scoring_protocol()
        homes = cohort(protocol, off=3, on=0)
        seed = protocol.study_seeds()[0]
        # One home in eight keeps an alert: 0.125 a home, below the margin,
        # with an interval that is not.
        homes[0] = a_home(
            seed,
            protocol,
            {
                (OUTAGE, OFF): a_run(in_window(seed, protocol, 3)),
                (OUTAGE, ON): a_run(in_window(seed, protocol, 1)),
            },
        )
        judged = score(homes, protocol)["criteria"]
        assert judged["C2"]["E2"] == 1 / 8 < protocol.margin_alerts
        assert judged["C2"]["E2_interval"]["high"] >= protocol.margin_alerts
        assert judged["C2"]["E3_interval"]["low"] > 0.0
        assert judged["C2"]["verdict"] == "partial"

    def test_the_other_horizon_is_scored_by_its_own_runs(self) -> None:
        protocol = scoring_protocol()
        homes = []
        for seed in protocol.study_seeds():
            homes.append(
                a_home(
                    seed,
                    protocol,
                    {
                        (OUTAGE, OFF): a_run(in_window(seed, protocol, 3)),
                        (OUTAGE, ON): a_run(in_window(seed, protocol, 0)),
                        (OUTAGE, "h24"): a_run(in_window(seed, protocol, 2)),
                        (STABLE, "h24"): a_run(
                            silence=(protocol.local(5, 3.0),),
                            refused=(protocol.local(5, 3.0).date(),),
                        ),
                    },
                )
            )
        results = score(homes, protocol)
        # The stable arm at that horizon has no alert in the window, so the
        # excess is the outage arm's own count.
        assert results["outage"]["h24"]["excess"]["estimate"] == 2.0
        assert results["outage"]["h24"]["removed"]["estimate"] == 1.0
        assert results["outage"][ON]["removed"]["estimate"] == 3.0
        assert results["stable"]["h24"]["silence_alerts"] == protocol.homes
        assert results["stable"]["h24"]["days_refused"] == protocol.homes
        assert results["stable"][ON]["silence_alerts"] == 0
        # The criteria are stated at the primary horizon alone.
        assert results["criteria"]["C2"]["E2"] == 0.0
        assert results["criteria"]["C3"]["verdict"] == "success"

    def test_a_residue_below_the_margin_is_still_a_success(self) -> None:
        protocol = scoring_protocol(homes=40)
        homes = cohort(protocol, off=2, on=0)
        # One home in forty keeps an alert: 0.025 a home, far below the margin.
        seed = protocol.study_seeds()[0]
        homes[0] = a_home(
            seed,
            protocol,
            {
                (OUTAGE, OFF): a_run(in_window(seed, protocol, 2)),
                (OUTAGE, ON): a_run(in_window(seed, protocol, 1)),
            },
        )
        judged = score(homes, protocol)["criteria"]
        assert judged["C2"]["verdict"] == "success"
        assert judged["C2"]["E2"] == pytest.approx(1 / 40)
        assert judged["C2"]["E2_interval"]["high"] < protocol.margin_alerts

    def test_a_clean_cohort_meets_every_criterion(self) -> None:
        protocol = scoring_protocol(outage_first_day=10, outage_last_day=30, days=70)
        alert = ((protocol.detection_window()[0] + 2 * DAY, SLEEP),)
        homes = []
        for home in cohort(protocol):
            runs = dict(home.runs)
            for arm in (CHANGE, CHANGE_AFTER_OUTAGE):
                for condition in (OFF, ON):
                    runs[(arm, condition)] = a_run(alert)
            homes.append(HomeRecord(home.seed, runs, home.observed))
        results = score(homes, protocol)
        assert {name: c["verdict"] for name, c in results["criteria"].items()} == {
            "C1": "reproduced",
            "C2": "success",
            "C3": "success",
            "C4": "non-inferior",
            "C5": "non-inferior",
            "C6": "success",
            "C7": "success",
            "C8": "confirmed",
        }
        assert criteria(results, protocol) == results["criteria"]


class TestRecord:
    def test_the_record_says_the_evidence_is_simulated(self, tmp_path: Path) -> None:
        protocol = scoring_protocol()
        closes = tuple(protocol.local(day, 0.0) for day in range(1, 4))
        homes = [
            HomeRecord(
                home.seed,
                {
                    pair: PipelineRun(
                        run.behavioural,
                        run.silence,
                        ((closes[0], "data_quality", "alert_rate"),),
                        closes,
                        3,
                        run.refused,
                        tuple(
                            ("persistent_change", "increase", "attention")
                            for _ in run.behavioural
                        ),
                    )
                    for pair, run in home.runs.items()
                },
                home.observed,
            )
            for home in cohort(protocol, off=3, on=1)
        ]
        record = record_of(homes, protocol, protocol_sha256="0" * 64)
        assert record.experiment == EXPERIMENT
        assert record.data_source == "simulator"
        assert SIMULATOR_NOTE in record.notes
        assert any("Pre-specified" in note for note in record.notes)
        assert any("No pilot informed the protocol" in note for note in record.notes)
        # What was run and read between the freeze and the run is in the record.
        told = record.configuration["between_the_freeze_and_the_run"]
        assert told == list(BETWEEN_THE_FREEZE_AND_THE_RUN)
        assert any("807611" in text for text in told)
        assert any("nothing in the protocol" in text for text in told)
        assert all(
            any(note.lower().startswith(text[:40].lower()) for note in record.notes)
            for text in told
        )
        assert list(record.seeds) == [
            protocol.seed_root,
            protocol.seed,
            *protocol.study_seeds(),
        ]
        assert record.configuration["protocol_sha256"] == protocol.sha256()
        assert record.configuration["result_schema"] == RESULT_SCHEMA
        assert set(record.mcse) == {
            "E1",
            "E2",
            "E3",
            "E5_on_minus_off",
            "E6_on_minus_off",
            "E9_off",
            "E9_on",
            "S1_E2_h24",
            "S1_E3_h24",
        }
        labels = [interval.label for interval in record.intervals]
        assert labels[0] == "E1: excess alerts per home, outage, rule off"
        assert len(labels) == len(set(labels)) == 14
        assert all(interval.unit == "seed" for interval in record.intervals)
        assert set(record.household_metrics["simulated_homes"]) == {
            str(seed) for seed in protocol.study_seeds()
        }

        path = record.write(tmp_path / "record.json")
        payload = load_record(path)
        assert payload["results"]["criteria"]["C2"]["verdict"] == "partial"
        assert payload["results"]["runs"][f"{OUTAGE}/{ON}"]["days_closed"] == 3 * 8
        assert payload["mcse"]["E1"] == 0.0
        assert (
            json.loads(path.read_text(encoding="utf-8"))["data_source"] == "simulator"
        )


# ----------------------------------------------------------------------------
# A small simulated cohort, on seeds that are not the protocol's
# ----------------------------------------------------------------------------
def small_protocol() -> SilentHomeProtocol:
    """Three homes of 34 days: two outages seen in time and one seen late.

    The record crosses the day the clocks change, and the outage is the
    declared one, sixty hours long.
    """
    return SilentHomeProtocol(
        seed_root=2,
        homes=3,
        days=34,
        step_minutes=30,
        outage_first_day=16,
        outage_last_day=17,
        common_day=16,
        change_day=21,
        max_delay_days=7.0,
        follow_days=6,
        resamples=200,
    )


def micro(moment: datetime) -> int:
    return (moment - EPOCH) // MICROSECOND


@pytest.fixture(scope="module")
def simulated() -> dict[str, object]:
    protocol = small_protocol()
    homes = run_homes(protocol, jobs=2)
    return {"protocol": protocol, "homes": homes, "results": score(homes, protocol)}


@pytest.fixture(scope="module")
def delivered() -> dict[str, object]:
    protocol = small_protocol()
    seed = protocol.study_seeds()[0]
    return {"protocol": protocol, "seed": seed, "home": deliver_home(seed, protocol)}


class TestDeliveredArms:
    """The arms are what the protocol says, before anything is run."""

    def test_the_change_is_the_detection_studys_step(self) -> None:
        change = change_of(declared_protocol())
        assert (change.start_day, change.ramp_days) == (56, 0)
        assert change.sleep_delta_hours == 1.6
        assert change.night_bathroom_extra == 1.2
        assert change.outing_probability_delta == 0.0

    def test_only_the_event_sensors_are_delivered(self, delivered: dict) -> None:
        home = delivered["home"]
        kept = {spec.sensor_id for spec in home.registry}
        assert all(spec.expected_interval is None for spec in home.registry)
        for record in home.records.values():
            assert {observation.sensor_id for observation in record} <= kept
            moments = [observation.timestamp for observation in record]
            assert moments == sorted(moments)

    def test_an_outage_arm_is_its_pair_without_the_window(
        self, delivered: dict
    ) -> None:
        protocol: SilentHomeProtocol = delivered["protocol"]
        seed, home = delivered["seed"], delivered["home"]
        pairs = (
            (OUTAGE, STABLE, protocol.outage(seed)),
            (SHORT_OUTAGE, STABLE, protocol.short_outage(seed)),
            (COMMON, STABLE, protocol.common_outage()),
            (CHANGE_AFTER_OUTAGE, CHANGE, protocol.outage(seed)),
        )
        for arm, pair, (begin, end) in pairs:
            whole = home.records[pair]
            outside = [o for o in whole if not begin <= o.timestamp < end]
            # Something is taken out, and nothing else is touched.
            assert len(outside) < len(whole)
            assert list(home.records[arm]) == outside
        short = len(home.records[STABLE]) - len(home.records[SHORT_OUTAGE])
        long = len(home.records[STABLE]) - len(home.records[OUTAGE])
        assert 0 < short < long

    def test_the_changed_home_is_not_the_stable_home(self, delivered: dict) -> None:
        protocol: SilentHomeProtocol = delivered["protocol"]
        home = delivered["home"]
        assert list(home.records[CHANGE]) != list(home.records[STABLE])

        def first_of_the_day(arm: str, day: int) -> float:
            """Local hour of the first bedroom-to-elsewhere activity of a day."""
            begin = protocol.local(day, 3.0)
            return min(
                (o.timestamp - begin) / HOUR
                for o in home.records[arm]
                if o.sensor_id == "kitchen_motion" and o.timestamp >= begin
            )

        after = range(protocol.change_day, protocol.days - 1)
        earlier = [
            first_of_the_day(STABLE, day) - first_of_the_day(CHANGE, day)
            for day in after
        ]
        # The resident wakes earlier after the change, by about the step.
        assert np.median(earlier) > 0.5


class TestSimulatedHomes:
    def test_the_homes_are_not_the_protocols(self, simulated: dict) -> None:
        protocol: SilentHomeProtocol = simulated["protocol"]
        seeds = protocol.study_seeds()
        assert not set(seeds) & set(declared_protocol().study_seeds())
        assert [home.seed for home in simulated["homes"]] == list(seeds)
        assert sorted(protocol.group(seed) for seed in seeds) == [
            LATE,
            IN_TIME,
            IN_TIME,
        ]
        # They are named in the record of what was run before the run.
        assert all(
            str(seed) in " ".join(BETWEEN_THE_FREEZE_AND_THE_RUN) for seed in seeds
        )

    def test_every_declared_run_is_made_and_every_day_is_closed(
        self, simulated: dict
    ) -> None:
        protocol: SilentHomeProtocol = simulated["protocol"]
        for home in simulated["homes"]:
            assert set(home.runs) == set(protocol.runs())
            for run in home.runs.values():
                assert len(run.closes) == protocol.days
                assert run.other == ()
                assert len(run.described) == len(run.behavioural)

    def test_with_the_rule_off_nothing_is_silent(self, simulated: dict) -> None:
        for home in simulated["homes"]:
            for (_, condition), run in home.runs.items():
                if condition == OFF:
                    assert run.silence == ()
                    assert run.refused == ()

    def test_with_the_rule_off_the_outage_raises_alerts(self, simulated: dict) -> None:
        protocol: SilentHomeProtocol = simulated["protocol"]
        results = simulated["results"]
        off = results["outage"][OFF]
        assert off["alerts_in_the_window"] >= 2
        assert off["stable_alerts_in_the_window"] == 0
        assert off["homes_with_an_excess"] >= 2
        # They are raised when the outage's third day closes.
        for home in simulated["homes"]:
            begin, _ = protocol.outage(home.seed)
            for at, _subject in home.runs[(OUTAGE, OFF)].behavioural:
                assert 2 * DAY < at - begin < 4 * DAY
        kinds = {
            verdict
            for home in simulated["homes"]
            for verdict, _, _ in home.runs[(OUTAGE, OFF)].described
        }
        assert kinds <= {"persistent_change", "abrupt_change"}

    def test_with_the_rule_on_those_alerts_are_not_raised(
        self, simulated: dict
    ) -> None:
        results = simulated["results"]
        for condition in (ON, "h24"):
            assert results["outage"][condition]["alerts_in_the_window"] == 0
            assert results["outage"][condition]["removed"]["estimate"] > 0.0
        assert sum(simulated["results"]["timeline"]["alerts"][ON][OUTAGE]) == 0
        assert sum(simulated["results"]["timeline"]["alerts"][OFF][OUTAGE]) >= 2

    def test_with_the_rule_on_the_outage_is_reported_and_its_days_refused(
        self, simulated: dict
    ) -> None:
        protocol: SilentHomeProtocol = simulated["protocol"]
        zone = ZoneInfo("Europe/Lisbon")
        for home in simulated["homes"]:
            begin, end = protocol.outage(home.seed)
            first_day = begin.astimezone(zone).date()
            last_day = end.astimezone(zone).date()
            late = protocol.group(home.seed) == LATE
            for arm in (OUTAGE, CHANGE_AFTER_OUTAGE):
                run = home.runs[(arm, ON)]
                assert len(run.silence) == 3
                assert all(begin <= at <= end for at in run.silence)
                # Reported about a horizon after it began, and not before.
                first = (run.silence[0] - begin) / HOUR
                assert 10.0 < first <= 12.5
                # Every day the outage touched is refused, apart from the
                # first when the horizon had not passed as it closed.
                touched = [
                    first_day + timedelta(days=k)
                    for k in range((last_day - first_day).days + 1)
                ]
                assert list(run.refused) == (touched[1:] if late else touched)
                assert run.usable == protocol.days - len(run.refused)
            # A whole day's horizon never sees the outage before its first
            # day has closed.
            later = home.runs[(OUTAGE, "h24")]
            assert len(later.silence) == 2
            assert (later.silence[0] - begin) / HOUR > 22.0
            assert first_day not in later.refused
            assert set(later.refused) <= set(home.runs[(OUTAGE, ON)].refused)

    def test_the_first_outage_day_is_told_per_home(self, simulated: dict) -> None:
        protocol: SilentHomeProtocol = simulated["protocol"]
        rows = simulated["results"]["per_home"]
        for home in simulated["homes"]:
            row = rows[str(home.seed)]
            late = protocol.group(home.seed) == LATE
            assert row["group"] == protocol.group(home.seed)
            assert row[f"{OUTAGE}/{ON}/first_outage_day_refused"] is not late
            assert row[f"{OUTAGE}/h24/first_outage_day_refused"] is False
            assert row[f"{OUTAGE}/{ON}/reported"] is True
            assert row[f"{OUTAGE}/{ON}/silence_alerts_inside_the_outage"] == 3

    def test_the_rule_does_nothing_where_nothing_is_wrong(
        self, simulated: dict
    ) -> None:
        raised = 0
        for home in simulated["homes"]:
            for arm in (STABLE, CHANGE, SHORT_OUTAGE):
                assert (
                    home.runs[(arm, ON)].behavioural
                    == home.runs[(arm, OFF)].behavioural
                )
                assert home.runs[(arm, ON)].silence == ()
                assert home.runs[(arm, ON)].refused == ()
                raised += len(home.runs[(arm, OFF)].behavioural)
        # The comparison is of alerts that exist.
        assert raised >= 1

    def test_the_fleet_arms_differ_only_by_their_outage(self, simulated: dict) -> None:
        protocol: SilentHomeProtocol = simulated["protocol"]
        for home in simulated["homes"]:
            stable = home.observed[STABLE]
            assert (np.diff(stable) >= 0).all()
            for arm, (begin, end) in (
                (OUTAGE, protocol.outage(home.seed)),
                (COMMON, protocol.common_outage()),
            ):
                inside = (stable >= micro(begin)) & (stable < micro(end))
                assert inside.sum() > 100
                assert np.array_equal(home.observed[arm], stable[~inside])

    def test_a_home_is_a_function_of_its_seed(
        self, simulated: dict, tmp_path: Path
    ) -> None:
        protocol: SilentHomeProtocol = simulated["protocol"]
        first = simulated["homes"][0]
        again = run_home(first.seed, protocol)
        assert again.runs == first.runs
        for arm, moments in first.observed.items():
            assert np.array_equal(again.observed[arm], moments)

    def test_the_cohort_is_scored(self, simulated: dict) -> None:
        protocol: SilentHomeProtocol = simulated["protocol"]
        results = simulated["results"]
        assert results["result_schema"] == RESULT_SCHEMA
        assert results["homes"] == 3
        assert set(results["runs"]) == {f"{a}/{c}" for a, c in protocol.runs()}
        assert results["reporting"][ON]["reported"]["estimate"] == 1.0
        assert results["criteria"]["C3"]["verdict"] == "success"
        assert results["criteria"]["C6"]["verdict"] == "success"
        assert results["criteria"]["C8"]["verdict"] == "confirmed"
        assert results["fleet"][STABLE]["stretches"] == []
        assert len(results["fleet"][COMMON]["stretches"]) == 1
        raw = results["raw"]["homes"]
        assert set(raw) == {str(seed) for seed in protocol.study_seeds()}
        for home in simulated["homes"]:
            for (arm, condition), run in home.runs.items():
                kept = raw[str(home.seed)][f"{arm}/{condition}"]
                assert [a[0] for a in kept["behavioural_alerts"]] == [
                    at.isoformat() for at, _ in run.behavioural
                ]
                assert all(len(a) == 5 and a[2] for a in kept["behavioural_alerts"])
                assert len(kept["silence_alerts"]) == len(run.silence)
                assert len(kept["days_refused"]) == len(run.refused)
        json.dumps(results, allow_nan=False)


class TestCheckpoint:
    def test_a_finished_home_is_read_back_and_not_run_again(
        self, simulated: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pickle

        from sensor_modeling.datasets import silent_home_experiment as experiment

        protocol: SilentHomeProtocol = simulated["protocol"]
        for home in simulated["homes"]:
            with (tmp_path / f"home-{home.seed}.pickle").open("wb") as handle:
                pickle.dump(home, handle)

        def refuse(seed: int, protocol: SilentHomeProtocol) -> HomeRecord:
            raise AssertionError(f"home {seed} was run again")

        monkeypatch.setattr(experiment, "run_home", refuse)
        seen = []
        homes = run_homes(
            protocol, checkpoint=tmp_path, progress=lambda done, n: seen.append(done)
        )
        assert seen == [1, 2, 3]
        assert [home.seed for home in homes] == list(protocol.study_seeds())
        for kept, home in zip(homes, simulated["homes"]):
            assert kept.runs == home.runs
            assert all(
                np.array_equal(kept.observed[arm], home.observed[arm])
                for arm in home.observed
            )
        assert score(homes, protocol) == simulated["results"]

    def test_a_home_is_kept_as_soon_as_it_is_finished(
        self, simulated: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sensor_modeling.datasets import silent_home_experiment as experiment

        protocol: SilentHomeProtocol = simulated["protocol"]
        by_seed = {home.seed: home for home in simulated["homes"]}
        last = protocol.study_seeds()[-1]

        def cheap(seed: int, protocol: SilentHomeProtocol) -> HomeRecord:
            if seed == last:
                raise RuntimeError("the run was interrupted")
            return by_seed[seed]

        monkeypatch.setattr(experiment, "run_home", cheap)
        with pytest.raises(RuntimeError, match="interrupted"):
            run_homes(protocol, checkpoint=tmp_path / "homes")
        kept = sorted(path.name for path in (tmp_path / "homes").iterdir())
        assert kept == sorted(
            f"home-{seed}.pickle" for seed in protocol.study_seeds()[:-1]
        )


def test_the_silence_alerts_subject_has_one_name() -> None:
    assert SILENCE_SUBJECT == HOME_SILENCE == "home_silence"


# ----------------------------------------------------------------------------
# The description, on a synthetic TIHM cohort
# ----------------------------------------------------------------------------
TIHM_START = date(2019, 4, 1)
TIHM_DAYS = 28
TIHM_HOMES = ("home1", "home2", "home3", "home4")
SHARED_SILENCE = (18, 19, 20, 21)
OWN_SILENCE = ("home1", 12)
RATES = {"Bedroom": 0.4, "Bathroom": 0.3, "Kitchen": 0.8, "Lounge": 1.0, "Hallway": 0.5}


def write_tihm(directory: Path) -> Path:
    """Four homes; three share four silent days, and one has another of its own."""
    rng = np.random.default_rng(11)
    directory.mkdir(parents=True, exist_ok=True)
    activity: list[tuple[str, str, str]] = []
    for home in TIHM_HOMES:
        for offset in range(TIHM_DAYS):
            shared = offset in SHARED_SILENCE and home != "home4"
            if shared or (home, offset) == OWN_SILENCE:
                continue
            day = datetime.combine(
                TIHM_START + timedelta(days=offset), datetime.min.time()
            )
            for hour in range(24):
                for room, rate in RATES.items():
                    for _ in range(rng.poisson(rate)):
                        moment = day + timedelta(
                            hours=hour, seconds=int(rng.integers(0, 3600))
                        )
                        activity.append(
                            (home, room, moment.strftime("%Y-%m-%d %H:%M:%S"))
                        )
    for name, header, rows in (
        ("Activity.csv", ("patient_id", "location_name", "date"), activity),
        ("Labels.csv", ("patient_id", "date", "type"), []),
        (
            "Demographics.csv",
            ("patient_id", "age", "sex"),
            [(home, "(80, 90]", "Male") for home in TIHM_HOMES],
        ),
    ):
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    return directory


@pytest.fixture(scope="module")
def tihm(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    data = write_tihm(tmp_path_factory.mktemp("tihm") / "Dataset")
    protocol = declared_protocol()
    alert_burden = TihmProtocol(step_minutes=30, resamples=200)
    runs = run_tihm_homes(TihmAdapter(data), protocol, alert_burden, jobs=2)
    published = {
        "monitored_days": sum(len(run.days) for run in runs[OFF]),
        "behavioural_alerts": sum(
            len(day.alerts) for run in runs[OFF] for day in run.days
        ),
    }
    return {
        "data": data,
        "protocol": protocol,
        "alert_burden": alert_burden,
        "runs": runs,
        "published": published,
        "results": describe(runs, protocol, published=published),
    }


class TestTihmDescription:
    def test_every_home_is_run_under_every_condition(self, tihm: dict) -> None:
        runs = tihm["runs"]
        assert list(runs) == [OFF, "h12", "h24"]
        for group in runs.values():
            assert [run.household for run in group] == list(TIHM_HOMES)

    def test_nothing_is_reported_unless_the_published_run_is_reproduced(
        self, tihm: dict
    ) -> None:
        with pytest.raises(ValueError, match="nothing is reported"):
            describe(tihm["runs"], tihm["protocol"])
        wrong = {**tihm["published"], "behavioural_alerts": 10_000}
        with pytest.raises(ValueError, match="not the published record's"):
            describe(tihm["runs"], tihm["protocol"], published=wrong)

    def test_with_the_rule_off_a_silent_day_is_a_usable_day(self, tihm: dict) -> None:
        off = tihm["results"]["conditions"][OFF]
        assert off["monitored_days"] == len(TIHM_HOMES) * TIHM_DAYS
        assert off["usable_days"] == off["monitored_days"]
        assert off["days_refused_because_of_the_rule"]["all"] == 0
        # The silent days are read as behaviour, and they raise alerts.
        assert off["behavioural_alerts"]["all"] > 0
        assert (
            off["behavioural_alerts"]["all"] == tihm["published"]["behavioural_alerts"]
        )
        assert off["silence_alerts"]["all"] == 0
        assert off["behavioural_alerts"]["raised_only_with_the_rule_off"] == 0
        assert off["behavioural_alerts"]["raised_only_under_this_condition"] == 0

    def test_with_the_rule_on_the_silent_days_are_refused_and_reported(
        self, tihm: dict
    ) -> None:
        conditions = tihm["results"]["conditions"]
        off, on = conditions[OFF], conditions["h12"]
        assert on["monitored_days"] == off["monitored_days"]
        refused = on["days_refused_because_of_the_rule"]
        assert refused["homes_with_one"] == 3
        assert refused["per_home"]["home4"] == 0
        assert refused["per_home"]["home2"] >= len(SHARED_SILENCE)
        assert refused["per_home"]["home1"] > refused["per_home"]["home2"]
        assert on["usable_days"] == off["usable_days"] - refused["all"]
        silences = on["silence_alerts"]
        assert silences["homes_with_one"] == 3
        assert silences["per_home"]["home4"] == 0
        assert silences["all"] == sum(silences["per_home"].values())
        assert silences["per_monitored_day"] == silences["all"] / on["monitored_days"]
        alerts = on["behavioural_alerts"]
        assert (
            alerts["also_raised_with_the_rule_off"]
            + alerts["raised_only_under_this_condition"]
            == alerts["all"]
        )
        assert (
            alerts["also_raised_with_the_rule_off"]
            + alerts["raised_only_with_the_rule_off"]
            == off["behavioural_alerts"]["all"]
        )
        assert sum(alerts["by_day_summarised"].values()) == alerts["all"]
        assert sum(alerts["by_date_raised"].values()) == alerts["all"]
        counted = off["behavioural_alerts"]
        assert sum(counted["by_day_summarised"].values()) == counted["all"] > 0
        # An alert is raised when its day closes, which is the next local date.
        assert {
            date.fromisoformat(day) + timedelta(days=1)
            for day in counted["by_day_summarised"]
        } == {date.fromisoformat(day) for day in counted["by_date_raised"]}
        # Every alert of this cohort came from its silent days.
        assert (
            alerts["raised_only_with_the_rule_off"] == off["behavioural_alerts"]["all"]
        )
        assert alerts["all"] == 0

    def test_the_shared_silence_is_one_stretch_and_the_private_one_is_not(
        self, tihm: dict
    ) -> None:
        results = tihm["results"]
        (stretch,) = results["fleet"]["stretches"]
        assert stretch["homes"] == 3
        assert stretch["monitored"] == 4
        first = datetime.combine(
            TIHM_START + timedelta(days=SHARED_SILENCE[0]), datetime.min.time()
        ).replace(tzinfo=timezone.utc)
        assert first - 2 * DAY < datetime.fromisoformat(stretch["since"]) < first + DAY
        assert results["fleet"]["largest_share_silent"] == 0.75
        silences = results["conditions"]["h12"]["silence_alerts"]
        # home1's own silent day is reported by an alert outside the stretch.
        assert 0 < silences["inside_a_stretch_of_common_silence"] < silences["all"]

    def test_every_home_is_compared_with_the_published_record(self, tihm: dict) -> None:
        runs, protocol, published = tihm["runs"], tihm["protocol"], tihm["published"]
        table = {
            run.household: {
                "monitored_days": len(run.days),
                "behavioural_alerts": sum(len(day.alerts) for day in run.days),
                "usable_days": 0,
            }
            for run in runs[OFF]
        }
        results = describe(runs, protocol, published=published, published_homes=table)
        assert results["check"]["homes_compared_one_by_one"] == len(TIHM_HOMES)
        assert tihm["results"]["check"]["homes_compared_one_by_one"] == 0
        # The totals agree and two homes do not: an alert moved from one home
        # to another.
        with_one = next(n for n, row in table.items() if row["behavioural_alerts"])
        without = next(n for n, row in table.items() if not row["behavioural_alerts"])
        moved = {name: dict(row) for name, row in table.items()}
        moved[with_one]["behavioural_alerts"] -= 1
        moved[without]["behavioural_alerts"] += 1
        with pytest.raises(ValueError, match=r"in 2 home\(s\)"):
            describe(runs, protocol, published=published, published_homes=moved)
        # A home the published record does not have.
        missing = {name: row for name, row in table.items() if name != "home4"}
        with pytest.raises(ValueError, match="home4"):
            describe(runs, protocol, published=published, published_homes=missing)

    def test_alerts_are_compared_with_the_rule_off_by_home_day_and_subject(
        self,
    ) -> None:
        zone = ZoneInfo("Europe/London")
        start = datetime(2019, 5, 1, tzinfo=zone)

        def home(name: str, alerts: dict[int, tuple[str, ...]]) -> HouseholdRun:
            days = tuple(
                DayRecord(
                    day=(start + timedelta(days=k)).date(),
                    usable=True,
                    coverage=1.0,
                    observed=1.0,
                    abstention=0.0,
                    kinds={"sleeping_hours": "ordinary"},
                    deviations={"sleeping_hours": 0.0},
                    alerts=tuple((s, "attention") for s in alerts.get(k, ())),
                    events=10,
                    labels=frozenset(),
                    silent=0.25 if k == 3 else 0.0,
                )
                for k in range(6)
            )
            return HouseholdRun(
                household=name,
                days=days,
                system_health_alerts=0,
                data_quality_alerts=0,
                hourly={},
                label_times={},
                dispositions={},
                issues={},
                closes=tuple(start + timedelta(days=k + 1) for k in range(6)),
                silence_alerts=(start + timedelta(days=3, hours=14),),
                observed=tuple(start + timedelta(hours=h) for h in range(6 * 24)),
            )

        off = [
            home("a", {1: ("sleeping_hours",), 2: ("sleeping_hours", "away_hours")}),
            home("b", {4: ("sleeping_hours",)}),
            home("c", {}),
        ]
        on = [
            # Kept, removed, and the same subject on another day: a new alert.
            home("a", {1: ("sleeping_hours",), 5: ("away_hours",)}),
            # The same day and subject in another home is not the same alert.
            home("b", {}),
            home("c", {4: ("sleeping_hours",)}),
        ]
        protocol = declared_protocol()
        results = describe(
            {OFF: off, "h12": on, "h24": off},
            protocol,
            published={"monitored_days": 18, "behavioural_alerts": 4},
            zone=zone,
        )
        alerts = results["conditions"]["h12"]["behavioural_alerts"]
        assert alerts["all"] == 3
        assert alerts["also_raised_with_the_rule_off"] == 1
        assert alerts["raised_only_with_the_rule_off"] == 3
        assert alerts["raised_only_under_this_condition"] == 2
        assert alerts["homes_with_one"] == 2
        assert alerts["by_feature"] == {"away_hours": 1, "sleeping_hours": 2}
        assert alerts["by_day_summarised"] == {
            "2019-05-02": 1,
            "2019-05-05": 1,
            "2019-05-06": 1,
        }
        assert alerts["by_date_raised"] == {
            "2019-05-03": 1,
            "2019-05-06": 1,
            "2019-05-07": 1,
        }
        same = results["conditions"]["h24"]["behavioural_alerts"]
        assert same["raised_only_with_the_rule_off"] == 0
        assert same["raised_only_under_this_condition"] == 0
        refused = results["conditions"]["h12"]["days_refused_because_of_the_rule"]
        assert refused == {
            "all": 3,
            "homes_with_one": 3,
            "per_home": dict.fromkeys("abc", 1),
        }
        silences = results["conditions"]["h12"]["silence_alerts"]
        assert silences["all"] == 3
        assert silences["per_monitored_day"] == 3 / 18
        # Hourly records without a gap: no stretch for an alert to be inside.
        assert results["fleet"]["stretches"] == []
        assert silences["inside_a_stretch_of_common_silence"] == 0

    def test_the_record_is_a_description(self, tihm: dict, tmp_path: Path) -> None:
        result = describe_tihm(
            TihmAdapter(tihm["data"]),
            tihm["protocol"],
            tihm["alert_burden"],
            protocol_sha256="0" * 64,
            output_dir=tmp_path,
            jobs=1,
            published=tihm["published"],
        )
        assert result.path == tmp_path / f"{TIHM_EXPERIMENT}.json"
        payload = load_record(result.path)
        assert payload["experiment"] == TIHM_EXPERIMENT
        assert payload["data_source"] == "tihm"
        assert payload["results"]["result_schema"] == TIHM_SCHEMA
        assert payload["results"] == json.loads(json.dumps(tihm["results"]))
        assert any("A description, not a test" in note for note in payload["notes"])
        assert SIMULATOR_NOTE not in payload["notes"]
        assert payload["configuration"]["protocol_sha256"] == tihm["protocol"].sha256()
