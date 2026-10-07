"""Tests for silence across a fleet of homes.

One silent home says nothing about the service that collects every home's
records. Most of the homes silent together does. The tests check that the
fleet check tells those apart, dates a common silence from the last home still
speaking, and does not mistake a home that has not joined, or has left, for a
silent one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sensor_modeling.health import (
    CommonSilence,
    FleetConfig,
    assess_fleet,
    common_silences,
    replay_fleet,
)

T0 = datetime(2024, 5, 1, tzinfo=timezone.utc)
HORIZON = timedelta(hours=12)
CONFIG = FleetConfig(horizon=HORIZON)


def hours(*values: float) -> list[datetime]:
    return [T0 + timedelta(hours=value) for value in values]


def hourly(first: int, last: int, gap: tuple[int, int] | None = None) -> list[datetime]:
    """One observation an hour from *first* to *last*, without those in *gap*."""
    return [
        T0 + timedelta(hours=hour)
        for hour in range(first, last + 1)
        if gap is None or not gap[0] < hour < gap[1]
    ]


class TestAssessment:
    def test_homes_that_are_reporting_share_no_silence(self) -> None:
        now = T0 + timedelta(hours=20)
        seen = {name: now - timedelta(hours=1) for name in "abcd"}
        verdict = assess_fleet(seen, now, CONFIG)
        assert verdict.silent == ()
        assert not verdict.common_cause
        assert verdict.since is None
        assert verdict.share == 0.0

    def test_most_homes_silent_together_is_a_common_cause(self) -> None:
        now = T0 + timedelta(hours=40)
        seen = {
            "a": T0 + timedelta(hours=20),
            "b": T0 + timedelta(hours=22),
            "c": T0 + timedelta(hours=21),
            "d": T0 + timedelta(hours=39),
        }
        verdict = assess_fleet(seen, now, CONFIG)
        assert verdict.silent == ("a", "b", "c")
        assert verdict.share == 0.75
        assert verdict.common_cause
        # It can only have begun after the last silent home was still speaking.
        assert verdict.since == T0 + timedelta(hours=22)
        assert verdict.to_dict()["monitored"] == 4

    def test_a_home_is_silent_only_from_the_horizon(self) -> None:
        seen = {name: T0 for name in "abc"}
        early = assess_fleet(seen, T0 + HORIZON - timedelta(seconds=1), CONFIG)
        assert early.silent == () and not early.common_cause
        assert assess_fleet(seen, T0 + HORIZON, CONFIG).common_cause

    def test_a_few_silent_homes_among_many_are_their_own_affair(self) -> None:
        now = T0 + timedelta(hours=40)
        seen = {f"home{k}": now - timedelta(hours=1) for k in range(7)}
        seen.update({name: T0 for name in "abc"})
        verdict = assess_fleet(seen, now, CONFIG)
        assert len(verdict.silent) == 3
        assert verdict.share == 0.3
        assert not verdict.common_cause

    def test_two_silent_homes_are_not_a_fleet(self) -> None:
        now = T0 + timedelta(hours=40)
        verdict = assess_fleet({"a": T0, "b": T0}, now, CONFIG)
        assert verdict.share == 1.0
        assert not verdict.common_cause
        assert assess_fleet(
            {"a": T0, "b": T0}, now, FleetConfig(horizon=HORIZON, min_homes=2)
        ).common_cause

    def test_a_home_that_never_reported_is_not_counted(self) -> None:
        now = T0 + timedelta(hours=40)
        seen = {"a": T0, "b": T0, "c": T0, "d": None, "e": None, "f": None}
        verdict = assess_fleet(seen, now, CONFIG)
        assert verdict.monitored == ("a", "b", "c")
        assert verdict.common_cause

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"horizon": timedelta(0)},
            {"horizon": HORIZON, "fraction": 0.0},
            {"horizon": HORIZON, "fraction": 1.5},
            {"horizon": HORIZON, "min_homes": 1},
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            FleetConfig(**kwargs)

    def test_the_configuration_serialises(self) -> None:
        assert CONFIG.to_dict() == {
            "horizon_hours": 12.0,
            "fraction": 0.6,
            "min_homes": 3,
        }


class TestRecordedFleet:
    def test_a_shared_gap_is_one_common_silence(self) -> None:
        fleet = {name: hourly(0, 200, gap=(60, 100)) for name in "abcde"}
        found = common_silences(fleet, CONFIG)
        assert len(found) == 1
        silence = found[0]
        assert isinstance(silence, CommonSilence)
        assert silence.since == T0 + timedelta(hours=60)
        assert silence.detected == T0 + timedelta(hours=72)
        assert silence.until == T0 + timedelta(hours=99)
        assert silence.homes == 5 and silence.monitored == 5
        assert silence.covers(T0 + timedelta(hours=80))
        assert not silence.covers(T0 + timedelta(hours=59))
        assert silence.to_dict()["homes"] == 5

    def test_gaps_that_do_not_overlap_are_each_homes_own(self) -> None:
        fleet = {
            name: hourly(0, 400, gap=(40 + 60 * k, 70 + 60 * k))
            for k, name in enumerate("abcde")
        }
        assert common_silences(fleet, CONFIG) == []

    def test_a_fleet_with_no_gap_has_no_common_silence(self) -> None:
        fleet = {name: hourly(0, 100) for name in "abcd"}
        assert common_silences(fleet, CONFIG) == []

    def test_homes_not_yet_joined_or_already_gone_are_not_silent(self) -> None:
        fleet = {
            "a": hourly(0, 300),
            "b": hourly(0, 300),
            "c": hourly(0, 300),
            # Three homes that join late and three that leave early.
            "late1": hourly(200, 300),
            "late2": hourly(200, 300),
            "late3": hourly(200, 300),
            "gone1": hourly(0, 50),
            "gone2": hourly(0, 50),
            "gone3": hourly(0, 50),
        }
        assert common_silences(fleet, CONFIG) == []

    def test_the_onset_follows_the_last_home_still_speaking(self) -> None:
        fleet = {
            "a": hourly(0, 200, gap=(60, 100)),
            "b": hourly(0, 200, gap=(62, 100)),
            "c": hourly(0, 200, gap=(65, 100)),
        }
        (silence,) = common_silences(fleet, CONFIG)
        assert silence.since == T0 + timedelta(hours=65)
        assert silence.detected == T0 + timedelta(hours=77)

    def test_the_spacing_of_assessments_bounds_the_detection_time(self) -> None:
        fleet = {name: hourly(0, 200, gap=(60, 100)) for name in "abc"}
        (silence,) = common_silences(fleet, CONFIG, every=timedelta(hours=5))
        assert silence.since == T0 + timedelta(hours=60)
        assert silence.detected == T0 + timedelta(hours=75)

    def test_unordered_observations_are_refused(self) -> None:
        with pytest.raises(ValueError, match="not in time order"):
            common_silences({"a": hours(3, 1, 2)}, CONFIG)

    def test_the_spacing_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="every must be positive"):
            common_silences({"a": hours(1, 2)}, CONFIG, every=timedelta(0))

    def test_an_empty_fleet_has_nothing_to_find(self) -> None:
        assert common_silences({}, CONFIG) == []
        assert common_silences({"a": []}, CONFIG) == []


class TestReplay:
    def test_it_assesses_the_fleet_at_every_spacing_from_first_to_last(self) -> None:
        fleet = {name: hourly(0, 48) for name in "abc"}
        verdicts = list(replay_fleet(fleet, CONFIG))
        assert [v.at for v in verdicts] == [T0 + timedelta(hours=h) for h in range(49)]
        assert all(v.monitored == ("a", "b", "c") for v in verdicts)
        assert all(v.share == 0.0 and not v.common_cause for v in verdicts)

    def test_it_gives_the_share_of_homes_silent_at_each_assessment(self) -> None:
        fleet = {
            "a": hourly(0, 100, gap=(20, 60)),
            "b": hourly(0, 100),
            "c": hourly(0, 100),
            "d": hourly(0, 100),
        }
        shares = {v.at: v.share for v in replay_fleet(fleet, CONFIG)}
        # One home in four is silent from twelve hours after it was last heard.
        assert shares[T0 + timedelta(hours=31)] == 0.0
        assert shares[T0 + timedelta(hours=32)] == 0.25
        assert shares[T0 + timedelta(hours=59)] == 0.25
        assert shares[T0 + timedelta(hours=60)] == 0.0
        assert max(shares.values()) == 0.25
        # A quarter of the homes is not a common cause.
        assert common_silences(fleet, CONFIG) == []

    def test_the_stretches_are_the_assessments_that_call_a_common_cause(self) -> None:
        fleet = {name: hourly(0, 200, gap=(60, 100)) for name in "abc"}
        called = [v.at for v in replay_fleet(fleet, CONFIG) if v.common_cause]
        (silence,) = common_silences(fleet, CONFIG)
        assert silence.detected == called[0]
        assert silence.until == called[-1]

    def test_it_refuses_what_the_stretches_refuse(self) -> None:
        with pytest.raises(ValueError, match="not in time order"):
            list(replay_fleet({"a": hours(3, 1, 2)}, CONFIG))
        with pytest.raises(ValueError, match="every must be positive"):
            list(replay_fleet({"a": hours(1, 2)}, CONFIG, every=timedelta(0)))
        assert list(replay_fleet({}, CONFIG)) == []
