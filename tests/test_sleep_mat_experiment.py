"""The sleep-mat comparison's machinery, on homes and mats built by hand."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from scipy import stats

from sensor_modeling.baseline.adaptive import AdaptiveBaseline, BaselineConfig
from sensor_modeling.datasets.sleep_mat_experiment import (
    MatDay,
    MatHome,
    Pairs,
    PipelineDay,
    SimulatedHome,
    agreement,
    alerts_beside_the_mat,
    check_published,
    correlation,
    deviations,
    home_statistics,
    level_reading,
    limits_of_agreement,
    mat_verdicts,
    mean_over_homes,
    observed_days,
    pairs_of,
    pipeline_days,
    read_mat,
    score_simulated,
    silent_days,
    tracking_reading,
    true_sleep_hours,
)
from sensor_modeling.datasets.sleep_mat_protocol import (
    IN_BED,
    SLEEP,
    SleepMatProtocol,
)
from sensor_modeling.datasets.tihm_experiment import DayRecord, HouseholdRun
from sensor_modeling.simulation.household import Episode, GroundTruth
from sensor_modeling.states.ontology import BehaviouralState

HEADER = "patient_id,date,state,heart_rate,respiratory_rate,snoring\n"
D0 = date(2019, 5, 1)


def day(n: int) -> date:
    return D0 + timedelta(days=n)


def write_mat(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path.write_text(
        HEADER + "".join(f"{h},{t},{s},60.0,14.0,False\n" for h, t, s in rows),
        encoding="utf-8",
    )
    return path


def minutes(
    home: str, start: str, count: int, state: str
) -> list[tuple[str, str, str]]:
    begin = datetime.fromisoformat(start)
    return [
        (home, (begin + timedelta(minutes=k)).strftime("%Y-%m-%d %H:%M:%S"), state)
        for k in range(count)
    ]


def a_mat(home: str, values: dict[date, tuple[float, float]]) -> MatHome:
    days = {d: MatDay(sleep=s, in_bed=b) for d, (s, b) in values.items()}
    return MatHome(home=home, days=days, observed=observed_days(days))


def a_day(n: int, value: float, **kwargs: object) -> PipelineDay:
    defaults: dict[str, object] = {
        "usable": True,
        "kind": "ordinary",
        "deviation": 0.0,
        "alerts": 0,
        "events": 100,
    }
    defaults.update(kwargs)
    return PipelineDay(day=day(n), value=value, **defaults)  # type: ignore[arg-type]


class TestTheMat:
    def test_a_record_is_a_minute_and_awake_is_in_bed_not_asleep(
        self, tmp_path: Path
    ) -> None:
        rows = (
            minutes("h1", "2019-05-01 22:00:00", 60, "AWAKE")
            + minutes("h1", "2019-05-01 23:00:00", 120, "LIGHT")
            + minutes("h1", "2019-05-02 01:00:00", 90, "DEEP")
            + minutes("h1", "2019-05-02 02:30:00", 30, "REM")
        )
        homes = read_mat(write_mat(tmp_path / "Sleep.csv", rows))
        h1 = homes["h1"]
        assert h1.days[day(0)] == MatDay(sleep=1.0, in_bed=2.0)
        assert h1.days[day(1)] == MatDay(sleep=3.0, in_bed=3.0)

    def test_a_shift_moves_minutes_across_midnight(self, tmp_path: Path) -> None:
        rows = minutes("h1", "2019-05-01 23:30:00", 60, "LIGHT")
        path = write_mat(tmp_path / "Sleep.csv", rows)
        assert read_mat(path)["h1"].days[day(0)].sleep == 0.5
        later = read_mat(path, shift_hours=1.0)["h1"].days
        assert day(0) not in later and later[day(1)].sleep == 1.0
        earlier = read_mat(path, shift_hours=-1.0)["h1"].days
        assert earlier[day(0)].sleep == 1.0 and day(1) not in earlier

    def test_a_day_is_observed_only_between_two_days_with_records(self) -> None:
        days = [day(0), day(1), day(2), day(4), day(5), day(6), day(7)]
        assert observed_days(days) == {day(1), day(5), day(6)}

    def test_an_unknown_state_or_a_repeated_minute_is_refused(
        self, tmp_path: Path
    ) -> None:
        bad = write_mat(tmp_path / "a.csv", [("h1", "2019-05-01 22:00:00", "NAP")])
        with pytest.raises(ValueError, match="unknown state"):
            read_mat(bad)
        twice = write_mat(
            tmp_path / "b.csv",
            [("h1", "2019-05-01 22:00:00", "REM")] * 2,
        )
        with pytest.raises(ValueError, match="repeats"):
            read_mat(twice)

    def test_other_columns_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "c.csv"
        path.write_text("patient_id,date,state\nh1,2019-05-01 22:00:00,REM\n")
        with pytest.raises(ValueError, match="columns"):
            read_mat(path)


class TestMatching:
    def test_a_matched_day_is_usable_and_observed_by_the_mat(self) -> None:
        mat = a_mat("h", {day(n): (7.0 + n, 8.0 + n) for n in range(0, 6)})
        days = [a_day(n, 10.0 + n) for n in range(0, 7)]
        days[2] = a_day(2, 12.0, usable=False)
        pairs = pairs_of("h", days, mat)
        assert pairs.days == (day(1), day(3), day(4))
        assert list(pairs.pipeline) == [11.0, 13.0, 14.0]
        assert list(pairs.references[SLEEP]) == [8.0, 10.0, 11.0]
        assert list(pairs.references[IN_BED]) == [9.0, 11.0, 12.0]

    def test_a_home_without_a_mat_has_no_matched_day(self) -> None:
        assert pairs_of("h", [a_day(1, 9.0)], None).size == 0

    def test_pipeline_days_read_the_run(self) -> None:
        record = DayRecord(
            day=day(3),
            usable=True,
            coverage=1.0,
            observed=1.0,
            abstention=0.0,
            kinds={"sleeping_hours": "abrupt_change", "away_hours": "ordinary"},
            deviations={"sleeping_hours": -3.4, "away_hours": 0.2},
            alerts=(("sleeping_hours", "attention"), ("away_hours", "information")),
            events=42,
            labels=frozenset(),
            hours={"sleeping": 7.25, "away": 1.0},
        )
        run = HouseholdRun(
            household="h",
            days=(record,),
            system_health_alerts=0,
            data_quality_alerts=0,
            hourly={},
            label_times={},
            dispositions={},
            issues={},
            closes=(),
        )
        (only,) = pipeline_days(run)
        assert only == PipelineDay(
            day=day(3),
            usable=True,
            value=7.25,
            kind="abrupt_change",
            deviation=-3.4,
            alerts=1,
            events=42,
        )
        assert only.has_verdict


class TestPerHome:
    def test_correlations_are_scipys_and_none_when_constant(self) -> None:
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        y = np.array([2.0, 2.5, 2.0, 6.0, 3.0])
        assert correlation(x, y, "spearman") == pytest.approx(stats.spearmanr(x, y)[0])
        assert correlation(x, y, "pearson") == pytest.approx(stats.pearsonr(x, y)[0])
        assert correlation(x, np.full(5, 7.0), "spearman") is None
        assert correlation(x[:2], y[:2], "spearman") is None

    def test_a_home_needs_the_declared_number_of_matched_days(self) -> None:
        protocol = SleepMatProtocol()
        n = protocol.min_matched_days
        values = np.arange(n, dtype=float)
        pairs = Pairs(
            "h",
            tuple(day(k) for k in range(n)),
            values + 2.0,
            {SLEEP: values, IN_BED: values + 1.0},
        )
        row = home_statistics(pairs, protocol)
        assert row["included"] and row["matched_days"] == n
        assert row[SLEEP]["mean_difference"] == pytest.approx(2.0)
        assert row[IN_BED]["mean_difference"] == pytest.approx(1.0)
        assert row[SLEEP]["variance_of_differences"] == pytest.approx(0.0)
        fewer = Pairs(
            "h",
            pairs.days[:-1],
            pairs.pipeline[:-1],
            {SLEEP: values[:-1], IN_BED: values[:-1]},
        )
        assert not home_statistics(fewer, protocol)["included"]

    def test_the_t_interval_is_the_textbook_one(self) -> None:
        values = [0.2, 0.5, 0.4, 0.8]
        found = mean_over_homes(values, 0.95)
        half = stats.t.ppf(0.975, 3) * np.std(values, ddof=1) / 2.0
        assert found["estimate"] == pytest.approx(0.475)
        assert found["interval"]["low"] == pytest.approx(0.475 - half)
        assert found["interval"]["high"] == pytest.approx(0.475 + half)
        assert mean_over_homes([], 0.95)["estimate"] is None
        assert mean_over_homes([0.3], 0.95)["interval"] is None

    def test_limits_of_agreement_add_the_two_spreads(self) -> None:
        found = limits_of_agreement([1.0, 3.0], [4.0, 0.0])
        spread = 1.96 * math.sqrt(2.0 + 2.0)
        assert found["bias"] == 2.0
        assert (found["low"], found["high"]) == pytest.approx((2 - spread, 2 + spread))


class TestCriteria:
    @pytest.mark.parametrize(
        ("low", "high", "homes", "reading"),
        [
            (0.51, 0.8, 14, "follows"),
            (0.1, 0.49, 14, "does_not_follow"),
            (0.4, 0.6, 14, "inconclusive"),
            (0.5, 0.7, 14, "inconclusive"),
            (0.6, 0.8, 2, "not_estimable"),
        ],
    )
    def test_c1(self, low: float, high: float, homes: int, reading: str) -> None:
        estimate = {"interval": {"low": low, "high": high}, "homes": homes}
        assert tracking_reading(estimate, 0.5) == reading

    @pytest.mark.parametrize(
        ("low", "high", "reading"),
        [
            (-0.9, 0.9, "agrees"),
            (-1.0, 0.5, "inconclusive"),
            (1.0, 2.0, "inconclusive"),
            (1.01, 2.0, "does_not_agree"),
            (-3.0, -1.01, "does_not_agree"),
        ],
    )
    def test_c2(self, low: float, high: float, reading: str) -> None:
        estimate = {"interval": {"low": low, "high": high}, "homes": 14}
        assert level_reading(estimate, 1.0) == reading

    def test_agreement_reads_only_included_homes(self) -> None:
        protocol = SleepMatProtocol()
        rng = np.random.default_rng(5)
        table = {}
        for k in range(4):
            n = protocol.min_matched_days + (k if k < 3 else -1)
            x = rng.standard_normal(n)
            pairs = Pairs(
                f"h{k}",
                tuple(day(j) for j in range(n)),
                x + 3.0,
                {SLEEP: x + rng.standard_normal(n), IN_BED: x},
            )
            table[f"h{k}"] = home_statistics(pairs, protocol)
        found = agreement(table, protocol)
        assert found["homes"] == ["h0", "h1", "h2"]
        expected = np.mean([table[h][SLEEP]["spearman"] for h in ("h0", "h1", "h2")])
        assert found[SLEEP]["spearman"]["estimate"] == pytest.approx(expected)
        assert found[IN_BED]["mean_difference"]["estimate"] == pytest.approx(3.0)


class TestWhatElse:
    def test_the_mats_verdicts_are_a_default_baselines(self) -> None:
        values = {day(n): (7.0 + (n % 3) * 0.5, 9.0) for n in range(0, 30)}
        mat = a_mat("h", values)
        baseline = AdaptiveBaseline("sleeping_hours", BaselineConfig())
        expected = {}
        for d in sorted(mat.observed):
            change = baseline.observe(d, values[d][0])
            expected[d] = (change.kind.value, change.deviation)
        assert mat_verdicts(mat) == pytest.approx(expected)

    def test_deviations_count_days_past_the_threshold(self) -> None:
        protocol = SleepMatProtocol()
        values = {day(n): (7.0 + 0.1 * (n % 5), 9.0) for n in range(0, 40)}
        values[day(30)] = (12.0, 13.0)
        mat = a_mat("h", values)
        verdicts = mat_verdicts(mat)
        assert verdicts[day(30)][1] > protocol.alert_threshold
        days = [a_day(n, 8.0, deviation=0.1) for n in range(0, 40)]
        days[30] = a_day(30, 14.0, deviation=4.0)
        days[31] = a_day(31, 3.0, deviation=-5.0)
        found = deviations({"h": days}, {"h": mat}, protocol)
        assert found["pipeline_past_threshold"] == 2
        assert found["same_sign"] == 1 + (verdicts[day(31)][1] < 0)
        assert found["also_past_threshold"] >= 1

    def test_an_alert_is_judged_by_the_mats_side_of_its_median(self) -> None:
        mat = a_mat("h", {day(n): (6.0 + n, 9.0) for n in range(0, 6)})
        days = [a_day(n, 8.0) for n in range(0, 7)]
        days[1] = a_day(1, 4.0, deviation=-3.5, alerts=1)  # mat 7, below the median
        days[4] = a_day(4, 12.0, deviation=3.5, alerts=2)  # mat 10, above it
        days[6] = a_day(6, 12.0, deviation=3.5, alerts=1)  # not matched
        pairs = {"h": pairs_of("h", days, mat)}
        found = alerts_beside_the_mat({"h": days}, {"h": mat}, pairs)
        assert found["alerts"] == 4 and found["on_matched_days"] == 3
        assert found["same_side"] == 3 and found["opposite"] == 0

    def test_silent_days_are_days_without_an_activity_record(self) -> None:
        mat = a_mat("h", {day(n): (7.0, 8.0 + n) for n in range(0, 5)})
        days = [a_day(n, 8.0) for n in range(0, 6)]
        days[2] = a_day(2, 23.5, events=0)
        days[5] = a_day(5, 24.0, events=0)
        found = silent_days({"h": days}, {"h": mat})
        assert found["silent_days"] == 2 and found["mat_observed"] == 1
        assert found["median_pipeline_hours"] == 23.5
        assert found["median_mat_in_bed_hours"] == 10.0


class TestTheCheck:
    def test_a_run_that_is_not_the_published_one_reports_nothing(self) -> None:
        record = DayRecord(
            day=day(0),
            usable=True,
            coverage=1.0,
            observed=1.0,
            abstention=0.0,
            kinds={},
            deviations={},
            alerts=(("sleeping_hours", "attention"),),
            events=1,
            labels=frozenset(),
        )
        run = HouseholdRun("h", (record,), 0, 0, {}, {}, {}, {}, ())
        good = {"h": {"monitored_days": 1, "behavioural_alerts": 1}}
        assert check_published({"h": run}, good)["reproduced"]
        with pytest.raises(ValueError, match="nothing is reported"):
            check_published(
                {"h": run}, {"h": {"monitored_days": 1, "behavioural_alerts": 0}}
            )
        with pytest.raises(ValueError, match="nothing is reported"):
            check_published({"h": run}, {})


class TestTheSimulatedReference:
    def test_sleep_is_split_at_local_midnight(self) -> None:
        zone = ZoneInfo("Europe/London")
        start = datetime(2019, 5, 1, 22, 0, tzinfo=zone).astimezone(timezone.utc)
        episodes = (
            Episode(
                start, start + timedelta(hours=9), BehaviouralState.SLEEPING, "bedroom"
            ),
            Episode(
                start + timedelta(hours=9),
                start + timedelta(hours=10),
                BehaviouralState.AWAY,
                None,
            ),
        )
        truth = GroundTruth(episodes=episodes, visitors=(), tz=zone)
        assert true_sleep_hours(truth, zone) == {day(0): 2.0, day(1): 7.0}

    def test_the_reference_scores_the_truth(self) -> None:
        protocol = SleepMatProtocol()
        n = protocol.min_matched_days
        rng = np.random.default_rng(2)
        homes = []
        for seed in (1, 2, 3):
            truth = 7.0 + rng.standard_normal(n)
            homes.append(
                SimulatedHome(
                    seed=seed,
                    days=tuple(day(k) for k in range(n)),
                    pipeline=tuple(truth + 1.5),
                    truth=tuple(truth),
                    usable_days=n,
                    closed_days=n + 2,
                )
            )
        found = score_simulated(homes, protocol)
        assert found["included"] == 3
        assert found[SLEEP]["spearman"]["estimate"] == pytest.approx(1.0)
        assert found[SLEEP]["mean_difference"]["estimate"] == pytest.approx(1.5)
