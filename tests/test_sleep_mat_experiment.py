"""The sleep-mat comparison's machinery, on homes and mats built by hand."""

from __future__ import annotations

import dataclasses
import json
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
    check_rule,
    correlation,
    deviations,
    home_statistics,
    hourly_alignment,
    level_reading,
    limits_of_agreement,
    mat_verdicts,
    matched_days,
    mean_over_homes,
    observed_days,
    pairs_of,
    pipeline_days,
    read_mat,
    run_simulated_home,
    score,
    score_simulated,
    silenced_days,
    silent_days,
    staged_awake_homes,
    tihm_record,
    tracking_reading,
    true_sleep_hours,
)
from sensor_modeling.datasets.sleep_mat_protocol import (
    IN_BED,
    PRIMARY,
    RULE,
    SLEEP,
    STAGED_AWAKE_HOMES,
    WITH_SILENT_DAYS,
    SleepMatProtocol,
)
from sensor_modeling.datasets.sleep_mat_summary import render_page
from sensor_modeling.datasets.tihm_experiment import (
    DayRecord,
    HouseholdRun,
    _per_household,
)
from sensor_modeling.datasets.tihm_protocol import declared_protocol as tihm_protocol
from sensor_modeling.simulation.household import Episode, GroundTruth
from sensor_modeling.states.ontology import BehaviouralState

HEADER = "patient_id,date,state,heart_rate,respiratory_rate,snoring\n"
D0 = date(2019, 5, 1)
PROTOCOL = SleepMatProtocol()


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


def a_mat(
    home: str,
    values: dict[date, tuple[float, float]],
    hourly: dict[tuple[date, int], int] | None = None,
) -> MatHome:
    days = {d: MatDay(sleep=s, in_bed=b) for d, (s, b) in values.items()}
    return MatHome(
        home=home, days=days, observed=observed_days(days), hourly=hourly or {}
    )


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


def a_record(n: int, sleeping: float, **kwargs: object) -> DayRecord:
    fields: dict[str, object] = {
        "day": day(n),
        "usable": True,
        "coverage": 1.0,
        "observed": 1.0,
        "abstention": 0.0,
        "kinds": {"sleeping_hours": "ordinary"},
        "deviations": {"sleeping_hours": 0.0},
        "alerts": (),
        "events": 50,
        "labels": frozenset(),
        "hours": {"sleeping": sleeping},
    }
    fields.update(kwargs)
    return DayRecord(**fields)  # type: ignore[arg-type]


def a_run(home: str, records: list[DayRecord], silences: int = 0) -> HouseholdRun:
    hourly = {}
    for record in records:
        for hour in range(24):
            # Activity in the day, none at night, scaled by the day's events.
            hourly[(record.day, hour)] = (
                0 if hour < 7 or hour >= 23 else record.events // 16
            )
    return HouseholdRun(
        household=home,
        days=tuple(records),
        system_health_alerts=0,
        data_quality_alerts=silences,
        hourly=hourly,
        label_times={},
        dispositions={},
        issues={},
        closes=tuple(
            datetime.combine(
                r.day + timedelta(days=1), datetime.min.time(), timezone.utc
            )
            for r in records
        ),
        silence_alerts=tuple(
            datetime(2019, 5, 2, tzinfo=timezone.utc) for _ in range(silences)
        ),
    )


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
        h1 = read_mat(write_mat(tmp_path / "Sleep.csv", rows))["h1"]
        assert h1.days[day(0)] == MatDay(sleep=1.0, in_bed=2.0)
        assert h1.days[day(1)] == MatDay(sleep=3.0, in_bed=3.0)
        assert h1.records == 300
        assert h1.hourly[(day(0), 22)] == 60 and h1.hourly[(day(1), 2)] == 60

    def test_a_shift_moves_minutes_across_midnight(self, tmp_path: Path) -> None:
        rows = minutes("h1", "2019-05-01 23:30:00", 60, "LIGHT")
        path = write_mat(tmp_path / "Sleep.csv", rows)
        assert read_mat(path)["h1"].days[day(0)].sleep == 0.5
        later = read_mat(path, shift_hours=1.0)["h1"]
        assert day(0) not in later.days and later.days[day(1)].sleep == 1.0
        assert later.hourly[(day(1), 0)] == 30
        earlier = read_mat(path, shift_hours=-1.0)["h1"].days
        assert earlier[day(0)].sleep == 1.0 and day(1) not in earlier

    def test_a_day_is_observed_only_between_two_days_with_records(self) -> None:
        days = [day(0), day(1), day(2), day(4), day(5), day(6), day(7)]
        assert observed_days(days) == {day(1), day(5), day(6)}

    def test_an_unknown_state_a_repeated_minute_or_an_offset_is_refused(
        self, tmp_path: Path
    ) -> None:
        bad = write_mat(tmp_path / "a.csv", [("h1", "2019-05-01 22:00:00", "NAP")])
        with pytest.raises(ValueError, match="unknown state"):
            read_mat(bad)
        twice = write_mat(
            tmp_path / "b.csv", [("h1", "2019-05-01 22:00:00", "REM")] * 2
        )
        with pytest.raises(ValueError, match="repeats"):
            read_mat(twice)
        offset = write_mat(
            tmp_path / "c.csv", [("h1", "2019-05-01T22:00:00+01:00", "REM")]
        )
        with pytest.raises(ValueError):
            read_mat(offset)

    def test_other_columns_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "d.csv"
        path.write_text("patient_id,date,state\nh1,2019-05-01 22:00:00,REM\n")
        with pytest.raises(ValueError, match="columns"):
            read_mat(path)

    def test_the_homes_staged_mostly_awake_are_found(self) -> None:
        awake = a_mat("a", {day(n): (2.0, 8.0) for n in range(5)})
        asleep = a_mat("b", {day(n): (7.0, 8.0) for n in range(5)})
        assert staged_awake_homes({"a": awake, "b": asleep}) == ("a",)
        assert asleep.staged_asleep_share() == pytest.approx(7.0 / 8.0)


class TestMatching:
    def test_a_matched_day_is_usable_observed_and_not_at_either_end(self) -> None:
        mat = a_mat("h", {day(n): (7.0 + n, 8.0 + n) for n in range(0, 8)})
        days = [a_day(n, 10.0 + n) for n in range(1, 8)]
        days[2] = a_day(3, 12.0, usable=False)
        pairs = pairs_of("h", days, mat)
        # Day 1 is the home's first day and day 7 its last; day 3 is not usable.
        assert pairs.days == (day(2), day(4), day(5), day(6))
        assert list(pairs.pipeline) == [12.0, 14.0, 15.0, 16.0]
        assert list(pairs.references[SLEEP]) == [9.0, 11.0, 12.0, 13.0]
        assert list(pairs.references[IN_BED]) == [10.0, 12.0, 13.0, 14.0]

    def test_a_silenced_day_is_left_out_when_asked(self) -> None:
        mat = a_mat("h", {day(n): (7.0, 8.0) for n in range(0, 8)})
        off = [a_day(n, 9.0) for n in range(0, 8)]
        off[3] = a_day(3, 23.8, events=0)
        rule = [a_day(n, 9.0) for n in range(0, 8)]
        rule[3] = a_day(3, 0.0, events=0, silent=1.0, usable=False)
        rule[5] = a_day(5, 6.0, silent=0.3, usable=False)
        silenced = silenced_days(off, rule)
        assert silenced == {day(3), day(5)}
        kept = [d.day for d in matched_days(off, mat)]
        left = [d.day for d in matched_days(off, mat, silenced)]
        assert day(3) in kept and day(5) in kept
        assert day(3) not in left and day(5) not in left

    def test_a_home_without_a_mat_has_no_matched_day(self) -> None:
        assert pairs_of("h", [a_day(1, 9.0)], None).size == 0

    def test_pipeline_days_read_the_run(self) -> None:
        record = a_record(
            3,
            7.25,
            kinds={"sleeping_hours": "abrupt_change", "away_hours": "ordinary"},
            deviations={"sleeping_hours": -3.4, "away_hours": 0.2},
            alerts=(("sleeping_hours", "attention"), ("away_hours", "information")),
            events=42,
        )
        (only,) = pipeline_days(a_run("h", [record]))
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
        n = PROTOCOL.min_matched_days
        values = np.arange(n, dtype=float)
        pairs = Pairs(
            "h",
            tuple(day(k) for k in range(n)),
            values + 2.0,
            {SLEEP: values, IN_BED: values + 1.0},
        )
        row = home_statistics(pairs, PROTOCOL)
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
        assert not home_statistics(fewer, PROTOCOL)["included"]

    def test_a_constant_series_counts_as_no_correlation(self) -> None:
        n = PROTOCOL.min_matched_days
        pairs = Pairs(
            "h",
            tuple(day(k) for k in range(n)),
            np.full(n, 24.0),
            {SLEEP: np.arange(n, dtype=float), IN_BED: np.arange(n, dtype=float)},
        )
        row = home_statistics(pairs, PROTOCOL)
        assert row[SLEEP]["constant"] and row[SLEEP]["spearman"] == 0.0
        found = agreement({"h": row}, PROTOCOL)
        assert found[SLEEP]["homes_with_constant_values"] == ["h"]
        assert found[SLEEP]["spearman"]["estimate"] == 0.0

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
        rng = np.random.default_rng(5)
        table = {}
        for k in range(4):
            n = PROTOCOL.min_matched_days + (k if k < 3 else -1)
            x = rng.standard_normal(n)
            pairs = Pairs(
                f"h{k}",
                tuple(day(j) for j in range(n)),
                x + 3.0,
                {SLEEP: x + rng.standard_normal(n), IN_BED: x},
            )
            table[f"h{k}"] = home_statistics(pairs, PROTOCOL)
        found = agreement(table, PROTOCOL)
        assert found["homes"] == ["h0", "h1", "h2"]
        expected = np.mean([table[h][SLEEP]["spearman"] for h in ("h0", "h1", "h2")])
        assert found[SLEEP]["spearman"]["estimate"] == pytest.approx(expected)
        assert found[IN_BED]["mean_difference"]["estimate"] == pytest.approx(3.0)
        assert agreement(table, PROTOCOL, leave_out=["h1"])["homes"] == ["h0", "h2"]


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
        values = {day(n): (7.0 + 0.1 * (n % 5), 9.0) for n in range(0, 40)}
        values[day(30)] = (12.0, 13.0)
        values[day(31)] = (3.0, 4.0)
        mat = a_mat("h", values)
        verdicts = mat_verdicts(mat)
        assert verdicts[day(30)][1] > PROTOCOL.alert_threshold
        days = [a_day(n, 8.0, deviation=0.1) for n in range(0, 40)]
        days[30] = a_day(30, 14.0, deviation=4.0)
        days[31] = a_day(31, 3.0, deviation=-5.0)
        pairs = {"h": pairs_of("h", days, mat)}
        found = deviations({"h": days}, pairs, {"h": verdicts}, PROTOCOL)
        assert found["pipeline_past_threshold"] == 2
        same = 1 + int(verdicts[day(31)][1] < 0)
        also = int(verdicts[day(30)][1] >= 3) + int(verdicts[day(31)][1] <= -3)
        assert found["same_sign"] == same
        assert found["also_past_threshold"] == also

    def test_only_a_step_alert_is_judged_and_by_both_rules(self) -> None:
        mat = a_mat("h", {day(n): (6.0 + n, 9.0) for n in range(0, 8)})
        days = [a_day(n, 8.0) for n in range(0, 8)]
        # Matched days are 1 to 6, mat sleep 7 to 12, median 9.5.
        days[2] = a_day(2, 4.0, kind="abrupt_change", deviation=-3.5, alerts=1)
        days[5] = a_day(5, 12.0, kind="persistent_change", deviation=3.2, alerts=2)
        # A drift alert with its deviation on the other side: counted, not judged.
        days[4] = a_day(4, 9.0, kind="gradual_drift", deviation=0.4, alerts=1)
        # The home's last day is not matched: counted, not judged.
        days[7] = a_day(7, 12.0, kind="abrupt_change", deviation=3.5, alerts=1)
        pairs = {"h": pairs_of("h", days, mat)}
        verdicts = {
            "h": {day(2): ("ordinary", -1.0), day(5): ("insufficient_data", 0.0)}
        }
        found = alerts_beside_the_mat({"h": days}, {"h": mat}, pairs, verdicts)
        assert found["alerts"] == 5 and found["judged"] == 3
        assert found["by_kind"] == {
            "abrupt_change": 2,
            "persistent_change": 2,
            "gradual_drift": 1,
        }
        assert found["median"] == {"same_side": 3, "opposite": 0, "at_the_median": 0}
        assert found["mat_baseline"] == {
            "same_sign": 1,
            "opposite": 0,
            "zero": 0,
            "no_verdict": 2,
        }
        assert found["homes"]["h"]["judged"] == 3

    def test_silent_days_are_days_without_an_activity_record(self) -> None:
        mat = a_mat("h", {day(n): (7.0, 8.0 + n) for n in range(0, 5)})
        days = [a_day(n, 8.0) for n in range(0, 6)]
        days[2] = a_day(2, 23.5, events=0)
        days[5] = a_day(5, 24.0, events=0, usable=False)
        found = silent_days({"h": days}, {"h": mat})
        assert found["silent_days"] == 2 and found["usable"] == 1
        assert found["mat_observed"] == 1
        assert found["median_pipeline_hours"] == 23.5
        assert found["median_mat_in_bed_hours"] == 10.0
        assert found["homes"]["h"] == {
            "silent_days": 2,
            "usable": 1,
            "mat_observed": 1,
        }

    def test_the_clocks_agree_where_activity_and_bed_are_most_opposed(self) -> None:
        protocol = dataclasses.replace(PROTOCOL, min_matched_days=3)
        run = a_run("h", [a_record(n, 8.0) for n in range(0, 8)])
        # In bed from 23:00 to 07:00 by the activity clock, recorded an hour
        # early, as a mat clock an hour behind would record it.
        hourly = {}
        for n in range(0, 9):
            for hour in (22, 23):
                hourly[(day(n), hour)] = 60
            for hour in range(0, 6):
                hourly[(day(n), hour)] = 60
        mat = a_mat("h", {day(n): (7.0, 8.0) for n in range(0, 9)}, hourly)
        pairs = {"h": pairs_of("h", pipeline_days(run), mat)}
        found = hourly_alignment({"h": run}, {"h": mat}, pairs, ["h"], protocol)
        assert found["most_negative"] == "+1"
        assert (
            found["lags"]["+1"]["spearman"]["estimate"]
            < found["lags"]["+0"]["spearman"]["estimate"]
        )


class TestTheChecks:
    def test_a_run_that_is_not_the_published_one_reports_nothing(self) -> None:
        run = a_run("h", [a_record(0, 8.0, alerts=(("sleeping_hours", "attention"),))])
        good = _per_household([run], BaselineConfig().deviation_threshold)
        assert check_published({"h": run}, good)["reproduced"]
        bad = {"h": {**good["h"], "events": good["h"]["events"] + 1}}
        with pytest.raises(ValueError, match="nothing is reported"):
            check_published({"h": run}, bad)
        with pytest.raises(ValueError, match="nothing is reported"):
            check_published({"h": run}, {})

    def test_a_run_with_the_rule_that_is_not_the_silent_home_one_reports_nothing(
        self,
    ) -> None:
        off = a_run("h", [a_record(n, 8.0) for n in range(3)])
        on = a_run(
            "h",
            [
                a_record(0, 8.0),
                a_record(1, 2.0, silent=0.5, usable=False),
                a_record(2, 8.0),
            ],
            silences=1,
        )
        record = {
            "conditions": {
                "h12": {
                    "days_refused_because_of_the_rule": {"per_home": {"h": 1}},
                    "silence_alerts": {"per_home": {"h": 1}},
                }
            }
        }
        assert check_rule({"h": on}, {"h": off}, record, "h12")["reproduced"]
        record["conditions"]["h12"]["silence_alerts"]["per_home"]["h"] = 2
        with pytest.raises(ValueError, match="nothing is reported"):
            check_rule({"h": on}, {"h": off}, record, "h12")


def synthetic_cohort() -> tuple[dict, dict, dict, dict]:
    """Twenty mat homes of 40 days, three of them staged mostly awake."""
    rng = np.random.default_rng(11)
    homes = [*STAGED_AWAKE_HOMES, *(f"m{k:02d}" for k in range(17))]
    runs_off, runs_on, mats = {}, {}, {}
    for index, home in enumerate(homes):
        sleep = 7.0 + rng.standard_normal(40)
        share = 0.4 if home in STAGED_AWAKE_HOMES else 0.9
        values = {day(n): (share * (sleep[n] + 1.0), sleep[n] + 1.0) for n in range(40)}
        mats[home] = a_mat(home, values)
        records = []
        for n in range(40):
            silent = n == 20 and index % 4 == 0
            records.append(
                a_record(
                    n,
                    23.8 if silent else sleep[n] + 2.0 + 0.5 * rng.standard_normal(),
                    events=0 if silent else 60,
                    deviations={"sleeping_hours": float(rng.standard_normal())},
                )
            )
        runs_off[home] = a_run(home, records)
        runs_on[home] = a_run(
            home,
            [
                (
                    dataclasses.replace(r, usable=False, silent=0.4)
                    if not r.events or (index == 1 and n == 25)
                    else r
                )
                for n, r in enumerate(records)
            ],
        )
    runs = {"off": runs_off, "rule_12h": runs_on}
    published = _per_household(
        list(runs_off.values()), BaselineConfig().deviation_threshold
    )
    silent_home = {
        "conditions": {
            "h12": {
                "days_refused_because_of_the_rule": {
                    "per_home": {
                        h: sum(1 for d in r.days if d.silent > 0)
                        for h, r in runs_on.items()
                    }
                },
                "silence_alerts": {"per_home": dict.fromkeys(runs_on, 0)},
            }
        }
    }
    return runs, mats, published, silent_home


class TestTheWhole:
    def test_score_gives_every_estimand_and_reads_the_criteria(self) -> None:
        runs, mats, _, _ = synthetic_cohort()
        results = score(runs, mats, {-1.0: mats, 1.0: mats}, PROTOCOL)
        assert set(results["analyses"]) == {PRIMARY, WITH_SILENT_DAYS, RULE}
        primary = results["analyses"][PRIMARY]
        assert len(primary["homes"]) == 20
        assert primary[SLEEP]["mean_difference"]["estimate"] > 1.0
        c1 = results["criteria"]["C1_the_pipeline_follows_the_mat"]
        assert c1["reading"] in ("follows", "does_not_follow", "inconclusive")
        with_silent = results["analyses"][WITH_SILENT_DAYS]
        assert (
            with_silent[SLEEP]["mean_difference"]["estimate"]
            > primary[SLEEP]["mean_difference"]["estimate"]
        )
        assert results["E12_staged_asleep"]["left_out"] == list(STAGED_AWAKE_HOMES)
        assert len(results["E12_staged_asleep"]["homes"]) == 17
        assert set(results["E10_alignment"]) == {"-1", "+1"}
        assert set(results["E11_clocks"]["lags"]) == {
            "-3",
            "-2",
            "-1",
            "+0",
            "+1",
            "+2",
            "+3",
        }
        assert results["homes"]["m00"]["silent_matched_days"] == 0
        assert results["homes"][STAGED_AWAKE_HOMES[0]]["silent_matched_days"] == 1
        partly = results["homes"][STAGED_AWAKE_HOMES[1]]
        assert partly["partly_silent_matched_days"] == 1
        assert partly[PRIMARY]["matched_days"] == 37
        assert partly[WITH_SILENT_DAYS]["matched_days"] == 38
        # The synthetic mats hold no hourly minutes, so no lag is found.
        assert results["homes"]["m00"]["most_negative_lag"] is None

    def test_score_refuses_a_mat_that_stages_other_homes_awake(self) -> None:
        runs, mats, _, _ = synthetic_cohort()
        mats = dict(mats)
        mats["m00"] = a_mat("m00", {day(n): (1.0, 8.0) for n in range(40)})
        with pytest.raises(ValueError, match="staged mostly awake"):
            score(runs, mats, {}, PROTOCOL)

    def test_the_record_and_its_page(self) -> None:
        runs, mats, published, silent_home = synthetic_cohort()
        record = tihm_record(
            runs,
            mats,
            {-1.0: mats, 1.0: mats},
            PROTOCOL,
            tihm_protocol(),
            published,
            silent_home,
            protocol_sha256="0" * 64,
            code_changed={"sources": [], "distributions": [], "defaults": []},
        )
        payload = json.loads(record.to_json())
        assert payload["results"]["check"]["off"]["reproduced"]
        assert payload["results"]["check"]["rule_12h"]["reproduced"]
        assert payload["household_metrics"][PRIMARY]["m00"]["matched_days"] == 38
        labels = [entry["label"] for entry in payload["intervals"]]
        assert "off: sleep: spearman" in labels
        simulated = {
            "results": score_simulated(
                [
                    SimulatedHome(
                        seed=seed,
                        days=tuple(day(k) for k in range(20)),
                        pipeline=tuple(float(k % 7) + 1.0 for k in range(20)),
                        truth=tuple(float(k % 7) for k in range(20)),
                        usable_days=20,
                        closed_days=22,
                    )
                    for seed in (1, 2, 3)
                ],
                PROTOCOL,
            )
        }
        page = render_page(payload, simulated)
        assert page.startswith("# The pipeline's hours of sleep beside a sleep mat")
        assert "| C1 | The pipeline follows the mat | **" in page
        assert "the 20 mat homes give every count" in page
        assert "## Between the freeze and the run" not in page
        assert "Surrey and Borders Partnership NHS Foundation Trust" in page


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

    def test_the_night_the_clocks_change_has_the_hours_it_had(self) -> None:
        zone = ZoneInfo("Europe/Lisbon")
        start = datetime(2024, 3, 30, 23, 0, tzinfo=zone)
        end = datetime(2024, 3, 31, 7, 0, tzinfo=zone)
        episodes = (Episode(start, end, BehaviouralState.SLEEPING, "bedroom"),)
        truth = GroundTruth(episodes=episodes, visitors=(), tz=zone)
        hours = true_sleep_hours(truth, zone)
        assert hours == {date(2024, 3, 30): 1.0, date(2024, 3, 31): 6.0}

    def test_the_reference_scores_the_truth(self) -> None:
        n = PROTOCOL.min_matched_days
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
        found = score_simulated(homes, PROTOCOL)
        assert found["included"] == 3
        assert found[SLEEP]["spearman"]["estimate"] == pytest.approx(1.0)
        assert found[SLEEP]["mean_difference"]["estimate"] == pytest.approx(1.5)

    def test_a_short_simulated_home_runs_and_leaves_out_its_ends(self) -> None:
        small = dataclasses.replace(PROTOCOL, sim_days=4)
        home = run_simulated_home((999, small))
        first = small.sim_start()
        assert home.closed_days >= 3
        assert all(first < d < first + timedelta(days=3) for d in home.days)
        assert len(home.days) == len(home.pipeline) == len(home.truth)
        assert all(0.0 <= value <= 24.0 for value in home.truth)


class TestTheFigures:
    def test_every_figure_is_drawn_from_the_record(self, tmp_path: Path) -> None:
        from sensor_modeling.datasets.sleep_mat_figures import FIGURES, draw_figures

        runs, mats, published, silent_home = synthetic_cohort()
        record = tihm_record(
            runs,
            mats,
            {-1.0: mats, 1.0: mats},
            PROTOCOL,
            tihm_protocol(),
            published,
            silent_home,
            protocol_sha256="0" * 64,
        )
        paths = draw_figures(json.loads(record.to_json()), tmp_path)
        assert set(paths) == set(FIGURES)
        assert all(
            path.read_text(encoding="utf-8").startswith("<?xml")
            for path in paths.values()
        )
