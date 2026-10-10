"""The sleep-gap description: its plan, its minute-by-minute arithmetic and pages."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from sensor_modeling.datasets.sleep_gap_experiment import (
    MINUTES,
    Activations,
    BeliefRun,
    DayBeliefs,
    _summaries,
    belief_run,
    day_beliefs,
    distance_bin,
    home_gap,
    mat_timeline,
    period_of,
    since_bin,
    within_home,
)
from sensor_modeling.datasets.sleep_gap_figures import draw_figures
from sensor_modeling.datasets.sleep_gap_plan import (
    AN_HOUR_LATER,
    AS_RECORDED,
    ASLEEP,
    AWAKE_ON_THE_MAT,
    CLOCKS,
    MAT_CLASSES,
    NOTHING_YET,
    OFF_THE_MAT,
    PINNED_RECORDS,
    check_frozen_plan,
    declared_plan,
    file_sha256,
    write_plan,
)
from sensor_modeling.datasets.sleep_gap_summary import (
    PLAN_PAGE,
    RESULTS_PAGE,
    render_page,
    render_plan,
)
from sensor_modeling.online.pipeline import BehaviouralSensingPipeline, PipelineConfig
from sensor_modeling.simulation.household import HouseholdConfig, simulate

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts" / "sleep_gap" / "sleep_gap_plan.json"
RECORD = ROOT / "artifacts" / "sleep_gap" / "sleep-gap.json"
ZONE = ZoneInfo("Europe/London")
STATES = (
    "away",
    "home_active",
    "home_inactive",
    "sleeping",
    "bed_awake",
    "bathroom_activity",
    "kitchen_activity",
)


class TestPlan:
    def test_the_declaration_is_stable(self) -> None:
        plan = declared_plan()
        assert plan.sha256() == declared_plan().sha256()
        assert json.loads(json.dumps(plan.to_dict())) == plan.to_dict()

    def test_a_written_plan_is_checked(self, tmp_path) -> None:
        plan = declared_plan()
        path = tmp_path / "plan.json"
        write_plan(plan, path)
        check_frozen_plan(plan, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "changed"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="differs"):
            check_frozen_plan(plan, path)

    def test_the_pinned_records_are_the_files(self) -> None:
        for name, digest in PINNED_RECORDS.items():
            assert file_sha256(ROOT / name) == digest

    def test_the_plan_page_renders(self) -> None:
        payload = {**declared_plan().to_dict(), "plan_sha256": "0" * 64}
        page = render_plan(payload)
        assert "This page reports no result" in page
        assert "no mat record" in page


class TestMinutes:
    def test_the_mat_classes_and_the_shift(self) -> None:
        day = date(2019, 5, 2)
        minutes = {
            datetime(2019, 5, 2, 1, 0): "DEEP",
            datetime(2019, 5, 2, 1, 1): "AWAKE",
            datetime(2019, 5, 1, 23, 30): "LIGHT",
        }
        codes, distance = mat_timeline(minutes, 0.0).day(day)
        assert codes[60] == MAT_CLASSES.index(ASLEEP)
        assert codes[61] == MAT_CLASSES.index(AWAKE_ON_THE_MAT)
        assert codes[0] == MAT_CLASSES.index(OFF_THE_MAT)
        assert distance[60] == 0.0 and distance[0] == 30.0 and distance[100] == 39.0
        later, _ = mat_timeline(minutes, 1.0).day(day)
        # Counted an hour after its stamp, 23:30 the day before falls at 00:30.
        assert later[30] == MAT_CLASSES.index(ASLEEP)
        assert later[120] == MAT_CLASSES.index(ASLEEP)
        assert later[60] == MAT_CLASSES.index(OFF_THE_MAT)
        empty, far = mat_timeline({}, 0.0).day(day)
        assert (empty == MAT_CLASSES.index(OFF_THE_MAT)).all() and np.isinf(far).all()

    def test_the_last_activation_and_the_bins(self) -> None:
        at = datetime(2019, 5, 2, 8, 0, 10, tzinfo=ZONE)
        activations = Activations.of(
            [(at, "Kitchen"), (at + timedelta(hours=2), "Lounge")]
        )
        moments = np.array(
            [
                (at - timedelta(minutes=1)).timestamp(),
                (at + timedelta(minutes=20)).timestamp(),
                (at + timedelta(hours=2)).timestamp(),
            ]
        )
        sensors, since = activations.last(moments)
        assert sensors == [NOTHING_YET, "Kitchen", "Lounge"]
        assert np.isinf(since[0]) and since[1] == pytest.approx(20.0)
        assert since[2] == 0.0
        assert since_bin(5) == "under_10_minutes"
        assert since_bin(60) == "1_to_3_hours"
        assert since_bin(500) == "3_hours_or_more"
        assert since_bin(float("inf")) == NOTHING_YET
        assert distance_bin(0) == "under_30_minutes"
        assert distance_bin(400) == "6_hours_or_more"
        assert period_of(7 * 60) == "day" and period_of(22 * 60) == "night"

    def test_the_minutes_sum_to_the_pipelines_days(self) -> None:
        result = simulate(HouseholdConfig(days=3, seed=11))
        events = list(result.observations)
        zone = result.config.tz
        pipeline = BehaviouralSensingPipeline(
            result.registry, config=PipelineConfig(tz=zone, step=timedelta(minutes=10))
        )
        steps = pipeline.run(events)
        steps.extend(pipeline.close(events[-1].timestamp))
        run = belief_run("sim", steps, [(o.timestamp, o.sensor_id) for o in events])
        spread = day_beliefs(run, zone, timedelta(minutes=30))
        checked = 0
        for day, hours in run.day_hours.items():
            if day not in spread:
                continue
            totals = spread[day].belief.sum(axis=0) / 3600.0
            for k, state in enumerate(run.states):
                assert totals[k] == pytest.approx(hours[state], abs=1e-9)
            assert spread[day].watched.max() <= 60.0 + 1e-9
            watched = ~np.isnan(spread[day].moment)
            assert watched.sum() > 0.9 * MINUTES
            checked += 1
        assert checked >= 2


def _run(day: date, days: int = 1) -> tuple[BeliefRun, dict[date, DayBeliefs]]:
    """A home asleep from 00:00 to 08:00 by belief, active otherwise."""
    sleep = STATES.index("sleeping")
    active = STATES.index("home_active")
    belief = np.zeros((MINUTES, len(STATES)))
    belief[: 8 * 60, sleep] = 60.0
    belief[8 * 60 :, active] = 60.0
    moment = 60.0 * np.arange(MINUTES) + 60.0
    spread = {
        day + timedelta(days=k): DayBeliefs(belief, np.full(MINUTES, 60.0), moment)
        for k in range(days)
    }
    start = datetime.combine(day, datetime.min.time(), tzinfo=ZONE)
    run = BeliefRun(
        home="h",
        states=STATES,
        steps=(),
        day_hours={},
        observations=((start - timedelta(hours=1), "Bedroom"),),
    )
    return run, spread


class TestHomeGap:
    def test_the_composition_adds_up(self) -> None:
        day = date(2019, 5, 2)
        run, spread = _run(day)
        # The mat: asleep from 01:00 to 07:00, awake in bed 00:00 to 01:00.
        base = datetime(2019, 5, 2)
        mat = {base + timedelta(minutes=k): "LIGHT" for k in range(60, 7 * 60)}
        mat |= {base + timedelta(minutes=k): "AWAKE" for k in range(60)}
        gap = home_gap(
            run,
            spread,
            [day],
            mat_timeline(mat, 0.0),
            {day: 5},
            Activations.of(run.observations),
            ZONE,
            declared_plan(),
        )
        means = gap.means
        assert means["pipeline"] == pytest.approx(8.0)
        assert means["pipeline_asleep"] == pytest.approx(6.0)
        assert means["pipeline_awake_on_the_mat"] == pytest.approx(1.0)
        assert means["pipeline_off_the_mat"] == pytest.approx(1.0)
        assert means["mat_sleep"] == pytest.approx(6.0)
        assert means["mat_in_bed"] == pytest.approx(7.0)
        assert means["mat_sleep_not_counted"] == pytest.approx(0.0)
        assert means["mat_in_bed_not_counted"] == pytest.approx(0.0)
        assert means["difference"] == pytest.approx(
            means["pipeline_awake_on_the_mat"]
            + means["pipeline_off_the_mat"]
            - means["mat_sleep_not_counted"]
        )
        assert means["difference_in_bed"] == pytest.approx(
            means["pipeline_off_the_mat"] - means["mat_in_bed_not_counted"]
        )
        assert gap.hourly["pipeline_off_the_mat"][7] == pytest.approx(1.0)
        assert gap.beliefs[ASLEEP][STATES.index("sleeping")] == pytest.approx(1.0)
        cells = gap.cells
        assert cells["sensor"]["Bedroom"]["pipeline_sleep_hours"] == pytest.approx(
            8.0 - 7.0
        )
        assert cells["period"]["day"]["pipeline_sleep_hours"] == pytest.approx(1.0)
        assert cells["distance"]["under_30_minutes"]["pipeline_sleep_hours"] == (
            pytest.approx(29 / 60)
        )
        assert gap.daily["quiet_hours"] == [pytest.approx(24.0)]
        assert gap.short_nights == 0

    def test_correlations_references_and_shares(self) -> None:
        day = date(2019, 5, 2)
        run, spread = _run(day, days=6)
        days = sorted(spread)
        base = datetime(2019, 5, 2)
        mat = {
            base + timedelta(days=d, minutes=m): "LIGHT"
            for d in range(6)
            for m in range(60 + 20 * d, 7 * 60)
        }
        gap = home_gap(
            run,
            spread,
            days,
            mat_timeline(mat, 0.0),
            {},
            Activations.of(run.observations),
            ZONE,
            declared_plan(),
        )
        stats = within_home(gap)
        # The pipeline's hours never change, so they count as constant.
        assert stats["constant"]["mat_sleep~pipeline"] is True
        assert stats["correlations"]["mat_sleep~pipeline"] == 0.0
        # Sleep counted on the mat grows with the mat's own hours, tracking or
        # not; the swapped-mask reference says so.
        assert stats["correlations"]["mat_sleep~pipeline_on_the_mat"] == pytest.approx(
            1.0
        )
        assert stats["references"]["mat_sleep~pipeline_on_the_mat"] == pytest.approx(
            1.0
        )
        assert stats["shares"]["pipeline_on_the_mat"] is None


def _record() -> dict:
    day = date(2019, 5, 2)
    run, spread = _run(day, days=4)
    days = sorted(spread)
    base = datetime(2019, 5, 2)
    plan = declared_plan()
    gaps = {}
    for k, home in enumerate(("a", "b", "c")):
        mat = {
            base + timedelta(days=d, minutes=m): "LIGHT"
            for d in range(4)
            for m in range(60 + 10 * k + 5 * d, 7 * 60)
        }
        gaps[home] = home_gap(
            run,
            spread,
            days,
            mat_timeline(mat, 0.0),
            {d: 9 + k for k, d in enumerate(days)},
            Activations.of(run.observations),
            ZONE,
            plan,
        )
    clocks = {}
    for clock in CLOCKS:
        summary = _summaries(gaps, ["a", "b", "c"], plan)
        clocks[clock] = {
            **summary,
            "D6": {
                "left_out": [],
                **{k: v for k, v in summary.items() if k not in ("per_home", "homes")},
            },
        }
    return {
        "experiment": "sleep-gap",
        "recorded_at": datetime(2026, 10, 10, tzinfo=timezone.utc).isoformat(),
        "environment": {"git_commit": "abcdef0123"},
        "configuration": {
            "plan_sha256": "0" * 64,
            "code_changed_since_the_freeze": {},
            "short_night_hours": 4.0,
        },
        "notes": ["A note."],
        "results": {
            "check": {
                "off": {"homes": 17},
                "pairs": {"matched_days": 12},
                "beliefs": {"largest_difference_hours": 1e-12},
                "difference": {"hours": 1.5},
            },
            "states": list(STATES),
            "clocks": clocks,
        },
    }


class TestPages:
    def test_a_record_renders_and_draws(self, tmp_path) -> None:
        record = _record()
        page = render_page(record)
        assert "## The hours of sleep, by what the mat says (D1)" in page
        assert "An hour later" in page
        assert "swapped-mask reference" in page
        assert "bed is empty" not in page
        paths = draw_figures(record, tmp_path)
        assert sorted(paths) == ["composition", "context", "hourly"]
        assert all(path.exists() for path in paths.values())


@pytest.mark.skipif(not PLAN.exists(), reason="the plan is not frozen yet")
class TestFrozenPlan:
    def test_the_frozen_plan_is_the_declared_one(self) -> None:
        check_frozen_plan(declared_plan(), PLAN)

    def test_the_plan_page_is_its_rendering(self) -> None:
        payload = json.loads(PLAN.read_text(encoding="utf-8"))
        page = (ROOT / "docs" / PLAN_PAGE).read_text(encoding="utf-8")
        assert page == render_plan(payload)


@pytest.mark.skipif(not RECORD.exists(), reason="the description has not been run")
class TestPublishedRecord:
    def record(self) -> dict:
        return json.loads(RECORD.read_text(encoding="utf-8"))

    def test_every_check_held(self) -> None:
        check = self.record()["results"]["check"]
        for name in (
            "off",
            "rule_12h",
            "pairs",
            "clock",
            "beliefs",
            "mat",
            "difference",
        ):
            assert check[name]["reproduced"] is True

    def test_both_clocks_are_reported(self) -> None:
        clocks = self.record()["results"]["clocks"]
        assert set(clocks) == {AS_RECORDED, AN_HOUR_LATER}

    def test_the_results_page_is_its_rendering(self) -> None:
        page = (ROOT / "docs" / RESULTS_PAGE).read_text(encoding="utf-8")
        assert page == render_page(self.record())
