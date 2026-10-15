"""The matched-sensor study's runs, statistics and pages, on small inputs."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from sensor_modeling.datasets.matched_sensors_experiment import (
    HomeRun,
    RunRecord,
    check,
    collapse_reading,
    correlations,
    detected,
    excess,
    record_of,
    run_home,
    score,
    spearman,
    survival_reading,
    tracking_reading,
)
from sensor_modeling.datasets.matched_sensors_figures import draw_figures
from sensor_modeling.datasets.matched_sensors_planning import MOMENTS
from sensor_modeling.datasets.matched_sensors_protocol import (
    C1,
    C2,
    C3,
    MATCHED,
    SENSITIVITY,
    STANDARD,
    STATE_HOURS,
    declared_protocol,
)
from sensor_modeling.datasets.matched_sensors_summary import render_page
from sensor_modeling.datasets.threshold_calibration_protocol import CHANGE, STABLE

PROTOCOL = declared_protocol()
STUDY = PROTOCOL.study
BEGIN, END = STUDY.detection_window(CHANGE)


def _run(alerts=(), days=20, seed=0, flat=False) -> RunRecord:
    rng = np.random.default_rng(seed)
    truth = {s: tuple(rng.normal(5, 1, days)) for s in STATE_HOURS}
    hours = {
        s: (
            tuple(0.0 for _ in range(days))
            if flat
            else tuple(np.array(truth[s]) + rng.normal(0, 0.3, days))
        )
        for s in STATE_HOURS
    }
    states = tuple(STATE_HOURS)
    return RunRecord(
        behavioural=tuple(alerts),
        closes=(BEGIN,),
        usable=days,
        days=tuple(date(2024, 3, 5) + timedelta(days=k) for k in range(days)),
        hours=hours,
        truth=truth,
        moments=dict.fromkeys(MOMENTS, 1.0),
        verdicts={"change_verdicts": 3, "raised": 1, "not_attributable_enough": 1},
        attribution=(0.9, 0.8),
        beliefs={
            "kitchen": (2, tuple(2.0 / len(states) for _ in states)),
            "bathroom": (0, tuple(0.0 for _ in states)),
        },
        states=states,
    )


def _home(
    seed: int, standard_hit: bool, matched_hit: bool, false: bool = False
) -> HomeRun:
    hit = (
        (BEGIN + timedelta(hours=1), "sleeping_hours", "gradual_drift", "decrease", 60),
    )
    stable = hit if false else ()
    return HomeRun(
        seed=seed,
        runs={
            STANDARD: {
                STABLE: _run(stable, seed=seed),
                CHANGE: _run(hit if standard_hit else (), seed=seed),
            },
            MATCHED: {
                STABLE: _run(stable, seed=seed + 1),
                CHANGE: _run(hit if matched_hit else (), seed=seed + 1),
            },
        },
    )


class TestOneRun:
    def test_alerts_are_written_as_the_published_record_writes_them(self) -> None:
        run = _run(
            [
                (BEGIN, "sleeping_hours", "gradual_drift", "decrease", 57),
                (BEGIN, "away_hours", "persistent_change", "increase", 25),
            ]
        )
        assert run.encoded() == "57sG- 25aP+"

    def test_only_alerts_about_the_feature_inside_the_window_count(self) -> None:
        run = _run(
            [
                (
                    BEGIN - timedelta(seconds=1),
                    "sleeping_hours",
                    "abrupt_change",
                    "decrease",
                    56,
                ),
                (BEGIN, "away_hours", "abrupt_change", "decrease", 57),
                (END, "sleeping_hours", "abrupt_change", "decrease", 78),
            ]
        )
        assert run.first("sleeping_hours", BEGIN, END) is None

    def test_a_constant_series_has_no_correlation(self) -> None:
        assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0
        assert spearman([0, 1e-12, 0, 2e-12], [1, 2, 3, 4]) == 0.0
        assert spearman([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


class TestTheReadings:
    @pytest.mark.parametrize(
        ("low", "high", "reading"),
        [
            (-0.05, 0.02, "survives"),
            (-0.3, -0.12, "does_not_survive"),
            (-0.2, 0.0, "inconclusive"),
        ],
    )
    def test_survival(self, low: float, high: float, reading: str) -> None:
        entry = {"estimate": (low + high) / 2, "interval": {"low": low, "high": high}}
        assert survival_reading(entry, 0.1) == reading

    @pytest.mark.parametrize(
        ("low", "high", "reading"),
        [
            (0.55, 0.7, "follows"),
            (0.1, 0.4, "does_not_follow"),
            (0.45, 0.6, "inconclusive"),
        ],
    )
    def test_tracking(self, low: float, high: float, reading: str) -> None:
        entry = {"estimate": (low + high) / 2, "interval": {"low": low, "high": high}}
        assert tracking_reading(entry, 0.5) == reading

    def test_collapse(self) -> None:
        assert collapse_reading(None, 0.01, PROTOCOL) == "not_reproduced"
        assert collapse_reading(0.1, 0.01, PROTOCOL) == "reproduced"
        assert collapse_reading(0.1, 0.2, PROTOCOL) == "not_reproduced"
        assert collapse_reading(1.5, 0.01, PROTOCOL) == "not_reproduced"


class TestTheHomes:
    def test_detections_and_excess(self) -> None:
        homes = [
            _home(1, True, False),
            _home(2, True, True, false=True),
            _home(3, False, True),
        ]
        assert list(detected(homes, STANDARD, CHANGE, PROTOCOL)) == [True, True, False]
        assert list(excess(homes, MATCHED, PROTOCOL)) == [0.0, 0.0, 1.0]

    def test_a_home_with_too_few_days_enters_no_correlation(self) -> None:
        short = _home(4, True, True)
        runs = {**short.runs, MATCHED: {**short.runs[MATCHED], STABLE: _run(days=5)}}
        rows = correlations([HomeRun(4, runs)], MATCHED, "sleeping_hours", PROTOCOL)
        assert rows == [None]

    def test_the_check_names_every_run_that_differs(self) -> None:
        homes = [_home(1, True, False)]
        published = {"1": {STABLE: "", CHANGE: "60sG-"}}
        assert check(homes, published)["reproduced"]
        assert check(homes, {"1": {STABLE: "", CHANGE: ""}})["runs_that_differ"] == [
            "1/change"
        ]


def _planning() -> dict:
    return {
        "results": {
            "tihm": {"moments": {m: {"median": 1.0, "homes": 3} for m in MOMENTS}}
        }
    }


class TestTheScore:
    def test_nothing_is_reported_when_the_check_fails(self) -> None:
        homes = [_home(1, True, False)]
        failed = {"1": {STABLE: "x", CHANGE: ""}}
        results = score(homes, PROTOCOL, _planning(), failed)
        assert set(results) == {"check"}
        record = record_of(
            homes, PROTOCOL, planning=_planning(), published=failed, protocol_sha256="x"
        )
        assert not record.household_metrics
        assert not record.intervals
        payload = json.loads(json.dumps(record.to_dict(), default=str))
        assert "nothing is reported" in render_page(payload)
        assert draw_figures(payload, Path("unused")) == {}

    def test_the_sensitivity_profile_is_described_on_its_homes(self) -> None:
        homes = [_home(k, True, k % 2 == 0) for k in range(1, 7)]
        with_it = [
            HomeRun(h.seed, {**h.runs, SENSITIVITY: h.runs[MATCHED]}) for h in homes[:3]
        ] + homes[3:]
        published = {
            str(h.seed): {a: h.runs[STANDARD][a].encoded() for a in (STABLE, CHANGE)}
            for h in homes
        }
        results = score(with_it, PROTOCOL, _planning(), published)
        assert results["E9"]["homes"] == 3

    def test_a_record_and_its_page_and_figures(self, tmp_path) -> None:
        homes = [
            _home(k, k % 3 != 0, k % 2 == 0, false=k % 5 == 0) for k in range(1, 13)
        ]
        published = {
            str(h.seed): {
                arm: h.runs[STANDARD][arm].encoded() for arm in (STABLE, CHANGE)
            }
            for h in homes
        }
        record = record_of(
            homes,
            PROTOCOL,
            planning=_planning(),
            published=published,
            protocol_sha256="x",
        )
        payload = json.loads(json.dumps(record.to_dict(), default=str))
        results = payload["results"]
        assert results["check"]["reproduced"]
        assert results["E1"]["estimate"] == pytest.approx(
            np.mean(
                excess(homes, MATCHED, PROTOCOL) - excess(homes, STANDARD, PROTOCOL)
            )
        )
        assert set(results["criteria"]) == {C1, C2, C3}
        assert results["E8"][MATCHED]["kitchen"]["steps"] == 2 * len(homes)
        assert results["E7"][STANDARD]["raised_an_alert"] == len(homes)
        page = render_page(payload)
        assert "## The criteria" in page
        paths = draw_figures(payload, tmp_path)
        assert all(path.exists() for path in paths.values())


class TestARealHome:
    def test_both_profiles_run_on_one_plan(self) -> None:
        home = run_home(999, PROTOCOL, days=16)
        standard, matched = (home.runs[p][STABLE] for p in (STANDARD, MATCHED))
        truth = {
            profile: dict(zip(run.days, run.truth["sleeping"]))
            for profile, run in ((STANDARD, standard), (MATCHED, matched))
        }
        common = set(truth[STANDARD]) & set(truth[MATCHED])
        assert common
        assert all(truth[STANDARD][d] == truth[MATCHED][d] for d in common)
        assert matched.moments["hall_per_day"] is not None
        assert standard.moments["hall_per_day"] is None
        assert home.runs[MATCHED][CHANGE].closes
