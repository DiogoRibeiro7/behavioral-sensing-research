"""Tests for the TIHM alert-burden evaluation, on synthetic files in the dataset's format.

The synthetic cohort has three homes with a daily rhythm. One has agitation
labels stamped at noon on days whose mornings are busier, and another reports
nothing for four days, which the pipeline reads as sleep. The tests check the
protocol's declaration, the statistics, that a whole run is written as a valid
record that says what it could and could not do, that the descriptions made
after a run are kept apart from it, and that the published records, page and
figures are what the code gives.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.baseline.adaptive import AdaptiveBaseline, BaselineConfig
from sensor_modeling.datasets.external_figures import data_sha256
from sensor_modeling.datasets.tihm_experiment import (
    NO_LABEL,
    OTHER_SLOT,
    DayRecord,
    HouseholdRun,
    concordance,
    difference_estimate,
    expanding_z,
    flag_weights,
    history_share,
    published_folds,
    ratio_estimate,
    read_physiology,
    run_tihm,
    slot_of,
)
from sensor_modeling.datasets.tihm_figures import (
    FIGURES,
    POST_HOC_FIGURES,
    draw_figures,
    figure_data,
    post_hoc_figure_data,
)
from sensor_modeling.datasets.tihm_post_hoc import (
    POST_HOC_EXPERIMENT,
    POST_HOC_SCHEMA,
    STATUS,
    PostHocConfig,
    describe,
    estimated_exceedance,
    features,
    is_silent,
    nominal_exceedance,
    null_exceedance,
    recorded_values,
    replay,
    replay_differences,
    run_lengths,
    run_post_hoc,
    same_results,
    shuffled_values,
    stationary_values,
)
from sensor_modeling.datasets.tihm_protocol import (
    EXPERIMENT,
    PRIMARY_LABEL,
    PUBLISHED_LOGISTIC,
    RESULT_SCHEMA,
    TihmProtocol,
    check_frozen_protocol,
    declared_protocol,
    published_alerts,
    published_positives,
    published_rows,
    write_protocol,
)
from sensor_modeling.datasets.tihm_summary import render_page, render_protocol
from sensor_modeling.evaluation import load_record
from sensor_modeling.external.tihm import FILES, TIHM_MAPPING, TihmAdapter

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts" / "tihm" / "alert_burden_protocol.json"
START = date(2019, 4, 1)
DAYS = 32
HOMES = ("home1", "home2", "home3")
LABELLED_HOME = "home1"
LABEL_DAYS = (18, 19, 24, 29)
SILENT_HOME = "home3"
SILENT_DAYS = (22, 23, 24, 25)
ROOMS = ("Bedroom", "Bathroom", "Kitchen", "Lounge", "Hallway")
RATES = {
    "Bedroom": (0.3, 2.0, 0.5, 2.5),
    "Bathroom": (0.2, 2.5, 1.0, 1.5),
    "Kitchen": (0.05, 5.0, 3.0, 2.0),
    "Lounge": (0.05, 2.0, 6.0, 5.0),
    "Hallway": (0.1, 3.0, 3.0, 2.0),
}


def stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def write_world(directory: Path) -> Path:
    """Three homes, 32 days, with noon agitation labels after busy mornings.

    One home reports nothing at all for four days in a row.
    """
    rng = np.random.default_rng(7)
    directory.mkdir(parents=True, exist_ok=True)
    activity: list[tuple[str, str, str]] = []
    labels: list[tuple[str, str, str]] = []
    physiology: list[tuple[str, str, str, str, str]] = []
    for home in HOMES:
        for offset in range(DAYS):
            day = datetime.combine(START + timedelta(days=offset), datetime.min.time())
            busy = home == LABELLED_HOME and offset in LABEL_DAYS
            silent = home == SILENT_HOME and offset in SILENT_DAYS
            for hour in () if silent else range(24):
                block = hour // 6
                for room in ROOMS:
                    rate = RATES[room][block] * (3.0 if busy and block == 1 else 1.0)
                    for _ in range(rng.poisson(rate)):
                        moment = day + timedelta(
                            hours=hour, seconds=int(rng.integers(0, 3600))
                        )
                        activity.append((home, room, stamp(moment)))
            if not silent:
                activity.append((home, "Fridge Door", stamp(day + timedelta(hours=8))))
                activity.append((home, "Front Door", stamp(day + timedelta(hours=10))))
            if busy:
                labels.append(
                    (home, stamp(day + timedelta(hours=12, minutes=1)), PRIMARY_LABEL)
                )
            physiology.append(
                (
                    home,
                    stamp(day + timedelta(hours=9)),
                    "Heart rate",
                    "104.0" if offset == 20 else "70.0",
                    "beats/min",
                )
            )
    labels.append(
        ("home2", stamp(datetime(2019, 4, 21, 15, 40, 0)), "Pulse"),
    )
    for name, header, rows in (
        ("Activity.csv", ("patient_id", "location_name", "date"), activity),
        ("Labels.csv", ("patient_id", "date", "type"), labels),
        (
            "Physiology.csv",
            ("patient_id", "date", "device_type", "value", "unit"),
            physiology,
        ),
        (
            "Demographics.csv",
            ("patient_id", "age", "sex"),
            [(home, "(80, 90]", "Male") for home in HOMES],
        ),
    ):
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    return directory


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    data = write_world(tmp_path_factory.mktemp("tihm") / "Dataset")
    output = tmp_path_factory.mktemp("results")
    protocol = TihmProtocol(step_minutes=30, resamples=200)
    result = run_tihm(
        TihmAdapter(data),
        read_physiology(data / "Physiology.csv"),
        protocol,
        protocol_sha256="0" * 64,
        output_dir=output,
    )
    assert result.path is not None
    payload = load_record(result.path)
    later = run_post_hoc(
        TihmAdapter(data),
        read_physiology(data / "Physiology.csv"),
        protocol,
        payload,
        config=PostHocConfig(replicates=3, null_draws=4000),
        output_dir=output,
    )
    assert later.path is not None
    return {
        "data": data,
        "protocol": protocol,
        "path": result.path,
        "payload": payload,
        "post_hoc_path": later.path,
        "post_hoc": load_record(later.path),
    }


# ----------------------------------------------------------------------------
# The protocol
# ----------------------------------------------------------------------------
class TestProtocol:
    def test_the_frozen_file_is_the_declared_protocol(self) -> None:
        digest = check_frozen_protocol(declared_protocol(), FROZEN)
        assert len(digest) == 64

    def test_the_protocol_page_is_generated_from_the_frozen_file(self) -> None:
        payload = json.loads(FROZEN.read_text(encoding="utf-8"))
        page = ROOT / "docs" / "TIHM_ALERT_BURDEN_PROTOCOL.md"
        assert page.read_text(encoding="utf-8") == render_protocol(payload)

    def test_a_changed_protocol_is_refused(self) -> None:
        with pytest.raises(ValueError, match="differs from the one this code declares"):
            check_frozen_protocol(TihmProtocol(step_minutes=5), FROZEN)

    def test_it_declares_itself_exploratory_and_says_what_was_seen(self) -> None:
        declared = declared_protocol().to_dict()
        assert declared["status"].startswith("exploratory")
        assert len(declared["inspected_before"]) == 3
        assert declared["pipeline"]["fitted_on_tihm"] == "nothing"
        assert declared["mapping"]["sha256"] == TIHM_MAPPING.sha256()
        assert declared["mapping"]["scored_states"] == []
        assert declared["bootstrap"]["unit"] == "households"

    def test_writing_it_round_trips(self, tmp_path: Path) -> None:
        protocol = TihmProtocol(resamples=300)
        digest = write_protocol(protocol, tmp_path / "protocol.json")
        assert digest == protocol.sha256()
        payload = json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8"))
        assert payload["protocol_sha256"] == digest
        assert payload["bootstrap"]["resamples"] == 300

    def test_the_published_counts_are_consistent(self) -> None:
        assert published_rows() == (357, 314, 310, 307, 304)
        assert published_positives() == (10, 8, 13, 17, 23)
        assert published_alerts() == (86, 45, 58, 68, 84)
        assert sum(m[1][1] for m in PUBLISHED_LOGISTIC) == 55
        assert sum(published_alerts()) == 341


# ----------------------------------------------------------------------------
# The statistics
# ----------------------------------------------------------------------------
class TestStatistics:
    def test_flags_go_to_the_highest_scores(self) -> None:
        weights = flag_weights(np.array([0.1, 0.9, 0.5, 0.7]), 2)
        assert weights.tolist() == [0.0, 1.0, 0.0, 1.0]

    def test_ties_at_the_cut_share_what_remains(self) -> None:
        weights = flag_weights(np.array([1.0, 0.5, 0.5, 0.5, 0.0]), 2)
        assert weights.tolist() == pytest.approx([1.0, 1 / 3, 1 / 3, 1 / 3, 0.0])
        assert weights.sum() == pytest.approx(2.0)

    def test_flag_counts_at_the_edges(self) -> None:
        score = np.array([0.2, 0.4])
        assert flag_weights(score, 0).tolist() == [0.0, 0.0]
        assert flag_weights(score, 5).tolist() == [1.0, 1.0]
        assert flag_weights(np.array([]), 3).size == 0

    def test_concordance_counts_pairs_and_halves_ties(self) -> None:
        score = np.array([3.0, 1.0, 1.0, 0.0])
        label = np.array([True, True, False, False])
        concordant, pairs = concordance(score, label)
        assert pairs == 4.0
        assert concordant == pytest.approx(3.5)

    def test_concordance_without_both_classes_has_no_pairs(self) -> None:
        assert concordance(np.array([1.0, 2.0]), np.array([True, True])) == (0.0, 0.0)

    def test_expanding_z_reads_only_earlier_values(self) -> None:
        z = expanding_z([10, 12, 8, 10, 30], 3)
        assert z[:3].tolist() == [0.0, 0.0, 0.0]
        assert z[3] == pytest.approx(0.0)
        earlier = np.array([10, 12, 8, 10], dtype=float)
        assert z[4] == pytest.approx((30 - earlier.mean()) / earlier.std(ddof=1))
        changed = expanding_z([10, 12, 8, 10, 30, 999], 3)
        assert changed[:5].tolist() == z.tolist()

    def test_expanding_z_of_a_constant_series_is_zero(self) -> None:
        assert expanding_z([5, 5, 5, 5, 5], 2).tolist() == [0.0] * 5

    def test_history_share_excludes_the_day_itself(self) -> None:
        share = history_share([False, True, True, False], overall=0.1, strength=2.0)
        assert share.tolist() == pytest.approx(
            [0.2 / 2.0, 0.2 / 3.0, 1.2 / 4.0, 2.2 / 5.0]
        )

    def test_a_ratio_of_sums_and_its_interval(self) -> None:
        protocol = TihmProtocol(resamples=400)
        result = ratio_estimate([1, 0, 3, 2], [10, 10, 10, 10], protocol)
        assert result["estimate"] == pytest.approx(0.15)
        assert result["households"] == 4
        assert result["interval"]["low"] <= 0.15 <= result["interval"]["high"]
        assert result["interval"]["method"] == "percentile"

    def test_a_ratio_with_no_denominator_has_no_estimate(self) -> None:
        result = ratio_estimate([0, 0], [0, 0], TihmProtocol(resamples=200))
        assert result["estimate"] is None
        assert result["interval"] is None

    def test_a_difference_is_paired_on_the_same_households(self) -> None:
        protocol = TihmProtocol(resamples=400)
        same = difference_estimate([1, 2, 3], [5, 5, 5], [1, 2, 3], [5, 5, 5], protocol)
        assert same["estimate"] == pytest.approx(0.0)
        assert same["interval"]["low"] == pytest.approx(0.0)
        assert same["interval"]["high"] == pytest.approx(0.0)

    def test_published_folds_step_back_a_week_at_a_time(self) -> None:
        days = [START + timedelta(days=k) for k in range(60)]
        labelled = [k in (10, 55) for k in range(60)]
        folds = published_folds(days, labelled)
        assert len(folds) == 5
        train, test = folds[0]
        assert test.tolist() == list(range(49, 56))
        assert train.tolist() == list(range(49))
        assert folds[1][1].tolist() == list(range(42, 49))

    def test_no_label_gives_no_fold(self) -> None:
        assert published_folds([START], [False]) == []

    def test_a_label_is_on_a_slot_only_just_after_the_hour(self) -> None:
        assert slot_of(datetime(2019, 5, 1, 12, 1, 5)) == "12:00"
        assert slot_of(datetime(2019, 5, 1, 18, 2, 59)) == "18:00"
        assert slot_of(datetime(2019, 5, 1, 12, 3, 0)) == OTHER_SLOT
        assert slot_of(datetime(2019, 5, 1, 9, 0, 0)) == OTHER_SLOT

    def test_a_day_is_evaluable_only_with_a_verdict(self) -> None:
        day = DayRecord(
            day=START,
            usable=True,
            coverage=1.0,
            observed=1.0,
            abstention=0.0,
            kinds={"a": "insufficient_data", "b": "ordinary"},
            deviations={"a": 9.0, "b": -3.5},
            alerts=(),
            events=10,
            labels=frozenset(),
        )
        assert day.evaluable
        assert day.score == pytest.approx(3.5)
        assert day.deviating(3.0)
        assert not day.deviating(4.0)
        waiting = DayRecord(
            day=START,
            usable=True,
            coverage=1.0,
            observed=1.0,
            abstention=0.0,
            kinds={"a": "insufficient_data"},
            deviations={"a": 9.0},
            alerts=(),
            events=10,
            labels=frozenset(),
        )
        assert not waiting.evaluable
        assert waiting.score == 0.0
        assert not waiting.deviating(3.0)


# ----------------------------------------------------------------------------
# A whole run
# ----------------------------------------------------------------------------
class TestRun:
    def test_the_record_is_valid_and_names_its_protocol(
        self, world: dict[str, object]
    ) -> None:
        payload = world["payload"]
        assert payload["experiment"] == EXPERIMENT
        assert payload["data_source"] == "tihm"
        assert payload["results"]["result_schema"] == RESULT_SCHEMA
        assert payload["configuration"]["protocol_sha256"] == world["protocol"].sha256()
        assert payload["inference"]["mode"] == "online_filter"
        assert all(i["unit"] == "household" for i in payload["intervals"])
        assert any("Exploratory" in note for note in payload["notes"])

    def test_nothing_is_scored_against_a_state(self, world: dict[str, object]) -> None:
        contract = world["payload"]["results"]["contract"]
        assert contract["households"] == contract["converted"] == len(HOMES)
        assert contract["scorable_annotated_seconds"] == 0
        assert contract["point_annotations"] == len(LABEL_DAYS) + 1
        assert contract["refused"] == {}

    def test_days_are_counted_once_per_household(
        self, world: dict[str, object]
    ) -> None:
        results = world["payload"]["results"]
        table = results["households"]
        assert set(table) == set(HOMES)
        for home in HOMES:
            assert table[home]["monitored_days"] == DAYS
            assert table[home]["evaluable_days"] <= table[home]["usable_days"] <= DAYS
        assert results["monitoring"]["monitored_days"] == DAYS * len(HOMES)
        assert results["monitoring"]["evaluable_days"] > 0
        assert table[LABELLED_HOME]["label_days"] == len(LABEL_DAYS)
        assert table["home2"]["label_days"] == 0
        assert table["home2"]["any_label_days"] == 1

    def test_the_burden_is_alerts_over_days(self, world: dict[str, object]) -> None:
        results = world["payload"]["results"]
        burden, table = results["burden"], results["households"]
        alerts = sum(table[h]["behavioural_alerts"] for h in HOMES)
        assert burden["behavioural_alerts"] == alerts
        assert burden["B1_per_monitored_day"]["estimate"] == pytest.approx(
            alerts / (DAYS * len(HOMES))
        )
        assert sum(burden["by_feature"].values()) == alerts
        assert burden["simulator_reference"]["stable_arm"] == 0.010
        assert alerts > 0
        assert table[SILENT_HOME]["behavioural_alerts"] == alerts
        assert set(burden["by_feature"]) == {"sleeping_hours"}
        assert burden["households_with_an_alert"] == 1
        assert burden["B3_per_evaluable_day"]["estimate"] == pytest.approx(
            alerts / results["monitoring"]["evaluable_days"]
        )

    def test_the_association_is_over_evaluable_days(
        self, world: dict[str, object]
    ) -> None:
        results = world["payload"]["results"]
        flagged = results["association"]["A2_deviating_days"]
        assert flagged["days"] == results["monitoring"]["evaluable_days"]
        assert flagged["label_days"] <= len(LABEL_DAYS)
        assert flagged["flagged_days"] == results["burden"]["deviating_days"]
        assert flagged["label_days"] == len(LABEL_DAYS)
        alerted = results["association"]["A1_alert_days"]
        assert alerted["flagged_days"] == results["burden"]["alert_days"] > 0
        assert alerted["label_days_flagged"] == 0
        assert alerted["share_of_label_days_flagged"]["estimate"] == 0.0
        assert alerted["share_of_other_days_flagged"]["estimate"] == pytest.approx(
            alerted["flagged_days"] / (alerted["days"] - len(LABEL_DAYS))
        )
        scored = results["association"]["A3_deviation_score"]
        others = results["households"][LABELLED_HOME]["evaluable_days"] - len(
            LABEL_DAYS
        )
        assert scored["households_with_pairs"] == 1
        assert scored["pairs"] == len(LABEL_DAYS) * others

    def test_each_reference_gets_the_pipelines_number_of_flags(
        self, world: dict[str, object]
    ) -> None:
        results = world["payload"]["results"]
        references = results["references"]
        assert references["flags"] == results["burden"]["deviating_days"]
        assert references["flags"] >= len(LABEL_DAYS)
        assert references["label_days"] == len(LABEL_DAYS)
        for name in ("label_history", "event_count"):
            caught = references[name]["label_days_caught"]
            assert 0.0 <= caught <= references["label_days"]
        # Label days are the busy ones, so the event count ranks them first.
        events = references["event_count"]
        assert events["label_days_caught"] == len(LABEL_DAYS)
        assert events["concordance"]["estimate"] == pytest.approx(1.0)
        # The pipeline's deviating days are the silent home's, not the labelled one's.
        assert references["pipeline_deviating_days"]["label_days_caught"] == 0
        assert events["minus_pipeline"]["estimate"] == pytest.approx(1.0)

    def test_other_test_periods_are_not_compared_with_the_published_ones(
        self, world: dict[str, object]
    ) -> None:
        published = world["payload"]["results"]["published_baseline"]
        assert published["reproduced_test_periods"] is False
        assert "label_history" not in published
        assert published["published"]["label_days_caught"] == 55
        assert published["published"]["alerts"] == 341

    def test_labels_are_described_by_slot_and_by_limit(
        self, world: dict[str, object]
    ) -> None:
        labels = world["payload"]["results"]["labels"]
        primary = labels["primary"]
        assert primary["rows"] == primary["label_days"] == len(LABEL_DAYS)
        assert primary["rows_by_slot"] == {"12:00": len(LABEL_DAYS)}
        assert primary["households"] == 1
        assert primary["households_never_labelled"] == len(HOMES) - 1
        profile = labels["hourly_profile"]
        assert profile["12:00"]["days"] == len(LABEL_DAYS)
        assert profile["12:00"]["block_mean_z"]["06-12"] > 1.0
        assert abs(profile[NO_LABEL]["block_mean_z"]["06-12"]) < 0.5
        pulse = labels["stated_limits"]["Pulse"]
        assert pulse["label_days"] == 1
        assert pulse["days_meeting_the_limits"] == len(HOMES)
        assert pulse["label_days_meeting_the_limits"] == 1

    def test_a_second_run_gives_the_same_results(
        self, world: dict[str, object], tmp_path: Path
    ) -> None:
        data = world["data"]
        again = run_tihm(
            TihmAdapter(data),
            read_physiology(data / "Physiology.csv"),
            world["protocol"],
            protocol_sha256="0" * 64,
            output_dir=tmp_path,
        )
        assert again.path is not None
        assert load_record(again.path)["results"] == world["payload"]["results"]


# ----------------------------------------------------------------------------
# The page and the figures
# ----------------------------------------------------------------------------
class TestPage:
    def test_the_page_and_figures_come_from_the_record(
        self, world: dict[str, Any], tmp_path: Path
    ) -> None:
        payload = world["payload"]
        page = render_page(payload)
        assert page == render_page(load_record(world["path"]))
        assert "## After the run" not in page
        for home in HOMES:
            assert f"| `{home}` |" in page
        first = draw_figures(payload, tmp_path / "a")
        second = draw_figures(payload, tmp_path / "b")
        data = figure_data(payload)
        assert set(first) == set(FIGURES)
        for name in FIGURES:
            assert first[name].read_bytes() == second[name].read_bytes()
            assert f"data sha256 {data_sha256(data[name])}" in first[name].read_text(
                encoding="utf-8"
            )

    def test_unreproduced_test_periods_are_said_not_to_be_reported(
        self, world: dict[str, Any]
    ) -> None:
        page = render_page(world["payload"])
        assert "**Not reported.**" in page
        assert "published logistic regression, quoted" not in page

    def test_another_record_is_refused(self, world: dict[str, Any]) -> None:
        other = json.loads(json.dumps(world["payload"]))
        other["results"]["result_schema"] = "other/1"
        with pytest.raises(ValueError, match="not a tihm-alert-burden/1 record"):
            render_page(other)
        with pytest.raises(ValueError, match="not a tihm-alert-burden/1 record"):
            figure_data(other)
        with pytest.raises(ValueError, match="post-hoc/1 record"):
            render_page(world["payload"], world["payload"])
        with pytest.raises(ValueError, match="post-hoc/1 record"):
            post_hoc_figure_data(world["payload"])

    def test_the_post_hoc_part_comes_last_and_says_what_it_is(
        self, world: dict[str, Any], tmp_path: Path
    ) -> None:
        page = render_page(world["payload"], world["post_hoc"])
        assert page == render_page(
            world["payload"], load_record(world["post_hoc_path"])
        )
        head, tail = page.split("## After the run: post hoc descriptions")
        assert "## Every household" in head
        assert f"**Status: {STATUS}.**" in tail
        assert "no retuning and no added reference after scoring" in tail
        assert "### Post hoc: silent days" in tail
        assert "The date on which most households are silent is" in tail
        assert "not the protocol's A2" in tail
        assert tail.count("\n### ") == tail.count("\n### Post hoc: ") == 6
        paths = draw_figures(world["payload"], tmp_path, world["post_hoc"])
        assert set(paths) == {*FIGURES, *POST_HOC_FIGURES}
        data = post_hoc_figure_data(world["post_hoc"])
        for name in POST_HOC_FIGURES:
            svg = paths[name].read_text(encoding="utf-8")
            assert f"data sha256 {data_sha256(data[name])}" in svg
            assert "TIHM alert burden, post hoc" in svg
        assert "post hoc" not in paths[FIGURES[0]].read_text(encoding="utf-8")

    def test_a_cohort_without_a_silent_day_is_said_to_have_none(
        self, world: dict[str, Any]
    ) -> None:
        later = json.loads(json.dumps(world["post_hoc"]))
        later["results"]["calendar"]["most_silent_date"] = None
        later["results"]["calendar"]["day_after_the_most_silent_date"] = None
        page = render_page(world["payload"], later)
        assert "No household has a silent day." in page


# ----------------------------------------------------------------------------
# The descriptions made after the run
# ----------------------------------------------------------------------------
QUIET = (24, 25, 26)


def day_record(
    offset: int, values: dict[str, float], events: int = 100, label: bool = False
) -> DayRecord:
    return DayRecord(
        day=START + timedelta(days=offset),
        usable=True,
        coverage=1.0,
        observed=1.0,
        abstention=0.0,
        kinds={},
        deviations={},
        alerts=(),
        events=events,
        labels=frozenset({PRIMARY_LABEL}) if label else frozenset(),
        values=values,
    )


def household(days: list[DayRecord]) -> HouseholdRun:
    return HouseholdRun(
        household="home",
        days=tuple(days),
        system_health_alerts=0,
        data_quality_alerts=0,
        hourly={},
        label_times={},
        dispositions={},
        issues={},
        closes=(),
    )


def steady_home(
    quiet: tuple[int, ...] = QUIET, unused: int | None = None
) -> HouseholdRun:
    """Forty days of steady values; on a quiet day nothing fires and all is sleep."""
    rng = np.random.default_rng(3)
    days = []
    for offset in range(40):
        values = {name: float(rng.normal(8.0, 0.5)) for name in features()}
        if offset in quiet:
            values = {**values, "sleeping_hours": 23.8}
        if offset == unused:
            values = dict.fromkeys(features(), math.nan)
        days.append(day_record(offset, values, events=0 if offset in quiet else 100))
    return household(days)


def judged_home() -> HouseholdRun:
    """The steady home, with the verdicts the baseline gives its values."""
    run = steady_home()
    models = {name: AdaptiveBaseline(name, BaselineConfig()) for name in features()}
    days = []
    for day in run.days:
        changes = {
            name: models[name].observe(day.day, day.values[name]) for name in models
        }
        days.append(
            dataclasses.replace(
                day,
                kinds={name: change.kind.value for name, change in changes.items()},
                deviations={
                    name: float(change.deviation) for name, change in changes.items()
                },
            )
        )
    return household(days)


class TestPostHoc:
    def test_a_silent_day_has_no_event(self) -> None:
        assert is_silent(day_record(0, {}, events=0))
        assert not is_silent(day_record(0, {}, events=1))

    def test_runs_of_days_are_measured(self) -> None:
        assert run_lengths([False, True, True, False, True]) == [2, 1]
        assert run_lengths([True]) == [1]
        assert run_lengths([]) == []

    def test_a_small_reference_reaches_the_threshold_often(self) -> None:
        rng = np.random.default_rng(0)
        small = null_exceedance(4, 3.0, 40_000, rng)
        large = null_exceedance(60, 3.0, 40_000, rng)
        nominal = nominal_exceedance(3.0)
        assert nominal == pytest.approx(0.0027, abs=1e-4)
        assert small > 0.15 > 0.02 > large > nominal

    def test_an_exceedance_rate_needs_a_size_and_draws(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="must be positive"):
            null_exceedance(0, 3.0, 10, rng)
        with pytest.raises(ValueError, match="must be positive"):
            null_exceedance(4, 3.0, 0, rng)

    def test_a_configuration_is_validated(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            PostHocConfig(replicates=0)
        with pytest.raises(ValueError, match="between 0 and 1"):
            PostHocConfig(confidence=1.0)

    def test_a_verdict_needs_the_baselines_history(self) -> None:
        baseline = BaselineConfig()
        runs = [steady_home(quiet=())]
        count = replay(runs, baseline, recorded_values(runs))
        assert count.evaluable_days == 40 - baseline.min_samples
        assert count.label_days == 0

    def test_an_unused_day_is_left_out_of_the_history(self) -> None:
        baseline = BaselineConfig()
        runs = [steady_home(quiet=(), unused=30)]
        count = replay(runs, baseline, recorded_values(runs))
        assert count.evaluable_days == 40 - baseline.min_samples - 1

    def test_leaving_silent_days_out_removes_what_they_raised(self) -> None:
        baseline = BaselineConfig()
        runs = [steady_home()]
        values = recorded_values(runs)
        kept = replay(runs, baseline, values)
        left_out = replay(runs, baseline, values, skip_silent=True)
        assert kept.evaluable_days == 40 - baseline.min_samples
        assert left_out.evaluable_days == kept.evaluable_days - len(QUIET)
        assert kept.deviating_days >= left_out.deviating_days + len(QUIET)
        assert kept.change_verdicts > left_out.change_verdicts

    def test_silent_days_are_counted_and_placed_on_the_calendar(self) -> None:
        results, mcse = describe(
            [steady_home()], TihmProtocol(), PostHocConfig(replicates=2, null_draws=10)
        )
        silent, calendar = results["silent_days"], results["calendar"]
        assert silent["monitored_days"] == 40
        assert silent["silent_days"] == silent["usable"] == len(QUIET)
        assert silent["runs_by_length"] == {str(len(QUIET)): 1}
        assert silent["longest_run"] == len(QUIET)
        assert len(calendar["daily"]) == 40
        first = (START + timedelta(days=QUIET[0])).isoformat()
        following = (START + timedelta(days=QUIET[0] + 1)).isoformat()
        assert calendar["most_silent_date"]["date"] == first
        assert calendar["most_silent_date"]["silent_households"] == 1
        assert calendar["day_after_the_most_silent_date"]["date"] == following
        replays = results["replays"]
        assert (
            replays["recorded"]["change_verdicts"]
            > replays["silent_days_skipped"]["change_verdicts"]
        )
        assert replays["shuffled"]["replicates"] == 2
        assert all(value >= 0.0 for value in mcse.values())

    def test_stationary_values_keep_the_days_and_not_the_values(self) -> None:
        runs = [steady_home(unused=30)]
        values = recorded_values(runs)
        drawn = stationary_values(values, np.random.default_rng(0))
        for key, series in values.items():
            assert np.array_equal(np.isnan(series), np.isnan(drawn[key]))
            assert not np.array_equal(series, drawn[key], equal_nan=True)
        sleeping = drawn[("home", "sleeping_hours")]
        assert np.nanmax(sleeping) < 20.0

    def test_shuffled_values_keep_the_values_and_not_their_order(self) -> None:
        runs = [steady_home(unused=30)]
        values = recorded_values(runs)
        drawn = shuffled_values(values, np.random.default_rng(0))
        for key, series in values.items():
            assert np.array_equal(np.isnan(series), np.isnan(drawn[key]))
            assert np.array_equal(np.sort(series), np.sort(drawn[key]), equal_nan=True)
            assert not np.array_equal(series, drawn[key], equal_nan=True)

    def test_a_shuffled_day_keeps_its_features_together(self) -> None:
        runs = [steady_home(unused=30)]
        values = recorded_values(runs)
        drawn = shuffled_values(values, np.random.default_rng(0))
        keys = sorted(values)
        before = {tuple(values[key][k] for key in keys) for k in range(40) if k != 30}
        after = {tuple(drawn[key][k] for key in keys) for k in range(40) if k != 30}
        assert before == after
        assert all(np.isnan(drawn[key][30]) for key in keys)

    def test_a_replay_is_compared_with_the_run_feature_day_by_feature_day(
        self,
    ) -> None:
        baseline = BaselineConfig()
        run = judged_home()
        same = replay_differences([run], baseline)
        assert same == {
            "feature_days_compared": 40 * len(features()),
            "feature_days_differing": 0,
        }
        days = list(run.days)
        feature = features()[0]
        days[20] = dataclasses.replace(
            days[20],
            deviations={**days[20].deviations, feature: 99.0},
        )
        days[21] = dataclasses.replace(
            days[21], kinds={**days[21].kinds, feature: "abrupt_change"}
        )
        changed = replay_differences([household(days)], baseline)
        assert changed["feature_days_differing"] == 2

    def test_a_mean_and_sd_of_few_days_is_not_three_sigma_either(self) -> None:
        assert estimated_exceedance(4, 3.0) == pytest.approx(0.07484, abs=1e-5)
        assert estimated_exceedance(14, 3.0) == pytest.approx(0.01245, abs=1e-5)
        assert estimated_exceedance(10_000, 3.0) == pytest.approx(
            nominal_exceedance(3.0), rel=1e-2
        )
        with pytest.raises(ValueError, match="at least two values"):
            estimated_exceedance(1, 3.0)

    def test_results_are_compared_with_a_tolerance_and_by_shape(self) -> None:
        assert same_results({"a": [1.0, 2]}, {"a": [1.0 + 1e-12, 2]})
        assert not same_results({"a": [1.0, 2]}, {"a": [1.001, 2]})
        assert not same_results({"a": 1}, {"b": 1})
        assert not same_results([1, 2], [1])
        assert not same_results(True, 1)
        assert same_results({"a": None, "b": "x"}, {"a": None, "b": "x"})

    def test_the_record_says_what_it_describes(self, world: dict[str, Any]) -> None:
        later, payload = world["post_hoc"], world["payload"]
        results = later["results"]
        assert later["experiment"] == POST_HOC_EXPERIMENT
        assert results["result_schema"] == POST_HOC_SCHEMA
        assert results["status"] == STATUS
        assert later["configuration"]["status"] == STATUS
        assert results["describes"] == {
            "experiment": EXPERIMENT,
            "recorded_at": payload["recorded_at"],
            "git_commit": payload["environment"]["git_commit"],
            "protocol_sha256": payload["configuration"]["protocol_sha256"],
            "results_reproduced": True,
        }
        sizes = results["reference_size"]["gaussian_null"]
        assert len(later["mcse"]) == 4 + len(sizes)
        assert all(error >= 0.0 for error in later["mcse"].values())

    def test_the_recorded_replay_gives_the_runs_verdicts(
        self, world: dict[str, Any]
    ) -> None:
        results = world["payload"]["results"]
        recorded = world["post_hoc"]["results"]["replays"]["recorded"]
        assert recorded["reproduces_the_run"] is True
        assert recorded["feature_days_differing"] == 0
        assert recorded["feature_days_compared"] == DAYS * len(HOMES) * len(features())
        assert recorded["change_verdicts"] > 0
        assert recorded["evaluable_days"] == results["monitoring"]["evaluable_days"]
        assert recorded["deviating_days"] == results["burden"]["deviating_days"]
        verdicts = dict(results["monitoring"]["verdicts"])
        verdicts.pop("insufficient_data", None)
        assert recorded["verdicts"] == verdicts

    def test_every_description_is_about_the_same_days(
        self, world: dict[str, Any]
    ) -> None:
        results = world["payload"]["results"]
        later = world["post_hoc"]["results"]
        assert (
            later["state_hours"]["usable_days"] == results["monitoring"]["usable_days"]
        )
        assert "sleeping" in later["state_hours"]["hours"]
        silent = later["silent_days"]
        assert silent["monitored_days"] == results["monitoring"]["monitored_days"]
        assert silent["behavioural_alerts"] == results["burden"]["behavioural_alerts"]
        daily = later["calendar"]["daily"]
        assert sum(row["monitored_households"] for row in daily) == (
            results["monitoring"]["monitored_days"]
        )
        assert sum(row["behavioural_alerts"] for row in daily) == (
            results["burden"]["behavioural_alerts"]
        )
        groups = later["direction"]["event_count_z"]
        assert groups["all_evaluable_days"]["days"] == (
            results["monitoring"]["evaluable_days"]
        )
        assert sum(later["alerts_by_feature_and_verdict"].values()) == (
            results["burden"]["behavioural_alerts"]
        )

    def test_a_silent_run_is_usable_read_as_sleep_and_alerted_on(
        self, world: dict[str, Any]
    ) -> None:
        later = world["post_hoc"]["results"]
        silent, replays = later["silent_days"], later["replays"]
        assert silent["silent_days"] == len(SILENT_DAYS)
        assert silent["usable"] == silent["evaluable"] == len(SILENT_DAYS)
        assert silent["households"] == 1
        assert silent["runs_by_length"] == {str(len(SILENT_DAYS)): 1}
        hours = silent["median_sleeping_hours"]
        assert hours["silent_days"] > 20.0 > hours["other_days"]
        assert 0 < silent["alerts_on_silent_days"] <= silent["behavioural_alerts"]
        assert 0 < silent["change_verdicts_on_silent_days"] <= silent["change_verdicts"]
        assert silent["change_verdicts"] == replays["recorded"]["change_verdicts"]
        share = silent["share_silent"]
        assert share["alert_days"]["silent"] == silent["alerts_on_silent_days"]
        assert share["deviating_days"]["silent"] == len(SILENT_DAYS)
        assert share["label_days"]["silent"] == 0
        assert share["all_evaluable_days"]["silent"] == len(SILENT_DAYS)
        peak = later["calendar"]["most_silent_date"]
        assert peak["date"] == (START + timedelta(days=SILENT_DAYS[0])).isoformat()
        assert peak["silent_households"] == 1
        assert peak["monitored_households"] == len(HOMES)
        left_out = replays["silent_days_skipped"]
        assert left_out["evaluable_days"] == (
            replays["recorded"]["evaluable_days"] - len(SILENT_DAYS)
        )
        assert left_out["change_verdicts"] < replays["recorded"]["change_verdicts"]
        quiet = later["direction"]["event_count_z"]
        assert quiet["alert_days"]["mean"] < 0.0 < quiet["label_days"]["mean"]
        drivers = later["direction"]["deviating_day_drivers"]["other_days"]
        assert drivers["sleeping_hours: above"] >= len(SILENT_DAYS)

    def test_reference_sizes_come_with_their_gaussian_rate(
        self, world: dict[str, Any]
    ) -> None:
        size = world["post_hoc"]["results"]["reference_size"]
        assert set(size["features"]) == set(features())
        assert size["nominal"] == pytest.approx(nominal_exceedance(size["threshold"]))
        for entry in size["features"].values():
            assert (
                entry["days"]
                == entry["pooled"]["days"] + entry["weekday_aware"]["days"]
            )
            for count in entry["weekday_aware_by_size"]:
                assert count in size["gaussian_null"]
                assert int(count) >= size["weekday_min_samples"]
            # Thirty-two days hold at most four earlier days of one weekday, so a
            # same-weekday reference is sized by those four and not by the history.
            assert set(entry["weekday_aware_by_size"]) == {"4"}
            assert entry["weekday_aware"]["largest_reference"] == 4
            assert entry["pooled"]["smallest_reference"] == size["min_samples"]
            assert entry["pooled"]["largest_reference"] < 28
        assert set(size["gaussian_null"]) == set(size["gaussian_null_mcse"])
        assert size["mean_and_sd_estimated"]["4"] == pytest.approx(
            estimated_exceedance(4, size["threshold"])
        )

    def test_another_runs_results_are_not_described(
        self, world: dict[str, Any]
    ) -> None:
        data = world["data"]
        other = json.loads(json.dumps(world["payload"]))
        other["results"]["monitoring"]["monitored_days"] += 1
        with pytest.raises(ValueError, match="do not reproduce the published"):
            run_post_hoc(
                TihmAdapter(data),
                read_physiology(data / "Physiology.csv"),
                world["protocol"],
                other,
                config=PostHocConfig(replicates=1, null_draws=100),
            )
        other["results"]["result_schema"] = "other/1"
        with pytest.raises(ValueError, match="not a tihm-alert-burden/1 record"):
            run_post_hoc(
                TihmAdapter(data),
                read_physiology(data / "Physiology.csv"),
                world["protocol"],
                other,
            )
        with pytest.raises(ValueError, match="made under another protocol"):
            run_post_hoc(
                TihmAdapter(data),
                read_physiology(data / "Physiology.csv"),
                TihmProtocol(step_minutes=30, resamples=201),
                world["payload"],
            )


# ----------------------------------------------------------------------------
# The published record
# ----------------------------------------------------------------------------
PUBLISHED = ROOT / "artifacts" / "tihm" / "tihm-alert-burden.json"
PUBLISHED_POST_HOC = ROOT / "artifacts" / "tihm" / "tihm-alert-burden-post-hoc.json"
DOC = ROOT / "docs" / "TIHM_ALERT_BURDEN_RESULTS.md"
FIGURE_DIR = ROOT / "docs" / "figures"


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@pytest.fixture(scope="module")
def published() -> dict[str, Any]:
    return load_record(PUBLISHED)


@pytest.fixture(scope="module")
def described() -> dict[str, Any]:
    return load_record(PUBLISHED_POST_HOC)


class TestPublishedRecord:
    """The run on the 56 TIHM homes, as published."""

    def test_it_ran_the_frozen_protocol_from_a_clean_commit(
        self, published: dict[str, Any]
    ) -> None:
        declared = declared_protocol()
        frozen = check_frozen_protocol(declared, FROZEN)
        assert published["configuration"] == {
            **declared.to_dict(),
            "protocol_sha256": declared.sha256(),
            "protocol_file_sha256": frozen,
            "result_schema": RESULT_SCHEMA,
        }
        assert published["environment"]["git_dirty"] == "false"
        assert published["environment"]["git_commit"].startswith("b15126f")
        digests = {i["name"]: i["sha256"] for i in published["inputs"]}
        assert digests[FROZEN.name] == frozen
        assert {name: digests[name] for name in FILES} == dict(FILES)

    def test_nothing_was_scored_against_a_state(
        self, published: dict[str, Any]
    ) -> None:
        contract = published["results"]["contract"]
        assert contract["households"] == contract["converted"] == 56
        assert contract["refused"] == {}
        assert contract["scorable_annotated_seconds"] == 0

    def test_the_published_test_periods_were_reproduced(
        self, published: dict[str, Any]
    ) -> None:
        baseline = published["results"]["published_baseline"]
        assert baseline["reproduced_test_periods"] is True
        assert [p["person_days"] for p in baseline["test_periods"]] == list(
            published_rows()
        )
        assert [p["label_days"] for p in baseline["test_periods"]] == list(
            published_positives()
        )

    def test_every_household_is_in_the_record(self, published: dict[str, Any]) -> None:
        results = published["results"]
        table = results["households"]
        assert len(table) == 56
        assert sum(row["monitored_days"] for row in table.values()) == (
            results["monitoring"]["monitored_days"]
        )
        assert sum(row["behavioural_alerts"] for row in table.values()) == (
            results["burden"]["behavioural_alerts"]
        )

    def test_the_page_is_exactly_the_rendering_of_the_records(
        self, published: dict[str, Any], described: dict[str, Any]
    ) -> None:
        committed = DOC.read_text(encoding="utf-8").replace("\r\n", "\n")
        assert committed == render_page(published, described)
        for name in published["results"]["households"]:
            assert f"| `{name}` |" in committed

    def test_the_figures_carry_the_records_data(
        self, published: dict[str, Any], described: dict[str, Any]
    ) -> None:
        data = {**figure_data(published), **post_hoc_figure_data(described)}
        assert set(data) == {*FIGURES, *POST_HOC_FIGURES}
        for name, plotted in data.items():
            svg = (FIGURE_DIR / f"tihm-alert-burden-{name}.svg").read_text(
                encoding="utf-8"
            )
            assert f"data sha256 {data_sha256(plotted)}" in svg


class TestPublishedPostHoc:
    """The descriptions of that run, as published. None is part of the protocol."""

    def test_it_describes_the_published_run_from_a_clean_commit(
        self, published: dict[str, Any], described: dict[str, Any]
    ) -> None:
        results = described["results"]
        assert results["status"] == STATUS
        assert results["describes"] == {
            "experiment": EXPERIMENT,
            "recorded_at": published["recorded_at"],
            "git_commit": published["environment"]["git_commit"],
            "protocol_sha256": published["configuration"]["protocol_sha256"],
            "results_reproduced": True,
        }
        assert described["environment"]["git_dirty"] == "false"
        assert described["environment"]["git_commit"].startswith("bc2ea4e")
        digests = {i["name"]: i["sha256"] for i in described["inputs"]}
        assert digests[PUBLISHED.name] == text_sha256(PUBLISHED)
        assert {name: digests[name] for name in FILES} == dict(FILES)
        assert described["configuration"]["replicates"] == PostHocConfig().replicates

    def test_the_recorded_replay_gives_the_published_verdicts(
        self, published: dict[str, Any], described: dict[str, Any]
    ) -> None:
        recorded = described["results"]["replays"]["recorded"]
        results = published["results"]
        assert recorded["reproduces_the_run"] is True
        assert recorded["deviating_days"] == results["burden"]["deviating_days"]
        verdicts = dict(results["monitoring"]["verdicts"])
        verdicts.pop("insufficient_data")
        assert recorded["verdicts"] == verdicts

    def test_every_silent_day_and_alert_is_accounted_for(
        self, published: dict[str, Any], described: dict[str, Any]
    ) -> None:
        results, later = published["results"], described["results"]
        silent, daily = later["silent_days"], later["calendar"]["daily"]
        alerts = results["burden"]["behavioural_alerts"]
        assert silent["behavioural_alerts"] == alerts
        assert sum(row["behavioural_alerts"] for row in daily) == alerts
        assert sum(later["alerts_by_feature_and_verdict"].values()) == alerts
        assert sum(row["silent_households"] for row in daily) == silent["silent_days"]
        assert sum(row["monitored_households"] for row in daily) == (
            results["monitoring"]["monitored_days"]
        )
        assert sum(
            int(length) * count for length, count in silent["runs_by_length"].items()
        ) == (silent["silent_days"])
        assert sum(row["events"] for row in daily) == (
            results["contract"]["events"]["emitted"]
        )
