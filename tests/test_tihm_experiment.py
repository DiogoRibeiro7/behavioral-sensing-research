"""Tests for the TIHM alert-burden evaluation, on synthetic files in the dataset's format.

The synthetic cohort has three homes with a daily rhythm. One has agitation
labels stamped at noon on days whose mornings are busier. The tests check the
protocol's declaration, the statistics, and that a whole run is written as a
valid record that says what it could and could not do.
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from sensor_modeling.datasets.tihm_experiment import (
    NO_LABEL,
    OTHER_SLOT,
    DayRecord,
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
from sensor_modeling.datasets.tihm_summary import render_protocol
from sensor_modeling.evaluation import load_record
from sensor_modeling.external.tihm import TIHM_MAPPING, TihmAdapter

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts" / "tihm" / "alert_burden_protocol.json"
START = date(2019, 4, 1)
DAYS = 32
HOMES = ("home1", "home2", "home3")
LABELLED_HOME = "home1"
LABEL_DAYS = (18, 19, 24, 29)
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
    """Three homes, 32 days, with noon agitation labels after busy mornings."""
    rng = np.random.default_rng(7)
    directory.mkdir(parents=True, exist_ok=True)
    activity: list[tuple[str, str, str]] = []
    labels: list[tuple[str, str, str]] = []
    physiology: list[tuple[str, str, str, str, str]] = []
    for home in HOMES:
        for offset in range(DAYS):
            day = datetime.combine(START + timedelta(days=offset), datetime.min.time())
            busy = home == LABELLED_HOME and offset in LABEL_DAYS
            for hour in range(24):
                block = hour // 6
                for room in ROOMS:
                    rate = RATES[room][block] * (3.0 if busy and block == 1 else 1.0)
                    for _ in range(rng.poisson(rate)):
                        moment = day + timedelta(
                            hours=hour, seconds=int(rng.integers(0, 3600))
                        )
                        activity.append((home, room, stamp(moment)))
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
    return {
        "data": data,
        "protocol": protocol,
        "path": result.path,
        "payload": load_record(result.path),
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

    def test_the_association_is_over_evaluable_days(
        self, world: dict[str, object]
    ) -> None:
        results = world["payload"]["results"]
        flagged = results["association"]["A2_deviating_days"]
        assert flagged["days"] == results["monitoring"]["evaluable_days"]
        assert flagged["label_days"] <= len(LABEL_DAYS)
        assert flagged["flagged_days"] == results["burden"]["deviating_days"]
        share = flagged["share_of_label_days_flagged"]["estimate"]
        assert share is None or 0.0 <= share <= 1.0

    def test_each_reference_gets_the_pipelines_number_of_flags(
        self, world: dict[str, object]
    ) -> None:
        results = world["payload"]["results"]
        references = results["references"]
        assert references["flags"] == results["burden"]["deviating_days"]
        for name in ("label_history", "event_count"):
            caught = references[name]["label_days_caught"]
            assert 0.0 <= caught <= references["label_days"]

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
