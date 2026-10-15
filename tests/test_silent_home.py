"""Tests for the silent-home protocol's declaration and its published records.

The protocol was frozen before any simulated home had been run with the rule
on, and the first tests check the declaration alone: that the frozen file is
what the code declares, that its page is generated from it, and that the
homes, the outages and the windows it fixes are reproducible and fit the
record.

The last tests read the published records. They check that the run was the
frozen protocol's, from a clean commit, that its criteria are what its
estimands decide and its estimands what its homes' own values give, and that
the results page and its figures are generated from the records. None of them
runs a home.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from sensor_modeling.datasets.external_figures import data_sha256
from sensor_modeling.datasets.silent_home_experiment import (
    BETWEEN_THE_FREEZE_AND_THE_RUN,
    criteria,
)
from sensor_modeling.datasets.silent_home_figures import (
    FIGURES,
    TIHM_FIGURES,
    figure_data,
    tihm_figure_data,
)
from sensor_modeling.datasets.silent_home_protocol import (
    CHANGE,
    CHANGE_AFTER_OUTAGE,
    COMMON,
    FLEET_ARMS,
    IN_TIME,
    LATE,
    OFF,
    OUTAGE,
    PIPELINE_ARMS,
    RESULT_SCHEMA,
    SHORT_OUTAGE,
    STABLE,
    TIHM_PUBLISHED,
    TIHM_SCHEMA,
    SilentHomeProtocol,
    check_frozen_protocol,
    condition_name,
    declared_protocol,
    event_sensors,
    write_protocol,
)
from sensor_modeling.datasets.silent_home_summary import (
    FIGURE_DIR,
    PROTOCOL_FILE,
    PROTOCOL_PAGE,
    RECORD_FILE,
    RESULTS_PAGE,
    TIHM_RECORD_FILE,
    render_page,
    render_protocol,
)
from sensor_modeling.evaluation import load_record
from sensor_modeling.external.tihm import FILES
from sensor_modeling.simulation.household import build_registry

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / PROTOCOL_FILE
LISBON = ZoneInfo("Europe/Lisbon")


class TestFrozenProtocol:
    def test_the_frozen_file_is_the_declared_protocol(self) -> None:
        digest = check_frozen_protocol(declared_protocol(), FROZEN)
        assert len(digest) == 64

    def test_the_protocol_page_is_generated_from_the_frozen_file(self) -> None:
        payload = json.loads(FROZEN.read_text(encoding="utf-8"))
        page = ROOT / "docs" / PROTOCOL_PAGE
        assert page.read_text(encoding="utf-8") == render_protocol(payload)

    @pytest.mark.parametrize(
        "changed",
        [
            {"horizons_hours": (18.0, 24.0)},
            {"margin_alerts": 0.5},
            {"homes": 120},
            {"follow_days": 14},
        ],
    )
    def test_a_changed_protocol_is_refused(self, changed: dict) -> None:
        with pytest.raises(ValueError, match="differs from the one this code declares"):
            check_frozen_protocol(SilentHomeProtocol(**changed), FROZEN)

    def test_writing_it_round_trips(self, tmp_path: Path) -> None:
        protocol = SilentHomeProtocol(resamples=300)
        path = tmp_path / "protocol.json"
        digest = write_protocol(protocol, path)
        assert digest == protocol.sha256()
        assert json.loads(path.read_text(encoding="utf-8"))["protocol_sha256"] == digest
        check_frozen_protocol(protocol, path)
        with pytest.raises(ValueError):
            check_frozen_protocol(declared_protocol(), path)

    def test_it_says_what_it_is_and_what_had_been_seen(self) -> None:
        declared = declared_protocol().to_dict()
        assert declared["status"].startswith("pre-specified on simulated homes")
        assert "Nothing on TIHM is a test" in declared["status"]
        assert any(
            "No simulated household had been run with the rule on" in item
            for item in declared["inspected_before"]
        )
        assert declared["simulator"]["evidence"].startswith("simulated")
        assert declared["tihm"]["standing"].startswith("a description, not a test")
        assert declared["the_rule"]["default"] == "off"
        assert declared["bootstrap"]["unit"] == "homes"
        assert len(declared["what_the_simulator_cannot_show"]) == 3

    def test_every_criterion_names_an_estimand_that_is_declared(self) -> None:
        declared = declared_protocol().to_dict()
        assert list(declared["estimands"]) == [
            "E1",
            "E2",
            "E3",
            "E2_by_group",
            "E4",
            "E5",
            "E6",
            "E7",
            "E8",
            "E9",
            "S1",
        ]
        assert [name.split("_")[0] for name in declared["criteria"]] == [
            f"C{k}" for k in range(1, 9)
        ]


class TestHomes:
    def test_the_seeds_come_from_the_root_and_are_distinct(self) -> None:
        protocol = declared_protocol()
        seeds = protocol.study_seeds()
        assert len(seeds) == len(set(seeds)) == protocol.homes == 100
        assert seeds == tuple(sorted(seeds))
        assert seeds == declared_protocol().study_seeds()
        assert SilentHomeProtocol(seed_root=1).study_seeds() != seeds
        # The household the run time was measured on is not one of them.
        assert 999 not in seeds

    def test_the_homes_keep_only_sensors_that_promise_nothing(self) -> None:
        registry = build_registry()
        kept = event_sensors()
        assert kept
        assert all(registry.get(name).expected_interval is None for name in kept)
        left_out = {spec.sensor_id for spec in registry} - set(kept)
        assert left_out == set(
            declared_protocol().to_dict()["simulator"]["sensors_left_out"]
        )
        assert all(
            registry.get(name).expected_interval is not None for name in left_out
        )

    def test_every_home_has_an_outage_of_its_own(self) -> None:
        protocol = declared_protocol()
        starts = protocol.outage_starts()
        assert set(starts) == set(protocol.study_seeds())
        assert starts == declared_protocol().outage_starts()
        for day, hour in starts.values():
            assert protocol.outage_first_day <= day <= protocol.outage_last_day
            assert 0 <= hour <= 23
        assert len(set(starts.values())) > protocol.homes // 2
        listed = protocol.to_dict()["simulator"]["outages"]
        assert [(home["seed"], home["day"], home["hour"]) for home in listed] == [
            (seed, *starts[seed]) for seed in protocol.study_seeds()
        ]

    def test_both_groups_are_in_the_homes(self) -> None:
        protocol = declared_protocol()
        groups = [protocol.group(seed) for seed in protocol.study_seeds()]
        assert groups.count(IN_TIME) + groups.count(LATE) == protocol.homes
        assert groups.count(IN_TIME) >= 30
        assert groups.count(LATE) >= 30
        for seed in protocol.study_seeds():
            _, hour = protocol.outage_starts()[seed]
            in_time = hour + protocol.primary_hours < 24
            assert protocol.group(seed) == (IN_TIME if in_time else LATE)


class TestWindows:
    def test_an_outage_begins_at_its_local_hour(self) -> None:
        protocol = declared_protocol()
        for seed in protocol.study_seeds()[:10]:
            day, hour = protocol.outage_starts()[seed]
            begin, end = protocol.outage(seed)
            local = begin.astimezone(LISBON)
            assert local.date() == protocol.start() + timedelta(days=day)
            assert (local.hour, local.minute) == (hour, 0)
            assert end - begin == timedelta(hours=protocol.outage_hours)
            short_begin, short_end = protocol.short_outage(seed)
            assert short_begin == begin
            assert short_end - short_begin == timedelta(
                hours=protocol.short_outage_hours
            )

    def test_the_alerts_of_an_outage_are_counted_for_four_weeks_after_it(self) -> None:
        protocol = declared_protocol()
        seed = protocol.study_seeds()[0]
        for arm, (begin, end) in (
            (OUTAGE, protocol.outage(seed)),
            (SHORT_OUTAGE, protocol.short_outage(seed)),
        ):
            assert protocol.window(seed, arm) == (
                begin,
                end + timedelta(days=protocol.follow_days),
            )
        with pytest.raises(KeyError):
            protocol.window(seed, STABLE)

    def test_every_window_fits_the_record(self) -> None:
        protocol = declared_protocol()
        record_ends = protocol.local(protocol.days, 0.0)
        for seed in protocol.study_seeds():
            assert protocol.window(seed, OUTAGE)[1] <= record_ends
            # The change follows every outage by at least a week.
            assert protocol.change_begins() - protocol.outage(seed)[1] >= timedelta(
                days=7
            )
        assert protocol.common_outage()[1] <= record_ends
        assert protocol.detection_window()[1] <= record_ends

    def test_no_outage_falls_on_the_day_the_clocks_change(self) -> None:
        protocol = declared_protocol()
        first = protocol.local(protocol.outage_first_day, 0.0).astimezone(LISBON)
        last = protocol.local(protocol.days, 0.0).astimezone(LISBON)
        assert first.utcoffset() == last.utcoffset() == timedelta(hours=1)

    def test_a_detection_needs_a_day_of_the_change(self) -> None:
        protocol = declared_protocol()
        begin, end = protocol.detection_window()
        # The change day closes a day after it begins; the alert raised at the
        # close of the day before the change is outside the window.
        assert begin - protocol.change_begins() == timedelta(days=1)
        assert end - begin == timedelta(days=protocol.max_delay_days)

    def test_the_common_outage_is_one_window_for_every_home(self) -> None:
        protocol = declared_protocol()
        begin, end = protocol.common_outage()
        local = begin.astimezone(LISBON)
        assert local.date() == protocol.start() + timedelta(days=protocol.common_day)
        assert local.hour == protocol.common_hour
        assert end - begin == timedelta(hours=protocol.outage_hours)


class TestRuns:
    def test_every_arm_is_run_off_and_at_the_primary_horizon(self) -> None:
        protocol = declared_protocol()
        primary = condition_name(protocol.primary_hours)
        assert primary == "h12"
        runs = protocol.runs()
        assert len(runs) == len(set(runs)) == 12
        for arm in PIPELINE_ARMS:
            assert (arm, OFF) in runs
            assert (arm, primary) in runs
        others = {(arm, c) for arm, c in runs if c not in (OFF, primary)}
        assert others == {(STABLE, "h24"), (OUTAGE, "h24")}

    def test_the_common_outage_is_read_by_the_fleet_check_alone(self) -> None:
        protocol = declared_protocol()
        assert COMMON not in PIPELINE_ARMS
        assert COMMON in FLEET_ARMS
        assert all(arm != COMMON for arm, _ in protocol.runs())
        assert {CHANGE, CHANGE_AFTER_OUTAGE}.isdisjoint(FLEET_ARMS)

    def test_a_condition_gives_its_horizon(self) -> None:
        protocol = declared_protocol()
        assert protocol.conditions == (OFF, "h12", "h24")
        assert protocol.horizon(OFF) is None
        assert protocol.horizon("h12") == timedelta(hours=12)
        assert protocol.horizon("h24") == timedelta(hours=24)
        with pytest.raises(KeyError):
            protocol.horizon("h6")


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"homes": 1},
            {"horizons_hours": ()},
            {"horizons_hours": (12.0, 12.0)},
            {"horizons_hours": (-1.0,)},
            {"short_outage_hours": 12.0},
            {"outage_hours": 24.0},
            {"outage_first_day": 50, "outage_last_day": 40},
            {"common_hour": 24},
            {"change_day": 47},
            {"days": 70},
        ],
    )
    def test_a_declaration_that_cannot_be_run_is_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            SilentHomeProtocol(**kwargs)


# ----------------------------------------------------------------------------
# The published records
# ----------------------------------------------------------------------------
RECORD = ROOT / RECORD_FILE
TIHM_RECORD = ROOT / TIHM_RECORD_FILE
ALERT_BURDEN = ROOT / "artifacts" / "tihm" / "tihm-alert-burden.json"
PAGE = ROOT / "docs" / RESULTS_PAGE
FIGURES_AT = ROOT / "docs" / FIGURE_DIR
RUN_COMMIT = "18b1e42"
ON = "h12"


@pytest.fixture(scope="module")
def published() -> dict[str, Any]:
    return load_record(RECORD)


@pytest.fixture(scope="module")
def described() -> dict[str, Any]:
    return load_record(TIHM_RECORD)


class TestPublishedRun:
    """The record of the protocol's test, as published."""

    def test_it_ran_the_frozen_protocol_from_a_clean_commit(
        self, published: dict[str, Any]
    ) -> None:
        protocol = declared_protocol()
        configuration = published["configuration"]
        assert configuration["protocol_sha256"] == protocol.sha256()
        assert configuration["protocol_file_sha256"] == check_frozen_protocol(
            protocol, FROZEN
        )
        assert published["inputs"][0]["sha256"] == configuration["protocol_file_sha256"]
        assert published["environment"]["git_dirty"] == "false"
        assert published["environment"]["git_commit"].startswith(RUN_COMMIT)
        assert published["data_source"] == "simulator"
        assert configuration["result_schema"] == RESULT_SCHEMA
        assert configuration["between_the_freeze_and_the_run"] == list(
            BETWEEN_THE_FREEZE_AND_THE_RUN
        )

    def test_every_home_and_every_run_of_the_protocol_is_in_it(
        self, published: dict[str, Any]
    ) -> None:
        protocol = declared_protocol()
        results = published["results"]
        seeds = protocol.study_seeds()
        assert results["homes"] == protocol.homes == 100
        assert published["seeds"] == [protocol.seed_root, protocol.seed, *seeds]
        assert set(results["per_home"]) == {str(seed) for seed in seeds}
        assert set(results["raw"]["homes"]) == {str(seed) for seed in seeds}
        assert set(results["runs"]) == {f"{a}/{c}" for a, c in protocol.runs()}
        for totals in results["runs"].values():
            assert totals["days_closed"] == protocol.homes * protocol.days
        for seed in seeds:
            day, hour = protocol.outage_starts()[seed]
            row = results["per_home"][str(seed)]
            assert (row["outage_day"], row["outage_hour"]) == (day, hour)
            assert row["group"] == protocol.group(seed)

    def test_the_criteria_are_what_the_estimands_decide(
        self, published: dict[str, Any]
    ) -> None:
        results = published["results"]
        assert criteria(results, declared_protocol()) == results["criteria"]
        assert sorted(results["criteria"]) == [f"C{k}" for k in range(1, 9)]

    def test_the_estimands_are_the_homes_own_values(
        self, published: dict[str, Any]
    ) -> None:
        results = published["results"]
        rows = list(results["per_home"].values())
        for condition, entry in results["outage"].items():
            excess = [
                row[f"{OUTAGE}/{condition}/alerts_in_the_outage_window"]
                - row[f"{STABLE}/{condition}/alerts_in_the_outage_window"]
                for row in rows
            ]
            assert entry["excess"]["estimate"] == pytest.approx(sum(excess) / len(rows))
            assert entry["homes_with_an_excess"] == sum(1 for e in excess if e > 0)
        for arm in (CHANGE, CHANGE_AFTER_OUTAGE):
            for condition in (OFF, ON):
                detected = results["detection"][arm][condition]["detected"]
                assert detected["count"] == sum(
                    1
                    for row in rows
                    if row[f"{arm}/{condition}/meets_the_detection_definition"]
                )
        reported = results["reporting"][ON]["reported"]
        assert reported["count"] == sum(
            1 for row in rows if row[f"{OUTAGE}/{ON}/reported"]
        )
        stable = results["stable"]
        for condition in (OFF, ON, "h24"):
            assert stable[condition]["behavioural_alerts"] == sum(
                row[f"{STABLE}/{condition}/behavioural_alerts"] for row in rows
            )

    def test_the_alerts_it_kept_give_the_counts_it_reports(
        self, published: dict[str, Any]
    ) -> None:
        protocol = declared_protocol()
        results = published["results"]
        for condition, entry in results["outage"].items():
            for arm, key in (
                (OUTAGE, "alerts_in_the_window"),
                (STABLE, "stable_alerts_in_the_window"),
            ):
                count = 0
                for seed, runs in results["raw"]["homes"].items():
                    begin, end = protocol.window(int(seed), OUTAGE)
                    count += sum(
                        1
                        for alert in runs[f"{arm}/{condition}"]["behavioural_alerts"]
                        if begin <= datetime.fromisoformat(alert[0]) < end
                    )
                assert count == entry[key]
        kept = sum(
            len(run["silence_alerts"])
            for runs in results["raw"]["homes"].values()
            for run in runs.values()
        )
        assert kept == sum(t["silence_alerts"] for t in results["runs"].values())

    def test_the_page_is_exactly_the_rendering_of_the_records(
        self, published: dict[str, Any], described: dict[str, Any]
    ) -> None:
        committed = PAGE.read_text(encoding="utf-8").replace("\r\n", "\n")
        assert committed == render_page(published, described)
        for seed in declared_protocol().study_seeds():
            assert f"| `{seed}` | day " in committed
        assert "**The evidence is simulated.** 100 paired simulated homes" in committed

    def test_the_page_says_what_the_verdicts_do_not(
        self, published: dict[str, Any]
    ) -> None:
        committed = PAGE.read_text(encoding="utf-8")
        results = published["results"]
        # An excess below zero and an inconclusive difference that excludes
        # zero are each said in words, beside the verdict.
        assert results["outage"][ON]["excess"]["interval"]["high"] < 0.0
        assert "**E2 lies below zero.**" in committed
        difference = results["detection"][CHANGE_AFTER_OUTAGE]["on_minus_off"]
        assert results["criteria"]["C5"]["verdict"] == "inconclusive"
        assert difference["interval"]["high"] < 0.0
        assert "**C5 is inconclusive, which is not no difference.**" in committed
        # The change arm never had the rule act, and the page says so.
        assert results["runs"][f"{CHANGE}/{ON}"]["silence_alerts"] == 0
        assert "In this arm the rule never acted" in committed
        # A statement of the protocol that the record does not bear out.
        rows = results["per_home"].values()
        refused = sum(
            1 for row in rows if row[f"{OUTAGE}/h24/first_outage_day_refused"]
        )
        assert refused > 0
        assert (
            f"Under `h24` the day on which the outage began was refused in "
            f"{refused} of the 100 homes." in committed
        )
        assert "the record does not bear that out" in committed

    def test_the_figures_carry_the_records_data(
        self, published: dict[str, Any], described: dict[str, Any]
    ) -> None:
        data = {**figure_data(published), **tihm_figure_data(described)}
        assert set(data) == {*FIGURES, *TIHM_FIGURES}
        for name, plotted in data.items():
            svg = (FIGURES_AT / f"silent-home-{name}.svg").read_text(encoding="utf-8")
            assert f"data sha256 {data_sha256(plotted)}" in svg


class TestPublishedDescription:
    """The record of the description on TIHM, as published. Not a test."""

    def test_it_reproduces_the_alert_burden_run_home_by_home(
        self, described: dict[str, Any]
    ) -> None:
        burden = load_record(ALERT_BURDEN)["results"]
        check = described["results"]["check"]
        assert check["published"] == TIHM_PUBLISHED
        assert check["monitored_days"] == burden["monitoring"]["monitored_days"] == 2850
        assert check["behavioural_alerts"] == burden["burden"]["behavioural_alerts"]
        assert check["homes_compared_one_by_one"] == len(burden["households"]) == 56
        off = described["results"]["conditions"][OFF]
        assert off["usable_days"] == burden["monitoring"]["usable_days"]
        assert off["evaluable_days"] == burden["monitoring"]["evaluable_days"]
        assert off["behavioural_alerts"]["by_feature"] == burden["burden"]["by_feature"]
        assert off["days_refused_because_of_the_rule"]["all"] == 0
        assert off["silence_alerts"]["all"] == 0

    def test_it_was_made_from_a_clean_commit_on_the_pinned_files(
        self, described: dict[str, Any]
    ) -> None:
        protocol = declared_protocol()
        configuration = described["configuration"]
        assert described["environment"]["git_dirty"] == "false"
        assert described["environment"]["git_commit"].startswith(RUN_COMMIT)
        assert described["data_source"] == "tihm"
        assert configuration["protocol_sha256"] == protocol.sha256()
        assert configuration["result_schema"] == TIHM_SCHEMA
        digests = {item["name"]: item["sha256"] for item in described["inputs"]}
        assert {name: digests[name] for name in FILES} == dict(FILES)
        assert any("A description, not a test" in note for note in described["notes"])

    def test_every_condition_keeps_every_monitored_day(
        self, described: dict[str, Any]
    ) -> None:
        conditions = described["results"]["conditions"]
        assert list(declared_protocol().conditions) == sorted(
            conditions, key=[OFF, ON, "h24"].index
        )
        for entry in conditions.values():
            assert entry["monitored_days"] == 2850
            refused = entry["days_refused_because_of_the_rule"]
            assert refused["all"] == sum(refused["per_home"].values())
            assert len(refused["per_home"]) == 56
            alerts = entry["behavioural_alerts"]
            assert sum(alerts["by_day_summarised"].values()) == alerts["all"]
            assert (
                alerts["also_raised_with_the_rule_off"]
                + alerts["raised_only_with_the_rule_off"]
                == 183
            )
