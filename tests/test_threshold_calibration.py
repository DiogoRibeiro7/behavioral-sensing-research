"""Tests for the threshold-calibration protocol's declaration.

The protocol was frozen before any simulated home had been run with the
calibrated reference. These tests check the declaration alone: that the frozen
file is what the code declares, that its page is generated from it, that the
records it rests on are the ones in the repository, and that the homes, the
conditions and the windows it fixes are reproducible and fit the record. None
of them runs a home.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from sensor_modeling.baseline import BaselineConfig
from sensor_modeling.datasets.silent_home_protocol import (
    declared_protocol as silent_home_protocol,
)
from sensor_modeling.datasets.threshold_calibration_protocol import (
    ARMS,
    BETWEEN_FILE,
    CALIBRATED,
    CHANGE,
    CHANGE_ARMS,
    DEFAULT,
    GRADUAL_CHANGE,
    NULL_RECORD,
    PINNED_DISTRIBUTIONS,
    PINNED_SCRIPTS,
    PLANNING_RECORD,
    SMALL_CHANGE,
    STABLE,
    TIHM_PUBLISHED,
    TIHM_RECORDS,
    ThresholdCalibrationProtocol,
    between_the_freeze_and_the_run,
    check_frozen_protocol,
    check_inputs,
    code_changes,
    code_now,
    condition_name,
    declared_protocol,
    equivalent_threshold,
    expected_from,
    file_sha256,
    nominal,
    pinned_sources,
    planned_from,
    write_protocol,
)
from sensor_modeling.datasets.threshold_calibration_summary import (
    PROTOCOL_FILE,
    PROTOCOL_PAGE,
    render_protocol,
)
from sensor_modeling.datasets.threshold_null import ThresholdNull
from sensor_modeling.evaluation import load_record

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / PROTOCOL_FILE
LISBON = ZoneInfo("Europe/Lisbon")


def pins(protocol: ThresholdCalibrationProtocol) -> dict:
    """The records a protocol is pinned to, as keyword arguments."""
    return {
        name: getattr(protocol, name)
        for name in (
            "null_record_sha256",
            "null_expected",
            "planning_record_sha256",
            "planning_expected",
            "tihm_records_sha256",
        )
    }


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
            {"checked_homes": 20},
            {"check_scales": (1.0, 0.7, 0.5)},
            {"calibration_thresholds": (1.5, 3.0)},
            {"informative_detection": 0.1},
            {"planned_sd": 0.5},
            {"tihm_checked_homes": 3},
            {"resamples": 2000},
            {"calibration_tolerance": 0.5},
            {"homes": 100},
            {"gradual_max_delay_days": 28.0},
            {"seed_root": 20261007},
        ],
    )
    def test_a_changed_protocol_is_refused(self, changed: dict) -> None:
        declared = declared_protocol()
        other = ThresholdCalibrationProtocol(**{**pins(declared), **changed})
        with pytest.raises(ValueError, match="differs from the one this code declares"):
            check_frozen_protocol(other, FROZEN)

    def test_the_records_it_rests_on_are_the_ones_in_the_repository(self) -> None:
        declared = declared_protocol()
        check_inputs(declared, ROOT)
        assert declared.null_record_sha256 == file_sha256(ROOT / NULL_RECORD)
        assert dict(declared.tihm_records_sha256) == {
            name: file_sha256(ROOT / name) for name in TIHM_RECORDS
        }

    def test_a_protocol_that_pins_nothing_is_not_frozen(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not pin the measurement"):
            write_protocol(ThresholdCalibrationProtocol(), tmp_path / "protocol.json")
        declared = declared_protocol()
        for name in ("null_record_sha256", "planning_record_sha256"):
            elsewhere = ThresholdCalibrationProtocol(
                **{**pins(declared), name: "0" * 64}
            )
            with pytest.raises(ValueError, match="is not the record the protocol"):
                check_inputs(elsewhere, ROOT)
        unplanned = ThresholdCalibrationProtocol(
            **{**pins(declared), "planning_expected": ()}
        )
        with pytest.raises(ValueError, match="does not pin the trials"):
            check_inputs(unplanned, ROOT)
        unpinned = ThresholdCalibrationProtocol(
            **{**pins(declared), "tihm_records_sha256": ()}
        )
        with pytest.raises(ValueError, match="published TIHM records"):
            check_inputs(unpinned, ROOT)

    def test_what_synthetic_days_gave_is_what_their_record_holds(self) -> None:
        declared = declared_protocol()
        measured = load_record(ROOT / NULL_RECORD)
        assert declared.null_expected == expected_from(measured)
        operating = measured["results"]["operating"]
        expected = {name: dict(values) for name, values in declared.null_expected}
        assert set(expected) == {"plain", "weekly_rhythm"}
        plain = expected["plain"]
        assert plain["default_false"] == (
            operating["none"]["default"]["1"]["found"]["share"]
        )
        assert plain["default_found"] == (
            operating["step2"]["default"]["1"]["found"]["share"]
        )
        matched = f"{plain['same_false_alerts']:g}"
        assert plain["at_the_same_false_alerts_found"] == (
            operating["step2"]["calibrated"][matched]["found"]["share"]
        )
        assert plain["at_the_same_false_alerts_false"] <= plain["default_false"]
        assert set(plain) == {
            "same_false_alerts",
            "default_false",
            "default_found",
            "at_the_same_false_alerts_false",
            "at_the_same_false_alerts_found",
        }
        rhythm = expected["weekly_rhythm"]
        assert rhythm["default_false"] == (
            operating["weekend"]["default"]["1"]["found"]["share"]
        )

    def test_a_protocol_that_misstates_the_measurement_is_refused(self) -> None:
        declared = declared_protocol()
        wrong = tuple(
            (name, tuple((key, value + 0.01) for key, value in values))
            for name, values in declared.null_expected
        )
        other = ThresholdCalibrationProtocol(
            **{**pins(declared), "null_expected": wrong}
        )
        with pytest.raises(ValueError, match="is not what it holds"):
            check_inputs(other, ROOT)
        misquoted = tuple(
            (key, value + 0.001) for key, value in declared.planning_expected
        )
        other = ThresholdCalibrationProtocol(
            **{**pins(declared), "planning_expected": misquoted}
        )
        with pytest.raises(ValueError, match="is not what it holds"):
            check_inputs(other, ROOT)

    def test_trials_made_with_other_scoring_functions_are_refused(
        self, tmp_path: Path
    ) -> None:
        declared = declared_protocol()
        root = tmp_path / "root"
        for name in (NULL_RECORD, PLANNING_RECORD, *TIHM_RECORDS):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes((ROOT / name).read_bytes())
        check_inputs(declared, root)
        tried = json.loads((root / PLANNING_RECORD).read_text(encoding="utf-8"))
        tried["configuration"]["scoring_functions"]["crossing"] = "0" * 64
        (root / PLANNING_RECORD).write_text(json.dumps(tried), encoding="utf-8")
        moved = ThresholdCalibrationProtocol(
            **{
                **pins(declared),
                "planning_record_sha256": file_sha256(root / PLANNING_RECORD),
            }
        )
        with pytest.raises(ValueError, match="other scoring functions"):
            check_inputs(moved, root)

    def test_the_trials_quoted_are_read_off_the_right_cells(self) -> None:
        # A record whose every range is distinct, so that a range read off
        # the wrong ratio, outcome or default is seen.
        counter = iter(range(1, 1000))

        def ranges() -> dict:
            return {
                outcome: {"low": next(counter) / 1000, "high": next(counter) / 1000}
                for outcome in (
                    "better",
                    "worse",
                    "not_shown",
                    "not_bracketed",
                    "mean_estimate",
                    "sd_of_estimates",
                )
            }

        ratios = ("1", "0.5", "0.75", "1.5")
        shapes = ("most", "nearly_all", "few")
        quiet = {ratio: ranges() for ratio in ratios}
        quiet["by_shape"] = {shape: {r: ranges() for r in ratios} for shape in shapes}
        kept = {"by_shape": {shape: {r: ranges() for r in ratios} for shape in shapes}}
        record = {
            "configuration": {"offsets": [0.45, 0.6, 1.1]},
            "results": {
                "summary": {"same_false_alerts": quiet, "same_detection": kept}
            },
        }
        found = dict(planned_from(record))
        assert (found["offset_low"], found["offset_high"]) == (0.45, 1.1)
        assert found["same_curve_better_low"] == quiet["1"]["better"]["low"]
        assert found["same_curve_worse_high"] == quiet["1"]["worse"]["high"]
        assert found["same_curve_estimate_low"] == quiet["1"]["mean_estimate"]["low"]
        assert found["error_high"] == quiet["1"]["sd_of_estimates"]["high"]
        for shape in shapes:
            by = quiet["by_shape"][shape]
            assert found[f"half_better_{shape}_low"] == by["0.5"]["better"]["low"]
            assert found[f"quarter_fewer_better_{shape}_high"] == (
                by["0.75"]["better"]["high"]
            )
            assert found[f"half_more_worse_{shape}_low"] == by["1.5"]["worse"]["low"]
        other = kept["by_shape"]["nearly_all"]
        assert found["same_detection_same_curve_better_high"] == (
            other["1"]["better"]["high"]
        )
        assert found["same_detection_half_more_not_bracketed_low"] == (
            other["1.5"]["not_bracketed"]["low"]
        )
        assert found["same_detection_half_more_estimate_high"] == (
            other["1.5"]["mean_estimate"]["high"]
        )
        assert len(found) == 2 * 17
        other["1.5"]["mean_estimate"] = None
        assert "same_detection_half_more_estimate_low" not in dict(planned_from(record))

    def test_the_sentences_say_what_the_trials_gave(self) -> None:
        declared = declared_protocol()
        planned = dict(declared.planning_expected)
        tried = declared.to_dict()["inspected_before"][-1]
        low, high = planned["same_curve_worse_low"], planned["same_curve_worse_high"]
        assert f"worse in {100 * low:.0f}" in tried
        assert f"to {100 * high:.0f}" in tried or f"{100 * low:.0f}" == (
            f"{100 * high:.0f}"
        )
        planning = declared.to_dict()["simulator"]["planning"]
        assert "where it finds nearly all" in planning
        why = declared.to_dict()["matching"]["why_not_the_same_detection"]
        assert "where the truth is more" in why or (
            "same_detection_half_more_estimate_low" not in planned
        )

    def test_the_measurement_was_made_at_multiples_the_homes_are_run_at(
        self,
    ) -> None:
        grid = declared_protocol().curve_scales
        assert set(ThresholdNull().scales) <= set(grid)
        assert len(grid) == 42 and grid[0] == 0.3 and grid[-1] == 2.0
        # Neighbouring multiples are close where a match can fall, and the
        # grid is coarse only above 1.5, as the protocol says.
        ratios = [high / low for low, high in zip(grid, grid[1:])]
        assert max(ratio for low, ratio in zip(grid, ratios) if 0.4 <= low < 1.5) < 1.07
        assert max(ratio for low, ratio in zip(grid, ratios) if low >= 1.5) > 1.1

    def test_it_says_what_it_is_and_what_had_been_seen(self) -> None:
        declared = declared_protocol().to_dict()
        assert declared["status"].startswith("pre-specified on simulated homes")
        assert "Nothing on TIHM is a test" in declared["status"]
        seen = " ".join(declared["inspected_before"])
        assert "No simulated household had been run with it" in seen
        assert "reviewed before it was frozen" in seen
        assert "The calibrated reference was not run on them" in seen
        assert "A second review, before the freeze" in seen
        assert "curves written down by hand" in seen
        assert seen.count("no simulated home") >= 2
        assert declared["simulator"]["evidence"].startswith("simulated")
        assert declared["tihm"]["standing"].startswith("a description, not a test")
        assert "gradual-drift verdicts" in declared["what_the_outcomes_turn_on"]


class TestCodeAtTheFreeze:
    def test_the_frozen_file_records_the_code_it_was_frozen_against(self) -> None:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["at_freeze"]
        assert set(frozen) == {"sources", "distributions", "defaults"}
        assert all(len(digest) == 64 for digest in frozen["sources"].values())
        assert set(PINNED_SCRIPTS) <= set(frozen["sources"])
        assert set(frozen["distributions"]) == {"python", *PINNED_DISTRIBUTIONS}
        assert {"baseline", "pipeline", "alert_policy", "health", "household"} <= set(
            frozen["defaults"]
        )
        assert frozen["defaults"]["baseline"]["calibrated"] is False
        assert frozen["defaults"]["baseline"]["deviation_threshold"] == 3.0

    def test_every_module_of_the_package_is_recorded(self) -> None:
        names = pinned_sources(ROOT)
        assert list(names[: -len(PINNED_SCRIPTS)]) == sorted(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "sensor_modeling").rglob("*.py")
        )
        # What a day passes through before a threshold is asked anything, and
        # what the thresholds are then asked by.
        for name in (
            "sensor_modeling/baseline/adaptive.py",
            "sensor_modeling/alerts/alert.py",
            "sensor_modeling/online/pipeline.py",
            "sensor_modeling/online/replay.py",
            "sensor_modeling/simulation/household.py",
            "sensor_modeling/evaluation/resampling.py",
            "sensor_modeling/models/change_point_detection/pelt.py",
            "sensor_modeling/datasets/threshold_calibration_experiment.py",
            "sensor_modeling/datasets/threshold_calibration_protocol.py",
            "sensor_modeling/datasets/silent_home_experiment.py",
        ):
            assert name in names
        assert not any("__pycache__" in name for name in names)
        assert set(code_now(ROOT)["sources"]) == set(names)

    def test_it_is_not_part_of_the_protocol_s_digest(self, tmp_path: Path) -> None:
        declared = declared_protocol()
        path = tmp_path / "protocol.json"
        assert write_protocol(declared, path) == declared.sha256()
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["protocol_sha256"] == declared.sha256()
        changed = "sensor_modeling/baseline/adaptive.py"
        written["at_freeze"]["sources"][changed] = "0" * 64
        path.write_text(json.dumps(written), encoding="utf-8")
        # The declaration is still the frozen one, and the change is seen.
        check_frozen_protocol(declared, path)
        assert code_changes(path, ROOT) == {
            "sources": [changed],
            "distributions": [],
            "defaults": [],
        }

    def test_a_file_added_or_removed_is_seen(self, tmp_path: Path) -> None:
        path = tmp_path / "protocol.json"
        write_protocol(declared_protocol(), path)
        written = json.loads(path.read_text(encoding="utf-8"))
        del written["at_freeze"]["sources"]["sensor_modeling/online/replay.py"]
        written["at_freeze"]["sources"]["sensor_modeling/gone.py"] = "0" * 64
        path.write_text(json.dumps(written), encoding="utf-8")
        assert code_changes(path, ROOT)["sources"] == [
            "sensor_modeling/gone.py",
            "sensor_modeling/online/replay.py",
        ]

    def test_a_changed_default_or_library_is_seen(self, tmp_path: Path) -> None:
        path = tmp_path / "protocol.json"
        write_protocol(declared_protocol(), path)
        assert code_changes(path, ROOT) == {
            "sources": [],
            "distributions": [],
            "defaults": [],
        }
        written = json.loads(path.read_text(encoding="utf-8"))
        written["at_freeze"]["defaults"]["baseline"]["min_samples"] = 10
        written["at_freeze"]["distributions"]["scipy"] = "0.0"
        path.write_text(json.dumps(written), encoding="utf-8")
        found = code_changes(path, ROOT)
        assert found["defaults"] == ["baseline"]
        assert found["distributions"] == ["scipy"]

    def test_a_file_without_it_cannot_be_compared(self, tmp_path: Path) -> None:
        path = tmp_path / "protocol.json"
        path.write_text(json.dumps(declared_protocol().to_dict()), encoding="utf-8")
        with pytest.raises(ValueError, match="does not record the code"):
            code_changes(path, ROOT)

    def test_a_digest_does_not_depend_on_line_endings(self, tmp_path: Path) -> None:
        unix, windows = tmp_path / "a.py", tmp_path / "b.py"
        unix.write_bytes(b"one\ntwo\n")
        windows.write_bytes(b"one\r\ntwo\r\n")
        assert file_sha256(unix) == file_sha256(windows)

    def test_what_happened_before_the_run_is_a_list_of_sentences(
        self, tmp_path: Path
    ) -> None:
        assert between_the_freeze_and_the_run(tmp_path) == ()
        path = tmp_path / BETWEEN_FILE
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(["one thing was run"]), encoding="utf-8")
        assert between_the_freeze_and_the_run(tmp_path) == ("one thing was run",)
        path.write_text(json.dumps({"a": "b"}), encoding="utf-8")
        with pytest.raises(ValueError, match="list of sentences"):
            between_the_freeze_and_the_run(tmp_path)
        path.write_text(json.dumps(["one", ""]), encoding="utf-8")
        with pytest.raises(ValueError, match="list of sentences"):
            between_the_freeze_and_the_run(tmp_path)
        # It is not code, so adding to it changes none of the recorded sources.
        assert not BETWEEN_FILE.endswith(".py")
        assert BETWEEN_FILE not in pinned_sources(ROOT)


class TestDeclaration:
    def test_the_homes_come_from_the_root_alone(self) -> None:
        declared = declared_protocol()
        seeds = declared.study_seeds()
        assert len(seeds) == len(set(seeds)) == declared.homes == 400
        assert list(seeds) == sorted(seeds)
        assert seeds == ThresholdCalibrationProtocol().study_seeds()

    def test_the_homes_checked_are_the_first_by_seed(self) -> None:
        declared = declared_protocol()
        assert declared.checked_seeds() == declared.study_seeds()[:10]
        assert declared.to_dict()["simulator"]["replay"]["checked_seeds"] == list(
            declared.checked_seeds()
        )

    def test_no_home_has_been_seen_before(self) -> None:
        seeds = set(declared_protocol().study_seeds())
        assert not seeds & set(silent_home_protocol().study_seeds())
        # Homes run while this and the silent-home scoring were written.
        seen = {999, 2024, 807611, 563265, 679832, 785210, 126987, 157858, 244682}
        seen |= {140752, 164848, 303487, 313864, 318792, 764755}
        assert not seeds & seen

    def test_the_number_of_homes_is_planned_on_the_criterion(self) -> None:
        declared = declared_protocol()
        assert declared.standard_error_planned() == pytest.approx(0.6 / 400**0.5)
        assert declared.standard_error_planned(100) == pytest.approx(0.06)
        # A difference of 2.8 standard errors is shown in 80 runs of 100.
        assert declared.planned_difference(0.8) == pytest.approx(0.084, abs=0.0005)
        assert declared.planned_power(declared.planned_difference(0.8)) == (
            pytest.approx(0.8)
        )
        assert declared.planned_power(0.0) == pytest.approx(0.025)
        assert declared.planned_power(0.05) == pytest.approx(0.385, abs=0.005)
        assert declared.planned_power(0.05, homes=100) < 0.15
        planning = declared.to_dict()["simulator"]["planning"]
        assert "found 0.084 more of the homes" in planning
        assert "curves written down by hand" in planning

    def test_the_conditions(self) -> None:
        declared = declared_protocol()
        assert (declared.default, declared.declared) == ("default@1", "calibrated@1")
        assert len(declared.conditions) == len(set(declared.conditions)) == 84
        assert declared.checked_conditions == (
            "calibrated@1",
            "calibrated@0.6",
            "calibrated@0.5",
        )
        assert set(declared.checked_conditions) <= set(declared.conditions)
        assert declared.baseline("default@1") == BaselineConfig()
        kept = declared.baseline("calibrated@0.6")
        assert kept.calibrated
        assert kept.deviation_threshold == pytest.approx(1.8)
        assert kept.trend_threshold == pytest.approx(2.1)
        raised = declared.baseline("default@2")
        assert not raised.calibrated and raised.deviation_threshold == 6.0
        with pytest.raises(KeyError):
            declared.baseline("calibrated@0.52")
        with pytest.raises(KeyError):
            condition_name("robust", 1.0)
        assert condition_name(CALIBRATED, 1.25) == "calibrated@1.25"
        assert condition_name(DEFAULT, 1.0) == "default@1"

    def test_the_conditions_described_in_full_follow_the_match(self) -> None:
        declared = declared_protocol()
        assert declared.shown(0.55) == ("default@1", "calibrated@1", "calibrated@0.55")
        assert declared.shown(1.0) == ("default@1", "calibrated@1", "calibrated@1")
        assert set(declared.shown(2.0)) <= set(declared.conditions)

    def test_the_arms(self) -> None:
        declared = declared_protocol()
        assert declared.shift(STABLE) is None
        step, small, gradual = (declared.shift(arm) for arm in CHANGE_ARMS)
        assert (step.start_day, step.sleep_delta_hours, step.ramp_days) == (56, 1.6, 0)
        assert step.night_bathroom_extra == 1.2
        assert small.start_day == 56
        assert small.sleep_delta_hours == pytest.approx(0.8)
        assert small.night_bathroom_extra == pytest.approx(0.6)
        assert (gradual.start_day, gradual.ramp_days) == (35, 28)
        assert gradual.sleep_delta_hours == 1.6
        with pytest.raises(KeyError):
            declared.shift("outage")
        assert set(declared.to_dict()["simulator"]["arms"]) == set(ARMS)

    def test_a_change_is_dated_from_the_first_day_it_moves_anything(self) -> None:
        declared = declared_protocol()
        assert declared.first_changed_day(CHANGE) == 56
        assert declared.first_changed_day(SMALL_CHANGE) == 56
        gradual = declared.shift(GRADUAL_CHANGE)
        assert gradual.strength_on(35) == 0.0
        assert gradual.strength_on(36) == pytest.approx(1 / 28)
        assert gradual.strength_on(63) == 1.0
        assert declared.first_changed_day(GRADUAL_CHANGE) == 36
        with pytest.raises(KeyError):
            declared.first_changed_day(STABLE)

    def test_the_windows_fit_the_record_and_open_at_local_midnight(self) -> None:
        declared = declared_protocol()
        end = declared.local(declared.days)
        for arm, first, days in (
            (CHANGE, 56, 21),
            (SMALL_CHANGE, 56, 21),
            (GRADUAL_CHANGE, 36, 42),
        ):
            begin, close = declared.detection_window(arm)
            local = begin.astimezone(LISBON)
            assert (local.hour, local.minute) == (0, 0)
            assert local.date() == declared.start() + timedelta(days=first + 1)
            assert close - begin == timedelta(days=days)
            assert close <= end
            assert declared.begins(arm) == declared.local(first)
        with pytest.raises(KeyError):
            declared.detection_window(STABLE)

    def test_the_calendar_is_what_the_protocol_says(self) -> None:
        declared = declared_protocol()
        assert declared.start().weekday() == 0
        assert (declared.start() + timedelta(days=56)).weekday() == 0
        assert (declared.start() + timedelta(days=35)).weekday() == 0
        # The day the clocks go forward is an hour short.
        short = declared.local(28) - declared.local(27)
        assert short == timedelta(hours=23)
        assert declared.local(27) - declared.local(26) == timedelta(hours=24)

    def test_what_a_threshold_states_and_its_inverse(self) -> None:
        declared = declared_protocol()
        assert declared.calibration_thresholds == (1.5, 2.0, 3.0)
        assert nominal(3.0) == pytest.approx(0.0027, abs=1e-4)
        for threshold in declared.calibration_thresholds:
            assert equivalent_threshold(nominal(threshold)) == pytest.approx(threshold)

    def test_the_published_tihm_counts_are_the_records(self) -> None:
        described = load_record(ROOT / TIHM_RECORDS[1])["results"]["conditions"]
        for rule, published in TIHM_PUBLISHED.items():
            assert described[rule]["monitored_days"] == published["monitored_days"]
            assert described[rule]["behavioural_alerts"]["all"] == (
                published["behavioural_alerts"]
            )
        burden = load_record(ROOT / TIHM_RECORDS[0])["results"]["households"]
        assert sum(home["monitored_days"] for home in burden.values()) == 2850
        assert sum(home["behavioural_alerts"] for home in burden.values()) == 183

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"homes": 1},
            {"checked_homes": 0},
            {"checked_homes": 401},
            {"tihm_checked_homes": 0},
            {"curve_scales": (0.5, 0.6, 1.0, 1.0)},
            {"curve_scales": (0.6, 0.5, 1.0)},
            {"curve_scales": (0.5, 0.6)},
            {"check_scales": (0.52,)},
            {"calibration_thresholds": ()},
            {"small_fraction": 1.0},
            {"calibration_tolerance": 0.0},
            {"planned_sd": 0.0},
            {"gradual_day": 20},
            {"gradual_ramp_days": 0},
            {"days": 70},
            {"gradual_max_delay_days": 60.0},
        ],
    )
    def test_a_declaration_that_makes_no_sense_is_refused(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            ThresholdCalibrationProtocol(**kwargs)

    def test_the_declaration_serialises_and_orders_itself(self) -> None:
        declared = declared_protocol().to_dict()
        assert json.loads(json.dumps(declared, allow_nan=False)) == declared
        assert declared["conditions"]["role_order"] == [
            "default",
            "declared",
            "same_false_alerts",
        ]
        assert set(declared["conditions"]["roles"]) == set(
            declared["conditions"]["role_order"]
        )
        assert declared["simulator"]["arm_order"] == list(ARMS)
        definitions = declared["definitions"]
        assert set(definitions["order"]) == set(definitions) - {"order"}
        assert list(declared["estimands"]) == [f"E{k}" for k in range(1, 7)]
        criteria = declared["criteria"]
        assert criteria["order"] == [
            "C1_more_is_detected_at_the_same_false_alerts",
            "C2_the_threshold_means_what_it_says",
        ]
        assert set(criteria) == {
            *criteria["order"],
            "order",
            "margin_order",
            "margins",
            "readings",
        }
        assert criteria["margin_order"] == list(criteria["margins"])
        for name in ("matching", "bootstrap"):
            assert set(declared[name]["order"]) == set(declared[name]) - {"order"}

    def test_the_match_is_at_the_same_false_alerts_and_nowhere_else(self) -> None:
        declared = declared_protocol().to_dict()
        matching = declared["matching"]
        assert "smallest multiple" in matching["same_false_alerts"]
        assert "No estimand does" in matching["why_not_the_same_detection"]
        assert (
            "E1" in declared["criteria"]["C1_more_is_detected_at_the_same_false_alerts"]
        )
        assert "at the same false alerts" in declared["estimands"]["E1"]
        assert "same false alerts" in declared["estimands"]["E4"]
        assert not any(
            "same detection" in text for text in declared["estimands"].values()
        )


# ----------------------------------------------------------------------------
# The published run and description
# ----------------------------------------------------------------------------
RUN_COMMIT = "b0f6035"
RECORD = ROOT / "artifacts/threshold_calibration/threshold-calibration.json"
TIHM_RECORD = ROOT / "artifacts/threshold_calibration/threshold-calibration-tihm.json"


@pytest.fixture(scope="module")
def published() -> dict:
    return load_record(RECORD)


@pytest.fixture(scope="module")
def described() -> dict:
    return load_record(TIHM_RECORD)


def flags(text: str) -> list[int]:
    return [int(character) for character in text]


class TestPublishedRun:
    """The record of the protocol's test, as published."""

    def test_it_ran_the_frozen_protocol_from_a_clean_commit(
        self, published: dict
    ) -> None:
        protocol = declared_protocol()
        configuration = published["configuration"]
        assert configuration["protocol_sha256"] == protocol.sha256()
        assert configuration["protocol_file_sha256"] == check_frozen_protocol(
            protocol, FROZEN
        )
        assert published["inputs"][0]["sha256"] == configuration["protocol_file_sha256"]
        assert {entry["source"] for entry in published["inputs"]} >= {
            NULL_RECORD,
            PLANNING_RECORD,
        }
        assert published["environment"]["git_dirty"] == "false"
        assert published["environment"]["git_commit"].startswith(RUN_COMMIT)
        assert published["data_source"] == "simulator"
        assert configuration["code_changed_since_the_freeze"] == {
            "sources": [],
            "distributions": [],
            "defaults": [],
        }
        assert configuration["between_the_freeze_and_the_run"] == list(
            between_the_freeze_and_the_run(ROOT)
        )

    def test_every_home_of_the_protocol_is_in_it(self, published: dict) -> None:
        protocol = declared_protocol()
        results, seeds = published["results"], protocol.study_seeds()
        assert results["homes"] == protocol.homes == 400
        assert published["seeds"] == [protocol.seed_root, protocol.seed, *seeds]
        assert results["raw"]["seeds"] == list(seeds)
        rows = published["household_metrics"]["simulated_homes"]
        assert set(rows) == {str(seed) for seed in seeds}
        check = results["check"]
        assert check["default_replays"] == 4 * 400
        assert check["default_replays_that_differ"] == []
        assert check["calibrated_runs_checked"] == 10 * 4 * 3
        assert check["calibrated_runs_that_differ"] == []

    def test_the_criteria_are_what_the_estimands_decide(self, published: dict) -> None:
        from sensor_modeling.datasets.threshold_calibration_experiment import (
            criteria,
        )

        results = published["results"]
        assert criteria(results, declared_protocol()) == results["criteria"]

    def test_the_match_is_what_the_homes_own_counts_give(self, published: dict) -> None:
        import numpy as np

        from sensor_modeling.datasets import threshold_calibration_experiment as e

        protocol = declared_protocol()
        conditions = published["results"]["raw"]["conditions"]
        homes = protocol.homes

        def counts(name: str) -> np.ndarray:
            return np.array(
                [int(v) for v in conditions[name]["false_alerts"].split()],
                dtype=np.int64,
            )

        def net(name: str, arm: str) -> np.ndarray:
            found = conditions[name]
            return np.array(flags(found[f"{arm}/detected"])) - np.array(
                flags(found[f"{arm}/false_detection"])
            )

        names = [condition_name(CALIBRATED, s) for s in protocol.curve_scales]
        samples = e._samples(homes, protocol)
        false = samples @ np.column_stack([counts(name) for name in names])
        default_false = samples @ counts(protocol.default)
        place, share, status = e.crossing(-false[:, ::-1], -default_false)
        recorded = published["results"]["matched"]["excess_detection"]
        for arm in CHANGE_ARMS:
            grid = samples @ np.column_stack([net(name, arm) for name in names])
            default = samples @ net(protocol.default, arm)
            values = e.read_at(grid[:, ::-1] / homes, place, share) - default / homes
            again = e.matched_estimate(values, status, protocol, homes)
            assert again["estimate"] == pytest.approx(recorded[arm]["estimate"])
            assert again["interval"] == recorded[arm]["interval"]
            assert again["resamples"] == recorded[arm]["resamples"]

    def test_the_page_and_its_figures_are_the_records_rendering(
        self, published: dict, described: dict
    ) -> None:
        import re

        from sensor_modeling.datasets.threshold_calibration_figures import (
            FIGURES,
            PREFIX,
            TIHM_FIGURES,
            data_sha256,
            figure_data,
            tihm_figure_data,
        )
        from sensor_modeling.datasets.threshold_calibration_summary import (
            RESULTS_PAGE,
            render_page,
        )

        page = (ROOT / "docs" / RESULTS_PAGE).read_text(encoding="utf-8")
        assert page == render_page(published, described)
        data = {**figure_data(published), **tihm_figure_data(described)}
        for name in (*FIGURES, *TIHM_FIGURES):
            svg = (ROOT / "docs" / "figures" / f"{PREFIX}-{name}.svg").read_text(
                encoding="utf-8"
            )
            found = re.search(r"data sha256 ([0-9a-f]{64})", svg)
            assert found is not None
            assert found.group(1) == data_sha256(data[name])


class TestPublishedDescription:
    """The record of the description on TIHM, as published. Not a test."""

    def test_it_was_made_from_a_clean_commit_on_the_published_runs(
        self, described: dict
    ) -> None:
        protocol = declared_protocol()
        configuration = described["configuration"]
        assert described["environment"]["git_dirty"] == "false"
        assert described["environment"]["git_commit"].startswith(RUN_COMMIT)
        assert described["data_source"] == "tihm"
        assert configuration["protocol_sha256"] == protocol.sha256()
        assert configuration["code_changed_since_the_freeze"] == {
            "sources": [],
            "distributions": [],
            "defaults": [],
        }
        results = described["results"]
        assert results["homes"] == 56
        check = results["check"]
        for rule, published in TIHM_PUBLISHED.items():
            assert check[rule]["monitored_days"] == published["monitored_days"]
            assert check[rule]["behavioural_alerts"] == published["behavioural_alerts"]
        assert check["homes_compared_one_by_one"] == 56
        assert check["homes_that_differ"] == []
        assert check["calibrated_runs"]["that_differ"] == []
        assert len(check["calibrated_runs"]["homes"]) == protocol.tihm_checked_homes
