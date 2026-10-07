"""Tests for the measurement of the baseline's thresholds on synthetic days."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from sensor_modeling.baseline import BaselineConfig, ChangeKind
from sensor_modeling.datasets.threshold_calibration_figures import (
    NULL_FIGURES,
    null_figure_data,
)
from sensor_modeling.datasets.threshold_null import (
    CALIBRATED,
    DEFAULT,
    KINDS,
    NONE,
    RAMP2,
    REFERENCES,
    RESULT_SCHEMA,
    SHAPES,
    STEP2,
    STEP3,
    WEEKEND,
    WEEKEND_STEP2,
    ThresholdNull,
    baseline_config,
    clustered_rate,
    judge,
    layout,
    matching_multiples,
    measure,
    nominal,
    paired,
    reference_deviations,
    run_threshold_null,
    share,
)
from sensor_modeling.datasets.threshold_null_summary import (
    dominated,
    render_page,
)
from sensor_modeling.evaluation import load_record

TINY = ThresholdNull(
    rate_series=6,
    series=4,
    scales=(0.5, 1.0),
    sizes=(4,),
    quantile_draws=1000,
)


class TestDeclaration:
    def test_the_first_day_is_a_monday(self) -> None:
        assert ThresholdNull().start.weekday() == 0
        with pytest.raises(ValueError, match="Monday"):
            ThresholdNull(start=date(2024, 3, 5))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"series": 1},
            {"rate_series": 1},
            {"step_day": 110},
            {"ramp_day": 60},
            {"scales": (0.5, 0.6)},
            {"scales": (1.0, -1.0)},
            {"scales": (1.0, 1.0, 0.5)},
            {"sizes": (1,)},
            {"quantile_draws": 10},
        ],
    )
    def test_settings_that_make_no_sense_are_refused(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            ThresholdNull(**kwargs)

    def test_it_meets_the_floor_for_a_simulation_study(self) -> None:
        declared = ThresholdNull()
        assert declared.series >= 100 and declared.rate_series >= 100
        assert len(declared.scales) == 13

    def test_what_each_shape_adds(self) -> None:
        declared = ThresholdNull()
        assert not declared.offsets(NONE).any()
        weekend = declared.offsets(WEEKEND)
        assert list(weekend[:7]) == [0, 0, 0, 0, 0, 1.5, 2.0]
        assert list(weekend[7:14]) == list(weekend[:7])
        step = declared.offsets(STEP2)
        assert step[55] == 0.0 and step[56] == -2.0 and step[-1] == -2.0
        assert declared.offsets(STEP3)[56] == -3.0
        ramp = declared.offsets(RAMP2)
        assert ramp[42] == 0.0 and ramp[56] == -1.0
        assert ramp[70] == -2.0 and ramp[-1] == -2.0
        both = declared.offsets(WEEKEND_STEP2)
        assert list(both[:7]) == [0, 0, 0, 0, 0, 1.5, 2.0]
        assert list(both[56:63]) == [-2, -2, -2, -2, -2, -0.5, 0.0]
        with pytest.raises(KeyError):
            declared.offsets("spike")

    def test_the_window_is_the_three_weeks_from_the_step(self) -> None:
        window = ThresholdNull().window()
        assert (window.start, window.stop) == (56, 77)

    def test_the_days_come_from_the_root_alone(self) -> None:
        declared = ThresholdNull()
        first = declared.noise(0, 3, 5)
        assert first.shape == (5, declared.days)
        assert np.array_equal(first, declared.noise(0, 3, 5))
        assert not np.array_equal(first, declared.noise(1, 3, 5))
        assert not np.array_equal(first, declared.noise(0, 4, 5))
        other = ThresholdNull(seed_root=declared.seed_root + 1)
        assert not np.array_equal(first, other.noise(0, 3, 5))

    def test_the_thresholds_are_multiplied_together(self) -> None:
        plain, scaled = baseline_config(DEFAULT), baseline_config(CALIBRATED, 0.6)
        assert plain == BaselineConfig()
        assert scaled.calibrated
        assert scaled.deviation_threshold == pytest.approx(1.8)
        assert scaled.trend_threshold == pytest.approx(2.1)
        with pytest.raises(KeyError):
            baseline_config("robust")

    def test_the_declaration_serialises(self) -> None:
        payload = ThresholdNull().to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert set(payload["shapes"]) == set(SHAPES)
        assert set(payload["references"]) == set(REFERENCES)


class TestJudging:
    def test_a_day_without_a_verdict_has_no_deviation(self) -> None:
        declared = ThresholdNull()
        values = 8.0 + declared.noise(0, 0, 1)[0]
        deviations, kinds = judge(values, BaselineConfig(), declared.start)
        insufficient = KINDS.index(ChangeKind.INSUFFICIENT_DATA)
        assert np.isnan(deviations[:14]).all() and not np.isnan(deviations[14:]).any()
        assert (kinds[:14] == insufficient).all() and (kinds[14:] != insufficient).all()
        assert (deviations[14:] >= 0).all()

    def test_the_reference_of_a_day_depends_on_the_day_alone(self) -> None:
        places = layout(ThresholdNull())
        assert places[14] == (False, 14)
        assert places[27] == (False, 27)
        assert places[28] == (True, 4)
        assert places[34] == (True, 4)
        assert places[35] == (True, 5)
        assert places[119] == (True, 17)


class TestStatistics:
    def test_what_a_threshold_states(self) -> None:
        assert nominal(3.0) == pytest.approx(0.0026998, rel=1e-4)
        assert nominal(1.96) == pytest.approx(0.05, rel=1e-3)

    def test_a_share_of_days_has_the_series_as_its_unit(self) -> None:
        hits, counts = np.array([1, 0, 3]), np.array([10, 10, 10])
        found = clustered_rate(hits, counts)
        assert found["rate"] == pytest.approx(4 / 30)
        assert found["days"] == 30
        # With equal counts it is the standard error of the mean of the
        # series' own shares.
        shares = hits / counts
        assert found["se"] == pytest.approx(shares.std(ddof=1) / math.sqrt(3))

    def test_days_that_share_a_series_do_not_count_as_independent(self) -> None:
        together = clustered_rate(np.array([10, 0, 0, 0]), np.array([10, 10, 10, 10]))
        binomial = math.sqrt(0.25 * 0.75 / 40)
        assert together["se"] > 2 * binomial

    def test_a_share_with_nothing_behind_it_is_not_given(self) -> None:
        assert clustered_rate(np.array([0]), np.array([0]))["rate"] is None
        assert clustered_rate(np.array([1]), np.array([4]))["se"] is None

    def test_a_share_of_series(self) -> None:
        found = share(np.array([True, False, False, True, True]))
        assert found["share"] == pytest.approx(0.6)
        assert found["count"] == 3 and found["series"] == 5
        assert found["se"] == pytest.approx(math.sqrt(0.6 * 0.4 / 5))

    def test_a_paired_difference(self) -> None:
        first, second = np.array([1, 1, 0, 1]), np.array([0, 1, 0, 0])
        found = paired(first, second)
        assert found["estimate"] == pytest.approx(0.5)
        assert found["se"] == pytest.approx(np.std([1, 0, 0, 1], ddof=1) / 2.0)

    def test_four_days_put_a_gaussian_day_past_three_one_time_in_six(self) -> None:
        deviations = reference_deviations(4, 200_000, np.random.default_rng(1))
        assert deviations.shape == (200_000,) and (deviations >= 0).all()
        assert float((deviations >= 3.0).mean()) == pytest.approx(0.175, abs=0.01)
        with pytest.raises(ValueError):
            reference_deviations(1, 10, np.random.default_rng(1))


@pytest.fixture(scope="module")
def tiny() -> dict:
    return measure(TINY)


class TestMeasure:
    def test_every_part_is_there(self, tiny: dict) -> None:
        assert tiny["result_schema"] == RESULT_SCHEMA
        assert set(tiny["rates"]) == {NONE, WEEKEND}
        assert set(tiny["operating"]) == set(SHAPES)
        for shape in SHAPES:
            for reference in REFERENCES:
                assert set(tiny["operating"][shape][reference]) == {"0.5", "1"}
        assert set(tiny["matching"]) == {"plain", "weekly_rhythm"}
        for entry in tiny["matching"].values():
            assert {"1"} <= set(entry["calibrated_minus_default"]) <= {"1", "0.5"}
        assert set(tiny["needed_threshold"]["by_size"]) == {"4"}

    def test_the_days_are_all_counted(self, tiny: dict) -> None:
        for reference in REFERENCES:
            cell = tiny["rates"][NONE][reference]
            assert cell["evaluable_days"] == 6 * (120 - 14)
            assert cell["phase"]["all"]["3"]["days"] == 6 * 106
            assert cell["phase"]["pooled"]["3"]["days"] == 6 * 14
            assert cell["phase"]["weekday_aware"]["3"]["days"] == 6 * 92
            assert cell["by_weekday_size"]["4"]["3"]["days"] == 6 * 7
            assert cell["by_weekday_size"]["17"]["3"]["days"] == 6 * 1
            assert sum(v["3"]["days"] for v in cell["by_week"].values()) == 6 * 106
            verdicts = sum(v["rate"] for v in cell["verdicts"].values())
            assert verdicts == pytest.approx(1.0)

    def test_a_lower_threshold_is_passed_no_less_often(self, tiny: dict) -> None:
        for reference in REFERENCES:
            cell = tiny["rates"][NONE][reference]["phase"]["all"]
            rates = [cell[f"{t:g}"]["rate"] for t in TINY.rate_thresholds]
            assert rates == sorted(rates, reverse=True)

    def test_the_comparison_is_the_difference_of_the_two_shares(
        self, tiny: dict
    ) -> None:
        for setting in tiny["matching"].values():
            for scale, shapes in setting["calibrated_minus_default"].items():
                assert set(shapes) == set(SHAPES)
                for shape, entry in shapes.items():
                    mine = tiny["operating"][shape][CALIBRATED][scale]["found"]
                    other = tiny["operating"][shape][DEFAULT]["1"]["found"]
                    assert entry["estimate"] == pytest.approx(
                        mine["share"] - other["share"]
                    )

    def test_the_result_does_not_depend_on_how_many_run_at_once(
        self, tiny: dict
    ) -> None:
        spread = ThresholdNull(
            rate_series=150,
            series=4,
            scales=(1.0,),
            sizes=(4,),
            quantile_draws=1000,
        )
        stages: list[str] = []
        alone = measure(spread, progress=lambda stage, *_: stages.append(stage))
        assert measure(spread, jobs=2) == alone
        assert stages == ["rates", "rates", "operating"]

    def test_the_record_says_the_values_are_synthetic(self, tmp_path: Path) -> None:
        result = run_threshold_null(TINY, output_dir=tmp_path)
        assert result.path == tmp_path / "threshold-null.json"
        payload = load_record(result.path)
        assert payload["data_source"] == "synthetic"
        assert payload["seeds"] == [TINY.seed_root]
        assert any("synthetic" in note for note in payload["notes"])
        assert payload["configuration"]["result_schema"] == RESULT_SCHEMA
        assert payload["mcse"]
        page = render_page(payload)
        assert page.startswith("# The baseline's thresholds")
        assert set(null_figure_data(payload)) == set(NULL_FIGURES)


def operating(points: dict[str, dict[str, tuple[float, float]]]) -> dict:
    """A table of operating points from (false reports, found) for each scale."""
    return {
        shape: {
            reference: {
                scale: {"found": {"share": pair[place]}}
                for scale, pair in scales.items()
            }
            for reference, scales in points.items()
        }
        for place, shape in ((0, NONE), (1, STEP2), (0, WEEKEND), (1, WEEKEND_STEP2))
    }


class TestMatching:
    FALSE = {0.4: 0.30, 0.5: 0.12, 0.6: 0.04, 1.0: 0.00}
    FOUND = {0.4: 1.00, 0.5: 0.98, 0.6: 0.85, 1.0: 0.10}

    def test_the_strictest_multiple_that_finds_as_much(self) -> None:
        found = matching_multiples(self.FALSE, self.FOUND, 0.10, 0.85)
        assert found["same_detection"] == 0.6
        assert found["same_detection_was_reached"] is True
        stricter = matching_multiples(self.FALSE, self.FOUND, 0.10, 0.86)
        assert stricter["same_detection"] == 0.5

    def test_the_most_sensitive_multiple_that_reports_falsely_no_more(self) -> None:
        found = matching_multiples(self.FALSE, self.FOUND, 0.12, 0.85)
        assert found["same_false_reports"] == 0.5
        assert found["same_false_reports_was_reached"] is True
        quieter = matching_multiples(self.FALSE, self.FOUND, 0.11, 0.85)
        assert quieter["same_false_reports"] == 0.6

    def test_a_point_that_cannot_be_matched_is_said_to_be(self) -> None:
        found = matching_multiples(self.FALSE, self.FOUND, -0.01, 1.01)
        assert found == {
            "same_detection": 0.4,
            "same_detection_was_reached": False,
            "same_false_reports": 1.0,
            "same_false_reports_was_reached": False,
        }

    def test_each_setting_is_matched_in_its_own_shapes(self, tiny: dict) -> None:
        plain, rhythm = tiny["matching"]["plain"], tiny["matching"]["weekly_rhythm"]
        assert (plain["nothing_changed"], plain["a_step"]) == (NONE, STEP2)
        assert (rhythm["nothing_changed"], rhythm["a_step"]) == (
            WEEKEND,
            WEEKEND_STEP2,
        )
        for entry in (plain, rhythm):
            assert entry["same_detection"] in TINY.scales
            assert entry["same_false_reports"] in TINY.scales


class TestDominated:
    def test_a_point_is_matched_by_one_no_worse_on_both_counts(self) -> None:
        table = operating(
            {
                DEFAULT: {"1": (0.10, 0.80), "2": (0.01, 0.30)},
                CALIBRATED: {"0.6": (0.02, 0.85), "1": (0.00, 0.10)},
            }
        )
        assert dominated(table, STEP2) == ["1"]

    def test_fewer_false_reports_alone_do_not_match_a_point(self) -> None:
        table = operating(
            {DEFAULT: {"1": (0.10, 0.80)}, CALIBRATED: {"1": (0.00, 0.79)}}
        )
        assert dominated(table, STEP2) == []

    def test_a_step_with_a_rhythm_is_set_against_the_rhythm_alone(self) -> None:
        table = operating(
            {DEFAULT: {"1": (0.10, 0.80)}, CALIBRATED: {"0.6": (0.05, 0.90)}}
        )
        table[WEEKEND][CALIBRATED]["0.6"]["found"]["share"] = 0.50
        assert dominated(table, STEP2) == ["1"]
        assert dominated(table, WEEKEND_STEP2) == []
