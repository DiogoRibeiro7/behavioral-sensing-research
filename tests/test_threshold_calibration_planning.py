"""Tests for the trials the threshold-calibration protocol was planned on.

The outcomes are drawn from curves written down by hand. No home is simulated
and no reference is run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sensor_modeling.datasets import threshold_calibration_planning as planning
from sensor_modeling.datasets.threshold_calibration_planning import (
    BETTER,
    NOT_BRACKETED,
    NOT_SHOWN,
    OUTCOMES,
    SAME_DETECTION,
    SAME_FALSE_ALERTS,
    SHAPES,
    WORSE,
    Cell,
    Planning,
    curves,
    draw,
    plan,
    run_cell,
    run_planning,
    summary,
    trial,
)
from sensor_modeling.datasets.threshold_calibration_protocol import (
    BRACKETED,
    ThresholdCalibrationProtocol,
)


def small(**changes: object) -> Planning:
    """Few trials of small cohorts, for tests."""
    settings: dict[str, object] = {
        "replications": 100,
        "homes": 60,
        "resamples": 80,
        "ratios": (1.0, 0.25),
        "offsets": (0.6,),
    }
    settings.update(changes)
    return Planning(**settings)  # type: ignore[arg-type]


class TestCurves:
    def test_the_default_has_the_silent_home_records_sizes_where_it_ships(
        self,
    ) -> None:
        false, found, falsely = curves("most", np.array([1.0]))
        assert false[0] == pytest.approx(1.0)
        assert found[0] == pytest.approx(0.75)
        assert falsely[0] == pytest.approx(0.12)
        assert curves("nearly_all", np.array([1.0]))[1][0] == pytest.approx(0.95)
        assert curves("few", np.array([1.0]))[1][0] == pytest.approx(0.42)

    def test_every_curve_falls_as_the_thresholds_rise(self) -> None:
        multiples = np.linspace(0.3, 3.5, 60)
        for shape in SHAPES:
            for values in curves(shape, multiples):
                assert (np.diff(values) <= 1e-12).all()

    def test_false_alerts_are_a_straight_line_in_their_logarithm(self) -> None:
        false = curves("most", np.array([0.4, 0.45, 0.5]))[0]
        assert false[1] == pytest.approx(np.sqrt(16.0 * 10.0))


class TestDraw:
    def test_a_cohort_is_whole_numbers_that_fall_with_the_thresholds(self) -> None:
        config = small()
        cohort = draw(np.random.default_rng(1), Cell("most", 1.0, 0.6), config)
        assert set(cohort) == {"default", "calibrated"}
        for counts, net in cohort.values():
            assert counts.shape == net.shape == (config.homes, len(config.scales))
            assert counts.dtype == net.dtype == np.int64
            assert (np.diff(counts, axis=1) <= 0).all()
            assert set(np.unique(net)) <= {-1, 0, 1}

    def test_the_ratio_multiplies_the_calibrated_references_false_alerts(
        self,
    ) -> None:
        config = small(homes=4000)
        place = config.scales.index(0.6)
        means = {}
        for ratio in (1.0, 0.25):
            cohort = draw(np.random.default_rng(2), Cell("most", ratio, 0.6), config)
            means[ratio] = cohort["calibrated"][0][:, place].mean()
            # The calibrated reference at 0.6 is the default at 0.6 / 0.6.
            default = cohort["default"][0][:, config.scales.index(1.0)].mean()
            assert default == pytest.approx(1.0, abs=0.08)
        assert means[1.0] == pytest.approx(1.0, abs=0.08)
        assert means[0.25] == pytest.approx(0.25, abs=0.04)

    def test_what_is_found_follows_the_curve_and_the_offset(self) -> None:
        config = small(homes=4000)
        cohort = draw(np.random.default_rng(3), Cell("most", 1.0, 0.6), config)
        declared, moved = config.scales.index(1.0), config.scales.index(0.6)
        # Found in 75 of 100 less 12 of 100 falsely.
        assert cohort["default"][1][:, declared].mean() == pytest.approx(0.63, abs=0.04)
        assert cohort["calibrated"][1][:, moved].mean() == pytest.approx(0.63, abs=0.04)

    def test_a_cohort_follows_from_its_generator_alone(self) -> None:
        config, cell = small(), Cell("few", 0.75, 0.6)
        one = draw(np.random.default_rng(4), cell, config)
        two = draw(np.random.default_rng(4), cell, config)
        for reference in one:
            assert np.array_equal(one[reference][0], two[reference][0])
            assert np.array_equal(one[reference][1], two[reference][1])


class TestTrial:
    def test_both_matches_are_read_on_one_cohort(self) -> None:
        config = small()
        found = trial(np.random.default_rng(5), Cell("most", 1.0, 0.6), config)
        assert set(found) == {SAME_FALSE_ALERTS, SAME_DETECTION}
        quiet = found[SAME_FALSE_ALERTS]
        assert quiet["verdict"] in OUTCOMES
        assert quiet["homes"] == config.homes
        assert quiet["resamples"]["all"] == config.resamples
        assert quiet["interval"]["confidence"] == config.confidence

    @pytest.mark.parametrize(
        ("low", "high", "more", "verdict"),
        [
            (0.01, 0.2, True, BETTER),
            (0.01, 0.2, False, WORSE),
            (-0.2, -0.01, True, WORSE),
            (-0.2, -0.01, False, BETTER),
            (-0.1, 0.1, True, NOT_SHOWN),
            (0.0, 0.1, True, NOT_SHOWN),
            (None, None, True, NOT_SHOWN),
            (None, -0.01, True, WORSE),
        ],
    )
    def test_a_verdict_is_read_off_the_interval(
        self, low: float | None, high: float | None, more: bool, verdict: str
    ) -> None:
        estimate = {"status": BRACKETED, "interval": {"low": low, "high": high}}
        assert planning._verdict(estimate, more_is_better=more) == verdict
        estimate["status"] = "not_reached"
        assert planning._verdict(estimate, more_is_better=more) == NOT_BRACKETED

    def test_a_reference_with_a_quarter_of_the_false_alerts_is_called_better(
        self,
    ) -> None:
        config = small(homes=400, resamples=200)
        rng = np.random.default_rng(6)
        cell = Cell("most", 0.25, 0.6)
        verdicts = [
            trial(rng, cell, config)[SAME_FALSE_ALERTS]["verdict"] for _ in range(20)
        ]
        assert verdicts.count(BETTER) == 20


class TestPlan:
    def test_the_cells_are_every_default_at_every_truth(self) -> None:
        config = Planning()
        cells = config.cells()
        assert len(cells) == len({cell.name for cell in cells}) == 3 * 4 * 5 + 4
        assert {cell.shape for cell in cells} == set(SHAPES)
        assert [cell for cell in cells if not cell.usual] == [
            Cell("most", 1.0, 0.6, correlation=0.2),
            Cell("most", 1.0, 0.6, dispersion=0.6),
            Cell("most", 1.0, 0.6, alerts=0.25),
            Cell("most", 1.0, 0.6, alerts=4.0),
        ]
        # With the same curve the match falls near the offset, and the
        # offsets cover the grid from below 0.5 to above the declared
        # thresholds.
        assert min(config.offsets) < 0.5 and max(config.offsets) > 1.0
        assert config.to_dict()["cells"] == [cell.name for cell in cells]
        assert json.loads(json.dumps(config.to_dict())) == config.to_dict()

    def test_the_cohort_is_the_protocols(self) -> None:
        config, protocol = Planning(), ThresholdCalibrationProtocol()
        assert config.homes == protocol.homes
        assert config.scales == protocol.curve_scales
        assert config.confidence == protocol.confidence

    @pytest.mark.parametrize(
        "changes", [{"replications": 50}, {"ratios": (0.5,)}, {"homes": 1}]
    )
    def test_settings_that_make_no_sense_are_refused(self, changes: dict) -> None:
        with pytest.raises(ValueError):
            Planning(**changes)

    def test_the_result_does_not_depend_on_how_many_run_at_once(self) -> None:
        config = small(homes=40, resamples=40)
        alone = plan(config)
        assert plan(config, jobs=2) == alone
        assert [cell["cell"] for cell in alone["cells"]] == [
            cell.name for cell in config.cells()
        ]

    def test_a_cell_counts_what_its_trials_concluded(self) -> None:
        config = small()
        cell = config.cells()[0]
        seed = np.random.SeedSequence(config.seed_root).spawn(len(config.cells()))[0]
        found = run_cell((cell, config, seed, None))
        for match in (SAME_FALSE_ALERTS, SAME_DETECTION):
            shares = [found[match][outcome]["share"] for outcome in OUTCOMES]
            assert sum(shares) == pytest.approx(1.0)
            assert found[match]["trials_with_an_estimate"] == round(
                config.replications * (1.0 - found[match][NOT_BRACKETED]["share"])
            )
        worse = found[SAME_FALSE_ALERTS][WORSE]
        assert worse["se"] == pytest.approx(
            np.sqrt(worse["share"] * (1 - worse["share"]) / config.replications)
        )

    def test_the_summary_is_the_range_over_the_cells_of_a_truth(self) -> None:
        config = small(homes=40, resamples=40)
        cells = plan(config)["cells"]
        found = summary(cells, config)
        same = [cell for cell in cells if cell["ratio"] == 1.0]
        quiet = found[SAME_FALSE_ALERTS]
        assert quiet["1"]["cells"] == len(same) == 3 + 4
        better = [cell[SAME_FALSE_ALERTS][BETTER]["share"] for cell in same]
        assert quiet["1"][BETTER] == {"low": min(better), "high": max(better)}
        errors = [cell[SAME_FALSE_ALERTS]["sd_of_estimates"] for cell in same]
        assert quiet["1"]["sd_of_estimates"]["high"] == max(errors)
        means = [cell[SAME_FALSE_ALERTS]["mean_estimate"] for cell in same]
        assert quiet["1"]["mean_estimate"]["low"] == min(means)
        # By default, over the main cells alone.
        assert set(quiet["by_shape"]) == set(SHAPES)
        most = [
            cell
            for cell in same
            if cell["shape"] == "most"
            and (cell["correlation"], cell["alerts"]) == (0.6, 1.0)
            and cell["dispersion"] == 1.9
        ]
        assert quiet["by_shape"]["most"]["1"]["cells"] == len(most) == 1
        assert set(found[SAME_DETECTION]["by_shape"]) == set(SHAPES)
        one = [
            cell
            for cell in cells
            if (cell["shape"], cell["ratio"]) == ("nearly_all", 1.0)
        ]
        assert found[SAME_DETECTION]["by_shape"]["nearly_all"]["1"][BETTER][
            "high"
        ] == max(cell[SAME_DETECTION][BETTER]["share"] for cell in one)

    def test_a_kept_cell_is_read_back_and_not_run_again(self, tmp_path) -> None:
        config = small(homes=40, resamples=40)
        first = plan(config, checkpoint=tmp_path)
        kept = sorted(tmp_path.glob("cell-*.json"))
        assert len(kept) == len(config.cells())
        assert plan(config, checkpoint=tmp_path) == first == plan(config)
        # A kept cell is named by the settings: other settings run again.
        other = small(homes=40, resamples=40, seed_root=7)
        plan(other, checkpoint=tmp_path)
        assert len(sorted(tmp_path.glob("cell-*.json"))) == 2 * len(kept)

    def test_the_scoring_functions_it_decides_with_are_recorded(self) -> None:
        sources = Planning().to_dict()["scoring_functions"]
        assert set(sources) == {"crossing", "read_at", "matched_estimate"}
        assert all(len(digest) == 64 for digest in sources.values())
        assert sources == planning.scoring_sources()

    def test_the_record_says_that_nothing_was_simulated(self) -> None:
        record = run_planning(small(homes=40, resamples=40)).record
        payload = record.to_dict()
        assert payload["experiment"] == "threshold-calibration-planning"
        assert payload["data_source"] == "synthetic"
        assert "No home is simulated" in payload["notes"][0]
        assert payload["seeds"] == [20261011]
        assert payload["results"]["result_schema"] == planning.RESULT_SCHEMA
        assert len(payload["mcse"]) == len(payload["results"]["cells"]) * 4
        record.to_json()


class TestWhatTheDrawsAssume:
    def test_homes_differ_in_their_false_alerts_as_the_record_s_homes_do(
        self,
    ) -> None:
        config = small(homes=4000)
        cohort = draw(np.random.default_rng(8), Cell("most", 1.0, 0.6), config)
        counts = cohort["default"][0][:, config.scales.index(1.0)]
        # A gamma rate of shape 1.9 under a Poisson count of mean one.
        assert counts.std(ddof=1) == pytest.approx(np.sqrt(1 + 1 / 1.9), abs=0.06)

    def test_a_home_found_under_one_reference_tends_to_be_under_the_other(
        self,
    ) -> None:
        config = small(homes=4000)
        cohort = draw(np.random.default_rng(9), Cell("most", 1.0, 0.6), config)
        default = cohort["default"][1][:, config.scales.index(1.0)]
        calibrated = cohort["calibrated"][1][:, config.scales.index(0.6)]
        assert np.corrcoef(default, calibrated)[0, 1] > 0.2
        # A false detection is drawn apart from a detection, so a home can
        # be falsely detected and not found.
        assert (default == -1).any()

    def test_the_other_match_says_better_with_fewer_false_alerts(self) -> None:
        config = small(homes=400, resamples=200)
        rng = np.random.default_rng(10)
        found = [trial(rng, Cell("most", 0.25, 0.6), config) for _ in range(5)]
        for one in found:
            assert one[SAME_DETECTION]["estimate"] < 0.0
            assert one[SAME_DETECTION]["verdict"] == BETTER

    def test_a_match_at_the_end_of_the_grid_is_not_bracketed(self) -> None:
        estimate = {"status": "at_the_end_of_the_grid", "interval": {}}
        assert planning._verdict(estimate, more_is_better=True) == NOT_BRACKETED

    def test_a_cell_s_summary_is_computed_from_its_trials(self) -> None:
        trials = [
            {
                "verdict": verdict,
                "estimate": estimate,
                "mcse": 0.1,
                "resamples": {"bracketed": bracketed},
            }
            for verdict, estimate, bracketed in (
                (BETTER, 0.3, 10),
                (NOT_SHOWN, -0.1, 10),
                (NOT_BRACKETED, None, 4),
                (WORSE, -0.5, 8),
            )
        ]
        found = planning._summarise(trials, resamples=10)
        assert found[BETTER]["share"] == 0.25
        assert found[NOT_BRACKETED]["share"] == 0.25
        assert found["trials_with_an_estimate"] == 3
        assert found["mean_estimate"] == pytest.approx(-0.1)
        assert found["sd_of_estimates"] == pytest.approx(
            np.std([0.3, -0.1, -0.5], ddof=1)
        )
        assert found["resamples_bracketed"] == pytest.approx(0.8)

    def test_the_other_match_is_summarised_over_the_main_cells(self) -> None:
        config = small(homes=40, resamples=40)
        cells = plan(config)["cells"]
        found = summary(cells, config)[SAME_DETECTION]["by_shape"]["most"]["1"]
        assert found["cells"] == 1
        main = next(
            cell
            for cell in cells
            if (cell["shape"], cell["ratio"]) == ("most", 1.0)
            and cell["cell"].endswith("correlation=0.6/dispersion=1.9/alerts=1")
        )
        share = main[SAME_DETECTION][NOT_BRACKETED]["share"]
        assert found[NOT_BRACKETED] == {"low": share, "high": share}
