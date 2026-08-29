"""Tests for the naive versus occupancy-aware attribution comparison."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from sensor_modeling.evaluation import (
    Scenario,
    compare_scenario,
    run_attribution_study,
    standard_scenarios,
)
from sensor_modeling.online import BehaviouralSensingPipeline, PipelineConfig
from sensor_modeling.simulation import HouseholdConfig, simulate

STEP = timedelta(minutes=30)


def alone(days: int = 4, seed: int = 31) -> Scenario:
    """A household nobody else ever enters."""
    return Scenario(
        "alone",
        HouseholdConfig(
            days=days,
            seed=seed,
            visitor_probability=0.0,
            carer_weekday_visits=False,
        ),
        description="nobody else present",
    )


def with_carer(days: int = 4, seed: int = 31) -> Scenario:
    """A household with a regular weekday carer round."""
    return Scenario(
        "carer",
        HouseholdConfig(
            days=days,
            seed=seed,
            visitor_probability=0.0,
            carer_weekday_visits=True,
        ),
        description="weekday carer",
    )


class TestNaiveMode:
    def test_disabling_attribution_credits_every_event_to_the_resident(self) -> None:
        """The naive arm must actually be naive, or the comparison is empty."""
        result = simulate(HouseholdConfig(days=3, seed=9))
        pipeline = BehaviouralSensingPipeline(
            result.registry,
            config=PipelineConfig(
                tz=result.config.tz, step=STEP, attribute_activity=False
            ),
        )
        steps = pipeline.run(result.observations)
        assert steps
        shares = {
            contribution.attribution
            for step in steps
            for contribution in step.state.evidence
        }
        assert shares == {1.0}

    def test_the_default_discounts_unattributable_evidence(self) -> None:
        result = simulate(HouseholdConfig(days=3, seed=9, carer_weekday_visits=True))
        pipeline = BehaviouralSensingPipeline(
            result.registry, config=PipelineConfig(tz=result.config.tz, step=STEP)
        )
        steps = pipeline.run(result.observations)
        shares = {
            contribution.attribution
            for step in steps
            for contribution in step.state.evidence
        }
        assert any(share < 1.0 for share in shares)


class TestComparison:
    def test_attribution_changes_no_conclusion_when_nobody_else_is_present(
        self,
    ) -> None:
        """The property that shows the mechanism is not just adding noise.

        The conclusions are identical. The probabilities differ by order
        1e-4, because the occupancy model never becomes *certain* the
        resident is alone and that residual uncertainty propagates. That is
        the honest behaviour of a probabilistic layer, not a defect: a model
        that reached certainty here would be overconfident.
        """
        comparison = compare_scenario(alone(), step=STEP)
        assert comparison.contaminated_fraction == 0.0
        assert comparison.balanced_accuracy_gain == pytest.approx(0.0, abs=1e-9)
        assert abs(comparison.calibration_gain) < 1e-3
        assert comparison.occupancy_aware.mean_attribution > 0.9

    def test_a_contaminated_scenario_is_recognised_as_such(self) -> None:
        comparison = compare_scenario(with_carer(), step=STEP)
        assert comparison.contaminated_fraction > 0.0

    def test_both_arms_see_the_same_trajectory(self) -> None:
        """Pairing is what makes the difference attributable to attribution."""
        comparison = compare_scenario(with_carer(), step=STEP)
        assert comparison.naive.states.n == comparison.occupancy_aware.states.n

    def test_the_naive_arm_reports_full_attribution(self) -> None:
        comparison = compare_scenario(with_carer(), step=STEP)
        assert comparison.naive.mean_attribution == pytest.approx(1.0)

    def test_a_comparison_serialises(self) -> None:
        payload = compare_scenario(alone(), step=STEP).to_dict()
        assert payload["scenario"] == "alone"
        assert "naive" in payload and "occupancy_aware" in payload
        assert "visitor_detection" in payload

    def test_results_are_reproducible_from_the_seed(self) -> None:
        first = compare_scenario(with_carer(), step=STEP).to_dict()
        second = compare_scenario(with_carer(), step=STEP).to_dict()
        assert first == second


class TestScenarios:
    def test_the_standard_set_covers_the_named_situations(self) -> None:
        names = {s.name for s in standard_scenarios()}
        assert names == {
            "resident_alone",
            "resident_goes_out",
            "short_visitor",
            "prolonged_visitor",
            "carer_visits",
            "visitor_and_carer",
            "resident_without_wearable",
            "no_radar",
            "sparse_coverage",
        }

    def test_every_scenario_documents_what_it_isolates(self) -> None:
        assert all(s.description for s in standard_scenarios())

    def test_a_degraded_scenario_withholds_records(self) -> None:
        scenario = next(
            s for s in standard_scenarios(days=3) if s.name == "sparse_coverage"
        )
        result, observations = scenario.build()
        assert len(observations) < len(result.observations)

    def test_a_study_reports_the_contaminated_subset(self) -> None:
        study = run_attribution_study([alone(), with_carer()], step=STEP)
        assert [c.scenario for c in study.contaminated] == ["carer"]
        assert "mean_gain_when_contaminated" in study.to_dict()

    def test_looking_up_a_missing_scenario_raises(self) -> None:
        study = run_attribution_study([alone()], step=STEP)
        with pytest.raises(KeyError, match="no scenario named"):
            study.by_name("nonexistent")

    def test_an_empty_study_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one scenario"):
            run_attribution_study([])

    def test_scenarios_can_be_reconfigured(self) -> None:
        scenario = alone()
        longer = replace(scenario, household=replace(scenario.household, days=5))
        assert longer.household.days == 5
