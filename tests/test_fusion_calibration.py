"""Calibration and scenario tests for state inference.

Calibration is the property that makes a probabilistic system usable for
longitudinal monitoring: a stated confidence of 0.9 has to mean something, or
downstream thresholds are arbitrary. These tests check it against the
simulator's ground truth, and check the scenarios a deployment actually
encounters -- ambiguity, contradiction, failure, silence and transition.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from sensor_modeling.evaluation import state_metrics
from sensor_modeling.fusion import Explanation
from sensor_modeling.online import BehaviouralSensingPipeline, PipelineConfig
from sensor_modeling.simulation import (
    DegradationConfig,
    HouseholdConfig,
    degrade,
    dropout,
    not_worn,
    simulate,
)
from sensor_modeling.states import BehaviouralState as S

STEP = timedelta(minutes=15)


@pytest.fixture(scope="module")
def household() -> object:
    """A ten-day synthetic household shared across the module."""
    return simulate(HouseholdConfig(days=10, seed=606))


def run(result: object, observations: object = None) -> list:
    """Drive a record through a fresh pipeline."""
    pipeline = BehaviouralSensingPipeline(
        result.registry,  # type: ignore[attr-defined]
        config=PipelineConfig(tz=result.config.tz, step=STEP),  # type: ignore[attr-defined]
    )
    records = observations if observations is not None else result.observations  # type: ignore[attr-defined]
    steps = pipeline.run(records)  # type: ignore[arg-type]
    steps.extend(pipeline.close(result.end))  # type: ignore[attr-defined]
    return steps


def truth_for(result: object, steps: list) -> list:
    """Ground-truth states aligned with a run of steps."""
    return result.truth.states_at([step.at for step in steps])  # type: ignore[attr-defined]


class TestCalibration:
    def test_stated_confidence_tracks_observed_accuracy(
        self, household: object
    ) -> None:
        """The defining property of a calibrated probabilistic system.

        Compared across confidence quantiles rather than fixed bands, since
        the filter is confident most of the time and a fixed low band can be
        empty for a given seed.
        """
        steps = run(household)
        truth = truth_for(household, steps)
        pairs = sorted(
            (
                (step.state.confidence, actual is step.state.most_likely)
                for actual, step in zip(truth, steps)
                if actual is not None
            ),
            key=lambda pair: pair[0],
        )
        assert len(pairs) > 100

        third = len(pairs) // 3
        least_confident = pairs[:third]
        most_confident = pairs[-third:]
        low_accuracy = sum(1 for _, hit in least_confident if hit) / len(
            least_confident
        )
        high_accuracy = sum(1 for _, hit in most_confident if hit) / len(most_confident)

        assert high_accuracy > low_accuracy
        assert high_accuracy > 0.75

    def test_expected_calibration_error_is_bounded(self, household: object) -> None:
        steps = run(household)
        metrics = state_metrics(truth_for(household, steps), [s.state for s in steps])
        assert metrics.calibration_error < 0.2

    def test_losing_sensors_does_not_make_the_model_overconfident(
        self, household: object
    ) -> None:
        """The dangerous failure is confidence staying high as accuracy falls."""
        clean = state_metrics(
            truth_for(household, run(household)),
            [s.state for s in run(household)],
        )
        degraded_records, _ = degrade(
            household.observations,  # type: ignore[attr-defined]
            DegradationConfig(missing_rate=0.4, seed=5),
        )
        degraded_steps = run(household, degraded_records)
        degraded = state_metrics(
            truth_for(household, degraded_steps),
            [s.state for s in degraded_steps],
        )
        assert degraded.balanced_accuracy <= clean.balanced_accuracy
        # Calibration is allowed to worsen, but not without bound.
        assert degraded.calibration_error < clean.calibration_error + 0.15

    def test_probabilities_are_proper_across_a_whole_run(
        self, household: object
    ) -> None:
        for step in run(household):
            belief = step.state.belief
            assert np.all(np.isfinite(belief))
            assert belief.min() >= 0.0
            assert np.isclose(belief.sum(), 1.0)


class TestScenarios:
    def test_an_unambiguous_night_is_inferred_confidently(
        self, household: object
    ) -> None:
        steps = run(household)
        truth = truth_for(household, steps)
        asleep = [step for actual, step in zip(truth, steps) if actual is S.SLEEPING]
        assert asleep
        mean_confidence = float(np.mean([s.state.confidence for s in asleep]))
        assert mean_confidence > 0.6

    def test_a_single_sensor_failure_is_absorbed(self, household: object) -> None:
        faults = (
            dropout(
                "bed_pressure",
                household.start + timedelta(days=2),  # type: ignore[attr-defined]
                timedelta(days=2),
            ),
        )
        records, _ = degrade(
            household.observations, DegradationConfig(faults=faults)  # type: ignore[attr-defined]
        )
        metrics = state_metrics(
            truth_for(household, run(household, records)),
            [s.state for s in run(household, records)],
        )
        assert metrics.balanced_accuracy > 0.55

    def test_multiple_simultaneous_failures_degrade_without_collapsing(
        self, household: object
    ) -> None:
        start = household.start + timedelta(days=2)  # type: ignore[attr-defined]
        faults = (
            dropout("bed_pressure", start, timedelta(days=3)),
            dropout("living_radar", start, timedelta(days=3)),
            *not_worn(["wearable_motion", "resident_beacon"], start, timedelta(days=3)),
        )
        records, _ = degrade(
            household.observations, DegradationConfig(faults=faults)  # type: ignore[attr-defined]
        )
        steps = run(household, records)
        during = [
            step for step in steps if start <= step.at < start + timedelta(days=3)
        ]
        assert during
        # Ambient sensors alone cannot resolve much, so the model should be
        # visibly less certain rather than confidently wrong.
        assert float(np.mean([s.state.completeness for s in during])) < 0.75

    def test_no_observations_at_all_yields_no_steps(self, household: object) -> None:
        pipeline = BehaviouralSensingPipeline(household.registry)  # type: ignore[attr-defined]
        assert pipeline.run([]) == []

    def test_state_transitions_are_followed(self, household: object) -> None:
        """The filter must move, not settle on one state for the whole run."""
        steps = run(household)
        reported = [step.state.state for step in steps]
        changes = sum(1 for a, b in zip(reported, reported[1:]) if a is not b)
        assert changes > 20
        assert len(set(reported)) >= 4


class TestStructuredExplanation:
    def test_every_estimate_can_explain_itself(self, household: object) -> None:
        for step in run(household)[:50]:
            explanation = step.state.explanation()
            assert isinstance(explanation, Explanation)
            assert explanation.state is step.state.state
            assert 0.0 <= explanation.probability <= 1.0

    def test_the_rendered_block_names_every_evidence_category(
        self, household: object
    ) -> None:
        rendered = run(household)[len(run(household)) // 2].state.explanation().render()
        for heading in (
            "State:",
            "Probability:",
            "Sensor coverage:",
            "Supporting evidence:",
            "Contradictory:",
            "Quiet but working:",
            "No evidence from:",
        ):
            assert heading in rendered

    def test_an_abstention_explains_why_it_declined(self, household: object) -> None:
        dead = tuple(
            dropout(
                sensor_id,
                household.start + timedelta(days=2),  # type: ignore[attr-defined]
                timedelta(days=2),
            )
            for sensor_id in household.registry.sensor_ids()  # type: ignore[attr-defined]
        )
        records, _ = degrade(
            household.observations, DegradationConfig(faults=dead)  # type: ignore[attr-defined]
        )
        abstentions = [
            step.state.explanation()
            for step in run(household, records)
            if step.state.abstained
        ]
        assert abstentions
        assert abstentions[0].reason
        assert "coverage" in abstentions[0].reason or "clearly" in abstentions[0].reason

    def test_silent_and_missing_are_reported_separately(
        self, household: object
    ) -> None:
        """A quiet room and a broken sensor must not look the same.

        The radar drops out partway through rather than from the start: a
        sensor that has *never* reported is UNKNOWN, since it may simply not
        be installed, and only one that stops after establishing a cadence
        can be judged missing.
        """
        faults = (
            dropout(
                "living_radar",
                household.start + timedelta(days=3),  # type: ignore[attr-defined]
                timedelta(days=4),
            ),
        )
        records, _ = degrade(
            household.observations, DegradationConfig(faults=faults)  # type: ignore[attr-defined]
        )
        explanations = [step.state.explanation() for step in run(household, records)]
        assert any("living_radar" in e.missing for e in explanations)
        assert any(e.silent for e in explanations)

    def test_a_sensor_that_never_reported_is_not_called_missing(
        self, household: object
    ) -> None:
        """It may simply not be installed yet."""
        faults = (
            dropout(
                "living_radar",
                household.start,  # type: ignore[attr-defined]
                timedelta(days=30),
            ),
        )
        records, _ = degrade(
            household.observations, DegradationConfig(faults=faults)  # type: ignore[attr-defined]
        )
        explanations = [step.state.explanation() for step in run(household, records)]
        assert not any("living_radar" in e.missing for e in explanations)

    def test_the_explanation_serialises_with_the_estimate(
        self, household: object
    ) -> None:
        payload = run(household)[10].state.to_dict()
        assert "explanation_parts" in payload
        assert set(payload["explanation_parts"]) >= {  # type: ignore[arg-type]
            "state",
            "supporting",
            "contradicting",
            "silent",
            "missing",
        }
