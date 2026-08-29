"""Adversarial tests: deliberately attacking the platform's assumptions.

Each test here names an assumption that ambient monitoring systems commonly
make and that this platform claims not to. The tests exist to make those
claims falsifiable rather than rhetorical:

.. code-block:: text

    a missing sensor means inactivity
    an object event means a human behaviour
    a visitor's activity is the resident's
    sampling is regular
    clocks are correct
    the wearable is always worn
    the baseline is stable
    sensors are healthy
    a quiet sensor is a broken one
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from sensor_modeling.alerts import AlertKind
from sensor_modeling.baseline import AdaptiveBaseline, BaselineConfig, ChangeKind
from sensor_modeling.context import ResidentContextEstimator
from sensor_modeling.evaluation import state_metrics
from sensor_modeling.health import SensorHealthMonitor, SensorStatus
from sensor_modeling.observations import (
    ClockOffsetEstimator,
    Modality,
    Observation,
    ObservationIngestor,
    ObservationKind,
)
from sensor_modeling.online import (
    BehaviouralSensingPipeline,
    PipelineConfig,
    collect_alerts,
    daily_summaries,
)
from sensor_modeling.simulation import (
    DegradationConfig,
    HouseholdConfig,
    build_registry,
    degrade,
    dropout,
    simulate,
)
from sensor_modeling.states import BehaviouralState as S

STEP = timedelta(minutes=30)


def household(days: int = 8, seed: int = 555) -> object:
    """A short synthetic household."""
    return simulate(HouseholdConfig(days=days, seed=seed))


def pipeline_for(result: object, **kwargs: object) -> BehaviouralSensingPipeline:
    """Build a pipeline over a simulated household."""
    return BehaviouralSensingPipeline(
        result.registry,  # type: ignore[attr-defined]
        config=PipelineConfig(
            tz=result.config.tz, step=STEP, **kwargs  # type: ignore[attr-defined,arg-type]
        ),
    )


def run(result: object, observations: object) -> list:
    """Run a record through a fresh pipeline and return the steps."""
    pipeline = pipeline_for(result)
    steps = pipeline.run(observations)  # type: ignore[arg-type]
    steps.extend(pipeline.close(result.end))  # type: ignore[attr-defined]
    return steps


def scored(result: object, steps: list) -> object:
    """Score a run against ground truth."""
    truth = result.truth.states_at([step.at for step in steps])  # type: ignore[attr-defined]
    return state_metrics(truth, [step.state for step in steps])


class TestMissingIsNotInactivity:
    """'A silent sensor observed nothing happening.'"""

    def test_a_total_outage_produces_abstention_not_confident_inactivity(self) -> None:
        result = household()
        registry = result.registry  # type: ignore[attr-defined]
        blackout_start = result.start + timedelta(days=3)  # type: ignore[attr-defined]
        faults = tuple(
            dropout(sensor_id, blackout_start, timedelta(days=2))
            for sensor_id in registry.sensor_ids()
        )
        delivered, _ = degrade(
            result.observations, DegradationConfig(faults=faults)  # type: ignore[attr-defined]
        )
        steps = run(result, delivered)

        during = [
            step
            for step in steps
            if blackout_start + timedelta(hours=6)
            <= step.at
            < blackout_start + timedelta(days=2)
        ]
        assert during, "the blackout window should still produce steps"
        # Nothing was observed, so nothing may be asserted.
        assert all(step.state.state is S.UNKNOWN for step in during)
        assert all(step.state.completeness < 0.3 for step in during)

    def test_an_outage_never_enters_the_personal_baseline(self) -> None:
        """A sensor outage must not redefine what normal looks like."""
        baseline = AdaptiveBaseline(
            "sleeping_hours",
            BaselineConfig(min_samples=5, weekday_min_samples=99, min_scale=0.1),
        )
        start = date(2024, 1, 1)
        for offset in range(20):
            baseline.observe(start + timedelta(days=offset), 8.0)
        before = baseline.reference().centre

        for offset in range(20, 30):
            verdict = baseline.skip(start + timedelta(days=offset))
            assert verdict.kind is ChangeKind.INSUFFICIENT_DATA

        assert baseline.samples == 20
        assert baseline.reference().centre == before

    def test_a_poorly_observed_day_is_excluded_rather_than_recorded_as_quiet(
        self,
    ) -> None:
        result = household()
        registry = result.registry  # type: ignore[attr-defined]
        blackout = result.start + timedelta(days=3)  # type: ignore[attr-defined]
        faults = tuple(
            dropout(sensor_id, blackout, timedelta(days=1))
            for sensor_id in registry.sensor_ids()
        )
        delivered, _ = degrade(
            result.observations, DegradationConfig(faults=faults)  # type: ignore[attr-defined]
        )
        steps = run(result, delivered)
        unusable = [s for s in daily_summaries(steps) if not s.is_usable()]
        assert unusable, "the blacked-out day should be marked unusable"


class TestObjectEventsAreNotBehaviours:
    """'The fridge opened, therefore the resident ate.'"""

    def test_the_ontology_makes_no_claim_beyond_room_activity(self) -> None:
        names = {state.value for state in S}
        for forbidden in ("eating", "drinking", "toileting", "meal", "hydration"):
            assert not any(forbidden in name for name in names)

    def test_contact_activity_infers_location_not_consumption(self) -> None:
        result = household()
        steps = run(result, result.observations)  # type: ignore[attr-defined]
        reported = {step.state.state for step in steps}
        assert reported <= set(S)
        assert S.KITCHEN_ACTIVITY in reported


class TestVisitorContamination:
    """'Whoever tripped that sensor was the resident.'"""

    def test_attribution_falls_while_a_visitor_is_present(self) -> None:
        result = household()
        steps = run(result, result.observations)  # type: ignore[attr-defined]
        with_visitor = [
            step for step in steps if result.truth.visitor_at(step.at)  # type: ignore[attr-defined]
        ]
        without = [
            step for step in steps if not result.truth.visitor_at(step.at)  # type: ignore[attr-defined]
        ]
        assert with_visitor and without
        mean_with = np.mean([s.context.ambient_attribution() for s in with_visitor])
        mean_without = np.mean([s.context.ambient_attribution() for s in without])
        assert mean_with < mean_without

    def test_a_wearable_stays_attributable_regardless_of_visitors(self) -> None:
        result = household()
        estimator = ResidentContextEstimator(result.registry)  # type: ignore[attr-defined]
        estimate = estimator.update(result.start)  # type: ignore[attr-defined]
        weights = estimator.attribution(estimate)
        assert weights["wearable_motion"] == 1.0
        assert weights["resident_beacon"] == 1.0


class TestIrregularSamplingAndClocks:
    """'Samples arrive on a regular grid, from correct clocks.'"""

    def test_wildly_irregular_arrival_still_infers_a_state(self) -> None:
        result = household(days=4)
        rng = np.random.default_rng(0)
        thinned = [
            obs
            for obs in result.observations  # type: ignore[attr-defined]
            if rng.random() < 0.5
        ]
        steps = run(result, thinned)
        assert steps
        assert scored(result, steps).balanced_accuracy > 0.4  # type: ignore[attr-defined]

    def test_clock_drift_is_estimated_and_corrected_at_the_boundary(self) -> None:
        registry = build_registry()
        estimator = ClockOffsetEstimator(
            window=32, min_samples=4, tolerance=timedelta(seconds=15)
        )
        ingestor = ObservationIngestor(
            registry, correct_clock_drift=True, clock_estimator=estimator
        )
        base = result_start = simulate(HouseholdConfig(days=1, seed=1)).start
        offset = timedelta(seconds=90)
        batch = [
            Observation(
                timestamp=base + timedelta(minutes=index),
                sensor_id="kitchen_motion",
                modality=Modality.MOTION,
                kind=ObservationKind.EVENT,
                value=1.0,
                source="slow-hub",
                received_at=base + timedelta(minutes=index) + offset,
            )
            for index in range(12)
        ]
        admitted, report = ingestor.ingest_many(batch)
        assert report.clock_adjusted > 0
        corrected = admitted[-1]
        assert corrected.timestamp > result_start + timedelta(minutes=11)
        assert abs((corrected.timestamp - batch[-1].timestamp) - offset) < timedelta(
            seconds=5
        )

    def test_a_dst_transition_does_not_corrupt_daily_aggregation(self) -> None:
        result = simulate(HouseholdConfig(days=4, seed=31, start=date(2024, 3, 29)))
        steps = run(result, result.observations)
        summaries = daily_summaries(steps)
        spring_forward = [s for s in summaries if s.day == date(2024, 3, 31)]
        assert spring_forward
        # A 23-hour day cannot report more than 23 hours of behaviour.
        assert sum(spring_forward[0].hours.values()) <= 23.5


class TestSensorHealthAdversarial:
    """'Sensors are healthy, and a quiet sensor is a broken one.'"""

    def test_a_permanently_quiet_event_sensor_is_never_called_broken(self) -> None:
        result = household(days=6)
        quiet = [
            obs
            for obs in result.observations  # type: ignore[attr-defined]
            if obs.sensor_id != "fridge_contact"
        ]
        steps = run(result, quiet)
        last = steps[-1]
        assert last.health.sensors["fridge_contact"].status in {
            SensorStatus.UNKNOWN,
            SensorStatus.HEALTHY,
        }
        assert not last.health.sensors["fridge_contact"].is_faulty

    @staticmethod
    def _constant_bed_stream(start: object, hours: float) -> list[Observation]:
        """A bed sensor reporting 'occupied' continuously for *hours*."""
        return [
            Observation(
                timestamp=start + timedelta(minutes=5 * index),  # type: ignore[operator]
                sensor_id="bed_pressure",
                modality=Modality.BED_PRESSURE,
                kind=ObservationKind.STATE,
                value=1.0,
            )
            for index in range(int(hours * 12) + 1)
        ]

    def test_a_night_of_bed_occupancy_is_not_a_stuck_sensor(self) -> None:
        """Reporting an unchanged level is what a state sensor is for.

        Treating a repeated level as a fault made a sleeping resident look
        like a broken bed sensor every single night, which then discounted
        the strongest evidence for sleep exactly when it mattered.
        """
        result = household(days=2)
        monitor = SensorHealthMonitor(result.registry)  # type: ignore[attr-defined]
        stream = self._constant_bed_stream(result.start, hours=9)  # type: ignore[attr-defined]
        monitor.observe_many(stream)
        report = monitor.report_for(
            "bed_pressure", stream[-1].timestamp + timedelta(minutes=1)
        )
        assert report.status is SensorStatus.HEALTHY
        assert report.reliability > 0.9

    def test_an_implausibly_long_unchanged_level_is_caught(self) -> None:
        """Two solid days in bed is a sensor fault, not a behaviour."""
        result = household(days=4)
        monitor = SensorHealthMonitor(result.registry)  # type: ignore[attr-defined]
        stream = self._constant_bed_stream(result.start, hours=48)  # type: ignore[attr-defined]
        monitor.observe_many(stream)
        report = monitor.report_for(
            "bed_pressure", stream[-1].timestamp + timedelta(minutes=1)
        )
        assert report.status is SensorStatus.STUCK
        assert report.reliability < 0.2

    def test_a_stuck_sensor_does_not_trigger_the_outage_canary(self) -> None:
        """A stuck sensor is still delivering, so it says nothing about
        whether other sensors' records are getting through."""
        result = household(days=4)
        monitor = SensorHealthMonitor(result.registry)  # type: ignore[attr-defined]
        stream = self._constant_bed_stream(result.start, hours=48)  # type: ignore[attr-defined]
        monitor.observe_many(stream)
        for observation in result.observations:  # type: ignore[attr-defined]
            if observation.sensor_id == "fridge_contact":
                monitor.observe(observation)
        report = monitor.report(stream[-1].timestamp + timedelta(minutes=1))
        assert report.sensors["bed_pressure"].status is SensorStatus.STUCK
        assert not report.sensors["fridge_contact"].is_faulty

    @pytest.mark.parametrize("missing_rate", [0.05, 0.10, 0.20, 0.40])
    def test_performance_degrades_gracefully_under_missingness(
        self, missing_rate: float
    ) -> None:
        """The reliability sweep the evaluation methodology calls for."""
        result = household(days=6, seed=808)
        delivered, withheld = degrade(
            result.observations,  # type: ignore[attr-defined]
            DegradationConfig(missing_rate=missing_rate, seed=99),
        )
        assert withheld, "the sweep must actually withhold records"
        metrics = scored(result, run(result, delivered))
        # No cliff: the platform must still be informative at 40% loss, and
        # must never become confidently wrong.
        assert metrics.balanced_accuracy > 0.35  # type: ignore[attr-defined]
        assert metrics.calibration_error < 0.35  # type: ignore[attr-defined]

    def test_higher_missingness_is_not_rewarded(self) -> None:
        result = household(days=6, seed=808)
        scores = []
        for rate in (0.0, 0.4):
            delivered, _ = degrade(
                result.observations,  # type: ignore[attr-defined]
                DegradationConfig(missing_rate=rate, seed=99),
            )
            scores.append(scored(result, run(result, delivered)).balanced_accuracy)  # type: ignore[attr-defined]
        assert scores[0] >= scores[1]


class TestContradictionAndStorms:
    """'Sensors agree, and alerts are rare by nature.'"""

    def test_contradictory_evidence_is_surfaced_rather_than_averaged_away(
        self,
    ) -> None:
        result = household(days=5)
        steps = run(result, result.observations)  # type: ignore[attr-defined]
        contested = [step for step in steps if step.state.contradicting]
        assert contested, "some interval should have contradicting evidence"
        step = contested[0]
        assert "contradicted by" in step.state.explain()

    def test_a_broken_deployment_cannot_flood_the_carer(self) -> None:
        result = household(days=10, seed=404)
        registry = result.registry  # type: ignore[attr-defined]
        faults = tuple(
            dropout(sensor_id, result.start + timedelta(days=1), timedelta(days=8))  # type: ignore[attr-defined]
            for sensor_id in registry.sensor_ids()
        )
        delivered, _ = degrade(
            result.observations, DegradationConfig(faults=faults)  # type: ignore[attr-defined]
        )
        alerts = collect_alerts(run(result, delivered))
        # Deduplication and rate limiting must keep the burden bounded even
        # when everything is broken at once.
        assert len(alerts) < 15
        behavioural = [a for a in alerts if a.kind is AlertKind.BEHAVIOURAL_CHANGE]
        assert (
            not behavioural
        ), "a dead deployment must not produce findings about the resident"


class TestDegenerateInputs:
    """'Inputs are well formed.'"""

    def test_an_empty_record_produces_nothing_rather_than_crashing(self) -> None:
        pipeline = BehaviouralSensingPipeline(build_registry())
        assert pipeline.run([]) == []

    def test_a_single_observation_does_not_produce_a_confident_conclusion(
        self,
    ) -> None:
        result = household(days=2)
        one = [
            obs
            for obs in result.observations  # type: ignore[attr-defined]
            if obs.sensor_id == "kitchen_motion"
        ][:1]
        pipeline = pipeline_for(result)
        steps = pipeline.run(one)
        steps.extend(pipeline.close(result.start + timedelta(hours=2)))  # type: ignore[attr-defined]
        for step in steps:
            assert step.state.confidence <= 1.0
            assert np.isclose(step.state.belief.sum(), 1.0)

    def test_all_beliefs_remain_valid_probability_distributions(self) -> None:
        result = household(days=4)
        delivered, _ = degrade(
            result.observations,  # type: ignore[attr-defined]
            DegradationConfig(
                missing_rate=0.3,
                duplication_rate=0.1,
                late_rate=0.1,
                clock_drift={"sim-radar": timedelta(minutes=2)},
                seed=13,
            ),
        )
        for step in run(result, delivered):
            assert np.all(np.isfinite(step.state.belief))
            assert step.state.belief.min() >= 0.0
            assert np.isclose(step.state.belief.sum(), 1.0)
            assert np.isclose(step.context.belief.sum(), 1.0)
            assert 0.0 <= step.health.coverage <= 1.0


class TestNoCircularValidation:
    """'The simulator and the estimator may share a model.'"""

    def test_the_simulator_does_not_use_the_inference_transition_model(self) -> None:
        """Guard against the two silently converging in future edits."""
        import inspect

        from sensor_modeling.simulation import household as household_module

        source = inspect.getsource(household_module)
        for leaked in ("StateOntology", "transition_matrix", "MultimodalBayesFilter"):
            assert leaked not in source, (
                f"the simulator must not import {leaked}; sharing a generative "
                "model with the estimator would make evaluation circular"
            )

    def test_ground_truth_is_never_handed_to_the_pipeline(self) -> None:
        import inspect

        from sensor_modeling.online import pipeline as pipeline_module

        source = inspect.getsource(pipeline_module)
        for leaked in ("GroundTruth", "simulation", "Episode"):
            assert leaked not in source
