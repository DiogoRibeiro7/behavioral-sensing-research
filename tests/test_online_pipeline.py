"""Integration tests for the online end-to-end pipeline."""

from __future__ import annotations

from datetime import timedelta

import pytest

from sensor_modeling.alerts import AlertKind
from sensor_modeling.fusion import EmissionDefaults, default_emissions
from sensor_modeling.observations import Modality, Observation, ObservationKind
from sensor_modeling.online import (
    BehaviouralSensingPipeline,
    PipelineConfig,
    collect_alerts,
    daily_summaries,
)
from sensor_modeling.simulation import (
    DegradationConfig,
    HouseholdConfig,
    degrade,
    dropout,
    not_worn,
    simulate,
)
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology


def run_pipeline(
    result: object, **kwargs: object
) -> tuple[BehaviouralSensingPipeline, list]:
    """Drive a simulation through the pipeline and return both."""
    config = PipelineConfig(
        tz=result.config.tz,  # type: ignore[attr-defined]
        step=timedelta(minutes=5),
        **kwargs,  # type: ignore[arg-type]
    )
    pipeline = BehaviouralSensingPipeline(result.registry, config=config)  # type: ignore[attr-defined]
    steps = pipeline.run(result.observations)  # type: ignore[attr-defined]
    steps.extend(pipeline.close(result.end))  # type: ignore[attr-defined]
    return pipeline, steps


def accuracy(result: object, steps: list) -> float:
    """Fraction of steps whose reported state matches the ground truth."""
    truth = result.truth.states_at([step.at for step in steps])  # type: ignore[attr-defined]
    pairs = [
        (actual, step.state.state)
        for actual, step in zip(truth, steps)
        if actual is not None
    ]
    return sum(1 for actual, reported in pairs if actual == reported) / len(pairs)


@pytest.fixture(scope="module")
def household() -> object:
    """A twelve-day synthetic household, shared across tests."""
    return simulate(HouseholdConfig(days=12, seed=2024))


class TestEndToEndRecovery:
    def test_the_pipeline_recovers_the_simulated_behaviour(
        self, household: object
    ) -> None:
        """The headline claim: a Markov filter that knows nothing about
        schedules recovers a schedule-generated day from ambient sensors."""
        _, steps = run_pipeline(household)
        assert accuracy(household, steps) > 0.85

    def test_sleep_is_distinguished_from_being_inactive_at_home(
        self, household: object
    ) -> None:
        """Bed occupancy plus low wearable motion must separate these two,
        and the bedroom sensor's silence must not argue against sleep."""
        _, steps = run_pipeline(household)
        truth = household.truth.states_at([step.at for step in steps])  # type: ignore[attr-defined]
        asleep = [
            step.state.state
            for actual, step in zip(truth, steps)
            if actual is S.SLEEPING
        ]
        recall = sum(1 for state in asleep if state is S.SLEEPING) / len(asleep)
        assert recall > 0.85

    def test_inferred_daily_hours_track_the_truth(self, household: object) -> None:
        _, steps = run_pipeline(household)
        true_sleep = household.truth.daily_hours(S.SLEEPING)  # type: ignore[attr-defined]
        summaries = [
            summary for summary in daily_summaries(steps) if summary.observed > 0.95
        ]
        assert len(summaries) >= 8
        errors = [
            abs(summary.hours_in(S.SLEEPING) - true_sleep[summary.day])
            for summary in summaries
            if summary.day in true_sleep
        ]
        assert sum(errors) / len(errors) < 1.0

    def test_every_step_reports_state_context_and_health(
        self, household: object
    ) -> None:
        _, steps = run_pipeline(household)
        step = steps[len(steps) // 2]
        payload = step.to_dict()
        assert set(payload) >= {"at", "state", "context", "health"}
        assert 0.0 <= step.context.resident_home <= 1.0
        assert 0.0 <= step.health.coverage <= 1.0
        assert step.state.explain()


class TestStreamRobustness:
    def test_out_of_order_arrivals_are_reordered_within_tolerance(
        self, household: object
    ) -> None:
        late, _ = degrade(
            household.observations,  # type: ignore[attr-defined]
            DegradationConfig(late_rate=0.2, late_delay=timedelta(minutes=4), seed=3),
        )
        ordered_pipeline, ordered_steps = run_pipeline(household)
        pipeline = BehaviouralSensingPipeline(
            household.registry,  # type: ignore[attr-defined]
            config=PipelineConfig(
                tz=household.config.tz,  # type: ignore[attr-defined]
                step=timedelta(minutes=5),
                lateness_tolerance=timedelta(minutes=10),
            ),
        )
        steps = pipeline.run(late)
        steps.extend(pipeline.close(household.end))  # type: ignore[attr-defined]
        assert pipeline.too_late == 0
        assert accuracy(household, steps) == pytest.approx(
            accuracy(household, ordered_steps), abs=0.05
        )
        assert ordered_pipeline.too_late == 0

    def test_records_later_than_the_tolerance_are_counted_not_absorbed(
        self, household: object
    ) -> None:
        late, _ = degrade(
            household.observations,  # type: ignore[attr-defined]
            DegradationConfig(late_rate=0.3, late_delay=timedelta(hours=3), seed=4),
        )
        pipeline = BehaviouralSensingPipeline(
            household.registry,  # type: ignore[attr-defined]
            config=PipelineConfig(
                tz=household.config.tz,  # type: ignore[attr-defined]
                step=timedelta(minutes=5),
                lateness_tolerance=timedelta(minutes=10),
            ),
        )
        pipeline.run(late)
        assert pipeline.too_late > 0

    def test_duplicate_delivery_does_not_change_the_conclusion(
        self, household: object
    ) -> None:
        duplicated, _ = degrade(
            household.observations,  # type: ignore[attr-defined]
            DegradationConfig(duplication_rate=0.2, seed=9),
        )
        _, clean_steps = run_pipeline(household)
        pipeline = BehaviouralSensingPipeline(
            household.registry,  # type: ignore[attr-defined]
            config=PipelineConfig(tz=household.config.tz, step=timedelta(minutes=5)),  # type: ignore[attr-defined]
        )
        steps = pipeline.run(duplicated)
        steps.extend(pipeline.close(household.end))  # type: ignore[attr-defined]
        assert accuracy(household, steps) == pytest.approx(
            accuracy(household, clean_steps), abs=0.05
        )

    def test_malformed_records_are_rejected_without_stopping_the_stream(
        self, household: object
    ) -> None:
        stray = Observation(
            timestamp=household.start,  # type: ignore[attr-defined]
            sensor_id="not_installed",
            modality=Modality.VIBRATION,
            kind=ObservationKind.EVENT,
            value=1.0,
        )
        pipeline = BehaviouralSensingPipeline(
            household.registry,  # type: ignore[attr-defined]
            config=PipelineConfig(tz=household.config.tz, step=timedelta(minutes=5)),  # type: ignore[attr-defined]
        )
        assert pipeline.push(stray) is False
        steps = pipeline.run(household.observations)  # type: ignore[attr-defined]
        assert steps
        assert len(pipeline.ingestion.rejected) == 1

    def test_a_sensor_dropout_does_not_become_observed_inactivity(
        self, household: object
    ) -> None:
        """The load-bearing claim of the whole codebase, end to end."""
        start = household.start + timedelta(days=3)  # type: ignore[attr-defined]
        faults = (dropout("bed_pressure", start, timedelta(days=2)),)
        degraded, _ = degrade(
            household.observations, DegradationConfig(faults=faults)  # type: ignore[attr-defined]
        )
        pipeline = BehaviouralSensingPipeline(
            household.registry,  # type: ignore[attr-defined]
            config=PipelineConfig(tz=household.config.tz, step=timedelta(minutes=5)),  # type: ignore[attr-defined]
        )
        steps = pipeline.run(degraded)
        steps.extend(pipeline.close(household.end))  # type: ignore[attr-defined]

        during = [
            step for step in steps if start <= step.at < start + timedelta(days=2)
        ]
        assert during
        # The bed sensor is discounted, not read as an empty bed, so the
        # remaining modalities still recover sleep.
        truth = household.truth.states_at([step.at for step in during])  # type: ignore[attr-defined]
        asleep = [
            step.state.state
            for actual, step in zip(truth, during)
            if actual is S.SLEEPING
        ]
        assert asleep
        assert sum(1 for state in asleep if state is S.SLEEPING) / len(asleep) > 0.5
        assert any("bed_pressure" in step.state.missing for step in during)

    def test_losing_the_wearable_degrades_gracefully(self, household: object) -> None:
        start = household.start + timedelta(days=2)  # type: ignore[attr-defined]
        faults = not_worn(
            ["wearable_motion", "resident_beacon"], start, timedelta(days=3)
        )
        degraded, _ = degrade(
            household.observations, DegradationConfig(faults=faults)  # type: ignore[attr-defined]
        )
        pipeline = BehaviouralSensingPipeline(
            household.registry,  # type: ignore[attr-defined]
            config=PipelineConfig(tz=household.config.tz, step=timedelta(minutes=5)),  # type: ignore[attr-defined]
        )
        steps = pipeline.run(degraded)
        steps.extend(pipeline.close(household.end))  # type: ignore[attr-defined]
        assert accuracy(household, steps) > 0.7

    def test_a_degraded_deployment_raises_a_system_health_alert(
        self, household: object
    ) -> None:
        start = household.start + timedelta(days=1)  # type: ignore[attr-defined]
        faults = (
            dropout("bed_pressure", start, timedelta(days=4)),
            dropout("living_radar", start, timedelta(days=4)),
            *not_worn(["wearable_motion", "resident_beacon"], start, timedelta(days=4)),
        )
        degraded, _ = degrade(
            household.observations, DegradationConfig(faults=faults)  # type: ignore[attr-defined]
        )
        pipeline = BehaviouralSensingPipeline(
            household.registry,  # type: ignore[attr-defined]
            config=PipelineConfig(tz=household.config.tz, step=timedelta(minutes=5)),  # type: ignore[attr-defined]
        )
        steps = pipeline.run(degraded)
        health_alerts = [
            alert
            for alert in collect_alerts(steps)
            if alert.kind is AlertKind.SYSTEM_HEALTH
        ]
        assert health_alerts
        assert any("not the resident" in c for c in health_alerts[0].caveats)


class TestPipelineMechanics:
    def test_snapshot_and_restore_continue_the_same_run(
        self, household: object
    ) -> None:
        observations = list(household.observations)  # type: ignore[attr-defined]
        halfway = len(observations) // 2

        straight = BehaviouralSensingPipeline(
            household.registry,  # type: ignore[attr-defined]
            config=PipelineConfig(tz=household.config.tz, step=timedelta(minutes=5)),  # type: ignore[attr-defined]
        )
        straight.run(observations)

        first = BehaviouralSensingPipeline(
            household.registry,  # type: ignore[attr-defined]
            config=PipelineConfig(tz=household.config.tz, step=timedelta(minutes=5)),  # type: ignore[attr-defined]
        )
        first.run(observations[:halfway])
        state = first.snapshot()

        resumed = BehaviouralSensingPipeline(
            household.registry,  # type: ignore[attr-defined]
            config=PipelineConfig(tz=household.config.tz, step=timedelta(minutes=5)),  # type: ignore[attr-defined]
        )
        resumed.restore(state)
        resumed.run(observations[halfway:])

        assert resumed.filter.at == straight.filter.at
        assert resumed.baselines["sleeping"].samples == pytest.approx(
            straight.baselines["sleeping"].samples, abs=1
        )

    def test_a_snapshot_round_trips_through_json(self, household: object) -> None:
        import json

        pipeline, _ = run_pipeline(household)
        payload = json.loads(json.dumps(pipeline.snapshot()))
        restored = BehaviouralSensingPipeline(
            household.registry,  # type: ignore[attr-defined]
            config=PipelineConfig(tz=household.config.tz, step=timedelta(minutes=5)),  # type: ignore[attr-defined]
        )
        restored.restore(payload)
        assert restored.filter.at == pipeline.filter.at

    def test_custom_emissions_are_used_when_supplied(self, household: object) -> None:
        ontology = StateOntology()
        emissions = default_emissions(
            household.registry, ontology, EmissionDefaults(active_rate=60.0)  # type: ignore[attr-defined]
        )
        pipeline = BehaviouralSensingPipeline(
            household.registry,  # type: ignore[attr-defined]
            ontology,
            emissions,
            PipelineConfig(tz=household.config.tz, step=timedelta(minutes=5)),  # type: ignore[attr-defined]
        )
        assert set(pipeline.filter.emissions) == {
            spec.sensor_id for spec in household.registry  # type: ignore[attr-defined]
        }

    def test_an_empty_stream_produces_no_steps(self, household: object) -> None:
        pipeline = BehaviouralSensingPipeline(household.registry)  # type: ignore[attr-defined]
        assert pipeline.run([]) == []
        assert pipeline.close(household.start) == ()  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"step": timedelta(0)},
            {"lateness_tolerance": timedelta(seconds=-1)},
            {"features": ()},
            {"min_day_coverage": 1.5},
            {"min_day_observed": -0.1},
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            PipelineConfig(**kwargs)
