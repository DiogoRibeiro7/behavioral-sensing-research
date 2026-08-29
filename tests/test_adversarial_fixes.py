"""Regression tests for defects found by adversarial review.

Each test corresponds to a finding in ``docs/ADVERSARIAL_REVIEW.md`` and
fails if that defect returns.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sensor_modeling.fusion import MultimodalBayesFilter, default_emissions
from sensor_modeling.health import HealthConfig, SensorHealthMonitor, SensorStatus
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
)
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

T0 = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
ONTOLOGY = StateOntology()


def kitchen_registry(copies: int, *, group: str | None) -> SensorRegistry:
    """A registry of *copies* sensors all watching the same kitchen."""
    return SensorRegistry.from_specs(
        [
            SensorSpec(
                f"kitchen_{index}",
                Modality.MOTION,
                room="kitchen",
                redundancy_group=group,
            )
            for index in range(copies)
        ]
    )


def belief_after_one_event(copies: int, *, group: str | None) -> float:
    """Return P(kitchen_activity) after every copy reports one activation."""
    registry = kitchen_registry(copies, group=group)
    bayes = MultimodalBayesFilter(
        ONTOLOGY, default_emissions(registry, ONTOLOGY), registry=registry
    )
    bayes.update(T0, [])
    now = T0 + timedelta(minutes=5)
    observations = [
        Observation(
            now - timedelta(seconds=10),
            f"kitchen_{index}",
            Modality.MOTION,
            ObservationKind.EVENT,
            1.0,
        )
        for index in range(copies)
    ]
    return bayes.update(now, observations).probabilities[S.KITCHEN_ACTIVITY]


class TestRedundancy:
    """M1: redundant sensors drove the posterior to false certainty."""

    def test_undeclared_redundancy_still_compounds(self) -> None:
        """Documents the hazard the declaration exists to prevent.

        Without a declared group the filter has no way to know two sensors
        watch the same thing, so the same evidence is applied once per copy.
        The assertion is on how far the posterior moves, not on which way:
        the compounding pushes toward or away from a state depending on
        whether the observed count sits above or below what that state's
        rate predicts, and it is wrong in either direction.
        """
        single = belief_after_one_event(1, group=None)
        assert abs(belief_after_one_event(8, group=None) - single) > abs(
            belief_after_one_event(2, group=None) - single
        )

    def test_a_declared_group_makes_copies_count_once(self) -> None:
        single = belief_after_one_event(1, group="kitchen_pir")
        for copies in (2, 4, 8):
            assert belief_after_one_event(copies, group="kitchen_pir") == pytest.approx(
                single, abs=1e-9
            )

    def test_grouping_removes_the_compounding_entirely(self) -> None:
        """With the group declared, eight copies move the posterior no further
        than one does."""
        single = belief_after_one_event(1, group="kitchen_pir")
        assert belief_after_one_event(8, group="kitchen_pir") == pytest.approx(
            single, abs=1e-9
        )

    def test_the_group_is_recorded_in_the_registry_export(self) -> None:
        exported = kitchen_registry(2, group="kitchen_pir").to_dict()
        assert exported["kitchen_0"]["redundancy_group"] == "kitchen_pir"

    def test_ungrouped_sensors_keep_their_full_weight(self) -> None:
        registry = SensorRegistry.from_specs(
            [SensorSpec("kitchen_0", Modality.MOTION, room="kitchen")]
        )
        emission = default_emissions(registry, ONTOLOGY)[0]
        assert emission.weight == pytest.approx(1.0)


class TestUnderDelivery:
    """M2: a sensor dropping records still looked healthy."""

    @staticmethod
    def registry() -> SensorRegistry:
        return SensorRegistry.from_specs(
            [
                SensorSpec(
                    "hall_temp",
                    Modality.ENVIRONMENTAL,
                    kind=ObservationKind.SAMPLE,
                    expected_interval=timedelta(minutes=1),
                )
            ]
        )

    @staticmethod
    def verdict(spacing_seconds: int, *, config: HealthConfig | None = None) -> object:
        """Report health immediately after a run at the given spacing.

        Reporting one second after the latest observation keeps the silence
        check from firing, so only the delivery-rate check can produce a
        verdict.
        """
        monitor = SensorHealthMonitor(
            TestUnderDelivery.registry(), config or HealthConfig(stuck_samples=999)
        )
        last = T0
        for index in range(40):
            last = T0 + timedelta(seconds=spacing_seconds * index)
            monitor.observe(
                Observation(
                    last,
                    "hall_temp",
                    Modality.ENVIRONMENTAL,
                    ObservationKind.SAMPLE,
                    20.0 + (index % 5) * 0.1,
                )
            )
        return monitor.report_for("hall_temp", last + timedelta(seconds=1))

    def test_a_sensor_meeting_its_cadence_is_healthy(self) -> None:
        assert self.verdict(60).status is SensorStatus.HEALTHY  # type: ignore[attr-defined]

    def test_a_mild_shortfall_is_tolerated(self) -> None:
        """67% of the promised rate sits above the default floor."""
        assert self.verdict(90).status is SensorStatus.HEALTHY  # type: ignore[attr-defined]

    def test_a_sensor_dropping_most_records_is_degraded(self) -> None:
        report = self.verdict(150)
        assert report.status is SensorStatus.DEGRADED  # type: ignore[attr-defined]
        assert "promised reporting rate" in report.detail  # type: ignore[attr-defined]
        assert report.reliability < 0.7  # type: ignore[attr-defined]

    def test_the_shortfall_is_quantified_in_the_verdict(self) -> None:
        assert "25%" in self.verdict(240).detail  # type: ignore[attr-defined]

    def test_the_floor_is_configurable(self) -> None:
        lenient = HealthConfig(stuck_samples=999, delivery_floor=0.2)
        assert self.verdict(150, config=lenient).status is SensorStatus.HEALTHY  # type: ignore[attr-defined]

    def test_an_event_sensor_without_a_cadence_is_not_judged(self) -> None:
        """It never promised a rate, so a drop is indistinguishable from a
        quieter resident."""
        registry = SensorRegistry.from_specs(
            [SensorSpec("fridge", Modality.CONTACT, room="kitchen")]
        )
        monitor = SensorHealthMonitor(registry)
        for index in range(40):
            monitor.observe(
                Observation(
                    T0 + timedelta(minutes=30 * index),
                    "fridge",
                    Modality.CONTACT,
                    ObservationKind.EVENT,
                    1.0,
                )
            )
        report = monitor.report_for("fridge", T0 + timedelta(minutes=30 * 40))
        assert not report.is_faulty

    def test_invalid_delivery_configuration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="delivery_floor"):
            HealthConfig(delivery_floor=0.0)
        with pytest.raises(ValueError, match="delivery_window"):
            HealthConfig(delivery_window=1)

    def test_delivery_history_survives_a_restart(self) -> None:
        monitor = SensorHealthMonitor(self.registry(), HealthConfig(stuck_samples=999))
        last = T0
        for index in range(40):
            last = T0 + timedelta(seconds=150 * index)
            monitor.observe(
                Observation(
                    last,
                    "hall_temp",
                    Modality.ENVIRONMENTAL,
                    ObservationKind.SAMPLE,
                    20.0 + (index % 5) * 0.1,
                )
            )
        restarted = SensorHealthMonitor(
            self.registry(), HealthConfig(stuck_samples=999)
        )
        restarted.restore(monitor.snapshot())
        report = restarted.report_for("hall_temp", last + timedelta(seconds=1))
        assert report.status is SensorStatus.DEGRADED
