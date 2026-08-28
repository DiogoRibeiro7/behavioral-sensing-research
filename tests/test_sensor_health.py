"""Tests for online sensor health estimation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sensor_modeling.health import (
    HealthConfig,
    SensorHealthMonitor,
    SensorStatus,
    status_reliability,
)
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
    Unit,
)

T0 = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
HEARTBEAT = timedelta(minutes=5)


def registry() -> SensorRegistry:
    """A deployment with one heartbeat sensor and one purely event sensor."""
    return SensorRegistry.from_specs(
        [
            SensorSpec(
                "hall_temp",
                Modality.ENVIRONMENTAL,
                unit=Unit.CELSIUS,
                room="hall",
                expected_interval=HEARTBEAT,
                value_range=(-20.0, 60.0),
            ),
            SensorSpec("fridge_contact", Modality.CONTACT, room="kitchen"),
        ]
    )


def temperature(minute: int, value: float, quality: float = 1.0) -> Observation:
    """A sampled temperature reading *minute* minutes after the epoch."""
    return Observation(
        timestamp=T0 + timedelta(minutes=minute),
        sensor_id="hall_temp",
        modality=Modality.ENVIRONMENTAL,
        kind=ObservationKind.SAMPLE,
        value=value,
        unit=Unit.CELSIUS,
        quality=quality,
    )


def fridge(minute: int) -> Observation:
    """A contact activation *minute* minutes after the epoch."""
    return Observation(
        timestamp=T0 + timedelta(minutes=minute),
        sensor_id="fridge_contact",
        modality=Modality.CONTACT,
        kind=ObservationKind.EVENT,
        value=1.0,
    )


class TestSilence:
    def test_a_reporting_sensor_is_healthy(self) -> None:
        monitor = SensorHealthMonitor(registry())
        monitor.observe_many(temperature(m, 20.0 + m * 0.01) for m in range(0, 30, 5))
        report = monitor.report_for("hall_temp", T0 + timedelta(minutes=26))
        assert report.status is SensorStatus.HEALTHY
        assert report.reliability > 0.9

    @pytest.mark.parametrize(
        ("minutes", "expected"),
        [
            (12, SensorStatus.DEGRADED),
            (30, SensorStatus.DROPOUT),
            (120, SensorStatus.MISSING),
        ],
    )
    def test_silence_escalates_for_a_sensor_that_promised_to_report(
        self, minutes: int, expected: SensorStatus
    ) -> None:
        monitor = SensorHealthMonitor(registry())
        monitor.observe(temperature(0, 20.0))
        report = monitor.report_for("hall_temp", T0 + timedelta(minutes=minutes))
        assert report.status is expected

    def test_a_missing_sensor_supplies_no_evidence(self) -> None:
        monitor = SensorHealthMonitor(registry())
        monitor.observe(temperature(0, 20.0))
        report = monitor.report_for("hall_temp", T0 + timedelta(hours=4))
        assert report.status is SensorStatus.MISSING
        assert report.reliability == 0.0
        assert report.is_faulty

    def test_silence_of_an_event_sensor_is_never_called_a_failure(self) -> None:
        """A quiet cupboard sensor must not be turned into a broken sensor."""
        monitor = SensorHealthMonitor(registry())
        monitor.observe(fridge(0))
        report = monitor.report_for("fridge_contact", T0 + timedelta(days=3))
        assert report.status is SensorStatus.HEALTHY
        assert not report.is_faulty
        assert report.silence == timedelta(days=3)

    def test_a_sensor_that_never_reported_is_unknown_not_broken(self) -> None:
        monitor = SensorHealthMonitor(registry())
        report = monitor.report_for("fridge_contact", T0)
        assert report.status is SensorStatus.UNKNOWN
        assert report.last_seen is None
        assert report.silence is None


class TestValueVerdicts:
    def test_a_stuck_sampled_sensor_is_detected(self) -> None:
        monitor = SensorHealthMonitor(registry(), HealthConfig(stuck_samples=5))
        monitor.observe_many(temperature(m, 21.0) for m in range(0, 30, 5))
        report = monitor.report_for("hall_temp", T0 + timedelta(minutes=26))
        assert report.status is SensorStatus.STUCK
        assert report.reliability < 0.2

    def test_repeated_event_activations_are_not_stuck(self) -> None:
        """Every contact activation reports the same value; that is normal."""
        monitor = SensorHealthMonitor(registry(), HealthConfig(stuck_samples=3))
        monitor.observe_many(fridge(m) for m in range(20))
        report = monitor.report_for("fridge_contact", T0 + timedelta(minutes=20))
        assert report.status is SensorStatus.HEALTHY

    def test_implausible_values_are_flagged_after_the_tolerance(self) -> None:
        monitor = SensorHealthMonitor(
            registry(), HealthConfig(out_of_range_tolerance=2)
        )
        monitor.observe(temperature(0, 20.0))
        monitor.observe(temperature(5, 900.0))
        assert (
            monitor.report_for("hall_temp", T0 + timedelta(minutes=6)).status
            is SensorStatus.HEALTHY
        )
        monitor.observe(temperature(10, 900.0))
        assert (
            monitor.report_for("hall_temp", T0 + timedelta(minutes=11)).status
            is SensorStatus.OUT_OF_RANGE
        )

    def test_a_returning_sensor_recovers(self) -> None:
        monitor = SensorHealthMonitor(
            registry(), HealthConfig(out_of_range_tolerance=1)
        )
        monitor.observe(temperature(0, 900.0))
        assert (
            monitor.report_for("hall_temp", T0 + timedelta(minutes=1)).status
            is SensorStatus.OUT_OF_RANGE
        )
        monitor.observe(temperature(5, 21.0))
        assert (
            monitor.report_for("hall_temp", T0 + timedelta(minutes=6)).status
            is SensorStatus.HEALTHY
        )

    def test_low_reported_quality_degrades_a_sensor(self) -> None:
        monitor = SensorHealthMonitor(registry(), HealthConfig(quality_floor=0.7))
        monitor.observe_many(
            temperature(m, 20.0 + m * 0.01, quality=0.1) for m in range(0, 100, 5)
        )
        report = monitor.report_for("hall_temp", T0 + timedelta(minutes=96))
        assert report.status is SensorStatus.DEGRADED
        assert report.reliability < 0.3

    def test_calibration_drift_is_reported(self) -> None:
        config = HealthConfig(
            drift_reference=20, drift_recent=10, drift_sigma=3.0, stuck_samples=50
        )
        monitor = SensorHealthMonitor(registry(), config)
        wobble = [0.0, 0.1, -0.1, 0.2, -0.2]
        for step in range(20):
            monitor.observe(temperature(step * 5, 20.0 + wobble[step % 5]))
        for step in range(20, 40):
            monitor.observe(temperature(step * 5, 26.0 + wobble[step % 5]))
        report = monitor.report_for("hall_temp", T0 + timedelta(minutes=40 * 5))
        assert report.status is SensorStatus.DRIFTING
        assert "shifted" in report.detail

    def test_stable_readings_are_not_called_drift(self) -> None:
        config = HealthConfig(drift_reference=20, drift_recent=10, stuck_samples=100)
        monitor = SensorHealthMonitor(registry(), config)
        wobble = [0.0, 0.1, -0.1, 0.2, -0.2]
        for step in range(40):
            monitor.observe(temperature(step * 5, 20.0 + wobble[step % 5]))
        report = monitor.report_for("hall_temp", T0 + timedelta(minutes=40 * 5))
        assert report.status is SensorStatus.HEALTHY

    def test_drift_floor_suppresses_trivial_shifts(self) -> None:
        config = HealthConfig(
            drift_reference=20,
            drift_recent=10,
            drift_sigma=3.0,
            drift_floor=50.0,
            stuck_samples=100,
        )
        monitor = SensorHealthMonitor(registry(), config)
        wobble = [0.0, 0.1, -0.1, 0.2, -0.2]
        for step in range(20):
            monitor.observe(temperature(step * 5, 20.0 + wobble[step % 5]))
        for step in range(20, 40):
            monitor.observe(temperature(step * 5, 26.0 + wobble[step % 5]))
        report = monitor.report_for("hall_temp", T0 + timedelta(minutes=40 * 5))
        assert report.status is SensorStatus.HEALTHY


class TestSystemReport:
    def test_system_health_is_separable_from_behaviour(self) -> None:
        monitor = SensorHealthMonitor(registry())
        monitor.observe(temperature(0, 20.0))
        monitor.observe(fridge(0))
        report = monitor.report(T0 + timedelta(hours=4))
        assert report.faulty == ["hall_temp"]
        assert 0.0 < report.coverage < 1.0
        assert report.reliabilities()["hall_temp"] == 0.0
        assert report.reliabilities()["fridge_contact"] > 0.0
        assert report.to_dict()["faulty"] == ["hall_temp"]

    def test_unregistered_sensors_are_ignored(self) -> None:
        monitor = SensorHealthMonitor(registry())
        stray = Observation(
            timestamp=T0,
            sensor_id="not_registered",
            modality=Modality.MOTION,
            kind=ObservationKind.EVENT,
            value=1.0,
        )
        monitor.observe(stray)
        assert set(monitor.report(T0).sensors) == {"hall_temp", "fridge_contact"}

    def test_snapshot_and_restore_survive_a_restart(self) -> None:
        monitor = SensorHealthMonitor(registry(), HealthConfig(stuck_samples=5))
        monitor.observe_many(temperature(m, 21.0) for m in range(0, 30, 5))
        state = monitor.snapshot()

        restarted = SensorHealthMonitor(registry(), HealthConfig(stuck_samples=5))
        restarted.restore(state)
        report = restarted.report_for("hall_temp", T0 + timedelta(minutes=26))
        assert report.status is SensorStatus.STUCK
        assert report.observations == 6


class TestConfiguration:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"degraded_after": 10.0, "dropout_after": 5.0},
            {"stuck_samples": 1},
            {"quality_floor": 1.5},
            {"quality_smoothing": 0.0},
            {"drift_reference": 1},
            {"drift_sigma": 0.0},
            {"drift_floor": -1.0},
            {"out_of_range_tolerance": 0},
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            HealthConfig(**kwargs)  # type: ignore[arg-type]

    def test_status_reliability_orders_faults_below_health(self) -> None:
        assert status_reliability(SensorStatus.HEALTHY) == 1.0
        assert status_reliability(SensorStatus.MISSING) == 0.0
        assert status_reliability(SensorStatus.UNKNOWN) > status_reliability(
            SensorStatus.STUCK
        )
