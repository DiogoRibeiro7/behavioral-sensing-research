"""Tests for the restraint applied between behavioural change and alert."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from sensor_modeling.alerts import (
    AlertEngine,
    AlertKind,
    AlertPolicy,
    AlertSeverity,
)
from sensor_modeling.baseline import BaselineReference, BehaviouralChange, ChangeKind
from sensor_modeling.health import SensorHealthReport, SensorStatus, SystemHealthReport

T0 = datetime(2024, 5, 1, 9, 0, tzinfo=timezone.utc)
DAY = date(2024, 5, 1)


def change(
    *,
    kind: ChangeKind = ChangeKind.PERSISTENT_CHANGE,
    deviation: float = -6.0,
    duration_days: int = 5,
    feature: str = "sleep_hours",
    weekday_aware: bool = True,
) -> BehaviouralChange:
    """Build a behavioural change verdict."""
    return BehaviouralChange(
        feature=feature,
        day=DAY,
        kind=kind,
        value=3.0,
        reference=BaselineReference(
            centre=8.0,
            scale=0.8,
            samples=40,
            weekday_samples=6,
            weekday_aware=weekday_aware,
        ),
        deviation=deviation,
        duration_days=duration_days,
        slope_per_day=0.0,
        change_point=None,
        detail="test change",
    )


def health(coverage_sensors: dict[str, float], at: datetime = T0) -> SystemHealthReport:
    """Build a system health report from per-sensor reliabilities."""
    return SystemHealthReport(
        at=at,
        sensors={
            sensor_id: SensorHealthReport(
                sensor_id=sensor_id,
                status=(SensorStatus.HEALTHY if value > 0.5 else SensorStatus.MISSING),
                reliability=value,
                last_seen=at,
                silence=timedelta(0),
                observations=10,
                detail="test",
            )
            for sensor_id, value in coverage_sensors.items()
        },
    )


class TestRestraint:
    def test_an_ordinary_day_never_alerts(self) -> None:
        engine = AlertEngine()
        assert engine.consider(change(kind=ChangeKind.ORDINARY), at=T0) is None

    def test_a_temporary_disturbance_never_alerts(self) -> None:
        """An unusual day that has already reverted is not a finding."""
        engine = AlertEngine()
        verdict = change(kind=ChangeKind.TEMPORARY_DISTURBANCE, duration_days=1)
        assert engine.consider(verdict, at=T0) is None

    def test_insufficient_data_never_alerts_about_behaviour(self) -> None:
        engine = AlertEngine()
        assert engine.consider(change(kind=ChangeKind.INSUFFICIENT_DATA), at=T0) is None

    def test_a_sustained_change_alerts(self) -> None:
        engine = AlertEngine()
        alert = engine.consider(change(), at=T0)
        assert alert is not None
        assert alert.kind is AlertKind.BEHAVIOURAL_CHANGE
        assert "sleep_hours" in alert.summary
        assert alert.confidence == pytest.approx(1.0)

    def test_a_change_seen_through_poor_coverage_does_not_alert(self) -> None:
        engine = AlertEngine()
        assert engine.consider(change(), at=T0, coverage=0.2) is None

    def test_a_change_that_may_be_a_visitor_does_not_alert(self) -> None:
        """A daughter's visit must not be reported as the resident changing."""
        engine = AlertEngine()
        assert engine.consider(change(), at=T0, attribution=0.3) is None

    def test_partial_attribution_is_recorded_as_a_caveat(self) -> None:
        engine = AlertEngine()
        alert = engine.consider(change(), at=T0, attribution=0.6)
        assert alert is not None
        assert any("attributable" in caveat for caveat in alert.caveats)

    def test_partial_coverage_is_recorded_as_a_caveat(self) -> None:
        engine = AlertEngine()
        alert = engine.consider(change(), at=T0, coverage=0.6)
        assert alert is not None
        assert any("coverage" in caveat for caveat in alert.caveats)

    def test_a_pooled_reference_is_recorded_as_a_caveat(self) -> None:
        engine = AlertEngine()
        alert = engine.consider(change(weekday_aware=False), at=T0)
        assert alert is not None
        assert any("same-weekday" in caveat for caveat in alert.caveats)


class TestSeverity:
    def test_longer_and_larger_changes_score_higher(self) -> None:
        engine = AlertEngine(AlertPolicy(cooldown=timedelta(0)))
        mild = engine.consider(change(deviation=-3.5, duration_days=1), at=T0)
        severe = engine.consider(
            change(deviation=-12.0, duration_days=10), at=T0 + timedelta(days=1)
        )
        assert mild is not None and severe is not None
        assert severe.score > mild.score

    def test_severity_bands_follow_the_policy(self) -> None:
        policy = AlertPolicy(
            min_score=0.1, attention_score=0.4, urgent_score=0.8, cooldown=timedelta(0)
        )
        engine = AlertEngine(policy)
        urgent = engine.consider(change(deviation=-20.0, duration_days=14), at=T0)
        assert urgent is not None
        assert urgent.severity is AlertSeverity.URGENT

    def test_importance_weights_scale_the_score(self) -> None:
        quiet = AlertEngine(AlertPolicy(importance={"sleep_hours": 0.1}))
        loud = AlertEngine(AlertPolicy(importance={"sleep_hours": 1.0}))
        assert quiet.consider(change(), at=T0) is None
        assert loud.consider(change(), at=T0) is not None


class TestDeduplicationAndStorms:
    def test_a_repeat_of_the_same_finding_is_suppressed(self) -> None:
        engine = AlertEngine(AlertPolicy(cooldown=timedelta(hours=20)))
        assert engine.consider(change(), at=T0) is not None
        assert engine.consider(change(), at=T0 + timedelta(hours=2)) is None

    def test_the_same_finding_reappears_after_the_cooldown(self) -> None:
        engine = AlertEngine(AlertPolicy(cooldown=timedelta(hours=20)))
        engine.consider(change(), at=T0)
        assert engine.consider(change(), at=T0 + timedelta(hours=21)) is not None

    def test_an_escalating_finding_breaks_through_the_cooldown(self) -> None:
        policy = AlertPolicy(min_score=0.1, attention_score=0.35, urgent_score=0.65)
        engine = AlertEngine(policy)
        first = engine.consider(change(deviation=-3.2, duration_days=1), at=T0)
        second = engine.consider(
            change(deviation=-20.0, duration_days=14), at=T0 + timedelta(hours=1)
        )
        assert first is not None and second is not None
        assert second.severity is AlertSeverity.URGENT

    def test_a_burst_is_capped_and_summarised(self) -> None:
        policy = AlertPolicy(max_per_window=3, cooldown=timedelta(0))
        engine = AlertEngine(policy)
        raised = [
            engine.consider(change(feature=f"feature_{index}"), at=T0)
            for index in range(6)
        ]
        emitted = [alert for alert in raised if alert is not None]
        storms = [a for a in emitted if a.kind is AlertKind.DATA_QUALITY]
        assert len(emitted) == 4
        assert len(storms) == 1
        assert "withheld" in storms[0].summary

    def test_a_storm_notice_is_not_repeated(self) -> None:
        policy = AlertPolicy(max_per_window=2, cooldown=timedelta(0))
        engine = AlertEngine(policy)
        for index in range(8):
            engine.consider(change(feature=f"feature_{index}"), at=T0)
        later = engine.consider(change(feature="another"), at=T0 + timedelta(hours=1))
        assert later is None

    def test_review_returns_every_alert_raised(self) -> None:
        engine = AlertEngine(AlertPolicy(cooldown=timedelta(0)))
        alerts = engine.review(
            [change(feature="sleep_hours"), change(feature="kitchen_hours")], at=T0
        )
        assert {alert.subject for alert in alerts} == {"sleep_hours", "kitchen_hours"}


class TestSystemHealthAlerts:
    def test_a_healthy_deployment_does_not_alert(self) -> None:
        engine = AlertEngine()
        assert engine.consider_health(health({"a": 1.0, "b": 0.9})) is None

    def test_a_degraded_deployment_alerts_about_the_apparatus(self) -> None:
        engine = AlertEngine()
        alert = engine.consider_health(health({"a": 0.0, "b": 0.0, "c": 0.9}))
        assert alert is not None
        assert alert.kind is AlertKind.SYSTEM_HEALTH
        assert alert.subject == "deployment"
        assert sorted(alert.evidence["faulty"]) == ["a", "b"]  # type: ignore[arg-type]

    def test_a_health_alert_disclaims_any_behavioural_reading(self) -> None:
        """A broken sensor must never be presented as a finding about a person."""
        engine = AlertEngine()
        alert = engine.consider_health(health({"a": 0.0, "b": 0.0}))
        assert alert is not None
        assert any("not the resident" in caveat for caveat in alert.caveats)

    def test_health_alerts_are_deduplicated(self) -> None:
        engine = AlertEngine()
        report = health({"a": 0.0, "b": 0.0})
        assert engine.consider_health(report, at=T0) is not None
        assert engine.consider_health(report, at=T0 + timedelta(hours=1)) is None


class TestEngineMechanics:
    def test_alerts_serialise_with_a_stable_identifier(self) -> None:
        engine = AlertEngine()
        alert = engine.consider(change(), at=T0)
        assert alert is not None
        payload = alert.to_dict()
        assert payload["id"] == alert.identifier
        assert payload["kind"] == "behavioural_change"
        assert isinstance(payload["evidence"], dict)

    def test_snapshot_and_restore_preserve_deduplication(self) -> None:
        engine = AlertEngine()
        engine.consider(change(), at=T0)
        state = engine.snapshot()

        restarted = AlertEngine()
        restarted.restore(state)
        assert restarted.consider(change(), at=T0 + timedelta(hours=2)) is None

    def test_restore_rejects_malformed_payloads(self) -> None:
        engine = AlertEngine()
        with pytest.raises(TypeError):
            engine.restore({"last_emitted": []})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_score": 0.9, "attention_score": 0.5},
            {"min_confidence": 1.5},
            {"health_coverage_floor": -0.1},
            {"cooldown": timedelta(seconds=-1)},
            {"storm_window": timedelta(0)},
            {"max_per_window": 0},
            {"importance": {"a": -1.0}},
        ],
    )
    def test_invalid_policy_is_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            AlertPolicy(**kwargs)
