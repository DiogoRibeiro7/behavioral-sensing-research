"""Tests for replaying a run's days under another configuration.

The home is hand-built: three event sensors, a routine that changes part of
the way through, and a stretch in which nothing reports. It is not a simulated
household.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sensor_modeling.alerts import AlertKind, AlertPolicy
from sensor_modeling.baseline import BaselineConfig
from sensor_modeling.health import HealthConfig
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
)
from sensor_modeling.online import (
    BehaviouralSensingPipeline,
    PipelineConfig,
    PipelineStep,
    collect_alerts,
    daily_summaries,
    replay_days,
    reproduces,
)

START = datetime(2024, 3, 4, tzinfo=timezone.utc)
DAYS = 44
CHANGE_DAY = 26
# The record stops in the evening of its last day, so that the last day is
# closed by the pipeline's ``close`` and not by the arrival of the next.
END = START + timedelta(days=DAYS - 1, hours=20)
# Nothing is heard from the evening of day 15 until the morning of day 18.
LAST_HEARD = START + timedelta(days=15, hours=22)
HEARD_AGAIN = START + timedelta(days=18, hours=7)
CONFIG = PipelineConfig(tz=timezone.utc, step=timedelta(minutes=30))
# Longer than the home's longest night, which is thirteen hours after the change.
HEALTH = HealthConfig(home_silence_horizon=timedelta(hours=16))
PLAIN = BaselineConfig(min_samples=5)
CALIBRATED = BaselineConfig(min_samples=5, calibrated=True, deviation_threshold=2.0)


def home() -> SensorRegistry:
    """Three event sensors and nothing that reports on a cadence."""
    return SensorRegistry.from_specs(
        [
            SensorSpec("front_door", Modality.DOOR, room="hall"),
            SensorSpec("living_motion", Modality.MOTION, room="living"),
            SensorSpec("kitchen_motion", Modality.MOTION, room="kitchen"),
        ]
    )


def record() -> list[Observation]:
    """Motion through the day, from 07:00 and then, after the change, from 11:00.

    The hour the day begins at moves a little from day to day, so that the
    baseline has a spread to compare against.
    """
    records = []
    for day in range(DAYS):
        wobble = 20 * ((day * 3) % 4)
        first = (11 * 60 if day >= CHANGE_DAY else 7 * 60) + wobble
        for minute in range(first, 23 * 60, 20):
            moment = START + timedelta(days=day, minutes=minute)
            if LAST_HEARD < moment < HEARD_AGAIN or moment >= END:
                continue
            records.append(
                Observation(
                    timestamp=moment,
                    sensor_id=(
                        "kitchen_motion" if minute % 240 == 0 else "living_motion"
                    ),
                    modality=Modality.MOTION,
                    kind=ObservationKind.EVENT,
                    value=1.0,
                )
            )
    return records


def run(
    baseline_config: BaselineConfig = PLAIN, alert_policy: AlertPolicy | None = None
) -> list[PipelineStep]:
    """Run the home through the pipeline with the silent-home rule on."""
    pipeline = BehaviouralSensingPipeline(
        home(),
        config=CONFIG,
        health_config=HEALTH,
        baseline_config=baseline_config,
        alert_policy=alert_policy,
    )
    steps = pipeline.run(record())
    steps.extend(pipeline.close(END))
    return steps


def kinds(steps: list[PipelineStep]) -> list[AlertKind]:
    return [alert.kind for alert in collect_alerts(steps)]


@pytest.fixture(scope="module")
def plain() -> list[PipelineStep]:
    return run()


@pytest.fixture(scope="module")
def calibrated() -> list[PipelineStep]:
    return run(CALIBRATED)


class TestReplay:
    def test_the_home_raises_alerts_of_both_kinds(
        self, plain: list[PipelineStep]
    ) -> None:
        """Otherwise the tests below would compare two empty lists."""
        raised = kinds(plain)
        assert AlertKind.BEHAVIOURAL_CHANGE in raised
        assert AlertKind.DATA_QUALITY in raised

    def test_a_replay_under_the_run_s_own_configuration_is_the_run(
        self, plain: list[PipelineStep]
    ) -> None:
        replayed = replay_days(plain, CONFIG, baseline_config=PLAIN)
        assert reproduces(plain, replayed)
        assert [a.to_dict() for step in replayed for a in step.alerts] == [
            a.to_dict() for a in collect_alerts(plain)
        ]

    def test_the_days_do_not_depend_on_the_baseline(
        self, plain: list[PipelineStep], calibrated: list[PipelineStep]
    ) -> None:
        assert [day.to_dict() for day in daily_summaries(plain)] == [
            day.to_dict() for day in daily_summaries(calibrated)
        ]

    def test_a_replay_under_another_baseline_is_the_run_made_with_it(
        self, plain: list[PipelineStep], calibrated: list[PipelineStep]
    ) -> None:
        replayed = replay_days(plain, CONFIG, baseline_config=CALIBRATED)
        assert reproduces(calibrated, replayed)
        # The two configurations do not conclude the same, so the replay is
        # not passing by repeating the run it was given.
        assert not reproduces(plain, replayed)

    def test_a_replay_under_another_alert_policy_is_the_run_made_with_it(
        self, plain: list[PipelineStep]
    ) -> None:
        policy = AlertPolicy(
            cooldown=timedelta(hours=1),
            storm_window=timedelta(hours=30),
            max_per_window=1,
        )
        made = run(PLAIN, policy)
        replayed = replay_days(
            plain, CONFIG, baseline_config=PLAIN, alert_policy=policy
        )
        assert reproduces(made, replayed)
        assert not reproduces(plain, replayed)
        assert "alert_rate" in {alert.subject for alert in collect_alerts(made)}

    def test_the_record_s_last_day_is_replayed(self, plain: list[PipelineStep]) -> None:
        replayed = replay_days(plain, CONFIG, baseline_config=PLAIN)
        closed = [step.day_closed.day for step in replayed if step.day_closed]
        assert closed == [day.day for day in daily_summaries(plain)]
        assert plain[-1].day_closed is not None
        assert plain[-1].state is plain[-2].state

    def test_only_steps_that_concluded_something_are_kept(
        self, plain: list[PipelineStep]
    ) -> None:
        replayed = replay_days(plain, CONFIG, baseline_config=PLAIN)
        assert all(step.day_closed is not None or step.alerts for step in replayed)
        assert len(replayed) < len(plain)

    def test_the_defaults_are_the_pipeline_s(self) -> None:
        pipeline = BehaviouralSensingPipeline(home(), config=CONFIG)
        steps = pipeline.run(record())
        assert reproduces(steps, replay_days(steps, CONFIG))

    def test_nothing_replayed_is_nothing(self) -> None:
        assert replay_days([]) == []
        assert reproduces([], [])
