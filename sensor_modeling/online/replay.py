"""Ask what another baseline or alert configuration would have concluded.

The pipeline's inference runs from observations to a summary of each day, and
none of it depends on how the personal baseline or the alert engine is
configured: those two read the day's summary and nothing upstream of it. So
the question "what would this home's alerts have been at another threshold"
does not need the home to be run again. :func:`replay_days` takes the steps a
pipeline produced and passes the same days, at the same moments, through
fresh baselines and a fresh alert engine under the configuration asked for.

It calls the function the pipeline itself calls when it closes a day,
:func:`~sensor_modeling.online.pipeline.judge_day`, and it reviews every
health report in the order the pipeline did, because the alert engine counts
its recent alerts of every kind before it lets another through. Replayed under
the configuration the steps were produced with, it returns the pipeline's own
verdicts and alerts; :func:`reproduces` checks that.

A replay answers a question about thresholds. It cannot answer one about
anything upstream of the day's summary: a different step, another emission
model, the rule for a silent home. Those change the summaries, and the home
must be run again.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from ..alerts.alert import Alert, AlertEngine, AlertPolicy
from ..baseline.adaptive import AdaptiveBaseline, BaselineConfig, BehaviouralChange
from ..baseline.features import DailySummary
from .pipeline import PipelineConfig, PipelineStep, judge_day


@dataclass(frozen=True)
class ReplayedStep:
    """What a step concluded under the replayed configuration.

    Only the steps at which something was concluded are kept: those that
    closed a day, and those that raised an alert.

    Attributes
    ----------
    at
        The moment of the step.
    day_closed
        The day the step closed, as the pipeline summarised it, if it closed
        one.
    changes
        The baselines' verdicts on that day.
    alerts
        Every alert of the step: those the day's review raised, then the one
        about the apparatus, as the pipeline orders them.
    """

    at: datetime
    day_closed: DailySummary | None
    changes: tuple[BehaviouralChange, ...]
    alerts: tuple[Alert, ...]


def replay_days(
    steps: Iterable[PipelineStep],
    config: PipelineConfig | None = None,
    *,
    baseline_config: BaselineConfig | None = None,
    alert_policy: AlertPolicy | None = None,
) -> list[ReplayedStep]:
    """Replay a run's days through fresh baselines and a fresh alert engine.

    Parameters
    ----------
    steps
        Every step of one run, from its first, in order. The baselines and the
        alert engine start empty, as the pipeline's did.
    config
        The configuration the run was made with. Its features and its tests of
        a usable day are applied as the pipeline applied them.
    baseline_config, alert_policy
        The configuration to replay under. The defaults are the pipeline's.

    Returns
    -------
    list[ReplayedStep]
        The steps that closed a day or raised an alert, in order.
    """
    config = config or PipelineConfig()
    baseline_config = baseline_config or BaselineConfig()
    engine = AlertEngine(alert_policy)
    baselines = {
        state.value: AdaptiveBaseline(f"{state.value}_hours", baseline_config)
        for state in config.features
    }
    replayed: list[ReplayedStep] = []
    previous: PipelineStep | None = None
    for step in steps:
        changes: tuple[BehaviouralChange, ...] = ()
        alerts: tuple[Alert, ...] = ()
        if step.day_closed is not None:
            changes, alerts = judge_day(
                step.day_closed,
                step.at,
                attribution=step.context.ambient_attribution(),
                baselines=baselines,
                engine=engine,
                config=config,
                baseline_config=baseline_config,
            )
        # The step that closes a record repeats the estimate of the step
        # before it, and the pipeline reviews no health report there.
        if previous is None or step.state is not previous.state:
            health_alert = engine.consider_health(step.health, at=step.at)
            if health_alert is not None:
                alerts = alerts + (health_alert,)
        if step.day_closed is not None or alerts:
            replayed.append(ReplayedStep(step.at, step.day_closed, changes, alerts))
        previous = step
    return replayed


def reproduces(steps: Iterable[PipelineStep], replayed: Sequence[ReplayedStep]) -> bool:
    """Whether a replay returned the verdicts and alerts the run itself made.

    True of a replay under the configuration the run was made with. It is the
    check to make before a replay under another configuration is believed.
    """
    made = [
        (
            step.at,
            [change.to_dict() for change in step.changes],
            [alert.to_dict() for alert in step.alerts],
        )
        for step in steps
        if step.day_closed is not None or step.alerts
    ]
    again = [
        (
            step.at,
            [change.to_dict() for change in step.changes],
            [alert.to_dict() for alert in step.alerts],
        )
        for step in replayed
    ]
    return _same(made, again)


def _same(first: object, second: object) -> bool:
    """Equality in which a value that is not a number equals itself."""
    if isinstance(first, float) and isinstance(second, float):
        return first == second or (first != first and second != second)
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            _same(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(
            _same(a, b) for a, b in zip(first, second)
        )
    return first == second
