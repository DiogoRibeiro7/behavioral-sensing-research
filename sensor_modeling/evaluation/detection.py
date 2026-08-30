"""Measuring whether behavioural change detection is usable.

Four numbers decide whether a monitoring system is worth deploying, and they
trade against each other rather than improving together:

.. code-block:: text

    detection delay          how long a real change goes unreported
    false alert burden       how often a carer is contacted for nothing
    missed changes           how often a real change is never reported
    calibration              whether the confidence attached means anything

A system tuned to detect everything floods its recipients; one tuned for
silence misses what matters. This module measures the trade rather than
asserting a favourable point on it, and includes arms where nothing at all has
changed, because a detector's behaviour on a stable record is as informative
as its behaviour on a changed one.

Arms are paired by seed, so the stable and changed runs of a given seed differ
only in the injected change.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Any

import numpy as np

from ..alerts.alert import AlertKind
from ..online.pipeline import BehaviouralSensingPipeline, PipelineConfig, collect_alerts
from ..simulation.faults import DegradationConfig, degrade
from ..simulation.household import BehaviourShift, HouseholdConfig, simulate
from .metrics import DetectionMetrics, detection_metrics

logger = logging.getLogger(__name__)

#: Feature the injected changes act on. Sleep is used because the simulator
#: can shift it cleanly and because it is the quantity most ambient-monitoring
#: studies report on.
TRACKED_FEATURE = "sleeping_hours"


@dataclass(frozen=True)
class ChangeArm:
    """One configuration of injected change and sensor degradation."""

    name: str
    shift: BehaviourShift | None
    degradation: DegradationConfig | None = None
    description: str = ""

    @property
    def has_change(self) -> bool:
        """Whether a real behavioural change was injected."""
        return self.shift is not None


@dataclass(frozen=True)
class ArmOutcome:
    """What one arm produced on one seed."""

    arm: str
    seed: int
    metrics: DetectionMetrics
    behavioural_alerts: int
    person_days: float
    mean_alert_confidence: float

    @property
    def alerts_per_person_day(self) -> float:
        """Total behavioural alert burden, matched or not."""
        return self.behavioural_alerts / self.person_days if self.person_days else 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the outcome."""
        return {
            "arm": self.arm,
            "seed": self.seed,
            "behavioural_alerts": self.behavioural_alerts,
            "alerts_per_person_day": self.alerts_per_person_day,
            "mean_alert_confidence": self.mean_alert_confidence,
            "detection": self.metrics.to_dict(),
        }


@dataclass
class DetectionStudy:
    """Detection outcomes across arms and seeds."""

    outcomes: list[ArmOutcome] = field(default_factory=list)

    def arms(self) -> list[str]:
        """Arm names in first-seen order."""
        seen: dict[str, None] = {}
        for outcome in self.outcomes:
            seen.setdefault(outcome.arm, None)
        return list(seen)

    def for_arm(self, arm: str) -> list[ArmOutcome]:
        """Every outcome for one arm, ordered by seed."""
        return sorted((o for o in self.outcomes if o.arm == arm), key=lambda o: o.seed)

    def summary(self, arm: str) -> dict[str, float]:
        """Aggregate one arm across its seeds."""
        outcomes = self.for_arm(arm)
        if not outcomes:
            raise KeyError(f"no outcomes for arm '{arm}'")
        # Pool the individual delays rather than averaging each seed's median.
        # A mean of medians is not a median, it weights a seed that detected one
        # change as heavily as a seed that detected twenty, and it is not the
        # quantity the provenance record defines "median_delay_days" to be.
        delays = [delay for o in outcomes for delay in o.metrics.delays_days]
        seed_medians = [
            o.metrics.median_delay_days
            for o in outcomes
            if not np.isnan(o.metrics.median_delay_days)
        ]
        return {
            "seeds": float(len(outcomes)),
            "recall": float(np.mean([o.metrics.recall for o in outcomes])),
            "median_delay_days": float(np.median(delays)) if delays else float("nan"),
            "mean_seed_median_delay_days": (
                float(np.mean(seed_medians)) if seed_medians else float("nan")
            ),
            "detected_changes": float(len(delays)),
            "alerts_per_person_day": float(
                np.mean([o.alerts_per_person_day for o in outcomes])
            ),
            "false_positives_per_person_day": float(
                np.mean([o.metrics.false_positives_per_person_day for o in outcomes])
            ),
            "mean_alert_confidence": float(
                np.mean([o.mean_alert_confidence for o in outcomes])
            ),
        }

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the study."""
        return {
            "arms": {arm: self.summary(arm) for arm in self.arms()},
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def standard_arms(
    days: int = 70, change_day: int = 40, magnitude: float = 1.6
) -> list[ChangeArm]:
    """Build the arms needed to characterise the detection trade-off.

    Every arm that injects a change is matched by one that does not, so the
    alert burden attributable to the change is separable from the burden the
    detector produces anyway.
    """
    step = BehaviourShift(
        start_day=change_day,
        sleep_delta_hours=magnitude,
        night_bathroom_extra=1.2,
    )
    gradual = BehaviourShift(
        start_day=change_day // 2,
        sleep_delta_hours=magnitude,
        night_bathroom_extra=1.2,
        ramp_days=max(days - change_day // 2 - 5, 5),
    )
    missing = DegradationConfig(missing_rate=0.3, seed=1)

    return [
        ChangeArm(
            "stable",
            None,
            description="Nothing changed. Every alert here is a false alarm.",
        ),
        ChangeArm(
            "abrupt_change",
            step,
            description="A step change in sleep on a known day.",
        ),
        ChangeArm(
            "gradual_change",
            gradual,
            description="The same magnitude reached slowly, which no single "
            "day's deviation can reveal.",
        ),
        ChangeArm(
            "stable_degraded",
            None,
            missing,
            description="Nothing changed, and a third of records are lost. "
            "Missingness must not manufacture findings.",
        ),
        ChangeArm(
            "abrupt_change_degraded",
            step,
            missing,
            description="A real change seen through a degraded record.",
        ),
    ]


def _evaluate(
    arm: ChangeArm,
    seed: int,
    *,
    days: int,
    change_day: int,
    step: timedelta,
    max_delay_days: float,
) -> ArmOutcome:
    """Run one arm on one seed and score the alerts it delivered."""
    household = HouseholdConfig(days=days, seed=seed, shift=arm.shift)
    result = simulate(household)

    observations: Sequence[Any] = result.observations
    if arm.degradation is not None:
        observations, _ = degrade(observations, replace(arm.degradation, seed=seed))

    pipeline = BehaviouralSensingPipeline(
        result.registry, config=PipelineConfig(tz=result.config.tz, step=step)
    )
    steps = pipeline.run(observations)
    steps.extend(pipeline.close(result.end))

    alerts = [
        alert
        for alert in collect_alerts(steps)
        if alert.kind is AlertKind.BEHAVIOURAL_CHANGE
        and TRACKED_FEATURE in str(alert.subject)
    ]
    detected: list[date] = [alert.at.date() for alert in alerts]

    true_changes: list[date] = []
    if arm.shift is not None:
        true_changes.append(household.start + timedelta(days=arm.shift.start_day))

    metrics = detection_metrics(
        detected,
        true_changes,
        person_days=float(days),
        max_delay_days=max_delay_days,
    )
    return ArmOutcome(
        arm=arm.name,
        seed=seed,
        metrics=metrics,
        behavioural_alerts=len(alerts),
        person_days=float(days),
        mean_alert_confidence=(
            float(np.mean([a.confidence for a in alerts])) if alerts else float("nan")
        ),
    )


def run_detection_study(
    arms: Iterable[ChangeArm] | None = None,
    *,
    seeds: Iterable[int] = (11, 22, 33),
    days: int = 70,
    change_day: int = 40,
    step: timedelta = timedelta(minutes=15),
    max_delay_days: float = 21.0,
) -> DetectionStudy:
    """Characterise detection delay against alert burden across arms.

    A gradual arm's change is dated from where it *begins*, not from where it
    becomes visible, so its reported delay includes the time the trend spent
    too small to detect. That is the honest accounting: the resident was
    already declining.
    """
    selected = list(arms) if arms is not None else standard_arms(days, change_day)
    if not selected:
        raise ValueError("at least one arm is required")

    study = DetectionStudy()
    for seed in seeds:
        for arm in selected:
            outcome = _evaluate(
                arm,
                seed,
                days=days,
                change_day=change_day,
                step=step,
                max_delay_days=max_delay_days,
            )
            study.outcomes.append(outcome)
            logger.info(
                "seed %s | %-24s | alerts %d | recall %.2f | delay %.1f",
                seed,
                arm.name,
                outcome.behavioural_alerts,
                outcome.metrics.recall,
                outcome.metrics.median_delay_days,
            )
    return study
