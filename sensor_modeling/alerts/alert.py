"""Structured, explainable alerts with deliberate restraint.

Four things are kept strictly separate in this codebase, and this module is
where the last boundary is drawn:

.. code-block:: text

    observation          a sensor reported something
    state                the resident is probably doing something
    behavioural change   that something has shifted against their own history
    alert                a person should look at this

An unusual observation is not an alert. A behavioural change is not
automatically an alert either. Raising one is a claim on somebody's attention,
and in this domain a stream of false alarms is not a minor annoyance -- it is
the failure mode that gets monitoring switched off entirely.

So an alert has to survive several filters. The change must be large enough,
have lasted long enough, have been observed well enough, and be attributable
to the resident rather than to a visitor. Alerts that repeat are deduplicated,
and a burst is capped rather than delivered.

Behavioural alerts and system-health alerts are separate kinds. A failing
sensor produces a system-health alert about the apparatus, never a behavioural
alert about the resident. Alerts describe observed changes in sensor-derived
behaviour; they do not diagnose.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..baseline.adaptive import BehaviouralChange, ChangeKind
from ..health.monitor import SystemHealthReport

logger = logging.getLogger(__name__)


class AlertKind(str, Enum):
    """What an alert is about."""

    BEHAVIOURAL_CHANGE = "behavioural_change"
    """A sustained change in the resident's own behavioural pattern."""

    SYSTEM_HEALTH = "system_health"
    """The sensing apparatus is degraded. Says nothing about the resident."""

    DATA_QUALITY = "data_quality"
    """Too little was observed to say anything about behaviour at all."""


class AlertSeverity(str, Enum):
    """How much attention an alert asks for."""

    INFORMATION = "information"
    ATTENTION = "attention"
    URGENT = "urgent"


@dataclass(frozen=True)
class Alert:
    """A structured, explainable request for human attention.

    Attributes
    ----------
    at
        When the alert was raised.
    kind
        Whether this concerns behaviour, the apparatus, or data quality.
    severity
        How much attention is being asked for.
    subject
        The feature, sensor, or subsystem the alert is about.
    summary
        One-line description, phrased as an observation rather than a
        diagnosis.
    score
        The graded strength that produced the severity, in ``[0, 1]``.
    confidence
        How well-evidenced the alert is, combining sensor coverage and
        attribution.
    evidence
        Structured detail supporting the alert.
    caveats
        Reasons to treat the alert cautiously, stated explicitly rather than
        left for the reader to infer.
    """

    at: datetime
    kind: AlertKind
    severity: AlertSeverity
    subject: str
    summary: str
    score: float
    confidence: float
    evidence: Mapping[str, object] = field(default_factory=dict)
    caveats: tuple[str, ...] = ()

    @property
    def dedup_key(self) -> str:
        """Stable key identifying repeats of the same finding."""
        return f"{self.kind.value}:{self.subject}"

    @property
    def identifier(self) -> str:
        """Deterministic identifier for this alert instance."""
        digest = hashlib.sha256(
            f"{self.dedup_key}|{self.at.isoformat()}|{self.severity.value}".encode()
        )
        return digest.hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the alert."""
        return {
            "id": self.identifier,
            "at": self.at.isoformat(),
            "kind": self.kind.value,
            "severity": self.severity.value,
            "subject": self.subject,
            "summary": self.summary,
            "score": self.score,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
            "caveats": list(self.caveats),
        }


@dataclass
class AlertPolicy:
    """Thresholds and restraints governing when an alert is raised.

    Parameters
    ----------
    min_score
        Graded strength below which nothing is raised at all.
    attention_score, urgent_score
        Strength thresholds for the two higher severities.
    min_confidence
        Combined coverage-and-attribution confidence required before a
        behavioural alert may be raised. Below this the change is real as an
        observation but not attributable well enough to act on.
    importance
        Per-feature multiplier. Some behavioural features matter more than
        others, and the deployment says which rather than the algorithm.
    cooldown
        Period during which a repeat of the same finding is suppressed,
        unless its severity has increased.
    storm_window, max_per_window
        A burst larger than this is capped, and a single summary alert is
        raised in place of the flood.
    health_coverage_floor
        System coverage below which a system-health alert is raised.
    """

    min_score: float = 0.25
    attention_score: float = 0.45
    urgent_score: float = 0.7
    min_confidence: float = 0.4
    importance: Mapping[str, float] = field(default_factory=dict)
    cooldown: timedelta = timedelta(hours=20)
    storm_window: timedelta = timedelta(hours=24)
    max_per_window: int = 6
    health_coverage_floor: float = 0.5

    def __post_init__(self) -> None:
        """Validate the policy."""
        if not 0.0 <= self.min_score < self.attention_score < self.urgent_score <= 1.0:
            raise ValueError(
                "scores must satisfy 0 <= min_score < attention_score < "
                "urgent_score <= 1"
            )
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must lie in [0, 1]")
        if not 0.0 <= self.health_coverage_floor <= 1.0:
            raise ValueError("health_coverage_floor must lie in [0, 1]")
        if self.cooldown < timedelta(0):
            raise ValueError("cooldown must be non-negative")
        if self.storm_window <= timedelta(0):
            raise ValueError("storm_window must be positive")
        if self.max_per_window < 1:
            raise ValueError("max_per_window must be at least 1")
        if any(float(v) < 0.0 for v in self.importance.values()):
            raise ValueError("importance weights must be non-negative")

    def severity_for(self, score: float) -> AlertSeverity:
        """Return the severity band a graded score falls into."""
        if score >= self.urgent_score:
            return AlertSeverity.URGENT
        if score >= self.attention_score:
            return AlertSeverity.ATTENTION
        return AlertSeverity.INFORMATION


class AlertEngine:
    """Turn behavioural changes and health reports into restrained alerts.

    The engine holds bounded state -- the last emission per finding plus a
    short window of recent alerts -- so it runs indefinitely on an edge
    device and survives a restart through :meth:`snapshot`.
    """

    def __init__(self, policy: AlertPolicy | None = None) -> None:
        self.policy = policy or AlertPolicy()
        self._last_emitted: dict[str, tuple[datetime, AlertSeverity]] = {}
        self._recent: list[datetime] = []
        self._storm_notified: datetime | None = None

    # ------------------------------------------------------------------
    def _grade(
        self,
        change: BehaviouralChange,
        deviation_threshold: float,
        trend_threshold: float,
    ) -> float:
        """Score a behavioural change on how large it is and how long it held.

        Both matter and neither is sufficient: a huge one-day excursion is a
        disturbance, and a marginal shift that holds for a fortnight is worth
        more attention than its size alone suggests.

        A gradual drift is graded differently, because it has no deviation
        streak to measure -- that is precisely what distinguishes it from a
        step. Grading it on the same axes would score every slow decline at
        zero and make the most clinically interesting pattern in ambient
        monitoring permanently unalertable. Its magnitude is the movement
        that identified it, and it carries a duration credit by
        construction, since a drift is only ever declared across a whole
        trend window.

        The offset is set so that a drift which has only just crossed the
        baseline's threshold lands in the middle band rather than the top
        one: qualifying as a drift is a low bar, and a slow decline that has
        only just become visible does not warrant the same response as one
        moving twice as fast.
        """
        if change.kind is ChangeKind.GRADUAL_DRIFT:
            magnitude = min(change.trend_strength / (2.0 * trend_threshold), 1.0)
            return 0.35 + 0.5 * magnitude
        magnitude = min(abs(change.deviation) / (2.0 * deviation_threshold), 1.0)
        duration = min(change.duration_days / 7.0, 1.0)
        return 0.5 * magnitude + 0.5 * duration

    def _suppressed(self, key: str, at: datetime, severity: AlertSeverity) -> bool:
        """Whether this finding was raised too recently to raise again."""
        previous = self._last_emitted.get(key)
        if previous is None:
            return False
        last_at, last_severity = previous
        if at - last_at >= self.policy.cooldown:
            return False
        escalated = _severity_rank(severity) > _severity_rank(last_severity)
        return not escalated

    def _prune(self, at: datetime) -> None:
        """Drop recent-alert records that have left the storm window."""
        cutoff = at - self.policy.storm_window
        self._recent = [moment for moment in self._recent if moment > cutoff]

    def _storming(self, at: datetime) -> bool:
        """Whether the recent alert rate has exceeded the cap."""
        self._prune(at)
        return len(self._recent) >= self.policy.max_per_window

    def _record(self, alert: Alert) -> Alert:
        """Register an emitted alert for deduplication and rate control."""
        self._last_emitted[alert.dedup_key] = (alert.at, alert.severity)
        self._recent.append(alert.at)
        return alert

    # ------------------------------------------------------------------
    def consider(
        self,
        change: BehaviouralChange,
        *,
        at: datetime,
        coverage: float = 1.0,
        attribution: float = 1.0,
        deviation_threshold: float = 3.0,
        trend_threshold: float = 3.5,
    ) -> Alert | None:
        """Decide whether a behavioural change warrants an alert.

        Parameters
        ----------
        change
            The verdict from the adaptive baseline.
        at
            When the decision is being made.
        coverage
            Mean sensor coverage behind the change, in ``[0, 1]``.
        attribution
            Probability the change reflects the resident's own behaviour
            rather than a visitor's, in ``[0, 1]``.
        deviation_threshold
            The baseline's deviation threshold, used to scale magnitude.
        trend_threshold
            The baseline's trend threshold, used to scale a drift.

        Returns
        -------
        Alert | None
            An alert, or ``None`` when the change does not warrant one.
        """
        if not change.is_change:
            return None

        confidence = float(coverage) * float(attribution)
        importance = float(self.policy.importance.get(change.feature, 1.0))
        score = min(
            importance
            * confidence
            * self._grade(change, deviation_threshold, trend_threshold),
            1.0,
        )

        if confidence < self.policy.min_confidence:
            logger.debug(
                "Change in '%s' not attributable enough to alert (confidence %.2f)",
                change.feature,
                confidence,
            )
            return None
        if score < self.policy.min_score:
            return None

        severity = self.policy.severity_for(score)
        key = f"{AlertKind.BEHAVIOURAL_CHANGE.value}:{change.feature}"
        if self._suppressed(key, at, severity):
            logger.debug("Suppressing repeat alert for '%s'", change.feature)
            return None
        if self._storming(at):
            return self._storm_alert(at)

        caveats = []
        if coverage < 0.8:
            caveats.append(f"sensor coverage was {coverage:.0%} over the period")
        if attribution < 0.8:
            caveats.append(
                f"only {attribution:.0%} of the activity is attributable to the resident"
            )
        if not change.reference.weekday_aware:
            caveats.append(
                "compared against pooled history; not enough same-weekday days yet"
            )

        return self._record(
            Alert(
                at=at,
                kind=AlertKind.BEHAVIOURAL_CHANGE,
                severity=severity,
                subject=change.feature,
                summary=_summarise(change),
                score=score,
                confidence=confidence,
                evidence={
                    "change": change.to_dict(),
                    "coverage": coverage,
                    "attribution": attribution,
                },
                caveats=tuple(caveats),
            )
        )

    def consider_health(
        self, report: SystemHealthReport, *, at: datetime | None = None
    ) -> Alert | None:
        """Decide whether the sensing apparatus warrants an alert.

        This is deliberately a different kind of alert. A failing sensor is a
        maintenance problem; presenting it as a behavioural finding would be
        exactly the confusion the platform exists to prevent.
        """
        moment = at if at is not None else report.at
        if report.coverage >= self.policy.health_coverage_floor:
            return None

        shortfall = 1.0 - report.coverage
        severity = self.policy.severity_for(shortfall)
        key = f"{AlertKind.SYSTEM_HEALTH.value}:deployment"
        if self._suppressed(key, moment, severity):
            return None

        faulty = report.faulty
        return self._record(
            Alert(
                at=moment,
                kind=AlertKind.SYSTEM_HEALTH,
                severity=severity,
                subject="deployment",
                summary=(
                    f"Sensor coverage has fallen to {report.coverage:.0%}; "
                    f"{len(faulty)} sensor(s) are not reporting usable data"
                ),
                score=shortfall,
                confidence=1.0,
                evidence={"faulty": faulty, "coverage": report.coverage},
                caveats=(
                    "This concerns the sensing apparatus, not the resident. "
                    "Behavioural conclusions over this period are unreliable.",
                ),
            )
        )

    def _storm_alert(self, at: datetime) -> Alert | None:
        """Replace a burst of alerts with a single notice about the burst."""
        if (
            self._storm_notified is not None
            and at - self._storm_notified < self.policy.storm_window
        ):
            return None
        self._storm_notified = at
        logger.warning("Alert rate exceeded %d per window", self.policy.max_per_window)
        return Alert(
            at=at,
            kind=AlertKind.DATA_QUALITY,
            severity=AlertSeverity.ATTENTION,
            subject="alert_rate",
            summary=(
                f"More than {self.policy.max_per_window} alerts were raised within "
                "the review window; further alerts are being withheld"
            ),
            score=1.0,
            confidence=1.0,
            evidence={"recent_alerts": len(self._recent)},
            caveats=(
                "A burst of alerts usually indicates a sensing or configuration "
                "problem rather than a sudden change in the resident.",
            ),
        )

    def review(
        self,
        changes: Sequence[BehaviouralChange],
        *,
        at: datetime,
        coverage: float = 1.0,
        attribution: float = 1.0,
        deviation_threshold: float = 3.0,
        trend_threshold: float = 3.5,
    ) -> list[Alert]:
        """Consider several changes at once, returning the alerts raised."""
        raised = []
        for change in changes:
            alert = self.consider(
                change,
                at=at,
                coverage=coverage,
                attribution=attribution,
                deviation_threshold=deviation_threshold,
                trend_threshold=trend_threshold,
            )
            if alert is not None:
                raised.append(alert)
        return raised

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, object]:
        """Return restartable engine state."""
        return {
            "last_emitted": {
                key: [moment.isoformat(), severity.value]
                for key, (moment, severity) in self._last_emitted.items()
            },
            "recent": [moment.isoformat() for moment in self._recent],
            "storm_notified": (
                self._storm_notified.isoformat() if self._storm_notified else None
            ),
        }

    def restore(self, state: Mapping[str, object]) -> None:
        """Restore engine state produced by :meth:`snapshot`."""
        emitted = state.get("last_emitted", {})
        if emitted is None:
            emitted = {}
        if not isinstance(emitted, Mapping):
            raise TypeError("snapshot 'last_emitted' must be a mapping")
        self._last_emitted = {
            str(key): (
                datetime.fromisoformat(str(value[0])),
                AlertSeverity(str(value[1])),
            )
            for key, value in emitted.items()
        }
        recent = state.get("recent", [])
        if recent is None:
            recent = []
        if isinstance(recent, str) or not isinstance(recent, Sequence):
            raise TypeError("snapshot 'recent' must be a sequence of timestamps")
        self._recent = [datetime.fromisoformat(str(moment)) for moment in recent]
        notified = state.get("storm_notified")
        self._storm_notified = (
            datetime.fromisoformat(str(notified)) if notified else None
        )


def _summarise(change: BehaviouralChange) -> str:
    """Phrase a change as an observation about sensor-derived behaviour.

    Deliberately descriptive: it reports what was measured against what the
    resident's own history predicted, and offers no explanation for it.
    """
    if change.kind is ChangeKind.GRADUAL_DRIFT:
        return (
            f"Gradual {change.direction} in {change.feature}: trending "
            f"{change.slope_per_day:+.3f} per day, {change.trend_strength:.1f} "
            f"robust SD of movement against a personal reference of "
            f"{change.reference.centre:.2f}"
        )
    return (
        f"Sustained {change.direction} in {change.feature}: {change.value:.2f} "
        f"against a personal reference of {change.reference.centre:.2f}, held "
        f"for {change.duration_days} day(s)"
    )


def _severity_rank(severity: AlertSeverity) -> int:
    """Return an ordering rank for a severity band."""
    return [
        AlertSeverity.INFORMATION,
        AlertSeverity.ATTENTION,
        AlertSeverity.URGENT,
    ].index(severity)


def unresolved_kinds(changes: Sequence[BehaviouralChange]) -> set[ChangeKind]:
    """Return the distinct change kinds present in *changes*, for reporting."""
    return {change.kind for change in changes}
