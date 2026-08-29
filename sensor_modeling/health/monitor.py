"""Online estimation of sensor health from the observation stream.

:class:`SensorHealthMonitor` watches ingested observations and maintains, per
sensor, an interpretable verdict about whether that sensor can currently be
trusted. It holds bounded state -- a few short deques per sensor -- so it runs
unchanged on an edge device for months.

Three design choices matter scientifically:

*Silence is only evidence of failure when the sensor promised to speak.* A
sensor that declares an ``expected_interval`` is expected to report on that
cadence, so prolonged silence is diagnostic. A purely event-driven contact
sensor makes no such promise, and its silence is genuinely ambiguous between
"broken" and "nobody opened the cupboard". For those sensors the monitor
declines to call a failure and leaves the status at ``UNKNOWN``.

*Verdicts are separated from behaviour.* The monitor reads values but never
interprets them as activity. Its output is consumed by fusion as an evidence
weight, which is what stops a dead sensor from being read as a quiet resident.

*Drift is reported, not corrected.* The monitor cannot distinguish sensor
drift from a genuine environmental change without redundant sensing, so it
flags the shift and leaves the judgement to the analyst.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

from ..observations.observation import Observation
from ..observations.registry import SensorRegistry, SensorSpec
from ..observations.types import ObservationKind
from .status import FAULTY_STATUSES, SensorStatus, status_reliability

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensorHealthReport:
    """The health verdict for one sensor at one moment.

    Attributes
    ----------
    sensor_id
        Sensor the verdict refers to.
    status
        Assessed operating condition.
    reliability
        Evidence weight in ``[0, 1]`` that fusion should apply to this
        sensor's observations.
    last_seen
        Timestamp of the most recent observation, if any.
    silence
        How long the sensor has been quiet at the time of the report.
    observations
        Number of observations seen since the monitor was started or reset.
    detail
        Short human-readable explanation of the verdict.
    """

    sensor_id: str
    status: SensorStatus
    reliability: float
    last_seen: datetime | None
    silence: timedelta | None
    observations: int
    detail: str

    @property
    def is_faulty(self) -> bool:
        """Whether the sensor currently cannot be trusted as evidence."""
        return self.status in FAULTY_STATUSES

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the report."""
        return {
            "sensor_id": self.sensor_id,
            "status": self.status.value,
            "reliability": self.reliability,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "silence_seconds": (
                self.silence.total_seconds() if self.silence is not None else None
            ),
            "observations": self.observations,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SystemHealthReport:
    """Deployment-wide health, observable independently of behaviour."""

    at: datetime
    sensors: dict[str, SensorHealthReport]

    @property
    def faulty(self) -> list[str]:
        """Sensors that currently cannot be trusted, sorted by identifier."""
        return sorted(sid for sid, r in self.sensors.items() if r.is_faulty)

    @property
    def coverage(self) -> float:
        """Mean reliability across registered sensors, in ``[0, 1]``.

        This is a system-integrity measure. It says how much of the sensing
        apparatus is working, and says nothing at all about the resident.
        """
        if not self.sensors:
            return 0.0
        return sum(r.reliability for r in self.sensors.values()) / len(self.sensors)

    def reliabilities(self) -> dict[str, float]:
        """Return the per-sensor evidence weights fusion should apply."""
        return {sid: report.reliability for sid, report in self.sensors.items()}

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the system report."""
        return {
            "at": self.at.isoformat(),
            "coverage": self.coverage,
            "faulty": self.faulty,
            "sensors": {sid: r.to_dict() for sid, r in self.sensors.items()},
        }


@dataclass
class HealthConfig:
    """Thresholds governing health verdicts.

    Parameters
    ----------
    degraded_after
        Multiple of the declared reporting interval after which a sensor is
        called degraded.
    dropout_after
        Multiple after which a fault is suspected.
    missing_after
        Multiple after which the sensor is treated as supplying no evidence.
    stuck_samples
        Consecutive identical readings that indicate a stuck *sampled*
        sensor. Not applied to state sensors: reporting an unchanged level is
        what a state sensor is for, and a bed that reads occupied for eight
        hours is a sleeping resident, not a fault.
    stuck_state_hours
        How long a *state* sensor may report one unchanged level before it is
        suspected of being stuck. Set well beyond any plausible real
        duration of the states being reported.
    quality_floor
        Smoothed device-reported quality below which a sensor is degraded.
    quality_smoothing
        Weight of each new quality reading in the exponential average.
    drift_reference
        Readings retained as the calibration reference for a sampled sensor.
    drift_recent
        Readings compared against that reference.
    drift_sigma
        Robust standard deviations of shift that count as drift.
    drift_floor
        Absolute shift below which drift is never reported, in sensor units.
        Prevents a very stable sensor from being flagged for trivial moves.
    out_of_range_tolerance
        Consecutive implausible readings tolerated before flagging.
    outage_canary_fraction
        Fraction of the sensors that *can* be checked for silence which must
        be failing before the silence of the sensors that cannot be checked
        is also distrusted. See :meth:`SensorHealthMonitor.report`.
    """

    degraded_after: float = 2.0
    dropout_after: float = 5.0
    missing_after: float = 20.0
    stuck_samples: int = 12
    stuck_state_hours: float = 36.0
    quality_floor: float = 0.5
    quality_smoothing: float = 0.2
    drift_reference: int = 64
    drift_recent: int = 32
    drift_sigma: float = 6.0
    drift_floor: float = 0.0
    out_of_range_tolerance: int = 2
    outage_canary_fraction: float = 0.6

    def __post_init__(self) -> None:
        """Validate threshold configuration."""
        if not 0.0 < self.degraded_after < self.dropout_after < self.missing_after:
            raise ValueError(
                "silence thresholds must be increasing and positive: "
                "degraded_after < dropout_after < missing_after"
            )
        if self.stuck_samples < 2:
            raise ValueError("stuck_samples must be at least 2")
        if self.stuck_state_hours <= 0:
            raise ValueError("stuck_state_hours must be positive")
        if not 0.0 <= self.quality_floor <= 1.0:
            raise ValueError("quality_floor must lie in [0, 1]")
        if not 0.0 < self.quality_smoothing <= 1.0:
            raise ValueError("quality_smoothing must lie in (0, 1]")
        if self.drift_reference < 2 or self.drift_recent < 2:
            raise ValueError("drift windows must contain at least 2 readings")
        if self.drift_sigma <= 0:
            raise ValueError("drift_sigma must be positive")
        if self.drift_floor < 0:
            raise ValueError("drift_floor must be non-negative")
        if self.out_of_range_tolerance < 1:
            raise ValueError("out_of_range_tolerance must be at least 1")
        if not 0.0 < self.outage_canary_fraction <= 1.0:
            raise ValueError("outage_canary_fraction must lie in (0, 1]")


@dataclass
class _SensorState:
    """Bounded per-sensor state maintained by the monitor."""

    spec: SensorSpec
    quality: float
    observations: int = 0
    last_seen: datetime | None = None
    last_value: float | None = None
    repeats: int = 0
    repeat_since: datetime | None = None
    out_of_range_run: int = 0
    reference: deque[float] = field(default_factory=deque)
    recent: deque[float] = field(default_factory=deque)


class SensorHealthMonitor:
    """Track the operating condition of every registered sensor.

    Parameters
    ----------
    registry
        Declared sensors. Observations from unregistered sensors are ignored,
        because their expected behaviour is unknown.
    config
        Thresholds governing the verdicts.
    """

    def __init__(
        self, registry: SensorRegistry, config: HealthConfig | None = None
    ) -> None:
        self.registry = registry
        self.config = config or HealthConfig()
        self._states: dict[str, _SensorState] = {
            spec.sensor_id: _SensorState(
                spec=spec,
                quality=spec.prior_reliability,
                reference=deque(maxlen=self.config.drift_reference),
                recent=deque(maxlen=self.config.drift_recent),
            )
            for spec in registry
        }

    # ------------------------------------------------------------------
    def observe(self, observation: Observation) -> None:
        """Update health state from a single ingested observation."""
        state = self._states.get(observation.sensor_id)
        if state is None:
            logger.debug(
                "Ignoring health update for unregistered sensor '%s'",
                observation.sensor_id,
            )
            return

        state.observations += 1
        if state.last_seen is None or observation.timestamp > state.last_seen:
            state.last_seen = observation.timestamp

        smoothing = self.config.quality_smoothing
        state.quality = (
            1.0 - smoothing
        ) * state.quality + smoothing * observation.quality

        if state.last_value is not None and observation.value == state.last_value:
            state.repeats += 1
        else:
            state.repeats = 1
            state.repeat_since = observation.timestamp
        state.last_value = observation.value

        if state.spec.contains(observation.value):
            state.out_of_range_run = 0
        else:
            state.out_of_range_run += 1

        if observation.kind is ObservationKind.SAMPLE:
            if len(state.reference) < state.reference.maxlen:  # type: ignore[operator]
                state.reference.append(observation.value)
            state.recent.append(observation.value)

    def observe_many(self, observations: Iterable[Observation]) -> None:
        """Update health state from many observations."""
        for observation in observations:
            self.observe(observation)

    # ------------------------------------------------------------------
    def _silence(self, state: _SensorState, now: datetime) -> timedelta | None:
        """Return how long a sensor has been quiet, if it has ever reported."""
        if state.last_seen is None:
            return None
        return max(now - state.last_seen, timedelta(0))

    def _silence_status(
        self, state: _SensorState, silence: timedelta | None
    ) -> tuple[SensorStatus, str] | None:
        """Return a silence-based verdict, or ``None`` when silence is mute.

        A sensor that never declared a reporting interval made no promise to
        speak, so its silence cannot be turned into a failure claim without
        also turning a quiet resident into a broken sensor.
        """
        interval = state.spec.expected_interval
        if interval is None or silence is None:
            return None
        elapsed = silence / interval
        if elapsed >= self.config.missing_after:
            return SensorStatus.MISSING, f"silent for {elapsed:.1f} expected intervals"
        if elapsed >= self.config.dropout_after:
            return SensorStatus.DROPOUT, f"silent for {elapsed:.1f} expected intervals"
        if elapsed >= self.config.degraded_after:
            return SensorStatus.DEGRADED, f"silent for {elapsed:.1f} expected intervals"
        return None

    def _drift(self, state: _SensorState) -> tuple[SensorStatus, str] | None:
        """Compare the recent calibration of a sampled sensor to its own past."""
        reference, recent = state.reference, state.recent
        if len(reference) < 2 or len(recent) < max(2, recent.maxlen or 2):
            return None
        reference_median = statistics.median(reference)
        recent_median = statistics.median(recent)
        shift = abs(recent_median - reference_median)
        deviations = [abs(value - reference_median) for value in reference]
        scale = 1.4826 * statistics.median(deviations)
        threshold = max(self.config.drift_sigma * scale, self.config.drift_floor)
        if threshold <= 0.0 or shift <= threshold:
            return None
        return (
            SensorStatus.DRIFTING,
            f"median shifted by {shift:.3g} against a reference scale of {scale:.3g}",
        )

    def _value_status(self, state: _SensorState) -> tuple[SensorStatus, str] | None:
        """Return a verdict based on the values themselves."""
        if state.out_of_range_run >= self.config.out_of_range_tolerance:
            return (
                SensorStatus.OUT_OF_RANGE,
                f"{state.out_of_range_run} consecutive implausible readings",
            )
        # Repeated identical readings mean different things per kind. An
        # event sensor reports the same value on every activation, and a
        # state sensor reports an unchanged level for as long as the level is
        # unchanged; neither is a fault. Only a sampled sensor, which is
        # supposed to track a varying quantity, is suspicious when it stops
        # moving.
        if (
            state.spec.kind is ObservationKind.SAMPLE
            and state.repeats >= self.config.stuck_samples
        ):
            return (
                SensorStatus.STUCK,
                f"{state.repeats} identical readings of {state.last_value}",
            )
        # A state sensor is only stuck once its level has persisted beyond
        # any plausible real duration.
        if (
            state.spec.kind is ObservationKind.STATE
            and state.repeat_since is not None
            and state.last_seen is not None
        ):
            held = state.last_seen - state.repeat_since
            if held >= timedelta(hours=self.config.stuck_state_hours):
                return (
                    SensorStatus.STUCK,
                    f"held {state.last_value} for {held.total_seconds() / 3600:.0f} h",
                )
        return self._drift(state)

    def report_for(self, sensor_id: str, now: datetime) -> SensorHealthReport:
        """Return the health verdict for one sensor as of *now*."""
        state = self._states[sensor_id]
        silence = self._silence(state, now)

        verdict = self._silence_status(state, silence) or self._value_status(state)
        if verdict is None:
            if state.observations == 0:
                verdict = (SensorStatus.UNKNOWN, "no observations received")
            elif state.quality < self.config.quality_floor:
                verdict = (
                    SensorStatus.DEGRADED,
                    f"reported quality averaging {state.quality:.2f}",
                )
            else:
                verdict = (SensorStatus.HEALTHY, "reporting as declared")

        status, detail = verdict
        reliability = (
            state.spec.prior_reliability * state.quality * status_reliability(status)
        )
        return SensorHealthReport(
            sensor_id=sensor_id,
            status=status,
            reliability=min(max(reliability, 0.0), 1.0),
            last_seen=state.last_seen,
            silence=silence,
            observations=state.observations,
            detail=detail,
        )

    def report(self, now: datetime) -> SystemHealthReport:
        """Return the deployment-wide health verdict as of *now*.

        Individual verdicts are then corrected for one failure mode no single
        sensor can detect on its own. A purely event-driven sensor makes no
        promise to report, so its silence is normally uninformative about its
        health -- but if every sensor that *did* promise has simultaneously
        gone missing, the most likely explanation is that the pathway
        carrying all of them has failed, not that the resident stopped using
        every room at once.

        The sensors with a declared cadence therefore act as canaries for the
        whole delivery path. When enough of them fail, event sensors that
        have been silent for the same period are downgraded to ``DROPOUT``,
        so their silence stops being read as observed inactivity.
        """
        reports = {sid: self.report_for(sid, now) for sid in self._states}

        verifiable = [
            sid
            for sid, state in self._states.items()
            if state.spec.expected_interval is not None
        ]
        if not verifiable:
            return SystemHealthReport(at=now, sensors=reports)

        # Only *silence* indicates a broken delivery path. A stuck or
        # out-of-range sensor is still delivering records, so it says nothing
        # about whether other sensors' records are getting through.
        failing = [
            sid
            for sid in verifiable
            if reports[sid].status in (SensorStatus.DROPOUT, SensorStatus.MISSING)
        ]
        # One dead canary is more likely a dead canary than a dead mine.
        if len(failing) < 2 or len(failing) < self.config.outage_canary_fraction * len(
            verifiable
        ):
            return SystemHealthReport(at=now, sensors=reports)

        # The outage can only have begun after the last canary still speaking.
        last_heard: list[datetime] = [
            seen for sid in failing if (seen := reports[sid].last_seen) is not None
        ]
        if not last_heard:
            return SystemHealthReport(at=now, sensors=reports)
        outage_since = max(last_heard)

        for sid, state in self._states.items():
            if state.spec.expected_interval is not None:
                continue
            if state.last_seen is not None and state.last_seen > outage_since:
                continue
            reports[sid] = replace(
                reports[sid],
                status=SensorStatus.DROPOUT,
                reliability=state.spec.prior_reliability
                * state.quality
                * status_reliability(SensorStatus.DROPOUT),
                detail=(
                    "silent throughout a deployment-wide outage; its silence "
                    "cannot be read as an absence of activity"
                ),
            )
        return SystemHealthReport(at=now, sensors=reports)

    def reliabilities(self, now: datetime) -> dict[str, float]:
        """Return per-sensor evidence weights, the fusion layer's input."""
        return self.report(now).reliabilities()

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, object]:
        """Return restartable monitor state."""
        return {
            sensor_id: {
                "quality": state.quality,
                "observations": state.observations,
                "last_seen": (state.last_seen.isoformat() if state.last_seen else None),
                "last_value": state.last_value,
                "repeats": state.repeats,
                "repeat_since": (
                    state.repeat_since.isoformat() if state.repeat_since else None
                ),
                "out_of_range_run": state.out_of_range_run,
                "reference": list(state.reference),
                "recent": list(state.recent),
            }
            for sensor_id, state in self._states.items()
        }

    def restore(self, snapshot: Mapping[str, object]) -> None:
        """Restore monitor state produced by :meth:`snapshot`."""
        for sensor_id, payload in snapshot.items():
            state = self._states.get(sensor_id)
            if state is None or not isinstance(payload, Mapping):
                continue
            last_seen = payload.get("last_seen")
            state.quality = float(payload.get("quality", state.quality))
            state.observations = int(payload.get("observations", 0))
            state.last_seen = (
                datetime.fromisoformat(str(last_seen)) if last_seen else None
            )
            last_value = payload.get("last_value")
            state.last_value = None if last_value is None else float(last_value)
            state.repeats = int(payload.get("repeats", 0))
            repeat_since = payload.get("repeat_since")
            state.repeat_since = (
                datetime.fromisoformat(str(repeat_since)) if repeat_since else None
            )
            state.out_of_range_run = int(payload.get("out_of_range_run", 0))
            state.reference = deque(
                (float(v) for v in payload.get("reference", ()) or ()),
                maxlen=self.config.drift_reference,
            )
            state.recent = deque(
                (float(v) for v in payload.get("recent", ()) or ()),
                maxlen=self.config.drift_recent,
            )
