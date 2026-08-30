"""Boundary validation and repair for incoming observations.

:class:`ObservationIngestor` is the single door through which sensor records
enter the platform. It is deliberately narrow in scope: it checks that a
record is *structurally* valid and consistent with its declared contract,
repairs what can be repaired deterministically, and records what it did.

Judgements about whether a sensor is *behaving* -- stuck, drifting, dead --
belong to :mod:`sensor_modeling.health`, which observes the ingested stream
over time. Keeping the two apart means a malfunctioning sensor never looks
like a malformed message, and vice versa.

The ingestor holds bounded state (one timestamp plus a short latency window
per source), so it is usable unchanged in a long-running edge deployment.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .observation import Observation
from .registry import SensorContractError, SensorRegistry, UnknownSensorError
from .types import ObservationFlag

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RejectedObservation:
    """A record that could not be admitted, with the reason why."""

    reason: str
    sensor_id: str
    detail: str


@dataclass
class IngestionReport:
    """Counters describing what happened to a batch of incoming records."""

    accepted: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    late_arrivals: int = 0
    clock_adjusted: int = 0
    unit_converted: int = 0
    rejected: list[RejectedObservation] = field(default_factory=list)

    @property
    def received(self) -> int:
        """Total number of records offered to the ingestor."""
        return self.accepted + self.duplicates + len(self.rejected)

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable summary of the report."""
        return {
            "received": self.received,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "out_of_order": self.out_of_order,
            "late_arrivals": self.late_arrivals,
            "clock_adjusted": self.clock_adjusted,
            "unit_converted": self.unit_converted,
            "rejected": [
                {"reason": r.reason, "sensor_id": r.sensor_id, "detail": r.detail}
                for r in self.rejected
            ],
        }


@dataclass
class ClockOffsetEstimator:
    """Estimate a per-source clock offset from observed arrival latencies.

    For a source whose clock is ahead of the receiving system by ``delta``,
    every reported timestamp is ``delta`` too large, so the measured latency
    ``received_at - timestamp`` is ``delta`` too small. Transport delay is
    bounded below by some minimum, so the smallest latency seen from a source
    estimates ``minimum_delay - delta``, and the correction to apply to its
    timestamps is ``min_latency - reference_delay``.

    This is the same minimum-filtering idea used by network time protocols.
    It is intentionally simple and inspectable: it assumes the receiving clock
    is the reference and that each source occasionally delivers a record with
    close to the minimum transport delay.

    Parameters
    ----------
    window
        Number of recent latencies retained per source.
    min_samples
        Latencies required before an offset is reported at all.
    tolerance
        Offsets smaller than this are treated as noise and not applied.
    reference_delay
        Assumed minimum transport delay shared by all sources.
    """

    window: int = 64
    min_samples: int = 8
    tolerance: timedelta = timedelta(seconds=30)
    reference_delay: timedelta = timedelta(0)
    _latencies: dict[str, deque[float]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Validate estimator configuration."""
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        if self.min_samples > self.window:
            raise ValueError("min_samples must not exceed window")
        if self.tolerance < timedelta(0):
            raise ValueError("tolerance must be non-negative")

    def observe(self, source: str, latency: timedelta) -> None:
        """Record an observed arrival *latency* for *source*."""
        samples = self._latencies.setdefault(source, deque(maxlen=self.window))
        samples.append(latency.total_seconds())

    def offset(self, source: str) -> timedelta:
        """Return the timestamp correction to apply to records from *source*."""
        samples = self._latencies.get(source)
        if samples is None or len(samples) < self.min_samples:
            return timedelta(0)
        estimate = timedelta(seconds=min(samples)) - self.reference_delay
        if abs(estimate) < self.tolerance:
            return timedelta(0)
        return estimate

    def offsets(self) -> dict[str, float]:
        """Return the current offset per source in seconds, for reporting."""
        return {
            source: self.offset(source).total_seconds() for source in self._latencies
        }

    def median_latency(self, source: str) -> float | None:
        """Return the median observed latency in seconds, if known."""
        samples = self._latencies.get(source)
        if not samples:
            return None
        return statistics.median(samples)


@dataclass
class ObservationIngestor:
    """Validate, normalise, and repair observations at the system boundary.

    Parameters
    ----------
    registry
        Declared sensors. Records from unregistered sensors are rejected
        rather than guessed at.
    late_threshold
        Arrival delay beyond which a record is flagged as a late arrival.
    correct_clock_drift
        Whether to apply the estimated per-source clock offset. Corrections
        are always flagged so downstream stages know the timestamp was
        adjusted rather than measured.
    clock_estimator
        Estimator used when ``correct_clock_drift`` is enabled.
    """

    registry: SensorRegistry
    late_threshold: timedelta = timedelta(minutes=5)
    correct_clock_drift: bool = False
    clock_estimator: ClockOffsetEstimator = field(default_factory=ClockOffsetEstimator)
    _newest: datetime | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate ingestor configuration."""
        if self.late_threshold <= timedelta(0):
            raise ValueError("late_threshold must be positive")

    @property
    def newest_timestamp(self) -> datetime | None:
        """The latest observation timestamp admitted so far."""
        return self._newest

    def ingest(
        self, observation: Observation, report: IngestionReport | None = None
    ) -> Observation | None:
        """Admit a single observation.

        Returns
        -------
        Observation | None
            The normalised observation, or ``None`` when it was rejected.
            Rejections are recorded in *report* rather than raised, so that a
            single malformed record cannot stop a live stream.
        """
        outcome = report if report is not None else IngestionReport()

        try:
            normalised = self.registry.normalise(observation)
        except UnknownSensorError as exc:
            outcome.rejected.append(
                RejectedObservation("unknown_sensor", observation.sensor_id, str(exc))
            )
            return None
        except SensorContractError as exc:
            outcome.rejected.append(
                RejectedObservation("contract", observation.sensor_id, str(exc))
            )
            return None

        if ObservationFlag.UNIT_CONVERTED in normalised.flags:
            outcome.unit_converted += 1

        latency = normalised.latency
        if latency is not None:
            self.clock_estimator.observe(normalised.source, latency)
            if latency > self.late_threshold:
                normalised = normalised.with_flags(ObservationFlag.LATE_ARRIVAL)
                outcome.late_arrivals += 1

        if self.correct_clock_drift:
            offset = self.clock_estimator.offset(normalised.source)
            if offset:
                normalised = normalised.shifted(offset)
                outcome.clock_adjusted += 1

        if self._newest is not None and normalised.timestamp < self._newest:
            normalised = normalised.with_flags(ObservationFlag.OUT_OF_ORDER)
            outcome.out_of_order += 1
        else:
            self._newest = normalised.timestamp

        outcome.accepted += 1
        return normalised

    def ingest_many(
        self, observations: Iterable[Observation]
    ) -> tuple[list[Observation], IngestionReport]:
        """Admit many observations and return the survivors with a report."""
        report = IngestionReport()
        admitted = [
            result
            for result in (self.ingest(obs, report) for obs in observations)
            if result is not None
        ]
        if report.rejected:
            logger.warning(
                "Rejected %d of %d observations during ingestion",
                len(report.rejected),
                report.received,
            )
        return admitted, report

    def snapshot(self) -> dict[str, object]:
        """Return restartable ingestor state."""
        return {
            "newest": self._newest.isoformat() if self._newest else None,
            "clock_offsets": self.clock_estimator.offsets(),
        }

    def restore(self, state: dict[str, object]) -> None:
        """Restore the newest-timestamp watermark from :meth:`snapshot`."""
        newest = state.get("newest")
        self._newest = (
            datetime.fromisoformat(str(newest)).astimezone(timezone.utc)
            if newest
            else None
        )
