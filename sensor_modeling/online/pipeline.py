"""The incremental end-to-end pipeline.

.. code-block:: text

    observation arrives
            v
    validate at the boundary
            v
    update sensor health
            v
    update person context
            v
    update latent behavioural state
            v
    close the day, update the personal baseline
            v
    evaluate behavioural change
            v
    emit a structured, explainable result

Every stage keeps bounded state, so the pipeline runs indefinitely on an edge
device, and every stage can be snapshotted and restored, so a restart or an
intermittent uplink does not lose the resident's accumulated history.

The one piece of machinery that only exists because streams are real is the
lateness buffer. Observations arrive out of order; the filter is causal and
refuses to move backwards. So the pipeline holds records behind a watermark
long enough to reorder them, releases them in timestamp order, and counts
anything later than the tolerance rather than quietly folding a stale record
into the current belief.

This module owns orchestration only. It contains no inference of its own, and
nothing here knows about files, HTTP, dashboards or storage.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Any

from ..alerts.alert import Alert, AlertEngine, AlertPolicy
from ..baseline.adaptive import AdaptiveBaseline, BaselineConfig, BehaviouralChange
from ..baseline.features import DailySummary, summarise_days
from ..context.occupancy import ContextConfig, ContextEstimate, ResidentContextEstimator
from ..fusion.defaults import default_emissions
from ..fusion.emissions import EmissionModel
from ..fusion.estimate import StateEstimate
from ..fusion.filter import FusionConfig, MultimodalBayesFilter
from ..health.monitor import HealthConfig, SensorHealthMonitor, SystemHealthReport
from ..observations.ingest import IngestionReport, ObservationIngestor
from ..observations.observation import Observation
from ..observations.registry import SensorRegistry
from ..states.ontology import BehaviouralState, StateOntology

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration of the online pipeline.

    Parameters
    ----------
    tz
        Timezone whose calendar days bound the baseline. Defaults to the
        registry's deployment timezone when not given.
    step
        How often the state filter advances. Between steps, observations
        accumulate and are applied together.
    lateness_tolerance
        How far behind real time the processing watermark sits. Records that
        arrive later than this are counted and discarded rather than folded
        into an already-advanced belief.
    features
        Behavioural states whose daily hours get an adaptive baseline.
    min_day_coverage, min_day_observed
        Quality a day must reach before it may inform the baseline.
    attribute_activity
        Whether ambient evidence is discounted by the probability that the
        monitored resident generated it. Setting this false makes the
        pipeline attribute every ambient event to the resident, which is the
        naive behaviour most ambient monitoring assumes. It exists so that
        assumption can be *measured* against the occupancy-aware default
        rather than argued about.
    """

    tz: tzinfo | None = None
    step: timedelta = timedelta(minutes=5)
    lateness_tolerance: timedelta = timedelta(minutes=10)
    features: tuple[BehaviouralState, ...] = (
        BehaviouralState.SLEEPING,
        BehaviouralState.KITCHEN_ACTIVITY,
        BehaviouralState.BATHROOM_ACTIVITY,
        BehaviouralState.AWAY,
    )
    min_day_coverage: float = 0.5
    min_day_observed: float = 0.6
    attribute_activity: bool = True

    def __post_init__(self) -> None:
        """Validate the pipeline configuration."""
        if self.step <= timedelta(0):
            raise ValueError("step must be positive")
        if self.lateness_tolerance < timedelta(0):
            raise ValueError("lateness_tolerance must be non-negative")
        if not self.features:
            raise ValueError("at least one baseline feature is required")
        for name in ("min_day_coverage", "min_day_observed"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class PipelineStep:
    """Everything the pipeline concluded at one moment."""

    at: datetime
    state: StateEstimate
    context: ContextEstimate
    health: SystemHealthReport
    alerts: tuple[Alert, ...] = ()
    changes: tuple[BehaviouralChange, ...] = ()
    day_closed: DailySummary | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form of the step."""
        return {
            "at": self.at.isoformat(),
            "state": self.state.to_dict(),
            "context": self.context.to_dict(),
            "health": self.health.to_dict(),
            "alerts": [alert.to_dict() for alert in self.alerts],
            "changes": [change.to_dict() for change in self.changes],
            "day_closed": (
                self.day_closed.to_dict() if self.day_closed is not None else None
            ),
        }


class BehaviouralSensingPipeline:
    """Incremental orchestration of the full inference chain.

    Parameters
    ----------
    registry
        Declared sensors.
    ontology
        Latent behavioural states and their dynamics.
    emissions
        Observation models. Defaults are derived from the registry.
    config
        Pipeline configuration.
    fusion_config, health_config, context_config, baseline_config, alert_policy
        Configuration for each stage.
    """

    def __init__(
        self,
        registry: SensorRegistry,
        ontology: StateOntology | None = None,
        emissions: Iterable[EmissionModel] | None = None,
        config: PipelineConfig | None = None,
        *,
        fusion_config: FusionConfig | None = None,
        health_config: HealthConfig | None = None,
        context_config: ContextConfig | None = None,
        baseline_config: BaselineConfig | None = None,
        alert_policy: AlertPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.ontology = ontology or StateOntology()
        self.config = config or PipelineConfig()

        self.ingestor = ObservationIngestor(registry)
        self.health = SensorHealthMonitor(registry, health_config)
        self.context = ResidentContextEstimator(registry, context_config)
        self.filter = MultimodalBayesFilter(
            self.ontology,
            (
                emissions
                if emissions is not None
                else default_emissions(registry, self.ontology)
            ),
            registry=registry,
            config=fusion_config,
        )
        self.alerts = AlertEngine(alert_policy)
        self.baselines = {
            state.value: AdaptiveBaseline(
                f"{state.value}_hours", baseline_config or BaselineConfig()
            )
            for state in self.config.features
        }
        self._baseline_config = baseline_config or BaselineConfig()

        self._buffer: list[Observation] = []
        self._pending: list[Observation] = []
        self._day_estimates: list[StateEstimate] = []
        self._last_context: ContextEstimate | None = None
        self._current_day: date | None = None
        self._next_step: datetime | None = None
        self.ingestion = IngestionReport()
        self.too_late = 0

    # ------------------------------------------------------------------
    @property
    def tz(self) -> tzinfo | None:
        """Timezone whose calendar days bound the baseline."""
        return self.config.tz

    def _local_day(self, moment: datetime) -> date:
        """Return the local calendar date of *moment*."""
        zone = self.config.tz
        return (moment.astimezone(zone) if zone is not None else moment).date()

    def push(self, observation: Observation) -> bool:
        """Admit one observation into the lateness buffer.

        Returns
        -------
        bool
            ``True`` when the record was buffered, ``False`` when it was
            rejected at the boundary or arrived beyond the lateness
            tolerance. Rejections are counted, never raised: one malformed
            record must not stop a live stream.
        """
        admitted = self.ingestor.ingest(observation, self.ingestion)
        if admitted is None:
            return False
        if (
            self._next_step is not None
            and admitted.timestamp < self._next_step - self.config.step
        ):
            self.too_late += 1
            logger.debug(
                "Discarding record from '%s' that arrived after its window closed",
                admitted.sensor_id,
            )
            return False
        bisect.insort(self._buffer, admitted, key=lambda obs: obs.timestamp)
        return True

    def _release(self, watermark: datetime) -> list[Observation]:
        """Release buffered records at or before *watermark*, in time order."""
        cut = bisect.bisect_right([obs.timestamp for obs in self._buffer], watermark)
        released, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return released

    # ------------------------------------------------------------------
    def advance(self, now: datetime) -> list[PipelineStep]:
        """Advance the pipeline to *now*, returning one result per step.

        Observations are released from the buffer behind the lateness
        watermark, so a record that arrived out of order still reaches the
        filter in the right place provided it was not later than the
        tolerance.
        """
        watermark = now - self.config.lateness_tolerance
        self._pending.extend(self._release(watermark))
        self._pending.sort(key=lambda obs: obs.timestamp)

        if self._next_step is None:
            if not self._pending:
                return []
            self._next_step = self._pending[0].timestamp

        steps: list[PipelineStep] = []
        while self._next_step is not None and self._next_step <= watermark:
            boundary = self._next_step
            cut = bisect.bisect_right(
                [obs.timestamp for obs in self._pending], boundary
            )
            batch, self._pending = self._pending[:cut], self._pending[cut:]
            steps.append(self._step(boundary, batch))
            self._next_step = boundary + self.config.step
        return steps

    def _step(self, moment: datetime, batch: Sequence[Observation]) -> PipelineStep:
        """Run one pipeline step over the observations of a single interval."""
        self.health.observe_many(batch)
        health = self.health.report(moment)
        reliabilities = health.reliabilities()

        context = self.context.update(moment, batch, reliabilities=reliabilities)
        self._last_context = context
        attribution = (
            self.context.attribution(context)
            if self.config.attribute_activity
            else None
        )

        estimate = self.filter.update(
            moment, batch, reliabilities=reliabilities, attribution=attribution
        )

        day = self._local_day(moment)
        closed: DailySummary | None = None
        changes: tuple[BehaviouralChange, ...] = ()
        alerts: tuple[Alert, ...] = ()

        if self._current_day is None:
            self._current_day = day
        elif day != self._current_day:
            closed, changes, alerts = self._close_day(
                self._current_day, moment, context
            )
            self._current_day = day
            self._day_estimates = []

        self._day_estimates.append(estimate)

        health_alert = self.alerts.consider_health(health, at=moment)
        if health_alert is not None:
            alerts = alerts + (health_alert,)

        return PipelineStep(
            at=moment,
            state=estimate,
            context=context,
            health=health,
            alerts=alerts,
            changes=changes,
            day_closed=closed,
        )

    def _close_day(
        self, day: date, moment: datetime, context: ContextEstimate
    ) -> tuple[DailySummary | None, tuple[BehaviouralChange, ...], tuple[Alert, ...]]:
        """Summarise a completed day and update the baselines from it."""
        summaries = summarise_days(
            self._day_estimates, tz=self.config.tz, max_interval=self.config.step * 3
        )
        summary = next((item for item in summaries if item.day == day), None)
        if summary is None:
            return None, (), ()

        usable = summary.is_usable(
            self.config.min_day_coverage, self.config.min_day_observed
        )
        changes = []
        for state in self.config.features:
            baseline = self.baselines[state.value]
            if usable:
                changes.append(baseline.observe(day, summary.hours_in(state)))
            else:
                changes.append(
                    baseline.skip(
                        day,
                        f"day observed at {summary.observed:.0%} with "
                        f"{summary.coverage:.0%} sensor coverage",
                    )
                )

        alerts = self.alerts.review(
            changes,
            at=moment,
            coverage=summary.coverage,
            attribution=context.ambient_attribution(),
            deviation_threshold=self._baseline_config.deviation_threshold,
            trend_threshold=self._baseline_config.trend_threshold,
        )
        return summary, tuple(changes), tuple(alerts)

    # ------------------------------------------------------------------
    def run(
        self, observations: Iterable[Observation], *, until: datetime | None = None
    ) -> list[PipelineStep]:
        """Process a whole record offline, in arrival order.

        A convenience wrapper over :meth:`push` and :meth:`advance` for batch
        experiments. It drives exactly the same online code path, so results
        match what a live deployment would have produced.
        """
        steps: list[PipelineStep] = []
        latest: datetime | None = None
        for observation in observations:
            arrival = observation.received_at or observation.timestamp
            self.push(observation)
            if latest is None or arrival > latest:
                latest = arrival
                steps.extend(self.advance(arrival))
        if latest is not None:
            end = (
                until if until is not None else latest + self.config.lateness_tolerance
            )
            steps.extend(self.advance(end + self.config.lateness_tolerance))
        return steps

    def close(self, now: datetime) -> tuple[PipelineStep, ...]:
        """Flush the buffer and close the final partial day at *now*.

        The final day is usually incomplete, so its summary will often fail
        the coverage test and be recorded as unusable rather than entering
        the baseline as an unusually quiet day.
        """
        steps = tuple(self.advance(now + self.config.lateness_tolerance * 2))
        if self._current_day is None or not self._day_estimates:
            return steps

        last = self._day_estimates[-1]
        context = self._last_context
        if context is None:  # pragma: no cover - close() before any step
            return steps

        summary, changes, alerts = self._close_day(self._current_day, last.at, context)
        self._day_estimates = []
        self._current_day = None
        if summary is None:
            return steps
        return steps + (
            PipelineStep(
                at=last.at,
                state=last,
                context=context,
                health=self.health.report(last.at),
                alerts=alerts,
                changes=changes,
                day_closed=summary,
            ),
        )

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Return restartable state for every stage.

        The buffer is deliberately not included. Buffered records have not
        been folded into any belief yet, so re-delivering them after a
        restart is correct; persisting them here would risk counting them
        twice.
        """
        return {
            "ingestor": self.ingestor.snapshot(),
            "health": self.health.snapshot(),
            "context": self.context.snapshot(),
            "filter": self.filter.snapshot(),
            "alerts": self.alerts.snapshot(),
            "baselines": {
                name: baseline.snapshot() for name, baseline in self.baselines.items()
            },
            "current_day": self._current_day.isoformat() if self._current_day else None,
            "next_step": self._next_step.isoformat() if self._next_step else None,
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        """Restore pipeline state produced by :meth:`snapshot`."""
        self.ingestor.restore(state["ingestor"])
        self.health.restore(state["health"])
        self.context.restore(state["context"])
        self.filter.restore(state["filter"])
        self.alerts.restore(state["alerts"])
        for name, payload in (state.get("baselines") or {}).items():
            baseline = self.baselines.get(name)
            if baseline is not None:
                baseline.restore(payload)
        current_day = state.get("current_day")
        self._current_day = date.fromisoformat(current_day) if current_day else None
        next_step = state.get("next_step")
        self._next_step = datetime.fromisoformat(next_step) if next_step else None
        self._day_estimates = []


def local_midnight(day: date, zone: tzinfo | None) -> datetime:
    """Return the instant local midnight begins on *day*."""
    naive = datetime.combine(day, time.min)
    return naive.replace(tzinfo=zone) if zone is not None else naive


def collect_alerts(steps: Iterable[PipelineStep]) -> list[Alert]:
    """Return every alert raised across a run, in order."""
    return [alert for step in steps for alert in step.alerts]


def collect_changes(steps: Iterable[PipelineStep]) -> list[BehaviouralChange]:
    """Return every baseline verdict produced across a run, in order."""
    return [change for step in steps for change in step.changes]


def daily_summaries(steps: Iterable[PipelineStep]) -> list[DailySummary]:
    """Return the day summaries closed across a run, in order."""
    return [step.day_closed for step in steps if step.day_closed is not None]
