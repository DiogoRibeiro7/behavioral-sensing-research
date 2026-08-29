"""Measuring whether the online pipeline is genuinely edge-capable.

Claiming bounded memory is easy; demonstrating it is the point of this module.
The measurement that matters is not peak allocation on one run -- that varies
with the interpreter and tells you little -- but whether the *retained* state
grows with how long the pipeline has been running.

So the central measurement here compares the serialised snapshot after a short
run against the snapshot after a much longer one. If the pipeline is bounded,
those are close to the same size no matter how many observations went through.
If it is quietly accumulating, the second is larger, and no amount of docstring
prose about bounded deques will hide it.

Correctness came first. Nothing in the pipeline has been optimised, and these
numbers exist to establish a baseline and catch regressions, not to advertise
performance.
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from ..baseline.adaptive import BaselineConfig
from ..simulation.household import HouseholdConfig, simulate
from .pipeline import BehaviouralSensingPipeline, PipelineConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineBenchmark:
    """Throughput, latency and retained-state measurements for one run.

    Attributes
    ----------
    days, observations, steps
        Size of the workload.
    wall_seconds
        Total processing time.
    observations_per_second
        Ingestion throughput.
    microseconds_per_observation
        Mean cost of admitting one record.
    step_latency_ms
        Mean, median, 95th percentile and maximum cost of one pipeline step,
        which is where health, context, fusion and baselines all run.
    snapshot_bytes
        Size of the serialised restartable state. This is the number that
        decides whether the pipeline fits on a constrained device.
    peak_memory_mb
        Peak traced allocation during the run, including the simulated
        record itself, so it is an upper bound rather than the pipeline's
        own footprint.
    """

    days: int
    observations: int
    steps: int
    wall_seconds: float
    observations_per_second: float
    microseconds_per_observation: float
    step_latency_ms: dict[str, float]
    snapshot_bytes: int
    peak_memory_mb: float

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the benchmark."""
        return {
            "days": self.days,
            "observations": self.observations,
            "steps": self.steps,
            "wall_seconds": self.wall_seconds,
            "observations_per_second": self.observations_per_second,
            "microseconds_per_observation": self.microseconds_per_observation,
            "step_latency_ms": self.step_latency_ms,
            "snapshot_bytes": self.snapshot_bytes,
            "peak_memory_mb": self.peak_memory_mb,
        }


def benchmark_pipeline(
    days: int = 7,
    *,
    seed: int = 4242,
    step: timedelta = timedelta(minutes=15),
    trace_memory: bool = True,
) -> PipelineBenchmark:
    """Run the pipeline over a simulated household and measure it.

    Latency is measured per *step* rather than per observation, because a
    step is where the work happens: observations between steps only accumulate
    in a buffer, while a step runs health, context, fusion and -- at a day
    boundary -- the baselines and alerting.
    """
    result = simulate(HouseholdConfig(days=days, seed=seed))
    pipeline = BehaviouralSensingPipeline(
        result.registry, config=PipelineConfig(tz=result.config.tz, step=step)
    )

    if trace_memory:
        tracemalloc.start()

    latencies: list[float] = []
    started = time.perf_counter()
    latest = None
    for observation in result.observations:
        arrival = observation.received_at or observation.timestamp
        pipeline.push(observation)
        if latest is None or arrival > latest:
            latest = arrival
            before = time.perf_counter()
            produced = pipeline.advance(arrival)
            elapsed = time.perf_counter() - before
            if produced:
                latencies.extend([elapsed / len(produced)] * len(produced))
    pipeline.close(result.end)
    wall = time.perf_counter() - started

    peak_mb = 0.0
    if trace_memory:
        peak_mb = tracemalloc.get_traced_memory()[1] / 1_000_000
        tracemalloc.stop()

    snapshot = json.dumps(pipeline.snapshot(), default=str)
    count = len(result.observations)
    samples = np.array(latencies) * 1000.0 if latencies else np.zeros(1)

    return PipelineBenchmark(
        days=days,
        observations=count,
        steps=len(latencies),
        wall_seconds=wall,
        observations_per_second=count / wall if wall > 0 else 0.0,
        microseconds_per_observation=(wall / count * 1e6) if count else 0.0,
        step_latency_ms={
            "mean": float(samples.mean()),
            "median": float(np.median(samples)),
            "p95": float(np.percentile(samples, 95)),
            "max": float(samples.max()),
        },
        snapshot_bytes=len(snapshot.encode("utf-8")),
        peak_memory_mb=peak_mb,
    )


@dataclass(frozen=True)
class BoundedStateResult:
    """Evidence for or against the bounded-memory claim."""

    short_days: int
    long_days: int
    short_snapshot_bytes: int
    long_snapshot_bytes: int

    @property
    def growth_ratio(self) -> float:
        """How much retained state grew for a much longer run.

        A bounded pipeline stays near one. A value tracking the ratio of the
        run lengths would mean state is accumulating with the stream.
        """
        if self.short_snapshot_bytes == 0:
            return float("inf")
        return self.long_snapshot_bytes / self.short_snapshot_bytes

    @property
    def workload_ratio(self) -> float:
        """How much longer the long run was."""
        return self.long_days / self.short_days if self.short_days else float("inf")

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the result."""
        return {
            "short_days": self.short_days,
            "long_days": self.long_days,
            "short_snapshot_bytes": self.short_snapshot_bytes,
            "long_snapshot_bytes": self.long_snapshot_bytes,
            "growth_ratio": self.growth_ratio,
            "workload_ratio": self.workload_ratio,
        }


def measure_bounded_state(
    short_days: int = 3,
    long_days: int = 21,
    *,
    seed: int = 4242,
    step: timedelta = timedelta(minutes=30),
    baseline_config: BaselineConfig | None = None,
) -> BoundedStateResult:
    """Compare retained state after a short run against a much longer one.

    This is the direct test of the bounded-memory claim. Every stage but the
    baselines keeps a fixed-size belief, and the baselines keep a history
    capped at ``history_days``.

    That cap is why the growth ratio is not exactly one by default: over a few
    weeks the history is still filling, so a longer run legitimately retains
    more. Pass a *baseline_config* whose ``history_days`` both runs exceed to
    observe the asymptotic behaviour, where retained state stops growing
    however long the pipeline runs.
    """
    if short_days >= long_days:
        raise ValueError("long_days must exceed short_days")

    sizes: dict[int, int] = {}
    for days in (short_days, long_days):
        result = simulate(HouseholdConfig(days=days, seed=seed))
        pipeline = BehaviouralSensingPipeline(
            result.registry,
            config=PipelineConfig(tz=result.config.tz, step=step),
            baseline_config=baseline_config,
        )
        pipeline.run(result.observations)
        pipeline.close(result.end)
        sizes[days] = len(json.dumps(pipeline.snapshot(), default=str).encode("utf-8"))

    outcome = BoundedStateResult(
        short_days=short_days,
        long_days=long_days,
        short_snapshot_bytes=sizes[short_days],
        long_snapshot_bytes=sizes[long_days],
    )
    logger.info(
        "State grew %.2fx for a %.1fx longer run",
        outcome.growth_ratio,
        outcome.workload_ratio,
    )
    return outcome


if __name__ == "__main__":  # pragma: no cover - manual invocation
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(benchmark_pipeline().to_dict(), indent=2))
    print(json.dumps(measure_bounded_state().to_dict(), indent=2))
