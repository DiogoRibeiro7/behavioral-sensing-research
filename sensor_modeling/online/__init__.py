"""Incremental, edge-capable orchestration of the inference chain.

This package owns orchestration only. It performs no inference of its own and
knows nothing about files, HTTP, dashboards, or storage, which keeps the
scientific layers usable without it.
"""

from .benchmarks import (
    BoundedStateResult,
    PipelineBenchmark,
    benchmark_pipeline,
    measure_bounded_state,
)
from .pipeline import (
    BehaviouralSensingPipeline,
    PipelineConfig,
    PipelineStep,
    collect_alerts,
    collect_changes,
    daily_summaries,
    local_midnight,
)

__all__ = [
    "BehaviouralSensingPipeline",
    "BoundedStateResult",
    "PipelineBenchmark",
    "PipelineConfig",
    "PipelineStep",
    "benchmark_pipeline",
    "collect_alerts",
    "collect_changes",
    "daily_summaries",
    "local_midnight",
    "measure_bounded_state",
]
