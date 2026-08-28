"""Incremental, edge-capable orchestration of the inference chain.

This package owns orchestration only. It performs no inference of its own and
knows nothing about files, HTTP, dashboards, or storage, which keeps the
scientific layers usable without it.
"""

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
    "PipelineConfig",
    "PipelineStep",
    "collect_alerts",
    "collect_changes",
    "daily_summaries",
    "local_midnight",
]
