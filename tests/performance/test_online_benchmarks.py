"""Performance and bounded-memory checks for the online pipeline.

Thresholds here are deliberately loose. They exist to catch a regression that
changes behaviour by an order of magnitude, not to pin down performance on a
particular machine, and CI runners vary far too much for tight bounds to mean
anything.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from sensor_modeling.baseline import BaselineConfig
from sensor_modeling.online import benchmark_pipeline, measure_bounded_state


class TestThroughput:
    def test_the_pipeline_processes_a_week_in_reasonable_time(self) -> None:
        result = benchmark_pipeline(days=3, step=timedelta(minutes=30))
        assert result.observations > 1000
        assert result.steps > 50
        assert result.wall_seconds < 120

    def test_per_observation_cost_stays_in_microseconds(self) -> None:
        result = benchmark_pipeline(days=3, step=timedelta(minutes=30))
        assert result.microseconds_per_observation < 2000

    def test_step_latency_is_reported_across_percentiles(self) -> None:
        result = benchmark_pipeline(days=3, step=timedelta(minutes=30))
        latency = result.step_latency_ms
        assert latency["median"] <= latency["p95"] <= latency["max"]
        assert latency["p95"] < 500

    def test_a_benchmark_serialises(self) -> None:
        payload = benchmark_pipeline(days=2, step=timedelta(hours=1)).to_dict()
        assert set(payload) >= {
            "observations",
            "steps",
            "observations_per_second",
            "step_latency_ms",
            "snapshot_bytes",
        }

    def test_memory_tracing_can_be_disabled(self) -> None:
        result = benchmark_pipeline(days=2, step=timedelta(hours=1), trace_memory=False)
        assert result.peak_memory_mb == 0.0


class TestBoundedState:
    def test_retained_state_is_small_enough_for_a_constrained_device(self) -> None:
        result = benchmark_pipeline(days=3, step=timedelta(minutes=30))
        assert result.snapshot_bytes < 200_000

    def test_state_grows_far_more_slowly_than_the_workload(self) -> None:
        result = measure_bounded_state(
            short_days=3, long_days=15, step=timedelta(hours=1)
        )
        assert result.workload_ratio == pytest.approx(5.0)
        # Sub-linear by a wide margin: a five-fold longer run must not retain
        # anything like five times as much.
        assert result.growth_ratio < 2.0

    def test_state_stops_growing_once_the_history_cap_is_reached(self) -> None:
        """The asymptotic property that makes indefinite operation possible.

        Below the cap the baseline history is still filling, so some growth is
        legitimate. Once both runs exceed it, retained state must be flat
        however long the pipeline has been running.
        """
        result = measure_bounded_state(
            short_days=12,
            long_days=36,
            step=timedelta(hours=1),
            baseline_config=BaselineConfig(history_days=7, min_samples=3),
        )
        assert result.growth_ratio < 1.1

    def test_the_comparison_requires_a_longer_second_run(self) -> None:
        with pytest.raises(ValueError, match="must exceed"):
            measure_bounded_state(short_days=10, long_days=5)

    def test_a_bounded_state_result_serialises(self) -> None:
        payload = measure_bounded_state(
            short_days=2, long_days=5, step=timedelta(hours=1)
        ).to_dict()
        assert payload["workload_ratio"] == pytest.approx(2.5)
        assert payload["growth_ratio"] > 0
