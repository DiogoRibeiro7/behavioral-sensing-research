"""Pruned Exact Linear Time (PELT) change-point detector.

The detector segments a univariate signal by minimizing

``sum(segment_cost) + penalty * n_change_points``

with exact dynamic programming and PELT pruning. Returned indices are
0-based change-point locations that mark the first sample of a new segment.

The default ``"l2"`` cost models piecewise-constant mean shifts and uses
prefix sums for fast segment evaluation. The alternative ``"l1"`` cost and
custom cost callables are supported for robustness or experimentation, but
they recompute segment costs directly and are therefore slower. As with
standard PELT, observed runtime is often near-linear when pruning is effective
and can degrade to quadratic in adversarial cases.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

SegmentCost = Callable[[np.ndarray], float]


@dataclass
class PELTChangePointDetector:
    """Detect univariate change points with configurable penalties and costs."""

    penalty: float = 1.0
    min_segment_length: int = 2
    cost: str | SegmentCost = "l2"

    def _validate_signal(self, signal: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return *signal* as a finite 1D float array."""
        arr = np.asarray(signal, dtype=float)
        if arr.ndim != 1:
            raise ValueError("PELTChangePointDetector only supports 1D signals")
        if not np.all(np.isfinite(arr)):
            raise ValueError("signal must contain only finite numeric values")
        return arr

    def _cost_function(self, signal: np.ndarray) -> Callable[[int, int], float]:
        """Build a segment cost function over half-open intervals."""
        if callable(self.cost):
            return lambda start, end: float(self.cost(signal[start:end]))

        if self.cost == "l2":
            prefix_sum = np.concatenate(([0.0], np.cumsum(signal)))
            prefix_sq = np.concatenate(([0.0], np.cumsum(signal * signal)))

            def l2_cost(start: int, end: int) -> float:
                seg_sum = prefix_sum[end] - prefix_sum[start]
                seg_sq = prefix_sq[end] - prefix_sq[start]
                length = end - start
                return float(seg_sq - (seg_sum * seg_sum) / length)

            return l2_cost

        if self.cost == "l1":
            return lambda start, end: float(
                np.abs(signal[start:end] - np.median(signal[start:end])).sum()
            )

        raise ValueError("cost must be 'l1', 'l2', or a callable segment cost")

    def detect(self, signal: Sequence[float] | np.ndarray) -> List[int]:
        """Detect change points in a univariate signal.

        The returned indices are 0-based segment starts, excluding ``0`` and
        ``len(signal)``. For example, ``[20, 35]`` means segments
        ``signal[:20]``, ``signal[20:35]``, and ``signal[35:]``.
        """
        if self.penalty <= 0:
            raise ValueError("penalty must be positive")
        if self.min_segment_length < 1:
            raise ValueError("min_segment_length must be at least 1")

        arr = self._validate_signal(signal)
        n_samples = len(arr)
        if n_samples < 2 * self.min_segment_length:
            return []

        segment_cost = self._cost_function(arr)
        best_cost = np.full(n_samples + 1, np.inf)
        previous = np.full(n_samples + 1, -1, dtype=int)
        best_cost[0] = -self.penalty
        candidates = [0]

        for end in range(self.min_segment_length, n_samples + 1):
            valid_candidates = [
                start
                for start in candidates
                if end - start >= self.min_segment_length and np.isfinite(best_cost[start])
            ]
            if not valid_candidates:
                continue

            totals = [
                best_cost[start] + segment_cost(start, end) + self.penalty
                for start in valid_candidates
            ]
            best_idx = int(np.argmin(totals))
            best_start = valid_candidates[best_idx]
            best_cost[end] = totals[best_idx]
            previous[end] = best_start

            retained = [
                start
                for start in candidates
                if end - start < self.min_segment_length
                or best_cost[start] + segment_cost(start, end) <= best_cost[end]
            ]
            candidates = retained + [end]

        if not np.isfinite(best_cost[n_samples]):
            return []

        change_points: List[int] = []
        end = n_samples
        while previous[end] > 0:
            change_points.append(int(previous[end]))
            end = int(previous[end])
        change_points.reverse()
        logger.info(
            "PELT detected %d change points in signal of length %d",
            len(change_points),
            n_samples,
        )
        return change_points
