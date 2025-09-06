"""Genetic algorithm based change point tuning.

Simplified GA approach inspired by Awais et al. (2016).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np

from ..utils.plotting import plot_change_points

logger = logging.getLogger(__name__)


@dataclass
class GeneticOptimizationCPD:
    """Use a tiny genetic search to tune the detection threshold."""

    population: int = 5
    generations: int = 5

    def fit(self, series: np.ndarray) -> "GeneticOptimizationCPD":
        self.series = np.asarray(series)
        self.threshold_ = self._ga_search()
        logger.debug("Optimized threshold to %.3f", self.threshold_)
        return self

    def _ga_search(self) -> float:
        rng = np.random.default_rng(0)
        candidates = rng.uniform(0.1, 2.0, self.population)
        for _ in range(self.generations):
            scores = [self._fitness(th) for th in candidates]
            best = np.argmin(scores)
            # mutate around best
            candidates = candidates[best] + rng.normal(0, 0.1, self.population)
            candidates = np.clip(candidates, 0.1, 2.0)
        return candidates[0]

    def _fitness(self, threshold: float) -> float:
        diffs = np.abs(np.diff(self.series))
        cps = np.where(diffs > threshold)[0]
        # simple fitness: prefer at least one cp around middle
        target = len(self.series) // 2
        if len(cps) == 0:
            return target
        return min(abs(cp - target) for cp in cps)

    def predict(self, plot: bool = False) -> np.ndarray:
        """Detect change points using GA-optimized threshold."""
        if not hasattr(self, "threshold_"):
            raise ValueError("Model must be fitted before prediction")
        diffs = np.abs(np.diff(self.series))
        cps = np.where(diffs > self.threshold_)[0] + 1
        if plot:
            plot_change_points(self.series, cps, title="GA Optimization CPD")
        logger.info("GeneticOptimizationCPD detected %d change points", len(cps))
        return cps
