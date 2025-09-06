"""Adaptive normalization for real-time change point detection.

Based on Gupta et al. (2022) real-time preprocessing approach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np

from ..utils.plotting import plot_change_points

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveNormalizer:
    """Normalize data online before detecting change points."""

    window: int = 20

    def fit(self, series: np.ndarray) -> "AdaptiveNormalizer":
        self.series = np.asarray(series)
        return self

    def _normalize(self) -> np.ndarray:
        normed = []
        for i in range(len(self.series)):
            start = max(0, i - self.window)
            segment = self.series[start : i + 1]
            normed.append((self.series[i] - segment.mean()) / (segment.std() + 1e-6))
        return np.array(normed)

    def predict(self, threshold: float = 3.0, plot: bool = False) -> np.ndarray:
        """Detect change points after adaptive normalization."""
        if not hasattr(self, "series"):
            raise ValueError("Model must be fitted before prediction")
        normed = self._normalize()
        diffs = np.abs(np.diff(normed))
        cps = np.where(diffs > threshold)[0] + 1
        if plot:
            plot_change_points(self.series, cps, title="Adaptive Normalization CPD")
        logger.info("AdaptiveNormalizer detected %d change points", len(cps))
        return cps
