"""Energy-efficient change point detection.

Simplified implementation of the CPAM algorithm from Cook et al. (2020).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np

from ..utils.plotting import plot_change_points

logger = logging.getLogger(__name__)


@dataclass
class EnergyEfficientCPD:
    """Detect change points using energy-efficient statistics."""

    window: int = 10

    def fit(self, series: np.ndarray) -> "EnergyEfficientCPD":
        self.series = np.asarray(series)
        return self

    def predict(self, threshold: float = 0.5, plot: bool = False) -> np.ndarray:
        """Detect change points based on energy across windows."""
        if not hasattr(self, "series"):
            raise ValueError("Model must be fitted before prediction")
        n = len(self.series)
        energies = [
            np.sum((self.series[i : i + self.window] - self.series[i : i + self.window].mean()) ** 2)
            for i in range(n - self.window)
        ]
        diffs = np.abs(np.diff(energies))
        cps = np.where(diffs > threshold)[0] + 1
        if plot:
            plot_change_points(self.series, cps, title="Energy Efficient CPD")
        logger.info("EnergyEfficientCPD detected %d change points", len(cps))
        return cps
