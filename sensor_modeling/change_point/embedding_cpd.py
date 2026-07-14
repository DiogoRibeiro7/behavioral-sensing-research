"""Neural embedding change point detection.

Implementation inspired by Dadi et al. (2021) "ADAF".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..utils.plotting import plot_change_points
from ._validation import (
    validate_positive_int,
    validate_positive_threshold,
    validate_series,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingCPD:
    """Detect change points using simple neural embeddings.

    This simplified detector follows the spirit of the neural embedding
    approach by Dadi et al. (2021). It computes rolling mean embeddings and
    flags large embedding differences as change points.
    """

    window: int = 5

    def __post_init__(self) -> None:
        validate_positive_int(self.window, "window")

    def fit(self, series: np.ndarray) -> EmbeddingCPD:
        """Learn embeddings from the input series."""
        self.series = validate_series(series)
        kernel = np.ones(self.window) / self.window
        self.embeddings = np.convolve(self.series, kernel, mode="valid")
        logger.debug("Computed embeddings of length %d", len(self.embeddings))
        return self

    def predict(self, threshold: float = 0.2, plot: bool = False) -> np.ndarray:
        """Return detected change point indices.

        Args:
            threshold: Difference threshold on successive embeddings.
            plot: Whether to plot results using utility helpers.
        """
        if not hasattr(self, "embeddings"):
            raise ValueError("Model must be fitted before prediction")
        threshold = validate_positive_threshold(threshold)
        diffs = np.abs(np.diff(self.embeddings))
        cps = np.where(diffs > threshold)[0] + 1
        if plot:
            plot_change_points(self.series, cps, title="Embedding CPD")
        logger.info("EmbeddingCPD detected %d change points", len(cps))
        return cps
