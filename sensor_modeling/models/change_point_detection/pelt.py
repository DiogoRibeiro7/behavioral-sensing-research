"""Simple placeholder PELT change-point detector."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PELTChangePointDetector:
    penalty: float = 1.0

    def detect(self, signal: np.ndarray) -> List[int]:
        """Detect change points in *signal*.

        This placeholder implementation returns an empty list and simply logs the call.
        """
        logger.info("Running placeholder PELT on signal of length %d", len(signal))
        return []
