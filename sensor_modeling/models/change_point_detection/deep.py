"""Simple deep-learning-based change-point detector.

This implementation uses a small multilayer perceptron from scikit-learn
as a lightweight stand-in for more sophisticated deep architectures.
It trains on sliding windows of the input sequence and classifies whether
the next time step contains a change point.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier


@dataclass
class DeepCPDConfig:
    """Configuration for :class:`DeepChangePointDetector`."""

    window: int = 5
    hidden: int = 32
    threshold: float = 0.5


class DeepChangePointDetector:
    """Detect change points using a small neural network."""

    def __init__(self, cfg: DeepCPDConfig | None = None):
        """Initialize the detector with optional configuration."""
        self.cfg = cfg or DeepCPDConfig()
        hidden = (self.cfg.hidden,)
        self.model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=200)

    def _build_dataset(self, seq: Sequence[float]):
        """Construct sliding-window features and change labels."""
        x, y = [], []
        w = self.cfg.window
        arr = np.asarray(seq, dtype=float)
        for i in range(len(arr) - w - 1):
            window = arr[i : i + w]  # noqa: E203
            label = 1 if arr[i + w] != arr[i + w - 1] else 0
            x.append(window)
            y.append(label)
        return np.vstack(x), np.array(y)

    def fit(self, seq: Sequence[float]) -> DeepChangePointDetector:
        """Train the underlying classifier on a sequence."""
        X, y = self._build_dataset(seq)
        if X.size:
            self.model.fit(X, y)
        return self

    def detect(self, seq: Sequence[float]) -> np.ndarray:
        """Return an array with 1 where a change point is predicted."""
        X, _ = self._build_dataset(seq)
        if X.size == 0:
            return np.zeros(len(seq), dtype=int)
        probs = self.model.predict_proba(X)[:, 1]
        preds = probs > self.cfg.threshold
        result = np.zeros(len(seq), dtype=int)
        start = self.cfg.window + 1
        stop = start + len(preds)
        result[start:stop] = preds
        return result
