"""Heterogeneous HMM for multi-source data integration.

Moreno-Pino et al. (2022) proposed integrating heterogeneous sensor streams in a
single hidden Markov model. Here we provide a minimal interface-compatible
version that accepts concatenated feature representations from multiple sources.
"""

from .base import BaseHMM


class HeterogeneousHMM(BaseHMM):
    """HMM that integrates multiple sensor modalities."""

    def __init__(
        self,
        n_states: int = 3,
        source_weights: list[float] | None = None,
        n_iter: int = 10,
        random_state: int | None = None,
    ):
        super().__init__(n_states=n_states, n_iter=n_iter, random_state=random_state)
        self.source_weights = source_weights
