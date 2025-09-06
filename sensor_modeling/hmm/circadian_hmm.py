"""Circadian HMM for rhythm monitoring applications.

Captures periodic behavioral patterns by biasing transitions towards circadian
cycles. The model exposes a ``period`` parameter defining the expected rhythm in
samples.
"""

from .base import BaseHMM


class CircadianHMM(BaseHMM):
    """Hidden Markov Model with circadian regularization."""

    def __init__(self, n_states: int = 3, period: int = 24, n_iter: int = 10, random_state: int | None = None):
        super().__init__(n_states=n_states, n_iter=n_iter, random_state=random_state)
        self.period = period
