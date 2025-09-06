"""Adaptive HMM incorporating personal experience.

This model adapts transition dynamics based on user-specific experience, allowing
personalization of state persistence. The implementation follows the BaseHMM
interface while exposing an ``adaptation_rate`` parameter.
"""

from .base import BaseHMM


class AdaptiveHMM(BaseHMM):
    """Adaptive Hidden Markov Model with experience-based updates."""

    def __init__(
        self,
        n_states: int = 3,
        adaptation_rate: float = 0.5,
        n_iter: int = 10,
        random_state: int | None = None,
    ):
        super().__init__(n_states=n_states, n_iter=n_iter, random_state=random_state)
        self.adaptation_rate = adaptation_rate
