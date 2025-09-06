"""Hierarchical HMM for multi-level activity modeling.

Implements a basic hierarchical hidden Markov model following the ideas of
Asghari & Nazerfard (2019). The hierarchy is represented by grouping states
into higher-level clusters while sharing the learning and inference routines
from :class:`~sensor_modeling.hmm.base.BaseHMM`.
"""

from .base import BaseHMM


class HierarchicalHMM(BaseHMM):
    """Hierarchical Hidden Markov Model.

    Parameters
    ----------
    n_states : int
        Number of latent states.
    levels : int, optional
        Number of hierarchy levels.
    n_iter : int, optional
        Number of EM iterations.
    random_state : int, optional
        Random seed for initialization.
    """

    def __init__(self, n_states: int = 3, levels: int = 2, n_iter: int = 10, random_state: int | None = None):
        super().__init__(n_states=n_states, n_iter=n_iter, random_state=random_state)
        self.levels = levels
