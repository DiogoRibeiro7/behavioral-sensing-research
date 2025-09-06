"""Scaled Dirichlet HMM with variational inference.

Implements the variational inference approach of Manouchehri & Bouguila (2023)
for scaled Dirichlet mixture emissions. This simplified version focuses on the
interface compatibility and leverages :class:`~sensor_modeling.hmm.base.BaseHMM`
for core learning routines.
"""

from .base import BaseHMM


class ScaledDirichletHMM(BaseHMM):
    """HMM with scaled Dirichlet emissions."""

    def __init__(
        self,
        n_states: int = 3,
        concentration: float = 1.0,
        n_iter: int = 10,
        random_state: int | None = None,
    ):
        super().__init__(n_states=n_states, n_iter=n_iter, random_state=random_state)
        self.concentration = concentration
