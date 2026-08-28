"""Configurable ontology of latent behavioural states."""

from .markov import build_generator, stationary_distribution, transition_matrix
from .ontology import (
    DEFAULT_DWELL,
    DEFAULT_JUMPS,
    DEFAULT_ROOMS,
    DEFAULT_STATES,
    BehaviouralState,
    StateOntology,
)

__all__ = [
    "DEFAULT_DWELL",
    "DEFAULT_JUMPS",
    "DEFAULT_ROOMS",
    "DEFAULT_STATES",
    "BehaviouralState",
    "StateOntology",
    "build_generator",
    "stationary_distribution",
    "transition_matrix",
]
