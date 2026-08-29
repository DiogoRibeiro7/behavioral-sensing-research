"""Multimodal sensor fusion into a probabilistic behavioural state.

The fusion layer answers ``P(Z_t | O_1:t)`` for asynchronous, heterogeneous,
partially missing observations, and reports the answer together with the
evidence that produced it.
"""

from .defaults import EmissionDefaults, default_emission_for, default_emissions
from .emissions import (
    BernoulliEmission,
    BetaEmission,
    EmissionModel,
    GaussianEmission,
    PoissonEventEmission,
)
from .estimate import (
    EvidenceContribution,
    Explanation,
    StateEstimate,
    belief_from_mapping,
    belief_matrix,
)
from .filter import (
    FusionConfig,
    MultimodalBayesFilter,
    NonMonotonicUpdateError,
)

__all__ = [
    "BernoulliEmission",
    "BetaEmission",
    "EmissionDefaults",
    "EmissionModel",
    "EvidenceContribution",
    "Explanation",
    "FusionConfig",
    "GaussianEmission",
    "MultimodalBayesFilter",
    "NonMonotonicUpdateError",
    "PoissonEventEmission",
    "StateEstimate",
    "belief_from_mapping",
    "default_emission_for",
    "default_emissions",
    "belief_matrix",
]
