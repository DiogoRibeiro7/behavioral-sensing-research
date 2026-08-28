"""Multimodal sensor fusion into a probabilistic behavioural state.

The fusion layer answers ``P(Z_t | O_1:t)`` for asynchronous, heterogeneous,
partially missing observations, and reports the answer together with the
evidence that produced it.
"""

from .emissions import (
    BernoulliEmission,
    BetaEmission,
    EmissionModel,
    GaussianEmission,
    PoissonEventEmission,
)
from .estimate import (
    EvidenceContribution,
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
    "EmissionModel",
    "EvidenceContribution",
    "FusionConfig",
    "GaussianEmission",
    "MultimodalBayesFilter",
    "NonMonotonicUpdateError",
    "PoissonEventEmission",
    "StateEstimate",
    "belief_from_mapping",
    "belief_matrix",
]
