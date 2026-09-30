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
    UpdateObserver,
    WindowTerms,
)
from .history import (
    ChannelHistory,
    HistoryAwareBayesFilter,
    HistoryConfig,
    HistoryModel,
    PosteriorDecomposition,
)
from .regime import (
    ONLINE,
    EvidenceLeakageError,
    EvidenceSummary,
    InferenceMode,
    InferenceRegime,
    NotEnumerated,
    ReportedEstimate,
    assert_respects_horizon,
    check_evidence,
    check_evidence_access,
    evidence_from_dict,
    regime_beliefs,
    regime_estimates,
    require_online,
    require_same_regime,
)
from .smoothing import smooth_beliefs, smooth_estimates

__all__ = [
    "UpdateObserver",
    "WindowTerms",
    "ONLINE",
    "EvidenceLeakageError",
    "EvidenceSummary",
    "InferenceMode",
    "InferenceRegime",
    "NotEnumerated",
    "ReportedEstimate",
    "assert_respects_horizon",
    "check_evidence",
    "check_evidence_access",
    "evidence_from_dict",
    "regime_beliefs",
    "regime_estimates",
    "require_online",
    "require_same_regime",
    "smooth_beliefs",
    "smooth_estimates",
    "BernoulliEmission",
    "BetaEmission",
    "EmissionDefaults",
    "EmissionModel",
    "EvidenceContribution",
    "Explanation",
    "FusionConfig",
    "GaussianEmission",
    "ChannelHistory",
    "HistoryAwareBayesFilter",
    "HistoryConfig",
    "HistoryModel",
    "PosteriorDecomposition",
    "MultimodalBayesFilter",
    "NonMonotonicUpdateError",
    "PoissonEventEmission",
    "StateEstimate",
    "belief_from_mapping",
    "default_emission_for",
    "default_emissions",
    "belief_matrix",
]
