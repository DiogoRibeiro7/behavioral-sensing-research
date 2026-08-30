"""Interoperability exports that keep measurement and inference distinct.

Once a behavioural conclusion is written into a clinical record it looks like
every other entry there. This package exports measurements, derived features,
inferred states and algorithmic alerts as distinguishable resources carrying
explicit provenance, so a downstream reader can always tell which is which.

The exports are FHIR-*style* prototypes for interoperability research. They
are not validated FHIR profiles and their codes are not drawn from any
recognised terminology.
"""

from .fhir import (
    CODE_SYSTEM,
    PROVENANCE_DERIVED_FEATURE,
    PROVENANCE_INFERRED,
    PROVENANCE_MEASURED,
    RESEARCH_NOTE,
    alert_resource,
    bundle,
    measured_only,
    observation_resource,
    state_resource,
    summarise_provenance,
)
from .privacy import (
    DEFAULT_REDACTED_KEYS,
    Pseudonymiser,
    RedactionPolicy,
    SaltError,
    identifiers_in,
    redact,
    redact_bundle,
    research_identifier,
)

__all__ = [
    "CODE_SYSTEM",
    "DEFAULT_REDACTED_KEYS",
    "PROVENANCE_DERIVED_FEATURE",
    "PROVENANCE_INFERRED",
    "PROVENANCE_MEASURED",
    "RESEARCH_NOTE",
    "Pseudonymiser",
    "RedactionPolicy",
    "SaltError",
    "alert_resource",
    "bundle",
    "identifiers_in",
    "measured_only",
    "observation_resource",
    "redact",
    "redact_bundle",
    "research_identifier",
    "state_resource",
    "summarise_provenance",
]
