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

__all__ = [
    "CODE_SYSTEM",
    "PROVENANCE_DERIVED_FEATURE",
    "PROVENANCE_INFERRED",
    "PROVENANCE_MEASURED",
    "RESEARCH_NOTE",
    "alert_resource",
    "bundle",
    "measured_only",
    "observation_resource",
    "state_resource",
    "summarise_provenance",
]
