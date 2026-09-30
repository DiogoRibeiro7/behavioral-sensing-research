"""Phase 5: the contract for independently collected, annotated smart-home datasets.

- :mod:`.contract`: what an adapter exposes, in the dataset's own terms.
- :mod:`.mapping`: the declared correspondence of native labels, sensor types
  and locations to the repository's ontology.
- :mod:`.validation`: checks of a household against the contract and mapping.
- :mod:`.canonical`: conversion into the repository's canonical recording, with
  everything left behind counted.

No model is scored here. See ``docs/EXTERNAL_DATASET_CONTRACT.md``.
"""

from .canonical import CanonicalHousehold, to_canonical
from .contract import (
    Annotation,
    AnnotationSemantics,
    ContractError,
    CsvAdapter,
    CsvLayout,
    DatasetAdapter,
    DatasetProvenance,
    HouseholdData,
    InMemoryAdapter,
    OccupancyPeriod,
    RawEvent,
    SensorDescription,
)
from .mapping import (
    CANONICAL_ROOMS,
    MISSING,
    UNDECLARED,
    MappingEntry,
    MappingError,
    OntologyMapping,
    Outcome,
    Resolution,
    SensorSemantics,
    ambiguous,
    approximate,
    exact,
    unmappable,
)
from .validation import (
    DatasetReport,
    Issue,
    Severity,
    ValidationReport,
    validate_dataset,
    validate_household,
)

__all__ = [
    "CANONICAL_ROOMS",
    "MISSING",
    "UNDECLARED",
    "Annotation",
    "AnnotationSemantics",
    "CanonicalHousehold",
    "ContractError",
    "CsvAdapter",
    "CsvLayout",
    "DatasetAdapter",
    "DatasetProvenance",
    "DatasetReport",
    "HouseholdData",
    "InMemoryAdapter",
    "Issue",
    "MappingEntry",
    "MappingError",
    "OccupancyPeriod",
    "OntologyMapping",
    "Outcome",
    "RawEvent",
    "Resolution",
    "SensorDescription",
    "SensorSemantics",
    "Severity",
    "ValidationReport",
    "ambiguous",
    "approximate",
    "exact",
    "to_canonical",
    "unmappable",
    "validate_dataset",
    "validate_household",
]
