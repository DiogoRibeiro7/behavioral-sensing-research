"""Canonical, hardware-neutral representation of sensor observations.

This package defines the single data model every other stage of the pipeline
consumes. The central claim it encodes is that a sensor record is *evidence*,
not behaviour: an :class:`Observation` carries the value, the unit it was
measured in, the reliability attached to it, and the provenance of any repair
applied during ingestion, so that later stages can weight it honestly.
"""

from .ingest import (
    ClockOffsetEstimator,
    IngestionReport,
    ObservationIngestor,
    RejectedObservation,
)
from .observation import Observation, default_kind, require_aware
from .registry import (
    SensorContractError,
    SensorRegistry,
    SensorSpec,
    UnknownSensorError,
)
from .stream import Gap, ObservationStream
from .types import Modality, ObservationFlag, ObservationKind
from .units import Unit, convert, to_canonical

__all__ = [
    "ClockOffsetEstimator",
    "Gap",
    "IngestionReport",
    "Modality",
    "Observation",
    "ObservationFlag",
    "ObservationIngestor",
    "ObservationKind",
    "ObservationStream",
    "RejectedObservation",
    "SensorContractError",
    "SensorRegistry",
    "SensorSpec",
    "Unit",
    "UnknownSensorError",
    "convert",
    "default_kind",
    "require_aware",
    "to_canonical",
]
