"""Core enumerations for the canonical sensor observation model.

The types in this module are deliberately hardware-neutral. A concrete
device is mapped onto a :class:`Modality` and an :class:`ObservationKind`
by an adapter, so that downstream inference never depends on a specific
manufacturer, protocol, or product line.
"""

from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    """Sensing modality of an observation.

    The modality describes *what kind of physical evidence* a sensor
    produces, not what behaviour it implies. A ``CONTACT`` observation on a
    fridge door records that the door moved; it does not record eating.
    """

    CONTACT = "contact"
    """Binary open/close contact on an object (cupboard, fridge, drawer)."""

    DOOR = "door"
    """Contact sensor on an entrance or room door used for transitions."""

    MOTION = "motion"
    """Passive infrared or equivalent binary movement detection."""

    VIBRATION = "vibration"
    """Accelerometer-derived vibration on furniture or appliances."""

    ENVIRONMENTAL = "environmental"
    """Ambient scalar measurement (temperature, humidity, light, CO2)."""

    BED_PRESSURE = "bed_pressure"
    """Bed or chair occupancy from pressure or load-cell sensing."""

    WEARABLE_MOTION = "wearable_motion"
    """Accelerometer-derived activity counts or magnitude from a wearable."""

    WEARABLE_PHYSIOLOGY = "wearable_physiology"
    """Physiological signal from a wearable (heart rate, skin temperature)."""

    ROOM_OCCUPANCY = "room_occupancy"
    """Room-level occupancy estimate produced by an upstream device."""

    RADAR = "radar"
    """Derived feature from mmWave/radar sensing; never raw radar cubes."""

    PROXIMITY = "proximity"
    """Short-range presence beacon (BLE-style) associated with an identity."""

    OTHER = "other"
    """Modality not covered above; adapters should document the semantics."""


class ObservationKind(str, Enum):
    """Temporal semantics of an observation.

    This distinction controls what may legitimately be done with gaps in the
    record, and is the reason event streams are never forward-filled.
    """

    EVENT = "event"
    """Instantaneous occurrence. Absence of an event is *not* a zero value."""

    STATE = "state"
    """A level that persists until the next reported change (e.g. door open)."""

    SAMPLE = "sample"
    """A measurement of a continuously existing quantity at a point in time."""


class ObservationFlag(str, Enum):
    """Provenance and integrity annotations attached to an observation.

    Flags are set by ingestion and validation code, never by inference. They
    let downstream stages distinguish a genuinely measured value from one
    that was repaired, reordered, or reconstructed.
    """

    LATE_ARRIVAL = "late_arrival"
    """Observation reached the system materially after its own timestamp."""

    OUT_OF_ORDER = "out_of_order"
    """Observation was inserted before an already-ingested later observation."""

    DUPLICATE_VALUE = "duplicate_value"
    """An identical observation was already present and was collapsed."""

    UNIT_CONVERTED = "unit_converted"
    """The value was converted from the reported unit to the canonical unit."""

    CLOCK_ADJUSTED = "clock_adjusted"
    """The timestamp was corrected using an estimated per-source clock offset."""

    IMPUTED = "imputed"
    """The value was reconstructed rather than measured; treat as weak evidence."""


#: Modalities whose default temporal semantics are event-like. Adapters may
#: override the kind explicitly, but these defaults keep the common case safe.
EVENT_LIKE_MODALITIES = frozenset(
    {Modality.CONTACT, Modality.DOOR, Modality.MOTION, Modality.VIBRATION}
)
