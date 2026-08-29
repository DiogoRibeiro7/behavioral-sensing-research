"""The canonical, hardware-neutral sensor observation.

Every ingestion adapter in the toolkit converts device-specific payloads into
:class:`Observation` instances. Downstream stages -- health monitoring,
fusion, state inference, baselines, alerting -- consume only this type, which
keeps scientific code independent of any particular sensor product.

An observation records *evidence*, not behaviour. It carries the metadata
needed to weight that evidence honestly: the unit it was measured in, the
quality the device reported, the confidence an upstream estimator attached to
it, and flags describing anything that was repaired during ingestion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any

from .types import EVENT_LIKE_MODALITIES, Modality, ObservationFlag, ObservationKind
from .units import Unit, to_canonical


def require_aware(value: Any, name: str) -> datetime:
    """Return *value* as a timezone-aware :class:`datetime`.

    Naive timestamps are rejected rather than assumed to be UTC or local: in
    a longitudinal deployment that assumption silently shifts every event by
    hours and quietly breaks daily-rhythm analysis across DST boundaries.
    """
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} is not a valid ISO-8601 timestamp") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        to_pydatetime = getattr(value, "to_pydatetime", None)
        if to_pydatetime is None:
            raise TypeError(f"{name} must be a datetime or ISO-8601 string")
        parsed = to_pydatetime()

    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed


def _validate_unit_interval(value: float, name: str) -> float:
    """Return *value* after checking it lies in ``[0, 1]``."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return number


@dataclass(frozen=True, slots=True)
class Observation:
    """A single hardware-neutral sensor observation.

    Parameters
    ----------
    timestamp
        Timezone-aware instant the observation refers to.
    sensor_id
        Stable identifier of the reporting sensor within a deployment.
    modality
        What kind of physical evidence the sensor produces.
    kind
        Temporal semantics: event, persisting state, or periodic sample.
    value
        Numeric value in *unit*. Binary activations use ``0.0``/``1.0``.
    unit
        Unit of *value*; use :meth:`in_canonical_unit` to normalise.
    quality
        Device-reported measurement quality in ``[0, 1]``.
    confidence
        Confidence that *value* is correct, in ``[0, 1]``. Derived features
        such as a radar presence probability should set this below one.
    source
        Identifier of the gateway, hub, or adapter that produced the record.
    sampling_interval
        Nominal interval between samples, for ``SAMPLE`` observations.
    received_at
        When the record reached the system, used to detect late arrivals.
    flags
        Provenance annotations added during ingestion and validation.
    context
        Free-form string metadata (room, placement, firmware version).
    """

    timestamp: datetime
    sensor_id: str
    modality: Modality
    kind: ObservationKind
    value: float
    unit: Unit = Unit.NONE
    quality: float = 1.0
    confidence: float = 1.0
    source: str = ""
    sampling_interval: timedelta | None = None
    received_at: datetime | None = None
    flags: frozenset[ObservationFlag] = frozenset()
    context: Mapping[str, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Validate and normalise the record at the system boundary."""
        set_field = object.__setattr__

        set_field(self, "timestamp", require_aware(self.timestamp, "timestamp"))
        if self.received_at is not None:
            set_field(
                self, "received_at", require_aware(self.received_at, "received_at")
            )

        if not isinstance(self.sensor_id, str) or not self.sensor_id.strip():
            raise ValueError("sensor_id must be a non-empty string")
        set_field(self, "sensor_id", self.sensor_id.strip())

        set_field(self, "modality", Modality(self.modality))
        set_field(self, "kind", ObservationKind(self.kind))
        set_field(self, "unit", Unit(self.unit))

        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("value must be finite; use a missing observation instead")
        if self.unit is Unit.PROBABILITY and not 0.0 <= value <= 1.0:
            raise ValueError("probability-valued observations must lie in [0, 1]")
        if self.unit is Unit.COUNT and (value < 0 or value != int(value)):
            raise ValueError("count-valued observations must be non-negative integers")
        set_field(self, "value", value)

        set_field(self, "quality", _validate_unit_interval(self.quality, "quality"))
        set_field(
            self, "confidence", _validate_unit_interval(self.confidence, "confidence")
        )

        if not isinstance(self.source, str):
            raise TypeError("source must be a string")

        if self.sampling_interval is not None:
            if not isinstance(self.sampling_interval, timedelta):
                raise TypeError("sampling_interval must be a timedelta")
            if self.sampling_interval <= timedelta(0):
                raise ValueError("sampling_interval must be positive")

        set_field(self, "flags", frozenset(ObservationFlag(f) for f in self.flags))

        if not isinstance(self.context, Mapping):
            raise TypeError("context must be a mapping of strings to strings")
        context = {str(k): str(v) for k, v in self.context.items()}
        set_field(self, "context", MappingProxyType(context))

    # ------------------------------------------------------------------
    @property
    def is_event(self) -> bool:
        """Whether absence of this observation carries no information."""
        return self.kind is ObservationKind.EVENT

    @property
    def latency(self) -> timedelta | None:
        """Delay between the observation instant and its arrival, if known."""
        if self.received_at is None:
            return None
        return self.received_at - self.timestamp

    def evidence_weight(self) -> float:
        """Return the combined reliability of this record in ``[0, 1]``.

        Quality describes the sensing hardware and confidence describes the
        value itself. Both must hold for the record to count as strong
        evidence, so they combine multiplicatively.
        """
        return self.quality * self.confidence

    def in_canonical_unit(self) -> Observation:
        """Return an equivalent observation expressed in the canonical unit."""
        converted, unit = to_canonical(self.value, self.unit)
        if unit is self.unit:
            return self
        return replace(
            self,
            value=converted,
            unit=unit,
            flags=self.flags | {ObservationFlag.UNIT_CONVERTED},
        )

    def with_flags(self, *flags: ObservationFlag) -> Observation:
        """Return a copy with additional provenance *flags* attached."""
        if not flags:
            return self
        return replace(self, flags=self.flags | frozenset(flags))

    def shifted(self, offset: timedelta, *, flag: bool = True) -> Observation:
        """Return a copy whose timestamp is moved by *offset*.

        Used by clock-drift correction, which records the adjustment as a
        flag so that later stages can see the timestamp was not measured.
        """
        extra = frozenset({ObservationFlag.CLOCK_ADJUSTED}) if flag else frozenset()
        return replace(
            self, timestamp=self.timestamp + offset, flags=self.flags | extra
        )

    def identity(self) -> tuple[datetime, str, float]:
        """Return the key used to recognise an exact duplicate record."""
        return (self.timestamp.astimezone(timezone.utc), self.sensor_id, self.value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the observation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "sensor_id": self.sensor_id,
            "modality": self.modality.value,
            "kind": self.kind.value,
            "value": self.value,
            "unit": self.unit.value,
            "quality": self.quality,
            "confidence": self.confidence,
            "source": self.source,
            "sampling_interval": (
                self.sampling_interval.total_seconds()
                if self.sampling_interval is not None
                else None
            ),
            "received_at": (
                self.received_at.isoformat() if self.received_at is not None else None
            ),
            "flags": sorted(f.value for f in self.flags),
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Observation:
        """Rebuild an observation from :meth:`to_dict` output."""
        interval = payload.get("sampling_interval")
        return cls(
            timestamp=payload["timestamp"],
            sensor_id=payload["sensor_id"],
            modality=Modality(payload["modality"]),
            kind=ObservationKind(payload["kind"]),
            value=float(payload["value"]),
            unit=Unit(payload.get("unit", Unit.NONE)),
            quality=float(payload.get("quality", 1.0)),
            confidence=float(payload.get("confidence", 1.0)),
            source=str(payload.get("source", "")),
            sampling_interval=(
                timedelta(seconds=float(interval)) if interval is not None else None
            ),
            received_at=payload.get("received_at"),
            flags=frozenset(ObservationFlag(f) for f in payload.get("flags", ()) or ()),
            context=payload.get("context") or {},
        )


def default_kind(modality: Modality) -> ObservationKind:
    """Return the conventional :class:`ObservationKind` for *modality*.

    Adapters should override this when a device genuinely reports something
    else, but the defaults keep the dangerous case -- treating an event
    stream as a continuously sampled signal -- from happening by accident.
    """
    return (
        ObservationKind.EVENT
        if modality in EVENT_LIKE_MODALITIES
        else ObservationKind.SAMPLE
    )
