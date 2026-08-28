"""Declarative descriptions of the sensors in a deployment.

A :class:`SensorRegistry` is the place where a deployment states what each
sensor *is*: its modality, its temporal semantics, the unit it reports in, the
room it observes, how often it is expected to report, and whether its
activations can be attributed to a specific person.

Inference code reads the registry instead of pattern-matching on sensor names,
which is what keeps the platform sensor-agnostic. Ingestion uses it to
normalise units and to reject records that contradict the declared contract.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import timedelta

from .observation import Observation, default_kind
from .types import Modality, ObservationFlag, ObservationKind
from .units import Unit, canonical_unit, convert

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensorSpec:
    """Static description of one sensor in a deployment.

    Parameters
    ----------
    sensor_id
        Stable identifier matching :attr:`Observation.sensor_id`.
    modality
        The kind of physical evidence produced.
    kind
        Temporal semantics; defaults to the convention for *modality*.
    unit
        Unit the sensor is expected to report in. Observations arriving in a
        convertible unit are converted; incompatible units are rejected.
    room
        Room or zone observed, when the sensor is spatially localised.
    expected_interval
        Nominal reporting interval. ``None`` for purely event-driven sensors,
        which cannot be checked for silence in the same way.
    value_range
        Inclusive ``(low, high)`` bound on plausible values in *unit*.
    prior_reliability
        Prior probability in ``(0, 1]`` that this sensor is working. Used as
        the starting point for online health estimation.
    attributable
        Whether an activation identifies *who* generated it. True only for
        person-bound sensing such as a worn device or a personal beacon.
    description
        Human-readable note recording the precise semantics of the value.
    """

    sensor_id: str
    modality: Modality
    kind: ObservationKind | None = None
    unit: Unit = Unit.NONE
    room: str | None = None
    expected_interval: timedelta | None = None
    value_range: tuple[float, float] | None = None
    prior_reliability: float = 0.99
    attributable: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the declared sensor contract."""
        set_field = object.__setattr__
        if not isinstance(self.sensor_id, str) or not self.sensor_id.strip():
            raise ValueError("sensor_id must be a non-empty string")
        set_field(self, "sensor_id", self.sensor_id.strip())
        set_field(self, "modality", Modality(self.modality))
        set_field(
            self,
            "kind",
            (
                default_kind(self.modality)
                if self.kind is None
                else ObservationKind(self.kind)
            ),
        )
        set_field(self, "unit", Unit(self.unit))

        if not 0.0 < float(self.prior_reliability) <= 1.0:
            raise ValueError("prior_reliability must lie in (0, 1]")
        set_field(self, "prior_reliability", float(self.prior_reliability))

        if self.expected_interval is not None:
            if not isinstance(self.expected_interval, timedelta):
                raise TypeError("expected_interval must be a timedelta")
            if self.expected_interval <= timedelta(0):
                raise ValueError("expected_interval must be positive")

        if self.value_range is not None:
            low, high = (float(v) for v in self.value_range)
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError("value_range must be a finite (low, high) pair")
            set_field(self, "value_range", (low, high))

    def contains(self, value: float) -> bool:
        """Whether *value* lies inside the declared plausible range."""
        if self.value_range is None:
            return True
        low, high = self.value_range
        return low <= value <= high


class UnknownSensorError(KeyError):
    """Raised when an observation refers to a sensor that is not registered."""


class SensorContractError(ValueError):
    """Raised when an observation contradicts its declared :class:`SensorSpec`."""


@dataclass
class SensorRegistry:
    """A collection of :class:`SensorSpec` records keyed by sensor identifier."""

    specs: dict[str, SensorSpec] = field(default_factory=dict)

    @classmethod
    def from_specs(cls, specs: Iterable[SensorSpec]) -> SensorRegistry:
        """Build a registry from an iterable of specifications."""
        registry = cls()
        for spec in specs:
            registry.add(spec)
        return registry

    def add(self, spec: SensorSpec) -> None:
        """Register *spec*, replacing any previous entry for the same sensor."""
        if spec.sensor_id in self.specs:
            logger.warning("Replacing existing spec for sensor '%s'", spec.sensor_id)
        self.specs[spec.sensor_id] = spec

    def __contains__(self, sensor_id: object) -> bool:
        return sensor_id in self.specs

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self) -> Iterator[SensorSpec]:
        return iter(self.specs.values())

    def __getitem__(self, sensor_id: str) -> SensorSpec:
        try:
            return self.specs[sensor_id]
        except KeyError as exc:
            raise UnknownSensorError(f"sensor '{sensor_id}' is not registered") from exc

    def get(self, sensor_id: str) -> SensorSpec | None:
        """Return the spec for *sensor_id*, or ``None`` when unregistered."""
        return self.specs.get(sensor_id)

    def sensor_ids(self) -> list[str]:
        """Return registered sensor identifiers in insertion order."""
        return list(self.specs)

    def by_modality(self, modality: Modality) -> list[SensorSpec]:
        """Return every spec with the given *modality*."""
        return [spec for spec in self.specs.values() if spec.modality is modality]

    def by_room(self, room: str) -> list[SensorSpec]:
        """Return every spec observing *room*."""
        return [spec for spec in self.specs.values() if spec.room == room]

    def rooms(self) -> list[str]:
        """Return the distinct rooms covered by the registry, sorted."""
        return sorted({spec.room for spec in self.specs.values() if spec.room})

    def subset(self, sensor_ids: Iterable[str]) -> SensorRegistry:
        """Return a registry restricted to *sensor_ids*.

        Sensor-ablation experiments use this to build a deployment with a
        modality removed without touching any inference code.
        """
        wanted = list(sensor_ids)
        missing = [sid for sid in wanted if sid not in self.specs]
        if missing:
            raise UnknownSensorError(f"unregistered sensors: {sorted(missing)}")
        return SensorRegistry({sid: self.specs[sid] for sid in wanted})

    # ------------------------------------------------------------------
    def normalise(self, observation: Observation) -> Observation:
        """Validate *observation* against its spec and normalise its unit.

        Returns
        -------
        Observation
            The observation expressed in the unit declared by the spec.

        Raises
        ------
        UnknownSensorError
            If the sensor is not registered. Unregistered sensors are refused
            rather than guessed at, because a guessed modality would silently
            change what the value is taken to mean.
        SensorContractError
            If the declared unit is incompatible, or the modality disagrees.
        """
        spec = self[observation.sensor_id]

        if observation.modality is not spec.modality:
            raise SensorContractError(
                f"sensor '{spec.sensor_id}' is registered as {spec.modality.value} "
                f"but reported {observation.modality.value}"
            )

        result = observation
        if observation.unit is not spec.unit:
            if canonical_unit(observation.unit) is not canonical_unit(spec.unit):
                raise SensorContractError(
                    f"sensor '{spec.sensor_id}' expects {spec.unit.value} but "
                    f"reported {observation.unit.value}"
                )
            converted = convert(observation.value, observation.unit, spec.unit)
            result = replace(result, value=converted, unit=spec.unit)
            result = result.with_flags(ObservationFlag.UNIT_CONVERTED)

        if spec.kind is not None and result.kind is not spec.kind:
            result = replace(result, kind=spec.kind)
        return result

    def in_range(self, observation: Observation) -> bool:
        """Whether *observation* falls inside its declared plausible range."""
        spec = self.get(observation.sensor_id)
        return True if spec is None else spec.contains(observation.value)

    def to_dict(self) -> dict[str, Mapping[str, object]]:
        """Return a serialisable description of the registry."""
        return {
            spec.sensor_id: {
                "modality": spec.modality.value,
                "kind": spec.kind.value if spec.kind is not None else None,
                "unit": spec.unit.value,
                "room": spec.room,
                "expected_interval": (
                    spec.expected_interval.total_seconds()
                    if spec.expected_interval is not None
                    else None
                ),
                "value_range": list(spec.value_range) if spec.value_range else None,
                "prior_reliability": spec.prior_reliability,
                "attributable": spec.attributable,
                "description": spec.description,
            }
            for spec in self.specs.values()
        }
