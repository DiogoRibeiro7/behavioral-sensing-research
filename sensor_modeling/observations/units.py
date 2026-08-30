"""Unit handling for canonical sensor observations.

Ambient deployments routinely mix units: one gateway reports Celsius, another
Fahrenheit; one radar reports metres, another centimetres. Silently mixing
them corrupts every downstream statistic, so units are explicit on every
observation and converted to a canonical unit at the ingestion boundary.
"""

from __future__ import annotations

from enum import Enum


class Unit(str, Enum):
    """Units supported by the canonical observation model."""

    NONE = "none"
    """Dimensionless value, including binary 0/1 activations."""

    PROBABILITY = "probability"
    """Value constrained to ``[0, 1]``, used for derived presence features."""

    COUNT = "count"
    """Non-negative integer count (tracks, steps, activation counts)."""

    CELSIUS = "degC"
    FAHRENHEIT = "degF"
    PERCENT = "percent"
    LUX = "lux"
    PPM = "ppm"
    METRE = "m"
    CENTIMETRE = "cm"
    METRE_PER_SECOND = "m/s"
    CENTIMETRE_PER_SECOND = "cm/s"
    G = "g"
    """Acceleration in multiples of standard gravity."""

    MILLI_G = "mg"
    BPM = "bpm"
    """Beats or breaths per minute."""

    SECOND = "s"
    MINUTE = "min"


#: Multiplicative conversions to the canonical unit of each dimension.
#: Affine conversions (temperature) are handled separately in :func:`convert`.
_SCALE_TO_CANONICAL: dict[Unit, tuple[Unit, float]] = {
    Unit.CENTIMETRE: (Unit.METRE, 0.01),
    Unit.CENTIMETRE_PER_SECOND: (Unit.METRE_PER_SECOND, 0.01),
    Unit.MILLI_G: (Unit.G, 0.001),
    Unit.MINUTE: (Unit.SECOND, 60.0),
}

#: Units that are already canonical for their dimension.
_CANONICAL_UNITS = frozenset(
    {
        Unit.NONE,
        Unit.PROBABILITY,
        Unit.COUNT,
        Unit.CELSIUS,
        Unit.PERCENT,
        Unit.LUX,
        Unit.PPM,
        Unit.METRE,
        Unit.METRE_PER_SECOND,
        Unit.G,
        Unit.BPM,
        Unit.SECOND,
    }
)


def canonical_unit(unit: Unit) -> Unit:
    """Return the canonical unit for the dimension of *unit*."""
    if unit is Unit.FAHRENHEIT:
        return Unit.CELSIUS
    scaled = _SCALE_TO_CANONICAL.get(unit)
    return scaled[0] if scaled is not None else unit


def convert(value: float, from_unit: Unit, to_unit: Unit) -> float:
    """Convert *value* between two units of the same dimension.

    Raises
    ------
    ValueError
        If the units belong to different dimensions and no conversion is
        defined. Mismatched dimensions are a data-integrity problem, not
        something to paper over with a pass-through.
    """
    if from_unit is to_unit:
        return float(value)

    if from_unit is Unit.FAHRENHEIT and to_unit is Unit.CELSIUS:
        return (float(value) - 32.0) * 5.0 / 9.0
    if from_unit is Unit.CELSIUS and to_unit is Unit.FAHRENHEIT:
        return float(value) * 9.0 / 5.0 + 32.0

    from_scaled = _SCALE_TO_CANONICAL.get(from_unit)
    to_scaled = _SCALE_TO_CANONICAL.get(to_unit)
    from_base, from_factor = from_scaled if from_scaled else (from_unit, 1.0)
    to_base, to_factor = to_scaled if to_scaled else (to_unit, 1.0)
    if from_base is to_base:
        return float(value) * from_factor / to_factor

    raise ValueError(
        f"cannot convert {from_unit.value} to {to_unit.value}: different dimensions"
    )


def to_canonical(value: float, unit: Unit) -> tuple[float, Unit]:
    """Return *value* expressed in the canonical unit for its dimension."""
    target = canonical_unit(unit)
    if target is unit:
        return float(value), unit
    return convert(value, unit, target), target
