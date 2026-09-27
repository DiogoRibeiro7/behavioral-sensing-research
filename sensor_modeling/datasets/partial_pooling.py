"""Partial pooling of household-specific parameters toward a population.

Three ways to give a household its own parameter sit on one scale:

- **Population structure.** Every household gets the population value
  ``θ_pop``, fitted on other households.
- **Household adaptation.** Each household gets ``θ_pop`` plus a deviation
  estimated from its own data and shrunk toward zero.
- **Unconstrained per-household fitting.** Each household gets its own raw
  estimate ``θ_raw = sum / n`` and ignores the population.

The framework here is the middle one, for any parameter estimated as a mean:
a probability from successes and trials, or a mean count from a total and a
number of windows. Given a household's ``n`` observations with total ``s``::

    θ_raw = s / n
    w     = n / (n + κ)
    θ̂     = θ_pop + w · (θ_raw − θ_pop)      which equals (s + κ θ_pop) / (n + κ)

- ``θ_pop + δ`` with ``δ = w · (θ_raw − θ_pop)``: the population parameter plus
  a household deviation, shrunk toward zero by the factor ``w``.
- ``κ`` is the pooling strength: how many observations the population is worth.
  It is the one setting, declared in :class:`PoolingConfig`.
- With no data, ``w = 0`` and the household gets the population value exactly.
  As ``n`` grows, ``w → 1`` and the estimate approaches the raw one. ``κ → 0`` is
  unconstrained per-household fitting, and ``κ → ∞`` is the population alone.
- ``1 − w = κ / (n + κ)`` is the effective shrinkage: the share of the estimate
  that comes from the population.

For a probability, ``θ̂`` is the posterior mean under a Beta prior with mean
``θ_pop`` and ``κ`` pseudo-observations. For a Poisson mean, it is the posterior
mean under a Gamma prior with mean ``θ_pop`` and ``κ`` pseudo-windows. It is
closed-form, so fitting is deterministic, and it is always between the raw and
the population value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoolingConfig:
    """How strongly household estimates are pooled toward the population.

    Attributes
    ----------
    strength
        ``κ``, the population's weight in observations. The default, 288
        five-minute windows, is 24 labelled hours: the same pooling strength
        the Phase 3.1 periodic prior declares for its household deviations.
    """

    strength: float = 288.0

    def __post_init__(self) -> None:
        """Validate the strength."""
        value = float(self.strength)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("strength must be positive and finite")
        object.__setattr__(self, "strength", value)

    def to_dict(self) -> dict[str, float]:
        """Return a serialisable form."""
        return {"strength": self.strength}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PoolingConfig:
        """Rebuild a configuration written by :meth:`to_dict`."""
        return cls(strength=float(payload["strength"]))


def _vector(values: Any, name: str) -> np.ndarray:
    array = np.array(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


@dataclass(frozen=True)
class PooledEstimate:
    """A household's partially pooled estimate, element by element.

    Attributes
    ----------
    population
        ``θ_pop``, the population value each element is shrunk toward.
    total, count
        The household's own sufficient statistics: the sum of its
        observations and their number.
    strength
        ``κ``, the pooling strength used.
    """

    population: np.ndarray
    total: np.ndarray
    count: np.ndarray
    strength: float

    def __post_init__(self) -> None:
        """Validate the inputs and freeze copies of them."""
        population = _vector(self.population, "population")
        total = _vector(self.total, "total")
        count = _vector(self.count, "count")
        if not population.shape == total.shape == count.shape:
            raise ValueError("population, total and count need the same length")
        if not np.all(np.isfinite(population)):
            raise ValueError("population values must be finite")
        if not np.all(np.isfinite(total)) or (total < 0).any():
            raise ValueError("totals must be finite and non-negative")
        if not np.all(np.isfinite(count)) or (count < 0).any():
            raise ValueError("counts must be finite and non-negative")
        if ((count == 0) & (total > 0)).any():
            raise ValueError("a total needs at least one observation")
        strength = float(self.strength)
        if not math.isfinite(strength) or strength <= 0.0:
            raise ValueError("strength must be positive and finite")
        for array in (population, total, count):
            array.setflags(write=False)
        object.__setattr__(self, "population", population)
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "strength", strength)

    @property
    def weight(self) -> np.ndarray:
        """``w = n / (n + κ)``: how much the household's own data counts."""
        values: np.ndarray = self.count / (self.count + self.strength)
        return values

    @property
    def shrinkage(self) -> np.ndarray:
        """``1 − w = κ / (n + κ)``: the share that comes from the population."""
        values: np.ndarray = self.strength / (self.count + self.strength)
        return values

    @property
    def raw(self) -> np.ndarray:
        """``θ_raw = s / n``, unconstrained; NaN where the household has no data."""
        values: np.ndarray = np.divide(
            self.total,
            self.count,
            out=np.full_like(self.total, np.nan),
            where=self.count > 0,
        )
        return values

    @property
    def deviation(self) -> np.ndarray:
        """``δ = w · (θ_raw − θ_pop)``: the household deviation, shrunk toward zero."""
        values: np.ndarray = self.pooled - self.population
        return values

    @property
    def pooled(self) -> np.ndarray:
        """``θ̂ = θ_pop + w · (θ_raw − θ_pop)``, the household's estimate.

        It equals ``(s + κ θ_pop) / (n + κ)``. It is computed in this form so
        that a household with no data gets ``θ_pop`` exactly, not up to
        rounding.
        """
        values: np.ndarray = self.population.copy()
        seen = self.count > 0
        values[seen] = self.population[seen] + self.weight[seen] * (
            self.total[seen] / self.count[seen] - self.population[seen]
        )
        return values

    def to_dict(self) -> dict[str, Any]:
        """Return the inputs and every derived quantity, serialisable."""

        def listed(values: np.ndarray) -> list[float | None]:
            return [float(v) if math.isfinite(v) else None for v in values]

        return {
            "population": listed(self.population),
            "total": listed(self.total),
            "count": listed(self.count),
            "strength": self.strength,
            "raw": listed(self.raw),
            "pooled": listed(self.pooled),
            "deviation": listed(self.deviation),
            "shrinkage": listed(self.shrinkage),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PooledEstimate:
        """Rebuild an estimate written by :meth:`to_dict`.

        Only the inputs are read. The derived quantities are recomputed, and
        a payload whose derived quantities disagree with its inputs is refused.
        """
        estimate = cls(
            np.array(payload["population"], dtype=float),
            np.array(payload["total"], dtype=float),
            np.array(payload["count"], dtype=float),
            float(payload["strength"]),
        )
        if estimate.to_dict() != dict(payload):
            raise ValueError("the payload's derived values disagree with its inputs")
        return estimate


def pool(
    population: Any, total: Any, count: Any, config: PoolingConfig
) -> PooledEstimate:
    """Pool a household's own statistics toward *population*.

    Parameters
    ----------
    population
        ``θ_pop`` per element, fitted without this household.
    total, count
        The household's sum of observations and their number, per element.
        Zero counts give the population value.
    config
        The pooling strength.
    """
    return PooledEstimate(
        np.asarray(population, dtype=float),
        np.asarray(total, dtype=float),
        np.asarray(count, dtype=float),
        config.strength,
    )
