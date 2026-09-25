"""The resampling engine shared by every comparison in this package.

A bootstrap resamples whatever units the caller passes one value for: held-out
households in a real-data comparison, simulated trajectories in a simulation
study. Timestamps are never the unit. Within one home they are strongly
dependent, and resampling them would treat one household as thousands of
independent experiments; see ``docs/EVALUATION_DESIGN.md``.

Two interval methods are provided:

- **percentile**: the quantiles of the bootstrap distribution. It is simple and
  is the method every earlier result in this package used.
- **BCa** (bias-corrected and accelerated): the percentile interval corrected
  for median bias and skew in the bootstrap distribution, with the skew
  estimated by the jackknife. It returns ``None`` where the correction is
  undefined, so a caller can fall back to the percentile interval instead of
  reporting a meaningless one.

Resamples are drawn from ``numpy.random.default_rng(seed)``. The same seed
draws the same resamples on any machine running the same NumPy.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

#: Smallest number of resamples accepted. Fewer makes the interval endpoints
#: themselves noisy.
MIN_RESAMPLES = 100

_NORMAL = NormalDist()


@dataclass(frozen=True)
class Interval:
    """A confidence interval and how it was computed."""

    low: float
    high: float
    confidence: float
    method: str

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form."""
        return {
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "method": self.method,
        }


def check_settings(confidence: float, resamples: int) -> None:
    """Reject a confidence level or resample count that cannot give an interval."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if resamples < MIN_RESAMPLES:
        raise ValueError(f"resamples must be at least {MIN_RESAMPLES}")


def resample_indices(units: int, resamples: int, seed: int) -> np.ndarray:
    """Return ``(resamples, units)`` unit indices drawn with replacement."""
    if units < 1:
        raise ValueError("at least one unit is required to resample")
    return np.random.default_rng(seed).integers(0, units, size=(resamples, units))


def percentile_interval(replicates: np.ndarray, confidence: float) -> Interval:
    """The central *confidence* quantiles of the bootstrap replicates."""
    tail = (1.0 - confidence) / 2.0
    return Interval(
        low=float(np.quantile(replicates, tail)),
        high=float(np.quantile(replicates, 1.0 - tail)),
        confidence=confidence,
        method="percentile",
    )


def jackknife(
    values: np.ndarray, statistic: Callable[[np.ndarray], float]
) -> np.ndarray:
    """The statistic recomputed with each unit left out in turn."""
    if values.size < 2:
        raise ValueError("the jackknife needs at least two units")
    return np.array(
        [statistic(np.delete(values, position)) for position in range(values.size)]
    )


def bca_interval(
    replicates: np.ndarray,
    estimate: float,
    leave_one_out: np.ndarray,
    confidence: float,
) -> Interval | None:
    """The bias-corrected and accelerated interval, or ``None`` if undefined.

    The bias correction uses the share of replicates below the estimate, with
    replicates equal to it counted as half. A discrete statistic, such as the
    median of a few households, often reproduces the estimate exactly; counting
    those ties as half keeps them from biasing the correction. The acceleration
    is estimated from the jackknife values *leave_one_out*.

    ``None`` is returned when every replicate lies on one side of the estimate,
    or when the acceleration is so large that the adjusted levels stop being
    monotone. In both cases the correction has no meaning.
    """
    below = float(np.sum(replicates < estimate))
    ties = float(np.sum(replicates == estimate))
    share = (below + 0.5 * ties) / replicates.size
    if not 0.0 < share < 1.0:
        return None
    bias = _NORMAL.inv_cdf(share)

    spread = leave_one_out.mean() - leave_one_out
    scale = float(np.sum(spread**2))
    acceleration = float(np.sum(spread**3)) / (6.0 * scale**1.5) if scale > 0 else 0.0

    tail = (1.0 - confidence) / 2.0
    levels = []
    for z in (_NORMAL.inv_cdf(tail), _NORMAL.inv_cdf(1.0 - tail)):
        shifted = bias + z
        denominator = 1.0 - acceleration * shifted
        if denominator <= 0.0:
            return None
        levels.append(_NORMAL.cdf(bias + shifted / denominator))
    if not levels[0] < levels[1]:
        return None
    return Interval(
        low=float(np.quantile(replicates, levels[0])),
        high=float(np.quantile(replicates, levels[1])),
        confidence=confidence,
        method="bca",
    )


def monte_carlo_standard_error(replicates: np.ndarray | list[float]) -> float:
    """Monte Carlo standard error of the mean of independent replicates.

    For a simulation study this is the precision the replication count bought,
    ``s / sqrt(n)``; see ``docs/SIMULATION_PROTOCOLS.md``. It is zero when every
    replicate agrees.
    """
    values = np.asarray(replicates, dtype=float)
    if values.size < 2:
        raise ValueError("a Monte Carlo standard error needs at least two replicates")
    if not np.all(np.isfinite(values)):
        raise ValueError("replicates must be finite")
    spread = float(values.std(ddof=1))
    return spread / math.sqrt(values.size) if spread > 0 else 0.0
