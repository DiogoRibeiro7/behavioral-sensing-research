"""Time of day for matched-information experiments: definition and encodings.

Definition
----------
The time of day of a prediction moment ``t`` in a household whose clocks
follow zone ``Z`` is the local wall-clock hour

.. math:: h(t) = \\text{hour of } t \\text{ read in } Z, \\qquad h \\in \\{0, \\dots, 23\\}.

This is the quantity the optional circadian prior reads
(``StateOntology.transition`` uses ``at.hour``), so an information set that
includes time of day gives every model exactly what the filter can use.

Encodings
---------
An encoding re-expresses ``h`` for a model. It adds no information, so every
encoding below stays within the same information set.

- **Ordinal:** ``h`` itself. Suits trees, which can split anywhere.
- **One-hot:** 24 indicators, one free parameter per hour.
- **Cyclic:** the angle of the midpoint of hour ``h`` on the 24-hour clock,

  .. math:: \\theta(h) = \\frac{2\\pi\\,(h + \\tfrac12)}{24},

  encoded with ``K`` harmonics as
  :math:`(\\sin k\\theta, \\cos k\\theta)` for :math:`k = 1, \\dots, K`, which
  gives ``2K`` columns. The hours lie evenly on a circle, so hour 23 is exactly
  as close to hour 0 as hour 0 is to hour 1.

With one harmonic, a linear score :math:`\\beta \\sin\\theta + \\gamma
\\cos\\theta` equals :math:`A \\cos(\\theta - \\varphi)`, where
:math:`A = \\sqrt{\\beta^2 + \\gamma^2}` and :math:`\\varphi =
\\operatorname{atan2}(\\beta, \\gamma)`. That is one daily cycle, with amplitude
``A``, peaking at clock hour :math:`24\\varphi / 2\\pi - \\tfrac12`
(:func:`peak_hour`). Further harmonics add shorter cycles, such as a morning and
an evening peak. With ``K = 11`` the encoding has 22 of the 24 degrees of
freedom of one-hot, so ``K`` is limited to 1–11.

Daylight saving time
--------------------
Time of day follows the clock, not elapsed time since midnight. A day with a
daylight-saving change still runs 00:00 to 23:59 on the clock, even though it
lasts 23 or 25 hours.

- **Spring forward.** Local times from 02:00 to 02:59 do not exist. An
  absolute instant never maps to them. The evaluation grid, however, steps by
  wall-clock arithmetic, as the online pipeline does, and can name such a
  time. Its hour is then read as written (2). That is the same reading the
  circadian prior makes, so filter and features stay matched.
- **Fall back.** Local times from 01:00 to 01:59 occur twice. Both occurrences
  read the same clock time, so they get the same hour and the same encoding.
  They differ only in UTC offset, which time of day does not describe.

Every value is a function of the moment alone, so time features cannot use
future observations.
"""

from __future__ import annotations

import math
from datetime import datetime, tzinfo

import numpy as np

from ..observations import require_aware

HOURS_PER_DAY = 24

#: Largest number of harmonics the 24 hourly values support in sine-cosine pairs.
MAX_HARMONICS = 11


def local_hour(moment: datetime, zone: tzinfo) -> int:
    """The wall-clock hour of *moment* in *zone*, 0 to 23.

    A moment already expressed in *zone* is read as written, as the circadian
    prior reads it. Any other aware moment is converted to *zone* first. A
    naive moment is refused, since its zone would be a guess.
    """
    return require_aware(moment, "moment").astimezone(zone).hour


def _hours(values: np.ndarray | list[float]) -> np.ndarray:
    hours = np.asarray(values, dtype=float)
    if hours.ndim != 1:
        raise ValueError("hours must be one-dimensional")
    if not np.all(np.isfinite(hours)) or np.any(hours != np.round(hours)):
        raise ValueError("hours must be whole numbers")
    if np.any((hours < 0) | (hours >= HOURS_PER_DAY)):
        raise ValueError("hours must lie in 0-23")
    return hours


def _harmonics(harmonics: int) -> int:
    if isinstance(harmonics, bool) or not isinstance(harmonics, int):
        raise ValueError("harmonics must be an integer")
    if not 1 <= harmonics <= MAX_HARMONICS:
        raise ValueError(f"harmonics must lie in 1-{MAX_HARMONICS}")
    return harmonics


def hour_angle(hours: np.ndarray | list[float]) -> np.ndarray:
    """The clock angle of each hour's midpoint, ``2 pi (h + 1/2) / 24``."""
    return 2.0 * math.pi * (_hours(hours) + 0.5) / HOURS_PER_DAY


def cyclic_hour_columns(
    harmonics: int = 1, prefix: str = "hour_of_day"
) -> tuple[str, ...]:
    """Column names of :func:`cyclic_hour_features`, in the same order."""
    return tuple(
        f"{prefix}_{kind}{k}"
        for k in range(1, _harmonics(harmonics) + 1)
        for kind in ("sin", "cos")
    )


def cyclic_hour_features(
    hours: np.ndarray | list[float], harmonics: int = 1
) -> np.ndarray:
    """Encode hours on the 24-hour circle with *harmonics* sine-cosine pairs.

    Returns an array of shape ``(len(hours), 2 * harmonics)`` with columns
    ``sin(k theta), cos(k theta)`` for ``k = 1, ..., harmonics``.
    """
    theta = hour_angle(hours)
    k = np.arange(1, _harmonics(harmonics) + 1)
    angles = theta[:, None] * k[None, :]
    features = np.empty((theta.size, 2 * k.size))
    features[:, 0::2] = np.sin(angles)
    features[:, 1::2] = np.cos(angles)
    return features


def peak_hour(sin_coefficient: float, cos_coefficient: float) -> float:
    """The clock hour at which a first-harmonic score peaks, in ``[0, 24)``.

    For a score ``b sin(theta) + g cos(theta)`` over the encoding above, the
    maximum lies at ``theta = atan2(b, g)``, the midpoint of the hour returned
    here. A score with both coefficients zero has no peak.
    """
    if sin_coefficient == 0.0 and cos_coefficient == 0.0:
        raise ValueError("a score with both coefficients zero has no peak")
    phase = math.atan2(sin_coefficient, cos_coefficient) % (2.0 * math.pi)
    return (HOURS_PER_DAY * phase / (2.0 * math.pi) - 0.5) % HOURS_PER_DAY
