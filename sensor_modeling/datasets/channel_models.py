"""Fitted observation models for the generative filter's evidence channels.

The filter's declared model gives every sensor a Poisson rate chosen by
reasoning, and pools a channel's sensors into one Poisson count. The Phase 3.3
diagnostic found that those declared rates overstate the evidence of silence
more than dependence between channels does. It also found real counts far more
often zero than a Poisson with their observed mean allows. This module fits
the channel observation models from labelled training households, keeping the
channels conditionally independent given the state, so that only their
marginal distributions change.

Two fitted families are provided, alongside the declared one:

The hurdle model
----------------
For channel ``c``, state ``s`` and a window's pooled count ``n``::

    P(n = 0     | s) = π
    P(n = k ≥ 1 | s) = (1 − π) · μ^k e^{−μ} / (k! (1 − e^{−μ}))

with ``π = π_c(s)``, the probability that the channel is silent, and
``μ = μ_c(s)``, the rate of the zero-truncated Poisson that describes how much
activity there is once there is any. Its log-likelihood is::

    log P(n | s) = log π                                   if n = 0
                 = log(1 − π) − log(e^μ − 1) + n log μ − log n!   if n ≥ 1

The ``log n!`` term is shared by every state and dropped. Silence and activity
each have their own parameter, which is what the diagnostic's zero excess
needs. When ``π = e^{−μ}`` the hurdle model is exactly a Poisson with rate
``μ``.

The hurdle negative binomial
----------------------------
The same hurdle, with a zero-truncated negative binomial for the active count::

    P(n = k ≥ 1 | s) = (1 − π) · NB(k; μ, α) / (1 − NB(0; μ, α))

``NB(k; μ, α)`` has mean ``μ`` and variance ``μ + α μ²``. The posterior
predictive checks found the zero-truncated Poisson's active count
systematically under-dispersed, with a variance growing with the square of
the mean: the negative binomial's relationship. ``α = 0`` is the Poisson, so the
family contains the hurdle model as its limit. See
``docs/HURDLE_NEGATIVE_BINOMIAL.md``.

The fitted Poisson
------------------
``P(n = k | s) = λ^k e^{−λ} / k!`` with ``λ = λ_c(s)``, the channel's mean count
per window. This is the declared family with fitted values, and it is the
comparison that shows whether silence needs its own parameter.

Fitting
-------
Parameters are per channel and state, pooled over the labelled windows of the
training households only. Each is a posterior mean that shrinks toward the
declared model with ``κ`` pseudo-windows::

    π̂ = (Z + κ π₀) / (W + κ)
    m̂ = (S + κ m₀) / (P + κ),      μ̂ solves μ / (1 − e^{−μ}) = m̂
    λ̂ = (S + κ λ₀) / (W + κ)

``W`` is the number of windows, ``Z`` the silent ones, ``P = W − Z`` the active
ones and ``S`` the total count. ``λ₀`` is the declared expected count, ``π₀ =
e^{−λ₀}`` the declared silence probability and ``m₀ = λ₀ / (1 − e^{−λ₀})`` the
declared mean count in an active window, each averaged over the training
households' windows. With much data the prior does not matter. With none, the
fitted model is the declared one. A channel no training household has is
therefore given the held-out household's own declared values, which use its
sensor registry but none of its labels.

The negative binomial keeps the hurdle's ``π̂`` and active mean ``m̂``. Its
dispersion ``α̂`` maximises the profile likelihood of the training households'
active-count table plus ``κ`` pseudo-windows shaped as the zero-truncated
Poisson with mean ``m̂``, which shrinks it toward the Poisson limit, ``α = 0``,
with the weight of ``κ`` windows. ``μ̂`` then solves the zero-truncated negative
binomial's mean for ``m̂``.

Uninstrumented channels are absent, not silent: they have no model and no
column. Nothing here clips a probability.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import betaln, gammaln

from ..observations import Modality, SensorRegistry
from ..states.ontology import BehaviouralState, StateOntology
from .casas import CasasRecording
from .information_sets import (
    EvidenceChannel,
    EvidenceResolution,
    InformationComponent,
    InformationSet,
    build_feature_table,
    evidence_column,
)
from .matched_evaluation import _Household, _regular_moments
from .partial_pooling import PooledEstimate, PoolingConfig, pool
from .restricted_filter import ChannelLikelihood, ChannelModel, channel_likelihoods

HURDLE = "hurdle"
HURDLE_NB = "hurdle_nb"
POISSON = "poisson"
DECLARED = "declared"
FAMILIES = (DECLARED, HURDLE, HURDLE_NB, POISSON)

#: Smallest positive rate of the truncated Poisson: the limit of a channel whose
#: active windows all hold exactly one activation.
MIN_TRUNCATED_RATE = 1e-9


def _log_expm1(values: np.ndarray) -> np.ndarray:
    """``log(e^x − 1)``, accurate for small and large ``x``."""
    values = np.asarray(values, dtype=float)
    large = values > 30.0
    out = np.empty_like(values)
    out[large] = values[large] + np.log1p(-np.exp(-values[large]))
    out[~large] = np.log(np.expm1(values[~large]))
    return out


def truncated_poisson_rate(mean: float) -> float:
    """The rate ``μ`` whose zero-truncated Poisson has mean *mean*.

    It solves ``μ / (1 − e^{−μ}) = mean``, which has one positive root for any
    mean above 1. A mean of 1, every active window holding exactly one
    activation, is the limit ``μ → 0``, returned as ``MIN_TRUNCATED_RATE``.
    """
    if not math.isfinite(mean) or mean < 1.0:
        raise ValueError("a zero-truncated Poisson mean must be at least 1")
    if mean <= 1.0 + 1e-9:
        return MIN_TRUNCATED_RATE

    def excess(rate: float) -> float:
        return rate + mean * math.expm1(-rate)

    return float(brentq(excess, MIN_TRUNCATED_RATE, mean, xtol=1e-14, rtol=1e-12))


def _checked_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    if not np.all(np.isfinite(counts)) or (counts < 0).any():
        raise ValueError("counts must be finite and non-negative")
    if not np.array_equal(counts, np.round(counts)):
        raise ValueError("counts must be whole numbers")
    return counts


@dataclass(frozen=True)
class HurdleChannel:
    """A channel's hurdle observation model: silence, then how much activity.

    Attributes
    ----------
    channel
        The evidence channel.
    silence
        ``π_c(s)``, the probability of a silent window, per state.
    rate
        ``μ_c(s)``, the zero-truncated Poisson rate of an active window's
        count, per state.
    """

    channel: EvidenceChannel
    silence: np.ndarray
    rate: np.ndarray

    def __post_init__(self) -> None:
        """Validate the parameters and freeze copies of them."""
        silence = np.array(self.silence, dtype=float)
        rate = np.array(self.rate, dtype=float)
        if silence.shape != rate.shape or silence.ndim != 1:
            raise ValueError("silence and rate need one value per state")
        if not np.all((silence > 0.0) & (silence < 1.0)):
            raise ValueError("silence probabilities must lie strictly in (0, 1)")
        if not np.all(np.isfinite(rate) & (rate > 0.0)):
            raise ValueError("rates must be positive and finite")
        silence.setflags(write=False)
        rate.setflags(write=False)
        object.__setattr__(self, "silence", silence)
        object.__setattr__(self, "rate", rate)

    @property
    def zero_term(self) -> np.ndarray:
        """``log π``: the log-likelihood of a silent window, per state."""
        values: np.ndarray = np.log(self.silence)
        return values

    @property
    def active_term(self) -> np.ndarray:
        """``log(1 − π) − log(e^μ − 1)``: the log-likelihood of any activity."""
        values: np.ndarray = np.log1p(-self.silence) - _log_expm1(self.rate)
        return values

    @property
    def per_activation(self) -> np.ndarray:
        """``log μ``: the log-likelihood each activation adds, per state."""
        values: np.ndarray = np.log(self.rate)
        return values

    def loglik(self, counts: np.ndarray) -> np.ndarray:
        """``(rows, states)`` log-likelihood of each pooled count, up to ``log n!``."""
        counts = _checked_counts(counts)
        active = self.active_term[None, :] + counts[:, None] * self.per_activation
        values: np.ndarray = np.where(
            (counts == 0)[:, None], self.zero_term[None, :], active
        )
        return values

    def decompose(self, count: float) -> dict[str, np.ndarray]:
        """One window's log-likelihood split into silence and activity parts.

        ``silence`` is the part that says whether the channel fired at all:
        ``log π`` for a silent window, ``log(1 − π)`` otherwise. ``activity`` is
        how much it fired, given that it did: zero for a silent window.
        """
        counts = _checked_counts(np.array([count]))
        if counts[0] == 0:
            return {"silence": self.zero_term, "activity": np.zeros_like(self.rate)}
        return {
            "silence": np.log1p(-self.silence),
            "activity": counts[0] * self.per_activation - _log_expm1(self.rate),
        }


def poisson_channel(
    channel: EvidenceChannel, mean: np.ndarray, sensors: Sequence[str] = ()
) -> ChannelLikelihood:
    """A Poisson channel with *mean* expected activations per window, per state."""
    mean = np.asarray(mean, dtype=float)
    if mean.ndim != 1 or not np.all(np.isfinite(mean) & (mean > 0.0)):
        raise ValueError("Poisson means must be positive and finite")
    per_activation = np.log(mean)
    return ChannelLikelihood(
        channel,
        tuple(sorted(sensors)),
        per_activation - per_activation.mean(),
        -mean + mean.mean(),
        mean.copy(),
        1.0,
    )


# ----------------------------------------------------------------------------
# The over-dispersed active count: a zero-truncated negative binomial
# ----------------------------------------------------------------------------
#: Largest dispersion the fit may return: an active count whose variance
#: exceeds its mean by a hundred times the square of the mean. As ``α`` grows
#: with the mean held fixed, the zero-truncated negative binomial tends to the
#: logarithmic series, and a count table with many ones and a long tail can
#: have its likelihood rise toward that limit without a maximum. The bound
#: stops it there, where the likelihood is flat; a fit at the bound is flagged.
MAX_DISPERSION = 100.0

#: Smallest positive dispersion the fit searches. Below it the variance exceeds
#: a Poisson's by less than a ten-thousandth of the squared mean, and the fit
#: compares with the Poisson limit, dispersion zero, directly.
MIN_DISPERSION = 1e-4

#: Log-likelihood per unit of weight a positive dispersion must gain over the
#: Poisson limit to be preferred. Near ``α = 0`` the likelihood is flat to second
#: order, and a smaller gain is below the formula's rounding.
MIN_GAIN = 1e-6

#: Tail mass below which a tabulated count distribution is cut off.
_TAIL = 1e-12


def _log1mexp(values: np.ndarray) -> np.ndarray:
    """``log(1 − e^x)`` for ``x < 0``, accurate near 0 and far below it."""
    values = np.asarray(values, dtype=float)
    near = values > -math.log(2.0)
    out = np.empty_like(values)
    out[near] = np.log(-np.expm1(values[near]))
    out[~near] = np.log1p(-np.exp(values[~near]))
    return out


def nb_log_zero(rate: np.ndarray | float, dispersion: np.ndarray | float) -> np.ndarray:
    """``log P(X = 0)`` of the negative binomial with mean *rate* and *dispersion*.

    It is ``−log(1 + αμ) / α``, and ``−μ`` at ``α = 0``, the Poisson.
    """
    mu = np.asarray(rate, dtype=float)
    alpha = np.asarray(dispersion, dtype=float)
    poisson_limit = alpha <= 0.0
    safe = np.where(poisson_limit, 1.0, alpha)
    values: np.ndarray = np.where(poisson_limit, -mu, -np.log1p(safe * mu) / safe)
    return values


def ztnb_log_pmf(
    counts: np.ndarray | float,
    rate: np.ndarray | float,
    dispersion: np.ndarray | float,
) -> np.ndarray:
    """``log P(X = k | X ≥ 1)`` of the zero-truncated negative binomial, for ``k ≥ 1``.

    *counts* broadcast against *rate* and *dispersion*. With ``r = 1/α``::

        log P(k) = log Γ(k + r) − log Γ(r) − log k! + r log(r / (r + μ))
                   + k log(μ / (r + μ)) − log(1 − P(0))

    ``log Γ(k + r) − log Γ(r) − log k!`` is evaluated as ``−log k − log B(k, r)``,
    which stays accurate as ``r`` grows. At ``α = 0`` it is the zero-truncated
    Poisson.
    """
    k = np.asarray(counts, dtype=float)
    mu = np.asarray(rate, dtype=float)
    alpha = np.asarray(dispersion, dtype=float)
    k, mu, alpha = np.broadcast_arrays(k, mu, alpha)
    poisson_limit = alpha <= 0.0
    safe = np.where(poisson_limit, 1.0, alpha)
    size = 1.0 / safe
    log_zero = nb_log_zero(mu, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        negative_binomial = (
            -np.log(k)
            - betaln(k, size)
            + k * (np.log(safe * mu) - np.log1p(safe * mu))
            + log_zero
        )
        poisson = k * np.log(mu) - mu - gammaln(k + 1.0)
    values: np.ndarray = np.where(poisson_limit, poisson, negative_binomial) - (
        _log1mexp(log_zero)
    )
    return values


def ztnb_moments(
    rate: np.ndarray | float, dispersion: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of the zero-truncated negative binomial.

    The untruncated count has mean ``μ`` and second moment
    ``μ + (1 + α) μ²``; truncation divides both by ``1 − P(0)``.
    """
    mu = np.asarray(rate, dtype=float)
    alpha = np.asarray(dispersion, dtype=float)
    active = -np.expm1(nb_log_zero(mu, alpha))
    mean = mu / active
    second = (mu + (1.0 + alpha) * mu * mu) / active
    return mean, second - mean * mean


def ztnb_rate(mean: float, dispersion: float) -> float:
    """The mean ``μ`` whose zero-truncated negative binomial has mean *mean*.

    It solves ``μ / (1 − P(0)) = mean``, which has one root for any mean above 1,
    since the truncated mean rises from 1 as ``μ`` does. A mean of 1, every
    active window holding one activation, is the limit ``μ → 0``, returned as
    ``MIN_TRUNCATED_RATE``. At dispersion zero it is
    :func:`truncated_poisson_rate`.
    """
    if not math.isfinite(dispersion) or dispersion < 0.0:
        raise ValueError("dispersion must be finite and non-negative")
    if dispersion == 0.0:
        return truncated_poisson_rate(mean)
    if not math.isfinite(mean) or mean < 1.0:
        raise ValueError("a zero-truncated mean must be at least 1")
    if mean <= 1.0 + 1e-9:
        return MIN_TRUNCATED_RATE

    def excess(rate: float) -> float:
        return rate + mean * math.expm1(-math.log1p(dispersion * rate) / dispersion)

    return float(brentq(excess, MIN_TRUNCATED_RATE, mean, xtol=1e-14, rtol=1e-12))


@dataclass(frozen=True)
class HurdleNBChannel:
    """A channel's hurdle model with a negative-binomial active count.

    For state ``s``, ``π = π_c(s)``, ``μ = μ_c(s)`` and ``α = α_c(s)``::

        P(n = 0     | s) = π
        P(n = k ≥ 1 | s) = (1 − π) · NB(k; μ, α) / (1 − NB(0; μ, α))

    ``NB(k; μ, α)`` is the negative binomial with mean ``μ`` and variance
    ``μ + α μ²``. ``α = 0`` is the Poisson, so the model is then exactly the
    :class:`HurdleChannel` with the same ``π`` and ``μ``.

    Attributes
    ----------
    channel
        The evidence channel.
    silence
        ``π``, the probability of a silent window, per state.
    rate
        ``μ``, the mean of the untruncated negative binomial, per state.
    dispersion
        ``α ≥ 0``, per state: how much faster than its mean the active count's
        variance grows.
    """

    channel: EvidenceChannel
    silence: np.ndarray
    rate: np.ndarray
    dispersion: np.ndarray

    def __post_init__(self) -> None:
        """Validate the parameters and freeze copies of them."""
        silence = np.array(self.silence, dtype=float)
        rate = np.array(self.rate, dtype=float)
        dispersion = np.array(self.dispersion, dtype=float)
        if not (silence.shape == rate.shape == dispersion.shape) or silence.ndim != 1:
            raise ValueError("silence, rate and dispersion need one value per state")
        if not np.all((silence > 0.0) & (silence < 1.0)):
            raise ValueError("silence probabilities must lie strictly in (0, 1)")
        if not np.all(np.isfinite(rate) & (rate > 0.0)):
            raise ValueError("rates must be positive and finite")
        if not np.all(np.isfinite(dispersion) & (dispersion >= 0.0)):
            raise ValueError("dispersions must be finite and non-negative")
        for values in (silence, rate, dispersion):
            values.setflags(write=False)
        object.__setattr__(self, "silence", silence)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "dispersion", dispersion)

    @property
    def zero_term(self) -> np.ndarray:
        """``log π``: the log-likelihood of a silent window, per state."""
        values: np.ndarray = np.log(self.silence)
        return values

    def active_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance of an active window's count, per state."""
        return ztnb_moments(self.rate, self.dispersion)

    def log_pmf(self, counts: np.ndarray) -> np.ndarray:
        """``(rows, states)`` log-probability of each count, normalised over counts."""
        counts = _checked_counts(counts)
        active = np.log1p(-self.silence)[None, :] + ztnb_log_pmf(
            np.maximum(counts, 1.0)[:, None],
            self.rate[None, :],
            self.dispersion[None, :],
        )
        values: np.ndarray = np.where(
            (counts == 0)[:, None], self.zero_term[None, :], active
        )
        return values

    def loglik(self, counts: np.ndarray) -> np.ndarray:
        """``(rows, states)`` log-likelihood of each pooled count, up to ``log n!``.

        The ``log n!`` shared by every state is dropped, as :class:`HurdleChannel`
        drops it, so the two families' terms are directly comparable.
        """
        counts = _checked_counts(counts)
        values: np.ndarray = self.log_pmf(counts) + gammaln(counts + 1.0)[:, None]
        return values

    def decompose(self, count: float) -> dict[str, np.ndarray]:
        """One window's log-likelihood split into silence and activity parts."""
        counts = _checked_counts(np.array([count]))
        if counts[0] == 0:
            return {"silence": self.zero_term, "activity": np.zeros_like(self.rate)}
        return {
            "silence": np.log1p(-self.silence),
            "activity": ztnb_log_pmf(counts[0], self.rate, self.dispersion)
            + gammaln(counts[0] + 1.0),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form."""
        return {
            "family": HURDLE_NB,
            **_channel_key(self.channel),
            "silence": self.silence.tolist(),
            "rate": self.rate.tolist(),
            "dispersion": self.dispersion.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HurdleNBChannel:
        """Rebuild a model written by :meth:`to_dict`."""
        if payload.get("family") != HURDLE_NB:
            raise ValueError(f"not a {HURDLE_NB} channel model")
        return cls(
            EvidenceChannel(payload["room"], Modality(payload["modality"])),
            np.array(payload["silence"], dtype=float),
            np.array(payload["rate"], dtype=float),
            np.array(payload["dispersion"], dtype=float),
        )


@dataclass(frozen=True)
class ActiveCounts:
    """How often each active count occurred, per state, on one channel.

    The negative binomial's dispersion is not determined by a count's sum and
    sum of squares, so its fit needs the whole distribution of active counts.

    Attributes
    ----------
    values
        The distinct counts of at least one, increasing.
    frequencies
        ``(states, values)``: how many active windows of each state held each
        count.
    """

    values: np.ndarray
    frequencies: np.ndarray

    def __post_init__(self) -> None:
        """Validate and freeze copies."""
        values = np.array(self.values, dtype=float)
        frequencies = np.array(self.frequencies, dtype=float)
        if values.ndim != 1 or frequencies.ndim != 2:
            raise ValueError("values are one-dimensional, frequencies two")
        if frequencies.shape[1] != values.size:
            raise ValueError("one frequency column per value")
        if values.size and (
            values.min() < 1.0
            or np.any(np.diff(values) <= 0.0)
            or not np.array_equal(values, np.round(values))
        ):
            raise ValueError("values must be increasing whole counts of at least 1")
        if not np.all(np.isfinite(frequencies)) or (frequencies < 0.0).any():
            raise ValueError("frequencies must be finite and non-negative")
        values.setflags(write=False)
        frequencies.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "frequencies", frequencies)

    @classmethod
    def from_counts(
        cls, counts: np.ndarray, labels: np.ndarray, states: int
    ) -> ActiveCounts:
        """Tabulate the active windows among *counts*, by their state *labels*."""
        counts = _checked_counts(counts)
        active = counts > 0
        values, index = np.unique(counts[active], return_inverse=True)
        frequencies = np.zeros((states, values.size))
        np.add.at(frequencies, (np.asarray(labels)[active], index), 1.0)
        return cls(values, frequencies)

    @classmethod
    def combine(cls, parts: Sequence[ActiveCounts]) -> ActiveCounts:
        """One table holding every part's active windows."""
        if not parts:
            raise ValueError("nothing to combine")
        values = np.unique(np.concatenate([p.values for p in parts]))
        states = {p.frequencies.shape[0] for p in parts}
        if len(states) != 1:
            raise ValueError("every part needs the same states")
        frequencies = np.zeros((states.pop(), values.size))
        for part in parts:
            frequencies[:, np.searchsorted(values, part.values)] += part.frequencies
        return cls(values, frequencies)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form."""
        return {
            "values": [int(v) for v in self.values],
            "frequencies": self.frequencies.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActiveCounts:
        """Rebuild a table written by :meth:`to_dict`."""
        return cls(
            np.array(payload["values"], dtype=float),
            np.array(payload["frequencies"], dtype=float),
        )


def _tail_count(rate: float) -> int:
    """A count beyond which the zero-truncated Poisson with *rate* holds under ``_TAIL``."""
    from scipy.stats import poisson

    return max(1, int(poisson.isf(_TAIL, rate)) + 1)


def _ztp_table(rate: float, top: int) -> np.ndarray:
    """The zero-truncated Poisson's probabilities of the counts 1 to *top*."""
    from scipy.stats import poisson

    k = np.arange(1, top + 1)
    probabilities: np.ndarray = poisson.pmf(k, rate) / -math.expm1(-rate)
    return probabilities


def fit_dispersion(
    values: np.ndarray,
    weights: np.ndarray,
    active_mean: float,
    *,
    max_dispersion: float = MAX_DISPERSION,
) -> float:
    """The dispersion ``α`` that maximises the profile likelihood of an active-count table.

    For each ``α``, ``μ`` is the one whose truncated mean is *active_mean*, which
    is the maximum-likelihood ``μ`` for that ``α`` when *active_mean* is the
    table's own mean: with ``α`` fixed, the truncated negative binomial is an
    exponential family in ``k``. The log-likelihood ``Σ w_k log P(k)`` is then
    maximised over ``α`` in ``{0} ∪ [MIN_DISPERSION, max_dispersion]``: a grid in
    ``log α``, refined by bounded Brent search around the best point. A positive
    ``α`` is preferred to the Poisson limit only if it gains more than
    ``MIN_GAIN`` per unit of weight, so a table no more dispersed than a Poisson
    returns exactly 0.

    Parameters
    ----------
    values, weights
        Distinct active counts and how much each weighs, such as its
        frequency.
    active_mean
        The mean the fitted distribution must keep.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape or values.ndim != 1:
        raise ValueError("values and weights must be matching one-dimensional arrays")
    if not math.isfinite(max_dispersion) or max_dispersion <= MIN_DISPERSION:
        raise ValueError("max_dispersion must exceed MIN_DISPERSION")
    if weights.sum() <= 0.0 or active_mean <= 1.0 + 1e-9:
        return 0.0

    def loss(log_alpha: float | None) -> float:
        alpha = 0.0 if log_alpha is None else math.exp(log_alpha)
        rate = ztnb_rate(active_mean, alpha)
        return -float(np.dot(weights, ztnb_log_pmf(values, rate, alpha)))

    best = loss(None)
    grid = np.linspace(math.log(MIN_DISPERSION), math.log(max_dispersion), 41)
    losses = [loss(float(u)) for u in grid]
    at = int(np.argmin(losses))
    low = grid[max(at - 1, 0)]
    high = grid[min(at + 1, grid.size - 1)]
    refined = minimize_scalar(
        loss,
        bounds=(float(low), float(high)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    candidates = [
        (float(grid[at]), losses[at]),
        (float(refined.x), float(refined.fun)),
    ]
    log_alpha, value = min(candidates, key=lambda c: c[1])
    if best - value <= MIN_GAIN * float(weights.sum()):
        return 0.0
    if log_alpha >= math.log(max_dispersion) - 1e-9:
        return max_dispersion  # exactly, so a fit at the bound is flagged
    return min(max(math.exp(log_alpha), MIN_DISPERSION), max_dispersion)


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelStatistics:
    """Sufficient statistics of one channel's labelled windows, per state.

    ``active_counts`` tabulates the active windows, which the negative
    binomial's dispersion needs. It is optional, so statistics built from sums
    alone still fit the hurdle and Poisson families, and it is never part of a
    fit's serialised form, so adding it changes no published digest.
    """

    windows: np.ndarray
    silent: np.ndarray
    total: np.ndarray
    squares: np.ndarray
    declared: np.ndarray
    active_counts: ActiveCounts | None = None

    @property
    def active(self) -> np.ndarray:
        """Windows with at least one activation."""
        values: np.ndarray = self.windows - self.silent
        return values


def _statistics(
    counts: np.ndarray, labels: np.ndarray, states: int, declared: np.ndarray
) -> ChannelStatistics:
    windows = np.bincount(labels, minlength=states).astype(float)
    silent = np.bincount(labels, weights=(counts == 0).astype(float), minlength=states)
    total = np.bincount(labels, weights=counts, minlength=states)
    squares = np.bincount(labels, weights=counts * counts, minlength=states)
    return ChannelStatistics(
        windows,
        silent,
        total,
        squares,
        declared * windows,
        ActiveCounts.from_counts(counts, labels, states),
    )


def household_channel_counts(
    recording: CasasRecording,
    resolution: EvidenceResolution,
    ontology: StateOntology,
    *,
    household: str,
) -> tuple[dict[EvidenceChannel, np.ndarray], np.ndarray, np.ndarray, list[Any]]:
    """Each instrumented channel's current-window counts, and the labels.

    Returns the counts at every moment, the labelled rows, their state
    indices, and the moments.
    """
    info = InformationSet(
        "current", frozenset({InformationComponent.CURRENT_EVIDENCE}), resolution
    )
    moments = _regular_moments(recording, resolution.step)
    table = build_feature_table(recording, info, moments, household=household)
    built = _Household.build(household, "fit", table, recording)
    states = tuple(ontology.states)
    labels = np.array([states.index(label) for label in built.labels], dtype=int)
    instrumented = sorted(set(resolution.channels) - set(table.uninstrumented))
    counts = {
        channel: table.column(evidence_column(channel.name, 0))
        for channel in instrumented
    }
    return counts, built.labelled, labels, list(table.moments)


@dataclass(frozen=True)
class FittedChannels:
    """The fitted hurdle and Poisson parameters of every channel, for one fold.

    Attributes
    ----------
    fitted_on
        The training households, sorted.
    states
        State order of every per-state array.
    pseudo_windows
        ``κ``, the prior's weight in windows.
    statistics
        Sufficient statistics per channel, pooled over the training households.
    """

    fitted_on: tuple[str, ...]
    states: tuple[BehaviouralState, ...]
    pseudo_windows: float
    statistics: Mapping[EvidenceChannel, ChannelStatistics]

    def parameters(
        self, channel: EvidenceChannel, declared: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """``silence``, ``rate``, ``mean`` and ``active_mean`` per state for *channel*.

        ``active_mean`` is the mean count of an active window that ``rate``
        is solved from.

        *declared* is the held-out household's declared expected count, used as
        the prior where the training households have no window of a state on
        this channel. Without it, such a state's parameters are NaN: they
        depend on the household they are used for.
        """
        kappa = self.pseudo_windows
        size = len(self.states)
        stats = self.statistics.get(channel)
        windows = stats.windows if stats is not None else np.zeros(size)
        seen = windows > 0
        prior_mean = np.full(size, np.nan)
        if stats is not None:
            prior_mean[seen] = stats.declared[seen] / windows[seen]
        if declared is not None:
            prior_mean[~seen] = np.asarray(declared, dtype=float)[~seen]
        prior_silence = np.exp(-prior_mean)
        prior_active_mean = prior_mean / -np.expm1(-prior_mean)
        if stats is None:
            silent = total = active = np.zeros(size)
        else:
            silent, total, active = stats.silent, stats.total, stats.active
        silence = (silent + kappa * prior_silence) / (windows + kappa)
        active_mean = (total + kappa * prior_active_mean) / (active + kappa)
        rate = np.array(
            [
                truncated_poisson_rate(float(m)) if math.isfinite(m) else math.nan
                for m in active_mean
            ]
        )
        mean = (total + kappa * prior_mean) / (windows + kappa)
        return {
            "silence": silence,
            "rate": rate,
            "mean": mean,
            "active_mean": active_mean,
        }

    def dispersion(
        self,
        channel: EvidenceChannel,
        declared: np.ndarray | None = None,
        *,
        max_dispersion: float = MAX_DISPERSION,
    ) -> np.ndarray:
        """The negative binomial's dispersion ``α`` per state for *channel*.

        Each state's ``α`` maximises the profile likelihood of the training
        households' active-count table plus ``κ`` pseudo-windows distributed as
        the zero-truncated Poisson with the same active mean, keeping the
        hurdle's active mean (:func:`fit_dispersion`). The pseudo-windows carry
        no information about the mean, only the Poisson's shape, so they
        penalise ``α`` toward 0 with the weight of ``κ`` windows: with little
        data the fit is the hurdle model, and with none it is the declared one.

        *declared* is as for :meth:`parameters`. A state whose parameters are
        undefined without it gets NaN.

        Raises
        ------
        ValueError
            If the statistics carry no active-count table.
        """
        fitted = self.parameters(channel, declared)
        stats = self.statistics.get(channel)
        if stats is not None and stats.active_counts is None:
            raise ValueError(
                "the negative binomial's dispersion needs active-count tables; "
                "build the statistics with home_statistics or combine_statistics"
            )
        size = len(self.states)
        out = np.full(size, np.nan)
        for i in range(size):
            mean = float(fitted["active_mean"][i])
            if not math.isfinite(mean):
                continue
            shape_rate = truncated_poisson_rate(mean)
            top = _tail_count(shape_rate)
            observed_values = (
                stats.active_counts.values
                if stats is not None and stats.active_counts is not None
                else np.zeros(0)
            )
            observed_weights = (
                stats.active_counts.frequencies[i]
                if stats is not None and stats.active_counts is not None
                else np.zeros(0)
            )
            values = np.union1d(observed_values, np.arange(1.0, top + 1.0))
            weights = np.zeros(values.size)
            weights[np.searchsorted(values, observed_values)] += observed_weights
            weights[: int(top)] += self.pseudo_windows * _ztp_table(
                shape_rate, int(top)
            )
            out[i] = fit_dispersion(
                values, weights, mean, max_dispersion=max_dispersion
            )
        return out

    def dispersion_to_dict(self) -> dict[str, Any]:
        """The negative binomial's fitted parameters, serialisable.

        Per channel and state: the active windows the dispersion was fitted on,
        ``α``, whether it reached ``MAX_DISPERSION``, the size ``1/α`` (``None``
        at the Poisson limit), ``μ`` and the active mean. Kept apart from
        :meth:`to_dict`, whose digest identifies published fits.
        """
        channels: dict[str, Any] = {}
        for channel in sorted(self.statistics):
            stats = self.statistics[channel]
            fitted = self.parameters(channel)
            alpha = self.dispersion(channel)
            channels[channel.name] = {}
            for i, state in enumerate(self.states):
                a, m = float(alpha[i]), float(fitted["active_mean"][i])
                defined = math.isfinite(a) and math.isfinite(m)
                channels[channel.name][state.value] = {
                    "active_windows": int(stats.active[i]),
                    "dispersion": a if defined else None,
                    "at_bound": (a >= MAX_DISPERSION) if defined else None,
                    "size": (1.0 / a if a > 0.0 else None) if defined else None,
                    "rate": ztnb_rate(m, a) if defined else None,
                    "active_mean": m if defined else None,
                }
        return {
            "model": "hurdle with a zero-truncated negative binomial active count",
            "fitted_on": list(self.fitted_on),
            "pseudo_windows": self.pseudo_windows,
            "max_dispersion": MAX_DISPERSION,
            "channels": channels,
        }

    def models(
        self,
        registry: SensorRegistry,
        resolution: EvidenceResolution,
        family: str,
        ontology: StateOntology | None = None,
    ) -> dict[EvidenceChannel, ChannelModel]:
        """One household's channel models of *family*, for its instrumented channels."""
        ontology = ontology or StateOntology()
        declared, _ = channel_likelihoods(registry, resolution, ontology)
        if family == DECLARED:
            return dict(declared)
        out: dict[EvidenceChannel, ChannelModel] = {}
        for channel, terms in declared.items():
            fitted = self.parameters(channel, terms.expected)
            if family == HURDLE:
                out[channel] = HurdleChannel(channel, fitted["silence"], fitted["rate"])
            elif family == HURDLE_NB:
                alpha = self.dispersion(channel, terms.expected)
                out[channel] = HurdleNBChannel(
                    channel,
                    fitted["silence"],
                    np.array(
                        [
                            ztnb_rate(float(m), float(a))
                            for m, a in zip(fitted["active_mean"], alpha, strict=True)
                        ]
                    ),
                    alpha,
                )
            elif family == POISSON:
                out[channel] = poisson_channel(channel, fitted["mean"], terms.sensors)
            else:
                raise ValueError(f"family must be one of {FAMILIES}")
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form: statistics and fitted parameters."""
        channels: dict[str, Any] = {}
        for channel in sorted(self.statistics):
            stats = self.statistics[channel]
            fitted = self.parameters(channel)
            active = stats.active
            positive_variance = np.where(
                active > 1,
                (stats.squares - stats.total**2 / np.maximum(active, 1))
                / np.maximum(active - 1, 1),
                np.nan,
            )
            channels[channel.name] = {
                state.value: {
                    "windows": int(stats.windows[i]),
                    "silent": int(stats.silent[i]),
                    "total": int(stats.total[i]),
                    "declared_mean": (
                        float(stats.declared[i] / stats.windows[i])
                        if stats.windows[i] > 0
                        else None
                    ),
                    **{
                        key: (
                            float(fitted[key][i])
                            if math.isfinite(fitted[key][i])
                            else None
                        )
                        for key in ("silence", "rate", "mean")
                    },
                    "active_mean": (
                        float(stats.total[i] / active[i]) if active[i] > 0 else None
                    ),
                    "active_variance": (
                        float(positive_variance[i])
                        if math.isfinite(positive_variance[i])
                        else None
                    ),
                }
                for i, state in enumerate(self.states)
            }
        return {
            "model": "hurdle and Poisson channel observation models, fitted",
            "fitted_on": list(self.fitted_on),
            "pseudo_windows": self.pseudo_windows,
            "channels": channels,
        }

    def sha256(self) -> str:
        """SHA-256 of :meth:`to_dict`."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def fit_channel_models(
    recordings: Mapping[str, CasasRecording],
    *,
    resolution: EvidenceResolution,
    pseudo_windows: float,
    ontology: StateOntology | None = None,
) -> FittedChannels:
    """Fit every channel's hurdle and Poisson parameters from *recordings*.

    Pass the training households only: every labelled window of every
    recording given contributes.
    """
    if not recordings:
        raise ValueError("at least one training household is required")
    if not math.isfinite(pseudo_windows) or pseudo_windows <= 0.0:
        raise ValueError("pseudo_windows must be positive")
    if not isinstance(resolution.step, timedelta):
        raise ValueError("the resolution needs a step")
    ontology = ontology or StateOntology()
    return combine_statistics(
        {
            home: home_statistics(
                recordings[home], resolution, ontology, household=home
            )
            for home in sorted(recordings)
        },
        states=tuple(ontology.states),
        pseudo_windows=pseudo_windows,
    )


def home_statistics(
    recording: CasasRecording,
    resolution: EvidenceResolution,
    ontology: StateOntology,
    *,
    household: str,
) -> dict[EvidenceChannel, ChannelStatistics]:
    """One household's channel statistics over all its labelled windows."""
    counts, rows, labels, _ = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    declared, _ = channel_likelihoods(recording.registry, resolution, ontology)
    return {
        channel: _statistics(
            column[rows], labels, len(ontology.states), declared[channel].expected
        )
        for channel, column in counts.items()
    }


def combine_statistics(
    parts: Mapping[str, Mapping[EvidenceChannel, ChannelStatistics]],
    *,
    states: tuple[BehaviouralState, ...],
    pseudo_windows: float,
) -> FittedChannels:
    """The population fitted on the households in *parts*, from their statistics.

    Households are combined in sorted order, as :func:`fit_channel_models`
    combines them, so both give the same parameters to the last bit.
    """
    if not parts:
        raise ValueError("at least one training household is required")
    if not math.isfinite(pseudo_windows) or pseudo_windows <= 0.0:
        raise ValueError("pseudo_windows must be positive")
    pooled: dict[EvidenceChannel, list[ChannelStatistics]] = {}
    for home in sorted(parts):
        for channel, stats in parts[home].items():
            pooled.setdefault(channel, []).append(stats)
    statistics = {
        channel: ChannelStatistics(
            *(
                np.sum([getattr(s, name) for s in pieces], axis=0)
                for name in ("windows", "silent", "total", "squares", "declared")
            ),
            active_counts=(
                ActiveCounts.combine([s.active_counts for s in pieces])  # type: ignore[misc]
                if all(s.active_counts is not None for s in pieces)
                else None
            ),
        )
        for channel, pieces in pooled.items()
    }
    return FittedChannels(
        tuple(sorted(parts)), states, float(pseudo_windows), statistics
    )


def total_loglik(
    models: Mapping[EvidenceChannel, ChannelModel],
    counts: Mapping[EvidenceChannel, np.ndarray],
) -> np.ndarray:
    """``(windows, states)`` sum over channels of each channel's log-likelihood.

    Every channel in *models* must have an observed count in every window.
    """
    channels = sorted(models)
    if set(counts) != set(channels):
        raise ValueError("counts must cover exactly the modelled channels")
    summed: np.ndarray = np.sum([models[c].loglik(counts[c]) for c in channels], axis=0)
    return summed


def filter_recursion(
    loglik: np.ndarray, transition: np.ndarray, prior: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The filter's recursion over consecutive windows.

    Starting from *prior* one step before the first window, each window is a
    transition and then the window's log-likelihood. Returns the prediction
    before each window and the posterior after it, ``(windows, states)`` each.
    """
    predicted = np.empty_like(loglik, dtype=float)
    posterior = np.empty_like(predicted)
    belief = np.asarray(prior, dtype=float)
    for row in range(loglik.shape[0]):
        belief = belief @ transition
        predicted[row] = belief
        weights = belief * np.exp(loglik[row] - loglik[row].max())
        belief = weights / weights.sum()
        posterior[row] = belief
    return predicted, posterior


# ----------------------------------------------------------------------------
# Household adaptation, by partial pooling
# ----------------------------------------------------------------------------
def household_statistics(
    recording: CasasRecording,
    resolution: EvidenceResolution,
    ontology: StateOntology,
    *,
    household: str,
    until: datetime,
) -> dict[EvidenceChannel, ChannelStatistics]:
    """A household's channel statistics from its labelled windows up to *until*.

    Only windows that close at or before *until* count, so a household adapted
    this way can be scored on its later windows without reading them twice.
    """
    counts, rows, labels, moments = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    keep = np.array([moments[int(row)] <= until for row in rows], dtype=bool)
    declared, _ = channel_likelihoods(recording.registry, resolution, ontology)
    return {
        channel: _statistics(
            column[rows[keep]],
            labels[keep],
            len(ontology.states),
            declared[channel].expected,
        )
        for channel, column in counts.items()
    }


def _channel_key(channel: EvidenceChannel) -> dict[str, str]:
    return {"room": channel.room, "modality": channel.modality.value}


@dataclass(frozen=True)
class HouseholdChannels:
    """One household's hurdle parameters, pooled toward a fitted population.

    For each instrumented channel and state, the silence probability ``π`` is
    pooled from the household's silent windows among its windows, and the mean
    count of an active window from its total count among its active windows.
    The activity rate ``μ`` is then solved from the pooled mean, as the
    population's is.

    Attributes
    ----------
    household
        Whose parameters these are.
    states
        State order of every estimate.
    config
        The pooling strength.
    population_sha256, population_fitted_on
        Which fitted population the household was pooled toward, and which
        households that population was fitted on.
    until
        ISO time of the last window the household's own statistics read, or
        ``None`` if it contributed none: the population alone.
    silence, activity
        The pooled estimates per channel.
    """

    household: str
    states: tuple[BehaviouralState, ...]
    config: PoolingConfig
    population_sha256: str
    population_fitted_on: tuple[str, ...]
    until: str | None
    silence: Mapping[EvidenceChannel, PooledEstimate]
    activity: Mapping[EvidenceChannel, PooledEstimate]

    def __post_init__(self) -> None:
        """Check the estimates cover the same channels and states."""
        if set(self.silence) != set(self.activity):
            raise ValueError("silence and activity must cover the same channels")
        for channel in self.silence:
            for estimate in (self.silence[channel], self.activity[channel]):
                if estimate.population.size != len(self.states):
                    raise ValueError("every estimate needs one value per state")
                if estimate.strength != self.config.strength:
                    raise ValueError("every estimate must use the declared strength")

    def models(
        self, dispersion: Mapping[EvidenceChannel, np.ndarray] | None = None
    ) -> dict[EvidenceChannel, ChannelModel]:
        """The household's channel models, for the restricted recursion.

        Without *dispersion*, the hurdle models. With the population's
        dispersion per channel, from :meth:`FittedChannels.dispersion`, the
        hurdle negative binomial: its silence and active mean are the pooled
        ones, and its dispersion is the population's. A household's own
        dispersion is not pooled, because a few active windows cannot estimate
        it stably.
        """
        out: dict[EvidenceChannel, ChannelModel] = {}
        for channel in sorted(self.silence):
            means = self.activity[channel].pooled
            if dispersion is None:
                out[channel] = HurdleChannel(
                    channel,
                    self.silence[channel].pooled,
                    np.array([truncated_poisson_rate(float(m)) for m in means]),
                )
                continue
            alpha = np.asarray(dispersion[channel], dtype=float)
            out[channel] = HurdleNBChannel(
                channel,
                self.silence[channel].pooled,
                np.array(
                    [
                        ztnb_rate(float(m), float(a))
                        for m, a in zip(means, alpha, strict=True)
                    ]
                ),
                alpha,
            )
        return out

    def diagnostics(self) -> list[dict[str, Any]]:
        """Raw, pooled and population estimates, and the effective shrinkage.

        One row per channel, state and parameter. ``raw`` is the household's
        unconstrained estimate, ``None`` without data; ``shrinkage`` is the
        share of ``pooled`` that comes from ``population``.
        """
        rows: list[dict[str, Any]] = []
        for channel in sorted(self.silence):
            for parameter, estimate in (
                ("silence", self.silence[channel]),
                ("active_mean", self.activity[channel]),
            ):
                for i, state in enumerate(self.states):
                    raw = float(estimate.raw[i])
                    rows.append(
                        {
                            "channel": channel.name,
                            "state": state.value,
                            "parameter": parameter,
                            "observations": int(estimate.count[i]),
                            "raw": raw if math.isfinite(raw) else None,
                            "pooled": float(estimate.pooled[i]),
                            "population": float(estimate.population[i]),
                            "shrinkage": float(estimate.shrinkage[i]),
                        }
                    )
        return rows

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form, with its provenance."""
        return {
            "model": "hurdle channel parameters pooled toward a fitted population",
            "household": self.household,
            "states": [state.value for state in self.states],
            "pooling": self.config.to_dict(),
            "population": {
                "sha256": self.population_sha256,
                "fitted_on": list(self.population_fitted_on),
            },
            "until": self.until,
            "channels": [
                {
                    **_channel_key(channel),
                    "silence": self.silence[channel].to_dict(),
                    "active_mean": self.activity[channel].to_dict(),
                }
                for channel in sorted(self.silence)
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HouseholdChannels:
        """Rebuild parameters written by :meth:`to_dict`."""
        silence: dict[EvidenceChannel, PooledEstimate] = {}
        activity: dict[EvidenceChannel, PooledEstimate] = {}
        for entry in payload["channels"]:
            channel = EvidenceChannel(entry["room"], Modality(entry["modality"]))
            silence[channel] = PooledEstimate.from_dict(entry["silence"])
            activity[channel] = PooledEstimate.from_dict(entry["active_mean"])
        return cls(
            household=str(payload["household"]),
            states=tuple(BehaviouralState(s) for s in payload["states"]),
            config=PoolingConfig.from_dict(payload["pooling"]),
            population_sha256=str(payload["population"]["sha256"]),
            population_fitted_on=tuple(payload["population"]["fitted_on"]),
            until=payload["until"],
            silence=silence,
            activity=activity,
        )

    def sha256(self) -> str:
        """SHA-256 of :meth:`to_dict`."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def adapt_channels(
    population: FittedChannels,
    recording: CasasRecording,
    *,
    household: str,
    resolution: EvidenceResolution,
    config: PoolingConfig,
    until: datetime | None,
    ontology: StateOntology | None = None,
) -> HouseholdChannels:
    """Pool *household*'s hurdle parameters toward *population*.

    Parameters
    ----------
    population
        Channel models fitted on other households. A household it was fitted
        on is refused: its data would count twice, and a held-out score would
        no longer be held out.
    recording
        The household's recording. Its registry sets its channels and, for a
        state the population never saw, the declared prior.
    until
        The adaptation reads the household's labelled windows that close at
        or before this moment, and nothing later. ``None`` reads none: an
        unseen household gets the population alone.
    """
    ontology = ontology or StateOntology()
    own = (
        household_statistics(
            recording, resolution, ontology, household=household, until=until
        )
        if until is not None
        else {}
    )
    return pool_channels(
        population,
        recording.registry,
        own,
        household=household,
        resolution=resolution,
        config=config,
        until=until,
        ontology=ontology,
    )


def pool_channels(
    population: FittedChannels,
    registry: SensorRegistry,
    own: Mapping[EvidenceChannel, ChannelStatistics],
    *,
    household: str,
    resolution: EvidenceResolution,
    config: PoolingConfig,
    until: datetime | None,
    ontology: StateOntology | None = None,
) -> HouseholdChannels:
    """Pool a household's own statistics toward *population*.

    :func:`adapt_channels` computes *own* from the recording; this takes it
    precomputed, so one household's statistics can be pooled at several
    strengths without being counted again. *own* must hold only the
    household's labelled windows that close at or before *until*, as
    :func:`household_statistics` returns them, and be empty when *until* is
    ``None``.
    """
    if household in population.fitted_on:
        raise ValueError(
            f"household {household!r} was used to fit the population; adapting "
            "it would count its data twice"
        )
    if until is None and own:
        raise ValueError(
            "statistics without a cut-off cannot be attributed to a window"
        )
    ontology = ontology or StateOntology()
    states = tuple(ontology.states)
    if states != population.states:
        raise ValueError("the ontology's states differ from the population's")
    declared, _ = channel_likelihoods(registry, resolution, ontology)
    zeros = np.zeros(len(states))
    silence: dict[EvidenceChannel, PooledEstimate] = {}
    activity: dict[EvidenceChannel, PooledEstimate] = {}
    for channel, terms in declared.items():
        fitted = population.parameters(channel, terms.expected)
        stats = own.get(channel)
        silence[channel] = pool(
            fitted["silence"],
            stats.silent if stats is not None else zeros,
            stats.windows if stats is not None else zeros,
            config,
        )
        activity[channel] = pool(
            fitted["active_mean"],
            stats.total if stats is not None else zeros,
            stats.active if stats is not None else zeros,
            config,
        )
    return HouseholdChannels(
        household=household,
        states=states,
        config=config,
        population_sha256=population.sha256(),
        population_fitted_on=population.fitted_on,
        until=until.isoformat() if until is not None else None,
        silence=silence,
        activity=activity,
    )
