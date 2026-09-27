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

Uninstrumented channels are absent, not silent: they have no model and no
column. Nothing here clips a probability.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
from scipy.optimize import brentq

from ..observations import SensorRegistry
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
from .restricted_filter import ChannelLikelihood, ChannelModel, channel_likelihoods

HURDLE = "hurdle"
POISSON = "poisson"
DECLARED = "declared"
FAMILIES = (DECLARED, HURDLE, POISSON)

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
# Fitting
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelStatistics:
    """Sufficient statistics of one channel's labelled windows, per state."""

    windows: np.ndarray
    silent: np.ndarray
    total: np.ndarray
    squares: np.ndarray
    declared: np.ndarray

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
    return ChannelStatistics(windows, silent, total, squares, declared * windows)


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
        """``silence``, ``rate`` and ``mean`` per state for *channel*.

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
        return {"silence": silence, "rate": rate, "mean": mean}

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
    states = tuple(ontology.states)
    pooled: dict[EvidenceChannel, list[ChannelStatistics]] = {}
    for home in sorted(recordings):
        recording = recordings[home]
        counts, rows, labels, _ = household_channel_counts(
            recording, resolution, ontology, household=home
        )
        declared, _ = channel_likelihoods(recording.registry, resolution, ontology)
        for channel, column in counts.items():
            pooled.setdefault(channel, []).append(
                _statistics(
                    column[rows], labels, len(states), declared[channel].expected
                )
            )
    statistics = {
        channel: ChannelStatistics(
            *(
                np.sum([getattr(s, name) for s in parts], axis=0)
                for name in ("windows", "silent", "total", "squares", "declared")
            )
        )
        for channel, parts in pooled.items()
    }
    return FittedChannels(
        tuple(sorted(recordings)), states, float(pseudo_windows), statistics
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
