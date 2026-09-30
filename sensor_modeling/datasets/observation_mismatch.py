"""Observation-model mismatch of the fitted channel models, window by window.

The question is how surprising a window's evidence is under every behavioural
state, not which state it favours. :mod:`sensor_modeling.evaluation.mismatch`
defines the measures. This module computes their raw terms from the fitted
channel models, exactly and in log space, and assembles a household's trace.

The count law
-------------
Every channel family used here is a hurdle: silence with probability ``π``,
and otherwise a zero-truncated negative binomial with mean ``μ`` and
dispersion ``α`` before truncation (:mod:`.channel_models`)::

    P(X = 0 | s) = π
    P(X = k | s) = (1 − π) · NB(k; μ, α) / (1 − NB(0; μ, α)),   k ≥ 1

``α = 0`` is the zero-truncated Poisson, which gives the hurdle model. The
declared and fitted Poisson channels are the hurdle with ``π = e^{−λ}`` and
``μ = λ``, which is exactly the Poisson with mean ``λ``.

Unlike the models' ``loglik``, which drops the ``log k!`` every state shares,
these probabilities are normalised over the counts: a surprise must be
absolute to compare with what the state itself produces. The tails are exact:

- the upper tail, ``P(X ≥ x | s)``, from the negative binomial's or Poisson's
  survival function;
- the lower tail, ``P(X ≤ x | s)``, by summing the probabilities of every count
  up to ``x``;
- where the survival function nears double precision's underflow, the upper
  tail is summed directly in log space over its first terms, with a geometric
  bound on the rest.

The expected surprise and its variance, the entropy and varentropy of the law,
sum the probabilities of every count up to the one whose upper tail is below
``10⁻¹⁵``, or :data:`MOMENT_LIMIT` counts.

Unavailable evidence
--------------------
A known sensor failure is given as intervals per channel. A window overlapping
one is ``failed`` on that channel: its count enters no measure, and the
recursion's prediction also leaves it out, as a filter aware of the failure
would. A missing count, NaN, is ``missing``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from scipy.special import logsumexp
from scipy.stats import nbinom, poisson

from ..evaluation.mismatch import AVAILABLE, FAILED, MISSING, MismatchTrace
from ..states.ontology import StateOntology
from .casas import CasasRecording
from .channel_models import (
    FittedChannels,
    HurdleChannel,
    HurdleNBChannel,
    _log1mexp,
    filter_recursion,
    household_channel_counts,
    nb_log_zero,
    ztnb_log_pmf,
)
from .information_sets import EvidenceChannel, EvidenceResolution
from .restricted_filter import ChannelLikelihood, ChannelModel

#: Upper-tail mass beyond which the moments stop summing.
MOMENT_TAIL = 1e-15

#: Most counts the moments sum over.
MOMENT_LIMIT = 1_000_000

#: Largest count the lower tail enumerates; beyond it, the complement is used.
ENUMERATION_LIMIT = 1_000_000

#: A log survival below this, about ``10⁻²⁹⁰``, is near double precision's
#: underflow, and the upper tail is summed directly instead.
UNDERFLOW = -668.0

#: Terms the direct upper-tail sum takes before its geometric bound.
FAR_TERMS = 64


def _checked_counts(counts: Any) -> np.ndarray:
    values = np.atleast_1d(np.asarray(counts, dtype=float))
    if values.ndim != 1:
        raise ValueError("counts must be one-dimensional")
    if not np.all(np.isfinite(values)) or (values < 0).any():
        raise ValueError("counts must be finite and non-negative")
    if not np.array_equal(values, np.round(values)):
        raise ValueError("counts must be whole numbers")
    return values


@dataclass(frozen=True)
class CountLaw:
    """One channel's normalised count distribution under each state.

    Attributes
    ----------
    silence
        ``π``, the probability of a silent window, per state, in ``(0, 1)``.
    rate
        ``μ``, the untruncated mean of an active window's count, per state.
    dispersion
        ``α ≥ 0``, per state; ``0`` is the Poisson.
    """

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
    def _log_active_nb(self) -> np.ndarray:
        """``log(1 − NB(0))``: the untruncated law's probability of any activity."""
        return _log1mexp(nb_log_zero(self.rate, self.dispersion))

    def _log_nb_sf(self, counts: np.ndarray) -> np.ndarray:
        """``(counts, states)``: ``log P(K > k)`` of the untruncated law."""
        k = counts[:, None]
        poisson_limit = self.dispersion <= 0.0
        alpha = np.where(poisson_limit, 1.0, self.dispersion)
        with np.errstate(divide="ignore"):
            values: np.ndarray = np.where(
                poisson_limit,
                poisson.logsf(k, self.rate),
                nbinom.logsf(k, 1.0 / alpha, 1.0 / (1.0 + alpha * self.rate)),
            )
        return values

    def _log_positive(self, counts: np.ndarray) -> np.ndarray:
        """``(counts, states)``: the zero-truncated law's log-probability of each count ≥ 1."""
        return ztnb_log_pmf(counts[:, None], self.rate, self.dispersion)

    def log_pmf(self, counts: Any) -> np.ndarray:
        """``(counts, states)``: ``log P(X = x | s)``, normalised over the counts."""
        k = _checked_counts(counts)
        active = np.log1p(-self.silence) + self._log_positive(np.maximum(k, 1.0))
        values: np.ndarray = np.where(
            (k == 0)[:, None], np.log(self.silence)[None, :], active
        )
        return values

    def _far_upper(self, counts: np.ndarray) -> np.ndarray:
        """``(counts, states)``: ``log P(X ≥ x)`` summed directly, for tails that underflow.

        The first :data:`FAR_TERMS` probabilities are summed in log space, and
        the rest bounded by a geometric series with the last ratio of
        successive terms.
        """
        grid = counts[:, None] + np.arange(FAR_TERMS + 1.0)[None, :]
        log_p = self.log_pmf(grid.ravel()).reshape(counts.size, FAR_TERMS + 1, -1)
        head = logsumexp(log_p[:, :FAR_TERMS], axis=1)
        ratio = np.minimum(log_p[:, FAR_TERMS] - log_p[:, FAR_TERMS - 1], -1e-12)
        rest = log_p[:, FAR_TERMS] - _log1mexp(ratio)
        values: np.ndarray = np.logaddexp(head, rest)
        return values

    def log_upper(self, counts: Any) -> np.ndarray:
        """``(counts, states)``: ``log P(X ≥ x | s)``."""
        k = _checked_counts(counts)
        truncated = self._log_nb_sf(np.maximum(k - 1.0, 0.0)) - self._log_active_nb
        values = np.where((k == 0)[:, None], 0.0, np.log1p(-self.silence) + truncated)
        underflow = values < UNDERFLOW
        if underflow.any():
            rows = np.flatnonzero(underflow.any(axis=1))
            values[rows] = np.where(
                underflow[rows], self._far_upper(k[rows]), values[rows]
            )
        # The tail holds the count itself.
        out: np.ndarray = np.minimum(np.maximum(values, self.log_pmf(k)), 0.0)
        return out

    def log_lower(self, counts: Any) -> np.ndarray:
        """``(counts, states)``: ``log P(X ≤ x | s)``."""
        k = _checked_counts(counts)
        size = len(self.silence)
        truncated = np.zeros((k.size, size))
        enumerable = (k >= 1) & (k <= ENUMERATION_LIMIT)
        if enumerable.any():
            top = int(k[enumerable].max())
            grid = np.arange(1.0, top + 1.0)
            cumulative = np.logaddexp.accumulate(self._log_positive(grid), axis=0)
            truncated[enumerable] = cumulative[k[enumerable].astype(int) - 1]
        far = k > ENUMERATION_LIMIT
        if far.any():
            above = np.minimum(self._log_nb_sf(k[far]) - self._log_active_nb, -1e-300)
            truncated[far] = _log1mexp(above)
        log_silence = np.log(self.silence)
        values = np.where(
            (k == 0)[:, None],
            log_silence[None, :],
            np.logaddexp(log_silence, np.log1p(-self.silence) + truncated),
        )
        out: np.ndarray = np.minimum(np.maximum(values, self.log_pmf(k)), 0.0)
        return out

    def _moment_top(self) -> int:
        """A count beyond which every state's law holds under :data:`MOMENT_TAIL`.

        Found by doubling on the log survival function, which stays accurate
        where the quantile functions fail for so small a tail. It is at most
        twice the smallest such count, and at most :data:`MOMENT_LIMIT`.
        """
        # The truncated law's tail is the untruncated one's over 1 − NB(0).
        log_mass = math.log(MOMENT_TAIL) + self._log_active_nb
        top = 1.0
        while top < MOMENT_LIMIT:
            above = np.diagonal(self._log_nb_sf(np.full(len(self.rate), top)))
            if np.all(above <= log_mass):
                break
            top *= 2.0
        return int(min(top, MOMENT_LIMIT))

    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        """The entropy and varentropy of each state's law: the mean and variance of ``−log P(X)``."""
        grid = np.arange(0.0, self._moment_top() + 1.0)
        log_p = self.log_pmf(grid)
        p = np.exp(log_p)
        entropy = -(p * log_p).sum(axis=0)
        varentropy = (p * (-log_p - entropy) ** 2).sum(axis=0)
        return entropy, varentropy


def count_law(model: ChannelModel) -> CountLaw:
    """The normalised count law of a fitted or declared channel model."""
    if isinstance(model, HurdleNBChannel):
        return CountLaw(model.silence, model.rate, model.dispersion)
    if isinstance(model, HurdleChannel):
        return CountLaw(model.silence, model.rate, np.zeros_like(model.rate))
    if isinstance(model, ChannelLikelihood):
        mean = np.asarray(model.expected, dtype=float)
        return CountLaw(np.exp(-mean), mean, np.zeros_like(mean))
    raise TypeError(f"no count law for a {type(model).__name__}")


def training_support(
    fitted: FittedChannels, channels: Iterable[EvidenceChannel]
) -> dict[str, np.ndarray]:
    """The largest count each state's training windows showed on each channel.

    NaN where the fit holds no windows of the state, or no table of active
    counts.
    """
    size = len(fitted.states)
    out: dict[str, np.ndarray] = {}
    for channel in channels:
        stats = fitted.statistics.get(channel)
        top = np.full(size, np.nan)
        if stats is not None and stats.active_counts is not None:
            table = stats.active_counts
            for state in range(size):
                seen = table.values[table.frequencies[state] > 0]
                if seen.size:
                    top[state] = float(seen.max())
                elif stats.windows[state] > 0:
                    top[state] = 0.0
        out[channel.name] = top
    return out


@dataclass(frozen=True)
class PatternReference:
    """Which channels were instrumented and active in each training window.

    Attributes
    ----------
    channels
        The channel names, in bit order.
    states
        The state order of ``labels``.
    households
        The training households the windows come from.
    instrumented, active
        ``(windows,)`` bit masks over ``channels``.
    labels
        ``(windows,)`` each window's labelled state index.
    """

    channels: tuple[str, ...]
    states: tuple[str, ...]
    households: tuple[str, ...]
    instrumented: np.ndarray
    active: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        """Validate the layout and freeze the arrays."""
        if len(self.channels) > 62 or len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must be distinct, at most 62")
        instrumented = np.asarray(self.instrumented, dtype=np.int64)
        active = np.asarray(self.active, dtype=np.int64)
        labels = np.asarray(self.labels, dtype=int)
        if not (instrumented.shape == active.shape == labels.shape) or labels.ndim != 1:
            raise ValueError("one mask pair and label per window")
        if np.any(active & ~instrumented):
            raise ValueError("an active channel must be instrumented")
        if labels.size and (labels.min() < 0 or labels.max() >= len(self.states)):
            raise ValueError("labels must index the states")
        for name, values in (
            ("instrumented", instrumented),
            ("active", active),
            ("labels", labels),
        ):
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "households", tuple(sorted(self.households)))

    def support(
        self, channels: Sequence[str], available: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """How many training windows showed each window's activity pattern.

        A training window is comparable when it instrumented every available
        channel. It shows the pattern when, on those channels, exactly the
        active ones fired. Returns the ``(windows, states)`` counts that show it
        and the ``(windows,)`` comparable counts. A channel the reference does
        not know leaves no training window comparable.
        """
        bit = {name: 1 << i for i, name in enumerate(self.channels)}
        unknown = 1 << 62
        weights = np.array([bit.get(c, unknown) for c in channels], dtype=np.int64)
        masks = (np.asarray(available, dtype=np.int64) * weights).sum(axis=1)
        patterns = (np.asarray(active, dtype=np.int64) * weights).sum(axis=1)
        support = np.zeros((masks.size, len(self.states)), dtype=int)
        comparable = np.zeros(masks.size, dtype=int)
        keys = np.stack([masks, patterns], axis=1)
        for mask, pattern in np.unique(keys, axis=0):
            rows = (masks == mask) & (patterns == pattern)
            covered = (self.instrumented & mask) == mask
            shows = covered & ((self.active & mask) == pattern)
            comparable[rows] = int(covered.sum())
            support[rows] = np.bincount(self.labels[shows], minlength=len(self.states))
        return support, comparable


def pattern_reference(
    recordings: Mapping[str, CasasRecording],
    households: Sequence[str],
    *,
    resolution: EvidenceResolution,
    ontology: StateOntology | None = None,
) -> PatternReference:
    """The activity patterns of the training households' labelled windows."""
    ontology = ontology or StateOntology()
    channels = tuple(c.name for c in resolution.channels)
    bit = {name: 1 << i for i, name in enumerate(channels)}
    instrumented: list[np.ndarray] = []
    active: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for home in sorted(households):
        counts, rows, truth, _ = household_channel_counts(
            recordings[home], resolution, ontology, household=home
        )
        mask = sum(bit[c.name] for c in counts)
        fired = np.zeros(rows.size, dtype=np.int64)
        for channel, column in counts.items():
            fired |= np.where(column[rows] > 0, bit[channel.name], 0)
        instrumented.append(np.full(rows.size, mask, dtype=np.int64))
        active.append(fired)
        labels.append(truth)
    return PatternReference(
        channels,
        tuple(s.value for s in ontology.states),
        tuple(households),
        np.concatenate(instrumented) if instrumented else np.zeros(0, np.int64),
        np.concatenate(active) if active else np.zeros(0, np.int64),
        np.concatenate(labels) if labels else np.zeros(0, int),
    )


def _column_table(
    law: CountLaw, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The log pmf, upper and lower tails of *counts*, computed once per distinct count."""
    values, index = np.unique(counts, return_inverse=True)
    return (
        law.log_pmf(values)[index],
        law.log_upper(values)[index],
        law.log_lower(values)[index],
    )


def mismatch_trace(
    laws: Mapping[str, CountLaw],
    counts: Mapping[str, np.ndarray],
    *,
    household: str,
    timestamps: Sequence[str],
    states: Sequence[str],
    failed: Mapping[str, np.ndarray] | None = None,
    predicted: np.ndarray | None = None,
    training: Mapping[str, np.ndarray] | None = None,
    patterns: PatternReference | None = None,
    truth: np.ndarray | None = None,
) -> MismatchTrace:
    """Score every window's evidence against every state's count laws.

    Parameters
    ----------
    laws
        Each channel's count law, by channel name. The channels are ordered by
        name.
    counts
        Each channel's ``(windows,)`` counts; NaN where missing.
    household, timestamps, states
        Whose windows, their moments, and the state order of the laws.
    failed
        Each channel's ``(windows,)`` known sensor failures. A failed count
        enters no measure.
    predicted
        ``(windows, states)``: the prediction before each window's evidence.
    training
        Each channel's largest training count per state, from
        :func:`training_support`.
    patterns
        The training windows' activity patterns, from :func:`pattern_reference`.
    truth
        Each window's labelled state index, or ``-1``.
    """
    channels = tuple(sorted(laws))
    if set(counts) != set(channels):
        raise ValueError("counts must cover exactly the channels with a law")
    size, windows = len(states), len(timestamps)
    for name in channels:
        if laws[name].silence.shape != (size,):
            raise ValueError(f"the law of {name!r} needs one value per state")
    status = np.full((windows, len(channels)), AVAILABLE, dtype=int)
    table = np.full((windows, len(channels)), np.nan)
    shape = (windows, len(channels), size)
    log_pmf, log_upper, log_lower = (np.full(shape, np.nan) for _ in range(3))
    entropy = np.zeros((len(channels), size))
    varentropy = np.zeros((len(channels), size))
    for j, name in enumerate(channels):
        column = np.asarray(counts[name], dtype=float)
        if column.shape != (windows,):
            raise ValueError(f"the counts of {name!r} need one value per window")
        broken = (
            np.asarray(failed[name], dtype=bool)
            if failed is not None and name in failed
            else np.zeros(windows, dtype=bool)
        )
        if broken.shape != (windows,):
            raise ValueError(f"the failures of {name!r} need one flag per window")
        status[np.isnan(column), j] = MISSING
        status[broken, j] = FAILED
        table[:, j] = column
        on = status[:, j] == AVAILABLE
        law = laws[name]
        if on.any():
            pmf, upper, lower = _column_table(law, column[on])
            log_pmf[on, j], log_upper[on, j], log_lower[on, j] = pmf, upper, lower
        entropy[j], varentropy[j] = law.moments()
    available = status == AVAILABLE
    support = comparable = None
    if patterns is not None:
        if tuple(patterns.states) != tuple(states):
            raise ValueError("the pattern reference uses another state order")
        active = available & (np.nan_to_num(table) > 0)
        support, comparable = patterns.support(channels, available, active)
    training_max = None
    if training is not None:
        training_max = np.stack(
            [
                np.asarray(training.get(name, np.full(size, np.nan)), dtype=float)
                for name in channels
            ]
        )
    return MismatchTrace(
        household=household,
        timestamps=tuple(timestamps),
        states=tuple(states),
        channels=channels,
        status=status,
        counts=np.where(status == MISSING, np.nan, table),
        log_pmf=log_pmf,
        log_upper=log_upper,
        log_lower=log_lower,
        entropy=entropy,
        varentropy=varentropy,
        silence=np.stack([laws[name].silence for name in channels]),
        truth=(
            np.full(windows, -1, dtype=int)
            if truth is None
            else np.asarray(truth, dtype=int)
        ),
        predicted=predicted,
        training_max=training_max,
        pattern_support=support,
        pattern_comparable=comparable,
    )


def failure_flags(
    moments: Sequence[datetime],
    step: Any,
    intervals: Sequence[tuple[datetime, datetime]],
) -> np.ndarray:
    """Which windows, each ending at a moment and spanning *step*, overlap an interval.

    An interval ``[start, end)`` marks a known sensor failure.
    """
    flags = np.zeros(len(moments), dtype=bool)
    for start, end in intervals:
        if end <= start:
            raise ValueError("a failure interval must end after it starts")
        flags |= np.array([start < m and end > m - step for m in moments], dtype=bool)
    return flags


def household_mismatch(
    recording: CasasRecording,
    models: Mapping[EvidenceChannel, ChannelModel],
    *,
    household: str,
    resolution: EvidenceResolution,
    ontology: StateOntology | None = None,
    fitted: FittedChannels | None = None,
    patterns: PatternReference | None = None,
    failures: (
        Mapping[EvidenceChannel, Sequence[tuple[datetime, datetime]]] | None
    ) = None,
    until: datetime | None = None,
) -> MismatchTrace:
    """A household's labelled windows scored against its channel models.

    The prediction before each window is the filter's recursion over every
    window of the recording, from the stationary distribution, with the
    models' likelihoods, less any failed channel's. The scored windows are the
    labelled ones, and with *until* only those closing after it.

    *fitted*, the population the models come from, supplies the training
    support, and *patterns* the training activity patterns. Neither may
    include the household.
    """
    ontology = ontology or StateOntology()
    if fitted is not None and household in fitted.fitted_on:
        raise ValueError(
            f"household {household!r} was used to fit the population; its training "
            "support would include its own windows"
        )
    if patterns is not None and household in patterns.households:
        raise ValueError(f"household {household!r} is in the pattern reference")
    counts, rows, labels, moments = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    if set(models) != set(counts):
        raise ValueError("the models must cover exactly the instrumented channels")
    failures = failures or {}
    unknown = set(failures) - set(models)
    if unknown:
        raise ValueError(f"failures of channels with no model: {sorted(unknown)}")
    flags = {
        channel: failure_flags(moments, resolution.step, failures.get(channel, ()))
        for channel in models
    }
    loglik = np.sum(
        [
            np.where(flags[c][:, None], 0.0, models[c].loglik(counts[c]))
            for c in sorted(models)
        ],
        axis=0,
    )
    predicted, _ = filter_recursion(
        loglik, ontology.transition(resolution.step), ontology.stationary()
    )
    if until is not None:
        after = np.array([moments[int(r)] > until for r in rows], dtype=bool)
        rows, labels = rows[after], labels[after]
    return mismatch_trace(
        {c.name: count_law(models[c]) for c in models},
        {c.name: np.asarray(counts[c], dtype=float)[rows] for c in models},
        household=household,
        timestamps=[moments[int(r)].isoformat() for r in rows],
        states=[s.value for s in ontology.states],
        failed={c.name: flags[c][rows] for c in models},
        predicted=predicted[rows],
        training=training_support(fitted, models) if fitted is not None else None,
        patterns=patterns,
        truth=labels,
    )


def model_description(family: str, fitted: FittedChannels | None) -> dict[str, Any]:
    """The ``model`` entry of a report: the family and its training households."""
    return {
        "family": family,
        "fitted_on": list(fitted.fitted_on) if fitted is not None else [],
    }
