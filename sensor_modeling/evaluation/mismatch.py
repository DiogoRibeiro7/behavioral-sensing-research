"""How surprising a window's evidence is under every behavioural state.

A posterior can be confident while every state explains the evidence poorly:
normalisation hides how improbable the evidence was under all of them. This
module measures that, observation-model mismatch, from each channel's
normalised predictive distribution under each state. It is diagnostic only. It
changes no decision and sets no threshold.

For each window it keeps the raw structure, and derives from it:

- each channel's log-probability under each state, and the window's, summed
  over the channels the model treats as independent given the state;
- the best achievable log-probability, over the states;
- each channel's exact upper and lower tail under each state, and a two-sided
  tail, twice the smaller, which is a valid p-value;
- the state-conditional surprise, ``−log p(x | s)``, standardised by its exact
  mean and variance under ``s``: how many standard deviations more surprising
  the evidence is than evidence ``s`` itself produces;
- each channel's excess surprise, its surprise minus its expected surprise,
  which sums to the window's;
- the posterior predictive log-probability and tails under the filter's
  prediction before the window, and how far that falls below the best state;
- the pattern of active channels: its probability, exact tail and standardised
  surprise under each state, and how many training windows showed it;
- each count against the largest count training windows showed.

Unavailable evidence
--------------------
A channel is ``available``, ``missing`` (no count recorded) or ``failed`` (a
known sensor failure). Only available channels enter any measure: a failed
sensor's stuck counts or silence are not behaviour, so they are never scored as
novelty. A window with no available channel has no evidence, and every measure
of it is ``None``: it is neither typical nor surprising.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

#: The format of a written trace.
TRACE_FORMAT = "observation-mismatch-trace/1"

#: Tail probabilities the summaries report shares at. They describe the
#: distribution of each tail; none of them is a decision threshold.
LEVELS = (1e-2, 1e-4, 1e-6, 1e-9)

#: Channel statuses in a window, by code.
AVAILABLE, MISSING, FAILED = 0, 1, 2
STATUS = ("available", "missing", "failed")

#: Most available channels whose activity patterns are enumerated for an
#: exact tail; beyond it the pattern tail is ``None``.
MAX_PATTERN_CHANNELS = 20

#: Relative tolerance under which two log-probabilities tie.
_TIE = 1e-9

_LOG_TWO = math.log(2.0)


def _frozen(values: np.ndarray) -> np.ndarray:
    values = np.array(values)
    values.setflags(write=False)
    return values


def _optional(values: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    return _frozen(array)


def _nullable(values: np.ndarray) -> Any:
    """Nested lists with ``None`` for NaN, as strict JSON needs."""
    array = np.asarray(values, dtype=float)
    out = array.astype(object)
    out[np.isnan(array)] = None
    return out.tolist()


def _number(value: float) -> float | None:
    return None if not math.isfinite(value) else float(value)


def _distribution(values: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "q10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def _share(flags: Any, among: np.ndarray) -> float | None:
    count = int(among.sum())
    return float(np.asarray(flags)[among].sum() / count) if count else None


def _masked_max(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The maximum over the last axis where *mask* holds, NaN elsewhere."""
    filled = np.where(np.isnan(values), -np.inf, values).max(axis=-1)
    out: np.ndarray = np.where(mask, filled, np.nan)
    return out


def _bernoulli_moments(silence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Entropy and varentropy of silent-or-active, per channel and state."""
    log_silent, log_active = np.log(silence), np.log1p(-silence)
    entropy = -(silence * log_silent + (1.0 - silence) * log_active)
    varentropy = silence * (1.0 - silence) * (log_silent - log_active) ** 2
    return entropy, varentropy


@dataclass(frozen=True)
class MismatchTrace:
    """Every window's evidence scored against every state, for one household.

    Build it with :func:`sensor_modeling.datasets.observation_mismatch.mismatch_trace`.
    It holds the raw per-channel terms. Every other quantity is derived from
    them, so a trace read back from a file is checked by recomputing it.

    Attributes
    ----------
    household, timestamps, states, channels
        Whose windows, their moments, the state order and the channel order.
    status
        ``(windows, channels)``: ``AVAILABLE``, ``MISSING`` or ``FAILED``.
    counts
        ``(windows, channels)``: each count, NaN when missing. A failed
        channel's count is kept, and used by nothing.
    log_pmf, log_upper, log_lower
        ``(windows, channels, states)``: the natural log of ``P(X = x | s)``,
        ``P(X ≥ x | s)`` and ``P(X ≤ x | s)``; NaN unless available.
    entropy, varentropy
        ``(channels, states)``: the mean and variance of ``−log P(X | s)``
        under ``s``.
    silence
        ``(channels, states)``: ``P(X = 0 | s)``, for activity patterns.
    predicted
        ``(windows, states)``: the filter's prediction before each window's
        evidence, or ``None``.
    training_max
        ``(channels, states)``: the largest count training windows of each
        state showed, NaN when unknown; or ``None``.
    pattern_support
        ``(windows, states)``: how many training windows of each state showed
        the window's activity pattern, or ``None``.
    pattern_comparable
        ``(windows,)``: how many training windows instrumented every available
        channel, so could show the pattern; with ``pattern_support``.
    truth
        Each window's labelled state index, or ``-1``; used by nothing here.
    """

    household: str
    timestamps: tuple[str, ...]
    states: tuple[str, ...]
    channels: tuple[str, ...]
    status: np.ndarray
    counts: np.ndarray
    log_pmf: np.ndarray
    log_upper: np.ndarray
    log_lower: np.ndarray
    entropy: np.ndarray
    varentropy: np.ndarray
    silence: np.ndarray
    truth: np.ndarray
    predicted: np.ndarray | None = None
    training_max: np.ndarray | None = None
    pattern_support: np.ndarray | None = None
    pattern_comparable: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate the layout and freeze the arrays."""
        w, c, s = len(self.timestamps), len(self.channels), len(self.states)
        if not s or len(set(self.states)) != s:
            raise ValueError("states must be distinct and non-empty")
        if not c or len(set(self.channels)) != c:
            raise ValueError("channels must be distinct and non-empty")
        status = np.asarray(self.status, dtype=int)
        if (
            status.shape != (w, c)
            or not np.isin(status, (AVAILABLE, MISSING, FAILED)).all()
        ):
            raise ValueError("status needs one code per window and channel")
        available = status == AVAILABLE
        counts = np.asarray(self.counts, dtype=float)
        if counts.shape != (w, c):
            raise ValueError("counts must be (windows, channels)")
        if not np.array_equal(np.isnan(counts), status == MISSING):
            raise ValueError("a count is NaN exactly when it is missing")
        present = counts[~np.isnan(counts)]
        if (present < 0).any() or not np.array_equal(present, np.round(present)):
            raise ValueError("counts must be whole and non-negative")
        for name in ("log_pmf", "log_upper", "log_lower"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (w, c, s):
                raise ValueError(f"{name} must be (windows, channels, states)")
            if not np.array_equal(
                ~np.isnan(values), np.repeat(available[:, :, None], s, 2)
            ):
                raise ValueError(f"{name} is defined exactly for available channels")
            finite = values[~np.isnan(values)]
            if not np.all(np.isfinite(finite)) or (finite > 1e-9).any():
                raise ValueError(f"{name} must hold finite log-probabilities")
            object.__setattr__(self, name, _frozen(values))
        for name in ("entropy", "varentropy"):
            values = np.asarray(getattr(self, name), dtype=float)
            if (
                values.shape != (c, s)
                or not np.all(np.isfinite(values))
                or (values < 0).any()
            ):
                raise ValueError(
                    f"{name} must be finite and non-negative, (channels, states)"
                )
            object.__setattr__(self, name, _frozen(values))
        silence = np.asarray(self.silence, dtype=float)
        if silence.shape != (c, s) or not np.all((silence > 0.0) & (silence < 1.0)):
            raise ValueError("silence must lie strictly in (0, 1), (channels, states)")
        truth = np.asarray(self.truth, dtype=int)
        if (
            truth.shape != (w,)
            or truth.min(initial=0) < -1
            or truth.max(initial=-1) >= s
        ):
            raise ValueError("truth needs one state index or -1 per window")
        predicted = _optional(self.predicted, (w, s))
        if predicted is not None and (
            (predicted < 0).any() or np.any(np.abs(predicted.sum(axis=1) - 1.0) > 1e-6)
        ):
            raise ValueError("every prediction must be a distribution over the states")
        training = _optional(self.training_max, (c, s))
        if training is not None:
            known = training[~np.isnan(training)]
            if (known < 0).any() or not np.array_equal(known, np.round(known)):
                raise ValueError("training maxima must be whole and non-negative")
        if (self.pattern_support is None) != (self.pattern_comparable is None):
            raise ValueError("pattern support and comparable windows go together")
        if self.pattern_support is not None:
            support = np.asarray(self.pattern_support, dtype=int)
            comparable = np.asarray(self.pattern_comparable, dtype=int)
            if support.shape != (w, s) or comparable.shape != (w,):
                raise ValueError(
                    "pattern support is (windows, states), comparable (windows,)"
                )
            if (support < 0).any() or (support.sum(axis=1) > comparable).any():
                raise ValueError("pattern support cannot exceed the comparable windows")
            object.__setattr__(self, "pattern_support", _frozen(support))
            object.__setattr__(self, "pattern_comparable", _frozen(comparable))
        object.__setattr__(self, "timestamps", tuple(self.timestamps))
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "status", _frozen(status))
        object.__setattr__(self, "counts", _frozen(counts))
        object.__setattr__(self, "silence", _frozen(silence))
        object.__setattr__(self, "truth", _frozen(truth))
        object.__setattr__(self, "predicted", predicted)
        object.__setattr__(self, "training_max", training)

    # ------------------------------------------------------------------
    # Availability
    @cached_property
    def available(self) -> np.ndarray:
        """``(windows, channels)``: which channels enter the measures."""
        values: np.ndarray = self.status == AVAILABLE
        return values

    @cached_property
    def evidence(self) -> np.ndarray:
        """``(windows,)``: whether any channel is available."""
        values: np.ndarray = np.asarray(self.available.any(axis=1))
        return values

    def _rows(self, values: np.ndarray) -> np.ndarray:
        """*values* with every window without evidence set to NaN."""
        mask = self.evidence.reshape((-1,) + (1,) * (values.ndim - 1))
        out: np.ndarray = np.where(mask, values, np.nan)
        return out

    # ------------------------------------------------------------------
    # The window under each state
    @cached_property
    def loglik(self) -> np.ndarray:
        """``(windows, states)``: ``log p(x | s)``, summed over available channels."""
        terms = np.where(self.available[:, :, None], self.log_pmf, 0.0)
        return self._rows(terms.sum(axis=1))

    @cached_property
    def expected_surprise(self) -> np.ndarray:
        """``(windows, states)``: the mean of ``−log p(X | s)`` under ``s``."""
        return self._rows(self.available.astype(float) @ self.entropy)

    @cached_property
    def surprise_variance(self) -> np.ndarray:
        """``(windows, states)``: the variance of ``−log p(X | s)`` under ``s``."""
        return self._rows(self.available.astype(float) @ self.varentropy)

    @cached_property
    def standardised(self) -> np.ndarray:
        """``(windows, states)``: the surprise's excess over its mean, in standard deviations."""
        excess = -self.loglik - self.expected_surprise
        variance = self.surprise_variance
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(variance > 0.0, excess / np.sqrt(variance), 0.0)
        return self._rows(values)

    @cached_property
    def excess(self) -> np.ndarray:
        """``(windows, channels, states)``: each channel's surprise minus its mean.

        Over the available channels it sums to the window's excess surprise.
        """
        values: np.ndarray = -self.log_pmf - self.entropy[None, :, :]
        return values

    @cached_property
    def best_state(self) -> np.ndarray:
        """``(windows,)``: the state under which the evidence is most probable, or -1."""
        filled = np.where(np.isnan(self.loglik), -np.inf, self.loglik)
        values: np.ndarray = np.where(self.evidence, np.argmax(filled, axis=1), -1)
        return values

    @cached_property
    def max_loglik(self) -> np.ndarray:
        """``(windows,)``: the best achievable log-probability, over the states."""
        return _masked_max(self.loglik, self.evidence)

    @cached_property
    def least_standardised(self) -> np.ndarray:
        """``(windows,)``: the smallest standardised surprise over the states."""
        return -_masked_max(-self.standardised, self.evidence)

    # ------------------------------------------------------------------
    # Channels
    @cached_property
    def log_two_sided(self) -> np.ndarray:
        """``(windows, channels, states)``: the log two-sided tail, ``2 min(upper, lower)``."""
        values: np.ndarray = np.minimum(
            0.0, _LOG_TWO + np.minimum(self.log_upper, self.log_lower)
        )
        return values

    def _best(self, values: np.ndarray) -> np.ndarray:
        """The largest value over the states: the most lenient state's."""
        return _masked_max(values, self.available)

    @cached_property
    def best_two_sided(self) -> np.ndarray:
        """``(windows, channels)``: the two-sided tail under the most lenient state."""
        return self._best(self.log_two_sided)

    @cached_property
    def best_upper(self) -> np.ndarray:
        """``(windows, channels)``: the upper tail under the most lenient state: bursts."""
        return self._best(self.log_upper)

    @cached_property
    def best_lower(self) -> np.ndarray:
        """``(windows, channels)``: the lower tail under the most lenient state."""
        return self._best(self.log_lower)

    @cached_property
    def best_log_pmf(self) -> np.ndarray:
        """``(windows, channels)``: each channel's best achievable log-probability."""
        return self._best(self.log_pmf)

    @cached_property
    def beyond_training(self) -> np.ndarray | None:
        """``(windows, channels)``: counts above every count training showed, or ``None``."""
        if self.training_max is None:
            return None
        known = ~np.isnan(self.training_max)
        seen = np.where(known, self.training_max, -np.inf).max(axis=1)
        top = np.where(known.any(axis=1), seen, np.inf)
        values: np.ndarray = self.available & (
            np.nan_to_num(self.counts) > top[None, :]
        )
        return values

    # ------------------------------------------------------------------
    # The posterior predictive
    def _predictive(self, values: np.ndarray) -> np.ndarray | None:
        if self.predicted is None:
            return None
        with np.errstate(divide="ignore"):
            log_prior = np.log(self.predicted)[:, None, :]
        filled = np.where(np.isnan(values), 0.0, values)
        out: np.ndarray = np.minimum(
            np.where(self.available, logsumexp(log_prior + filled, axis=2), np.nan),
            0.0,
        )
        return out

    @cached_property
    def predictive_log_pmf(self) -> np.ndarray | None:
        """``(windows, channels)``: ``log p(x_c | past)``, or ``None`` without a prediction."""
        return self._predictive(self.log_pmf)

    @cached_property
    def predictive_two_sided(self) -> np.ndarray | None:
        """``(windows, channels)``: the log two-sided tail of the predictive distribution."""
        upper, lower = self._predictive(self.log_upper), self._predictive(
            self.log_lower
        )
        if upper is None or lower is None:
            return None
        values: np.ndarray = np.minimum(0.0, _LOG_TWO + np.minimum(upper, lower))
        return values

    @cached_property
    def predictive_loglik(self) -> np.ndarray | None:
        """``(windows,)``: ``log p(x | past)``, the whole window under the prediction."""
        if self.predicted is None:
            return None
        with np.errstate(divide="ignore"):
            log_prior = np.log(self.predicted)
        filled = np.where(np.isnan(self.loglik), 0.0, self.loglik)
        values: np.ndarray = np.where(
            self.evidence, logsumexp(log_prior + filled, axis=1), np.nan
        )
        return values

    @cached_property
    def predictive_gap(self) -> np.ndarray | None:
        """``(windows,)``: the predictive log-probability minus the best state's; at most 0.

        Near zero, the prediction expected a state that explains the evidence.
        Far below it, the evidence fits some state the prediction did not expect.
        """
        if self.predictive_loglik is None:
            return None
        values: np.ndarray = np.minimum(self.predictive_loglik - self.max_loglik, 0.0)
        return values

    # ------------------------------------------------------------------
    # The pattern of active channels
    @cached_property
    def active(self) -> np.ndarray:
        """``(windows, channels)``: available channels with at least one activation."""
        values: np.ndarray = self.available & (np.nan_to_num(self.counts) > 0)
        return values

    @cached_property
    def pattern_loglik(self) -> np.ndarray:
        """``(windows, states)``: the log-probability of the window's activity pattern."""
        log_silent, log_active = np.log(self.silence), np.log1p(-self.silence)
        terms = np.where(self.active[:, :, None], log_active[None], log_silent[None])
        terms = np.where(self.available[:, :, None], terms, 0.0)
        return self._rows(terms.sum(axis=1))

    @cached_property
    def pattern_standardised(self) -> np.ndarray:
        """``(windows, states)``: the pattern's standardised surprise under each state."""
        entropy, varentropy = _bernoulli_moments(self.silence)
        weights = self.available.astype(float)
        excess = -self.pattern_loglik - weights @ entropy
        variance = weights @ varentropy
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(variance > 0.0, excess / np.sqrt(variance), 0.0)
        return self._rows(values)

    @cached_property
    def pattern_tail(self) -> np.ndarray:
        """``(windows, states)``: the log probability of a pattern at most as probable.

        Exact, by enumerating every pattern of the available channels; NaN
        beyond :data:`MAX_PATTERN_CHANNELS` channels or without evidence.
        """
        out = np.full((len(self.timestamps), len(self.states)), np.nan)
        log_silent, log_active = np.log(self.silence), np.log1p(-self.silence)
        masks = np.unique(self.available[self.evidence], axis=0)
        for mask in masks:
            columns = np.flatnonzero(mask)
            if columns.size > MAX_PATTERN_CHANNELS:
                continue
            bits = (np.arange(2**columns.size)[:, None] >> np.arange(columns.size)) & 1
            table = bits @ log_active[columns] + (1 - bits) @ log_silent[columns]
            order = np.sort(table, axis=0)
            cumulative = np.logaddexp.accumulate(order, axis=0)
            rows = np.flatnonzero(np.all(self.available == mask, axis=1))
            observed = self.pattern_loglik[rows]
            for state in range(len(self.states)):
                tolerance = _TIE * np.maximum(1.0, np.abs(observed[:, state]))
                position = np.searchsorted(
                    order[:, state], observed[:, state] + tolerance, side="right"
                )
                out[rows, state] = cumulative[np.maximum(position - 1, 0), state]
        tails: np.ndarray = np.minimum(out, 0.0)
        return tails

    @cached_property
    def best_pattern_tail(self) -> np.ndarray:
        """``(windows,)``: the pattern tail under the most lenient state; NaN if not enumerated."""
        enumerated = ~np.isnan(self.pattern_tail).all(axis=1)
        return _masked_max(self.pattern_tail, enumerated)

    @cached_property
    def unseen_pattern(self) -> np.ndarray | None:
        """``(windows,)``: patterns no comparable training window showed, or ``None``.

        A window no training window can be compared with is not unseen: nothing
        is known about it.
        """
        if self.pattern_support is None or self.pattern_comparable is None:
            return None
        values: np.ndarray = (
            self.evidence
            & (self.pattern_comparable > 0)
            & (self.pattern_support.sum(axis=1) == 0)
        )
        return values

    # ------------------------------------------------------------------
    def windows(self) -> list[dict[str, Any]]:
        """Every window's record: the raw terms and every derived measure."""
        states, channels = self.states, self.channels

        def per_state(values: np.ndarray) -> dict[str, float | None]:
            return {s: _number(v) for s, v in zip(states, values.tolist())}

        out: list[dict[str, Any]] = []
        for t, moment in enumerate(self.timestamps):
            record: dict[str, Any] = {
                "timestamp": moment,
                "truth": states[int(self.truth[t])] if self.truth[t] >= 0 else None,
                "evidence": bool(self.evidence[t]),
                "channels": {},
            }
            for j, name in enumerate(channels):
                status = STATUS[int(self.status[t, j])]
                entry: dict[str, Any] = {
                    "status": status,
                    "count": _number(self.counts[t, j]),
                }
                if self.available[t, j]:
                    entry.update(
                        {
                            "log_pmf": per_state(self.log_pmf[t, j]),
                            "log_upper": per_state(self.log_upper[t, j]),
                            "log_lower": per_state(self.log_lower[t, j]),
                            "log_two_sided": per_state(self.log_two_sided[t, j]),
                            "excess": per_state(self.excess[t, j]),
                            "best_log_pmf": _number(self.best_log_pmf[t, j]),
                            "best_two_sided": _number(self.best_two_sided[t, j]),
                            "best_upper": _number(self.best_upper[t, j]),
                            "best_lower": _number(self.best_lower[t, j]),
                            "beyond_training": (
                                bool(self.beyond_training[t, j])
                                if self.beyond_training is not None
                                else None
                            ),
                            "predictive_log_pmf": (
                                _number(self.predictive_log_pmf[t, j])
                                if self.predictive_log_pmf is not None
                                else None
                            ),
                            "predictive_two_sided": (
                                _number(self.predictive_two_sided[t, j])
                                if self.predictive_two_sided is not None
                                else None
                            ),
                        }
                    )
                record["channels"][name] = entry
            if self.evidence[t]:
                best = int(self.best_state[t])
                record.update(
                    {
                        "loglik": per_state(self.loglik[t]),
                        "expected_surprise": per_state(self.expected_surprise[t]),
                        "standardised": per_state(self.standardised[t]),
                        "best_state": states[best],
                        "max_loglik": _number(self.max_loglik[t]),
                        "least_standardised": _number(self.least_standardised[t]),
                        "predictive_loglik": (
                            _number(self.predictive_loglik[t])
                            if self.predictive_loglik is not None
                            else None
                        ),
                        "predictive_gap": (
                            _number(self.predictive_gap[t])
                            if self.predictive_gap is not None
                            else None
                        ),
                        "pattern": {
                            "active": [
                                c for j, c in enumerate(channels) if self.active[t, j]
                            ],
                            "loglik": per_state(self.pattern_loglik[t]),
                            "standardised": per_state(self.pattern_standardised[t]),
                            "log_tail": per_state(self.pattern_tail[t]),
                            "best_log_tail": _number(self.best_pattern_tail[t]),
                            "training_support": (
                                {
                                    s: int(v)
                                    for s, v in zip(states, self.pattern_support[t])
                                }
                                if self.pattern_support is not None
                                else None
                            ),
                            "training_comparable": (
                                int(self.pattern_comparable[t])
                                if self.pattern_comparable is not None
                                else None
                            ),
                        },
                    }
                )
            out.append(record)
        return out

    def summary(self) -> dict[str, Any]:
        """The household's windows, described: availability, distributions and tail shares.

        Shares are over the windows with evidence. At each of :data:`LEVELS`:

        ``channel_tail``
            some channel's two-sided tail is below the level under every state;
        ``burst`` / ``low``
            some channel's upper or lower tail is below it under every state;
        ``pattern_tail``
            the activity pattern's tail is below it under every state;
        ``predictive_tail``
            some channel's two-sided tail under the prediction is below it.
        """
        evidence, available = self.evidence, self.available
        windows = len(self.timestamps)
        channel_windows = {
            name: int((self.status == code).sum()) for code, name in enumerate(STATUS)
        }

        def below(values: np.ndarray, level: float) -> np.ndarray:
            """Where a log tail lies below *level*; NaN, not scored, never does."""
            flags: np.ndarray = np.nan_to_num(values, nan=0.0) < math.log(level)
            return flags

        levels: dict[str, Any] = {}
        pattern_known = evidence & ~np.isnan(self.best_pattern_tail)
        predictive = self.predictive_two_sided
        for level in LEVELS:
            levels[f"{level:g}"] = {
                "channel_tail": _share(
                    below(self.best_two_sided, level).any(axis=1), evidence
                ),
                "burst": _share(below(self.best_upper, level).any(axis=1), evidence),
                "low": _share(below(self.best_lower, level).any(axis=1), evidence),
                "pattern_tail": _share(
                    below(self.best_pattern_tail, level), pattern_known
                ),
                "predictive_tail": (
                    _share(below(predictive, level).any(axis=1), evidence)
                    if predictive is not None
                    else None
                ),
            }
        beyond = self.beyond_training
        unseen = self.unseen_pattern
        comparable = (
            evidence & (self.pattern_comparable > 0)
            if self.pattern_comparable is not None
            else None
        )
        best = np.clip(self.best_state, 0, None)
        best_excess = np.take_along_axis(
            self.excess, np.repeat(best[:, None, None], len(self.channels), 1), 2
        )[:, :, 0]
        channels: dict[str, Any] = {}
        for j, name in enumerate(self.channels):
            on = available[:, j]
            channels[name] = {
                **{
                    status: int((self.status[:, j] == code).sum())
                    for code, status in enumerate(STATUS)
                },
                "mean_excess_at_best_state": (
                    float(best_excess[on, j].mean()) if on.any() else None
                ),
                "beyond_training": (
                    int(beyond[:, j].sum()) if beyond is not None else None
                ),
                "burst": {
                    f"{level:g}": _share(below(self.best_upper[:, j], level), on)
                    for level in LEVELS
                },
            }
        return {
            "windows": windows,
            "evidence_windows": int(evidence.sum()),
            "no_evidence_windows": int(windows - evidence.sum()),
            "channel_windows": channel_windows,
            "max_loglik": _distribution(self.max_loglik),
            "least_standardised": _distribution(self.least_standardised),
            "least_pattern_standardised": _distribution(
                -_masked_max(-self.pattern_standardised, evidence)
            ),
            "predictive_gap": (
                _distribution(self.predictive_gap)
                if self.predictive_gap is not None
                else None
            ),
            "levels": levels,
            "beyond_training": (
                _share(beyond.any(axis=1), evidence) if beyond is not None else None
            ),
            "unseen_pattern": (
                _share(unseen, comparable)
                if unseen is not None and comparable is not None
                else None
            ),
            "channels": channels,
        }

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """The trace's raw terms, from which every other quantity is recomputed."""

        def optional(values: np.ndarray | None) -> Any:
            return _nullable(values) if values is not None else None

        return {
            "format": TRACE_FORMAT,
            "household": self.household,
            "timestamps": list(self.timestamps),
            "states": list(self.states),
            "channels": list(self.channels),
            "status": self.status.tolist(),
            "counts": _nullable(self.counts),
            "log_pmf": _nullable(self.log_pmf),
            "log_upper": _nullable(self.log_upper),
            "log_lower": _nullable(self.log_lower),
            "entropy": self.entropy.tolist(),
            "varentropy": self.varentropy.tolist(),
            "silence": self.silence.tolist(),
            "truth": self.truth.tolist(),
            "predicted": optional(self.predicted),
            "training_max": optional(self.training_max),
            "pattern_support": (
                self.pattern_support.tolist()
                if self.pattern_support is not None
                else None
            ),
            "pattern_comparable": (
                self.pattern_comparable.tolist()
                if self.pattern_comparable is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MismatchTrace:
        """Rebuild a trace written by :meth:`to_dict`."""
        if payload.get("format") != TRACE_FORMAT:
            raise ValueError(f"not a {TRACE_FORMAT} trace")
        shape = (
            len(payload["timestamps"]),
            len(payload["channels"]),
            len(payload["states"]),
        )

        def array(key: str, dims: tuple[int, ...]) -> np.ndarray:
            return np.array(payload[key], dtype=float).reshape(dims)

        def optional(key: str, dims: tuple[int, ...], kind: type = float) -> Any:
            if payload[key] is None:
                return None
            return np.array(payload[key], dtype=kind).reshape(dims)

        w, c, s = shape
        return cls(
            household=str(payload["household"]),
            timestamps=tuple(payload["timestamps"]),
            states=tuple(payload["states"]),
            channels=tuple(payload["channels"]),
            status=np.array(payload["status"], dtype=int).reshape(w, c),
            counts=array("counts", (w, c)),
            log_pmf=array("log_pmf", shape),
            log_upper=array("log_upper", shape),
            log_lower=array("log_lower", shape),
            entropy=array("entropy", (c, s)),
            varentropy=array("varentropy", (c, s)),
            silence=array("silence", (c, s)),
            truth=np.array(payload["truth"], dtype=int),
            predicted=optional("predicted", (w, s)),
            training_max=optional("training_max", (c, s)),
            pattern_support=optional("pattern_support", (w, s), int),
            pattern_comparable=optional("pattern_comparable", (w,), int),
        )

    def write(self, path: Path) -> str:
        """Write the trace as canonical gzip-compressed JSON; return its SHA-256.

        The gzip header carries no time or name, so writing the same trace twice
        gives the same bytes.
        """
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
            handle.write(text.encode("utf-8"))
        data = buffer.getvalue()
        Path(path).write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def read(cls, path: Path, sha256: str | None = None) -> MismatchTrace:
        """Read a trace written by :meth:`write`, checking its digest if given."""
        data = Path(path).read_bytes()
        if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError(f"the trace at {path} does not match its recorded digest")
        return cls.from_dict(json.loads(gzip.decompress(data).decode("utf-8")))


# ----------------------------------------------------------------------------
# In an experiment record
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class MismatchReport:
    """What an experiment record keeps of an observation-mismatch analysis.

    Attributes
    ----------
    model
        The observation model scored against: its ``family``, and the training
        households it was ``fitted_on``, which the training support comes from.
    states
        The state order.
    households
        Per household: the summary of its trace, and where the trace is
        written, with its SHA-256, or ``None`` if it is not.
    """

    model: Mapping[str, Any]
    states: tuple[str, ...]
    households: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        """Validate the report through its serialised form."""
        problems = validate_mismatch(self.to_dict())
        if problems:
            raise ValueError("; ".join(problems))

    @classmethod
    def from_traces(
        cls,
        traces: Sequence[MismatchTrace],
        model: Mapping[str, Any],
        files: Mapping[str, Mapping[str, str]] | None = None,
    ) -> MismatchReport:
        """A report of *traces*, one per household, with each trace's file if written."""
        if not traces:
            raise ValueError("a report needs at least one household's trace")
        states = traces[0].states
        if any(trace.states != states for trace in traces):
            raise ValueError("every trace must use the same states")
        if len({trace.household for trace in traces}) != len(traces):
            raise ValueError("one trace per household")
        households = {
            trace.household: {
                "summary": trace.summary(),
                "trace": (
                    dict(files[trace.household])
                    if files and trace.household in files
                    else None
                ),
            }
            for trace in traces
        }
        return cls(dict(model), states, households)

    def to_dict(self) -> dict[str, Any]:
        """Return the serialisable form an experiment record carries."""
        return {
            "trace_format": TRACE_FORMAT,
            "levels": list(LEVELS),
            "model": dict(self.model),
            "states": list(self.states),
            "households": {h: dict(e) for h, e in sorted(self.households.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MismatchReport:
        """Rebuild a report written by :meth:`to_dict`."""
        return cls(
            model=dict(payload["model"]),
            states=tuple(payload["states"]),
            households=dict(payload["households"]),
        )


def _check_shares(value: Any, where: str, problems: list[str]) -> None:
    if value is not None and not (
        isinstance(value, (int, float)) and 0.0 <= value <= 1.0
    ):
        problems.append(f"{where} must be null or a share in [0, 1]")


def _check_summary(summary: Any, where: str, problems: list[str]) -> None:
    if not isinstance(summary, Mapping):
        problems.append(f"{where} must be an object")
        return
    windows = summary.get("windows")
    evidence = summary.get("evidence_windows")
    empty = summary.get("no_evidence_windows")
    if not all(isinstance(v, int) and v >= 0 for v in (windows, evidence, empty)):
        problems.append(f"{where} must count its windows")
        return
    if int(evidence) + int(empty) != int(windows):  # type: ignore[arg-type]
        problems.append(f"{where}: evidence and no-evidence windows must add up")
    channel_windows = summary.get("channel_windows")
    if not isinstance(channel_windows, Mapping) or set(channel_windows) != set(STATUS):
        problems.append(f"{where}.channel_windows must count {list(STATUS)}")
    levels = summary.get("levels")
    if not isinstance(levels, Mapping) or set(levels) != {f"{v:g}" for v in LEVELS}:
        problems.append(f"{where}.levels must cover {[f'{v:g}' for v in LEVELS]}")
    else:
        for level, shares in levels.items():
            for key, value in dict(shares).items():
                _check_shares(value, f"{where}.levels.{level}.{key}", problems)
    for key in ("beyond_training", "unseen_pattern"):
        _check_shares(summary.get(key), f"{where}.{key}", problems)


def validate_mismatch(payload: Mapping[str, Any]) -> list[str]:
    """Every problem with a record's observation-mismatch section."""
    problems: list[str] = []
    expected = {"trace_format", "levels", "model", "states", "households"}
    if not isinstance(payload, Mapping) or set(payload) != expected:
        return [f"observation_mismatch must have exactly {sorted(expected)}"]
    if payload["trace_format"] != TRACE_FORMAT:
        problems.append(f"observation_mismatch.trace_format must be {TRACE_FORMAT!r}")
    if list(payload["levels"]) != list(LEVELS):
        problems.append(f"observation_mismatch.levels must be {list(LEVELS)}")
    model = payload["model"]
    if (
        not isinstance(model, Mapping)
        or not isinstance(model.get("family"), str)
        or not model["family"].strip()
        or not isinstance(model.get("fitted_on"), list)
    ):
        problems.append(
            "observation_mismatch.model must name its family and list fitted_on"
        )
    states = payload["states"]
    if not isinstance(states, list) or not states or len(set(states)) != len(states):
        problems.append("observation_mismatch.states must be distinct and non-empty")
    households = payload["households"]
    if not isinstance(households, Mapping) or not households:
        problems.append("observation_mismatch.households must not be empty")
        return problems
    fitted_on = (
        set(model.get("fitted_on") or []) if isinstance(model, Mapping) else set()
    )
    for home, entry in households.items():
        where = f"observation_mismatch.households.{home}"
        if home in fitted_on:
            problems.append(f"{where} was used to fit the model it is scored against")
        if not isinstance(entry, Mapping) or set(entry) != {"summary", "trace"}:
            problems.append(f"{where} must have exactly summary and trace")
            continue
        trace = entry["trace"]
        if trace is not None and (
            not isinstance(trace, Mapping)
            or set(trace) != {"file", "sha256"}
            or not isinstance(trace["sha256"], str)
            or len(trace["sha256"]) != 64
        ):
            problems.append(f"{where}.trace must be null or a file with its SHA-256")
        _check_summary(entry["summary"], f"{where}.summary", problems)
    return problems
