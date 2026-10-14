"""Activity sub-states: within-state temporal dependence in the channel model.

The posterior predictive checks of the fitted hurdle channels found silence in
runs far longer than independence given the state allows, and active counts
that mix a low and a high intensity. Both are what a latent level of activity
that persists within a behavioural state produces. This module gives chosen
behavioural states two hidden sub-states, ``quiet`` and ``busy``, each with its
own hurdle parameters on every channel, and a Markov chain between them.

The model
---------
A joint state is a behavioural state ``s`` and one of its sub-states ``k``.
The channels stay conditionally independent given the joint state, each with
the hurdle model of :mod:`~sensor_modeling.datasets.channel_models`. Over one
step::

    P((s, k) → (s, k'))  = P(s → s) · A_s[k, k']
    P((s, k) → (s', k')) = P(s → s') · e_s'[k']        for s' ≠ s

``P`` is the behavioural transition, here the time prior's at the window's
hour, ``A_s`` the sub-state chain within ``s`` and ``e_s`` the sub-state in
which ``s`` is entered. Leaving ``s`` does not depend on the sub-state, so the
behavioural layer keeps the time prior's transition exactly. A state that is
not split has one sub-state with the fitted hurdle's parameters. The belief in
a behavioural state is the sum over its sub-states.

When a state's two sub-states share their parameters, the model is exactly the
hurdle model: the behavioural belief does not depend on ``A_s`` or ``e_s``.

Fitting
-------
On the training households' labelled windows only. Each split state's windows
form stretches of consecutive labelled windows, and the sub-states are hidden
within them. EM runs a fixed number of iterations from a fixed start:

- **E-step.** Forward-backward within each stretch, entered from ``e_s``.
- **M-step.** Each sub-state's hurdle is the fitted hurdle of
  :class:`~sensor_modeling.datasets.channel_models.FittedChannels` with each
  window weighted by its sub-state probability, the same ``κ`` pseudo-windows
  toward the declared model included; ``A_s`` and ``e_s`` are the expected
  transition and entry counts, each with one pseudo-count.
- **The start.** Both sub-states start from the fitted hurdle's rates. The
  quiet one has each channel's odds of silence multiplied by
  ``odds_factor``, the busy one divided by it; the chain starts with
  ``persistence`` on its diagonal and the entry even.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..observations import SensorRegistry
from ..states.ontology import BehaviouralState, StateOntology
from .casas import CasasRecording
from .channel_models import (
    HURDLE,
    ChannelStatistics,
    FittedChannels,
    HurdleChannel,
    household_channel_counts,
    total_loglik,
    truncated_poisson_rate,
)
from .information_sets import EvidenceChannel, EvidenceResolution
from .predictive_checks import run_windows, stretches_of
from .restricted_filter import channel_likelihoods

#: Sub-states of a split state, in order.
QUIET, BUSY = 0, 1
SUBSTATES = ("quiet", "busy")


@dataclass(frozen=True)
class SubstateConfig:
    """What the sub-state model fixes before it is fitted.

    Attributes
    ----------
    split
        The behavioural states given two sub-states.
    iterations
        EM iterations, always all of them.
    odds_factor
        How far the quiet and busy sub-states start from the fitted hurdle's
        silence, as a factor on its odds.
    persistence
        The starting probability that a sub-state persists over one step.
    pseudo_count
        Added to every expected transition and entry count.
    """

    split: tuple[BehaviouralState, ...]
    iterations: int = 100
    odds_factor: float = 4.0
    persistence: float = 0.9
    pseudo_count: float = 1.0

    def __post_init__(self) -> None:
        """Validate the declaration."""
        split = tuple(BehaviouralState(s) for s in self.split)
        if not split or len(set(split)) != len(split):
            raise ValueError("split must name distinct states")
        if any(s not in StateOntology().states for s in split):
            raise ValueError("every split state must be an ontology state")
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations < 1
        ):
            raise ValueError("iterations must be a positive integer")
        if not math.isfinite(self.odds_factor) or self.odds_factor <= 1.0:
            raise ValueError("odds_factor must exceed 1")
        if not 0.0 < self.persistence < 1.0:
            raise ValueError("persistence must lie in (0, 1)")
        if not math.isfinite(self.pseudo_count) or self.pseudo_count <= 0.0:
            raise ValueError("pseudo_count must be positive")
        object.__setattr__(self, "split", split)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form."""
        return {
            "split": [s.value for s in self.split],
            "substates": list(SUBSTATES),
            "iterations": self.iterations,
            "odds_factor": self.odds_factor,
            "persistence": self.persistence,
            "pseudo_count": self.pseudo_count,
        }


# ----------------------------------------------------------------------------
# The joint state space
# ----------------------------------------------------------------------------
def joint_states(
    states: Sequence[BehaviouralState], split: Sequence[BehaviouralState]
) -> list[tuple[int, int]]:
    """Each joint state as ``(behavioural index, sub-state)``, in ontology order."""
    chosen = set(split)
    return [
        (s, k)
        for s, state in enumerate(states)
        for k in range(len(SUBSTATES) if state in chosen else 1)
    ]


@dataclass(frozen=True)
class SubstateChannels:
    """A fold's sub-state channel model, fitted on its training households.

    Attributes
    ----------
    reference
        The fold's fitted hurdle channels. A state that is not split keeps
        their parameters.
    substates
        One set of fitted channels per sub-state: the reference's statistics,
        with each split state's replaced by the sub-state's weighted ones.
    split
        The split states.
    entry
        ``e_s`` per split state.
    within
        ``A_s`` per split state.
    trace
        Per split state, the training log-likelihood at each E-step.
    windows
        Per split state, the training windows and stretches EM used.
    """

    reference: FittedChannels
    substates: tuple[FittedChannels, ...]
    split: tuple[BehaviouralState, ...]
    entry: Mapping[BehaviouralState, np.ndarray]
    within: Mapping[BehaviouralState, np.ndarray]
    trace: Mapping[BehaviouralState, tuple[float, ...]] = field(default_factory=dict)
    windows: Mapping[BehaviouralState, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the chain and freeze copies of it."""
        if len(self.substates) != len(SUBSTATES):
            raise ValueError(f"a split state has {len(SUBSTATES)} sub-states")
        size = len(SUBSTATES)
        entry, within = {}, {}
        for state in self.split:
            e = np.array(self.entry[state], dtype=float)
            a = np.array(self.within[state], dtype=float)
            if e.shape != (size,) or a.shape != (size, size):
                raise ValueError("each split state needs an entry and a chain")
            if not (np.allclose(e.sum(), 1.0) and np.allclose(a.sum(axis=1), 1.0)):
                raise ValueError("the entry and each row of the chain must sum to 1")
            if (e < 0).any() or (a < 0).any():
                raise ValueError("probabilities must be non-negative")
            e.setflags(write=False)
            a.setflags(write=False)
            entry[state], within[state] = e, a
        object.__setattr__(self, "split", tuple(self.split))
        object.__setattr__(self, "entry", entry)
        object.__setattr__(self, "within", within)

    @property
    def states(self) -> tuple[BehaviouralState, ...]:
        """The behavioural states, in the reference's order."""
        return self.reference.states

    @property
    def joint(self) -> list[tuple[int, int]]:
        """Each joint state as ``(behavioural index, sub-state)``."""
        return joint_states(self.states, self.split)

    def aggregation(self) -> np.ndarray:
        """``(joint, states)``: one where a joint state belongs to a state."""
        joint = self.joint
        out = np.zeros((len(joint), len(self.states)))
        for j, (s, _) in enumerate(joint):
            out[j, s] = 1.0
        return out

    def models(
        self,
        registry: SensorRegistry,
        resolution: EvidenceResolution,
        ontology: StateOntology,
    ) -> list[dict[EvidenceChannel, HurdleChannel]]:
        """One household's hurdle channels in each sub-state."""
        out = []
        for fitted in self.substates:
            models = fitted.models(registry, resolution, HURDLE, ontology)
            out.append(
                {c: m for c, m in models.items() if isinstance(m, HurdleChannel)}
            )
        return out

    def loglik(
        self,
        registry: SensorRegistry,
        resolution: EvidenceResolution,
        counts: Mapping[EvidenceChannel, np.ndarray],
        ontology: StateOntology,
    ) -> np.ndarray:
        """``(windows, joint)`` log-likelihood of each window in each joint state."""
        per = [
            total_loglik(models, counts)
            for models in self.models(registry, resolution, ontology)
        ]
        return np.stack([per[k][:, s] for s, k in self.joint], axis=1)

    def factors(self) -> np.ndarray:
        """``(joint, joint)``: ``A_s`` within a state and ``e_s'`` into another."""
        joint = self.joint
        out = np.zeros((len(joint), len(joint)))
        for j, (s, k) in enumerate(joint):
            for i, (t, m) in enumerate(joint):
                state = self.states[t]
                split = state in self.within
                if t == s:
                    out[j, i] = self.within[state][k, m] if split else 1.0
                else:
                    out[j, i] = self.entry[state][m] if split else 1.0
        return out

    def transitions(self, behavioural: np.ndarray) -> np.ndarray:
        """The joint transition for each behavioural one, ``(..., joint, joint)``."""
        index = np.array([s for s, _ in self.joint])
        expanded: np.ndarray = behavioural[..., index[:, None], index[None, :]]
        joint: np.ndarray = expanded * self.factors()
        return joint

    def start(self, prior: np.ndarray) -> np.ndarray:
        """The joint belief from a behavioural one: each state entered from ``e_s``."""
        prior = np.asarray(prior, dtype=float)
        return np.array(
            [
                prior[s]
                * (
                    self.entry[self.states[s]][k]
                    if self.states[s] in self.entry
                    else 1.0
                )
                for s, k in self.joint
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """The fitted chains and each sub-state's parameters, serialisable."""
        states: dict[str, Any] = {}
        for state in self.split:
            s = self.states.index(state)
            channels: dict[str, Any] = {}
            for channel in sorted(self.reference.statistics):
                entries = []
                for fitted in self.substates:
                    stats = fitted.statistics[channel]
                    parameters = fitted.parameters(channel)
                    defined = stats.windows[s] > 0.0
                    entries.append(
                        {
                            "windows": float(stats.windows[s]),
                            "silence": (
                                float(parameters["silence"][s]) if defined else None
                            ),
                            "rate": float(parameters["rate"][s]) if defined else None,
                        }
                    )
                channels[channel.name] = dict(zip(SUBSTATES, entries, strict=True))
            windows, stretches = self.windows.get(state, (0, 0))
            trace = [float(x) for x in self.trace.get(state, ())]
            shares = [
                sum(entry["windows"] for entry in per.values())
                for per in (
                    {c: v[name] for c, v in channels.items()} for name in SUBSTATES
                )
            ]
            silences = [
                [v[name]["silence"] for v in channels.values() if v[name]["silence"]]
                for name in SUBSTATES
            ]
            states[state.value] = {
                "windows": windows,
                "stretches": stretches,
                "entry": [float(x) for x in self.entry[state]],
                "within": [[float(x) for x in row] for row in self.within[state]],
                "share": [
                    float(x / sum(shares)) if sum(shares) > 0 else None for x in shares
                ],
                "mean_silence": [
                    float(np.mean(values)) if values else None for values in silences
                ],
                "trace": trace,
                "last_change": trace[-1] - trace[-2] if len(trace) > 1 else None,
                "channels": channels,
            }
        return {
            "model": "two activity sub-states per split state, hurdle channels "
            "within each, fitted by EM",
            "fitted_on": list(self.reference.fitted_on),
            "pseudo_windows": self.reference.pseudo_windows,
            "split": [s.value for s in self.split],
            "states": states,
        }

    def sha256(self) -> str:
        """SHA-256 of :meth:`to_dict`."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def chain(persistence: float, size: int = len(SUBSTATES)) -> np.ndarray:
    """A chain that stays with *persistence* and otherwise moves evenly."""
    off = (1.0 - persistence) / (size - 1)
    values: np.ndarray = np.full((size, size), off) + np.eye(size) * (persistence - off)
    return values


def tied(
    reference: FittedChannels,
    split: Sequence[BehaviouralState],
    entry: Sequence[float] = (0.5, 0.5),
    persistence: float = 0.7,
) -> SubstateChannels:
    """The model whose sub-states share the reference's parameters.

    Its chain does not matter: the model is exactly the hurdle model.
    """
    split = tuple(split)
    return SubstateChannels(
        reference,
        (reference,) * len(SUBSTATES),
        split,
        {state: np.asarray(entry, dtype=float) for state in split},
        {state: chain(persistence) for state in split},
    )


# ----------------------------------------------------------------------------
# Training windows
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class HouseholdWindows:
    """One household's channel counts, labels and declared expected counts.

    Attributes
    ----------
    counts
        Each instrumented channel's count at every moment.
    rows
        The labelled moments.
    labels
        Their state indices.
    hours
        The local hour at the end of every moment.
    declared
        Each instrumented channel's declared expected count per state.
    registry
        The household's sensor registry.
    """

    counts: Mapping[EvidenceChannel, np.ndarray]
    rows: np.ndarray
    labels: np.ndarray
    hours: tuple[int, ...]
    declared: Mapping[EvidenceChannel, np.ndarray]
    registry: SensorRegistry


def household_windows(
    recording: CasasRecording,
    resolution: EvidenceResolution,
    ontology: StateOntology,
    *,
    household: str,
) -> HouseholdWindows:
    """Read one household's windows once, for fitting and for scoring."""
    counts, rows, labels, moments = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    declared, _ = channel_likelihoods(recording.registry, resolution, ontology)
    return HouseholdWindows(
        counts,
        np.asarray(rows, dtype=int),
        np.asarray(labels, dtype=int),
        tuple(moment.hour for moment in moments),
        {channel: np.asarray(declared[channel].expected) for channel in counts},
        recording.registry,
    )


@dataclass(frozen=True)
class _Stretches:
    """One split state's training windows, packed by offset within a stretch.

    Stretches are sorted longest first, so at offset ``t`` the stretches still
    running are the first ``sizes[t]``, and their windows lie contiguously at
    ``bounds[t]:bounds[t + 1]``. Forward-backward then runs one vector step per
    offset rather than one step per window.
    """

    sizes: np.ndarray
    bounds: np.ndarray
    values: Mapping[EvidenceChannel, np.ndarray]
    present: Mapping[EvidenceChannel, np.ndarray]
    declared: Mapping[EvidenceChannel, np.ndarray]
    stretches: int

    @property
    def windows(self) -> int:
        """Windows in all stretches."""
        return int(self.bounds[-1])


def _stretches(
    homes: Mapping[str, HouseholdWindows],
    s: int,
    channels: Sequence[EvidenceChannel],
) -> _Stretches | None:
    starts: list[int] = []
    lengths: list[int] = []
    columns: dict[EvidenceChannel, list[np.ndarray]] = {c: [] for c in channels}
    present: dict[EvidenceChannel, list[np.ndarray]] = {c: [] for c in channels}
    declared: dict[EvidenceChannel, list[np.ndarray]] = {c: [] for c in channels}
    offset = 0
    for home in sorted(homes):
        data = homes[home]
        chosen = data.rows[data.labels == s]
        if chosen.size == 0:
            continue
        stretch = stretches_of(chosen)
        first = np.flatnonzero(np.concatenate([[True], stretch[1:] != stretch[:-1]]))
        starts.extend(int(offset + f) for f in first)
        lengths.extend(np.diff(np.append(first, chosen.size)).tolist())
        for channel in channels:
            if channel in data.counts:
                columns[channel].append(data.counts[channel][chosen].astype(float))
                present[channel].append(np.ones(chosen.size, dtype=bool))
                declared[channel].append(
                    np.full(chosen.size, float(data.declared[channel][s]))
                )
            else:
                columns[channel].append(np.zeros(chosen.size))
                present[channel].append(np.zeros(chosen.size, dtype=bool))
                declared[channel].append(np.zeros(chosen.size))
        offset += chosen.size
    if not starts:
        return None
    order = np.argsort(-np.asarray(lengths), kind="stable")
    sorted_starts = np.asarray(starts)[order]
    sorted_lengths = np.asarray(lengths)[order]
    longest = int(sorted_lengths[0])
    sizes = np.array([int(np.sum(sorted_lengths > t)) for t in range(longest)])
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    index = np.empty(int(bounds[-1]), dtype=int)
    for t in range(longest):
        index[bounds[t] : bounds[t + 1]] = sorted_starts[: sizes[t]] + t
    return _Stretches(
        sizes,
        bounds,
        {c: np.concatenate(columns[c])[index] for c in channels},
        {c: np.concatenate(present[c])[index] for c in channels},
        {c: np.concatenate(declared[c])[index] for c in channels},
        len(starts),
    )


def _hurdle_loglik(
    channel: EvidenceChannel, values: np.ndarray, silence: float, rate: float
) -> np.ndarray:
    """The hurdle's log-likelihood of each count, up to ``log n!``."""
    model = HurdleChannel(channel, np.array([silence]), np.array([rate]))
    values = model.loglik(values)[:, 0]
    return values


def _parameters(
    windows: float, silent: float, total: float, declared: float, kappa: float
) -> tuple[float, float]:
    """The weighted hurdle fit, exactly as :meth:`FittedChannels.parameters`."""
    prior_mean = declared / windows
    prior_silence = math.exp(-prior_mean)
    prior_active_mean = prior_mean / -math.expm1(-prior_mean)
    silence = (silent + kappa * prior_silence) / (windows + kappa)
    active_mean = (total + kappa * prior_active_mean) / (windows - silent + kappa)
    return silence, truncated_poisson_rate(active_mean)


def forward_backward(
    loglik: np.ndarray,
    sizes: np.ndarray,
    bounds: np.ndarray,
    entry: np.ndarray,
    within: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Sub-state posteriors within packed stretches.

    Returns each window's sub-state probabilities, the expected transition
    counts, the expected entry counts and the log-likelihood.
    """
    size = loglik.shape[1]
    alpha = np.empty_like(loglik)
    scaled = np.empty_like(loglik)
    norm = np.empty(loglik.shape[0])
    total = 0.0
    for t in range(sizes.size):
        lo, hi = int(bounds[t]), int(bounds[t + 1])
        if t == 0:
            predicted = np.broadcast_to(entry, (hi - lo, size))
        else:
            back = int(bounds[t - 1])
            predicted = alpha[back : back + (hi - lo)] @ within
        top = loglik[lo:hi].max(axis=1, keepdims=True)
        scaled[lo:hi] = np.exp(loglik[lo:hi] - top)
        joint = predicted * scaled[lo:hi]
        norm[lo:hi] = joint.sum(axis=1)
        alpha[lo:hi] = joint / norm[lo:hi, None]
        total += float(np.log(norm[lo:hi]).sum() + top.sum())
    beta = np.ones_like(loglik)
    transitions = np.zeros((size, size))
    for t in range(sizes.size - 2, -1, -1):
        lo = int(bounds[t])
        nxt, end = int(bounds[t + 1]), int(bounds[t + 2])
        carried = scaled[nxt:end] * beta[nxt:end] / norm[nxt:end, None]
        beta[lo : lo + (end - nxt)] = carried @ within.T
        transitions += within * (alpha[lo : lo + (end - nxt)].T @ carried)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    entries = gamma[int(bounds[0]) : int(bounds[1])].sum(axis=0)
    return gamma, transitions, entries, total


def _initial(
    reference: FittedChannels,
    channels: Sequence[EvidenceChannel],
    s: int,
    config: SubstateConfig,
) -> list[dict[EvidenceChannel, tuple[float, float]]]:
    out: list[dict[EvidenceChannel, tuple[float, float]]] = [{}, {}]
    for channel in channels:
        fitted = reference.parameters(channel)
        silence, rate = float(fitted["silence"][s]), float(fitted["rate"][s])
        if not (math.isfinite(silence) and math.isfinite(rate)):
            continue
        odds = silence / (1.0 - silence)
        for k, factor in (
            (QUIET, config.odds_factor),
            (BUSY, 1.0 / config.odds_factor),
        ):
            out[k][channel] = (odds * factor / (1.0 + odds * factor), rate)
    return out


def _fit_state(
    packed: _Stretches,
    reference: FittedChannels,
    s: int,
    config: SubstateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, ...]]:
    """EM for one split state: the weights, entry, chain and trace."""
    size = len(SUBSTATES)
    kappa = reference.pseudo_windows
    channels = sorted(packed.values)
    parameters = _initial(reference, channels, s, config)
    entry = np.full(size, 1.0 / size)
    within = chain(config.persistence, size)
    trace: list[float] = []
    gamma = np.full((packed.windows, size), 1.0 / size)
    for _ in range(config.iterations):
        loglik = np.zeros((packed.windows, size))
        for k in range(size):
            for channel in channels:
                if channel not in parameters[k]:
                    continue
                mask = packed.present[channel]
                silence, rate = parameters[k][channel]
                loglik[mask, k] += _hurdle_loglik(
                    channel, packed.values[channel][mask], silence, rate
                )
        gamma, transitions, entries, total = forward_backward(
            loglik, packed.sizes, packed.bounds, entry, within
        )
        trace.append(total)
        entry = (entries + config.pseudo_count) / (
            entries.sum() + size * config.pseudo_count
        )
        counts = transitions + config.pseudo_count
        within = counts / counts.sum(axis=1, keepdims=True)
        for k in range(size):
            for channel in channels:
                mask = packed.present[channel]
                weight = gamma[mask, k]
                windows = float(weight.sum())
                if windows <= 0.0:
                    parameters[k].pop(channel, None)
                    continue
                values = packed.values[channel][mask]
                parameters[k][channel] = _parameters(
                    windows,
                    float(weight[values == 0].sum()),
                    float(weight @ values),
                    float(weight @ packed.declared[channel][mask]),
                    kappa,
                )
    return gamma, entry, within, tuple(trace)


def _weighted(
    stats: ChannelStatistics,
    s: int,
    packed: _Stretches,
    channel: EvidenceChannel,
    weight: np.ndarray,
) -> ChannelStatistics:
    """*stats* with state *s* replaced by the windows weighted by *weight*."""
    mask = packed.present[channel]
    w = weight[mask]
    values = packed.values[channel][mask]
    arrays = {
        "windows": float(w.sum()),
        "silent": float(w[values == 0].sum()),
        "total": float(w @ values),
        "squares": float(w @ (values * values)),
        "declared": float(w @ packed.declared[channel][mask]),
    }
    out: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        column = np.array(getattr(stats, name), dtype=float)
        column[s] = value
        out[name] = column
    return ChannelStatistics(
        out["windows"], out["silent"], out["total"], out["squares"], out["declared"]
    )


def _mean_silence(
    packed: _Stretches, weight: np.ndarray, reference: FittedChannels
) -> float:
    """A sub-state's silence probability, averaged over the channels it was fitted on."""
    values = []
    for channel in sorted(packed.values):
        mask = packed.present[channel]
        w = weight[mask]
        if not mask.any() or w.sum() <= 0.0:
            continue
        counts = packed.values[channel][mask]
        silence, _ = _parameters(
            float(w.sum()),
            float(w[counts == 0].sum()),
            float(w @ counts),
            float(w @ packed.declared[channel][mask]),
            reference.pseudo_windows,
        )
        values.append(silence)
    return float(np.mean(values)) if values else math.nan


def fit_substates(
    homes: Mapping[str, HouseholdWindows],
    reference: FittedChannels,
    config: SubstateConfig,
) -> SubstateChannels:
    """Fit the sub-states on the training households' labelled windows.

    *reference* is the fold's fitted hurdle, fitted on the same households.
    A split state with no training window keeps two sub-states with the
    reference's parameters, which is exactly the hurdle model there. After
    fitting, the sub-state whose silence probability, averaged over the
    channels, is higher is called quiet; the order changes no belief.
    """
    states = reference.states
    channels = sorted(reference.statistics)
    statistics: list[dict[EvidenceChannel, ChannelStatistics]] = [
        dict(reference.statistics) for _ in SUBSTATES
    ]
    size = len(SUBSTATES)
    entry: dict[BehaviouralState, np.ndarray] = {}
    within: dict[BehaviouralState, np.ndarray] = {}
    trace: dict[BehaviouralState, tuple[float, ...]] = {}
    windows: dict[BehaviouralState, tuple[int, int]] = {}
    for state in config.split:
        s = states.index(state)
        packed = _stretches(homes, s, channels)
        if packed is None:
            entry[state] = np.full(size, 1.0 / size)
            within[state] = chain(config.persistence, size)
            windows[state] = (0, 0)
            continue
        gamma, entry[state], within[state], trace[state] = _fit_state(
            packed, reference, s, config
        )
        silences = [_mean_silence(packed, gamma[:, k], reference) for k in range(size)]
        if silences[QUIET] < silences[BUSY]:
            # EM does not keep the sub-states in order; the quieter is "quiet".
            gamma = gamma[:, ::-1]
            entry[state] = entry[state][::-1]
            within[state] = within[state][::-1, ::-1]
        windows[state] = (packed.windows, packed.stretches)
        for k in range(size):
            for channel in channels:
                if packed.present[channel].any():
                    statistics[k][channel] = _weighted(
                        statistics[k][channel], s, packed, channel, gamma[:, k]
                    )
    return SubstateChannels(
        reference,
        tuple(
            FittedChannels(
                reference.fitted_on, states, reference.pseudo_windows, statistics[k]
            )
            for k in range(size)
        ),
        config.split,
        entry,
        within,
        trace,
        windows,
    )


# ----------------------------------------------------------------------------
# The predictive distribution of quiet runs
# ----------------------------------------------------------------------------
def sample_silence(
    rng: np.random.Generator,
    silence: np.ndarray,
    entry: np.ndarray,
    within: np.ndarray,
    stretch: np.ndarray,
    replicates: int,
) -> np.ndarray:
    """``(replicates, windows)``: whether each window is silent, drawn from the model.

    The sub-state is drawn from *entry* at the start of each stretch and from
    the chain *within* after it; a window is silent with its sub-state's
    *silence*.
    """
    stretch = np.asarray(stretch)
    size = stretch.size
    path = np.empty((replicates, size), dtype=int)
    cumulative = np.cumsum(within, axis=1)
    starts = np.concatenate([[True], stretch[1:] != stretch[:-1]]) if size else []
    for t in range(size):
        uniforms = rng.random(replicates)
        if starts[t]:
            path[:, t] = np.searchsorted(np.cumsum(entry), uniforms, side="right")
        else:
            path[:, t] = (uniforms[:, None] >= cumulative[path[:, t - 1]]).sum(axis=1)
    path = np.minimum(path, len(entry) - 1)
    drawn: np.ndarray = rng.random((replicates, size)) < np.asarray(silence)[path]
    return drawn


def predicted_quiet_runs(
    rng: np.random.Generator,
    silence: np.ndarray,
    entry: np.ndarray,
    within: np.ndarray,
    stretch: np.ndarray,
    replicates: int,
    length: int,
) -> float:
    """The model's mean number of windows in quiet runs of at least *length*."""
    silent = sample_silence(rng, silence, entry, within, stretch, replicates)
    return float(np.mean(run_windows(silent, stretch, length)))


__all__ = [
    "BUSY",
    "QUIET",
    "SUBSTATES",
    "HouseholdWindows",
    "SubstateChannels",
    "SubstateConfig",
    "chain",
    "fit_substates",
    "forward_backward",
    "household_windows",
    "joint_states",
    "predicted_quiet_runs",
    "sample_silence",
    "tied",
]
