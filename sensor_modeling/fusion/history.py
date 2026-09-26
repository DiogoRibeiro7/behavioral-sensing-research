"""An explicit, bounded recent-history state for the generative filter.

The filter's memoryless Poisson emission assumes a window's activations depend
only on the current state. Real activity is bursty and has refractory
periods. A bathroom visit, for example, often starts with a burst and then
falls quiet while the state persists. Phase 1 found that a supervised model
extracts far more from the three previous windows than the filter's posterior
carries. This module gives the generative model an explicit history state
instead of relying on the posterior alone.

Model
-----
For channel ``c`` (a room and a modality) and step windows of length ``Δ``:

- ``n_{c,t}`` is the channel's activations in window ``t``;
- ``h_{c,t} = n_{c,t-1} + … + n_{c,t-k}`` is its history over the ``k``
  previous windows, ``k = 3``, the Phase 1 history depth;
- the history is **observed** only if every one of those windows was observed
  for the channel, and **missing** otherwise.

The emission rate of each sensor ``i`` of the channel becomes

.. math::

    \\lambda_i(s \\mid h) = \\lambda_i(s)\\, e^{\\eta_c(s, h)}, \\qquad
    \\eta_c(s, h) = \\beta_{s, r(c, s)}
                   \\big(\\log(1 + h) - \\mathrm{E}_s[\\log(1 + H)]\\big),

with ``η = 0`` when the history is missing. ``E_s`` is taken under the
memoryless model itself, with state ``s`` held for the history window:
``H ~ Poisson(k Δ Λ_c(s))``, where ``Λ_c`` sums the channel's rates.

- **Interpretation.** A history that matches what state ``s`` predicts leaves
  the rate unchanged. ``β > 0`` means recent activity beyond that prediction
  makes more activity now likely; ``β < 0`` means it makes it less likely.
- **Silence and missingness.** Observed silence is informative. Missing
  history is exactly neutral.
- **Groups.** ``r(c, s)`` is ``own room`` or ``other room`` for a state that
  names a room, and ``any room`` otherwise. That gives one coefficient per
  group, eleven for the default ontology.

A window's counts enter the likelihood exactly once, as the current window;
the history only changes how likely they are. The filter is therefore the
exact forward recursion of a Markov-switching autoregressive model, with no
evidence counted twice. With every ``β = 0`` it is the original filter.

Diagnostics
-----------
Each prediction's log posterior splits exactly into three terms:

.. math::

    \\log p_t(s) = \\underbrace{\\log \\hat p_t(s)}_{\\text{prior and transitions}}
                 + \\underbrace{C_t(s)}_{\\text{current window}}
                 + \\underbrace{R_t(s)}_{\\text{recent history}} - \\log Z_t,

and :class:`PosteriorDecomposition` reports each term, channel by channel for
the history.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from scipy.special import gammaln, logsumexp

from ..observations.observation import Observation, require_aware
from ..observations.registry import SensorRegistry
from ..states.ontology import BehaviouralState, StateOntology
from .emissions import EmissionModel, PoissonEventEmission
from .estimate import EvidenceContribution, StateEstimate
from .filter import (
    FusionConfig,
    MultimodalBayesFilter,
    NonMonotonicUpdateError,
    _kl_divergence,
    _support_for,
)

OWN_ROOM = "own room"
OTHER_ROOM = "other room"
ANY_ROOM = "any room"


def relation(state_room: str | None, channel_room: str) -> str:
    """How a channel's room relates to the room a state names."""
    if state_room is None:
        return ANY_ROOM
    return OWN_ROOM if channel_room == state_room else OTHER_ROOM


def expected_log1p_poisson(means: np.ndarray | Sequence[float]) -> np.ndarray:
    """``E[log(1 + N)]`` for ``N ~ Poisson(mean)``, element by element."""
    values = np.asarray(means, dtype=float)
    out = np.zeros_like(values)
    for index, mean in np.ndenumerate(values):
        if not math.isfinite(mean) or mean < 0.0:
            raise ValueError("Poisson means must be finite and non-negative")
        if mean == 0.0:
            continue
        top = int(math.ceil(mean + 12.0 * math.sqrt(mean) + 30.0))
        counts = np.arange(top + 1)
        log_pmf = counts * math.log(mean) - mean - gammaln(counts + 1.0)
        out[index] = float(np.sum(np.exp(log_pmf) * np.log1p(counts)))
    return out


def history_log_likelihood(
    count: float | np.ndarray,
    expected: np.ndarray,
    eta: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """What conditioning on the history adds to a Poisson log-likelihood.

    For *count* activations in a window whose memoryless expected count is
    *expected* per state, a rate multiplier ``e^η`` adds
    ``weight (count η − expected (e^η − 1))`` to each state's log-likelihood.
    """
    contribution: np.ndarray = weight * (count * eta - expected * np.expm1(eta))
    return contribution


@dataclass(frozen=True)
class HistoryConfig:
    """Declared settings of the history state. None is tuned on data.

    Attributes
    ----------
    window_steps
        ``k``, the previous windows the history covers. Three is the Phase 1
        history depth, the lagged windows of the ``I2`` and ``I3`` sets.
    precision
        Ridge on each coefficient when it is fitted. It keeps a group with few
        training windows near zero, the memoryless model.
    """

    window_steps: int = 3
    precision: float = 1.0

    def __post_init__(self) -> None:
        """Validate the settings."""
        if (
            isinstance(self.window_steps, bool)
            or not isinstance(self.window_steps, int)
            or self.window_steps < 1
        ):
            raise ValueError("window_steps must be a positive integer")
        if not math.isfinite(self.precision) or self.precision <= 0.0:
            raise ValueError("precision must be positive and finite")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serialisable form."""
        return {
            "window_steps": self.window_steps,
            "precision": float(self.precision),
            "feature": "log(1 + activations in the history window)",
            "reference": "its expectation under the memoryless model in each state",
        }


def _key(state: BehaviouralState, group: str) -> str:
    return f"{state.value}|{group}"


@dataclass(frozen=True, eq=False)
class HistoryModel:
    """Coefficients ``β`` of the history-conditioned emission, one per group.

    Attributes
    ----------
    states
        The ontology states, in order.
    rooms
        The room each state names, or ``None``, which fixes its groups.
    config
        The declared settings.
    coefficients
        ``β`` by ``"<state>|<relation>"``.
    fitted_on
        Households whose labels fitted the coefficients, for provenance.
    support
        Training windows behind each coefficient.
    """

    states: tuple[BehaviouralState, ...]
    rooms: tuple[str | None, ...]
    config: HistoryConfig
    coefficients: Mapping[str, float]
    fitted_on: tuple[str, ...] = ()
    support: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check every group has a finite coefficient."""
        states = tuple(BehaviouralState(s) for s in self.states)
        if len(states) != len(self.rooms):
            raise ValueError("every state needs its room, or None")
        expected = {_key(s, g) for s, g in _groups(states, self.rooms)}
        if set(self.coefficients) != expected:
            raise ValueError(
                f"coefficients must cover exactly the groups {sorted(expected)}"
            )
        if not all(math.isfinite(v) for v in self.coefficients.values()):
            raise ValueError("coefficients must be finite")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "rooms", tuple(self.rooms))
        object.__setattr__(
            self,
            "coefficients",
            {k: float(self.coefficients[k]) for k in sorted(expected)},
        )
        object.__setattr__(self, "fitted_on", tuple(sorted(self.fitted_on)))
        object.__setattr__(self, "support", dict(sorted(self.support.items())))

    @classmethod
    def null(
        cls, ontology: StateOntology | None = None, config: HistoryConfig | None = None
    ) -> HistoryModel:
        """Every coefficient zero: exactly the original, memoryless model."""
        ontology = ontology or StateOntology()
        rooms = tuple(ontology.room_of(state) for state in ontology.states)
        return cls(
            states=tuple(ontology.states),
            rooms=rooms,
            config=config or HistoryConfig(),
            coefficients={_key(s, g): 0.0 for s, g in _groups(ontology.states, rooms)},
        )

    def check(self, ontology: StateOntology) -> None:
        """Refuse an ontology whose states or rooms differ from the model's."""
        if (
            tuple(ontology.states) != self.states
            or tuple(ontology.room_of(state) for state in ontology.states) != self.rooms
        ):
            raise ValueError("the history model was built for another ontology")

    def betas(self, channel_room: str) -> np.ndarray:
        """``β`` for each state, for a channel in *channel_room*."""
        return np.array(
            [
                self.coefficients[_key(state, relation(room, channel_room))]
                for state, room in zip(self.states, self.rooms)
            ]
        )

    def modulation(
        self, channel_room: str, history: float | None, reference: np.ndarray
    ) -> np.ndarray:
        """``η`` per state, or zeros when the history is missing.

        *reference* is ``E_s[log(1 + H)]`` per state, from
        :func:`expected_log1p_poisson` of the channel's memoryless expected
        count over the history window.
        """
        if history is None:
            return np.zeros(len(self.states))
        eta: np.ndarray = self.betas(channel_room) * (math.log1p(history) - reference)
        return eta

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serialisable form."""
        return {
            "model": "history-conditioned Poisson emission",
            "states": [state.value for state in self.states],
            "rooms": list(self.rooms),
            "config": self.config.to_dict(),
            "coefficients": dict(self.coefficients),
            "fitted_on": list(self.fitted_on),
            "support": dict(self.support),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HistoryModel:
        """Rebuild a model from :meth:`to_dict`."""
        config = payload["config"]
        if not isinstance(config, Mapping):
            raise ValueError("a history model needs its config")
        return cls(
            states=tuple(BehaviouralState(s) for s in payload["states"]),  # type: ignore[attr-defined]
            rooms=tuple(payload["rooms"]),  # type: ignore[arg-type]
            config=HistoryConfig(
                window_steps=int(config["window_steps"]),
                precision=float(config["precision"]),
            ),
            coefficients=dict(payload["coefficients"]),  # type: ignore[call-overload]
            fitted_on=tuple(payload.get("fitted_on") or ()),  # type: ignore[arg-type]
            support=dict(payload.get("support") or {}),  # type: ignore[call-overload]
        )

    def sha256(self) -> str:
        """SHA-256 of the canonical serialised model."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def _groups(
    states: Iterable[BehaviouralState], rooms: Iterable[str | None]
) -> list[tuple[BehaviouralState, str]]:
    groups: list[tuple[BehaviouralState, str]] = []
    for state, room in zip(states, rooms):
        if room is None:
            groups.append((state, ANY_ROOM))
        else:
            groups.extend([(state, OWN_ROOM), (state, OTHER_ROOM)])
    return groups


@dataclass(frozen=True)
class ChannelHistory:
    """One channel's history at a prediction, and what it contributed.

    Attributes
    ----------
    channel
        ``<room>_<modality>``.
    count
        Activations in the history window, or ``None`` when missing.
    reason
        Why the history is missing, or ``None`` when it was observed.
    modulation
        ``η`` per state; zeros when missing.
    contribution
        Its log-likelihood contribution per state.
    """

    channel: str
    count: int | None
    reason: str | None
    modulation: tuple[float, ...]
    contribution: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form."""
        return {
            "channel": self.channel,
            "count": self.count,
            "reason": self.reason,
            "modulation": list(self.modulation),
            "contribution": list(self.contribution),
        }


@dataclass(frozen=True, eq=False)
class PosteriorDecomposition:
    """Where a prediction's posterior came from.

    ``log posterior = prior_transition + current + history − log Z``, exactly.
    Each term is defined up to a constant shared by every state, so compare
    states within a term, for example with :meth:`log_odds`.

    Attributes
    ----------
    at
        The prediction moment.
    states
        State order of every array.
    prior_transition
        Log of the predicted belief: the prior, or the previous posterior
        carried by the transition model.
    current
        The current window's memoryless log-likelihood, summed over sensors.
    history
        What conditioning on the recent history added, summed over channels.
    posterior
        The resulting belief.
    channels
        Each history channel's state and contribution.
    history_applied
        ``False`` when the update did not cover exactly one step, so no
        history term was applied.
    """

    at: datetime
    states: tuple[BehaviouralState, ...]
    prior_transition: np.ndarray
    current: np.ndarray
    history: np.ndarray
    posterior: np.ndarray
    channels: tuple[ChannelHistory, ...]
    history_applied: bool

    def log_odds(
        self, state: BehaviouralState, versus: BehaviouralState
    ) -> dict[str, float]:
        """Each term's share of the log posterior odds of *state* over *versus*.

        The three shares sum to ``posterior``, the log posterior odds.
        """
        a, b = self.states.index(state), self.states.index(versus)
        return {
            "prior_transition": float(
                self.prior_transition[a] - self.prior_transition[b]
            ),
            "current": float(self.current[a] - self.current[b]),
            "history": float(self.history[a] - self.history[b]),
            "posterior": float(np.log(self.posterior[a]) - np.log(self.posterior[b])),
        }

    def explain(self) -> dict[str, object]:
        """The winner against the runner-up, term by term and channel by channel."""
        order = np.argsort(-self.posterior, kind="stable")
        winner, runner = self.states[int(order[0])], self.states[int(order[1])]
        a, b = int(order[0]), int(order[1])
        return {
            "at": self.at.isoformat(),
            "state": winner.value,
            "versus": runner.value,
            "log_odds": self.log_odds(winner, runner),
            "history_applied": self.history_applied,
            "history_by_channel": {
                c.channel: {
                    "count": c.count,
                    "reason": c.reason,
                    "log_odds": c.contribution[a] - c.contribution[b],
                }
                for c in self.channels
            },
        }

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form."""
        return {
            "at": self.at.isoformat(),
            "states": [state.value for state in self.states],
            "prior_transition": self.prior_transition.tolist(),
            "current": self.current.tolist(),
            "history": self.history.tolist(),
            "posterior": self.posterior.tolist(),
            "channels": [c.to_dict() for c in self.channels],
            "history_applied": self.history_applied,
        }


#: Why a channel's history is missing.
COLD_START = (
    "fewer than k complete windows since the filter started, resumed or had a gap"
)
UNRELIABLE = "a sensor of the channel was unreliable in the history window"
IRREGULAR = "the update did not cover exactly one step"


class HistoryAwareBayesFilter(MultimodalBayesFilter):
    """The generative filter with an explicit, bounded recent-history state.

    It is :class:`MultimodalBayesFilter` with every Poisson event sensor's rate
    conditioned on its channel's history over the previous ``k`` windows, as
    the module documentation describes. The history state is a ring buffer of
    ``k`` windows of per-channel counts, so memory is bounded.

    A window enters the buffer only when an update covers exactly one
    ``step``. The first update, a longer gap, a shorter update, or an
    unreliable sensor records the window as missing rather than silent.
    :meth:`explain` returns the last prediction's
    :class:`PosteriorDecomposition`.
    """

    def __init__(
        self,
        ontology: StateOntology,
        emissions: Iterable[EmissionModel],
        registry: SensorRegistry,
        history: HistoryModel,
        *,
        step: timedelta,
        config: FusionConfig | None = None,
        prior: np.ndarray | None = None,
    ) -> None:
        super().__init__(ontology, emissions, registry, config, prior)
        history.check(ontology)
        if step <= timedelta(0):
            raise ValueError("step must be positive")
        self.history = history
        self.step = step
        self._rates: dict[str, np.ndarray] = {}
        self._channel_of: dict[str, str] = {}
        members: dict[str, list[str]] = {}
        self._room_of: dict[str, str] = {}
        for sensor_id, emission in sorted(self.emissions.items()):
            spec = registry.get(sensor_id)
            if not isinstance(emission, PoissonEventEmission) or spec is None:
                continue
            if not spec.room:
                continue
            channel = f"{spec.room}_{spec.modality.value}"
            self._channel_of[sensor_id] = channel
            self._room_of[channel] = spec.room
            members.setdefault(channel, []).append(sensor_id)
            self._rates[sensor_id] = emission.rates_per_second(ontology)
        self._members = {
            channel: tuple(ids) for channel, ids in sorted(members.items())
        }
        window = history.config.window_steps * step.total_seconds()
        self._reference = {
            channel: expected_log1p_poisson(
                window * np.sum([self._rates[i] for i in ids], axis=0)
            )
            for channel, ids in self._members.items()
        }
        self._at_before: datetime | None = None
        self._buffer: deque[dict[str, int | None]] = deque(
            maxlen=history.config.window_steps
        )
        self._last: PosteriorDecomposition | None = None

    @property
    def channels(self) -> tuple[str, ...]:
        """Channels with a history state, sorted."""
        return tuple(self._members)

    def reset(self, prior: np.ndarray | None = None) -> None:
        """Return to the initial belief and clear the clock and the history."""
        super().reset(prior)
        self._buffer.clear()
        self._last = None

    def explain(self) -> PosteriorDecomposition | None:
        """The last prediction's decomposition, or ``None`` before any update."""
        return self._last

    def _histories(self) -> dict[str, tuple[int | None, str | None]]:
        """Each channel's history from the buffer, before the current window."""
        # An empty window stands for one that was not a full step: the first
        # update, a gap or an irregular update.
        complete = len(self._buffer) == self.history.config.window_steps and all(
            self._buffer
        )
        result: dict[str, tuple[int | None, str | None]] = {}
        for channel in self._members:
            if not complete:
                result[channel] = (None, COLD_START)
                continue
            counts = [window.get(channel) for window in self._buffer]
            if any(count is None for count in counts):
                result[channel] = (None, UNRELIABLE)
            else:
                result[channel] = (int(sum(counts)), None)  # type: ignore[arg-type]
        return result

    def _record(
        self,
        elapsed: timedelta,
        grouped: Mapping[str, Sequence[Observation]],
        reliabilities: Mapping[str, float] | float | None,
    ) -> None:
        """Push the window just used, or missing windows, into the buffer."""
        if self._at_before is None or elapsed != self.step:
            gaps = 1 if elapsed <= self.step else round(elapsed / self.step)
            for _ in range(min(max(gaps, 1), self.history.config.window_steps)):
                self._buffer.append({})
            return
        window: dict[str, int | None] = {}
        for channel, sensors in self._members.items():
            reliable = all(
                self._weight_for(reliabilities, sensor, 1.0)
                >= self.config.evidence_floor
                for sensor in sensors
            )
            window[channel] = (
                sum(
                    1
                    for sensor in sensors
                    for observation in grouped.get(sensor, ())
                    if observation.value != 0.0
                )
                if reliable
                else None
            )
        self._buffer.append(window)

    def update(
        self,
        now: datetime,
        observations: Sequence[Observation] = (),
        *,
        reliabilities: Mapping[str, float] | float | None = None,
        attribution: Mapping[str, float] | float | None = None,
    ) -> StateEstimate:
        """Advance to *now*, fold in the window's evidence and its history.

        Identical to :meth:`MultimodalBayesFilter.update` except that each
        Poisson event sensor's likelihood is conditioned on its channel's
        history, which is read from the buffer before this window is added.
        """
        moment = require_aware(now, "now")
        if self._at is not None and moment < self._at:
            raise NonMonotonicUpdateError(
                f"update at {moment.isoformat()} precedes the filter clock at "
                f"{self._at.isoformat()}"
            )
        self._at_before = self._at
        elapsed = moment - self._at if self._at is not None else moment - moment
        predicted = self._belief @ self.ontology.transition(elapsed, at=moment)

        grouped: dict[str, list[Observation]] = {}
        for observation in observations:
            if observation.sensor_id in self.emissions:
                grouped.setdefault(observation.sensor_id, []).append(observation)

        histories = self._histories()
        applied = self._at is not None and elapsed == self.step
        prior_transition = np.log(np.maximum(predicted, 1e-300))
        current = np.zeros(self.ontology.size)
        history = np.zeros(self.ontology.size)
        per_channel = {c: np.zeros(self.ontology.size) for c in self._members}
        etas = {
            channel: (
                self.history.modulation(
                    self._room_of[channel], count, self._reference[channel]
                )
                if applied
                else np.zeros(self.ontology.size)
            )
            for channel, (count, _) in histories.items()
        }
        likelihoods: dict[str, np.ndarray] = {}
        weights: dict[str, tuple[float, float]] = {}
        reliability_total = 0.0
        seconds = elapsed.total_seconds()
        for sensor_id, emission in self.emissions.items():
            reliability = self._weight_for(reliabilities, sensor_id, 1.0)
            share = self._weight_for(attribution, sensor_id, 1.0)
            reliability_total += min(max(reliability, 0.0), 1.0)
            weights[sensor_id] = (reliability, share)
            likelihood = emission.log_likelihood(
                self.ontology,
                grouped.get(sensor_id, []),
                elapsed,
                reliability=reliability,
                attribution=share,
            )
            current = current + likelihood
            channel = self._channel_of.get(sensor_id)
            temper = emission.weight * float(reliability) * float(share)
            if applied and channel is not None and temper > 0.0:
                count = sum(1 for o in grouped.get(sensor_id, ()) if o.value != 0.0)
                term = history_log_likelihood(
                    count, self._rates[sensor_id] * seconds, etas[channel], temper
                )
                history = history + term
                per_channel[channel] = per_channel[channel] + term
                likelihood = likelihood + term
            likelihoods[sensor_id] = likelihood

        log_belief = prior_transition + current + history
        self._belief = np.exp(log_belief - logsumexp(log_belief))
        information_gain = _kl_divergence(self._belief, predicted)
        self._at = moment
        self._record(elapsed, grouped, reliabilities)

        self._last = PosteriorDecomposition(
            at=moment,
            states=tuple(self.ontology.states),
            prior_transition=prior_transition,
            current=current,
            history=history,
            posterior=self._belief.copy(),
            channels=tuple(
                ChannelHistory(
                    channel=channel,
                    count=histories[channel][0],
                    reason=(histories[channel][1] or (None if applied else IRREGULAR)),
                    modulation=tuple(float(v) for v in etas[channel]),
                    contribution=tuple(float(v) for v in per_channel[channel]),
                )
                for channel in self._members
            ),
            history_applied=applied,
        )

        winner = int(np.argmax(self._belief))
        contributions = tuple(
            EvidenceContribution(
                sensor_id=sensor_id,
                modality=self._modality_of(sensor_id),
                support=_support_for(likelihoods[sensor_id], winner),
                reliability=(
                    weights[sensor_id][0]
                    if weights[sensor_id][0] >= self.config.evidence_floor
                    else 0.0
                ),
                attribution=weights[sensor_id][1],
                observations=len(grouped.get(sensor_id, [])),
            )
            for sensor_id in self.emissions
        )
        return StateEstimate(
            at=moment,
            ontology=self.ontology,
            belief=self._belief.copy(),
            evidence=contributions,
            completeness=reliability_total / len(self.emissions),
            min_confidence=self.config.min_confidence,
            min_completeness=self.config.min_completeness,
            information_gain=information_gain,
        )

    def snapshot(self) -> dict[str, object]:
        """Restartable state: the belief, the clock and the history buffer."""
        return {
            **super().snapshot(),
            "history": [dict(window) for window in self._buffer],
            "history_model_sha256": self.history.sha256(),
            "step_seconds": self.step.total_seconds(),
        }

    def restore(self, state: Mapping[str, object]) -> None:
        """Restore state from :meth:`snapshot`, refusing another model's."""
        if state.get("history_model_sha256") != self.history.sha256():
            raise ValueError("snapshot was taken under a different history model")
        if state.get("step_seconds") != self.step.total_seconds():
            raise ValueError("snapshot was taken with a different step")
        windows = state.get("history")
        if (
            not isinstance(windows, list)
            or len(windows) > self.history.config.window_steps
        ):
            raise ValueError("snapshot history must list at most k windows")
        restored: list[dict[str, int | None]] = []
        for window in windows:
            if not isinstance(window, Mapping) or (
                window and set(window) != set(self._members)
            ):
                raise ValueError("snapshot history names different channels")
            restored.append(
                {c: (None if v is None else int(v)) for c, v in window.items()}
            )
        super().restore(state)
        self._buffer.clear()
        self._buffer.extend(restored)
        self._last = None
