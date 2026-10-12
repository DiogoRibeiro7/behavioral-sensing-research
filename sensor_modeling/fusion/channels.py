"""Emission models of channels: every sensor of one room and modality, together.

The default emission models read one sensor each, and the filter adds their
log-likelihoods. A channel pools the sensors of one room and modality and
counts their activations in a window. For a Poisson model the two agree,
because a Poisson log-likelihood is linear in the count. A hurdle model does
not factor that way: whether a channel was silent in a window depends on all
of its sensors at once. Its emission model therefore reads the records of
every sensor it pools, which it lists in :attr:`EmissionModel.sources`, and
the filter gives it each of those records.

:class:`HurdleChannelEmission` is the hurdle model of
:mod:`sensor_modeling.datasets.channel_models`, with parameters given rather
than fitted. It is defined for one window length, the one it was fitted at.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from ..observations.observation import Observation
from ..observations.types import Modality
from ..states.ontology import BehaviouralState, StateOntology
from .emissions import EmissionModel


def _log_expm1(values: np.ndarray) -> np.ndarray:
    """``log(e^x − 1)``, accurate for small and large ``x``."""
    values = np.asarray(values, dtype=float)
    large = values > 30.0
    out = np.empty_like(values)
    out[large] = values[large] + np.log1p(-np.exp(-values[large]))
    out[~large] = np.log(np.expm1(values[~large]))
    return out


@dataclass
class HurdleChannelEmission(EmissionModel):
    """A channel's hurdle model: silence, then how much activity.

    In one window, the channel's pooled count ``n`` has log-likelihood
    ``log π(s)`` when ``n = 0``, and otherwise
    ``log(1 − π(s)) − log(e^{μ(s)} − 1) + n log μ(s)``, up to ``log n!``: the
    probability of any activity, then a zero-truncated Poisson count. A record
    counts as an activation when its value is not zero, as for
    :class:`~sensor_modeling.fusion.emissions.PoissonEventEmission`.

    Parameters
    ----------
    sensor_id
        The model's key in the filter. It names the channel and must not be
        the identifier of a registered sensor.
    sensors
        The registered sensors pooled into the channel.
    silence
        ``π(s)``, the probability of a silent window, for every state.
    rate
        ``μ(s)``, the zero-truncated Poisson rate of an active window's count,
        for every state.
    window
        The window the parameters describe. An update must advance exactly one
        window, or no time with no records.
    modality
        The channel's modality, reported with its evidence.
    """

    sensors: tuple[str, ...] = ()
    silence: Mapping[BehaviouralState, float] = field(default_factory=dict)
    rate: Mapping[BehaviouralState, float] = field(default_factory=dict)
    window: timedelta = timedelta(minutes=5)
    modality: Modality | None = None

    def __post_init__(self) -> None:
        """Validate the channel and its parameters."""
        super().__post_init__()
        self.sensors = tuple(self.sensors)
        if not self.sensors or len(set(self.sensors)) != len(self.sensors):
            raise ValueError("a channel needs distinct sensors")
        if self.sensor_id in self.sensors:
            raise ValueError("a channel's key must not be one of its sensors")
        if set(self.silence) != set(self.rate):
            raise ValueError("silence and rate must name the same states")
        for state in self.silence:
            silence, rate = float(self.silence[state]), float(self.rate[state])
            if not 0.0 < silence < 1.0:
                raise ValueError(f"silence for {state} must lie strictly in (0, 1)")
            if not math.isfinite(rate) or rate <= 0.0:
                raise ValueError(f"rate for {state} must be positive and finite")
        if self.window <= timedelta(0):
            raise ValueError("window must be positive")

    @property
    def sources(self) -> tuple[str, ...]:
        """Every sensor pooled into the channel."""
        return self.sensors

    def _vector(
        self, ontology: StateOntology, values: Mapping[BehaviouralState, float]
    ) -> np.ndarray:
        missing = [state.value for state in ontology.states if state not in values]
        if missing:
            raise ValueError(
                f"channel '{self.sensor_id}' has no parameter for {missing}"
            )
        return np.array([float(values[state]) for state in ontology.states])

    def _raw_log_likelihood(
        self,
        ontology: StateOntology,
        observations: Sequence[Observation],
        elapsed: timedelta,
    ) -> np.ndarray:
        if elapsed == timedelta(0) and not observations:
            return np.zeros(ontology.size)
        if elapsed != self.window:
            raise ValueError(
                f"channel '{self.sensor_id}' is defined per window of "
                f"{self.window.total_seconds():g} s; an update advanced "
                f"{elapsed.total_seconds():g} s"
            )
        silence = self._vector(ontology, self.silence)
        count = sum(1 for observation in observations if observation.value != 0.0)
        if count == 0:
            silent: np.ndarray = np.log(silence)
            return silent
        rate = self._vector(ontology, self.rate)
        active: np.ndarray = (
            np.log1p(-silence) - _log_expm1(rate) + count * np.log(rate)
        )
        return active
