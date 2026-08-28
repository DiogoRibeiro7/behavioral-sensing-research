"""The configurable ontology of latent behavioural states.

The states here are deliberately weaker than the activities of daily living a
clinician would name. ``KITCHEN_ACTIVITY`` says the resident appears to be
active in the kitchen; it does not say they ate. Claiming food intake needs
evidence that a contact sensor cannot supply, so the ontology stops where the
evidence stops and leaves the stronger claim to be made -- or not -- further
downstream.

Transitions are modelled in continuous time. Ambient observations arrive
asynchronously and irregularly, so the transition operator has to be defined
for an arbitrary elapsed interval rather than for a fixed time step. A
continuous-time Markov chain gives exactly that: the generator encodes how
long a state typically persists, and the transition matrix over any interval
follows from its matrix exponential.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from functools import lru_cache

import numpy as np
from scipy.linalg import expm


class BehaviouralState(str, Enum):
    """A latent behavioural state of the monitored resident.

    ``UNKNOWN`` is not a latent state the chain can occupy. It is the value an
    estimator returns when it declines to commit, and is excluded from the
    ontology's state vector.
    """

    AWAY = "away"
    """Not in the home."""

    HOME_ACTIVE = "home_active"
    """At home and moving about, without a more specific location."""

    HOME_INACTIVE = "home_inactive"
    """At home, awake, and largely stationary."""

    SLEEPING = "sleeping"
    """In bed with sustained low movement."""

    BED_AWAKE = "bed_awake"
    """In bed but moving; distinguishable from sleep only with bed sensing."""

    BATHROOM_ACTIVITY = "bathroom_activity"
    """Active in the bathroom. Not a claim about toileting."""

    KITCHEN_ACTIVITY = "kitchen_activity"
    """Active in the kitchen. Not a claim about eating or drinking."""

    UNKNOWN = "unknown"
    """Insufficient evidence to commit to any state."""


#: The states a resident can actually occupy, in canonical vector order.
DEFAULT_STATES: tuple[BehaviouralState, ...] = (
    BehaviouralState.AWAY,
    BehaviouralState.HOME_ACTIVE,
    BehaviouralState.HOME_INACTIVE,
    BehaviouralState.SLEEPING,
    BehaviouralState.BED_AWAKE,
    BehaviouralState.BATHROOM_ACTIVITY,
    BehaviouralState.KITCHEN_ACTIVITY,
)

#: Typical persistence of each state, used to build the generator matrix.
DEFAULT_DWELL: dict[BehaviouralState, timedelta] = {
    BehaviouralState.AWAY: timedelta(hours=3),
    BehaviouralState.HOME_ACTIVE: timedelta(minutes=20),
    BehaviouralState.HOME_INACTIVE: timedelta(hours=1),
    BehaviouralState.SLEEPING: timedelta(hours=3),
    BehaviouralState.BED_AWAKE: timedelta(minutes=20),
    BehaviouralState.BATHROOM_ACTIVITY: timedelta(minutes=6),
    BehaviouralState.KITCHEN_ACTIVITY: timedelta(minutes=15),
}

#: Room each state implies, where it implies one. States without a room make
#: no spatial claim and receive no room-specific evidence.
DEFAULT_ROOMS: dict[BehaviouralState, str | None] = {
    BehaviouralState.AWAY: None,
    BehaviouralState.HOME_ACTIVE: None,
    BehaviouralState.HOME_INACTIVE: None,
    BehaviouralState.SLEEPING: "bedroom",
    BehaviouralState.BED_AWAKE: "bedroom",
    BehaviouralState.BATHROOM_ACTIVITY: "bathroom",
    BehaviouralState.KITCHEN_ACTIVITY: "kitchen",
}

#: Plausible state-to-state moves. Reaching or leaving the home passes through
#: a general at-home state rather than teleporting out of bed to the street.
DEFAULT_JUMPS: dict[BehaviouralState, tuple[BehaviouralState, ...]] = {
    BehaviouralState.AWAY: (BehaviouralState.HOME_ACTIVE,),
    BehaviouralState.HOME_ACTIVE: (
        BehaviouralState.AWAY,
        BehaviouralState.HOME_INACTIVE,
        BehaviouralState.BED_AWAKE,
        BehaviouralState.BATHROOM_ACTIVITY,
        BehaviouralState.KITCHEN_ACTIVITY,
    ),
    BehaviouralState.HOME_INACTIVE: (
        BehaviouralState.HOME_ACTIVE,
        BehaviouralState.BED_AWAKE,
        BehaviouralState.BATHROOM_ACTIVITY,
        BehaviouralState.KITCHEN_ACTIVITY,
    ),
    BehaviouralState.SLEEPING: (BehaviouralState.BED_AWAKE,),
    BehaviouralState.BED_AWAKE: (
        BehaviouralState.SLEEPING,
        BehaviouralState.HOME_ACTIVE,
        BehaviouralState.BATHROOM_ACTIVITY,
    ),
    BehaviouralState.BATHROOM_ACTIVITY: (
        BehaviouralState.HOME_ACTIVE,
        BehaviouralState.HOME_INACTIVE,
        BehaviouralState.BED_AWAKE,
    ),
    BehaviouralState.KITCHEN_ACTIVITY: (
        BehaviouralState.HOME_ACTIVE,
        BehaviouralState.HOME_INACTIVE,
    ),
}


@dataclass(frozen=True)
class StateOntology:
    """A configurable set of latent states with continuous-time dynamics.

    Parameters
    ----------
    states
        Latent states in canonical vector order. ``UNKNOWN`` is not allowed.
    dwell
        Mean persistence of each state. Longer dwell means the chain is more
        reluctant to leave, which is what supplies temporal smoothing.
    rooms
        Room each state implies, or ``None`` when it makes no spatial claim.
    jumps
        Permitted destinations from each state. Defaults to every other state.
    """

    states: tuple[BehaviouralState, ...] = DEFAULT_STATES
    dwell: Mapping[BehaviouralState, timedelta] = field(
        default_factory=lambda: dict(DEFAULT_DWELL)
    )
    rooms: Mapping[BehaviouralState, str | None] = field(
        default_factory=lambda: dict(DEFAULT_ROOMS)
    )
    jumps: Mapping[BehaviouralState, Sequence[BehaviouralState]] = field(
        default_factory=lambda: dict(DEFAULT_JUMPS)
    )

    def __post_init__(self) -> None:
        """Validate the ontology and precompute its generator."""
        if len(self.states) < 2:
            raise ValueError("an ontology needs at least two states")
        if len(set(self.states)) != len(self.states):
            raise ValueError("states must be unique")
        if BehaviouralState.UNKNOWN in self.states:
            raise ValueError(
                "UNKNOWN is an estimator abstention, not an occupiable state"
            )
        for state in self.states:
            duration = self.dwell.get(state)
            if duration is None:
                raise ValueError(f"no mean dwell time declared for {state.value}")
            if duration <= timedelta(0):
                raise ValueError(f"mean dwell time for {state.value} must be positive")
        object.__setattr__(self, "_generator", self._build_generator())

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        """Number of latent states."""
        return len(self.states)

    def index(self, state: BehaviouralState) -> int:
        """Return the vector position of *state*."""
        return self.states.index(state)

    def room_of(self, state: BehaviouralState) -> str | None:
        """Return the room *state* implies, if any."""
        return self.rooms.get(state)

    def states_in_room(self, room: str) -> tuple[BehaviouralState, ...]:
        """Return the states that imply presence in *room*."""
        return tuple(s for s in self.states if self.rooms.get(s) == room)

    # ------------------------------------------------------------------
    def _build_generator(self) -> np.ndarray:
        """Build the continuous-time transition rate matrix.

        A state with mean dwell ``d`` leaves at total rate ``1/d``, split
        evenly across its permitted destinations. Rates are expressed per
        second so that any observation interval can be handled directly.
        """
        size = self.size
        generator = np.zeros((size, size), dtype=float)
        for row, state in enumerate(self.states):
            rate = 1.0 / self.dwell[state].total_seconds()
            destinations = [
                self.index(target)
                for target in self.jumps.get(state, self.states)
                if target in self.states and target is not state
            ]
            if not destinations:
                destinations = [i for i in range(size) if i != row]
            share = rate / len(destinations)
            for column in destinations:
                generator[row, column] = share
            generator[row, row] = -rate
        return generator

    @property
    def generator(self) -> np.ndarray:
        """The continuous-time generator matrix, in transitions per second."""
        return np.asarray(object.__getattribute__(self, "_generator"))

    def transition(self, elapsed: timedelta) -> np.ndarray:
        """Return ``P(Z_{t+elapsed} | Z_t)`` as a row-stochastic matrix.

        A non-positive interval yields the identity, so repeated updates at
        the same instant leave the belief untouched.
        """
        seconds = elapsed.total_seconds()
        if seconds <= 0.0:
            return np.eye(self.size)
        return _transition_matrix(
            self.generator.tobytes(), self.size, round(seconds, 3)
        )

    def stationary(self) -> np.ndarray:
        """Return the stationary distribution implied by the generator.

        Used as the default prior: before any evidence arrives, the most
        defensible belief is the long-run behaviour of the declared dynamics.
        """
        size = self.size
        system = np.vstack([self.generator.T, np.ones(size)])
        target = np.zeros(size + 1)
        target[-1] = 1.0
        solution, *_ = np.linalg.lstsq(system, target, rcond=None)
        distribution = np.clip(solution, 0.0, None)
        total = distribution.sum()
        if total <= 0.0:
            return np.full(size, 1.0 / size)
        normalised: np.ndarray = distribution / total
        return normalised

    def uniform(self) -> np.ndarray:
        """Return a uniform belief over the latent states."""
        return np.full(self.size, 1.0 / self.size)

    def labels(self) -> list[str]:
        """Return the state names in vector order."""
        return [state.value for state in self.states]


@lru_cache(maxsize=512)
def _transition_matrix(generator_bytes: bytes, size: int, seconds: float) -> np.ndarray:
    """Return ``expm(Q * seconds)`` with caching on the rounded interval.

    Ambient streams produce the same handful of intervals over and over, so
    caching keeps the matrix exponential off the hot path of online updates.
    """
    generator = np.frombuffer(generator_bytes, dtype=float).reshape(size, size)
    matrix = np.asarray(expm(generator * seconds), dtype=float)
    matrix = np.clip(matrix, 0.0, None)
    row_sums = matrix.sum(axis=1, keepdims=True)
    stochastic: np.ndarray = matrix / np.where(row_sums > 0.0, row_sums, 1.0)
    return stochastic
