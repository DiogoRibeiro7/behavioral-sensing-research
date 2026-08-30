"""Recursive multimodal fusion into a latent behavioural state.

:class:`MultimodalBayesFilter` maintains ``P(Z_t | O_1:t)`` -- the posterior
over latent states given every observation so far, from every modality.

It is a forward filter over a continuous-time Markov chain, which is what lets
it accept the traffic an ambient deployment actually produces: observations
arriving asynchronously, at different rates per modality, with modalities
appearing and disappearing as sensors fail and recover. Between updates the
belief is propagated by the chain's transition operator for exactly the
elapsed interval; at each update every sensor that had something to say
contributes a tempered log-likelihood.

Three properties follow from the construction rather than from special cases:

*A failed sensor is silent, not negative.* Reliability enters as a tempering
exponent, so a sensor with reliability zero contributes a flat likelihood and
leaves the belief entirely to the other modalities.

*Absence of evidence widens the posterior.* With nothing to condition on, the
prediction step relaxes the belief toward the chain's stationary distribution,
confidence falls, and the estimate eventually abstains instead of coasting on
a stale conclusion.

*Contradiction is preserved, not resolved.* Sensors pointing at different
states pull the posterior apart, which shows up as a lower confidence and as
explicitly recorded contradicting evidence, rather than being averaged away.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.special import logsumexp

from ..observations.observation import Observation, require_aware
from ..observations.registry import SensorRegistry
from ..observations.types import Modality
from ..states.ontology import StateOntology
from .emissions import EmissionModel
from .estimate import EvidenceContribution, StateEstimate

logger = logging.getLogger(__name__)


class NonMonotonicUpdateError(ValueError):
    """Raised when a filter update would move backwards in time.

    The filter is causal by construction. Reordering late-arriving records is
    the job of the surrounding pipeline, which owns a buffer and can decide
    how much lateness to tolerate; silently folding a stale record into the
    current belief would corrupt it without any trace.
    """


@dataclass
class FusionConfig:
    """Thresholds governing when the filter declines to name a state.

    Parameters
    ----------
    min_confidence
        Posterior mass the leading state must hold to be reported.
    min_completeness
        Mean sensor reliability required before any posterior is trusted.
    evidence_floor
        Reliability below which a sensor is recorded as supplying nothing.
    """

    min_confidence: float = 0.35
    min_completeness: float = 0.25
    evidence_floor: float = 0.05

    def __post_init__(self) -> None:
        """Validate the abstention thresholds."""
        for name in ("min_confidence", "min_completeness", "evidence_floor"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            setattr(self, name, value)


class MultimodalBayesFilter:
    """Forward filtering of latent behavioural state over asynchronous evidence.

    Parameters
    ----------
    ontology
        Latent states and their continuous-time dynamics.
    emissions
        One observation model per sensor that should inform the state.
    registry
        Sensor declarations, used to label evidence with its modality.
    config
        Abstention thresholds.
    prior
        Initial belief. Defaults to the ontology's stationary distribution,
        which is the most defensible starting point before any evidence.
    """

    def __init__(
        self,
        ontology: StateOntology,
        emissions: Iterable[EmissionModel],
        registry: SensorRegistry | None = None,
        config: FusionConfig | None = None,
        prior: np.ndarray | None = None,
    ) -> None:
        self.ontology = ontology
        self.registry = registry
        self.config = config or FusionConfig()
        self.emissions: dict[str, EmissionModel] = {}
        for emission in emissions:
            if emission.sensor_id in self.emissions:
                raise ValueError(
                    f"duplicate emission model for sensor '{emission.sensor_id}'"
                )
            self.emissions[emission.sensor_id] = emission
        if not self.emissions:
            raise ValueError("at least one emission model is required")

        self._prior: np.ndarray = self._validated_prior(prior)
        self._belief: np.ndarray = self._prior.copy()
        self._at: datetime | None = None

    def _validated_prior(self, prior: np.ndarray | None) -> np.ndarray:
        """Return a normalised prior belief vector."""
        if prior is None:
            return self.ontology.stationary()
        vector = np.asarray(prior, dtype=float)
        if vector.shape != (self.ontology.size,):
            raise ValueError("prior must have one entry per ontology state")
        if not np.all(np.isfinite(vector)) or vector.min() < 0.0 or vector.sum() <= 0.0:
            raise ValueError("prior must be finite, non-negative, and non-zero")
        normalised: np.ndarray = vector / vector.sum()
        return normalised

    # ------------------------------------------------------------------
    @property
    def at(self) -> datetime | None:
        """Time of the most recent update, if the filter has been used."""
        return self._at

    @property
    def belief(self) -> np.ndarray:
        """A copy of the current posterior over latent states."""
        current: np.ndarray = self._belief.copy()
        return current

    def reset(self, prior: np.ndarray | None = None) -> None:
        """Return the filter to its initial belief and clear its clock."""
        self._prior = self._validated_prior(prior)
        self._belief = self._prior.copy()
        self._at = None

    # ------------------------------------------------------------------
    @staticmethod
    def _weight_for(
        weights: Mapping[str, float] | float | None, sensor_id: str, default: float
    ) -> float:
        """Resolve a per-sensor weight from a mapping, scalar, or default."""
        if weights is None:
            return default
        if isinstance(weights, Mapping):
            return float(weights.get(sensor_id, default))
        return float(weights)

    def _modality_of(self, sensor_id: str) -> Modality:
        """Return the modality of *sensor_id*, or ``OTHER`` when unknown."""
        if self.registry is None:
            return Modality.OTHER
        spec = self.registry.get(sensor_id)
        return spec.modality if spec is not None else Modality.OTHER

    def update(
        self,
        now: datetime,
        observations: Sequence[Observation] = (),
        *,
        reliabilities: Mapping[str, float] | float | None = None,
        attribution: Mapping[str, float] | float | None = None,
    ) -> StateEstimate:
        """Advance the filter to *now* and fold in the interval's evidence.

        Parameters
        ----------
        now
            Timezone-aware instant to advance to. Must not precede the
            previous update.
        observations
            Records covering the interval since the previous update. Records
            from sensors without an emission model are ignored.
        reliabilities
            Per-sensor evidence weights from the health monitor. A missing
            entry defaults to full reliability.
        attribution
            Per-sensor probability that the monitored resident generated the
            observation. A missing entry defaults to full attribution.

        Returns
        -------
        StateEstimate
            The posterior with its supporting and contradicting evidence.
        """
        moment = require_aware(now, "now")
        if self._at is not None and moment < self._at:
            raise NonMonotonicUpdateError(
                f"update at {moment.isoformat()} precedes the filter clock at "
                f"{self._at.isoformat()}"
            )

        elapsed = moment - self._at if self._at is not None else moment - moment
        # The moment is passed so a circadian ontology can vary its dynamics by
        # hour. A time-homogeneous one ignores it.
        predicted = self._belief @ self.ontology.transition(elapsed, at=moment)

        grouped: dict[str, list[Observation]] = {}
        for observation in observations:
            if observation.sensor_id in self.emissions:
                grouped.setdefault(observation.sensor_id, []).append(observation)
            else:
                logger.debug(
                    "No emission model for sensor '%s'; observation ignored",
                    observation.sensor_id,
                )

        log_belief = np.log(np.maximum(predicted, 1e-300))
        likelihoods: dict[str, np.ndarray] = {}
        weights: dict[str, tuple[float, float]] = {}
        reliability_total = 0.0

        for sensor_id, emission in self.emissions.items():
            reliability = self._weight_for(reliabilities, sensor_id, 1.0)
            share = self._weight_for(attribution, sensor_id, 1.0)
            reliability_total += min(max(reliability, 0.0), 1.0)
            weights[sensor_id] = (reliability, share)
            likelihoods[sensor_id] = emission.log_likelihood(
                self.ontology,
                grouped.get(sensor_id, []),
                elapsed,
                reliability=reliability,
                attribution=share,
            )
            log_belief = log_belief + likelihoods[sensor_id]

        self._belief = np.exp(log_belief - logsumexp(log_belief))
        self._at = moment

        # Support is measured against the state the posterior actually
        # settled on, so a sensor pointing somewhere else is recorded as
        # contradicting the conclusion rather than as backing its own guess.
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
        )

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, object]:
        """Return restartable filter state."""
        return {
            "belief": self._belief.tolist(),
            "at": self._at.isoformat() if self._at else None,
            "states": self.ontology.labels(),
        }

    def restore(self, state: Mapping[str, object]) -> None:
        """Restore filter state produced by :meth:`snapshot`.

        The stored state labels are checked against the current ontology, so
        a belief saved under a different state set is rejected rather than
        silently reinterpreted position by position.
        """
        labels = state.get("states")
        if labels is not None and list(labels) != self.ontology.labels():  # type: ignore[call-overload]
            raise ValueError("snapshot was taken under a different state ontology")
        belief = np.asarray(state["belief"], dtype=float)
        if belief.shape != (self.ontology.size,):
            raise ValueError("snapshot belief does not match the ontology size")
        total = belief.sum()
        if not np.all(np.isfinite(belief)) or belief.min() < 0.0 or total <= 0.0:
            raise ValueError(
                "snapshot belief must be finite, non-negative, and non-zero"
            )
        self._belief = belief / total
        moment = state.get("at")
        self._at = datetime.fromisoformat(str(moment)) if moment else None


def _support_for(likelihood: np.ndarray, winner: int) -> float:
    """Return how much a sensor favours *winner* over the best alternative.

    Positive means the sensor backs the reported state, negative means it
    points elsewhere, and zero means it is indifferent -- which is what an
    uninformative or fully discounted sensor produces.
    """
    if likelihood.size < 2:
        return 0.0
    others = np.delete(likelihood, winner)
    return float(likelihood[winner] - others.max())
