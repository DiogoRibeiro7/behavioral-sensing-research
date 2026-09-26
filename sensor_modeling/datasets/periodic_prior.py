"""A hierarchical periodic prior over behavioural states by time of day.

The generative filter's optional circadian term rescales how quickly each state
is left at each hour, but nothing states how probable a state is at a given
hour. This module supplies that statement as an explicit, fitted probability
model, with population structure and shrunk household adaptation.

Model
-----
For local wall-clock hour ``h`` (0 to 23, read as in
:mod:`~sensor_modeling.datasets.time_features`), let ``θ(h) = 2π(h + ½)/24`` be
the clock angle of the hour's midpoint, and

.. math::

    \\phi(h) = (1, \\sin\\theta, \\cos\\theta, \\dots, \\sin K\\theta, \\cos K\\theta)

a Fourier basis with ``K`` harmonics. Household ``g``'s prior probability of
state ``s`` at hour ``h`` is a softmax of periodic log-probabilities:

.. math::

    \\eta_{g,s}(h) = (w_s + d_{g,s})^\\top \\phi(h), \\qquad
    \\pi_{g,h}(s) = \\frac{\\exp \\eta_{g,s}(h)}{\\sum_{s'} \\exp \\eta_{g,s'}(h)}.

``w`` is the population effect and ``d_g`` the household's deviation. Each row
of ``w`` is interpretable: its intercept sets how common the state is, and each
sine-cosine pair adds one daily cycle with its own amplitude and peak hour.

Estimation
----------
``n_g(h, s)`` is the labelled time, in hours, that household ``g`` spent in
state ``s`` during hour ``h``. The population effect is fitted from the
permitted training households only:

.. math::

    \\hat w = \\arg\\max_w \\sum_g \\sum_{h,s} n_g(h,s) \\log \\pi_h(s; w)
              - \\tfrac{\\lambda_0}{2} \\lVert w \\rVert^2 .

A household's deviation is fitted from that household's own labelled time:

.. math::

    \\hat d_g = \\arg\\max_d \\sum_{h,s} n_g(h,s) \\log \\pi_h(s; \\hat w + d)
                - \\tfrac{\\lambda}{2} \\lVert d \\rVert^2 .

This is the posterior mode under the prior ``d_g ~ N(0, λ⁻¹ I)``, so the
deviation is shrunk toward the population with precision ``λ``. Consequences:

- A household with no labelled time gets ``d = 0``, the population prior.
- A larger ``λ`` pools more; with more labelled time, the household moves
  further from the population.

Both objectives are strictly concave, so the fit is unique. It is computed by
Newton's method from zero, which is deterministic. ``λ₀`` is a small ridge that
makes the population fit unique, since adding the same function of the hour to
every state leaves the softmax unchanged.

In the generative model
-----------------------
:meth:`PeriodicStatePrior.ontology` feeds the prior to the ontology's existing
circadian term, so time of day enters the filter through one mechanism, not
two. With stickiness ``c_s(h) = π_h(s) / π(s)``, where ``π`` is the base
ontology's stationary distribution, the generator at hour ``h`` becomes
``Q_h = diag(1/c(h)) Q``. Two facts follow directly:

- ``(π ⊙ c(h)) Q_h = πQ = 0``, so the chain's equilibrium at hour ``h`` is
  exactly ``π_h``.
- The ``π_h``-weighted mean exit rate equals the base chain's.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

import numpy as np
from scipy.special import logsumexp

from ..states.ontology import DEFAULT_STATES, BehaviouralState, StateOntology
from .casas import CasasRecording, truth_series
from .time_features import HOURS_PER_DAY, MAX_HARMONICS, hour_angle, local_hour

#: Largest number of Newton iterations a fit may take.
_MAX_ITERATIONS = 200


@dataclass(frozen=True)
class PeriodicPriorConfig:
    """Settings of a periodic state prior. None is tuned on data.

    Attributes
    ----------
    harmonics
        Fourier harmonics ``K``, 1 to 11. Two describe a morning and an evening
        structure.
    population_precision
        Ridge ``λ₀`` on the population coefficients, in labelled hours. It only
        makes the fit unique; the population is fitted from thousands of hours.
    household_precision
        Prior precision ``λ`` of a household's deviation, in labelled hours.
        A deviation coefficient informed by ``I`` hours' worth of Fisher
        information is shrunk by roughly ``λ / (λ + I)``. The default, one
        day, pools strongly until a household has several days of labels.
    """

    harmonics: int = 2
    population_precision: float = 1.0
    household_precision: float = 24.0

    def __post_init__(self) -> None:
        """Validate the settings."""
        if (
            isinstance(self.harmonics, bool)
            or not isinstance(self.harmonics, int)
            or not 1 <= self.harmonics <= MAX_HARMONICS
        ):
            raise ValueError(f"harmonics must be an integer in 1-{MAX_HARMONICS}")
        for name in ("population_precision", "household_precision"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serialisable form."""
        return {
            "harmonics": self.harmonics,
            "population_precision": float(self.population_precision),
            "household_precision": float(self.household_precision),
            "basis": "1, sin(k theta), cos(k theta); theta = 2 pi (hour + 1/2) / 24",
            "unit": "labelled hours",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PeriodicPriorConfig:
        """Rebuild settings from :meth:`to_dict`."""
        return cls(
            harmonics=int(payload["harmonics"]),  # type: ignore[call-overload]
            population_precision=float(payload["population_precision"]),  # type: ignore[arg-type]
            household_precision=float(payload["household_precision"]),  # type: ignore[arg-type]
        )


def angle_basis(angles: np.ndarray | Sequence[float], harmonics: int) -> np.ndarray:
    """``(n, 2K + 1)`` Fourier basis ``(1, sin kθ, cos kθ, ...)`` at *angles*."""
    theta = np.asarray(angles, dtype=float).reshape(-1)
    k = np.arange(1, harmonics + 1)
    basis = np.empty((theta.size, 1 + 2 * harmonics))
    basis[:, 0] = 1.0
    basis[:, 1::2] = np.sin(theta[:, None] * k[None, :])
    basis[:, 2::2] = np.cos(theta[:, None] * k[None, :])
    return basis


def hour_basis(hours: np.ndarray | Sequence[float], harmonics: int) -> np.ndarray:
    """The Fourier basis at the midpoints of local *hours*."""
    return angle_basis(hour_angle(np.asarray(hours, dtype=float)), harmonics)


def _objective(
    deviation: np.ndarray,
    base: np.ndarray,
    counts: np.ndarray,
    basis: np.ndarray,
    precision: float,
) -> float:
    logits = basis @ (base + deviation).T
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    return float(-(counts * logp).sum() + 0.5 * precision * (deviation**2).sum())


def _map_deviation(
    counts: np.ndarray, base: np.ndarray, precision: float, harmonics: int
) -> np.ndarray:
    """The deviation from *base* maximising the penalised log-likelihood.

    Newton's method with a backtracking line search, from zero. The objective
    is strictly concave in the deviation, so the result is unique.
    """
    states, size = base.shape
    if counts.sum() <= 0.0:
        return np.zeros_like(base)
    basis = hour_basis(np.arange(HOURS_PER_DAY), harmonics)
    totals = counts.sum(axis=1)
    deviation = np.zeros_like(base)
    outer = np.einsum("hp,hq->hpq", basis, basis)
    for _ in range(_MAX_ITERATIONS):
        logits = basis @ (base + deviation).T
        probabilities = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        gradient = (
            probabilities * totals[:, None] - counts
        ).T @ basis + precision * deviation
        hessian = precision * np.eye(states * size)
        for hour in np.flatnonzero(totals > 0.0):
            p = probabilities[hour]
            hessian += totals[hour] * np.kron(np.diag(p) - np.outer(p, p), outer[hour])
        step = np.linalg.solve(hessian, gradient.reshape(-1)).reshape(states, size)
        current = _objective(deviation, base, counts, basis, precision)
        # The Newton decrement: the improvement the quadratic model predicts.
        # Once it is below rounding error of the objective, the full step is
        # exact to working precision and a line search can no longer tell.
        decrease = float(gradient.reshape(-1) @ step.reshape(-1))
        if decrease <= 1e-12 * (1.0 + abs(current)):
            return deviation - step
        length = 1.0
        while (
            _objective(deviation - length * step, base, counts, basis, precision)
            > current - 1e-4 * length * decrease
            and length > 1e-12
        ):
            length *= 0.5
        deviation = deviation - length * step
    raise RuntimeError("the periodic prior fit did not converge")


def _frozen(array: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    values = np.array(array, dtype=float)
    values.setflags(write=False)
    return values


@dataclass(frozen=True, eq=False)
class PeriodicStatePrior:
    """A fitted hierarchical periodic prior over states.

    Attributes
    ----------
    states
        States in the order of the coefficient rows.
    config
        The settings it was fitted with.
    population
        ``(states, 2K + 1)`` population coefficients ``w``.
    deviations
        Household deviations ``d_g``, by household. A household not listed is
        unseen and gets the population prior.
    fitted_on
        Households whose labels fitted the population, for provenance.
    labelled_hours
        Labelled hours behind the population fit.
    """

    states: tuple[BehaviouralState, ...]
    config: PeriodicPriorConfig
    population: np.ndarray
    deviations: Mapping[str, np.ndarray] = field(default_factory=dict)
    fitted_on: tuple[str, ...] = ()
    labelled_hours: float = 0.0

    def __post_init__(self) -> None:
        """Check shapes and freeze copies of every coefficient array."""
        states = tuple(BehaviouralState(s) for s in self.states)
        if not states or len(set(states)) != len(states):
            raise ValueError("a periodic prior needs distinct states")
        shape = (len(states), 1 + 2 * self.config.harmonics)
        population = _frozen(self.population)
        if population.shape != shape or not np.all(np.isfinite(population)):
            raise ValueError(f"population coefficients must be finite, shape {shape}")
        deviations: dict[str, np.ndarray] = {}
        for household in sorted(self.deviations):
            deviation = _frozen(self.deviations[household])
            if deviation.shape != shape or not np.all(np.isfinite(deviation)):
                raise ValueError(
                    f"deviation of {household!r} must be finite, shape {shape}"
                )
            deviations[household] = deviation
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "population", population)
        object.__setattr__(self, "deviations", deviations)
        object.__setattr__(self, "fitted_on", tuple(sorted(self.fitted_on)))

    def coefficients(self, household: str | None = None) -> np.ndarray:
        """``w + d_g``, or ``w`` for the population or an unseen household."""
        deviation = self.deviations.get(household) if household is not None else None
        if deviation is None:
            return np.array(self.population)
        return np.array(self.population + deviation)

    def log_probabilities_at(
        self, angles: np.ndarray | Sequence[float], household: str | None = None
    ) -> np.ndarray:
        """``(n, states)`` log-probabilities at clock *angles*, in radians.

        This is the continuous periodic function whose values at the hour
        midpoints are the prior. It has period ``2π``.
        """
        logits = (
            angle_basis(angles, self.config.harmonics) @ self.coefficients(household).T
        )
        log_probabilities: np.ndarray = logits - logsumexp(
            logits, axis=1, keepdims=True
        )
        return log_probabilities

    def log_probabilities(
        self, hours: np.ndarray | Sequence[float], household: str | None = None
    ) -> np.ndarray:
        """``(n, states)`` log-probabilities at local *hours*."""
        return self.log_probabilities_at(
            hour_angle(np.asarray(hours, dtype=float)), household
        )

    def probabilities(
        self, hours: np.ndarray | Sequence[float], household: str | None = None
    ) -> np.ndarray:
        """``(n, states)`` prior probabilities at local *hours*."""
        probabilities: np.ndarray = np.exp(self.log_probabilities(hours, household))
        return probabilities

    def hourly(self, household: str | None = None) -> np.ndarray:
        """``(24, states)`` prior probabilities, one row per local hour."""
        return self.probabilities(np.arange(HOURS_PER_DAY), household)

    def adapt(self, household: str, counts: np.ndarray) -> PeriodicStatePrior:
        """A copy with *household*'s deviation fitted from its own labelled time.

        Parameters
        ----------
        household
            The household the counts belong to.
        counts
            ``(24, states)`` labelled hours by hour and state, from
            :func:`hour_state_counts`. They must come from the household's
            own permitted labels, recorded before any moment it is used for.
            All-zero counts give a zero deviation, the population prior.
        """
        values = _checked_counts(counts, len(self.states))
        deviation = _map_deviation(
            values,
            np.array(self.population),
            self.config.household_precision,
            self.config.harmonics,
        )
        return replace(self, deviations={**self.deviations, household: deviation})

    def stickiness(
        self, base: StateOntology | None = None, household: str | None = None
    ) -> dict[BehaviouralState, tuple[float, ...]]:
        """Circadian multipliers ``π_h(s) / π(s)`` for the base ontology.

        With them, the ontology's equilibrium at hour ``h`` is exactly this
        prior's ``π_h``.
        """
        ontology = base or StateOntology()
        if ontology.circadian is not None:
            raise ValueError("the base ontology already has a circadian profile")
        if tuple(ontology.states) != self.states:
            raise ValueError("the base ontology's states differ from the prior's")
        stationary = ontology.stationary()
        if np.any(stationary <= 0.0):
            raise ValueError("the base ontology's stationary distribution has zeros")
        ratio = self.hourly(household) / stationary[None, :]
        return {
            state: tuple(float(v) for v in ratio[:, index])
            for index, state in enumerate(self.states)
        }

    def ontology(
        self, base: StateOntology | None = None, household: str | None = None
    ) -> StateOntology:
        """The base ontology with this prior as its circadian term."""
        ontology = base or StateOntology()
        return replace(ontology, circadian=self.stickiness(ontology, household))

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serialisable form."""
        return {
            "model": "periodic state prior",
            "states": [state.value for state in self.states],
            "config": self.config.to_dict(),
            "population": self.population.tolist(),
            "deviations": {h: d.tolist() for h, d in self.deviations.items()},
            "fitted_on": list(self.fitted_on),
            "labelled_hours": float(self.labelled_hours),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PeriodicStatePrior:
        """Rebuild a prior from :meth:`to_dict`."""
        deviations = payload.get("deviations") or {}
        return cls(
            states=tuple(BehaviouralState(s) for s in payload["states"]),  # type: ignore[attr-defined]
            config=PeriodicPriorConfig.from_dict(payload["config"]),  # type: ignore[arg-type]
            population=np.array(payload["population"], dtype=float),
            deviations={
                str(h): np.array(d, dtype=float)
                for h, d in deviations.items()  # type: ignore[attr-defined]
            },
            fitted_on=tuple(payload.get("fitted_on") or ()),  # type: ignore[arg-type]
            labelled_hours=float(payload.get("labelled_hours") or 0.0),  # type: ignore[arg-type]
        )

    def sha256(self) -> str:
        """SHA-256 of the canonical serialised prior."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def _checked_counts(counts: np.ndarray, states: int) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    if values.shape != (HOURS_PER_DAY, states):
        raise ValueError(f"counts must have shape {(HOURS_PER_DAY, states)}")
    if not np.all(np.isfinite(values)) or (values < 0.0).any():
        raise ValueError("counts must be finite and non-negative")
    return values


def fit_periodic_prior(
    counts: Mapping[str, np.ndarray],
    *,
    states: Sequence[BehaviouralState] = DEFAULT_STATES,
    config: PeriodicPriorConfig | None = None,
) -> PeriodicStatePrior:
    """Fit the population prior from permitted households' labelled time.

    Parameters
    ----------
    counts
        ``(24, states)`` labelled hours for each training household, from
        :func:`hour_state_counts`. Only households whose labels the
        evaluation permits for fitting may be passed; held-out households
        never are.
    states
        The label space, in the order of the counts' columns.
    config
        Settings; defaults to :class:`PeriodicPriorConfig`.

    The fit pools the households' labelled time, so each labelled hour counts
    once. No household deviation is fitted; see :meth:`PeriodicStatePrior.adapt`.
    """
    settings = config or PeriodicPriorConfig()
    order = tuple(states)
    if not counts:
        raise ValueError("at least one household is required")
    pooled = sum(
        (_checked_counts(counts[h], len(order)) for h in sorted(counts)),
        np.zeros((HOURS_PER_DAY, len(order))),
    )
    if pooled.sum() <= 0.0:
        raise ValueError("the households have no labelled time")
    size = 1 + 2 * settings.harmonics
    population = _map_deviation(
        pooled,
        np.zeros((len(order), size)),
        settings.population_precision,
        settings.harmonics,
    )
    return PeriodicStatePrior(
        states=order,
        config=settings,
        population=population,
        fitted_on=tuple(counts),
        labelled_hours=float(pooled.sum()),
    )


def hour_state_counts(
    recording: CasasRecording,
    moments: Sequence[datetime],
    *,
    states: Sequence[BehaviouralState] = DEFAULT_STATES,
    step: timedelta,
    until: datetime | None = None,
) -> np.ndarray:
    """Labelled hours by local hour and state, counted at *moments*.

    Each labelled moment counts *step* of time in its state and in its local
    hour, read with :func:`~sensor_modeling.datasets.time_features.local_hour`
    in the recording's zone. That is the hour an information set declares for
    the same moment, including across daylight-saving changes. With *until*,
    only moments at or before it count, so an adapted prior can use only
    labels recorded before it is used.
    """
    if not recording.observations:
        raise ValueError("recording contains no observations")
    zone = recording.observations[0].timestamp.tzinfo
    if zone is None:  # pragma: no cover - observations are always aware
        raise ValueError("observation timestamps must be timezone-aware")
    order = {state: index for index, state in enumerate(states)}
    counts = np.zeros((HOURS_PER_DAY, len(order)))
    unit = step / timedelta(hours=1)
    for moment, label in zip(
        moments, truth_series(recording.activities, list(moments))
    ):
        if label is None or label not in order:
            continue
        if until is not None and moment > until:
            continue
        counts[local_hour(moment, zone), order[label]] += unit
    return counts
