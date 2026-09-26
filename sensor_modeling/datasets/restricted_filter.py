"""The generative filter's model, restricted to what an information set declares.

The online filter is recursive. Its belief at ``t`` conditions on every earlier
window, so the filter itself cannot be placed inside a declared information
set. Its generative model can. Under the filter's own prior, transition and
emission models, the posterior given only the windows a set declares is

.. math::

    p(s_t \\mid x_{t-k}, \\dots, x_t) \\propto
    \\Big[\\big((\\pi \\odot \\ell_{t-k})\\, T \\odot \\ell_{t-k+1}\\big)\\, T
    \\cdots\\Big] \\odot \\ell_t,

where :math:`\\pi` is the ontology's stationary distribution, the filter's own
belief before any evidence, :math:`T` the transition over one step, and
:math:`\\ell_j` the likelihood of window ``j``. That is what the production
:class:`~sensor_modeling.fusion.MultimodalBayesFilter` reports when it starts
one step before the first declared window and is fed those windows alone. The
tests check this equality. ``k`` is 0 for current evidence and
``history_steps`` with recent history.

What the restricted model receives
----------------------------------
- The per-channel activation counts of the information set's table, the same
  numbers every other model receives.
- The household's sensor registry, through the filter's default emission
  models: which channels are instrumented and how many sensors pool into each.
  The filter always has this deployment metadata. The discriminative models see
  only whether a channel is instrumented.
- Full reliability and attribution for every sensor. The production pipeline
  derives both from the whole observation history, which the set does not
  include.

The Poisson emission's log-likelihood is linear in a window's activation count,
and sensors on one channel share one rate, so the pooled channel count gives
exactly the likelihood of the separate sensor counts. The coefficients are read
from the filter's own emission models, not re-derived, and a registry where
pooling would not be exact is refused.

Where it cannot go
------------------
The current generative model has no time-of-day input, so sets with time of day
are unsupported (:func:`unsupported_reason`). Its default ontology is
time-homogeneous. Its optional circadian term rescales transition rates by
hour, which acts only through the recursion and does not state how probable a
state is at an hour. The one fitted profile, v0.3, was fitted on every
development home, so it has no held-out score on that panel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from ..fusion.defaults import default_emissions
from ..fusion.emissions import EmissionModel, PoissonEventEmission
from ..observations import Observation, ObservationKind, SensorRegistry
from ..online.pipeline import BehaviouralSensingPipeline, PipelineConfig, scoring_steps
from ..states.ontology import BehaviouralState, StateOntology
from .casas import CasasRecording
from .information_sets import (
    EvidenceChannel,
    EvidenceResolution,
    FeatureTable,
    InformationComponent,
    InformationSet,
    _route_sensors,
    evidence_column,
)

#: Components the current generative model can consume exactly.
SUPPORTED_COMPONENTS = frozenset(
    {InformationComponent.CURRENT_EVIDENCE, InformationComponent.RECENT_HISTORY}
)

#: Why a set with time of day has no matched generative model.
TIME_OF_DAY_UNSUPPORTED = (
    "the current generative model has no time-of-day input: its default "
    "ontology is time-homogeneous, and its optional circadian term rescales "
    "transition rates, so it acts only through unbounded recursion, not as a "
    "statement about the state at an hour; the one fitted profile (v0.3) was "
    "fitted on every development home and has no held-out score there"
)

_PROBE_TIME = datetime(2000, 1, 1, tzinfo=timezone.utc)


def unsupported_reason(information_set: InformationSet) -> str | None:
    """Why the generative model cannot consume *information_set*, or ``None``."""
    components = information_set.components
    if InformationComponent.CURRENT_EVIDENCE not in components:
        return "the generative model conditions on the current window by construction"
    if InformationComponent.TIME_OF_DAY in components:
        return TIME_OF_DAY_UNSUPPORTED
    extra = sorted(c.value for c in components - SUPPORTED_COMPONENTS)
    if extra:
        return f"the generative model has no input for {', '.join(extra)}"
    return None


@dataclass(frozen=True)
class ChannelLikelihood:
    """The filter's log-likelihood terms for one channel's step window.

    Attributes
    ----------
    channel
        The evidence channel.
    sensors
        Registered sensors pooled into it, sorted.
    per_activation
        Log-likelihood added per activation, one value per ontology state.
    per_window
        Log-likelihood of a window with no activation, summed over the
        channel's sensors, one value per ontology state.

    Both are defined up to a constant shared by every state, which a posterior
    does not depend on.
    """

    channel: EvidenceChannel
    sensors: tuple[str, ...]
    per_activation: np.ndarray
    per_window: np.ndarray


def _probe(
    emission: EmissionModel,
    ontology: StateOntology,
    spec_kind: ObservationKind | None,
    modality: object,
    activations: int,
    step: timedelta,
) -> np.ndarray:
    """The emission's own log-likelihood of *activations* in one step window."""
    observations = [
        Observation(
            _PROBE_TIME,
            emission.sensor_id,
            modality,  # type: ignore[arg-type]
            spec_kind or ObservationKind.EVENT,
            1.0,
        )
        for _ in range(activations)
    ]
    return emission.log_likelihood(ontology, observations, step)


def _centred(values: np.ndarray) -> np.ndarray:
    centred: np.ndarray = values - values.mean()
    return centred


def channel_likelihoods(
    registry: SensorRegistry,
    resolution: EvidenceResolution,
    ontology: StateOntology | None = None,
) -> tuple[dict[EvidenceChannel, ChannelLikelihood], tuple[str, ...]]:
    """Per-channel likelihood terms of the filter's default emission models.

    Returns
    -------
    tuple
        The terms of every instrumented declared channel, and the sensors whose
        emission models lie outside the declared channels. The restricted model
        ignores those sensors, because the information set carries none of
        their readings.

    Raises
    ------
    ValueError
        If a channel's sensors are not Poisson event models sharing one rate,
        so that the pooled count would not determine their likelihood.
    """
    ontology = ontology or StateOntology()
    step = resolution.step
    routed, _ = _route_sensors(registry, resolution.channels)
    emissions = {
        model.sensor_id: model for model in default_emissions(registry, ontology)
    }
    ignored = tuple(sorted(set(emissions) - set(routed)))

    members: dict[EvidenceChannel, list[str]] = {}
    for sensor_id, channel in routed.items():
        members.setdefault(channel, []).append(sensor_id)

    terms: dict[EvidenceChannel, ChannelLikelihood] = {}
    for channel in sorted(members):
        sensors = tuple(sorted(members[channel]))
        increments: list[np.ndarray] = []
        silence = np.zeros(ontology.size)
        for sensor_id in sensors:
            emission = emissions.get(sensor_id)
            spec = registry.get(sensor_id)
            if not isinstance(emission, PoissonEventEmission) or spec is None:
                raise ValueError(
                    f"sensor {sensor_id!r} on channel {channel.name!r} has no "
                    "Poisson event emission, so its likelihood is not a "
                    "function of the channel's activation count"
                )
            levels = [
                _probe(emission, ontology, spec.kind, spec.modality, n, step)
                for n in range(3)
            ]
            first, second = levels[1] - levels[0], levels[2] - levels[1]
            if not np.allclose(_centred(first), _centred(second), atol=1e-9):
                raise ValueError(
                    f"the emission of {sensor_id!r} is not linear in its count"
                )
            increments.append(_centred(first))
            silence = silence + levels[0]
        if any(not np.allclose(i, increments[0], atol=1e-9) for i in increments):
            raise ValueError(
                f"channel {channel.name!r} pools sensors with different rates, "
                "so its pooled count does not determine their likelihood"
            )
        terms[channel] = ChannelLikelihood(
            channel, sensors, increments[0], _centred(silence)
        )
    return terms, ignored


def restricted_posteriors(
    table: FeatureTable,
    likelihoods: Mapping[EvidenceChannel, ChannelLikelihood],
    ontology: StateOntology | None = None,
) -> np.ndarray:
    """The generative model's posterior at each row, given only the row's windows.

    Parameters
    ----------
    table
        One household's features under a supported information set.
    likelihoods
        :func:`channel_likelihoods` for the same household.
    ontology
        The time-homogeneous ontology whose prior and transitions are used.
        Defaults to the filter's default.

    Returns
    -------
    numpy.ndarray
        ``(rows, states)`` posteriors in the ontology's state order.

    A window that closes before the recording starts carries no evidence and is
    skipped. The belief there is the stationary prior, which the transition
    leaves unchanged.
    """
    information_set = table.information_set
    reason = unsupported_reason(information_set)
    if reason is not None:
        raise ValueError(f"information set {information_set.name!r}: {reason}")
    ontology = ontology or StateOntology()
    if ontology.circadian is not None:
        raise ValueError(TIME_OF_DAY_UNSUPPORTED)
    resolution = information_set.resolution
    instrumented = set(resolution.channels) - set(table.uninstrumented)
    if set(likelihoods) != instrumented:
        raise ValueError(
            "likelihoods must cover exactly the table's instrumented channels"
        )

    depth = (
        resolution.history_steps
        if InformationComponent.RECENT_HISTORY in information_set.components
        else 0
    )
    transition = ontology.transition(resolution.step)
    belief = np.tile(ontology.stationary(), (len(table.moments), 1))
    for lag in range(depth, -1, -1):
        belief = belief @ transition
        loglik = np.zeros_like(belief)
        for channel, terms in likelihoods.items():
            counts = table.column(evidence_column(channel.name, lag))
            seen = ~np.isnan(counts)
            loglik[seen] += (
                np.outer(counts[seen], terms.per_activation) + terms.per_window
            )
        loglik -= loglik.max(axis=1, keepdims=True)
        belief = belief * np.exp(loglik)
        belief /= belief.sum(axis=1, keepdims=True)
    return belief


def filter_posteriors(
    recording: CasasRecording,
    moments: Sequence[datetime],
    *,
    step: timedelta,
) -> tuple[np.ndarray, tuple[BehaviouralState, ...]]:
    """The production pipeline's posterior at each moment, unrestricted.

    The pipeline runs with its default configuration at *step*, over the
    whole recording, exactly as ``evaluate_recording`` runs it. Its belief
    conditions on every earlier window and on its health and attribution
    layers, so it belongs to no declared information set.

    Returns
    -------
    tuple
        ``(moments, states)`` posteriors and the state order of their columns.

    Raises
    ------
    ValueError
        If the pipeline produced no scoring step at one of *moments*.
    """
    if not recording.observations:
        raise ValueError("recording contains no observations")
    zone = recording.observations[0].timestamp.tzinfo
    pipeline = BehaviouralSensingPipeline(
        recording.registry, config=PipelineConfig(tz=zone, step=step)
    )
    steps = pipeline.run(recording.observations)
    steps.extend(pipeline.close(recording.observations[-1].timestamp))
    beliefs = {s.at: s.state.belief for s in scoring_steps(steps)}
    missing = [moment for moment in moments if moment not in beliefs]
    if missing:
        raise ValueError(
            f"the pipeline produced no step at {len(missing)} moments, "
            f"first {missing[0].isoformat()}"
        )
    stacked = (
        np.vstack([beliefs[moment] for moment in moments])
        if moments
        else (np.empty((0, pipeline.ontology.size)))
    )
    return stacked, tuple(pipeline.ontology.states)
