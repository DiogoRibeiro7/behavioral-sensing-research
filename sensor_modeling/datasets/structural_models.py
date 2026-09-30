"""Model specifications for structural disagreement, built from supported assumptions only.

:mod:`~sensor_modeling.evaluation.disagreement` measures how much several model
specifications disagree. This module says which specifications may take part.
It then produces their posteriors for a household in the filter's recursion
over every window, with the deployed decision rule unchanged.

A specification chooses one variant on each of three axes:

``observation``
    ``hurdle``, the fitted hurdle-Poisson, or ``hurdle_nb``, the fitted hurdle
    negative binomial.
``parameters``
    ``population``, fitted on the fold's training households, or ``pooled``,
    the household partially pooled toward it on its labelled windows before a
    cut-off.
``parameter_sample``
    ``all`` of the fold's training households, or one of two fixed halves of
    them, ``half_a`` and ``half_b``. This is the population's estimation
    uncertainty over training households, which reads no held-out household
    and no future window: allowed, causal parameter uncertainty.

Every variant cites the published evidence that supports it. No variant is an
arbitrary perturbation, and an ensemble is capped at a few specifications.

Two variations are refused, with their reasons:

- the Phase 3.1 time prior as the recursion's transition, which was evaluated
  only on declared information sets, never as a recursion;
- fixed-lag smoothing, which reads evidence after the window.

The declared rates are not a fitted specification, and are not offered.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from ..evaluation.disagreement import (
    DisagreementReport,
    DisagreementTrace,
    compare_posteriors,
)
from ..states.ontology import StateOntology
from .casas import CasasRecording
from .channel_models import (
    HURDLE,
    HURDLE_NB,
    ChannelStatistics,
    FittedChannels,
    combine_statistics,
    filter_recursion,
    household_channel_counts,
    household_statistics,
    pool_channels,
    total_loglik,
)
from .information_sets import EvidenceChannel, EvidenceResolution
from .partial_pooling import PoolingConfig
from .restricted_filter import ChannelModel, channel_likelihoods

#: The most specifications an ensemble may compare.
MAX_SPECIFICATIONS = 6

#: The halves a parameter sample may take, with the training households each keeps.
HALVES = ("half_a", "half_b")


@dataclass(frozen=True)
class Support:
    """The published result an assumption rests on."""

    record: str
    estimand: str
    finding: str

    def to_dict(self) -> dict[str, str]:
        """Return a serialisable form."""
        return {
            "record": self.record,
            "estimand": self.estimand,
            "finding": self.finding,
        }


@dataclass(frozen=True)
class Assumption:
    """One supported variant on one axis of a model specification."""

    axis: str
    variant: str
    description: str
    support: tuple[Support, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form."""
        return {
            "axis": self.axis,
            "variant": self.variant,
            "description": self.description,
            "support": [s.to_dict() for s in self.support],
        }


_FITTED_RATES = "artifacts/phase3/phase3-fitted-rates.json"
_NEGATIVE_BINOMIAL = "artifacts/phase3/phase3-hurdle-negative-binomial.json"
_POOLING = "artifacts/phase3/phase3-partial-pooling.json"

#: Every supported variant, by axis and name.
CATALOGUE: dict[tuple[str, str], Assumption] = {
    ("observation", HURDLE): Assumption(
        "observation",
        HURDLE,
        "fitted hurdle-Poisson channels: a silence probability and a "
        "zero-truncated Poisson active count per channel and state",
        (Support(_FITTED_RATES, "FR", "success"),),
    ),
    ("observation", HURDLE_NB): Assumption(
        "observation",
        HURDLE_NB,
        "fitted hurdle negative-binomial channels: the same silence, and a "
        "zero-truncated negative-binomial active count",
        (Support(_NEGATIVE_BINOMIAL, "NR", "adopt"),),
    ),
    ("parameters", "population"): Assumption(
        "parameters",
        "population",
        "every held-out household gets its fold's population parameters",
        (Support(_FITTED_RATES, "FR", "success"),),
    ),
    ("parameters", "pooled"): Assumption(
        "parameters",
        "pooled",
        "the household's silence and active mean pooled toward the population "
        "on its labelled windows before a cut-off, with the Phase 3.4 strength; "
        "supported with current windows, inconclusive in the recursion",
        (
            Support(_POOLING, "P1", "success"),
            Support(_POOLING, "P2", "inconclusive"),
        ),
    ),
    ("parameter_sample", "all"): Assumption(
        "parameter_sample",
        "all",
        "the population fitted on every training household of the fold",
        (Support(_FITTED_RATES, "FR", "success"),),
    ),
    ("parameter_sample", "half_a"): Assumption(
        "parameter_sample",
        "half_a",
        "the same population fitted on the first half of the fold's training "
        "households, in sorted order, alternating: its estimation uncertainty "
        "over training households",
        (Support(_FITTED_RATES, "FR", "success"),),
    ),
    ("parameter_sample", "half_b"): Assumption(
        "parameter_sample",
        "half_b",
        "the same on the other half",
        (Support(_FITTED_RATES, "FR", "success"),),
    ),
}

#: Variations considered and refused, with the reason.
REFUSED: dict[tuple[str, str], str] = {
    ("observation", "declared"): "the declared rates are not a fitted "
    "specification, and the fitted families improve on them in every measure "
    "the Phase 3.3 follow-up scored",
    ("transition", "periodic"): "the Phase 3.1 time prior was evaluated only on "
    "declared information sets, never as the recursion's transition",
    ("inference", "fixed_lag"): "a fixed-lag smoother reads evidence after the "
    "window, which an online diagnostic may not",
}

AXES = ("observation", "parameters", "parameter_sample")


@dataclass(frozen=True)
class Specification:
    """One model specification: a supported variant on every axis.

    The default is the reference, the deployed recursion: the Phase 3.3
    follow-up's fitted hurdle-Poisson with population parameters from every
    training household.
    """

    observation: str = HURDLE
    parameters: str = "population"
    parameter_sample: str = "all"

    def __post_init__(self) -> None:
        """Refuse any variant the catalogue does not support."""
        for axis in AXES:
            variant = getattr(self, axis)
            if (axis, variant) in REFUSED:
                raise ValueError(
                    f"{axis} {variant!r} is not a supported variant: "
                    f"{REFUSED[(axis, variant)]}"
                )
            if (axis, variant) not in CATALOGUE:
                raise ValueError(f"{axis} {variant!r} is not in the catalogue")

    @property
    def name(self) -> str:
        """``observation/parameters/parameter_sample``."""
        return f"{self.observation}/{self.parameters}/{self.parameter_sample}"

    @property
    def assumptions(self) -> tuple[Assumption, ...]:
        """The catalogue entry of each of its variants."""
        return tuple(CATALOGUE[(axis, getattr(self, axis))] for axis in AXES)

    def differs(self, other: Specification) -> tuple[str, ...]:
        """The axes on which it differs from *other*."""
        return tuple(a for a in AXES if getattr(self, a) != getattr(other, a))

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form, with every assumption's support."""
        return {
            "name": self.name,
            "assumptions": [a.to_dict() for a in self.assumptions],
        }


#: The deployed recursion.
REFERENCE = Specification()

#: One supported alternative on each axis, and the two halves of the sample.
DEFAULT_ENSEMBLE: tuple[Specification, ...] = (
    REFERENCE,
    Specification(observation=HURDLE_NB),
    Specification(parameters="pooled"),
    Specification(parameter_sample="half_a"),
    Specification(parameter_sample="half_b"),
)


def check_ensemble(
    specifications: Sequence[Specification], reference: Specification = REFERENCE
) -> None:
    """Refuse an ensemble that is not a few distinct, supported specifications.

    It must hold the reference and at most :data:`MAX_SPECIFICATIONS`
    specifications, each different.
    """
    if reference not in specifications:
        raise ValueError("the ensemble must include the reference specification")
    if len(set(specifications)) != len(specifications):
        raise ValueError("every specification in an ensemble must differ")
    if not 2 <= len(specifications) <= MAX_SPECIFICATIONS:
        raise ValueError(
            f"an ensemble compares 2 to {MAX_SPECIFICATIONS} specifications, not "
            "dozens of perturbations"
        )


def halves(training: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """The two fixed halves of the training households: alternate, in sorted order."""
    ordered = sorted(training)
    if len(ordered) < 2:
        raise ValueError("halves need at least two training households")
    return {"half_a": tuple(ordered[0::2]), "half_b": tuple(ordered[1::2])}


def fit_samples(
    statistics: Mapping[str, Mapping[EvidenceChannel, ChannelStatistics]],
    training: Sequence[str],
    *,
    states: tuple[Any, ...],
    pseudo_windows: float,
) -> dict[str, FittedChannels]:
    """The population on every training household, and on each half of them."""
    samples = {"all": tuple(sorted(training)), **halves(training)}
    return {
        sample: combine_statistics(
            {h: statistics[h] for h in homes},
            states=states,
            pseudo_windows=pseudo_windows,
        )
        for sample, homes in samples.items()
    }


def _channel_models(
    specification: Specification,
    recording: CasasRecording,
    population: FittedChannels,
    *,
    household: str,
    resolution: EvidenceResolution,
    ontology: StateOntology,
    config: PoolingConfig,
    until: datetime | None,
) -> Mapping[EvidenceChannel, ChannelModel]:
    if specification.parameters == "population":
        return population.models(
            recording.registry, resolution, specification.observation, ontology
        )
    if until is None:
        raise ValueError("pooled parameters need a cut-off before the scored windows")
    own = household_statistics(
        recording, resolution, ontology, household=household, until=until
    )
    pooled = pool_channels(
        population,
        recording.registry,
        own,
        household=household,
        resolution=resolution,
        config=config,
        until=until,
        ontology=ontology,
    )
    if specification.observation == HURDLE:
        return pooled.models()
    declared, _ = channel_likelihoods(recording.registry, resolution, ontology)
    return pooled.models(
        dispersion={
            channel: population.dispersion(channel, terms.expected)
            for channel, terms in declared.items()
        }
    )


def household_trace(
    recording: CasasRecording,
    specifications: Sequence[Specification],
    samples: Mapping[str, FittedChannels],
    *,
    household: str,
    resolution: EvidenceResolution,
    ontology: StateOntology | None = None,
    config: PoolingConfig | None = None,
    until: datetime | None = None,
    reference: Specification = REFERENCE,
) -> DisagreementTrace:
    """Every specification's recursion posterior at the household's scored windows.

    Every specification runs the same recursion over every window of the
    recording, from the stationary distribution, with the same transition. Only
    its channel models differ. The scored windows are the labelled ones, and
    with *until* only those closing after it, the same for every
    specification: a pooled specification has read the labelled windows up to
    *until*.

    The reference specification's most probable state is the decision, which
    the trace records and does not change.
    """
    check_ensemble(specifications, reference)
    ontology = ontology or StateOntology()
    config = config or PoolingConfig(288.0)
    if household in samples["all"].fitted_on:
        raise ValueError(
            f"household {household!r} was used to fit the population; its windows "
            "would be scored by parameters fitted on them"
        )
    counts, rows, labels, moments = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    if until is not None:
        after = np.array([moments[int(r)] > until for r in rows], dtype=bool)
        rows, labels = rows[after], labels[after]
    transition = ontology.transition(resolution.step)
    posteriors = {}
    for specification in specifications:
        models = _channel_models(
            specification,
            recording,
            samples[specification.parameter_sample],
            household=household,
            resolution=resolution,
            ontology=ontology,
            config=config,
            until=until,
        )
        _, posterior = filter_recursion(
            total_loglik(models, counts), transition, ontology.stationary()
        )
        posteriors[specification.name] = posterior[rows]
    return compare_posteriors(
        posteriors,
        household=household,
        timestamps=[moments[int(r)].isoformat() for r in rows],
        states=[s.value for s in ontology.states],
        reference=reference.name,
        truth=labels,
    )


def ensemble_report(
    traces: Sequence[DisagreementTrace],
    specifications: Sequence[Specification],
    files: Mapping[str, Mapping[str, str]] | None = None,
) -> DisagreementReport:
    """The record section for *traces*, with every specification's support."""
    return DisagreementReport.from_traces(
        traces,
        [s.to_dict() for s in sorted(specifications, key=lambda s: s.name)],
        files,
    )
