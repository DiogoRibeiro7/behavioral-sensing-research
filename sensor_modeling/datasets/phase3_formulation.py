"""The confirmed Phase 3 formulation, with frozen parameters, for the online pipeline.

The formulation the held-out confirmation scored is the fitted hurdle channels
of Phase 3.3 with the hierarchical time prior of Phase 3.1 as an
hour-dependent transition, in the filter's recursion. Its parameters were
fitted once on the 20 development homes
(:func:`~sensor_modeling.datasets.heldout_experiment.training_fit`), and the
held-out record digests them. This module reads those parameters from the file
the package ships, refuses a file whose parameters are not the ones the record
digests, and gives what
:class:`~sensor_modeling.online.BehaviouralSensingPipeline` needs to run the
formulation on one home:

- :meth:`Phase3Formulation.emissions`, a
  :class:`~sensor_modeling.fusion.HurdleChannelEmission` for each channel the
  home's registry routes a sensor to;
- :meth:`Phase3Formulation.ontology`, the default ontology with the time prior
  as its circadian term;
- :meth:`Phase3Formulation.start`, the time prior's distribution at an hour,
  from which the recursion starts one window before its first.

What the parameters were fitted on
----------------------------------
Single-resident CASAS ``hh`` homes, with the CASAS sensor layout and labelling,
in windows of five minutes, with the hour read on the homes' local clock. A
state the training homes never showed on a channel takes the home's declared
expected count as its parameters, exactly as the research recursion does, so
those parameters depend on the home's registry.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from ..fusion.channels import HurdleChannelEmission
from ..observations.registry import SensorRegistry
from ..states.ontology import BehaviouralState, StateOntology
from .channel_models import FittedChannels, truncated_poisson_rate
from .information_sets import EvidenceResolution
from .periodic_prior import PeriodicStatePrior
from .restricted_filter import channel_likelihoods

SCHEMA = "phase3-parameters/1"

#: The file the package ships, under ``sensor_modeling/data``.
PARAMETERS_FILE = "phase3_parameters.json"

#: The record whose training fit the parameters must be.
HELDOUT_RECORD = "artifacts/heldout/casas-heldout-confirmation.json"

#: ``results.training.channels_sha256`` of the held-out record.
CHANNELS_SHA256 = "15a63d668737ede83e3c8b10d7c97917b60849e2e253285c84a27a775b352ed6"

#: ``results.training.periodic_prior_sha256`` of the held-out record.
PERIODIC_PRIOR_SHA256 = (
    "99f979d697985d9bfe056cbeb66dbd56bf5a623af0427e780c75cfa511409b7b"
)

#: The window the channels were fitted at, and so the pipeline's step.
STEP = timedelta(minutes=5)

#: Prefix of a channel's key in the filter, which no CASAS sensor carries.
CHANNEL_KEY = "channel:"


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """SHA-256 of *payload* as the fitted objects digest themselves."""
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def parameters_payload(
    channels: FittedChannels,
    prior: PeriodicStatePrior,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """The parameter file's contents for a fit, with where it came from."""
    return {
        "schema": SCHEMA,
        "formulation": (
            "fitted hurdle channels with the hierarchical time prior as an "
            "hour-dependent transition, in the filter's recursion"
        ),
        "step_seconds": int(STEP.total_seconds()),
        "channels": channels.to_dict(),
        "channels_sha256": channels.sha256(),
        "periodic_prior": prior.to_dict(),
        "periodic_prior_sha256": prior.sha256(),
        "source": dict(source),
    }


def write_parameters(payload: Mapping[str, Any], path: Path) -> Path:
    """Write a parameter file, with LF line endings."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def _read(path: Path | None) -> str:
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    shipped = resources.files("sensor_modeling") / "data" / PARAMETERS_FILE
    if not shipped.is_file():
        raise FileNotFoundError(
            f"sensor_modeling/data/{PARAMETERS_FILE} is not installed, so the "
            "phase3 formulation has no parameters"
        )
    return shipped.read_text(encoding="utf-8")


def _unseen(expected: float, kappa: float) -> tuple[float, float]:
    """Silence and rate of a state the training homes never showed on a channel.

    :meth:`FittedChannels.parameters` with no training window: the prior from
    the home's declared expected count *expected*, with the same arithmetic.
    """
    prior_mean = np.float64(expected)
    prior_silence = np.exp(-prior_mean)
    prior_active_mean = prior_mean / -np.expm1(-prior_mean)
    silence = (0.0 + kappa * prior_silence) / (0.0 + kappa)
    active_mean = (0.0 + kappa * prior_active_mean) / (0.0 + kappa)
    return float(silence), truncated_poisson_rate(float(active_mean))


@dataclass(frozen=True)
class Phase3Formulation:
    """The frozen parameters of the confirmed formulation.

    Attributes
    ----------
    channels
        The fitted channels as :meth:`FittedChannels.to_dict` gives them: per
        channel and state, the silence probability and truncated rate, or
        ``None`` where no training window had the state.
    prior
        The hierarchical time prior.
    """

    channels: Mapping[str, Any]
    prior: PeriodicStatePrior

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        channels_sha256: str | None = None,
        periodic_prior_sha256: str | None = None,
    ) -> Phase3Formulation:
        """Read the parameters, refusing any but the ones expected.

        Parameters
        ----------
        path
            A parameter file. By default, the one the package ships.
        channels_sha256, periodic_prior_sha256
            The digests the parameters must have. By default, the ones the
            held-out record gives for its training fit.

        Raises
        ------
        FileNotFoundError
            If there is no parameter file.
        ValueError
            If the file is not a parameter file, or its parameters differ from
            the ones expected.
        """
        payload = json.loads(_read(path))
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} parameter file")
        if payload.get("step_seconds") != int(STEP.total_seconds()):
            raise ValueError("the parameters were not fitted at a five-minute window")
        channels = payload["channels"]
        expected = channels_sha256 or CHANNELS_SHA256
        if canonical_sha256(channels) != expected:
            raise ValueError(
                "the channel parameters are not the ones the held-out "
                "confirmation fitted"
            )
        prior = PeriodicStatePrior.from_dict(payload["periodic_prior"])
        if prior.sha256() != (periodic_prior_sha256 or PERIODIC_PRIOR_SHA256):
            raise ValueError(
                "the time prior is not the one the held-out confirmation fitted"
            )
        states = tuple(StateOntology().states)
        if prior.states != states:
            raise ValueError("the time prior's states are not the default ontology's")
        known = {channel.name for channel in EvidenceResolution().channels}
        unknown = sorted(set(channels["channels"]) - known)
        if unknown:
            raise ValueError(f"the parameters name channels outside CASAS's: {unknown}")
        return cls(channels, prior)

    def ontology(self) -> StateOntology:
        """The default ontology with the time prior as its circadian term."""
        return self.prior.ontology(StateOntology())

    def start(self, at: datetime) -> np.ndarray:
        """The time prior's distribution at the local hour of *at*."""
        start: np.ndarray = self.prior.probabilities([at.hour])[0]
        return start

    def emissions(self, registry: SensorRegistry) -> list[HurdleChannelEmission]:
        """One home's channel emissions, for every channel its sensors route to.

        A sensor routes to a channel when it is an event sensor whose room and
        modality name one of the CASAS channels, as in the research recursion.
        Other sensors inform no channel.

        Raises
        ------
        ValueError
            If no sensor routes to a channel, or a channel pools sensors whose
            declared rates differ.
        """
        ontology = StateOntology()
        terms, _ = channel_likelihoods(registry, EvidenceResolution(), ontology)
        if not terms:
            raise ValueError(
                "no event sensor of the registry routes to a channel the "
                "formulation was fitted on"
            )
        kappa = float(self.channels["pseudo_windows"])
        fitted: Mapping[str, Any] = self.channels["channels"]
        emissions = []
        for channel in sorted(terms):
            term = terms[channel]
            stored = fitted.get(channel.name, {})
            silence: dict[BehaviouralState, float] = {}
            rate: dict[BehaviouralState, float] = {}
            for index, state in enumerate(ontology.states):
                entry = stored.get(state.value) or {}
                if entry.get("silence") is None or entry.get("rate") is None:
                    silence[state], rate[state] = _unseen(
                        float(term.expected[index]), kappa
                    )
                else:
                    silence[state] = float(entry["silence"])
                    rate[state] = float(entry["rate"])
            emissions.append(
                HurdleChannelEmission(
                    f"{CHANNEL_KEY}{channel.name}",
                    sensors=term.sensors,
                    silence=silence,
                    rate=rate,
                    window=STEP,
                    modality=channel.modality,
                )
            )
        return emissions
