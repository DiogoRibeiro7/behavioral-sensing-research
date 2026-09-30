"""Phase 5: the external-generalisation protocol, frozen before any external scoring.

The external dataset is the UCI "Activities of Daily Living Recognition Using
Binary Sensors" dataset (:mod:`sensor_modeling.external.ordonez`). It holds two
single-resident homes in Spain, collected independently of CASAS, with a
materially different sensing layout. It is legally and reproducibly
obtainable: CC BY 4.0, from a DOI, and verified by file digests.

Two conditions are frozen:

``zero_shot``
    The Phase 3.3 follow-up's fitted hurdle-Poisson recursion, with its
    population fitted on the 20 CASAS development homes, and no parameter
    fitted on external data.
``adapted``
    The same, with each home's channel parameters partially pooled toward the
    population. Only the labelled windows of the home's adaptation period, its
    first seven days, are used, at the Phase 3.4 strength.

Both are scored on the same windows: every labelled window of the scored
period, the days after the adaptation period, whose annotation maps to a state.
The scored period influences neither the mapping nor any model setting.

This module declares the protocol. :data:`POPULATION_SHA256` pins the model
versions, and :func:`declared_protocol` is frozen in
``artifacts/phase5/external_protocol.json``. The scoring run must refuse any
other protocol, and must reproduce these digests before it scores.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..external.contract import HouseholdData
from ..external.mapping import OntologyMapping
from ..external.ordonez import (
    ARCHIVE_SHA256,
    ARCHIVE_URL,
    HOUSEHOLDS,
    LABELS,
    ORDONEZ_MAPPING,
    PROVENANCE,
    SENSORS,
    TIMEZONE,
    native_type,
    period_start,
    sensor_id,
)
from ..external.validation import Severity, ValidationReport
from ..observations import ObservationKind
from ..states import BehaviouralState
from .information_sets import HH_EVIDENCE_CHANNELS, EvidenceChannel, EvidenceResolution
from .partial_pooling import PoolingConfig
from .recoverable_gap import GENERATIVE_CONFIGURATION, FrozenSplits
from .structural_models import halves
from .uncertainty_experiment import ENSEMBLE

PROTOCOL_SCHEMA = "external-protocol/1"

ZERO_SHOT = "zero_shot"
ADAPTED = "adapted"
DECLARED = "declared"

#: The days at the start of each home that adaptation may use.
ADAPTATION_DAYS = 7

#: Labelled days of each home, from its description file.
LABELLED_DAYS = {"OrdonezA": 14, "OrdonezB": 21}

#: The CASAS development population's ``FittedChannels.sha256()``, fitted with
#: 12 pseudo-windows on all 20 development homes, and on each fixed half.
POPULATION_SHA256 = {
    "all": "50e00c3da80c2ae814575b69b0423b75584c0278307cdd54554c8b3ca91da7a4",
    "half_a": "dd9381bb3388e2b9ee3f2584c59f70d56f1eac12b37cbde2bdfbfba45121b0a6",
    "half_b": "f6f3b9f1e1271edf6c698e88ef8afb4aca8a542bbd6547746231e7043a6de611",
}

#: Validation errors that leave the offending items unscored rather than
#: exclude the household.
TOLERATED = frozenset(
    {"impossible_interval", "undeclared_label", "undeclared_sensor", "unknown_sensor"}
)

MINIMAL_DIFFERENCES = {
    "balanced_accuracy": 0.02,
    "log_loss": 0.05,
    "brier": 0.01,
    "calibration_error": 0.02,
    "aurc": 0.01,
    "chance": 0.05,
}


def model_channel(sensor: tuple[str, str, str]) -> tuple[str | None, str]:
    """The CASAS-trained model's channel a sensor feeds, or ``None`` with the reason."""
    location, kind, place = sensor
    semantics = ORDONEZ_MAPPING.sensor_type(native_type(location, kind))
    room = ORDONEZ_MAPPING.location(place)
    if not semantics.usable:
        return None, f"its type is {semantics.status}: {semantics.rationale}"
    target = semantics.target
    if target.kind is not ObservationKind.EVENT:
        return None, f"a {target.kind.value} sensor; the channels count events"
    if not room.usable:
        return None, f"its room is {room.status}"
    channel = EvidenceChannel(room.target, target.modality)
    if channel not in HH_EVIDENCE_CHANNELS:
        return None, f"no {channel.name} channel in the CASAS-trained model"
    return channel.name, "routed"


def sensor_table(household: str) -> list[dict[str, Any]]:
    """Every sensor of a home: its mapping and the model channel it feeds, if any."""
    rows = []
    for sensor in SENSORS[household]:
        location, kind, place = sensor
        semantics = ORDONEZ_MAPPING.sensor_type(native_type(location, kind))
        room = ORDONEZ_MAPPING.location(place)
        channel, reason = model_channel(sensor)
        rows.append(
            {
                "sensor": sensor_id(location, kind, place),
                "native_type": native_type(location, kind),
                "place": place,
                "type_status": semantics.status,
                "modality": (
                    semantics.target.modality.value if semantics.usable else None
                ),
                "kind": semantics.target.kind.value if semantics.usable else None,
                "room": room.target if room.usable else None,
                "room_status": room.status,
                "model_channel": channel,
                "unsupported_because": None if channel else reason,
            }
        )
    return rows


def scored_states(mapping: OntologyMapping = ORDONEZ_MAPPING) -> list[str]:
    """The states some label maps to: the only states that can be scored as truth."""
    states = {
        mapping.label(label).target for label in LABELS if mapping.label(label).usable
    }
    return [s.value for s in BehaviouralState if s in states]


def unsupported_states(mapping: OntologyMapping = ORDONEZ_MAPPING) -> dict[str, str]:
    """Every latent state no label maps to, with why."""
    covered = set(scored_states(mapping))
    reasons: dict[str, str] = {}
    for state in BehaviouralState:
        if state is BehaviouralState.UNKNOWN or state.value in covered:
            continue
        ambiguous = [
            label
            for label in LABELS
            if mapping.label(label).status == "ambiguous"
            and state in mapping.label(label).targets
        ]
        reasons[state.value] = (
            f"a candidate of the ambiguous labels {', '.join(ambiguous)}, which are "
            "not scored"
            if ambiguous
            else "no label names it"
        )
    return reasons


def eligibility(report: ValidationReport) -> tuple[bool, list[str]]:
    """Whether a validated home is eligible, and the errors that exclude it.

    Errors in :data:`TOLERATED` leave their items unscored and reported. Any
    other error excludes the home.
    """
    blocking = sorted(
        {i.code for i in report.issues if i.severity is Severity.ERROR} - set(TOLERATED)
    )
    return not blocking, blocking


def adaptation_end(data: HouseholdData) -> datetime:
    """Local midnight ending the adaptation period, and starting the scored one."""
    return period_start(data, ADAPTATION_DAYS)


@dataclass(frozen=True)
class ExternalProtocol:
    """Everything the external evaluation fixes before any external scoring."""

    splits: FrozenSplits
    mapping: OntologyMapping = ORDONEZ_MAPPING
    resolution: EvidenceResolution = field(default_factory=EvidenceResolution)
    pseudo_windows: float = 12.0
    pooling: PoolingConfig = field(default_factory=lambda: PoolingConfig(288.0))
    adaptation_days: int = ADAPTATION_DAYS
    resamples: int = 10_000
    confidence: float = 0.95
    seed: int = 0
    minimal_differences: Mapping[str, float] = field(
        default_factory=lambda: dict(MINIMAL_DIFFERENCES)
    )
    name: str = "phase5-external-ordonez"

    @property
    def development_homes(self) -> tuple[str, ...]:
        """The 20 CASAS development homes the population is fitted on."""
        return tuple(sorted(self.splits.homes))

    def to_dict(self) -> dict[str, Any]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        homes = self.development_homes
        return {
            "schema": PROTOCOL_SCHEMA,
            "name": self.name,
            "status": "frozen before any external scoring; no model performance on "
            "the external dataset was computed or viewed",
            "question": "do the fitted channel models, and the Phase 4 structural "
            "signal, keep their direction and practical value on an "
            "independently collected dataset with a different sensing layout",
            "dataset": {
                "provenance": PROVENANCE.to_dict(),
                "archive": {"url": ARCHIVE_URL, "sha256": ARCHIVE_SHA256},
                "redistribution": "none: the scoring run downloads the archive "
                "and verifies every file's digest",
                "choice": "it is independently collected, single-resident by "
                "its documentation, annotated with intervals, and legally and "
                "reproducibly obtainable: a stated licence, a DOI and a stable "
                "archive; datasets without a stated licence, or with more than one "
                "resident per home, were not considered",
            },
            "households": {
                "candidates": list(HOUSEHOLDS),
                "labelled_days": dict(LABELLED_DAYS),
                "inclusion": "every candidate home, single-resident by the "
                "dataset's documentation",
                "exclusion": "a home whose validation under the frozen mapping "
                "has an error other than "
                + ", ".join(sorted(TOLERATED))
                + "; tolerated errors leave their items unscored and reported",
                "eligibility": f"at least {self.adaptation_days} labelled days "
                "after the adaptation period, a declared timezone, and at least "
                "two channels the model can use",
            },
            "periods": {
                "timezone": TIMEZONE,
                "adaptation": f"local midnight of the home's first annotated day, "
                f"for {self.adaptation_days} days",
                "scored": "from the end of the adaptation period to the end of "
                "the recording",
                "scored_days": {
                    home: LABELLED_DAYS[home] - self.adaptation_days
                    for home in HOUSEHOLDS
                },
                "use": "the adaptation period may inform the mapping and the "
                "adapted condition; the scored period informs nothing",
            },
            "mapping": {
                "file": "artifacts/phase5/ordonez_mapping.json",
                "sha256": self.mapping.sha256(),
                "written_from": self.mapping.description,
                "labels": {
                    label: {
                        "outcome": self.mapping.label(label).status,
                        "targets": [t.value for t in self.mapping.label(label).targets],
                    }
                    for label in LABELS
                },
                "new_values": "a label, sensor type or room first seen in the "
                "scored period is left unscored and reported; the frozen mapping "
                "is never edited",
            },
            "sensor_mapping": {home: sensor_table(home) for home in HOUSEHOLDS},
            "unsupported_states": unsupported_states(self.mapping),
            "scored_states": scored_states(self.mapping),
            "unsupported_sensors": {
                home: {
                    row["sensor"]: row["unsupported_because"]
                    for row in sensor_table(home)
                    if row["model_channel"] is None
                }
                for home in HOUSEHOLDS
            },
            "uninstrumented_channels": {
                home: sorted(
                    {c.name for c in HH_EVIDENCE_CHANNELS}
                    - {
                        row["model_channel"]
                        for row in sensor_table(home)
                        if row["model_channel"]
                    }
                )
                for home in HOUSEHOLDS
            },
            "preprocessing": {
                "adapter": "sensor_modeling.external.ordonez.read_household: each "
                "activation row an ON event at its start and an OFF event at its "
                "end; each annotation row an interval",
                "conversion": "sensor_modeling.external.to_canonical with "
                "strict=False: events sorted, exact duplicates kept once, naive "
                f"times placed in {TIMEZONE}, overlapping annotations split, "
                "conflicting spans unscored, everything left out counted",
                "windows": f"the evidence resolution's "
                f"{int(self.resolution.step.total_seconds() // 60)}-minute steps "
                "from each recording's first observation to its last",
                "channels": [c.name for c in self.resolution.channels],
                "labels": "truth_series of the canonical segments at each step; "
                "unlabelled, unmappable, ambiguous and conflicting steps are not "
                "scored",
                "scored_windows": "labelled steps in the scored period whose "
                "state is one of the scored states",
            },
            "models": {
                DECLARED: {
                    "specification": "declared",
                    "description": "the filter's declared Poisson channel rates, "
                    "from the external registry's default emissions, in the same "
                    "recursion; nothing fitted",
                    "configuration": dict(GENERATIVE_CONFIGURATION),
                    "use": "the reference for the transfer estimand",
                },
                ZERO_SHOT: {
                    "family": "hurdle",
                    "specification": "hurdle/population/all",
                    "population": "fit_samples(...)['all'] of every development "
                    "home's labelled windows",
                    "development_homes": list(homes),
                    "splits_sha256": self.splits.sha256,
                    "pseudo_windows": self.pseudo_windows,
                    "population_sha256": POPULATION_SHA256["all"],
                    "inference": "filter_recursion over every step of the external "
                    "recording from the stationary distribution, with the "
                    "ontology's default transition",
                },
                ADAPTED: {
                    "from": ZERO_SHOT,
                    "adaptation": "pool_channels of the home's own statistics "
                    "toward the population, with household_statistics(until=the "
                    "end of the adaptation period)",
                    "pooling": self.pooling.to_dict(),
                    "strength_source": "the Phase 3.4 declared strength: 288 "
                    "five-minute windows, 24 labelled hours",
                },
                "structural_ensemble": {
                    "specifications": [s.name for s in ENSEMBLE],
                    "halves": {k: list(v) for k, v in halves(homes).items()},
                    "population_sha256": dict(POPULATION_SHA256),
                    "source": "the Phase 4 population-only ensemble, fitted on "
                    "the 20 development homes and their fixed halves",
                },
                "code": "the scoring run's code must check this frozen file and "
                "reproduce every population digest before it scores",
            },
            "conditions": {
                ZERO_SHOT: "no parameter is fitted on external data; the "
                "adaptation period's events only run the recursion",
                ADAPTED: "each home's channel parameters are pooled toward the "
                "population on its adaptation period's labelled windows",
                "same_windows": "both conditions, and the declared model, are "
                "scored on the same scored windows",
            },
            "adaptation_rules": {
                "permitted": "the adaptation period's events and mapped labels",
                "forbidden": "any event or label of the scored period, and any "
                "setting chosen by looking at external results",
                "adapted": "each routed channel's silence probability and active "
                "mean, per state, by partial pooling",
                "not_adapted": "the transition, the prior, the mapping, the channel "
                "set, the dispersion and the ensemble",
            },
            "metrics": {
                "primary": ["balanced_accuracy", "log_loss", "calibration_error"],
                "secondary": ["brier", "macro_f1", "per_class_recall"],
                "selective": "error AURC of confidence and of the structural "
                "ensemble's normalised discordance over the default coverage "
                "grid, within the home",
                "chance": "one over the number of scored states present in the "
                "home's scored windows",
                "scoring": "per home, over the scored windows; predictions over "
                "every latent state; a prediction of an unsupported state is an "
                "error",
            },
            "bootstrap": {
                "unit": "local calendar days of the scored period, within a home",
                "reason": "two homes support no between-household inference; days "
                "keep within-day dependence intact",
                "resamples": self.resamples,
                "confidence": self.confidence,
                "interval": "percentile",
                "paired": "every model and condition is resampled on the same days",
                "seed": self.seed,
            },
            "minimal_differences": dict(self.minimal_differences),
            "estimands": {
                "T": "transfer: zero-shot minus declared, per home",
                "A": "adaptation: adapted minus zero-shot, per home",
                "C": "above chance: zero-shot balanced accuracy minus chance, per "
                "home",
                "S": "Phase 4 direction: confidence's error AURC minus the "
                "structural signal's, per home, zero-shot",
            },
            "verdicts": "per home and metric, oriented so that positive favours "
            "the first model: favours it when the mean is at least the minimal "
            "difference and the day-bootstrap interval lies above 0; favours the "
            "second when at most minus it with the interval below 0; negligible "
            "when the interval lies within plus or minus it; uncertain otherwise",
            "criteria": {
                "T": "transfers when balanced accuracy favours zero-shot in both "
                "homes and log loss favours the declared model in neither; does "
                "not transfer when balanced accuracy is negligible or favours the "
                "declared model in both; inconclusive otherwise",
                "A": "helps when balanced accuracy favours adapted in both homes "
                "and log loss favours zero-shot in neither; does not help when "
                "balanced accuracy is negligible or favours zero-shot in both; "
                "inconclusive otherwise",
                "C": "above chance when the difference favours zero-shot in both "
                "homes; not above chance when it is negligible or favours chance in "
                "both; inconclusive otherwise",
                "S": "the direction survives when the AURC favours the structural "
                "signal in both homes; reverses when it favours confidence in "
                "both; inconclusive otherwise",
                "claims": "two homes: every conclusion is about these two homes, "
                "not a population of homes",
            },
            "reporting": "every home's results, every estimand and every negative "
            "result; household-level first, pooled only as description; no "
            "threshold, no retuning, no added model or metric",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_protocol(splits: FrozenSplits) -> ExternalProtocol:
    """The protocol as declared for the frozen development panel."""
    return ExternalProtocol(splits)


def check_frozen_protocol(protocol: ExternalProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after scoring begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def write_protocol(protocol: ExternalProtocol, path: Path) -> str:
    """Write the declaration with its digest; return the digest."""
    payload = {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return protocol.sha256()
