"""Phase 4 again, for the reference with the time prior.

The first Phase 4 comparison ranked the predictions of the hurdle recursion
without the time prior, then the reference formulation: structural
disagreement was materially better than confidence, and the minority-state
guard covered only ``bed_awake``. The reference has since become the fitted
hurdle channels with the time prior's hour-dependent transition, which changes
every recursion the signals read. This experiment repeats the comparison, with
the same signals, ensemble, estimands and decision rule, on the new reference,
before an abstention rule is selected for it.

What changes, and nothing else:

- **The predictions.** ``filter_periodic_hurdle``, the combined-prior recursion,
  whose scores the run must first give back.
- **The ensemble.** Every specification runs the time prior's recursion, with
  the fold's prior fitted on all its training households, so only their channel
  models differ, as before.
- **The evidence groups' context.** The time prior recursion's prediction
  before the window.

``predictive_mismatch`` reads the count laws alone, never the prediction, so it
is the Phase 4 signal unchanged.

Everything is declared in :class:`PriorUncertaintyProtocol` and frozen, with
the code the run depends on, before any household is scored, and the run is
one-shot. The protocol also declares, before scoring, how the abstention rule's
signal is chosen from the result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.disagreement import DisagreementTrace, compare_posteriors
from ..evaluation.metrics import PredictionMetrics
from ..evaluation.provenance import ExperimentRecord, InputArtifact, ModelRecord
from ..evaluation.selective import SelectiveData, Signal, evaluate_signals
from ..fusion.regime import ONLINE
from ..states.ontology import StateOntology
from . import combined_prior_experiment as combined
from . import pinning
from . import uncertainty_experiment as phase4
from .casas import CasasRecording
from .channel_models import (
    HURDLE,
    ChannelStatistics,
    FittedChannels,
    home_statistics,
    household_channel_counts,
    total_loglik,
)
from .combined_prior_experiment import (
    FILTER_PERIODIC_HURDLE,
    CombinedPriorProtocol,
    check_fits,
    periodic_recursion,
    periodic_transitions,
)
from .information_sets import EvidenceChannel
from .matched_evaluation import _finite, online_evidence
from .observation_mismatch import household_mismatch
from .partial_pooling import PoolingConfig
from .periodic_prior import PeriodicStatePrior
from .pinning import file_sha256, read_amendments
from .rates_experiment import fold_fits
from .recoverable_gap import FrozenSplits, _score
from .structural_models import (
    REFERENCE,
    Specification,
    _channel_models,
    check_ensemble,
    fit_samples,
)
from .substate_experiment import check_reference_scores
from .uncertainty_experiment import (
    COMPARATOR,
    CONFIDENCE,
    DIRECTIONS,
    ENSEMBLE,
    ENTROPY,
    EVIDENCE,
    MISMATCH,
    MISSING,
    SIGNALS,
    STRUCTURAL,
    STRUCTURAL_SIGNALS,
    UncertaintyProtocol,
    difficult_minority_states,
    evidence_discordance,
    normalised_entropy,
    paired_comparisons,
    rejection_bias,
    reproduction,
)

PROTOCOL_SCHEMA = "prior-uncertainty-protocol/1"
#: The record keeps the Phase 4 layout, so its page renders the same way.
RESULT_SCHEMA = phase4.RESULT_SCHEMA
EXPERIMENT = "phase4-uncertainty-time-prior"

PROTOCOL_FILE = "artifacts/phase4/prior_uncertainty_protocol.json"
RECORD_FILE = f"artifacts/phase4/{EXPERIMENT}.json"
AMENDMENTS_FILE = "artifacts/phase4/prior_uncertainty_amendments.json"
RESULTS_BRANCH = f"run-artifacts/{EXPERIMENT}"

PHASE4_PROTOCOL = "artifacts/phase4/uncertainty_protocol.json"
PHASE4_PROTOCOL_SHA256 = (
    "537dd3d319e1ec3964d2d8df370cac3d05a72fd267a10eaa148b0148f59782e1"
)
PHASE4_RECORD = "artifacts/phase4/phase4-uncertainty-diagnostics.json"
COMBINED_PRIOR_PROTOCOL = "artifacts/phase3/combined_prior_protocol.json"
COMBINED_PRIOR_RECORD = "artifacts/phase3/phase3-combined-prior.json"
PINNED_RECORDS = {
    **combined.PINNED_RECORDS,
    COMBINED_PRIOR_PROTOCOL: (
        "3a0c5cad58187609601181ae841099b9144707ba2327af4173c9442812dc52a5"
    ),
    COMBINED_PRIOR_RECORD: (
        "57db137f57f5b298054d7a4cfe8c837d854ca9a8793329857d7803401477db59"
    ),
    PHASE4_PROTOCOL: PHASE4_PROTOCOL_SHA256,
    PHASE4_RECORD: ("faeb9d679ee251b493586292b9d2dd55913cab6cb986f9ad14cc4e8f3bc5c2cb"),
    # Cited in what had been seen.
    "artifacts/heldout/casas-heldout-confirmation.json": (
        "785f6757ed9e9890dc7dfbf471660403779cc7dcc6a246586a87fd2323e2c97a"
    ),
    "artifacts/phase3/phase3-activity-substates.json": (
        "73e19a933de8d3bd1cf97aed8981a04bf446673f92d29b56a47f679cb525b1df"
    ),
}

#: The sources whose content the freeze records and the run requires unchanged.
WORKFLOW = ".github/workflows/run-phase4-time-prior.yml"
PINNED_FILES = (
    "pyproject.toml",
    "scripts/freeze_phase4_time_prior.py",
    "scripts/run_phase4_time_prior.py",
    WORKFLOW,
)
PINNED_DISTRIBUTIONS = ("numpy", "scipy", "pandas")
#: The versions the workflow installs, which the freeze must be made with.
RUN_VERSIONS = {"numpy": "2.3.5", "pandas": "2.3.3", "scipy": "1.15.3"}
_REPOSITORY = Path(__file__).resolve().parents[2]

INSPECTED_BEFORE = (
    "The Phase 4 record, for the hurdle recursion without the time prior: "
    "structural_disagreement is materially better than confidence, its AURC "
    "lower by +0.059 [+0.031, +0.087]; entropy is negligibly different, "
    "+0.002 [+0.001, +0.003]; evidence_disagreement, −0.149 [−0.183, −0.110], "
    "and predictive_mismatch, −0.040 [−0.060, −0.020], are worse; the "
    "minority-state guard covers bed_awake alone, 68 windows in 5 households; "
    "structural_disagreement retains less than confidence of "
    "bathroom_activity and kitchen_activity at coverages 0.5 and 0.7; no "
    "threshold was selected.",
    "The combined-prior record: in the recursion, the fitted hurdle channels "
    "with the time prior score a mean household balanced accuracy of 0.517 "
    "against 0.456 without the prior, K1 +0.061 [+0.044, +0.077], with "
    "calibration error 0.213 against 0.284.",
    "The held-out record: the same formulation, fitted once on the 20 "
    "development homes, scored once on 43 held-out CASAS homes, with every "
    "home's scores published.",
    "The activity sub-state record: two activity sub-states were inconclusive "
    "against this reference, which stays the reference.",
    "What the reference's predictions already show: the combined-prior record "
    "publishes every household's confusion matrix, so the pooled table the "
    "difficult-minority rule reads is known. bed_awake is the only state below "
    "the minority share whose recall, 0.044 (3 of 68 windows), is below the "
    "pooled balanced accuracy, 0.473; bathroom_activity's recall, 0.487, is "
    "above it, and kitchen_activity's, 0.547. The guard will cover bed_awake "
    "alone, in 5 households.",
    "Confidence on this recursion: its calibration error in every household is "
    "published by the combined-prior, held-out and activity sub-state records. "
    "Entropy, structural_disagreement and evidence_disagreement had not been "
    "computed on it for any CASAS household before this protocol was frozen; "
    "predictive_mismatch reads no prediction, so its values are the Phase 4 "
    "record's. The code was exercised on simulated homes only.",
)

#: How the abstention rule's signal is chosen from this result, declared before it.
RULE_SIGNAL = (
    "the structural signal whose decision is materially better, the one with the "
    "largest mean AURC difference against confidence when several are; "
    "confidence when none is; a tie in that mean goes to the signal first in "
    "alphabetical order. Entropy is not a candidate: it ranks windows by the same "
    "posterior as confidence, and Phase 4 found it negligibly different"
)


def rule_signal(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The abstention rule's signal, by the declared rule."""
    better = {
        name: comparisons[name]["aurc"]["comparison"]["mean"]["estimate"]
        for name in STRUCTURAL_SIGNALS
        if comparisons[name]["decision"] == "materially better"
    }
    if not better:
        return {
            "signal": COMPARATOR,
            "reason": "no structural signal is materially better than confidence",
        }
    chosen = max(sorted(better), key=lambda name: better[name])
    return {
        "signal": chosen,
        "reason": "materially better than confidence"
        + (
            ", with the largest mean AURC difference of "
            + ", ".join(f"`{name}`" for name in sorted(better))
            if len(better) > 1
            else ""
        ),
    }


@dataclass(frozen=True)
class PriorUncertaintyProtocol:
    """Everything the experiment fixes before any household is scored.

    Attributes
    ----------
    base
        The frozen Phase 4 protocol: its signals, ensemble, evidence groups,
        coverage grid, estimands, minimal differences, bootstrap, guard and
        decision rule are used unchanged.
    combined
        The frozen combined-prior protocol, whose fold fits and recursion with
        the time prior give the predictions.
    tolerance
        How far a reproduced score or fitted parameter may lie from the
        published one before nothing is reported.
    """

    base: UncertaintyProtocol
    combined: CombinedPriorProtocol
    tolerance: float = 1e-6
    inspected_before: tuple[str, ...] = INSPECTED_BEFORE
    name: str = EXPERIMENT

    def __post_init__(self) -> None:
        """Refuse bases that do not describe the same households and channels."""
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        rates = self.combined.base
        if tuple(self.base.folds) != tuple(rates.folds):
            raise ValueError("the two base protocols must have the same folds")
        if self.base.resolution != rates.information_sets[0].resolution:
            raise ValueError("the two base protocols must read the same channels")
        if self.base.pseudo_windows != rates.pseudo_windows:
            raise ValueError("the two base protocols must fit the same channels")

    @property
    def homes(self) -> tuple[str, ...]:
        """Every held-out household, sorted."""
        return self.base.homes

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        declared = self.base.to_dict()
        ensemble = dict(declared["ensemble"])  # type: ignore[call-overload]
        evidence = dict(declared["evidence_groups"])  # type: ignore[call-overload]
        evidence["groups"] = (
            "motion: motion channels; contact: door channels; context: the time "
            "prior recursion's prediction before the window"
        )
        ensemble["reason"] = (
            f"{ensemble['reason']}; every specification runs the time prior's "
            "recursion, with the fold's prior fitted on all its training "
            "households, so only their channel models differ"
        )
        ensemble["transition"] = (
            "structural_models refuses the time prior as the recursion's "
            "transition because it had been evaluated only on declared "
            "information sets; the combined-prior protocol has since evaluated it "
            "in the recursion, a success, and it is the reference's transition "
            "here, shared by every specification rather than varied between them"
        )
        declared.update(
            {
                "schema": PROTOCOL_SCHEMA,
                "name": self.name,
                "status": "pre-specified, on the development panel, which earlier "
                "work has inspected; one run; not a held-out claim",
                "question": "does any structural diagnostic produce a materially "
                "better selective-risk curve than confidence for the fitted "
                "channels with the time prior, without disproportionately "
                "rejecting difficult minority states",
                "inspected_before": list(self.inspected_before),
                "base_protocol": {
                    "file": PHASE4_PROTOCOL,
                    "sha256": PHASE4_PROTOCOL_SHA256,
                    "protocol_sha256": self.base.sha256(),
                    "used": "its signals, directions and missing policies, its "
                    "ensemble's specifications, the evidence groups' channels and "
                    "floor, the coverage grid and inspection levels, the estimands "
                    "and minimal differences, the bootstrap, the difficult-minority "
                    "rule, the guard and the decision rule, unchanged",
                    "changed": "the predictions every signal ranks; the transition "
                    "every specification of the ensemble runs; the evidence "
                    "groups' context, now the time prior recursion's prediction",
                    "unchanged_values": "predictive_mismatch reads each channel's "
                    "count laws and the window's counts, never a prediction, so "
                    "its value in every window is the Phase 4 record's",
                },
                "time_prior": {
                    "file": COMBINED_PRIOR_PROTOCOL,
                    "sha256": PINNED_RECORDS[COMBINED_PRIOR_PROTOCOL],
                    "protocol_sha256": self.combined.sha256(),
                    "used": "its fold fits of the channels and the time prior on "
                    "training households only, which are the fitted-rates "
                    "protocol's, and its recursion with the time prior",
                },
                "predictions": {
                    "model": FILTER_PERIODIC_HURDLE,
                    "description": "the combined-prior recursion: fitted "
                    "hurdle-Poisson channels with the fold's population "
                    "parameters and the time prior's transition at each window's "
                    "hour, the population prior for every held-out household, fed "
                    "every window of the recording from the prior's distribution "
                    "at the first window's hour; the predicted state is the "
                    "posterior's most probable, and every signal ranks these same "
                    "predictions",
                    "pseudo_windows": self.base.pseudo_windows,
                    "scored": "every labelled window of every held-out household",
                },
                "ensemble": ensemble,
                "evidence_groups": evidence,
                "the_check": {
                    "what": [
                        "every fold's fitted channels and time prior equal the "
                        "fitted-rates record's, parameter by parameter",
                        "the ensemble's population on every training household is "
                        "the fold's fitted channels, exactly",
                        f"every household's scores of {FILTER_PERIODIC_HURDLE} "
                        "equal the combined-prior record's",
                    ],
                    "tolerance": self.tolerance,
                    "otherwise": "nothing is reported, not even per household",
                },
                "rule_signal": {
                    "rule": RULE_SIGNAL,
                    "use": "the abstention rule is selected afterwards, under its "
                    "own protocol, for the signal this rule gives",
                },
                "one_run": {
                    "runs": "one scoring run, dispatched on develop",
                    "branch": f"the run writes {RESULTS_BRANCH}, and the workflow "
                    "refuses to run when that branch exists",
                    "sources": "the freeze records the digest of every file of the "
                    "sensor_modeling package and of "
                    + ", ".join(PINNED_FILES)
                    + ", and the versions of "
                    + ", ".join(PINNED_DISTRIBUTIONS)
                    + "; the run refuses when a source differs",
                    "amendments": "a change to a pinned source after the freeze is "
                    f"declared in {AMENDMENTS_FILE} before the run, with its reason "
                    "and the digest of every source it changes; the run accepts a "
                    "changed source only at the digest an amendment gives, and the "
                    "record lists the amendments as an input",
                    "crash": "a run that stops before writing its record reports "
                    "nothing; it is fixed without looking at any outcome, the fix "
                    "is declared as an amendment, and it is run again. A run that "
                    "stops after writing its record is the run: the workflow still "
                    "commits what it wrote, and a missing page or figure is "
                    "generated afterwards from the record by the same functions",
                    "undefined": "a signal whose missing policy is refuse and which "
                    "is undefined in a scored window stops the run before it "
                    "writes a record: a crash, whose amendment may make the "
                    "signal's computation finite but may not change its policy",
                    "libraries": "the run refuses when numpy, scipy or pandas is at "
                    "another version than the freeze recorded",
                    "where": "the run is made only by its workflow, dispatched on "
                    "develop, and the workflow refuses when the frozen protocol "
                    "file has changed since the commit that froze it",
                },
                "pinned_records": dict(PINNED_RECORDS),
                "what_this_cannot_show": [
                    "A held-out claim: the 20 homes have been inspected in earlier "
                    "work, by Phase 4 among others.",
                    "A confirmation on the 43 held-out CASAS homes, which are used "
                    "up for this formulation: its scores in every one of them are "
                    "published.",
                    "A threshold: none is selected here.",
                    "Uncertainty in the fits: each fold's training homes share one "
                    "fit, and the household bootstrap resamples the scored homes "
                    "only.",
                    "Another formulation: every signal ranks the predictions of "
                    f"{FILTER_PERIODIC_HURDLE} alone.",
                    "An unbiased signal choice: the rule picks the abstention "
                    "rule's signal on these 20 homes, and the largest mean among "
                    "several winners is biased upward. No CASAS homes remain "
                    "untouched to confirm the signal or the rule chosen with it.",
                    "Independence from Phase 4: the homes, windows and folds are "
                    "the same, and predictive_mismatch is the same window by "
                    "window; only the predictions and the ensemble's recursion "
                    "differ.",
                    "A guard with reach: it covers bed_awake, 68 windows in 5 "
                    "households; windows cluster within homes, and the homes of a "
                    "fold share one fit.",
                ],
            }
        )
        return declared

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def check_pinned_records(root: Path = Path(".")) -> None:
    """Refuse unless every pinned record under *root* is the file pinned."""
    for name, digest in PINNED_RECORDS.items():
        if file_sha256(Path(root) / name) != digest:
            raise ValueError(f"{name} is not the file the protocol pins")


def declared_protocol(
    splits: FrozenSplits, root: Path = Path(".")
) -> PriorUncertaintyProtocol:
    """The protocol as declared, on the frozen Phase 4 and combined-prior protocols."""
    base = phase4.declared_protocol(splits)
    phase4.check_frozen_protocol(base, Path(root) / PHASE4_PROTOCOL)
    prior = combined.declared_protocol(splits, root)
    combined.check_frozen_protocol(prior, Path(root) / COMBINED_PRIOR_PROTOCOL)
    check_pinned_records(root)
    return PriorUncertaintyProtocol(base, prior)


def code_now(root: Path | None = None) -> dict[str, Any]:
    """The digest of every pinned source file, and the libraries' versions, now."""
    return pinning.code_now(
        _REPOSITORY if root is None else Path(root), PINNED_FILES, PINNED_DISTRIBUTIONS
    )


def library_changes(path: Path) -> list[str]:
    """The pinned libraries whose version differs now from the freeze at *path*."""
    return pinning.library_changes(path, PINNED_DISTRIBUTIONS)


def code_changes(
    path: Path,
    root: Path | None = None,
    amendments: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    """The pinned sources that differ now from the freeze and its amendments."""
    return pinning.code_changes(
        path, _REPOSITORY if root is None else Path(root), PINNED_FILES, amendments
    )


def write_protocol(
    protocol: PriorUncertaintyProtocol, path: Path, root: Path | None = None
) -> str:
    """Write the declaration, its digest, the moment and the code it was frozen with."""
    payload = {
        **protocol.to_dict(),
        "protocol_sha256": protocol.sha256(),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "at_freeze": code_now(root),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return protocol.sha256()


def check_frozen_protocol(protocol: PriorUncertaintyProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    frozen.pop("frozen_at", None)
    frozen.pop("at_freeze", None)
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after scoring begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


# ----------------------------------------------------------------------------
# One household
# ----------------------------------------------------------------------------
def periodic_trace(
    recording: CasasRecording,
    specifications: Sequence[Specification],
    samples: Mapping[str, FittedChannels],
    *,
    household: str,
    counts: Mapping[EvidenceChannel, np.ndarray],
    rows: np.ndarray,
    labels: np.ndarray,
    moments: Sequence[Any],
    transitions: np.ndarray,
    start: np.ndarray,
    protocol: UncertaintyProtocol,
    ontology: StateOntology,
) -> DisagreementTrace:
    """Every specification's posterior at the scored windows, under the time prior.

    As :func:`~sensor_modeling.datasets.structural_models.household_trace`,
    but every specification runs the time prior's recursion, from *start*, with
    the transition into each window at its hour. Only the channel models differ.
    """
    check_ensemble(specifications, REFERENCE)
    if household in samples["all"].fitted_on:
        raise ValueError(
            f"household {household!r} was used to fit the population; its windows "
            "would be scored by parameters fitted on them"
        )
    hours = [moment.hour for moment in moments]
    posteriors = {}
    for specification in specifications:
        models = _channel_models(
            specification,
            recording,
            samples[specification.parameter_sample],
            household=household,
            resolution=protocol.resolution,
            ontology=ontology,
            config=PoolingConfig(288.0),
            until=None,
        )
        _, posterior = periodic_recursion(
            total_loglik(models, counts), transitions, hours, start
        )
        posteriors[specification.name] = posterior[rows]
    return compare_posteriors(
        posteriors,
        household=household,
        timestamps=[moments[int(r)].isoformat() for r in rows],
        states=[s.value for s in ontology.states],
        reference=REFERENCE.name,
        truth=labels,
    )


def household_signals(
    recording: CasasRecording,
    protocol: PriorUncertaintyProtocol,
    samples: Mapping[str, FittedChannels],
    periodic: PeriodicStatePrior,
    *,
    household: str,
    ontology: StateOntology,
) -> dict[str, Any]:
    """The reference predictions, their scores and every signal at the scored windows."""
    base = protocol.base
    resolution = base.resolution
    space = tuple(ontology.states)
    population = samples["all"]
    models = population.models(recording.registry, resolution, HURDLE, ontology)
    counts, rows, labels, moments = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    loglik = {c: models[c].loglik(counts[c]) for c in sorted(models)}
    hours = [moment.hour for moment in moments]
    transitions = periodic_transitions(periodic, ontology, resolution.step)
    start = periodic.probabilities([hours[0]])[0]
    predicted, posterior = periodic_recursion(
        total_loglik(models, counts), transitions, hours, start
    )
    scored = posterior[rows]

    trace = periodic_trace(
        recording,
        ENSEMBLE,
        samples,
        household=household,
        counts=counts,
        rows=rows,
        labels=labels,
        moments=moments,
        transitions=transitions,
        start=start,
        protocol=base,
        ontology=ontology,
    )
    if not np.array_equal(
        trace.posteriors[:, trace.models.index(REFERENCE.name)], scored
    ):
        raise RuntimeError("the ensemble's reference is not the scored recursion")

    mismatch = household_mismatch(
        recording,
        models,
        household=household,
        resolution=resolution,
        ontology=ontology,
        fitted=population,
    )
    if tuple(mismatch.timestamps) != tuple(trace.timestamps) or not (
        np.array_equal(mismatch.truth, trace.truth)
        and np.array_equal(trace.truth, labels)
    ):
        raise RuntimeError("the signals' traces do not align with the scored windows")
    evidence = evidence_discordance(
        loglik,
        counts,
        predicted,
        rows,
        moments,
        states=[s.value for s in space],
        step=resolution.step,
        evidence_floor=base.evidence_floor,
    )
    return {
        "truth": labels,
        "predicted": np.argmax(scored, axis=1),
        "score": _score([space[int(i)] for i in labels], scored, space),
        "signals": {
            CONFIDENCE: scored.max(axis=1),
            ENTROPY: normalised_entropy(scored),
            STRUCTURAL: trace.normalised_discordance,
            EVIDENCE: evidence,
            MISMATCH: mismatch.least_standardised,
        },
        "windows": int(rows.size),
        "channels": [c.name for c in sorted(models)],
        "undefined": {
            EVIDENCE: int(np.isnan(evidence).sum()),
            MISMATCH: int(np.isnan(mismatch.least_standardised).sum()),
        },
    }


# ----------------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PriorUncertaintyResult:
    """A completed experiment: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def run_prior_uncertainty(
    recordings: Mapping[str, CasasRecording],
    protocol: PriorUncertaintyProtocol,
    *,
    fitted_rates: Mapping[str, Any],
    combined_prior: Mapping[str, Any],
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> PriorUncertaintyResult:
    """Run the pre-specified comparison; report nothing unless the checks hold.

    *fitted_rates* and *combined_prior* are the published records, as loaded.
    """
    base = protocol.base
    rates = protocol.combined.base
    homes = sorted({h for fold in base.folds for h in (*fold.train, *fold.test)})
    missing = sorted(set(homes) - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    ontology = StateOntology()
    space = tuple(ontology.states)
    states = [s.value for s in space]
    fits = fold_fits(recordings, rates, ontology)
    check: dict[str, Any] = {
        "fits": check_fits(fits, fitted_rates["results"]["fitted"], protocol.tolerance)
    }
    statistics: dict[str, dict[EvidenceChannel, ChannelStatistics]] = {
        home: home_statistics(
            recordings[home], base.resolution, ontology, household=home
        )
        for home in homes
    }
    scored: dict[str, dict[str, Any]] = {}
    scores: dict[str, PredictionMetrics] = {}
    fold_of: dict[str, str] = {}
    for fold in base.folds:
        samples = fit_samples(
            statistics, fold.train, states=space, pseudo_windows=base.pseudo_windows
        )
        if samples["all"].to_dict() != fits[fold.name].channels.to_dict():
            raise ValueError(
                f"the ensemble's population in {fold.name} is not the fold's fitted "
                "channels; nothing is reported"
            )
        for home in sorted(fold.test):
            entry = household_signals(
                recordings[home],
                protocol,
                samples,
                fits[fold.name].periodic,
                household=home,
                ontology=ontology,
            )
            scores[home] = entry.pop("score")
            scored[home] = entry
            fold_of[home] = fold.name
    check["population"] = {"folds": sorted(fits), "identical": True}
    check["scores"] = check_reference_scores(
        scores,
        combined_prior["household_metrics"][f"{FILTER_PERIODIC_HURDLE}@R"],
        protocol.tolerance,
    )
    scored = dict(sorted(scored.items()))

    data = SelectiveData.from_households(
        states,
        {h: e["truth"] for h, e in scored.items()},
        {h: e["predicted"] for h, e in scored.items()},
        {h: e["signals"][CONFIDENCE] for h, e in scored.items()},
    )
    signals = [
        Signal(
            name,
            data.align({h: e["signals"][name] for h, e in scored.items()}),
            DIRECTIONS[name],
            missing=MISSING[name],
        )
        for name in SIGNALS
    ]
    report = evaluate_signals(
        data,
        signals,
        coverages=base.coverages,
        confidence=base.confidence,
        resamples=base.curve_resamples,
        seed=base.seed,
        bins=base.calibration_bins,
    )
    payload = report.to_dict()
    table = difficult_minority_states(
        data.truth, data.predicted, states, base.minority_share
    )
    difficult = [s for s, e in table.items() if e["difficult_minority"]]
    comparisons = paired_comparisons(payload, base, difficult)
    decisions = {name: comparisons[name]["decision"] for name in comparisons}
    better = [s for s in STRUCTURAL_SIGNALS if decisions[s] == "materially better"]
    declared = protocol.to_dict()
    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified; development panel, which earlier work has inspected",
        "check": check,
        "states": table,
        "difficult_minority_states": difficult,
        "signals": {
            name: {
                "direction": DIRECTIONS[name],
                "missing": MISSING[name],
                "undefined_windows": sum(
                    e["undefined"].get(name, 0) for e in scored.values()
                ),
                "summary": payload["signals"][name]["summary"],
            }
            for name in SIGNALS
        },
        "comparisons": comparisons,
        "rejection_bias": rejection_bias(payload, base),
        "decisions": decisions,
        "key_question": {
            "question": declared["question"],
            "answer": "yes" if better else "no",
            "materially_better": better,
        },
        "rule_signal": rule_signal(comparisons),
        "reproduction": reproduction(payload, None),
        "households": {
            home: {
                "fold": fold_of[home],
                "windows": entry["windows"],
                "channels": entry["channels"],
            }
            for home, entry in scored.items()
        },
    }
    record = ExperimentRecord(
        experiment=protocol.name,
        configuration={**declared, "protocol_sha256": protocol.sha256()},
        inference=ONLINE,
        evidence=online_evidence(recordings, protocol.homes, base.resolution.step),
        seeds=[base.seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions={
            "coverage": "The share of scored windows retained.",
            "error": "The share of retained predictions whose state is wrong.",
            "aurc": "The selective error averaged over the coverage grid by the "
            "trapezoid rule. Lower is better.",
            "gain": "(AURC of random rejection minus AURC) over (AURC of random "
            "rejection minus the oracle's): 1 for the oracle, 0 for random.",
            "balanced_accuracy": "The mean recall over the states among retained "
            "predictions.",
            "calibration_error": "The expected calibration error of the retained "
            "predictions over 10 confidence bins.",
            "state_coverage": "The share of a state's windows retained.",
            "ratio": "A state's retained share over the coverage: 1 for "
            "proportional rejection, below 1 when the state is rejected more.",
            **{name: phase4.DEFINITIONS[name] for name in SIGNALS},
        },
        notes=[
            "Every signal, estimand, minimal difference and decision rule, and the "
            "rule that picks the abstention rule's signal, was declared in the "
            "protocol before any household was scored.",
            "Every signal ranks the same predictions: the combined-prior recursion "
            "with the time prior, cross-fitted on the frozen development folds, "
            "whose scores the run reproduced first.",
            "Uncertainty resamples households, never timestamps.",
            "No threshold or operating coverage is selected.",
            "The development panel has been inspected in earlier work; this is "
            "not a held-out claim.",
        ],
        inputs=list(inputs),
        split={"folds": [{**f.to_dict(), "sha256": f.sha256()} for f in base.folds]},
        preprocessing={
            "moments": "one per step from each recording's first observation to "
            "its last",
            "labels": "truth_series of each recording's annotations; unlabelled "
            "moments are not scored",
            "step_seconds": base.resolution.step.total_seconds(),
        },
        models=[
            ModelRecord(
                spec.name,
                {
                    "assumptions": [a.to_dict() for a in spec.assumptions],
                    "transition": "the time prior's at each window's hour",
                },
                "periodic_trace",
            )
            for spec in ENSEMBLE
        ],
        household_metrics={
            name: {
                home: {
                    key: payload["signals"][name]["households"][home][key]
                    for key in ("aurc", "excess_aurc", "gain")
                }
                for home in scored
            }
            for name in SIGNALS
        },
        intervals=phase4._intervals(comparisons),
        selective_prediction=report,
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return PriorUncertaintyResult(record=record, path=path)


__all__ = [
    "AMENDMENTS_FILE",
    "EXPERIMENT",
    "PINNED_RECORDS",
    "PROTOCOL_FILE",
    "RECORD_FILE",
    "RESULTS_BRANCH",
    "RULE_SIGNAL",
    "RUN_VERSIONS",
    "PriorUncertaintyProtocol",
    "PriorUncertaintyResult",
    "check_frozen_protocol",
    "check_pinned_records",
    "code_changes",
    "code_now",
    "declared_protocol",
    "file_sha256",
    "household_signals",
    "library_changes",
    "periodic_trace",
    "read_amendments",
    "rule_signal",
    "run_prior_uncertainty",
    "write_protocol",
]
