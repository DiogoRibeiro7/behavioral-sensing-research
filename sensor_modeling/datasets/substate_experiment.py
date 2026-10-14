"""Phase 3.3 follow-up: activity sub-states in the recursion, on the development panel.

The posterior predictive checks of the fitted hurdle channels routed the work
to within-state temporal dependence: silence comes in runs far longer than
independence given the state allows, and the active counts mix a low and a
high intensity. The fitted channels lose 0.231 of ``home_active`` recall, and
neither the negative binomial nor the time prior recovers it. This experiment
gives the common states two hidden activity sub-states
(:mod:`~sensor_modeling.datasets.substate_channels`) and asks whether, in the
filter's recursion with the time prior, they recover ``home_active`` recall
without costing balanced accuracy or calibration.

Everything is declared in :class:`SubstateProtocol` and frozen in a committed
file before any household is scored. The folds, the fold fits and the time
prior are the combined-prior protocol's. Before anything is reported, the run
must give back three published results: the fold fits, the reference's scores
in the recursion, and the predictive checks' quiet-run statistic against the
population hurdle.

Models, both scored in the recursion ``R`` on identical windows:

| Model | Channels | Transition |
| --- | --- | --- |
| ``filter_periodic_hurdle`` | fitted hurdle | the time prior's at each window's hour |
| ``filter_periodic_substates`` | fitted hurdle in two sub-states of each common state | the time prior's between states; the sub-state chain within one |
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.households import household_values, recall_of
from ..evaluation.metrics import PredictionMetrics
from ..evaluation.provenance import ExperimentRecord, InputArtifact, ModelRecord
from ..fusion.regime import ONLINE
from ..states.ontology import BehaviouralState, StateOntology
from . import combined_prior_experiment as combined
from . import predictive_checks
from .casas import CasasRecording
from .channel_models import HURDLE, HurdleChannel, total_loglik
from .combined_prior_experiment import (
    FILTER_PERIODIC_HURDLE,
    CombinedPriorProtocol,
    _close,
    check_fits,
    periodic_recursion,
    periodic_transitions,
)
from .history_experiment import Estimand
from .matched_evaluation import MATCHED_METRIC_DEFINITIONS, _finite, online_evidence
from .predictive_checks import (
    POPULATION,
    PredictiveProtocol,
    across_households,
    check_cell,
    log_ratios,
    stretches_of,
)
from .rates_experiment import FoldFit, fold_fits
from .recoverable_gap import FrozenSplits, _cell, _paired, _score
from .substate_channels import (
    HouseholdWindows,
    SubstateChannels,
    SubstateConfig,
    fit_substates,
    household_windows,
    predicted_quiet_runs,
)
from .time_prior_experiment import verdict

PROTOCOL_SCHEMA = "activity-substates-protocol/1"
RESULT_SCHEMA = "activity-substates-results/1"
EXPERIMENT = "phase3-activity-substates"
RECURSION = "R"

S = BehaviouralState
REFERENCE = FILTER_PERIODIC_HURDLE
MODEL = "filter_periodic_substates"
MODELS: dict[str, dict[str, str]] = {
    REFERENCE: {
        "channels": "fitted hurdle, one per channel and state",
        "transition": "the time prior's at each window's hour",
    },
    MODEL: {
        "channels": "fitted hurdle, one per channel and joint state: two "
        "sub-states in each split state, one in every other",
        "transition": "the time prior's between behavioural states; within a "
        "split state, its sub-state chain; a state is entered in its fitted "
        "entry distribution",
    },
}

#: The states given two sub-states: those whose quiet runs or bursts the
#: predictive checks found in excess against each cell's own hurdle.
SPLIT = (S.AWAY, S.HOME_ACTIVE, S.HOME_INACTIVE, S.SLEEPING)
#: The state whose recall the primary rule reads.
FOCUS = S.HOME_ACTIVE
#: The states the mechanism reads: those whose quiet runs the predictive checks
#: found in excess against the population hurdle that inference uses.
MECHANISM_STATES = (S.HOME_ACTIVE, S.HOME_INACTIVE)

COMBINED_PRIOR_PROTOCOL = "artifacts/phase3/combined_prior_protocol.json"
COMBINED_PRIOR_PROTOCOL_SHA256 = (
    "3a0c5cad58187609601181ae841099b9144707ba2327af4173c9442812dc52a5"
)
COMBINED_PRIOR_RECORD = "artifacts/phase3/phase3-combined-prior.json"
COMBINED_PRIOR_RECORD_SHA256 = (
    "57db137f57f5b298054d7a4cfe8c837d854ca9a8793329857d7803401477db59"
)
PREDICTIVE_PROTOCOL = "artifacts/phase3/predictive_protocol.json"
PREDICTIVE_PROTOCOL_SHA256 = (
    "5bbef7ef1dcb417fced2fdafe925ee4548e8e2893bb874da753072d077e72dbd"
)
PREDICTIVE_RECORD = "artifacts/phase3/phase3-hurdle-predictive-checks.json"
PREDICTIVE_RECORD_SHA256 = (
    "024fb929a8c71bc901b05e7758bcc79287860a03142ce601dd1dd7546bf76b34"
)
PINNED_RECORDS = {
    **combined.PINNED_RECORDS,
    COMBINED_PRIOR_PROTOCOL: COMBINED_PRIOR_PROTOCOL_SHA256,
    COMBINED_PRIOR_RECORD: COMBINED_PRIOR_RECORD_SHA256,
    PREDICTIVE_PROTOCOL: PREDICTIVE_PROTOCOL_SHA256,
    PREDICTIVE_RECORD: PREDICTIVE_RECORD_SHA256,
    # Cited in what had been seen.
    "artifacts/phase3/phase3-hurdle-negative-binomial.json": (
        "e07e62e188f17bdb69c0ff1eb1657153bb864e6cb1ed32d01245c5a0cccdfc23"
    ),
    "artifacts/heldout/casas-heldout-confirmation.json": (
        "785f6757ed9e9890dc7dfbf471660403779cc7dcc6a246586a87fd2323e2c97a"
    ),
    "artifacts/contextact/contextact-transfer.json": (
        "1614872031cc35b1607fef79b55bafbe767d91f84d9e8f303e9a8ec5ed34d9b8"
    ),
}
#: The sources whose content the freeze records and the run requires unchanged.
WORKFLOW = ".github/workflows/run-phase3-substates.yml"
PINNED_PACKAGE = "sensor_modeling"
PINNED_FILES = (
    "pyproject.toml",
    "scripts/freeze_phase3_substates.py",
    "scripts/run_phase3_substates.py",
    WORKFLOW,
)
PINNED_DISTRIBUTIONS = ("numpy", "scipy", "pandas")
_REPOSITORY = Path(__file__).resolve().parents[2]
#: Fixes after the freeze, each with its reason and the sources it changes.
AMENDMENTS_FILE = "artifacts/phase3/substates_amendments.json"
#: The branch the one scoring run writes, which the workflow refuses to rewrite.
RESULTS_BRANCH = "run-artifacts/phase3-activity-substates"
#: Households a per-state recall verdict needs.
MIN_HOUSEHOLDS = 5

ESTIMANDS: tuple[Estimand, ...] = (
    Estimand(
        "W1",
        "primary",
        "formulation",
        "what two activity sub-states in the common states add to the reference "
        "formulation, the fitted hurdle channels with the time prior, in the "
        "recursion",
        MODEL,
        RECURSION,
        REFERENCE,
        RECURSION,
    ),
)
#: The metrics whose verdicts the primary rule reads beside the focus recall.
GUARDED = ("balanced_accuracy", "log_loss", "calibration_error")
#: The other common states, whose recall the primary rule also guards.
GUARDED_STATES = (S.AWAY, S.HOME_INACTIVE, S.SLEEPING)

INSPECTED_BEFORE = (
    "The combined-prior record: in the recursion, the fitted hurdle channels "
    "with the time prior score a mean household balanced accuracy of 0.517 "
    "against 0.456 without the prior, K1 +0.061 [+0.044, +0.077], a success; "
    "K1's home_active recall difference is +0.002 [−0.002, +0.007], "
    "negligible, with home_active recall 0.268 with the prior and 0.266 "
    "without it.",
    "The fitted-rates record: in the recursion the fitted hurdle channels lose "
    "0.231 [0.172, 0.290] of home_active recall to the declared channels (FR).",
    "The predictive-check record: against each cell's own hurdle, windows in "
    "quiet runs of at least an hour are 1.336 times the predicted in away, "
    "5.187 in home_active, in the 15 households with enough runs, and 2.568 in "
    "home_inactive, each positive, and 1.116 in sleeping, negligible; bursts "
    "are positive in home_inactive, 1.596, and sleeping, 2.896, and negative "
    "in bathroom_activity and kitchen_activity; against the fold's population "
    "hurdle, quiet runs are 6.173 times the predicted in home_active, in 15 "
    "households, and 2.584 in home_inactive, positive, and negligible in away "
    "and sleeping; the own-reference quiet-run excess is positive on every "
    "channel with enough households, from 1.410 on bathroom motion to 2.561 on "
    "living-room motion; the active count's variance is 4.342 to 7.418 times "
    "the model's in the common states, above one in every household with a "
    "value; and its page describes the active counts as having too many "
    "single activations, too few mid-range counts and far too many large ones.",
    "The negative-binomial record: against the hurdle, the zero-truncated "
    "negative binomial's home_active recall difference is −0.055 [−0.087, "
    "−0.024] with current windows, favouring the reference, and +0.029 "
    "[−0.004, +0.063] in the recursion, uncertain; it was not adopted.",
    "The held-out record: on 43 held-out CASAS homes, K1's home_active recall "
    "difference is −0.000 [−0.003, +0.002], negligible.",
    "The ContextAct transfer record: for one independent resident, the time "
    "prior's gain held, K1 +0.049 [+0.013, +0.087], a success; with the "
    "prior, the CASAS-fitted hurdle channels lost 0.149 [0.076, 0.221] of "
    "balanced accuracy to the declared channels, with better log loss, K3 "
    "inconclusive.",
    "No sub-state model had been fitted to, or run on, any CASAS household "
    "before this protocol was frozen; the code was exercised on simulated "
    "homes only.",
)

FAMILY = {
    "chosen": "activity sub-states: each split state has two hidden "
    "sub-states, quiet and busy, each with its own hurdle parameters on every "
    "channel, and a Markov chain between them within the state",
    "why_one_family": [
        "The time prior's hour-dependent transition stays on the behavioural "
        "layer, so a sub-state's exit from its state is the prior's whichever "
        "sub-state it leaves. A Markov-modulated emission whose modulating chain "
        "is shared by the channels is then the same model as activity "
        "sub-states, and the two named families differ only in whether the "
        "modulation is shared or per channel.",
        "Shared, not per channel: the predictive checks found the quiet-run "
        "excess on every channel with enough households, and the same mixture "
        "of a low and a high intensity in every household's active counts, "
        "which one level of activity shared by the channels produces; they did "
        "not examine dependence between channels, so nothing in them asks for a "
        "chain per channel. A chain per channel would be a factorial model, 64 "
        "sub-states per state with six channels.",
    ],
    "split_states": "the states whose quiet runs or bursts the predictive "
    "checks found positive against each cell's own hurdle",
    "not_split": "the short, busy states bathroom_activity and "
    "kitchen_activity, whose bursts are fewer than independence predicts, and "
    "bed_awake, with too few homes",
    "output": "the belief in a behavioural state is the sum over its sub-states",
    "exactness": "with each split state's two sub-states sharing the hurdle's "
    "parameters the model is exactly the reference, whatever its chain; tested "
    "to 1e-10 on simulated homes",
    "not_tested": [
        "exits that depend on the sub-state, which would replace the time "
        "prior's transition rather than add to it",
        "a modulating chain per channel",
        "more than two sub-states",
    ],
}


def substate_conclusion(
    verdicts: Mapping[str, str], recall: str, others: Mapping[str, str]
) -> str:
    """Success, trade-off, failure or inconclusive, by the declared rule.

    *recall* is the verdict on the focus state's recall, and *others* those on
    the other common states' recall.
    """
    if recall in ("negligible", "favours reference"):
        return "failure"
    if recall == "favours model":
        guards = [verdicts[m] for m in GUARDED] + list(others.values())
        if all(v != "favours reference" for v in guards):
            return "success"
        return "trade-off"
    return "inconclusive"


#: How a mechanism verdict reads: the household mean of how much closer to
#: the observed quiet runs the model's prediction is than the reference's.
MECHANISM_READING = {
    "positive": "falls",
    "negative": "rises",
    "negligible": "unchanged",
    "uncertain": "uncertain",
    "insufficient": "insufficient",
}


def mechanism_conclusion(readings: Mapping[str, str]) -> str:
    """Whether the excess of long quiet runs falls, over the mechanism states."""
    values = list(readings.values())
    if values and all(v == "falls" for v in values):
        return "falls"
    if any(v == "falls" for v in values):
        return "falls in part"
    if values and all(v in ("unchanged", "rises") for v in values):
        return "does not fall"
    return "inconclusive"


@dataclass(frozen=True)
class SubstateProtocol:
    """Everything the experiment fixes before any household is scored.

    Attributes
    ----------
    base
        The combined-prior protocol, whose folds, fold fits, time prior,
        metrics, minimal differences and bootstrap are used unchanged.
    predictive
        The predictive-check protocol, whose cells, quiet-run statistic,
        replicates, seeds and household reading the mechanism uses.
    substates
        The sub-state model's declaration.
    tolerance
        How far a reproduced figure may lie from the published one before
        nothing is reported.
    """

    base: CombinedPriorProtocol
    predictive: PredictiveProtocol
    substates: SubstateConfig = field(default_factory=lambda: SubstateConfig(SPLIT))
    tolerance: float = 1e-6
    inspected_before: tuple[str, ...] = INSPECTED_BEFORE
    name: str = EXPERIMENT

    def __post_init__(self) -> None:
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if FOCUS not in self.substates.split:
            raise ValueError(f"{FOCUS.value} must be split")
        if not set(MECHANISM_STATES) <= set(self.substates.split):
            raise ValueError("every mechanism state must be split")
        homes = self.base.base.homes
        if self.predictive.homes != homes:
            raise ValueError("the predictive checks must cover the same households")

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        base = self.base.base
        predictive = self.predictive
        step = base.information_sets[0].resolution.step.total_seconds()
        return {
            "schema": PROTOCOL_SCHEMA,
            "name": self.name,
            "status": "pre-specified, on the development panel, which earlier work "
            "has inspected; not a held-out claim",
            "question": "whether two activity sub-states in the common states, "
            "entering the filter's recursion with the time prior, recover "
            "home_active recall without costing balanced accuracy, log loss or "
            "calibration",
            "inspected_before": list(self.inspected_before),
            "base_protocol": {
                "file": COMBINED_PRIOR_PROTOCOL,
                "sha256": COMBINED_PRIOR_PROTOCOL_SHA256,
                "protocol_sha256": self.base.sha256(),
                "used": "its folds; its fold fits of the hurdle channels and the "
                "time prior on training households only, which are the "
                "fitted-rates protocol's; its recursion with the time prior; its "
                "metrics, minimal differences and household bootstrap",
            },
            "predictive_protocol": {
                "file": PREDICTIVE_PROTOCOL,
                "sha256": PREDICTIVE_PROTOCOL_SHA256,
                "protocol_sha256": predictive.sha256(),
                "used": "its cells, stretches, quiet-run statistic, replicates, "
                "seeds, sufficiency rule, minimal ratio and household reading",
            },
            "households": {
                "folds": [
                    {**fold.to_dict(), "sha256": fold.sha256()} for fold in base.folds
                ],
                "splits_sha256": base.splits_sha256,
            },
            "family": FAMILY,
            "substates": self.substates.to_dict(),
            "fitting": {
                "unit": "one fit per fold and split state, on the labelled windows "
                "of the fold's training households, in stretches of consecutive "
                "labelled windows of the state",
                "e_step": "forward-backward within each stretch, each stretch "
                "entered from the entry distribution",
                "m_step": "each sub-state's hurdle is the base protocol's hurdle "
                "fit with every window weighted by its sub-state probability, "
                f"with the same {base.pseudo_windows:g} pseudo-windows toward the "
                "declared model; the chain and the entry are the expected "
                "transition and entry counts, each with the pseudo-count",
                "start": "both sub-states start from the fold's fitted hurdle "
                "rates; the quiet one with each channel's odds of silence "
                "multiplied by the odds factor, the busy one divided by it; the "
                "chain with the persistence on its diagonal; the entry even",
                "iterations": "all of them, with no stopping rule; the training "
                "log-likelihood at each E-step is recorded",
                "no_window": "a split state with no training window keeps two "
                "sub-states with the hurdle's parameters",
                "unsplit_states": "the fold's fitted hurdle, unchanged",
                "entry": "estimated at the first window of every stretch, which "
                "is an entry from another state or a restart after an unlabelled "
                "gap",
                "names": "after fitting, the sub-state whose silence probability, "
                "averaged over the channels, is higher is called quiet; the names "
                "change no belief",
                "collapse": "nothing stops a sub-state from holding few windows; it "
                "then keeps its own pseudo-windows' pull toward the declared model "
                "and still enters inference; each sub-state's share of the "
                "training windows is reported",
                "convergence": "the change in the training log-likelihood over the "
                "last iteration is reported, not used; with the pseudo-windows the "
                "M-step does not maximise that likelihood, so it need not rise at "
                "every iteration",
                "settings": "the odds factor, the persistence, a sub-state lasting "
                "50 minutes on average, and the iterations were set before any "
                "CASAS household was fitted, and were not selected on data",
            },
            "recursion": {
                "windows": f"every window of the recording at the base step of "
                f"{step:g} seconds; both models receive the same windows, and "
                "only labelled windows are scored",
                "transition": "into each window, the time prior's at the local "
                "hour at the window's end, the population prior for every "
                "held-out household, expanded to the joint states: within a "
                "split state, the prior's probability of staying times the "
                "sub-state chain; into another state, the prior's probability "
                "times that state's entry distribution",
                "start": "one step before the first window, the prior's "
                "distribution at that window's hour, each split state in its "
                "entry distribution",
                "joint_states": "11: two for each of the four split states and "
                "one for each of the three others",
                "scored": "the belief in each behavioural state, the sum over its "
                "sub-states, scored by its most probable state",
                "regime": "online: no estimate reads a window after its own",
            },
            "runtime": "measured on simulated homes before the freeze: EM costs "
            "about 1.4 microseconds per training window and iteration, under 20 "
            "seconds per fold for the roughly 110,000 training windows a fold "
            "has in the split states; the recursion over 11 joint states costs "
            "about 14 microseconds per window",
            "models": MODELS,
            "estimands": [
                {**estimand.to_dict(), "rule": "activity sub-states"}
                for estimand in ESTIMANDS
            ],
            "criteria": {
                "verdicts": "favours model: mean >= delta and lower bound > 0; "
                "favours reference: mean <= -delta and upper bound < 0; negligible: "
                "the whole interval within (-delta, delta); uncertain: anything else",
                "activity sub-states": f"success when {FOCUS.value} recall "
                "favours the model and none of balanced accuracy, log loss, "
                "calibration error and the recall of "
                + ", ".join(s.value for s in GUARDED_STATES)
                + " favours the reference; trade-off when "
                f"{FOCUS.value} recall favours the model and at least one of "
                "them favours the reference; failure when "
                f"{FOCUS.value} recall is negligible or favours the reference; "
                "inconclusive otherwise",
                "primary": "W1 alone decides whether the sub-states replace the "
                "hurdle channels as the reference formulation on the development "
                "panel: only a success does; a trade-off does not; a success "
                "needs an independent dataset not yet scored with this family to "
                "be confirmed",
                "multiplicity": "one primary; the mechanism and per-state recall "
                "are read by their own rules and change no conclusion of W1",
            },
            "per_state_recall": {
                "estimands": [e.key for e in ESTIMANDS],
                "states": "every state",
                "verdict": "the same verdicts with the recall minimal difference; "
                f"insufficient with fewer than {MIN_HOUSEHOLDS} households in "
                "which the state is labelled",
            },
            "mechanism": {
                "key": "M1",
                "question": "whether the excess of long quiet runs falls",
                "statistic": "the predictive checks' quiet_runs: windows in runs of "
                f"at least {predictive.quiet_run_windows} silent windows within "
                "stretches, observed over the mean of "
                f"{predictive.replicates} replicates of the cell's windows",
                "reference": "the fold's population hurdle, drawn exactly as the "
                "predictive checks drew it, with their seeds",
                "model": "the fold's sub-state model: each replicate draws the "
                "sub-state path of every stretch from the entry distribution and "
                "the chain, then each window's silence from its sub-state",
                "cells": "the cells whose quiet-run statistic the predictive checks "
                "judged against the population hurdle",
                "household_value": "for each mechanism state, the mean over a "
                "household's cells of the absolute value of the reference's log "
                "ratio minus that of the model's; positive means the model's "
                "prediction is nearer the observed long quiet runs, whichever "
                "side of them it lies",
                "states": "the states whose quiet runs the predictive checks "
                "found positive against the population hurdle: "
                + " and ".join(s.value for s in MECHANISM_STATES),
                "reading": "the predictive checks' household reading, with the "
                f"minimal ratio {predictive.minimal_ratio:g}: positive is falls, "
                "negative rises, negligible unchanged",
                "conclusion": "falls when it falls in every mechanism state; falls "
                "in part when it falls in some; does not fall when it is "
                "unchanged or rises in every one; inconclusive otherwise",
                "described": "each split state's remaining log ratio under the "
                "model and the reference, signed and read the same way, is "
                "described beside it",
            },
            "one_run": {
                "runs": "one scoring run, dispatched on develop",
                "branch": f"the run writes {RESULTS_BRANCH}, and the workflow "
                "refuses to run when that branch exists",
                "sources": f"the freeze records the digest of every file of the "
                f"{PINNED_PACKAGE} package and of "
                + ", ".join(PINNED_FILES)
                + ", and the versions of "
                + ", ".join(PINNED_DISTRIBUTIONS)
                + "; the run refuses when a source differs",
                "crash": "a run that stops before writing its record reports "
                "nothing; it is fixed without looking at any outcome, the fix is "
                "declared as an amendment, and it is run again",
                "amendments": "a change to a pinned source after the freeze, a fix "
                "or a change merged on develop, is declared in "
                f"{AMENDMENTS_FILE} before the run, with its reason and the digest "
                "of every source it changes; the run accepts a changed source "
                "only at the digest an amendment gives, and the record lists the "
                "amendments as an input",
            },
            "metrics": list(base.metrics),
            "minimal_differences": dict(base.minimal_differences),
            "bootstrap": {
                "unit": "household",
                "statistics": "mean and median paired differences",
                "resamples": base.resamples,
                "confidence": base.confidence,
                "seed": base.seed,
                "interval": "percentile",
            },
            "the_check": {
                "what": [
                    "every fold's fitted channels and time prior equal the "
                    "fitted-rates record's, parameter by parameter",
                    f"every household's scores of {REFERENCE} in the recursion "
                    "equal the combined-prior record's",
                    "every cell's quiet-run statistic against the population "
                    "hurdle equals the predictive-check record's",
                ],
                "tolerance": self.tolerance,
                "otherwise": "nothing is reported, not even per household",
            },
            "pinned_records": dict(PINNED_RECORDS),
            "what_this_cannot_show": [
                "A held-out claim: the 20 homes have been inspected in earlier "
                "work, and the 43 held-out CASAS homes are used up for "
                "confirmation. A success here needs an independent dataset not "
                "yet scored with this family to be confirmed.",
                "Which within-state dependence the data have: a success shows "
                "that two shared sub-states help, not that the dependence is "
                "shared by the channels or has two levels.",
                "Uncertainty in the fits: each fold's ten training homes share "
                "one fit, and the household bootstrap resamples the scored homes "
                "only, so the intervals leave out how the fits would vary.",
                "Selection: the family, the split states and the mechanism states "
                "were chosen from the predictive checks on these same 20 homes, "
                "so M1 is not an independent test of the mechanism.",
                "Transfer: the sub-states add parameters to channels that did not "
                "transfer to the one independent resident of ContextAct, so a "
                "gain here may not survive a change of homes, sensors or "
                "annotation.",
                "Dependence between channels: given the joint state, the channels "
                "are still independent.",
                "Anything about TIHM or the online pipeline, whose defaults this "
                "does not change.",
            ],
            "tuning": "none: every setting is the base protocols' or declared here",
        }

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
        raw = (Path(root) / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"{name} is not the file the protocol pins")


def declared_protocol(splits: FrozenSplits, root: Path = Path(".")) -> SubstateProtocol:
    """The protocol as declared, on the frozen combined-prior and predictive protocols.

    Refuses when a base protocol is not the one frozen in the repository at
    *root*, or a pinned record is not the file pinned.
    """
    base = combined.declared_protocol(splits, root)
    combined.check_frozen_protocol(base, Path(root) / COMBINED_PRIOR_PROTOCOL)
    predictive = predictive_checks.declared_protocol(splits)
    predictive_checks.check_frozen_protocol(
        predictive, Path(root) / PREDICTIVE_PROTOCOL
    )
    check_pinned_records(root)
    return SubstateProtocol(base, predictive)


def file_sha256(path: Path) -> str:
    """SHA-256 of a text file, whatever its line endings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def code_now(root: Path | None = None) -> dict[str, Any]:
    """The digest of every pinned source file, and the libraries' versions, now."""
    root = _REPOSITORY if root is None else Path(root)
    package = sorted(
        path.relative_to(root).as_posix()
        for path in (root / PINNED_PACKAGE).rglob("*.py")
        if "__pycache__" not in path.parts
    )
    return {
        "sources": {
            name: file_sha256(root / name) for name in (*package, *PINNED_FILES)
        },
        "distributions": {
            "python": platform.python_version(),
            **{name: metadata.version(name) for name in PINNED_DISTRIBUTIONS},
        },
    }


def read_amendments(path: Path) -> list[dict[str, Any]]:
    """The declared amendments, or none when the file does not exist.

    Each amendment names its ``reason`` and, in ``sources``, the digest of
    every pinned source it changes, ``None`` for one it removes.
    """
    if not Path(path).exists():
        return []
    amendments = json.loads(Path(path).read_text(encoding="utf-8"))["amendments"]
    for amendment in amendments:
        if not str(amendment.get("reason", "")).strip():
            raise ValueError("every amendment needs its reason")
        if not isinstance(amendment.get("sources"), dict) or not amendment["sources"]:
            raise ValueError("every amendment names the sources it changes")
    return list(amendments)


def code_changes(
    path: Path,
    root: Path | None = None,
    amendments: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    """The pinned sources that differ now from the freeze and its amendments.

    A source an amendment changes is expected at the digest the amendment
    gives, the last amendment's when several change it.
    """
    frozen = json.loads(Path(path).read_text(encoding="utf-8")).get("at_freeze")
    if frozen is None:
        raise ValueError(f"{path} does not record the code it was frozen against")
    now = code_now(root)["sources"]
    then = dict(frozen["sources"])
    for amendment in amendments:
        for name, digest in amendment["sources"].items():
            if digest is None:
                then.pop(name, None)
            else:
                then[name] = digest
    return sorted(
        name for name in set(then) | set(now) if then.get(name) != now.get(name)
    )


def write_protocol(
    protocol: SubstateProtocol, path: Path, root: Path | None = None
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


def check_frozen_protocol(protocol: SubstateProtocol, path: Path) -> str:
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
# The checks
# ----------------------------------------------------------------------------
def check_reference_scores(
    scores: Mapping[str, PredictionMetrics],
    published: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    """Refuse reference scores that are not the combined-prior record's."""
    differing = []
    identical = True
    for home, metrics in scores.items():
        mine = json.loads(json.dumps(metrics.to_dict()))
        if home not in published or not _close(mine, published[home], tolerance):
            differing.append(home)
        else:
            identical = identical and mine == published[home]
    if differing or set(published) != set(scores):
        raise ValueError(
            f"{len(differing)} reference scores differ from the combined-prior "
            "record's; nothing is reported"
        )
    return {
        "model": REFERENCE,
        "households": len(scores),
        "reproduced": True,
        "identical": identical,
    }


def check_windows(
    scores: Mapping[str, Mapping[str, PredictionMetrics]],
) -> dict[str, Any]:
    """Refuse unless both models scored the same labelled windows in every home.

    The number of windows and the labels' counts, the confusion matrix's rows,
    must agree.
    """
    differing = []
    for home, reference in scores[REFERENCE].items():
        model = scores[MODEL].get(home)
        if model is None:
            differing.append(home)
            continue
        mine: dict[str, Any] = json.loads(json.dumps(model.to_dict()))
        theirs: dict[str, Any] = json.loads(json.dumps(reference.to_dict()))
        rows = [
            [sum(row) for row in entry["confusion"]["counts"]]
            for entry in (mine, theirs)
        ]
        if mine["n"] != theirs["n"] or rows[0] != rows[1]:
            differing.append(home)
    if differing or set(scores[MODEL]) != set(scores[REFERENCE]):
        raise ValueError(
            f"the models scored different windows in {len(differing)} homes; "
            "nothing is reported"
        )
    return {"households": len(scores[REFERENCE]), "same_windows": True}


def check_runs(
    cells: Mapping[str, Sequence[Mapping[str, Any]]],
    published: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    """Refuse quiet-run statistics that are not the predictive-check record's."""
    differing = 0
    identical = True
    count = 0
    for home, mine in cells.items():
        theirs = {
            (c["state"], c["channel"]): c["references"][POPULATION]["statistics"].get(
                "quiet_runs"
            )
            for c in published.get(home, {}).get("cells", [])
        }
        ours = {(c["state"], c["channel"]): c["reference"] for c in mine}
        if set(ours) != set(theirs):
            differing += 1
            continue
        for key, value in ours.items():
            count += 1
            entry = json.loads(json.dumps(value))
            if not _close(entry, theirs[key], tolerance):
                differing += 1
            else:
                identical = identical and entry == theirs[key]
    if differing or set(cells) != set(published):
        raise ValueError(
            f"{differing} quiet-run statistics differ from the predictive-check "
            "record's; nothing is reported"
        )
    return {
        "statistic": "quiet_runs",
        "cells": count,
        "reproduced": True,
        "identical": identical,
    }


# ----------------------------------------------------------------------------
# Scoring one household
# ----------------------------------------------------------------------------
def household_cells(
    data: HouseholdWindows,
    fit: FoldFit,
    model: SubstateChannels,
    protocol: SubstateProtocol,
    *,
    index: int,
    ontology: StateOntology,
) -> list[dict[str, Any]]:
    """Each cell's quiet-run statistic against the population hurdle and the model."""
    predictive = protocol.predictive
    states = tuple(ontology.states)
    resolution = predictive.resolution
    population = fit.channels.models(data.registry, resolution, HURDLE, ontology)
    by_substate = model.models(data.registry, resolution, ontology)
    cells: list[dict[str, Any]] = []
    for s, state in enumerate(states):
        chosen = data.rows[data.labels == s]
        if chosen.size < predictive.min_windows:
            continue
        stretch = stretches_of(chosen)
        for channel in sorted(data.counts):
            c = resolution.channels.index(channel)
            values = data.counts[channel][chosen]
            hurdle = population[channel]
            if not isinstance(hurdle, HurdleChannel):  # pragma: no cover - by family
                raise TypeError("the population family must be the hurdle")
            rng = np.random.default_rng([predictive.seed, index, s, c, 1])
            checked = check_cell(
                values,
                stretch,
                POPULATION,
                (float(hurdle.silence[s]), float(hurdle.rate[s])),
                predictive,
                rng,
            )
            reference = checked["statistics"]["quiet_runs"]
            cell: dict[str, Any] = {
                "state": state.value,
                "channel": channel.name,
                "windows": int(values.size),
                "reference": reference,
            }
            if state in model.split:
                silence = np.array(
                    [float(models[channel].silence[s]) for models in by_substate]
                )
                predicted = predicted_quiet_runs(
                    np.random.default_rng([predictive.seed, index, s, c, 2]),
                    silence,
                    model.entry[state],
                    model.within[state],
                    stretch,
                    predictive.replicates,
                    predictive.quiet_run_windows,
                )
                ratio = float(
                    log_ratios(
                        "quiet_runs",
                        np.array([reference["observed"]]),
                        np.array([predicted]),
                    )[0]
                )
                cell["model"] = {"predicted": predicted, "log_ratio": ratio}
            cells.append(cell)
    return cells


def _judged(cell: Mapping[str, Any]) -> bool:
    return (
        "model" in cell
        and cell["reference"].get("log_ratio") is not None
        and cell["reference"].get("verdict") != "not estimable"
    )


def mechanism(
    cells: Mapping[str, Sequence[Mapping[str, Any]]],
    protocol: SubstateProtocol,
) -> dict[str, Any]:
    """M1, and each split state's remaining excess, from the households' cells."""

    def values(state: str, which: str) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for home, entries in sorted(cells.items()):
            chosen = [c for c in entries if c["state"] == state and _judged(c)]
            if not chosen:
                continue
            if which == "fall":
                ratios = [
                    abs(c["reference"]["log_ratio"]) - abs(c["model"]["log_ratio"])
                    for c in chosen
                ]
            else:
                ratios = [c[which]["log_ratio"] for c in chosen]
            out[home] = float(np.mean(ratios))
        return out

    states: dict[str, Any] = {}
    for state in MECHANISM_STATES:
        summary = across_households(values(state.value, "fall"), protocol.predictive)
        states[state.value] = {
            **summary,
            "reading": MECHANISM_READING[summary["verdict"]],
        }
    readings = {key: entry["reading"] for key, entry in states.items()}
    remaining = {
        state.value: {
            which: across_households(values(state.value, which), protocol.predictive)
            for which in ("reference", "model")
        }
        for state in protocol.substates.split
    }
    return {
        "key": "M1",
        "states": states,
        "conclusion": mechanism_conclusion(readings),
        "remaining": remaining,
    }


# ----------------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------------
def _estimand_entry(
    estimand: Estimand,
    scores: Mapping[str, Mapping[str, PredictionMetrics]],
    protocol: SubstateProtocol,
    states: tuple[BehaviouralState, ...],
) -> dict[str, Any]:
    base = protocol.base.base
    gap = base.as_gap_protocol()
    comparisons = {
        metric: _paired(
            household_values(scores[estimand.model], metric),
            household_values(scores[estimand.reference], metric),
            metric,
            gap,
        )
        for metric in base.metrics
    }
    verdicts = {
        metric: verdict(comparison, base.minimal_differences[metric])
        for metric, comparison in comparisons.items()
    }
    recall: dict[str, Any] = {}
    for state in states:
        comparison = _paired(
            household_values(scores[estimand.model], recall_of(state)),
            household_values(scores[estimand.reference], recall_of(state)),
            "balanced_accuracy",
            gap,
        )
        recall[state.value] = {
            "comparison": comparison,
            "verdict": (
                verdict(comparison, base.minimal_differences["recall"])
                if comparison["n"] >= MIN_HOUSEHOLDS
                else "insufficient"
            ),
        }
    focus = recall[FOCUS.value]["verdict"]
    others = {state.value: recall[state.value]["verdict"] for state in GUARDED_STATES}
    return {
        **estimand.to_dict(),
        "rule": "activity sub-states",
        "comparisons": comparisons,
        "verdicts": verdicts,
        "focus": {"state": FOCUS.value, "verdict": focus},
        "guarded_states": others,
        "per_state_recall": recall,
        "conclusion": substate_conclusion(verdicts, focus, others),
    }


@dataclass(frozen=True)
class SubstateResult:
    """A completed experiment: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def run_substates(
    recordings: Mapping[str, CasasRecording],
    protocol: SubstateProtocol,
    *,
    fitted_rates: Mapping[str, Any],
    combined_prior: Mapping[str, Any],
    predictive: Mapping[str, Any],
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> SubstateResult:
    """Run the pre-specified experiment; report nothing unless the checks hold.

    *fitted_rates*, *combined_prior* and *predictive* are the published
    records, as loaded.
    """
    base = protocol.base.base
    missing = sorted(set(base.homes) - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    ontology = StateOntology()
    states = tuple(ontology.states)
    resolution = base.information_sets[0].resolution
    fits = fold_fits(recordings, base, ontology)
    check: dict[str, Any] = {
        "fits": check_fits(fits, fitted_rates["results"]["fitted"], protocol.tolerance)
    }
    windows = {
        home: household_windows(recordings[home], resolution, ontology, household=home)
        for home in base.homes
    }
    models = {
        fold.name: fit_substates(
            {home: windows[home] for home in fold.train},
            fits[fold.name].channels,
            protocol.substates,
        )
        for fold in base.folds
    }
    order = {home: i for i, home in enumerate(protocol.predictive.homes)}
    scores: dict[str, dict[str, PredictionMetrics]] = {REFERENCE: {}, MODEL: {}}
    cells: dict[str, list[dict[str, Any]]] = {}
    folds: dict[str, str] = {}
    for fold in base.folds:
        fit, model = fits[fold.name], models[fold.name]
        timed = periodic_transitions(fit.periodic, ontology, resolution.step)
        joint = model.transitions(timed)
        aggregation = model.aggregation()
        for home in sorted(fold.test):
            folds[home] = fold.name
            data = windows[home]
            truth = [states[int(i)] for i in data.labels]
            start = fit.periodic.probabilities([data.hours[0]])[0]
            hurdle = fit.channels.models(data.registry, resolution, HURDLE, ontology)
            _, posterior = periodic_recursion(
                total_loglik(hurdle, data.counts), timed, data.hours, start
            )
            scores[REFERENCE][home] = _score(truth, posterior[data.rows], states)
            _, posterior = periodic_recursion(
                model.loglik(data.registry, resolution, data.counts, ontology),
                joint,
                data.hours,
                model.start(start),
            )
            scores[MODEL][home] = _score(
                truth, (posterior @ aggregation)[data.rows], states
            )
            cells[home] = household_cells(
                data, fit, model, protocol, index=order[home], ontology=ontology
            )
    check["scores"] = check_reference_scores(
        scores[REFERENCE],
        combined_prior["household_metrics"][f"{REFERENCE}@{RECURSION}"],
        protocol.tolerance,
    )
    check["windows"] = check_windows(scores)
    check["runs"] = check_runs(
        cells, predictive["results"]["households"], protocol.tolerance
    )
    gap = base.as_gap_protocol()
    estimands = [_estimand_entry(e, scores, protocol, states) for e in ESTIMANDS]
    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified; development panel, which earlier work has inspected",
        "check": check,
        "states": [state.value for state in states],
        "folds": dict(sorted(folds.items())),
        "cells": [_cell(name, RECURSION, scores[name], states, gap) for name in MODELS],
        "estimands": estimands,
        "mechanism": mechanism(cells, protocol),
        "households": {home: {"cells": cells[home]} for home in sorted(cells)},
        "fitted": {
            fold: {**model.to_dict(), "sha256": model.sha256()}
            for fold, model in models.items()
        },
        "conclusion": estimands[0]["conclusion"],
    }
    record = ExperimentRecord(
        experiment=protocol.name,
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        inference=ONLINE,
        evidence=online_evidence(recordings, base.homes, resolution.step),
        seeds=sorted({base.seed, protocol.predictive.seed}),
        results=_finite(results),
        data_source=data_source,
        metric_definitions=MATCHED_METRIC_DEFINITIONS,
        notes=[
            "Every setting, estimand and criterion was declared in the protocol "
            "before any household was scored.",
            "The fold fits are the fitted-rates protocol's, on training households "
            "only; the sub-states are fitted on the same households, and the "
            "reference and the quiet-run statistic reproduce the published "
            "records.",
            "Verdicts compare effect sizes and household bootstrap intervals with "
            "declared minimal differences; they are not significance tests.",
        ],
        inputs=list(inputs),
        split={"folds": [{**f.to_dict(), "sha256": f.sha256()} for f in base.folds]},
        models=[
            ModelRecord(REFERENCE, MODELS[REFERENCE], "periodic_recursion"),
            ModelRecord(MODEL, MODELS[MODEL], "fit_substates, periodic_recursion"),
        ],
        household_metrics={
            f"{name}@{RECURSION}": {h: m.to_dict() for h, m in per_home.items()}
            for name, per_home in scores.items()
        },
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return SubstateResult(record=record, path=path)


__all__ = [
    "ESTIMANDS",
    "MECHANISM_STATES",
    "MODEL",
    "PINNED_RECORDS",
    "REFERENCE",
    "SPLIT",
    "SubstateProtocol",
    "SubstateResult",
    "check_frozen_protocol",
    "check_pinned_records",
    "check_reference_scores",
    "check_runs",
    "check_windows",
    "code_changes",
    "read_amendments",
    "code_now",
    "declared_protocol",
    "household_cells",
    "mechanism",
    "mechanism_conclusion",
    "run_substates",
    "substate_conclusion",
    "write_protocol",
]
