"""Phase 4: an abstention rule for the fitted channels with the time prior.

Step 1, Phase 4 for the time prior, found structural disagreement materially
better than confidence and picked it as the rule's signal by a rule declared
before scoring. It also found that the signal keeps less than confidence of
``bathroom_activity`` and ``kitchen_activity``. This experiment selects the
rule on the development panel: a coverage target, raised until a guard keeps
``away`` and the two activity states at a declared fraction of their
proportional share, and the threshold on the signal that gives it.

Selection is cross-fitted. For each fold, the threshold is chosen on the
fold's training homes, each scored by models fitted on the fold's other
training homes, so the choice reads nothing of the fold's held-out homes; it
is then applied to the held-out homes, scored as step 1 scored them. The rule
that is frozen chooses its threshold the same way on all 20 homes, each scored
by models fitted on the other 19.
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

from ..evaluation.households import compare_households
from ..evaluation.provenance import ExperimentRecord, InputArtifact
from ..fusion.regime import ONLINE
from ..states.ontology import BehaviouralState, StateOntology
from . import pinning
from . import prior_uncertainty_experiment as step1
from .casas import CasasRecording
from .channel_models import (
    HURDLE,
    home_statistics,
    household_channel_counts,
    total_loglik,
)
from .combined_prior_experiment import (
    check_fits,
    periodic_recursion,
    periodic_transitions,
)
from .matched_evaluation import _finite, _regular_moments, online_evidence
from .periodic_prior import fit_periodic_prior, hour_state_counts
from .pinning import file_sha256, read_amendments
from .prior_uncertainty_experiment import PriorUncertaintyProtocol, periodic_trace
from .rates_experiment import fold_fits
from .recoverable_gap import FrozenSplits, _score
from .structural_models import fit_samples
from .substate_experiment import check_reference_scores
from .time_prior_experiment import verdict
from .uncertainty_experiment import ENSEMBLE, STRUCTURAL

PROTOCOL_SCHEMA = "abstention-protocol/1"
RESULT_SCHEMA = "abstention-results/1"
EXPERIMENT = "phase4-abstention-rule"
PROTOCOL_FILE = "artifacts/phase4/abstention_protocol.json"
RECORD_FILE = f"artifacts/phase4/{EXPERIMENT}.json"
AMENDMENTS_FILE = "artifacts/phase4/abstention_amendments.json"
RESULTS_BRANCH = f"run-artifacts/{EXPERIMENT}"
WORKFLOW = ".github/workflows/run-abstention-rule.yml"
PINNED_FILES = (
    "pyproject.toml",
    "scripts/freeze_abstention_rule.py",
    "scripts/run_abstention_rule.py",
    WORKFLOW,
)
PINNED_DISTRIBUTIONS = step1.PINNED_DISTRIBUTIONS
RUN_VERSIONS = step1.RUN_VERSIONS
_REPOSITORY = Path(__file__).resolve().parents[2]

STEP1_PROTOCOL = step1.PROTOCOL_FILE
STEP1_RECORD = step1.RECORD_FILE
PINNED_RECORDS = {
    **step1.PINNED_RECORDS,
    STEP1_PROTOCOL: "0cd450ef7ba3b5058da025d76a637983e48b2722b20d1b9b62403f2349dc316a",
    STEP1_RECORD: "99addb681e893df0a988156ddb23056c7aa7fc145471501f3def895591f48022",
}

S = BehaviouralState
SIGNAL = STRUCTURAL
GUARDED = (S.AWAY, S.BATHROOM_ACTIVITY, S.KITCHEN_ACTIVITY)

INSPECTED_BEFORE = (
    "The step 1 record, for the fitted channels with the time prior: "
    "structural_disagreement is materially better than confidence, household "
    "error AURC 0.048 [0.033, 0.062] lower, and the declared rule picks it; "
    "full-coverage error is 0.364 pooled, 0.360 as a mean over homes. Pooled "
    "over the panel, it keeps of the proportional share of bathroom_activity "
    "and kitchen_activity 0.55 and 0.50 at coverage 0.5 and 0.70 and 0.60 at "
    "0.7, and of away 1.17 and 1.11.",
    "What that record implies for this selection, pooled over all 20 homes: at "
    "coverages 0.8, 0.85, 0.9 and 0.95 the kept share of bathroom_activity is "
    "0.76, 0.78, 0.82 and 0.86, of kitchen_activity 0.67, 0.72, 0.79 and 0.87, "
    "and of away 1.07, 1.05, 1.03 and 1.02, so the guard first holds at 0.95, "
    "and away cannot bind; there the error is 0.347 against 0.364, and the mean "
    "gain over homes is 0.017, below the minimal difference of 0.02. A1 is "
    "therefore unlikely to favour the rule, and the decision likely to be not "
    "supported or inconclusive.",
    "The target coverage and the guard's fraction were declared after those "
    "curves were seen, and were not changed to move that outcome. The guarded "
    "states are those the roadmap named before step 1 was run, the activity "
    "states and away; step 1's guard covered bed_awake, 68 windows in 5 homes, "
    "which this guard leaves out.",
)

WHAT_THIS_CANNOT_SHOW = (
    "A confirmation: the 20 development homes have been inspected, the 43 "
    "held-out CASAS homes are used up for this formulation, and ContextAct's "
    "per-day scores of it are published too. A confirmation needs an "
    "independent dataset this formulation has not been scored on, with the "
    "rule frozen here applied once, unchanged.",
    "Independence of the selection: step 1 saw these homes' curves before the "
    "target and fraction were declared.",
    "The signal's choice: it was picked on these homes, and is not confirmed.",
    "Uncertainty in the fits and the selection: the homes of a fold share one "
    "fit and one threshold, and the household bootstrap resamples the scored "
    "homes without selecting the threshold again.",
    "The frozen threshold's own performance: A1 judges the procedure, with "
    "thresholds chosen on fits to nine homes and applied to fits to ten; the "
    "frozen threshold is chosen on fits to 19 and has no held-out estimate.",
)


@dataclass(frozen=True)
class AbstentionProtocol:
    """Everything the selection fixes before any household is scored.

    Attributes
    ----------
    step1
        Step 1's frozen protocol, whose predictions, signal and folds are used.
    target
        The coverage the rule aims for: abstain on at most one window in five.
    fraction
        The guard: each guarded state keeps at least this fraction of its
        proportional share.
    minimal
        The minimal difference of error and balanced accuracy.
    """

    step1: PriorUncertaintyProtocol
    target: float = 0.8
    fraction: float = 0.8
    minimal: float = 0.02
    tolerance: float = 1e-6
    name: str = EXPERIMENT

    def __post_init__(self) -> None:
        """Validate the declaration."""
        grid = np.array(self.step1.base.coverages)
        if not np.isclose(grid, self.target).any() or not 0.0 < self.target < 1.0:
            raise ValueError("the target must be a coverage on the grid, below 1")
        for name in ("fraction", "minimal", "tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")

    @property
    def coverages(self) -> tuple[float, ...]:
        """The coverages the selection may choose, from the target to 1."""
        return tuple(c for c in self.step1.base.coverages if c >= self.target - 1e-9)

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        base = self.step1.base
        return {
            "schema": PROTOCOL_SCHEMA,
            "name": self.name,
            "status": "pre-specified selection on the development panel, which "
            "earlier work has inspected; one run; not a confirmation",
            "question": "which threshold on structural disagreement keeps the "
            "coverage near the target while the guarded states keep their "
            "share, and does it lower the error on homes it was not chosen on",
            "inspected_before": list(INSPECTED_BEFORE),
            "step1": {
                "file": STEP1_PROTOCOL,
                "sha256": PINNED_RECORDS[STEP1_PROTOCOL],
                "protocol_sha256": self.step1.sha256(),
                "used": "its folds, fits, predictions and ensemble; the signal is "
                "its record's rule_signal, which the run checks",
            },
            "signal": SIGNAL,
            "selection": {
                "coverages": list(self.coverages),
                "threshold": "at each coverage from the target upward, the "
                "smallest signal value at or below which that share of the "
                "selection windows lies, pooled; a window is kept when its "
                "signal is at or below the threshold, so ties are kept",
                "guard": f"each of {', '.join(s.value for s in GUARDED)} keeps at "
                f"least {self.fraction:g} of its proportional share: its kept "
                "share over the coverage kept, pooled over the selection windows; "
                "a state with no selection window is not judged",
                "rule": "the first coverage at which the guard holds; at 1, the "
                "rule never abstains",
                "undefined": "a signal undefined in any selection or held-out "
                "window stops the run before it writes a record",
            },
            "cross_fitting": "for each fold, the threshold is selected on its "
            "training homes, each scored by the ensemble and time prior fitted on "
            "the fold's other training homes; it is applied to the fold's held-out "
            "homes, scored by the fold's fits as step 1 scored them",
            "final_rule": "the threshold selected the same way on all 20 homes, "
            "each scored by the ensemble and time prior fitted on the other 19",
            "estimands": {
                "A1": "primary: per held-out home, the error at full coverage "
                "minus the error of the predictions the rule keeps; positive "
                "favours the rule",
                "A2": "per held-out home, the balanced accuracy of the kept "
                "predictions minus that at full coverage",
                "guard_held_out": "the guard, pooled over the held-out homes' "
                "windows",
                "coverage": "the coverage kept, pooled and per home, described",
            },
            "verdicts": "favours the rule: mean at least the minimal difference "
            "and lower bound above 0; favours no abstention: the mirror; "
            "negligible: the interval within plus or minus it; uncertain: "
            "anything else",
            "decision": "supported on development data when A1 favours the rule, "
            "A2 does not favour no abstention and the guard holds on the held-out "
            "homes; not supported when A1 is negligible or favours no abstention, "
            "or the guard fails on the held-out homes; inconclusive otherwise",
            "edge_cases": "a fold whose threshold never abstains gives its homes "
            "A1 and A2 of exactly 0; a home with no kept window has no A1 or A2 "
            "and is excluded from the comparison and listed as excluded; kept "
            "balanced accuracy averages the states with a "
            "kept window",
            "adoption": "the frozen rule is adopted as the formulation's "
            "abstention rule only when the decision is supported; otherwise it is "
            "recorded and not adopted, and the online pipeline is not changed "
            "either way",
            "parameters": {
                "target": self.target,
                "fraction": self.fraction,
                "minimal": self.minimal,
            },
            "bootstrap": {
                "unit": "household",
                "resamples": base.resamples,
                "confidence": base.confidence,
                "seed": base.seed,
            },
            "the_check": {
                "what": [
                    "every fold's fitted channels and time prior equal the "
                    "fitted-rates record's",
                    "every held-out home's reference scores equal the "
                    "combined-prior record's",
                    "step 1's record picked this signal",
                ],
                "tolerance": self.tolerance,
                "otherwise": "nothing is reported",
            },
            "one_run": {
                "runs": f"one run, by {WORKFLOW} dispatched on develop; the script "
                "refuses anywhere else",
                "branch": f"the run writes {RESULTS_BRANCH}, and the workflow "
                "refuses when it exists, or when the frozen protocol file has "
                "more than its freeze commit",
                "sources": "the freeze records every file of the package and "
                + ", ".join(PINNED_FILES)
                + ", and the versions of "
                + ", ".join(PINNED_DISTRIBUTIONS)
                + "; the run refuses when one differs",
                "amendments": f"a change to a pinned source after the freeze is "
                f"declared in {AMENDMENTS_FILE} with its reason and new digest; "
                "the run accepts it only at that digest and lists the file as an "
                "input",
                "crash": "a run that stops before writing its record reports "
                "nothing; it is fixed without looking at any outcome, the fix is "
                "declared as an amendment, and it is run again; a run that wrote "
                "its record is the run",
            },
            "pinned_records": dict(PINNED_RECORDS),
            "what_this_cannot_show": list(WHAT_THIS_CANNOT_SHOW),
            "tuning": "none: every setting is declared here, before scoring",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_protocol(
    splits: FrozenSplits, root: Path = Path(".")
) -> AbstentionProtocol:
    """The protocol as declared, on step 1's frozen protocol; refuses changed pins."""
    first = step1.declared_protocol(splits, root)
    step1.check_frozen_protocol(first, Path(root) / STEP1_PROTOCOL)
    for name, digest in PINNED_RECORDS.items():
        if file_sha256(Path(root) / name) != digest:
            raise ValueError(f"{name} is not the file the protocol pins")
    return AbstentionProtocol(first)


def write_protocol(
    protocol: AbstentionProtocol, path: Path, root: Path | None = None
) -> str:
    """Write the declaration, its digest, the moment and the code it was frozen with."""
    payload = {
        **protocol.to_dict(),
        "protocol_sha256": protocol.sha256(),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "at_freeze": pinning.code_now(
            _REPOSITORY if root is None else Path(root),
            PINNED_FILES,
            PINNED_DISTRIBUTIONS,
        ),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return protocol.sha256()


def check_frozen_protocol(protocol: AbstentionProtocol, path: Path) -> str:
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


def code_changes(path: Path, root: Path | None = None) -> list[str]:
    """The pinned sources that differ now from the freeze and its amendments."""
    root = _REPOSITORY if root is None else Path(root)
    return pinning.code_changes(
        path, root, PINNED_FILES, read_amendments(root / AMENDMENTS_FILE)
    )


# ----------------------------------------------------------------------------
# The selection
# ----------------------------------------------------------------------------
def kept_shares(
    signal: np.ndarray, truth: np.ndarray, threshold: float, states: Sequence[str]
) -> dict[str, Any]:
    """The coverage kept at *threshold*, and each guarded state's share over it."""
    kept = signal <= threshold
    coverage = float(kept.mean()) if kept.size else math.nan
    ratios: dict[str, float | None] = {}
    for state in GUARDED:
        mask = truth == states.index(state.value)
        ratios[state.value] = (
            float(kept[mask].mean()) / coverage if mask.any() and coverage > 0 else None
        )
    return {"coverage": coverage, "ratios": ratios}


def select_threshold(
    signal: np.ndarray,
    truth: np.ndarray,
    states: Sequence[str],
    protocol: AbstentionProtocol,
) -> dict[str, Any]:
    """The declared selection on pooled windows: the first coverage the guard allows."""
    signal = np.asarray(signal, dtype=float)
    if not np.isfinite(signal).all():
        raise ValueError("the signal must be defined in every selection window")
    tried = []
    for coverage in protocol.coverages:
        threshold = (
            math.inf
            if coverage >= 1.0
            else float(np.quantile(signal, coverage, method="inverted_cdf"))
        )
        shares = kept_shares(signal, truth, threshold, states)
        holds = all(
            r is None or r >= protocol.fraction for r in shares["ratios"].values()
        )
        tried.append({"target": coverage, **shares, "guard_holds": holds})
        if holds or coverage >= 1.0:
            return {
                "coverage": coverage,
                "threshold": None if math.isinf(threshold) else threshold,
                "abstains": not math.isinf(threshold),
                "tried": tried,
            }
    raise AssertionError("the grid ends at 1")  # pragma: no cover


# ----------------------------------------------------------------------------
# Scoring homes
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Scored:
    """One home's labelled windows: truth, prediction, reference posterior, signal."""

    truth: np.ndarray
    predicted: np.ndarray
    posterior: np.ndarray
    signal: np.ndarray


def score_home(
    recording: CasasRecording,
    samples: Mapping[str, Any],
    periodic: Any,
    protocol: AbstentionProtocol,
    *,
    household: str,
    ontology: StateOntology,
) -> Scored:
    """The reference recursion and structural disagreement at a home's labelled windows."""
    base = protocol.step1.base
    resolution = base.resolution
    models = samples["all"].models(recording.registry, resolution, HURDLE, ontology)
    counts, rows, labels, moments = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    transitions = periodic_transitions(periodic, ontology, resolution.step)
    start = periodic.probabilities([moments[0].hour])[0]
    _, posterior = periodic_recursion(
        total_loglik(models, counts), transitions, [m.hour for m in moments], start
    )
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
    scored = posterior[rows]
    return Scored(
        labels, np.argmax(scored, axis=1), scored, trace.normalised_discordance
    )


def _metrics(scored: Scored, kept: np.ndarray, size: int) -> dict[str, float | None]:
    if not kept.any():
        return {"error": None, "balanced_accuracy": None}
    truth, predicted = scored.truth[kept], scored.predicted[kept]
    recalls = [
        float((predicted[truth == k] == k).mean())
        for k in range(size)
        if (truth == k).any()
    ]
    return {
        "error": float((truth != predicted).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
    }


@dataclass(frozen=True)
class AbstentionResult:
    """A completed run: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def run_abstention(
    recordings: Mapping[str, CasasRecording],
    protocol: AbstentionProtocol,
    *,
    fitted_rates: Mapping[str, Any],
    combined_prior: Mapping[str, Any],
    step1_record: Mapping[str, Any],
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> AbstentionResult:
    """Run the declared selection; report nothing unless the checks hold."""
    if step1_record["results"]["rule_signal"]["signal"] != SIGNAL:
        raise ValueError("step 1's record did not pick this signal")
    base = protocol.step1.base
    rates = protocol.step1.combined.base
    ontology = StateOntology()
    space = tuple(ontology.states)
    states = [s.value for s in space]
    homes = sorted({h for fold in base.folds for h in (*fold.train, *fold.test)})
    fits = fold_fits(recordings, rates, ontology)
    check: dict[str, Any] = {
        "fits": check_fits(fits, fitted_rates["results"]["fitted"], protocol.tolerance),
        "signal": SIGNAL,
    }
    statistics = {
        h: home_statistics(recordings[h], base.resolution, ontology, household=h)
        for h in homes
    }
    hours = {
        h: hour_state_counts(
            recordings[h],
            _regular_moments(recordings[h], base.resolution.step),
            states=space,
            step=base.resolution.step,
        )
        for h in homes
    }

    def fitted_on(training: Sequence[str]) -> tuple[dict[str, Any], Any]:
        samples = fit_samples(
            statistics, training, states=space, pseudo_windows=base.pseudo_windows
        )
        periodic = fit_periodic_prior(
            {h: hours[h] for h in training}, states=space, config=rates.periodic_prior
        )
        return samples, periodic

    def left_out(pool: Sequence[str]) -> dict[str, Scored]:
        out = {}
        for home in sorted(pool):
            samples, periodic = fitted_on([h for h in pool if h != home])
            out[home] = score_home(
                recordings[home],
                samples,
                periodic,
                protocol,
                household=home,
                ontology=ontology,
            )
        return out

    def pooled(scored: Mapping[str, Scored]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.concatenate([scored[h].signal for h in sorted(scored)]),
            np.concatenate([scored[h].truth for h in sorted(scored)]),
        )

    folds: dict[str, Any] = {}
    held_out: dict[str, Scored] = {}
    thresholds: dict[str, float] = {}
    for fold in base.folds:
        selection = select_threshold(*pooled(left_out(fold.train)), states, protocol)
        samples = fit_samples(
            statistics, fold.train, states=space, pseudo_windows=base.pseudo_windows
        )
        if samples["all"].to_dict() != fits[fold.name].channels.to_dict():
            raise ValueError("the ensemble's population is not the fold's fit")
        periodic = fits[fold.name].periodic
        for home in sorted(fold.test):
            held_out[home] = score_home(
                recordings[home],
                samples,
                periodic,
                protocol,
                household=home,
                ontology=ontology,
            )
            if not np.isfinite(held_out[home].signal).all():
                raise ValueError(f"the signal is undefined in {home}")
            thresholds[home] = (
                math.inf if selection["threshold"] is None else selection["threshold"]
            )
        folds[fold.name] = {"selection": selection, "test": sorted(fold.test)}
    check["scores"] = check_reference_scores(
        {
            h: _score([space[int(i)] for i in s.truth], s.posterior, space)
            for h, s in held_out.items()
        },
        combined_prior["household_metrics"]["filter_periodic_hurdle@R"],
        protocol.tolerance,
    )

    per_home: dict[str, Any] = {}
    for home, scored in sorted(held_out.items()):
        kept = scored.signal <= thresholds[home]
        everything = np.ones_like(kept)
        per_home[home] = {
            "coverage": float(kept.mean()),
            "kept": _metrics(scored, kept, len(space)),
            "full": _metrics(scored, everything, len(space)),
        }

    def paired(key: str, higher_is_better: bool) -> dict[str, Any]:
        rule = {h: e["kept"][key] for h, e in per_home.items()}
        full = {h: e["full"][key] for h, e in per_home.items()}
        comparison = compare_households(
            rule,
            full,
            higher_is_better=higher_is_better,
            confidence=base.confidence,
            resamples=base.resamples,
            seed=base.seed,
        ).to_dict()
        label = {
            "favours model": "favours the rule",
            "favours reference": "favours no abstention",
        }
        raw = verdict(comparison, protocol.minimal)
        return {"comparison": comparison, "verdict": label.get(raw, raw)}

    estimands = {"A1": paired("error", False), "A2": paired("balanced_accuracy", True)}
    signal, truth = pooled(held_out)
    kept = np.concatenate(
        [held_out[h].signal <= thresholds[h] for h in sorted(held_out)]
    )
    guard: dict[str, Any] = {
        "coverage": float(kept.mean()),
        "ratios": {
            state.value: (
                float(kept[truth == states.index(state.value)].mean())
                / float(kept.mean())
                if (truth == states.index(state.value)).any()
                else None
            )
            for state in GUARDED
        },
    }
    guard["holds"] = all(
        r is None or r >= protocol.fraction for r in guard["ratios"].values()
    )
    a1, a2 = estimands["A1"]["verdict"], estimands["A2"]["verdict"]
    if a1 == "favours the rule" and a2 != "favours no abstention" and guard["holds"]:
        decision = "supported on development data"
    elif a1 in ("negligible", "favours no abstention") or not guard["holds"]:
        decision = "not supported"
    else:
        decision = "inconclusive"
    final = select_threshold(*pooled(left_out(homes)), states, protocol)

    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified selection; development panel, which earlier work "
        "has inspected",
        "check": check,
        "folds": folds,
        "households": per_home,
        "estimands": estimands,
        "guard_held_out": guard,
        "decision": decision,
        "rule": {"signal": SIGNAL, **final},
    }
    record = ExperimentRecord(
        experiment=protocol.name,
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        inference=ONLINE,
        evidence=online_evidence(recordings, base.homes, base.resolution.step),
        seeds=[base.seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions={
            "coverage": "The share of a home's labelled windows the rule keeps.",
            "error": "The share of kept predictions whose state is wrong.",
            "balanced_accuracy": "The mean recall over the states among the kept "
            "predictions.",
            "ratio": "A state's kept share over the coverage kept: 1 for "
            "proportional abstention.",
        },
        notes=[
            "Every setting and rule was declared before any household was scored.",
            "Each threshold was selected on homes scored by models fitted without "
            "the homes it was applied to.",
        ],
        inputs=list(inputs),
        split={"folds": [{**f.to_dict(), "sha256": f.sha256()} for f in base.folds]},
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return AbstentionResult(record=record, path=path)
