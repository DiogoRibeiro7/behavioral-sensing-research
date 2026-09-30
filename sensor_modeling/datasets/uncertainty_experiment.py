"""Phase 4: the first pre-specified comparison of richer uncertainty diagnostics.

Paper 1 found that confidence, entropy, margin, evidence strength and
information gain fail as standalone uncertainty signals. Phase 4 built three
structural diagnostics:

- ``structural_disagreement``: how much supported model specifications
  disagree (:mod:`~sensor_modeling.evaluation.disagreement`);
- ``evidence_disagreement``: how much the evidence groups disagree
  (:mod:`~sensor_modeling.evaluation.evidence_groups`);
- ``predictive_mismatch``: how surprising the evidence is under every state
  (:mod:`~sensor_modeling.evaluation.mismatch`).

This experiment compares them with posterior confidence and entropy as signals
for rejecting the same predictions, with the selective-prediction framework
(:mod:`~sensor_modeling.evaluation.selective`), on the development households.

The question
------------
Does any structural diagnostic produce a materially better selective-risk
curve than confidence, without disproportionately rejecting difficult minority
states?

Every signal, setting, estimand, minimal difference and decision rule is
declared in :class:`UncertaintyProtocol` and frozen in a committed file before
any household is scored. No threshold is selected.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.evidence_groups import group_window
from ..evaluation.households import compare_households
from ..evaluation.provenance import (
    ExperimentRecord,
    InputArtifact,
    ModelRecord,
    ReportedInterval,
)
from ..evaluation.selective import (
    CALIBRATION_BINS,
    DEFAULT_COVERAGES,
    HIGHER_IS_RISKIER,
    HIGHER_IS_SAFER,
    SelectiveData,
    Signal,
    check_coverages,
    evaluate_signals,
)
from ..fusion.filter import FusionConfig, WindowTerms
from ..fusion.regime import ONLINE
from ..states.ontology import StateOntology
from .casas import CasasRecording
from .channel_models import (
    HURDLE,
    HURDLE_NB,
    ChannelStatistics,
    FittedChannels,
    filter_recursion,
    home_statistics,
    household_channel_counts,
    total_loglik,
)
from .information_sets import EvidenceChannel, EvidenceResolution
from .matched_evaluation import HouseholdSplit, _finite, online_evidence
from .observation_mismatch import household_mismatch
from .recoverable_gap import FrozenSplits, GapProtocol
from .structural_models import (
    REFERENCE,
    Specification,
    check_ensemble,
    fit_samples,
    household_trace,
)
from .time_prior_experiment import verdict

PROTOCOL_SCHEMA = "uncertainty-protocol/1"
RESULT_SCHEMA = "uncertainty-results/1"

CONFIDENCE = "confidence"
ENTROPY = "entropy"
STRUCTURAL = "structural_disagreement"
EVIDENCE = "evidence_disagreement"
MISMATCH = "predictive_mismatch"

#: Every signal, in report order.
SIGNALS = (CONFIDENCE, ENTROPY, STRUCTURAL, EVIDENCE, MISMATCH)

#: The signal every other is compared with.
COMPARATOR = CONFIDENCE

#: The structural diagnostics the key question is about.
STRUCTURAL_SIGNALS = (STRUCTURAL, EVIDENCE, MISMATCH)

#: Each signal's direction and missing policy.
DIRECTIONS = {
    CONFIDENCE: HIGHER_IS_SAFER,
    ENTROPY: HIGHER_IS_RISKIER,
    STRUCTURAL: HIGHER_IS_RISKIER,
    EVIDENCE: HIGHER_IS_RISKIER,
    MISMATCH: HIGHER_IS_RISKIER,
}
MISSING = {
    CONFIDENCE: "refuse",
    ENTROPY: "refuse",
    STRUCTURAL: "refuse",
    EVIDENCE: "reject_first",
    MISMATCH: "refuse",
}

DEFINITIONS = {
    CONFIDENCE: "the reference recursion's posterior probability of its "
    "predicted state",
    ENTROPY: "the reference posterior's Shannon entropy divided by log of the "
    "number of states, in [0, 1]",
    STRUCTURAL: "the normalised discordance of the ensemble's posteriors: their "
    "generalised Jensen-Shannon divergence in bits over log2 of the number of "
    "specifications, in [0, 1]",
    EVIDENCE: "the normalised discordance of the identifiable evidence groups of "
    "the reference recursion's window: motion (the motion channels), contact "
    "(the door channel) and context (the prediction before the window), each "
    "group's posterior its channels' fitted likelihoods normalised with no "
    "prior; undefined, and rejected first, when fewer than two groups are "
    "identifiable",
    MISMATCH: "the least standardised surprise over the states: min over s of "
    "(-log p(x | s) minus its mean under s) over its standard deviation under "
    "s, with the reference's normalised hurdle count laws",
}

#: The population-only ensemble: the reference, the other supported
#: observation model, and the population fitted on each half of the training
#: households. The pooled specification reads the household's own labelled
#: windows, which a deployed signal cannot, so it is left out.
ENSEMBLE: tuple[Specification, ...] = (
    REFERENCE,
    Specification(observation=HURDLE_NB),
    Specification(parameter_sample="half_a"),
    Specification(parameter_sample="half_b"),
)

#: Estimands at each inspection coverage, with whether higher is better.
AT_LEVEL = {
    "balanced_accuracy": True,
    "error": False,
    "calibration_error": False,
}

DECISIONS = (
    "materially better",
    "better but rejects difficult minority states",
    "negligible difference",
    "worse",
    "uncertain",
)


def _default_minimal() -> dict[str, float]:
    return {
        "aurc": 0.01,
        "balanced_accuracy_area": 0.02,
        "balanced_accuracy": 0.02,
        "error": 0.02,
        "calibration_error": 0.02,
        "state_coverage": 0.05,
    }


@dataclass(frozen=True)
class UncertaintyProtocol:
    """Everything the experiment fixes before any household is scored."""

    folds: tuple[HouseholdSplit, ...]
    splits_sha256: str
    resolution: EvidenceResolution = field(default_factory=EvidenceResolution)
    pseudo_windows: float = 12.0
    coverages: tuple[float, ...] = DEFAULT_COVERAGES
    inspection: tuple[float, ...] = (0.5, 0.7, 0.9)
    calibration_bins: int = CALIBRATION_BINS
    evidence_floor: float = FusionConfig().evidence_floor
    curve_resamples: int = 2000
    resamples: int = 10_000
    confidence: float = 0.95
    seed: int = 0
    minority_share: float = 0.10
    minimal_differences: Mapping[str, float] = field(default_factory=_default_minimal)
    name: str = "phase4-uncertainty-diagnostics"

    def __post_init__(self) -> None:
        """Validate the declaration."""
        base = GapProtocol(
            tuple(self.folds),
            seed=self.seed,
            resamples=self.resamples,
            confidence=self.confidence,
        )
        if not isinstance(self.splits_sha256, str) or len(self.splits_sha256) != 64:
            raise ValueError("splits_sha256 must be a SHA-256 hex digest")
        grid = check_coverages(self.coverages)
        if not self.inspection or any(
            not np.isclose(grid, level).any() or level >= 1.0
            for level in self.inspection
        ):
            raise ValueError("inspection levels must lie on the grid, below 1")
        if not 0.0 < self.minority_share < 1.0:
            raise ValueError("minority_share must lie in (0, 1)")
        if not math.isfinite(self.pseudo_windows) or self.pseudo_windows <= 0.0:
            raise ValueError("pseudo_windows must be positive")
        if self.curve_resamples < 100:
            raise ValueError("curve_resamples must be at least 100")
        expected = set(_default_minimal())
        if set(self.minimal_differences) != expected or any(
            not math.isfinite(v) or v <= 0.0 for v in self.minimal_differences.values()
        ):
            raise ValueError(
                f"minimal differences must be positive for {sorted(expected)}"
            )
        check_ensemble(ENSEMBLE)
        object.__setattr__(self, "folds", base.folds)
        object.__setattr__(self, "coverages", tuple(float(c) for c in grid))
        object.__setattr__(self, "inspection", tuple(float(c) for c in self.inspection))
        object.__setattr__(self, "minimal_differences", dict(self.minimal_differences))

    @property
    def homes(self) -> tuple[str, ...]:
        """Every held-out household, sorted."""
        return tuple(sorted(home for fold in self.folds for home in fold.test))

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        return {
            "schema": PROTOCOL_SCHEMA,
            "name": self.name,
            "question": "does any structural diagnostic produce a materially "
            "better selective-risk curve than confidence, without "
            "disproportionately rejecting difficult minority states",
            "households": {
                "splits_file": "artifacts/phase1/household_splits.json",
                "splits_sha256": self.splits_sha256,
                "folds": [
                    {**fold.to_dict(), "sha256": fold.sha256()} for fold in self.folds
                ],
                "use": "each household is held out once; its fold's training "
                "households fit every channel model; the development panel, not "
                "a held-out claim",
            },
            "predictions": {
                "model": REFERENCE.name,
                "description": "the Phase 3.3 follow-up's recursion: fitted "
                "hurdle-Poisson channels with the fold's population parameters, "
                "fed every window of the recording from the stationary "
                "distribution; the predicted state is the posterior's most "
                "probable, and every signal ranks these same predictions",
                "pseudo_windows": self.pseudo_windows,
                "scored": "every labelled window of every held-out household",
            },
            "signals": {
                name: {
                    "direction": DIRECTIONS[name],
                    "missing": MISSING[name],
                    "definition": DEFINITIONS[name],
                }
                for name in SIGNALS
            },
            "ensemble": {
                "specifications": [s.to_dict() for s in ENSEMBLE],
                "reason": "the default ensemble without its pooled specification, "
                "which adapts to the household's own labelled windows; a deployed "
                "signal has no labels, and a cut-off would change the scored "
                "windows of every signal",
            },
            "evidence_groups": {
                "groups": "motion: motion channels; contact: door channels; "
                "context: the prediction before the window",
                "evidence_floor": self.evidence_floor,
                "reliability": "1 for every channel: CASAS records no sensor health",
            },
            "comparator": COMPARATOR,
            "structural_signals": list(STRUCTURAL_SIGNALS),
            "compared": [s for s in SIGNALS if s != COMPARATOR],
            "evaluation": {
                "framework": "sensor_modeling.evaluation.selective",
                "coverages": list(self.coverages),
                "inspection": list(self.inspection),
                "selection": "pooled over the panel for the pooled curves; within "
                "each household for the household curves; ties at the boundary "
                "retained in part",
                "loss": "0-1: the selective risk is the error rate",
                "calibration_bins": self.calibration_bins,
                "curve_bootstrap": {
                    "unit": "household",
                    "resamples": self.curve_resamples,
                    "confidence": self.confidence,
                    "seed": self.seed,
                },
            },
            "estimands": {
                "aurc": "primary: each household's error AURC over the grid, "
                "confidence minus signal; positive favours the signal",
                "balanced_accuracy_area": "each household's selective balanced "
                "accuracy averaged over the grid by the trapezoid rule, signal "
                "minus confidence",
                "at_inspection": "at each inspection coverage, each household's "
                "selective balanced accuracy (signal minus confidence), error and "
                "calibration error (confidence minus signal)",
                "state_coverage": "at each inspection coverage, each household's "
                "retained share of each state it has, signal minus confidence: "
                "negative when the signal rejects the state more",
                "pairing": "households with a value under both signals; every "
                "household counts once",
            },
            "bootstrap": {
                "unit": "household",
                "statistics": "mean and median paired differences",
                "resamples": self.resamples,
                "confidence": self.confidence,
                "interval": "percentile",
                "seed": self.seed,
            },
            "minimal_differences": dict(self.minimal_differences),
            "verdicts": "favours signal: mean at least the minimal difference and "
            "interval above 0; favours confidence: mean at most minus it and "
            "interval below 0; negligible: interval within plus or minus it; "
            "uncertain otherwise",
            "difficult_minority_states": {
                "rule": f"a state is a difficult minority state when its share of "
                f"the scored windows is below {self.minority_share:g} and the "
                "reference's recall on it at full coverage is below the "
                "reference's balanced accuracy, both pooled over the scored "
                "windows; fixed from the reference's predictions, before any "
                "signal is compared",
                "minority_share": self.minority_share,
            },
            "rejection_bias": {
                "inspected": "every state's retained share at every inspection "
                "coverage, pooled and paired by household against confidence",
                "guard": "a signal disproportionately rejects difficult minority "
                "states when, for any such state at any inspection coverage, "
                "the paired comparison of its retained share favours confidence: "
                "it retains less of the state than confidence does, by at least "
                "the minimal difference, with the interval below 0",
                "absolute": "the retained share over the coverage, pooled, is "
                "reported for every signal and state: 1 is proportional "
                "rejection",
            },
            "decision": {
                "materially better": "the AURC verdict favours the signal and the "
                "guard holds",
                "better but rejects difficult minority states": "the AURC verdict "
                "favours the signal and the guard fails",
                "negligible difference": "the AURC verdict is negligible",
                "worse": "the AURC verdict favours confidence",
                "uncertain": "otherwise",
                "key_question": "yes when any structural signal is materially "
                "better; no otherwise",
            },
            "thresholds": "none selected: no coverage level or signal value is "
            "chosen for deployment",
            "inference": "online filter: every prediction and signal reads no "
            "window after the scored one",
            "tuning": "none: every setting is declared here, before scoring",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_protocol(splits: FrozenSplits) -> UncertaintyProtocol:
    """The protocol as declared for the development panel's frozen folds."""
    return UncertaintyProtocol(splits.folds, splits.sha256)


def check_frozen_protocol(protocol: UncertaintyProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after scoring begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


# ----------------------------------------------------------------------------
# One household's predictions and signals
# ----------------------------------------------------------------------------
def normalised_entropy(posterior: np.ndarray) -> np.ndarray:
    """Each row's Shannon entropy over the log of the number of states."""
    p = np.asarray(posterior, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, p * np.log(p), 0.0)
    values: np.ndarray = -terms.sum(axis=1) / math.log(p.shape[1])
    return values


def evidence_discordance(
    loglik: Mapping[EvidenceChannel, np.ndarray],
    counts: Mapping[EvidenceChannel, np.ndarray],
    predicted: np.ndarray,
    rows: np.ndarray,
    moments: Sequence[Any],
    *,
    states: Sequence[str],
    step: Any,
    evidence_floor: float,
) -> np.ndarray:
    """The normalised discordance of the evidence groups at each scored row.

    NaN where fewer than two groups are identifiable.
    """
    channels = sorted(loglik)
    modalities = {c.name: c.modality for c in channels}
    out = np.full(rows.size, np.nan)
    for k, row in enumerate(rows):
        r = int(row)
        terms = WindowTerms(
            at=moments[r],
            elapsed=step,
            predicted=predicted[r],
            likelihoods={c.name: loglik[c][r] for c in channels},
            reliability={c.name: 1.0 for c in channels},
            attribution={c.name: 1.0 for c in channels},
            observations={c.name: int(counts[c][r]) for c in channels},
        )
        window = group_window(terms, modalities, states, evidence_floor=evidence_floor)
        discordance = window.discordance()
        if discordance is not None:
            out[k] = discordance / math.log2(len(window.identifiable))
    return out


def household_signals(
    recording: CasasRecording,
    protocol: UncertaintyProtocol,
    samples: Mapping[str, FittedChannels],
    *,
    household: str,
    ontology: StateOntology,
) -> dict[str, Any]:
    """The reference predictions and every signal at a household's scored windows."""
    resolution = protocol.resolution
    population = samples["all"]
    labels_of = [s.value for s in ontology.states]
    models = population.models(recording.registry, resolution, HURDLE, ontology)
    counts, rows, labels, moments = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    loglik = {c: models[c].loglik(counts[c]) for c in sorted(models)}
    predicted, posterior = filter_recursion(
        total_loglik(models, counts),
        ontology.transition(resolution.step),
        ontology.stationary(),
    )
    scored = posterior[rows]

    trace = household_trace(
        recording,
        ENSEMBLE,
        samples,
        household=household,
        resolution=resolution,
        ontology=ontology,
    )
    reference = trace.posteriors[:, trace.models.index(REFERENCE.name)]
    if not np.array_equal(reference, scored):
        raise RuntimeError("the ensemble's reference is not the scored recursion")

    mismatch = household_mismatch(
        recording,
        models,
        household=household,
        resolution=resolution,
        ontology=ontology,
        fitted=population,
    )
    assert mismatch.predicted is not None
    if not np.allclose(mismatch.predicted, predicted[rows], rtol=0.0, atol=1e-12):
        raise RuntimeError("the mismatch trace's prediction is not the recursion's")

    evidence = evidence_discordance(
        loglik,
        counts,
        predicted,
        rows,
        moments,
        states=labels_of,
        step=resolution.step,
        evidence_floor=protocol.evidence_floor,
    )
    return {
        "truth": labels,
        "predicted": np.argmax(scored, axis=1),
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
# Comparisons
# ----------------------------------------------------------------------------
def _grid_mean(values: np.ndarray, grid: np.ndarray) -> float:
    """A curve's trapezoidal mean over the grid, as the AURC is computed."""
    widths = np.diff(grid)
    area = float(((values[1:] + values[:-1]) / 2.0 * widths).sum())
    return area / float(grid[-1] - grid[0])


def _compare(
    signal: Mapping[str, float | None],
    reference: Mapping[str, float | None],
    *,
    higher_is_better: bool,
    minimal: float,
    protocol: UncertaintyProtocol,
) -> dict[str, Any]:
    """A paired household comparison oriented so that positive favours the signal."""
    present = [
        h
        for h in signal
        if signal[h] is not None
        and reference.get(h) is not None
        and math.isfinite(float(signal[h]))  # type: ignore[arg-type]
        and math.isfinite(float(reference[h]))  # type: ignore[arg-type]
    ]
    if not present:
        return {"comparison": None, "verdict": "not compared", "minimal": minimal}
    comparison = compare_households(
        signal,
        reference,
        higher_is_better=higher_is_better,
        confidence=protocol.confidence,
        resamples=protocol.resamples,
        seed=protocol.seed,
    ).to_dict()
    return {
        "comparison": comparison,
        "verdict": verdict(comparison, minimal),
        "minimal": minimal,
    }


def _label(verdict_: str) -> str:
    return {
        "favours model": "favours signal",
        "favours reference": "favours confidence",
    }.get(verdict_, verdict_)


def decide(aurc: str, guard_holds: bool) -> str:
    """The declared decision from the AURC verdict and the minority-state guard."""
    if aurc == "favours signal":
        return (
            "materially better"
            if guard_holds
            else "better but rejects difficult minority states"
        )
    if aurc == "negligible":
        return "negligible difference"
    if aurc == "favours confidence":
        return "worse"
    return "uncertain"


def difficult_minority_states(
    truth: np.ndarray, predicted: np.ndarray, states: Sequence[str], share: float
) -> dict[str, dict[str, Any]]:
    """Each state's share and reference recall, and whether it is a difficult minority."""
    out: dict[str, dict[str, Any]] = {}
    recalls = {}
    for k, state in enumerate(states):
        mask = truth == k
        if mask.any():
            recalls[state] = float((predicted[mask] == k).mean())
    balanced = float(np.mean(list(recalls.values())))
    for k, state in enumerate(states):
        fraction = float((truth == k).mean())
        recall = recalls.get(state)
        out[state] = {
            "share": fraction,
            "recall": recall,
            "difficult_minority": bool(
                recall is not None and fraction < share and recall < balanced
            ),
        }
    return out


def paired_comparisons(
    payload: Mapping[str, Any],
    protocol: UncertaintyProtocol,
    difficult: Sequence[str],
) -> dict[str, Any]:
    """Every compared signal against confidence, household by household."""
    grid = np.array(protocol.coverages)
    minimal = protocol.minimal_differences
    states = list(payload["states"])
    signals = payload["signals"]

    def per_home(name: str, pick: Any) -> dict[str, float | None]:
        return {h: pick(e) for h, e in signals[name]["households"].items()}

    def at(values: list[float | None], level: float) -> float | None:
        return values[int(np.argmin(np.abs(grid - level)))]

    def area(values: list[float | None]) -> float | None:
        array = np.array([np.nan if v is None else v for v in values], dtype=float)
        return _grid_mean(array, grid) if np.all(np.isfinite(array)) else None

    out: dict[str, Any] = {}
    for name in [s for s in SIGNALS if s != COMPARATOR]:
        entry: dict[str, Any] = {
            "aurc": _compare(
                per_home(name, lambda e: e["aurc"]),
                per_home(COMPARATOR, lambda e: e["aurc"]),
                higher_is_better=False,
                minimal=minimal["aurc"],
                protocol=protocol,
            ),
            "balanced_accuracy_area": _compare(
                per_home(name, lambda e: area(e["balanced_accuracy"])),
                per_home(COMPARATOR, lambda e: area(e["balanced_accuracy"])),
                higher_is_better=True,
                minimal=minimal["balanced_accuracy_area"],
                protocol=protocol,
            ),
            "at": {},
            "state_coverage": {},
        }
        for level in protocol.inspection:
            key = f"{level:g}"
            entry["at"][key] = {
                metric: _compare(
                    per_home(name, lambda e, m=metric, c=level: at(e[m], c)),
                    per_home(COMPARATOR, lambda e, m=metric, c=level: at(e[m], c)),
                    higher_is_better=better,
                    minimal=minimal[metric],
                    protocol=protocol,
                )
                for metric, better in AT_LEVEL.items()
            }
            entry["state_coverage"][key] = {
                state: _compare(
                    per_home(
                        name, lambda e, s=state, c=level: at(e["state_coverage"][s], c)
                    ),
                    per_home(
                        COMPARATOR,
                        lambda e, s=state, c=level: at(e["state_coverage"][s], c),
                    ),
                    higher_is_better=True,
                    minimal=minimal["state_coverage"],
                    protocol=protocol,
                )
                for state in states
            }
        for key in ("aurc", "balanced_accuracy_area"):
            entry[key]["verdict"] = _label(entry[key]["verdict"])
        for level in entry["at"].values():
            for item in level.values():
                item["verdict"] = _label(item["verdict"])
        for level in entry["state_coverage"].values():
            for item in level.values():
                item["verdict"] = _label(item["verdict"])
        failures = [
            {"state": state, "coverage": key}
            for key, level in entry["state_coverage"].items()
            for state, item in level.items()
            if state in difficult and item["verdict"] == "favours confidence"
        ]
        entry["guard"] = {
            "difficult_minority_states": list(difficult),
            "failures": failures,
            "holds": not failures,
        }
        entry["decision"] = decide(entry["aurc"]["verdict"], not failures)
        out[name] = entry
    return out


def rejection_bias(
    payload: Mapping[str, Any], protocol: UncertaintyProtocol
) -> dict[str, Any]:
    """Each signal's pooled retained share of each state, and its ratio to the coverage."""
    grid = np.array(protocol.coverages)
    out: dict[str, Any] = {}
    for name in SIGNALS:
        table = payload["signals"][name]["pooled"]["state_coverage"]
        out[name] = {}
        for level in protocol.inspection:
            index = int(np.argmin(np.abs(grid - level)))
            out[name][f"{level:g}"] = {
                state: {
                    "retained": values[index],
                    "ratio": (
                        values[index] / float(grid[index])
                        if values[index] is not None
                        else None
                    ),
                }
                for state, values in table.items()
            }
    return out


def reproduction(
    payload: Mapping[str, Any], published: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Each household's full-coverage metrics, against the published recursion."""
    entries = payload["signals"][CONFIDENCE]["households"]
    ours = {
        home: {
            "accuracy": 1.0 - e["error"][-1],
            "balanced_accuracy": e["balanced_accuracy"][-1],
            "calibration_error": e["calibration_error"][-1],
        }
        for home, e in entries.items()
    }
    out: dict[str, Any] = {"households": ours, "published": None}
    if published is not None:
        cell = published["household_metrics"]["hurdle@R"]
        gaps = [
            abs(values[metric] - cell[home][metric])
            for home, values in ours.items()
            for metric in values
        ]
        out["published"] = {
            "experiment": published["experiment"],
            "cell": "hurdle@R",
            "git_commit": published["environment"]["git_commit"],
            "max_abs_difference": max(gaps),
            "households": len(ours),
        }
    return out


@dataclass(frozen=True)
class UncertaintyResult:
    """A completed experiment: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def run_uncertainty(
    recordings: Mapping[str, CasasRecording],
    protocol: UncertaintyProtocol,
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    published: Mapping[str, Any] | None = None,
    output_dir: Path | None = None,
) -> UncertaintyResult:
    """Run the pre-specified comparison and build its record.

    *published* is the Phase 3.3 follow-up's negative-binomial record, whose
    hurdle recursion the reference predictions must reproduce.
    """
    homes = sorted({h for fold in protocol.folds for h in (*fold.train, *fold.test)})
    missing = sorted(set(homes) - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    ontology = StateOntology()
    space = tuple(ontology.states)
    states = [s.value for s in space]
    statistics: dict[str, dict[EvidenceChannel, ChannelStatistics]] = {
        home: home_statistics(
            recordings[home], protocol.resolution, ontology, household=home
        )
        for home in homes
    }
    scored: dict[str, dict[str, Any]] = {}
    fold_of: dict[str, str] = {}
    for fold in protocol.folds:
        samples = fit_samples(
            statistics, fold.train, states=space, pseudo_windows=protocol.pseudo_windows
        )
        for home in sorted(fold.test):
            scored[home] = household_signals(
                recordings[home], protocol, samples, household=home, ontology=ontology
            )
            fold_of[home] = fold.name
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
        coverages=protocol.coverages,
        confidence=protocol.confidence,
        resamples=protocol.curve_resamples,
        seed=protocol.seed,
        bins=protocol.calibration_bins,
    )
    payload = report.to_dict()
    table = difficult_minority_states(
        data.truth, data.predicted, states, protocol.minority_share
    )
    difficult = [s for s, e in table.items() if e["difficult_minority"]]
    comparisons = paired_comparisons(payload, protocol, difficult)
    decisions = {name: comparisons[name]["decision"] for name in comparisons}
    better = [s for s in STRUCTURAL_SIGNALS if decisions[s] == "materially better"]
    results = {
        "result_schema": RESULT_SCHEMA,
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
        "rejection_bias": rejection_bias(payload, protocol),
        "decisions": decisions,
        "key_question": {
            "question": protocol.to_dict()["question"],
            "answer": "yes" if better else "no",
            "materially_better": better,
        },
        "reproduction": reproduction(payload, published),
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
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        inference=ONLINE,
        evidence=online_evidence(recordings, protocol.homes, protocol.resolution.step),
        seeds=[protocol.seed],
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
            **{name: DEFINITIONS[name] for name in SIGNALS},
        },
        notes=[
            "Every signal, estimand, minimal difference and decision rule was "
            "declared in the protocol before any household was scored.",
            "Every signal ranks the same predictions: the Phase 3.3 follow-up's "
            "hurdle recursion, cross-fitted on the frozen development folds.",
            "Uncertainty resamples households, never timestamps.",
            "No threshold or operating coverage is selected.",
            "The development panel has been inspected in earlier work; this is "
            "not a held-out claim.",
        ],
        inputs=list(inputs),
        split={
            "folds": [{**f.to_dict(), "sha256": f.sha256()} for f in protocol.folds]
        },
        preprocessing={
            "moments": "one per step from each recording's first observation to "
            "its last",
            "labels": "truth_series of each recording's annotations; unlabelled "
            "moments are not scored",
            "step_seconds": protocol.resolution.step.total_seconds(),
        },
        models=[
            ModelRecord(
                spec.name,
                {"assumptions": [a.to_dict() for a in spec.assumptions]},
                "household_trace",
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
        intervals=_intervals(comparisons),
        selective_prediction=report,
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return UncertaintyResult(record=record, path=path)


def _intervals(comparisons: Mapping[str, Any]) -> list[ReportedInterval]:
    reported: list[ReportedInterval] = []
    for name, entry in comparisons.items():
        for key in ("aurc", "balanced_accuracy_area"):
            comparison = entry[key]["comparison"]
            if comparison is None or comparison["mean"]["interval"] is None:
                continue
            estimate = comparison["mean"]
            reported.append(
                ReportedInterval(
                    label=f"{name} against confidence: mean {key} difference",
                    estimate=estimate["estimate"],
                    low=estimate["interval"]["low"],
                    high=estimate["interval"]["high"],
                    confidence=estimate["interval"]["confidence"],
                    method=estimate["interval"]["method"],
                    unit="household",
                    n=comparison["n"],
                )
            )
    return reported
