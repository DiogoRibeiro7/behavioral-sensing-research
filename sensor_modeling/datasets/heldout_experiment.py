"""The held-out confirmation: the Phase 3 formulation on CASAS homes outside the panel.

On the 20 development homes the time prior and the fitted hurdle channels
combine in the filter's recursion (the combined-prior record, K1 a success).
Those homes had been inspected many times. This experiment fits everything on
all 20 of them, freezes it, and scores it once on the 43 single-resident
CASAS homes of the frozen external cohort, beside the Phase 2 baselines and
the supervised diagnostic fitted on the same 20 homes.

The recursions are scored exactly as in the combined-prior experiment, by
:func:`~sensor_modeling.datasets.combined_prior_experiment.household_scores`.
The baselines are fitted and scored by the matched runner at ``I3``, the hour
and the last three windows, which is strictly less than the recursion reads.
Every model must score the same number of windows in each true state in each
home, or nothing is reported.

Before the freeze, :func:`inputs_report` reads every recording, records what
each home instruments and which windows are labelled, and runs the recursions
to check they are finite, without scoring anything. The protocol declares
that report. Everything is declared in :class:`HeldOutProtocol` and frozen,
with the digest of every source file, in a committed file before any held-out
home is scored.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.households import household_values
from ..evaluation.metrics import PredictionMetrics
from ..evaluation.provenance import ExperimentRecord, InputArtifact, ModelRecord
from ..fusion.regime import ONLINE
from ..states.ontology import StateOntology
from . import rates_experiment
from .baseline_models import baseline_suite
from .casas import CasasRecording, truth_series
from .channel_models import filter_recursion, fit_channel_models, total_loglik
from .combined_prior_experiment import ESTIMANDS as DEVELOPMENT_ESTIMANDS
from .combined_prior_experiment import (
    FILTER_PERIODIC_HURDLE,
    MODELS,
    RECURSION,
    RULES,
    CombinedPriorProtocol,
    _estimand_entry,
    household_scores,
    interaction,
    periodic_recursion,
    periodic_transitions,
)
from .history_experiment import Estimand
from .information_sets import InformationSet, build_feature_table, evidence_column
from .matched_evaluation import (
    MATCHED_METRIC_DEFINITIONS,
    HouseholdSplit,
    ModelSpec,
    _finite,
    _regular_moments,
    held_out_metrics,
    online_evidence,
    run_matched_evaluation,
)
from .periodic_prior import fit_periodic_prior, hour_state_counts
from .rates_experiment import FittedRatesProtocol, FoldFit
from .recoverable_gap import (
    DIAGNOSTIC,
    LOGISTIC,
    FrozenSplits,
    _cell,
    _paired,
    gap_models,
)
from .time_prior_experiment import verdict

PROTOCOL_SCHEMA = "heldout-protocol/1"
RESULT_SCHEMA = "heldout-results/1"
INPUTS_SCHEMA = "heldout-inputs/1"
EXPERIMENT = "casas-heldout-confirmation"
BASELINE_SET = "I3"
#: The fold name the matched runner records for the one training set.
SPLIT = "development-20-heldout-43"
#: A state's K1 recall verdict needs at least this many held-out homes that
#: have the state; below it the difference is described, not judged.
MIN_RECALL_HOMES = 5

COHORT_MANIFEST = "artifacts/v03/external_cohort_manifest.json"
INPUTS_FILE = "artifacts/heldout/heldout_inputs.json"
#: The records the protocol rests on, and their digests.
PINNED_RECORDS = {
    COHORT_MANIFEST: "6e3e157327c6eb40edfae498c2c4cc679a9e7c5b999899cf31683ed8161bc332",
    "artifacts/v03/external_primary_result.json": (
        "66501d48099f48ce7c834839988830090812414b24ce76b193f84ed70783f23f"
    ),
    "artifacts/phase3/fitted_rates_protocol.json": (
        "f695f972fcfcd11266a8a1a720932442e8b7d455bdec7dd6304ecab41843075d"
    ),
    "artifacts/phase3/combined_prior_protocol.json": (
        "3a0c5cad58187609601181ae841099b9144707ba2327af4173c9442812dc52a5"
    ),
    "artifacts/phase3/phase3-combined-prior.json": (
        "57db137f57f5b298054d7a4cfe8c837d854ca9a8793329857d7803401477db59"
    ),
}
#: The sources whose content the freeze records and the run requires unchanged.
PINNED_PACKAGE = "sensor_modeling"
PINNED_FILES = (
    "pyproject.toml",
    "scripts/freeze_heldout_confirmation.py",
    "scripts/run_heldout_confirmation.py",
)
PINNED_DISTRIBUTIONS = ("numpy", "scipy", "pandas", "scikit-learn")
_REPOSITORY = Path(__file__).resolve().parents[2]

#: The gap estimands: a baseline at ``I3`` against the combination in the
#: recursion. Positive favours the baseline.
GAP_ESTIMANDS: tuple[Estimand, ...] = (
    Estimand(
        "G1",
        "secondary",
        "gap",
        "how far the supervised diagnostic, given the hour and the last three "
        "windows only, is from the combination in the recursion",
        DIAGNOSTIC,
        BASELINE_SET,
        FILTER_PERIODIC_HURDLE,
        RECURSION,
    ),
    Estimand(
        "G2",
        "secondary",
        "gap",
        "how far regularised logistic regression, given the hour and the last "
        "three windows only, is from the combination in the recursion",
        LOGISTIC,
        BASELINE_SET,
        FILTER_PERIODIC_HURDLE,
        RECURSION,
    ),
)
GAP_READING = {
    "favours model": "the baseline leads",
    "favours reference": "the combination leads",
    "negligible": "no gap",
    "uncertain": "uncertain",
}

INSPECTED_BEFORE = (
    "The 43 homes were scored once, in the frozen v0.3 external test, by the "
    "v0.2 production filter and the v0.3 candidate with its circadian profile: a "
    "median paired gain of +0.0091 [+0.0054, +0.0117] balanced accuracy, 37 of "
    "43 homes improved; pooled balanced accuracy 0.496 and 0.504, log loss 4.30 "
    "and 4.17. Every per-home metric of those two models was seen, in "
    "artifacts/v03/external_primary_result.json.",
    "The adapter learned to read the sensor names and activities of the tm, rw, "
    "mn and ihs families while the cohort was screened, before that test; see "
    "docs/EXTERNAL_COHORT_COMPOSITION.md.",
    "Neither Phase 3 part, the fitted hurdle channels or the hierarchical time "
    "prior, nor any baseline or the diagnostic, has been scored on any of the 43 "
    "homes; the v0.3 candidate's time-of-day term was its circadian profile.",
    "Before this protocol was frozen, every recording was read once to write the "
    "inputs report it declares: each home's instrumented channels, windows, "
    "labelled windows and the states that occur. The recursions were run on the "
    "held-out homes only to check that every posterior is finite; no prediction "
    "was compared with a label.",
    "The development result this confirms: on the 20 development homes the "
    "time prior adds +0.061 [+0.044, +0.077] balanced accuracy to the fitted "
    "hurdle channels in the recursion, 0.517 against 0.456 (K1, a success); K2 "
    "and K3 are successes too.",
    "On the development homes, cross-fitted, the diagnostic at I3 scored a mean "
    "household balanced accuracy of 0.590 and a mean log loss of 2.455, "
    "logistic regression at I3 0.498 and 1.314, and the combination 0.517 and "
    "1.728.",
)


def family(home: str) -> str:
    """The CASAS deployment family of a home, from its identifier."""
    match = re.match(r"[a-z]+", home)
    if match is None:
        raise ValueError(f"no family in household identifier {home!r}")
    return match.group(0)


def load_cohort(path: Path) -> tuple[dict[str, str], ...]:
    """The frozen cohort's homes, each with its file name and digest."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen-primary-external-cohort":
        raise ValueError(f"{path} is not the frozen external cohort")
    homes = tuple(
        {"id": h["id"], "filename": h["filename"], "sha256": h["sha256"]}
        for h in sorted(manifest["eligible_homes"], key=lambda h: h["id"])
        if h["eligible"]
    )
    if len(homes) != manifest["eligible_count"]:
        raise ValueError("the cohort's eligible homes do not match its count")
    return homes


def file_sha256(path: Path) -> str:
    """SHA-256 of a text file, whatever its line endings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_pinned_records(root: Path = Path(".")) -> None:
    """Refuse unless every pinned record under *root* is the file pinned."""
    for name, digest in PINNED_RECORDS.items():
        if file_sha256(Path(root) / name) != digest:
            raise ValueError(f"{name} is not the file the protocol pins")


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


def code_changes(path: Path, root: Path | None = None) -> list[str]:
    """The pinned sources that differ now from those recorded at the freeze."""
    frozen = json.loads(Path(path).read_text(encoding="utf-8")).get("at_freeze")
    if frozen is None:
        raise ValueError(f"{path} does not record the code it was frozen against")
    now = code_now(root)["sources"]
    then = frozen["sources"]
    return sorted(
        name for name in set(then) | set(now) if then.get(name) != now.get(name)
    )


@dataclass(frozen=True)
class HeldOutProtocol:
    """Everything the confirmation fixes before any held-out home is scored.

    Attributes
    ----------
    base
        The fitted-rates protocol, whose channel and time-prior settings,
        information sets, metrics, minimal differences and bootstrap are used
        unchanged. Its 20 households are the training set.
    cohort
        The held-out homes, each with its file name and digest.
    inputs
        The inputs report, :func:`inputs_report`, with its digest under
        ``"sha256"``. A protocol without one cannot be frozen.
    """

    base: FittedRatesProtocol
    cohort: tuple[Mapping[str, str], ...]
    inputs: Mapping[str, Any] | None = None
    inspected_before: tuple[str, ...] = INSPECTED_BEFORE
    name: str = EXPERIMENT

    def __post_init__(self) -> None:
        test = [home["id"] for home in self.cohort]
        if not test or len(set(test)) != len(test):
            raise ValueError("the cohort must name distinct homes")
        if set(test) & set(self.training):
            raise ValueError("a held-out home is a training home")
        if self.inputs is not None and set(self.inputs["homes"]) != {
            *self.training,
            *test,
        }:
            raise ValueError("the inputs report does not cover exactly these homes")

    @property
    def training(self) -> tuple[str, ...]:
        """The training households: every development household."""
        return self.base.homes

    @property
    def test(self) -> tuple[str, ...]:
        """The held-out households, sorted."""
        return tuple(sorted(home["id"] for home in self.cohort))

    @property
    def split(self) -> HouseholdSplit:
        """The one training set and the held-out homes, for the matched runner."""
        return HouseholdSplit(SPLIT, train=self.training, test=self.test)

    @property
    def scoring(self) -> CombinedPriorProtocol:
        """The combined-prior settings the recursions and estimands reuse."""
        return CombinedPriorProtocol(self.base)

    def baselines(self) -> tuple[ModelSpec, ...]:
        """The Phase 2 baselines at ``I3``, with the diagnostic."""
        diagnostic = next(spec for spec in gap_models() if spec.name == DIAGNOSTIC)
        return (*baseline_suite(self.baseline_set()), diagnostic)

    def baseline_set(self) -> InformationSet:
        """The baselines' information set, ``I3``."""
        labels = self.base.as_gap_protocol().labels
        return next(
            info
            for info in self.base.information_sets
            if labels[info.name] == BASELINE_SET
        )

    def _instrumentation(self) -> dict[str, Any]:
        if self.inputs is None:
            return {}
        homes = self.inputs["homes"]
        everywhere = set.intersection(
            *(set(homes[home]["instrumented"]) for home in self.training)
        )
        lacking = {
            home: sorted(everywhere - set(homes[home]["instrumented"]))
            for home in self.test
            if everywhere - set(homes[home]["instrumented"])
        }
        states = sorted({s for home in self.test for s in homes[home]["states"]})
        return {
            "report": {"file": INPUTS_FILE, "sha256": self.inputs["sha256"]},
            "test": {
                home: {
                    key: homes[home][key]
                    for key in ("instrumented", "windows", "labelled", "states")
                }
                for home in self.test
            },
            "channels_every_training_home_instruments": sorted(everywhere),
            "held_out_homes_lacking_one": lacking,
            "held_out_homes_with_each_state": {
                state: sum(state in homes[home]["states"] for home in self.test)
                for state in states
            },
        }

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        base = self.base
        families: dict[str, int] = {}
        for home in self.test:
            families[family(home)] = families.get(family(home), 0) + 1
        return {
            "schema": PROTOCOL_SCHEMA,
            "name": self.name,
            "status": "pre-specified, one-shot, on CASAS homes outside the "
            "development panel; the homes were scored once before, by the v0.2 "
            "and v0.3 production filter",
            "question": "whether the time prior's gain over the fitted hurdle "
            "channels in the recursion, a success on the development panel, holds "
            "on CASAS homes outside it",
            "inspected_before": list(self.inspected_before),
            "base_protocol": {
                "file": "artifacts/phase3/fitted_rates_protocol.json",
                "protocol_sha256": base.sha256(),
                "used": "its channel families and shrinkage, its time prior, its "
                "information sets, metrics, minimal differences and household "
                "bootstrap; not its folds",
            },
            "households": {
                "training": {
                    "homes": list(self.training),
                    "splits_sha256": base.splits_sha256,
                    "fitted": "the channels, the time prior and every baseline "
                    "are fitted once, on all 20 development homes",
                },
                "test": {
                    "manifest": COHORT_MANIFEST,
                    "homes": [dict(home) for home in self.cohort],
                    "families": families,
                    "reading": "every recording is verified against its frozen "
                    "digest and read with the America/Los_Angeles zone, as every "
                    "CASAS result here was; the hour the time prior reads is the "
                    "recorded wall-clock hour",
                },
            },
            "instrumentation": self._instrumentation(),
            "models": {
                "recursion": {
                    name: {"channels": channels, "time_prior": timed}
                    for name, (channels, timed) in MODELS.items()
                },
                "baselines": {
                    "information_set": BASELINE_SET,
                    "models": [spec.to_dict() for spec in self.baselines()],
                    "why": "the richest declared information set, the hour and the "
                    "last three windows, is strictly less than the recursion reads",
                },
                "missing_channels": "a channel a home does not instrument has no "
                "term in the recursion; the baselines see its columns as missing, "
                "encoded as zero beside a missing indicator, and where every "
                "training home instruments the channel that indicator never varied "
                "in training, so the baselines read the absent channel as silence. "
                "The homes this affects are listed under instrumentation",
            },
            "windows": "every window at the base protocol's step of "
            f"{base.information_sets[0].resolution.step.total_seconds():g} "
            "seconds; only labelled windows are scored; in every home every model "
            "must score the same number of windows in each true state, or nothing "
            "is reported",
            "unscored": "a held-out home with no labelled window is listed as "
            "unscored and enters no estimand; any other home without a score from "
            "every model stops the run",
            "estimands": [
                {**e.to_dict(), "rule": rule}
                for e, rule in (
                    *((e, RULES[e.key]) for e in DEVELOPMENT_ESTIMANDS),
                    *((e, "gap") for e in GAP_ESTIMANDS),
                )
            ],
            "criteria": {
                "verdicts": "favours model: mean >= delta and lower bound > 0; "
                "favours reference: mean <= -delta and upper bound < 0; negligible: "
                "the whole interval within (-delta, delta); uncertain: anything else",
                "time prior": "success when balanced accuracy favours the model and "
                "neither log loss nor calibration error favours the reference; "
                "failure when balanced accuracy is negligible or favours the "
                "reference; inconclusive otherwise",
                "fitted channels": "success when calibration error favours the model "
                "and neither log loss nor balanced accuracy favours the reference; "
                "trade-off when calibration error favours the model and balanced "
                "accuracy the reference; failure when calibration error is "
                "negligible or favours the reference; inconclusive otherwise",
                "gap": "read by balanced accuracy: the baseline leads, the "
                "combination leads, no gap, or uncertain; every metric's verdict is "
                "reported",
                "primary": "K1 alone decides: success confirms that the time prior "
                "adds to the fitted hurdle channels on held-out CASAS homes; failure "
                "does not; inconclusive is reported as such",
                "multiplicity": "K2, K3, G1 and G2 are secondary; nothing is "
                "adjusted",
            },
            "per_state_recall": {
                "estimands": ["K1"],
                "verdict": "the same verdicts with the recall minimal difference, "
                f"for a state that occurs in at least {MIN_RECALL_HOMES} held-out "
                "homes; for fewer the difference is described, not judged",
            },
            "interaction": "K1 minus K2 per household, described as in the "
            "combined-prior protocol",
            "by_family": "each family's mean of every recursion's balanced accuracy "
            "and of K1, described, not judged",
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
            "one_shot": "the run refuses if its record exists, in the repository "
            "or on its results branch, and if any pinned source file differs from "
            "the freeze; whatever it reports stands, and nothing may be changed in "
            "response and rescored as the same confirmation. A run that stops "
            "before writing its record reports nothing; a fix that leaves this "
            "declaration unchanged, made without seeing any outcome, is disclosed "
            "beside the record and the run repeated",
            "pinned_records": dict(PINNED_RECORDS),
            "what_this_cannot_show": [
                "Independence from CASAS: the homes share its sensors, labelling and "
                "annotation, and 8 of them its hh family.",
                "Why a failure happened: training is all hh, and 35 of the 43 homes "
                "come from families with other instrumentation and sensor names the "
                "adapter learned while screening them, so non-transfer and the "
                "mapping cannot be separated.",
                "A first look at these homes: they were scored once before, by the "
                "production filter.",
                "Intervals that allow for families: the bootstrap resamples homes, "
                "and 25 of the 43 are of the tm family.",
                "Anything about the online pipeline as shipped: it offers neither "
                "the fitted channels nor the time prior.",
            ],
            "tuning": "none: every setting is the base protocol's or declared here",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def load_inputs(path: Path) -> dict[str, Any]:
    """The inputs report, with its file's digest under ``"sha256"``."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("schema") != INPUTS_SCHEMA:
        raise ValueError(f"{path} is not an inputs report")
    return {**report, "sha256": file_sha256(Path(path))}


def declared_protocol(
    splits: FrozenSplits, root: Path = Path("."), *, inputs: bool = True
) -> HeldOutProtocol:
    """The protocol as declared, on the frozen fitted-rates protocol and cohort.

    With *inputs*, the committed inputs report is part of it; without, the
    protocol is the one the inputs check runs under and cannot be frozen.
    """
    base = rates_experiment.declared_protocol(splits)
    rates_experiment.check_frozen_protocol(
        base, Path(root) / "artifacts/phase3/fitted_rates_protocol.json"
    )
    check_pinned_records(root)
    return HeldOutProtocol(
        base,
        load_cohort(Path(root) / COHORT_MANIFEST),
        load_inputs(Path(root) / INPUTS_FILE) if inputs else None,
    )


def write_protocol(protocol: HeldOutProtocol, path: Path) -> str:
    """Write the declaration, its digest, the moment and the code it was frozen with."""
    if protocol.inputs is None:
        raise ValueError("a protocol cannot be frozen without its inputs report")
    payload = {
        **protocol.to_dict(),
        "protocol_sha256": protocol.sha256(),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "at_freeze": code_now(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return protocol.sha256()


def check_frozen_protocol(protocol: HeldOutProtocol, path: Path) -> str:
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
# Fitting on the training homes, and the inputs report
# ----------------------------------------------------------------------------
def training_fit(
    recordings: Mapping[str, CasasRecording],
    training: Sequence[str],
    base: FittedRatesProtocol,
    ontology: StateOntology,
) -> FoldFit:
    """The channels, time prior and shares, fitted once on *training*.

    :func:`~sensor_modeling.datasets.rates_experiment.fold_fits` for one
    training set that no fold of the base protocol has.
    """
    resolution = base.information_sets[0].resolution
    states = tuple(ontology.states)
    homes = {home: recordings[home] for home in training}
    counts = {
        home: hour_state_counts(
            recording,
            _regular_moments(recording, resolution.step),
            states=states,
            step=resolution.step,
        )
        for home, recording in homes.items()
    }
    pooled = sum(counts.values(), np.zeros((24, len(states))))
    return FoldFit(
        fit_channel_models(
            homes,
            resolution=resolution,
            pseudo_windows=base.pseudo_windows,
            ontology=ontology,
        ),
        fit_periodic_prior(counts, states=states, config=base.periodic_prior),
        {
            state.value: float(share)
            for state, share in zip(states, pooled.sum(axis=0) / pooled.sum())
        },
    )


def _finite_recursions(
    recording: CasasRecording, fit: FoldFit, table: Any, base: FittedRatesProtocol
) -> bool:
    """Whether every recursion's posterior is finite, from sensors alone."""
    ontology = StateOntology()
    resolution = base.information_sets[0].resolution
    instrumented = sorted(set(resolution.channels) - set(table.uninstrumented))
    counts = {c: table.column(evidence_column(c.name, 0)) for c in instrumented}
    hours = [moment.hour for moment in table.moments]
    timed = periodic_transitions(fit.periodic, ontology, resolution.step)
    start = fit.periodic.probabilities([hours[0]])[0]
    finite = True
    for channels, with_prior in MODELS.values():
        models = fit.channels.models(recording.registry, resolution, channels, ontology)
        loglik = total_loglik(models, counts)
        if with_prior:
            _, posterior = periodic_recursion(loglik, timed, hours, start)
        else:
            _, posterior = filter_recursion(
                loglik, ontology.transition(resolution.step), ontology.stationary()
            )
        finite = finite and bool(np.isfinite(posterior).all())
    return finite


def inputs_report(
    recordings: Mapping[str, CasasRecording], protocol: HeldOutProtocol
) -> dict[str, Any]:
    """What every home instruments and labels, and whether the recursions run.

    Reads each home's sensors to build its current-window features, and of its
    annotations only which windows are labelled and which states occur. Fits on
    the training homes, runs every recursion on each held-out home to check its
    posterior is finite, and builds the baselines' rows. No model is scored.
    """
    base = protocol.base
    current = base.information_sets[0]
    started = time.perf_counter()
    fit = training_fit(recordings, protocol.training, base, StateOntology())
    fitted = time.perf_counter() - started
    homes: dict[str, Any] = {}
    for home in (*protocol.training, *protocol.test):
        recording = recordings[home]
        moments = _regular_moments(recording, current.resolution.step)
        table = build_feature_table(recording, current, moments, household=home)
        truth = [
            s for s in truth_series(recording.activities, moments) if s is not None
        ]
        entry: dict[str, Any] = {
            "role": "training" if home in protocol.training else "held out",
            "family": family(home),
            "observations": len(recording.observations),
            "windows": len(moments),
            "labelled": len(truth),
            "states": sorted({state.value for state in truth}),
            "instrumented": sorted(
                c.name
                for c in set(current.resolution.channels) - set(table.uninstrumented)
            ),
            "excluded_sensors": len(table.excluded_sensors),
        }
        if home in protocol.test:
            entry["recursions_finite"] = _finite_recursions(recording, fit, table, base)
            build_feature_table(  # the baselines' rows, built only to see they build
                recording, protocol.baseline_set(), moments, household=home
            )
        homes[home] = entry
    return {
        "schema": INPUTS_SCHEMA,
        "homes": homes,
        "fit_seconds": round(fitted, 1),
        "peak_memory_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        ),
        "scored": "nothing: no prediction was compared with a label",
    }


# ----------------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------------
def truth_counts(metrics: PredictionMetrics) -> dict[str, int]:
    """How many scored windows a model had in each true state."""
    confusion = metrics.confusion.to_dict()
    rows = zip(confusion["truth"], confusion["counts"])  # type: ignore[call-overload]
    return {state: int(sum(row)) for state, row in rows if sum(row)}


def _gap_entry(
    estimand: Estimand,
    scores: Mapping[str, Mapping[str, PredictionMetrics]],
    protocol: HeldOutProtocol,
) -> dict[str, Any]:
    base = protocol.base
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
    return {
        **estimand.to_dict(),
        "rule": "gap",
        "comparisons": comparisons,
        "verdicts": verdicts,
        "conclusion": GAP_READING[verdicts["balanced_accuracy"]],
    }


def by_family(
    scores: Mapping[str, Mapping[str, PredictionMetrics]], homes: Sequence[str]
) -> dict[str, Any]:
    """Each family's mean balanced accuracy per recursion, and of K1."""
    out: dict[str, Any] = {}
    for name in sorted({family(home) for home in homes}):
        members = [home for home in homes if family(home) == name]

        def mean(model: str, members: list[str] = members) -> float | None:
            values = [
                v
                for v in (scores[model][h].balanced_accuracy for h in members)
                if v is not None and np.isfinite(v)
            ]
            return float(np.mean(values)) if values else None

        levels = {model: mean(model) for model in MODELS}
        first, second = levels[FILTER_PERIODIC_HURDLE], levels["filter_hurdle"]
        out[name] = {
            "households": len(members),
            "balanced_accuracy": levels,
            "k1": None if first is None or second is None else first - second,
        }
    return out


def _recall_verdicts(entry: dict[str, Any]) -> None:
    for state in entry.get("per_state_recall", {}).values():
        if state["comparison"]["n"] < MIN_RECALL_HOMES:
            state["verdict"] = "too few homes"


@dataclass(frozen=True)
class HeldOutResult:
    """A completed confirmation: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def run_heldout(
    recordings: Mapping[str, CasasRecording],
    protocol: HeldOutProtocol,
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> HeldOutResult:
    """Fit on the training homes, score the held-out homes once, and record it."""
    base = protocol.base
    missing = sorted({*protocol.training, *protocol.test} - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    ontology = StateOntology()
    states = tuple(ontology.states)
    gap = base.as_gap_protocol()
    step = base.information_sets[0].resolution.step
    unscored = [
        home
        for home in protocol.test
        if all(
            label is None
            for label in truth_series(
                recordings[home].activities, _regular_moments(recordings[home], step)
            )
        )
    ]
    scored = [home for home in protocol.test if home not in unscored]
    fit = training_fit(recordings, protocol.training, base, ontology)

    run = run_matched_evaluation(
        recordings,
        split=protocol.split,
        information_set=protocol.baseline_set(),
        models=protocol.baselines(),
        seed=base.seed,
        data_source=data_source,
        metrics=base.metrics,
        resamples=base.resamples,
        confidence=base.confidence,
        inputs=inputs,
        experiment=f"{protocol.name}-{BASELINE_SET}",
    )
    scores: dict[str, dict[str, PredictionMetrics]] = {
        spec.name: held_out_metrics(run, spec.name) for spec in protocol.baselines()
    }
    absent = sorted(
        {h for per_home in scores.values() for h in set(scored) - set(per_home)}
    )
    if absent:
        raise ValueError(f"the baselines did not score {absent}; nothing is reported")
    for model in MODELS:
        scores[model] = {}
    for home in scored:
        for model, metrics in household_scores(
            recordings[home],
            fit,
            household=home,
            ontology=ontology,
            protocol=protocol.scoring,
        ).items():
            scores[model][home] = metrics
    differing = sorted(
        home
        for home in scored
        if len(
            {
                json.dumps(truth_counts(per_home[home]), sort_keys=True)
                for per_home in scores.values()
            }
        )
        != 1
    )
    if differing:
        raise ValueError(
            f"the models scored different windows in {differing}; nothing is reported"
        )

    estimands = [
        _estimand_entry(e, scores, protocol.scoring, states)
        for e in DEVELOPMENT_ESTIMANDS
    ]
    for entry in estimands:
        if entry["key"] == "K1":
            _recall_verdicts(entry)
        else:
            entry.pop("per_state_recall", None)
    estimands += [_gap_entry(e, scores, protocol) for e in GAP_ESTIMANDS]
    cells = [
        _cell(model, RECURSION, scores[model], states, gap) for model in MODELS
    ] + [
        _cell(spec.name, BASELINE_SET, scores[spec.name], states, gap)
        for spec in protocol.baselines()
    ]
    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified, one-shot; CASAS homes outside the development "
        "panel, scored once before by the production filter",
        "check": {
            "windows": "identical",
            "households": len(scored),
            "unscored": unscored,
        },
        "states": [state.value for state in states],
        "training": {
            "homes": list(protocol.training),
            "channels_sha256": fit.channels.sha256(),
            "periodic_prior_sha256": fit.periodic.sha256(),
        },
        "cells": cells,
        "estimands": estimands,
        "interaction": interaction(scores, protocol.scoring),
        "by_family": by_family(scores, scored),
        "conclusion": next(e["conclusion"] for e in estimands if e["key"] == "K1"),
    }
    record = ExperimentRecord(
        experiment=protocol.name,
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        inference=ONLINE,
        evidence=online_evidence(recordings, scored, step),
        seeds=[base.seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions=MATCHED_METRIC_DEFINITIONS,
        notes=[
            "Every setting, estimand and criterion was declared in the protocol "
            "before any held-out home was scored.",
            f"Everything was fitted once, on the {len(protocol.training)} "
            "development homes only.",
            "Verdicts compare effect sizes and household bootstrap intervals with "
            "declared minimal differences; they are not significance tests.",
        ],
        inputs=list(inputs),
        split={"fold": protocol.split.to_dict()},
        models=[
            *(
                ModelRecord(
                    name,
                    {"channels": channels, "time_prior": timed},
                    "periodic_recursion" if timed else "filter_recursion",
                )
                for name, (channels, timed) in MODELS.items()
            ),
            *(
                ModelRecord(spec.name, dict(spec.configuration), BASELINE_SET)
                for spec in protocol.baselines()
            ),
        ],
        household_metrics={
            f"{model}@{RECURSION if model in MODELS else BASELINE_SET}": {
                h: m.to_dict() for h, m in per_home.items()
            }
            for model, per_home in scores.items()
        },
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return HeldOutResult(record=record, path=path)


__all__ = [
    "GAP_ESTIMANDS",
    "PINNED_RECORDS",
    "HeldOutProtocol",
    "HeldOutResult",
    "by_family",
    "check_frozen_protocol",
    "check_pinned_records",
    "code_changes",
    "code_now",
    "declared_protocol",
    "family",
    "inputs_report",
    "load_cohort",
    "load_inputs",
    "run_heldout",
    "training_fit",
    "truth_counts",
    "write_protocol",
]
