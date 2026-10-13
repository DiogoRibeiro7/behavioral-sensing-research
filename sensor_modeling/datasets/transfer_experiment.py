"""The transfer test: the confirmed formulation on ContextAct@A4H, an independent dataset.

The Phase 3 formulation, fitted hurdle channels with the hierarchical time prior
in the filter's recursion, was confirmed on 43 held-out CASAS homes. Those homes
share CASAS's sensors and labelling. This experiment asks whether the time
prior's gain over the fitted channels holds on data collected independently of
CASAS: the one resident of ContextAct@A4H, in a smart apartment in France, over
two periods of 7 and 21 days, read by
:mod:`sensor_modeling.external.contextact` under its frozen mapping.

Everything is fitted once, on the 20 CASAS development homes, as the held-out
confirmation fitted it (:func:`~sensor_modeling.datasets.heldout_experiment.training_fit`);
nothing is fitted on ContextAct. The run first reproduces that fit: the fitted
channels must have the held-out record's digest, and the time prior must agree
with the parameters the package ships, which are then used. It then checks that
the online pipeline's ``phase3`` opt-in gives the recursion's beliefs in both
periods, so the result is about the formulation as it ships.

With one resident, households cannot be resampled. The unit is a local calendar
day. Each model is scored on each day's labelled windows, an estimate is a mean
over days, and the bootstrap resamples days. The recursions run over each period
as a whole and are scored day by day. The baselines are scored by the matched
runner with one unit per day, reading the hour and the last three windows from
the whole period. Every model must score the same number of windows in each true
state on each day, or nothing is reported.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.households import household_values
from ..evaluation.metrics import PredictionMetrics
from ..evaluation.provenance import (
    ExperimentRecord,
    InputArtifact,
    ModelRecord,
    load_record,
)
from ..external import to_canonical, validate_household
from ..external.contextact import (
    CONTEXTACT_MAPPING,
    FILES,
    HOUSEHOLDS,
    MAPPING_FILE,
    TIMEZONE,
    ContextActAdapter,
    eligibility,
)
from ..fusion.regime import ONLINE, EvidenceSummary
from ..online.pipeline import (
    BehaviouralSensingPipeline,
    PipelineConfig,
    scoring_steps,
)
from ..states.ontology import BehaviouralState, StateOntology
from . import rates_experiment
from .baseline_models import baseline_suite
from .casas import ActivityInterval, CasasRecording
from .channel_models import filter_recursion, household_channel_counts, total_loglik
from .combined_prior_experiment import ESTIMANDS as DEVELOPMENT_ESTIMANDS
from .combined_prior_experiment import (
    FILTER_DECLARED,
    FILTER_PERIODIC_HURDLE,
    MODELS,
    RECURSION,
    RULES,
    CombinedPriorProtocol,
    _estimand_entry,
    interaction,
    periodic_recursion,
    periodic_transitions,
)
from .heldout_experiment import (
    BASELINE_SET,
    GAP_ESTIMANDS,
    _gap_entry,
    file_sha256,
    training_fit,
    truth_counts,
)
from .history_experiment import Estimand
from .information_sets import EvidenceResolution, InformationSet, build_feature_table
from .matched_evaluation import (
    MATCHED_METRIC_DEFINITIONS,
    HouseholdSplit,
    ModelSpec,
    _finite,
    _regular_moments,
    held_out_metrics,
    run_matched_evaluation,
)
from .phase3_formulation import (
    CHANNELS_SHA256,
    HELDOUT_RECORD,
    PARAMETERS_FILE,
    Phase3Formulation,
)
from .rates_experiment import FittedRatesProtocol, FoldFit
from .recoverable_gap import (
    DIAGNOSTIC,
    FrozenSplits,
    _cell,
    _paired,
    _score,
    gap_models,
)
from .time_prior_experiment import verdict

PROTOCOL_SCHEMA = "contextact-transfer-protocol/1"
RESULT_SCHEMA = "contextact-transfer-results/1"
INPUTS_SCHEMA = "contextact-transfer-inputs/1"
EXPERIMENT = "contextact-transfer"
#: The fold name the matched runner records: the development homes, and days.
SPLIT = "development-20-contextact-days"

PROTOCOL_FILE = "artifacts/contextact/transfer_protocol.json"
INPUTS_FILE = "artifacts/contextact/transfer_inputs.json"
RECORD_FILE = f"artifacts/contextact/{EXPERIMENT}.json"
#: The package file the shipped parameters are read from.
PARAMETERS_PATH = f"sensor_modeling/data/{PARAMETERS_FILE}"

#: A day enters the estimands only with at least this many labelled windows,
#: one hour at the five-minute step.
MIN_DAY_WINDOWS = 12
#: A state's K1 recall verdict needs at least this many scored days with the
#: state; below it the difference is described, not judged.
MIN_RECALL_DAYS = 5
#: How far the refitted time prior's coefficients may be from the shipped ones.
#: Its Newton fit rounds differently on other CPUs, by about 1e-13.
PRIOR_TOLERANCE = 1e-9
#: How far the online pipeline's beliefs may be from the recursion's.
PIPELINE_TOLERANCE = 1e-9
#: The minimal difference for balanced accuracy against chance, as in Phase 5.
CHANCE_DIFFERENCE = 0.05
#: K1 is judged only with at least this many scored days; with fewer, its
#: conclusion is inconclusive.
MIN_SCORED_DAYS = 10
#: A day enters C1 only with at least this many labelled states: with one, chance
#: is 1 and cannot be beaten.
MIN_CHANCE_STATES = 2
#: The library versions the run must use: the held-out confirmation's.
ENVIRONMENT = {
    "python": "3.13",
    "numpy": "2.3.5",
    "pandas": "2.3.3",
    "scipy": "1.15.3",
    "scikit-learn": "1.6.1",
}
#: The workflow that runs the protocol; only its runs on develop count.
WORKFLOW = ".github/workflows/run-contextact-transfer.yml"

#: The records the protocol rests on, and their digests.
PINNED_RECORDS = {
    HELDOUT_RECORD: "785f6757ed9e9890dc7dfbf471660403779cc7dcc6a246586a87fd2323e2c97a",
    "artifacts/heldout/heldout_protocol.json": (
        "a1196911c9f07ded4e06ddad4e2df56aff344ef8223695e456bdb657ae77b23a"
    ),
    "artifacts/phase3/fitted_rates_protocol.json": (
        "f695f972fcfcd11266a8a1a720932442e8b7d455bdec7dd6304ecab41843075d"
    ),
    MAPPING_FILE: "f837a24f2b305d21914d628c0d9c0534fe39a13f7cbd9113949d9465a1db5d58",
}
#: The sources whose content the freeze records and the run requires unchanged.
PINNED_PACKAGE = "sensor_modeling"
PINNED_FILES = (
    "pyproject.toml",
    PARAMETERS_PATH,
    "scripts/freeze_contextact_transfer.py",
    "scripts/run_contextact_transfer.py",
    WORKFLOW,
)
PINNED_DISTRIBUTIONS = ("numpy", "scipy", "pandas", "scikit-learn")
_REPOSITORY = Path(__file__).resolve().parents[2]

#: The combination against the declared model, in the recursion.
LEVEL_ESTIMANDS: tuple[Estimand, ...] = (
    Estimand(
        "L1",
        "secondary",
        "level",
        "how far the combination is from the declared model, the production "
        "filter's own emissions without the time prior",
        FILTER_PERIODIC_HURDLE,
        RECURSION,
        FILTER_DECLARED,
        RECURSION,
    ),
)
LEVEL_READING = {
    "favours model": "the combination leads",
    "favours reference": "the declared model leads",
    "negligible": "no gap",
    "uncertain": "uncertain",
}
#: The combination's balanced accuracy against chance.
CHANCE = Estimand(
    "C1",
    "secondary",
    "chance",
    "whether the combination's balanced accuracy is above chance, one over the "
    "number of states labelled that day, on days with at least two",
    FILTER_PERIODIC_HURDLE,
    RECURSION,
    "chance",
    "day",
)
CHANCE_READING = {
    "favours model": "above chance",
    "favours reference": "below chance",
    "negligible": "not above chance",
    "uncertain": "uncertain",
}

INSPECTED_BEFORE = (
    "Nobody on this project had read ContextAct@A4H before the search for an "
    "independent dataset for this test. Its README, description file, apartment "
    "plan and paper were then read in full.",
    "Before the inputs check, from its change logs were read: each log's time "
    "span, the sensor identifiers present, and the value tokens of the motion, "
    "presence, door and bed sensors; of the annotations, only the distinct "
    "activity labels and markers, to declare each in the mapping. No "
    "annotation's time, duration or count was read then, nor any self-annotated "
    "position or number of people.",
    "The adapter was run on both periods to check that the mapping declares "
    "every label, sensor type and location, and to list which sensors feed which "
    "channel. No model was run.",
    "Before this protocol was frozen, the inputs check, a run of the workflow on "
    "develop named in the report it writes, read both periods to write that "
    "report: each period's validation issue codes and counts, the conversion's "
    "dispositions, including annotated seconds left unscored by reason, its "
    "channels, windows, labelled windows on each day and the states that occur. "
    "It recorded the digest of the protocol as declared without the report, so "
    "every setting here is shown to precede it. It ran the recursions to check "
    "their posteriors are finite and the online pipeline to check it gives the "
    "recursion's beliefs, and ran the whole experiment once on surrogate labels "
    "assigned by the hour, keeping only that it completed; no prediction was "
    "compared with a true label.",
    "The development result: on the 20 development homes, cross-fitted, the time "
    "prior adds +0.061 [+0.044, +0.077] balanced accuracy to the fitted hurdle "
    "channels in the recursion, 0.517 against 0.456 (K1, a success).",
    "The held-out confirmation: fitted on the 20 development homes and scored "
    "once on 43 held-out CASAS homes, +0.077 [+0.070, +0.084], 0.567 against "
    "0.490 (K1, a success); the declared model scored 0.482 and the diagnostic "
    "at I3 0.557, not distinguishable from the combination, -0.010 "
    "[-0.033, +0.014].",
    "On the first independent dataset, the two homes of the UCI ADL Binary "
    "dataset, the CASAS-trained model of Phase 5 did not transfer, and its "
    "balanced accuracy was not above chance in either home.",
)


# ----------------------------------------------------------------------------
# Provenance of the code
# ----------------------------------------------------------------------------
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


def code_digest(root: Path | None = None) -> str:
    """SHA-256 of every pinned source's digest, now."""
    return hashlib.sha256(
        json.dumps(code_now(root)["sources"], sort_keys=True).encode("utf-8")
    ).hexdigest()


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


# ----------------------------------------------------------------------------
# The protocol
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TransferProtocol:
    """Everything the transfer test fixes before ContextAct is scored.

    Attributes
    ----------
    base
        The fitted-rates protocol, whose channel and time-prior settings,
        information sets, metrics, minimal differences and bootstrap settings
        are used unchanged. Its 20 households are the training set.
    periods
        The ContextAct recording periods, the test set.
    files
        Every ContextAct file read, with its SHA-256.
    inputs
        The inputs report, :func:`inputs_report`, with its digest under
        ``"sha256"``. A protocol without one cannot be frozen.
    channels_sha256
        The digest the refitted channels must have: the held-out record's.
    """

    base: FittedRatesProtocol
    periods: tuple[str, ...] = HOUSEHOLDS
    files: Mapping[str, str] = field(default_factory=lambda: dict(FILES))
    inputs: Mapping[str, Any] | None = None
    channels_sha256: str = CHANNELS_SHA256
    inspected_before: tuple[str, ...] = INSPECTED_BEFORE
    name: str = EXPERIMENT

    def __post_init__(self) -> None:
        if not self.periods or len(set(self.periods)) != len(self.periods):
            raise ValueError("the protocol must name distinct periods")
        if set(self.periods) & set(self.training):
            raise ValueError("a period shares its name with a training home")
        if self.inputs is not None and set(self.inputs["periods"]) != set(self.periods):
            raise ValueError("the inputs report does not cover exactly these periods")

    @property
    def training(self) -> tuple[str, ...]:
        """The training households: every development household."""
        return self.base.homes

    @property
    def scoring(self) -> CombinedPriorProtocol:
        """The combined-prior settings the recursions and estimands reuse."""
        return CombinedPriorProtocol(self.base)

    @property
    def resolution(self) -> EvidenceResolution:
        """The evidence resolution of every information set."""
        return self.base.information_sets[0].resolution

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

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        base = self.base
        return {
            "schema": PROTOCOL_SCHEMA,
            "name": self.name,
            "status": "pre-specified, one-shot, on a dataset collected "
            "independently of CASAS and never scored by any model in this "
            "repository",
            "question": "whether the time prior's gain over the fitted hurdle "
            "channels in the recursion, a success on the development panel and on "
            "43 held-out CASAS homes, holds on the one resident of ContextAct@A4H",
            "inspected_before": list(self.inspected_before),
            "base_protocol": {
                "file": "artifacts/phase3/fitted_rates_protocol.json",
                "protocol_sha256": base.sha256(),
                "used": "its channel families and shrinkage, its time prior, its "
                "information sets, metrics, minimal differences and bootstrap "
                "settings; not its folds",
            },
            "training": {
                "homes": list(self.training),
                "splits_sha256": base.splits_sha256,
                "fitted": "the channels, the time prior and every baseline are "
                "fitted once, on all 20 development homes, exactly as the held-out "
                "confirmation fitted them; nothing is fitted on ContextAct",
                "reproduction": {
                    "channels_sha256": self.channels_sha256,
                    "channels": "the refitted channels must have this digest, the "
                    "held-out record's, or nothing is reported",
                    "time_prior": "the refitted prior's coefficients must be within "
                    f"{PRIOR_TOLERANCE:g} of those in {PARAMETERS_PATH}, or nothing "
                    "is reported; the shipped prior is then the one used",
                    "pipeline": "in each period the online pipeline with "
                    "formulation phase3 must give the combination's beliefs to "
                    f"within {PIPELINE_TOLERANCE:g} at every window, or nothing "
                    "is reported",
                },
            },
            "dataset": {
                "name": "ContextAct@A4H",
                "periods": list(self.periods),
                "files": dict(sorted(self.files.items())),
                "mapping": {
                    "file": MAPPING_FILE,
                    "mapping_sha256": CONTEXTACT_MAPPING.sha256(),
                },
                "reading": "each period is read by "
                "sensor_modeling.external.contextact, validated under the frozen "
                "mapping and converted to the canonical form; a period whose "
                "validation has an error other than unpaired annotations or "
                "undeclared labels is excluded and listed. Local times are read "
                f"in the {TIMEZONE} zone, and the hour the time prior reads is the "
                "local hour; a period whose offset from UTC changes, across a "
                "daylight-saving change, stops the run. If only one period is "
                "eligible, the estimands use its days alone",
                "adaptation": "none: no parameter is fitted or adapted on ContextAct",
            },
            "instrumentation": (
                {
                    "report": {"file": INPUTS_FILE, "sha256": self.inputs["sha256"]},
                    **self.inputs["summary"],
                }
                if self.inputs is not None
                else {}
            ),
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
                "missing_channels": "a channel the apartment does not instrument, "
                "hall motion, has no term in the recursion; the baselines see its "
                "columns as missing, encoded as zero beside a missing indicator, "
                "and where every training home instruments it that indicator never "
                "varied in training, so they read the absent channel as a silent hall",
            },
            "units": {
                "unit": "a local calendar day of a period",
                "scored": "a day's labelled windows, at the base protocol's step of "
                f"{self.resolution.step.total_seconds():g} seconds",
                "minimum": f"a day with fewer than {MIN_DAY_WINDOWS} labelled "
                "windows is listed as unscored and enters no estimand",
                "recursions": "each period's recursion runs over the whole period "
                "from its first window, and is scored day by day",
                "baselines": "the matched runner scores each day as a unit, reading "
                "the hour and the last three windows from the whole period",
                "identical windows": "on every day every model must score the same "
                "number of windows in each true state, or nothing is reported",
            },
            "estimands": [
                {**e.to_dict(), "rule": rule}
                for e, rule in (
                    *((e, RULES[e.key]) for e in DEVELOPMENT_ESTIMANDS),
                    *((e, "gap") for e in GAP_ESTIMANDS),
                    *((e, "level") for e in LEVEL_ESTIMANDS),
                    (CHANCE, "chance"),
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
                "level": "read by balanced accuracy: the combination leads, the "
                "declared model leads, no gap, or uncertain; every metric's verdict "
                "is reported",
                "chance": "the combination's balanced accuracy minus chance per day, "
                f"on the days with at least {MIN_CHANCE_STATES} labelled states, "
                f"with a minimal difference of {CHANCE_DIFFERENCE:g}: above chance, "
                "below chance, not above chance when negligible, or uncertain; the "
                "days left out are counted",
                "primary": "K1 alone decides: success means the time prior adds to "
                "the fitted hurdle channels for this resident of an independent "
                "dataset; failure that it does not; inconclusive is reported as "
                f"such. With fewer than {MIN_SCORED_DAYS} scored days, K1 is "
                "inconclusive whatever its verdicts",
                "multiplicity": "K2, K3, G1, G2, L1 and C1 are secondary; nothing is "
                "adjusted",
            },
            "per_state_recall": {
                "estimands": ["K1"],
                "verdict": "the same verdicts with the recall minimal difference, "
                f"for a state that occurs on at least {MIN_RECALL_DAYS} scored days; "
                "for fewer the difference is described, not judged, and a state "
                "labelled on no scored day is listed as never labelled",
            },
            "interaction": "K1 minus K2 per day, described as in the combined-prior "
            "protocol",
            "by_period": "each period's mean of every recursion's balanced accuracy "
            "and of K1, described, not judged",
            "metrics": list(base.metrics),
            "minimal_differences": {
                **dict(base.minimal_differences),
                "chance": CHANCE_DIFFERENCE,
            },
            "bootstrap": {
                "unit": "day",
                "reason": "one resident supports no between-household inference; "
                "days keep within-day dependence intact",
                "statistics": "mean and median paired differences over days",
                "resamples": base.resamples,
                "confidence": base.confidence,
                "seed": base.seed,
                "interval": "percentile",
            },
            "environment": dict(ENVIRONMENT),
            "one_shot": "only a run of the workflow "
            f"{WORKFLOW} on develop counts. The workflow refuses if its results "
            "branch exists, and the run refuses if its record exists in the "
            "repository, if any pinned source file differs from the freeze, or if "
            "the installed Python or libraries are not the declared ones. Whatever "
            "it reports stands, and nothing may be changed in response and rescored "
            "as the same test. A run that stops before writing its record reports "
            "nothing; a fix made without seeing any outcome is frozen as a new "
            "protocol file that names this one and the fix, and the run repeated",
            "pinned_records": dict(PINNED_RECORDS),
            "what_this_cannot_show": [
                "Anything about a population of homes or residents: ContextAct@A4H "
                "is one resident, in one smart apartment, who was part of the "
                "research group that ran the experiment.",
                "Intervals for other residents: the bootstrap resamples days of one "
                "person, which are not independent of one another, so the "
                "intervals are narrower than the uncertainty about this resident, "
                "and say nothing about anyone else.",
                "Independence of the two periods: they are the same person and "
                "apartment, months apart, and their days are pooled.",
                "The states no label names: bed_awake has no label, and the "
                "ambiguous labels, Musique and Loisir, are not scored. The "
                "resident annotated part of each day, reported missing some "
                "annotations, and labelled her own activities, not the "
                "CASAS annotators' states.",
                "Why a failure happened: the apartment's open ground floor, its "
                "pooled living-room channel, the missing hall motion, the "
                "different annotation conventions and the resident's own routine "
                "cannot be separated from non-transfer.",
                "A first look at the data: the documentation, the sensor encodings "
                "and the label vocabulary were read, and the inputs report was "
                "written, before the freeze.",
                "A fair test of the baselines: they read the missing hall motion as "
                "a silent hall, a shift that penalises G1 and G2 alone.",
                "Days of equal information: every scored day counts equally, from "
                f"{MIN_DAY_WINDOWS} labelled windows to a whole day.",
                "The clock: the logs give no timezone; local time in "
                f"{TIMEZONE} is assumed, and if they were in UTC the time prior "
                "would read hours one or two hours off.",
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
) -> TransferProtocol:
    """The protocol as declared, on the frozen fitted-rates protocol.

    With *inputs*, the committed inputs report is part of it; without, the
    protocol is the one the inputs check runs under and cannot be frozen.
    """
    base = rates_experiment.declared_protocol(splits)
    rates_experiment.check_frozen_protocol(
        base, Path(root) / "artifacts/phase3/fitted_rates_protocol.json"
    )
    check_pinned_records(root)
    return TransferProtocol(
        base, inputs=load_inputs(Path(root) / INPUTS_FILE) if inputs else None
    )


def write_protocol(protocol: TransferProtocol, path: Path) -> str:
    """Write the declaration, its digest, the moment and the code it was frozen with."""
    if protocol.inputs is None:
        raise ValueError("a protocol cannot be frozen without its inputs report")
    declared = replace(protocol, inputs=None).sha256()
    if protocol.inputs.get("declared_protocol_sha256") != declared:
        raise ValueError(
            "the inputs report was written under another declaration; every "
            "setting must precede the report"
        )
    if protocol.inputs.get("code_sha256") != code_digest():
        raise ValueError(
            "the pinned sources changed after the inputs report was written"
        )
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


def check_frozen_protocol(protocol: TransferProtocol, path: Path) -> str:
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
# Reading ContextAct, the fit and the checks
# ----------------------------------------------------------------------------
def read_periods(
    directory: Path, periods: Sequence[str] = HOUSEHOLDS
) -> tuple[dict[str, CasasRecording], dict[str, Any]]:
    """Each eligible period in the canonical form, and every period's validation.

    The validation entries name each issue's code, severity and count, and no
    example: examples could quote annotations.
    """
    adapter = ContextActAdapter(Path(directory))
    recordings: dict[str, CasasRecording] = {}
    reports: dict[str, Any] = {}
    for period in periods:
        data = adapter.load(period)
        report = validate_household(data, CONTEXTACT_MAPPING)
        eligible, blocking = eligibility(report)
        entry: dict[str, Any] = {
            "eligible": eligible,
            "blocking": blocking,
            "issues": {
                issue.code: {"severity": issue.severity.value, "count": issue.count}
                for issue in report.issues
            },
        }
        if eligible:
            converted = to_canonical(
                data, CONTEXTACT_MAPPING, strict=False, source="contextact-a4h"
            )
            recordings[period] = converted.recording
            entry["dispositions"] = _finite(dict(converted.dispositions))
        reports[period] = entry
    return recordings, reports


def reproduced_fit(
    training: Mapping[str, CasasRecording],
    protocol: TransferProtocol,
    formulation: Phase3Formulation,
) -> tuple[FoldFit, float]:
    """The held-out confirmation's fit, refitted, checked and with the shipped prior.

    Returns the fit and how far the refitted prior's coefficients were from the
    shipped ones.
    """
    fit = training_fit(training, protocol.training, protocol.base, StateOntology())
    if fit.channels.sha256() != protocol.channels_sha256:
        raise ValueError(
            "the refitted channels are not the held-out confirmation's; nothing "
            "is reported"
        )
    shipped = formulation.prior
    if fit.periodic.population.shape != shipped.population.shape:
        raise ValueError("the refitted time prior has another shape than the shipped")
    difference = float(np.abs(fit.periodic.population - shipped.population).max())
    if difference > PRIOR_TOLERANCE:
        raise ValueError(
            f"the refitted time prior is {difference:.3g} from the shipped one; "
            "nothing is reported"
        )
    return replace(fit, periodic=shipped), difference


def day_key(period: str, moment: datetime) -> str:
    """The unit a window belongs to: its period and local calendar date."""
    return f"{period}/{moment.date().isoformat()}"


def recursion_posteriors(
    recording: CasasRecording,
    fit: FoldFit,
    *,
    period: str,
    resolution: EvidenceResolution,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[BehaviouralState], list[datetime]]:
    """Every recursion's posterior over a period, as the held-out run computes it.

    Returns the posteriors, ``(windows, states)`` each, the labelled rows, their
    states and every window's moment.
    """
    ontology = StateOntology()
    states = tuple(ontology.states)
    counts, rows, labels, moments = household_channel_counts(
        recording, resolution, ontology, household=period
    )
    hours = [moment.hour for moment in moments]
    homogeneous = ontology.transition(resolution.step)
    timed = periodic_transitions(fit.periodic, ontology, resolution.step)
    start = fit.periodic.probabilities([hours[0]])[0]
    posteriors: dict[str, np.ndarray] = {}
    for name, (family, with_prior) in MODELS.items():
        models = fit.channels.models(recording.registry, resolution, family, ontology)
        loglik = total_loglik(models, counts)
        if with_prior:
            _, posteriors[name] = periodic_recursion(loglik, timed, hours, start)
        else:
            _, posteriors[name] = filter_recursion(
                loglik, homogeneous, ontology.stationary()
            )
    return posteriors, rows, [states[int(i)] for i in labels], list(moments)


def check_clock(
    recording: CasasRecording, period: str, moments: Sequence[datetime]
) -> None:
    """Refuse a period whose offset from UTC changes, across daylight saving."""
    if len({moment.utcoffset() for moment in moments}) != 1:
        raise ValueError(
            f"{period} crosses a change of UTC offset; nothing is reported"
        )


def labelled_days(
    period: str, rows: np.ndarray, moments: Sequence[datetime]
) -> dict[str, int]:
    """How many labelled windows each day of a period has."""
    return dict(sorted(Counter(day_key(period, moments[int(r)]) for r in rows).items()))


def split_days(counts: Mapping[str, int]) -> tuple[list[str], dict[str, int]]:
    """The days with enough labelled windows, and the others with their counts."""
    kept = [day for day, n in counts.items() if n >= MIN_DAY_WINDOWS]
    return kept, {day: n for day, n in counts.items() if n < MIN_DAY_WINDOWS}


def day_scores(
    period: str,
    posteriors: Mapping[str, np.ndarray],
    rows: np.ndarray,
    truth: Sequence[BehaviouralState],
    moments: Sequence[datetime],
    days: Sequence[str],
) -> dict[str, dict[str, PredictionMetrics]]:
    """Each recursion's metrics on each of *days*, from the period's posteriors."""
    states = tuple(StateOntology().states)
    members: dict[str, list[int]] = {}
    for position, row in enumerate(rows):
        members.setdefault(day_key(period, moments[int(row)]), []).append(position)
    scores: dict[str, dict[str, PredictionMetrics]] = {model: {} for model in MODELS}
    for day in days:
        positions = members[day]
        selected = rows[positions]
        labels = [truth[p] for p in positions]
        for model in MODELS:
            scores[model][day] = _score(labels, posteriors[model][selected], states)
    return scores


def pipeline_difference(
    recording: CasasRecording, posterior: np.ndarray, moments: Sequence[datetime]
) -> float:
    """How far the online pipeline's beliefs are from the combination's, at most.

    The pipeline runs with ``formulation="phase3"``, on the parameters the
    package ships.
    """
    pipeline = BehaviouralSensingPipeline(
        recording.registry, config=PipelineConfig(formulation="phase3")
    )
    steps = scoring_steps(pipeline.run(recording.observations))
    beliefs = {step.at: step.state.belief for step in steps}
    missing = [moment for moment in moments if moment not in beliefs]
    if missing:
        raise ValueError(
            f"the pipeline did not step at {len(missing)} of the recursion's windows"
        )
    return float(
        max(np.abs(beliefs[m] - posterior[i]).max() for i, m in enumerate(moments))
    )


#: The states the dry run's surrogate labels cycle through, one per hour.
SURROGATE_STATES = (
    BehaviouralState.SLEEPING,
    BehaviouralState.AWAY,
    BehaviouralState.HOME_ACTIVE,
    BehaviouralState.KITCHEN_ACTIVITY,
    BehaviouralState.HOME_INACTIVE,
    BehaviouralState.BATHROOM_ACTIVITY,
)


def surrogate(recording: CasasRecording) -> CasasRecording:
    """The recording with its labels replaced by states assigned by the hour.

    The dry run scores these, so that the whole experiment runs once before the
    freeze without any true label being compared with a prediction.
    """
    stamps = [o.timestamp for o in recording.observations]
    start = min(stamps).replace(minute=0, second=0, microsecond=0)
    end = max(stamps)
    activities = []
    moment = start
    while moment < end:
        following = moment + timedelta(hours=1)
        state = SURROGATE_STATES[moment.hour % len(SURROGATE_STATES)]
        activities.append(ActivityInterval(state.value, moment, following, state))
        moment = following
    return replace(recording, activities=tuple(activities))


def _peak_memory_mb() -> int | None:
    """Peak resident memory in MB, where the platform reports it."""
    try:
        import resource
    except ImportError:  # pragma: no cover - not on Windows
        return None
    usage = resource.getrusage(  # type: ignore[attr-defined, unused-ignore]
        resource.RUSAGE_SELF  # type: ignore[attr-defined, unused-ignore]
    )
    return int(round(usage.ru_maxrss / 1024))


def _instrumented(
    recording: CasasRecording, period: str, protocol: TransferProtocol
) -> tuple[list[str], int, int]:
    current = protocol.base.information_sets[0]
    moments = _regular_moments(recording, current.resolution.step)
    table = build_feature_table(recording, current, moments, household=period)
    channels = sorted(
        c.name for c in set(current.resolution.channels) - set(table.uninstrumented)
    )
    return channels, len(table.excluded_sensors), len(moments)


def inputs_report(
    training: Mapping[str, CasasRecording],
    periods: Mapping[str, CasasRecording],
    validation: Mapping[str, Any],
    protocol: TransferProtocol,
) -> dict[str, Any]:
    """What each period instruments and labels, and whether everything runs.

    Fits on the training homes and checks the fit reproduces. For each eligible
    period it reads the sensors to build the current-window features and, of its
    annotations, only which windows are labelled, on which day, and which states
    occur. It runs every recursion to check its posterior is finite, the online
    pipeline to check it gives the combination's beliefs, and builds the
    baselines' rows. No model is scored.
    """
    if protocol.inputs is not None:
        raise ValueError("the inputs check runs under the protocol without its report")
    shipped = Phase3Formulation.load()
    started = time.perf_counter()
    fit, prior_difference = reproduced_fit(training, protocol, shipped)
    fitted = time.perf_counter() - started
    entries: dict[str, Any] = {}
    for period in protocol.periods:
        entry: dict[str, Any] = {"validation": validation[period]}
        recording = periods.get(period)
        if recording is not None:
            posteriors, rows, truth, moments = recursion_posteriors(
                recording, fit, period=period, resolution=protocol.resolution
            )
            check_clock(recording, period, moments)
            days = labelled_days(period, rows, moments)
            channels, excluded, windows = _instrumented(recording, period, protocol)
            entry.update(
                {
                    "observations": len(recording.observations),
                    "instrumented": channels,
                    "excluded_sensors": excluded,
                    "windows": windows,
                    "labelled": len(rows),
                    "states": sorted({state.value for state in truth}),
                    "labelled_windows_by_day": days,
                    "scored_days": split_days(days)[0],
                    "recursions_finite": all(
                        bool(np.isfinite(p).all()) for p in posteriors.values()
                    ),
                    "pipeline_difference": pipeline_difference(
                        recording, posteriors[FILTER_PERIODIC_HURDLE], moments
                    ),
                }
            )
            build_feature_table(  # the baselines' rows, built only to see they build
                recording, protocol.baseline_set(), moments, household=period
            )
        entries[period] = entry
    eligible = [p for p in protocol.periods if periods.get(p) is not None]
    from .transfer_summary import render_page  # the pages import this module

    checked = time.perf_counter()
    with tempfile.TemporaryDirectory() as scratch:
        dry = run_transfer(
            training,
            {p: surrogate(periods[p]) for p in eligible},
            validation,
            protocol,
            data_source="surrogate labels",
            output_dir=Path(scratch),
        )
        if dry.path is None:  # pragma: no cover - an output directory is given
            raise ValueError("the dry run wrote no record")
        render_page(load_record(dry.path))
    return {
        "schema": INPUTS_SCHEMA,
        "declared_protocol_sha256": protocol.sha256(),
        "code_sha256": code_digest(),
        "run": os.environ.get("GITHUB_RUN_ID"),
        "commit": os.environ.get("GITHUB_SHA"),
        "periods": entries,
        "summary": {
            "eligible_periods": eligible,
            "instrumented": {p: entries[p]["instrumented"] for p in eligible},
            "scored_days": sum(len(entries[p]["scored_days"]) for p in eligible),
            "periods_with_each_state": dict(
                sorted(
                    Counter(
                        state for p in eligible for state in entries[p]["states"]
                    ).items()
                )
            ),
        },
        "fit": {
            "channels_sha256": fit.channels.sha256(),
            "prior_max_difference": prior_difference,
            "seconds": round(fitted, 1),
        },
        "dry_run": {
            "completed": True,
            "seconds": round(time.perf_counter() - checked, 1),
            "reading": "the whole experiment, scored on surrogate labels assigned "
            "by the hour; its record was discarded unread",
        },
        "peak_memory_mb": _peak_memory_mb(),
        "scored": "nothing: no prediction was compared with a true label",
    }


# ----------------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------------
def baseline_day_scores(
    training: Mapping[str, CasasRecording],
    periods: Mapping[str, CasasRecording],
    days: Mapping[str, Sequence[str]],
    protocol: TransferProtocol,
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
) -> dict[str, dict[str, PredictionMetrics]]:
    """Each baseline's metrics on each day, from the matched runner."""
    step = protocol.resolution.step
    recordings: dict[str, CasasRecording] = dict(training)
    moments: dict[str, Sequence[datetime]] = {
        home: _regular_moments(training[home], step) for home in protocol.training
    }
    for period, members in days.items():
        grid = _regular_moments(periods[period], step)
        for day in members:
            recordings[day] = periods[period]
            moments[day] = [m for m in grid if day_key(period, m) == day]
    base = protocol.base
    run = run_matched_evaluation(
        recordings,
        split=HouseholdSplit(
            SPLIT,
            train=protocol.training,
            test=tuple(d for members in days.values() for d in members),
        ),
        information_set=protocol.baseline_set(),
        models=protocol.baselines(),
        seed=base.seed,
        data_source=data_source,
        metrics=base.metrics,
        moments=moments,
        resamples=base.resamples,
        confidence=base.confidence,
        inputs=inputs,
        experiment=f"{protocol.name}-{BASELINE_SET}",
    )
    return {
        spec.name: held_out_metrics(run, spec.name) for spec in protocol.baselines()
    }


def _level_entry(
    estimand: Estimand,
    scores: Mapping[str, Mapping[str, PredictionMetrics]],
    protocol: TransferProtocol,
) -> dict[str, Any]:
    entry = _gap_entry(estimand, scores, protocol)  # type: ignore[arg-type]
    entry["rule"] = "level"
    entry["conclusion"] = LEVEL_READING[entry["verdicts"]["balanced_accuracy"]]
    return entry


def chance_of(metrics: PredictionMetrics) -> float:
    """One over the number of states labelled in a unit's scored windows."""
    return 1.0 / len(truth_counts(metrics))


def _chance_entry(
    scores: Mapping[str, Mapping[str, PredictionMetrics]], protocol: TransferProtocol
) -> dict[str, Any]:
    combination = {
        day: metrics
        for day, metrics in scores[CHANCE.model].items()
        if len(truth_counts(metrics)) >= MIN_CHANCE_STATES
    }
    left_out = len(scores[CHANCE.model]) - len(combination)
    if not combination:
        return {
            **CHANCE.to_dict(),
            "rule": "chance",
            "comparisons": {},
            "verdicts": {},
            "conclusion": "no day with two labelled states",
            "days_left_out": left_out,
        }
    comparison = _paired(
        household_values(combination, "balanced_accuracy"),
        {day: chance_of(metrics) for day, metrics in combination.items()},
        "balanced_accuracy",
        protocol.base.as_gap_protocol(),
    )
    found = verdict(comparison, CHANCE_DIFFERENCE)
    return {
        **CHANCE.to_dict(),
        "rule": "chance",
        "comparisons": {"balanced_accuracy": comparison},
        "verdicts": {"balanced_accuracy": found},
        "conclusion": CHANCE_READING[found],
        "days_left_out": left_out,
    }


def by_period(
    scores: Mapping[str, Mapping[str, PredictionMetrics]], days: Sequence[str]
) -> dict[str, Any]:
    """Each period's mean daily balanced accuracy per recursion, and of K1."""
    out: dict[str, Any] = {}
    for period in sorted({day.split("/")[0] for day in days}):
        members = [day for day in days if day.split("/")[0] == period]

        def mean(model: str, members: list[str] = members) -> float | None:
            values = [
                v
                for v in (scores[model][d].balanced_accuracy for d in members)
                if v is not None and np.isfinite(v)
            ]
            return float(np.mean(values)) if values else None

        levels = {model: mean(model) for model in MODELS}
        first, second = levels[FILTER_PERIODIC_HURDLE], levels["filter_hurdle"]
        out[period] = {
            "days": len(members),
            "balanced_accuracy": levels,
            "k1": None if first is None or second is None else first - second,
        }
    return out


@dataclass(frozen=True)
class TransferResult:
    """A completed transfer test: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def run_transfer(
    training: Mapping[str, CasasRecording],
    periods: Mapping[str, CasasRecording],
    validation: Mapping[str, Any],
    protocol: TransferProtocol,
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> TransferResult:
    """Fit on the development homes, score ContextAct once by day, and record it.

    *periods* holds the eligible periods in the canonical form and *validation*
    every period's validation, as :func:`read_periods` returns them.
    """
    missing = sorted(set(protocol.training) - set(training))
    if missing:
        raise ValueError(f"no recording for training homes {missing}")
    if set(periods) - set(protocol.periods):
        raise ValueError("a period the protocol does not name was given")
    ontology = StateOntology()
    states = tuple(ontology.states)
    gap = protocol.base.as_gap_protocol()
    fit, prior_difference = reproduced_fit(training, protocol, Phase3Formulation.load())

    scores: dict[str, dict[str, PredictionMetrics]] = {model: {} for model in MODELS}
    scored_days: dict[str, list[str]] = {}
    unscored_days: dict[str, int] = {}
    pipeline: dict[str, float] = {}
    evidence: list[datetime] = []
    for period in protocol.periods:
        recording = periods.get(period)
        if recording is None:
            continue
        posteriors, rows, truth, moments = recursion_posteriors(
            recording, fit, period=period, resolution=protocol.resolution
        )
        check_clock(recording, period, moments)
        difference = pipeline_difference(
            recording, posteriors[FILTER_PERIODIC_HURDLE], moments
        )
        if difference > PIPELINE_TOLERANCE:
            raise ValueError(
                f"in {period} the online pipeline is {difference:.3g} from the "
                "recursion; nothing is reported"
            )
        pipeline[period] = difference
        kept, below = split_days(labelled_days(period, rows, moments))
        unscored_days.update(below)
        if not kept:
            continue
        scored_days[period] = kept
        for model, per_day in day_scores(
            period, posteriors, rows, truth, moments, kept
        ).items():
            scores[model].update(per_day)
        evidence += [
            moments[int(r)] for r in rows if day_key(period, moments[int(r)]) in kept
        ]
    days = [d for members in scored_days.values() for d in members]
    if not days:
        raise ValueError("no day has enough labelled windows; nothing is reported")

    baselines = baseline_day_scores(
        training,
        periods,
        scored_days,
        protocol,
        data_source=data_source,
        inputs=inputs,
    )
    absent = sorted(
        {d for per_day in baselines.values() for d in set(days) - set(per_day)}
    )
    if absent:
        raise ValueError(f"the baselines did not score {absent}; nothing is reported")
    scores.update(baselines)
    differing = sorted(
        day
        for day in days
        if len(
            {
                json.dumps(truth_counts(per_day[day]), sort_keys=True)
                for per_day in scores.values()
            }
        )
        != 1
    )
    if differing:
        raise ValueError(
            f"the models scored different windows on {differing}; nothing is reported"
        )

    # Recall is compared only for states labelled on some scored day: a state
    # labelled on none has no recall to compare.
    present = tuple(
        state
        for state in states
        if any(
            state.value in truth_counts(scores[FILTER_PERIODIC_HURDLE][day])
            for day in days
        )
    )
    estimands = [
        _estimand_entry(e, scores, protocol.scoring, present)
        for e in DEVELOPMENT_ESTIMANDS
    ]
    for entry in estimands:
        if entry["key"] != "K1":
            entry.pop("per_state_recall", None)
            continue
        recall = entry.get("per_state_recall", {})
        for found in recall.values():
            if found["comparison"]["n"] < MIN_RECALL_DAYS:
                found["verdict"] = "too few days"
        entry["per_state_recall"] = {
            state.value: recall.get(
                state.value, {"comparison": None, "verdict": "never labelled"}
            )
            for state in states
        }
        if len(days) < MIN_SCORED_DAYS:
            entry["conclusion"] = "inconclusive"
            entry["note"] = f"fewer than {MIN_SCORED_DAYS} scored days"
    estimands += [
        _gap_entry(e, scores, protocol) for e in GAP_ESTIMANDS  # type: ignore[arg-type]
    ]
    estimands += [_level_entry(e, scores, protocol) for e in LEVEL_ESTIMANDS]
    estimands.append(_chance_entry(scores, protocol))
    cells = [
        _cell(model, RECURSION, scores[model], present, gap) for model in MODELS
    ] + [
        _cell(spec.name, BASELINE_SET, scores[spec.name], present, gap)
        for spec in protocol.baselines()
    ]
    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified, one-shot; the one resident of ContextAct@A4H, "
        "collected independently of CASAS",
        "check": {
            "windows": "identical",
            "days": len(days),
            "periods": {
                period: len(members) for period, members in scored_days.items()
            },
            "unscored_days": unscored_days,
            "excluded_periods": [p for p in protocol.periods if p not in periods],
            "pipeline_max_difference": pipeline,
        },
        "states": [state.value for state in states],
        "training": {
            "homes": list(protocol.training),
            "channels_sha256": fit.channels.sha256(),
            "periodic_prior_sha256": fit.periodic.sha256(),
            "prior_max_difference": prior_difference,
        },
        "validation": dict(validation),
        "cells": cells,
        "estimands": estimands,
        "interaction": interaction(scores, protocol.scoring),
        "by_period": by_period(scores, days),
        "conclusion": next(e["conclusion"] for e in estimands if e["key"] == "K1"),
    }
    base = protocol.base
    record = ExperimentRecord(
        experiment=protocol.name,
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        inference=ONLINE,
        evidence=EvidenceSummary.online(sorted(evidence)),
        seeds=[base.seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions=MATCHED_METRIC_DEFINITIONS,
        notes=[
            "Every setting, estimand and criterion was declared in the protocol "
            "before ContextAct was scored.",
            f"Everything was fitted once, on the {len(protocol.training)} CASAS "
            "development homes only; nothing was fitted on ContextAct.",
            "The unit is a day of one resident. Verdicts compare effect sizes and "
            "day bootstrap intervals with declared minimal differences; they are "
            "not significance tests, and they describe this resident only.",
        ],
        inputs=list(inputs),
        split={
            "fold": HouseholdSplit(
                SPLIT, train=protocol.training, test=tuple(days)
            ).to_dict()
        },
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
                d: m.to_dict() for d, m in per_day.items()
            }
            for model, per_day in scores.items()
        },
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return TransferResult(record=record, path=path)


__all__ = [
    "CHANCE",
    "LEVEL_ESTIMANDS",
    "PINNED_RECORDS",
    "TransferProtocol",
    "TransferResult",
    "baseline_day_scores",
    "by_period",
    "check_frozen_protocol",
    "check_pinned_records",
    "check_clock",
    "code_changes",
    "code_digest",
    "code_now",
    "day_key",
    "day_scores",
    "declared_protocol",
    "inputs_report",
    "labelled_days",
    "load_inputs",
    "pipeline_difference",
    "read_periods",
    "recursion_posteriors",
    "reproduced_fit",
    "run_transfer",
    "split_days",
    "write_protocol",
]
