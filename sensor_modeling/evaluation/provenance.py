"""Making an experimental result interpretable without the code that made it.

A JSON file containing ``{"balanced_accuracy": 0.814}`` is nearly worthless six
months later. Balanced accuracy of what, over which sensors, with which seeds,
under which version, and computed how? Every one of those has to be recoverable
from the artefact itself, because the code will have moved on and the person
reading it may not be the person who ran it.

An :class:`ExperimentRecord` therefore carries the results *and* everything
needed to interpret and reproduce them: the configuration, the seeds, the
software and library versions, the sensor subset, and a written definition of
every metric reported.

The layout is versioned (:data:`SCHEMA_VERSION`) and checked: a record is
validated before it is written and when it is read, older versions are migrated
on load, and every file is strict, deterministic JSON. The schema is documented
in ``docs/EXPERIMENT_ARTIFACTS.md``.

Artefacts are written to a results directory that is deliberately excluded
from version control. Results are regenerated from their seed rather than
committed, so the repository does not accumulate large generated files.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Default location for generated artefacts. Excluded from version control.
RESULTS_DIR = Path("results")

#: Version of the artefact layout, ``MAJOR.MINOR``.
#:
#: A minor version only adds optional fields. A reader loading an older minor
#: fills those fields with their defaults. A major version would change what an
#: existing field means and would need an explicit migration. A reader refuses
#: any version newer than its own rather than guessing at fields it does not
#: know. See ``docs/EXPERIMENT_ARTIFACTS.md``.
SCHEMA_VERSION = "1.1"

#: Versions :func:`load_record` accepts, oldest first.
READABLE_VERSIONS = ("1.0", "1.1")

#: Note carried by every record whose observations came from the simulator.
SIMULATOR_NOTE = (
    "Generated from the bundled simulator. Not validated against real sensor "
    "data; see docs/limitations.md."
)

#: Repository root, used to ask git which commit the code came from.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Written definitions of every metric this package reports.
#:
#: These travel with the artefact so a stored number can be interpreted
#: without the source. Where a metric has a convention that could reasonably
#: go the other way -- notably how abstentions are scored -- the convention is
#: stated rather than left implicit.
METRIC_DEFINITIONS: dict[str, str] = {
    "accuracy": (
        "Fraction of scored estimates whose reported state equals the true "
        "state. A reported UNKNOWN is never correct, so abstentions count as "
        "errors."
    ),
    "selective_accuracy": (
        "Accuracy among only the estimates that committed to a state. Read "
        "together with abstention_rate."
    ),
    "abstention_rate": (
        "Fraction of estimates that declined to name a state, because "
        "confidence or sensor coverage fell below threshold."
    ),
    "balanced_accuracy": (
        "Unweighted mean of per-class recall. Used in preference to accuracy "
        "because the states are heavily imbalanced and a model that always "
        "predicts the majority state scores highly on raw accuracy."
    ),
    "macro_f1": "Unweighted mean F1 across the classes present in the truth.",
    "log_loss": (
        "Mean negative log probability assigned to the true state. Punishes "
        "confident errors far more than hedged ones."
    ),
    "brier": (
        "Multiclass Brier score over the full posterior, in [0, 2]. A proper "
        "scoring rule; lower is better."
    ),
    "calibration_error": (
        "Expected calibration error: the bin-weighted gap between stated "
        "confidence and observed accuracy. Answers whether a confidence of "
        "0.9 means anything."
    ),
    "per_class_recall": "Recall for each state present in the ground truth.",
    "confusion": (
        "Counts of labelled true state (rows) against reported state "
        "(columns). The final column is UNKNOWN, so abstentions stay visible."
    ),
    "precision": "True positives divided by predicted positives.",
    "recall": "True positives divided by actual positives.",
    "f1": "Harmonic mean of precision and recall.",
    "median_delay_days": (
        "Median days between a true change occurring and an alert being "
        "delivered for it. A detection counts only if it falls at or after "
        "the change and within max_delay_days. Where an arm aggregates several "
        "seeds, the individual delays are pooled before taking the median, so "
        "this is a median over detections and not a mean of per-seed medians."
    ),
    "mean_seed_median_delay_days": (
        "Mean across seeds of each seed's own median delay. Reported alongside "
        "median_delay_days because it weights every seed equally regardless of "
        "how many changes it detected; it is not a median."
    ),
    "detected_changes": (
        "Number of true changes detected across the seeds in an arm, which is "
        "the sample size behind median_delay_days."
    ),
    "false_positives_per_person_day": (
        "Delivered alerts not matched to a true change, divided by monitored "
        "person-days. The alert burden a recipient experiences."
    ),
    "mean_difference": (
        "Mean paired difference between two configurations evaluated on "
        "identical simulated trajectories."
    ),
    "ci_low, ci_high": (
        "Bootstrap confidence interval on the mean paired difference. "
        "Reported instead of a p-value: with simulations, significance is a "
        "statement about how long the computer ran."
    ),
    "effect_size": (
        "Cohen's dz for the paired difference. Note that pairing removes "
        "between-household variance, so dz is larger than an unpaired field "
        "study of the same size would produce."
    ),
    "contaminated_fraction": (
        "Fraction of simulated time during which someone other than the "
        "monitored resident was present."
    ),
}


def git_state() -> dict[str, str]:
    """Return the commit the code came from, and whether it was modified.

    The package version is not enough to identify code. A version stays fixed
    across many commits during development, so two materially different
    implementations can produce records that claim the same provenance. The
    commit resolves that; ``git_dirty`` records whether the working tree had
    uncommitted changes, because a dirty run is not reproducible from the
    commit alone.

    Returns ``"unknown"`` when git is unavailable or the package was installed
    from a distribution rather than a checkout, which is not an error.
    """

    def _run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = _run("rev-parse", "HEAD")
    if commit is None:
        return {"git_commit": "unknown", "git_dirty": "unknown"}
    status = _run("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": "unknown" if status is None else str(bool(status)).lower(),
    }


def resolved_defaults() -> dict[str, Any]:
    """Snapshot the algorithm defaults an experiment actually ran under.

    Recording only the options a user typed is not enough to interpret a result
    later. Most of what determines the outcome lives in configuration defaults,
    so a record that omits them cannot be distinguished from one produced after
    those defaults changed. Capturing the resolved values freezes the model
    specification alongside the numbers.
    """
    from ..baseline.adaptive import BaselineConfig
    from ..context.occupancy import ContextConfig
    from ..health.monitor import HealthConfig
    from ..online.pipeline import PipelineConfig
    from ..simulation.household import HouseholdConfig

    snapshot: dict[str, Any] = {}
    for name, factory in (
        ("pipeline", PipelineConfig),
        ("baseline", BaselineConfig),
        ("context", ContextConfig),
        ("health", HealthConfig),
        ("household", HouseholdConfig),
    ):
        try:
            snapshot[name] = dataclasses.asdict(factory())
        except Exception:  # pragma: no cover - a config needing arguments
            logger.debug("could not snapshot %s defaults", name, exc_info=True)
            snapshot[name] = "unavailable"
    return snapshot


def environment() -> dict[str, str]:
    """Capture the software environment a result was produced in."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    # scikit-learn fits the baseline models, so its version changes their numbers.
    for name in ("numpy", "scipy", "pandas", "sklearn", "sensor_modeling"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:  # pragma: no cover - all are hard dependencies
            versions[name] = "not installed"
    versions.update(git_state())
    return versions


class ArtifactError(ValueError):
    """An experiment artefact does not satisfy its schema."""


@dataclass(frozen=True)
class InputArtifact:
    """A file the experiment read, identified by its content, not its path.

    Attributes
    ----------
    name
        File name or other stable identifier, such as ``"hh101.csv"``.
    sha256
        Lower-case hexadecimal SHA-256 of the file's bytes.
    role
        What the file was, such as ``"recording"`` or ``"frozen profile"``.
    source
        Where it can be obtained, such as ``"zenodo:15708568"``.
    """

    name: str
    sha256: str
    role: str
    source: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return a serialisable form."""
        return {
            "name": self.name,
            "sha256": self.sha256,
            "role": self.role,
            "source": self.source,
        }


@dataclass(frozen=True)
class ModelRecord:
    """A model's name, how it was built, and every setting that affects it."""

    name: str
    configuration: Mapping[str, Any]
    build: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form."""
        return {
            "name": self.name,
            "build": self.build,
            "configuration": dict(self.configuration),
        }


@dataclass(frozen=True)
class ReportedInterval:
    """One estimate with its uncertainty interval, ready to quote.

    Attributes
    ----------
    label
        What is estimated, such as ``"tree vs state_frequency: mean
        balanced_accuracy improvement"``.
    estimate
        The point estimate, or ``None`` where none exists.
    low, high
        Interval bounds.
    confidence
        Nominal coverage, in ``(0, 1)``.
    method
        How the interval was computed, such as ``"percentile"`` or ``"bca"``.
    unit
        What was resampled: ``"household"`` for real homes, ``"seed"`` for
        simulated trajectories. Never timestamps.
    n
        Number of units behind the interval.
    """

    label: str
    estimate: float | None
    low: float
    high: float
    confidence: float
    method: str
    unit: str
    n: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form."""
        return {
            "label": self.label,
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "method": self.method,
            "unit": self.unit,
            "n": self.n,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentRecord:
    """A result together with everything needed to interpret and repeat it.

    The execution time and software environment are captured when the record
    is created, not when it is written, so writing the same record twice gives
    byte-identical files.

    Parameters
    ----------
    experiment
        Identifier of the experiment, matching the command that produces it.
    configuration
        Every parameter that affects the outcome and has no dedicated field
        below. A reader must be able to reconstruct the run from the record.
    seeds
        Random seeds used. Listed separately because they are the first thing
        anyone reproducing a result reaches for.
    results
        Aggregate results. Their layout belongs to the experiment; a runner
        names it in ``configuration["result_schema"]``.
    sensor_subset
        Sensors the experiment ran over, when it varies them.
    notes
        Anything a reader needs in order not to over-read the result.
    data_source
        Dataset identifier, such as ``"casas-hh"`` or ``"simulator"``. Only
        simulator records carry :data:`SIMULATOR_NOTE`; saying so of a
        real-data result would be false provenance.
    metric_definitions
        Definitions of the metrics this record reports. Defaults to
        :data:`METRIC_DEFINITIONS`.
    inputs
        Files the experiment read, identified by digest.
    split
        Household split, when the experiment has one.
    information_set
        Information-set declaration, when the experiment has one.
    preprocessing
        How raw data became the rows that were scored.
    models
        Every model compared, with its settings.
    household_metrics
        Metrics per model, then per household.
    intervals
        Estimates with uncertainty intervals, ready to quote.
    mcse
        Monte Carlo standard errors of simulation-derived summaries, by name.
    recorded_at, environment, resolved_defaults
        Captured at creation. Pass them only when rebuilding a record that
        was already written, as :meth:`load` does.
    migrated_from
        The schema version the record was read from, when it was migrated.
    """

    experiment: str
    configuration: Mapping[str, Any]
    seeds: Sequence[int] = field(default_factory=list)
    results: Mapping[str, Any] = field(default_factory=dict)
    sensor_subset: Sequence[str] | None = None
    notes: Sequence[str] = field(default_factory=list)
    data_source: str = "simulator"
    metric_definitions: Mapping[str, str] | None = None
    inputs: Sequence[InputArtifact] = field(default_factory=list)
    split: Mapping[str, Any] | None = None
    information_set: Mapping[str, Any] | None = None
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    models: Sequence[ModelRecord] = field(default_factory=list)
    household_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(
        default_factory=dict
    )
    intervals: Sequence[ReportedInterval] = field(default_factory=list)
    mcse: Mapping[str, float] = field(default_factory=dict)
    recorded_at: str = field(default_factory=_now)
    environment: Mapping[str, str] = field(default_factory=environment)
    resolved_defaults: Mapping[str, Any] = field(default_factory=resolved_defaults)
    migrated_from: str | None = None

    def __post_init__(self) -> None:
        """Validate that the record is self-describing."""
        if not str(self.experiment).strip():
            raise ValueError("an experiment record needs a name")
        if not str(self.data_source).strip():
            raise ValueError("an experiment record needs a data source")
        notes = list(self.notes)
        if self.data_source == "simulator" and SIMULATOR_NOTE not in notes:
            notes.append(SIMULATOR_NOTE)
        self.notes = notes

    def to_dict(self) -> dict[str, Any]:
        """Return the artefact as strict JSON-ready data.

        Non-finite numbers become ``None``, because JSON has no NaN or
        infinity. Values JSON cannot hold are converted as described in
        :func:`json_safe`.
        """
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": self.experiment,
            "recorded_at": self.recorded_at,
            "environment": dict(self.environment),
            "data_source": self.data_source,
            "inputs": [item.to_dict() for item in self.inputs],
            "split": dict(self.split) if self.split is not None else None,
            "information_set": (
                dict(self.information_set) if self.information_set is not None else None
            ),
            "preprocessing": dict(self.preprocessing),
            "models": [model.to_dict() for model in self.models],
            "configuration": dict(self.configuration),
            "resolved_defaults": dict(self.resolved_defaults),
            "seeds": list(self.seeds),
            "sensor_subset": (
                list(self.sensor_subset) if self.sensor_subset is not None else None
            ),
            "metric_definitions": dict(
                self.metric_definitions
                if self.metric_definitions is not None
                else METRIC_DEFINITIONS
            ),
            "results": dict(self.results),
            "household_metrics": {
                model: {house: dict(values) for house, values in houses.items()}
                for model, houses in self.household_metrics.items()
            },
            "intervals": [interval.to_dict() for interval in self.intervals],
            "mcse": dict(self.mcse),
            "notes": list(self.notes),
        }
        if self.migrated_from is not None:
            payload["migrated_from"] = self.migrated_from
        safe: dict[str, Any] = json_safe(payload)
        return safe

    def to_json(self) -> str:
        """The canonical text: sorted keys, strict JSON, one trailing newline."""
        payload = self.to_dict()
        validate_record(payload)
        return (
            json.dumps(
                payload, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False
            )
            + "\n"
        )

    def write(self, path: Path | None = None) -> Path:
        """Validate the artefact, write it as JSON and return where it went.

        The file is UTF-8 with LF line endings on every platform, so a digest
        of it does not depend on the machine that wrote it.
        """
        target = path if path is not None else RESULTS_DIR / f"{self.experiment}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8", newline="\n")
        logger.info("Wrote experiment record to %s", target)
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentRecord:
        """Rebuild a record from a validated payload of the current version."""
        validate_record(payload)
        return cls(
            experiment=payload["experiment"],
            configuration=payload["configuration"],
            seeds=payload["seeds"],
            results=payload["results"],
            sensor_subset=payload["sensor_subset"],
            notes=payload["notes"],
            data_source=payload["data_source"],
            metric_definitions=payload["metric_definitions"],
            inputs=[InputArtifact(**item) for item in payload["inputs"]],
            split=payload["split"],
            information_set=payload["information_set"],
            preprocessing=payload["preprocessing"],
            models=[ModelRecord(**item) for item in payload["models"]],
            household_metrics=payload["household_metrics"],
            intervals=[ReportedInterval(**item) for item in payload["intervals"]],
            mcse=payload["mcse"],
            recorded_at=payload["recorded_at"],
            environment=payload["environment"],
            resolved_defaults=payload["resolved_defaults"],
            migrated_from=payload.get("migrated_from"),
        )

    @classmethod
    def load(cls, path: Path) -> ExperimentRecord:
        """Read, migrate if needed, validate and rebuild a written record."""
        return cls.from_dict(load_record(path))


def json_safe(value: Any, path: str = "$") -> Any:
    """Convert *value* to data that strict JSON can hold, deterministically.

    - A non-finite float becomes ``None``.
    - A NumPy scalar or array becomes the corresponding Python value.
    - An enum becomes its value.
    - A date or datetime becomes ISO 8601 text.
    - A time delta, time zone or path becomes text, as earlier records wrote
      them.
    - Tuples become lists, and mapping keys become text.

    Anything else raises :class:`TypeError` naming where it was found, rather
    than being written as an opaque string.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        return json_safe(value.value, path)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item(), path)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist(), path)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (timedelta, tzinfo)):
        return str(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(json_safe(key, path)): json_safe(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path}: {type(value).__name__} cannot be written to JSON")


#: Fields every record has, with the type each must have.
_REQUIRED: Mapping[str, type | tuple[type, ...]] = {
    "schema_version": str,
    "experiment": str,
    "recorded_at": str,
    "environment": dict,
    "data_source": str,
    "configuration": dict,
    "resolved_defaults": dict,
    "seeds": list,
    "sensor_subset": (list, type(None)),
    "metric_definitions": dict,
    "results": dict,
    "notes": list,
}

#: Fields added in 1.1, with the default a 1.0 record is given.
_ADDED_IN_1_1: Mapping[str, Any] = {
    "inputs": [],
    "split": None,
    "information_set": None,
    "preprocessing": {},
    "models": [],
    "household_metrics": {},
    "intervals": [],
    "mcse": {},
}

_ENVIRONMENT_KEYS = ("git_commit", "git_dirty", "python", "sensor_modeling")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _non_finite(value: Any, path: str) -> list[str]:
    """Every place a non-finite number or a non-JSON value appears."""
    if isinstance(value, float):
        return [] if math.isfinite(value) else [f"{path} is not a finite number"]
    if value is None or isinstance(value, (bool, int, str)):
        return []
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _non_finite(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in _non_finite(v, f"{path}[{i}]")]
    return [f"{path} holds a {type(value).__name__}, which is not JSON"]


def _check_sections(payload: Mapping[str, Any]) -> list[str]:
    """Type and content checks for every field."""
    problems: list[str] = []
    if not payload["experiment"].strip():
        problems.append("experiment is empty")
    if not payload["data_source"].strip():
        problems.append("data_source is empty")
    try:
        if datetime.fromisoformat(payload["recorded_at"]).tzinfo is None:
            problems.append("recorded_at has no time zone")
    except ValueError:
        problems.append("recorded_at is not an ISO 8601 timestamp")
    environment_ = payload["environment"]
    problems += [
        f"environment lacks {key}"
        for key in _ENVIRONMENT_KEYS
        if key not in environment_
    ]
    problems += [
        f"environment.{k} is not text"
        for k, v in environment_.items()
        if not isinstance(v, str)
    ]
    if not all(
        isinstance(s, int) and not isinstance(s, bool) for s in payload["seeds"]
    ):
        problems.append("seeds must be integers")
    if payload["sensor_subset"] is not None and not all(
        isinstance(s, str) for s in payload["sensor_subset"]
    ):
        problems.append("sensor_subset must be text")
    if not all(
        isinstance(v, str) and v for v in payload["metric_definitions"].values()
    ):
        problems.append("every metric definition must be non-empty text")
    if not all(isinstance(note, str) for note in payload["notes"]):
        problems.append("notes must be text")

    for index, item in enumerate(payload["inputs"]):
        where = f"inputs[{index}]"
        if not isinstance(item, dict) or set(item) != {
            "name",
            "sha256",
            "role",
            "source",
        }:
            problems.append(f"{where} must have exactly name, sha256, role and source")
            continue
        if not item["name"] or not item["role"]:
            problems.append(f"{where} needs a name and a role")
        if not isinstance(item["sha256"], str) or not _SHA256.match(item["sha256"]):
            problems.append(f"{where}.sha256 is not a lower-case SHA-256")

    for key in ("split", "information_set"):
        if payload[key] is not None and not isinstance(payload[key], dict):
            problems.append(f"{key} must be an object or null")
    if not isinstance(payload["preprocessing"], dict):
        problems.append("preprocessing must be an object")

    names: list[str] = []
    for index, item in enumerate(payload["models"]):
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "build", "configuration"}
            or not isinstance(item["configuration"], dict)
        ):
            problems.append(f"models[{index}] must have name, build and configuration")
            continue
        names.append(item["name"])
    if len(set(names)) != len(names):
        problems.append("model names must be unique")

    for model, houses in payload["household_metrics"].items():
        if not isinstance(houses, dict) or not all(
            isinstance(values, dict) for values in houses.values()
        ):
            problems.append(f"household_metrics.{model} must map households to metrics")

    fields_ = {"label", "estimate", "low", "high", "confidence", "method", "unit", "n"}
    for index, item in enumerate(payload["intervals"]):
        where = f"intervals[{index}]"
        if not isinstance(item, dict) or set(item) != fields_:
            problems.append(f"{where} must have exactly {sorted(fields_)}")
            continue
        if not (_is_number(item["low"]) and _is_number(item["high"])):
            problems.append(f"{where} bounds must be numbers")
        elif item["low"] > item["high"]:
            problems.append(f"{where} has low above high")
        if item["estimate"] is not None and not _is_number(item["estimate"]):
            problems.append(f"{where}.estimate must be a number or null")
        if not (_is_number(item["confidence"]) and 0 < item["confidence"] < 1):
            problems.append(f"{where}.confidence must lie in (0, 1)")
        if not item["label"] or not item["method"] or not item["unit"]:
            problems.append(f"{where} needs a label, a method and a unit")
        if item["n"] is not None and not (
            isinstance(item["n"], int) and not isinstance(item["n"], bool)
        ):
            problems.append(f"{where}.n must be an integer or null")

    if not isinstance(payload["mcse"], dict) or not all(
        _is_number(v) and v >= 0 for v in payload["mcse"].values()
    ):
        problems.append("mcse must map names to non-negative numbers")
    return problems


def validate_record(payload: Mapping[str, Any]) -> None:
    """Check a current-version payload against the schema.

    Raises
    ------
    ArtifactError
        Listing every problem found, not only the first.
    """
    if not isinstance(payload, Mapping):
        raise ArtifactError("an experiment artefact must be a JSON object")
    # Structural problems stop the field-level checks, which rely on every
    # field being present with the right type. Everything else is collected.
    structural: list[str] = []
    missing = sorted((set(_REQUIRED) | set(_ADDED_IN_1_1)) - set(payload))
    structural += [f"missing required field {key!r}" for key in missing]
    expected_types: dict[str, type | tuple[type, ...]] = {
        **_REQUIRED,
        "inputs": list,
        "models": list,
        "intervals": list,
        "household_metrics": dict,
        "mcse": dict,
        "preprocessing": dict,
    }
    structural += [
        f"{key} has the wrong type"
        for key, expected in expected_types.items()
        if key in payload and not isinstance(payload[key], expected)
    ]

    problems = list(structural)
    if payload.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version is {payload.get('schema_version')!r}, expected "
            f"{SCHEMA_VERSION!r}; load older files with load_record"
        )
    known = set(_REQUIRED) | set(_ADDED_IN_1_1) | {"migrated_from"}
    problems += [f"unknown field {key!r}" for key in sorted(set(payload) - known)]
    problems += _non_finite(dict(payload), "$")
    if not structural:
        problems += _check_sections(payload)
    if problems:
        raise ArtifactError("invalid experiment artefact: " + "; ".join(problems))


def _migrate_1_0(payload: dict[str, Any]) -> dict[str, Any]:
    """Bring a 1.0 record to the current version without changing its content.

    1.0 had no ``data_source`` before 0.6.0. A record without one is marked
    ``"simulator"`` when it carries the simulator note, which every 1.0
    simulator record did, and ``"unknown"`` otherwise. The 1.0 writer allowed
    NaN and infinity, so non-finite numbers become ``None``.
    """
    migrated = json_safe(payload)
    if "data_source" not in migrated:
        notes = migrated.get("notes") or []
        simulated = any("bundled simulator" in note for note in notes)
        migrated["data_source"] = "simulator" if simulated else "unknown"
    for key, default in _ADDED_IN_1_1.items():
        migrated.setdefault(key, json.loads(json.dumps(default)))
    migrated["schema_version"] = SCHEMA_VERSION
    migrated["migrated_from"] = "1.0"
    result: dict[str, Any] = migrated
    return result


def _parse_version(value: Any) -> tuple[int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+", value):
        raise ArtifactError(f"schema_version {value!r} is not MAJOR.MINOR")
    major, minor = value.split(".")
    return int(major), int(minor)


def load_record(path: Path) -> dict[str, Any]:
    """Read an artefact, migrating an older schema version, and validate it.

    Returns the payload at the current schema version.

    Raises
    ------
    ArtifactError
        If the file is not a valid artefact, uses a schema version newer than
        this reader or of another major version, or, for a current-version
        file, contains NaN or infinity, which strict JSON cannot hold.
    """
    constants: list[str] = []

    def remember(token: str) -> float:
        constants.append(token)
        return float(token.replace("Infinity", "inf"))

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"), parse_constant=remember
        )
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"artefact at {path} is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"artefact at {path} is not a JSON object")
    if "schema_version" not in payload:
        raise ArtifactError(f"artefact at {path} has no schema_version")

    version = payload["schema_version"]
    major, minor = _parse_version(version)
    current = _parse_version(SCHEMA_VERSION)
    if major != current[0]:
        raise ArtifactError(
            f"artefact at {path} uses schema {version}; this reader reads major "
            f"version {current[0]} only"
        )
    if minor > current[1]:
        raise ArtifactError(
            f"artefact at {path} uses schema {version}, newer than {SCHEMA_VERSION}; "
            "upgrade sensor_modeling to read it"
        )
    if version == "1.0":
        payload = _migrate_1_0(payload)
    elif constants:
        raise ArtifactError(
            f"artefact at {path} contains {sorted(set(constants))}, which strict "
            "JSON cannot hold"
        )
    validate_record(payload)
    loaded: dict[str, Any] = payload
    return loaded
