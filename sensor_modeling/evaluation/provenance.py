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

Artefacts are written to a results directory that is deliberately excluded
from version control. Results are regenerated from their seed rather than
committed, so the repository does not accumulate large generated files.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Default location for generated artefacts. Excluded from version control.
RESULTS_DIR = Path("results")

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
    "precision": "True positives divided by predicted positives.",
    "recall": "True positives divided by actual positives.",
    "f1": "Harmonic mean of precision and recall.",
    "median_delay_days": (
        "Median days between a true change occurring and an alert being "
        "delivered for it. A detection counts only if it falls at or after "
        "the change and within max_delay_days."
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


def environment() -> dict[str, str]:
    """Capture the software environment a result was produced in."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("numpy", "scipy", "pandas", "sensor_modeling"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:  # pragma: no cover - all are hard dependencies
            versions[name] = "not installed"
    return versions


@dataclass
class ExperimentRecord:
    """A result together with everything needed to interpret and repeat it.

    Parameters
    ----------
    experiment
        Name of the experiment, matching the command that produces it.
    configuration
        Every parameter that affects the outcome. A reader must be able to
        reconstruct the run from this alone.
    seeds
        Random seeds used. Listed separately from the configuration because
        they are the first thing anyone reproducing a result reaches for.
    results
        The findings themselves.
    sensor_subset
        Sensors the experiment ran over, when it varies them.
    notes
        Anything a reader needs in order not to over-read the result.
    """

    experiment: str
    configuration: Mapping[str, Any]
    seeds: Sequence[int] = field(default_factory=list)
    results: Mapping[str, Any] = field(default_factory=dict)
    sensor_subset: Sequence[str] | None = None
    notes: Sequence[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate that the record is self-describing."""
        if not str(self.experiment).strip():
            raise ValueError("an experiment record needs a name")

    def to_dict(self) -> dict[str, Any]:
        """Return the full artefact, results and provenance together."""
        return {
            "experiment": self.experiment,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "environment": environment(),
            "configuration": dict(self.configuration),
            "seeds": list(self.seeds),
            "sensor_subset": (
                list(self.sensor_subset) if self.sensor_subset is not None else None
            ),
            "metric_definitions": METRIC_DEFINITIONS,
            "results": dict(self.results),
            "notes": [
                *self.notes,
                "Generated from the bundled simulator. Not validated against "
                "real sensor data; see docs/limitations.rst.",
            ],
        }

    def write(self, path: Path | None = None) -> Path:
        """Write the artefact as JSON and return where it went.

        A ``recorded_at`` stamp is added at write time, so two writes of the
        same record are not byte-identical. The *results* they contain are,
        which is the property reproducibility actually needs.
        """
        target = path if path is not None else RESULTS_DIR / f"{self.experiment}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        logger.info("Wrote experiment record to %s", target)
        return target


def load_record(path: Path) -> dict[str, Any]:
    """Read an artefact back, checking it carries its provenance."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [
        key
        for key in ("experiment", "environment", "configuration", "metric_definitions")
        if key not in payload
    ]
    if missing:
        raise ValueError(
            f"artefact at {path} is missing provenance fields: {sorted(missing)}"
        )
    result: dict[str, Any] = payload
    return result
