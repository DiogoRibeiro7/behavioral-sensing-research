"""The TIHM alert-burden protocol: the online pipeline on a clinical cohort.

The TIHM dataset (:mod:`sensor_modeling.external.tihm`) holds 56 homes of
people living with dementia, with alerts a clinical monitoring team verified.
It cannot score state inference: its annotations are alerts, and the mapping
declares every one of them unmappable. What it can describe is the other end of
the pipeline, which nothing else in this repository has measured on real homes:
how many alerts the pipeline raises per person-day at its declared defaults,
how much of the record it will not use, and whether what it flags has any
relation to what a clinical team confirmed.

**This protocol is exploratory.** It was written after the dataset's labels had
been analysed, and :attr:`TihmProtocol.inspected_before` says exactly what had
been seen. It fixes every definition before any pipeline output is compared
with a label, so that the comparison cannot be shaped by its own result. It
does not make the result a held-out or confirmatory claim.

Nothing is fitted on TIHM. The pipeline runs with the declared emissions, the
default baseline and the default alert policy.

:func:`declared_protocol` is frozen in
``artifacts/tihm/alert_burden_protocol.json``. The scoring run refuses any
other protocol.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..alerts.alert import AlertPolicy
from ..baseline.adaptive import BaselineConfig
from ..external.mapping import OntologyMapping
from ..external.tihm import (
    ARCHIVE_SHA256,
    ARCHIVE_URL,
    FILES,
    LABELS,
    LOCATIONS,
    PROVENANCE,
    READ_FILES,
    TIHM_MAPPING,
    TIMEZONE,
)
from ..online.pipeline import PipelineConfig

#: The record's name and the layout of its results.
EXPERIMENT = "tihm-alert-burden"
RESULT_SCHEMA = "tihm-alert-burden/1"

#: The label type the association and reference analyses are about.
PRIMARY_LABEL = "Agitation"

#: The baseline feature the simulator's detection study tracks.
SIMULATOR_FEATURE = "sleeping_hours"

#: The simulator's behavioural alert burden for that feature, per person-day,
#: as ``docs/ADVERSARIAL_REVIEW.md`` (item i4) reports it for 70-day runs.
SIMULATOR_REFERENCE = {
    "stable_arm": 0.010,
    "changed_arm": 0.033,
    "feature": SIMULATOR_FEATURE,
    "source": "docs/ADVERSARIAL_REVIEW.md, item i4; docs/RESEARCH_QUESTIONS.md, RQ4",
}

#: Hours an agitation label is stamped at, and how far past the hour counts.
LABEL_SLOTS = (8, 12, 18)
SLOT_TOLERANCE_SECONDS = 180

#: Six-hour blocks of the day, as ``(name, first hour, hour after the last)``.
BLOCKS = (("00-06", 0, 6), ("06-12", 6, 12), ("12-18", 12, 18), ("18-24", 18, 24))

#: The alert limits the dataset paper states for the daily measurements, as
#: ``(device_type, below, above)``. A day meets a label's rule when any of its
#: readings of a listed device is below or above its limit.
STATED_LIMITS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "Blood pressure": (
        ("Systolic blood pressure", 80.0, 190.0),
        ("Diastolic blood pressure", 50.0, 110.0),
    ),
    "Pulse": (("Heart rate", 55.0, 100.0),),
    "Body temperature": (("Body Temperature", 35.0, 37.6),),
    "Body water": (("Total body water", 40.0, 70.0),),
}

#: The notebook published with the dataset, and the logistic-regression
#: confusion matrices it prints for its five test weeks, newest week first, as
#: ``[[true negatives, false positives], [false negatives, true positives]]``.
PUBLISHED_NOTEBOOK = "example_code/example_baseline_classfication.ipynb"
PUBLISHED_NOTEBOOK_SHA256 = (
    "49d22859216685014bc0000877a56c4d010be75cad335f5a6b69f5582380d735"
)
PUBLISHED_LOGISTIC: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((271, 76), (0, 10)),
    ((268, 38), (1, 7)),
    ((248, 49), (4, 9)),
    ((233, 57), (6, 11)),
    ((215, 66), (5, 18)),
)
PUBLISHED_TEST_DAYS = 7
PUBLISHED_FOLDS = 5


def published_alerts() -> tuple[int, ...]:
    """Alerts the published model raised in each test week, newest first."""
    return tuple(matrix[0][1] + matrix[1][1] for matrix in PUBLISHED_LOGISTIC)


def published_rows() -> tuple[int, ...]:
    """Person-days in each published test week, newest first."""
    return tuple(sum(sum(row) for row in matrix) for matrix in PUBLISHED_LOGISTIC)


def published_positives() -> tuple[int, ...]:
    """Labelled days in each published test week, newest first."""
    return tuple(sum(matrix[1]) for matrix in PUBLISHED_LOGISTIC)


@dataclass(frozen=True)
class TihmProtocol:
    """Every definition of the TIHM alert-burden evaluation.

    Attributes
    ----------
    step_minutes
        The pipeline's inference step, the one the CASAS evaluation used.
    resamples, confidence, seed
        The household bootstrap.
    history_strength
        Pseudo-days by which a household's label share is shrunk towards the
        overall share in the label-history reference.
    min_previous_days
        Previous monitored days a household needs before a day's event count
        gets a deviation in the event-count reference.
    mapping
        The declared correspondence to the ontology.
    """

    step_minutes: int = 10
    resamples: int = 5000
    confidence: float = 0.95
    seed: int = 0
    history_strength: float = 7.0
    min_previous_days: int = 5
    mapping: OntologyMapping = field(default=TIHM_MAPPING)

    #: What had been seen when the protocol was written.
    inspected_before: tuple[str, ...] = (
        "Labels.csv, Activity.csv and Physiology.csv had been analysed outside "
        "this repository: label counts by type and participant, label "
        "timestamps, hourly activity around agitation labels, the stated "
        "limits against the readings, and a rebuild of the published baseline. "
        "The label and reference definitions below were written with that "
        "knowledge",
        "one household, c55f8, had been run through the pipeline to measure run "
        "time; its step count and its days' usability were read, and none of "
        "its verdicts or alerts",
        "no pipeline alert, verdict or deviation had been compared with any " "label",
    )

    def to_dict(self) -> dict[str, Any]:
        """Return the full declaration, in a stable serialisable form."""
        baseline = BaselineConfig()
        policy = AlertPolicy()
        pipeline = PipelineConfig()
        return {
            "schema": "tihm-alert-burden-protocol/1",
            "name": EXPERIMENT,
            "status": "exploratory: descriptive of these homes; no threshold, "
            "mapping or model setting is chosen from them, and nothing here is a "
            "held-out or confirmatory claim",
            "question": "at its declared defaults, how many alerts does the online "
            "pipeline raise per person-day on a clinical cohort, how much of the "
            "record will it not use, and does what it flags relate to the alerts "
            "a clinical monitoring team verified",
            "inspected_before": list(self.inspected_before),
            "dataset": {
                "name": PROVENANCE.name,
                "version": PROVENANCE.version,
                "source": PROVENANCE.source,
                "licence": PROVENANCE.licence,
                "citation": PROVENANCE.citation,
                "archive": ARCHIVE_URL,
                "archive_sha256": ARCHIVE_SHA256,
                "files": dict(FILES),
                "read_through_the_contract": list(READ_FILES),
                "read_outside_the_contract": {
                    "Physiology.csv": "only to describe the labels: which days "
                    "have a reading, and whether a day's readings meet the "
                    "limits the dataset paper states. No pipeline input",
                },
                "timezone": TIMEZONE,
            },
            "mapping": {
                "sha256": self.mapping.sha256(),
                "labels": {label: "unmappable" for label in LABELS},
                "sensors": dict(LOCATIONS),
                "scored_states": [],
                "consequence": "no annotated time maps to a behavioural state, so "
                "state inference is not scored on this dataset",
            },
            "households": "every participant in Demographics.csv; a household "
            "that fails the contract is reported and not run",
            "pipeline": {
                "what": "BehaviouralSensingPipeline with the registry the "
                "canonical conversion builds, default emissions derived from "
                "that registry, and default configuration throughout",
                "step_minutes": self.step_minutes,
                "features": [f"{state.value}_hours" for state in pipeline.features],
                "min_day_coverage": pipeline.min_day_coverage,
                "min_day_observed": pipeline.min_day_observed,
                "attribute_activity": pipeline.attribute_activity,
                "baseline": {
                    "min_samples": baseline.min_samples,
                    "deviation_threshold": baseline.deviation_threshold,
                    "persistence_days": baseline.persistence_days,
                    "trend_window": baseline.trend_window,
                    "trend_threshold": baseline.trend_threshold,
                },
                "alert_policy": {
                    "min_score": policy.min_score,
                    "min_confidence": policy.min_confidence,
                    "cooldown_hours": policy.cooldown.total_seconds() / 3600.0,
                    "max_per_window": policy.max_per_window,
                    "health_coverage_floor": policy.health_coverage_floor,
                },
                "fitted_on_tihm": "nothing",
            },
            "definitions": {
                "monitored_day": "a local calendar day the pipeline closed with a "
                "daily summary",
                "usable_day": "a monitored day whose summary passes the pipeline's "
                "coverage and observed-fraction tests",
                "evaluable_day": "a monitored day on which at least one baseline "
                "feature received a verdict other than insufficient_data",
                "behavioural_alert": "an alert of kind behavioural_change, "
                "attributed to the day its verdict is about",
                "alert_day": "a monitored day with at least one behavioural alert",
                "deviating_day": "an evaluable day on which at least one feature's "
                "robust deviation reached the baseline's deviation threshold",
                "deviation_score": "the largest absolute robust deviation among "
                "the day's features with a verdict; zero where there is none",
                "label_day": f"a local calendar day with at least one "
                f"{PRIMARY_LABEL} label; a label's day is the date of its "
                "timestamp",
                "any_label_day": "the same for a label of any type",
            },
            "estimands": {
                "B1": "behavioural alerts per monitored person-day, all features",
                "B2": f"the same for {SIMULATOR_FEATURE} alone, beside the "
                "simulator's figures, which are a reference and not a test",
                "B3": "behavioural alerts per evaluable person-day",
                "H1": "system-health and data-quality alerts per monitored "
                "person-day, and the share of monitored days that are usable "
                "and evaluable",
                "A1": "on evaluable days: the share of label days that are alert "
                "days, minus the share of other days that are",
                "A2": "the same for deviating days",
                "A3": "the probability that a label day has a higher deviation "
                "score than another evaluable day of the same household, ties "
                "counted as half",
                "R1": "label days caught by each reference when it flags as many "
                "evaluable days as the pipeline has deviating days",
                "P1": "label days caught by each reference on the published "
                "protocol's five test weeks, with the published model's alert "
                "count in each week, beside the published count",
            },
            "references": {
                "label_history": "the share of the household's earlier monitored "
                f"days that were label days, shrunk by {self.history_strength:g} "
                "pseudo-days towards the cohort's overall share on evaluable "
                "days; it uses no sensor",
                "event_count": "the day's sensor event count as a z-score against "
                "the household's previous monitored days, zero until "
                f"{self.min_previous_days} exist",
                "random": "the expected catch of flagging the same number of days "
                "at random",
                "ties": "days tied at the cut share the remaining flags equally",
            },
            "published_baseline": {
                "notebook": PUBLISHED_NOTEBOOK,
                "notebook_sha256": PUBLISHED_NOTEBOOK_SHA256,
                "model": "logistic regression",
                "confusion_matrices_newest_first": [
                    [list(row) for row in matrix] for matrix in PUBLISHED_LOGISTIC
                ],
                "protocol": f"{PUBLISHED_FOLDS} test periods of "
                f"{PUBLISHED_TEST_DAYS} days stepping back from the last "
                f"{PRIMARY_LABEL} day; training on every earlier day; a person-day "
                "is a day with any activity row or non-zero physiology reading",
                "check": "the run refuses to report this section unless each test "
                "period reproduces the published number of person-days and "
                "label days",
                "label_history_here": "the household's share of label days in the "
                "training period, shrunk the same way towards the training "
                "period's overall share; a household with no training day gets "
                "that overall share",
                "not_done": "the published model is not retrained here; its counts "
                "are quoted from the notebook's printed output",
            },
            "label_description": {
                "slots": list(LABEL_SLOTS),
                "slot_tolerance_seconds": SLOT_TOLERANCE_SECONDS,
                "hourly_profile": "sensor events per hour as a z-score against the "
                "same household and hour over its full monitored days, averaged "
                "over label days grouped by the slot of the day's first label",
                "blocks": [name for name, _, _ in BLOCKS],
                "full_day": "a monitored day that is neither the household's first "
                "nor its last",
                "stated_limits": {
                    label: [list(limit) for limit in limits]
                    for label, limits in STATED_LIMITS.items()
                },
            },
            "bootstrap": {
                "unit": "households",
                "reason": "days within a home are dependent; see "
                "docs/EVALUATION_DESIGN.md",
                "resamples": self.resamples,
                "confidence": self.confidence,
                "interval": "percentile",
                "seed": self.seed,
                "statistic": "every estimand is a ratio, or a difference of "
                "ratios, of sums over households, recomputed on each resample",
            },
            "simulator_reference": dict(SIMULATOR_REFERENCE),
            "reporting": "every estimand and every household, whatever it shows; "
            "no threshold, no retuning and no added reference after scoring",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_protocol() -> TihmProtocol:
    """The protocol as declared."""
    return TihmProtocol()


def write_protocol(protocol: TihmProtocol, path: Path) -> str:
    """Write the declaration with its digest; return the digest."""
    payload = {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return protocol.sha256()


def check_frozen_protocol(protocol: TihmProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after scoring begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
