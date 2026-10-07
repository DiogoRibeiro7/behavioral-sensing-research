"""The TIHM alert-burden evaluation: what the online pipeline raises on a clinical cohort.

This runs the protocol of :mod:`.tihm_protocol`. Each household of the TIHM
dataset is read through the external-dataset contract, converted to the
canonical recording, and run through
:class:`~sensor_modeling.online.pipeline.BehaviouralSensingPipeline` at its
declared defaults. Nothing is fitted. The run then describes four things:

- **the contract**: what converts, and that no annotated time can be scored
  against a state;
- **the alert burden**: behavioural, system-health and data-quality alerts per
  person-day, and how much of the record the pipeline will not use;
- **the association**: whether alert days, deviating days and the deviation
  score relate to the days a clinical team verified an agitation alert,
  beside two references that use no model;
- **the labels**: when they are stamped, what the sensors show around them,
  and how they relate to the limits the dataset paper states.

Every interval resamples households. The result is exploratory; see the
protocol.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from ..alerts.alert import AlertKind
from ..baseline.adaptive import BaselineConfig, ChangeKind
from ..evaluation import ExperimentRecord, InputArtifact, ModelRecord, ReportedInterval
from ..evaluation.resampling import percentile_interval, resample_indices
from ..external.canonical import to_canonical
from ..external.contract import DatasetAdapter, HouseholdData
from ..external.validation import localise, validate_household
from ..fusion import ONLINE, EvidenceSummary
from ..online.pipeline import BehaviouralSensingPipeline, PipelineConfig
from .tihm_protocol import (
    BLOCKS,
    EXPERIMENT,
    LABEL_SLOTS,
    PRIMARY_LABEL,
    PUBLISHED_FOLDS,
    PUBLISHED_LOGISTIC,
    PUBLISHED_TEST_DAYS,
    RESULT_SCHEMA,
    SIMULATOR_FEATURE,
    SIMULATOR_REFERENCE,
    SLOT_TOLERANCE_SECONDS,
    STATED_LIMITS,
    TihmProtocol,
    published_alerts,
    published_positives,
    published_rows,
)

DAYS_PER_MONTH = 365.25 / 12.0

#: The group of a label day whose first label is on no declared slot.
OTHER_SLOT = "other"

#: The group of full monitored days without a label.
NO_LABEL = "none"


# ----------------------------------------------------------------------------
# One household through the pipeline
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DayRecord:
    """What the pipeline concluded about one monitored day, and what was labelled.

    Attributes
    ----------
    day
        The local calendar date.
    usable
        Whether the day's summary passed the pipeline's quality tests.
    coverage, observed, abstention
        The summary's data-quality fractions.
    kinds, deviations
        Each baseline feature's verdict and robust deviation.
    alerts
        The features a behavioural alert was raised about, with its severity.
    events
        Canonical observations on the day.
    labels
        The label types stamped on the day.
    """

    day: date
    usable: bool
    coverage: float
    observed: float
    abstention: float
    kinds: Mapping[str, str]
    deviations: Mapping[str, float]
    alerts: tuple[tuple[str, str], ...]
    events: int
    labels: frozenset[str]

    @property
    def evaluable(self) -> bool:
        """Whether any feature received a verdict other than insufficient data."""
        return any(
            kind != ChangeKind.INSUFFICIENT_DATA.value for kind in self.kinds.values()
        )

    @property
    def score(self) -> float:
        """The largest absolute deviation among the features with a verdict."""
        values = [
            abs(self.deviations[feature])
            for feature, kind in self.kinds.items()
            if kind != ChangeKind.INSUFFICIENT_DATA.value
        ]
        return max(values) if values else 0.0

    def deviating(self, threshold: float) -> bool:
        """Whether any feature's deviation reached the baseline's threshold."""
        return self.evaluable and self.score >= threshold


@dataclass(frozen=True)
class HouseholdRun:
    """One household's run: its days, its other alerts and what conversion left out.

    Attributes
    ----------
    household
        The dataset's identifier.
    days
        Every monitored day, in order.
    system_health_alerts, data_quality_alerts
        Alerts about the apparatus and about alert bursts, over the whole run.
    hourly
        Canonical observations by ``(day, hour)``.
    label_times
        Every label's local time, by type.
    dispositions, issues
        What the conversion did not carry, and the validation issue counts.
    closes
        The moments the pipeline closed a day at.
    """

    household: str
    days: tuple[DayRecord, ...]
    system_health_alerts: int
    data_quality_alerts: int
    hourly: Mapping[tuple[date, int], int]
    label_times: Mapping[str, tuple[datetime, ...]]
    dispositions: Mapping[str, Any]
    issues: Mapping[str, int]
    closes: tuple[datetime, ...]


def run_household(data: HouseholdData, protocol: TihmProtocol) -> HouseholdRun:
    """Run one household through the contract and the pipeline at its defaults."""
    report = validate_household(data, protocol.mapping)
    canonical = to_canonical(data, protocol.mapping, source="tihm")
    recording = canonical.recording
    if data.timezone is None:  # pragma: no cover - to_canonical already refused it
        raise ValueError(f"household {data.household!r} has no timezone")
    zone = ZoneInfo(data.timezone)

    events: Counter[date] = Counter()
    hourly: Counter[tuple[date, int]] = Counter()
    for observation in recording.observations:
        local = observation.timestamp.astimezone(zone)
        events[local.date()] += 1
        hourly[(local.date(), local.hour)] += 1

    labels: dict[date, set[str]] = defaultdict(set)
    label_times: dict[str, list[datetime]] = defaultdict(list)
    for annotation in data.annotations:
        moment = localise(annotation.start, zone)
        labels[moment.date()].add(annotation.label)
        label_times[annotation.label].append(moment)

    pipeline = BehaviouralSensingPipeline(
        recording.registry,
        config=PipelineConfig(tz=zone, step=timedelta(minutes=protocol.step_minutes)),
    )
    steps = pipeline.run(recording.observations)
    steps.extend(pipeline.close(recording.observations[-1].timestamp))

    days: list[DayRecord] = []
    closes: list[datetime] = []
    other: Counter[str] = Counter()
    for step in steps:
        for alert in step.alerts:
            if alert.kind is not AlertKind.BEHAVIOURAL_CHANGE:
                other[alert.kind.value] += 1
        summary = step.day_closed
        if summary is None:
            continue
        closes.append(step.at)
        days.append(
            DayRecord(
                day=summary.day,
                usable=summary.is_usable(
                    pipeline.config.min_day_coverage, pipeline.config.min_day_observed
                ),
                coverage=float(summary.coverage),
                observed=float(summary.observed),
                abstention=float(summary.abstention),
                kinds={c.feature: c.kind.value for c in step.changes},
                deviations={c.feature: float(c.deviation) for c in step.changes},
                alerts=tuple(
                    (alert.subject, alert.severity.value)
                    for alert in step.alerts
                    if alert.kind is AlertKind.BEHAVIOURAL_CHANGE
                ),
                events=int(events.get(summary.day, 0)),
                labels=frozenset(labels.get(summary.day, ())),
            )
        )
    return HouseholdRun(
        household=data.household,
        days=tuple(days),
        system_health_alerts=other[AlertKind.SYSTEM_HEALTH.value],
        data_quality_alerts=other[AlertKind.DATA_QUALITY.value],
        hourly=dict(hourly),
        label_times={k: tuple(sorted(v)) for k, v in label_times.items()},
        dispositions=canonical.dispositions,
        issues={issue.code: issue.count for issue in report.issues},
        closes=tuple(closes),
    )


# ----------------------------------------------------------------------------
# Household-level statistics
# ----------------------------------------------------------------------------
def ratio_estimate(
    numerator: Sequence[float], denominator: Sequence[float], protocol: TihmProtocol
) -> dict[str, Any]:
    """A ratio of sums over households, with its household-bootstrap interval.

    Each household contributes one numerator and one denominator. A resample
    whose denominator sums to zero has no ratio and is left out; how many
    remained is reported.
    """
    return difference_estimate(numerator, denominator, None, None, protocol)


def difference_estimate(
    numerator: Sequence[float],
    denominator: Sequence[float],
    other_numerator: Sequence[float] | None,
    other_denominator: Sequence[float] | None,
    protocol: TihmProtocol,
) -> dict[str, Any]:
    """A ratio of sums minus another, both resampled on the same households."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    paired = other_numerator is not None and other_denominator is not None
    num2 = np.asarray(other_numerator, dtype=float) if paired else np.zeros_like(num)
    den2 = np.asarray(other_denominator, dtype=float) if paired else np.ones_like(den)

    def value(n: float, d: float, n2: float, d2: float) -> float | None:
        if d <= 0 or d2 <= 0:
            return None
        return float(n / d - (n2 / d2 if paired else 0.0))

    estimate = value(num.sum(), den.sum(), num2.sum(), den2.sum())
    result: dict[str, Any] = {
        "estimate": estimate,
        "interval": None,
        "households": int(num.size),
        "resamples_defined": 0,
    }
    if num.size < 2 or estimate is None:
        return result
    index = resample_indices(num.size, protocol.resamples, protocol.seed)
    n, d = num[index].sum(axis=1), den[index].sum(axis=1)
    n2, d2 = num2[index].sum(axis=1), den2[index].sum(axis=1)
    defined = (d > 0) & (d2 > 0)
    result["resamples_defined"] = int(defined.sum())
    if not defined.any():
        return result
    replicates = n[defined] / d[defined]
    if paired:
        replicates = replicates - n2[defined] / d2[defined]
    result["interval"] = percentile_interval(replicates, protocol.confidence).to_dict()
    return result


def flag_weights(score: np.ndarray, flags: int) -> np.ndarray:
    """Flag the *flags* highest scores; ties at the cut share what remains equally."""
    weights = np.zeros(score.size, dtype=float)
    if flags <= 0 or score.size == 0:
        return weights
    if flags >= score.size:
        return np.ones(score.size, dtype=float)
    cut = np.sort(score)[::-1][flags - 1]
    above, tie = score > cut, score == cut
    weights[above] = 1.0
    weights[tie] = (flags - above.sum()) / tie.sum()
    return weights


def concordance(score: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    """Concordant pairs and all pairs of one labelled and one unlabelled day.

    A pair is concordant when the labelled day has the higher score; a tie
    counts as half.
    """
    positive, negative = score[label], score[~label]
    if positive.size == 0 or negative.size == 0:
        return 0.0, 0.0
    higher = (positive[:, None] > negative[None, :]).sum()
    equal = (positive[:, None] == negative[None, :]).sum()
    return float(higher + 0.5 * equal), float(positive.size * negative.size)


def expanding_z(values: Sequence[float], min_previous: int) -> np.ndarray:
    """Each value as a z-score against the values before it; zero until enough exist."""
    series = np.asarray(values, dtype=float)
    scores = np.zeros(series.size, dtype=float)
    for position in range(min_previous, series.size):
        previous = series[:position]
        spread = float(previous.std(ddof=1)) if previous.size > 1 else 0.0
        if spread > 0:
            scores[position] = (series[position] - previous.mean()) / spread
    return scores


def history_share(
    labelled: Sequence[bool], overall: float, strength: float
) -> np.ndarray:
    """The share of earlier days that were labelled, shrunk towards *overall*."""
    flags = np.asarray(labelled, dtype=float)
    before = np.concatenate([[0.0], np.cumsum(flags)[:-1]])
    days = np.arange(flags.size, dtype=float)
    return (before + strength * overall) / (days + strength)


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def _contract(runs: Sequence[HouseholdRun], households: int) -> dict[str, Any]:
    issues: Counter[str] = Counter()
    events: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    seconds = 0.0
    points = 0
    for run in runs:
        issues.update(run.issues)
        events.update(run.dispositions["events"])
        labels.update(run.dispositions["unscored_labels"])
        seconds += sum(
            value
            for outcome, value in run.dispositions["annotated_seconds"].items()
            if outcome in ("exact", "approximate")
        )
        points += int(run.dispositions["point_annotations"])
    return {
        "households": households,
        "converted": len(runs),
        "issues": dict(sorted(issues.items())),
        "events": dict(sorted(events.items())),
        "point_annotations": points,
        "scorable_annotated_seconds": seconds,
    }


def _per_household(
    runs: Sequence[HouseholdRun], threshold: float
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for run in runs:
        days = run.days
        table[run.household] = {
            "monitored_days": len(days),
            "usable_days": sum(d.usable for d in days),
            "evaluable_days": sum(d.evaluable for d in days),
            "behavioural_alerts": sum(len(d.alerts) for d in days),
            "simulator_feature_alerts": sum(
                1
                for d in days
                for subject, _ in d.alerts
                if subject == SIMULATOR_FEATURE
            ),
            "alert_days": sum(bool(d.alerts) for d in days),
            "deviating_days": sum(d.deviating(threshold) for d in days),
            "label_days": sum(PRIMARY_LABEL in d.labels for d in days),
            "any_label_days": sum(bool(d.labels) for d in days),
            "system_health_alerts": run.system_health_alerts,
            "data_quality_alerts": run.data_quality_alerts,
            "events": sum(d.events for d in days),
        }
    return table


def _column(table: Mapping[str, Mapping[str, Any]], name: str) -> list[float]:
    return [float(table[h][name]) for h in sorted(table)]


def _monitoring(
    runs: Sequence[HouseholdRun],
    table: Mapping[str, Mapping[str, Any]],
    protocol: TihmProtocol,
) -> dict[str, Any]:
    days = [d for run in runs for d in run.days]
    monitored = _column(table, "monitored_days")
    usable = [d for d in days if d.usable]
    kinds: Counter[str] = Counter(kind for d in days for kind in d.kinds.values())
    return {
        "monitored_days": len(days),
        "usable_days": len(usable),
        "evaluable_days": sum(d.evaluable for d in days),
        "households_with_an_evaluable_day": sum(
            1 for h in table if table[h]["evaluable_days"] > 0
        ),
        "usable_share": ratio_estimate(
            _column(table, "usable_days"), monitored, protocol
        ),
        "evaluable_share": ratio_estimate(
            _column(table, "evaluable_days"), monitored, protocol
        ),
        "mean_coverage": float(np.mean([d.coverage for d in days])) if days else None,
        "mean_observed": float(np.mean([d.observed for d in days])) if days else None,
        "mean_abstention": (
            float(np.mean([d.abstention for d in usable])) if usable else None
        ),
        "verdicts": dict(sorted(kinds.items())),
        "system_health_alerts": int(sum(_column(table, "system_health_alerts"))),
        "data_quality_alerts": int(sum(_column(table, "data_quality_alerts"))),
        "system_health_per_monitored_day": ratio_estimate(
            _column(table, "system_health_alerts"), monitored, protocol
        ),
        "data_quality_per_monitored_day": ratio_estimate(
            _column(table, "data_quality_alerts"), monitored, protocol
        ),
    }


def _burden(
    runs: Sequence[HouseholdRun],
    table: Mapping[str, Mapping[str, Any]],
    protocol: TihmProtocol,
) -> dict[str, Any]:
    monitored = _column(table, "monitored_days")
    evaluable = _column(table, "evaluable_days")
    alerts = _column(table, "behavioural_alerts")
    by_feature: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    for run in runs:
        for day in run.days:
            for subject, severity in day.alerts:
                by_feature[subject] += 1
                by_severity[severity] += 1
    per_household = sorted(int(a) for a in alerts)
    return {
        "behavioural_alerts": int(sum(alerts)),
        "alert_days": int(sum(_column(table, "alert_days"))),
        "deviating_days": int(sum(_column(table, "deviating_days"))),
        "by_feature": dict(sorted(by_feature.items())),
        "by_severity": dict(sorted(by_severity.items())),
        "households_with_an_alert": sum(1 for a in alerts if a > 0),
        "largest_household_count": per_household[-1] if per_household else 0,
        "B1_per_monitored_day": ratio_estimate(alerts, monitored, protocol),
        "B2_simulator_feature_per_monitored_day": ratio_estimate(
            _column(table, "simulator_feature_alerts"), monitored, protocol
        ),
        "B3_per_evaluable_day": ratio_estimate(alerts, evaluable, protocol),
        "alert_day_share_of_evaluable_days": ratio_estimate(
            _column(table, "alert_days"), evaluable, protocol
        ),
        "deviating_day_share_of_evaluable_days": ratio_estimate(
            _column(table, "deviating_days"), evaluable, protocol
        ),
        "simulator_reference": dict(SIMULATOR_REFERENCE),
    }


@dataclass(frozen=True)
class _Evaluable:
    """The evaluable days of every household, as aligned arrays."""

    household: np.ndarray
    label: np.ndarray
    any_label: np.ndarray
    alert: np.ndarray
    deviating: np.ndarray
    score: np.ndarray
    history: np.ndarray
    events: np.ndarray
    names: tuple[str, ...]


def _evaluable(
    runs: Sequence[HouseholdRun], protocol: TihmProtocol, threshold: float
) -> _Evaluable:
    overall_days = [d for run in runs for d in run.days if d.evaluable]
    overall = (
        sum(PRIMARY_LABEL in d.labels for d in overall_days) / len(overall_days)
        if overall_days
        else 0.0
    )
    columns: dict[str, list[Any]] = defaultdict(list)
    names = tuple(sorted(run.household for run in runs))
    position = {name: k for k, name in enumerate(names)}
    for run in runs:
        labelled = [PRIMARY_LABEL in d.labels for d in run.days]
        history = history_share(labelled, overall, protocol.history_strength)
        z = expanding_z([d.events for d in run.days], protocol.min_previous_days)
        for k, day in enumerate(run.days):
            if not day.evaluable:
                continue
            columns["household"].append(position[run.household])
            columns["label"].append(labelled[k])
            columns["any_label"].append(bool(day.labels))
            columns["alert"].append(bool(day.alerts))
            columns["deviating"].append(day.deviating(threshold))
            columns["score"].append(day.score)
            columns["history"].append(float(history[k]))
            columns["events"].append(float(z[k]))
    return _Evaluable(
        household=np.asarray(columns["household"], dtype=int),
        label=np.asarray(columns["label"], dtype=bool),
        any_label=np.asarray(columns["any_label"], dtype=bool),
        alert=np.asarray(columns["alert"], dtype=bool),
        deviating=np.asarray(columns["deviating"], dtype=bool),
        score=np.asarray(columns["score"], dtype=float),
        history=np.asarray(columns["history"], dtype=float),
        events=np.asarray(columns["events"], dtype=float),
        names=names,
    )


def _sums(values: np.ndarray, household: np.ndarray, households: int) -> np.ndarray:
    return np.bincount(household, weights=values.astype(float), minlength=households)


def _flagged(
    flag: np.ndarray, label: np.ndarray, table: _Evaluable, protocol: TihmProtocol
) -> dict[str, Any]:
    """How a day-level flag relates to a day-level label, by household sums."""
    n = len(table.names)
    weights = flag.astype(float)
    hit = _sums(weights * label, table.household, n)
    labelled = _sums(label, table.household, n)
    false = _sums(weights * ~label, table.household, n)
    other = _sums(~label, table.household, n)
    flagged = hit + false
    return {
        "days": int(label.size),
        "label_days": int(label.sum()),
        "flagged_days": float(weights.sum()),
        "label_days_flagged": float((weights * label).sum()),
        "households_with_a_label_day": int((labelled > 0).sum()),
        "share_of_label_days_flagged": ratio_estimate(hit, labelled, protocol),
        "share_of_other_days_flagged": ratio_estimate(false, other, protocol),
        "difference": difference_estimate(hit, labelled, false, other, protocol),
        "share_of_flagged_days_labelled": ratio_estimate(hit, flagged, protocol),
        "label_day_share": ratio_estimate(labelled, labelled + other, protocol),
    }


def _concordance(
    score: np.ndarray, label: np.ndarray, table: _Evaluable, protocol: TihmProtocol
) -> dict[str, Any]:
    n = len(table.names)
    concordant, pairs = np.zeros(n), np.zeros(n)
    for k in range(n):
        mask = table.household == k
        concordant[k], pairs[k] = concordance(score[mask], label[mask])
    return {
        **ratio_estimate(concordant, pairs, protocol),
        "pairs": float(pairs.sum()),
        "households_with_pairs": int((pairs > 0).sum()),
    }


def _association(table: _Evaluable, protocol: TihmProtocol) -> dict[str, Any]:
    return {
        "A1_alert_days": _flagged(table.alert, table.label, table, protocol),
        "A2_deviating_days": _flagged(table.deviating, table.label, table, protocol),
        "A3_deviation_score": _concordance(table.score, table.label, table, protocol),
        "any_label": {
            "alert_days": _flagged(table.alert, table.any_label, table, protocol),
            "deviating_days": _flagged(
                table.deviating, table.any_label, table, protocol
            ),
            "deviation_score": _concordance(
                table.score, table.any_label, table, protocol
            ),
        },
    }


def _caught(
    weights: np.ndarray, table: _Evaluable, protocol: TihmProtocol
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(table.names)
    hit = _sums(weights * table.label, table.household, n)
    labelled = _sums(table.label, table.household, n)
    return hit, {
        "label_days_caught": float(hit.sum()),
        "share_of_label_days": ratio_estimate(hit, labelled, protocol),
    }


def _references(table: _Evaluable, protocol: TihmProtocol) -> dict[str, Any]:
    n = len(table.names)
    budget = int(table.deviating.sum())
    labelled = _sums(table.label, table.household, n)
    pipeline_hit, pipeline = _caught(table.deviating.astype(float), table, protocol)
    result: dict[str, Any] = {
        "flags": budget,
        "days": int(table.label.size),
        "label_days": int(table.label.sum()),
        "random_expected_caught": (
            budget / table.label.size * float(table.label.sum())
            if table.label.size
            else None
        ),
        "pipeline_deviating_days": pipeline,
    }
    for name, score in (
        ("label_history", table.history),
        ("event_count", table.events),
    ):
        hit, entry = _caught(flag_weights(score, budget), table, protocol)
        entry["minus_pipeline"] = difference_estimate(
            hit, labelled, pipeline_hit, labelled, protocol
        )
        result[name] = entry
    result["event_count"]["concordance"] = _concordance(
        table.events, table.label, table, protocol
    )
    return result


# ----------------------------------------------------------------------------
# The published baseline's protocol
# ----------------------------------------------------------------------------
def read_physiology(path: Path) -> list[tuple[str, datetime, str, float]]:
    """The physiology table's rows: participant, time, device and value."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        expected = ["patient_id", "date", "device_type", "value", "unit"]
        if header != expected:
            raise ValueError(f"{Path(path).name} has columns {header}, not {expected}")
        return [
            (
                row[0],
                datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S"),
                row[2],
                float(row[3]),
            )
            for row in reader
            if row
        ]


def published_folds(
    days: Sequence[date], labelled: Sequence[bool]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The published protocol's train and test positions, newest test week first."""
    when = np.asarray([d.toordinal() for d in days])
    positive = np.asarray(labelled, dtype=bool)
    if not positive.any():
        return []
    end = int(when[positive].max())
    folds = []
    for _ in range(PUBLISHED_FOLDS):
        split = end - PUBLISHED_TEST_DAYS
        folds.append(
            (
                np.flatnonzero(when <= split),
                np.flatnonzero((when > split) & (when <= end)),
            )
        )
        end = split
    return folds


def _published(
    runs: Sequence[HouseholdRun],
    physiology: Sequence[tuple[str, datetime, str, float]],
    protocol: TihmProtocol,
) -> dict[str, Any]:
    """Two references on the published test weeks, at the published alert counts."""
    matrices = np.asarray(PUBLISHED_LOGISTIC)
    alerts, rows, positives = (
        published_alerts(),
        published_rows(),
        published_positives(),
    )
    tp, fp = int(matrices[:, 1, 1].sum()), int(matrices[:, 0, 1].sum())
    fn = int(matrices[:, 1, 0].sum())
    published = {
        "test_person_days": int(sum(rows)),
        "label_days": int(sum(positives)),
        "alerts": tp + fp,
        "label_days_caught": tp,
        "alerts_on_other_days": fp,
        "share_of_label_days_caught": tp / (tp + fn),
        "share_of_alerts_on_label_days": tp / (tp + fp),
        "mean_weekly_sensitivity": float(
            np.mean(matrices[:, 1, 1] / matrices[:, 1].sum(axis=1))
        ),
        "mean_weekly_specificity": float(
            np.mean(matrices[:, 0, 0] / matrices[:, 0].sum(axis=1))
        ),
        "alerts_per_person_month": (tp + fp) / sum(rows) * DAYS_PER_MONTH,
        "alerts_on_other_days_per_person_month": fp / sum(rows) * DAYS_PER_MONTH,
        "random_expected_caught": (tp + fp) / sum(rows) * sum(positives),
    }

    counts: dict[tuple[str, date], int] = defaultdict(int)
    for run in runs:
        for (day, _), count in run.hourly.items():
            counts[(run.household, day)] += count
    for household, moment, _, value in physiology:
        if value != 0:
            counts.setdefault((household, moment.date()), 0)
    label_days = {
        (run.household, moment.date())
        for run in runs
        for moment in run.label_times.get(PRIMARY_LABEL, ())
    }
    keys = sorted(counts)
    names = tuple(sorted({household for household, _ in keys}))
    position = {name: k for k, name in enumerate(names)}
    household = np.asarray([position[h] for h, _ in keys], dtype=int)
    labelled = np.asarray([key in label_days for key in keys], dtype=bool)
    events = np.zeros(len(keys))
    for name in names:
        mask = household == position[name]
        events[mask] = expanding_z(
            [counts[key] for key in keys if key[0] == name], protocol.min_previous_days
        )
    folds = published_folds([day for _, day in keys], labelled)
    reproduced = [
        (int(test.size), int(labelled[test].sum())) for _, test in folds
    ] == list(zip(rows, positives))
    result: dict[str, Any] = {
        "published": published,
        "reproduced_test_periods": reproduced,
        "test_periods": [
            {"person_days": int(test.size), "label_days": int(labelled[test].sum())}
            for _, test in folds
        ],
    }
    if not reproduced:
        return result

    n = len(names)
    weights = {"label_history": np.zeros(len(keys)), "event_count": np.zeros(len(keys))}
    tested = np.zeros(len(keys), dtype=bool)
    for (train, test), flags in zip(folds, alerts):
        overall = float(labelled[train].mean())
        seen = np.bincount(household[train], minlength=n).astype(float)
        hits = np.bincount(household[train], weights=labelled[train], minlength=n)
        share = (hits + protocol.history_strength * overall) / (
            seen + protocol.history_strength
        )
        weights["label_history"][test] = flag_weights(share[household[test]], flags)
        weights["event_count"][test] = flag_weights(events[test], flags)
        tested[test] = True
    in_test = _sums(tested & labelled, household, n)
    result["households_in_test"] = int(np.unique(household[tested]).size)
    result["households_with_a_label_day"] = int((in_test > 0).sum())
    ordered = np.sort(in_test)[::-1]
    result["top_five_households_share_of_label_days"] = float(
        ordered[:5].sum() / in_test.sum()
    )
    for name, weight in weights.items():
        hit = _sums(weight * labelled * tested, household, n)
        result[name] = {
            "label_days_caught": float(hit.sum()),
            "share_of_label_days": ratio_estimate(hit, in_test, protocol),
        }
    return result


# ----------------------------------------------------------------------------
# What the labels are
# ----------------------------------------------------------------------------
def slot_of(moment: datetime) -> str:
    """The declared slot a label is stamped on, or :data:`OTHER_SLOT`."""
    past = moment.minute * 60 + moment.second
    if moment.hour in LABEL_SLOTS and past < SLOT_TOLERANCE_SECONDS:
        return f"{moment.hour:02d}:00"
    return OTHER_SLOT


def _hourly_profile(runs: Sequence[HouseholdRun]) -> dict[str, Any]:
    """Hourly events on label days against the household's own hours, by slot."""
    by_group: dict[str, list[np.ndarray]] = defaultdict(list)
    blocks: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        days = sorted({day for day, _ in run.hourly})
        full = days[1:-1]
        if len(full) < 2:
            continue
        counts = np.array(
            [[run.hourly.get((day, hour), 0) for hour in range(24)] for day in full],
            dtype=float,
        )
        spread = counts.std(axis=0, ddof=1)
        z = np.where(spread > 0, (counts - counts.mean(axis=0)) / spread, np.nan)
        first: dict[date, datetime] = {}
        for moment in run.label_times.get(PRIMARY_LABEL, ()):
            first.setdefault(moment.date(), moment)
        block_counts = {
            name: counts[:, low:high].sum(axis=1) for name, low, high in BLOCKS
        }
        for k, day in enumerate(full):
            group = slot_of(first[day]) if day in first else NO_LABEL
            by_group[group].append(z[k])
            for name, values in block_counts.items():
                scale = values.std(ddof=1)
                if scale > 0:
                    blocks[group][name].append(
                        float((values[k] - values.mean()) / scale)
                    )
    return {
        group: {
            "days": len(rows),
            "hourly_mean_z": [
                None if np.isnan(value) else float(value)
                for value in _nanmean(np.vstack(rows))
            ],
            "block_mean_z": {
                name: (
                    float(np.mean(blocks[group][name])) if blocks[group][name] else None
                )
                for name, _, _ in BLOCKS
            },
        }
        for group, rows in sorted(by_group.items())
    }


def _nanmean(matrix: np.ndarray) -> np.ndarray:
    counted = (~np.isnan(matrix)).sum(axis=0)
    total = np.nansum(matrix, axis=0)
    return np.where(counted > 0, total / np.maximum(counted, 1), np.nan)


def _limits(
    runs: Sequence[HouseholdRun],
    physiology: Sequence[tuple[str, datetime, str, float]],
) -> dict[str, Any]:
    """Label days against the limits the dataset paper states for each reading."""
    readings: dict[str, dict[tuple[str, date], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for household, moment, device, value in physiology:
        readings[device][(household, moment.date())].append(value)
    label_days: dict[str, set[tuple[str, date]]] = defaultdict(set)
    for run in runs:
        for label, moments in run.label_times.items():
            label_days[label].update((run.household, m.date()) for m in moments)
    result: dict[str, Any] = {}
    for label, limits in STATED_LIMITS.items():
        with_reading: set[tuple[str, date]] = set()
        met: set[tuple[str, date]] = set()
        for device, below, above in limits:
            for key, values in readings[device].items():
                with_reading.add(key)
                if min(values) < below or max(values) > above:
                    met.add(key)
        labelled = label_days.get(label, set())
        per_household = Counter(household for household, _ in labelled)
        top = per_household.most_common(1)
        result[label] = {
            "label_days": len(labelled),
            "days_with_a_reading": len(with_reading),
            "days_meeting_the_limits": len(met),
            "label_days_meeting_the_limits": len(met & labelled),
            "label_days_without_a_reading": len(labelled - with_reading),
            "days_meeting_the_limits_without_a_label": len(met - labelled),
            "households_with_a_label": len(per_household),
            "largest_household_label_days": top[0][1] if top else 0,
            "largest_household_readings": (
                sum(
                    len(values)
                    for device, _, _ in limits
                    for (household, _), values in readings[device].items()
                    if household == top[0][0]
                )
                if top
                else 0
            ),
            "reading_range": {
                device: (
                    [
                        float(min(min(v) for v in readings[device].values())),
                        float(max(max(v) for v in readings[device].values())),
                    ]
                    if readings[device]
                    else None
                )
                for device, _, _ in limits
            },
        }
    return result


def _labels(
    runs: Sequence[HouseholdRun],
    physiology: Sequence[tuple[str, datetime, str, float]],
) -> dict[str, Any]:
    rows: Counter[str] = Counter()
    days: Counter[str] = Counter()
    households: Counter[str] = Counter()
    for run in runs:
        for label, moments in run.label_times.items():
            rows[label] += len(moments)
            days[label] += len({m.date() for m in moments})
            households[label] += 1
    moments = [m for run in runs for m in run.label_times.get(PRIMARY_LABEL, ())]
    slots = Counter(slot_of(m) for m in moments)
    on_slot = [m.minute * 60 + m.second for m in moments if slot_of(m) != OTHER_SLOT]
    per_household = sorted(
        (
            len({m.date() for m in run.label_times.get(PRIMARY_LABEL, ())})
            for run in runs
        ),
        reverse=True,
    )
    total = sum(per_household)
    after_label = after_none = label_then_label = none_then_label = 0
    for run in runs:
        labelled = {d.day: PRIMARY_LABEL in d.labels for d in run.days}
        for day, today in labelled.items():
            yesterday = labelled.get(day - timedelta(days=1))
            if yesterday is None:
                continue
            if yesterday:
                after_label += 1
                label_then_label += today
            else:
                after_none += 1
                none_then_label += today
    return {
        "rows_by_type": dict(sorted(rows.items())),
        "label_days_by_type": dict(sorted(days.items())),
        "households_by_type": dict(sorted(households.items())),
        "households_without_any_label": sum(1 for run in runs if not run.label_times),
        "primary": {
            "label": PRIMARY_LABEL,
            "rows": len(moments),
            "label_days": total,
            "households": sum(1 for count in per_household if count > 0),
            "households_never_labelled": sum(
                1 for count in per_household if count == 0
            ),
            "largest_household_share": per_household[0] / total if total else None,
            "five_largest_households_share": (
                sum(per_household[:5]) / total if total else None
            ),
            "rows_by_slot": dict(sorted(slots.items())),
            "latest_seconds_past_a_slot": max(on_slot) if on_slot else None,
            "share_labelled_after_a_label_day": (
                label_then_label / after_label if after_label else None
            ),
            "share_labelled_after_another_day": (
                none_then_label / after_none if after_none else None
            ),
            "days_after_a_label_day": after_label,
        },
        "hourly_profile": _hourly_profile(runs),
        "stated_limits": _limits(runs, physiology),
    }


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------
def score(
    runs: Sequence[HouseholdRun],
    physiology: Sequence[tuple[str, datetime, str, float]],
    protocol: TihmProtocol,
    *,
    households: int | None = None,
    refused: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Every result of the protocol, from the households' runs."""
    threshold = BaselineConfig().deviation_threshold
    table = _per_household(runs, threshold)
    evaluable = _evaluable(runs, protocol, threshold)
    return {
        "result_schema": RESULT_SCHEMA,
        "contract": {
            **_contract(runs, households if households is not None else len(runs)),
            "refused": {h: list(codes) for h, codes in (refused or {}).items()},
        },
        "monitoring": _monitoring(runs, table, protocol),
        "burden": _burden(runs, table, protocol),
        "association": _association(evaluable, protocol),
        "references": _references(evaluable, protocol),
        "published_baseline": _published(runs, physiology, protocol),
        "labels": _labels(runs, physiology),
        "households": table,
    }


def _intervals(results: Mapping[str, Any], protocol: TihmProtocol) -> list[Any]:
    quoted = {
        "B1: behavioural alerts per monitored person-day": results["burden"][
            "B1_per_monitored_day"
        ],
        f"B2: {SIMULATOR_FEATURE} alerts per monitored person-day": results["burden"][
            "B2_simulator_feature_per_monitored_day"
        ],
        "B3: behavioural alerts per evaluable person-day": results["burden"][
            "B3_per_evaluable_day"
        ],
        "H1: usable share of monitored days": results["monitoring"]["usable_share"],
        "H1: evaluable share of monitored days": results["monitoring"][
            "evaluable_share"
        ],
        "A1: label days that are alert days, minus other days": results["association"][
            "A1_alert_days"
        ]["difference"],
        "A2: label days that are deviating days, minus other days": results[
            "association"
        ]["A2_deviating_days"]["difference"],
        "A3: within-household concordance of the deviation score": results[
            "association"
        ]["A3_deviation_score"],
    }
    intervals = []
    for label, entry in quoted.items():
        if entry.get("interval") is None:
            continue
        intervals.append(
            ReportedInterval(
                label=label,
                estimate=entry["estimate"],
                low=entry["interval"]["low"],
                high=entry["interval"]["high"],
                confidence=protocol.confidence,
                method="percentile",
                unit="household",
                n=entry["households"],
            )
        )
    return intervals


@dataclass(frozen=True)
class TihmResult:
    """The record of a run and where it was written."""

    record: ExperimentRecord
    path: Path | None


def run_tihm(
    adapter: DatasetAdapter,
    physiology: Sequence[tuple[str, datetime, str, float]],
    protocol: TihmProtocol,
    *,
    protocol_sha256: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> TihmResult:
    """Run every household and write the record of the protocol's results."""
    runs: list[HouseholdRun] = []
    refused: dict[str, list[str]] = {}
    names = list(adapter.households())
    for name in names:
        data = adapter.load(name)
        report = validate_household(data, protocol.mapping)
        if not report.ok:
            refused[name] = sorted({issue.code for issue in report.errors})
            continue
        runs.append(run_household(data, protocol))
    if not runs:
        raise ValueError("no household passed the contract, so there is nothing to run")
    results = score(runs, physiology, protocol, households=len(names), refused=refused)
    baseline, pipeline = BaselineConfig(), PipelineConfig()
    record = ExperimentRecord(
        experiment=EXPERIMENT,
        configuration={
            **protocol.to_dict(),
            "protocol_sha256": protocol.sha256(),
            "protocol_file_sha256": protocol_sha256,
            "result_schema": RESULT_SCHEMA,
        },
        inference=ONLINE,
        evidence=EvidenceSummary.online(
            moment for run in runs for moment in run.closes
        ),
        seeds=[protocol.seed],
        results=results,
        data_source="tihm",
        metric_definitions=dict(protocol.to_dict()["definitions"]),
        notes=[
            "Exploratory. The dataset's labels had been analysed before the "
            "protocol was written; the protocol was committed before any "
            "pipeline output was compared with a label.",
            "Nothing was fitted on TIHM. The pipeline's emissions, baseline and "
            "alert policy are the declared defaults.",
            "The labels are alerts a monitoring team verified after an earlier "
            "model raised them. A day without a label is not a day without an "
            "episode.",
            "Intervals resample households. Every figure describes these homes "
            "and is not an estimate for other homes.",
        ],
        inputs=list(inputs),
        preprocessing={
            "contract": "external-dataset contract, strict conversion",
            "mapping_sha256": protocol.mapping.sha256(),
            "step_minutes": protocol.step_minutes,
        },
        models=[
            ModelRecord(
                "online_pipeline_defaults",
                {
                    "features": [f"{s.value}_hours" for s in pipeline.features],
                    "baseline_min_samples": baseline.min_samples,
                    "deviation_threshold": baseline.deviation_threshold,
                    "persistence_days": baseline.persistence_days,
                },
                "BehaviouralSensingPipeline",
            )
        ],
        household_metrics={"online_pipeline_defaults": results["households"]},
        intervals=_intervals(results, protocol),
    )
    path = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = record.write(Path(output_dir) / f"{EXPERIMENT}.json")
    return TihmResult(record, path)
