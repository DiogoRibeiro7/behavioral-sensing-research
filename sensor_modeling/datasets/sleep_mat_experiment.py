"""The sleep-mat comparison: the pipeline's hours of sleep beside a sleep mat.

What is compared, on which days and homes, and how it is judged are declared
in :mod:`sensor_modeling.datasets.sleep_mat_protocol` and frozen before any
value of the pipeline is set beside any record of the mat. This module reads
the mat, runs the pipeline on the homes that have one, matches their days, and
computes the declared estimands. It also computes the same estimands on
simulated homes against the simulator's true hours of sleep, as a reference.

TIHM is by Palermo et al., *Scientific Data* 10, 606 (2023), under CC BY 4.0.
Surrey and Borders Partnership NHS Foundation Trust and Howz are acknowledged,
as the dataset asks. Nothing of the dataset is redistributed: the records hold
per-home summaries, never a day's values.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy import stats

from ..baseline.adaptive import AdaptiveBaseline, BaselineConfig, ChangeKind
from ..evaluation import ExperimentRecord, InputArtifact, ModelRecord, ReportedInterval
from ..external.contract import DatasetAdapter, HouseholdData
from ..external.tihm import TIMEZONE as TIHM_TIMEZONE
from ..external.validation import validate_household
from ..fusion import ONLINE, EvidenceSummary
from ..health.monitor import HealthConfig
from ..online.pipeline import (
    BehaviouralSensingPipeline,
    PipelineConfig,
    daily_summaries,
)
from ..simulation.faults import DegradationConfig, degrade
from ..simulation.household import GroundTruth, HouseholdConfig, simulate
from ..states.ontology import BehaviouralState
from .sleep_mat_protocol import (
    AGREES,
    ANALYSES,
    ASLEEP_STATES,
    DOES_NOT_AGREE,
    DOES_NOT_FOLLOW,
    EXPERIMENT,
    FOLLOWS,
    IN_BED,
    INCONCLUSIVE,
    JUDGED_KINDS,
    MAT_COLUMNS,
    MAT_FILE,
    MAT_STATES,
    MAT_TIME_FORMAT,
    NOT_ESTIMABLE,
    PRIMARY,
    REFERENCES,
    RESULT_SCHEMA,
    RUN_OFF,
    RUN_RULE,
    RUNS,
    SIMULATED_EXPERIMENT,
    SIMULATED_SCHEMA,
    SLEEP,
    STAGED_ASLEEP_SHARE,
    STAGED_AWAKE_HOMES,
    TRACKED_FEATURE,
    SleepMatProtocol,
    event_sensors,
)
from .tihm_experiment import HouseholdRun, _per_household, run_household
from .tihm_protocol import TihmProtocol

#: The pipeline's name for the state compared.
SLEEPING = BehaviouralState.SLEEPING.value

#: Fewer included homes than this and a criterion is not estimable.
MIN_HOMES = 3

INSUFFICIENT = ChangeKind.INSUFFICIENT_DATA.value


# ----------------------------------------------------------------------------
# The mat
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class MatDay:
    """What the mat recorded on one local day, in hours."""

    sleep: float
    in_bed: float

    def of(self, reference: str) -> float:
        """The day's hours under *reference*."""
        if reference == SLEEP:
            return self.sleep
        if reference == IN_BED:
            return self.in_bed
        raise KeyError(f"no reference '{reference}'")


@dataclass(frozen=True)
class MatHome:
    """One home's mat records, by local day.

    Attributes
    ----------
    home
        The dataset's identifier.
    days
        Every day with at least one record.
    observed
        The days whose day before and day after also have a record.
    hourly
        Minutes in bed by local ``(day, hour)``.
    records
        The number of records.
    """

    home: str
    days: Mapping[date, MatDay]
    observed: frozenset[date]
    hourly: Mapping[tuple[date, int], int] = field(default_factory=dict)
    records: int = 0

    def staged_asleep_share(self) -> float | None:
        """The share of minutes in bed on observed days staged as asleep."""
        in_bed = sum(self.days[d].in_bed for d in self.observed)
        if in_bed == 0.0:
            return None
        return sum(self.days[d].sleep for d in self.observed) / in_bed


def observed_days(days: Iterable[date]) -> frozenset[date]:
    """The days whose neighbours on both sides are also among *days*."""
    present = set(days)
    one = timedelta(days=1)
    return frozenset(d for d in present if d - one in present and d + one in present)


def read_mat(path: Path, shift_hours: float = 0.0) -> dict[str, MatHome]:
    """Read the mat's file into each home's days.

    Each record is a minute. Its day is the local date of its timestamp, read
    as a local clock time in the file's exact format and moved by
    *shift_hours* first. A record whose state is not one the device reports is
    refused, as is a repeated minute.
    """
    shift = timedelta(hours=shift_hours)
    asleep: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))
    in_bed: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))
    hourly: dict[str, dict[tuple[date, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    seen: set[tuple[str, datetime]] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MAT_COLUMNS:
            raise ValueError(
                f"{MAT_FILE} has columns {reader.fieldnames}, not {list(MAT_COLUMNS)}"
            )
        for row in reader:
            home, state = row["patient_id"], row["state"]
            if state not in MAT_STATES:
                raise ValueError(f"{MAT_FILE} has an unknown state {state!r}")
            moment = datetime.strptime(row["date"], MAT_TIME_FORMAT)
            if (home, moment) in seen:
                raise ValueError(f"{MAT_FILE} repeats the minute {moment} of {home}")
            seen.add((home, moment))
            local = moment + shift
            day = local.date()
            in_bed[home][day] += 1
            hourly[home][(day, local.hour)] += 1
            if state in ASLEEP_STATES:
                asleep[home][day] += 1
    homes: dict[str, MatHome] = {}
    for home in sorted(in_bed):
        days = {
            day: MatDay(sleep=asleep[home][day] / 60.0, in_bed=minutes / 60.0)
            for day, minutes in sorted(in_bed[home].items())
        }
        homes[home] = MatHome(
            home=home,
            days=days,
            observed=observed_days(days),
            hourly=dict(hourly[home]),
            records=sum(in_bed[home].values()),
        )
    return homes


def staged_awake_homes(mats: Mapping[str, MatHome]) -> tuple[str, ...]:
    """The homes whose mat stages less than the declared share as asleep."""
    out = []
    for home, mat in sorted(mats.items()):
        share = mat.staged_asleep_share()
        if share is not None and share < STAGED_ASLEEP_SHARE:
            out.append(home)
    return tuple(out)


# ----------------------------------------------------------------------------
# The pipeline's days
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineDay:
    """What the pipeline concluded about one day of a home, for this comparison.

    Attributes
    ----------
    day
        The local calendar date.
    usable
        Whether the day's summary passed the pipeline's quality tests.
    value
        The expected hours of sleeping in the day's summary.
    kind, deviation
        The baseline's verdict on the hours of sleep and its deviation.
    alerts
        Behavioural alerts about the hours of sleep raised when the day closed.
    events
        Activity records on the day.
    """

    day: date
    usable: bool
    value: float
    kind: str
    deviation: float
    alerts: int
    events: int

    @property
    def has_verdict(self) -> bool:
        """Whether the baseline gave the day a verdict on the hours of sleep."""
        return self.usable and self.kind not in ("", INSUFFICIENT)


def pipeline_days(run: HouseholdRun) -> tuple[PipelineDay, ...]:
    """The days of one run, as this comparison reads them."""
    return tuple(
        PipelineDay(
            day=record.day,
            usable=record.usable,
            value=float(record.hours.get(SLEEPING, 0.0)),
            kind=record.kinds.get(TRACKED_FEATURE, ""),
            deviation=float(record.deviations.get(TRACKED_FEATURE, 0.0)),
            alerts=sum(1 for subject, _ in record.alerts if subject == TRACKED_FEATURE),
            events=record.events,
        )
        for record in run.days
    )


# ----------------------------------------------------------------------------
# Matching and the per-home statistics
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Pairs:
    """A home's matched days: the pipeline's value beside the mat's."""

    home: str
    days: tuple[date, ...]
    pipeline: np.ndarray
    references: Mapping[str, np.ndarray]

    @property
    def size(self) -> int:
        """The number of matched days."""
        return len(self.days)


def matched_days(
    days: Sequence[PipelineDay], mat: MatHome | None, keep_silent: bool
) -> list[PipelineDay]:
    """The usable days the mat observed, other than the home's first and last.

    A silent day is left out unless *keep_silent*.
    """
    if mat is None or not days:
        return []
    first, last = days[0].day, days[-1].day
    return [
        d
        for d in days
        if d.usable
        and d.day in mat.observed
        and first < d.day < last
        and (keep_silent or d.events > 0)
    ]


def pairs_of(
    home: str,
    days: Sequence[PipelineDay],
    mat: MatHome | None,
    keep_silent: bool = False,
) -> Pairs:
    """A home's matched days, as arrays."""
    chosen = matched_days(days, mat, keep_silent)
    return Pairs(
        home=home,
        days=tuple(d.day for d in chosen),
        pipeline=np.array([d.value for d in chosen], dtype=float),
        references={
            reference: np.array(
                [mat.days[d.day].of(reference) for d in chosen] if mat else [],
                dtype=float,
            )
            for reference in REFERENCES
        },
    )


def correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float | None:
    """Spearman's or Pearson's correlation; ``None`` when either is constant."""
    if x.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    if kind == "spearman":
        return float(stats.spearmanr(x, y)[0])
    if kind == "pearson":
        return float(stats.pearsonr(x, y)[0])
    raise KeyError(f"no correlation '{kind}'")


def home_statistics(pairs: Pairs, protocol: SleepMatProtocol) -> dict[str, Any]:
    """Everything an estimand reads from one home, against each reference.

    A home with constant values on either source has no correlation; as the
    protocol declares, it counts as 0, and ``constant`` says so.
    """
    included = pairs.size >= protocol.min_matched_days
    out: dict[str, Any] = {"matched_days": pairs.size, "included": included}
    for reference, values in pairs.references.items():
        difference = pairs.pipeline - values
        spearman = correlation(pairs.pipeline, values, "spearman")
        pearson = correlation(pairs.pipeline, values, "pearson")
        constant = pairs.size >= 3 and spearman is None
        out[reference] = {
            "spearman": 0.0 if constant else spearman,
            "pearson": 0.0 if constant else pearson,
            "constant": constant,
            "mean_difference": float(difference.mean()) if pairs.size else None,
            "variance_of_differences": (
                float(difference.var(ddof=1)) if pairs.size >= 2 else None
            ),
            "median_pipeline": float(np.median(pairs.pipeline)) if pairs.size else None,
            "median_mat": float(np.median(values)) if pairs.size else None,
        }
    return out


def mean_over_homes(values: Sequence[float], confidence: float) -> dict[str, Any]:
    """A mean over homes with its Student t interval."""
    array = np.asarray(values, dtype=float)
    homes = int(array.size)
    if homes == 0:
        return {"estimate": None, "interval": None, "homes": 0, "sd": None}
    estimate = float(array.mean())
    if homes < 2:
        return {"estimate": estimate, "interval": None, "homes": homes, "sd": None}
    sd = float(array.std(ddof=1))
    half = float(
        stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, homes - 1) * sd / math.sqrt(homes)
    )
    return {
        "estimate": estimate,
        "interval": {
            "low": estimate - half,
            "high": estimate + half,
            "confidence": confidence,
            "method": "student_t",
        },
        "homes": homes,
        "sd": sd,
    }


def limits_of_agreement(
    means: Sequence[float], variances: Sequence[float]
) -> dict[str, Any]:
    """E4: the bias plus and minus 1.96 of the between- and within-home spread."""
    if len(means) < 2:
        return {"bias": None, "low": None, "high": None}
    between = float(np.var(means, ddof=1))
    within = float(np.mean(variances))
    spread = 1.96 * math.sqrt(between + within)
    bias = float(np.mean(means))
    return {
        "bias": bias,
        "low": bias - spread,
        "high": bias + spread,
        "between_home_sd": math.sqrt(between),
        "within_home_sd": math.sqrt(within),
    }


def tracking_reading(estimate: Mapping[str, Any], margin: float) -> str:
    """C1's reading of E1."""
    interval = estimate.get("interval")
    if estimate.get("homes", 0) < MIN_HOMES or not interval:
        return NOT_ESTIMABLE
    if interval["low"] > margin:
        return FOLLOWS
    if interval["high"] < margin:
        return DOES_NOT_FOLLOW
    return INCONCLUSIVE


def level_reading(estimate: Mapping[str, Any], margin: float) -> str:
    """C2's reading of E2."""
    interval = estimate.get("interval")
    if estimate.get("homes", 0) < MIN_HOMES or not interval:
        return NOT_ESTIMABLE
    if -margin < interval["low"] and interval["high"] < margin:
        return AGREES
    if interval["high"] < -margin or interval["low"] > margin:
        return DOES_NOT_AGREE
    return INCONCLUSIVE


def agreement(
    table: Mapping[str, Mapping[str, Any]],
    protocol: SleepMatProtocol,
    leave_out: Iterable[str] = (),
) -> dict[str, Any]:
    """E1 to E5 over the included homes of one analysis."""
    left_out = set(leave_out)
    included = [
        home
        for home, row in sorted(table.items())
        if row["included"] and home not in left_out
    ]
    out: dict[str, Any] = {"homes": included}
    for reference in REFERENCES:
        rows = [table[home][reference] for home in included]
        means = [r["mean_difference"] for r in rows]
        variances = [r["variance_of_differences"] for r in rows]
        out[reference] = {
            "spearman": mean_over_homes(
                [r["spearman"] for r in rows], protocol.confidence
            ),
            "pearson": mean_over_homes(
                [r["pearson"] for r in rows], protocol.confidence
            ),
            "homes_with_constant_values": [
                home for home in included if table[home][reference]["constant"]
            ],
            "mean_difference": mean_over_homes(means, protocol.confidence),
            "limits_of_agreement": limits_of_agreement(means, variances),
        }
    return out


# ----------------------------------------------------------------------------
# E6 to E8 and E11
# ----------------------------------------------------------------------------
def mat_verdicts(mat: MatHome) -> dict[date, tuple[str, float]]:
    """The verdicts a default personal baseline gives the mat's sleep.

    The mat's observed days are passed in order, as the pipeline passes its
    usable days; a day the mat did not observe is not passed.
    """
    baseline = AdaptiveBaseline(TRACKED_FEATURE, BaselineConfig())
    verdicts: dict[date, tuple[str, float]] = {}
    for day in sorted(mat.observed):
        change = baseline.observe(day, mat.days[day].sleep)
        verdicts[day] = (change.kind.value, float(change.deviation))
    return verdicts


def deviations(
    days: Mapping[str, Sequence[PipelineDay]],
    pairs: Mapping[str, Pairs],
    verdicts: Mapping[str, Mapping[date, tuple[str, float]]],
    protocol: SleepMatProtocol,
) -> dict[str, Any]:
    """E6: the deviations each source's personal baseline gives the same days."""
    threshold = protocol.alert_threshold
    correlations: list[float] = []
    per_home: dict[str, Any] = {}
    flagged = same_sign = also_past = 0
    for home in sorted(pairs):
        matched = set(pairs[home].days)
        mat = verdicts.get(home, {})
        both = [
            (d.deviation, mat[d.day][1])
            for d in days[home]
            if d.day in matched
            and d.has_verdict
            and d.day in mat
            and mat[d.day][0] != INSUFFICIENT
        ]
        x = np.array([a for a, _ in both], dtype=float)
        y = np.array([b for _, b in both], dtype=float)
        rho = None
        if len(both) >= protocol.min_matched_days:
            found = correlation(x, y, "spearman")
            rho = 0.0 if found is None else found
            correlations.append(rho)
        past = [(a, b) for a, b in both if abs(a) >= threshold]
        home_same = sum(1 for a, b in past if a * b > 0)
        home_also = sum(1 for a, b in past if a * b > 0 and abs(b) >= threshold)
        flagged += len(past)
        same_sign += home_same
        also_past += home_also
        per_home[home] = {
            "days_with_both_verdicts": len(both),
            "spearman": rho,
            "pipeline_past_threshold": len(past),
            "same_sign": home_same,
            "also_past_threshold": home_also,
        }
    return {
        "threshold": threshold,
        "spearman": mean_over_homes(correlations, protocol.confidence),
        "pipeline_past_threshold": flagged,
        "same_sign": same_sign,
        "also_past_threshold": also_past,
        "homes": per_home,
    }


def alerts_beside_the_mat(
    days: Mapping[str, Sequence[PipelineDay]],
    mats: Mapping[str, MatHome],
    pairs: Mapping[str, Pairs],
    verdicts: Mapping[str, Mapping[date, tuple[str, float]]],
) -> dict[str, Any]:
    """E7: the alerts about sleep, and what the mat says on the days they came."""

    def empty() -> dict[str, Any]:
        return {
            "alerts": 0,
            "by_kind": {},
            "judged": 0,
            "median": {"same_side": 0, "opposite": 0, "at_the_median": 0},
            "mat_baseline": {"same_sign": 0, "opposite": 0, "no_verdict": 0},
        }

    total = empty()
    per_home: dict[str, Any] = {}
    for home in sorted(days):
        if home not in mats:
            continue
        mat, home_pairs = mats[home], pairs[home]
        centre = (
            float(np.median(home_pairs.references[SLEEP])) if home_pairs.size else None
        )
        matched = set(home_pairs.days)
        counts = empty()
        for d in days[home]:
            if not d.alerts:
                continue
            for target in (counts, total):
                target["alerts"] += d.alerts
                target["by_kind"][d.kind] = target["by_kind"].get(d.kind, 0) + d.alerts
            if d.kind not in JUDGED_KINDS or d.day not in matched or centre is None:
                continue
            sign = np.sign(d.deviation)
            side = (mat.days[d.day].sleep - centre) * sign
            median_key = (
                "same_side" if side > 0 else "opposite" if side < 0 else "at_the_median"
            )
            verdict = verdicts.get(home, {}).get(d.day)
            if verdict is None or verdict[0] == INSUFFICIENT:
                baseline_key = "no_verdict"
            else:
                baseline_key = "same_sign" if verdict[1] * sign > 0 else "opposite"
            for target in (counts, total):
                target["judged"] += d.alerts
                target["median"][median_key] += d.alerts
                target["mat_baseline"][baseline_key] += d.alerts
        per_home[home] = counts
    return {**total, "homes": per_home}


def silent_days(
    days: Mapping[str, Sequence[PipelineDay]], mats: Mapping[str, MatHome]
) -> dict[str, Any]:
    """E8: days with no activity record, and what the mat recorded on them."""
    per_home: dict[str, Any] = {}
    silent: list[tuple[str, PipelineDay]] = []
    observed: list[tuple[str, PipelineDay]] = []
    for home in sorted(days):
        if home not in mats:
            continue
        home_silent = [d for d in days[home] if not d.events]
        home_observed = [d for d in home_silent if d.day in mats[home].observed]
        silent += [(home, d) for d in home_silent]
        observed += [(home, d) for d in home_observed]
        per_home[home] = {
            "silent_days": len(home_silent),
            "usable": sum(1 for d in home_silent if d.usable),
            "mat_observed": len(home_observed),
        }
    return {
        "silent_days": len(silent),
        "usable": sum(1 for _, d in silent if d.usable),
        "mat_observed": len(observed),
        "median_pipeline_hours": (
            median(d.value for _, d in observed) if observed else None
        ),
        "median_mat_sleep_hours": (
            median(mats[home].days[d.day].sleep for home, d in observed)
            if observed
            else None
        ),
        "median_mat_in_bed_hours": (
            median(mats[home].days[d.day].in_bed for home, d in observed)
            if observed
            else None
        ),
        "homes": per_home,
    }


def hourly_alignment(
    runs: Mapping[str, HouseholdRun],
    mats: Mapping[str, MatHome],
    pairs: Mapping[str, Pairs],
    homes: Sequence[str],
    protocol: SleepMatProtocol,
) -> dict[str, Any]:
    """E11: hourly activity against minutes in bed, at each lag of the mat's clock.

    It reads the activity records per local hour and the mat, and no output of
    the pipeline. At a lag of *k* hours, a minute in bed recorded at local hour
    *h* is counted at hour *h* + *k*.
    """
    by_lag: dict[str, Any] = {}
    for lag in protocol.lags_hours:
        values: list[float] = []
        per_home: dict[str, float | None] = {}
        for home in homes:
            hourly, bed = runs[home].hourly, mats[home].hourly
            activity, minutes = [], []
            for day in pairs[home].days:
                for hour in range(24):
                    moment = datetime.combine(day, datetime.min.time()) + timedelta(
                        hours=hour - lag
                    )
                    activity.append(hourly.get((day, hour), 0))
                    minutes.append(bed.get((moment.date(), moment.hour), 0))
            rho = correlation(
                np.array(activity, dtype=float),
                np.array(minutes, dtype=float),
                "spearman",
            )
            per_home[home] = rho
            values.append(0.0 if rho is None else rho)
        by_lag[f"{lag:+d}"] = {
            "spearman": mean_over_homes(values, protocol.confidence),
            "homes": per_home,
        }
    estimates = {
        lag: entry["spearman"]["estimate"]
        for lag, entry in by_lag.items()
        if entry["spearman"]["estimate"] is not None
    }
    return {
        "lags": by_lag,
        "most_negative": (
            min(estimates, key=estimates.__getitem__) if estimates else None
        ),
    }


# ----------------------------------------------------------------------------
# Running the pipeline on the mat homes, and checking the runs
# ----------------------------------------------------------------------------
def _run(arguments: tuple[HouseholdData, TihmProtocol, float | None]) -> HouseholdRun:
    data, tihm, hours = arguments
    if hours is None:
        return run_household(data, tihm)
    return run_household(
        data, tihm, HealthConfig(home_silence_horizon=timedelta(hours=hours))
    )


def run_mat_homes(
    adapter: DatasetAdapter,
    homes: Sequence[str],
    protocol: SleepMatProtocol,
    tihm: TihmProtocol,
    *,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, dict[str, HouseholdRun]]:
    """Run each mat home under each run; refuse one the contract refuses."""
    loaded = []
    for name in homes:
        data = adapter.load(name)
        if not validate_household(data, tihm.mapping).ok:
            raise ValueError(f"the contract refuses mat home {name!r}")
        loaded.append(data)
    hours = {RUN_OFF: None, RUN_RULE: protocol.rule_hours}
    tasks = [(data, tihm, hours[run]) for run in RUNS for data in loaded]
    done: list[HouseholdRun] = []
    if jobs <= 1:
        for task in tasks:
            done.append(_run(task))
            if progress is not None:
                progress(len(done), len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for result in pool.map(_run, tasks):
                done.append(result)
                if progress is not None:
                    progress(len(done), len(tasks))
    size = len(loaded)
    return {
        run: {result.household: result for result in done[k * size : (k + 1) * size]}
        for k, run in enumerate(RUNS)
    }


def check_published(
    runs: Mapping[str, HouseholdRun], published: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Refuse a run with the rule off that is not the published one.

    Every count of every mat home's row in the published record is compared.
    """
    threshold = BaselineConfig().deviation_threshold
    here = _per_household(list(runs.values()), threshold)
    differing = []
    for home in sorted(runs):
        there = published.get(home)
        if there is None or {k: there.get(k) for k in here[home]} != here[home]:
            differing.append(home)
    if differing:
        raise ValueError(
            "with the rule off the run does not give the published record's "
            f"counts for {', '.join(differing)}; nothing is reported"
        )
    return {
        "homes": len(runs),
        "fields": sorted(next(iter(here.values()))) if here else [],
        "monitored_days": sum(row["monitored_days"] for row in here.values()),
        "behavioural_alerts": sum(row["behavioural_alerts"] for row in here.values()),
        "reproduced": True,
    }


def check_rule(
    runs: Mapping[str, HouseholdRun],
    off: Mapping[str, HouseholdRun],
    silent_home: Mapping[str, Any],
    condition: str,
) -> dict[str, Any]:
    """Refuse a run with the rule on that is not the silent-home record's."""
    record = silent_home["conditions"][condition]
    refused = record["days_refused_because_of_the_rule"]["per_home"]
    alerts = record["silence_alerts"]["per_home"]
    differing = []
    for home, run in sorted(runs.items()):
        here = (
            len(run.days),
            sum(1 for day in run.days if day.silent > 0.0),
            len(run.silence_alerts),
        )
        there = (len(off[home].days), refused.get(home), alerts.get(home))
        if here != there:
            differing.append(home)
    if differing:
        raise ValueError(
            "with the rule on the run does not give the silent-home record's "
            f"counts for {', '.join(differing)}; nothing is reported"
        )
    return {
        "homes": len(runs),
        "days_refused_because_of_the_rule": sum(
            sum(1 for day in run.days if day.silent > 0.0) for run in runs.values()
        ),
        "silence_alerts": sum(len(run.silence_alerts) for run in runs.values()),
        "reproduced": True,
    }


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def _analysis(
    days: Mapping[str, Sequence[PipelineDay]],
    mats: Mapping[str, MatHome],
    protocol: SleepMatProtocol,
    keep_silent: bool,
) -> tuple[dict[str, Pairs], dict[str, Any]]:
    pairs = {
        home: pairs_of(home, days[home], mats.get(home), keep_silent)
        for home in sorted(days)
    }
    table = {home: home_statistics(pairs[home], protocol) for home in sorted(days)}
    return pairs, table


def score(
    runs: Mapping[str, Mapping[str, HouseholdRun]],
    mats: Mapping[str, MatHome],
    shifted: Mapping[float, Mapping[str, MatHome]],
    protocol: SleepMatProtocol,
) -> dict[str, Any]:
    """Every estimand and criterion, from the runs and the mat."""
    found = staged_awake_homes(mats)
    if found != STAGED_AWAKE_HOMES:
        raise ValueError(
            f"the mat's file gives {list(found)} as the homes staged mostly awake, "
            f"not the declared {list(STAGED_AWAKE_HOMES)}"
        )
    by_run = {
        run: {home: pipeline_days(result) for home, result in runs[run].items()}
        for run in RUNS
    }
    analyses: dict[str, Any] = {}
    tables: dict[str, Any] = {}
    all_pairs: dict[str, dict[str, Pairs]] = {}
    for name, (run, keep_silent) in ANALYSES.items():
        pairs, table = _analysis(by_run[run], mats, protocol, keep_silent)
        all_pairs[name], tables[name] = pairs, table
        analyses[name] = agreement(table, protocol)
    primary, primary_pairs = analyses[PRIMARY], all_pairs[PRIMARY]
    verdicts = {home: mat_verdicts(mat) for home, mat in mats.items()}
    alignment = {}
    for hours, homes in sorted(shifted.items()):
        _, table = _analysis(by_run[RUN_OFF], homes, protocol, keep_silent=False)
        result = agreement(table, protocol)
        alignment[f"{hours:+g}"] = {
            "homes": result["homes"],
            SLEEP: {
                "spearman": result[SLEEP]["spearman"],
                "mean_difference": result[SLEEP]["mean_difference"],
            },
        }
    homes: dict[str, Any] = {}
    for home in sorted(mats):
        mat = mats[home]
        off_days = by_run[RUN_OFF].get(home, ())
        homes[home] = {
            "mat_records": mat.records,
            "mat_days": len(mat.days),
            "mat_observed_days": len(mat.observed),
            "staged_asleep_share": mat.staged_asleep_share(),
            "monitored_days": len(off_days),
            "usable_days": {
                run: sum(1 for d in by_run[run].get(home, ()) if d.usable)
                for run in RUNS
            },
            "silent_matched_days": len(matched_days(off_days, mat, keep_silent=True))
            - len(matched_days(off_days, mat, keep_silent=False)),
            **{name: tables[name].get(home) for name in ANALYSES},
        }
    return {
        "homes": homes,
        "analyses": analyses,
        "criteria": {
            "C1_the_pipeline_follows_the_mat": {
                "estimand": "E1",
                "analysis": PRIMARY,
                "estimate": primary[SLEEP]["spearman"],
                "margin": protocol.tracking_margin,
                "reading": tracking_reading(
                    primary[SLEEP]["spearman"], protocol.tracking_margin
                ),
            },
            "C2_the_pipeline_agrees_in_level": {
                "estimand": "E2",
                "analysis": PRIMARY,
                "estimate": primary[SLEEP]["mean_difference"],
                "margin": protocol.level_margin_hours,
                "reading": level_reading(
                    primary[SLEEP]["mean_difference"], protocol.level_margin_hours
                ),
            },
        },
        "E6_deviations": deviations(by_run[RUN_OFF], primary_pairs, verdicts, protocol),
        "E7_alerts": alerts_beside_the_mat(
            by_run[RUN_OFF], mats, primary_pairs, verdicts
        ),
        "E8_silent_days": silent_days(by_run[RUN_OFF], mats),
        "E10_alignment": alignment,
        "E11_clocks": hourly_alignment(
            runs[RUN_OFF], mats, primary_pairs, primary["homes"], protocol
        ),
        "E12_staged_asleep": {
            "share": STAGED_ASLEEP_SHARE,
            "left_out": list(STAGED_AWAKE_HOMES),
            **agreement(tables[PRIMARY], protocol, leave_out=STAGED_AWAKE_HOMES),
        },
    }


def _interval(
    label: str, estimate: Mapping[str, Any], unit: str = "household"
) -> ReportedInterval | None:
    interval = estimate.get("interval")
    if not interval:
        return None
    return ReportedInterval(
        label=label,
        estimate=estimate["estimate"],
        low=float(interval["low"]),
        high=float(interval["high"]),
        confidence=float(interval["confidence"]),
        method=str(interval["method"]),
        unit=unit,
        n=int(estimate["homes"]),
    )


def intervals_of(results: Mapping[str, Any]) -> list[ReportedInterval]:
    """Every interval the TIHM record reports, ready to quote."""
    out: list[ReportedInterval | None] = []
    for name in ANALYSES:
        for reference in REFERENCES:
            entry = results["analyses"][name][reference]
            out.append(_interval(f"{name}: {reference}: spearman", entry["spearman"]))
            out.append(_interval(f"{name}: {reference}: pearson", entry["pearson"]))
            out.append(
                _interval(
                    f"{name}: {reference}: mean difference, hours",
                    entry["mean_difference"],
                )
            )
    out.append(
        _interval("E6: deviations: spearman", results["E6_deviations"]["spearman"])
    )
    for reference in REFERENCES:
        entry = results["E12_staged_asleep"][reference]
        out.append(_interval(f"E12: {reference}: spearman", entry["spearman"]))
        out.append(
            _interval(
                f"E12: {reference}: mean difference, hours", entry["mean_difference"]
            )
        )
    return [item for item in out if item is not None]


# ----------------------------------------------------------------------------
# The simulated reference
# ----------------------------------------------------------------------------
def true_sleep_hours(truth: GroundTruth, zone: tzinfo) -> dict[date, float]:
    """The hours of each local day the simulator's resident spent asleep.

    Durations are elapsed time, so a night the clocks change on is given the
    hours it had.
    """
    hours: dict[date, float] = defaultdict(float)
    for episode in truth.episodes:
        if episode.state is not BehaviouralState.SLEEPING:
            continue
        start = episode.start.astimezone(timezone.utc)
        end = episode.end.astimezone(timezone.utc)
        while start < end:
            local = start.astimezone(zone)
            midnight = datetime.combine(
                local.date() + timedelta(days=1), datetime.min.time(), zone
            ).astimezone(timezone.utc)
            stop = min(end, midnight)
            hours[local.date()] += (stop - start).total_seconds() / 3600.0
            start = stop
    return dict(hours)


@dataclass(frozen=True)
class SimulatedHome:
    """One simulated home's matched days: the pipeline's value and the truth."""

    seed: int
    days: tuple[date, ...]
    pipeline: tuple[float, ...]
    truth: tuple[float, ...]
    usable_days: int
    closed_days: int
    closes: tuple[datetime, ...] = ()


def run_simulated_home(arguments: tuple[int, SleepMatProtocol]) -> SimulatedHome:
    """Simulate one home, run the pipeline and match its days to the truth."""
    seed, protocol = arguments
    sensors = event_sensors()
    result = simulate(HouseholdConfig(days=protocol.sim_days, seed=seed))
    delivered, _ = degrade(result.observations_for(sensors), DegradationConfig())
    config = PipelineConfig(
        tz=result.config.tz, step=timedelta(minutes=protocol.sim_step_minutes)
    )
    pipeline = BehaviouralSensingPipeline(
        result.registry.subset(sensors), config=config
    )
    steps = pipeline.run(delivered)
    steps.extend(pipeline.close(result.end))
    summaries = daily_summaries(steps)
    truth = true_sleep_hours(result.truth, result.config.tz)
    first = protocol.sim_start()
    last = first + timedelta(days=protocol.sim_days - 1)
    usable = [
        s
        for s in summaries
        if s.is_usable(config.min_day_coverage, config.min_day_observed)
    ]
    chosen = [s for s in usable if first < s.day < last]
    return SimulatedHome(
        seed=seed,
        days=tuple(s.day for s in chosen),
        pipeline=tuple(float(s.hours_in(BehaviouralState.SLEEPING)) for s in chosen),
        truth=tuple(float(truth.get(s.day, 0.0)) for s in chosen),
        usable_days=len(usable),
        closed_days=len(summaries),
        closes=tuple(step.at for step in steps if step.day_closed is not None),
    )


def run_simulated_homes(
    protocol: SleepMatProtocol,
    *,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> list[SimulatedHome]:
    """Every simulated home of the reference, in the order of its seeds."""
    tasks = [(seed, protocol) for seed in protocol.sim_seeds()]
    done: list[SimulatedHome] = []
    if jobs <= 1:
        for task in tasks:
            done.append(run_simulated_home(task))
            if progress is not None:
                progress(len(done), len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for home in pool.map(run_simulated_home, tasks):
                done.append(home)
                if progress is not None:
                    progress(len(done), len(tasks))
    return done


def score_simulated(
    homes: Sequence[SimulatedHome], protocol: SleepMatProtocol
) -> dict[str, Any]:
    """S1: E1, E2, E4 and E5 against the simulator's true hours of sleep."""
    table: dict[str, Any] = {}
    for home in homes:
        truth = np.array(home.truth, dtype=float)
        pairs = Pairs(
            home=str(home.seed),
            days=home.days,
            pipeline=np.array(home.pipeline, dtype=float),
            references={SLEEP: truth, IN_BED: truth},
        )
        row = home_statistics(pairs, protocol)
        table[str(home.seed)] = {
            "matched_days": row["matched_days"],
            "included": row["included"],
            "usable_days": home.usable_days,
            "closed_days": home.closed_days,
            SLEEP: {
                **row[SLEEP],
                "sd_truth": float(truth.std(ddof=1)) if truth.size >= 2 else None,
            },
        }
    included = [seed for seed, row in table.items() if row["included"]]
    rows = [table[seed][SLEEP] for seed in included]
    means = [r["mean_difference"] for r in rows]
    variances = [r["variance_of_differences"] for r in rows]
    return {
        "homes": table,
        "included": len(included),
        SLEEP: {
            "spearman": mean_over_homes(
                [r["spearman"] for r in rows], protocol.confidence
            ),
            "pearson": mean_over_homes(
                [r["pearson"] for r in rows], protocol.confidence
            ),
            "homes_with_constant_values": [
                seed for seed in included if table[seed][SLEEP]["constant"]
            ],
            "mean_difference": mean_over_homes(means, protocol.confidence),
            "limits_of_agreement": limits_of_agreement(means, variances),
            "median_within_home_sd_of_truth": (
                float(np.median([r["sd_truth"] for r in rows])) if rows else None
            ),
        },
    }


# ----------------------------------------------------------------------------
# The records
# ----------------------------------------------------------------------------
def tihm_record(
    runs: Mapping[str, Mapping[str, HouseholdRun]],
    mats: Mapping[str, MatHome],
    shifted: Mapping[float, Mapping[str, MatHome]],
    protocol: SleepMatProtocol,
    tihm: TihmProtocol,
    published: Mapping[str, Mapping[str, Any]],
    silent_home: Mapping[str, Any],
    *,
    protocol_sha256: str,
    inputs: Sequence[InputArtifact] = (),
    code_changed: Mapping[str, Sequence[str]] | None = None,
    between: Sequence[str] = (),
) -> ExperimentRecord:
    """The record of the comparison on TIHM.

    Nothing is scored unless both runs reproduce the records they must.
    """
    check = {
        RUN_OFF: check_published(runs[RUN_OFF], published),
        RUN_RULE: check_rule(
            runs[RUN_RULE], runs[RUN_OFF], silent_home, f"h{protocol.rule_hours:g}"
        ),
    }
    results = {"check": check, **score(runs, mats, shifted, protocol)}
    return ExperimentRecord(
        experiment=EXPERIMENT,
        configuration={
            "protocol_sha256": protocol.sha256(),
            "protocol_file_sha256": protocol_sha256,
            "tihm_protocol_sha256": tihm.sha256(),
            "analyses": {name: list(spec) for name, spec in ANALYSES.items()},
            "timezone": TIHM_TIMEZONE,
            "code_changed_since_the_freeze": {
                group: list(names) for group, names in (code_changed or {}).items()
            },
            "between_the_freeze_and_the_run": list(between),
            "result_schema": RESULT_SCHEMA,
        },
        inference=ONLINE,
        evidence=EvidenceSummary.online(
            moment
            for group in runs.values()
            for result in group.values()
            for moment in result.closes
        ),
        seeds=[],
        results=results,
        data_source="tihm",
        inputs=list(inputs),
        models=[
            ModelRecord(
                name=f"pipeline_{run}",
                configuration={
                    "step_minutes": protocol.tihm_step_minutes,
                    "home_silence_horizon_hours": (
                        None if run == RUN_OFF else protocol.rule_hours
                    ),
                },
            )
            for run in RUNS
        ],
        household_metrics={
            name: {
                home: {
                    "matched_days": row[name]["matched_days"],
                    "included": row[name]["included"],
                    **{
                        f"{reference}_{key}": row[name][reference][key]
                        for reference in REFERENCES
                        for key in ("spearman", "pearson", "mean_difference")
                    },
                }
                for home, row in results["homes"].items()
                if row.get(name) is not None
            }
            for name in ANALYSES
        },
        intervals=intervals_of(results),
        notes=[
            "A measurement on a real cohort under a protocol frozen before any "
            "value of the pipeline was set beside any record of the mat. "
            "Nothing in the pipeline was changed or fitted.",
            "The mat is the reference because the pipeline never sees it. Its "
            "stages are the device's own and are not validated here.",
            "Per-home summaries only: no day's values are recorded, and nothing "
            "of the dataset is redistributed.",
            "TIHM is by Palermo et al., Scientific Data 10, 606 (2023), under CC "
            "BY 4.0. Surrey and Borders Partnership NHS Foundation Trust and "
            "Howz are acknowledged, as the dataset asks.",
        ],
    )


def simulated_record(
    homes: Sequence[SimulatedHome],
    protocol: SleepMatProtocol,
    *,
    protocol_sha256: str,
    inputs: Sequence[InputArtifact] = (),
) -> ExperimentRecord:
    """The record of the simulated reference."""
    results = score_simulated(homes, protocol)
    return ExperimentRecord(
        experiment=SIMULATED_EXPERIMENT,
        configuration={
            "protocol_sha256": protocol.sha256(),
            "protocol_file_sha256": protocol_sha256,
            "simulated_reference": protocol.to_dict()["simulated_reference"],
            "result_schema": SIMULATED_SCHEMA,
        },
        inference=ONLINE,
        evidence=EvidenceSummary.online(
            moment for home in homes for moment in home.closes
        ),
        seeds=list(protocol.sim_seeds()),
        results=results,
        data_source="simulator",
        sensor_subset=list(event_sensors()),
        inputs=list(inputs),
        intervals=[
            item
            for item in (
                _interval("S1: spearman", results[SLEEP]["spearman"], "seed"),
                _interval("S1: pearson", results[SLEEP]["pearson"], "seed"),
                _interval(
                    "S1: mean difference, hours",
                    results[SLEEP]["mean_difference"],
                    "seed",
                ),
            )
            if item is not None
        ],
        notes=[
            "A reference, not a test: what the estimands give against the "
            "simulator's true hours of sleep, where the observation model is "
            "the one that made the data.",
        ],
    )
