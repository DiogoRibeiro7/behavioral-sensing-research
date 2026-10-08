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
from datetime import date, datetime, timedelta, tzinfo
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy import stats

from ..baseline.adaptive import AdaptiveBaseline, BaselineConfig, ChangeKind
from ..evaluation import ExperimentRecord, InputArtifact, ModelRecord
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
    ASLEEP_STATES,
    CONDITIONS,
    DOES_NOT_AGREE,
    DOES_NOT_FOLLOW,
    EXPERIMENT,
    FOLLOWS,
    IN_BED,
    INCONCLUSIVE,
    MAT_COLUMNS,
    MAT_FILE,
    MAT_STATES,
    NOT_ESTIMABLE,
    OFF,
    REFERENCES,
    RESULT_SCHEMA,
    RULE,
    SIMULATED_EXPERIMENT,
    SIMULATED_SCHEMA,
    SLEEP,
    TRACKED_FEATURE,
    SleepMatProtocol,
    event_sensors,
)
from .tihm_experiment import HouseholdRun, run_household
from .tihm_protocol import TihmProtocol

#: The pipeline's name for the state compared.
SLEEPING = BehaviouralState.SLEEPING.value

#: Fewer included homes than this and a criterion is not estimable.
MIN_HOMES = 3


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
    """

    home: str
    days: Mapping[date, MatDay]
    observed: frozenset[date]


def observed_days(days: Iterable[date]) -> frozenset[date]:
    """The days whose neighbours on both sides are also among *days*."""
    present = set(days)
    one = timedelta(days=1)
    return frozenset(d for d in present if d - one in present and d + one in present)


def read_mat(path: Path, shift_hours: float = 0.0) -> dict[str, MatHome]:
    """Read the mat's file into each home's days.

    Each record is a minute. Its day is the local date of its timestamp, read
    as a local clock time and moved by *shift_hours* first. A record whose
    state is not one the device reports is refused, as is a repeated minute.
    """
    shift = timedelta(hours=shift_hours)
    asleep: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))
    in_bed: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))
    seen: set[tuple[str, str]] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MAT_COLUMNS:
            raise ValueError(
                f"{MAT_FILE} has columns {reader.fieldnames}, not {list(MAT_COLUMNS)}"
            )
        for row in reader:
            home, stamp, state = row["patient_id"], row["date"], row["state"]
            if state not in MAT_STATES:
                raise ValueError(f"{MAT_FILE} has an unknown state {state!r}")
            if (home, stamp) in seen:
                raise ValueError(f"{MAT_FILE} repeats the minute {stamp} of {home}")
            seen.add((home, stamp))
            day = (datetime.fromisoformat(stamp) + shift).date()
            in_bed[home][day] += 1
            if state in ASLEEP_STATES:
                asleep[home][day] += 1
    homes: dict[str, MatHome] = {}
    for home in sorted(in_bed):
        days = {
            day: MatDay(sleep=asleep[home][day] / 60.0, in_bed=minutes / 60.0)
            for day, minutes in sorted(in_bed[home].items())
        }
        homes[home] = MatHome(home=home, days=days, observed=observed_days(days))
    return homes


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
        return self.usable and self.kind not in ("", ChangeKind.INSUFFICIENT_DATA.value)


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


def pairs_of(home: str, days: Sequence[PipelineDay], mat: MatHome | None) -> Pairs:
    """The usable days of a home that the mat observed."""
    chosen = (
        [] if mat is None else [d for d in days if d.usable and d.day in mat.observed]
    )
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
    """Everything an estimand reads from one home, against each reference."""
    included = pairs.size >= protocol.min_matched_days
    out: dict[str, Any] = {"matched_days": pairs.size, "included": included}
    for reference, values in pairs.references.items():
        difference = pairs.pipeline - values
        out[reference] = {
            "spearman": correlation(pairs.pipeline, values, "spearman"),
            "pearson": correlation(pairs.pipeline, values, "pearson"),
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
    table: Mapping[str, Mapping[str, Any]], protocol: SleepMatProtocol
) -> dict[str, Any]:
    """E1 to E5 over the included homes of one condition."""
    included = [home for home, row in sorted(table.items()) if row["included"]]
    out: dict[str, Any] = {"homes": included}
    for reference in REFERENCES:
        rows = [table[home][reference] for home in included]
        spearman = [r["spearman"] for r in rows if r["spearman"] is not None]
        pearson = [r["pearson"] for r in rows if r["pearson"] is not None]
        means = [r["mean_difference"] for r in rows]
        variances = [r["variance_of_differences"] for r in rows]
        out[reference] = {
            "spearman": mean_over_homes(spearman, protocol.confidence),
            "pearson": mean_over_homes(pearson, protocol.confidence),
            "homes_without_a_correlation": [
                home for home in included if table[home][reference]["spearman"] is None
            ],
            "mean_difference": mean_over_homes(means, protocol.confidence),
            "limits_of_agreement": limits_of_agreement(means, variances),
        }
    return out


# ----------------------------------------------------------------------------
# E6 to E8
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
    mats: Mapping[str, MatHome],
    protocol: SleepMatProtocol,
) -> dict[str, Any]:
    """E6: the deviations each source's personal baseline gives the same days."""
    threshold = protocol.alert_threshold
    correlations: list[float] = []
    per_home: dict[str, Any] = {}
    flagged = same_sign = also_past = 0
    for home in sorted(days):
        if home not in mats:
            continue
        mat = mats[home]
        verdicts = mat_verdicts(mat)
        both = [
            (d.deviation, verdicts[d.day][1])
            for d in days[home]
            if d.has_verdict
            and d.day in mat.observed
            and verdicts[d.day][0] != ChangeKind.INSUFFICIENT_DATA.value
        ]
        x = np.array([a for a, _ in both], dtype=float)
        y = np.array([b for _, b in both], dtype=float)
        rho = (
            correlation(x, y, "spearman")
            if len(both) >= protocol.min_matched_days
            else None
        )
        if rho is not None:
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
) -> dict[str, Any]:
    """E7: the alerts about sleep, and the mat's sleep on the days they came."""
    total = matched = same = opposite = at_median = 0
    per_home: dict[str, Any] = {}
    for home in sorted(days):
        if home not in mats:
            continue
        mat, home_pairs = mats[home], pairs[home]
        centre = (
            float(np.median(home_pairs.references[SLEEP])) if home_pairs.size else None
        )
        counts = {"alerts": 0, "on_matched_days": 0, "same_side": 0, "opposite": 0}
        matched_days = set(home_pairs.days)
        for d in days[home]:
            if not d.alerts:
                continue
            total += d.alerts
            counts["alerts"] += d.alerts
            if d.day not in matched_days or centre is None:
                continue
            matched += d.alerts
            counts["on_matched_days"] += d.alerts
            side = (mat.days[d.day].sleep - centre) * np.sign(d.deviation)
            if side > 0:
                same += d.alerts
                counts["same_side"] += d.alerts
            elif side < 0:
                opposite += d.alerts
                counts["opposite"] += d.alerts
            else:
                at_median += d.alerts
        per_home[home] = counts
    return {
        "alerts": total,
        "on_matched_days": matched,
        "same_side": same,
        "opposite": opposite,
        "at_the_median": at_median,
        "homes": per_home,
    }


def silent_days(
    days: Mapping[str, Sequence[PipelineDay]], mats: Mapping[str, MatHome]
) -> dict[str, Any]:
    """E8: days with no activity record, and what the mat recorded on them."""
    silent = [
        (home, d)
        for home in sorted(days)
        if home in mats
        for d in days[home]
        if not d.events
    ]
    observed = [(home, d) for home, d in silent if d.day in mats[home].observed]
    return {
        "silent_days": len(silent),
        "mat_observed": len(observed),
        "homes": sorted({home for home, _ in silent}),
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
    }


# ----------------------------------------------------------------------------
# Running the pipeline on the mat homes
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
    """Run each mat home under each condition; refuse one the contract refuses."""
    loaded = []
    for name in homes:
        data = adapter.load(name)
        if not validate_household(data, tihm.mapping).ok:
            raise ValueError(f"the contract refuses mat home {name!r}")
        loaded.append(data)
    hours = {OFF: None, RULE: protocol.rule_hours}
    tasks = [(data, tihm, hours[c]) for c in CONDITIONS for data in loaded]
    done: list[HouseholdRun] = []
    if jobs <= 1:
        for task in tasks:
            done.append(_run(task))
            if progress is not None:
                progress(len(done), len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for run in pool.map(_run, tasks):
                done.append(run)
                if progress is not None:
                    progress(len(done), len(tasks))
    size = len(loaded)
    return {
        condition: {run.household: run for run in done[k * size : (k + 1) * size]}
        for k, condition in enumerate(CONDITIONS)
    }


def check_published(
    runs: Mapping[str, HouseholdRun], published: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Refuse a run with the rule off that is not the published one, home by home."""
    differing = []
    for home, run in sorted(runs.items()):
        here = (len(run.days), sum(len(day.alerts) for day in run.days))
        row = published.get(home)
        there = (
            None
            if row is None
            else (int(row["monitored_days"]), int(row["behavioural_alerts"]))
        )
        if here != there:
            differing.append(f"{home}: {here} against {there}")
    if differing:
        raise ValueError(
            "with the rule off the run does not give the published record's "
            "monitored days and behavioural alerts: "
            + "; ".join(differing)
            + "; nothing is reported"
        )
    return {
        "homes": len(runs),
        "monitored_days": sum(len(run.days) for run in runs.values()),
        "behavioural_alerts": sum(
            len(day.alerts) for run in runs.values() for day in run.days
        ),
        "reproduced": True,
    }


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def _homes_table(
    days: Mapping[str, Sequence[PipelineDay]],
    mats: Mapping[str, MatHome],
    protocol: SleepMatProtocol,
) -> tuple[dict[str, Pairs], dict[str, Any]]:
    pairs = {home: pairs_of(home, days[home], mats.get(home)) for home in sorted(days)}
    table = {home: home_statistics(pairs[home], protocol) for home in sorted(days)}
    return pairs, table


def score(
    runs: Mapping[str, Mapping[str, HouseholdRun]],
    mats: Mapping[str, MatHome],
    shifted: Mapping[float, Mapping[str, MatHome]],
    protocol: SleepMatProtocol,
) -> dict[str, Any]:
    """Every estimand and criterion, from the runs and the mat."""
    by_condition = {
        condition: {home: pipeline_days(run) for home, run in runs[condition].items()}
        for condition in CONDITIONS
    }
    conditions: dict[str, Any] = {}
    tables: dict[str, Any] = {}
    off_pairs: dict[str, Pairs] = {}
    for condition in CONDITIONS:
        pairs, table = _homes_table(by_condition[condition], mats, protocol)
        tables[condition] = table
        conditions[condition] = agreement(table, protocol)
        if condition == OFF:
            off_pairs = pairs
    off = conditions[OFF]
    alignment = {}
    for hours, homes in sorted(shifted.items()):
        _, table = _homes_table(by_condition[OFF], homes, protocol)
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
        days_off = by_condition[OFF].get(home, ())
        homes[home] = {
            "mat_days": len(mat.days),
            "mat_observed_days": len(mat.observed),
            "monitored_days": len(days_off),
            "usable_days": {
                condition: sum(
                    1 for d in by_condition[condition].get(home, ()) if d.usable
                )
                for condition in CONDITIONS
            },
            **{condition: tables[condition].get(home) for condition in CONDITIONS},
        }
    return {
        "homes": homes,
        "conditions": conditions,
        "criteria": {
            "C1_the_pipeline_follows_the_mat": {
                "estimand": "E1",
                "estimate": off[SLEEP]["spearman"],
                "margin": protocol.tracking_margin,
                "reading": tracking_reading(
                    off[SLEEP]["spearman"], protocol.tracking_margin
                ),
            },
            "C2_the_pipeline_agrees_in_level": {
                "estimand": "E2",
                "estimate": off[SLEEP]["mean_difference"],
                "margin": protocol.level_margin_hours,
                "reading": level_reading(
                    off[SLEEP]["mean_difference"], protocol.level_margin_hours
                ),
            },
        },
        "E6_deviations": deviations(by_condition[OFF], mats, protocol),
        "E7_alerts": alerts_beside_the_mat(by_condition[OFF], mats, off_pairs),
        "E8_silent_days": silent_days(by_condition[OFF], mats),
        "E10_alignment": alignment,
    }


# ----------------------------------------------------------------------------
# The simulated reference
# ----------------------------------------------------------------------------
def true_sleep_hours(truth: GroundTruth, zone: tzinfo) -> dict[date, float]:
    """The hours of each local day the simulator's resident spent asleep."""
    hours: dict[date, float] = defaultdict(float)
    for episode in truth.episodes:
        if episode.state is not BehaviouralState.SLEEPING:
            continue
        start, end = episode.start.astimezone(zone), episode.end.astimezone(zone)
        while start < end:
            midnight = datetime.combine(
                start.date() + timedelta(days=1), datetime.min.time(), zone
            )
            stop = min(end, midnight)
            hours[start.date()] += (stop - start).total_seconds() / 3600.0
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


def run_simulated_home(
    arguments: tuple[int, SleepMatProtocol],
) -> SimulatedHome:
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
    zone = result.config.tz
    truth = true_sleep_hours(result.truth, zone)
    first = protocol.sim_start()
    last = first + timedelta(days=protocol.sim_days - 1)
    chosen = [
        s
        for s in summaries
        if s.is_usable(config.min_day_coverage, config.min_day_observed)
        and first < s.day < last
    ]
    return SimulatedHome(
        seed=seed,
        days=tuple(s.day for s in chosen),
        pipeline=tuple(float(s.hours_in(BehaviouralState.SLEEPING)) for s in chosen),
        truth=tuple(float(truth.get(s.day, 0.0)) for s in chosen),
        usable_days=sum(
            1
            for s in summaries
            if s.is_usable(config.min_day_coverage, config.min_day_observed)
        ),
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
        pairs = Pairs(
            home=str(home.seed),
            days=home.days,
            pipeline=np.array(home.pipeline, dtype=float),
            references={SLEEP: np.array(home.truth, dtype=float)},
        )
        included = pairs.size >= protocol.min_matched_days
        difference = pairs.pipeline - pairs.references[SLEEP]
        table[str(home.seed)] = {
            "matched_days": pairs.size,
            "usable_days": home.usable_days,
            "closed_days": home.closed_days,
            "included": included,
            SLEEP: {
                "spearman": correlation(
                    pairs.pipeline, pairs.references[SLEEP], "spearman"
                ),
                "pearson": correlation(
                    pairs.pipeline, pairs.references[SLEEP], "pearson"
                ),
                "mean_difference": float(difference.mean()) if pairs.size else None,
                "variance_of_differences": (
                    float(difference.var(ddof=1)) if pairs.size >= 2 else None
                ),
                "median_pipeline": (
                    float(np.median(pairs.pipeline)) if pairs.size else None
                ),
                "median_truth": (
                    float(np.median(pairs.references[SLEEP])) if pairs.size else None
                ),
                "sd_truth": (
                    float(pairs.references[SLEEP].std(ddof=1))
                    if pairs.size >= 2
                    else None
                ),
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
                [r["spearman"] for r in rows if r["spearman"] is not None],
                protocol.confidence,
            ),
            "pearson": mean_over_homes(
                [r["pearson"] for r in rows if r["pearson"] is not None],
                protocol.confidence,
            ),
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
@dataclass(frozen=True)
class SleepMatResult:
    """The records written, and where."""

    record: ExperimentRecord
    path: Path | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def tihm_record(
    runs: Mapping[str, Mapping[str, HouseholdRun]],
    mats: Mapping[str, MatHome],
    shifted: Mapping[float, Mapping[str, MatHome]],
    protocol: SleepMatProtocol,
    tihm: TihmProtocol,
    published: Mapping[str, Mapping[str, Any]],
    *,
    protocol_sha256: str,
    inputs: Sequence[InputArtifact] = (),
) -> ExperimentRecord:
    """The record of the comparison on TIHM."""
    check = check_published(runs[OFF], published)
    results = {"check": check, **score(runs, mats, shifted, protocol)}
    return ExperimentRecord(
        experiment=EXPERIMENT,
        configuration={
            "protocol_sha256": protocol.sha256(),
            "protocol_file_sha256": protocol_sha256,
            "tihm_protocol_sha256": tihm.sha256(),
            "conditions": list(CONDITIONS),
            "timezone": TIHM_TIMEZONE,
            "result_schema": RESULT_SCHEMA,
        },
        inference=ONLINE,
        evidence=EvidenceSummary.online(
            moment
            for group in runs.values()
            for run in group.values()
            for moment in run.closes
        ),
        seeds=[],
        results=results,
        data_source="tihm",
        inputs=list(inputs),
        models=[
            ModelRecord(
                name=f"pipeline_{condition}",
                configuration={
                    "step_minutes": protocol.tihm_step_minutes,
                    "home_silence_horizon_hours": (
                        None if condition == OFF else protocol.rule_hours
                    ),
                },
            )
            for condition in CONDITIONS
        ],
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
        results=score_simulated(homes, protocol),
        data_source="simulator",
        sensor_subset=list(event_sensors()),
        inputs=list(inputs),
        notes=[
            "A reference, not a test: what the estimands give against the "
            "simulator's true hours of sleep, where the observation model is "
            "the one that made the data.",
        ],
    )
