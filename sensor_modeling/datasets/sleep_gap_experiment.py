"""Where the pipeline's sleep parts from the sleep mat, minute by minute.

What is computed, on which homes and days, is declared in
:mod:`sensor_modeling.datasets.sleep_gap_plan` and frozen before anything is
computed. This module runs the pipeline on the mat homes a second time to keep
each step's belief, spreads every belief over the minutes its interval covers,
sets each minute beside what the mat recorded of it and what the home's
sensors had last reported, and computes the plan's estimands.

TIHM is by Palermo et al., *Scientific Data* 10, 606 (2023), under CC BY 4.0.
Surrey and Borders Partnership NHS Foundation Trust and Howz are acknowledged,
as the dataset asks. Nothing of the dataset is redistributed: the record holds
per-home summaries, never a day's or a minute's values.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from ..evaluation import ExperimentRecord, InputArtifact, ModelRecord, ReportedInterval
from ..external.canonical import to_canonical
from ..external.contract import DatasetAdapter, HouseholdData
from ..external.tihm import TIMEZONE
from ..external.validation import validate_household
from ..fusion import ONLINE, EvidenceSummary
from ..online.pipeline import BehaviouralSensingPipeline, PipelineConfig, PipelineStep
from ..states.ontology import BehaviouralState
from .sleep_gap_plan import (
    ASLEEP,
    AWAKE_ON_THE_MAT,
    CLOCKS,
    COMPOSITION,
    DAY,
    DAY_HOURS,
    DISTANCE_BINS,
    EVENTS,
    EXITS,
    EXPERIMENT,
    MAT_CLASSES,
    MAT_IN_BED,
    MAT_SLEEP,
    NIGHT,
    NOTHING_YET,
    OFF,
    OFF_THE_MAT,
    ON_THE_MAT,
    PIPELINE,
    PRIMARY_CLOCK,
    QUIET,
    QUIET_AFTER_MINUTES,
    RESULT_SCHEMA,
    SINCE_BINS,
    SleepGapPlan,
)
from .sleep_mat_experiment import (
    MatHome,
    Pairs,
    check_published,
    check_rule,
    correlation,
    home_statistics,
    mean_over_homes,
    pairs_of,
    pipeline_days,
    silenced_days,
)
from .sleep_mat_protocol import (
    ASLEEP_STATES,
    MAT_COLUMNS,
    MAT_FILE,
    MAT_STATES,
    MAT_TIME_FORMAT,
    PRIMARY,
    RUN_OFF,
    RUN_RULE,
    SLEEP,
    STAGED_AWAKE_HOMES,
    SleepMatProtocol,
)
from .tihm_experiment import HouseholdRun
from .tihm_protocol import TihmProtocol

MINUTES = 24 * 60
SLEEPING = BehaviouralState.SLEEPING.value
_CLASS_INDEX = {name: k for k, name in enumerate(MAT_CLASSES)}
_ASLEEP = _CLASS_INDEX[ASLEEP]
_AWAKE = _CLASS_INDEX[AWAKE_ON_THE_MAT]
_OFF = _CLASS_INDEX[OFF_THE_MAT]

#: The daily quantities D5 sets beside each other.
DAILY = (PIPELINE, ON_THE_MAT, OFF, MAT_SLEEP, MAT_IN_BED, QUIET, EVENTS)
#: The pairs D5 correlates, in the order the page lists them; those with a
#: part of the pipeline's sleep get a swapped-mask reference.
CORRELATED = (
    (MAT_SLEEP, PIPELINE),
    (MAT_SLEEP, ON_THE_MAT),
    (MAT_SLEEP, OFF),
    (MAT_IN_BED, PIPELINE),
    (MAT_IN_BED, ON_THE_MAT),
    (MAT_IN_BED, OFF),
    (PIPELINE, QUIET),
    (PIPELINE, EVENTS),
)
REFERENCED = (
    (MAT_SLEEP, ON_THE_MAT),
    (MAT_SLEEP, OFF),
    (MAT_IN_BED, ON_THE_MAT),
    (MAT_IN_BED, OFF),
)
PARTS = (ON_THE_MAT, OFF)
SPREAD = (PIPELINE, ON_THE_MAT, OFF)
#: The groupings of the minutes with no mat record, D4's and D7's.
GROUPS = ("sensor", "since", "joint", "period", "exit", "distance", "distance_period")


def pair_name(left: str, right: str) -> str:
    """The record's name for a correlation of *left* with *right*."""
    return f"{left}~{right}"


# ----------------------------------------------------------------------------
# The mat, minute by minute
# ----------------------------------------------------------------------------
def read_mat_minutes(
    path: Path, homes: Sequence[str] | None = None
) -> dict[str, dict[datetime, str]]:
    """Each home's mat records as a stage by naive timestamp, as the file has them.

    A record whose state is not one the device reports is refused, as is a
    repeated minute.
    """
    wanted = None if homes is None else set(homes)
    out: dict[str, dict[datetime, str]] = defaultdict(dict)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MAT_COLUMNS:
            raise ValueError(f"{MAT_FILE} has columns {reader.fieldnames}")
        for row in reader:
            home, state = row["patient_id"], row["state"]
            if wanted is not None and home not in wanted:
                continue
            if state not in MAT_STATES:
                raise ValueError(f"{MAT_FILE} has an unknown state {state!r}")
            moment = datetime.strptime(row["date"], MAT_TIME_FORMAT)
            if moment in out[home]:
                raise ValueError(f"{MAT_FILE} repeats the minute {moment} of {home}")
            out[home][moment] = state
    return dict(out)


@dataclass(frozen=True)
class MatTimeline:
    """One home's mat under one clock, minute by minute over its whole file.

    Attributes
    ----------
    origin
        The naive local minute of the first entry.
    codes
        Each minute's class: asleep, awake on the mat, or no record.
    distance
        Minutes from each minute to the nearest mat record, earlier or later;
        zero on a record, infinite in a home with none.
    """

    origin: datetime
    codes: np.ndarray
    distance: np.ndarray

    def day(self, day: date) -> tuple[np.ndarray, np.ndarray]:
        """The classes and distances of *day*'s minutes, from 00:00 local."""
        first = int((datetime.combine(day, time()) - self.origin).total_seconds() // 60)
        codes = np.full(MINUTES, _OFF, dtype=np.int8)
        distance = np.full(MINUTES, np.inf)
        low, high = max(first, 0), min(first + MINUTES, self.codes.size)
        if low < high:
            codes[low - first : high - first] = self.codes[low:high]
            distance[low - first : high - first] = self.distance[low:high]
        return codes, distance


def mat_timeline(minutes: Mapping[datetime, str], shift_hours: float) -> MatTimeline:
    """A home's mat under a clock *shift_hours* later than its timestamps.

    A record stamped at a minute covers that minute; under a shift of *k*
    hours it is counted *k* hours after its stamp, as the sleep-mat
    comparison's shifted reading counts it.
    """
    shift = timedelta(hours=shift_hours)
    if not minutes:
        origin = datetime(1970, 1, 1)
        return MatTimeline(origin, np.zeros(0, dtype=np.int8), np.zeros(0))
    stamps = sorted(minutes)
    origin = datetime.combine((stamps[0] + shift).date() - timedelta(days=1), time())
    end = datetime.combine((stamps[-1] + shift).date() + timedelta(days=2), time())
    size = int((end - origin).total_seconds() // 60)
    codes = np.full(size, _OFF, dtype=np.int8)
    where = np.empty(len(stamps), dtype=np.int64)
    for k, stamp in enumerate(stamps):
        index = int((stamp + shift - origin).total_seconds() // 60)
        where[k] = index
        codes[index] = _ASLEEP if minutes[stamp] in ASLEEP_STATES else _AWAKE
    grid = np.arange(size)
    right = np.searchsorted(where, grid, side="left")
    after = np.where(
        right < where.size, where[np.minimum(right, where.size - 1)] - grid, np.inf
    )
    before = np.where(right > 0, grid - where[np.maximum(right - 1, 0)], np.inf)
    distance = np.minimum(after, before).astype(float)
    return MatTimeline(origin, codes, distance)


# ----------------------------------------------------------------------------
# The pipeline's beliefs, minute by minute
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class BeliefRun:
    """The second run of one home: each step's belief, and the days it closed.

    Attributes
    ----------
    home
        The dataset's identifier.
    states
        The ontology's states, in the order of each belief.
    steps
        Every step's moment and belief, in order.
    day_hours
        The hours each closed day's summary gives each state.
    observations
        The moment and sensor of every canonical observation, in order.
    """

    home: str
    states: tuple[str, ...]
    steps: tuple[tuple[datetime, np.ndarray], ...]
    day_hours: Mapping[date, Mapping[str, float]]
    observations: tuple[tuple[datetime, str], ...]


def run_beliefs(data: HouseholdData, tihm: TihmProtocol) -> BeliefRun:
    """Run one home through the contract and the pipeline at its defaults.

    The pipeline is built as the alert-burden run builds it, with the rule off,
    and every step's belief is kept.
    """
    canonical = to_canonical(data, tihm.mapping, source="tihm")
    recording = canonical.recording
    if data.timezone is None:  # pragma: no cover - to_canonical already refused it
        raise ValueError(f"household {data.household!r} has no timezone")
    zone = ZoneInfo(data.timezone)
    pipeline = BehaviouralSensingPipeline(
        recording.registry,
        config=PipelineConfig(tz=zone, step=timedelta(minutes=tihm.step_minutes)),
    )
    steps = pipeline.run(recording.observations)
    steps.extend(pipeline.close(recording.observations[-1].timestamp))
    return belief_run(
        data.household,
        steps,
        tuple((o.timestamp, o.sensor_id) for o in recording.observations),
    )


def belief_run(
    home: str,
    steps: Sequence[PipelineStep],
    observations: Sequence[tuple[datetime, str]],
) -> BeliefRun:
    """Keep every step's belief and every closed day's hours of a run."""
    states = tuple(state.value for state in steps[0].state.ontology.states)
    kept: list[tuple[datetime, np.ndarray]] = []
    day_hours: dict[date, dict[str, float]] = {}
    for step in steps:
        kept.append((step.at, np.asarray(step.state.belief, dtype=float)))
        summary = step.day_closed
        if summary is not None:
            day_hours[summary.day] = {
                state.value: float(hours) for state, hours in summary.hours.items()
            }
    return BeliefRun(
        home=home,
        states=states,
        steps=tuple(kept),
        day_hours=day_hours,
        observations=tuple(observations),
    )


@dataclass
class DayBeliefs:
    """One day's beliefs spread over its minutes.

    ``belief`` holds, for each minute and state, the belief times the seconds
    of the minute its interval covers; ``watched`` the seconds covered; and
    ``moment`` the seconds after local midnight of the step whose interval
    covers the minute's middle, not a number where none does.
    """

    belief: np.ndarray
    watched: np.ndarray
    moment: np.ndarray


def day_beliefs(
    run: BeliefRun, zone: tzinfo, max_interval: timedelta
) -> dict[date, DayBeliefs]:
    """Every day's beliefs, spread minute by minute as its summary counts them.

    A step's belief describes the interval since the step before. Only
    intervals between two steps of the same local day, and no longer than
    *max_interval*, count.
    """
    size = len(run.states)
    out: dict[date, DayBeliefs] = {}
    middles = 60.0 * np.arange(MINUTES) + 30.0
    for (before, _), (moment, belief) in zip(run.steps, run.steps[1:]):
        span = moment - before
        if span <= timedelta(0) or span > max_interval:
            continue
        start, end = before.astimezone(zone), moment.astimezone(zone)
        if start.date() != end.date():
            continue
        day = end.date()
        if day not in out:
            out[day] = DayBeliefs(
                np.zeros((MINUTES, size)),
                np.zeros(MINUTES),
                np.full(MINUTES, np.nan),
            )
        midnight = datetime.combine(day, time(), tzinfo=zone)
        low = (start - midnight).total_seconds()
        high = (end - midnight).total_seconds()
        first, last = int(low // 60), min(int(math.ceil(high / 60.0)), MINUTES)
        for minute in range(first, last):
            covered = min(high, 60.0 * (minute + 1)) - max(low, 60.0 * minute)
            if covered > 0.0:
                out[day].belief[minute] += belief * covered
                out[day].watched[minute] += covered
            if low < middles[minute] <= high:
                out[day].moment[minute] = high
    return out


@dataclass(frozen=True)
class Activations:
    """A home's activations as seconds since the epoch, and each one's sensor."""

    seconds: np.ndarray
    sensors: tuple[str, ...]

    @classmethod
    def of(cls, observations: Sequence[tuple[datetime, str]]) -> Activations:
        """From each observation's moment and sensor, in order."""
        return cls(
            np.array([moment.timestamp() for moment, _ in observations], dtype=float),
            tuple(sensor for _, sensor in observations),
        )

    def last(self, moments: np.ndarray) -> tuple[list[str], np.ndarray]:
        """The sensor of the latest activation at or before each moment, and the
        minutes since it; ``nothing_yet`` and infinite before the first."""
        index = np.searchsorted(self.seconds, moments, side="right") - 1
        sensors = [self.sensors[k] if k >= 0 else NOTHING_YET for k in index]
        since = np.where(
            index >= 0, (moments - self.seconds[np.maximum(index, 0)]) / 60.0, np.inf
        )
        return sensors, since


def since_bin(minutes: float) -> str:
    """The plan's name for a time since the last activation."""
    if math.isinf(minutes):
        return NOTHING_YET
    for name, low, high in SINCE_BINS:
        if minutes >= low and (high is None or minutes < high):
            return name
    raise ValueError(f"no bin for {minutes} minutes")


def distance_bin(minutes: float) -> str:
    """The plan's name for a time to the nearest mat record."""
    for name, low, high in DISTANCE_BINS:
        if minutes >= low and (high is None or minutes < high):
            return name
    raise ValueError(f"no bin for {minutes} minutes")


def period_of(minute: int) -> str:
    """Day or night, by the minute's local hour."""
    return DAY if DAY_HOURS[0] <= minute // 60 < DAY_HOURS[1] else NIGHT


# ----------------------------------------------------------------------------
# One home's summaries
# ----------------------------------------------------------------------------
@dataclass
class HomeGap:
    """What one home contributes to each estimand, under one clock."""

    days: int
    daily: dict[str, list[float]] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    hourly: dict[str, list[float]] = field(default_factory=dict)
    beliefs: dict[str, list[float] | None] = field(default_factory=dict)
    cells: dict[str, dict[str, dict[str, float | None]]] = field(default_factory=dict)
    short_nights: int = 0
    short_night_share: float | None = None
    sleeping: np.ndarray = field(default_factory=lambda: np.zeros((0, MINUTES)))
    occupied: np.ndarray = field(default_factory=lambda: np.zeros((0, MINUTES)))


def _cell(cell: Sequence[float], days: int) -> dict[str, float | None]:
    hours, minutes, sleeping, watched = cell
    return {
        "pipeline_sleep_hours": hours / days,
        "hours_with_no_record": minutes / 60.0 / days,
        "sleeping_belief": sleeping / watched if watched > 0.0 else None,
    }


def home_gap(
    run: BeliefRun,
    beliefs: Mapping[date, DayBeliefs],
    days: Sequence[date],
    timeline: MatTimeline,
    events: Mapping[date, int],
    activations: Activations,
    zone: tzinfo,
    plan: SleepGapPlan,
) -> HomeGap:
    """One home's estimands over its matched *days* under one clock."""
    sleep = run.states.index(SLEEPING)
    size = len(run.states)
    daily: dict[str, list[float]] = {name: [] for name in DAILY}
    parts: dict[str, list[float]] = {
        "pipeline_asleep": [],
        "pipeline_awake_on_the_mat": [],
        "mat_asleep_unwatched": [],
    }
    hourly = {
        name: np.zeros(24)
        for name in (*(f"pipeline_{c}" for c in MAT_CLASSES), MAT_SLEEP, MAT_IN_BED)
    }
    class_belief = {name: np.zeros(size) for name in MAT_CLASSES}
    class_watched = dict.fromkeys(MAT_CLASSES, 0.0)
    cells: dict[str, dict[str, list[float]]] = {group: {} for group in GROUPS}
    by_hour = np.arange(MINUTES) // 60
    middles = 60.0 * np.arange(MINUTES) + 30.0
    sleeping_rows, occupied_rows = [], []
    for day in days:
        spread = beliefs[day]
        codes, distance = timeline.day(day)
        midnight = datetime.combine(day, time(), tzinfo=zone).timestamp()
        moments = np.where(np.isnan(spread.moment), middles, spread.moment)
        sensors, since = activations.last(midnight + moments)
        _, quiet_since = activations.last(midnight + middles)
        sleeping = spread.belief[:, sleep]
        asleep, awake, off = codes == _ASLEEP, codes == _AWAKE, codes == _OFF
        on_mat = ~off
        hours = {
            ASLEEP: float(sleeping[asleep].sum()) / 3600.0,
            AWAKE_ON_THE_MAT: float(sleeping[awake].sum()) / 3600.0,
            OFF_THE_MAT: float(sleeping[off].sum()) / 3600.0,
        }
        parts["pipeline_asleep"].append(hours[ASLEEP])
        parts["pipeline_awake_on_the_mat"].append(hours[AWAKE_ON_THE_MAT])
        parts["mat_asleep_unwatched"].append(
            float((60.0 - spread.watched[asleep]).clip(min=0.0).sum()) / 3600.0
        )
        daily[PIPELINE].append(float(sleeping.sum()) / 3600.0)
        daily[ON_THE_MAT].append(hours[ASLEEP] + hours[AWAKE_ON_THE_MAT])
        daily[OFF].append(hours[OFF_THE_MAT])
        daily[MAT_SLEEP].append(float(asleep.sum()) / 60.0)
        daily[MAT_IN_BED].append(float(on_mat.sum()) / 60.0)
        daily[QUIET].append(float((quiet_since >= QUIET_AFTER_MINUTES).sum()) / 60.0)
        daily[EVENTS].append(float(events.get(day, 0)))
        sleeping_rows.append(sleeping)
        occupied_rows.append(on_mat)
        for name, chosen in (
            (ASLEEP, asleep),
            (AWAKE_ON_THE_MAT, awake),
            (OFF_THE_MAT, off),
        ):
            class_belief[name] += spread.belief[chosen].sum(axis=0)
            class_watched[name] += float(spread.watched[chosen].sum())
            hourly[f"pipeline_{name}"] += (
                np.bincount(by_hour[chosen], weights=sleeping[chosen], minlength=24)
                / 3600.0
            )
        hourly[MAT_SLEEP] += np.bincount(by_hour[asleep], minlength=24) / 60.0
        hourly[MAT_IN_BED] += np.bincount(by_hour[on_mat], minlength=24) / 60.0
        for minute in np.flatnonzero(off):
            since_name = since_bin(float(since[minute]))
            sensor = sensors[minute]
            period = period_of(int(minute))
            far = distance_bin(float(distance[minute]))
            keys = {
                "sensor": sensor,
                "since": since_name,
                "joint": f"{sensor}|{since_name}|{period}",
                "period": period,
                "exit": (
                    NOTHING_YET
                    if sensor == NOTHING_YET
                    else ("exit" if sensor in EXITS else "other")
                ),
                "distance": far,
                "distance_period": f"{far}|{period}",
            }
            for group, key in keys.items():
                cell = cells[group].setdefault(key, [0.0, 0.0, 0.0, 0.0])
                cell[0] += float(sleeping[minute]) / 3600.0
                cell[1] += 1.0
                cell[2] += float(sleeping[minute])
                cell[3] += float(spread.watched[minute])
    count = len(days)
    gap = HomeGap(days=count, daily=daily)
    means = {name: float(np.mean(values)) for name, values in daily.items()}
    means |= {name: float(np.mean(values)) for name, values in parts.items()}
    means["mat_sleep_not_counted"] = means[MAT_SLEEP] - means["pipeline_asleep"]
    means["mat_in_bed_not_counted"] = means[MAT_IN_BED] - means[ON_THE_MAT]
    means["difference"] = means[PIPELINE] - means[MAT_SLEEP]
    means["difference_in_bed"] = means[PIPELINE] - means[MAT_IN_BED]
    gap.means = {name: means[name] for name in COMPOSITION}
    gap.hourly = {name: list(values / count) for name, values in hourly.items()}
    gap.beliefs = {
        name: (
            list(class_belief[name] / class_watched[name])
            if class_watched[name] > 0.0
            else None
        )
        for name in MAT_CLASSES
    }
    gap.cells = {
        group: {key: _cell(cell, count) for key, cell in sorted(cells[group].items())}
        for group in GROUPS
    }
    short = [
        k for k, hours in enumerate(daily[MAT_IN_BED]) if hours < plan.short_night_hours
    ]
    gap.short_nights = len(short)
    total_off = float(np.sum(daily[OFF]))
    gap.short_night_share = (
        float(np.sum([daily[OFF][k] for k in short])) / total_off
        if total_off > 0.0
        else None
    )
    gap.sleeping = np.asarray(sleeping_rows, dtype=float)
    gap.occupied = np.asarray(occupied_rows, dtype=float)
    return gap


def _correlation(x: np.ndarray, y: np.ndarray) -> tuple[float | None, bool]:
    """Spearman's correlation, and whether a constant side made it count as 0."""
    rho = correlation(x, y, "spearman")
    constant = x.size >= 3 and rho is None
    return (0.0 if constant else rho), constant


def swapped_reference(gap: HomeGap, mat: str, part: str) -> float | None:
    """The correlation of the mat's *mat* hours with the pipeline's *part* when
    every day's beliefs are set beside every other day's mat.

    For each cyclic shift of the days, a day's beliefs are counted on the mat
    minutes of the day that many places later, and set beside that day's mat
    hours; the reference is the mean over the shifts. A constant side counts as
    0, as in the correlation itself.
    """
    n = gap.days
    if n < 3:
        return None
    on = gap.sleeping @ gap.occupied.T / 3600.0
    total = gap.sleeping.sum(axis=1) / 3600.0
    reference = np.asarray(gap.daily[mat], dtype=float)
    values = []
    rows = np.arange(n)
    for shift in range(1, n):
        columns = (rows + shift) % n
        on_mat = on[rows, columns]
        counted = on_mat if part == ON_THE_MAT else total - on_mat
        rho, _ = _correlation(reference[columns], counted)
        values.append(0.0 if rho is None else rho)
    return float(np.mean(values))


def within_home(gap: HomeGap) -> dict[str, Any]:
    """D5's statistics of one home: correlations, references, shares, spreads."""
    arrays = {
        name: np.asarray(values, dtype=float) for name, values in gap.daily.items()
    }
    out: dict[str, Any] = {
        "correlations": {},
        "constant": {},
        "references": {},
        "excess": {},
        "shares": {},
        "spread": {},
    }
    for left, right in CORRELATED:
        name = pair_name(left, right)
        out["correlations"][name], out["constant"][name] = _correlation(
            arrays[left], arrays[right]
        )
    out["excess"] = {}
    for left, right in REFERENCED:
        name = pair_name(left, right)
        reference = swapped_reference(gap, left, right)
        out["references"][name] = reference
        rho = out["correlations"][name]
        out["excess"][name] = (
            None if reference is None or rho is None else rho - reference
        )
    total = arrays[PIPELINE]
    variance = float(total.var(ddof=1)) if total.size >= 2 else 0.0
    for part in PARTS:
        if variance > 0.0:
            covariance = float(np.cov(total, arrays[part], ddof=1)[0, 1])
            out["shares"][part] = covariance / variance
        else:
            out["shares"][part] = None
    for name in SPREAD:
        values = arrays[name]
        out["spread"][name] = float(values.std(ddof=1)) if values.size >= 2 else None
    return out


# ----------------------------------------------------------------------------
# Running the mat homes a second time, and the checks
# ----------------------------------------------------------------------------
def _beliefs(arguments: tuple[HouseholdData, TihmProtocol]) -> BeliefRun:
    return run_beliefs(*arguments)


def run_belief_homes(
    adapter: DatasetAdapter,
    homes: Sequence[str],
    tihm: TihmProtocol,
    *,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, BeliefRun]:
    """The repeat of the run with the rule off, keeping every step's belief."""
    tasks = []
    for name in homes:
        data = adapter.load(name)
        if not validate_household(data, tihm.mapping).ok:
            raise ValueError(f"the contract refuses mat home {name!r}")
        tasks.append((data, tihm))
    done: list[BeliefRun] = []
    if jobs <= 1:
        for task in tasks:
            done.append(_beliefs(task))
            if progress is not None:
                progress(len(done), len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for result in pool.map(_beliefs, tasks):
                done.append(result)
                if progress is not None:
                    progress(len(done), len(tasks))
    return {run.home: run for run in done}


def _close(a: Any, b: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tolerance) for k in a)
    if a is None or b is None:
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tolerance
    return bool(a == b)


def check_pairs(
    pairs: Mapping[str, Pairs],
    protocol: SleepMatProtocol,
    sleep_mat: Mapping[str, Any],
) -> dict[str, Any]:
    """Refuse matched days or per-home statistics other than the sleep-mat record's."""
    differing = []
    for home, home_pairs in sorted(pairs.items()):
        here = home_statistics(home_pairs, protocol)
        there = sleep_mat["homes"].get(home, {}).get(PRIMARY)
        if there is None or not _close(here, there):
            differing.append(home)
    if differing:
        raise ValueError(
            "the matched days or per-home statistics differ from the sleep-mat "
            f"record's for {', '.join(differing)}; nothing is reported"
        )
    included = sorted(
        home for home, p in pairs.items() if p.size >= protocol.min_matched_days
    )
    if included != sorted(sleep_mat["analyses"][PRIMARY]["homes"]):
        raise ValueError("the included homes are not the sleep-mat record's")
    return {
        "homes": len(pairs),
        "included": included,
        "matched_days": sum(pairs[home].size for home in included),
        "reproduced": True,
    }


def check_beliefs(
    runs: Mapping[str, HouseholdRun],
    beliefs: Mapping[str, BeliefRun],
    spread: Mapping[str, Mapping[date, DayBeliefs]],
    pairs: Mapping[str, Pairs],
    tolerance_hours: float,
) -> dict[str, Any]:
    """Refuse a repeat whose days differ from the run's, or minutes that lose time.

    On every matched day the minutes must sum, for every state, to the day's
    hours, and the watched minutes to the day's observed time.
    """
    differing, unsummed = [], []
    largest = 0.0
    for home, run in sorted(runs.items()):
        first = {record.day: dict(record.hours) for record in run.days if record.hours}
        second = {
            day: dict(hours)
            for day, hours in beliefs[home].day_hours.items()
            if day in first
        }
        if first != second:
            differing.append(home)
            continue
        records = {record.day: record for record in run.days}
        states = beliefs[home].states
        for day in pairs[home].days:
            record = records[day]
            day_spread = spread[home].get(day)
            if day_spread is None:
                unsummed.append(f"{home} {day.isoformat()}")
                continue
            sums = day_spread.belief.sum(axis=0) / 3600.0
            gaps = [
                abs(sums[k] - record.hours.get(state, 0.0))
                for k, state in enumerate(states)
            ]
            gaps.append(abs(day_spread.watched.sum() / 3600.0 - 24.0 * record.observed))
            largest = max(largest, *gaps)
            if max(gaps) > tolerance_hours:
                unsummed.append(f"{home} {day.isoformat()}")
    if differing or unsummed:
        raise ValueError(
            "the repeat does not give the run's days for "
            f"{', '.join(differing) or 'no home'}, or the minutes do not sum to "
            f"the day's hours on {len(unsummed)} days; nothing is reported"
        )
    return {"homes": len(runs), "largest_difference_hours": largest, "reproduced": True}


def check_mat(
    timelines: Mapping[str, Mapping[str, MatTimeline]],
    mats: Mapping[str, Mapping[str, MatHome]],
    pairs: Mapping[str, Pairs],
    homes: Sequence[str],
    tolerance_hours: float,
) -> dict[str, Any]:
    """Refuse mat minutes that do not sum to the day's mat hours under each clock."""
    bad = []
    for clock in CLOCKS:
        for home in homes:
            days = mats[clock][home].days
            for day in pairs[home].days:
                codes, _ = timelines[clock][home].day(day)
                sleep = float((codes == _ASLEEP).sum()) / 60.0
                in_bed = float((codes != _OFF).sum()) / 60.0
                there = days.get(day)
                expected = (there.sleep, there.in_bed) if there else (0.0, 0.0)
                if (
                    max(abs(sleep - expected[0]), abs(in_bed - expected[1]))
                    > tolerance_hours
                ):
                    bad.append(f"{clock} {home} {day.isoformat()}")
    if bad:
        raise ValueError(
            f"the mat's minutes do not give its hours on {len(bad)} days; "
            "nothing is reported"
        )
    return {"clocks": list(CLOCKS), "reproduced": True}


def check_clock(
    pairs: Mapping[str, Pairs], homes: Sequence[str], zone: tzinfo
) -> dict[str, Any]:
    """Refuse a matched day on which the local clock changes."""
    changed = []
    for home in homes:
        for day in pairs[home].days:
            start = datetime.combine(day, time(), tzinfo=zone).timestamp()
            end = datetime.combine(
                day + timedelta(days=1), time(), tzinfo=zone
            ).timestamp()
            if end - start != 86400.0:
                changed.append(f"{home} {day.isoformat()}")
    if changed:
        raise ValueError(f"{len(changed)} matched days have a clock change")
    return {"reproduced": True}


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def _mean(values: Sequence[float | None], confidence: float) -> dict[str, Any]:
    kept = [v for v in values if v is not None]
    return {
        **mean_over_homes(kept, confidence),
        "median": float(median(kept)) if kept else None,
    }


def _groups(
    gaps: Mapping[str, HomeGap],
    homes: Sequence[str],
    groups: Sequence[str],
    level: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in groups:
        keys = sorted({key for h in homes for key in gaps[h].cells[group]})
        out[group] = {}
        for key in keys:
            cells = [gaps[h].cells[group].get(key) for h in homes]
            out[group][key] = {
                "pipeline_sleep_hours": float(
                    np.mean(
                        [c["pipeline_sleep_hours"] or 0.0 if c else 0.0 for c in cells]
                    )
                ),
                "hours_with_no_record": float(
                    np.mean(
                        [c["hours_with_no_record"] or 0.0 if c else 0.0 for c in cells]
                    )
                ),
                "sleeping_belief": _mean(
                    [c["sleeping_belief"] if c else None for c in cells], level
                ),
            }
    return out


def _tracking(
    within: Mapping[str, Mapping[str, Any]], homes: Sequence[str], level: float
) -> dict[str, Any]:
    correlations = {
        name: {
            **_mean([within[h]["correlations"][name] for h in homes], level),
            "constant_homes": sum(1 for h in homes if within[h]["constant"][name]),
        }
        for name in within[homes[0]]["correlations"]
    }
    return {
        "correlations": correlations,
        "references": {
            name: _mean([within[h]["references"][name] for h in homes], level)
            for name in within[homes[0]]["references"]
        },
        "excess": {
            name: {
                **_mean([within[h]["excess"][name] for h in homes], level),
                "homes_above_zero": sum(
                    1
                    for h in homes
                    if (value := within[h]["excess"][name]) is not None and value > 0.0
                ),
            }
            for name in within[homes[0]]["excess"]
        },
        "shares": {
            part: _mean([within[h]["shares"][part] for h in homes], level)
            for part in PARTS
        },
        "spread": {
            name: _mean([within[h]["spread"][name] for h in homes], level)
            for name in SPREAD
        },
    }


def _summaries(
    gaps: Mapping[str, HomeGap], homes: Sequence[str], plan: SleepGapPlan
) -> dict[str, Any]:
    """D1 to D5 and D7 over *homes*."""
    level = plan.confidence
    within = {h: within_home(gaps[h]) for h in homes}
    hourly = {
        name: [
            float(np.mean([gaps[h].hourly[name][hour] for h in homes]))
            for hour in range(24)
        ]
        for name in gaps[homes[0]].hourly
    }
    beliefs: dict[str, Any] = {}
    for name in MAT_CLASSES:
        rows = [row for h in homes if (row := gaps[h].beliefs[name]) is not None]
        beliefs[name] = {
            "homes": len(rows),
            "mean": [float(v) for v in np.mean(rows, axis=0)] if rows else None,
        }
    return {
        "homes": list(homes),
        "D1": {
            name: _mean([gaps[h].means[name] for h in homes], level)
            for name in COMPOSITION
        },
        "D2": hourly,
        "D3": beliefs,
        "D4": _groups(
            gaps, homes, ("sensor", "since", "joint", "period", "exit"), level
        ),
        "D5": _tracking(within, homes, level),
        "D7": {
            **_groups(gaps, homes, ("distance", "distance_period"), level),
            "short_nights": {
                "days": sum(gaps[h].short_nights for h in homes),
                "homes": sum(1 for h in homes if gaps[h].short_nights),
                "share_of_sleep_with_no_record": _mean(
                    [gaps[h].short_night_share for h in homes], level
                ),
            },
        },
        "per_home": {
            h: {
                "matched_days": gaps[h].days,
                **gaps[h].means,
                "short_nights": gaps[h].short_nights,
                **{f"r[{k}]": v for k, v in within[h]["correlations"].items()},
                **{f"reference[{k}]": v for k, v in within[h]["references"].items()},
                **{f"excess[{k}]": v for k, v in within[h]["excess"].items()},
                **{f"share[{k}]": v for k, v in within[h]["shares"].items()},
            }
            for h in homes
        },
    }


@dataclass
class Checked:
    """What the checks leave for the estimands."""

    check: dict[str, Any]
    homes: list[str]
    pairs: dict[str, Pairs]
    spread: dict[str, dict[date, DayBeliefs]]
    timelines: dict[str, dict[str, MatTimeline]]


def run_checks(
    runs: Mapping[str, Mapping[str, HouseholdRun]],
    beliefs: Mapping[str, BeliefRun],
    mat_minutes: Mapping[str, Mapping[datetime, str]],
    mats: Mapping[str, Mapping[str, MatHome]],
    protocol: SleepMatProtocol,
    plan: SleepGapPlan,
    zone: tzinfo,
    sleep_mat: Mapping[str, Any],
) -> Checked:
    """Every check that reads no estimand; each raises when it does not hold.

    *mats* holds the mat's days as the sleep-mat comparison reads them, under
    each clock. Nothing of the description is computed.
    """
    off = {home: pipeline_days(run) for home, run in runs[RUN_OFF].items()}
    rule = {home: pipeline_days(run) for home, run in runs[RUN_RULE].items()}
    silenced = {home: silenced_days(off[home], rule.get(home, ())) for home in off}
    pairs = {
        home: pairs_of(home, off[home], mats[PRIMARY_CLOCK].get(home), silenced[home])
        for home in sorted(off)
    }
    check: dict[str, Any] = {"pairs": check_pairs(pairs, protocol, sleep_mat)}
    homes = list(check["pairs"]["included"])
    check["clock"] = check_clock(pairs, homes, zone)
    max_interval = timedelta(minutes=plan.step_minutes * plan.max_interval_steps)
    spread = {home: day_beliefs(beliefs[home], zone, max_interval) for home in homes}
    check["beliefs"] = check_beliefs(
        {home: runs[RUN_OFF][home] for home in homes},
        beliefs,
        spread,
        pairs,
        plan.tolerance_hours,
    )
    timelines = {
        clock: {home: mat_timeline(mat_minutes.get(home, {}), shift) for home in homes}
        for clock, shift in CLOCKS.items()
    }
    check["mat"] = check_mat(timelines, mats, pairs, homes, plan.tolerance_hours)
    return Checked(check, homes, pairs, spread, timelines)


def score(
    runs: Mapping[str, Mapping[str, HouseholdRun]],
    beliefs: Mapping[str, BeliefRun],
    mat_minutes: Mapping[str, Mapping[datetime, str]],
    mats: Mapping[str, Mapping[str, MatHome]],
    protocol: SleepMatProtocol,
    plan: SleepGapPlan,
    zone: tzinfo,
    sleep_mat: Mapping[str, Any],
) -> dict[str, Any]:
    """The checks and every estimand, from both runs, the repeat and the mat."""
    checked = run_checks(
        runs, beliefs, mat_minutes, mats, protocol, plan, zone, sleep_mat
    )
    check, homes = checked.check, checked.homes
    events = {
        home: {record.day: record.events for record in runs[RUN_OFF][home].days}
        for home in homes
    }
    activations = {home: Activations.of(beliefs[home].observations) for home in homes}
    by_clock: dict[str, Any] = {}
    for clock in CLOCKS:
        gaps = {
            home: home_gap(
                beliefs[home],
                checked.spread[home],
                checked.pairs[home].days,
                checked.timelines[clock][home],
                events[home],
                activations[home],
                zone,
                plan,
            )
            for home in homes
        }
        kept = [home for home in homes if home not in STAGED_AWAKE_HOMES]
        by_clock[clock] = {
            **_summaries(gaps, homes, plan),
            "D6": {
                "left_out": [h for h in homes if h in STAGED_AWAKE_HOMES],
                **{
                    k: v
                    for k, v in _summaries(gaps, kept, plan).items()
                    if k not in ("per_home", "homes")
                },
            },
        }
    published = sleep_mat["analyses"][PRIMARY][SLEEP]["mean_difference"]["estimate"]
    found = by_clock[PRIMARY_CLOCK]["D1"]["difference"]["estimate"]
    if abs(found - published) > plan.tolerance_hours:
        raise ValueError(
            f"D1's difference {found} is not the sleep-mat record's {published}; "
            "nothing is reported"
        )
    check["difference"] = {"hours": found, "reproduced": True}
    return {
        "check": check,
        "states": list(beliefs[homes[0]].states),
        "clocks": by_clock,
    }


def _interval(label: str, estimate: Mapping[str, Any]) -> ReportedInterval | None:
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
        unit="household",
        n=int(estimate["homes"]),
    )


def intervals_of(results: Mapping[str, Any]) -> list[ReportedInterval]:
    """Every interval of D1 and D5, ready to quote."""
    out: list[ReportedInterval | None] = []
    for clock, entry in results["clocks"].items():
        for name, estimate in entry["D1"].items():
            out.append(_interval(f"{clock}: D1: {name}, hours a day", estimate))
        for group in ("correlations", "references", "excess"):
            for name, estimate in entry["D5"][group].items():
                out.append(
                    _interval(f"{clock}: D5: {group}: spearman {name}", estimate)
                )
        for name, estimate in entry["D5"]["shares"].items():
            out.append(_interval(f"{clock}: D5: variance share of {name}", estimate))
    return [item for item in out if item is not None]


def sleep_gap_record(
    runs: Mapping[str, Mapping[str, HouseholdRun]],
    beliefs: Mapping[str, BeliefRun],
    mat_minutes: Mapping[str, Mapping[datetime, str]],
    mats: Mapping[str, Mapping[str, MatHome]],
    protocol: SleepMatProtocol,
    tihm: TihmProtocol,
    plan: SleepGapPlan,
    published: Mapping[str, Mapping[str, Any]],
    silent_home: Mapping[str, Any],
    sleep_mat: Mapping[str, Any],
    *,
    plan_sha256: str,
    inputs: Sequence[InputArtifact] = (),
    code_changed: Mapping[str, Sequence[str]] | None = None,
) -> ExperimentRecord:
    """The record of the description. Nothing is reported unless every check holds."""
    check = {
        RUN_OFF: check_published(runs[RUN_OFF], published),
        RUN_RULE: check_rule(
            runs[RUN_RULE], runs[RUN_OFF], silent_home, f"h{protocol.rule_hours:g}"
        ),
    }
    zone = ZoneInfo(TIMEZONE)
    scored = score(runs, beliefs, mat_minutes, mats, protocol, plan, zone, sleep_mat)
    results = {
        "check": {**check, **scored["check"]},
        "states": scored["states"],
        "clocks": scored["clocks"],
        "primary_clock": PRIMARY_CLOCK,
    }
    return ExperimentRecord(
        experiment=EXPERIMENT,
        configuration={
            "plan_sha256": plan.sha256(),
            "plan_file_sha256": plan_sha256,
            "sleep_mat_protocol_sha256": protocol.sha256(),
            "tihm_protocol_sha256": tihm.sha256(),
            "clocks_hours": dict(CLOCKS),
            "primary_clock": PRIMARY_CLOCK,
            "min_matched_days": protocol.min_matched_days,
            "step_minutes": plan.step_minutes,
            "max_interval_steps": plan.max_interval_steps,
            "quiet_after_minutes": QUIET_AFTER_MINUTES,
            "short_night_hours": plan.short_night_hours,
            "code_changed_since_the_freeze": {
                group: list(names) for group, names in (code_changed or {}).items()
            },
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
                name="pipeline_off_beliefs",
                configuration={
                    "step_minutes": plan.step_minutes,
                    "home_silence_horizon_hours": None,
                },
            )
        ],
        household_metrics={
            clock: entry["per_home"] for clock, entry in scored["clocks"].items()
        },
        intervals=intervals_of(results),
        notes=[
            "An exploratory description, planned after the sleep-mat comparison "
            "had been read and frozen before anything of it was computed. No "
            "criterion and no margin. Nothing in the pipeline was changed or "
            "fitted.",
            "The mat's stages are the device's own and are not validated here. A "
            "minute with no mat record may be an empty bed, a person asleep "
            "elsewhere or a mat that stopped recording.",
            "Per-home summaries only: no day's or minute's values are recorded, "
            "and nothing of the dataset is redistributed.",
            "TIHM is by Palermo et al., Scientific Data 10, 606 (2023), under CC "
            "BY 4.0. Surrey and Borders Partnership NHS Foundation Trust and "
            "Howz are acknowledged, as the dataset asks.",
        ],
    )
