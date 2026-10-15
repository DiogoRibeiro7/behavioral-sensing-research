"""The plan of a description: where the pipeline's sleep parts from the sleep mat.

The sleep-mat comparison found that, within a TIHM home, the pipeline's daily
``sleeping_hours`` barely follows the mat's hours asleep, and that it gives
about four hours a day more sleep than the mat. The matched-sensor study then
showed that, in the simulator, TIHM's measurable sensor properties do not by
themselves make the hours of sleep stop following the truth. Neither says
where in a day the pipeline's sleep and the mat's part.

This plan fixes how the pipeline's belief is set beside the mat minute by
minute, on the days and homes of the sleep-mat comparison's primary analysis,
and what is computed, before any of it is computed. **It is a description,
not a test.** The comparison it decomposes has been read, so no criterion and
no margin is declared; the plan only limits what is computed and says, in
advance, what each pattern would mean. Nothing in the pipeline is changed or
fitted.

:func:`declared_plan` is frozen in ``artifacts/sleep_gap/sleep_gap_plan.json``
with the digest of every source file the run passes through.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from ..alerts.alert import AlertPolicy
from ..evaluation.provenance import json_safe, resolved_defaults
from ..external.tihm import TIMEZONE

#: The record's name and the layout of its results.
EXPERIMENT = "sleep-gap"
RESULT_SCHEMA = "sleep-gap/1"

#: The mat's clocks: as the file writes it, and an hour later, the lag at which
#: the sleep-mat comparison found hourly activity most opposed to time in bed.
AS_RECORDED = "as_recorded"
AN_HOUR_LATER = "an_hour_later"
CLOCKS: dict[str, float] = {AS_RECORDED: 0.0, AN_HOUR_LATER: 1.0}
PRIMARY_CLOCK = AS_RECORDED

#: What the mat says of a minute.
ASLEEP = "asleep"
AWAKE_ON_THE_MAT = "awake_on_the_mat"
OFF_THE_MAT = "off_the_mat"
MAT_CLASSES = (ASLEEP, AWAKE_ON_THE_MAT, OFF_THE_MAT)

#: The context of a minute: the sensor that last reported before it, and how
#: long before. Bounds are in minutes, closed on the left.
NOTHING_YET = "nothing_yet"
SINCE_BINS: tuple[tuple[str, float, float | None], ...] = (
    ("under_10_minutes", 0.0, 10.0),
    ("10_to_60_minutes", 10.0, 60.0),
    ("1_to_3_hours", 60.0, 180.0),
    ("3_hours_or_more", 180.0, None),
)

#: A day's quiet hours are its minutes whose middle is at least this long
#: after the latest activation at or before it.
QUIET_AFTER_MINUTES = 60.0

#: The time from a minute with no mat record to the nearest mat record, in
#: minutes, closed on the left.
DISTANCE_BINS: tuple[tuple[str, float, float | None], ...] = (
    ("under_30_minutes", 0.0, 30.0),
    ("30_minutes_to_2_hours", 30.0, 120.0),
    ("2_to_6_hours", 120.0, 360.0),
    ("6_hours_or_more", 360.0, None),
)

#: The day period, from its first local hour to the hour it ends; the rest of
#: the day is night.
DAY_HOURS = (7, 22)
DAY = "day"
NIGHT = "night"

#: The exits among the activity file's locations.
EXITS = ("Front Door", "Back Door")

#: The daily quantities set beside each other within a home.
PIPELINE = "pipeline"
ON_THE_MAT = "pipeline_on_the_mat"
OFF = "pipeline_off_the_mat"
MAT_SLEEP = "mat_sleep"
MAT_IN_BED = "mat_in_bed"
QUIET = "quiet_hours"
EVENTS = "events"

#: D1's quantities, in the order the page lists them.
COMPOSITION = (
    PIPELINE,
    "pipeline_asleep",
    "pipeline_awake_on_the_mat",
    OFF,
    MAT_SLEEP,
    "mat_sleep_not_counted",
    "mat_asleep_unwatched",
    MAT_IN_BED,
    "mat_in_bed_not_counted",
    "difference",
    "difference_in_bed",
)

#: The records the run must reproduce or reads, and their digests.
SLEEP_MAT_PROTOCOL = "artifacts/sleep_mat/sleep_mat_protocol.json"
SLEEP_MAT_PROTOCOL_SHA256 = (
    "3e0f26fc82c4fde7ed241f469bd7c445afbdbe3e39e97169c8e0444ba5279f98"
)
SLEEP_MAT_RECORD = "artifacts/sleep_mat/sleep-mat.json"
SLEEP_MAT_RECORD_SHA256 = (
    "5c003cda0d83dfc32336a6603774d8d32a26e4170154172d4f50b3d33e5304d4"
)
ALERT_BURDEN_PROTOCOL = "artifacts/tihm/alert_burden_protocol.json"
ALERT_BURDEN_PROTOCOL_SHA256 = (
    "ed04c9885f427f055e099e066d518188dd665cfbaaa012c459136e98378d8a2e"
)
ALERT_BURDEN_RECORD = "artifacts/tihm/tihm-alert-burden.json"
ALERT_BURDEN_RECORD_SHA256 = (
    "5a5bb24167c8f089de3138f9c35ec6e07cd902ba385999f172458e40b40a64ff"
)
SILENT_HOME_RECORD = "artifacts/silent_home/silent-home-tihm.json"
SILENT_HOME_RECORD_SHA256 = (
    "96798aa2e573f281477faad624ee7369ff2508d5a8631c7003f7f75218bae6a9"
)
PINNED_RECORDS = {
    SLEEP_MAT_PROTOCOL: SLEEP_MAT_PROTOCOL_SHA256,
    SLEEP_MAT_RECORD: SLEEP_MAT_RECORD_SHA256,
    ALERT_BURDEN_PROTOCOL: ALERT_BURDEN_PROTOCOL_SHA256,
    ALERT_BURDEN_RECORD: ALERT_BURDEN_RECORD_SHA256,
    SILENT_HOME_RECORD: SILENT_HOME_RECORD_SHA256,
}

#: The sources whose content the freeze records.
PINNED_PACKAGE = "sensor_modeling"
PINNED_SCRIPTS = ("scripts/freeze_sleep_gap.py", "scripts/run_sleep_gap.py")
PINNED_DISTRIBUTIONS = ("numpy", "scipy", "pandas")

_REPOSITORY = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    """SHA-256 of a text file, whatever its line endings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def pinned_sources(root: Path | None = None) -> tuple[str, ...]:
    """Every source file whose content the freeze records, as a sorted tuple."""
    root = _REPOSITORY if root is None else Path(root)
    package = sorted(
        path.relative_to(root).as_posix()
        for path in (root / PINNED_PACKAGE).rglob("*.py")
        if "__pycache__" not in path.parts
    )
    return (*package, *PINNED_SCRIPTS)


def code_now(root: Path | None = None) -> dict[str, Any]:
    """The sources' digests, the libraries' versions and every default, now."""
    root = _REPOSITORY if root is None else Path(root)
    defaults: dict[str, Any] = json_safe(
        {**resolved_defaults(), "alert_policy": dataclasses.asdict(AlertPolicy())}
    )
    return {
        "sources": {name: file_sha256(root / name) for name in pinned_sources(root)},
        "distributions": {
            "python": platform.python_version(),
            **{name: metadata.version(name) for name in PINNED_DISTRIBUTIONS},
        },
        "defaults": defaults,
    }


#: What had been read or run before the plan was written.
INSPECTED_BEFORE = (
    "The sleep-mat comparison's record and pages, every estimand: within a home "
    "the pipeline's daily hours of sleep follow the mat's sleep at a mean "
    "Spearman correlation of 0.11 [0.03, 0.19] over 14 homes and its hours in "
    "bed at 0.14; the pipeline gives 3.95 hours a day more sleep than the mat "
    "and 1.97 more than its hours in bed; in one home its median is 19.9 hours "
    "of sleep a day; on the 19 silent days the mat observed its median is 23.8 "
    "hours; hourly activity is most opposed to time in bed with the mat's clock "
    "an hour later, at -0.58 against -0.54 as recorded, with overlapping "
    "intervals; with the mat an hour later the daily correlation is 0.13 over "
    "13 homes; and the per-home summaries the record holds.",
    "The alert-burden run and its description after the run: days with no "
    "record read as sleep, the pipeline abstaining on 0.06% of usable time, and "
    "its deviating days quiet ones.",
    "The matched-sensor study: in three TIHM homes, 0d5ef, 30a32 and 55cd4, the "
    "default pipeline's step-level beliefs after 10-minute windows that held "
    "only kitchen, or only bathroom, activations, 68% to 89% on "
    "`home_inactive`; those homes' hours a day in each state under the default "
    "and five changed emission declarations; and, in the simulator, hours of "
    "sleep that still follow the truth at 0.71 under sensors matched to TIHM's "
    "records.",
    "For this plan: the activity file's location names and their counts, the "
    "date range of the activity and mat files, 1 April to 30 June 2019, so "
    "every timestamp falls in British Summer Time and no clock change lies "
    "inside the data, and the pipeline's code: its steps start at a home's "
    "first observation and follow every 10 minutes, each step's belief "
    "describes the time since the step before, and a day's summary counts only "
    "the intervals between two steps of that day, so the interval across "
    "midnight is counted by neither day. No pipeline output had been "
    "aggregated by hour or set beside the mat at a finer grain than a day.",
    "An independent review of the draft read the code and the mat's file on "
    "its own: only minute records, sessions that start AWAKE 3,183 times in "
    "3,304 and end asleep about 71% of the time; 75 stretches without records "
    "of 3 hours or more beginning between 22:00 and 05:00, 61 of them in home "
    "55cd4; and, over every home's observed days, 5 with under 2 hours of "
    "records and 15 with under 4. It found that the draft's correlations of the "
    "mat with the pipeline's sleep on and off the mat are partly mechanical, "
    "that a minute without a record was called an empty bed, and that a "
    "minute's context was taken before the step its belief came from had seen "
    "all it saw. The plan was revised for each.",
    "The run's checks were rehearsed on the mat homes before the freeze, on "
    "the draft and again on the revision: both runs, the included homes and "
    "their matched days, the clock-change guard, the repeat's daily hours of "
    "every state and its observed time, and the mat's minutes under each "
    "clock, the last time with the run script's checks-only mode. Only "
    "whether each check held, the largest difference between a day's minutes "
    "and its hours, 1.4e-13 hours, and that three matched days are not "
    "observed by the mat an hour later were read.",
)

#: Readings written down before anything is computed.
WHAT_THE_PATTERNS_WOULD_MEAN = (
    "If the pipeline's sleep while the mat has a record follows the mat's "
    "sleep within a home by more than the swapped-mask reference gives, judged "
    "by each home's correlation minus its reference, and the whole day's does "
    "not, the daily number is diluted by sleep counted with no mat record; the "
    "covariance shares say which part moves the daily hours. If it follows no "
    "better than the reference, the pipeline also misses the night's own "
    "variation.",
    "If the sleep counted with no mat record falls mostly by day and after the "
    "lounge's sensor with a long quiet, it is consistent with sitting still "
    "read as sleep; after an exit door, front or back, with a long quiet, with "
    "absence read as sleep; within two hours of the mat's records, with the "
    "edges of the night, the mat's clock or the pipeline's timing of going to "
    "bed and getting up; far from any record at night, possibly with a mat that "
    "stopped recording.",
    "If the mat's clock an hour later moves much of the sleep counted near the "
    "mat's records onto them, the as-recorded clock puts that part off the "
    "mat. That the daily totals change little is known already from the "
    "sleep-mat comparison's shifted analysis.",
    "If, while the mat says asleep, the pipeline's belief sits on states other "
    "than sleeping, the pipeline misses sleep as well as adding it.",
    "None of these patterns says which of the pipeline and the mat is right "
    "about a minute: the mat's stages are the device's own and not validated, "
    "a minute with no record may be a person asleep elsewhere, out, or a mat "
    "that stopped, and a few homes dominate a mean over 14.",
)


@dataclass(frozen=True)
class SleepGapPlan:
    """What is computed, on which homes and days, before anything is computed.

    Attributes
    ----------
    confidence
        The level of every interval.
    min_matched_days
        The fewest matched days a home needs, the sleep-mat protocol's.
    step_minutes
        The pipeline's step on TIHM, the alert-burden protocol's.
    max_interval_steps
        The longest gap between two steps a step's belief is held to
        describe, in steps, as the pipeline's daily summary has it.
    tolerance_hours
        How far a day's hours, summed minute by minute, may lie from the run's
        daily summary, or the mat's, before nothing is reported.
    short_night_hours
        A matched day with fewer hours of mat records than this is counted in
        D7.
    """

    confidence: float = 0.95
    min_matched_days: int = 14
    step_minutes: int = 10
    max_interval_steps: int = 3
    tolerance_hours: float = 1e-9
    short_night_hours: float = 4.0
    inspected_before: tuple[str, ...] = INSPECTED_BEFORE

    def to_dict(self) -> dict[str, Any]:
        """Return the full declaration, in a stable serialisable form."""
        level = f"{self.confidence:.0%}"
        since = {
            name: (
                f"from {low:g} minutes"
                + (f" to under {high:g}" if high is not None else ", and on")
            )
            for name, low, high in SINCE_BINS
        }
        since[NOTHING_YET] = "no activation before it"
        distance = {
            name: (
                f"from {low:g} minutes"
                + (f" to under {high:g}" if high is not None else ", and on")
            )
            for name, low, high in DISTANCE_BINS
        }
        return {
            "schema": "sleep-gap-plan/1",
            "name": EXPERIMENT,
            "status": "exploratory description, planned after the sleep-mat "
            "comparison had been read and before any value of it was computed. "
            "No criterion and no margin is declared. Nothing in the pipeline is "
            "changed or fitted",
            "question": "where in the day, and on which of the mat's minutes, "
            "the pipeline puts its sleep, and which part of it moves the daily "
            "hours that do not follow the mat",
            "inspected_before": list(self.inspected_before),
            "homes_and_days": {
                "homes": "the homes the sleep-mat comparison's primary analysis "
                f"includes, those with at least {self.min_matched_days} matched "
                "days",
                "days": "that analysis's matched days: usable days of the run "
                "with the silent-home rule off that the mat observed, other than "
                "a home's first and last, leaving out every day touched by a "
                "silence of the whole home of 12 hours or more. The same days "
                "are kept under both of the mat's clocks, although three of them "
                "would not count as observed by the mat an hour later",
                "runs": "both runs of the sleep-mat comparison, with the "
                "alert-burden protocol's frozen settings and a step of "
                f"{self.step_minutes} minutes, the rule at 12 hours read only for "
                "the days it flags; the beliefs come from a repeat of the run "
                "with the rule off",
                "timezone": TIMEZONE,
            },
            "the_minutes": {
                "grid": "every minute of each matched local day, from 00:00 to "
                "24:00; a mat record stamped at a minute covers that minute",
                "the_mat": {
                    ASLEEP: "a mat record staged LIGHT, DEEP or REM",
                    AWAKE_ON_THE_MAT: "a mat record staged AWAKE",
                    OFF_THE_MAT: "no mat record. The bed may be empty, the person "
                    "may be asleep elsewhere, or the mat may have stopped "
                    "recording; the file cannot tell these apart",
                },
                "the_pipeline": "the belief of the step whose interval, from the "
                "step before to the step, covers the minute, weighted by the "
                "seconds it covers; only intervals between two steps of the same "
                "day, and no longer than "
                f"{self.max_interval_steps} steps, count, as the daily summary "
                "counts them. A minute no interval covers is unwatched: the "
                "interval across midnight is counted by neither day, so about "
                "ten minutes around midnight are unwatched every day, the mat's "
                "sleep in them is sleep the pipeline did not count, and the "
                "pipeline's hours at 00 and 23 are lower by construction",
                "clocks": {
                    AS_RECORDED: "the mat's timestamps as the file writes them, "
                    "as the sleep-mat comparison read them; the primary clock",
                    AN_HOUR_LATER: "each mat record counted an hour after its "
                    "timestamp, the lag at which the sleep-mat comparison found "
                    "hourly activity most opposed to time in bed, narrowly, and "
                    "where a mat clock in UTC would put it; every estimand is "
                    "computed under it as well, on the same days",
                },
                "context": {
                    "moment": "for a watched minute, the moment of the step whose "
                    "interval covers the minute's middle, which is what that "
                    "step's belief had seen; for an unwatched minute, its middle",
                    "sensor": "the sensor of the home's latest activation at or "
                    "before that moment, by its location in the activity file; "
                    "the front and back doors are the exits",
                    "since": {
                        **since,
                        "measured": "up to the context's moment",
                    },
                    "period": "day from 07:00 to 22:00, night from 22:00 to 07:00",
                    "distance_to_the_mat": distance,
                    "quiet_hours": "a day's minutes whose middle is at least "
                    f"{QUIET_AFTER_MINUTES:g} minutes after the latest activation "
                    "at or before it, in hours",
                },
            },
            "estimands": {
                "D1": "For each clock, the mean over homes, with its interval, "
                "and the median over homes of each home's mean over its matched "
                "days of: the pipeline's hours of sleep; its hours of sleep on "
                "the mat's asleep minutes, on its awake minutes and on minutes "
                "with no record; the mat's hours asleep and in bed; the mat's "
                "sleep and time in bed the pipeline did not count as sleep, and "
                "the mat's asleep hours on unwatched minutes; and the two "
                "differences, pipeline minus the mat's sleep, which equals "
                "sleep counted while awake on the mat plus sleep counted with no "
                "record minus the mat's sleep not counted, and pipeline minus "
                "the mat's time in bed, which equals sleep counted with no "
                "record minus the time in bed not counted. Each home's values "
                "are reported.",
                "D2": "For each clock and each local hour, the mean over homes of "
                "each home's mean hours a day in that hour of the pipeline's "
                "sleep on each class of the mat's minutes, and of the mat's sleep "
                "and time in bed.",
                "D3": "For each clock and each class of the mat's minutes, the "
                "mean over homes of each home's mean belief in each of the seven "
                "states over its watched minutes of that class.",
                "D4": "For each clock, over the minutes with no mat record: the "
                "pipeline's hours of sleep and the hours of such minutes, each a "
                "mean over homes of a home's mean a day, and the mean over homes "
                "of the pipeline's belief in sleeping over them, by the sensor "
                "that last reported, by the time since it, by whether that "
                "sensor was an exit, by period of the day, and by sensor, time "
                "since and period together.",
                "D5": "For each clock, the mean over homes of the within-home "
                "Spearman correlation over matched days of the mat's sleep, and "
                "of its time in bed, with: the pipeline's hours of sleep, its "
                "hours of sleep while the mat has a record, and its hours of "
                "sleep with no record; beside each correlation with a part, the "
                "same correlation with every day's beliefs set beside every other "
                "day's mat, the mean over all cyclic shifts of the days, which "
                "is what the mat's own hours give with no day-specific tracking, "
                "and each home's correlation minus its reference, as a mean over "
                "homes with the number of homes above zero; the share of "
                "the variance of the pipeline's daily hours of sleep that moves "
                "with each part, its covariance with the part over its variance, "
                "which sum to one; the correlation of the pipeline's hours of "
                "sleep with the day's quiet hours and with the day's "
                "activations; and the mean over homes of the within-home "
                "standard deviation of the pipeline's hours of sleep and of its "
                "two parts.",
                "D6": "D1 to D5 and D7 without the three homes the sleep-mat "
                "comparison found the mat staging mostly awake.",
                "D7": "For each clock, over the minutes with no mat record: the "
                "pipeline's hours of sleep, the hours of such minutes and the "
                "belief in sleeping, as in D4, by the time to the nearest mat "
                "record, earlier or later, in the home's whole file, and by that "
                "time and period of the day together; and the matched days with "
                f"fewer than {self.short_night_hours:g} hours of mat records, how "
                "many there are and the mean over homes of their share of the "
                "home's sleep counted with no record, a home with none of them "
                "counting as 0 and a home with no such sleep left out.",
            },
            "definitions": {
                "correlation": "as the sleep-mat protocol has it: Spearman's, "
                "over a home's matched days; a home whose values on either side "
                "are constant has no correlation and counts as 0, and how many "
                "did so is reported",
                "pipeline_on_the_mat": "the pipeline's hours of sleep on the "
                "mat's asleep and awake minutes",
                "variance_share": "a home's covariance between the pipeline's "
                "daily hours of sleep and a part, over the variance of the "
                "former; a home with constant hours has none and is left out",
            },
            "intervals": {
                "method": f"a Student t interval at {level} for a mean over "
                "homes, as the sleep-mat comparison's",
                "unit": "homes",
            },
            "the_check": {
                "what": [
                    "with the rule off, every mat home's run gives the published "
                    "alert-burden record's counts, and with the rule at 12 hours "
                    "the silent-home record's, as the sleep-mat run checked",
                    "every home's number of matched days, correlations and mean "
                    "differences against both of the mat's references equal the "
                    "sleep-mat record's, as do the included homes",
                    "the beliefs read minute by minute come from a repeat of the "
                    "run with the rule off whose daily summaries equal the first "
                    "run's; on every matched day the minutes sum, for every "
                    "state, to the day's hours, and the watched minutes to the "
                    "day's observed time, to within "
                    f"{self.tolerance_hours:g} hours",
                    "on every matched day the mat's minutes sum to the day's "
                    "hours asleep and in bed as the sleep-mat comparison reads "
                    "them, under each clock",
                    "with the mat's clock as recorded, D1's mean difference from "
                    "the mat's sleep equals the sleep-mat record's",
                    "no matched day has a clock change",
                ],
                "otherwise": "nothing is reported, not even per home",
            },
            "reporting": "every estimand under both clocks, whatever it shows; "
            "per-home summaries, never a day's or a minute's values",
            "what_the_patterns_would_mean": list(WHAT_THE_PATTERNS_WOULD_MEAN),
            "what_this_cannot_show": [
                "Whether the pipeline or the mat is right about a minute: the "
                "mat's stages are the device's own and are not validated.",
                "Why there is no mat record: a mat cannot tell sleep in a chair "
                "from a person who is out or from a mat that stopped.",
                "Anything beyond these 14 homes of people living with dementia, "
                "all in one study, or beyond the pipeline's default settings.",
                "A tested claim: the comparison it decomposes had been read, and "
                "nothing here is held to a margin.",
            ],
            "pinned_records": dict(PINNED_RECORDS),
            "acknowledgement": "TIHM is by Palermo et al., Scientific Data 10, "
            "606 (2023), under CC BY 4.0. Surrey and Borders Partnership NHS "
            "Foundation Trust and Howz are acknowledged, as the dataset asks. "
            "Nothing of the dataset is redistributed.",
            "settings": {
                "confidence": self.confidence,
                "min_matched_days": self.min_matched_days,
                "step_minutes": self.step_minutes,
                "max_interval_steps": self.max_interval_steps,
                "tolerance_hours": self.tolerance_hours,
                "short_night_hours": self.short_night_hours,
                "quiet_after_minutes": QUIET_AFTER_MINUTES,
                "day_hours": list(DAY_HOURS),
                "clocks_hours": dict(CLOCKS),
            },
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_plan() -> SleepGapPlan:
    """The plan as declared."""
    return SleepGapPlan()


def write_plan(plan: SleepGapPlan, path: Path) -> str:
    """Write the declaration with its digest and the code it was frozen against."""
    payload = {
        **plan.to_dict(),
        "plan_sha256": plan.sha256(),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "at_freeze": code_now(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return plan.sha256()


def check_frozen_plan(plan: SleepGapPlan, path: Path) -> str:
    """Refuse to run unless *plan* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    frozen.pop("frozen_at", None)
    frozen.pop("at_freeze", None)
    if frozen != {**plan.to_dict(), "plan_sha256": plan.sha256()}:
        raise ValueError(
            f"the plan in {path} differs from the one this code declares; "
            "a frozen plan cannot change once anything is computed"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def code_changes(path: Path, root: Path | None = None) -> dict[str, list[str]]:
    """What differs now from what the frozen file recorded at the freeze."""
    frozen = json.loads(Path(path).read_text(encoding="utf-8")).get("at_freeze")
    if frozen is None:
        raise ValueError(f"{path} does not record the code it was frozen against")
    now = code_now(root)
    return {
        group: sorted(
            name
            for name in set(frozen[group]) | set(now[group])
            if frozen[group].get(name) != now[group].get(name)
        )
        for group in ("sources", "distributions", "defaults")
    }
