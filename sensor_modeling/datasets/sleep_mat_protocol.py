"""The sleep-mat protocol: do the pipeline's hours of sleep follow a sleep mat.

Every alerting study in this repository counts its detections on one feature,
``sleeping_hours``: the hours of the day the pipeline's state filter gives to
``sleeping``, inferred from motion and door sensors. Nothing so far has set
that number beside a measurement of sleep. The TIHM run showed that a day with
no sensor event is read as nearly a whole day asleep, which says the number
can be wrong; it does not say how far it follows sleep on a day the sensors
reported.

In 17 of the 56 TIHM homes an under-the-mattress sleep mat recorded, for every
minute a person was in bed, a sleep stage computed by the device. It is a
measurement the pipeline never sees, so it can be the reference. This protocol
fixes how the two are put side by side, the estimands and the criteria, before
any value of the pipeline has been set beside any record of the mat.

**It is a measurement on a real cohort, not a test of a change.** Nothing in
the pipeline is altered or fitted: it runs with the alert-burden protocol's
frozen settings. The mat is not a gold standard either: its stages are the
device's own, and they are not validated here.

**The simulated reference is not a test.** The same estimands are computed on
simulated homes against the simulator's true hours of sleep, to say what
agreement looks like when the observation model is the one that made the data.

:func:`declared_protocol` is frozen in
``artifacts/sleep_mat/sleep_mat_protocol.json``, with the digest of every
source file the run passes through.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from ..alerts.alert import AlertPolicy
from ..baseline.adaptive import BaselineConfig
from ..evaluation.provenance import json_safe, resolved_defaults
from ..external.tihm import ARCHIVE_SHA256, FILES, PROVENANCE, TIMEZONE
from ..online.pipeline import PipelineConfig
from ..simulation.household import HouseholdConfig, build_registry

#: The records' names and the layout of their results.
EXPERIMENT = "sleep-mat"
RESULT_SCHEMA = "sleep-mat/1"
SIMULATED_EXPERIMENT = "sleep-mat-simulated"
SIMULATED_SCHEMA = "sleep-mat-simulated/1"

#: The feature compared, as the pipeline names it.
TRACKED_FEATURE = "sleeping_hours"

#: The file the mat's records are in, its columns and its timestamps' format.
MAT_FILE = "Sleep.csv"
MAT_COLUMNS = (
    "patient_id",
    "date",
    "state",
    "heart_rate",
    "respiratory_rate",
    "snoring",
)
MAT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

#: The stages the device reports, and those counted as asleep.
MAT_STATES = ("AWAKE", "LIGHT", "DEEP", "REM")
ASLEEP_STATES = ("LIGHT", "DEEP", "REM")

#: The two references, by what is counted on a day.
SLEEP = "sleep"
IN_BED = "in_bed"
REFERENCES = (SLEEP, IN_BED)

#: The two runs of the pipeline on each mat home.
RUN_OFF = "off"
RUN_RULE = "rule_12h"
RUNS = (RUN_OFF, RUN_RULE)

#: The analyses: which run, and whether days with no activity record are kept.
PRIMARY = "off"
WITH_SILENT_DAYS = "off_with_silent_days"
RULE = "rule_12h"
ANALYSES: dict[str, tuple[str, bool]] = {
    PRIMARY: (RUN_OFF, False),
    WITH_SILENT_DAYS: (RUN_OFF, True),
    RULE: (RUN_RULE, False),
}

#: The readings of a criterion stated against a margin.
FOLLOWS = "follows"
DOES_NOT_FOLLOW = "does_not_follow"
AGREES = "agrees"
DOES_NOT_AGREE = "does_not_agree"
INCONCLUSIVE = "inconclusive"
NOT_ESTIMABLE = "not_estimable"

#: The verdicts a behavioural alert is raised about, and those judged by the
#: sign of the day's deviation. A drift is judged by its slope, which the run
#: does not keep, so drift alerts are counted and not judged.
ALERT_KINDS = ("abrupt_change", "persistent_change", "gradual_drift")
JUDGED_KINDS = ("abrupt_change", "persistent_change")

#: The records the runs must reproduce, home by home, and their digests.
PUBLISHED_RECORD = "artifacts/tihm/tihm-alert-burden.json"
PUBLISHED_RECORD_SHA256 = (
    "5a5bb24167c8f089de3138f9c35ec6e07cd902ba385999f172458e40b40a64ff"
)
SILENT_HOME_RECORD = "artifacts/silent_home/silent-home-tihm.json"
SILENT_HOME_RECORD_SHA256 = (
    "96798aa2e573f281477faad624ee7369ff2508d5a8631c7003f7f75218bae6a9"
)

#: The planning record the interval rests on, its digest, the coverage each
#: interval reached in it without and with the outlying homes, and readings
#: the protocol quotes from it.
PLANNING_RECORD = "artifacts/sleep_mat/sleep-mat-planning.json"
PLANNING_RECORD_SHA256 = (
    "098876fe75003a8fe6af8e0933a0c31f6e16c5a9bf5936cd00d7b3ed7b8565e3"
)
PLANNED_COVERAGE = {
    "without_the_outlying_homes": {
        "percentile_bootstrap": (0.897, 0.929),
        "student_t": (0.932, 0.960),
    },
    "with_the_outlying_homes": {
        "percentile_bootstrap": (0.923, 1.0),
        "student_t": (0.948, 1.0),
    },
}
PLANNED_CHOICE = "student_t"
PLANNED_READINGS = {
    "c1_truth_every_home_at_0_5": 0.473,
    "c1_truth_homes_spread_0_6_about_0_5": 0.390,
    "c1_truth_outlying_homes_rest_at_0_6": 0.450,
    "c2_agrees_no_bias_spread_1": 0.864,
    "c2_agrees_no_bias_spread_2": 0.047,
}

#: The share of a home's minutes in bed, over its mat-observed days, that the
#: mat must stage as asleep for the home to stay in E12, and the homes that
#: share leaves out, from the mat's file alone.
STAGED_ASLEEP_SHARE = 0.6
STAGED_AWAKE_HOMES = ("16f4b", "d7a46", "f220c")

#: The sources whose content is recorded at the freeze and compared at the
#: run: every module of the package and the scripts that freeze and run it.
PINNED_PACKAGE = "sensor_modeling"
PINNED_SCRIPTS = ("scripts/freeze_sleep_mat.py", "scripts/run_sleep_mat.py")
PINNED_DISTRIBUTIONS = ("numpy", "scipy", "pandas")

#: What was run and read between the freezing of the protocol and its run, as
#: sentences in a file, so that adding to it changes no pinned source.
BETWEEN_FILE = "artifacts/sleep_mat/between_the_freeze_and_the_run.json"

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


def between_the_freeze_and_the_run(root: Path | None = None) -> tuple[str, ...]:
    """What was run and read between the freeze and the run, as recorded."""
    path = (_REPOSITORY if root is None else Path(root)) / BETWEEN_FILE
    if not path.exists():
        return ()
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not all(
        isinstance(entry, str) and entry for entry in entries
    ):
        raise ValueError(f"{BETWEEN_FILE} must hold a list of sentences")
    return tuple(entries)


def _span(bounds: tuple[float, float]) -> str:
    return f"{bounds[0]:.1%} to {bounds[1]:.1%}"


def event_sensors() -> tuple[str, ...]:
    """The simulated deployment's sensors that report on no cadence."""
    return tuple(
        spec.sensor_id for spec in build_registry() if spec.expected_interval is None
    )


@dataclass(frozen=True)
class SleepMatProtocol:
    """Every definition of the sleep-mat comparison.

    Attributes
    ----------
    tihm_step_minutes
        The pipeline's step on TIHM, the published run's.
    rule_hours
        The silent-home rule's horizon in the run that has it on.
    min_matched_days
        The matched days a home needs to enter an estimand.
    tracking_margin
        The mean within-home rank correlation C1 is judged against.
    level_margin_hours
        The bias in hours C2 is judged against.
    confidence
        The level of every interval.
    shifts_hours
        The shifts of the mat's clock E10 is made at.
    lags_hours
        The lags E11 compares hourly activity and time in bed at.
    sim_seed_root, sim_homes, sim_days, sim_step_minutes
        The simulated reference.
    planning_record_sha256
        The planning record the interval was chosen from.
    """

    tihm_step_minutes: int = 10
    rule_hours: float = 12.0
    min_matched_days: int = 14
    tracking_margin: float = 0.5
    level_margin_hours: float = 1.0
    confidence: float = 0.95
    shifts_hours: tuple[float, ...] = (-1.0, 1.0)
    lags_hours: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
    sim_seed_root: int = 20261008
    sim_homes: int = 100
    sim_days: int = 84
    sim_step_minutes: int = 10
    planning_record_sha256: str = PLANNING_RECORD_SHA256

    #: What had been seen when the protocol was written.
    inspected_before: tuple[str, ...] = (
        "the mat's file alone, for its structure: its columns, its four "
        "stages, its 461,423 records in 17 homes from 2019-04-01 to "
        "2019-06-30, one record a minute with no repeated minute, and no "
        "missing value",
        "the mat's coverage alone: the days each home has records on, the "
        "days whose neighbouring days also have records, and each home's "
        "median hours asleep and in bed on a day. 835 home-days have a "
        "record, 771 of them with records on the day before and the day "
        "after, and 14 homes have at least 14 such days. Four homes have a "
        "median under five hours asleep on the days they have records: 0f352 "
        "(3.7 asleep, 5.3 in bed), 16f4b (4.2, 9.4), d7a46 (2.6, 6.1) and "
        "f220c (1.8, 7.8)",
        "the share of each home's minutes in bed on its mat-observed days "
        "that the mat stages as asleep: under 0.5 in 16f4b, d7a46 and f220c, "
        "and 0.74 or more in every other home. E12's cut was set from it",
        "beside those day counts, each mat home's monitored and usable day "
        "counts in the published alert-burden record. That compares how many "
        "days each source has, not what either says about them",
        "the pipeline's output on TIHM had been read in the earlier studies: "
        "its alerts, its verdicts, its calendar, the silent days and the "
        "median 23.8 hours of sleep it infers on them. On 16 June 2019 all 47 "
        "monitored homes are silent. No value of the pipeline had been set "
        "beside any record of the mat, for any home or day",
        "the dataset paper's description of the mat: per-minute heart rate, "
        "breathing rate and sleep state while a person is in bed. It does not "
        "say whether any file's timestamps are local or UTC",
        "the interval was chosen on synthetic data, in the planning record, "
        "before the protocol was frozen. The planning was run twice: the "
        "second run added the outlying homes after the review below",
        "the code was rehearsed end to end on data that is not the "
        "comparison's: one TIHM home without a mat, run with the rule off and "
        "on and paired with a mat of random minutes generated for it, and one "
        "simulated home of ten days from seed 999, which is not a seed of the "
        "reference. Their run times and that every step completed were read, "
        "and none of their values",
        "an independent review of the draft read the mat's file alone, the "
        "published TIHM records and pages, and the first and last activity "
        "timestamp of each mat home; it ran no pipeline on TIHM. It found that "
        "the outage of 16 June falls on mat-observed days in 11 of the 14 "
        "homes, and the draft's primary was changed to leave days with no "
        "activity record out",
    )

    def __post_init__(self) -> None:
        """Validate the declaration."""
        if self.min_matched_days < 3:
            raise ValueError("a rank correlation needs at least three days")
        if not 0.0 < self.tracking_margin < 1.0:
            raise ValueError("tracking_margin must lie in (0, 1)")
        if self.level_margin_hours <= 0.0:
            raise ValueError("level_margin_hours must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")
        if self.sim_homes < 2:
            raise ValueError("at least two simulated homes are needed")
        if self.sim_days < 3:
            raise ValueError("a simulated record needs a day between its ends")
        if 0.0 in self.shifts_hours:
            raise ValueError("a shift of zero is the primary alignment")
        if 0 not in self.lags_hours:
            raise ValueError("the lags must include zero")

    @property
    def alert_threshold(self) -> float:
        """The baseline's deviation threshold, as the pipeline ships it."""
        return float(BaselineConfig().deviation_threshold)

    @property
    def rule(self) -> timedelta:
        """The silent-home rule's horizon in the run that has it on."""
        return timedelta(hours=self.rule_hours)

    def sim_seeds(self) -> tuple[int, ...]:
        """The simulated homes' seeds, derived from the root."""
        state = np.random.SeedSequence(self.sim_seed_root).generate_state(
            4 * self.sim_homes, dtype=np.uint32
        )
        unique: list[int] = []
        for value in state:
            seed = int(value % 1_000_000)
            if seed not in unique:
                unique.append(seed)
            if len(unique) == self.sim_homes:
                break
        if len(unique) < self.sim_homes:  # pragma: no cover - 4n draws from a million
            raise ValueError("the root did not give enough distinct seeds")
        return tuple(sorted(unique))

    def sim_start(self) -> date:
        """The local date every simulated record begins on."""
        begins: date = HouseholdConfig().start
        return begins

    def to_dict(self) -> dict[str, Any]:
        """Return the full declaration, in a stable serialisable form."""
        baseline, policy, pipeline = BaselineConfig(), AlertPolicy(), PipelineConfig()
        household = HouseholdConfig()
        level = f"{self.confidence:.0%}"
        without = PLANNED_COVERAGE["without_the_outlying_homes"]
        outlying = PLANNED_COVERAGE["with_the_outlying_homes"]
        readings = PLANNED_READINGS
        return {
            "schema": "sleep-mat-protocol/1",
            "name": EXPERIMENT,
            "status": "pre-specified measurement on a real cohort. The days, the "
            "homes, the estimands and the criteria are fixed before any value of "
            "the pipeline is set beside any record of the mat. Nothing in the "
            "pipeline is changed or fitted",
            "question": f"whether the pipeline's `{TRACKED_FEATURE}`, the hours "
            "of a day its state filter gives to sleeping, follows the sleep a "
            "mat under the mattress records, on the days the home's sensors "
            "reported: from one day to the next within a home, and in level",
            "inspected_before": list(self.inspected_before),
            "data": {
                "dataset": PROVENANCE.name,
                "citation": PROVENANCE.citation,
                "licence": PROVENANCE.licence,
                "archive_sha256": ARCHIVE_SHA256,
                "files": dict(FILES),
                "read": ["Activity.csv", "Demographics.csv", "Labels.csv", MAT_FILE],
                "not_read": ["Physiology.csv"],
                "redistributed": "no: the records hold per-home summaries, "
                "never a day's values",
            },
            "the_mat": {
                "file": MAT_FILE,
                "columns": list(MAT_COLUMNS),
                "time_format": MAT_TIME_FORMAT,
                "what_it_is": "an under-the-mattress sleep mat. The dataset "
                "paper says it records heart rate, breathing rate and a sleep "
                "state each minute while a person is in bed",
                "states": list(MAT_STATES),
                "asleep": list(ASLEEP_STATES),
                "clock": f"its timestamps are read as local time in {TIMEZONE}, "
                "as the adapter reads the activity records. The dataset does "
                "not say which clock either file is in. Every day of the "
                "release is in British Summer Time, so no day changes its "
                "offset",
                "standing": "the device's own stages, not validated here. The "
                "mat is the reference because the pipeline never sees it, not "
                "because it is right",
            },
            "pipeline": {
                "what": "`sensor_modeling.datasets.tihm_experiment."
                "run_household`, as the alert-burden run calls it: the contract, "
                "then `BehaviouralSensingPipeline` with default emissions derived "
                "from each home's registry; nothing is fitted",
                "step_minutes": self.tihm_step_minutes,
                "everything_else": "as frozen in "
                "`artifacts/tihm/alert_burden_protocol.json`",
                "runs": {
                    RUN_OFF: "the silent-home rule off: the pipeline as it "
                    "ships, and the run the alert-burden record was made from",
                    RUN_RULE: "`HealthConfig.home_silence_horizon` at "
                    f"{self.rule_hours:g} hours, as in the silent-home results",
                },
                "min_day_coverage": pipeline.min_day_coverage,
                "min_day_observed": pipeline.min_day_observed,
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
                },
                "check": {
                    RUN_OFF: "every mat home must give every count of its row in "
                    f"the published record, `{PUBLISHED_RECORD}`: monitored, "
                    "usable and evaluable days, events, behavioural alerts and "
                    "those about the hours of sleep, alert days, deviating "
                    "days, label days and alerts about the system and the data",
                    RUN_RULE: "every mat home must give its monitored days, the "
                    "days the baseline refused because of the rule and its "
                    "silence alerts as the silent-home record has them at "
                    f"{self.rule_hours:g} hours, `{SILENT_HOME_RECORD}`",
                    "records": {
                        PUBLISHED_RECORD: PUBLISHED_RECORD_SHA256,
                        SILENT_HOME_RECORD: SILENT_HOME_RECORD_SHA256,
                    },
                    "otherwise": "nothing is reported",
                },
            },
            "definitions": {
                "day": f"a local calendar day in {TIMEZONE}, as the pipeline "
                "closes them",
                "pipeline_value": "the expected hours of `sleeping` in the "
                "day's summary: the sum over the day's steps of the filter's "
                "probability of sleeping, times the step. The summary is made "
                "from the day's own estimates, so the step that spans midnight "
                "is in neither day, and a day covers at most 24 hours less one "
                "step",
                "usable_day": "a day whose summary passes "
                "`DailySummary.is_usable` with the pipeline's defaults; these "
                "are the days the baseline is given",
                "silent_day": "a monitored day on which no activity record of "
                "the home falls",
                "mat_sleep": "the day's records whose state is one of "
                f"{', '.join(ASLEEP_STATES)}, in hours: a record is a minute",
                "mat_in_bed": "every record of the day, in hours",
                "mat_observed_day": "a day with at least one record of the "
                "mat, whose day before and day after also have one. A day on "
                "the edge of a stretch of records is left out because part of "
                "its night may be outside the stretch. A day with no record "
                "cannot be told from a day the mat did not work, so it is left "
                "out too",
                "matched_day": "a usable day that is mat-observed and is "
                "neither the home's first nor its last monitored day, which "
                "are partial. In an analysis that leaves silent days out, it is "
                "also not a silent day",
                "included_home": "a mat home with at least "
                f"{self.min_matched_days} matched days in the analysis. "
                "Inclusion is decided in each analysis, and again at each shift "
                "of E10. E6 counts days with a verdict from both baselines "
                "instead; E7 and E8 read every mat home",
                "constant_values": "a home whose values on either source are "
                "all equal on its matched days has no correlation. It counts as "
                "0 in E1 and E5, as not following, and its difference counts "
                "in E2 as any other",
                "difference": "the pipeline's value minus the mat's, in hours",
            },
            "analyses": {
                PRIMARY: "the rule off, silent days left out. The primary "
                "analysis: it asks the question on the days the sensors "
                "reported. A silent day is read as nearly a whole day asleep, "
                "which the alert-burden descriptions measured and the "
                "silent-home rule exists for; kept, it would measure that "
                "again",
                WITH_SILENT_DAYS: "the rule off, every usable day: the pipeline "
                "as it ships and as its baseline sees the days. A description",
                RULE: f"the rule on at {self.rule_hours:g} hours, silent days "
                "left out. The rule also refuses a day that lost any time to a "
                "silence as long as the horizon. A description",
            },
            "estimands": {
                "E1": "the mean over included homes of the within-home "
                "Spearman correlation between the pipeline's value and the "
                "mat's sleep on matched days, in the primary analysis. It is "
                "the rank of a day among the home's own days that a personal "
                "baseline reads, so the correlation is within homes and every "
                "home counts once. It is on the days' values, so a trend or a "
                "weekly rhythm both sources share counts as following",
                "E2": "the mean over included homes of the home's mean "
                "difference from the mat's sleep, in the primary analysis",
                "E3": "E1 and E2 against the mat's hours in bed",
                "E4": "limits of agreement with the mat's sleep and with its "
                "hours in bed: E2 plus and minus 1.96 times the square root of "
                "the variance of the homes' mean differences plus the mean "
                "over homes of the variance of their differences about their "
                "mean",
                "E5": "the mean over included homes of the within-home Pearson "
                "correlation, against each reference",
                "E6": "the deviations a personal baseline gives each source, "
                "in the primary analysis. The mat's sleep on its observed days "
                "is passed in order through an `AdaptiveBaseline` with the "
                "default configuration. The pipeline's baseline was passed its "
                "own usable days, silent days among them, so the two histories "
                "differ. On matched days on which both have a verdict other "
                "than insufficient data: the mean, over homes with at least "
                f"{self.min_matched_days} such days, of the within-home Spearman "
                "correlation between the two deviations; and, pooled, of the "
                "days on which the pipeline's deviation is at or past "
                f"{self.alert_threshold:g} in absolute value, the number on "
                "which the mat's deviation has the same sign, and the number "
                "on which it is also at or past that value",
                "E7": f"the behavioural alerts about `{TRACKED_FEATURE}` raised "
                "in the mat homes with the rule off, counted by the verdict "
                "they were raised for. An alert for an abrupt or a persistent "
                "change on a matched day of the primary analysis is judged two "
                "ways: whether the day's mat sleep is on the same side of the "
                "home's median mat sleep over its matched days as the "
                "pipeline's deviation, and whether the mat baseline's deviation "
                "that day, where E6 gives it a verdict, has the same sign. An "
                "alert for a drift, or on a day that is not matched, is counted "
                "and not judged",
                "E8": "the silent days of the mat homes with the rule off: how "
                "many there are, how many are mat-observed, and on those, the "
                "median pipeline value and the median hours of mat sleep and in "
                "bed",
                "E9": f"E1 to E5 in the analyses `{WITH_SILENT_DAYS}` and " f"`{RULE}`",
                "E10": "E1 and E2 with the mat's clock shifted by "
                + " and by ".join(f"{h:+g}" for h in self.shifts_hours)
                + " hours, every definition otherwise unchanged. +1 stands for "
                "a mat clock in UTC while the activity clock is local, -1 for "
                "the reverse. A shift of an hour moves little of a night across "
                "midnight, so E10 says how much the result depends on the "
                "clock, and cannot find which clock is right",
                "E11": "an alignment of the two clocks from the inputs alone, "
                "which reads no output of the pipeline: at each lag of "
                + ", ".join(f"{h:+d}" for h in self.lags_hours)
                + " hours, the mean over included homes of the within-home "
                "Spearman correlation between the home's activity records in "
                "each local hour of its matched days and the mat's minutes in "
                "bed in the same hour with its clock shifted by the lag. A "
                "person in bed moves little, so the correlation is expected to "
                "be most negative where the clocks agree",
                "E12": "E1 to E5 in the primary analysis without the homes in "
                "which the mat stages less than "
                f"{STAGED_ASLEEP_SHARE:g} of the minutes in bed on its observed "
                "days as asleep. From the mat's file alone these are "
                + ", ".join(STAGED_AWAKE_HOMES)
                + "; the run refuses to go on if the file gives others",
                "S1": "the simulated reference: E1, E2, E4 and E5 against the "
                "simulator's true hours of sleep on simulated homes",
            },
            "criteria": {
                "C1_the_pipeline_follows_the_mat": "the primary criterion, on "
                f"E1. Follows when the interval lies above {self.tracking_margin:g}; "
                f"does not follow when it lies below {self.tracking_margin:g}; "
                "inconclusive otherwise",
                "C2_the_pipeline_agrees_in_level": "secondary, on E2, and read "
                "whatever C1 reads. Agrees when the interval lies inside "
                f"(-{self.level_margin_hours:g}, +{self.level_margin_hours:g}) "
                "hours; does not agree when it lies wholly outside "
                f"[-{self.level_margin_hours:g}, +{self.level_margin_hours:g}]; "
                "inconclusive otherwise",
                "estimable": "a criterion is not estimable when fewer than "
                "three homes are included",
                "multiplicity": "C1 is the only primary criterion. The two "
                "criteria are not adjusted for each other, and E3 to E12 and S1 "
                "are descriptions with intervals",
            },
            "margins": {
                "tracking": f"{self.tracking_margin:g}, a declared convention. "
                "If within a home the two sources' days were jointly Gaussian "
                "and stationary with correlation r, a day the pipeline put z "
                "standard deviations from the home's centre would be expected "
                "r times z from it on the mat, and at 0.5 a deviation at the "
                f"baseline's threshold of {self.alert_threshold:g} would stand "
                "for one of 1.5 in sleep. The baseline reads each day against "
                "a rolling, weekday-aware reference and not the whole period, "
                "so this is a guide, not a derivation. The margin is on the "
                "mean over homes of a finite-sample Spearman correlation: in the "
                "planning trials, when every home had a Gaussian correlation of "
                f"0.5 its true value was {readings['c1_truth_every_home_at_0_5']:.2f}, "
                "and when homes were spread about 0.5 by 0.6 on Fisher's scale "
                f"{readings['c1_truth_homes_spread_0_6_about_0_5']:.2f}",
                "level": f"{self.level_margin_hours:g} hour, a declared "
                "convention of the order of the changes the detection studies "
                "inject: the step change has the resident wake 1.6 hours "
                "earlier, with 1.2 more night-time bathroom trips. A bias that "
                "size changes what a day's hours say about a person. It does "
                "not touch a personal baseline, which reads each home against "
                "itself, so C2 is secondary",
            },
            "expected_readings": {
                "C2": "C2 can say the pipeline does not agree in level and can "
                "hardly say it agrees. With no bias and the homes' mean "
                "differences spread by an hour, the planning read agrees in "
                f"{readings['c2_agrees_no_bias_spread_1']:.0%} of studies; "
                "spread by two hours, in "
                f"{readings['c2_agrees_no_bias_spread_2']:.0%}; and with the "
                "three homes the mat stages as mostly awake four hours further "
                "off, almost never",
                "C1": "with those three homes not following at all and the "
                "rest at a Gaussian correlation of 0.6, the true value of E1 "
                f"was {readings['c1_truth_outlying_homes_rest_at_0_6']:.2f} in "
                "the planning: they alone can hold C1 below its margin, which "
                "is why E12 is declared",
            },
            "intervals": {
                "unit": "homes",
                "method": "for a mean over homes, a Student t interval with "
                f"one fewer degrees of freedom than homes, at {level}: the "
                "mean plus and minus the quantile times the standard deviation "
                "of the home values over the square root of their number",
                "why": "in the planning trials, synthetic studies with the mat "
                "homes' numbers of mat-observed days, a percentile bootstrap "
                "over homes covered "
                f"{_span(without['percentile_bootstrap'])} of the time and the "
                f"t interval {_span(without['student_t'])}. With the three "
                "outlying homes both covered more, "
                f"{_span(outlying['percentile_bootstrap'])} and "
                f"{_span(outlying['student_t'])}, since their difference was "
                "fixed and not drawn",
                "planning_days": "the numbers of mat-observed days with both "
                "neighbours, from the mat alone: upper bounds on the matched "
                "days",
                "counts": "a pooled count of days or alerts in E6 or E7 is given "
                "with no interval, since the days of one home are not "
                "independent",
                "planning_record": PLANNING_RECORD,
                "planning_record_sha256": self.planning_record_sha256,
                "chosen": PLANNED_CHOICE,
            },
            "simulated_reference": {
                "standing": "a reference, not a test: what the estimands give "
                "when the observation model is the one that made the data",
                "evidence": "simulated: every home comes from this repository's "
                "simulator",
                "homes": self.sim_homes,
                "seed_root": self.sim_seed_root,
                "seeds": "the first distinct values of "
                "`numpy.random.SeedSequence(seed_root).generate_state(4 * "
                "homes, uint32)` modulo 1,000,000, sorted",
                "days": self.sim_days,
                "start": self.sim_start().isoformat(),
                "timezone": str(household.tz),
                "household": "`HouseholdConfig` defaults apart from days and "
                "seed; nothing is injected",
                "sensors": list(event_sensors()),
                "why_these_sensors": "the sensors that report on no cadence, "
                "as in TIHM",
                "delivery": "`sensor_modeling.simulation.faults.degrade` with "
                "its default configuration",
                "step_minutes": self.sim_step_minutes,
                "truth": "the hours of each local day covered by the "
                "simulator's episodes of `sleeping`, measured in elapsed time, "
                "so the night the clocks change is not given an hour it did "
                "not have",
                "matched_day": "a usable day other than the record's first "
                "and last, which are partial",
            },
            "reporting": "every estimand, criterion, analysis and home, whatever "
            "it shows; no margin, rule or day is changed after the comparison is "
            "made. What was run and read between the freeze and the run is "
            f"listed in `{BETWEEN_FILE}`",
            "what_this_cannot_show": [
                "whether the mat's stages are right. A disagreement between "
                "the two may be the mat's",
                "anything about the 39 homes without a mat",
                "how a day with no mat record went: the person may have slept "
                "out of bed, been away, or the mat may not have worked",
            ],
            "acknowledgement": "TIHM is by Palermo et al., Scientific Data 10, "
            "606 (2023), under CC BY 4.0. Surrey and Borders Partnership NHS "
            "Foundation Trust and Howz are acknowledged, as the dataset asks",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_protocol() -> SleepMatProtocol:
    """The protocol as declared."""
    return SleepMatProtocol()


def write_protocol(protocol: SleepMatProtocol, path: Path) -> str:
    """Write the declaration with its digest and the code it was frozen against.

    The file records, under ``at_freeze``, the digest of every pinned source,
    the versions of the numerical libraries and every default setting, so a
    run can say what, if anything, changed since.
    """
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


def check_frozen_protocol(protocol: SleepMatProtocol, path: Path) -> str:
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
