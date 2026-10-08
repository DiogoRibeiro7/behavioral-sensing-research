"""The sleep-mat protocol: do the pipeline's hours of sleep follow a sleep mat.

Every alerting study in this repository counts its detections on one feature,
``sleeping_hours``: the hours of the day the pipeline's state filter gives to
``sleeping``, inferred from motion and door sensors. Nothing so far has set
that number beside a measurement of sleep. The TIHM run showed that a day with
no sensor event is read as nearly a whole day asleep, which says the number
can be wrong; it does not say how far it follows sleep on an ordinary day.

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
``artifacts/sleep_mat/sleep_mat_protocol.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..alerts.alert import AlertPolicy
from ..baseline.adaptive import BaselineConfig
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

#: The file the mat's records are in, and its columns.
MAT_FILE = "Sleep.csv"
MAT_COLUMNS = (
    "patient_id",
    "date",
    "state",
    "heart_rate",
    "respiratory_rate",
    "snoring",
)

#: The stages the device reports, and those counted as asleep.
MAT_STATES = ("AWAKE", "LIGHT", "DEEP", "REM")
ASLEEP_STATES = ("LIGHT", "DEEP", "REM")

#: The two references, by what is counted on a day.
SLEEP = "sleep"
IN_BED = "in_bed"
REFERENCES = (SLEEP, IN_BED)

#: The conditions the pipeline is run under on TIHM.
OFF = "off"
RULE = "rule_12h"
CONDITIONS = (OFF, RULE)

#: The three readings of a criterion stated against a margin.
FOLLOWS = "follows"
DOES_NOT_FOLLOW = "does_not_follow"
AGREES = "agrees"
DOES_NOT_AGREE = "does_not_agree"
INCONCLUSIVE = "inconclusive"
NOT_ESTIMABLE = "not_estimable"

#: The published alert-burden record. With the rule off, every mat home must
#: give its monitored days and behavioural alerts before anything is reported.
PUBLISHED_RECORD = "artifacts/tihm/tihm-alert-burden.json"

#: The planning record the interval method rests on, its digest, and the
#: lowest and highest coverage each interval reached in it, by method.
PLANNING_RECORD = "artifacts/sleep_mat/sleep-mat-planning.json"
PLANNING_RECORD_SHA256 = (
    "3776bf0061d9c4c140a9e9c9b6768363a4a0e33b17f5085a63b0b9f99b119e51"
)
PLANNED_COVERAGE = {
    "percentile_bootstrap": (0.897, 0.929),
    "student_t": (0.932, 0.961),
}
PLANNED_CHOICE = "student_t"


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
        The silent-home rule's horizon in the condition that has it on.
    min_matched_days
        The matched days a home needs to enter an estimand.
    tracking_margin
        The within-home rank correlation C1 is judged against.
    level_margin_hours
        The bias in hours C2 is judged against.
    confidence
        The level of every interval.
    shifts_hours
        The shifts of the mat's clock the alignment description is made at.
    sim_seed_root, sim_homes, sim_days, sim_step_minutes
        The simulated reference.
    planning_record_sha256
        The planning record the interval method was chosen from.
    """

    tihm_step_minutes: int = 10
    rule_hours: float = 12.0
    min_matched_days: int = 14
    tracking_margin: float = 0.5
    level_margin_hours: float = 1.0
    confidence: float = 0.95
    shifts_hours: tuple[float, ...] = (-1.0, 1.0)
    sim_seed_root: int = 20261008
    sim_homes: int = 100
    sim_days: int = 84
    sim_step_minutes: int = 10
    planning_record_sha256: str = PLANNING_RECORD_SHA256

    #: What had been seen when the protocol was written.
    inspected_before: tuple[str, ...] = (
        "the mat's file alone, for its structure: its columns, its four "
        "stages, its 461,423 records in 17 homes from 2019-04-01 to "
        "2019-06-30, one record a minute with no duplicate minute, and no "
        "missing value",
        "the mat's coverage alone: the days each home has records on, the "
        "days whose neighbouring days also have records, and each home's "
        "median hours in bed and asleep on a day. 835 home-days have a "
        "record, 771 of them with records on the day before and the day "
        "after, and 14 homes have at least 14 such days. In three homes the "
        "median hours asleep are under five while the median hours in bed "
        "are near eight or more",
        "beside those day counts, each mat home's monitored and usable day "
        "counts in the published alert-burden record. That compares how many "
        "days each source has, not what either says about them",
        "the pipeline's output on TIHM had been read in the earlier studies: "
        "its alerts, its verdicts, and the median 23.8 hours of sleep it "
        "infers on days with no sensor event. No value of the pipeline had "
        "been set beside any record of the mat, for any home or day",
        "the dataset paper's description of the mat: per-minute heart rate, "
        "breathing rate and sleep state while a person is in bed. It does not "
        "say whether any file's timestamps are local or UTC",
        "the interval method was chosen on synthetic data, in the planning "
        "record, before this protocol was frozen",
        "the code was rehearsed end to end on data that is not the "
        "comparison's: one TIHM home without a mat, run with the rule off and "
        "on and paired with a mat of random minutes generated for it, and one "
        "simulated home of ten days from seed 999, which is not a seed of the "
        "reference. Their run times and that every step completed were read, "
        "and none of their values",
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

    @property
    def alert_threshold(self) -> float:
        """The baseline's deviation threshold, as the pipeline ships it."""
        return float(BaselineConfig().deviation_threshold)

    @property
    def rule(self) -> timedelta:
        """The silent-home rule's horizon in the condition that has it on."""
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
        return {
            "schema": "sleep-mat-protocol/1",
            "name": EXPERIMENT,
            "status": "pre-specified measurement on a real cohort. The days, the "
            "homes, the estimands and the criteria are fixed before any value of "
            "the pipeline is set beside any record of the mat. Nothing in the "
            "pipeline is changed or fitted",
            "question": f"whether the pipeline's `{TRACKED_FEATURE}`, the hours "
            "of a day its state filter gives to sleeping, follows the sleep a "
            "mat under the mattress records: from one day to the next within a "
            "home, and in level",
            "inspected_before": list(self.inspected_before),
            "data": {
                "dataset": PROVENANCE.name,
                "citation": PROVENANCE.citation,
                "licence": PROVENANCE.licence,
                "archive_sha256": ARCHIVE_SHA256,
                "files": dict(FILES),
                "read": ["Activity.csv", "Demographics.csv", "Labels.csv", MAT_FILE],
                "not_read": ["Physiology.csv"],
                "redistributed": "no",
            },
            "the_mat": {
                "file": MAT_FILE,
                "columns": list(MAT_COLUMNS),
                "what_it_is": "an under-the-mattress sleep mat. The dataset "
                "paper says it records heart rate, breathing rate and a sleep "
                "state each minute while a person is in bed",
                "states": list(MAT_STATES),
                "asleep": list(ASLEEP_STATES),
                "clock": f"its timestamps are read as local time in {TIMEZONE}, "
                "as the adapter reads the activity records. The dataset does "
                "not say which clock either file is in. The alignment "
                "description shifts the mat's clock to show what an offset "
                "would change",
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
                "conditions": {
                    OFF: "the silent-home rule off: the pipeline as it ships, "
                    "and the run the alert-burden record was made from",
                    RULE: "`HealthConfig.home_silence_horizon` at "
                    f"{self.rule_hours:g} hours, as described in the "
                    "silent-home results",
                },
                "primary": OFF,
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
                "check": "with the rule off, every mat home must give the "
                "monitored days and the behavioural alerts of the published "
                f"record, `{PUBLISHED_RECORD}`, or nothing is reported",
            },
            "definitions": {
                "day": f"a local calendar day in {TIMEZONE}, as the pipeline "
                "closes them",
                "pipeline_value": "the expected hours of `sleeping` in the "
                "day's summary: the sum over the day's steps of the filter's "
                "probability of sleeping, times the step",
                "usable_day": "a day whose summary passes "
                "`DailySummary.is_usable` with the pipeline's defaults; these "
                "are the days the baseline is given",
                "mat_sleep": "the day's records whose state is one of "
                f"{', '.join(ASLEEP_STATES)}, in hours: a record is a minute",
                "mat_in_bed": "every record of the day, in hours",
                "mat_observed_day": "a day with at least one record of the "
                "mat, whose day before and day after also have one. A day on "
                "the edge of a stretch of records is left out because part of "
                "its night may be outside the stretch. A day with no record "
                "cannot be told from a day the mat did not work, so it is left "
                "out too",
                "matched_day": "a usable day that is mat-observed",
                "included_home": "a home with a mat and at least "
                f"{self.min_matched_days} matched days. The same homes enter "
                "every estimand of a condition, apart from E6, which needs "
                "verdicts",
                "difference": "the pipeline's value minus the mat's, in hours",
                "silent_day": "a monitored day on which no activity record of "
                "the home falls",
            },
            "estimands": {
                "E1": "the mean over included homes of the within-home "
                "Spearman correlation between the pipeline's value and the "
                "mat's sleep on matched days. It is the rank of a day among "
                "the home's own days that a personal baseline reads, so the "
                "correlation is within homes, and every home counts once",
                "E2": "the mean over included homes of the home's mean "
                "difference from the mat's sleep",
                "E3": "E1 and E2 against the mat's hours in bed",
                "E4": "limits of agreement with the mat's sleep and with its "
                "hours in bed: E2 plus and minus 1.96 times the square root of "
                "the variance of the homes' mean differences plus the mean "
                "over homes of the variance of their differences about their "
                "mean",
                "E5": "the mean over included homes of the within-home Pearson "
                "correlation, against each reference",
                "E6": "the deviations a personal baseline gives each source. "
                "The mat's sleep on its observed days is passed in order "
                "through an `AdaptiveBaseline` with the default configuration, "
                "as the pipeline passes its usable days. On matched days on "
                "which both have a verdict other than insufficient data: the "
                "mean over homes with at least "
                f"{self.min_matched_days} such days of the within-home Spearman "
                "correlation between the two deviations; and, pooled, of the "
                "days on which the pipeline's deviation is at or past "
                f"{self.alert_threshold:g} in absolute value, the share on "
                "which the mat's deviation has the same sign, and the share on "
                "which it is also at or past that value with the same sign",
                "E7": f"the behavioural alerts about `{TRACKED_FEATURE}` raised "
                "in the mat homes: how many fall on matched days, and of those, "
                "how many on a day whose mat sleep is on the same side of the "
                "home's median mat sleep over its matched days as the pipeline's "
                "deviation that day. An alert on a day that is not matched is "
                "counted and not judged",
                "E8": "the silent days of the mat homes: how many there are, how "
                "many are mat-observed, and on those, the median pipeline value "
                "and the median hours of mat sleep and in bed",
                "E9": f"E1 to E5 with the silent-home rule on, `{RULE}`",
                "E10": "E1 and E2 with the mat's clock shifted by "
                + " and by ".join(f"{h:+g}" for h in self.shifts_hours)
                + " hours, every definition otherwise unchanged",
                "S1": "the simulated reference: E1, E2, E4 and E5 against the "
                "simulator's true hours of sleep on simulated homes",
            },
            "criteria": {
                "C1_the_pipeline_follows_the_mat": "on E1, with the rule off. "
                f"Follows when the interval lies above {self.tracking_margin:g}; "
                f"does not follow when it lies below {self.tracking_margin:g}; "
                "inconclusive otherwise",
                "C2_the_pipeline_agrees_in_level": "on E2, with the rule off. "
                "Agrees when the interval lies inside "
                f"(-{self.level_margin_hours:g}, +{self.level_margin_hours:g}) "
                "hours; does not agree when it lies wholly outside "
                f"[-{self.level_margin_hours:g}, +{self.level_margin_hours:g}]; "
                "inconclusive otherwise",
                "estimable": "a criterion is not estimable when fewer than "
                "three homes are included",
            },
            "margins": {
                "tracking": f"{self.tracking_margin:g}. If a day's values on "
                "the two sources were jointly Gaussian within a home with "
                "correlation r, a day the pipeline puts z standard deviations "
                "from the home's centre would be expected r times z from it on "
                "the mat. At 0.5 a deviation at the baseline's threshold of "
                f"{self.alert_threshold:g} stands for one of 1.5 in sleep: the "
                "threshold would mean half of what it says. Spearman's "
                "correlation is used for its robustness, and for jointly "
                "Gaussian values 0.5 in Pearson's is 0.48 in Spearman's",
                "level": f"{self.level_margin_hours:g} hour, of the order of "
                "the changes the detection studies inject: the step change has "
                "the resident wake 1.6 hours earlier. A bias that size changes "
                "what a day's hours say about a person. It does not touch a "
                "personal baseline, which reads each home against itself, so "
                "C2 is secondary to C1",
            },
            "intervals": {
                "unit": "homes",
                "method": "for a mean over homes, a Student t interval with "
                "one fewer degrees of freedom than homes, at "
                f"{level}: the mean plus and minus the quantile times the "
                "standard deviation of the home values over the square root "
                "of their number",
                "why": "in the planning trials, synthetic studies with these "
                "homes' numbers of days, a percentile bootstrap over homes "
                f"covered {_span(PLANNED_COVERAGE['percentile_bootstrap'])} of "
                "the time, and the t interval "
                f"{_span(PLANNED_COVERAGE['student_t'])}",
                "shares": "a pooled count of days in E6 or E7 is given with "
                "no interval, since the days of one home are not independent",
                "planning_record": PLANNING_RECORD,
                "planning_record_sha256": self.planning_record_sha256,
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
                "simulator's episodes of `sleeping`",
                "matched_day": "a usable day other than the record's first "
                "and last, which are partial",
            },
            "reporting": "every estimand, criterion, condition and home, whatever "
            "it shows; no margin, rule or day is changed after the comparison is "
            "made",
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
    """Write the declaration with its digest; return the digest."""
    payload = {
        **protocol.to_dict(),
        "protocol_sha256": protocol.sha256(),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after scoring begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
