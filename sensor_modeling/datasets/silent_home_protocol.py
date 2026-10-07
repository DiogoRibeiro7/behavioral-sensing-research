"""The silent-home protocol: does the opt-in rule do what it was built for.

The TIHM run found that a home of event sensors which stops reporting is read
as a home asleep, and that such days raise behavioural alerts.
``HealthConfig.home_silence_horizon`` was built from that finding, so the homes
that showed the problem cannot also be the evidence that the rule solves it.

This protocol has two parts, and they have different standing.

**The simulator is the test.** Simulated homes are reduced to their event
sensors, so that, as in TIHM, nothing in them reports on a cadence. A
whole-home outage, a real behavioural change and both together are injected at
known times, and every home is run with the rule off and on. The estimands,
the criteria and the homes are fixed here, before any simulated home has been
run with the rule on.

**TIHM is a description.** The rule is also run on the 56 TIHM homes, at the
same horizons, to say what it does where the problem was found. That is not a
test of anything: the rule was designed from those homes.

Simulated evidence is evidence about the simulator. Its homes are never silent
for long unless a fault makes them so, so it can show that the rule removes
what an outage causes, and it cannot show how often a real home is silent for
a harmless reason. :func:`declared_protocol` is frozen in
``artifacts/silent_home/silent_home_protocol.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..alerts.alert import AlertPolicy
from ..baseline.adaptive import BaselineConfig
from ..external.tihm import ARCHIVE_SHA256, FILES, PROVENANCE
from ..online.pipeline import PipelineConfig
from ..simulation.household import HouseholdConfig, build_registry

#: The records' names and the layout of their results.
EXPERIMENT = "silent-home"
RESULT_SCHEMA = "silent-home/1"
TIHM_EXPERIMENT = "silent-home-tihm"
TIHM_SCHEMA = "silent-home-tihm/1"

#: The condition with the rule off, which is the pipeline's default.
OFF = "off"

#: The arms that are run through the pipeline, and what each injects.
STABLE = "stable"
OUTAGE = "outage"
SHORT_OUTAGE = "outage_short"
CHANGE = "change"
CHANGE_AFTER_OUTAGE = "change_after_outage"
PIPELINE_ARMS = (STABLE, OUTAGE, SHORT_OUTAGE, CHANGE, CHANGE_AFTER_OUTAGE)

#: The arm that only the fleet check reads: the long outage in every home at
#: once. To one home it is the same as its own outage, so it is not run
#: through the pipeline.
COMMON = "outage_common"

#: The arms the fleet check reads.
FLEET_ARMS = (STABLE, OUTAGE, COMMON)

#: The two groups a home's own outage falls in, by whether the primary horizon
#: has passed before the day on which the outage begins is closed.
IN_TIME = "seen_before_the_day_closes"
LATE = "seen_after_the_day_closes"

#: The feature a detection is counted on, as in the detection study.
TRACKED_FEATURE = "sleeping_hours"

#: The subject of the alert the rule raises.
SILENCE_SUBJECT = "home_silence"

#: What the published TIHM run recorded, which the description must reproduce
#: with the rule off before it reports anything.
TIHM_PUBLISHED = {"monitored_days": 2850, "behavioural_alerts": 183}


def event_sensors() -> tuple[str, ...]:
    """The simulated deployment's sensors that report on no cadence."""
    return tuple(
        spec.sensor_id for spec in build_registry() if spec.expected_interval is None
    )


def condition_name(hours: float) -> str:
    """The name of the condition with the rule on at a horizon of *hours*."""
    return f"h{hours:g}"


@dataclass(frozen=True)
class SilentHomeProtocol:
    """Every definition of the silent-home evaluation.

    Attributes
    ----------
    seed_root, homes
        The simulated homes: their seeds are drawn from the root.
    days, step_minutes
        Length of each simulated record and the pipeline's step.
    horizons_hours
        The horizons the rule is run at. The first is the primary one.
    outage_first_day, outage_last_day, outage_hours
        A home's own long outage: the days it may begin on, and how long no
        sensor reports. The day and the hour are drawn for each home.
    short_outage_hours
        The short outage, from the same moment: shorter than every horizon.
    common_day, common_hour
        The local day and hour at which the outage every home shares begins.
    change_day, change_sleep_delta_hours, change_night_bathroom_extra
        The injected behavioural change, the detection study's step.
    max_delay_days
        How long after the change a detection still counts.
    follow_days
        How long after an outage ends its alerts are still counted.
    margin_alerts, margin_recall, informative_recall, reported_share
        The criteria's margins.
    fleet_fraction, fleet_min_homes, fleet_every_hours
        The fleet check.
    resamples, confidence, seed
        The bootstrap over homes.
    tihm_step_minutes
        The pipeline's step on TIHM, the published run's.
    """

    seed_root: int = 20261007
    homes: int = 100
    days: int = 84
    step_minutes: int = 15
    horizons_hours: tuple[float, ...] = (12.0, 24.0)
    outage_first_day: int = 28
    outage_last_day: int = 45
    outage_hours: float = 60.0
    short_outage_hours: float = 8.0
    common_day: int = 42
    common_hour: int = 4
    change_day: int = 56
    change_sleep_delta_hours: float = 1.6
    change_night_bathroom_extra: float = 1.2
    max_delay_days: float = 21.0
    follow_days: int = 28
    margin_alerts: float = 0.25
    margin_recall: float = 0.10
    informative_recall: float = 0.20
    reported_share: float = 0.95
    fleet_fraction: float = 0.6
    fleet_min_homes: int = 3
    fleet_every_hours: float = 1.0
    resamples: int = 5000
    confidence: float = 0.95
    seed: int = 0
    tihm_step_minutes: int = 10

    #: What had been seen when the protocol was written.
    inspected_before: tuple[str, ...] = (
        "the TIHM alert-burden run and the descriptions made after it had been "
        "read: the silent days, the calendar, and the replay of the baseline "
        "with silent days left out. The rule was designed from them, which is "
        "why nothing on TIHM here is a test",
        "the rule had been exercised in unit tests on a hand-built home of "
        "three event sensors and on hand-built fleets. No simulated household "
        "had been run with the rule on",
        "the rule was changed once before this protocol was frozen, from "
        "reading the code and not from any run: a day that lost any time to a "
        "silence is refused by the baseline, where at first only a mostly "
        "silent day was",
        "one simulated household, seed 999, which is not a home of this "
        "protocol, had been run with the rule off and with its event sensors "
        "alone to measure run time; its numbers of observations and of steps "
        "were read, and none of its alerts, verdicts or summaries",
        "the simulator's source had been read, including the rates at which "
        "its sensors fire. How long a simulated or a TIHM home goes without an "
        "event had not been measured; the horizons were declared, not "
        "estimated",
    )

    def __post_init__(self) -> None:
        """Validate the declaration."""
        if self.homes < 2:
            raise ValueError("at least two homes are needed to resample")
        if not self.horizons_hours or any(h <= 0 for h in self.horizons_hours):
            raise ValueError("at least one positive horizon is required")
        if len(set(self.horizons_hours)) != len(self.horizons_hours):
            raise ValueError("horizons must be distinct")
        if self.short_outage_hours >= min(self.horizons_hours):
            raise ValueError("the short outage must be shorter than every horizon")
        if self.outage_hours <= max(self.horizons_hours):
            raise ValueError("the long outage must outlast every horizon")
        if not 0 <= self.outage_first_day <= self.outage_last_day:
            raise ValueError("the outage days are not in order")
        if not 0 <= self.common_hour < 24:
            raise ValueError("common_hour must be an hour of the day")
        # The latest moment an outage of a home's own can end, in days.
        latest_end = self.outage_last_day + (23.0 + self.outage_hours) / 24.0
        common_end = self.common_day + (self.common_hour + self.outage_hours) / 24.0
        if self.change_day <= latest_end:
            raise ValueError("the change must follow every outage")
        if (
            max(
                latest_end + self.follow_days,
                common_end,
                self.change_day + 1 + self.max_delay_days,
            )
            > self.days
        ):
            raise ValueError("the record is too short for its windows")

    @property
    def primary_hours(self) -> float:
        """The horizon the criteria are stated at."""
        return self.horizons_hours[0]

    @property
    def conditions(self) -> tuple[str, ...]:
        """The rule off, then on at each horizon, the primary first."""
        return (OFF, *(condition_name(hours) for hours in self.horizons_hours))

    def horizon(self, condition: str) -> timedelta | None:
        """The horizon a condition runs the rule at; ``None`` with it off."""
        if condition == OFF:
            return None
        for hours in self.horizons_hours:
            if condition_name(hours) == condition:
                return timedelta(hours=hours)
        raise KeyError(f"no condition '{condition}'")

    def runs(self) -> tuple[tuple[str, str], ...]:
        """Every arm and condition that is run through the pipeline.

        Every arm is run with the rule off and at the primary horizon. The
        other horizons are run on the stable arm and the long outage only:
        they say what depends on the horizon, and nothing else is asked of
        them.
        """
        primary = condition_name(self.primary_hours)
        pairs = [(arm, c) for arm in PIPELINE_ARMS for c in (OFF, primary)]
        pairs += [
            (arm, condition_name(hours))
            for hours in self.horizons_hours[1:]
            for arm in (STABLE, OUTAGE)
        ]
        return tuple(pairs)

    def study_seeds(self) -> tuple[int, ...]:
        """The homes' seeds, derived from the root so the set is auditable."""
        state = np.random.SeedSequence(self.seed_root).generate_state(
            4 * self.homes, dtype=np.uint32
        )
        unique: list[int] = []
        for value in state:
            seed = int(value % 1_000_000)
            if seed not in unique:
                unique.append(seed)
            if len(unique) == self.homes:
                break
        if len(unique) < self.homes:  # pragma: no cover - 4n draws from a million
            raise ValueError("the root did not give enough distinct seeds")
        return tuple(sorted(unique))

    def outage_starts(self) -> dict[int, tuple[int, int]]:
        """The day and the local hour each home's own outage begins at."""
        rng = np.random.default_rng([self.seed_root, 1])
        days = rng.integers(
            self.outage_first_day, self.outage_last_day + 1, size=self.homes
        )
        hours = rng.integers(0, 24, size=self.homes)
        return {
            seed: (int(day), int(hour))
            for seed, day, hour in zip(self.study_seeds(), days, hours)
        }

    def group(self, seed: int) -> str:
        """Whether the primary horizon passes before the outage's day closes.

        The rule sees a silence once it has lasted the horizon, counted from
        the last observation, which is at or before the outage's beginning. So
        an outage that begins at least a horizon before midnight is seen
        before its first day is closed, and one that begins later may not be.
        """
        _, hour = self.outage_starts()[seed]
        return IN_TIME if hour + self.primary_hours < 24.0 else LATE

    def start(self) -> date:
        """The local date every simulated record begins on."""
        begins: date = HouseholdConfig().start
        return begins

    def local(self, day: int, hour: float) -> datetime:
        """The instant of a local hour on a day of the record, in UTC."""
        zone = HouseholdConfig().tz
        naive = datetime.combine(self.start() + timedelta(days=day), time.min)
        return (
            (naive + timedelta(hours=hour))
            .replace(tzinfo=zone)
            .astimezone(timezone.utc)
        )

    def outage(self, seed: int) -> tuple[datetime, datetime]:
        """The window of a home's own long outage."""
        day, hour = self.outage_starts()[seed]
        begin = self.local(day, hour)
        return begin, begin + timedelta(hours=self.outage_hours)

    def short_outage(self, seed: int) -> tuple[datetime, datetime]:
        """The window of a home's short outage, from the same moment."""
        begin, _ = self.outage(seed)
        return begin, begin + timedelta(hours=self.short_outage_hours)

    def common_outage(self) -> tuple[datetime, datetime]:
        """The window of the outage every home shares."""
        begin = self.local(self.common_day, self.common_hour)
        return begin, begin + timedelta(hours=self.outage_hours)

    def window(self, seed: int, arm: str) -> tuple[datetime, datetime]:
        """The window in which an outage arm's alerts are counted for a home."""
        if arm == OUTAGE:
            begin, end = self.outage(seed)
        elif arm == SHORT_OUTAGE:
            begin, end = self.short_outage(seed)
        else:
            raise KeyError(f"'{arm}' has no outage of the home's own")
        return begin, end + timedelta(days=self.follow_days)

    def change_begins(self) -> datetime:
        """Local midnight at the start of the day the change takes effect."""
        return self.local(self.change_day, 0.0)

    def detection_window(self) -> tuple[datetime, datetime]:
        """The window in which an alert counts as detecting the change.

        The pipeline raises behavioural alerts when a day closes. The window
        opens when the change day closes, so that the alert raised at the
        close of the day before, which holds none of the change, is outside
        it, and it is closed on the left and open on the right.
        """
        begin = self.local(self.change_day + 1, 0.0)
        return begin, begin + timedelta(days=self.max_delay_days)

    def to_dict(self) -> dict[str, Any]:
        """Return the full declaration, in a stable serialisable form."""
        baseline, policy, pipeline = BaselineConfig(), AlertPolicy(), PipelineConfig()
        household = HouseholdConfig()
        primary = condition_name(self.primary_hours)
        common_begin, common_end = self.common_outage()
        starts = self.outage_starts()
        groups = [self.group(seed) for seed in self.study_seeds()]
        return {
            "schema": "silent-home-protocol/1",
            "name": EXPERIMENT,
            "status": "pre-specified on simulated homes; descriptive on TIHM. "
            "The simulator's homes, estimands and criteria are fixed before any "
            "simulated home is run with the rule on. Nothing on TIHM is a test",
            "question": "whether, when a home of event sensors stops "
            "reporting, the opt-in rule keeps the silence out of the personal "
            "baseline and reports it, without changing anything when nothing "
            "is wrong and without costing the detection of a real change",
            "inspected_before": list(self.inspected_before),
            "the_rule": {
                "setting": "HealthConfig.home_silence_horizon",
                "default": "off",
                "conditions": {
                    OFF: "the rule off: the pipeline's default",
                    **{
                        condition_name(hours): f"the rule on at {hours:g} hours"
                        for hours in self.horizons_hours
                    },
                },
                "primary": primary,
                "horizons": "declared, not estimated from any home",
            },
            "simulator": {
                "evidence": "simulated: every home comes from this repository's "
                "simulator, so every result is a statement about the simulator",
                "homes": self.homes,
                "seed_root": self.seed_root,
                "seeds": "the first distinct values of "
                "`numpy.random.SeedSequence(seed_root).generate_state(4 * "
                "homes, uint32)` modulo 1,000,000, sorted",
                "days": self.days,
                "start": self.start().isoformat(),
                "timezone": str(household.tz),
                "household": "`HouseholdConfig` defaults apart from days, "
                "seed and the injected change",
                "sensors": list(event_sensors()),
                "sensors_left_out": [
                    spec.sensor_id
                    for spec in build_registry()
                    if spec.expected_interval is not None
                ],
                "why_left_out": "they report on a cadence, so their silence is "
                "already evidence of a failure. What is left is a stream of "
                "activations, as in TIHM",
                "delivery": "every arm passes through "
                "`sensor_modeling.simulation.faults.degrade` with no loss, "
                "lateness or duplication. An outage is a dropout fault on every "
                "sensor over the same window",
                "pipeline": {
                    "step_minutes": self.step_minutes,
                    "what": "`BehaviouralSensingPipeline` over the event "
                    "sensors' registry, with default emissions derived from it; "
                    "nothing is fitted",
                    "features": [f"{s.value}_hours" for s in pipeline.features],
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
                },
                "pairing": "every arm and condition is run on every home, so a "
                "home with a fault differs from the same home without it in the "
                "fault alone, and the rule on differs from the rule off in the "
                "rule alone",
                "arms": {
                    STABLE: "nothing is injected",
                    OUTAGE: f"no sensor of the home reports for "
                    f"{self.outage_hours:g} hours. Each home has its own "
                    f"outage: the day, between day {self.outage_first_day} and "
                    f"day {self.outage_last_day}, and the local hour it begins "
                    "at, from 0 to 23, are drawn from "
                    "`numpy.random.default_rng([seed_root, 1])`, the days first",
                    SHORT_OUTAGE: "no sensor of the home reports for "
                    f"{self.short_outage_hours:g} hours from the same moment, "
                    "which is shorter than every horizon",
                    CHANGE: "the detection study's step change from day "
                    f"{self.change_day}: the resident wakes "
                    f"{self.change_sleep_delta_hours:g} hours earlier, with "
                    f"{self.change_night_bathroom_extra:g} more night-time "
                    "bathroom trips expected a night",
                    CHANGE_AFTER_OUTAGE: "the home's own long outage and then "
                    "the change, which begins at least a week after the outage "
                    "has ended",
                    COMMON: "no sensor of any home reports from "
                    f"{common_begin.isoformat()} to {common_end.isoformat()}: "
                    f"{self.outage_hours:g} hours from "
                    f"{self.common_hour:02d}:00 local on day {self.common_day}. "
                    "Only the fleet check reads this arm. To one home it is the "
                    "same as an outage of its own, so it is not run through the "
                    "pipeline",
                },
                "runs": [list(pair) for pair in self.runs()],
                "outage_groups": {
                    "why": "the rule sees a silence once it has lasted the "
                    "horizon. If the day on which an outage begins is closed "
                    "before that, the day is closed as it would be without the "
                    "rule, with the silent hours read as observed. The hour is "
                    "drawn so that both cases are in the homes",
                    IN_TIME: "the outage begins more than the primary horizon "
                    "before local midnight",
                    LATE: "it begins later than that",
                    "homes": {
                        IN_TIME: groups.count(IN_TIME),
                        LATE: groups.count(LATE),
                    },
                },
                "outage_days": {
                    "first": min(day for day, _ in starts.values()),
                    "last": max(day for day, _ in starts.values()),
                },
                "replications": f"{self.homes} paired homes; "
                "`docs/SIMULATION_PROTOCOLS.md` sets a floor of 100 for a "
                "simulation study",
                "outages": [
                    {
                        "seed": seed,
                        "day": starts[seed][0],
                        "hour": starts[seed][1],
                        "group": self.group(seed),
                    }
                    for seed in self.study_seeds()
                ],
            },
            "definitions": {
                "behavioural_alert": "an alert of kind behavioural_change, dated "
                "by the moment it was raised",
                "silence_alert": "an alert of kind data_quality about "
                f"`{SILENCE_SUBJECT}`",
                "outage_window": "from the moment a home's outage begins until "
                f"{self.follow_days} days after it ends, which covers the "
                "baseline's trend window",
                "excess_alerts": "a home's behavioural alerts in its outage "
                "window in an outage arm, minus the same home's in the same "
                "window in the stable arm, under the same condition",
                "detected": "a home with a behavioural alert about "
                f"`{TRACKED_FEATURE}` raised once the change day has closed and "
                f"less than {self.max_delay_days:g} days after that: from "
                f"local midnight at the end of day {self.change_day}. The "
                "pipeline raises behavioural alerts when a day closes, so "
                f"these are the alerts that {self.max_delay_days:g} days of "
                "changed behaviour can raise, and the alert raised at the "
                "close of the day before the change is not among them",
                "delay": "days from local midnight at the start of day "
                f"{self.change_day} to the first such alert",
                "false_detection": "the same definition met in the arm that "
                f"has no change: `{STABLE}` for `{CHANGE}`, and `{OUTAGE}` for "
                f"`{CHANGE_AFTER_OUTAGE}`",
                "reported": "a home with a silence alert dated between its "
                "outage's beginning and its end",
                "the_rule_changes_an_alert": "the behavioural alerts of a home "
                "with the rule on are not the same alerts, by subject and "
                "moment, as with it off",
                "person_days": "homes times days",
            },
            "estimands": {
                "E1": f"mean excess alerts per home in `{OUTAGE}` with the rule off",
                "E2": "the same with the rule on at the primary horizon",
                "E3": "E1 minus E2, paired by home",
                "E2_by_group": "E1, E2 and E3 within each outage group. No "
                "criterion is stated on them: they measure what the rule "
                "leaves when it sees a silence only after a day has closed",
                "E4": f"in `{STABLE}`: behavioural alerts per person-day with the "
                "rule off and on, the homes in which the rule changes an "
                "alert, and silence alerts per person-day with the rule on",
                "E5": f"in `{CHANGE}`: the share of homes detected with the rule "
                "off and on, their paired difference, the pooled median delay, "
                "and the share of false detections",
                "E6": f"the same in `{CHANGE_AFTER_OUTAGE}`",
                "E7": f"in `{OUTAGE}` with the rule on: the share of homes "
                "reported, the hours from the outage's beginning to the first "
                "silence alert, the silence alerts per home, and the days "
                "the baseline refused in each home",
                "E8": "the stretches of common silence the fleet check finds in "
                f"`{STABLE}`, `{OUTAGE}` and `{COMMON}`, and the largest share of "
                f"homes silent at one assessment in each; for `{COMMON}`, the "
                "hours from the outage's beginning to the first assessment "
                "that calls it common",
                "E9": f"mean excess alerts per home in `{SHORT_OUTAGE}` with the "
                "rule off and on, the homes in which the rule changes an "
                "alert, and the homes with a silence alert",
                "S1": "E2, E3 and E7 at each other horizon, with E4's counts "
                "there. At a horizon of a whole day no outage is seen before "
                "the day it begins on has closed",
            },
            "criteria": {
                "C1_the_outage_raises_alerts": "reproduced when E1's interval "
                "lies above zero; otherwise not reproduced in the simulator, "
                "and C2 is not testable",
                "C2_the_rule_removes_them": "success when E3's interval lies "
                "above zero and E2's interval lies below "
                f"{self.margin_alerts:g} alerts per home; failure when E3's "
                "interval does not lie above zero; partial otherwise",
                "C3_no_harm_when_nothing_is_wrong": f"success when, in `{STABLE}`, "
                "the rule changes no behavioural alert in any home and raises "
                "no silence alert; failure otherwise",
                "C4_detection_is_kept": f"in `{CHANGE}`: uninformative when the "
                "share detected with the rule off is below "
                f"{self.informative_recall:g}; otherwise non-inferior when the "
                "interval of the paired difference, on minus off, lies above "
                f"-{self.margin_recall:g}, inferior when it lies below it, and "
                "inconclusive otherwise",
                "C5_detection_after_an_outage": "the same rule applied to "
                f"`{CHANGE_AFTER_OUTAGE}`",
                "C6_the_silence_is_reported": "success when at least "
                f"{self.reported_share:g} of the homes are reported; failure "
                "otherwise",
                "C7_a_shared_silence_is_told_from_scattered_ones": "success "
                f"when exactly one stretch is found in `{COMMON}` and it overlaps "
                f"the outage, and none in `{STABLE}` or `{OUTAGE}`; failure "
                "otherwise",
                "C8_a_short_silence_is_not_seen": "the rule cannot see a "
                "silence shorter than its horizon. Confirmed when, in "
                f"`{SHORT_OUTAGE}`, the rule changes no behavioural alert and "
                "raises no silence alert. Otherwise the outage joined a quiet "
                "stretch of the home and passed the horizon, and the homes in "
                "which it did are counted. Either way this is a limit of the "
                "rule and not a success",
            },
            "fleet": {
                "horizon_hours": self.primary_hours,
                "fraction": self.fleet_fraction,
                "min_homes": self.fleet_min_homes,
                "every_hours": self.fleet_every_hours,
                "what": "`sensor_modeling.health.fleet.common_silences` over "
                "the times of each home's delivered observations",
                "what_it_does_not_test": f"in `{OUTAGE}` the homes' outages "
                f"are spread over {self.outage_last_day - self.outage_first_day + 1} "
                "days, so few homes are silent at once and the fraction is far "
                "away. C7 says the check is not tripped by outages that merely "
                "overlap, and nothing about where the fraction should be set",
            },
            "bootstrap": {
                "unit": "homes",
                "resamples": self.resamples,
                "confidence": self.confidence,
                "seed": self.seed,
                "interval": "percentile",
                "share_detected": "a Wilson interval, since homes are "
                "independent; its paired difference is resampled over homes",
                "monte_carlo_error": "the standard deviation of the paired "
                "differences over the square root of the number of homes, "
                "reported for every mean paired difference",
            },
            "what_the_simulator_cannot_show": [
                "a simulated home is not expected to be silent for a horizon "
                "unless a fault makes it so, so the simulator says nothing "
                "about how often a real home is silent for a harmless reason, "
                "or about which horizon a real home needs",
                "a resident who is away for days, or who needs help, is not "
                "simulated. The rule reports such a silence in the same words "
                "as an outage, and the simulator cannot say what that costs",
                "the simulated sensors fire at rates this project wrote down",
            ],
            "tihm": {
                "standing": "a description, not a test: the rule was designed "
                "from these homes",
                "dataset": PROVENANCE.name,
                "citation": PROVENANCE.citation,
                "licence": PROVENANCE.licence,
                "archive_sha256": ARCHIVE_SHA256,
                "files": dict(FILES),
                "step_minutes": self.tihm_step_minutes,
                "conditions": list(self.conditions),
                "everything_else": "as frozen in "
                "`artifacts/tihm/alert_burden_protocol.json`",
                "fleet": "the fleet check with the settings above, over the "
                "times of each home's activity records",
                "check": "with the rule off the run must give the published "
                f"record's {TIHM_PUBLISHED['monitored_days']:,} monitored days "
                f"and {TIHM_PUBLISHED['behavioural_alerts']} behavioural alerts, "
                "or nothing is reported",
                "reported": [
                    "monitored, usable and evaluable days",
                    "the days the baseline refused because of the rule, in all "
                    "and per home",
                    "behavioural alerts, in all and by date",
                    "silence alerts, in all and per home",
                    "the stretches of common silence the fleet check finds, "
                    "and the silence alerts dated inside one",
                ],
                "not_reported": "any relation to the dataset's labels. Whether "
                "the alerts that remain relate to what a clinical team verified "
                "is the alert-burden protocol's question and is not reopened "
                "here",
            },
            "reporting": "every estimand, criterion, arm and home, whatever it "
            "shows; no horizon, margin or window is changed after scoring",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_protocol() -> SilentHomeProtocol:
    """The protocol as declared."""
    return SilentHomeProtocol()


def write_protocol(protocol: SilentHomeProtocol, path: Path) -> str:
    """Write the declaration with its digest; return the digest."""
    payload = {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return protocol.sha256()


def check_frozen_protocol(protocol: SilentHomeProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after scoring begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
