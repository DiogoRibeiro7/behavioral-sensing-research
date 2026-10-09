"""The matched-sensor protocol: the simulator's homes with TIHM's sensors.

Every detection the repository reports in simulation comes from homes whose
event sensors fire at rates written into the simulator. The TIHM homes'
sensors do not behave like that. Every motion sensor in TIHM stays quiet for
61 seconds after it reports; most motion activations follow one from another
room; there is a hallway sensor; and a contact logs an opening as two rows.
On TIHM the pipeline gives a median of 0.06 hours a day to kitchen activity
and 0.00 to bathroom activity.

This protocol draws the event sensors of the threshold-calibration study's
400 simulated homes again, from the same plans, under a sensor profile matched
to TIHM's sensor records, and asks how much of what the simulator said about
the pipeline still holds: its detection of the step change, its hours of
sleep, and its hours in the kitchen and the bathroom.

**It is a test of the simulator, not of a change to the pipeline.** Nothing in
the pipeline is altered or fitted. The profile is fixed by the planning record
before any home of the study has been run with it.

:func:`declared_protocol` is frozen in
``artifacts/matched_sensors/matched_sensors_protocol.json``, with the digest
of every source file the run passes through.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from ..alerts.alert import AlertPolicy
from ..baseline.adaptive import BaselineConfig
from ..evaluation.provenance import json_safe, resolved_defaults
from ..external.tihm import PROVENANCE
from ..online.pipeline import PipelineConfig
from ..simulation.household import HouseholdConfig
from ..simulation.sensor_profile import (
    HALLWAY,
    PAIR_GAP_SECONDS,
    STANDARD_PROFILE,
    SensorProfile,
)
from .silent_home_protocol import event_sensors
from .threshold_calibration_protocol import (
    CHANGE,
    STABLE,
    ThresholdCalibrationProtocol,
)
from .threshold_calibration_protocol import (
    declared_protocol as threshold_declared,
)

#: The record's name and the layout of its results.
EXPERIMENT = "matched-sensors"
RESULT_SCHEMA = "matched-sensors/1"

#: The profiles, and the two arms of each home. The sensitivity profile is run
#: on the first homes only, as a description.
STANDARD = "standard"
MATCHED = "matched"
SENSITIVITY = "sensitivity"
PROFILES = (STANDARD, MATCHED)
ARMS = (STABLE, CHANGE)

#: The features compared with the truth, as the pipeline names them, and every
#: state whose hours E5 describes.
FEATURES = ("sleeping_hours", "kitchen_activity_hours", "bathroom_activity_hours")
TRACKED_FEATURE = "sleeping_hours"
STATE_HOURS = (
    "away",
    "home_active",
    "home_inactive",
    "sleeping",
    "bed_awake",
    "bathroom_activity",
    "kitchen_activity",
)

#: The readings of the criteria.
SURVIVES = "survives"
DOES_NOT_SURVIVE = "does_not_survive"
FOLLOWS = "follows"
DOES_NOT_FOLLOW = "does_not_follow"
REPRODUCED = "reproduced"
NOT_REPRODUCED = "not_reproduced"
INCONCLUSIVE = "inconclusive"

#: The criteria's names.
C1 = "C1_the_detection_survives_the_matched_profile"
C2 = "C2_the_hours_of_sleep_still_follow_the_truth"
C3 = "C3_the_kitchen_and_bathroom_collapse_is_reproduced"

#: The records the study rests on, and their digests.
THRESHOLD_PROTOCOL = (
    "artifacts/threshold_calibration/threshold_calibration_protocol.json"
)
THRESHOLD_PROTOCOL_SHA256 = (
    "459f90f8bb84752af537d34ac51f376c6815f83e593ed6360bc343c15ece7403"
)
THRESHOLD_RECORD = "artifacts/threshold_calibration/threshold-calibration.json"
THRESHOLD_RECORD_SHA256 = (
    "1f0b22b7c329f30f97f16686ec5563cd6eafa4071079c691387fa5508da078d0"
)
TIHM_POST_HOC_RECORD = "artifacts/tihm/tihm-alert-burden-post-hoc.json"
TIHM_POST_HOC_RECORD_SHA256 = (
    "2701e81efe7d126e753478e2c0187e8e7568716380ad359ac4a92f1f255723ac"
)
PLANNING_RECORD = "artifacts/matched_sensors/matched-sensors-planning.json"
PLANNING_RECORD_SHA256 = (
    "c713dc27dd0308d098da0e0dad623da81cc9ac84ea770219df2e9a16f0ae569a"
)

#: The profile the planning chose, its second-nearest grid point, and readings
#: the protocol quotes from the TIHM post hoc record.
HOLD_OFF_SECONDS = 61.0
MATCHED_PRESENCE_SCALE = 1.0
MATCHED_SPILL_RATE = 2.0
SENSITIVITY_PRESENCE_SCALE = 1.0
SENSITIVITY_SPILL_RATE = 2.5
TIHM_DAY_MEDIANS = {
    "kitchen_activity_hours": 0.0587,
    "bathroom_activity_hours": 0.0021,
}
TIHM_SILENT_DAYS = 128
TIHM_USABLE_DAYS = 2793

#: The default reference's condition in the published record.
PUBLISHED_CONDITION = "default@1"

#: The sources whose content is recorded at the freeze and compared at the run.
PINNED_PACKAGE = "sensor_modeling"
PINNED_SCRIPTS = (
    "scripts/freeze_matched_sensors.py",
    "scripts/run_matched_sensors.py",
)
PINNED_DISTRIBUTIONS = ("numpy", "scipy", "pandas")

#: What was run and read between the freeze and the run.
BETWEEN_FILE = "artifacts/matched_sensors/between_the_freeze_and_the_run.json"

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


def _profile_rows(profiles: dict[str, SensorProfile]) -> dict[str, Any]:
    """Each setting's meaning and unit, and its value in every profile."""
    meanings = {
        "hold_off_seconds": "the shortest gap, in seconds, between two "
        "activations one motion sensor reports",
        "presence_scale": "the multiple of the simulator's in-room rates of 45 "
        "an hour active, 6 still and 0.6 asleep, and of a visitor's 31.5",
        "spill_rate": "activations an hour of every motion sensor of a room the "
        "resident is not in, the bathroom's and the hallway's among them, while "
        "the resident is at home and awake, moving or still; asleep, out or "
        "with no resident at home, 0.12 an hour",
        "hallway": "a hallway motion sensor that reports at each change of the "
        "resident's or a visitor's room, fires at the visitor rate while a "
        "visitor is in the hallway, and otherwise spills over",
        "paired_contacts": "the fridge and the entrance door record each "
        f"activation twice, the second row {PAIR_GAP_SECONDS[0]:g} to "
        f"{PAIR_GAP_SECONDS[1]:g} seconds after the first, uniformly",
    }
    return {
        name: {
            "meaning": meaning,
            **{label: profile.to_dict()[name] for label, profile in profiles.items()},
        }
        for name, meaning in meanings.items()
    }


@dataclass(frozen=True)
class MatchedSensorsProtocol:
    """Every definition of the matched-sensor study.

    Attributes
    ----------
    presence_scale, spill_rate, hold_off_seconds
        The matched profile, from the planning record.
    sensitivity_presence_scale, sensitivity_spill_rate, sensitivity_homes
        The planning record's second-nearest grid point, run on the first homes
        as a description.
    survival_margin
        The loss of excess detection C1 is judged against.
    tracking_margin
        The mean within-home rank correlation C2 is judged against.
    kitchen_ceiling_hours, bathroom_ceiling_hours
        The medians under which C3 reads the collapse as reproduced.
    min_matched_days
        The matched days a home needs to enter a correlation.
    constant_tolerance
        A series whose range is no wider than this is constant.
    resamples, confidence, seed
        The home bootstrap.
    """

    presence_scale: float = MATCHED_PRESENCE_SCALE
    spill_rate: float = MATCHED_SPILL_RATE
    hold_off_seconds: float = HOLD_OFF_SECONDS
    sensitivity_presence_scale: float = SENSITIVITY_PRESENCE_SCALE
    sensitivity_spill_rate: float = SENSITIVITY_SPILL_RATE
    sensitivity_homes: int = 100
    survival_margin: float = 0.10
    tracking_margin: float = 0.5
    kitchen_ceiling_hours: float = 0.25
    bathroom_ceiling_hours: float = 0.05
    min_matched_days: int = 14
    constant_tolerance: float = 1e-9
    resamples: int = 5000
    confidence: float = 0.95
    seed: int = 0
    planning_record_sha256: str = PLANNING_RECORD_SHA256

    #: What had been seen when the protocol was written.
    inspected_before: tuple[str, ...] = (
        "TIHM's activity file, read for its sensors alone: over every home, the "
        "gaps between consecutive activations of each location's sensor. No "
        "two activations of one motion sensor are less than 61 seconds apart. "
        "The fridge and door contacts have 44% to 59% of their gaps under a "
        "minute, with a median of 7 to 9 seconds between two such rows",
        "in three TIHM homes, 0d5ef, 30a32 and 55cd4, the activations a day of "
        "each location and the runs of consecutive activations of one "
        "location; and the default pipeline's beliefs in those homes after the "
        "10-minute windows that held only kitchen, or only bathroom, "
        "activations: 68% to 89% on `home_inactive` and 0% to 5% on kitchen "
        "activity after the kitchen windows",
        "the same three homes' median hours a day in each state under the "
        "default emissions and under five changed declarations: the fridge left "
        "out, counts read as fired or not in each step, the active rate at 15 "
        "an hour, and two combinations of these. None put kitchen activity "
        "above 0.42 hours a day or bathroom activity above 0.21. The "
        "pipeline's output on TIHM had been read in the earlier studies as "
        "well. No label and no sleep-mat record was read for this protocol",
        "a pilot on four simulated homes from seeds 900001 to 900004, which "
        "belong to no protocol, over 42 days. With the simulator's motion "
        "sensors given a 61-second hold-off, the pipeline's median hours of "
        "kitchen activity fell from 1.59 to 0.96 against a true 1.93, their "
        "mean within-home correlation with the truth from 0.70 to 0.45, and "
        "that of sleep from 0.68 to 0.65. With a hallway sensor reporting at "
        "each change of room as well, kitchen activity was at 0.95 hours and "
        "0.51, and sleep at 0.63. These read the pipeline's output on "
        "simulated homes before the protocol was written, and shaped it: they "
        "are why the profile is changed as a whole, why it has spill-over, and "
        "why C3 is declared. Neither pilot had spill-over",
        "a coarse trial of the moments on six planning homes, and the planning "
        "record; neither runs the pipeline. The simulator's own records are far "
        "from TIHM's: 8% of its motion activations follow another room's "
        "against 63%, a median retrigger gap of 60 seconds against 153, no "
        "hallway, and 242 living-room activations a day against 77. The "
        "planning was run again after the review below corrected the profile's "
        "hallway, visitors, pair gap and streams, and the protocol pins the "
        "second record. It chose a spill-over of 2 an hour, at a distance of "
        "0.8946, with 2.5 second at 0.8952; the first record had chosen 2.5",
        "the published threshold-calibration results on the 400 homes under "
        "the default reference: 293 homes detected, 42 falsely detected and "
        "342 false alerts; and TIHM's pooled-day medians of 0.06 hours of "
        "kitchen activity and 0.00 of bathroom activity",
        "the standard profile's runs of the study's first two homes, 2739 and "
        "5651, in both arms, to check that they raise the published alerts: "
        "they do",
        "the code was rehearsed end to end on two homes outside the study, "
        "seeds 999 and 1001, under the standard profile and a draft matched "
        "profile, and the unit tests run one short home outside the study the "
        "same way. Their run times and that every step completed were read, "
        "and none of their values. No pipeline output under spill-over, or "
        "under the chosen profile, has been read for any home",
        "an independent review of the draft read the code, the draft "
        "protocol and the first planning record, and drew sensor records on "
        "seeds outside the study without running the pipeline. It found the "
        "hallway reporting at each record's first instant, visitors in the "
        "hallway not firing it, the two arms of a home sharing the matched "
        "profile's sensor noise, an undeclared minimum of matched days, "
        "per-home outcomes written when the check fails, and a flat grid "
        "minimum whose nearest point changes between fresh draws of planning "
        "homes. The profile was corrected, the planning run again, the "
        "sensitivity profile and E7 to E9 declared, and the mechanisms below "
        "written down",
        "a second review of the revision read the code and the second "
        "planning record. To check the code it ran the pipeline under the "
        "matched profile on seeds 999, 1001 and 4242 for 24 days, and under "
        "the sensitivity profile on seed 999, and built a results page and "
        "figures from those runs with every digit masked; it passed on no "
        "value. It found that under the standard profile the two arms of a "
        "home can share sensor draws, a mechanism wrongly stated for sleep, "
        "and the render script failing on a record whose check failed. All "
        "three were corrected, with other wording, before the freeze",
    )

    def __post_init__(self) -> None:
        """Validate the declaration."""
        if not 0.0 < self.survival_margin < 1.0:
            raise ValueError("survival_margin must lie in (0, 1)")
        if not 0.0 < self.tracking_margin < 1.0:
            raise ValueError("tracking_margin must lie in (0, 1)")
        if self.kitchen_ceiling_hours <= 0.0 or self.bathroom_ceiling_hours <= 0.0:
            raise ValueError("the ceilings must be positive")
        if self.min_matched_days < 3:
            raise ValueError("a rank correlation needs at least three days")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")
        if self.sensitivity_homes < 2:
            raise ValueError("the sensitivity profile needs at least two homes")

    @property
    def study(self) -> ThresholdCalibrationProtocol:
        """The threshold-calibration protocol, whose homes and arms are used."""
        return threshold_declared()

    def profile(self, name: str) -> SensorProfile:
        """A profile by name."""
        if name == STANDARD:
            return STANDARD_PROFILE
        settings = {
            MATCHED: (self.presence_scale, self.spill_rate),
            SENSITIVITY: (self.sensitivity_presence_scale, self.sensitivity_spill_rate),
        }
        if name not in settings:
            raise KeyError(f"no profile '{name}'")
        scale, spill = settings[name]
        return SensorProfile(
            hold_off_seconds=self.hold_off_seconds,
            presence_scale=scale,
            spill_rate=spill,
            hallway=True,
            paired_contacts=True,
        )

    def sensitivity_seeds(self) -> tuple[int, ...]:
        """The homes the sensitivity profile is run on: the study's first."""
        return self.study.study_seeds()[: self.sensitivity_homes]

    def stream(self, arm: str) -> int:
        """The generator stream an arm's re-drawn sensors come from."""
        return ARMS.index(arm)

    def to_dict(self) -> dict[str, Any]:
        """Return the full declaration, in a stable serialisable form."""
        study = self.study
        baseline, policy, pipeline = BaselineConfig(), AlertPolicy(), PipelineConfig()
        begin, end = study.detection_window(CHANGE)
        changed = study.start() + timedelta(days=study.change_day)
        level = f"{self.confidence:.0%}"
        return {
            "schema": "matched-sensors-protocol/1",
            "name": EXPERIMENT,
            "status": "pre-specified simulation study. The profile's hold-off is "
            "TIHM's, and two of its settings were chosen to match eight moments "
            "of TIHM's sensor records, in the planning record; its form is this "
            "project's, written after a pilot. "
            "No pipeline output under spill-over or under the chosen profile has "
            "been read for any home. Nothing in the pipeline is changed or "
            "fitted. Every result is a statement about the simulator",
            "question": "when the simulator's event sensors are drawn again from "
            "the same plans under a profile matched to TIHM's sensor records, how "
            "much of what the simulator said about the pipeline still holds: its "
            "detection of the step change, its hours of sleep, and its hours in "
            "the kitchen and the bathroom",
            "inspected_before": list(self.inspected_before),
            "homes": {
                "from": "the threshold-calibration protocol, "
                f"`{THRESHOLD_PROTOCOL}`, whose homes, days, change, detection "
                "window and pipeline settings are used unchanged",
                "seed_root": study.seed_root,
                "homes": study.homes,
                "days": study.days,
                "step_minutes": study.step_minutes,
                "start": study.start().isoformat(),
                "timezone": str(HouseholdConfig().tz),
                "arms": {
                    STABLE: "nothing is injected; every behavioural alert is "
                    "a false one",
                    CHANGE: f"from {changed.isoformat()}, day {study.change_day} "
                    "counting the first day as day 0, the resident wakes "
                    f"{study.change_sleep_delta_hours:g} hours earlier, with "
                    f"{study.change_night_bathroom_extra:g} more night-time "
                    "bathroom trips expected a night",
                },
                "window": f"from {begin.isoformat()} to {end.isoformat()}: from "
                "local midnight at the end of the first changed day, for "
                f"{study.max_delay_days:g} days, closed on the left and open on "
                "the right",
            },
            "profiles": {
                STANDARD: "the simulator's own event record, as `simulate` draws "
                "it: the threshold-calibration study's homes exactly. Sensors: "
                + ", ".join(f"`{s}`" for s in event_sensors()),
                MATCHED: "the event sensors drawn again from the same plan by "
                "`sensor_modeling.simulation.sensor_profile."
                "profile_observations`, with the home's seed, the arm's stream "
                "and this profile. Sensors: the same and "
                f"`{HALLWAY}`",
                SENSITIVITY: "the matched profile at the planning record's "
                "second-nearest grid point, on the study's first "
                f"{self.sensitivity_homes} homes, as a description",
                "settings": _profile_rows(
                    {
                        STANDARD: self.profile(STANDARD),
                        MATCHED: self.profile(MATCHED),
                        SENSITIVITY: self.profile(SENSITIVITY),
                    }
                ),
                "chosen_by": f"`{PLANNING_RECORD}`. The hold-off is the shortest "
                "gap between two activations of one motion sensor in TIHM's "
                "activity file. TIHM's records have a hallway sensor and log a "
                "contact's opening as two rows; the profile imitates both with "
                "rules of this project's. The presence scale and the spill-over "
                "rate are the grid point whose eight moments of the sensor "
                "records were nearest TIHM's, summed squared log ratios of "
                "medians over homes. The minimum is flat, and fresh draws of "
                "planning homes order the nearest points differently, so the "
                "second-nearest is run as a sensitivity profile",
                "pairing": "the profiles of one home and arm share the plan, so "
                "they differ in the sensors and not in the resident. The stable "
                "and changed arms of one home share the seed and not the days "
                "after the change. Under the standard profile the simulator draws "
                "the plan and the sensors from one generator, and two arms whose "
                "plans differ can fall back into step, so they share a part of "
                "their sensor draws before the change that depends on the home, "
                "from none of the weeks to all of them. Re-drawn sensors come "
                f"from the arm's own stream, {self.stream(STABLE)} for the stable "
                f"arm and {self.stream(CHANGE)} for the changed one, so under the "
                "matched and sensitivity profiles the two arms share only the rows "
                "the plan fixes: the hallway's reports at changes of room and the "
                "door's crossings. The matched and sensitivity profiles of one arm "
                "draw from the same stream. This changes how a home's detection "
                "and its false detection move together, not what either is "
                "expected to be",
                "delivery": "every record passes through "
                "`sensor_modeling.simulation.faults.degrade` with its default "
                "configuration: no loss, lateness, duplication or fault",
            },
            "pipeline": {
                "what": "`BehaviouralSensingPipeline` over each profile's "
                "registry, with default emissions derived from it and every "
                "default setting; nothing is fitted. The matched registry has "
                "the hallway's motion sensor, whose room no state names, as "
                "TIHM's has",
                "step_minutes": study.step_minutes,
                "features": [f"`{s.value}_hours`" for s in pipeline.features],
                "min_day_coverage": pipeline.min_day_coverage,
                "min_day_observed": pipeline.min_day_observed,
                "baseline": {
                    "min_samples": baseline.min_samples,
                    "deviation_threshold": baseline.deviation_threshold,
                    "weekday_min_samples": baseline.weekday_min_samples,
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
            "what_the_outcomes_turn_on": [
                "the default emissions expect 0.15 activations an hour from the "
                "motion sensor of a room other than the one a state names, so "
                "while the resident is in the kitchen, the bathroom or in bed "
                "awake each spill-over activation is strong evidence against "
                f"that state. Spill-over at {self.spill_rate:g} an hour therefore "
                "pushes belief away "
                "from kitchen and bathroom activity whenever the resident is "
                "awake",
                "the living room's and the hallway's sensors name no state's "
                "room, so the default emissions read them as evidence for "
                "`home_active` (10 an hour) and `home_inactive` (2.5 an hour)",
                f"the hold-off of {self.hold_off_seconds:g} seconds lowers the "
                "rate a room's sensor reports while the resident is active in it "
                "from 45 an hour to about 25, against the 40 the emissions expect "
                "of the room's state; this is what the pilot showed",
                "paired contact rows double the evidence a fridge opening gives "
                "for an active state and a door crossing gives for `away` and "
                "`home_active`, which works against C3",
                "the occupancy layer reads two rooms firing within 60 seconds of "
                "each other as evidence of a visitor. Spill-over makes such "
                "pairs common while the resident is alone, and the hallway's "
                "report at a change of room, with the next room's first "
                "activation, makes one at most changes of room even without "
                "spill-over. That lowers the "
                "attribution of activity to the resident, can hold an alert back "
                f"at the confidence gate of {policy.min_confidence:g}, and "
                "discounts the evidence. E7 describes it",
                "C3 reproduced would show that this sensor model suffices to "
                "give the pipeline TIHM's near-zero kitchen and bathroom hours "
                "in the simulator, not that it is why TIHM's are near zero. "
                "E8 shows whether the kitchen's own activations lose to "
                "`home_inactive` as they did in TIHM",
            ],
            "check": {
                "what": "in every home, the standard profile's run of each arm "
                "must raise exactly the behavioural alerts the published record "
                f"gives that home and arm under `{PUBLISHED_CONDITION}`: the "
                "same days, subjects, verdicts and directions",
                "record": f"`{THRESHOLD_RECORD}`",
                "otherwise": "nothing is reported, not even per home",
            },
            "definitions": {
                "detected": f"an alert about `{TRACKED_FEATURE}` in the changed "
                "arm's window, in either direction, as the published study "
                "counts it",
                "false_detection": "an alert about the same feature in the "
                "same window of the stable arm",
                "excess_detection": "a home's detection minus its false "
                "detection: 1, 0 or -1",
                "false_alerts": "the behavioural alerts of the stable arm, "
                "about any feature",
                "matched_day": "a usable day of the stable arm other than the "
                "record's first and last days, as the sleep-mat protocol's "
                "simulated reference has them. Each profile's matched days are "
                "its own usable days",
                "pipeline_value": "a state's expected hours in the day's " "summary",
                "truth": "the hours of the local day the simulator's episodes "
                "of the state cover, in elapsed time",
                "correlation": "the within-home Spearman correlation between "
                "the pipeline's value and the truth on the profile's matched "
                f"days, for a home with at least {self.min_matched_days} of "
                "them. A home whose values on either side span no more than "
                f"{self.constant_tolerance:g} hours has no correlation and "
                "counts as 0",
                "pooled_day_median": "the median over every matched day of "
                "every home, as TIHM's medians are pooled over its usable days",
            },
            "estimands": {
                "E1": "the mean over homes of the matched profile's excess "
                "detection minus the standard profile's",
                "E2": "for each profile, the share of homes detected, the share "
                "falsely detected, the mean excess detection, the same three "
                "counting only alerts of a decrease, and the median delay of a "
                "detection in days from the start of the first changed day; and "
                "the differences between the profiles in the shares detected and "
                "falsely detected",
                "E3": "for each profile, the mean over homes of the false "
                "alerts, and the stable arm's alerts about each feature per "
                "person-day",
                "E4": "for each profile and each of "
                + ", ".join(f"`{f}`" for f in FEATURES)
                + ", the mean over homes of the correlation; and the mean over "
                "homes in both of the matched profile's correlation minus the "
                "standard profile's, each on its own matched days",
                "E5": "for each profile and each of the seven states, the "
                "pooled-day median of the pipeline's value and of the truth; for "
                "the three features, the mean over homes of the home's mean "
                "difference, pipeline minus truth; and for kitchen and bathroom "
                "activity, a home-bootstrap interval for the pooled-day median "
                "of the pipeline's value",
                "E6": "the eight moments of the planning, on each profile's "
                "stable record of the study's homes, beside TIHM's from the "
                "planning record",
                "E7": "for each profile, in the stable arm: the change verdicts, "
                "those that raised an alert, and those that did not because the "
                "day's coverage times attribution was under the confidence gate "
                "or for another reason; the notices that alerts were held back "
                "in a burst; and the mean over homes of each home's mean ambient "
                "attribution of activity to the resident at its days' closes",
                "E8": "for each profile, in the stable arm: the mean belief in "
                "each state at the end of the steps whose window held motion "
                "activations of the kitchen's sensor and of no other motion "
                "sensor, and likewise of the bathroom's, with how many such "
                "steps there were. The study's steps are of 15 minutes; TIHM's "
                "beliefs were read after 10-minute windows, TIHM's step",
                "E9": "E1, E2 and E4 for the sensitivity profile against the "
                f"standard profile, on the study's first {self.sensitivity_homes} "
                "homes",
            },
            "criteria": {
                C1: "the primary criterion, on E1. Survives when the interval "
                f"lies above -{self.survival_margin:g}; does not survive when it "
                f"lies below -{self.survival_margin:g}; inconclusive otherwise",
                C2: "secondary, on E4's mean correlation of `sleeping_hours` "
                f"under the matched profile. Follows when the interval lies above "
                f"{self.tracking_margin:g}; does not follow when it lies below; "
                "inconclusive otherwise",
                C3: "secondary, on E5 under the matched profile. Reproduced when "
                "the pooled-day median of `kitchen_activity_hours` is under "
                f"{self.kitchen_ceiling_hours:g} hours and that of "
                f"`bathroom_activity_hours` under {self.bathroom_ceiling_hours:g} "
                "hours; not reproduced otherwise. The medians are read as points; "
                "their home-bootstrap intervals are reported beside them",
                "multiplicity": "C1 is the only primary criterion. The criteria "
                "are not adjusted for each other. E2 to E9 are descriptions",
            },
            "margins": {
                "survival": f"{self.survival_margin:g}, a declared convention: a "
                "tenth of the homes. The standard profile's excess detection in "
                "the published record is 0.63. Assuming a standard deviation of "
                "about 0.6 for a home's paired difference, which the "
                "threshold-calibration protocol assumed for its own, 400 homes "
                "would give an interval about 0.06 either side",
                "tracking": f"{self.tracking_margin:g}, the sleep-mat protocol's "
                "margin. At 0.5 a deviation at the baseline's threshold of "
                f"{baseline.deviation_threshold:g} would stand for one of 1.5 in "
                "sleep, under a Gaussian reading. Against the simulator's exact "
                "truth it is a lower bar than against a sleep mat",
                "ceilings": f"{self.kitchen_ceiling_hours:g} and "
                f"{self.bathroom_ceiling_hours:g} hours, declared conventions. "
                "TIHM's pooled-day medians are "
                f"{TIHM_DAY_MEDIANS['kitchen_activity_hours']:.2f} and "
                f"{TIHM_DAY_MEDIANS['bathroom_activity_hours']:.2f} hours in "
                f"`{TIHM_POST_HOC_RECORD}`, over {TIHM_USABLE_DAYS:,} usable "
                f"days of which {TIHM_SILENT_DAYS} had no record; the "
                "simulator's truth gives about two hours a day in the kitchen and "
                "a quarter of an hour in the bathroom",
            },
            "pinned_records": {
                THRESHOLD_PROTOCOL: THRESHOLD_PROTOCOL_SHA256,
                THRESHOLD_RECORD: THRESHOLD_RECORD_SHA256,
                TIHM_POST_HOC_RECORD: TIHM_POST_HOC_RECORD_SHA256,
                PLANNING_RECORD: self.planning_record_sha256,
            },
            "intervals": {
                "unit": "homes",
                "method": f"a percentile bootstrap over homes at {level}, with "
                f"{self.resamples:,} resamples from seed {self.seed}, as the "
                "published study's; a share of homes has a Wilson interval",
                "counts": "a pooled count has none",
            },
            "reporting": "every estimand, criterion, profile and home, whatever "
            "it shows; no margin, profile or day is changed after a home is run "
            "with the matched profile. What was run and read between the freeze "
            f"and the run is listed in `{BETWEEN_FILE}`",
            "what_this_cannot_show": [
                "how the pipeline does on TIHM: the study runs no TIHM home",
                "whether the profile matches TIHM's sensors in anything but the "
                "eight moments. The residents' behaviour is the simulator's, and "
                "TIHM's carers, back doors and the behaviour of people living "
                "with dementia are not in it",
                "why TIHM's motion activations so often follow another room's. "
                "The profile puts it down to sensors that see beyond their room; "
                "more movement between rooms, or a second person at home, would "
                "fit the moments as well",
                "which part of the profile does what: it is changed as a whole",
                "that the chosen point is the best: the grid's minimum is flat, "
                "and E9 runs the next point",
                "the TIHM moments count every day with a record, partial days "
                "among them, which lowers TIHM's counts a day by a few percent",
            ],
            "data": {
                "dataset": PROVENANCE.name,
                "read": "TIHM's Activity.csv, by the planning only; the study "
                "reads no TIHM file",
            },
            "acknowledgement": "TIHM is by Palermo et al., Scientific Data 10, "
            "606 (2023), under CC BY 4.0. Surrey and Borders Partnership NHS "
            "Foundation Trust and Howz are acknowledged, as the dataset's "
            "repository asks",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_protocol() -> MatchedSensorsProtocol:
    """The protocol as declared."""
    return MatchedSensorsProtocol()


def write_protocol(protocol: MatchedSensorsProtocol, path: Path) -> str:
    """Write the declaration with its digest and the code it was frozen against."""
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


def check_frozen_protocol(protocol: MatchedSensorsProtocol, path: Path) -> str:
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


def planned_choice(planning: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """The planning record's nearest and second-nearest grid points."""
    grid = sorted(
        planning["results"]["grid"],
        key=lambda row: (row["distance"], row["presence_scale"], row["spill_rate"]),
    )
    return {
        MATCHED: (grid[0]["presence_scale"], grid[0]["spill_rate"]),
        SENSITIVITY: (grid[1]["presence_scale"], grid[1]["spill_rate"]),
    }
