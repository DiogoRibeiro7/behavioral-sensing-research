"""The threshold-calibration protocol: is a threshold that means what it says better.

The TIHM run found that the personal baseline's deviation threshold is passed
far more often than it states, because a weekday-aware reference rests on four
or five days. ``BaselineConfig.calibrated`` was built from that finding and
designed on synthetic Gaussian days, where it does what it should. Neither is
evidence that it helps in a home: TIHM showed the problem, and the Gaussian
days chose the design.

This protocol has two parts, and they have different standing.

**The simulator is the test.** Simulated homes are reduced to their event
sensors, as in the silent-home protocol. Each is run once through the pipeline
with nothing injected, with a step change, with a smaller one and with a
gradual one, and the days each run closed are replayed under the default
reference and under the calibrated one, each at a grid of multiples of the
declared thresholds.

The calibrated reference is not compared at the default's thresholds, where it
reports far less of everything. Two rules can be compared when they report
falsely as often, and where that is for these two is not known in advance.
So the comparison is made where the calibrated
reference's operating curve, over its multiples, has the false alerts of the
default as it ships, and what it finds is read there. Where that place is is
part of what is estimated, so it is found again in every resample of homes,
and the interval carries the uncertainty of finding it.

A home counts as detected net of what the same reference raises in the same
window of the same home with nothing injected, since a reference that alerts
more often is found "detecting" more often by chance. The estimands, the
criteria, the matching rule and the homes are fixed here, before any simulated
home has been run with the calibrated reference.

**TIHM is a description.** The same references are run on the 56 TIHM homes,
to say how often a real day passes a threshold that is exact for Gaussian
days. Nothing there is a test: the option was built from those homes.

Simulated evidence is evidence about the simulator. :func:`declared_protocol`
is frozen in ``artifacts/threshold_calibration/threshold_calibration_protocol.json``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from importlib import metadata
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from ..alerts.alert import AlertPolicy
from ..baseline.adaptive import DOF_PER_RESIDUAL, BaselineConfig
from ..evaluation.provenance import json_safe, resolved_defaults
from ..external.tihm import ARCHIVE_SHA256, FILES, PROVENANCE
from ..online.pipeline import PipelineConfig
from ..simulation.household import BehaviourShift, HouseholdConfig, build_registry
from .silent_home_protocol import event_sensors

#: The records' names and the layout of their results.
EXPERIMENT = "threshold-calibration"
RESULT_SCHEMA = "threshold-calibration/1"
TIHM_EXPERIMENT = "threshold-calibration-tihm"
TIHM_SCHEMA = "threshold-calibration-tihm/1"

#: The two references.
DEFAULT = "default"
CALIBRATED = "calibrated"
REFERENCES = (DEFAULT, CALIBRATED)

#: The arms, and what each injects.
STABLE = "stable"
CHANGE = "change"
SMALL_CHANGE = "small_change"
GRADUAL_CHANGE = "gradual_change"
ARMS = (STABLE, CHANGE, SMALL_CHANGE, GRADUAL_CHANGE)
CHANGE_ARMS = (CHANGE, SMALL_CHANGE, GRADUAL_CHANGE)

#: The feature a detection is counted on, as in the detection study.
TRACKED_FEATURE = "sleeping_hours"

#: The condition on TIHM with the silent-home rule off.
RULE_OFF = "off"

#: What the published TIHM runs recorded, which the description must reproduce
#: under the default reference before it reports anything.
TIHM_PUBLISHED = {
    RULE_OFF: {"monitored_days": 2850, "behavioural_alerts": 183},
    "h12": {"monitored_days": 2850, "behavioural_alerts": 49},
}

#: The record of the measurement on synthetic days that the design rests on.
NULL_RECORD = "artifacts/threshold_calibration/threshold-null.json"

#: The record of the trials on curves written down by hand that the number
#: of homes and the choice of the match rest on.
PLANNING_RECORD = "artifacts/threshold_calibration/threshold-calibration-planning.json"

#: The published TIHM records the description must reproduce.
TIHM_RECORDS = (
    "artifacts/tihm/tihm-alert-burden.json",
    "artifacts/silent_home/silent-home-tihm.json",
)

#: The thresholds the share of deviating days is described at.
TAIL_THRESHOLDS = (1.5, 1.8, 2.0, 2.5, 3.0)

#: What the match of the calibrated reference's curve to the default is.
BRACKETED = "bracketed"
NOT_REACHED = "not_reached"
END_OF_THE_GRID = "at_the_end_of_the_grid"

#: The source whose content is recorded when the protocol is frozen and
#: compared when it is run: every module of the package, since a home's days
#: pass through most of it before a threshold is asked anything, and the
#: scripts that freeze and run the protocol.
PINNED_PACKAGE = "sensor_modeling"
PINNED_SCRIPTS = (
    "scripts/freeze_threshold_calibration.py",
    "scripts/run_threshold_calibration.py",
    "scripts/run_threshold_calibration_tihm.py",
)

#: The distributions whose versions are recorded beside the sources: the
#: numerical libraries a verdict passes through.
PINNED_DISTRIBUTIONS = ("numpy", "scipy", "pandas")

#: What was run and read between the freezing of the protocol and its run.
#: It is a list of sentences in a file, and not code, so that adding to it
#: changes none of the sources above.
BETWEEN_FILE = "artifacts/threshold_calibration/between_the_freeze_and_the_run.json"

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
    """The sources' digests, the libraries' versions and every default setting, now."""
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
    """What was run and read between the freeze and the run, as recorded.

    A file that is not there records nothing.
    """
    path = (_REPOSITORY if root is None else Path(root)) / BETWEEN_FILE
    if not path.exists():
        return ()
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not all(
        isinstance(entry, str) and entry for entry in entries
    ):
        raise ValueError(f"{BETWEEN_FILE} must hold a list of sentences")
    return tuple(entries)


def condition_name(reference: str, scale: float) -> str:
    """The name of a reference run at a multiple of the declared thresholds."""
    if reference not in REFERENCES:
        raise KeyError(f"no reference '{reference}'")
    return f"{reference}@{scale:g}"


def nominal(threshold: float) -> float:
    """The share of Gaussian values at or past *threshold* in absolute value."""
    return 2.0 * (1.0 - NormalDist().cdf(threshold))


def equivalent_threshold(share: float) -> float:
    """The threshold a share of deviating days corresponds to on Gaussian days.

    The inverse of :func:`nominal`. No day deviating is an unbounded
    threshold, and every day deviating is a threshold of zero.
    """
    if not 0.0 <= share <= 1.0:
        raise ValueError("a share lies between 0 and 1")
    if share == 0.0:
        return float("inf")
    return NormalDist().inv_cdf(1.0 - share / 2.0) if share < 1.0 else 0.0


@dataclass(frozen=True)
class ThresholdCalibrationProtocol:
    """Every definition of the threshold-calibration evaluation.

    Attributes
    ----------
    seed_root, homes
        The simulated homes: their seeds are drawn from the root.
    days, step_minutes
        Length of each simulated record and the pipeline's step.
    curve_scales
        The multiples of the declared thresholds both references are run at.
    check_scales
        The multiples at which the replay is checked against the pipeline run
        with the calibrated reference itself.
    change_day, change_sleep_delta_hours, change_night_bathroom_extra
        The step change, the detection study's and the silent-home protocol's.
    small_fraction
        The smaller step, as a fraction of the step change.
    gradual_day, gradual_ramp_days
        The day the gradual change is declared from and the days it takes to
        reach the step change's size. The first day it changes anything is
        the day after.
    max_delay_days, gradual_max_delay_days
        How long after a change begins a detection still counts.
    informative_detection, calibration_tolerance
        The criteria's margins: the default's excess detection below which C1
        is uninformative, and the band of C2.
    calibration_thresholds
        The thresholds at which the calibrated reference's deviation is asked
        to mean what it says.
    planned_sd
        The standard deviation of a home's difference in excess detection
        between two conditions that the number of homes was planned from.
    checked_homes
        The homes on which the replay is checked against the pipeline run
        with the calibrated reference itself: the first, in order of seed.
    resamples, confidence, seed
        The bootstrap over homes.
    tihm_step_minutes, tihm_horizon_hours
        The pipeline's step on TIHM, the published run's, and the horizon of
        the silent-home rule there.
    tihm_checked_homes
        The TIHM homes, the first by identifier, on which the replay is
        checked against the pipeline run with the calibrated reference
        itself.
    null_record_sha256, null_expected
        The digest of the record of the measurement on synthetic days, and
        where that record puts the calibrated reference at the default's
        false reports.
    planning_record_sha256, planning_expected
        The digest of the record of the trials on curves written down by
        hand, and the ranges of it that this declaration quotes.
    tihm_records_sha256
        The digests of the published TIHM records the description must
        reproduce.
    """

    seed_root: int = 20261008
    homes: int = 400
    days: int = 84
    step_minutes: int = 15
    curve_scales: tuple[float, ...] = (
        0.3,
        0.325,
        0.35,
        0.375,
        0.4,
        0.425,
        0.45,
        0.475,
        0.5,
        0.525,
        0.55,
        0.575,
        0.6,
        0.625,
        0.65,
        0.675,
        0.7,
        0.725,
        0.75,
        0.775,
        0.8,
        0.825,
        0.85,
        0.875,
        0.9,
        0.925,
        0.95,
        0.975,
        1.0,
        1.05,
        1.1,
        1.15,
        1.2,
        1.25,
        1.3,
        1.35,
        1.4,
        1.45,
        1.5,
        1.6,
        1.8,
        2.0,
    )
    check_scales: tuple[float, ...] = (1.0, 0.6, 0.5)
    change_day: int = 56
    change_sleep_delta_hours: float = 1.6
    change_night_bathroom_extra: float = 1.2
    small_fraction: float = 0.5
    gradual_day: int = 35
    gradual_ramp_days: int = 28
    max_delay_days: float = 21.0
    gradual_max_delay_days: float = 42.0
    informative_detection: float = 0.20
    calibration_tolerance: float = 0.25
    calibration_thresholds: tuple[float, ...] = (1.5, 2.0, 3.0)
    planned_sd: float = 0.6
    checked_homes: int = 10
    resamples: int = 5000
    confidence: float = 0.95
    seed: int = 0
    tihm_step_minutes: int = 10
    tihm_horizon_hours: float = 12.0
    tihm_checked_homes: int = 5
    null_record_sha256: str = ""
    null_expected: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    planning_record_sha256: str = ""
    planning_expected: tuple[tuple[str, float], ...] = ()
    tihm_records_sha256: tuple[tuple[str, str], ...] = ()

    #: What had been seen when the protocol was written.
    inspected_before: tuple[str, ...] = (
        "the TIHM alert-burden run and the descriptions made after it had been "
        "read, among them how often a Gaussian value passes the threshold "
        "against a median and MAD of a few others. The calibrated reference "
        "was built from that, which is why nothing on TIHM here is a test",
        "the silent-home record had been read, in its totals and then home by "
        "home. Its homes are of the same design as these, from another root, "
        "and it gives what the default reference does on them: 99 behavioural "
        "alerts in 8,400 person-days of stable homes, 92 of them about a "
        "gradual drift and 49 of them raised when a day of the fifth or the "
        "sixth week closed; "
        "the step change detected in 75 of 100 homes at a median of 10 days, "
        "first by a drift verdict in 69 of the 75; and 12 of 100 stable homes "
        "meeting the detection definition in the step's window. So what the "
        "default will show here was known in advance, and what the calibrated "
        "reference will show was not",
        "the alerts of simulated homes that are in neither protocol had been "
        "read when the silent-home scoring was tested and reviewed, under the "
        "default reference, as that record lists",
        "the calibrated reference had been run on synthetic Gaussian days: in "
        "exploratory runs that are not recorded, in one full measurement that "
        "is not published, and then in the measurement recorded in "
        f"`{NULL_RECORD}`. It was changed twice because of what such days "
        "showed. Its degrees of freedom went from half a degree for each "
        "residual to 0.368, because the first passed a threshold of three "
        "more often than it states. And its trend, which was fitted to the "
        "days as they are, was fitted instead to each day's distance from the "
        "centre of its own weekday, because a weekly rhythm made it report "
        "drifts",
        "the calibrated reference had been exercised in unit tests on "
        "hand-built series and on one hand-built home of three event "
        "sensors. No simulated household had been run with it",
        "one simulated household, seed 999, which is not a home of this "
        "protocol, had been run with the default reference, with its event "
        "sensors alone and with all its sensors, to measure run time; its "
        "numbers of observations and its run times were read, and none of "
        "its alerts, verdicts or summaries",
        "the simulator's source had been read. It gives the resident a longer "
        "morning at the weekend, so the hours of sleep it plans differ by "
        "weekday",
        "a draft of this protocol was reviewed before it was frozen. The "
        "reviewer read the silent-home record home by home and ran synthetic "
        "Gaussian series, and no simulated home. The number of homes, the "
        "detection net of chance, the tolerance of the criterion on what a "
        "threshold means and the window of the gradual change were set after "
        "that review",
        "the scoring code had been run end to end before the freeze on six "
        "simulated homes that are not the protocol's, drawn from root 31, "
        "with the default reference alone: the default at other multiples of "
        "its own thresholds stood in for the calibrated conditions. What the "
        "default reference does on those six homes was read, and so were "
        "their days: the hours of sleep the pipeline infers there have a "
        "standard deviation of about 1.3 hours and are more than an hour "
        "longer at the weekend. The calibrated reference was not run on them",
        "in a first draft the calibrated reference was to be tested at two "
        "multiples chosen on synthetic days. The measurement on synthetic "
        "days, and the weekly rhythm of the homes above, made it likely "
        "that such multiples would not match the default in a simulated "
        "home, and that the comparison would then decide nothing",
        "in a second draft the two multiples were read off a quarter of the "
        "protocol's homes and tested on the rest, with a margin for how much "
        "detection could be lost. A second review, before the freeze, drew "
        "outcomes from operating curves written down by hand and found that "
        "a calibrated reference with the default's own curve would then be "
        "called better at the same detection in about one run in ten, and "
        "worse at the same false alerts as often: a multiple of the grid "
        "lies to one side of the match, and the margin let a reference slide "
        "along its own curve. The match is therefore placed on the curve "
        "itself, between two multiples, and found again in every resample. "
        "That reviewer read the record and the page of the six homes above "
        "and the silent-home record's alerts home by home, ran the "
        "calibrated reference on synthetic Gaussian series, and ran no "
        "simulated home",
        "the scoring code, once the match was placed on the curve, was run "
        "once more on the six homes above, from the replays kept from the "
        "first time, in which the default at other multiples of its own "
        "thresholds stands in for the calibrated conditions. No home was run "
        "again. The stand-in has the default's own curve, and the match came "
        "out where it is known to be, with a difference of zero",
        "a third review, before the freeze, read the protocol, the scoring, "
        "the trials on hand-written curves, the pages, the scripts and the "
        "tests, and both records of the measurement on synthetic days. It "
        "ran the declaration and both pages on made-up values and on twelve "
        "homes built by hand, drew 72,000 cohorts from operating curves of "
        "its own and decided the criterion on them with the scoring's own "
        "functions, and tested the unit tests against altered copies of the "
        "code. It ran no simulated home. It found that the straight line "
        "between two multiples of the grid, which then had 13, moved the "
        "criterion well past its stated error where neighbouring multiples "
        "were far apart. The grid was made three times as fine after that, "
        "from a scan of smooth curves written down by hand, in which no draw "
        "was made and no home run, and the trials were widened over where "
        "the match can fall",
        "one home of the six above, seed 140752, was simulated once more "
        "with the default reference, to time a replay against a run of the "
        "pipeline. Only the times were read",
    )

    def __post_init__(self) -> None:
        """Validate the declaration."""
        if self.homes < 2:
            raise ValueError("there must be at least two homes")
        if not 1 <= self.checked_homes <= self.homes:
            raise ValueError("checked_homes must be some of the homes")
        if self.tihm_checked_homes < 1:
            raise ValueError("tihm_checked_homes must be at least 1")
        if sorted(self.curve_scales) != list(self.curve_scales):
            raise ValueError("curve scales must be in increasing order")
        scales = self.curve_scales
        if any(scale <= 0 for scale in scales) or len(set(scales)) != len(scales):
            raise ValueError("curve scales must be positive and distinct")
        for scale in (1.0, *self.check_scales):
            if scale not in scales:
                raise ValueError("a declared scale must be one of the curve's")
        if not 0.0 < self.small_fraction < 1.0:
            raise ValueError("small_fraction must lie between 0 and 1")
        if self.calibration_tolerance <= 0.0 or self.planned_sd <= 0.0:
            raise ValueError("calibration_tolerance and planned_sd must be positive")
        if not self.calibration_thresholds or min(self.calibration_thresholds) <= 0:
            raise ValueError("calibration thresholds must be positive")
        baseline = BaselineConfig()
        if min(self.change_day, self.gradual_day) < baseline.trend_window:
            raise ValueError("a change must begin after the trend window has filled")
        if self.gradual_ramp_days < 1:
            raise ValueError("the gradual change must take at least a day")
        if (
            max(
                self.change_day + 1 + self.max_delay_days,
                self.gradual_day + 2 + self.gradual_max_delay_days,
            )
            > self.days
        ):
            raise ValueError("the record is too short for its windows")

    # ------------------------------------------------------------------
    @property
    def default(self) -> str:
        """The default reference at its declared thresholds: what ships."""
        return condition_name(DEFAULT, 1.0)

    @property
    def declared(self) -> str:
        """The calibrated reference at the declared thresholds."""
        return condition_name(CALIBRATED, 1.0)

    @property
    def conditions(self) -> tuple[str, ...]:
        """Every condition that is replayed: both references at every scale."""
        return tuple(
            condition_name(reference, scale)
            for reference in REFERENCES
            for scale in self.curve_scales
        )

    @property
    def checked_conditions(self) -> tuple[str, ...]:
        """The calibrated conditions the replay is checked at."""
        return tuple(condition_name(CALIBRATED, scale) for scale in self.check_scales)

    def shown(self, near: float) -> tuple[str, str, str]:
        """The conditions described in full, once the match is placed.

        The default, the calibrated reference as declared, and the calibrated
        reference at *near*, the multiple of the grid given for the match at
        the same false alerts.
        """
        return (self.default, self.declared, condition_name(CALIBRATED, near))

    def baseline(self, condition: str) -> BaselineConfig:
        """The baseline's configuration under a condition."""
        reference, _, scale = condition.partition("@")
        if condition not in self.conditions:
            raise KeyError(f"no condition '{condition}'")
        declared = BaselineConfig()
        return BaselineConfig(
            calibrated=reference == CALIBRATED,
            deviation_threshold=declared.deviation_threshold * float(scale),
            trend_threshold=declared.trend_threshold * float(scale),
        )

    def shift(self, arm: str) -> BehaviourShift | None:
        """The behavioural change an arm injects; ``None`` for the stable arm."""
        if arm == STABLE:
            return None
        if arm == CHANGE:
            return BehaviourShift(
                start_day=self.change_day,
                sleep_delta_hours=self.change_sleep_delta_hours,
                night_bathroom_extra=self.change_night_bathroom_extra,
            )
        if arm == SMALL_CHANGE:
            return BehaviourShift(
                start_day=self.change_day,
                sleep_delta_hours=self.change_sleep_delta_hours * self.small_fraction,
                night_bathroom_extra=self.change_night_bathroom_extra
                * self.small_fraction,
            )
        if arm == GRADUAL_CHANGE:
            return BehaviourShift(
                start_day=self.gradual_day,
                sleep_delta_hours=self.change_sleep_delta_hours,
                night_bathroom_extra=self.change_night_bathroom_extra,
                ramp_days=self.gradual_ramp_days,
            )
        raise KeyError(f"no arm '{arm}'")

    def first_changed_day(self, arm: str) -> int:
        """The first day of the record on which an arm's change moves anything.

        A step is in full effect on the day it is declared from. A ramp is at
        none of its size on that day and at one part in its length on the
        next, so the first day it changes is the day after.
        """
        shift = self.shift(arm)
        if shift is None:
            raise KeyError(f"'{arm}' has no change")
        return next(
            day
            for day in range(shift.start_day, self.days)
            if shift.strength_on(day) > 0.0
        )

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

    def checked_seeds(self) -> tuple[int, ...]:
        """The homes on which the replay is checked against the pipeline."""
        return self.study_seeds()[: self.checked_homes]

    def start(self) -> date:
        """The local date every simulated record begins on."""
        begins: date = HouseholdConfig().start
        return begins

    def local(self, day: int, hour: float = 0.0) -> datetime:
        """The instant of a local hour on a day of the record, in UTC."""
        zone = HouseholdConfig().tz
        naive = datetime.combine(self.start() + timedelta(days=day), time.min)
        return (
            (naive + timedelta(hours=hour))
            .replace(tzinfo=zone)
            .astimezone(timezone.utc)
        )

    def begins(self, arm: str) -> datetime:
        """Local midnight at the start of the first day an arm's change moves."""
        return self.local(self.first_changed_day(arm))

    def detection_window(self, arm: str) -> tuple[datetime, datetime]:
        """The window in which an alert counts as detecting an arm's change.

        The pipeline raises behavioural alerts when a day closes. The window
        opens when the first changed day closes, so that an alert raised at
        the close of a day that holds none of the change is outside it. It is
        closed on the left and open on the right.
        """
        begin = self.local(self.first_changed_day(arm) + 1)
        days = (
            self.gradual_max_delay_days
            if arm == GRADUAL_CHANGE
            else self.max_delay_days
        )
        return begin, begin + timedelta(days=days)

    def standard_error_planned(self, homes: int | None = None) -> float:
        """The standard error of E1, as planned."""
        count = self.homes if homes is None else homes
        return float(self.planned_sd / np.sqrt(count))

    def planned_power(self, difference: float, homes: int | None = None) -> float:
        """How often C1 would be a success, as planned, at a true difference.

        C1 is a success when the lower bound of E1's interval is above zero,
        so the estimate must exceed the interval's half width. The interval
        is taken as normal.
        """
        error = self.standard_error_planned(homes)
        half = NormalDist().inv_cdf(1.0 - (1.0 - self.confidence) / 2.0)
        return NormalDist().cdf(difference / error - half)

    def planned_difference(self, power: float, homes: int | None = None) -> float:
        """The true difference at which C1 would be a success that often."""
        error = self.standard_error_planned(homes)
        half = NormalDist().inv_cdf(1.0 - (1.0 - self.confidence) / 2.0)
        return float((half + NormalDist().inv_cdf(power)) * error)

    # ------------------------------------------------------------------
    def _runs(self, name: str) -> str:
        """A range of the planning record's shares, as runs in 100."""
        planned = dict(self.planning_expected)
        low, high = 100.0 * planned[f"{name}_low"], 100.0 * planned[f"{name}_high"]
        if f"{low:.0f}" == f"{high:.0f}":
            return f"{low:.0f}"
        return f"{low:.0f} to {high:.0f}"

    def _by_default(self, name: str) -> str:
        """A range for each default, as runs in 100."""
        parts = [
            f"{self._runs(f'{name}_{shape}')} where "
            + ("the default" if place == 0 else "it")
            + f" finds {words}"
            + (" steps" if place == 0 else "")
            for place, (shape, words) in enumerate(PLANNED_DEFAULTS.items())
        ]
        return ", ".join(parts[:-1]) + " and " + parts[-1]

    def _signed(self, name: str) -> str:
        """A range of the planning record's estimates, with their signs."""
        planned = dict(self.planning_expected)
        low, high = planned[f"{name}_low"], planned[f"{name}_high"]
        return f"from {low:+.3f} to {high:+.3f}"

    def _tried(self) -> list[str]:
        """What the trials on curves written down by hand showed, as recorded."""
        if not self.planning_expected:
            return []
        planned = dict(self.planning_expected)
        return [
            "the criterion as it now stands was then tried, before the "
            "freeze, on outcomes drawn from operating curves written down by "
            f"hand, as recorded in `{PLANNING_RECORD}`: counts of false alerts "
            "and detections with the sizes the silent-home record gives the "
            "default, for a default that finds most steps, nearly all of them "
            "or few, for a calibrated reference with the same curve, a better "
            "one and a worse one, and with the match placed from "
            f"{planned['offset_low']:g} to {planned['offset_high']:g} times the "
            "declared thresholds. No home was simulated for it and no "
            "reference was run. Two matches were tried, and no estimand is "
            "stated at the same detection, for the reason the matching rule "
            "gives. At the same false alerts, a reference with the default's "
            f"own curve was called better in {self._runs('same_curve_better')} "
            f"runs in 100 and worse in {self._runs('same_curve_worse')}, where "
            f"{50.0 * (1.0 - self.confidence):g} of each are expected, and its "
            "estimate was on average "
            f"{self._signed('same_curve_estimate')}, where the truth is zero. "
            "The estimate's standard error was from "
            f"{planned['error_low']:.3f} to {planned['error_high']:.3f}",
        ]

    def _seen(self) -> dict[str, Any]:
        baseline = BaselineConfig()
        return {
            "the_option": {
                "setting": "BaselineConfig.calibrated",
                "default": "off",
                "what_it_changes": "the centre stays weekday-aware; the scale "
                "is pooled over every retained day, from how far each fell "
                "from the centre the other days of its weekday would have "
                "given it; the deviation is mapped through a Student t with "
                f"{DOF_PER_RESIDUAL:g} degrees of freedom for each day behind "
                "the scale; the trend is fitted to those same distances and "
                "not to the days as they are, and its movement is measured "
                "against the same scale and is not mapped",
                "designed_on": "synthetic Gaussian days, which are not evidence "
                "about a home",
                "null_record": NULL_RECORD,
                "null_record_sha256": self.null_record_sha256,
            },
            "planned_on": {
                "record": PLANNING_RECORD,
                "sha256": self.planning_record_sha256,
                "what": "trials of the criterion on outcomes drawn from "
                "operating curves written down by hand. No home is simulated "
                "in them and no reference is run",
                "quoted": dict(self.planning_expected),
            },
            "conditions": {
                "naming": "`reference@scale`: the reference, and the multiple "
                "of the declared thresholds it is run at. Both thresholds are "
                "multiplied together: at scale s the deviation threshold is "
                f"{baseline.deviation_threshold:g} s and the trend threshold "
                f"{baseline.trend_threshold:g} s",
                "default": self.default,
                "declared": self.declared,
                "roles": {
                    "default": "the default reference at its declared "
                    "thresholds: the pipeline as it ships",
                    "declared": "the calibrated reference at the same thresholds",
                    "same_false_alerts": "the calibrated reference at the "
                    "multiple of the grid nearest its match at the same false "
                    "alerts",
                },
                "role_order": ["default", "declared", "same_false_alerts"],
                "why_not_as_declared": "at the declared thresholds the "
                "calibrated reference reports far less of everything, so a "
                "comparison there would show fewer false alerts and fewer "
                "detections and decide nothing. Two rules can be compared when "
                "they report falsely as often",
                "described": list(self.conditions),
                "curve_scales": list(self.curve_scales),
            },
            "matching": {
                "order": [
                    "curve",
                    "same_false_alerts",
                    "bracketed",
                    "what_it_uses",
                    "why",
                    "why_not_the_same_detection",
                    "nearest_multiple",
                ],
                "curve": "a reference's operating curve is the points its "
                "multiples give, in the order of the multiples: mean false "
                "alerts per home, and mean excess detection in each change "
                "arm, over the homes. Between two neighbouring multiples the "
                "curve is the straight line between their two points, and a "
                "place on that line has the multiple the same share of the "
                "way between the two",
                "same_false_alerts": "take the smallest multiple of the "
                "calibrated reference, which is its most sensitive thresholds, "
                "whose mean false alerts per home are at most those of "
                f"`{self.default}`. The match is on the line from that multiple "
                "to the next smaller one, where the false alerts equal the "
                "default's. The calibrated reference's excess detection is "
                "read there, in each change arm",
                "bracketed": "the match is bracketed when some multiple has no "
                "more false alerts than the default and the smallest multiple "
                "of the grid has more. When no multiple has so few the match "
                "is not reached, and when the smallest multiple already has "
                "so few it lies at the end of the grid. In both cases nothing "
                "is read",
                "what_it_uses": "the rule is applied to the means over the "
                "homes in hand. In the bootstrap those are the resampled "
                "homes, so the match moves from one resample to the next and "
                "the interval carries that",
                "why": "where the two references match depends on how far a "
                "home's days are from Gaussian and on its weekly rhythm, "
                "which synthetic days do not settle, and no multiple of a grid "
                "falls exactly on a match. A multiple chosen on some homes and "
                "tested on others lies to one side of it, and that side decides "
                "the comparison",
                "why_not_the_same_detection": self._why_not_the_same_detection(),
                "nearest_multiple": "the tables that describe a condition in "
                "full need a multiple that was run. That is the multiple of "
                "the grid nearest the match on all the homes, the smaller of "
                "two equally near; the smallest multiple when the match lies "
                "at the end of the grid, and the largest when it is not "
                "reached. This condition is chosen from the data and "
                "describes; no criterion reads it",
            },
            "what_the_outcomes_turn_on": "under the default reference most "
            "alerts come from the trend and not from a run of deviating days: "
            "in the silent-home record 92 of 99 false alerts and 69 of 75 "
            "first detections are gradual-drift verdicts. So C1 mostly "
            "compares two trend rules: the days' own trend over a scale of a "
            "few same-weekday days, against the trend of the days' distance "
            "from their weekday over the pooled scale, at a lower threshold. "
            "The Student t, which is what makes the deviation threshold mean "
            "what it says, does not touch the trend. C2 is the criterion "
            "about the deviation threshold itself. The alert policy stands "
            "between a verdict and an alert and grades a deviation against "
            "its threshold, so a calibrated score, which is smaller than the "
            "raw one, is graded lower; how many verdicts become alerts is "
            "reported for every condition described in full. Whether a run of "
            "deviating days is called persistent or abrupt is decided, under "
            "both references, on the days as they are, so under the "
            "calibrated one that label, and no alert, can still turn on a "
            "weekly rhythm",
        }

    def _why_not_the_same_detection(self) -> str:
        reason = (
            "the other way to match two rules is where they find as much, and "
            "to read the false alerts there. No estimand does. Excess "
            "detection falls away at both ends of the grid: at strict "
            "thresholds little is found, and at loose ones the stable record "
            "is alerted on as often. So a curve can have the default's "
            "detection at two places, and where the default ships near the "
            "most its own curve finds, the same detection is found again at "
            "stricter thresholds with fewer false alerts, on the same curve"
        )
        planned = dict(self.planning_expected)
        if planned:
            reason += (
                ". In the draws from curves written down by hand, where the "
                "default finds nearly every step, a calibrated reference with "
                "the default's own curve was called better there in "
                f"{self._runs('same_detection_same_curve_better')} runs in "
                f"100, where {50.0 * (1.0 - self.confidence):g} are expected. "
                "One with half as many false alerts again had no match there "
                f"in {self._runs('same_detection_half_more_not_bracketed')} "
                "runs in 100"
            )
            if "same_detection_half_more_estimate_low" in planned:
                reason += (
                    ", and where it had one its estimate was on average "
                    f"{self._signed('same_detection_half_more_estimate')} "
                    "alerts a home, where the truth is more"
                )
        return reason + (
            ". False alerts are expected to fall as the thresholds rise, so "
            "the match at the same false alerts should have one place; where "
            "they do not, the rule takes the crossing at the most sensitive "
            "thresholds"
        )

    def _simulator(self) -> dict[str, Any]:
        baseline, policy, pipeline = BaselineConfig(), AlertPolicy(), PipelineConfig()
        household = HouseholdConfig()
        small = self.shift(SMALL_CHANGE)
        assert small is not None
        return {
            "evidence": "simulated: every home comes from this repository's "
            "simulator, so every result is a statement about the simulator",
            "homes": self.homes,
            "seed_root": self.seed_root,
            "seeds": "the first distinct values of "
            "`numpy.random.SeedSequence(seed_root).generate_state(4 * "
            "homes, uint32)` modulo 1,000,000, sorted. Two of them can be "
            "neighbouring numbers. Each is given whole to the simulator's "
            "generator and no arithmetic is done on a seed, so neighbours "
            "are as unrelated as any other two",
            "study_seeds": list(self.study_seeds()),
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
            "why_left_out": "as in the silent-home protocol, so that the "
            "homes are streams of activations, as in TIHM, and what the "
            "default reference does on them is already on record",
            "delivery": "every arm passes through "
            "`sensor_modeling.simulation.faults.degrade` with no loss, "
            "lateness, duplication or fault",
            "pipeline": {
                "step_minutes": self.step_minutes,
                "what": "`BehaviouralSensingPipeline` over the event "
                "sensors' registry, with default emissions derived from it "
                "and every default setting; nothing is fitted",
                "features": [f"{s.value}_hours" for s in pipeline.features],
                "min_day_coverage": pipeline.min_day_coverage,
                "min_day_observed": pipeline.min_day_observed,
                "baseline": {
                    "min_samples": baseline.min_samples,
                    "weekday_min_samples": baseline.weekday_min_samples,
                    "deviation_threshold": baseline.deviation_threshold,
                    "persistence_days": baseline.persistence_days,
                    "trend_window": baseline.trend_window,
                    "trend_threshold": baseline.trend_threshold,
                    "min_scale": baseline.min_scale,
                    "history_days": baseline.history_days,
                    "change_point_penalty": baseline.change_point_penalty,
                },
                "alert_policy": {
                    "min_score": policy.min_score,
                    "min_confidence": policy.min_confidence,
                    "cooldown_hours": policy.cooldown.total_seconds() / 3600.0,
                    "max_per_window": policy.max_per_window,
                    "storm_window_hours": policy.storm_window.total_seconds() / 3600.0,
                },
                "every_other_setting": "the defaults, which the frozen file "
                "records in full beside the digests of the source files "
                "they live in",
            },
            "replay": {
                "what": "each arm of each home is run through the pipeline "
                "once, with the default reference. Every condition is then "
                "`sensor_modeling.online.replay_days` over the steps of "
                "that run: the same days, at the same moments, through "
                "fresh baselines and a fresh alert engine under the "
                "condition's thresholds",
                "why": "nothing upstream of a day's summary depends on the "
                "baseline's configuration, so the conditions differ in the "
                "baseline alone, and "
                f"{len(self.conditions)} of them cost little more than one",
                "check": "in every home and arm the replay under "
                f"`{self.default}` must return the verdicts and alerts of the "
                f"pipeline's own run. In the {self.checked_homes} homes with "
                "the smallest seeds the pipeline "
                "is also run with the calibrated reference at "
                + ", ".join(f"{scale:g}" for scale in self.check_scales)
                + " times the declared thresholds, in every arm, and the "
                "replay must return those runs' verdicts and alerts. If any "
                "of this fails nothing is reported",
                "checked_conditions": list(self.checked_conditions),
                "checked_seeds": list(self.checked_seeds()),
            },
            "pairing": "every condition is replayed over the same run, so "
            "two conditions differ in the reference and its thresholds "
            "alone. A home with a change shares its seed with the same home "
            "without it and not its days: the simulator draws the sensor "
            "events after it has planned every day, and from the change on "
            "the days themselves are drawn again. So no estimand compares "
            "the two arms day by day; a detection is set against the same "
            "window of the stable record as a rate, home by home",
            "arm_order": list(ARMS),
            "arms": {
                STABLE: "nothing is injected. Every behavioural alert is "
                "a false one",
                CHANGE: "the detection study's step change from day "
                f"{self.change_day}: the resident wakes "
                f"{self.change_sleep_delta_hours:g} hours earlier, with "
                f"{self.change_night_bathroom_extra:g} more night-time "
                "bathroom trips expected a night",
                SMALL_CHANGE: f"a step from day {self.change_day} of "
                f"{self.small_fraction:g} of that size: "
                f"{small.sleep_delta_hours:g} hours and "
                f"{small.night_bathroom_extra:g} trips",
                GRADUAL_CHANGE: "the step change's size reached gradually, "
                f"declared from day {self.gradual_day} over "
                f"{self.gradual_ramp_days} days. The simulator leaves day "
                f"{self.gradual_day} itself unchanged, so the first changed "
                f"day is day {self.first_changed_day(GRADUAL_CHANGE)}, by one "
                f"part in {self.gradual_ramp_days}",
            },
            "calendar": "every home has the same calendar. Day 0 is a "
            "Monday, both changes are declared from a Monday, and the day "
            "the clocks go forward, which is an hour short, is day 27, "
            "before either change and inside every reference",
            "replications": f"{self.homes} paired homes. "
            "`docs/SIMULATION_PROTOCOLS.md` sets a floor of 100 and asks "
            "that the number follow from the error that can be accepted",
            "planning": self._planning(),
        }

    def _planning(self) -> str:
        """How many homes, and what that many can show."""
        error = self.standard_error_planned()
        planned = (
            f"C1 is planned with a standard error of {error:.3f} for E1, which "
            f"is a standard deviation of {self.planned_sd:g} for a home over "
            f"{self.homes} homes. It would then be a success in 80 runs of 100 "
            "if the calibrated reference found "
            f"{self.planned_difference(0.8):.3f} more of the homes than the "
            "default at the same false alerts, and in "
            f"{100.0 * self.planned_power(0.05):.0f} of 100 if it found 0.05 "
            "more"
        )
        if not self.planning_expected:
            return planned
        return (
            "no simulated home has been run with the calibrated reference, so "
            "the number of homes was planned on outcomes drawn from operating "
            f"curves written down by hand, as recorded in `{PLANNING_RECORD}`. "
            f"{planned}. In those draws a reference with half the default's "
            "false alerts at every detection was called better in "
            f"{self._by_default('half_better')}. One with a quarter fewer was "
            f"called better in {self._by_default('quarter_fewer_better')}. One "
            "with half as many false alerts again was called worse in "
            f"{self._by_default('half_more_worse')}. All are runs in 100. Where "
            "the default already finds nearly every step there is little more "
            "to be found, and a better reference shows itself more in false "
            "alerts, which C1 does not read"
        )

    def _definitions(self) -> dict[str, Any]:
        gradual_first = self.first_changed_day(GRADUAL_CHANGE)
        return {
            "order": [
                "behavioural_alert",
                "false_alerts",
                "first_changed_day",
                "window",
                "detected",
                "false_detection",
                "excess_detection",
                "same_false_alerts",
                "delay",
                "evaluable_feature_day",
                "deviating",
                "stated",
                "equivalent_threshold",
                "lies_below_and_above",
                "person_days",
            ],
            "behavioural_alert": "an alert of kind behavioural_change, dated "
            "by the moment it was raised",
            "false_alerts": "a home's behavioural alerts in the whole of "
            f"its `{STABLE}` record, about any feature",
            "first_changed_day": f"day {self.change_day} for the two steps, "
            f"and day {gradual_first} for the gradual change",
            "window": "from local midnight at the end of the first changed "
            f"day, for {self.max_delay_days:g} days, or "
            f"{self.gradual_max_delay_days:g} days for the gradual change; "
            "closed on the left and open on the right. The pipeline raises "
            "behavioural alerts when a day closes, so an alert raised at the "
            "close of a day that holds none of the change is outside it",
            "detected": "a home with a behavioural alert about "
            f"`{TRACKED_FEATURE}` raised in its arm's window. The direction "
            "of the alert is not asked",
            "false_detection": "the same definition met in the same home's "
            f"`{STABLE}` record, in the same window, under the same condition",
            "excess_detection": "for a home, an arm and a condition: one if "
            "detected and zero if not, minus one if a false detection and "
            "zero if not. Its mean over homes is the share detected minus "
            "the share of false detections, which is what a condition "
            "detects beyond what it would have raised anyway",
            "same_false_alerts": "the place on the calibrated reference's "
            "operating curve that has the false alerts per home of "
            f"`{self.default}`, as the matching rule places it",
            "delay": "days from local midnight at the start of the first "
            "changed day to the first alert that counts",
            "evaluable_feature_day": "a day and a feature on which the "
            "baseline gave a verdict other than insufficient data",
            "deviating": "an evaluable feature-day whose deviation is at "
            "or past a threshold in absolute value",
            "stated": "the share of Gaussian values at or past a "
            "threshold in absolute value: "
            + ", ".join(
                f"{nominal(threshold):.4f} at {threshold:g}"
                for threshold in self.calibration_thresholds
            ),
            "equivalent_threshold": "the threshold at which Gaussian values "
            "would be deviating as often as a share that was observed: the "
            "inverse of the line above",
            "lies_below_and_above": "an interval lies below a value when its "
            "upper bound is strictly below it, and above a value when its "
            "lower bound is strictly above it",
            "person_days": "homes times days",
        }

    def _estimands(self) -> dict[str, str]:
        default, declared = self.default, self.declared
        criterion = ", ".join(f"{t:g}" for t in self.calibration_thresholds)
        return {
            "E1": "the calibrated reference's mean excess detection in "
            f"`{CHANGE}` at the same false alerts, minus that of `{default}`",
            "E2": f"for `{declared}`, mean false alerts per home and mean "
            f"excess detection in `{CHANGE}`, each minus the same under "
            f"`{default}`, paired by home: what the calibrated reference costs "
            "and saves when the thresholds are left where they are. No "
            "criterion is stated on them",
            "E3": f"in the `{STABLE}` records, the share of evaluable "
            f"feature-days of `{TRACKED_FEATURE}` whose deviation under the "
            f"calibrated reference is at or past each of {criterion}, as a "
            "ratio of sums over homes, and the threshold it is equivalent to. "
            "Described beside it, with no criterion: the same under the "
            f"default reference; at each of {criterion}, the same before and "
            "after the reference becomes weekday-aware, and the same for each "
            "other feature; and the share at or past each of "
            + ", ".join(f"{t:g}" for t in TAIL_THRESHOLDS)
            + " for both references",
            "E4": f"in `{SMALL_CHANGE}` and `{GRADUAL_CHANGE}`: the calibrated "
            "reference's mean excess detection at the same false alerts, read "
            "at the same place on its curve as E1, minus that of "
            f"`{default}`. No criterion is stated on them",
            "E5": "descriptions under each condition described in full, with "
            "no criterion: false alerts per home and per person-day, by "
            "feature, by the verdict that raised them and by the week of the "
            "day whose close raised them; in each change arm the share of "
            "homes detected, the share of false detections, the mean excess "
            "detection and their paired differences from the default, the "
            "pooled median delay, and the verdict and the direction of each "
            "home's first alert that counts; the same shares when the alert "
            "must also be of a decrease; detections by week of the window; "
            "and in every arm the change verdicts by kind, how many raised an "
            "alert, and how many did not because the day was not attributable "
            "enough",
            "E6": "under every described condition: mean false alerts per "
            "home and, in each change arm, the share detected and the mean "
            "excess detection. These are the two references' operating "
            "curves, and no criterion is stated on them. Beside E1 they are "
            "read one more way, fixed here, for each change arm on its own: "
            "for each multiple the default reference is run at, whether some "
            "multiple of the calibrated one has no more false alerts and no "
            "less excess detection; and the same question the other way "
            "round. That reading compares estimates on the same homes, takes "
            "the best of the multiples and carries no uncertainty, so it "
            "describes the curves and tests nothing",
        }

    def _criteria(self) -> dict[str, Any]:
        default = self.default
        tolerance = self.calibration_tolerance
        thresholds = self.calibration_thresholds
        return {
            "order": [
                "C1_more_is_detected_at_the_same_false_alerts",
                "C2_the_threshold_means_what_it_says",
            ],
            "C1_more_is_detected_at_the_same_false_alerts": "uninformative "
            f"when the mean excess detection in `{CHANGE}` under `{default}` "
            f"is below {self.informative_detection:g}; otherwise not bracketed "
            "when the match is not bracketed on the homes as they are; "
            "otherwise success when E1's interval lies above zero, failure "
            "when it lies below zero, and inconclusive otherwise",
            "C2_the_threshold_means_what_it_says": "decided for the "
            "calibrated reference at each of "
            + ", ".join(f"{t:g}" for t in thresholds)
            + ", on the interval of the threshold E3's share is equivalent "
            f"to: it holds when the interval lies within {tolerance:g} of the "
            "threshold on both sides; it does not hold when the interval "
            "lies wholly outside that band; inconclusive otherwise. The "
            "verdicts are points on one curve, since a day's deviation does "
            "not depend on the threshold it is read against",
            "margin_order": ["informative_detection", "calibration_tolerance"],
            "margins": {
                "informative_detection": f"{self.informative_detection:g} of excess "
                "detection: below it the default finds too little beyond "
                "chance for a comparison to mean anything",
                "calibration_tolerance": f"{tolerance:g} on the scale of the "
                f"threshold. At a threshold of {max(thresholds):g} that is a "
                f"share between {nominal(max(thresholds) + tolerance):.4f} "
                f"and {nominal(max(thresholds) - tolerance):.4f}, and at "
                f"{min(thresholds):g} between "
                f"{nominal(min(thresholds) + tolerance):.3f} and "
                f"{nominal(min(thresholds) - tolerance):.3f}",
            },
            "readings": [
                "the calibrated reference is better when C1 is a success: at "
                "the false alerts of the default as it ships, it finds more "
                "of the step change",
                "it is worse when C1 is a failure",
                "anything else is not shown, which is not the same as no " "difference",
                "a match that is not bracketed is reported as such, with "
                "whether it was not reached or lies at the end of the grid, "
                "and no claim follows from it",
                "C2 is decided threshold by threshold. The option's claim "
                "holds on these homes when it holds at all three, does not "
                "hold when it does not hold at one of them, and is not shown "
                "otherwise",
                "C2 is about the option's claim and not about its "
                "usefulness: a threshold can be miscalibrated on a simulated "
                "home and still be the better rule, and the reverse",
                "the intervals are 95% intervals, one for each estimand, and "
                "are not adjusted for there being two criteria. Only C1 and "
                "C2 are confirmatory; everything else is description",
                "C1 is stated at one point of the default's curve, the one "
                "that ships. A reference that is better there can be worse at "
                "another level of false alerts, and E6 is where that would "
                "be seen",
                f"false alerts are counted over a record of {self.days} "
                "days, and half of the default's come in the fifth and sixth "
                "weeks, when its weekday reference rests on four or five "
                "days. Over a longer record the default would raise fewer a "
                "day, and the match would lie elsewhere on the calibrated "
                "reference's curve",
            ],
        }

    def _tihm(self) -> dict[str, Any]:
        return {
            "standing": "a description, not a test: the calibrated "
            "reference was built from what these homes showed",
            "dataset": PROVENANCE.name,
            "citation": PROVENANCE.citation,
            "licence": PROVENANCE.licence,
            "archive_sha256": ARCHIVE_SHA256,
            "files": dict(FILES),
            "step_minutes": self.tihm_step_minutes,
            "conditions": list(self.conditions),
            "silent_home_rule": [RULE_OFF, f"h{self.tihm_horizon_hours:g}"],
            "checked_homes": self.tihm_checked_homes,
            "replay": "each home is run through the pipeline with the "
            "default reference, once with the silent-home rule off and "
            f"once with it on at {self.tihm_horizon_hours:g} hours, and "
            "every described condition is replayed over each run",
            "everything_else": "as frozen in "
            "`artifacts/tihm/alert_burden_protocol.json`",
            "published_records": dict(self.tihm_records_sha256),
            "check": "the replay under the default reference must return "
            "each run's verdicts and alerts. The homes run must be the homes "
            "of the published alert-burden record, each once, and with the "
            "rule off every one of them must have the monitored days and the "
            "behavioural alerts that record gives it. In the "
            f"{self.tihm_checked_homes} homes that come first by identifier, "
            "with the rule off, the pipeline is also run with the calibrated "
            "reference at the declared thresholds, and the replay under "
            f"`{self.declared}` must return that run's verdicts and alerts. "
            "Under both settings of "
            "the rule the totals must be those of the published records: "
            + "; ".join(
                f"{values['monitored_days']:,} monitored days and "
                f"{values['behavioural_alerts']} behavioural alerts with the "
                f"rule {'off' if rule == RULE_OFF else 'on'}"
                for rule, values in TIHM_PUBLISHED.items()
            )
            + ". Otherwise nothing is reported",
            "reported": [
                "monitored, usable and evaluable days",
                "the share of evaluable feature-days that are deviating, "
                "for each feature and each reference, beside what the "
                "threshold states, and the threshold it is equivalent to",
                "under every described condition: change verdicts by kind, "
                "and behavioural alerts in all and by the verdict that raised "
                f"them; and per home under `{self.default}` and "
                f"`{self.declared}`",
            ],
            "not_reported": "any relation to the dataset's labels. Whether "
            "what the pipeline raises relates to what a clinical team "
            "verified is the alert-burden protocol's question. Those labels "
            "have been read against the pipeline's output before, and "
            "more conditions would be more looks at them",
            "what_it_cannot_say": "fewer alerts on these homes are not "
            "better alerts. Nothing here says whether an alert that is no "
            "longer raised was a false one",
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the full declaration, in a stable serialisable form."""
        return {
            "schema": "threshold-calibration-protocol/1",
            "name": EXPERIMENT,
            "status": "pre-specified on simulated homes; descriptive on TIHM. "
            "The simulator's homes, estimands, criteria and matching rule "
            "are fixed before any simulated home is run with the calibrated "
            "reference. Nothing on TIHM is a test",
            "question": "whether a reference whose deviation threshold means "
            "what it says does better than the default one in a simulated "
            "home, when the two are compared at thresholds that make them "
            "report falsely as often",
            "inspected_before": [*self.inspected_before, *self._tried()],
            **self._seen(),
            "simulator": self._simulator(),
            "definitions": self._definitions(),
            "estimands": self._estimands(),
            "criteria": self._criteria(),
            "expected_from_synthetic_days": {
                "what": "where the record of the measurement on synthetic "
                "Gaussian days puts the calibrated reference at the default's "
                "false reports, as a multiple of its grid and by its own "
                "rule, on days with no weekly rhythm and on days with one, "
                "and how often each reference then reports a change in the "
                "three weeks from a step: with no step added, and with a step "
                "of two standard deviations. A simulated home's days are not "
                "Gaussian, so these are what the match found here can be read "
                "against, and they decide nothing",
                "settings": {
                    setting: dict(values) for setting, values in self.null_expected
                },
            },
            "bootstrap": {
                "order": [
                    "unit",
                    "resamples",
                    "confidence",
                    "seed",
                    "interval",
                    "shares",
                    "matched",
                    "ratio",
                    "median_delay",
                    "monte_carlo_error",
                ],
                "unit": "homes",
                "resamples": self.resamples,
                "confidence": self.confidence,
                "seed": self.seed,
                "interval": "percentile",
                "shares": "a share of homes has a Wilson interval, since "
                "homes are independent; a mean of excess detections and every "
                "paired difference are resampled over homes",
                "matched": "E1 and E4 are recomputed from the start in "
                "every resample: the calibrated reference's curve and the "
                "default's point are means over the resampled homes, and the "
                "match is placed on that curve. A resample in which the match "
                "is not bracketed has no value and counts against whichever "
                "claim a bound could support: above every value when the "
                "upper bound is taken, and below every value when the lower "
                "one is. The bounds are resampled values, with no "
                "interpolation between neighbours, that leave "
                f"{50.0 * (1.0 - self.confidence):g}% of the resamples outside "
                "on each side. So a bound does not exist when more than that "
                "share of the resamples have no match, and no claim can rest "
                "on it. How many resamples were bracketed, not reached and at "
                "the end of the grid is reported",
                "ratio": "a share of feature-days is a ratio of sums over "
                "homes, and each resample recomputes both sums. The interval "
                "of its equivalent threshold is the interval of the share, "
                "carried through the inverse",
                "median_delay": "pooled over the homes that detected; a "
                "resample in which no home detected has no median and is left "
                "out, and how many remained is reported",
                "monte_carlo_error": "for a mean paired difference, the "
                "standard deviation of the paired differences over the square "
                "root of the number of homes. For E1 and E4, the standard "
                "deviation of the resampled values over the resamples in "
                "which the match was bracketed",
            },
            "what_the_simulator_cannot_show": [
                "the simulated resident's days are drawn from distributions "
                "this project wrote down. How far a real day's hours are from "
                "them, and so how a threshold behaves on a real home, is not "
                "something a simulated home can say",
                "one size of step, half of it and one gradual change are "
                "injected, in one feature's direction, from one weekday. A "
                "reference that finds these may not find another kind of "
                "change",
                "the homes have event sensors alone. With sensors that report "
                "on a cadence the day's hours are inferred differently",
                "every home has the same calendar, so nothing here varies "
                "the weekday a change begins on or where the short day of "
                "the clock change falls",
                "the match is placed on the same homes it is read on. The "
                "interval carries that, and what it does not say is how a "
                "multiple chosen on one population of homes would do on "
                "another: these homes come from one simulator with one set of "
                "settings",
                "between two multiples the curve is taken to be a straight "
                "line, which can lie to either side of the true curve. The "
                "grid is fine where a match is likely, so that the gap is "
                "small beside the interval, and the trials on curves written "
                "down by hand say how often it moved a verdict there. Where "
                "the grid is coarse, above 1.5 times the declared thresholds, "
                "it can move one more often",
                "no estimate is given of how many fewer false alerts the "
                "calibrated reference raises at the same detection, for the "
                "reason the matching rule gives",
            ],
            "tihm": self._tihm(),
            "reporting": "every estimand, criterion, arm, condition and home, "
            "whatever it shows; no multiple, margin, window or rule is changed "
            "after scoring. The first run of this protocol that completes is "
            "its result. A run that stops before it completes, refused by a "
            "check or interrupted and not resumed with the same code, is not "
            "read, and the record of the run that completes lists every such "
            "attempt and why it stopped",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


#: The digest of the committed record of the measurement on synthetic days,
#: and where it puts the two multiples.
NULL_RECORD_SHA256 = "513c48884b69f956a1f842dbb3c7a695cd4400556bd406f2ee44843fb3c4711c"
NULL_EXPECTED: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = (
    (
        "plain",
        (
            ("same_false_alerts", 0.55),
            ("default_false", 0.085),
            ("default_found", 0.842),
            ("at_the_same_false_alerts_false", 0.053),
            ("at_the_same_false_alerts_found", 0.9685),
        ),
    ),
    (
        "weekly_rhythm",
        (
            ("same_false_alerts", 0.5),
            ("default_false", 0.145),
            ("default_found", 0.86),
            ("at_the_same_false_alerts_false", 0.093),
            ("at_the_same_false_alerts_found", 0.9885),
        ),
    ),
)

#: The digest of the committed record of the trials on curves written down by
#: hand, and the ranges of it that the declaration quotes.
PLANNING_RECORD_SHA256 = (
    "8ae86baca7d3ff90c76f70f02996b637d9a74b941936260e6f559be9d8fc970a"
)
PLANNING_EXPECTED: tuple[tuple[str, float], ...] = (
    ("offset_low", 0.45),
    ("offset_high", 1.1),
    ("same_curve_better_low", 0.017),
    ("same_curve_better_high", 0.035),
    ("same_curve_worse_low", 0.016),
    ("same_curve_worse_high", 0.038),
    ("same_curve_estimate_low", -0.002782545821513541),
    ("same_curve_estimate_high", 0.0006042056666903963),
    ("error_low", 0.02243782244956684),
    ("error_high", 0.036566435136005805),
    ("half_better_most_low", 0.991),
    ("half_better_most_high", 0.996),
    ("half_better_nearly_all_low", 0.671),
    ("half_better_nearly_all_high", 0.707),
    ("half_better_few_low", 0.982),
    ("half_better_few_high", 0.988),
    ("quarter_fewer_better_most_low", 0.457),
    ("quarter_fewer_better_most_high", 0.496),
    ("quarter_fewer_better_nearly_all_low", 0.178),
    ("quarter_fewer_better_nearly_all_high", 0.2),
    ("quarter_fewer_better_few_low", 0.38),
    ("quarter_fewer_better_few_high", 0.416),
    ("half_more_worse_most_low", 0.773),
    ("half_more_worse_most_high", 0.803),
    ("half_more_worse_nearly_all_low", 0.582),
    ("half_more_worse_nearly_all_high", 0.625),
    ("half_more_worse_few_low", 0.605),
    ("half_more_worse_few_high", 0.634),
    ("same_detection_same_curve_better_low", 0.053),
    ("same_detection_same_curve_better_high", 0.065),
    ("same_detection_half_more_not_bracketed_low", 0.915),
    ("same_detection_half_more_not_bracketed_high", 0.955),
    ("same_detection_half_more_estimate_low", -0.5068273613565015),
    ("same_detection_half_more_estimate_high", -0.4603814665272998),
)

#: The digests of the published TIHM records the description must reproduce.
TIHM_RECORDS_SHA256: tuple[tuple[str, str], ...] = (
    (
        "artifacts/tihm/tihm-alert-burden.json",
        "5a5bb24167c8f089de3138f9c35ec6e07cd902ba385999f172458e40b40a64ff",
    ),
    (
        "artifacts/silent_home/silent-home-tihm.json",
        "96798aa2e573f281477faad624ee7369ff2508d5a8631c7003f7f75218bae6a9",
    ),
)


#: The defaults the trials on hand-written curves were run for, by how much of
#: the step each finds at its declared thresholds.
PLANNED_DEFAULTS = {"most": "most", "nearly_all": "nearly all", "few": "few"}


def planned_from(record: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    """What a record of the trials on hand-written curves gives the protocol.

    Where on the grid the match was tried. At the same false alerts, over
    every cell: how often a reference with the default's own curve is called
    better and worse, the mean of its estimate and the estimate's standard
    error; and for each default, over its main cells, how often one with half
    the false alerts and one with a quarter fewer are called better and one
    with half as many again is called worse. At the same detection, where the
    default finds nearly every step: how often a reference with the same
    curve is called better, and for one with half as many false alerts
    again, how often it has no match and the mean of its estimate where it
    has one. A range with no value in the record is left out.
    """
    summary = record["results"]["summary"]
    quiet = summary["same_false_alerts"]
    shapes = quiet["by_shape"]
    kept = summary["same_detection"]["by_shape"]["nearly_all"]
    offsets = record["configuration"]["offsets"]
    named: dict[str, Any] = {
        "offset": {"low": min(offsets), "high": max(offsets)},
        "same_curve_better": quiet["1"]["better"],
        "same_curve_worse": quiet["1"]["worse"],
        "same_curve_estimate": quiet["1"]["mean_estimate"],
        "error": quiet["1"]["sd_of_estimates"],
        **{
            f"{name}_{shape}": shapes[shape][ratio][outcome]
            for name, ratio, outcome in (
                ("half_better", "0.5", "better"),
                ("quarter_fewer_better", "0.75", "better"),
                ("half_more_worse", "1.5", "worse"),
            )
            for shape in PLANNED_DEFAULTS
        },
        "same_detection_same_curve_better": kept["1"]["better"],
        "same_detection_half_more_not_bracketed": kept["1.5"]["not_bracketed"],
        "same_detection_half_more_estimate": kept["1.5"]["mean_estimate"],
    }
    return tuple(
        (f"{name}_{end}", float(found[end]))
        for name, found in named.items()
        if found is not None
        for end in ("low", "high")
    )


def expected_from(
    record: Mapping[str, Any],
) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    """What a record of the measurement on synthetic days gives the protocol.

    For each setting: the multiple of that record's grid at which its rule
    puts the calibrated reference at the default's false reports, and the
    shares of series with a change verdict in the window, for the default as
    declared and for the calibrated reference at that multiple.
    """
    operating = record["results"]["operating"]
    expected = []
    for setting, entry in record["results"]["matching"].items():
        quiet, moved = entry["nothing_changed"], entry["a_step"]
        values: list[tuple[str, float]] = [
            ("same_false_alerts", float(entry["same_false_reports"])),
        ]
        named = (
            ("default", DEFAULT, "1"),
            (
                "at_the_same_false_alerts",
                CALIBRATED,
                f"{entry['same_false_reports']:g}",
            ),
        )
        for label, reference, scale in named:
            for kind, shape in (("false", quiet), ("found", moved)):
                values.append(
                    (
                        f"{label}_{kind}",
                        float(operating[shape][reference][scale]["found"]["share"]),
                    )
                )
        expected.append((setting, tuple(values)))
    return tuple(expected)


def declared_protocol() -> ThresholdCalibrationProtocol:
    """The protocol as declared."""
    return ThresholdCalibrationProtocol(
        null_record_sha256=NULL_RECORD_SHA256,
        null_expected=NULL_EXPECTED,
        planning_record_sha256=PLANNING_RECORD_SHA256,
        planning_expected=PLANNING_EXPECTED,
        tihm_records_sha256=TIHM_RECORDS_SHA256,
    )


def check_inputs(
    protocol: ThresholdCalibrationProtocol, root: Path | None = None
) -> None:
    """Refuse a protocol whose pinned records are missing or are not the ones named.

    The measurement on synthetic days shaped the design, the trials on curves
    written down by hand planned it, and the published TIHM records are what
    the description must reproduce. A protocol that did not say which files
    those are could have them replaced under it.
    """
    root = _REPOSITORY if root is None else Path(root)
    if not protocol.null_record_sha256 or not protocol.null_expected:
        raise ValueError("the protocol does not pin the measurement on synthetic days")
    if file_sha256(root / NULL_RECORD) != protocol.null_record_sha256:
        raise ValueError(f"{NULL_RECORD} is not the record the protocol names")
    measured = json.loads((root / NULL_RECORD).read_text(encoding="utf-8"))
    if expected_from(measured) != protocol.null_expected:
        raise ValueError(
            f"what the protocol says {NULL_RECORD} gives is not what it holds"
        )
    if not protocol.planning_record_sha256 or not protocol.planning_expected:
        raise ValueError("the protocol does not pin the trials it was planned on")
    if file_sha256(root / PLANNING_RECORD) != protocol.planning_record_sha256:
        raise ValueError(f"{PLANNING_RECORD} is not the record the protocol names")
    tried = json.loads((root / PLANNING_RECORD).read_text(encoding="utf-8"))
    if planned_from(tried) != protocol.planning_expected:
        raise ValueError(
            f"what the protocol says {PLANNING_RECORD} gives is not what it holds"
        )
    # The trials decided the criterion with the scoring's own functions; if
    # those have changed since, the trials are not of the criterion scored.
    from .threshold_calibration_planning import scoring_sources

    if tried["configuration"].get("scoring_functions") != scoring_sources():
        raise ValueError(
            f"{PLANNING_RECORD} was made with other scoring functions than these"
        )
    pinned = dict(protocol.tihm_records_sha256)
    if set(pinned) != set(TIHM_RECORDS):
        raise ValueError("the protocol does not pin the published TIHM records")
    for name, digest in pinned.items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"{name} is not the record the protocol names")


def write_protocol(protocol: ThresholdCalibrationProtocol, path: Path) -> str:
    """Freeze the declaration: write it with its digest; return the digest.

    The file also records, under ``at_freeze``, the digest of every source
    file of the package and of the scripts that run the protocol, the
    versions of the numerical libraries, and every default setting. They are
    not part of the declaration or of its digest, so that the frozen protocol
    stays valid when that code later moves on; the run compares them and says
    what it found.
    """
    check_inputs(protocol)
    payload = {
        **protocol.to_dict(),
        "protocol_sha256": protocol.sha256(),
        "at_freeze": code_now(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return protocol.sha256()


def check_frozen_protocol(protocol: ThresholdCalibrationProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    frozen.pop("at_freeze", None)
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after scoring begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def code_changes(path: Path, root: Path | None = None) -> dict[str, list[str]]:
    """What differs now from what the frozen file recorded at the freeze.

    Returns the source files whose content has changed, among them files
    added and removed, the libraries whose version has, and the groups of
    default settings that have. All three are empty when the code a run uses
    is the code the protocol was frozen against.
    """
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
