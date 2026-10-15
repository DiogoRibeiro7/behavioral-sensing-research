"""An adaptive, non-stationary personal baseline.

Normal behaviour is not a fixed calibration window. People's routines shift
with the seasons, with recovery from illness, with a new medication, with a
grandchild moving in. A baseline frozen at enrolment slowly turns every one of
those into a permanent alarm, and a baseline that adapts instantly turns a real
decline into the new normal before anyone notices.

The model here treats behaviour as ``X_t ~ P_t(X)`` with a slowly moving
distribution, and separates the reasons a day can look unusual:

.. code-block:: text

    ordinary variability      within the personal band
    weekly periodicity        Sundays differ from Tuesdays, by design
    temporary disturbance     a few unusual days that revert
    persistent change         a shift that holds
    gradual drift             a slow monotone trend
    abrupt change             a step, located by change-point detection
    insufficient data         the apparatus was not watching

Two properties keep it defensible. The reference is *robust*: medians and MAD,
so a single extraordinary day cannot redefine normal. And the reference is
*weekday-aware*: a quiet Sunday is compared against other Sundays, not against
the working week, because otherwise ordinary weekly rhythm reads as change.

A third is opt-in. A weekday-aware reference is built from a handful of days:
four Sundays, then five. The spread of four values is so uncertain that a
stationary Gaussian day lies three of its robust standard deviations out about
one time in six, where a three-sigma band states one in 370. With
``BaselineConfig.calibrated`` the centre stays weekday-aware and the scale is
pooled: it is the spread of how far each retained day fell from the centre the
other days of its weekday would have given it. The deviation is then put on
the scale where the threshold means what it says, by the Student t that scale
has, and the trend is fitted to those same distances, so that a weekly rhythm
is not read as a drift. Nothing changes unless it is set.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

import numpy as np
from scipy import special

from ..models.change_point_detection.pelt import PELTChangePointDetector

logger = logging.getLogger(__name__)

#: Scale factor making the median absolute deviation a consistent estimator
#: of the standard deviation for normally distributed data.
MAD_TO_SIGMA = 1.4826

#: Degrees of freedom a calibrated scale has for each residual behind it. The
#: median absolute deviation is 37% as efficient as the standard deviation for
#: Gaussian data, so N residuals fix the scale about as well as a standard
#: deviation on 0.368 N degrees of freedom. The constant was set on stationary
#: Gaussian days, and docs/THRESHOLD_CALIBRATION_NULL.md measures what it
#: gives on series it was not set on.
DOF_PER_RESIDUAL = 0.368

#: The largest normal-equivalent deviation reported. Beyond it the tail
#: probability is below what a double can hold.
MAX_EQUIVALENT_DEVIATION = 37.0


def normal_equivalent(score: float, dof: float) -> float:
    """Return the Gaussian value with the tail probability *score* has under a t.

    A deviation divided by an estimated scale follows a Student t, not a
    Gaussian, and the fewer the days behind the scale the heavier its tails.
    Reading it against a Gaussian threshold then overstates how unusual the
    day is. This maps it to the Gaussian value that is exactly as unusual, so
    that a threshold of three means one day in 370 whatever the sample.
    """
    if dof <= 0:
        raise ValueError("dof must be positive")
    if np.isnan(score):
        raise ValueError("score must be a number")
    if score == 0.0:
        return 0.0
    tail = float(special.stdtr(dof, -abs(score)))
    if tail <= 0.0:
        return float(np.sign(score)) * MAX_EQUIVALENT_DEVIATION
    equivalent = min(-float(special.ndtri(tail)), MAX_EQUIVALENT_DEVIATION)
    return float(np.sign(score)) * equivalent


def medians_without_each(values: Sequence[float]) -> list[float]:
    """Return, for each value, the median of the others.

    The values are sorted once, and the median of the rest is read from the
    sorted list with one place skipped, so a history is not sorted again for
    every day in it. A single value has no others, and its entry is itself.
    """
    if len(values) < 2:
        return [float(value) for value in values]
    order = sorted(range(len(values)), key=values.__getitem__)
    ranked = [values[index] for index in order]
    others = len(ranked) - 1
    low, high = (others - 1) // 2, others // 2
    medians = [0.0] * len(values)
    for rank, index in enumerate(order):
        # Place p of the list without this value is place p of the whole list
        # before the value's own rank, and place p + 1 from there on.
        below = ranked[low if low < rank else low + 1]
        above = ranked[high if high < rank else high + 1]
        medians[index] = (below + above) / 2.0
    return medians


class ChangeKind(str, Enum):
    """How an apparent deviation from baseline should be read."""

    ORDINARY = "ordinary"
    """Within the personal band. Not a finding."""

    TEMPORARY_DISTURBANCE = "temporary_disturbance"
    """Unusual days that have already reverted."""

    PERSISTENT_CHANGE = "persistent_change"
    """A shift that has held long enough to be worth reporting."""

    GRADUAL_DRIFT = "gradual_drift"
    """A slow monotone trend rather than a step."""

    ABRUPT_CHANGE = "abrupt_change"
    """A step change located by change-point detection."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Not enough well-observed days to say anything."""


@dataclass(frozen=True)
class BaselineReference:
    """The personal reference a day is compared against.

    Attributes
    ----------
    centre, scale
        The reference's location and its robust spread.
    samples, weekday_samples
        Well-observed days retained, and those of the day's own weekday.
    weekday_aware
        Whether the centre is that of the day's own weekday.
    scale_samples
        Residuals behind a calibrated scale. Zero when the reference is not
        calibrated: its scale is then the spread of the days behind its
        centre.
    dof
        Degrees of freedom of a calibrated scale, or ``None``.
    """

    centre: float
    scale: float
    samples: int
    weekday_samples: int
    weekday_aware: bool
    scale_samples: int = 0
    dof: float | None = None

    @property
    def calibrated(self) -> bool:
        """Whether the deviation is on the scale the threshold is stated on."""
        return self.dof is not None

    def deviation(self, value: float) -> float:
        """Return the robust z-score of *value* against this reference.

        For a calibrated reference it is the normal-equivalent score: the
        Gaussian value as far into its tail as the day is into the tail of
        the Student t its scale gives.
        """
        if self.scale <= 0.0:
            return (
                0.0
                if value == self.centre
                else float(np.sign(value - self.centre)) * np.inf
            )
        score = (value - self.centre) / self.scale
        return score if self.dof is None else normal_equivalent(score, self.dof)

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the reference."""
        return {
            "centre": self.centre,
            "scale": self.scale,
            "samples": self.samples,
            "weekday_samples": self.weekday_samples,
            "weekday_aware": self.weekday_aware,
            "scale_samples": self.scale_samples,
            "dof": self.dof,
        }


@dataclass(frozen=True)
class BehaviouralChange:
    """A verdict about one feature on one day.

    Attributes
    ----------
    feature
        Name of the tracked quantity.
    day
        Day the verdict is about.
    kind
        How the deviation should be read.
    value
        The day's observed value.
    reference
        The personal reference it was compared against.
    deviation
        Robust z-score of the day against that reference.
    duration_days
        Consecutive well-observed days the deviation has held.
    slope_per_day
        Robust trend estimate over the recent window, in units per day.
    trend_strength
        Total movement across the trend window, in robust standard
        deviations. This is what identifies a gradual drift, and it is
        carried on the verdict so that alerting can grade a drift without
        re-deriving it.
    change_point
        Day a step change was located at, when one was found.
    detail
        Short human-readable explanation.
    """

    feature: str
    day: date
    kind: ChangeKind
    value: float
    reference: BaselineReference
    deviation: float
    duration_days: int
    slope_per_day: float
    trend_strength: float
    change_point: date | None
    detail: str

    @property
    def is_change(self) -> bool:
        """Whether this verdict describes a behavioural change worth acting on."""
        return self.kind in {
            ChangeKind.PERSISTENT_CHANGE,
            ChangeKind.GRADUAL_DRIFT,
            ChangeKind.ABRUPT_CHANGE,
        }

    @property
    def direction(self) -> str:
        """Which way the behaviour moved.

        For a drift this is the sign of the trend, not of the day: a drift is
        identified by its slope, and a single day inside a slow decline can
        easily sit on the other side of the reference. Reading the direction
        off the day would let a verdict announce a decrease while reporting a
        rising slope.
        """
        signal = (
            self.slope_per_day
            if self.kind is ChangeKind.GRADUAL_DRIFT
            else self.deviation
        )
        if signal > 0:
            return "increase"
        return "decrease" if signal < 0 else "none"

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the verdict."""
        return {
            "feature": self.feature,
            "day": self.day.isoformat(),
            "kind": self.kind.value,
            "value": self.value,
            "deviation": self.deviation,
            "direction": self.direction,
            "duration_days": self.duration_days,
            "slope_per_day": self.slope_per_day,
            "trend_strength": self.trend_strength,
            "change_point": (
                self.change_point.isoformat() if self.change_point else None
            ),
            "reference": self.reference.to_dict(),
            "detail": self.detail,
        }


@dataclass
class BaselineConfig:
    """Configuration for the adaptive baseline.

    Parameters
    ----------
    history_days
        Days of well-observed history retained. Bounds memory and defines how
        far back "normal" reaches.
    min_samples
        Well-observed days required before any verdict other than
        ``INSUFFICIENT_DATA`` is issued.
    weekday_min_samples
        Same-weekday days required before the reference becomes weekday-aware.
        Below this it falls back to the pooled reference rather than trusting
        two or three Sundays.
    deviation_threshold
        Robust z-score beyond which a day counts as deviating.
    persistence_days
        Consecutive deviating days after which a disturbance is called a
        persistent change.
    trend_window
        Days examined for a gradual trend.
    trend_threshold
        Robust standard deviations of total movement across the trend window
        that count as drift. Kept well above one: the Theil-Sen slope of a
        stable but noisy series still accumulates more than a standard
        deviation of apparent movement across four weeks, so a lower
        threshold reports noise as decline.
    change_point_penalty
        Penalty passed to PELT when locating a step change.
    min_scale
        Floor on the reference scale, in feature units. Without it a person
        with an extremely regular routine would have every ordinary hour of
        variation reported as an enormous deviation.
    calibrated
        Whether the reference's scale is pooled over every retained day and
        the deviation reported on the scale the threshold is stated on. Off
        by default. The centre is weekday-aware either way. With it on, the
        scale is the spread of each retained day's distance from the centre
        the other days of its weekday would have given it, so it rests on all
        the days and not on the four or five of one weekday, and it already
        holds the uncertainty of a centre drawn from a few days. The
        deviation is then the Gaussian value as unusual as the day is under
        the Student t that scale has, and ``deviation_threshold`` keeps its
        Gaussian meaning. The trend is fitted to the same distances, each
        day's from the centre the others of its weekday give it, so a weekly
        rhythm is not read as a trend, and its movement is measured against
        the same scale.
    """

    history_days: int = 120
    min_samples: int = 14
    weekday_min_samples: int = 4
    deviation_threshold: float = 3.0
    persistence_days: int = 3
    trend_window: int = 28
    trend_threshold: float = 3.5
    change_point_penalty: float = 8.0
    min_scale: float = 0.25
    calibrated: bool = False

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.history_days < 2:
            raise ValueError("history_days must be at least 2")
        if not 1 <= self.min_samples <= self.history_days:
            raise ValueError("min_samples must lie between 1 and history_days")
        if self.weekday_min_samples < 1:
            raise ValueError("weekday_min_samples must be at least 1")
        if self.deviation_threshold <= 0:
            raise ValueError("deviation_threshold must be positive")
        if self.persistence_days < 1:
            raise ValueError("persistence_days must be at least 1")
        if self.trend_window < 3:
            raise ValueError("trend_window must be at least 3")
        if self.trend_threshold <= 0:
            raise ValueError("trend_threshold must be positive")
        if self.change_point_penalty <= 0:
            raise ValueError("change_point_penalty must be positive")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive")


@dataclass
class AdaptiveBaseline:
    """A robust, weekday-aware, non-stationary baseline for one feature.

    Parameters
    ----------
    feature
        Name of the tracked quantity, used in verdicts.
    config
        Thresholds governing the verdicts.
    """

    feature: str
    config: BaselineConfig = field(default_factory=BaselineConfig)
    _days: deque[date] = field(default_factory=deque, repr=False)
    _values: deque[float] = field(default_factory=deque, repr=False)
    _streak: int = field(default=0, repr=False)
    _streak_sign: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Size the bounded history from the configuration."""
        self._days = deque(self._days, maxlen=self.config.history_days)
        self._values = deque(self._values, maxlen=self.config.history_days)

    # ------------------------------------------------------------------
    @property
    def samples(self) -> int:
        """Number of well-observed days currently retained."""
        return len(self._values)

    def reference(self, for_day: date | None = None) -> BaselineReference:
        """Return the personal reference, weekday-aware where possible.

        Comparing a Sunday against other Sundays is what stops the ordinary
        weekly rhythm of a life from being reported as behavioural change.
        The pooled reference is used until enough same-weekday history has
        accumulated to make the weekday split meaningful.
        """
        values = list(self._values)
        if not values:
            return BaselineReference(0.0, self.config.min_scale, 0, 0, False)

        weekday_values: list[float] = []
        if for_day is not None:
            weekday_values = [
                value
                for day, value in zip(self._days, values)
                if day.weekday() == for_day.weekday()
            ]

        weekday_aware = len(weekday_values) >= self.config.weekday_min_samples
        selected = weekday_values if weekday_aware else values

        centre = statistics.median(selected)
        if self.config.calibrated:
            residuals = self._left_out_residuals()
            return BaselineReference(
                centre=centre,
                scale=max(
                    MAD_TO_SIGMA * statistics.median(abs(r) for r in residuals),
                    self.config.min_scale,
                ),
                samples=len(values),
                weekday_samples=len(weekday_values),
                weekday_aware=weekday_aware,
                scale_samples=len(residuals),
                dof=DOF_PER_RESIDUAL * len(residuals),
            )
        deviations = [abs(value - centre) for value in selected]
        scale = max(
            MAD_TO_SIGMA * statistics.median(deviations) if deviations else 0.0,
            self.config.min_scale,
        )
        return BaselineReference(
            centre=centre,
            scale=scale,
            samples=len(values),
            weekday_samples=len(weekday_values),
            weekday_aware=weekday_aware,
        )

    def _left_out_residuals(self) -> list[float]:
        """How far each retained day fell from the centre the others gave it.

        Each day is compared much as a new day would be: against the median
        of the other days of its weekday where that weekday has enough days
        for the reference to be weekday-aware, and against the median of all
        the other days where it has not. A retained day is one of its
        weekday's days, so its centre rests on one day fewer than a new
        day's: on three others where a new day would have four. The spread of
        these residuals is close to the spread a new day's distance from its
        reference has, the uncertainty of a centre drawn from a few days
        included. A single retained day has no other, and its residual is
        zero.
        """
        values = list(self._values)
        if len(values) < 2:
            return [0.0] * len(values)
        groups: dict[int, list[int]] = {}
        for index, day in enumerate(self._days):
            groups.setdefault(day.weekday(), []).append(index)
        needed = max(self.config.weekday_min_samples - 1, 1)
        centres = medians_without_each(values)
        for members in groups.values():
            if len(members) - 1 < needed:
                continue
            own = medians_without_each([values[index] for index in members])
            for index, centre in zip(members, own):
                centres[index] = centre
        return [value - centre for value, centre in zip(values, centres)]

    # ------------------------------------------------------------------
    def _slope(self) -> float:
        """Return a robust trend over the recent window, in units per day.

        Uses the Theil-Sen median of pairwise slopes, which tolerates the
        occasional extraordinary day without letting it set the trend.

        A calibrated baseline fits the trend to how far each day fell from
        the centre the other days of its weekday gave it, and not to the days
        as they are. A weekly rhythm is then no part of the trend, as it is
        no part of the scale the trend's movement is measured against.
        """
        values = self._left_out_residuals() if self.config.calibrated else self._values
        window = list(values)[-self.config.trend_window :]
        days = list(self._days)[-self.config.trend_window :]
        if len(window) < 3:
            return 0.0
        slopes = [
            (window[j] - window[i]) / gap
            for i in range(len(window))
            for j in range(i + 1, len(window))
            if (gap := (days[j] - days[i]).days) > 0
        ]
        return statistics.median(slopes) if slopes else 0.0

    def _change_point(self) -> date | None:
        """Locate the most recent step change in the retained history."""
        if len(self._values) < 2 * self.config.min_samples:
            return None
        detector = PELTChangePointDetector(
            penalty=self.config.change_point_penalty, min_segment_length=3
        )
        located = detector.detect(np.asarray(self._values, dtype=float))
        if not located:
            return None
        return list(self._days)[located[-1]]

    def _update_streak(self, deviating: bool, sign: int) -> None:
        """Track how long a deviation in one direction has held."""
        if deviating and sign == self._streak_sign:
            self._streak += 1
        elif deviating:
            self._streak = 1
            self._streak_sign = sign
        else:
            self._streak = 0
            self._streak_sign = 0

    def observe(self, day: date, value: float) -> BehaviouralChange:
        """Record a well-observed day and return the verdict for it.

        The day is classified *before* it joins the history, so a day is never
        compared against a reference it has already influenced.
        """
        if self._days and day <= self._days[-1]:
            raise ValueError("baseline days must be strictly increasing")
        if not np.isfinite(value):
            raise ValueError("baseline values must be finite")

        reference = self.reference(day)
        deviation = reference.deviation(value)
        deviating = abs(deviation) >= self.config.deviation_threshold
        self._update_streak(deviating, int(np.sign(deviation)) if deviating else 0)

        self._days.append(day)
        self._values.append(value)

        return self._classify(day, value, reference, deviation, deviating)

    def _classify(
        self,
        day: date,
        value: float,
        reference: BaselineReference,
        deviation: float,
        deviating: bool,
    ) -> BehaviouralChange:
        """Decide how a day's deviation should be read."""
        slope = self._slope()
        movement = (
            abs(slope) * self.config.trend_window / reference.scale
            if reference.scale > 0
            else 0.0
        )
        change_point = None
        kind = ChangeKind.ORDINARY
        detail = "within the personal band"
        unit = "normal-equivalent SD" if reference.calibrated else "robust SD"

        if reference.samples < self.config.min_samples:
            kind = ChangeKind.INSUFFICIENT_DATA
            detail = (
                f"only {reference.samples} well-observed days; "
                f"{self.config.min_samples} needed"
            )
        elif deviating and self._streak >= self.config.persistence_days:
            change_point = self._change_point()
            kind = (
                ChangeKind.ABRUPT_CHANGE
                if change_point is not None
                else ChangeKind.PERSISTENT_CHANGE
            )
            detail = (
                f"deviation of {deviation:+.1f} {unit} held for " f"{self._streak} days"
            )
        elif deviating:
            kind = ChangeKind.TEMPORARY_DISTURBANCE
            detail = (
                f"deviation of {deviation:+.1f} {unit} on "
                f"{self._streak} day(s), not yet persistent"
            )
        else:
            if len(self._values) >= self.config.trend_window and (
                movement >= self.config.trend_threshold
            ):
                kind = ChangeKind.GRADUAL_DRIFT
                detail = (
                    f"trend of {slope:+.3f} per day over "
                    f"{self.config.trend_window} days "
                    f"({movement:.1f} robust SD of movement)"
                )

        return BehaviouralChange(
            feature=self.feature,
            day=day,
            kind=kind,
            value=value,
            reference=reference,
            deviation=deviation,
            duration_days=self._streak,
            slope_per_day=slope,
            trend_strength=movement,
            change_point=change_point,
            detail=detail,
        )

    def skip(
        self, day: date, reason: str = "insufficient sensor coverage"
    ) -> BehaviouralChange:
        """Record that a day was not observed well enough to be used.

        The day is deliberately kept out of the history. Feeding a poorly
        observed day in as a low value would let a sensor outage rewrite the
        resident's definition of normal.
        """
        logger.debug("Skipping %s for feature '%s': %s", day, self.feature, reason)
        reference = self.reference(day)
        return BehaviouralChange(
            feature=self.feature,
            day=day,
            kind=ChangeKind.INSUFFICIENT_DATA,
            value=float("nan"),
            reference=reference,
            deviation=0.0,
            duration_days=self._streak,
            slope_per_day=0.0,
            trend_strength=0.0,
            change_point=None,
            detail=reason,
        )

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, object]:
        """Return restartable baseline state."""
        return {
            "feature": self.feature,
            "days": [day.isoformat() for day in self._days],
            "values": list(self._values),
            "streak": self._streak,
            "streak_sign": self._streak_sign,
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        """Restore baseline state produced by :meth:`snapshot`.

        The payload is validated rather than trusted: a snapshot round-trips
        through JSON on the way to and from an edge device, so it arrives as
        untyped data.
        """
        raw_days = state.get("days") or []
        raw_values = state.get("values") or []
        if not isinstance(raw_days, Sequence) or not isinstance(raw_values, Sequence):
            raise TypeError("snapshot 'days' and 'values' must be sequences")
        if len(raw_days) != len(raw_values):
            raise ValueError("snapshot days and values must be the same length")

        self._days = deque(
            (date.fromisoformat(str(day)) for day in raw_days),
            maxlen=self.config.history_days,
        )
        self._values = deque(
            (float(value) for value in raw_values), maxlen=self.config.history_days
        )
        self._streak = int(state.get("streak", 0))
        self._streak_sign = int(state.get("streak_sign", 0))
