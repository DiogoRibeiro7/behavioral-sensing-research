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

from ..models.change_point_detection.pelt import PELTChangePointDetector

logger = logging.getLogger(__name__)

#: Scale factor making the median absolute deviation a consistent estimator
#: of the standard deviation for normally distributed data.
MAD_TO_SIGMA = 1.4826


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
    """The personal reference a day is compared against."""

    centre: float
    scale: float
    samples: int
    weekday_samples: int
    weekday_aware: bool

    def deviation(self, value: float) -> float:
        """Return the robust z-score of *value* against this reference."""
        if self.scale <= 0.0:
            return (
                0.0
                if value == self.centre
                else float(np.sign(value - self.centre)) * np.inf
            )
        return (value - self.centre) / self.scale

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the reference."""
        return {
            "centre": self.centre,
            "scale": self.scale,
            "samples": self.samples,
            "weekday_samples": self.weekday_samples,
            "weekday_aware": self.weekday_aware,
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
        """Whether the day sits above or below the personal reference."""
        if self.deviation > 0:
            return "increase"
        return "decrease" if self.deviation < 0 else "none"

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
        Robust z-scores of total movement across the trend window that count
        as drift.
    change_point_penalty
        Penalty passed to PELT when locating a step change.
    min_scale
        Floor on the reference scale, in feature units. Without it a person
        with an extremely regular routine would have every ordinary hour of
        variation reported as an enormous deviation.
    """

    history_days: int = 120
    min_samples: int = 14
    weekday_min_samples: int = 4
    deviation_threshold: float = 3.0
    persistence_days: int = 3
    trend_window: int = 28
    trend_threshold: float = 2.0
    change_point_penalty: float = 8.0
    min_scale: float = 0.25

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

    # ------------------------------------------------------------------
    def _slope(self) -> float:
        """Return a robust trend over the recent window, in units per day.

        Uses the Theil-Sen median of pairwise slopes, which tolerates the
        occasional extraordinary day without letting it set the trend.
        """
        window = list(self._values)[-self.config.trend_window :]
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
        change_point = None
        kind = ChangeKind.ORDINARY
        detail = "within the personal band"

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
                f"deviation of {deviation:+.1f} robust SD held for "
                f"{self._streak} days"
            )
        elif deviating:
            kind = ChangeKind.TEMPORARY_DISTURBANCE
            detail = (
                f"deviation of {deviation:+.1f} robust SD on "
                f"{self._streak} day(s), not yet persistent"
            )
        else:
            movement = abs(slope) * self.config.trend_window / reference.scale
            if len(self._values) >= self.config.trend_window and (
                movement >= self.config.trend_threshold
            ):
                kind = ChangeKind.GRADUAL_DRIFT
                detail = (
                    f"trend of {slope:+.3f} per day over "
                    f"{self.config.trend_window} days"
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
