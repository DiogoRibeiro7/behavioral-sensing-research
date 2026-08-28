"""Daily behavioural features derived from the state posterior.

A baseline needs a small number of stable, interpretable quantities per day:
how long the resident appeared to be asleep, how often they were active in the
kitchen, how much the day was spent in states the model could not name.

Two decisions here matter for the honesty of everything downstream.

*Aggregate the posterior, not the argmax.* Time in a state is accumulated as
``sum P(state | t) * dt`` rather than by counting the intervals where a state
happened to win. A day made of 60%-confident guesses and a day made of
99%-confident conclusions are genuinely different, and collapsing to argmax
first would erase that difference before the baseline ever sees it.

*Record how much of the day was actually observed.* Every summary carries the
fraction of the day covered by estimates and the mean sensor coverage behind
them. Days where the apparatus was not working are identifiable as such, so
the baseline can refuse them instead of recording a broken sensor as a quiet
day.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo

from ..fusion.estimate import StateEstimate
from ..states.ontology import BehaviouralState

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class DailySummary:
    """Aggregated behaviour and data quality for one local calendar day.

    Attributes
    ----------
    day
        The local calendar date summarised.
    hours
        Expected hours spent in each latent state, from the posterior.
    transitions
        Expected number of state changes across the day.
    coverage
        Mean sensor coverage behind the day's estimates, in ``[0, 1]``.
    abstention
        Fraction of the observed day for which the model declined to name a
        state.
    observed
        Fraction of the day's 24 hours covered by estimates at all.
    """

    day: date
    hours: dict[BehaviouralState, float]
    transitions: float
    coverage: float
    abstention: float
    observed: float

    def hours_in(self, state: BehaviouralState) -> float:
        """Return expected hours spent in *state*."""
        return self.hours.get(state, 0.0)

    def is_usable(self, min_coverage: float = 0.5, min_observed: float = 0.6) -> bool:
        """Whether the day was observed well enough to inform a baseline.

        A day that fails this test is not a quiet day. It is a day the
        apparatus did not watch, and it must be excluded rather than
        recorded as low activity.
        """
        return self.coverage >= min_coverage and self.observed >= min_observed

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the summary."""
        return {
            "day": self.day.isoformat(),
            "hours": {state.value: value for state, value in self.hours.items()},
            "transitions": self.transitions,
            "coverage": self.coverage,
            "abstention": self.abstention,
            "observed": self.observed,
        }


def _local_day(moment: datetime, zone: tzinfo | None) -> date:
    """Return the local calendar date of *moment*."""
    return (moment.astimezone(zone) if zone is not None else moment).date()


def _split_by_day(
    start: datetime, end: datetime, zone: tzinfo | None
) -> list[tuple[date, float]]:
    """Split an interval into per-local-day durations in seconds.

    Days are bounded by local midnight, so a day containing a DST transition
    is correctly 23 or 25 hours long rather than assumed to be 24.
    """
    if end <= start:
        return []
    pieces: list[tuple[date, float]] = []
    cursor = start
    while cursor < end:
        local = cursor.astimezone(zone) if zone is not None else cursor
        next_midnight_local = datetime.combine(
            local.date() + timedelta(days=1), datetime.min.time(), tzinfo=local.tzinfo
        )
        boundary = min(next_midnight_local.astimezone(cursor.tzinfo), end)
        if boundary <= cursor:  # pragma: no cover - defensive against odd zones
            boundary = end
        pieces.append((local.date(), (boundary - cursor).total_seconds()))
        cursor = boundary
    return pieces


def summarise_days(
    estimates: Sequence[StateEstimate],
    *,
    tz: tzinfo | None = None,
    max_interval: timedelta = timedelta(hours=1),
) -> list[DailySummary]:
    """Aggregate a run of state estimates into per-day behavioural summaries.

    Each estimate is taken to describe the interval since the previous one,
    which is the standard filtering approximation: the posterior at ``t``
    conditions on everything up to ``t``.

    Parameters
    ----------
    estimates
        State estimates in non-decreasing time order.
    tz
        Timezone whose calendar days are used. Defaults to the timezone of
        the first estimate, so days align with the resident's local clock.
    max_interval
        Longest interval a single estimate may be held to describe. Gaps
        beyond this are left uncounted rather than being attributed to
        whatever state was current before the outage.

    Returns
    -------
    list[DailySummary]
        One summary per local day that has any observed time, in order.
    """
    if not estimates:
        return []
    ordered = list(estimates)
    for previous, current in zip(ordered, ordered[1:]):
        if current.at < previous.at:
            raise ValueError("estimates must be in non-decreasing time order")

    zone = tz if tz is not None else ordered[0].at.tzinfo
    states = ordered[0].ontology.states

    seconds: dict[date, dict[BehaviouralState, float]] = {}
    observed: dict[date, float] = {}
    coverage: dict[date, float] = {}
    abstained: dict[date, float] = {}
    transitions: dict[date, float] = {}

    for previous, current in zip(ordered, ordered[1:]):
        span = current.at - previous.at
        if span <= timedelta(0) or span > max_interval:
            continue
        for day, duration in _split_by_day(previous.at, current.at, zone):
            bucket = seconds.setdefault(day, dict.fromkeys(states, 0.0))
            for index, state in enumerate(states):
                bucket[state] += float(current.belief[index]) * duration
            observed[day] = observed.get(day, 0.0) + duration
            coverage[day] = coverage.get(day, 0.0) + current.completeness * duration
            if current.abstained:
                abstained[day] = abstained.get(day, 0.0) + duration

        # Expected number of state changes over the interval, from the
        # probability that the two posteriors disagree.
        change = 1.0 - float((previous.belief * current.belief).sum())
        day = _local_day(current.at, zone)
        transitions[day] = transitions.get(day, 0.0) + change

    summaries = []
    for day in sorted(observed):
        total = observed[day]
        summaries.append(
            DailySummary(
                day=day,
                hours={state: value / 3600.0 for state, value in seconds[day].items()},
                transitions=transitions.get(day, 0.0),
                coverage=coverage[day] / total if total > 0 else 0.0,
                abstention=abstained.get(day, 0.0) / total if total > 0 else 0.0,
                observed=total / SECONDS_PER_DAY,
            )
        )
    return summaries


def feature_series(
    summaries: Iterable[DailySummary],
    state: BehaviouralState,
    *,
    min_coverage: float = 0.5,
    min_observed: float = 0.6,
) -> tuple[list[date], list[float]]:
    """Extract a per-day series of hours in *state*, skipping unusable days.

    Days the apparatus did not observe are dropped rather than recorded as
    zeros. A missing day is missing evidence; treating it as an observation
    of no activity is the single most damaging mistake this codebase exists
    to avoid.
    """
    days: list[date] = []
    values: list[float] = []
    for summary in summaries:
        if not summary.is_usable(min_coverage, min_observed):
            logger.debug(
                "Excluding poorly observed day %s from the series", summary.day
            )
            continue
        days.append(summary.day)
        values.append(summary.hours_in(state))
    return days, values
