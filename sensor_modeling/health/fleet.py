"""Silence across a fleet of homes.

A home's own health monitor can say that the home has stopped reporting. It
cannot say why, and one of the possible reasons is not in the home at all:
the service that collects every home's records has failed. That is visible
only from above. When most of the homes being monitored are silent at the
same moment, the likeliest explanation is what they share, not that their
residents all stopped moving together.

This is the canary argument of :mod:`.monitor`, one level up. There, the
sensors that report on a cadence vouch for a home's delivery path. Here, the
homes vouch for the service's.

Two things are kept deliberately modest. The check uses the same horizon a
home uses for its own silence, so it classifies silences and does not race to
detect them. And it reports a common cause without naming one: a shared
network, a collection server and a public holiday on which every resident is
away would all look the same.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class FleetConfig:
    """When simultaneous silence is read as a common cause.

    Parameters
    ----------
    horizon
        How long a home must have been silent to count as silent. Use the
        horizon the homes' own monitors use.
    fraction
        Share of the monitored homes that must be silent together.
    min_homes
        Fewest silent homes that can be called a common cause. One silent
        home is more likely a silent home than a dead service.
    """

    horizon: timedelta
    fraction: float = 0.6
    min_homes: int = 3

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.horizon <= timedelta(0):
            raise ValueError("horizon must be positive")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("fraction must lie in (0, 1]")
        if self.min_homes < 2:
            raise ValueError("min_homes must be at least 2")

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the configuration."""
        return {
            "horizon_hours": self.horizon.total_seconds() / 3600.0,
            "fraction": self.fraction,
            "min_homes": self.min_homes,
        }


@dataclass(frozen=True)
class FleetSilence:
    """What the fleet's silence looks like at one moment.

    Attributes
    ----------
    at
        When the assessment was made.
    monitored
        Homes that have reported at least once, sorted.
    silent
        Those among them silent for at least the horizon, sorted.
    common_cause
        Whether enough homes are silent together to suspect what they share.
    since
        When the common silence can have begun: the last observation of the
        silent home heard from most recently. ``None`` without a common cause.
    """

    at: datetime
    monitored: tuple[str, ...]
    silent: tuple[str, ...]
    common_cause: bool
    since: datetime | None

    @property
    def share(self) -> float:
        """Share of the monitored homes that are silent."""
        return len(self.silent) / len(self.monitored) if self.monitored else 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the assessment."""
        return {
            "at": self.at.isoformat(),
            "monitored": len(self.monitored),
            "silent": list(self.silent),
            "share": self.share,
            "common_cause": self.common_cause,
            "since": self.since.isoformat() if self.since else None,
        }


def assess_fleet(
    last_seen: Mapping[str, datetime | None], at: datetime, config: FleetConfig
) -> FleetSilence:
    """Assess whether the fleet's silent homes share a cause, as of *at*.

    Parameters
    ----------
    last_seen
        Each monitored home's most recent observation. A home that has never
        reported is left out: there is nothing yet for it to have stopped.
    at
        The moment of the assessment.
    config
        When simultaneous silence is read as a common cause.
    """
    heard = {home: seen for home, seen in last_seen.items() if seen is not None}
    monitored = tuple(sorted(heard))
    silent = tuple(home for home in monitored if at - heard[home] >= config.horizon)
    common = len(silent) >= config.min_homes and len(silent) >= config.fraction * len(
        monitored
    )
    # The common silence can only have begun after the last silent home
    # was still speaking.
    since = max(heard[home] for home in silent) if common else None
    return FleetSilence(at, monitored, silent, common, since)


@dataclass(frozen=True)
class CommonSilence:
    """One stretch during which the fleet's silence had a common cause.

    Attributes
    ----------
    since
        When the common silence can have begun.
    detected
        The first assessment that called it a common cause.
    until
        The last assessment that still did.
    homes
        Most homes silent together at any assessment in the stretch.
    monitored
        Homes being monitored at that assessment.
    """

    since: datetime
    detected: datetime
    until: datetime
    homes: int
    monitored: int

    def covers(self, moment: datetime) -> bool:
        """Whether *moment* falls between the onset and the last assessment."""
        return self.since <= moment <= self.until

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the stretch."""
        return {
            "since": self.since.isoformat(),
            "detected": self.detected.isoformat(),
            "until": self.until.isoformat(),
            "homes": self.homes,
            "monitored": self.monitored,
        }


def common_silences(
    observed: Mapping[str, Sequence[datetime]],
    config: FleetConfig,
    *,
    every: timedelta = timedelta(hours=1),
) -> list[CommonSilence]:
    """Find the stretches of common silence in a recorded fleet.

    This replays :func:`assess_fleet` over a record that is already complete.
    A home counts as monitored between its first and its last observation, so
    homes that have not joined yet, or have left, are not mistaken for silent
    ones.

    Parameters
    ----------
    observed
        Each home's observation times, in non-decreasing order.
    config
        When simultaneous silence is read as a common cause.
    every
        Spacing of the assessments.
    """
    if every <= timedelta(0):
        raise ValueError("every must be positive")
    records = {home: list(times) for home, times in observed.items() if len(times)}
    for home, times in records.items():
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            raise ValueError(f"observations of '{home}' are not in time order")
    if not records:
        return []

    start = min(times[0] for times in records.values())
    end = max(times[-1] for times in records.values())
    found: list[CommonSilence] = []
    current: CommonSilence | None = None
    moment = start
    while moment <= end:
        last_seen: dict[str, datetime | None] = {}
        for home, times in records.items():
            if not times[0] <= moment <= times[-1]:
                continue
            last_seen[home] = times[bisect.bisect_right(times, moment) - 1]
        verdict = assess_fleet(last_seen, moment, config)
        if verdict.common_cause and verdict.since is not None:
            if current is None:
                current = CommonSilence(
                    since=verdict.since,
                    detected=moment,
                    until=moment,
                    homes=len(verdict.silent),
                    monitored=len(verdict.monitored),
                )
            else:
                larger = len(verdict.silent) > current.homes
                current = CommonSilence(
                    since=current.since,
                    detected=current.detected,
                    until=moment,
                    homes=len(verdict.silent) if larger else current.homes,
                    monitored=(len(verdict.monitored) if larger else current.monitored),
                )
        elif current is not None:
            found.append(current)
            current = None
        moment += every
    if current is not None:
        found.append(current)
    return found
