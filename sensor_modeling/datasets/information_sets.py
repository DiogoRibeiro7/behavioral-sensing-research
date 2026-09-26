"""Matched information sets for comparing behavioural-state models.

Phase 1 of the roadmap asks how much of the real-data performance gap is caused
by what the sensors can support and how much by the current probabilistic
formulation. That question has no answer unless every candidate model, the
generative filter and the supervised diagnostics alike, is compared on an
explicitly declared information set. A model that sees more has not
demonstrated a modelling advantage merely by scoring higher.

This module declares those sets and builds the matching feature tables from a
real recording. It fits nothing and knows nothing about any model: it turns a
recording and a sequence of prediction moments into numbers, and states exactly
which information each number carries.

The contract
------------
For a prediction at moment ``t`` in one household:

- **Current evidence** is the number of activations on each evidence channel in
  the step window ``(t - step, t]``. That is the window the online pipeline
  batches into its estimate at ``t``, and an activation is counted the way the
  Poisson emission model counts one: an event observation with a non-zero
  value.
- **Time of day** is the local wall-clock hour of ``t``, the quantity the
  optional circadian prior of :class:`~sensor_modeling.states.StateOntology`
  conditions on. Its definition, daylight-saving behaviour and encodings are
  in :mod:`~sensor_modeling.datasets.time_features`.
- **Recent history** is the same per-channel count in each of the
  ``history_steps`` preceding windows ``(t - (j + 1) step, t - j step]``.
- **History summaries** condense those step windows over longer lookbacks into
  a few interpretable columns: activation counts and room changes over each
  summary window, and, over the longest window (the horizon), how long each
  channel has been quiet and which rooms were active most recently. Each one is
  a function of the per-channel counts in whole step windows, the evidence the
  filter receives, so none of them reveals timing or ordering inside a step.

Every window closes at or before ``t``, so nothing observed after the prediction
moment can reach a feature. Nothing here reads the annotations, and each
household's table is built from that household's recording alone.

Absence is not zero
-------------------
Two situations produce no activations without being silence:

- A declared channel with no event sensor in this household is
  **uninstrumented**. Its columns are NaN, because nothing was there to fire.
- A history window that closes before the recording's first observation was
  never observed. Its columns are NaN. A summary is NaN while its lookback
  reaches such a window, unless an activation inside the recording already
  settles its value.

A zero from an instrumented channel inside the recording is evidence and is
reported as zero. Windows after the last observation are zero too: at the
prediction moment nobody can know that the recording is about to end, and
marking them missing would use that future knowledge.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, tzinfo
from enum import Enum
from itertools import pairwise

import numpy as np

from ..observations import Modality, ObservationKind, SensorRegistry, require_aware
from .casas import CasasRecording
from .casas_hh import HH_LOCATIONS
from .time_features import local_hour


class InformationComponent(str, Enum):
    """One kind of information a model may be given about a prediction moment."""

    CURRENT_EVIDENCE = "current_evidence"
    """Per-channel activation counts in the window ending at the moment."""

    TIME_OF_DAY = "time_of_day"
    """Local wall-clock hour of the moment."""

    RECENT_HISTORY = "recent_history"
    """Per-channel activation counts in the preceding windows."""

    HISTORY_SUMMARY = "history_summary"
    """Interpretable summaries of the step windows over longer lookbacks."""


@dataclass(frozen=True, order=True)
class EvidenceChannel:
    """One room-resolved stream of activations: a room and a sensing modality.

    Channels are keyed by modality as well as room because the filter's default
    observation model does not treat them as exchangeable: an entrance door has
    its own rates, distinct from motion in the same hall. Pooling the two would
    give the filter information its comparators do not get.
    """

    room: str
    modality: Modality

    def __post_init__(self) -> None:
        """Validate the room and normalise the modality."""
        if not isinstance(self.room, str) or not self.room.strip():
            raise ValueError("an evidence channel needs a non-empty room")
        object.__setattr__(self, "modality", Modality(self.modality))

    @property
    def name(self) -> str:
        """Column-name stem, ``<room>_<modality>``."""
        return f"{self.room}_{self.modality.value}"


#: Every event channel the CASAS ``hh`` adapter can produce, from its location map.
#:
#: Declaring the channels once, rather than reading them off each recording,
#: gives every household the same columns, so a model trained on some homes can
#: be scored on others. A household lacking one of them reports it as
#: uninstrumented instead of silently gaining or losing a column.
HH_EVIDENCE_CHANNELS: tuple[EvidenceChannel, ...] = tuple(
    sorted(
        {
            EvidenceChannel(room, modality)
            for room, modality in HH_LOCATIONS.values()
            if modality is not Modality.ENVIRONMENTAL
        }
    )
)

#: Column holding the local hour of the prediction moment, 0 to 23.
HOUR_OF_DAY = "hour_of_day"

_EVIDENCE_COLUMN = re.compile(r"^events_(?P<channel>.+)_lag(?P<lag>\d+)$")


def evidence_column(channel: str, lag: int) -> str:
    """Name of the column counting *channel* activations *lag* windows back."""
    return f"events_{channel}_lag{lag}"


def parse_evidence_column(name: str) -> tuple[str, int] | None:
    """Return ``(channel, lag)`` for an evidence column, ``None`` for any other."""
    match = _EVIDENCE_COLUMN.match(name)
    if match is None:
        return None
    return match.group("channel"), int(match.group("lag"))


#: Activations on one channel over a summary window.
SUMMARY_COUNT = "count"
#: Changes of active room between consecutive active steps over a summary window.
SUMMARY_ROOM_CHANGES = "room_changes"
#: Minutes of whole quiet steps since one channel's last activation, within the horizon.
SUMMARY_QUIET_MINUTES = "quiet_minutes"
#: Whether one room was active in the most recent active step within the horizon.
SUMMARY_LAST_ROOM = "last_room"

#: Summary features, in the order their columns are laid out.
SUMMARY_FEATURES = (
    SUMMARY_COUNT,
    SUMMARY_ROOM_CHANGES,
    SUMMARY_QUIET_MINUTES,
    SUMMARY_LAST_ROOM,
)

_SUMMARY_COLUMN = re.compile(
    r"^history_(?P<feature>count|room_changes|quiet_minutes|last_room)"
    r"(?:_(?P<subject>.+?))?_(?P<minutes>\d+)m$"
)
_MINUTE = timedelta(minutes=1)


def summary_column(feature: str, window: timedelta, subject: str | None = None) -> str:
    """Name of a history-summary column.

    *subject* is a channel name for counts and quiet minutes, a room for the
    last-room indicator, and ``None`` for room changes. The name ends in the
    lookback in whole minutes, so ``history_count_kitchen_motion_60m`` counts
    kitchen motion over the last hour.
    """
    if feature not in SUMMARY_FEATURES:
        raise ValueError(f"summary feature must be one of {SUMMARY_FEATURES}")
    if (subject is None) != (feature == SUMMARY_ROOM_CHANGES):
        needs = "no subject" if feature == SUMMARY_ROOM_CHANGES else "a subject"
        raise ValueError(f"summary feature {feature!r} takes {needs}")
    stem = feature if subject is None else f"{feature}_{subject}"
    return f"history_{stem}_{window // _MINUTE}m"


def parse_summary_column(name: str) -> tuple[str, str | None, int] | None:
    """Return ``(feature, subject, minutes)`` for a summary column, else ``None``."""
    match = _SUMMARY_COLUMN.match(name)
    if match is None:
        return None
    feature, subject = match.group("feature"), match.group("subject")
    if (subject is None) != (feature == SUMMARY_ROOM_CHANGES):
        return None
    return feature, subject, int(match.group("minutes"))


@dataclass(frozen=True)
class EvidenceResolution:
    """How finely evidence is resolved, shared by every set in one comparison.

    Information sets are only comparable when they agree on this. Two sets that
    differ in step width or channel vocabulary differ in more than the
    components they declare, and :meth:`InformationSet.is_nested_in` refuses to
    call them nested.

    Attributes
    ----------
    step
        Width of one evidence window. When the filter is among the models
        compared, this must equal its pipeline step so both see the same
        windows. Five minutes is the step behind every published real-data
        figure in ``docs/real_data.md``.
    channels
        Declared evidence channels, identical for every household. Stored
        sorted, so the declaration order does not change the columns.
    history_steps
        Number of preceding windows that ``RECENT_HISTORY`` adds. Three is what
        the documented supervised diagnostic used.
    summary_windows
        Lookbacks of ``HISTORY_SUMMARY``, in whole minutes, stored in
        increasing order. The longest is the horizon. A set that includes the
        summaries needs each window to be a whole number of steps. One hour
        spans the median away and inactive bouts, the states where the filter
        falls furthest below the diagnostic ceiling. Three hours spans the
        median sleep bout. ``docs/INFORMATION_SETS.md`` gives the measurements.
    """

    step: timedelta = timedelta(minutes=5)
    channels: tuple[EvidenceChannel, ...] = HH_EVIDENCE_CHANNELS
    history_steps: int = 3
    summary_windows: tuple[timedelta, ...] = (
        timedelta(minutes=60),
        timedelta(minutes=180),
    )

    def __post_init__(self) -> None:
        """Validate the resolution and store channels in canonical order."""
        if not isinstance(self.step, timedelta) or self.step <= timedelta(0):
            raise ValueError("step must be a positive timedelta")
        channels = tuple(self.channels)
        if not channels:
            raise ValueError("at least one evidence channel is required")
        if len(set(channels)) != len(channels):
            raise ValueError("evidence channels must be unique")
        names = [channel.name for channel in channels]
        if len(set(names)) != len(names):
            raise ValueError("evidence channel names must be unique")
        if (
            isinstance(self.history_steps, bool)
            or not isinstance(self.history_steps, int)
            or self.history_steps < 0
        ):
            raise ValueError("history_steps must be a non-negative integer")
        windows = tuple(self.summary_windows)
        if not windows:
            raise ValueError("at least one summary window is required")
        for window in windows:
            if (
                not isinstance(window, timedelta)
                or window <= timedelta(0)
                or window % _MINUTE
            ):
                raise ValueError(
                    "summary windows must be positive whole numbers of minutes"
                )
        if len(set(windows)) != len(windows):
            raise ValueError("summary windows must be unique")
        object.__setattr__(self, "channels", tuple(sorted(channels)))
        object.__setattr__(self, "summary_windows", tuple(sorted(windows)))

    @property
    def rooms(self) -> tuple[str, ...]:
        """Rooms of the declared channels, sorted and without repeats."""
        return tuple(sorted({channel.room for channel in self.channels}))

    @property
    def horizon(self) -> timedelta:
        """The longest summary window."""
        return self.summary_windows[-1]


@dataclass(frozen=True)
class _Column:
    """Where one feature column comes from; ``channel=None`` is the hour."""

    channel: EvidenceChannel | None
    lag: int = 0

    @property
    def name(self) -> str:
        """Public column name."""
        if self.channel is None:
            return HOUR_OF_DAY
        return evidence_column(self.channel.name, self.lag)


@dataclass(frozen=True)
class _Summary:
    """One history-summary column: *feature* of *subject* over *window*."""

    feature: str
    window: timedelta
    subject: str | None = None

    @property
    def name(self) -> str:
        """Public column name."""
        return summary_column(self.feature, self.window, self.subject)


@dataclass(frozen=True)
class InformationSet:
    """A declared, reproducible set of information a model may condition on.

    Attributes
    ----------
    name
        Label used in reports.
    components
        Which kinds of information are included.
    resolution
        Step width, channel vocabulary and history depth.

    Columns are laid out as current evidence, then hour of day, then history
    from the most recent window backwards, then history summaries, with
    channels in canonical order inside each block. A set nested in another
    therefore has its columns as an ordered subsequence of the larger set's,
    with identical values.
    """

    name: str
    components: frozenset[InformationComponent]
    resolution: EvidenceResolution = field(default_factory=EvidenceResolution)

    def __post_init__(self) -> None:
        """Validate the declaration."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("an information set needs a name")
        components = frozenset(
            InformationComponent(component) for component in self.components
        )
        if not components:
            raise ValueError(f"information set {self.name!r} has no components")
        if (
            InformationComponent.RECENT_HISTORY in components
            and self.resolution.history_steps < 1
        ):
            raise ValueError(
                f"information set {self.name!r} includes recent history but "
                "the resolution declares no history steps"
            )
        if InformationComponent.HISTORY_SUMMARY in components and any(
            window % self.resolution.step for window in self.resolution.summary_windows
        ):
            raise ValueError(
                f"information set {self.name!r} includes history summaries but "
                "its summary windows are not whole numbers of steps"
            )
        object.__setattr__(self, "components", components)

    def _layout(self) -> tuple[_Column | _Summary, ...]:
        """Return the column sources in their canonical order."""
        resolution = self.resolution
        channels = resolution.channels
        layout: list[_Column | _Summary] = []
        if InformationComponent.CURRENT_EVIDENCE in self.components:
            layout.extend(_Column(channel, 0) for channel in channels)
        if InformationComponent.TIME_OF_DAY in self.components:
            layout.append(_Column(None))
        if InformationComponent.RECENT_HISTORY in self.components:
            for lag in range(1, resolution.history_steps + 1):
                layout.extend(_Column(channel, lag) for channel in channels)
        if InformationComponent.HISTORY_SUMMARY in self.components:
            for window in resolution.summary_windows:
                layout.extend(
                    _Summary(SUMMARY_COUNT, window, channel.name)
                    for channel in channels
                )
                layout.append(_Summary(SUMMARY_ROOM_CHANGES, window))
            horizon = resolution.horizon
            layout.extend(
                _Summary(SUMMARY_QUIET_MINUTES, horizon, channel.name)
                for channel in channels
            )
            layout.extend(
                _Summary(SUMMARY_LAST_ROOM, horizon, room) for room in resolution.rooms
            )
        return tuple(layout)

    @property
    def columns(self) -> tuple[str, ...]:
        """Feature column names, in canonical order."""
        return tuple(column.name for column in self._layout())

    def is_nested_in(self, other: InformationSet) -> bool:
        """Whether *other* carries everything this set does, at the same resolution."""
        return (
            self.resolution == other.resolution and self.components <= other.components
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serialisable declaration.

        The summary windows appear only in a set that includes the summaries,
        so declarations, and digests, of sets without them are unchanged.
        """
        resolution = self.resolution
        declaration: dict[str, object] = {
            "name": self.name,
            "components": sorted(component.value for component in self.components),
            "step_seconds": resolution.step.total_seconds(),
            "channels": [
                {"room": channel.room, "modality": channel.modality.value}
                for channel in resolution.channels
            ],
            "history_steps": resolution.history_steps,
            "columns": list(self.columns),
        }
        if InformationComponent.HISTORY_SUMMARY in self.components:
            declaration["summary_windows_seconds"] = [
                window.total_seconds() for window in resolution.summary_windows
            ]
        return declaration

    def sha256(self) -> str:
        """Return the SHA-256 of the canonical serialised declaration."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def nested_information_sets(
    resolution: EvidenceResolution | None = None,
) -> tuple[InformationSet, ...]:
    """Return the four Phase 1 information sets, sharing one resolution.

    In order: current evidence; plus time of day; plus recent history; plus
    both. The second and third are not nested in each other, and both lie
    between the first and the fourth.
    """
    shared = resolution if resolution is not None else EvidenceResolution()
    current = InformationComponent.CURRENT_EVIDENCE
    hour = InformationComponent.TIME_OF_DAY
    history = InformationComponent.RECENT_HISTORY
    return (
        InformationSet("current", frozenset({current}), shared),
        InformationSet("current+time_of_day", frozenset({current, hour}), shared),
        InformationSet("current+history", frozenset({current, history}), shared),
        InformationSet(
            "current+time_of_day+history",
            frozenset({current, hour, history}),
            shared,
        ),
    )


@dataclass(frozen=True, eq=False)
class FeatureTable:
    """Features for one household under one information set.

    Attributes
    ----------
    household
        Identifier of the household the rows describe.
    information_set
        The declaration the table was built from.
    moments
        Prediction moments, one per row, in the recording's timezone.
    values
        Read-only ``(len(moments), len(columns))`` array. NaN marks information
        that does not exist, as described in the module documentation, never
        a count of zero.
    uninstrumented
        Declared channels with no event sensor in this household.
    excluded_sensors
        Registered sensors whose readings reach no column: sensors that are not
        event-kind, have no room, or sit on an undeclared channel.
    """

    household: str
    information_set: InformationSet
    moments: tuple[datetime, ...]
    values: np.ndarray
    uninstrumented: tuple[EvidenceChannel, ...] = ()
    excluded_sensors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Check the array matches the declaration, and freeze a copy of it."""
        values = np.array(self.values, dtype=np.float64)
        expected = (len(self.moments), len(self.information_set.columns))
        if values.shape != expected:
            raise ValueError(f"values have shape {values.shape}, expected {expected}")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @property
    def columns(self) -> tuple[str, ...]:
        """Feature column names, in the order of :attr:`values`."""
        return self.information_set.columns

    def column(self, name: str) -> np.ndarray:
        """Return one column by name."""
        return self.values[:, self.columns.index(name)]


def _route_sensors(
    registry: SensorRegistry, channels: Iterable[EvidenceChannel]
) -> tuple[dict[str, EvidenceChannel], tuple[str, ...]]:
    """Assign each registered sensor to its declared channel, or exclude it."""
    declared = set(channels)
    routed: dict[str, EvidenceChannel] = {}
    excluded: list[str] = []
    for spec in registry:
        channel = (
            EvidenceChannel(spec.room, spec.modality)
            if spec.kind is ObservationKind.EVENT and spec.room
            else None
        )
        if channel is not None and channel in declared:
            routed[spec.sensor_id] = channel
        else:
            excluded.append(spec.sensor_id)
    return routed, tuple(sorted(excluded))


def _local_moments(moments: Sequence[datetime], zone: tzinfo) -> tuple[datetime, ...]:
    """Validate prediction moments and express them in the recording's zone."""
    aware = [require_aware(moment, "moment") for moment in moments]
    for earlier, later in pairwise(aware):
        if later <= earlier:
            raise ValueError(
                "prediction moments must be strictly increasing; a repeated "
                "moment would weight one estimate twice (see scoring_steps)"
            )
    return tuple(moment.astimezone(zone) for moment in aware)


_EPOCH = datetime(1970, 1, 1)
_MICROSECOND = timedelta(microseconds=1)


def _wall_clock(moment: datetime) -> int:
    """Microseconds on the local clock, the order in which windows compare times."""
    return (moment.replace(tzinfo=None) - _EPOCH) // _MICROSECOND


def _history_summaries(
    moments: Sequence[datetime],
    start: datetime,
    streams: Mapping[EvidenceChannel, Sequence[datetime]],
    resolution: EvidenceResolution,
) -> dict[_Summary, np.ndarray]:
    """Every summary column of *resolution*, keyed by its source.

    Everything is computed from ``seen[:, j]``, the number of activations up
    to the end of lag window ``j``. So each value depends only on
    per-channel counts in whole step windows ending at or before the moment.
    """
    step = resolution.step
    windows = resolution.summary_windows
    horizon = resolution.horizon
    depth = horizon // step
    minutes_per_step = step / _MINUTE
    at = np.array([_wall_clock(moment) for moment in moments], dtype=np.int64)
    width = step // _MICROSECOND
    ends = at[:, None] - width * np.arange(depth + 1)[None, :]
    first = _wall_clock(start)

    def observed(steps: int) -> np.ndarray:
        """Rows whose last *steps* windows all close at or after the start."""
        closes: np.ndarray = at - width * (steps - 1) >= first
        return closes

    missing = np.full(len(at), math.nan)
    rooms = resolution.rooms
    bits = {room: 1 << position for position, room in enumerate(rooms)}
    occupied = np.zeros((len(at), depth), dtype=np.int64)
    summaries: dict[_Summary, np.ndarray] = {}
    for channel in resolution.channels:
        stamps = streams.get(channel)
        count = {w: _Summary(SUMMARY_COUNT, w, channel.name) for w in windows}
        quiet = _Summary(SUMMARY_QUIET_MINUTES, horizon, channel.name)
        if stamps is None:
            summaries.update({source: missing for source in count.values()})
            summaries[quiet] = missing
            continue
        keys = np.array([_wall_clock(stamp) for stamp in stamps], dtype=np.int64)
        seen = np.searchsorted(keys, ends, side="right")
        for window, source in count.items():
            steps = window // step
            summaries[source] = np.where(
                observed(steps), seen[:, 0] - seen[:, steps], math.nan
            )
        active = seen[:, :-1] > seen[:, 1:]
        occupied |= np.where(active, bits[channel.room], 0)
        summaries[quiet] = np.where(
            active.any(axis=1),
            active.argmax(axis=1) * minutes_per_step,
            np.where(observed(depth), depth * minutes_per_step, math.nan),
        )

    busy = occupied != 0
    anywhere = busy.any(axis=1)
    latest = occupied[np.arange(len(at)), busy.argmax(axis=1)]
    instrumented = {channel.room for channel in streams}
    for room in rooms:
        source = _Summary(SUMMARY_LAST_ROOM, horizon, room)
        if room not in instrumented:
            summaries[source] = missing
            continue
        summaries[source] = np.where(
            anywhere,
            (latest & bits[room]) != 0,
            np.where(observed(depth), 0.0, math.nan),
        )

    # For each active lag, the nearest more recent active lag, or -1.
    lags = np.arange(depth)
    reached = np.maximum.accumulate(np.where(busy, lags, -1), axis=1)
    newer = np.hstack([np.full((len(at), 1), -1), reached[:, :-1]])
    before = np.take_along_axis(occupied, np.maximum(newer, 0), axis=1)
    changes = np.cumsum(busy & (newer >= 0) & (before != occupied), axis=1)
    for window in windows:
        source = _Summary(SUMMARY_ROOM_CHANGES, window)
        steps = window // step
        summaries[source] = (
            np.where(observed(steps), changes[:, steps - 1], math.nan)
            if instrumented
            else missing
        )
    return summaries


def build_feature_table(
    recording: CasasRecording,
    information_set: InformationSet,
    moments: Sequence[datetime],
    *,
    household: str,
) -> FeatureTable:
    """Build one household's features under one information set.

    Parameters
    ----------
    recording
        The household's recording. Only its registry and observations are read;
        annotations are never consulted.
    information_set
        What the model may condition on.
    moments
        Prediction moments, strictly increasing. To compare against the
        filter on identical positions, pass the moments of its scoring steps,
        ``[step.at for step in scoring_steps(steps)]``.
    household
        Identifier recorded on the table.

    Raises
    ------
    ValueError
        If the recording has no observations, the household is unnamed, or the
        moments are not strictly increasing.
    """
    if not isinstance(household, str) or not household.strip():
        raise ValueError("a household identifier is required")
    if not recording.observations:
        raise ValueError(f"recording for {household!r} contains no observations")

    first = min(recording.observations, key=lambda observation: observation.timestamp)
    zone = first.timestamp.tzinfo
    if zone is None:  # pragma: no cover - observations are always aware
        raise ValueError("observation timestamps must be timezone-aware")
    start = first.timestamp.astimezone(zone)
    local = _local_moments(moments, zone)

    resolution = information_set.resolution
    routed, excluded = _route_sensors(recording.registry, resolution.channels)
    streams: dict[EvidenceChannel, list[datetime]] = {
        channel: [] for channel in routed.values()
    }
    for observation in recording.observations:
        channel = routed.get(observation.sensor_id)
        if channel is not None and observation.value != 0.0:
            streams[channel].append(observation.timestamp.astimezone(zone))
    for stream in streams.values():
        stream.sort()

    layout = information_set._layout()
    step = resolution.step
    values = np.full((len(local), len(layout)), math.nan)
    for row, moment in enumerate(local):
        for col, source in enumerate(layout):
            if isinstance(source, _Summary):
                continue
            if source.channel is None:
                values[row, col] = local_hour(moment, zone)
                continue
            stamps = streams.get(source.channel)
            end = moment - step * source.lag
            if stamps is None or end < start:
                continue
            values[row, col] = bisect_right(stamps, end) - bisect_right(
                stamps, end - step
            )
    summaries = [
        (col, source)
        for col, source in enumerate(layout)
        if isinstance(source, _Summary)
    ]
    if summaries:
        computed = _history_summaries(local, start, streams, resolution)
        for col, source in summaries:
            values[:, col] = computed[source]

    return FeatureTable(
        household=household,
        information_set=information_set,
        moments=local,
        values=values,
        uninstrumented=tuple(
            channel for channel in resolution.channels if channel not in streams
        ),
        excluded_sensors=excluded,
    )


def build_panel_features(
    recordings: Mapping[str, CasasRecording],
    information_set: InformationSet,
    moments: Mapping[str, Sequence[datetime]],
) -> dict[str, FeatureTable]:
    """Build every household's features, one recording at a time.

    Households are never pooled before features are built, so no window, and
    therefore no history, can span two homes. Splits into development and test
    homes are made on the returned keys, at the household level, exactly as
    they are fixed elsewhere; nothing here chooses them.

    Returns
    -------
    dict
        Tables keyed by household, in sorted household order.
    """
    if set(recordings) != set(moments):
        missing = sorted(set(recordings) ^ set(moments))
        raise ValueError(
            f"recordings and moments must name the same households; differ on {missing}"
        )
    return {
        household: build_feature_table(
            recordings[household],
            information_set,
            moments[household],
            household=household,
        )
        for household in sorted(recordings)
    }
