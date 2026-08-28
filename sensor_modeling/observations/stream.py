"""Ordered storage and tabular views of canonical observations.

An :class:`ObservationStream` keeps observations in timestamp order regardless
of the order in which they arrive, collapses exact duplicates, and can report
where a sensor went quiet. It also converts observations into the tabular
form the existing modelling code expects.

The conversion deliberately offers three different framings, because the three
:class:`~sensor_modeling.observations.types.ObservationKind` values cannot be
tabulated the same way. Event streams are counted, never forward-filled: an
empty bin means "no event was recorded", which is evidence about the sensor as
much as about the resident, and turning it into a zero would silently convert a
dead sensor into observed inactivity.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo

import numpy as np
import pandas as pd

from .observation import Observation, require_aware
from .types import ObservationKind

logger = logging.getLogger(__name__)


def _utc_key(observation: Observation) -> datetime:
    """Return the UTC instant used to order an observation."""
    return observation.timestamp.astimezone(timezone.utc)


@dataclass(frozen=True)
class Gap:
    """A period during which a sensor reported nothing.

    A gap is a statement about the *record*, not about the resident. Whether
    it reflects a broken sensor or a genuinely quiet period is decided by the
    health monitor, not here.
    """

    sensor_id: str
    start: datetime
    end: datetime

    @property
    def duration(self) -> timedelta:
        """Length of the silent period."""
        return self.end - self.start


@dataclass
class ObservationStream:
    """A timestamp-ordered, duplicate-free collection of observations."""

    _items: list[Observation] = field(default_factory=list, repr=False)
    _identities: set[tuple[datetime, str, float]] = field(
        default_factory=set, repr=False
    )

    @classmethod
    def from_observations(
        cls, observations: Iterable[Observation]
    ) -> ObservationStream:
        """Build a stream from *observations*, in any order."""
        stream = cls()
        stream.extend(observations)
        return stream

    # ------------------------------------------------------------------
    def add(self, observation: Observation) -> bool:
        """Insert *observation* in timestamp order.

        Returns
        -------
        bool
            ``True`` when the observation was stored, ``False`` when it was
            an exact duplicate of a record already present and was dropped.
        """
        identity = observation.identity()
        if identity in self._identities:
            logger.debug("Dropping duplicate observation %s", identity)
            return False
        bisect.insort(self._items, observation, key=_utc_key)
        self._identities.add(identity)
        return True

    def extend(self, observations: Iterable[Observation]) -> int:
        """Insert many observations and return how many were stored."""
        return sum(1 for obs in observations if self.add(obs))

    def would_be_out_of_order(self, observation: Observation) -> bool:
        """Whether *observation* predates the newest record already stored."""
        if not self._items:
            return False
        return _utc_key(observation) < _utc_key(self._items[-1])

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Observation]:
        return iter(self._items)

    def __getitem__(self, index: int) -> Observation:
        return self._items[index]

    @property
    def start(self) -> datetime | None:
        """Timestamp of the earliest observation, if any."""
        return self._items[0].timestamp if self._items else None

    @property
    def end(self) -> datetime | None:
        """Timestamp of the latest observation, if any."""
        return self._items[-1].timestamp if self._items else None

    def sensor_ids(self) -> list[str]:
        """Return the sorted set of sensors that appear in the stream."""
        return sorted({obs.sensor_id for obs in self._items})

    def by_sensor(self, sensor_id: str) -> list[Observation]:
        """Return every observation from *sensor_id*, in timestamp order."""
        return [obs for obs in self._items if obs.sensor_id == sensor_id]

    def between(self, start: datetime, end: datetime) -> list[Observation]:
        """Return observations in the half-open interval ``[start, end)``."""
        start_utc = require_aware(start, "start").astimezone(timezone.utc)
        end_utc = require_aware(end, "end").astimezone(timezone.utc)
        if end_utc < start_utc:
            raise ValueError("end must not precede start")
        keys = [_utc_key(obs) for obs in self._items]
        left = bisect.bisect_left(keys, start_utc)
        right = bisect.bisect_left(keys, end_utc)
        return self._items[left:right]

    # ------------------------------------------------------------------
    def gaps(self, sensor_id: str, max_interval: timedelta) -> list[Gap]:
        """Return periods where *sensor_id* was silent for longer than allowed.

        Only the interior of the record is examined. Silence before the first
        or after the last observation is not reported, because the stream
        cannot tell a missing sensor from one that has not started yet.
        """
        if max_interval <= timedelta(0):
            raise ValueError("max_interval must be positive")
        observations = self.by_sensor(sensor_id)
        found: list[Gap] = []
        for previous, current in zip(observations, observations[1:]):
            if current.timestamp - previous.timestamp > max_interval:
                found.append(Gap(sensor_id, previous.timestamp, current.timestamp))
        return found

    # ------------------------------------------------------------------
    def _grid(self, freq: str, tz: tzinfo | None) -> pd.DatetimeIndex:
        """Return the regular time grid the framing helpers bin onto.

        Bin edges are aligned to the local clock, because behavioural rhythms
        follow local time, but the grid itself is generated from a fixed
        offset so that its instants stay contiguous across DST transitions.
        """
        if not self._items:
            return pd.DatetimeIndex([], tz=tz or timezone.utc)
        zone = tz if tz is not None else self._items[0].timestamp.tzinfo
        step = pd.Timedelta(freq)
        if step <= pd.Timedelta(0):
            raise ValueError("freq must be a positive fixed frequency")
        first = pd.Timestamp(self._items[0].timestamp).tz_convert(zone)
        last = pd.Timestamp(self._items[-1].timestamp).tz_convert(zone)
        start = first.floor(freq, ambiguous=True, nonexistent="shift_backward")
        end = last.floor(freq, ambiguous=True, nonexistent="shift_backward")
        return pd.date_range(start=start, end=end, freq=step, tz=zone)

    @staticmethod
    def _positions(
        index: pd.DatetimeIndex, observations: Sequence[Observation], freq: str
    ) -> np.ndarray:
        """Map observations onto grid positions, or ``-1`` when outside it.

        The lookup is done on absolute UTC instants rather than local wall
        times. Flooring a wall time that falls inside a spring-forward gap
        produces a local instant that does not exist on the grid, which would
        drop every observation recorded during the transition.
        """
        if len(index) == 0 or not observations:
            return np.full(len(observations), -1, dtype=int)
        edges = index.tz_convert("UTC").to_numpy(dtype="datetime64[ns]").astype("int64")
        span = int(pd.Timedelta(freq).value)
        stamps = np.array(
            [
                pd.Timestamp(obs.timestamp).tz_convert("UTC").value
                for obs in observations
            ],
            dtype="int64",
        )
        positions = np.searchsorted(edges, stamps, side="right") - 1
        positions[stamps >= edges[-1] + span] = -1
        return positions

    def _selected(self, sensor_ids: Sequence[str] | None) -> list[str]:
        """Return the sensor columns to build, defaulting to all present."""
        return list(sensor_ids) if sensor_ids is not None else self.sensor_ids()

    def _placed(
        self,
        kind: ObservationKind,
        freq: str,
        index: pd.DatetimeIndex,
        columns: list[str],
    ) -> list[tuple[int, int, float]]:
        """Return ``(row, column, value)`` triples for one observation kind.

        Triples are produced in timestamp order, so a consumer that only
        wants the most recent value per cell can simply overwrite as it goes.
        """
        column_of = {name: position for position, name in enumerate(columns)}
        selected = [
            obs
            for obs in self._items
            if obs.kind is kind and obs.sensor_id in column_of
        ]
        rows = self._positions(index, selected, freq)
        placed = [
            (int(row), column_of[obs.sensor_id], obs.value)
            for obs, row in zip(selected, rows)
            if row >= 0
        ]
        dropped = len(selected) - len(placed)
        if dropped:
            logger.warning(
                "%d %s observations fell outside the framing grid", dropped, kind.value
            )
        return placed

    def event_counts(
        self,
        freq: str = "15min",
        *,
        sensor_ids: Sequence[str] | None = None,
        tz: tzinfo | None = None,
    ) -> pd.DataFrame:
        """Return the number of recorded activations per sensor per time bin.

        A zero means no event was *recorded* in that bin. It does not mean the
        sensor was working and observed nothing; pair this frame with
        :meth:`observed_mask` and sensor health output before treating zeros
        as evidence of inactivity.
        """
        index = self._grid(freq, tz)
        columns = self._selected(sensor_ids)
        counts = np.zeros((len(index), len(columns)))
        for row, column, value in self._placed(
            ObservationKind.EVENT, freq, index, columns
        ):
            counts[row, column] += 1.0 if value != 0.0 else 0.0
        return pd.DataFrame(counts, index=index, columns=columns)

    def sample_frame(
        self,
        freq: str = "15min",
        *,
        sensor_ids: Sequence[str] | None = None,
        tz: tzinfo | None = None,
    ) -> pd.DataFrame:
        """Return the mean sampled value per sensor per bin.

        Bins with no sample are ``NaN`` and are left that way. Filling them
        would fabricate measurements of a continuously existing quantity.
        """
        index = self._grid(freq, tz)
        columns = self._selected(sensor_ids)
        totals = np.zeros((len(index), len(columns)))
        counts = np.zeros((len(index), len(columns)))
        for row, column, value in self._placed(
            ObservationKind.SAMPLE, freq, index, columns
        ):
            totals[row, column] += value
            counts[row, column] += 1.0
        means = np.divide(
            totals, counts, out=np.full_like(totals, np.nan), where=counts > 0
        )
        return pd.DataFrame(means, index=index, columns=columns)

    def state_frame(
        self,
        freq: str = "15min",
        *,
        sensor_ids: Sequence[str] | None = None,
        tz: tzinfo | None = None,
        max_hold: timedelta | None = None,
    ) -> pd.DataFrame:
        """Return the last reported state per sensor per bin.

        State observations persist until the next reported change, so carrying
        the last value forward is meaningful here -- but only for *max_hold*.
        Beyond that the state is unknown rather than unchanged, and the cell
        becomes ``NaN``.
        """
        index = self._grid(freq, tz)
        columns = self._selected(sensor_ids)
        values = np.full((len(index), len(columns)), np.nan)
        for row, column, value in self._placed(
            ObservationKind.STATE, freq, index, columns
        ):
            values[row, column] = value

        frame = pd.DataFrame(values, index=index, columns=columns)
        if len(index) == 0 or max_hold is None:
            return frame.ffill() if len(index) else frame
        limit = max(int(max_hold / pd.Timedelta(freq)), 1)
        return frame.ffill(limit=limit)

    def observed_mask(
        self,
        freq: str = "15min",
        *,
        sensor_ids: Sequence[str] | None = None,
        tz: tzinfo | None = None,
    ) -> pd.DataFrame:
        """Return which bins contain at least one record per sensor.

        This is the companion to :meth:`event_counts`: it separates "the
        sensor reported nothing" from "the sensor reported no activity".
        """
        index = self._grid(freq, tz)
        columns = self._selected(sensor_ids)
        mask = np.zeros((len(index), len(columns)), dtype=bool)
        column_of = {name: position for position, name in enumerate(columns)}
        relevant = [obs for obs in self._items if obs.sensor_id in column_of]
        for obs, row in zip(relevant, self._positions(index, relevant, freq)):
            if row >= 0:
                mask[int(row), column_of[obs.sensor_id]] = True
        return pd.DataFrame(mask, index=index, columns=columns)

    def to_dicts(self) -> list[dict[str, object]]:
        """Return every observation as a JSON-serialisable mapping."""
        return [obs.to_dict() for obs in self._items]
