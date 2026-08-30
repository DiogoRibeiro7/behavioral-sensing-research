"""Converting legacy tabular sensor data into canonical observations.

The original toolkit represents sensor data as a :class:`~pandas.DataFrame`
indexed by timestamp with one column per sensor. That form carries no
modality, no unit, no quality, and -- most consequentially -- no distinction
between an event stream and a sampled signal.

These adapters bridge the two, but they cannot invent the missing information.
A caller must supply a :class:`~sensor_modeling.observations.SensorRegistry`
saying what each column actually is. That requirement is deliberate: guessing
a column's modality from its name is exactly the hardware-specific coupling
the canonical model exists to remove, and guessing its *kind* would decide,
invisibly, whether a gap in the data means "nothing happened" or "nothing was
measured".

Conversion runs in one direction only. Canonical observations can be framed
back into tables through :class:`~sensor_modeling.observations.ObservationStream`,
so the older models keep working unchanged, but nothing above the observation
layer consumes tables.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone, tzinfo

import pandas as pd

from .observation import Observation
from .registry import SensorRegistry, UnknownSensorError
from .types import ObservationKind

logger = logging.getLogger(__name__)


class LegacyConversionError(ValueError):
    """Raised when a legacy frame cannot be converted without guessing."""


def _localise(index: pd.DatetimeIndex, tz: tzinfo | None) -> pd.DatetimeIndex:
    """Return *index* as timezone-aware, without inventing an offset.

    A naive legacy frame carries no timezone, so one must be supplied. It is
    never assumed, because assuming UTC or local silently shifts every record
    by hours and breaks daily-rhythm analysis across DST boundaries.
    """
    if index.tz is not None:
        return index
    if tz is None:
        raise LegacyConversionError(
            "legacy frame has naive timestamps; pass tz= to say which timezone "
            "they were recorded in rather than having one assumed"
        )
    return index.tz_localize(tz)


def observations_from_frame(
    frame: pd.DataFrame,
    registry: SensorRegistry,
    *,
    tz: tzinfo | None = None,
    source: str = "legacy-frame",
    skip_unregistered: bool = False,
) -> Iterator[Observation]:
    """Yield canonical observations from a legacy sensor frame.

    Parameters
    ----------
    frame
        Timestamp-indexed frame with one column per sensor.
    registry
        Declarations for the columns. Every column must be registered, so that
        modality, unit and temporal semantics come from the deployment rather
        than from a guess.
    tz
        Timezone of a naive index. Required when the index is naive.
    source
        Value recorded as the observation's ``source``.
    skip_unregistered
        When true, unregistered columns are logged and skipped instead of
        raising. Useful for frames carrying derived columns alongside sensors.

    Yields
    ------
    Observation
        One record per non-null cell, in timestamp order.

    Notes
    -----
    Zero-valued cells are emitted for ``STATE`` and ``SAMPLE`` sensors, whose
    zeros are genuine measurements, but **not** for ``EVENT`` sensors, whose
    zeros are the absence of a record rather than an observation of nothing.
    Emitting them would fabricate exactly the evidence the canonical model
    exists to withhold.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise LegacyConversionError("legacy frame must be indexed by timestamp")

    index = _localise(frame.index, tz)
    columns = []
    for column in frame.columns:
        spec = registry.get(str(column))
        if spec is None:
            if skip_unregistered:
                logger.debug("Skipping unregistered column '%s'", column)
                continue
            raise UnknownSensorError(
                f"column '{column}' is not registered; declare it in the "
                "registry so its modality and temporal semantics are explicit"
            )
        columns.append((str(column), spec))

    if not columns:
        return

    for position, moment in enumerate(index):
        timestamp = moment.to_pydatetime()
        for name, spec in columns:
            value = frame.iloc[position][name]
            if pd.isna(value):
                continue
            numeric = float(value)
            if spec.kind is ObservationKind.EVENT and numeric == 0.0:
                continue
            yield Observation(
                timestamp=timestamp,
                sensor_id=name,
                modality=spec.modality,
                kind=spec.kind or ObservationKind.SAMPLE,
                value=numeric,
                unit=spec.unit,
                source=source,
                context={"origin": "legacy-frame"},
            )


def observations_from_dataset(
    dataset: object,
    registry: SensorRegistry,
    **kwargs: object,
) -> Iterator[Observation]:
    """Yield canonical observations from a legacy ``SensorDataset``.

    Accepts anything exposing ``to_dataframe()``, which keeps this adapter
    from importing the older data layer and creating a cycle.
    """
    to_dataframe = getattr(dataset, "to_dataframe", None)
    if to_dataframe is None:
        raise LegacyConversionError("dataset must expose to_dataframe()")
    return observations_from_frame(to_dataframe(), registry, **kwargs)  # type: ignore[arg-type]


def observations_from_records(
    records: object,
    registry: SensorRegistry,
    *,
    timestamp_key: str = "timestamp",
    sensor_key: str = "sensor_id",
    value_key: str = "value",
    source: str = "legacy-records",
) -> Iterator[Observation]:
    """Yield canonical observations from long-format records.

    Long format -- one row per reading, with a sensor column -- is what most
    event-driven gateways export, and it carries event semantics correctly
    because an absent row is simply an absent row.
    """
    for record in records:  # type: ignore[attr-defined]
        if not isinstance(record, Mapping):
            raise LegacyConversionError("each record must be a mapping")
        sensor_id = str(record[sensor_key])
        spec = registry[sensor_id]
        yield Observation(
            timestamp=record[timestamp_key],
            sensor_id=sensor_id,
            modality=spec.modality,
            kind=spec.kind or ObservationKind.SAMPLE,
            value=float(record[value_key]),
            unit=spec.unit,
            source=str(record.get("source", source)),
            context={"origin": "legacy-records"},
        )


def naive_utc(moment: datetime) -> datetime:
    """Attach UTC to a naive datetime, explicitly and visibly.

    Provided so that a caller who genuinely knows their legacy data is UTC can
    say so at the call site, rather than having the assumption buried in a
    conversion routine.
    """
    if moment.tzinfo is not None:
        return moment
    return moment.replace(tzinfo=timezone.utc)
