"""Tests for converting legacy tabular data into canonical observations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from sensor_modeling.observations import (
    LegacyConversionError,
    Modality,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
    Unit,
    UnknownSensorError,
    naive_utc,
    observations_from_dataset,
    observations_from_frame,
    observations_from_records,
)
from sensor_modeling.utils.data_io import SensorDataset

LISBON = ZoneInfo("Europe/Lisbon")
T0 = datetime(2024, 5, 1, 8, 0, tzinfo=timezone.utc)


def registry() -> SensorRegistry:
    """A registry covering an event, a state and a sampled sensor."""
    return SensorRegistry.from_specs(
        [
            SensorSpec("kitchen_motion", Modality.MOTION, room="kitchen"),
            SensorSpec(
                "bed",
                Modality.BED_PRESSURE,
                kind=ObservationKind.STATE,
                room="bedroom",
            ),
            SensorSpec(
                "hall_temp",
                Modality.ENVIRONMENTAL,
                kind=ObservationKind.SAMPLE,
                unit=Unit.CELSIUS,
                room="hall",
            ),
        ]
    )


def frame(*, tz: object = timezone.utc) -> pd.DataFrame:
    """A legacy wide frame with one column per sensor."""
    naive_start = T0.replace(tzinfo=None)
    index = pd.date_range(naive_start, periods=4, freq="15min", tz=tz)
    return pd.DataFrame(
        {
            "kitchen_motion": [1.0, 0.0, 1.0, 0.0],
            "bed": [0.0, 0.0, 1.0, 1.0],
            "hall_temp": [20.0, 20.5, 21.0, 20.8],
        },
        index=index,
    )


class TestWideFrameConversion:
    def test_every_registered_column_is_converted(self) -> None:
        observations = list(observations_from_frame(frame(), registry()))
        by_sensor = {obs.sensor_id for obs in observations}
        assert by_sensor == {"kitchen_motion", "bed", "hall_temp"}

    def test_event_zeros_are_not_emitted_as_observations(self) -> None:
        """A zero in an event column is an absent record, not an observation.

        Emitting it would fabricate the exact evidence the canonical model
        exists to withhold.
        """
        observations = list(observations_from_frame(frame(), registry()))
        motion = [o for o in observations if o.sensor_id == "kitchen_motion"]
        assert len(motion) == 2
        assert all(obs.value == 1.0 for obs in motion)

    def test_state_and_sample_zeros_are_kept(self) -> None:
        """A bed reading zero is a genuine measurement of an empty bed."""
        observations = list(observations_from_frame(frame(), registry()))
        bed = [o for o in observations if o.sensor_id == "bed"]
        assert len(bed) == 4
        assert bed[0].value == 0.0

    def test_declared_semantics_are_applied(self) -> None:
        observations = list(observations_from_frame(frame(), registry()))
        temp = next(o for o in observations if o.sensor_id == "hall_temp")
        assert temp.modality is Modality.ENVIRONMENTAL
        assert temp.kind is ObservationKind.SAMPLE
        assert temp.unit is Unit.CELSIUS

    def test_missing_cells_are_skipped_not_imputed(self) -> None:
        data = frame()
        data.loc[data.index[1], "hall_temp"] = None
        observations = list(observations_from_frame(data, registry()))
        assert len([o for o in observations if o.sensor_id == "hall_temp"]) == 3

    def test_an_unregistered_column_is_refused_rather_than_guessed(self) -> None:
        data = frame()
        data["mystery"] = 1.0
        with pytest.raises(UnknownSensorError, match="not registered"):
            list(observations_from_frame(data, registry()))

    def test_unregistered_columns_can_be_skipped_explicitly(self) -> None:
        data = frame()
        data["derived_feature"] = 1.0
        observations = list(
            observations_from_frame(data, registry(), skip_unregistered=True)
        )
        assert "derived_feature" not in {o.sensor_id for o in observations}

    def test_a_naive_index_requires_an_explicit_timezone(self) -> None:
        """Assuming UTC or local would silently shift every record."""
        with pytest.raises(LegacyConversionError, match="naive timestamps"):
            list(observations_from_frame(frame(tz=None), registry()))

    def test_a_supplied_timezone_is_applied(self) -> None:
        observations = list(
            observations_from_frame(frame(tz=None), registry(), tz=LISBON)
        )
        assert all(obs.timestamp.tzinfo is not None for obs in observations)
        assert observations[0].timestamp.utcoffset() is not None

    def test_a_non_timestamp_index_is_refused(self) -> None:
        data = frame().reset_index(drop=True)
        with pytest.raises(LegacyConversionError, match="indexed by timestamp"):
            list(observations_from_frame(data, registry()))

    def test_an_empty_registry_yields_nothing(self) -> None:
        observations = list(
            observations_from_frame(frame(), SensorRegistry(), skip_unregistered=True)
        )
        assert observations == []

    def test_converted_records_survive_registry_normalisation(self) -> None:
        deployment = registry()
        for observation in observations_from_frame(frame(), deployment):
            assert deployment.normalise(observation) is not None


class TestDatasetAndRecords:
    def test_a_legacy_dataset_converts(self) -> None:
        dataset = SensorDataset(frame())
        observations = list(observations_from_dataset(dataset, registry()))
        assert observations
        assert all(obs.timestamp.tzinfo is not None for obs in observations)

    def test_an_object_without_to_dataframe_is_refused(self) -> None:
        with pytest.raises(LegacyConversionError, match="to_dataframe"):
            list(observations_from_dataset(object(), registry()))

    def test_long_format_records_convert(self) -> None:
        records = [
            {"timestamp": T0, "sensor_id": "kitchen_motion", "value": 1.0},
            {
                "timestamp": T0 + timedelta(minutes=5),
                "sensor_id": "hall_temp",
                "value": 21.0,
            },
        ]
        observations = list(observations_from_records(records, registry()))
        assert [o.sensor_id for o in observations] == ["kitchen_motion", "hall_temp"]
        assert observations[1].unit is Unit.CELSIUS

    def test_malformed_records_are_refused(self) -> None:
        with pytest.raises(LegacyConversionError, match="mapping"):
            list(observations_from_records([("not", "a", "mapping")], registry()))

    def test_naive_utc_is_an_explicit_opt_in(self) -> None:
        naive = datetime(2024, 5, 1, 8, 0)
        assert naive_utc(naive).tzinfo is timezone.utc
        assert naive_utc(T0) is T0


class TestRoundTrip:
    def test_framing_recovers_the_event_counts_it_started_from(self) -> None:
        """A wide frame converted to observations and framed back must not
        gain or lose activations."""
        from sensor_modeling.observations import ObservationStream

        original = frame()
        stream = ObservationStream.from_observations(
            observations_from_frame(original, registry())
        )
        counts = stream.event_counts("15min", sensor_ids=["kitchen_motion"])
        assert counts["kitchen_motion"].sum() == original["kitchen_motion"].sum()
