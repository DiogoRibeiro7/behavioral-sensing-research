"""Tests for the canonical observation model, registry, ingestion, and stream."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from sensor_modeling.observations import (
    ClockOffsetEstimator,
    Modality,
    Observation,
    ObservationFlag,
    ObservationIngestor,
    ObservationKind,
    ObservationStream,
    SensorContractError,
    SensorRegistry,
    SensorSpec,
    Unit,
    UnknownSensorError,
    convert,
    default_kind,
)

LISBON = ZoneInfo("Europe/Lisbon")
T0 = datetime(2024, 5, 1, 8, 0, tzinfo=timezone.utc)


def make_observation(**overrides: object) -> Observation:
    """Build a valid observation with selective overrides."""
    payload: dict[str, object] = {
        "timestamp": T0,
        "sensor_id": "kitchen_motion",
        "modality": Modality.MOTION,
        "kind": ObservationKind.EVENT,
        "value": 1.0,
    }
    payload.update(overrides)
    return Observation(**payload)  # type: ignore[arg-type]


class TestObservationValidation:
    def test_naive_timestamp_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            make_observation(timestamp=datetime(2024, 5, 1, 8, 0))

    def test_iso_string_timestamp_is_parsed(self) -> None:
        obs = make_observation(timestamp="2024-05-01T08:00:00+00:00")
        assert obs.timestamp == T0

    def test_malformed_timestamp_string_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ISO-8601"):
            make_observation(timestamp="not-a-timestamp")

    def test_non_finite_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            make_observation(value=math.nan)

    def test_empty_sensor_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="sensor_id"):
            make_observation(sensor_id="   ")

    def test_probability_unit_bounds_are_enforced(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            make_observation(value=1.5, unit=Unit.PROBABILITY)

    def test_count_unit_requires_non_negative_integers(self) -> None:
        with pytest.raises(ValueError, match="non-negative integers"):
            make_observation(value=1.5, unit=Unit.COUNT)

    @pytest.mark.parametrize("field_name", ["quality", "confidence"])
    def test_reliability_fields_must_lie_in_unit_interval(
        self, field_name: str
    ) -> None:
        with pytest.raises(ValueError, match=field_name):
            make_observation(**{field_name: 1.2})

    def test_negative_sampling_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="sampling_interval"):
            make_observation(sampling_interval=timedelta(seconds=-1))

    def test_observation_is_hashable_despite_context_mapping(self) -> None:
        obs = make_observation(context={"room": "kitchen"})
        assert isinstance(hash(obs), int)
        assert obs.context["room"] == "kitchen"

    def test_context_is_immutable(self) -> None:
        obs = make_observation(context={"room": "kitchen"})
        with pytest.raises(TypeError):
            obs.context["room"] = "hall"  # type: ignore[index]

    def test_evidence_weight_combines_quality_and_confidence(self) -> None:
        obs = make_observation(quality=0.5, confidence=0.5)
        assert obs.evidence_weight() == pytest.approx(0.25)

    def test_dict_roundtrip_preserves_equality(self) -> None:
        obs = make_observation(
            unit=Unit.NONE,
            quality=0.8,
            confidence=0.9,
            source="hub-1",
            sampling_interval=timedelta(minutes=5),
            received_at=T0 + timedelta(seconds=3),
            flags=frozenset({ObservationFlag.LATE_ARRIVAL}),
            context={"room": "kitchen"},
        )
        assert Observation.from_dict(obs.to_dict()) == obs

    def test_canonical_unit_conversion_flags_the_change(self) -> None:
        obs = make_observation(
            modality=Modality.ENVIRONMENTAL,
            kind=ObservationKind.SAMPLE,
            value=212.0,
            unit=Unit.FAHRENHEIT,
        )
        canonical = obs.in_canonical_unit()
        assert canonical.value == pytest.approx(100.0)
        assert canonical.unit is Unit.CELSIUS
        assert ObservationFlag.UNIT_CONVERTED in canonical.flags

    def test_incompatible_unit_conversion_raises(self) -> None:
        with pytest.raises(ValueError, match="different dimensions"):
            convert(1.0, Unit.LUX, Unit.METRE)

    def test_default_kind_treats_contact_sensors_as_events(self) -> None:
        assert default_kind(Modality.CONTACT) is ObservationKind.EVENT
        assert default_kind(Modality.ENVIRONMENTAL) is ObservationKind.SAMPLE


class TestSensorRegistry:
    @staticmethod
    def registry() -> SensorRegistry:
        return SensorRegistry.from_specs(
            [
                SensorSpec("kitchen_motion", Modality.MOTION, room="kitchen"),
                SensorSpec(
                    "hall_temp",
                    Modality.ENVIRONMENTAL,
                    unit=Unit.CELSIUS,
                    room="hall",
                    value_range=(-20.0, 60.0),
                ),
            ]
        )

    def test_unknown_sensor_is_refused_rather_than_guessed(self) -> None:
        with pytest.raises(UnknownSensorError):
            self.registry().normalise(make_observation(sensor_id="ghost"))

    def test_modality_mismatch_is_a_contract_error(self) -> None:
        obs = make_observation(sensor_id="kitchen_motion", modality=Modality.DOOR)
        with pytest.raises(SensorContractError, match="registered as motion"):
            self.registry().normalise(obs)

    def test_convertible_unit_is_normalised_to_the_declared_unit(self) -> None:
        obs = make_observation(
            sensor_id="hall_temp",
            modality=Modality.ENVIRONMENTAL,
            kind=ObservationKind.SAMPLE,
            value=68.0,
            unit=Unit.FAHRENHEIT,
        )
        normalised = self.registry().normalise(obs)
        assert normalised.value == pytest.approx(20.0)
        assert normalised.unit is Unit.CELSIUS
        assert ObservationFlag.UNIT_CONVERTED in normalised.flags

    def test_incompatible_unit_is_a_contract_error(self) -> None:
        obs = make_observation(
            sensor_id="hall_temp",
            modality=Modality.ENVIRONMENTAL,
            kind=ObservationKind.SAMPLE,
            value=5.0,
            unit=Unit.LUX,
        )
        with pytest.raises(SensorContractError, match="expects degC"):
            self.registry().normalise(obs)

    def test_range_check_uses_the_declared_bounds(self) -> None:
        registry = self.registry()
        inside = make_observation(
            sensor_id="hall_temp",
            modality=Modality.ENVIRONMENTAL,
            kind=ObservationKind.SAMPLE,
            value=21.0,
            unit=Unit.CELSIUS,
        )
        outside = make_observation(
            sensor_id="hall_temp",
            modality=Modality.ENVIRONMENTAL,
            kind=ObservationKind.SAMPLE,
            value=250.0,
            unit=Unit.CELSIUS,
        )
        assert registry.in_range(inside)
        assert not registry.in_range(outside)

    def test_subset_supports_ablation_without_touching_inference(self) -> None:
        subset = self.registry().subset(["kitchen_motion"])
        assert subset.sensor_ids() == ["kitchen_motion"]
        with pytest.raises(UnknownSensorError):
            self.registry().subset(["nope"])

    def test_rooms_and_modality_lookups(self) -> None:
        registry = self.registry()
        assert registry.rooms() == ["hall", "kitchen"]
        assert [s.sensor_id for s in registry.by_modality(Modality.MOTION)] == [
            "kitchen_motion"
        ]
        assert len(registry) == 2
        assert "hall_temp" in registry

    def test_invalid_spec_configuration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="prior_reliability"):
            SensorSpec("s", Modality.MOTION, prior_reliability=0.0)
        with pytest.raises(ValueError, match="value_range"):
            SensorSpec("s", Modality.MOTION, value_range=(5.0, 1.0))
        with pytest.raises(ValueError, match="expected_interval"):
            SensorSpec("s", Modality.MOTION, expected_interval=timedelta(0))


class TestIngestion:
    @staticmethod
    def ingestor(**kwargs: object) -> ObservationIngestor:
        registry = SensorRegistry.from_specs(
            [SensorSpec("kitchen_motion", Modality.MOTION, room="kitchen")]
        )
        return ObservationIngestor(registry, **kwargs)  # type: ignore[arg-type]

    def test_unknown_sensors_are_reported_not_raised(self) -> None:
        admitted, report = self.ingestor().ingest_many(
            [make_observation(), make_observation(sensor_id="ghost")]
        )
        assert len(admitted) == 1
        assert report.accepted == 1
        assert [r.reason for r in report.rejected] == ["unknown_sensor"]
        assert report.received == 2

    def test_out_of_order_records_are_flagged_and_still_admitted(self) -> None:
        later = make_observation(timestamp=T0 + timedelta(minutes=10))
        earlier = make_observation(timestamp=T0)
        admitted, report = self.ingestor().ingest_many([later, earlier])
        assert report.out_of_order == 1
        assert ObservationFlag.OUT_OF_ORDER in admitted[1].flags
        assert ObservationFlag.OUT_OF_ORDER not in admitted[0].flags

    def test_late_arrivals_are_flagged(self) -> None:
        obs = make_observation(received_at=T0 + timedelta(minutes=30))
        _, report = self.ingestor(late_threshold=timedelta(minutes=5)).ingest_many(
            [obs]
        )
        assert report.late_arrivals == 1

    def test_clock_drift_correction_shifts_and_flags_timestamps(self) -> None:
        estimator = ClockOffsetEstimator(
            window=8, min_samples=3, tolerance=timedelta(seconds=10)
        )
        ingestor = self.ingestor(correct_clock_drift=True, clock_estimator=estimator)
        # A source whose clock lags by 60 s reports latencies of about +60 s.
        batch = [
            make_observation(
                timestamp=T0 + timedelta(minutes=i),
                source="hub-lagging",
                received_at=T0 + timedelta(minutes=i, seconds=60),
            )
            for i in range(6)
        ]
        admitted, report = ingestor.ingest_many(batch)
        assert report.clock_adjusted > 0
        adjusted = admitted[-1]
        assert ObservationFlag.CLOCK_ADJUSTED in adjusted.flags
        assert adjusted.timestamp > batch[-1].timestamp

    def test_no_correction_when_offset_is_within_tolerance(self) -> None:
        estimator = ClockOffsetEstimator(
            window=8, min_samples=3, tolerance=timedelta(seconds=30)
        )
        ingestor = self.ingestor(correct_clock_drift=True, clock_estimator=estimator)
        batch = [
            make_observation(
                timestamp=T0 + timedelta(minutes=i),
                source="hub-ok",
                received_at=T0 + timedelta(minutes=i, seconds=2),
            )
            for i in range(6)
        ]
        _, report = ingestor.ingest_many(batch)
        assert report.clock_adjusted == 0

    def test_snapshot_and_restore_preserve_the_ordering_watermark(self) -> None:
        ingestor = self.ingestor()
        ingestor.ingest_many([make_observation(timestamp=T0 + timedelta(hours=1))])
        state = ingestor.snapshot()

        restored = self.ingestor()
        restored.restore(state)
        _, report = restored.ingest_many([make_observation(timestamp=T0)])
        assert report.out_of_order == 1

    def test_invalid_estimator_configuration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_samples"):
            ClockOffsetEstimator(window=4, min_samples=8)


class TestObservationStream:
    @staticmethod
    def motion(minute: int, sensor: str = "kitchen_motion") -> Observation:
        return make_observation(
            timestamp=T0 + timedelta(minutes=minute), sensor_id=sensor
        )

    def test_insertion_keeps_timestamp_order_regardless_of_arrival_order(self) -> None:
        stream = ObservationStream.from_observations(
            [self.motion(30), self.motion(0), self.motion(15)]
        )
        assert [obs.timestamp.minute for obs in stream] == [0, 15, 30]

    def test_exact_duplicates_are_collapsed(self) -> None:
        stream = ObservationStream()
        assert stream.add(self.motion(0)) is True
        assert stream.add(self.motion(0)) is False
        assert len(stream) == 1

    def test_would_be_out_of_order_detects_late_records(self) -> None:
        stream = ObservationStream.from_observations([self.motion(30)])
        assert stream.would_be_out_of_order(self.motion(10))
        assert not stream.would_be_out_of_order(self.motion(40))

    def test_between_returns_a_half_open_interval(self) -> None:
        stream = ObservationStream.from_observations(
            [self.motion(0), self.motion(15), self.motion(30)]
        )
        window = stream.between(T0, T0 + timedelta(minutes=30))
        assert [obs.timestamp.minute for obs in window] == [0, 15]
        with pytest.raises(ValueError, match="end must not precede start"):
            stream.between(T0 + timedelta(minutes=30), T0)

    def test_gaps_report_interior_silence_only(self) -> None:
        stream = ObservationStream.from_observations(
            [self.motion(0), self.motion(5), self.motion(120)]
        )
        gaps = stream.gaps("kitchen_motion", timedelta(minutes=30))
        assert len(gaps) == 1
        assert gaps[0].duration == timedelta(minutes=115)
        with pytest.raises(ValueError, match="max_interval"):
            stream.gaps("kitchen_motion", timedelta(0))

    def test_event_counts_never_forward_fill(self) -> None:
        stream = ObservationStream.from_observations([self.motion(0), self.motion(120)])
        counts = stream.event_counts("1h")
        assert list(counts["kitchen_motion"]) == [1.0, 0.0, 1.0]

    def test_observed_mask_separates_silence_from_inactivity(self) -> None:
        stream = ObservationStream.from_observations([self.motion(0), self.motion(120)])
        mask = stream.observed_mask("1h")
        assert list(mask["kitchen_motion"]) == [True, False, True]

    def test_sample_frame_leaves_unsampled_bins_missing(self) -> None:
        samples = [
            make_observation(
                timestamp=T0 + timedelta(hours=hour),
                sensor_id="hall_temp",
                modality=Modality.ENVIRONMENTAL,
                kind=ObservationKind.SAMPLE,
                value=20.0 + hour,
                unit=Unit.CELSIUS,
            )
            for hour in (0, 2)
        ]
        frame = ObservationStream.from_observations(samples).sample_frame("1h")
        assert frame["hall_temp"].iloc[0] == pytest.approx(20.0)
        assert math.isnan(frame["hall_temp"].iloc[1])
        assert frame["hall_temp"].iloc[2] == pytest.approx(22.0)

    def test_state_frame_holds_a_state_only_for_max_hold(self) -> None:
        states = [
            make_observation(
                timestamp=T0,
                sensor_id="bed",
                modality=Modality.BED_PRESSURE,
                kind=ObservationKind.STATE,
                value=1.0,
            ),
            make_observation(
                timestamp=T0 + timedelta(hours=5),
                sensor_id="bed",
                modality=Modality.BED_PRESSURE,
                kind=ObservationKind.STATE,
                value=0.0,
            ),
        ]
        frame = ObservationStream.from_observations(states).state_frame(
            "1h", max_hold=timedelta(hours=2)
        )
        column = frame["bed"]
        assert column.iloc[0] == 1.0
        assert column.iloc[2] == 1.0
        assert math.isnan(column.iloc[4])
        assert column.iloc[5] == 0.0

    def test_framing_is_complete_across_a_dst_fall_back(self) -> None:
        start = datetime(2024, 10, 27, 0, 30, tzinfo=timezone.utc)
        events = [
            make_observation(
                timestamp=(start + timedelta(minutes=30 * i)).astimezone(LISBON)
            )
            for i in range(8)
        ]
        counts = ObservationStream.from_observations(events).event_counts("1h")
        assert counts.values.sum() == pytest.approx(len(events))

    def test_framing_is_complete_across_a_dst_spring_forward(self) -> None:
        start = datetime(2024, 3, 31, 0, 30, tzinfo=timezone.utc)
        events = [
            make_observation(
                timestamp=(start + timedelta(minutes=15 * i)).astimezone(LISBON)
            )
            for i in range(8)
        ]
        counts = ObservationStream.from_observations(events).event_counts("1h")
        assert counts.values.sum() == pytest.approx(len(events))

    def test_empty_stream_frames_are_empty_not_broken(self) -> None:
        stream = ObservationStream()
        assert stream.start is None and stream.end is None
        assert stream.event_counts("1h").empty
        assert stream.observed_mask("1h").empty
        assert stream.state_frame("1h").empty
