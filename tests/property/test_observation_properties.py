"""Property-based tests for the canonical observation model.

These assert invariants that must hold for *every* valid observation and
stream, rather than for the handful of cases an example-based test happens to
pick. They are the tests most likely to find an edge case nobody thought of.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    ObservationStream,
    Unit,
    convert,
    to_canonical,
)

EPOCH = datetime(2024, 5, 1, tzinfo=timezone.utc)

# Offsets rather than datetimes, and a small sensor vocabulary: the invariants
# under test do not depend on the richness of either, and cheap strategies keep
# the suite fast enough to run on every commit.
OFFSETS = st.integers(min_value=0, max_value=60 * 24 * 30)
SENSOR_IDS = st.sampled_from(["a", "b", "c", "kitchen_motion", "bed"])
UNIT_INTERVAL = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
VALUES = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)

# The strategies below are deliberately cheap -- sampled_from, integers and
# bounded floats. Hypothesis nonetheless attributes one-off warm-up cost to
# whichever example is drawn first, which trips the too_slow health check at
# random depending on test order. Suppressing it here is a statement about
# that measurement artefact, not a licence for an expensive strategy.
FAST = settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def observations(draw: st.DrawFn) -> Observation:
    """Generate an arbitrary valid observation."""
    return Observation(
        timestamp=EPOCH + timedelta(minutes=draw(OFFSETS)),
        sensor_id=draw(SENSOR_IDS),
        modality=draw(st.sampled_from(list(Modality))),
        kind=draw(st.sampled_from(list(ObservationKind))),
        value=draw(VALUES),
        quality=draw(UNIT_INTERVAL),
        confidence=draw(UNIT_INTERVAL),
    )


@given(observations())
@FAST
def test_every_observation_round_trips_through_its_dict(obs: Observation) -> None:
    """Serialisation must be lossless, or a snapshot silently corrupts state."""
    assert Observation.from_dict(obs.to_dict()) == obs


@given(observations())
@FAST
def test_evidence_weight_stays_in_the_unit_interval(obs: Observation) -> None:
    assert 0.0 <= obs.evidence_weight() <= 1.0


@given(observations())
@FAST
def test_every_observation_is_hashable(obs: Observation) -> None:
    assert isinstance(hash(obs), int)


@given(observations())
@FAST
def test_timestamps_are_always_timezone_aware(obs: Observation) -> None:
    assert obs.timestamp.tzinfo is not None
    assert obs.timestamp.tzinfo.utcoffset(obs.timestamp) is not None


@given(
    st.floats(min_value=-200.0, max_value=200.0, allow_nan=False),
    st.sampled_from([Unit.CELSIUS, Unit.FAHRENHEIT]),
)
def test_temperature_conversion_is_reversible(value: float, unit: Unit) -> None:
    other = Unit.FAHRENHEIT if unit is Unit.CELSIUS else Unit.CELSIUS
    assert abs(convert(convert(value, unit, other), other, unit) - value) < 1e-6


@given(
    st.floats(min_value=0.0, max_value=1e5, allow_nan=False),
    st.sampled_from(
        [Unit.CENTIMETRE, Unit.CENTIMETRE_PER_SECOND, Unit.MILLI_G, Unit.MINUTE]
    ),
)
def test_canonicalisation_preserves_magnitude_order(value: float, unit: Unit) -> None:
    """Converting to canonical units must not reorder two readings."""
    smaller, _ = to_canonical(value, unit)
    larger, _ = to_canonical(value + 1.0, unit)
    assert smaller <= larger


@given(st.lists(observations(), min_size=1, max_size=40))
@FAST
def test_a_stream_is_always_sorted_regardless_of_arrival_order(
    records: list[Observation],
) -> None:
    stream = ObservationStream.from_observations(records)
    stamps = [obs.timestamp.astimezone(timezone.utc) for obs in stream]
    assert stamps == sorted(stamps)


@given(st.lists(observations(), min_size=1, max_size=40))
@FAST
def test_a_stream_never_holds_duplicate_identities(
    records: list[Observation],
) -> None:
    stream = ObservationStream.from_observations(records)
    identities = [obs.identity() for obs in stream]
    assert len(identities) == len(set(identities))


@given(st.lists(observations(), min_size=1, max_size=30))
@FAST
def test_re_adding_every_record_changes_nothing(
    records: list[Observation],
) -> None:
    """Replaying a batch must be idempotent, since gateways redeliver."""
    stream = ObservationStream.from_observations(records)
    before = len(stream)
    stream.extend(records)
    assert len(stream) == before


@given(st.lists(observations(), min_size=2, max_size=30))
@FAST
def test_event_counts_never_exceed_the_events_supplied(
    records: list[Observation],
) -> None:
    """Framing may drop nothing into existence."""
    events = [
        obs
        for obs in ObservationStream.from_observations(records)
        if obs.kind is ObservationKind.EVENT and obs.value != 0.0
    ]
    assume(events)
    stream = ObservationStream.from_observations(records)
    counts = stream.event_counts("1h")
    assert counts.values.sum() <= len(events)


@given(st.lists(observations(), min_size=1, max_size=30))
@FAST
def test_the_observed_mask_is_never_true_where_nothing_arrived(
    records: list[Observation],
) -> None:
    stream = ObservationStream.from_observations(records)
    mask = stream.observed_mask("1h")
    assert mask.values.sum() <= len(stream)


@given(
    st.integers(min_value=1, max_value=48),
    st.integers(min_value=1, max_value=6),
)
def test_gaps_are_reported_only_between_recorded_observations(
    spacing_minutes: int, count: int
) -> None:
    start = datetime(2024, 5, 1, tzinfo=timezone.utc)
    records = [
        Observation(
            timestamp=start + timedelta(minutes=spacing_minutes * index),
            sensor_id="s",
            modality=Modality.MOTION,
            kind=ObservationKind.EVENT,
            value=1.0,
        )
        for index in range(count)
    ]
    stream = ObservationStream.from_observations(records)
    gaps = stream.gaps("s", timedelta(minutes=1))
    assert all(stream.start <= gap.start and gap.end <= stream.end for gap in gaps)
    assert len(gaps) <= max(count - 1, 0)
