"""Tests for reading CASAS recordings into the canonical model.

The adapter's job is to lose nothing quietly. Most of these tests are about what
it refuses to do: guess a timezone, invent a label, or let a sensor it does not
understand contribute evidence anyway.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasReadError,
    casas_sensor_specs,
    evaluate_recording,
    read_casas,
    truth_series,
)
from sensor_modeling.observations import Modality, ObservationKind
from sensor_modeling.states import BehaviouralState

UTC = timezone.utc
LISBON = timezone(timedelta(hours=1))

LINES = [
    "2010-11-04 00:03:50.209589\tM003\tON",
    "2010-11-04 00:03:57.399391\tM003\tOFF",
    "2010-11-04 00:15:08.984841\tT002\t21.5",
    "2010-11-04 01:00:00.000000\tM004\tON\tSleeping\tbegin",
    "2010-11-04 05:40:51.303739\tM004\tON\tSleeping\tend",
    "2010-11-04 06:00:00.000000\tL001\tON",
    "2010-11-04 07:00:00.000000\tM002\tON\tRespirate\tbegin",
    "2010-11-04 07:10:00.000000\tM002\tON\tRespirate\tend",
]


def at(hour: int, minute: int = 0, tz: timezone = UTC) -> datetime:
    return datetime(2010, 11, 4, hour, minute, tzinfo=tz)


class TestTimestamps:
    def test_timestamps_are_placed_in_the_stated_zone(self) -> None:
        """The zone is the caller's to supply, and it must actually be applied.

        CASAS records naive local wall-clock. Reading it as UTC would displace
        every event, which is invisible in aggregate counts and fatal to any
        analysis of daily rhythm.
        """
        recording = read_casas(LINES, timezone=LISBON)

        first = recording.observations[0]
        assert first.timestamp.utcoffset() == timedelta(hours=1)
        assert first.timestamp.hour == 0

    def test_the_same_file_in_two_zones_gives_different_instants(self) -> None:
        utc = read_casas(LINES, timezone=UTC).observations[0].timestamp
        lisbon = read_casas(LINES, timezone=LISBON).observations[0].timestamp

        assert utc != lisbon

    def test_observations_come_back_in_time_order(self) -> None:
        stamps = [o.timestamp for o in read_casas(LINES, timezone=UTC).observations]
        assert stamps == sorted(stamps)


class TestSensorMapping:
    def test_a_sensor_kind_it_cannot_interpret_is_excluded_and_reported(self) -> None:
        """An unrecognised sensor must not become evidence by default.

        Mapping every unknown prefix to ``OTHER`` would let a light switch or a
        power meter push the posterior around under a modality nobody had
        reasoned about. Excluding it silently would be just as bad, so it is
        excluded and named.
        """
        recording = read_casas(LINES, timezone=UTC)

        assert "L001" in recording.unmapped_sensors
        assert all(o.sensor_id != "L001" for o in recording.observations)

    def test_prefixes_map_to_the_expected_modality_and_kind(self) -> None:
        specs, _ = casas_sensor_specs(["M003", "D001", "T002"])
        by_id = {spec.sensor_id: spec for spec in specs}

        assert by_id["M003"].modality is Modality.MOTION
        assert by_id["M003"].kind is ObservationKind.EVENT
        assert by_id["D001"].modality is Modality.DOOR
        assert by_id["T002"].modality is Modality.ENVIRONMENTAL
        assert by_id["T002"].kind is ObservationKind.SAMPLE

    def test_rooms_are_left_unset_unless_supplied(self) -> None:
        """Room drives occupancy, and the raw files do not carry it."""
        specs, _ = casas_sensor_specs(["M003"])
        assert specs[0].room is None

        specs, _ = casas_sensor_specs(["M003"], rooms={"M003": "kitchen"})
        assert specs[0].room == "kitchen"


class TestEventSemantics:
    def test_closing_readings_are_counted_rather_than_emitted(self) -> None:
        """Emitting OFF as well as ON would double every event rate.

        The pipeline reads motion and door sensors as event-kind, where the rate
        carries the information. Admitting both halves of each pair would inflate
        every Poisson rate by a factor of two.
        """
        recording = read_casas(LINES, timezone=UTC)

        assert recording.deactivations == 1
        assert all(
            o.value == 1.0 for o in recording.observations if o.sensor_id == "M003"
        )
        assert sum(1 for o in recording.observations if o.sensor_id == "M003") == 1

    def test_sample_sensors_keep_their_numeric_value(self) -> None:
        recording = read_casas(LINES, timezone=UTC)
        temperature = [o for o in recording.observations if o.sensor_id == "T002"]

        assert len(temperature) == 1
        assert temperature[0].value == pytest.approx(21.5)


class TestAnnotations:
    def test_a_label_outside_the_ontology_is_reported_not_coerced(self) -> None:
        """An unmapped label must not silently become UNKNOWN.

        UNKNOWN is a claim the ontology makes about abstention. Reusing it for
        "this dataset had a word we do not model" would confuse a scoring
        convention with a gap in the mapping.
        """
        recording = read_casas(LINES, timezone=UTC)

        assert recording.unmapped_activities == {"Respirate": 1}
        respirate = [a for a in recording.activities if a.label == "Respirate"]
        assert respirate and respirate[0].state is None

    def test_a_mapped_label_becomes_an_interval_with_a_state(self) -> None:
        recording = read_casas(LINES, timezone=UTC)
        sleeping = [a for a in recording.activities if a.label == "Sleeping"]

        assert len(sleeping) == 1
        assert sleeping[0].state is BehaviouralState.SLEEPING
        assert sleeping[0].start == at(1)

    def test_an_end_without_a_begin_is_dropped(self) -> None:
        """A dangling end marker has no duration and cannot be scored."""
        recording = read_casas(
            ["2010-11-04 05:00:00.0\tM004\tON\tSleeping\tend"], timezone=UTC
        )
        assert recording.activities == ()

    def test_an_interval_cannot_end_before_it_starts(self) -> None:
        with pytest.raises(CasasReadError, match="before it starts"):
            ActivityInterval(label="Sleeping", start=at(5), end=at(1), state=None)

    def test_the_activity_map_can_be_overridden(self) -> None:
        recording = read_casas(
            LINES,
            timezone=UTC,
            activity_states={"Respirate": BehaviouralState.HOME_INACTIVE},
        )

        assert "Respirate" not in recording.unmapped_activities
        # Sleeping is no longer mapped under the replacement vocabulary.
        assert "Sleeping" in recording.unmapped_activities


class TestTruthSeries:
    def test_unlabelled_time_is_none_rather_than_a_guess(self) -> None:
        """Half a real recording is unannotated, and must not be scored.

        ``state_metrics`` skips ``None``. Filling the gaps with a plausible
        state would invent agreement or disagreement that no annotator recorded.
        """
        recording = read_casas(LINES, timezone=UTC)
        series = truth_series(recording.activities, [at(0), at(3), at(9)])

        assert series == [None, BehaviouralState.SLEEPING, None]

    def test_an_unmapped_interval_yields_none(self) -> None:
        recording = read_casas(LINES, timezone=UTC)
        assert truth_series(recording.activities, [at(7, 5)]) == [None]

    def test_intervals_are_end_exclusive(self) -> None:
        interval = ActivityInterval(
            label="Sleeping", start=at(1), end=at(5), state=BehaviouralState.SLEEPING
        )
        assert interval.contains(at(1))
        assert not interval.contains(at(5))


class TestRobustness:
    def test_unreadable_lines_are_counted(self) -> None:
        recording = read_casas([*LINES, "not a sensor report"], timezone=UTC)
        assert recording.unparsed_lines == 1

    def test_a_file_with_nothing_readable_is_an_error(self) -> None:
        with pytest.raises(CasasReadError, match="no readable sensor reports"):
            read_casas(["nonsense", "more nonsense"], timezone=UTC)

    def test_a_missing_file_is_an_error(self) -> None:
        with pytest.raises(CasasReadError, match="no CASAS recording"):
            read_casas("does/not/exist.txt", timezone=UTC)

    def test_seconds_without_fractions_are_accepted(self) -> None:
        recording = read_casas(["2010-11-04 00:03:50\tM003\tON"], timezone=UTC)
        assert len(recording.observations) == 1

    def test_labelled_fraction_reports_how_much_was_scored(self) -> None:
        """Accuracy over a recording means little without knowing the coverage."""
        recording = read_casas(LINES, timezone=UTC)

        assert 0.0 < recording.labelled_fraction < 1.0

    def test_the_summary_names_everything_discarded(self) -> None:
        summary = read_casas(LINES, timezone=UTC).summary()

        assert summary["unmapped_sensors"] == ["L001"]
        assert summary["unmapped_activities"] == {"Respirate": 1}
        assert summary["deactivations_ignored"] == 1


def _realistic_day() -> list[str]:
    """A day in CASAS format with plausible event density per activity.

    Density is what distinguishes the states: a sleeping resident triggers
    bedroom motion a handful of times a night, a cooking one triggers kitchen
    motion every couple of minutes. A fixture with uniform density would make
    the states indistinguishable and the test meaningless.
    """
    import random

    rng = random.Random(11)
    start = datetime(2010, 11, 4, 0, 0)
    lines: list[str] = []

    def stamp(moment: datetime) -> str:
        return moment.strftime("%Y-%m-%d %H:%M:%S.%f")

    def burst(sensor: str, begin: datetime, end: datetime, lo: int, hi: int) -> None:
        moment = begin
        while moment < end:
            moment += timedelta(minutes=rng.randint(lo, hi))
            if moment < end:
                lines.append(f"{stamp(moment)}\t{sensor}\tON")

    seven = start + timedelta(hours=7)
    lines.append(f"{stamp(start)}\tM004\tON\tSleeping\tbegin")
    burst("M004", start, seven, 40, 110)
    lines.append(f"{stamp(seven)}\tM004\tON\tSleeping\tend")

    kitchen_from = start + timedelta(hours=7, minutes=5)
    kitchen_to = start + timedelta(hours=7, minutes=45)
    lines.append(f"{stamp(kitchen_from)}\tM001\tON\tMeal_Preparation\tbegin")
    burst("M001", kitchen_from, kitchen_to, 1, 3)
    lines.append(f"{stamp(kitchen_to)}\tM001\tON\tMeal_Preparation\tend")

    relax_from = start + timedelta(hours=8)
    relax_to = start + timedelta(hours=11)
    lines.append(f"{stamp(relax_from)}\tM002\tON\tRelax\tbegin")
    burst("M002", relax_from, relax_to, 8, 20)
    lines.append(f"{stamp(relax_to)}\tM002\tON\tRelax\tend")

    lines.sort()
    return lines


ROOMS = {"M004": "bedroom", "M001": "kitchen", "M002": "living", "M003": "bathroom"}


class TestPipelineIntegration:
    """The adapter feeding the unmodified pipeline.

    These tests establish that a CASAS-shaped recording reaches the inference
    layers intact and is scored correctly. They are **not** real-data
    validation: the fixture is synthesised in CASAS format by this repository,
    so it says nothing about how the approach performs on an actual apartment.
    That requires a downloaded recording, which is not redistributed here.
    """

    def test_a_recording_runs_through_the_pipeline_and_is_scored(self) -> None:
        recording = read_casas(_realistic_day(), timezone=UTC, rooms=ROOMS)
        result = evaluate_recording(recording, step=timedelta(minutes=10))

        assert result.steps > 0
        assert 0 < result.scored <= result.steps
        assert 0.0 <= result.metrics.balanced_accuracy <= 1.0

    def test_declared_defaults_separate_the_states_without_tuning(self) -> None:
        """Nothing is refitted to the dataset, so this measures transfer.

        If the emission defaults only worked on the simulator that produced
        them, a differently shaped recording would score near chance.
        """
        recording = read_casas(_realistic_day(), timezone=UTC, rooms=ROOMS)
        result = evaluate_recording(recording, step=timedelta(minutes=10))

        assert result.metrics.balanced_accuracy > 0.5

    def test_coverage_is_reported_beside_the_score(self) -> None:
        """A score without its coverage invites reading it as the whole day."""
        recording = read_casas(_realistic_day(), timezone=UTC, rooms=ROOMS)
        payload = evaluate_recording(recording, step=timedelta(minutes=10)).to_dict()

        assert "scored_fraction" in payload
        assert "labelled_fraction" in payload
        assert payload["scored"] < payload["steps"]

    def test_a_recording_with_no_mapped_label_refuses_to_score(self) -> None:
        """Metrics over an empty sample would look like a result."""
        lines = [
            "2010-11-04 00:00:00.0\tM001\tON\tRespirate\tbegin",
            "2010-11-04 00:30:00.0\tM001\tON",
            "2010-11-04 01:00:00.0\tM001\tON\tRespirate\tend",
        ]
        recording = read_casas(lines, timezone=UTC, rooms=ROOMS)

        with pytest.raises(ValueError, match="nothing can be scored"):
            evaluate_recording(recording, step=timedelta(minutes=10))
