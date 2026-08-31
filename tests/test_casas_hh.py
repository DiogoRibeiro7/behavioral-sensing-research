"""Tests for the CASAS ``hh`` CSV export.

This export differs from the classic format in ways that silently break a reader
written for the other one: comma separation, a location where the sensor
identifier used to be, quoted ``begin``/``end`` markers, and a different
activity vocabulary. It also mixes two annotation styles in the same column,
which is the defect these tests exist to prevent recurring.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sensor_modeling.datasets import (
    HH_ACTIVITY_STATES,
    CasasReadError,
    hh_sensor_specs,
    read_casas_hh,
)
from sensor_modeling.datasets.casas import truth_series
from sensor_modeling.datasets.casas_hh import HH_LOCATIONS, normalise_location
from sensor_modeling.observations import Modality
from sensor_modeling.states import BehaviouralState

UTC = timezone.utc

LINES = [
    '2011-06-15,01:00:00.000000,Bedroom,ON,Sleep="begin"',
    "2011-06-15,01:00:01.000000,Bedroom,OFF",
    "2011-06-15,03:00:00.000000,Bedroom,ON",
    '2011-06-15,07:00:00.000000,Bedroom,OFF,Sleep="end"',
    '2011-06-15,08:00:00.000000,Kitchen,ON,Cook_Breakfast="begin"',
    '2011-06-15,08:30:00.000000,Kitchen,OFF,Cook_Breakfast="end"',
    "2011-06-15,09:00:00.000000,OutsideDoor,OPEN",
    "2011-06-15,09:00:05.000000,OutsideDoor,CLOSE",
    "2011-06-15,10:00:00.000000,Garage,ON",
]


def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2011, 6, 15, hour, minute, tzinfo=UTC)


class TestFormat:
    def test_comma_separated_rows_are_read(self) -> None:
        recording = read_casas_hh(LINES, timezone=UTC)
        assert recording.observations

    def test_a_location_becomes_a_sensor_carrying_its_room(self) -> None:
        """The export has already aggregated each room's sensors into one stream.

        The room is what the occupancy and fusion layers need, and here it
        arrives in the data instead of having to be read off a floor plan.
        """
        specs, unmapped = hh_sensor_specs(["Kitchen", "Bedroom", "OutsideDoor"])
        by_id = {spec.sensor_id: spec for spec in specs}

        assert not unmapped
        assert by_id["Kitchen"].room == "kitchen"
        assert by_id["Kitchen"].modality is Modality.MOTION
        assert by_id["OutsideDoor"].modality is Modality.DOOR
        assert by_id["OutsideDoor"].room == "hall"

    def test_an_unknown_location_is_excluded_and_reported(self) -> None:
        recording = read_casas_hh(LINES, timezone=UTC)

        assert "Garage" in recording.unmapped_sensors
        assert all(o.sensor_id != "Garage" for o in recording.observations)

    def test_closing_readings_are_counted_not_emitted(self) -> None:
        recording = read_casas_hh(LINES, timezone=UTC)

        # Three OFF and one CLOSE. The unmapped Garage row is excluded before
        # the value is examined, so it contributes to neither count.
        assert recording.deactivations == 4

    def test_timestamps_take_the_supplied_zone(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        recording = read_casas_hh(LINES, timezone=eastern)

        assert recording.observations[0].timestamp.utcoffset() == timedelta(hours=-5)


class TestAnnotationStyles:
    def test_quoted_begin_and_end_markers_form_intervals(self) -> None:
        recording = read_casas_hh(LINES, timezone=UTC)
        sleep = [a for a in recording.activities if a.label == "Sleep"]

        assert len(sleep) == 1
        assert sleep[0].state is BehaviouralState.SLEEPING
        assert sleep[0].start == at(1)
        assert sleep[0].end == at(7)

    def test_a_bare_label_annotates_events_rather_than_opening_an_interval(
        self,
    ) -> None:
        """The fifth column carries two different annotation styles.

        Real recordings contain rows like ``WorkArea,ON,Work_At_Table`` with no
        ``begin``/``end``. Treating those as unparsable discarded hundreds of
        genuinely annotated events; treating them as interval markers would
        leave an activity open forever. They label the individual event, so
        consecutive events sharing a label are coalesced into the span they
        cover.
        """
        lines = [
            "2011-06-15,14:00:00.000000,WorkArea,ON,Work_At_Table",
            "2011-06-15,14:10:00.000000,WorkArea,ON,Work_At_Table",
            "2011-06-15,14:20:00.000000,WorkArea,ON,Work_At_Table",
        ]
        recording = read_casas_hh(lines, timezone=UTC)

        assert recording.unparsed_lines == 0
        assert len(recording.activities) == 1
        interval = recording.activities[0]
        assert interval.label == "Work_At_Table"
        assert interval.start == at(14, 0)
        assert interval.end == at(14, 20)
        assert interval.state is BehaviouralState.HOME_ACTIVE

    def test_a_change_of_bare_label_closes_the_previous_run(self) -> None:
        lines = [
            "2011-06-15,14:00:00.000000,WorkArea,ON,Work_At_Table",
            "2011-06-15,14:10:00.000000,WorkArea,ON,Work_At_Table",
            "2011-06-15,15:00:00.000000,Kitchen,ON,Cook_Lunch",
            "2011-06-15,15:20:00.000000,Kitchen,ON,Cook_Lunch",
        ]
        recording = read_casas_hh(lines, timezone=UTC)

        labels = sorted(a.label for a in recording.activities)
        assert labels == ["Cook_Lunch", "Work_At_Table"]

    def test_a_single_event_label_spans_no_time_and_is_dropped(self) -> None:
        """One instant is not an interval, and would score nothing."""
        lines = ["2011-06-15,14:00:00.000000,WorkArea,ON,Work_At_Table"]
        assert read_casas_hh(lines, timezone=UTC).activities == ()


class TestVocabulary:
    def test_the_hh_vocabulary_is_not_the_classic_one(self) -> None:
        """A mapping written for one export is nearly useless on the other.

        The classic files label ``Sleeping`` and ``Meal_Preparation``; this one
        labels ``Sleep`` and ``Cook_Breakfast``. Sharing a single mapping would
        silently discard almost every annotation.
        """
        from sensor_modeling.datasets import CASAS_ACTIVITY_STATES

        overlap = set(HH_ACTIVITY_STATES) & set(CASAS_ACTIVITY_STATES)

        assert "Sleep" in HH_ACTIVITY_STATES
        assert "Sleeping" not in HH_ACTIVITY_STATES
        # A handful of labels coincide, but none of the frequent ones do: sleep,
        # cooking and washing are all spelled differently.
        assert overlap == {
            "Enter_Home",
            "Housekeeping",
            "Leave_Home",
            "Relax",
            "Wash_Dishes",
            "Work",
        }
        assert len(overlap) < len(HH_ACTIVITY_STATES) / 4

    def test_other_activity_is_deliberately_unmapped(self) -> None:
        """It is the annotator declining to categorise, not a state."""
        assert "Other_Activity" not in HH_ACTIVITY_STATES

        lines = [
            '2011-06-15,14:00:00.000000,Kitchen,ON,Other_Activity="begin"',
            '2011-06-15,14:30:00.000000,Kitchen,ON,Other_Activity="end"',
        ]
        recording = read_casas_hh(lines, timezone=UTC)

        assert recording.unmapped_activities == {"Other_Activity": 1}
        assert truth_series(recording.activities, [at(14, 10)]) == [None]

    def test_wake_up_maps_to_bed_awake(self) -> None:
        """The ontology has a state for exactly this, so use it."""
        assert HH_ACTIVITY_STATES["Wake_Up"] is BehaviouralState.BED_AWAKE
        assert HH_ACTIVITY_STATES["Go_To_Sleep"] is BehaviouralState.BED_AWAKE

    def test_eating_is_not_treated_as_kitchen_presence(self) -> None:
        """Meals are often eaten elsewhere; assuming otherwise fakes agreement."""
        for label in ("Eat_Breakfast", "Eat_Lunch", "Eat_Dinner"):
            assert HH_ACTIVITY_STATES[label] is BehaviouralState.HOME_ACTIVE


class TestRobustness:
    def test_nothing_readable_is_an_error(self) -> None:
        with pytest.raises(CasasReadError, match="no readable sensor reports"):
            read_casas_hh(["junk", "more junk"], timezone=UTC)

    def test_a_missing_file_is_an_error(self) -> None:
        with pytest.raises(CasasReadError, match="no CASAS recording"):
            read_casas_hh("nowhere/at/all.csv", timezone=UTC)

    def test_rows_with_too_few_fields_are_counted(self) -> None:
        recording = read_casas_hh([*LINES, "2011-06-15,01:00:00"], timezone=UTC)
        assert recording.unparsed_lines == 1


class TestDerivedAwayIntervals:
    """`Leave_Home` marks crossing the threshold, not the time spent out.

    Its median duration in real recordings is about twelve seconds. Scoring it
    as AWAY penalises an inference that correctly reports activity, because the
    resident is inside and walking to the door. The hours actually spent out
    carry no label at all, so they have to be derived from the gap.
    """

    LINES = [
        '2011-06-15,09:00:00.000000,OutsideDoor,OPEN,Leave_Home="begin"',
        '2011-06-15,09:00:12.000000,OutsideDoor,CLOSE,Leave_Home="end"',
        "2011-06-15,11:00:00.000000,Kitchen,ON",
        '2011-06-15,14:00:00.000000,OutsideDoor,OPEN,Enter_Home="begin"',
        '2011-06-15,14:00:12.000000,OutsideDoor,CLOSE,Enter_Home="end"',
    ]

    def test_the_gap_between_leaving_and_returning_becomes_away(self) -> None:
        recording = read_casas_hh(self.LINES, timezone=UTC)
        away = [a for a in recording.activities if a.state is BehaviouralState.AWAY]

        assert len(away) == 1
        assert away[0].start == at(9, 0) + timedelta(seconds=12)
        assert away[0].end == at(14, 0)

    def test_leaving_itself_is_not_away(self) -> None:
        """Twelve seconds of walking to the door is activity in the home."""
        recording = read_casas_hh(self.LINES, timezone=UTC)
        leaving = [a for a in recording.activities if a.label == "Leave_Home"]

        assert leaving and leaving[0].state is BehaviouralState.HOME_ACTIVE

    def test_the_derivation_can_be_turned_off(self) -> None:
        recording = read_casas_hh(self.LINES, timezone=UTC, derive_away=False)

        assert not [a for a in recording.activities if a.state is BehaviouralState.AWAY]

    def test_a_departure_with_no_return_yields_no_interval(self) -> None:
        """An open-ended absence has no end to score against."""
        lines = self.LINES[:2]
        recording = read_casas_hh(lines, timezone=UTC)

        assert not [a for a in recording.activities if a.state is BehaviouralState.AWAY]

    def test_deriving_away_increases_the_labelled_span(self) -> None:
        """The point of the derivation is that this time was previously unscored."""
        with_away = read_casas_hh(self.LINES, timezone=UTC).labelled_fraction
        without = read_casas_hh(
            self.LINES, timezone=UTC, derive_away=False
        ).labelled_fraction

        assert with_away > without


class TestExtendedLocationVocabulary:
    """Later CASAS releases name sensors `<Room><Instance><Fixture>`.

    A reader handling only the flat vocabulary silently produces no
    observations for those homes, which looks like an ineligible recording
    rather than a parser gap.
    """

    def test_a_fixture_sensor_resolves_to_its_room(self) -> None:
        assert normalise_location("BedroomABed") == ("bedroom", Modality.MOTION)
        assert normalise_location("KitchenARefrigerator") == (
            "kitchen",
            Modality.MOTION,
        )
        assert normalise_location("BathroomAToilet") == ("bathroom", Modality.MOTION)

    def test_a_bed_sensor_is_motion_not_presence(self) -> None:
        """The dataset uses PIR and magnetic door sensors only.

        A sensor aimed at a bed detects movement at the bed. Mapping it to a
        presence modality would invent a measurement the deployment does not
        make, which the validation contract forbids.
        """
        room, modality = normalise_location("BedroomABed")

        assert modality is Modality.MOTION
        assert modality is not Modality.BED_PRESSURE

    def test_the_main_door_is_the_only_door(self) -> None:
        """`BedroomADoor` is a motion sensor by the door, not a contact."""
        assert normalise_location("MainDoor") == ("hall", Modality.DOOR)
        assert normalise_location("BedroomADoor") == ("bedroom", Modality.MOTION)

    def test_temperature_is_a_sample_not_an_activation(self) -> None:
        room, modality = normalise_location("BathroomATemperature")
        assert modality is Modality.ENVIRONMENTAL

        lines = [
            "2015-10-04,00:09:26.771558,BathroomATemperature,20.0",
            "2015-10-04,00:19:26.771558,BathroomATemperature,21.5",
        ]
        recording = read_casas_hh(lines, timezone=UTC)
        values = [o.value for o in recording.observations]
        assert values == [pytest.approx(20.0), pytest.approx(21.5)]

    def test_an_unrecognised_room_is_still_refused(self) -> None:
        """The normaliser must not become a way of admitting anything."""
        assert normalise_location("Garage") is None
        assert normalise_location("Nonsense") is None

    def test_the_instance_letter_is_stripped(self) -> None:
        assert normalise_location("HallwayA") == ("hall", Modality.MOTION)
        assert normalise_location("BedroomBArea") == ("bedroom", Modality.MOTION)

    def test_the_flat_vocabulary_still_wins(self) -> None:
        """Existing names keep their explicit mapping rather than being parsed."""
        assert normalise_location("OutsideDoor") == HH_LOCATIONS["OutsideDoor"]
        assert normalise_location("LoungeChair") == HH_LOCATIONS["LoungeChair"]
