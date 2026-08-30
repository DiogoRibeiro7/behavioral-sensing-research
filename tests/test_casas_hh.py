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
