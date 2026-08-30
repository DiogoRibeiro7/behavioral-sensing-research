"""Regression tests for the parameter-free v0.3 inference primitives."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sensor_modeling.fusion.history import EventEvidenceWindow
from sensor_modeling.observations import Modality, Observation, ObservationKind
from sensor_modeling.states.circadian import CircadianBand, CircadianSchedule
from sensor_modeling.states.ontology import BehaviouralState, StateOntology

S = BehaviouralState
T0 = datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)


def event(at: datetime, sensor_id: str = "motion") -> Observation:
    return Observation(at, sensor_id, Modality.MOTION, ObservationKind.EVENT, 1.0)


def test_circadian_generator_preserves_declared_exit_rates() -> None:
    ontology = StateOntology()
    schedule = CircadianSchedule(
        [
            CircadianBand(0, {S.SLEEPING: 5.0, S.HOME_ACTIVE: 0.5}),
            CircadianBand(8, {S.SLEEPING: 0.2, S.HOME_ACTIVE: 3.0}),
        ]
    )
    generator = schedule.generator(ontology, T0)
    expected = np.array(
        [-1.0 / ontology.dwell[state].total_seconds() for state in ontology.states]
    )
    assert np.allclose(np.diag(generator), expected)
    assert np.allclose(generator.sum(axis=1), 0.0)


def test_circadian_destination_weights_change_transition_preference() -> None:
    ontology = StateOntology()
    schedule = CircadianSchedule(
        [
            CircadianBand(0, {S.BED_AWAKE: 8.0, S.HOME_INACTIVE: 0.2}),
            CircadianBand(12, {S.BED_AWAKE: 0.2, S.HOME_INACTIVE: 8.0}),
        ]
    )
    night = schedule.transition(ontology, T0, timedelta(minutes=10))
    noon = schedule.transition(
        ontology, T0.replace(hour=12), timedelta(minutes=10)
    )
    row = ontology.index(S.HOME_ACTIVE)
    bed = ontology.index(S.BED_AWAKE)
    inactive = ontology.index(S.HOME_INACTIVE)
    assert night[row, bed] > noon[row, bed]
    assert noon[row, inactive] > night[row, inactive]


def test_circadian_transition_is_row_stochastic() -> None:
    ontology = StateOntology()
    schedule = CircadianSchedule([CircadianBand(0, {S.SLEEPING: 2.0})])
    matrix = schedule.transition(ontology, T0, timedelta(minutes=15))
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert matrix.min() >= 0.0


def test_circadian_schedule_wraps_before_first_start_hour() -> None:
    schedule = CircadianSchedule([CircadianBand(6), CircadianBand(18)])
    assert schedule.band_at(T0).start_hour == 18


def test_invalid_circadian_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CircadianBand(0, {S.SLEEPING: -1.0})


def test_event_window_releases_each_event_once() -> None:
    window = EventEvidenceWindow(timedelta(minutes=30))
    first = event(T0)
    second = event(T0 + timedelta(minutes=10))
    window.add(first)
    window.add(second)
    assert window.release(T0 + timedelta(minutes=20)) == ()
    released = window.release(T0 + timedelta(minutes=30))
    assert released == (first, second)
    assert window.pending == ()
    assert window.release(T0 + timedelta(hours=1)) == ()


def test_event_window_rejects_non_event_observations() -> None:
    window = EventEvidenceWindow(timedelta(minutes=30))
    sample = Observation(
        T0,
        "wearable",
        Modality.WEARABLE_MOTION,
        ObservationKind.SAMPLE,
        0.5,
    )
    with pytest.raises(ValueError, match="event observations only"):
        window.add(sample)


def test_event_window_snapshot_round_trip_preserves_pending_evidence() -> None:
    window = EventEvidenceWindow(timedelta(minutes=30))
    window.add(event(T0))
    window.add(event(T0 + timedelta(minutes=5), "door"))
    restored = EventEvidenceWindow.from_snapshot(window.snapshot())
    assert restored.width == window.width
    assert restored.opened_at == window.opened_at
    assert [item.to_dict() for item in restored.pending] == [
        item.to_dict() for item in window.pending
    ]


def test_event_window_requires_positive_width() -> None:
    with pytest.raises(ValueError, match="positive"):
        EventEvidenceWindow(timedelta(0))
