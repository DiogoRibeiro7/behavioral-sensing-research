"""Tests for fitting circadian profiles from development-only annotations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    fit_circadian_profile,
)
from sensor_modeling.observations import Modality, SensorRegistry, SensorSpec
from sensor_modeling.states import BehaviouralState as S

UTC = timezone.utc


def recording(*intervals: ActivityInterval) -> CasasRecording:
    """Minimal annotated recording; the fitter deliberately ignores sensors."""
    registry = SensorRegistry.from_specs(
        [SensorSpec("M001", Modality.MOTION, room="bedroom")]
    )
    return CasasRecording(registry=registry, observations=(), activities=intervals)


def interval(hour: int, hours: float, state: S) -> ActivityInterval:
    start = datetime(2024, 1, 1, hour, tzinfo=UTC)
    return ActivityInterval(state.value, start, start + timedelta(hours=hours), state)


def test_sleeping_is_stickier_in_hours_where_it_is_overrepresented() -> None:
    fit = fit_circadian_profile(
        [
            recording(
                interval(0, 6, S.SLEEPING),
                interval(8, 8, S.HOME_ACTIVE),
                interval(16, 6, S.HOME_INACTIVE),
            )
        ],
        shrinkage_hours=0.5,
    )

    sleeping = fit.profile[S.SLEEPING]
    assert sleeping[2] > 1.0
    assert sleeping[12] < 1.0


def test_an_absent_state_gets_a_neutral_profile() -> None:
    fit = fit_circadian_profile(
        [recording(interval(9, 4, S.HOME_ACTIVE))], shrinkage_hours=1.0
    )
    assert fit.profile[S.BATHROOM_ACTIVITY] == pytest.approx(tuple([1.0] * 24))


def test_intervals_are_split_across_hour_boundaries() -> None:
    start = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    spans = ActivityInterval("sleep", start, start + timedelta(hours=2), S.SLEEPING)
    fit = fit_circadian_profile([recording(spans)], shrinkage_hours=0.0)

    # The interval contributes to 09, 10 and 11; with no competing labelled
    # state each observed hour has conditional share one and therefore the same
    # lift. Unobserved hours stay neutral.
    assert fit.profile[S.SLEEPING][9] == pytest.approx(1.0)
    assert fit.profile[S.SLEEPING][10] == pytest.approx(1.0)
    assert fit.profile[S.SLEEPING][11] == pytest.approx(1.0)
    assert fit.profile[S.SLEEPING][12] == pytest.approx(1.0)
    assert fit.labelled_seconds == pytest.approx(7200.0)


def test_sparse_cells_are_shrunk_toward_one() -> None:
    no_shrink = fit_circadian_profile(
        [recording(interval(0, 1, S.SLEEPING), interval(1, 10, S.HOME_ACTIVE))],
        shrinkage_hours=0.0,
        maximum_multiplier=10.0,
    )
    shrunk = fit_circadian_profile(
        [recording(interval(0, 1, S.SLEEPING), interval(1, 10, S.HOME_ACTIVE))],
        shrinkage_hours=4.0,
        maximum_multiplier=10.0,
    )

    assert abs(shrunk.profile[S.SLEEPING][0] - 1.0) < abs(
        no_shrink.profile[S.SLEEPING][0] - 1.0
    )


def test_multiplier_bounds_are_enforced() -> None:
    fit = fit_circadian_profile(
        [recording(interval(0, 1, S.SLEEPING), interval(1, 20, S.HOME_ACTIVE))],
        shrinkage_hours=0.0,
        minimum_multiplier=0.5,
        maximum_multiplier=2.0,
    )
    assert min(fit.profile[S.SLEEPING]) >= 0.5
    assert max(fit.profile[S.SLEEPING]) <= 2.0


def test_serialisation_and_hash_are_stable() -> None:
    recordings = [recording(interval(0, 6, S.SLEEPING), interval(8, 8, S.HOME_ACTIVE))]
    first = fit_circadian_profile(recordings)
    second = fit_circadian_profile(recordings)
    assert first.to_dict() == second.to_dict()
    assert first.sha256() == second.sha256()
    assert len(first.sha256()) == 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shrinkage_hours": -1.0},
        {"minimum_multiplier": 0.0},
        {"minimum_multiplier": 2.0, "maximum_multiplier": 1.0},
    ],
)
def test_invalid_fit_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        fit_circadian_profile([recording(interval(0, 1, S.SLEEPING))], **kwargs)
