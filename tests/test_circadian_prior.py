"""Tests for the optional circadian prior on state dynamics.

A dwell time says how long a state lasts. It cannot say when in the day that
state is plausible, so without this a resident motionless at 02:00 and one
motionless at 14:00 look identical to the prior.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

UTC = timezone.utc
STEP = timedelta(minutes=5)


def night_owl() -> dict:
    """Sleeping is sticky before 07:00 and slippery after."""
    return {S.SLEEPING: tuple(4.0 if hour < 7 else 0.25 for hour in range(24))}


class TestDefaults:
    def test_an_ontology_without_a_profile_is_time_homogeneous(self) -> None:
        """Every earlier release behaved this way and must continue to."""
        ontology = StateOntology()

        at_three = ontology.transition(STEP, at=datetime(2011, 6, 15, 3, tzinfo=UTC))
        at_two = ontology.transition(STEP, at=datetime(2011, 6, 15, 14, tzinfo=UTC))

        assert np.allclose(at_three, at_two)
        assert np.allclose(at_three, ontology.transition(STEP))

    def test_a_profile_without_a_moment_is_ignored(self) -> None:
        """Nothing can vary by hour if the hour is not supplied."""
        ontology = StateOntology(circadian=night_owl())

        assert np.allclose(ontology.transition(STEP), StateOntology().transition(STEP))


class TestModulation:
    def test_a_sticky_hour_makes_the_state_persist(self) -> None:
        ontology = StateOntology(circadian=night_owl())
        index = ontology.states.index(S.SLEEPING)

        night = ontology.transition(STEP, at=datetime(2011, 6, 15, 3, tzinfo=UTC))
        day = ontology.transition(STEP, at=datetime(2011, 6, 15, 14, tzinfo=UTC))

        assert night[index, index] > day[index, index]

    def test_the_result_is_still_a_probability_matrix(self) -> None:
        """Rescaling a generator row is only valid if the diagonal is rebuilt."""
        ontology = StateOntology(circadian=night_owl())

        for hour in (0, 6, 12, 23):
            matrix = ontology.transition(
                STEP, at=datetime(2011, 6, 15, hour, tzinfo=UTC)
            )
            assert np.all(matrix >= 0.0)
            assert np.allclose(matrix.sum(axis=1), 1.0)

    def test_states_without_a_profile_keep_their_generator_row(self) -> None:
        """A partial profile must not silently reshape the rest of the chain.

        Asserted on the generator rather than the transition matrix. The matrix
        is ``expm(Q t)``, which mixes every row: an unmodified state still ends
        up with different transition probabilities because paths *through* the
        modified state changed. That is correct behaviour, and it means a
        row-level invariant has to be checked where it actually holds.
        """
        ontology = StateOntology(circadian=night_owl())
        index = ontology.states.index(S.KITCHEN_ACTIVITY)

        modulated = ontology._circadian_generator(3)

        assert np.allclose(modulated[index], ontology.generator[index])

    def test_stickiness_changes_persistence_not_destination(self) -> None:
        """Being reluctant to leave is not the same as leaving somewhere else.

        Checked on the generator, where scaling a row is exact.
        """
        ontology = StateOntology(circadian=night_owl())
        index = ontology.states.index(S.SLEEPING)

        night = ontology._circadian_generator(3)[index]
        day = ontology._circadian_generator(14)[index]

        leaving_night = np.delete(night, index)
        leaving_day = np.delete(day, index)

        assert leaving_night.sum() < leaving_day.sum()
        assert np.allclose(
            leaving_night / leaving_night.sum(), leaving_day / leaving_day.sum()
        )


class TestValidation:
    def test_a_profile_needs_one_entry_per_hour(self) -> None:
        with pytest.raises(ValueError, match="one per hour"):
            StateOntology(circadian={S.SLEEPING: (1.0, 2.0, 3.0)})

    def test_a_zero_multiplier_is_refused(self) -> None:
        """It would divide the exit rate to zero, trapping the chain forever."""
        with pytest.raises(ValueError, match="positive"):
            StateOntology(circadian={S.SLEEPING: tuple([0.0] * 24)})

    def test_a_negative_multiplier_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            StateOntology(circadian={S.SLEEPING: tuple([-1.0] * 24)})

    def test_a_profile_for_an_unknown_state_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not in the ontology"):
            StateOntology(
                states=(S.AWAY, S.HOME_ACTIVE),
                circadian={S.KITCHEN_ACTIVITY: tuple([1.0] * 24)},
            )
