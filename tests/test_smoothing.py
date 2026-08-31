"""Tests for revising past estimates with later evidence.

Smoothing is the one way a recursive filter can use more context without
double-counting: it conditions on evidence that arrived *after* the estimate,
which the belief has not already absorbed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sensor_modeling.fusion import smooth_beliefs, smooth_estimates
from sensor_modeling.fusion.estimate import StateEstimate
from sensor_modeling.states import StateOntology

UTC = timezone.utc
STEP = timedelta(minutes=5)
ONTOLOGY = StateOntology()
SIZE = len(ONTOLOGY.states)


def beliefs(*peaks: int, confidence: float = 0.8) -> np.ndarray:
    """One row per step, each concentrated on the named state."""
    rows = []
    for peak in peaks:
        row = np.full(SIZE, (1.0 - confidence) / (SIZE - 1))
        row[peak] = confidence
        rows.append(row / row.sum())
    return np.vstack(rows)


def estimates(array: np.ndarray) -> list[StateEstimate]:
    start = datetime(2011, 6, 15, 12, 0, tzinfo=UTC)
    return [
        StateEstimate(
            at=start + index * STEP,
            ontology=ONTOLOGY,
            belief=row,
            evidence=(),
            completeness=1.0,
            min_confidence=0.0,
            min_completeness=0.0,
        )
        for index, row in enumerate(array)
    ]


class TestBeliefs:
    def test_zero_lag_changes_nothing(self) -> None:
        array = beliefs(0, 3, 0)
        assert np.allclose(
            smooth_beliefs(array, ONTOLOGY.transition(STEP), lag=0), array
        )

    def test_results_remain_distributions(self) -> None:
        array = beliefs(0, 3, 0, 5, 1)
        for lag in (1, 2, 4):
            out = smooth_beliefs(array, ONTOLOGY.transition(STEP), lag=lag)
            assert np.all(out >= 0.0)
            assert np.allclose(out.sum(axis=1), 1.0)

    def test_an_isolated_blip_is_pulled_toward_its_neighbours(self) -> None:
        """One odd step surrounded by agreement is what smoothing is for."""
        array = beliefs(0, 0, 3, 0, 0)
        out = smooth_beliefs(array, ONTOLOGY.transition(STEP), lag=2)

        assert out[2][3] < array[2][3]
        assert out[2][0] > array[2][0]

    def test_a_longer_lag_is_not_a_full_smoother(self) -> None:
        """Each correction must start from the filtered horizon, not a smoothed one.

        Starting from already-smoothed values chains corrections through the
        whole sequence, so every lag returns identical numbers and an
        offline-only result is reported as a five-minute one.
        """
        array = beliefs(0, 0, 3, 0, 0, 0, 0, 5, 0, 0)
        transition = ONTOLOGY.transition(STEP)

        short = smooth_beliefs(array, transition, lag=1)
        long = smooth_beliefs(array, transition, lag=6)

        assert not np.allclose(short, long)

    def test_a_negative_lag_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            smooth_beliefs(beliefs(0, 1), ONTOLOGY.transition(STEP), lag=-1)

    def test_a_mismatched_transition_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            smooth_beliefs(beliefs(0, 1), np.eye(SIZE + 1), lag=1)


class TestEstimates:
    def test_evidence_is_left_untouched(self) -> None:
        """It records what the sensors argued at the time, not afterwards."""
        original = estimates(beliefs(0, 3, 0))
        smoothed = smooth_estimates(original, lag=1)

        for before, after in zip(original, smoothed):
            assert after.evidence == before.evidence
            assert after.at == before.at

    def test_beliefs_are_revised(self) -> None:
        original = estimates(beliefs(0, 3, 0))
        smoothed = smooth_estimates(original, lag=2)

        assert not np.allclose(smoothed[1].belief, original[1].belief)

    def test_zero_lag_returns_the_input(self) -> None:
        original = estimates(beliefs(0, 3, 0))
        assert smooth_estimates(original, lag=0) == original

    def test_out_of_order_estimates_are_refused(self) -> None:
        """The backward recursion would otherwise mix unrelated moments."""
        original = estimates(beliefs(0, 3, 0))
        shuffled = [original[2], original[0], original[1]]

        with pytest.raises(ValueError, match="ascending time order"):
            smooth_estimates(shuffled, lag=1)

    def test_a_single_estimate_is_returned_unchanged(self) -> None:
        original = estimates(beliefs(0))
        assert smooth_estimates(original, lag=3) == original
