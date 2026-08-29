"""Tests for replicating the attribution comparison across seeds.

The single-seed study demonstrates that attribution behaves as designed. It
cannot estimate what attribution is worth, because one generated household is
one sample. These tests cover the replicated form and the Monte Carlo standard
error that makes a replication count interpretable.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensor_modeling.evaluation.attribution import (
    ReplicatedAttributionStudy,
    run_replicated_attribution_study,
)
from sensor_modeling.evaluation.metrics import paired_difference

FAST = {"days": 2}


class TestReplicationCount:
    def test_a_single_seed_is_refused(self) -> None:
        """One household is a demonstration, and must not look like a study."""
        with pytest.raises(ValueError, match="at least two seeds"):
            run_replicated_attribution_study([11], **FAST)

    def test_repeated_seeds_are_refused(self) -> None:
        """Duplicated seeds inflate n without adding information."""
        with pytest.raises(ValueError, match="distinct"):
            run_replicated_attribution_study([11, 11, 22], **FAST)

    def test_seeds_that_would_share_simulated_faults_are_refused(self) -> None:
        """Adjacent seeds are the obvious thing to type, and are wrong here.

        ``standard_scenarios`` derives degradation seeds as ``seed + 1`` to
        ``seed + 3``, so consecutive study seeds would hand two supposedly
        independent replications the same record loss and the same stuck
        sensors. Nothing downstream could detect that, so it is refused at the
        boundary.
        """
        with pytest.raises(ValueError, match="differ by at least"):
            run_replicated_attribution_study([11, 12, 13], **FAST)

    def test_widely_spaced_seeds_are_accepted(self) -> None:
        study = run_replicated_attribution_study([11, 33, 55], **FAST)
        assert study.seeds == (11, 33, 55)

    def test_every_scenario_is_estimated_over_every_seed(self) -> None:
        seeds = [11, 22, 33]
        study = run_replicated_attribution_study(seeds, **FAST)

        assert study.seeds == tuple(seeds)
        assert study.aggregates
        for aggregate in study.aggregates:
            assert aggregate.balanced_accuracy_gain.n == len(seeds)
            assert aggregate.visitor_recall["n"] == len(seeds)


class TestMonteCarloError:
    def test_mcse_is_reported_with_the_interval(self) -> None:
        rng = np.random.default_rng(0)
        control = rng.normal(0.5, 0.05, 60)
        treatment = control + rng.normal(0.01, 0.004, 60)

        result = paired_difference(treatment, control)

        assert result.mcse > 0.0
        assert "mcse" in result.to_dict()

    def test_mcse_falls_with_the_square_root_of_the_replication_count(self) -> None:
        """This is the property that lets a target MCSE choose n.

        Quadrupling the replications should roughly halve the standard error.
        Without that behaviour the protocol's sample-size arithmetic would not
        hold.
        """
        rng = np.random.default_rng(7)
        differences = rng.normal(0.01, 0.005, 4000)

        small = paired_difference(differences[:100], np.zeros(100))
        large = paired_difference(differences[:400], np.zeros(400))

        assert large.mcse == pytest.approx(small.mcse / 2.0, rel=0.25)

    def test_a_wide_spread_needs_more_replications(self) -> None:
        """MCSE must track the spread, not only the count."""
        rng = np.random.default_rng(3)
        tight = rng.normal(0.01, 0.001, 200)
        loose = rng.normal(0.01, 0.010, 200)

        assert (
            paired_difference(loose, np.zeros(200)).mcse
            > paired_difference(tight, np.zeros(200)).mcse
        )


class TestSerialisation:
    def test_the_study_serialises_with_its_replication_count(self) -> None:
        study = run_replicated_attribution_study([11, 22], **FAST)
        payload = study.to_dict()

        assert payload["replications"] == 2
        assert payload["seeds"] == [11, 22]
        first = payload["scenarios"][0]
        assert "mcse" in first["balanced_accuracy_gain"]
        assert "mcse" in first["visitor_recall"]

    def test_an_empty_study_reports_no_replications(self) -> None:
        assert ReplicatedAttributionStudy().to_dict()["replications"] == 0
