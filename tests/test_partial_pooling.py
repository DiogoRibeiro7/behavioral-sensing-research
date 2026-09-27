"""Tests for partial pooling of household parameters toward a population.

They cover the estimator on its own, with no data, a very small sample, a
large sample, identical households and strongly different ones, and its first
application, the hurdle channel parameters:
- an unseen household gets the population exactly;
- a household reads only its own windows up to the declared moment;
- a household the population was fitted on is refused;
- the fitted parameters round-trip through JSON with their provenance.
"""

from __future__ import annotations

import json
import math
from datetime import timedelta
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import ActivityInterval, CasasRecording
from sensor_modeling.datasets.channel_models import (
    HURDLE,
    FittedChannels,
    HouseholdChannels,
    adapt_channels,
    fit_channel_models,
    household_channel_counts,
)
from sensor_modeling.datasets.information_sets import EvidenceResolution
from sensor_modeling.datasets.partial_pooling import (
    PooledEstimate,
    PoolingConfig,
    pool,
)
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

ONTOLOGY = StateOntology()
RESOLUTION = EvidenceResolution()
CONFIG = PoolingConfig()


# ----------------------------------------------------------------------------
# The estimator
# ----------------------------------------------------------------------------
class TestEstimator:
    def test_zero_observations_give_the_population_exactly(self) -> None:
        estimate = pool([0.3, 0.8], [0.0, 0.0], [0.0, 0.0], CONFIG)
        np.testing.assert_array_equal(estimate.pooled, [0.3, 0.8])
        np.testing.assert_array_equal(estimate.shrinkage, [1.0, 1.0])
        np.testing.assert_array_equal(estimate.deviation, [0.0, 0.0])
        assert np.isnan(estimate.raw).all()

    def test_a_very_small_sample_stays_near_the_population(self) -> None:
        estimate = pool([0.5], [2.0], [2.0], CONFIG)
        assert estimate.raw[0] == 1.0
        assert estimate.shrinkage[0] == pytest.approx(288.0 / 290.0)
        assert estimate.pooled[0] == pytest.approx(0.5 + 0.5 * 2.0 / 290.0)
        assert abs(estimate.deviation[0]) < 0.01

    def test_a_large_sample_approaches_its_own_estimate(self) -> None:
        estimate = pool([0.5], [9_000_000.0], [10_000_000.0], CONFIG)
        assert estimate.pooled[0] == pytest.approx(0.9, abs=1e-4)
        assert estimate.shrinkage[0] < 1e-4

    def test_identical_households_get_identical_estimates(self) -> None:
        first = pool([0.4, 2.0], [30.0, 150.0], [100.0, 60.0], CONFIG)
        second = pool([0.4, 2.0], [30.0, 150.0], [100.0, 60.0], CONFIG)
        assert first.to_dict() == second.to_dict()
        at_population = pool([0.4], [40.0], [100.0], CONFIG)
        assert at_population.deviation[0] == pytest.approx(0.0)

    def test_strongly_different_households_are_shrunk_symmetrically(self) -> None:
        low = pool([0.5], [10.0], [100.0], CONFIG)
        high = pool([0.5], [90.0], [100.0], CONFIG)
        assert low.deviation[0] == pytest.approx(-high.deviation[0])
        assert low.deviation[0] < 0.0 < high.deviation[0]
        assert high.deviation[0] == pytest.approx(0.4 * 100.0 / 388.0)
        more = pool([0.5], [900.0], [1000.0], CONFIG)
        assert more.deviation[0] > high.deviation[0]

    def test_the_estimate_is_population_plus_shrunk_deviation(self) -> None:
        rng = np.random.default_rng(0)
        count = rng.integers(0, 5000, size=200).astype(float)
        total = np.floor(rng.random(200) * count)
        population = rng.random(200)
        estimate = pool(population, total, count, PoolingConfig(strength=50.0))
        seen = count > 0
        np.testing.assert_allclose(
            estimate.pooled[seen],
            population[seen]
            + (1.0 - estimate.shrinkage[seen])
            * (estimate.raw[seen] - population[seen]),
        )
        low = np.minimum(estimate.raw[seen], population[seen])
        high = np.maximum(estimate.raw[seen], population[seen])
        assert np.all(
            (estimate.pooled[seen] >= low - 1e-12)
            & (estimate.pooled[seen] <= high + 1e-12)
        )

    def test_the_limits_are_the_population_and_unconstrained_fitting(self) -> None:
        tight = pool([0.5], [30.0], [100.0], PoolingConfig(strength=1e12))
        loose = pool([0.5], [30.0], [100.0], PoolingConfig(strength=1e-9))
        assert tight.pooled[0] == pytest.approx(0.5)
        assert loose.pooled[0] == pytest.approx(0.3)

    def test_it_round_trips_through_json(self) -> None:
        estimate = pool([0.25, 3.5, 0.9], [1.0, 70.0, 0.0], [7.0, 20.0, 0.0], CONFIG)
        payload = json.loads(json.dumps(estimate.to_dict()))
        again = PooledEstimate.from_dict(payload)
        assert again.to_dict() == estimate.to_dict()
        payload["pooled"][0] = 0.99
        with pytest.raises(ValueError, match="disagree"):
            PooledEstimate.from_dict(payload)

    @pytest.mark.parametrize("strength", [0.0, -1.0, math.inf, math.nan])
    def test_invalid_strengths_are_refused(self, strength: float) -> None:
        with pytest.raises(ValueError, match="strength"):
            PoolingConfig(strength=strength)

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (([0.5], [1.0], [0.0]), "at least one observation"),
            (([0.5], [-1.0], [2.0]), "totals"),
            (([0.5], [1.0], [-2.0]), "counts"),
            (([math.nan], [1.0], [2.0]), "population"),
            (([0.5, 0.5], [1.0], [2.0]), "same length"),
        ],
    )
    def test_invalid_statistics_are_refused(
        self, arguments: tuple[list[float], ...], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            pool(arguments[0], arguments[1], arguments[2], CONFIG)


# ----------------------------------------------------------------------------
# The hurdle channel parameters
# ----------------------------------------------------------------------------
def simulated(seed: int) -> CasasRecording:
    sim = simulate(HouseholdConfig(days=4, seed=seed))
    return CasasRecording(
        sim.registry,
        sim.observations,
        tuple(
            ActivityInterval(e.state.value, e.start, e.end, e.state)
            for e in sim.truth.episodes
        ),
    )


def relabelled_after(recording: CasasRecording, moment: Any) -> CasasRecording:
    """The recording with every annotation that starts after *moment* made ``sleeping``."""
    return CasasRecording(
        recording.registry,
        recording.observations,
        tuple(
            (
                a
                if a.start <= moment
                else ActivityInterval("Sleep", a.start, a.end, S.SLEEPING)
            )
            for a in recording.activities
        ),
    )


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {f"sim{s}": simulated(s) for s in range(1, 4)}


@pytest.fixture(scope="module")
def population(homes: dict[str, CasasRecording]) -> FittedChannels:
    return fit_channel_models(
        {"sim2": homes["sim2"], "sim3": homes["sim3"]},
        resolution=RESOLUTION,
        pseudo_windows=12.0,
        ontology=ONTOLOGY,
    )


def midpoint(recording: CasasRecording) -> Any:
    first = recording.observations[0].timestamp
    return first + (recording.observations[-1].timestamp - first) / 2


def adapt(
    population: FittedChannels,
    recording: CasasRecording,
    until: Any,
    household: str = "sim1",
    config: PoolingConfig = CONFIG,
) -> HouseholdChannels:
    return adapt_channels(
        population,
        recording,
        household=household,
        resolution=RESOLUTION,
        config=config,
        until=until,
        ontology=ONTOLOGY,
    )


class TestHouseholdChannels:
    def test_an_unseen_household_gets_the_population_exactly(
        self, homes: dict[str, CasasRecording], population: FittedChannels
    ) -> None:
        adapted = adapt(population, homes["sim1"], None)
        assert adapted.until is None
        expected = population.models(
            homes["sim1"].registry, RESOLUTION, HURDLE, ONTOLOGY
        )
        counts = np.arange(0.0, 8.0)
        for channel, model in adapted.models().items():
            np.testing.assert_array_equal(
                model.loglik(counts), expected[channel].loglik(counts)
            )
        assert all(row["shrinkage"] == 1.0 for row in adapted.diagnostics())
        assert all(row["raw"] is None for row in adapted.diagnostics())

    def test_zero_household_observations_before_its_first_window(
        self, homes: dict[str, CasasRecording], population: FittedChannels
    ) -> None:
        recording = homes["sim1"]
        early = recording.observations[0].timestamp - timedelta(hours=1)
        adapted = adapt(population, recording, early)
        unseen = adapt(population, recording, None)
        for channel in adapted.silence:
            np.testing.assert_array_equal(
                adapted.silence[channel].pooled, unseen.silence[channel].pooled
            )
            np.testing.assert_array_equal(
                adapted.activity[channel].pooled, unseen.activity[channel].pooled
            )

    def test_the_diagnostics_show_every_estimate(
        self, homes: dict[str, CasasRecording], population: FittedChannels
    ) -> None:
        recording = homes["sim1"]
        adapted = adapt(population, recording, midpoint(recording))
        rows = adapted.diagnostics()
        assert len(rows) == len(adapted.silence) * 2 * ONTOLOGY.size
        observed = [row for row in rows if row["raw"] is not None]
        assert observed
        for row in observed:
            assert row["shrinkage"] == pytest.approx(
                288.0 / (row["observations"] + 288.0)
            )
            assert row["pooled"] == pytest.approx(
                row["population"]
                + (1.0 - row["shrinkage"]) * (row["raw"] - row["population"])
            )
        assert {row["parameter"] for row in rows} == {"silence", "active_mean"}

    def test_more_household_data_means_less_shrinkage(
        self, homes: dict[str, CasasRecording], population: FittedChannels
    ) -> None:
        recording = homes["sim1"]
        half = adapt(population, recording, midpoint(recording))
        full = adapt(population, recording, recording.observations[-1].timestamp)
        for channel in half.silence:
            assert np.all(
                full.silence[channel].shrinkage <= half.silence[channel].shrinkage
            )

    def test_it_reads_nothing_after_until(
        self, homes: dict[str, CasasRecording], population: FittedChannels
    ) -> None:
        recording = homes["sim1"]
        until = midpoint(recording)
        before = adapt(population, recording, until)
        later_changed = adapt(population, relabelled_after(recording, until), until)
        assert later_changed.to_dict() == before.to_dict()
        everything = adapt(
            population,
            relabelled_after(recording, until),
            recording.observations[-1].timestamp,
        )
        assert (
            everything.to_dict()
            != adapt(
                population, recording, recording.observations[-1].timestamp
            ).to_dict()
        )

    def test_a_household_the_population_was_fitted_on_is_refused(
        self, homes: dict[str, CasasRecording], population: FittedChannels
    ) -> None:
        with pytest.raises(ValueError, match="fit the population"):
            adapt(population, homes["sim2"], None, household="sim2")

    def test_identical_households_get_identical_parameters(
        self, homes: dict[str, CasasRecording], population: FittedChannels
    ) -> None:
        recording = homes["sim1"]
        until = midpoint(recording)
        first = adapt(population, recording, until, household="copy-a")
        second = adapt(population, recording, until, household="copy-b")
        assert first.to_dict()["channels"] == second.to_dict()["channels"]

    def test_strongly_different_households_move_apart(
        self, homes: dict[str, CasasRecording], population: FittedChannels
    ) -> None:
        recording = homes["sim1"]
        end = recording.observations[-1].timestamp
        start = recording.observations[0].timestamp
        ordinary = adapt(population, recording, end)
        asleep = adapt(
            population, relabelled_after(recording, start - timedelta(hours=1)), end
        )
        sleeping = ONTOLOGY.states.index(S.SLEEPING)
        moved = [
            abs(
                asleep.silence[c].pooled[sleeping]
                - ordinary.silence[c].pooled[sleeping]
            )
            for c in ordinary.silence
        ]
        assert max(moved) > 0.01
        for channel in ordinary.silence:
            for estimate in (asleep.silence[channel], ordinary.silence[channel]):
                seen = estimate.count > 0
                low = np.minimum(estimate.raw[seen], estimate.population[seen])
                high = np.maximum(estimate.raw[seen], estimate.population[seen])
                assert np.all(estimate.pooled[seen] >= low - 1e-12)
                assert np.all(estimate.pooled[seen] <= high + 1e-12)

    def test_parameters_round_trip_with_their_provenance(
        self, homes: dict[str, CasasRecording], population: FittedChannels
    ) -> None:
        recording = homes["sim1"]
        until = midpoint(recording)
        adapted = adapt(population, recording, until)
        payload = json.loads(json.dumps(adapted.to_dict()))
        assert payload["population"] == {
            "sha256": population.sha256(),
            "fitted_on": ["sim2", "sim3"],
        }
        assert payload["until"] == until.isoformat()
        assert payload["pooling"] == {"strength": 288.0}
        again = HouseholdChannels.from_dict(payload)
        assert again.to_dict() == adapted.to_dict()
        assert again.sha256() == adapted.sha256()
        counts = np.arange(0.0, 6.0)
        for channel, model in again.models().items():
            np.testing.assert_array_equal(
                model.loglik(counts), adapted.models()[channel].loglik(counts)
            )

    def test_fitting_is_deterministic(
        self, homes: dict[str, CasasRecording], population: FittedChannels
    ) -> None:
        recording = homes["sim1"]
        until = midpoint(recording)
        assert adapt(population, recording, until).sha256() == (
            adapt(population, recording, until).sha256()
        )

    def test_the_adapted_models_are_valid_hurdle_channels(
        self, homes: dict[str, CasasRecording], population: FittedChannels
    ) -> None:
        recording = homes["sim1"]
        adapted = adapt(population, recording, recording.observations[-1].timestamp)
        counts, _, _, _ = household_channel_counts(
            recording, RESOLUTION, ONTOLOGY, household="sim1"
        )
        for channel, model in adapted.models().items():
            values = model.loglik(counts[channel])
            assert np.all(np.isfinite(values))
            assert np.all((model.silence > 0.0) & (model.silence < 1.0))
