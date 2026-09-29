"""Tests for the hurdle model with a zero-truncated negative-binomial active count.

They check the likelihood itself, its fit and the family's place beside the
hurdle model:
- probabilities sum to one, and the moments match enumeration;
- dispersion zero is exactly the hurdle-Poisson model, and a tiny dispersion
  is numerically indistinguishable from it;
- a strongly over-dispersed process is recovered;
- zero counts, very large counts and channels with little data are handled;
- the fit uses the training households only and shrinks toward the Poisson;
- models and fits serialise, and adding the family changes no published digest.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from sensor_modeling.datasets import ActivityInterval, CasasRecording
from sensor_modeling.datasets.channel_models import (
    HURDLE,
    HURDLE_NB,
    MAX_DISPERSION,
    MIN_TRUNCATED_RATE,
    ActiveCounts,
    ChannelStatistics,
    FittedChannels,
    HurdleChannel,
    HurdleNBChannel,
    combine_statistics,
    filter_recursion,
    fit_channel_models,
    fit_dispersion,
    home_statistics,
    household_channel_counts,
    household_statistics,
    nb_log_zero,
    pool_channels,
    total_loglik,
    truncated_poisson_rate,
    ztnb_log_pmf,
    ztnb_moments,
    ztnb_rate,
)
from sensor_modeling.datasets.information_sets import (
    EvidenceChannel,
    EvidenceResolution,
)
from sensor_modeling.datasets.partial_pooling import PoolingConfig
from sensor_modeling.observations import Modality
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import StateOntology

ONTOLOGY = StateOntology()
STATES = len(ONTOLOGY.states)
RESOLUTION = EvidenceResolution()
KITCHEN = EvidenceChannel("kitchen", Modality.MOTION)


def model(silence: float, rate: float, dispersion: float) -> HurdleNBChannel:
    return HurdleNBChannel(
        KITCHEN,
        np.full(STATES, silence),
        np.full(STATES, rate),
        np.full(STATES, dispersion),
    )


def hurdle_sample(
    rng: np.random.Generator, silence: float, mean: float, dispersion: float, size: int
) -> np.ndarray:
    """Counts from the hurdle with a negative-binomial active count."""
    draws: list[np.ndarray] = []
    need = size
    while need > 0:
        if dispersion == 0.0:
            batch = rng.poisson(mean, 4 * need)
        else:
            r = 1.0 / dispersion
            batch = rng.negative_binomial(r, r / (r + mean), 4 * need)
        batch = batch[batch > 0]
        draws.append(batch)
        need -= batch.size
    counts: np.ndarray = np.concatenate(draws)[:size].astype(float)
    counts[rng.random(size) < silence] = 0.0
    return counts


def statistics_of(counts: np.ndarray, declared: float = 1.0) -> ChannelStatistics:
    """Statistics of *counts*, every window in the first state."""
    labels = np.zeros(counts.size, dtype=int)
    size = np.zeros(STATES)
    size[0] = counts.size
    return ChannelStatistics(
        windows=size.copy(),
        silent=np.where(np.arange(STATES) == 0, float(np.sum(counts == 0)), 0.0),
        total=np.where(np.arange(STATES) == 0, float(counts.sum()), 0.0),
        squares=np.where(np.arange(STATES) == 0, float((counts * counts).sum()), 0.0),
        declared=size * declared,
        active_counts=ActiveCounts.from_counts(counts, labels, STATES),
    )


def fitted_on(counts: np.ndarray, pseudo_windows: float = 12.0) -> FittedChannels:
    return FittedChannels(
        ("h",), tuple(ONTOLOGY.states), pseudo_windows, {KITCHEN: statistics_of(counts)}
    )


# ----------------------------------------------------------------------------
# The distribution
# ----------------------------------------------------------------------------
class TestDistribution:
    @pytest.mark.parametrize(
        ("silence", "rate", "dispersion"),
        [(0.3, 2.0, 0.0), (0.8, 0.4, 0.5), (0.1, 6.0, 3.0), (0.5, 0.05, 50.0)],
    )
    def test_probabilities_sum_to_one(
        self, silence: float, rate: float, dispersion: float
    ) -> None:
        counts = np.arange(0, 20_000, dtype=float)
        pmf = np.exp(model(silence, rate, dispersion).log_pmf(counts)[:, 0])
        assert pmf.sum() == pytest.approx(1.0, abs=1e-9)
        assert pmf[0] == pytest.approx(silence)

    @pytest.mark.parametrize(
        ("rate", "dispersion"), [(1.5, 0.0), (1.5, 0.7), (8.0, 2.5)]
    )
    def test_the_moments_match_enumeration(
        self, rate: float, dispersion: float
    ) -> None:
        k = np.arange(1, 20_000, dtype=float)
        pmf = np.exp(ztnb_log_pmf(k, rate, dispersion))
        mean, variance = ztnb_moments(rate, dispersion)
        assert float(mean) == pytest.approx(float((k * pmf).sum()), rel=1e-9)
        assert float(variance) == pytest.approx(
            float((k * k * pmf).sum() - (k * pmf).sum() ** 2), rel=1e-7
        )

    @pytest.mark.parametrize("dispersion", [1e-4, 0.3, 4.0])
    def test_it_is_the_negative_binomial_truncated_at_zero(
        self, dispersion: float
    ) -> None:
        k = np.arange(1, 60, dtype=float)
        rate, size = 2.7, 1.0 / dispersion
        reference = nbinom.logpmf(k, size, size / (size + rate)) - np.log1p(
            -nbinom.pmf(0, size, size / (size + rate))
        )
        np.testing.assert_allclose(
            ztnb_log_pmf(k, rate, dispersion), reference, rtol=1e-10, atol=1e-10
        )
        assert float(nb_log_zero(rate, dispersion)) == pytest.approx(
            nbinom.logpmf(0, size, size / (size + rate)), rel=1e-12
        )

    @pytest.mark.parametrize("mean", [1.0, 1.0000001, 1.3, 4.0, 60.0])
    @pytest.mark.parametrize("dispersion", [0.0, 0.2, 5.0, MAX_DISPERSION])
    def test_the_rate_keeps_the_active_mean(
        self, mean: float, dispersion: float
    ) -> None:
        rate = ztnb_rate(mean, dispersion)
        if mean == 1.0:
            assert rate == MIN_TRUNCATED_RATE
        else:
            assert float(ztnb_moments(rate, dispersion)[0]) == pytest.approx(
                mean, rel=1e-9
            )

    def test_invalid_parameters_are_refused(self) -> None:
        with pytest.raises(ValueError, match="dispersions"):
            model(0.5, 1.0, -0.1)
        with pytest.raises(ValueError, match="silence"):
            model(1.0, 1.0, 0.1)
        with pytest.raises(ValueError, match="rates"):
            model(0.5, 0.0, 0.1)
        with pytest.raises(ValueError, match="at least 1"):
            ztnb_rate(0.5, 0.3)
        with pytest.raises(ValueError, match="non-negative"):
            ztnb_rate(2.0, -1.0)


# ----------------------------------------------------------------------------
# The Poisson limit
# ----------------------------------------------------------------------------
class TestPoissonLimit:
    def test_dispersion_zero_is_the_hurdle_model(self) -> None:
        silence = np.linspace(0.1, 0.9, STATES)
        rate = np.linspace(0.2, 7.0, STATES)
        counts = np.array([0.0, 1.0, 2.0, 7.0, 40.0])
        negative_binomial = HurdleNBChannel(KITCHEN, silence, rate, np.zeros(STATES))
        hurdle = HurdleChannel(KITCHEN, silence, rate)
        np.testing.assert_allclose(
            negative_binomial.loglik(counts), hurdle.loglik(counts), rtol=0, atol=1e-12
        )
        for count in (0.0, 3.0):
            for part, values in negative_binomial.decompose(count).items():
                np.testing.assert_allclose(
                    values, hurdle.decompose(count)[part], atol=1e-12
                )

    @pytest.mark.parametrize("count", [1.0, 5.0, 50.0, 500.0])
    def test_a_tiny_dispersion_is_numerically_the_poisson(self, count: float) -> None:
        tiny = model(0.4, 3.0, 1e-12).loglik(np.array([count]))
        exact = model(0.4, 3.0, 0.0).loglik(np.array([count]))
        np.testing.assert_allclose(tiny, exact, atol=1e-6)

    def test_poisson_data_fit_a_dispersion_near_zero(self) -> None:
        counts = hurdle_sample(np.random.default_rng(1), 0.5, 2.0, 0.0, 40_000)
        alpha = fitted_on(counts).dispersion(KITCHEN)[0]
        assert alpha < 0.01
        hurdle = fitted_on(counts).parameters(KITCHEN)
        assert ztnb_rate(float(hurdle["active_mean"][0]), 0.0) == pytest.approx(
            float(hurdle["rate"][0])
        )


# ----------------------------------------------------------------------------
# Strong over-dispersion
# ----------------------------------------------------------------------------
class TestOverDispersion:
    @pytest.mark.parametrize(("mean", "dispersion"), [(3.0, 0.5), (4.0, 2.0)])
    def test_the_dispersion_is_recovered(self, mean: float, dispersion: float) -> None:
        counts = hurdle_sample(np.random.default_rng(2), 0.6, mean, dispersion, 60_000)
        fit = fitted_on(counts, pseudo_windows=1.0)
        alpha = float(fit.dispersion(KITCHEN)[0])
        assert alpha == pytest.approx(dispersion, rel=0.1)
        active = counts[counts > 0]
        rate = ztnb_rate(float(fit.parameters(KITCHEN)["active_mean"][0]), alpha)
        _, variance = ztnb_moments(rate, alpha)
        assert float(variance) == pytest.approx(float(active.var()), rel=0.1)

    def test_it_explains_over_dispersed_counts_better_than_the_poisson(self) -> None:
        counts = hurdle_sample(np.random.default_rng(3), 0.5, 4.0, 2.0, 20_000)
        fit = fitted_on(counts)
        registry_free = {
            family: _single_state_model(fit, family) for family in (HURDLE, HURDLE_NB)
        }
        gain = (
            registry_free[HURDLE_NB].loglik(counts)[:, 0].sum()
            - registry_free[HURDLE].loglik(counts)[:, 0].sum()
        )
        assert gain > 1000.0

    def test_a_log_series_like_table_stops_at_the_bound(self) -> None:
        # Many ones and a long tail: the likelihood rises toward the
        # logarithmic-series limit, and the fit stops at the declared bound.
        k = np.arange(1, 400, dtype=float)
        theta = 0.97
        weights = 5000.0 * theta**k / k
        mean = float((k * weights).sum() / weights.sum())
        assert fit_dispersion(k, weights, mean) == MAX_DISPERSION
        assert fit_dispersion(k, weights, mean, max_dispersion=20.0) == 20.0


def _single_state_model(fit: FittedChannels, family: str) -> Any:
    parameters = fit.parameters(KITCHEN)
    silence = np.full(STATES, float(parameters["silence"][0]))
    if family == HURDLE:
        return HurdleChannel(
            KITCHEN, silence, np.full(STATES, float(parameters["rate"][0]))
        )
    alpha = float(fit.dispersion(KITCHEN)[0])
    rate = ztnb_rate(float(parameters["active_mean"][0]), alpha)
    return HurdleNBChannel(
        KITCHEN, silence, np.full(STATES, rate), np.full(STATES, alpha)
    )


# ----------------------------------------------------------------------------
# Zero counts, large counts and little data
# ----------------------------------------------------------------------------
class TestEdges:
    def test_a_zero_count_is_the_silence_probability(self) -> None:
        silence = np.linspace(0.05, 0.95, STATES)
        found = HurdleNBChannel(
            KITCHEN, silence, np.full(STATES, 2.0), np.full(STATES, 3.0)
        ).loglik(np.zeros(4))
        np.testing.assert_allclose(found, np.tile(np.log(silence), (4, 1)))

    def test_an_all_silent_channel_fits_the_poisson_limit(self) -> None:
        fit = fitted_on(np.zeros(500))
        assert fit.dispersion(KITCHEN)[0] == 0.0
        assert fit.parameters(KITCHEN)["silence"][0] > 0.9

    @pytest.mark.parametrize("dispersion", [0.0, 1e-9, 0.5, MAX_DISPERSION])
    def test_very_large_counts_stay_finite(self, dispersion: float) -> None:
        counts = np.array([1e3, 1e5, 1e6])
        channel = model(0.5, 3.0, dispersion)
        assert np.all(np.isfinite(channel.loglik(counts)))
        probabilities = channel.log_pmf(counts)[:, 0]
        assert np.all(np.isfinite(probabilities))
        assert np.all(np.diff(probabilities) < 0.0)  # far into the tail, less likely

    def test_a_count_of_one_million_matches_the_reference(self) -> None:
        size = 2.0
        reference = nbinom.logpmf(1e6, size, size / (size + 5.0)) - np.log1p(
            -nbinom.pmf(0, size, size / (size + 5.0))
        )
        assert float(ztnb_log_pmf(1e6, 5.0, 0.5)) == pytest.approx(reference, rel=1e-12)

    def test_little_data_shrinks_toward_the_poisson(self) -> None:
        # Four active windows spread widely: alone they suggest a large
        # dispersion; twelve Poisson-shaped pseudo-windows pull it back.
        counts = np.array([0.0] * 40 + [1.0, 2.0, 5.0, 21.0])
        unshrunk = fitted_on(counts, pseudo_windows=1e-6).dispersion(KITCHEN)[0]
        shrunk = fitted_on(counts, pseudo_windows=12.0).dispersion(KITCHEN)[0]
        strong = fitted_on(counts, pseudo_windows=1e4).dispersion(KITCHEN)[0]
        # The penalty is second order in the dispersion near zero, so a single
        # extreme count keeps it slightly positive even under a strong prior;
        # it falls toward zero as the prior grows.
        assert unshrunk > 1.0
        assert shrunk < unshrunk / 2.0
        assert strong < shrunk / 10.0
        assert strong < 0.1

    def test_without_training_windows_the_fit_is_the_declared_model(self) -> None:
        declared = np.linspace(0.1, 3.0, STATES)
        empty = FittedChannels(("h",), tuple(ONTOLOGY.states), 12.0, {})
        alpha = empty.dispersion(KITCHEN, declared)
        np.testing.assert_array_equal(alpha, np.zeros(STATES))
        parameters = empty.parameters(KITCHEN, declared)
        np.testing.assert_allclose(parameters["silence"], np.exp(-declared))

    def test_a_state_without_windows_uses_the_households_prior(self) -> None:
        counts = hurdle_sample(np.random.default_rng(4), 0.5, 3.0, 1.0, 3000)
        fit = fitted_on(counts)
        assert np.all(np.isnan(fit.dispersion(KITCHEN)[1:]))
        declared = np.full(STATES, 0.8)
        alpha = fit.dispersion(KITCHEN, declared)
        assert np.all(alpha[1:] == 0.0) and alpha[0] > 0.5

    def test_sums_alone_cannot_fit_the_dispersion(self) -> None:
        stats = replace(statistics_of(np.array([0.0, 1.0, 3.0])), active_counts=None)
        fit = FittedChannels(("h",), tuple(ONTOLOGY.states), 12.0, {KITCHEN: stats})
        fit.parameters(KITCHEN)  # the hurdle family still fits
        with pytest.raises(ValueError, match="active-count tables"):
            fit.dispersion(KITCHEN)


# ----------------------------------------------------------------------------
# Tables, serialisation and the unchanged hurdle fit
# ----------------------------------------------------------------------------
class TestTablesAndSerialisation:
    def test_active_counts_tabulate_and_combine(self) -> None:
        a = ActiveCounts.from_counts(
            np.array([0, 1, 1, 4, 0, 2]), np.array([0, 0, 1, 1, 2, 2]), 3
        )
        assert a.values.tolist() == [1.0, 2.0, 4.0]
        assert a.frequencies.tolist() == [[1, 0, 0], [1, 0, 1], [0, 1, 0]]
        b = ActiveCounts.from_counts(np.array([3, 1]), np.array([0, 2]), 3)
        combined = ActiveCounts.combine([a, b])
        whole = ActiveCounts.from_counts(
            np.array([0, 1, 1, 4, 0, 2, 3, 1]), np.array([0, 0, 1, 1, 2, 2, 0, 2]), 3
        )
        np.testing.assert_array_equal(combined.values, whole.values)
        np.testing.assert_array_equal(combined.frequencies, whole.frequencies)

    def test_tables_and_models_round_trip(self) -> None:
        table = ActiveCounts.from_counts(np.array([0, 2, 9, 2]), np.zeros(4, int), 2)
        again = ActiveCounts.from_dict(json.loads(json.dumps(table.to_dict())))
        np.testing.assert_array_equal(again.values, table.values)
        np.testing.assert_array_equal(again.frequencies, table.frequencies)
        original = HurdleNBChannel(
            KITCHEN,
            np.linspace(0.2, 0.8, STATES),
            np.linspace(0.5, 4.0, STATES),
            np.linspace(0.0, 3.0, STATES),
        )
        payload = json.loads(json.dumps(original.to_dict(), allow_nan=False))
        rebuilt = HurdleNBChannel.from_dict(payload)
        counts = np.array([0.0, 1.0, 6.0, 30.0])
        np.testing.assert_array_equal(rebuilt.loglik(counts), original.loglik(counts))
        with pytest.raises(ValueError, match="hurdle_nb"):
            HurdleNBChannel.from_dict({**payload, "family": "hurdle"})

    def test_the_fit_serialises_and_flags_the_bound(self) -> None:
        counts = hurdle_sample(np.random.default_rng(5), 0.5, 3.0, 1.0, 5000)
        described = fitted_on(counts).dispersion_to_dict()
        json.dumps(described, allow_nan=False)
        first = described["channels"]["kitchen_motion"][ONTOLOGY.states[0].value]
        assert first["active_windows"] == int(np.sum(counts > 0))
        assert first["at_bound"] is False
        assert first["size"] == pytest.approx(1.0 / first["dispersion"])
        unseen = described["channels"]["kitchen_motion"][ONTOLOGY.states[1].value]
        assert unseen["dispersion"] is None

    def test_tables_change_no_hurdle_digest(self) -> None:
        counts = hurdle_sample(np.random.default_rng(6), 0.5, 3.0, 1.0, 2000)
        with_table = fitted_on(counts)
        without = FittedChannels(
            ("h",),
            tuple(ONTOLOGY.states),
            12.0,
            {KITCHEN: replace(statistics_of(counts), active_counts=None)},
        )
        assert with_table.to_dict() == without.to_dict()
        assert with_table.sha256() == without.sha256()


# ----------------------------------------------------------------------------
# On simulated households: training homes only, the recursion, pooling
# ----------------------------------------------------------------------------
def simulated(seed: int) -> CasasRecording:
    sim = simulate(HouseholdConfig(days=3, seed=seed))
    return CasasRecording(
        sim.registry,
        sim.observations,
        tuple(
            ActivityInterval(e.state.value, e.start, e.end, e.state)
            for e in sim.truth.episodes
        ),
    )


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {f"sim{s}": simulated(s) for s in range(1, 5)}


class TestOnHouseholds:
    def test_the_fit_reads_the_training_households_only(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        training = {h: homes[h] for h in ("sim1", "sim2", "sim3")}
        fit = fit_channel_models(training, resolution=RESOLUTION, pseudo_windows=12.0)
        assert fit.fitted_on == ("sim1", "sim2", "sim3")
        parts = {
            h: home_statistics(r, RESOLUTION, ONTOLOGY, household=h)
            for h, r in training.items()
        }
        for channel, stats in fit.statistics.items():
            assert stats.active_counts is not None
            tables = [parts[h][channel].active_counts for h in sorted(parts)]
            combined = ActiveCounts.combine(tables)  # type: ignore[arg-type]
            np.testing.assert_array_equal(stats.active_counts.values, combined.values)
            np.testing.assert_array_equal(
                stats.active_counts.frequencies, combined.frequencies
            )
            assert stats.active_counts.frequencies.sum(axis=1) == pytest.approx(
                stats.active
            )
        assert combine_statistics(
            parts, states=fit.states, pseudo_windows=12.0
        ).sha256() == (fit.sha256())

    def test_the_family_runs_in_the_recursion(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        fit = fit_channel_models(
            {h: homes[h] for h in ("sim1", "sim2", "sim3")},
            resolution=RESOLUTION,
            pseudo_windows=12.0,
        )
        recording = homes["sim4"]
        models = fit.models(recording.registry, RESOLUTION, HURDLE_NB, ONTOLOGY)
        assert all(isinstance(m, HurdleNBChannel) for m in models.values())
        counts, _, _, _ = household_channel_counts(
            recording, RESOLUTION, ONTOLOGY, household="sim4"
        )
        loglik = total_loglik(models, counts)
        assert np.all(np.isfinite(loglik))
        _, posterior = filter_recursion(
            loglik, ONTOLOGY.transition(RESOLUTION.step), ONTOLOGY.stationary()
        )
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0)

    def test_pooling_keeps_the_pooled_mean_and_the_population_dispersion(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        fit = fit_channel_models(
            {h: homes[h] for h in ("sim1", "sim2", "sim3")},
            resolution=RESOLUTION,
            pseudo_windows=12.0,
        )
        recording = homes["sim4"]
        cutoff = min(o.timestamp for o in recording.observations)
        own = household_statistics(
            recording,
            RESOLUTION,
            ONTOLOGY,
            household="sim4",
            until=cutoff + (recording.observations[-1].timestamp - cutoff) / 2,
        )
        pooled = pool_channels(
            fit,
            recording.registry,
            own,
            household="sim4",
            resolution=RESOLUTION,
            config=PoolingConfig(288.0),
            until=cutoff + (recording.observations[-1].timestamp - cutoff) / 2,
            ontology=ONTOLOGY,
        )
        hurdle = pooled.models()
        assert all(isinstance(m, HurdleChannel) for m in hurdle.values())
        declared = fit.models(recording.registry, RESOLUTION, HURDLE_NB, ONTOLOGY)
        alphas = {c: m.dispersion for c, m in declared.items()}  # type: ignore[attr-defined]
        negative_binomial = pooled.models(dispersion=alphas)
        for channel, channel_model in negative_binomial.items():
            assert isinstance(channel_model, HurdleNBChannel)
            np.testing.assert_array_equal(channel_model.dispersion, alphas[channel])
            np.testing.assert_allclose(
                channel_model.silence, pooled.silence[channel].pooled
            )
            mean, _ = channel_model.active_moments()
            np.testing.assert_allclose(mean, pooled.activity[channel].pooled, rtol=1e-9)


def test_the_truncated_poisson_rate_is_unchanged() -> None:
    assert ztnb_rate(3.2, 0.0) == truncated_poisson_rate(3.2)
    assert float(poisson.pmf(0, 1.0)) == pytest.approx(math.exp(-1.0))
