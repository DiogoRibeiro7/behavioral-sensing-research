"""Tests for the hierarchical periodic state prior.

They guard what makes the prior usable as a matched generative component:
- it is a proper, normalised probability over states at every hour;
- it is periodic and has no seam at midnight;
- it follows the clock through daylight-saving changes;
- households are shrunk toward the population, and exactly to it without data;
- fitting is deterministic and reads no labels it was not given;
- inside the filter it is the production filter started from ``π_h``, and
  reads only the hour the information set declares.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    PeriodicPriorConfig,
    PeriodicStatePrior,
    build_feature_table,
    fit_periodic_prior,
    hour_state_counts,
    nested_information_sets,
)
from sensor_modeling.datasets.matched_evaluation import _regular_moments
from sensor_modeling.datasets.periodic_prior import hour_basis
from sensor_modeling.datasets.restricted_filter import (
    PERIODIC_PRIOR_NEEDS_HOUR,
    TIME_OF_DAY_UNSUPPORTED,
    channel_likelihoods,
    restricted_posteriors,
    unsupported_reason,
)
from sensor_modeling.datasets.time_features import hour_angle
from sensor_modeling.fusion.defaults import default_emissions
from sensor_modeling.fusion.filter import MultimodalBayesFilter
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
)
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology, stationary_distribution
from sensor_modeling.states.ontology import DEFAULT_STATES

I0, I1, I2, I3 = nested_information_sets()
STEP = timedelta(minutes=5)
LA = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc
HOURS = np.arange(24)
SLEEP = DEFAULT_STATES.index(S.SLEEPING)


def true_coefficients(seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """Coefficients of a known two-harmonic prior, with a strong night sleep."""
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(0.0, 0.4 * scale, (len(DEFAULT_STATES), 5))
    coefficients[SLEEP, :3] = [0.5, 0.2, 1.8 * scale]  # sleeping peaks near midnight
    return coefficients


def expected_counts(coefficients: np.ndarray, hours_per_hour: float) -> np.ndarray:
    """Labelled hours by hour and state, exactly as the prior would allocate them."""
    logits = hour_basis(HOURS, 2) @ coefficients.T
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    counts: np.ndarray = hours_per_hour * probabilities
    return counts


def population(**settings: float) -> PeriodicStatePrior:
    config = PeriodicPriorConfig(**settings)  # type: ignore[arg-type]
    return fit_periodic_prior(
        {"a": expected_counts(true_coefficients(), 300.0)}, config=config
    )


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float((p * np.log(p / q)).sum())


def simulated(seed: int, days: int = 2) -> CasasRecording:
    sim = simulate(HouseholdConfig(days=days, seed=seed))
    return CasasRecording(
        sim.registry,
        sim.observations,
        tuple(
            ActivityInterval(e.state.value, e.start, e.end, e.state)
            for e in sim.truth.episodes
        ),
    )


class TestConfig:
    def test_defaults_and_round_trip(self) -> None:
        config = PeriodicPriorConfig()
        assert (config.harmonics, config.population_precision) == (2, 1.0)
        assert config.household_precision == 24.0
        payload = json.loads(json.dumps(config.to_dict(), allow_nan=False))
        assert PeriodicPriorConfig.from_dict(payload) == config

    @pytest.mark.parametrize(
        "settings",
        [
            {"harmonics": 0},
            {"harmonics": 12},
            {"harmonics": True},
            {"population_precision": 0.0},
            {"household_precision": -1.0},
            {"household_precision": math.inf},
            {"household_precision": math.nan},
        ],
    )
    def test_invalid_settings_are_refused(self, settings: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            PeriodicPriorConfig(**settings)  # type: ignore[arg-type]


class TestProbabilities:
    def test_every_hour_is_a_normalised_distribution(self) -> None:
        prior = population()
        adapted = prior.adapt("h", expected_counts(true_coefficients(1), 5.0))
        for probabilities in (prior.hourly(), adapted.hourly("h")):
            assert probabilities.shape == (24, len(DEFAULT_STATES))
            assert (probabilities > 0.0).all()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)

    def test_it_recovers_a_known_prior(self) -> None:
        truth = expected_counts(true_coefficients(), 1.0)
        fitted = fit_periodic_prior(
            {"a": expected_counts(true_coefficients(), 5000.0)}
        ).hourly()
        np.testing.assert_allclose(fitted, truth, atol=2e-3)


class TestPeriodicity:
    def test_the_prior_repeats_every_24_hours(self) -> None:
        prior = population()
        angles = np.linspace(0.0, 2.0 * np.pi, 97)
        once = prior.log_probabilities_at(angles)
        for turns in (1, 2, -3):
            np.testing.assert_allclose(
                prior.log_probabilities_at(angles + 2.0 * np.pi * turns),
                once,
                atol=1e-10,
            )

    def test_it_is_continuous_across_midnight(self) -> None:
        prior = population()
        before = prior.log_probabilities_at([2.0 * np.pi - 1e-9])
        after = prior.log_probabilities_at([1e-9])
        np.testing.assert_allclose(before, after, atol=1e-7)
        np.testing.assert_allclose(
            prior.log_probabilities([23, 0]),
            prior.log_probabilities_at(hour_angle([23, 0])),
        )

    @pytest.mark.parametrize("shift", [1, 7, 12])
    def test_midnight_is_not_a_special_hour(self, shift: int) -> None:
        """Rotating the data around the clock rotates the fitted prior with it."""
        counts = expected_counts(true_coefficients(2), 40.0)
        fitted = fit_periodic_prior({"a": counts}).hourly()
        rotated = fit_periodic_prior({"a": np.roll(counts, shift, axis=0)}).hourly()
        np.testing.assert_allclose(rotated, np.roll(fitted, shift, axis=0), atol=1e-8)


class TestHouseholds:
    def test_zero_household_data_gives_exactly_the_population(self) -> None:
        prior = population()
        adapted = prior.adapt("h", np.zeros((24, len(DEFAULT_STATES))))
        assert not adapted.deviations["h"].any()
        np.testing.assert_array_equal(adapted.hourly("h"), prior.hourly())

    def test_an_unseen_household_gets_the_population(self) -> None:
        prior = population().adapt("seen", expected_counts(true_coefficients(3), 20.0))
        np.testing.assert_array_equal(prior.hourly("unseen"), prior.hourly())
        np.testing.assert_array_equal(prior.hourly(None), prior.hourly())
        assert not np.allclose(prior.hourly("seen"), prior.hourly())

    def test_stronger_pooling_shrinks_the_deviation(self) -> None:
        own = expected_counts(true_coefficients(4, scale=1.5), 20.0)
        norms = [
            float(
                np.linalg.norm(
                    population(household_precision=p).adapt("h", own).deviations["h"]
                )
            )
            for p in (0.1, 1.0, 10.0, 100.0, 1000.0, 1e8)
        ]
        assert norms == sorted(norms, reverse=True)
        assert norms[-1] < 1e-4

    def test_more_household_data_moves_further_from_the_population(self) -> None:
        prior = population()
        empirical = expected_counts(true_coefficients(5, scale=1.5), 1.0)
        distances = []
        for hours in (1.0, 10.0, 100.0, 1000.0):
            adapted = prior.adapt("h", empirical * hours).hourly("h")
            distances.append(kl(empirical, adapted))
        assert distances == sorted(distances, reverse=True)
        assert kl(empirical, prior.hourly()) > distances[0]

    def test_the_adapted_prior_lies_between_population_and_household(self) -> None:
        prior = population()
        empirical = expected_counts(true_coefficients(6, scale=1.5), 1.0)
        adapted = prior.adapt("h", empirical * 20.0).hourly("h")
        unpooled = population(household_precision=1e-6).adapt("h", empirical * 20.0)
        assert kl(empirical, adapted) < kl(empirical, prior.hourly())
        assert kl(empirical, unpooled.hourly("h")) < kl(empirical, adapted)


class TestFitting:
    def test_fitting_is_deterministic_and_order_free(self) -> None:
        counts = {
            name: expected_counts(true_coefficients(seed), 15.0)
            for name, seed in (("a", 7), ("b", 8), ("c", 9))
        }
        first = fit_periodic_prior(counts)
        again = fit_periodic_prior(dict(reversed(list(counts.items()))))
        assert first.sha256() == again.sha256()
        assert first.fitted_on == ("a", "b", "c")
        assert first.labelled_hours == pytest.approx(15.0 * 24 * 3)

    def test_only_the_households_given_are_read(self) -> None:
        counts = {"a": expected_counts(true_coefficients(7), 15.0)}
        fitted = fit_periodic_prior(counts)
        held_out = {**counts, "b": expected_counts(true_coefficients(8), 15.0)}
        assert fit_periodic_prior(held_out).sha256() != fitted.sha256()
        assert fit_periodic_prior(counts).sha256() == fitted.sha256()

    @pytest.mark.parametrize(
        "counts",
        [
            {},
            {"a": np.zeros((24, 7))},
            {"a": np.ones((23, 7))},
            {"a": -np.ones((24, 7))},
        ],
    )
    def test_unusable_counts_are_refused(self, counts: dict[str, np.ndarray]) -> None:
        with pytest.raises(ValueError):
            fit_periodic_prior(counts)

    def test_it_round_trips_through_json(self) -> None:
        prior = population().adapt("h", expected_counts(true_coefficients(9), 3.0))
        payload = json.loads(json.dumps(prior.to_dict(), allow_nan=False))
        again = PeriodicStatePrior.from_dict(payload)
        assert again.sha256() == prior.sha256()
        np.testing.assert_array_equal(again.hourly("h"), prior.hourly("h"))


class TestCountsAndTime:
    def test_counts_follow_the_clock_through_the_fall_back_night(self) -> None:
        recording = la_recording()
        instants = [
            datetime(2011, 11, 6, 6, 0, tzinfo=UTC) + timedelta(minutes=30) * k
            for k in range(13)
        ]
        moments = [instant.astimezone(LA) for instant in instants]
        counts = hour_state_counts(recording, moments, step=timedelta(minutes=30))
        per_hour = counts.sum(axis=1)
        assert per_hour[1] == pytest.approx(2.0)  # 01:00-01:59 happened twice
        assert per_hour[0] == pytest.approx(1.0) and per_hour[2] == pytest.approx(1.0)
        assert per_hour.sum() == pytest.approx(len(moments) * 0.5)

    def test_both_fall_back_occurrences_get_the_same_prior(self) -> None:
        prior = population()
        first = datetime(2011, 11, 6, 8, 30, tzinfo=UTC).astimezone(LA)
        second = datetime(2011, 11, 6, 9, 30, tzinfo=UTC).astimezone(LA)
        assert (first.fold, second.fold, first.hour, second.hour) == (0, 1, 1, 1)
        np.testing.assert_array_equal(
            prior.probabilities([first.hour]), prior.probabilities([second.hour])
        )

    def test_a_grid_moment_in_the_skipped_hour_is_read_as_written(self) -> None:
        recording = la_recording(day=13, month=3)
        moment = datetime(2011, 3, 13, 1, 30, tzinfo=LA) + timedelta(minutes=60)
        counts = hour_state_counts(recording, [moment], step=STEP)
        assert counts.sum(axis=1)[2] == pytest.approx(STEP / timedelta(hours=1))

    def test_labels_after_the_cutoff_are_not_read(self) -> None:
        source = simulated(3)
        moments = _regular_moments(source, STEP)
        cutoff = moments[len(moments) // 2]
        relabelled = CasasRecording(
            source.registry,
            source.observations,
            tuple(
                ActivityInterval("x", a.start, a.end, S.AWAY) if a.start > cutoff else a
                for a in source.activities
            ),
        )
        early = hour_state_counts(source, moments, step=STEP, until=cutoff)
        np.testing.assert_array_equal(
            early, hour_state_counts(relabelled, moments, step=STEP, until=cutoff)
        )
        assert early.sum() < hour_state_counts(source, moments, step=STEP).sum()
        prior = population()
        np.testing.assert_array_equal(
            prior.adapt("h", early).hourly("h"),
            prior.adapt(
                "h", hour_state_counts(relabelled, moments, step=STEP, until=cutoff)
            ).hourly("h"),
        )


def la_recording(day: int = 6, month: int = 11) -> CasasRecording:
    """A Los Angeles recording labelled asleep across a daylight-saving night."""
    spec = SensorSpec(
        "Bedroom", Modality.MOTION, kind=ObservationKind.EVENT, room="bedroom"
    )
    start = datetime(2011, month, day - 1, 22, 0, tzinfo=LA)
    return CasasRecording(
        SensorRegistry.from_specs([spec]),
        (Observation(start, "Bedroom", Modality.MOTION, ObservationKind.EVENT, 1.0),),
        (
            ActivityInterval(
                "Sleep", start, datetime(2011, month, day, 6, 0, tzinfo=LA), S.SLEEPING
            ),
        ),
    )


class TestOntology:
    def test_the_equilibrium_at_every_hour_is_the_prior(self) -> None:
        prior = population()
        ontology = prior.ontology()
        for hour in range(24):
            np.testing.assert_allclose(
                stationary_distribution(ontology.generator_at_hour(hour)),
                prior.hourly()[hour],
                atol=1e-10,
            )

    def test_the_mean_exit_rate_is_unchanged(self) -> None:
        prior = population()
        base = StateOntology()
        ontology = prior.ontology(base)
        expected = float((base.stationary() * -np.diag(base.generator)).sum())
        for hour in range(24):
            rate = (
                prior.hourly()[hour] * -np.diag(ontology.generator_at_hour(hour))
            ).sum()
            assert rate == pytest.approx(expected, rel=1e-10)

    def test_the_hour_accessor_matches_the_moment_accessor(self) -> None:
        ontology = population().ontology()
        for hour in (0, 5, 23):
            at = datetime(2024, 3, 1, hour, 30, tzinfo=UTC)
            np.testing.assert_array_equal(
                ontology.transition_at_hour(STEP, hour), ontology.transition(STEP, at)
            )

    def test_the_original_ontology_is_unchanged(self) -> None:
        base = StateOntology()
        assert base.circadian is None
        for hour in (None, 3, 17):
            np.testing.assert_array_equal(
                base.transition_at_hour(STEP, hour), base.transition(STEP)
            )

    def test_unsuitable_bases_are_refused(self) -> None:
        prior = population()
        with pytest.raises(ValueError, match="already has a circadian"):
            prior.ontology(prior.ontology())
        with pytest.raises(ValueError, match="states differ"):
            prior.ontology(StateOntology(states=DEFAULT_STATES[:-1]))


class TestRestrictedFilter:
    def test_support_follows_the_declared_hour(self) -> None:
        assert unsupported_reason(I1, periodic_prior=True) is None
        assert unsupported_reason(I3, periodic_prior=True) is None
        assert unsupported_reason(I0, periodic_prior=True) == PERIODIC_PRIOR_NEEDS_HOUR
        assert unsupported_reason(I2, periodic_prior=True) == PERIODIC_PRIOR_NEEDS_HOUR
        assert unsupported_reason(I1) == TIME_OF_DAY_UNSUPPORTED

    @pytest.mark.parametrize(
        ("information_set", "depth"), [(I1, 0), (I3, 3)], ids=["I1", "I3"]
    )
    def test_it_is_the_production_filter_started_from_the_prior(
        self, information_set: object, depth: int
    ) -> None:
        source = simulated(5)
        other = simulated(6)
        prior = fit_periodic_prior(
            {
                "other": hour_state_counts(
                    other, _regular_moments(other, STEP), step=STEP
                )
            }
        )
        moments = _regular_moments(source, STEP)
        table = build_feature_table(source, information_set, moments, household="h")  # type: ignore[arg-type]
        terms, _ = channel_likelihoods(source.registry, I3.resolution)
        posteriors = restricted_posteriors(table, terms, periodic_prior=prior)

        ontology = prior.ontology()
        routed = {sensor for term in terms.values() for sensor in term.sensors}
        emissions = [
            e
            for e in default_emissions(source.registry, ontology)
            if e.sensor_id in routed
        ]
        observations = sorted(source.observations, key=lambda o: o.timestamp)
        checked = 0
        for row in range(depth + 1, len(moments), 23):
            moment = moments[row]
            if (moment - STEP * depth).hour != moment.hour:
                continue  # the filter would switch generator inside the span
            production = MultimodalBayesFilter(
                ontology,
                emissions,
                source.registry,
                prior=prior.probabilities([moment.hour])[0],
            )
            production.update(moment - STEP * (depth + 1))
            for lag in range(depth, -1, -1):
                end = moment - STEP * lag
                estimate = production.update(
                    end, [o for o in observations if end - STEP < o.timestamp <= end]
                )
            np.testing.assert_allclose(estimate.belief, posteriors[row], atol=1e-12)
            checked += 1
        assert checked > 10

    def test_it_reads_the_declared_hour_and_nothing_finer(self) -> None:
        spec = SensorSpec(
            "Kitchen", Modality.MOTION, kind=ObservationKind.EVENT, room="kitchen"
        )
        moments = [
            datetime(2024, 3, 1, h, m, tzinfo=UTC)
            for h, m in ((10, 25), (10, 55), (22, 25))
        ]
        events = [(m - timedelta(minutes=2)) for m in moments]
        source = CasasRecording(
            SensorRegistry.from_specs([spec]),
            tuple(
                Observation(at, "Kitchen", Modality.MOTION, ObservationKind.EVENT, 1.0)
                for at in [datetime(2024, 3, 1, 0, 0, tzinfo=UTC), *events]
            ),
            (),
        )
        table = build_feature_table(source, I1, moments, household="h")
        terms, _ = channel_likelihoods(source.registry, I1.resolution)
        posteriors = restricted_posteriors(table, terms, periodic_prior=population())
        np.testing.assert_array_equal(posteriors[0], posteriors[1])  # same hour
        assert not np.allclose(posteriors[0], posteriors[2])  # another hour

    def test_the_prior_needs_the_hour_and_the_hour_needs_the_prior(self) -> None:
        source = simulated(7)
        moments = _regular_moments(source, STEP)[:20]
        terms, _ = channel_likelihoods(source.registry, I0.resolution)
        without_hour = build_feature_table(source, I2, moments, household="h")
        with pytest.raises(ValueError, match="conditions on the local hour"):
            restricted_posteriors(without_hour, terms, periodic_prior=population())
        with_hour = build_feature_table(source, I3, moments, household="h")
        with pytest.raises(ValueError, match="time-of-day"):
            restricted_posteriors(with_hour, terms)

    def test_an_unseen_household_is_scored_with_the_population(self) -> None:
        source = simulated(8)
        moments = _regular_moments(source, STEP)[:50]
        table = build_feature_table(source, I3, moments, household="h")
        terms, _ = channel_likelihoods(source.registry, I3.resolution)
        prior = population().adapt(
            "someone else", expected_counts(true_coefficients(1), 30.0)
        )
        np.testing.assert_array_equal(
            restricted_posteriors(table, terms, periodic_prior=prior, household="h"),
            restricted_posteriors(table, terms, periodic_prior=prior),
        )
