"""Tests for the generative filter's explicit recent-history state.

They guard what the history state promises:
- zero coefficients give exactly the original filter;
- each prediction splits exactly into prior and transitions, current window,
  and history;
- history is bounded, read only from before the current window, and missing
  whenever it was not fully observed, which is never treated as silence;
- it survives snapshot and restore;
- the matched, restricted model reads nothing outside its information set.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    EvidenceResolution,
    build_feature_table,
    fit_history_model,
    nested_information_sets,
)
from sensor_modeling.datasets.history_fit import fit_coefficient
from sensor_modeling.datasets.matched_evaluation import _regular_moments
from sensor_modeling.datasets.restricted_filter import (
    HISTORY_NEEDS_LAGS,
    channel_likelihoods,
    restricted_posteriors,
    unsupported_reason,
)
from sensor_modeling.fusion import (
    HistoryAwareBayesFilter,
    HistoryConfig,
    HistoryModel,
    MultimodalBayesFilter,
)
from sensor_modeling.fusion.defaults import default_emissions
from sensor_modeling.fusion.history import (
    ANY_ROOM,
    COLD_START,
    OTHER_ROOM,
    OWN_ROOM,
    UNRELIABLE,
    expected_log1p_poisson,
    history_log_likelihood,
    relation,
)
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
)
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

I0, I1, I2, I3 = nested_information_sets()
STEP = timedelta(minutes=5)
UTC = timezone.utc
T0 = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)
ONTOLOGY = StateOntology()
SPECS = (
    SensorSpec(
        "Bathroom", Modality.MOTION, kind=ObservationKind.EVENT, room="bathroom"
    ),
    SensorSpec("Bedroom", Modality.MOTION, kind=ObservationKind.EVENT, room="bedroom"),
    SensorSpec("Kitchen", Modality.MOTION, kind=ObservationKind.EVENT, room="kitchen"),
    SensorSpec("FrontDoor", Modality.DOOR, kind=ObservationKind.EVENT, room="hall"),
)
REGISTRY = SensorRegistry.from_specs(SPECS)


def model(beta: float = 0.8, **overrides: float) -> HistoryModel:
    """A history model with every coefficient *beta*, except *overrides*."""
    null = HistoryModel.null(ONTOLOGY)
    coefficients = {key: overrides.get(key, beta) for key in null.coefficients}
    return HistoryModel(null.states, null.rooms, null.config, coefficients)


def history_filter(history: HistoryModel | None = None) -> HistoryAwareBayesFilter:
    return HistoryAwareBayesFilter(
        ONTOLOGY,
        default_emissions(REGISTRY, ONTOLOGY),
        REGISTRY,
        history or model(),
        step=STEP,
    )


def event(sensor: str, at: datetime) -> Observation:
    spec = REGISTRY.get(sensor)
    assert spec is not None
    return Observation(at, sensor, spec.modality, ObservationKind.EVENT, 1.0)


def run(
    filter_: HistoryAwareBayesFilter | MultimodalBayesFilter,
    windows: list[list[Observation]],
    start: datetime = T0,
    reliabilities: list[dict[str, float] | None] | None = None,
) -> list[np.ndarray]:
    """Update once at *start*, then once per step with each window's events."""
    filter_.update(start)
    beliefs = []
    for position, batch in enumerate(windows):
        at = start + STEP * (position + 1)
        extra = (
            {} if reliabilities is None else {"reliabilities": reliabilities[position]}
        )
        beliefs.append(filter_.update(at, batch, **extra).belief)
    return beliefs


def windows_with(pattern: dict[int, list[str]], count: int) -> list[list[Observation]]:
    """``count`` windows; window ``i`` holds one event per sensor in ``pattern[i]``."""
    return [
        [
            event(sensor, T0 + STEP * (i + 1) - timedelta(minutes=1))
            for sensor in pattern.get(i, [])
        ]
        for i in range(count)
    ]


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


class TestModel:
    def test_groups_and_the_null_model(self) -> None:
        null = HistoryModel.null(ONTOLOGY)
        assert len(null.coefficients) == 11
        assert set(null.coefficients.values()) == {0.0}
        assert "bathroom_activity|own room" in null.coefficients
        assert "away|any room" in null.coefficients
        assert relation("bathroom", "bathroom") == OWN_ROOM
        assert relation("bathroom", "kitchen") == OTHER_ROOM
        assert relation(None, "kitchen") == ANY_ROOM

    def test_invalid_models_are_refused(self) -> None:
        null = HistoryModel.null(ONTOLOGY)
        coefficients = dict(null.coefficients)
        coefficients.pop("away|any room")
        with pytest.raises(ValueError, match="groups"):
            HistoryModel(null.states, null.rooms, null.config, coefficients)
        with pytest.raises(ValueError, match="finite"):
            HistoryModel(
                null.states,
                null.rooms,
                null.config,
                {**null.coefficients, "away|any room": math.nan},
            )
        for settings in (
            {"window_steps": 0},
            {"window_steps": True},
            {"precision": 0.0},
        ):
            with pytest.raises(ValueError):
                HistoryConfig(**settings)

    def test_it_round_trips_through_json(self) -> None:
        original = model(0.3, **{"sleeping|own room": -0.4})
        payload = json.loads(json.dumps(original.to_dict(), allow_nan=False))
        again = HistoryModel.from_dict(payload)
        assert again.sha256() == original.sha256()
        assert again.coefficients == original.coefficients

    def test_the_reference_is_the_poisson_expectation(self) -> None:
        assert expected_log1p_poisson([0.0])[0] == 0.0
        mean = 2.5
        counts = np.arange(200)
        log_pmf = (
            counts * math.log(mean)
            - mean
            - np.array([math.lgamma(n + 1) for n in counts])
        )
        exact = float(np.sum(np.exp(log_pmf) * np.log1p(counts)))
        assert expected_log1p_poisson([mean])[0] == pytest.approx(exact, rel=1e-12)
        with pytest.raises(ValueError):
            expected_log1p_poisson([-1.0])

    def test_missing_history_is_neutral_and_silence_is_not(self) -> None:
        history = model(0.8)
        reference = expected_log1p_poisson(np.full(len(ONTOLOGY.states), 3.0))
        assert not history.modulation("kitchen", None, reference).any()
        silent = history.modulation("kitchen", 0, reference)
        assert (silent < 0).all()
        np.testing.assert_allclose(silent, 0.8 * (0.0 - reference))
        eta = np.zeros(3)
        assert not history_log_likelihood(4, np.ones(3), eta).any()


class TestFilter:
    def test_zero_coefficients_are_exactly_the_original_filter(self) -> None:
        source = simulated(4, days=1)
        emissions = default_emissions(source.registry, ONTOLOGY)
        original = MultimodalBayesFilter(ONTOLOGY, emissions, source.registry)
        history = HistoryAwareBayesFilter(
            ONTOLOGY,
            default_emissions(source.registry, ONTOLOGY),
            source.registry,
            HistoryModel.null(ONTOLOGY),
            step=STEP,
        )
        observations = sorted(source.observations, key=lambda o: o.timestamp)
        at, position = observations[0].timestamp, 0
        while at <= observations[-1].timestamp:
            batch = []
            while (
                position < len(observations) and observations[position].timestamp <= at
            ):
                batch.append(observations[position])
                position += 1
            np.testing.assert_allclose(
                original.update(at, batch).belief,
                history.update(at, batch).belief,
                atol=1e-12,
            )
            at += STEP

    def test_every_prediction_decomposes_exactly(self) -> None:
        filter_ = history_filter()
        run(filter_, windows_with({0: ["Kitchen"], 2: ["Bathroom"], 5: ["Kitchen"]}, 9))
        decomposition = filter_.explain()
        assert decomposition is not None and decomposition.history_applied
        total = (
            decomposition.prior_transition
            + decomposition.current
            + decomposition.history
        )
        np.testing.assert_allclose(
            np.exp(total - np.logaddexp.reduce(total)),
            decomposition.posterior,
            atol=1e-12,
        )
        odds = decomposition.log_odds(S.KITCHEN_ACTIVITY, S.AWAY)
        assert odds["posterior"] == pytest.approx(
            odds["prior_transition"] + odds["current"] + odds["history"]
        )
        by_channel = decomposition.explain()["history_by_channel"]
        assert isinstance(by_channel, dict)
        assert set(by_channel) == set(filter_.channels)
        channel_total = sum(np.array(c.contribution) for c in decomposition.channels)
        np.testing.assert_allclose(channel_total, decomposition.history, atol=1e-12)

    def test_cold_start(self) -> None:
        filter_ = history_filter()
        filter_.update(T0)
        first = filter_.explain()
        assert first is not None and not first.history_applied
        for step_number in range(1, 4):
            filter_.update(T0 + STEP * step_number)
            decomposition = filter_.explain()
            assert decomposition is not None
            assert all(c.count is None for c in decomposition.channels)
            assert all(c.reason == COLD_START for c in decomposition.channels)
            assert not decomposition.history.any()
        filter_.update(T0 + STEP * 4)
        filter_.update(T0 + STEP * 5)
        decomposition = filter_.explain()
        assert decomposition is not None
        assert all(c.count == 0 and c.reason is None for c in decomposition.channels)
        assert decomposition.history.any()

    def test_long_silence_is_observed_bounded_and_finite(self) -> None:
        filter_ = history_filter()
        beliefs = run(filter_, windows_with({0: ["Kitchen", "Bathroom"]}, 300))
        assert np.all(np.isfinite(beliefs[-1]))
        assert len(filter_.snapshot()["history"]) == 3  # type: ignore[arg-type]
        decomposition = filter_.explain()
        assert decomposition is not None
        assert all(c.count == 0 for c in decomposition.channels)
        assert (
            np.all(np.isfinite(decomposition.history)) and decomposition.history.any()
        )

    def test_an_unreliable_sensor_makes_its_channel_missing_not_silent(self) -> None:
        filter_ = history_filter()
        reliabilities: list[dict[str, float] | None] = [None] * 8
        reliabilities[5] = {"Kitchen": 0.0}
        run(filter_, windows_with({}, 8), reliabilities=reliabilities)
        decomposition = filter_.explain()
        assert decomposition is not None
        by_channel = {c.channel: c for c in decomposition.channels}
        assert by_channel["kitchen_motion"].count is None
        assert by_channel["kitchen_motion"].reason == UNRELIABLE
        assert not any(by_channel["kitchen_motion"].contribution)
        assert by_channel["bathroom_motion"].count == 0
        assert any(by_channel["bathroom_motion"].contribution)

    def test_only_instrumented_channels_have_a_history(self) -> None:
        assert history_filter().channels == (
            "bathroom_motion",
            "bedroom_motion",
            "hall_door",
            "kitchen_motion",
        )

    @pytest.mark.parametrize("elapsed", [STEP * 3, timedelta(minutes=2)])
    def test_an_irregular_update_truncates_the_history(
        self, elapsed: timedelta
    ) -> None:
        filter_ = history_filter()
        run(filter_, windows_with({}, 6))
        resumed = T0 + STEP * 6 + elapsed
        filter_.update(resumed)
        irregular = filter_.explain()
        assert irregular is not None and not irregular.history_applied
        filter_.update(resumed + STEP)
        after = filter_.explain()
        assert after is not None and after.history_applied
        assert all(c.count is None for c in after.channels)
        for k in range(2, 5):
            filter_.update(resumed + STEP * k)
        recovered = filter_.explain()
        assert recovered is not None
        assert all(c.count == 0 for c in recovered.channels)

    def test_the_history_is_read_before_the_current_window(self) -> None:
        filter_ = history_filter()
        run(
            filter_,
            windows_with(
                {3: ["Kitchen"], 4: ["Kitchen"], 5: ["Kitchen"], 6: ["Kitchen"]}, 7
            ),
        )
        decomposition = filter_.explain()
        assert decomposition is not None
        kitchen = next(
            c for c in decomposition.channels if c.channel == "kitchen_motion"
        )
        assert kitchen.count == 3  # windows 3 to 5, not the current window 6

    def test_future_observations_change_no_earlier_prediction(self) -> None:
        base = windows_with({1: ["Kitchen"], 4: ["Bathroom"]}, 10)
        changed = (
            base[:6]
            + windows_with({6: ["Bedroom"], 7: ["FrontDoor"], 8: ["Kitchen"]}, 10)[6:]
        )
        first, second = run(history_filter(), base), run(history_filter(), changed)
        for position in range(6):
            np.testing.assert_array_equal(first[position], second[position])
        assert not np.allclose(first[7], second[7])

    def test_replay_is_deterministic(self) -> None:
        windows = windows_with(
            {0: ["Kitchen"], 3: ["Bathroom", "Kitchen"], 7: ["FrontDoor"]}, 12
        )
        a, b = history_filter(), history_filter()
        run(a, windows)
        run(b, windows)
        assert a.explain().to_dict() == b.explain().to_dict()  # type: ignore[union-attr]

    def test_snapshot_and_restore_continue_exactly(self) -> None:
        windows = windows_with(
            {0: ["Kitchen"], 3: ["Bathroom"], 6: ["Kitchen"], 9: ["Bedroom"]}, 12
        )
        whole = history_filter()
        expected = run(whole, windows)
        first = history_filter()
        run(first, windows[:5])
        state = json.loads(json.dumps(first.snapshot()))
        second = history_filter()
        second.restore(state)
        for position in range(5, 12):
            at = T0 + STEP * (position + 1)
            np.testing.assert_allclose(
                second.update(at, windows[position]).belief,
                expected[position],
                atol=1e-12,
            )

    def test_a_snapshot_from_another_model_step_or_household_is_refused(self) -> None:
        source = history_filter()
        run(source, windows_with({0: ["Kitchen"]}, 5))
        state = source.snapshot()
        with pytest.raises(ValueError, match="history model"):
            history_filter(model(0.1)).restore(state)
        other_step = HistoryAwareBayesFilter(
            ONTOLOGY,
            default_emissions(REGISTRY, ONTOLOGY),
            REGISTRY,
            model(),
            step=STEP * 2,
        )
        with pytest.raises(ValueError, match="step"):
            other_step.restore(state)
        other_home = SensorRegistry.from_specs(SPECS[:2])
        elsewhere = HistoryAwareBayesFilter(
            ONTOLOGY,
            default_emissions(other_home, ONTOLOGY),
            other_home,
            model(),
            step=STEP,
        )
        with pytest.raises(ValueError, match="channels"):
            elsewhere.restore(state)

    def test_a_new_household_starts_cold(self) -> None:
        busy = history_filter()
        run(busy, windows_with({i: ["Kitchen"] for i in range(10)}, 10))
        fresh = history_filter()
        fresh.update(T0 + STEP * 20)
        decomposition = fresh.explain()
        assert decomposition is not None and not decomposition.history_applied
        assert all(c.reason == COLD_START for c in decomposition.channels)

    def test_reset_clears_the_history(self) -> None:
        filter_ = history_filter()
        run(filter_, windows_with({}, 6))
        filter_.reset()
        assert filter_.snapshot()["history"] == []
        assert filter_.explain() is None


class TestRestricted:
    def test_support_needs_the_lagged_windows(self) -> None:
        assert unsupported_reason(I2, history_steps=3) is None
        assert unsupported_reason(I0, history_steps=3) == HISTORY_NEEDS_LAGS
        shallow = nested_information_sets(EvidenceResolution(history_steps=2))[2]
        assert unsupported_reason(shallow, history_steps=3) == HISTORY_NEEDS_LAGS
        source = simulated(5)
        moments = _regular_moments(source, STEP)[:10]
        table = build_feature_table(source, I0, moments, household="h")
        terms, _ = channel_likelihoods(source.registry, I0.resolution)
        with pytest.raises(ValueError, match="history state"):
            restricted_posteriors(table, terms, history_model=model())

    def test_it_is_the_online_filter_fed_only_the_declared_windows(self) -> None:
        source = simulated(6)
        history = model(0.7, **{"sleeping|own room": -0.5, "away|any room": 1.2})
        moments = _regular_moments(source, STEP)
        table = build_feature_table(source, I2, moments, household="h")
        terms, _ = channel_likelihoods(source.registry, I2.resolution)
        posteriors = restricted_posteriors(table, terms, history_model=history)
        routed = {sensor for term in terms.values() for sensor in term.sensors}
        emissions = [
            e
            for e in default_emissions(source.registry, ONTOLOGY)
            if e.sensor_id in routed
        ]
        observations = sorted(source.observations, key=lambda o: o.timestamp)
        checked = 0
        for row in range(4, len(moments), 29):
            moment = moments[row]
            online = HistoryAwareBayesFilter(
                ONTOLOGY, emissions, source.registry, history, step=STEP
            )
            online.update(moment - STEP * 4)
            for lag in range(3, -1, -1):
                end = moment - STEP * lag
                estimate = online.update(
                    end, [o for o in observations if end - STEP < o.timestamp <= end]
                )
            np.testing.assert_allclose(estimate.belief, posteriors[row], atol=1e-10)
            checked += 1
        assert checked > 10

    def test_a_null_model_is_the_original_restricted_model(self) -> None:
        source = simulated(7)
        moments = _regular_moments(source, STEP)
        table = build_feature_table(source, I3, moments, household="h")
        terms, _ = channel_likelihoods(source.registry, I3.resolution)
        from sensor_modeling.datasets import fit_periodic_prior, hour_state_counts

        prior = fit_periodic_prior({"h": hour_state_counts(source, moments, step=STEP)})
        np.testing.assert_allclose(
            restricted_posteriors(
                table, terms, periodic_prior=prior, history_model=HistoryModel.null()
            ),
            restricted_posteriors(table, terms, periodic_prior=prior),
            atol=1e-12,
        )

    def test_history_before_the_recording_is_missing(self) -> None:
        source = simulated(8)
        moments = _regular_moments(source, STEP)[:3]
        table = build_feature_table(source, I2, moments, household="h")
        terms, _ = channel_likelihoods(source.registry, I2.resolution)
        np.testing.assert_allclose(
            restricted_posteriors(table, terms, history_model=model()),
            restricted_posteriors(table, terms),
            atol=1e-12,
        )


class TestFit:
    def test_it_recovers_a_known_coefficient(self) -> None:
        rng = np.random.default_rng(0)
        z = rng.normal(0.0, 1.0, 50_000)
        mu = rng.uniform(0.1, 3.0, z.size)
        y = rng.poisson(mu * np.exp(0.6 * z))
        assert fit_coefficient(z, y, mu, precision=1.0) == pytest.approx(0.6, abs=0.02)
        assert fit_coefficient(np.empty(0), np.empty(0), np.empty(0), 1.0) == 0.0
        assert abs(fit_coefficient(z[:50], y[:50], mu[:50], precision=1e6)) < 1e-3

    def test_fitting_reads_only_the_given_households_deterministically(self) -> None:
        homes = {name: simulated(seed) for name, seed in (("a", 1), ("b", 2))}
        fitted = fit_history_model(homes)
        assert fitted.fitted_on == ("a", "b")
        assert (
            fit_history_model(dict(reversed(list(homes.items())))).sha256()
            == fitted.sha256()
        )
        assert fit_history_model({"a": homes["a"]}).sha256() != fitted.sha256()
        assert sum(fitted.support.values()) > 0
        with pytest.raises(ValueError, match="at least one"):
            fit_history_model({})
