"""Tests for the fitted channel observation models and their experiment.

They guard:
- the hurdle likelihood: zeros, one and several active channels, missing
  windows, normalisation, extreme counts, and its equivalence to a Poisson;
- the fitting: its estimators, its shrinkage, and that only training
  households reach it;
- that the declared model is unchanged and still available;
- the frozen protocol, and a reproducible record and summary.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.special import gammaln

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    HouseholdSplit,
    nested_information_sets,
)
from sensor_modeling.datasets.channel_models import (
    DECLARED,
    HURDLE,
    POISSON,
    ChannelStatistics,
    FittedChannels,
    HurdleChannel,
    filter_recursion,
    fit_channel_models,
    household_channel_counts,
    poisson_channel,
    total_loglik,
    truncated_poisson_rate,
)
from sensor_modeling.datasets.information_sets import (
    EvidenceChannel,
    build_feature_table,
    evidence_column,
)
from sensor_modeling.datasets.matched_evaluation import _regular_moments
from sensor_modeling.datasets.rates_experiment import (
    CELLS,
    CRITERIA,
    ESTIMANDS,
    MECHANISMS,
    RECURSION,
    FittedRatesProtocol,
    FittedRatesResult,
    check_frozen_protocol,
    conclusion,
    declared_protocol,
    fold_fits,
    run_fitted_rates,
)
from sensor_modeling.datasets.rates_summary import render_summary
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.restricted_filter import (
    channel_likelihoods,
    restricted_posteriors,
)
from sensor_modeling.datasets.silence_dependence import filter_trajectory
from sensor_modeling.evaluation import load_record
from sensor_modeling.observations import Modality
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
PROTOCOL = ROOT / "artifacts" / "phase3" / "fitted_rates_protocol.json"
ONTOLOGY = StateOntology()
STATES = ONTOLOGY.size
STEP = timedelta(minutes=5)
KITCHEN = EvidenceChannel("kitchen", Modality.MOTION)
SETS = dict(zip(("I0", "I1", "I2", "I3"), nested_information_sets()))
RESOLUTION = SETS["I0"].resolution
FOLDS = (
    HouseholdSplit("a", train=("sim2", "sim4"), test=("sim1", "sim3")),
    HouseholdSplit("b", train=("sim1", "sim3"), test=("sim2", "sim4")),
)


def hurdle(silence: Any = None, rate: Any = None) -> HurdleChannel:
    silence = np.linspace(0.3, 0.9, STATES) if silence is None else silence
    rate = np.linspace(0.5, 4.0, STATES) if rate is None else rate
    return HurdleChannel(KITCHEN, np.asarray(silence), np.asarray(rate))


def normalised(values: np.ndarray) -> np.ndarray:
    shifted = np.exp(values - values.max(axis=-1, keepdims=True))
    result: np.ndarray = shifted / shifted.sum(axis=-1, keepdims=True)
    return result


# ----------------------------------------------------------------------------
# The hurdle likelihood
# ----------------------------------------------------------------------------
class TestHurdleLikelihood:
    def test_zero_events_are_evidence_through_the_silence_probability(self) -> None:
        model = hurdle()
        np.testing.assert_allclose(
            model.loglik(np.array([0.0]))[0], np.log(model.silence)
        )

    def test_one_active_window(self) -> None:
        model = hurdle()
        n = 3.0
        expected = (
            np.log1p(-model.silence)
            - np.log(np.expm1(model.rate))
            + n * np.log(model.rate)
        )
        np.testing.assert_allclose(model.loglik(np.array([n]))[0], expected)

    def test_the_decomposition_adds_up_to_the_likelihood(self) -> None:
        model = hurdle()
        for count in (0.0, 1.0, 5.0):
            parts = model.decompose(count)
            np.testing.assert_allclose(
                parts["silence"] + parts["activity"], model.loglik(np.array([count]))[0]
            )
        np.testing.assert_array_equal(model.decompose(0.0)["activity"], 0.0)

    def test_the_distribution_sums_to_one(self) -> None:
        model = hurdle()
        counts = np.arange(0, 200, dtype=float)
        log_pmf = model.loglik(counts) - gammaln(counts + 1.0)[:, None]
        np.testing.assert_allclose(np.exp(log_pmf).sum(axis=0), 1.0, atol=1e-10)

    def test_it_is_exactly_a_poisson_when_silence_matches(self) -> None:
        rate = np.linspace(0.05, 6.0, STATES)
        counts = np.array([0.0, 1.0, 2.0, 9.0, 40.0])
        as_hurdle = hurdle(np.exp(-rate), rate).loglik(counts)
        as_poisson = poisson_channel(KITCHEN, rate).loglik(counts)
        np.testing.assert_allclose(normalised(as_hurdle), normalised(as_poisson))

    def test_extreme_counts_and_parameters_stay_finite(self) -> None:
        model = hurdle(
            np.array([1e-12, 0.5, 1.0 - 1e-12, 0.2, 0.9, 0.99, 0.01]),
            np.array([1e-9, 1e-3, 1.0, 50.0, 300.0, 5.0, 0.5]),
        )
        values = model.loglik(np.array([0.0, 1.0, 1e6]))
        assert np.all(np.isfinite(values))
        assert np.all(np.isfinite(normalised(values)))
        np.testing.assert_allclose(normalised(values).sum(axis=1), 1.0)

    @pytest.mark.parametrize(
        ("silence", "rate", "message"),
        [
            (np.full(STATES, 1.0), np.ones(STATES), "silence"),
            (np.full(STATES, 0.0), np.ones(STATES), "silence"),
            (np.full(STATES, 0.5), np.zeros(STATES), "rates"),
            (np.full(STATES, 0.5), np.ones(STATES - 1), "one value per state"),
        ],
    )
    def test_invalid_parameters_are_refused(
        self, silence: np.ndarray, rate: np.ndarray, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            HurdleChannel(KITCHEN, silence, rate)

    @pytest.mark.parametrize("counts", [[-1.0], [0.5], [math.nan], [math.inf]])
    def test_invalid_counts_are_refused(self, counts: list[float]) -> None:
        with pytest.raises(ValueError, match="counts"):
            hurdle().loglik(np.array(counts))

    @pytest.mark.parametrize("mean", [1.0 + 1e-6, 1.5, 3.0, 40.0])
    def test_the_truncated_rate_has_the_requested_mean(self, mean: float) -> None:
        rate = truncated_poisson_rate(mean)
        assert rate / -math.expm1(-rate) == pytest.approx(mean, rel=1e-9)

    def test_a_mean_of_one_is_the_limit_and_below_one_is_refused(self) -> None:
        assert truncated_poisson_rate(1.0) < 1e-6
        with pytest.raises(ValueError, match="at least 1"):
            truncated_poisson_rate(0.9)


# ----------------------------------------------------------------------------
# In the restricted model and the recursion
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


@pytest.fixture(scope="module")
def fitted(homes: dict[str, CasasRecording]) -> FittedChannels:
    return fit_channel_models(
        {h: homes[h] for h in ("sim2", "sim4")},
        resolution=RESOLUTION,
        pseudo_windows=12.0,
        ontology=ONTOLOGY,
    )


class TestInTheModel:
    def test_several_active_channels_multiply(
        self, homes: dict[str, CasasRecording], fitted: FittedChannels
    ) -> None:
        recording = homes["sim1"]
        moments = _regular_moments(recording, STEP)[:300]
        table = build_feature_table(recording, SETS["I0"], moments, household="sim1")
        models = fitted.models(recording.registry, RESOLUTION, HURDLE, ONTOLOGY)
        beliefs = restricted_posteriors(table, models, ONTOLOGY)
        counts = {c: table.column(evidence_column(c.name, 0)) for c in models}
        manual = np.log(ONTOLOGY.stationary() @ ONTOLOGY.transition(STEP))
        manual = manual[None, :] + total_loglik(models, counts)
        np.testing.assert_allclose(beliefs, normalised(manual), atol=1e-12)
        assert (np.sum([c > 0 for c in counts.values()], axis=0) >= 2).any()

    def test_uninstrumented_channels_have_no_model(
        self, homes: dict[str, CasasRecording], fitted: FittedChannels
    ) -> None:
        recording = homes["sim1"]
        table = build_feature_table(
            recording,
            SETS["I0"],
            _regular_moments(recording, STEP)[:50],
            household="sim1",
        )
        models = fitted.models(recording.registry, RESOLUTION, HURDLE, ONTOLOGY)
        assert set(models) == set(RESOLUTION.channels) - set(table.uninstrumented)
        assert table.uninstrumented
        extra = {**models, **{c: hurdle() for c in table.uninstrumented}}
        with pytest.raises(ValueError, match="instrumented"):
            restricted_posteriors(table, extra, ONTOLOGY)

    def test_unobserved_windows_are_not_silence(
        self, homes: dict[str, CasasRecording], fitted: FittedChannels
    ) -> None:
        recording = homes["sim1"]
        moments = _regular_moments(recording, STEP)[:5]
        table = build_feature_table(recording, SETS["I2"], moments, household="sim1")
        models = fitted.models(recording.registry, RESOLUTION, HURDLE, ONTOLOGY)
        with_history = restricted_posteriors(table, models, ONTOLOGY)
        current = build_feature_table(recording, SETS["I0"], moments, household="sim1")
        without = restricted_posteriors(current, models, ONTOLOGY)
        # Before the recording starts, the lagged windows are missing: the first
        # row sees only its current window, exactly as in I0.
        np.testing.assert_allclose(with_history[0], without[0], atol=1e-12)

    def test_the_declared_model_is_unchanged(
        self, homes: dict[str, CasasRecording], fitted: FittedChannels
    ) -> None:
        recording = homes["sim1"]
        declared, _ = channel_likelihoods(recording.registry, RESOLUTION, ONTOLOGY)
        chosen = fitted.models(recording.registry, RESOLUTION, DECLARED, ONTOLOGY)
        assert set(chosen) == set(declared)
        for channel, terms in declared.items():
            np.testing.assert_array_equal(
                chosen[channel].loglik(np.arange(5.0)), terms.loglik(np.arange(5.0))
            )

    def test_the_history_state_needs_the_declared_terms(
        self, homes: dict[str, CasasRecording], fitted: FittedChannels
    ) -> None:
        from sensor_modeling.fusion.history import HistoryModel

        recording = homes["sim1"]
        table = build_feature_table(
            recording,
            SETS["I2"],
            _regular_moments(recording, STEP)[:20],
            household="sim1",
        )
        models = fitted.models(recording.registry, RESOLUTION, HURDLE, ONTOLOGY)
        history = HistoryModel.null(ONTOLOGY)
        with pytest.raises(ValueError, match="declared"):
            restricted_posteriors(table, models, ONTOLOGY, history_model=history)

    def test_the_recursion_matches_the_phase3_decomposition(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        recording = homes["sim2"]
        counts, _, _, _ = household_channel_counts(
            recording, RESOLUTION, ONTOLOGY, household="sim2"
        )
        declared, _ = channel_likelihoods(recording.registry, RESOLUTION, ONTOLOGY)
        channels = sorted(declared)
        _, posterior = filter_recursion(
            total_loglik(declared, counts),
            ONTOLOGY.transition(STEP),
            ONTOLOGY.stationary(),
        )
        reference = filter_trajectory(
            np.column_stack([counts[c] for c in channels]),
            np.stack([declared[c].per_activation for c in channels]),
            np.stack([declared[c].per_window for c in channels]),
            ONTOLOGY.transition(STEP),
            ONTOLOGY.stationary(),
        )
        np.testing.assert_allclose(posterior, reference.posterior, atol=1e-12)
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0)


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------
def statistics(windows: float, silent: float, total: float, declared: float) -> Any:
    size = np.zeros(STATES)
    return ChannelStatistics(
        size + windows,
        size + silent,
        size + total,
        size + total,
        size + declared * windows,
    )


class TestFitting:
    def test_the_estimators(self) -> None:
        fitted = FittedChannels(
            ("h",),
            tuple(ONTOLOGY.states),
            12.0,
            {KITCHEN: statistics(1000, 600, 1200, 0.8)},
        )
        values = fitted.parameters(KITCHEN)
        prior = 0.8
        assert values["silence"][0] == pytest.approx(
            (600 + 12 * math.exp(-prior)) / 1012
        )
        active_mean = (1200 + 12 * prior / -math.expm1(-prior)) / (400 + 12)
        assert values["rate"][0] == pytest.approx(truncated_poisson_rate(active_mean))
        assert values["mean"][0] == pytest.approx((1200 + 12 * prior) / 1012)

    def test_without_data_the_fitted_model_is_the_declared_one(self) -> None:
        declared = np.linspace(0.1, 3.0, STATES)
        values = FittedChannels(("h",), tuple(ONTOLOGY.states), 12.0, {}).parameters(
            KITCHEN, declared
        )
        np.testing.assert_allclose(values["silence"], np.exp(-declared))
        np.testing.assert_allclose(values["mean"], declared)
        model = HurdleChannel(KITCHEN, values["silence"], values["rate"])
        counts = np.array([0.0, 1.0, 4.0])
        np.testing.assert_allclose(
            normalised(model.loglik(counts)),
            normalised(poisson_channel(KITCHEN, declared).loglik(counts)),
            atol=1e-9,
        )

    def test_a_state_without_training_windows_takes_the_households_prior(
        self,
    ) -> None:
        stats = statistics(1000, 600, 1200, 0.8)
        empty = ChannelStatistics(
            *(
                np.where(np.arange(STATES) == 4, 0.0, getattr(stats, name))
                for name in ("windows", "silent", "total", "squares", "declared")
            )
        )
        fitted = FittedChannels(("h",), tuple(ONTOLOGY.states), 12.0, {KITCHEN: empty})
        described = fitted.to_dict()["channels"]["kitchen_motion"]
        assert described["bed_awake"]["silence"] is None
        assert described["sleeping"]["silence"] is not None
        declared = np.full(STATES, 2.0)
        values = fitted.parameters(KITCHEN, declared)
        assert values["silence"][4] == pytest.approx(math.exp(-2.0))
        assert np.all(np.isfinite(values["rate"]))

    def test_strong_shrinkage_returns_the_prior(self) -> None:
        fitted = FittedChannels(
            ("h",),
            tuple(ONTOLOGY.states),
            1e12,
            {KITCHEN: statistics(1000, 100, 5000, 0.8)},
        )
        assert fitted.parameters(KITCHEN)["silence"][0] == pytest.approx(math.exp(-0.8))

    def test_the_hurdle_recovers_its_parameters(self) -> None:
        rng = np.random.default_rng(0)
        silence, rate, windows = 0.7, 2.5, 200_000
        counts = rng.poisson(rate, size=windows * 3)
        counts = counts[counts > 0][:windows]
        counts[rng.random(windows) < silence] = 0
        fitted = FittedChannels(
            ("h",),
            tuple(ONTOLOGY.states),
            1.0,
            {
                KITCHEN: statistics(
                    windows, float(np.sum(counts == 0)), float(counts.sum()), 1.0
                )
            },
        )
        values = fitted.parameters(KITCHEN)
        assert values["silence"][0] == pytest.approx(silence, abs=0.005)
        assert values["rate"][0] == pytest.approx(rate, rel=0.02)

    def test_only_training_households_reach_a_fold(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        fits = fold_fits(homes, small_protocol(), ONTOLOGY)
        for fold in FOLDS:
            assert fits[fold.name].channels.fitted_on == tuple(sorted(fold.train))
            assert fits[fold.name].periodic.fitted_on == tuple(sorted(fold.train))
        changed = {
            home: relabelled(r) if home in FOLDS[0].test else r
            for home, r in homes.items()
        }
        again = fold_fits(changed, small_protocol(), ONTOLOGY)
        assert again["a"].channels.sha256() == fits["a"].channels.sha256()
        assert again["b"].channels.sha256() != fits["b"].channels.sha256()


def relabelled(recording: CasasRecording) -> CasasRecording:
    return CasasRecording(
        recording.registry,
        recording.observations,
        tuple(
            ActivityInterval("Sleep", a.start, a.end, S.SLEEPING)
            for a in recording.activities
        ),
    )


# ----------------------------------------------------------------------------
# The protocol
# ----------------------------------------------------------------------------
def small_protocol(**changes: Any) -> FittedRatesProtocol:
    settings: dict[str, Any] = {
        "folds": FOLDS,
        "splits_sha256": "0" * 64,
        "resamples": 200,
    }
    settings.update(changes)
    return FittedRatesProtocol(**settings)


class TestFrozenProtocol:
    def test_the_committed_protocol_is_the_declared_one(self) -> None:
        splits = load_frozen_splits(SPLITS)
        declared = declared_protocol(splits)
        check_frozen_protocol(declared, PROTOCOL)
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert frozen["protocol_sha256"] == declared.sha256()

    def test_it_freezes_everything_the_experiment_depends_on(self) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert "weakened" in frozen["routing"]
        assert frozen["cells"] == {
            model: list(labels) for model, labels in CELLS.items()
        }
        assert frozen["fitting"]["pseudo_windows"] == 12.0
        assert [e["key"] for e in frozen["estimands"] if e["role"] == "primary"] == [
            "F0",
            "FR",
        ]
        assert [m["key"] for m in frozen["mechanisms"]] == [m.key for m in MECHANISMS]
        assert frozen["criteria"] == CRITERIA
        assert frozen["bootstrap"]["resamples"] == 10_000
        assert frozen["silence_settings"]["min_expected"] == 5.0

    @pytest.mark.parametrize("change", [{"seed": 1}, {"pseudo_windows": 24.0}])
    def test_any_change_to_the_protocol_is_refused(
        self, change: dict[str, Any]
    ) -> None:
        changed = replace(declared_protocol(load_frozen_splits(SPLITS)), **change)
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(changed, PROTOCOL)

    def test_a_tampered_protocol_file_is_refused(self, tmp_path: Path) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        frozen["criteria"]["success"] = "anything"
        tampered = tmp_path / "protocol.json"
        tampered.write_text(json.dumps(frozen), encoding="utf-8")
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(
                declared_protocol(load_frozen_splits(SPLITS)), tampered
            )

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"splits_sha256": "abc"}, "SHA-256"),
            ({"pseudo_windows": 0.0}, "pseudo_windows"),
            ({"minimal_effects": {"slope": 0.02}}, "Phase 3.3 scale"),
            ({"folds": FOLDS[:1]}, "two folds"),
        ],
    )
    def test_invalid_protocols_are_refused(
        self, change: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            small_protocol(**change)

    @pytest.mark.parametrize(
        ("verdicts", "expected"),
        [
            (
                {
                    "calibration_error": "favours model",
                    "balanced_accuracy": "negligible",
                    "log_loss": "favours model",
                },
                "success",
            ),
            (
                {
                    "calibration_error": "favours model",
                    "balanced_accuracy": "favours reference",
                    "log_loss": "favours model",
                },
                "trade-off",
            ),
            (
                {
                    "calibration_error": "favours model",
                    "balanced_accuracy": "uncertain",
                    "log_loss": "favours reference",
                },
                "inconclusive",
            ),
            (
                {
                    "calibration_error": "negligible",
                    "balanced_accuracy": "favours model",
                    "log_loss": "favours model",
                },
                "failure",
            ),
            (
                {
                    "calibration_error": "favours reference",
                    "balanced_accuracy": "favours model",
                    "log_loss": "favours model",
                },
                "failure",
            ),
            (
                {
                    "calibration_error": "uncertain",
                    "balanced_accuracy": "favours model",
                    "log_loss": "favours model",
                },
                "inconclusive",
            ),
        ],
    )
    def test_the_conclusion_rule(self, verdicts: dict[str, str], expected: str) -> None:
        assert conclusion(verdicts) == expected


# ----------------------------------------------------------------------------
# End to end, on simulated homes
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[FittedRatesResult]:
    yield run_fitted_rates(
        homes,
        small_protocol(),
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("rates"),
    )


def results_of(outcome: FittedRatesResult) -> dict[str, Any]:
    return dict(outcome.record.to_dict()["results"])


class TestExperiment:
    def test_the_record_validates(self, result: FittedRatesResult) -> None:
        assert result.path is not None
        payload = load_record(result.path)
        assert payload["results"]["result_schema"] == "fitted-rates-results/1"
        assert payload["configuration"]["protocol_sha256"] == small_protocol().sha256()

    def test_every_declared_cell_is_scored(self, result: FittedRatesResult) -> None:
        cells = {
            (c["model"], c["information_set"]) for c in results_of(result)["cells"]
        }
        assert cells == {(m, s) for m, sets in CELLS.items() for s in sets}
        assert all(c["households"] == 4 for c in results_of(result)["cells"])

    def test_every_comparison_is_on_identical_information(self) -> None:
        for estimand in ESTIMANDS:
            assert estimand.kind == "formulation"
            assert estimand.model_set == estimand.reference_set
            assert (estimand.model_set == RECURSION) == (
                estimand.model.startswith("filter_")
            )

    def test_no_household_is_scored_with_models_fitted_on_it(
        self, result: FittedRatesResult
    ) -> None:
        results = results_of(result)
        for home, entry in results["households"].items():
            fitted = results["fitted"][entry["fold"]]
            assert home not in fitted["channels"]["fitted_on"]
            assert home not in fitted["periodic_prior"]["fitted_on"]

    def test_conclusions_and_mechanisms_follow_their_definitions(
        self, result: FittedRatesResult
    ) -> None:
        results = results_of(result)
        estimands = {e["key"]: e for e in results["estimands"]}
        for entry in estimands.values():
            assert entry["conclusion"] == conclusion(entry["verdicts"])
        assert results["conclusions"] == {
            "current_windows": estimands["F0"]["conclusion"],
            "recursion": estimands["FR"]["conclusion"],
        }
        for home, entry in results["households"].items():
            inflation = entry["log_inflation"]
            if inflation[DECLARED] is not None and inflation[HURDLE] is not None:
                assert entry["mechanisms"]["inflation_reduction_hurdle"] == (
                    pytest.approx(inflation[DECLARED] - inflation[HURDLE])
                ), home
            slope = entry["accumulation_slope"]
            if slope[DECLARED] is not None and slope[POISSON] is not None:
                assert entry["mechanisms"]["accumulation_reduction_poisson"] == (
                    pytest.approx(slope[DECLARED] - slope[POISSON])
                ), home

    def test_a_second_run_gives_the_same_record(
        self, homes: dict[str, CasasRecording], result: FittedRatesResult
    ) -> None:
        again = run_fitted_rates(homes, small_protocol(), data_source="simulator")
        first, second = result.record.to_dict(), again.record.to_dict()
        first.pop("recorded_at")
        second.pop("recorded_at")
        assert first == second

    def test_the_summary_is_generated_from_the_written_record(
        self, result: FittedRatesResult
    ) -> None:
        assert result.path is not None
        summary = render_summary(load_record(result.path))
        assert summary == render_summary(
            json.loads(json.dumps(result.record.to_dict()))
        )
        for heading in (
            "## Pre-specified conclusions",
            "## Every cell",
            "## Estimands",
            "## Per-state recall",
            "## Mechanisms",
            "## Quiet runs",
            "## Fitted silence",
        ):
            assert heading in summary
        payload = result.record.to_dict()
        payload["results"] = {"result_schema": "silence-results/1"}
        with pytest.raises(ValueError, match="fitted-rates-results"):
            render_summary(payload)
