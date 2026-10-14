"""Tests for the activity sub-state model and its experiment.

They guard what the result rests on:
- with tied sub-states the recursion is exactly the reference's, the hurdle
  recursion with the time prior, which is the production filter's;
- the joint transition keeps the time prior's on the behavioural layer;
- forward-backward over packed stretches is exact, and EM is deterministic
  and ignores what it should;
- the run reproduces the published fits, reference scores and quiet-run
  statistic before it reports anything, and refuses when it does not;
- the protocol is frozen, pins the records it reads, and refuses any change;
- the pages are generated from the frozen file and the record.
"""

from __future__ import annotations

import copy
import itertools
import json
import shutil
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    HouseholdSplit,
    fit_periodic_prior,
    hour_state_counts,
    nested_information_sets,
)
from sensor_modeling.datasets.channel_models import fit_channel_models, total_loglik
from sensor_modeling.datasets.combined_prior_experiment import (
    CombinedPriorProtocol,
    periodic_recursion,
    periodic_transitions,
    run_combined_prior,
)
from sensor_modeling.datasets.matched_evaluation import _regular_moments
from sensor_modeling.datasets.predictive_checks import (
    PredictiveProtocol,
    run_predictive_checks,
)
from sensor_modeling.datasets.rates_experiment import (
    FittedRatesProtocol,
    run_fitted_rates,
)
from sensor_modeling.datasets.recoverable_gap import _score, load_frozen_splits
from sensor_modeling.datasets.substate_channels import (
    SubstateConfig,
    chain,
    fit_substates,
    forward_backward,
    household_windows,
    joint_states,
    sample_silence,
    tied,
)
from sensor_modeling.datasets.substate_experiment import (
    AMENDMENTS_FILE,
    ESTIMANDS,
    MODEL,
    PINNED_RECORDS,
    PREDICTIVE_PROTOCOL,
    REFERENCE,
    SPLIT,
    SubstateProtocol,
    SubstateResult,
    check_frozen_protocol,
    check_pinned_records,
    check_windows,
    code_changes,
    declared_protocol,
    file_sha256,
    mechanism_conclusion,
    read_amendments,
    run_substates,
    substate_conclusion,
    write_protocol,
)
from sensor_modeling.datasets.substate_summary import (
    PROTOCOL_FILE,
    PROTOCOL_PAGE,
    RECORD_FILE,
    RESULTS_PAGE,
    render_page,
    render_protocol,
)
from sensor_modeling.evaluation import load_record
from sensor_modeling.observations import SensorRegistry
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
ONTOLOGY = StateOntology()
STATES = tuple(ONTOLOGY.states)
STEP = timedelta(minutes=5)
RESOLUTION = nested_information_sets()[0].resolution
FOLDS = (
    HouseholdSplit("a", train=("sim2", "sim4"), test=("sim1", "sim3")),
    HouseholdSplit("b", train=("sim1", "sim3"), test=("sim2", "sim4")),
)


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


def without_kitchen_or_bed_awake(recording: CasasRecording) -> CasasRecording:
    """A home with no kitchen sensors, and no `bed_awake` label."""
    kitchen = {spec.sensor_id for spec in recording.registry if spec.room == "kitchen"}
    return CasasRecording(
        SensorRegistry.from_specs(
            spec for spec in recording.registry if spec.sensor_id not in kitchen
        ),
        tuple(o for o in recording.observations if o.sensor_id not in kitchen),
        tuple(a for a in recording.activities if a.state is not S.BED_AWAKE),
    )


def without(recording: CasasRecording, state: S) -> CasasRecording:
    """The same home with no label of *state*."""
    return CasasRecording(
        recording.registry,
        recording.observations,
        tuple(a for a in recording.activities if a.state is not state),
    )


def small_protocol(**changes: Any) -> SubstateProtocol:
    base = FittedRatesProtocol(folds=FOLDS, splits_sha256="0" * 64, resamples=200)
    predictive = PredictiveProtocol(
        folds=FOLDS, splits_sha256="0" * 64, replicates=40, resamples=200
    )
    settings: dict[str, Any] = {
        "substates": SubstateConfig(SPLIT, iterations=8),
        **changes,
    }
    return SubstateProtocol(CombinedPriorProtocol(base), predictive, **settings)


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {
        "sim1": simulated(1),
        "sim2": simulated(2),
        "sim3": simulated(3),
        "sim4": without_kitchen_or_bed_awake(simulated(4)),
    }


@pytest.fixture(scope="module")
def windows(homes: dict[str, CasasRecording]) -> dict[str, Any]:
    return {
        home: household_windows(recording, RESOLUTION, ONTOLOGY, household=home)
        for home, recording in homes.items()
    }


@pytest.fixture(scope="module")
def reference(homes: dict[str, CasasRecording]) -> Any:
    return fit_channel_models(
        {h: homes[h] for h in ("sim2", "sim4")},
        resolution=RESOLUTION,
        pseudo_windows=12.0,
        ontology=ONTOLOGY,
    )


@pytest.fixture(scope="module")
def prior(homes: dict[str, CasasRecording]) -> Any:
    return fit_periodic_prior(
        {
            h: hour_state_counts(homes[h], _regular_moments(homes[h], STEP), step=STEP)
            for h in ("sim2", "sim4")
        }
    )


@pytest.fixture(scope="module")
def fitted(windows: dict[str, Any], reference: Any) -> Any:
    return fit_substates(
        {h: windows[h] for h in ("sim2", "sim4")},
        reference,
        SubstateConfig(SPLIT, iterations=20),
    )


def recursion(model: Any, data: Any, prior: Any) -> np.ndarray:
    timed = periodic_transitions(prior, ONTOLOGY, STEP)
    start = prior.probabilities([data.hours[0]])[0]
    _, posterior = periodic_recursion(
        model.loglik(data.registry, RESOLUTION, data.counts, ONTOLOGY),
        model.transitions(timed),
        data.hours,
        model.start(start),
    )
    marginal: np.ndarray = posterior @ model.aggregation()
    return marginal


def hurdle_recursion(reference: Any, data: Any, prior: Any) -> np.ndarray:
    timed = periodic_transitions(prior, ONTOLOGY, STEP)
    start = prior.probabilities([data.hours[0]])[0]
    models = reference.models(data.registry, RESOLUTION, "hurdle", ONTOLOGY)
    _, posterior = periodic_recursion(
        total_loglik(models, data.counts), timed, data.hours, start
    )
    return posterior


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------
class TestModel:
    @pytest.mark.parametrize(
        ("entry", "persistence"), [((0.5, 0.5), 0.7), ((0.9, 0.1), 0.99)]
    )
    def test_tied_sub_states_are_the_hurdle_recursion(
        self,
        windows: dict[str, Any],
        reference: Any,
        prior: Any,
        entry: tuple[float, float],
        persistence: float,
    ) -> None:
        for home in ("sim1", "sim3"):
            data = windows[home]
            model = tied(reference, SPLIT, entry, persistence)
            np.testing.assert_allclose(
                recursion(model, data, prior),
                hurdle_recursion(reference, data, prior),
                atol=1e-10,
            )

    def test_the_fitted_model_keeps_unsplit_states_and_differs_in_split_ones(
        self, windows: dict[str, Any], reference: Any, fitted: Any
    ) -> None:
        data = windows["sim1"]
        joint = fitted.loglik(data.registry, RESOLUTION, data.counts, ONTOLOGY)
        hurdle = total_loglik(
            reference.models(data.registry, RESOLUTION, "hurdle", ONTOLOGY),
            data.counts,
        )
        for j, (s, k) in enumerate(fitted.joint):
            if STATES[s] in SPLIT:
                assert not np.allclose(joint[:, j], hurdle[:, s])
            else:
                assert k == 0
                np.testing.assert_array_equal(joint[:, j], hurdle[:, s])

    def test_the_joint_transition_keeps_the_behavioural_one(
        self, fitted: Any, prior: Any
    ) -> None:
        timed = periodic_transitions(prior, ONTOLOGY, STEP)
        joint = fitted.transitions(timed)
        aggregation = fitted.aggregation()
        assert joint.shape == (24, 11, 11)
        np.testing.assert_allclose(joint.sum(axis=2), 1.0, atol=1e-12)
        for hour in range(24):
            for j, (s, _) in enumerate(fitted.joint):
                np.testing.assert_allclose(
                    joint[hour, j] @ aggregation, timed[hour, s], atol=1e-12
                )
        start = prior.probabilities([7])[0]
        np.testing.assert_allclose(fitted.start(start) @ aggregation, start, atol=1e-12)

    def test_the_joint_states_split_the_declared_states(self) -> None:
        joint = joint_states(STATES, SPLIT)
        assert len(joint) == 11
        for s, state in enumerate(STATES):
            assert sum(1 for t, _ in joint if t == s) == (2 if state in SPLIT else 1)

    def test_forward_backward_is_exact(self) -> None:
        rng = np.random.default_rng(3)
        lengths = [4, 1, 3, 4, 2]
        sequences = [rng.normal(size=(n, 2)) for n in lengths]
        entry = np.array([0.3, 0.7])
        within = np.array([[0.8, 0.2], [0.35, 0.65]])
        order = np.argsort(-np.array(lengths), kind="stable")
        sizes = np.array([sum(n > t for n in lengths) for t in range(max(lengths))])
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        packed = np.concatenate(
            [
                np.stack([sequences[i][t] for i in order[: sizes[t]]])
                for t in range(sizes.size)
            ]
        )
        gamma, transitions, entries, total = forward_backward(
            packed, sizes, bounds, entry, within
        )
        expected_total = 0.0
        expected_transitions = np.zeros((2, 2))
        expected_entries = np.zeros(2)
        for position, i in enumerate(order):
            loglik = sequences[i]
            paths = list(itertools.product(range(2), repeat=loglik.shape[0]))
            weights = np.array(
                [
                    entry[p[0]]
                    * np.prod([within[a, b] for a, b in itertools.pairwise(p)])
                    * np.exp(sum(loglik[t, p[t]] for t in range(len(p))))
                    for p in paths
                ]
            )
            expected_total += float(np.log(weights.sum()))
            weights /= weights.sum()
            for t in range(loglik.shape[0]):
                marginal = np.zeros(2)
                for weight, p in zip(weights, paths, strict=True):
                    marginal[p[t]] += weight
                np.testing.assert_allclose(
                    gamma[bounds[t] + position], marginal, atol=1e-12
                )
            for weight, p in zip(weights, paths, strict=True):
                expected_entries[p[0]] += weight
                for a, b in itertools.pairwise(p):
                    expected_transitions[a, b] += weight
        assert total == pytest.approx(expected_total, abs=1e-10)
        np.testing.assert_allclose(entries, expected_entries, atol=1e-12)
        np.testing.assert_allclose(transitions, expected_transitions, atol=1e-12)

    def test_em_is_deterministic_and_records_every_iteration(
        self, windows: dict[str, Any], reference: Any, fitted: Any
    ) -> None:
        again = fit_substates(
            {h: windows[h] for h in ("sim2", "sim4")},
            reference,
            SubstateConfig(SPLIT, iterations=20),
        )
        assert again.sha256() == fitted.sha256()
        for state in SPLIT:
            assert len(fitted.trace[state]) == 20
            assert fitted.trace[state][-1] > fitted.trace[state][0]
            np.testing.assert_allclose(fitted.within[state].sum(axis=1), 1.0)
            assert fitted.entry[state].sum() == pytest.approx(1.0)

    def test_the_quiet_sub_state_is_the_quieter(self, fitted: Any) -> None:
        for entry in fitted.to_dict()["states"].values():
            quiet, busy = entry["mean_silence"]
            assert quiet >= busy
            assert sum(entry["share"]) == pytest.approx(1.0)

    def test_the_sub_states_split_the_windows(
        self, windows: dict[str, Any], fitted: Any
    ) -> None:
        for state in SPLIT:
            s = STATES.index(state)
            for channel, stats in fitted.reference.statistics.items():
                pieces = [f.statistics[channel] for f in fitted.substates]
                assert sum(p.windows[s] for p in pieces) == pytest.approx(
                    stats.windows[s]
                )
                assert sum(p.silent[s] for p in pieces) == pytest.approx(
                    stats.silent[s]
                )
                assert sum(p.total[s] for p in pieces) == pytest.approx(stats.total[s])

    def test_an_unlabelled_home_changes_nothing(
        self, homes: dict[str, CasasRecording], windows: dict[str, Any], reference: Any
    ) -> None:
        blank = CasasRecording(homes["sim1"].registry, homes["sim1"].observations, ())
        data = household_windows(blank, RESOLUTION, ONTOLOGY, household="blank")
        config = SubstateConfig(SPLIT, iterations=3)
        training = {h: windows[h] for h in ("sim2", "sim4")}
        with_blank = fit_substates({**training, "blank": data}, reference, config)
        assert (
            with_blank.sha256() == fit_substates(training, reference, config).sha256()
        )

    def test_a_state_with_no_training_window_stays_tied(
        self, homes: dict[str, CasasRecording], windows: dict[str, Any], prior: Any
    ) -> None:
        training = {h: without(homes[h], S.AWAY) for h in ("sim2", "sim4")}
        reference = fit_channel_models(
            training, resolution=RESOLUTION, pseudo_windows=12.0, ontology=ONTOLOGY
        )
        model = fit_substates(
            {
                h: household_windows(r, RESOLUTION, ONTOLOGY, household=h)
                for h, r in training.items()
            },
            reference,
            SubstateConfig(SPLIT, iterations=3),
        )
        assert model.windows[S.AWAY] == (0, 0)
        assert S.AWAY not in model.trace
        data = windows["sim1"]
        joint = model.loglik(data.registry, RESOLUTION, data.counts, ONTOLOGY)
        quiet, busy = (j for j, (s, _) in enumerate(model.joint) if STATES[s] is S.AWAY)
        np.testing.assert_array_equal(joint[:, quiet], joint[:, busy])

    def test_a_home_missing_a_channel_is_scored_without_it(
        self, windows: dict[str, Any], fitted: Any, prior: Any
    ) -> None:
        data = windows["sim4"]
        assert len(data.counts) < len(windows["sim1"].counts)
        belief = recursion(fitted, data, prior)
        assert np.isfinite(belief).all()
        np.testing.assert_allclose(belief.sum(axis=1), 1.0, atol=1e-12)

    def test_a_persistent_chain_draws_longer_quiet_runs(self) -> None:
        stretch = np.zeros(500, dtype=int)
        silence = np.array([0.95, 0.05])
        entry = np.array([0.5, 0.5])
        rng = np.random.default_rng(0)
        sticky = sample_silence(rng, silence, entry, chain(0.99), stretch, 200)
        mixed = sample_silence(rng, silence, entry, chain(0.5), stretch, 200)
        assert sticky.mean() == pytest.approx(mixed.mean(), abs=0.05)
        assert sticky.shape == (200, 500)

        def longest(rows: np.ndarray) -> float:
            best = []
            for row in rows:
                run = top = 0
                for silent in row:
                    run = run + 1 if silent else 0
                    top = max(top, run)
                best.append(top)
            return float(np.mean(best))

        assert longest(sticky) > 3 * longest(mixed)

    def test_a_stretch_restarts_from_the_entry(self) -> None:
        stretch = np.repeat(np.arange(100), 3)
        silence = np.array([1.0, 0.0])
        rng = np.random.default_rng(0)
        drawn = sample_silence(
            rng, silence, np.array([1.0, 0.0]), np.eye(2)[::-1], stretch, 10
        )
        # Entered quiet, the chain always switches: silent, active, silent.
        np.testing.assert_array_equal(drawn[:, 0::3], True)
        np.testing.assert_array_equal(drawn[:, 1::3], False)
        np.testing.assert_array_equal(drawn[:, 2::3], True)

    @pytest.mark.parametrize(
        "changes",
        [
            {"split": ()},
            {"split": (S.AWAY, S.AWAY)},
            {"iterations": 0},
            {"odds_factor": 1.0},
            {"persistence": 1.0},
            {"pseudo_count": 0.0},
        ],
    )
    def test_invalid_configurations_are_refused(self, changes: dict[str, Any]) -> None:
        settings: dict[str, Any] = {"split": SPLIT, **changes}
        with pytest.raises(ValueError):
            SubstateConfig(**settings)


# ----------------------------------------------------------------------------
# The rules
# ----------------------------------------------------------------------------
class TestRules:
    @pytest.mark.parametrize(
        ("recall", "others", "expected"),
        [
            ("favours model", "uncertain", "success"),
            ("favours model", "negligible", "success"),
            ("favours model", "favours reference", "trade-off"),
            ("negligible", "favours model", "failure"),
            ("favours reference", "favours model", "failure"),
            ("uncertain", "favours model", "inconclusive"),
        ],
    )
    def test_the_primary_rule(self, recall: str, others: str, expected: str) -> None:
        verdicts = dict.fromkeys(
            ("balanced_accuracy", "log_loss", "calibration_error", "brier"), others
        )
        guards = dict.fromkeys(("away", "home_inactive", "sleeping"), others)
        assert substate_conclusion(verdicts, recall, guards) == expected

    def test_one_guarded_metric_favouring_the_reference_is_a_trade_off(self) -> None:
        verdicts = {
            "balanced_accuracy": "favours model",
            "log_loss": "favours model",
            "calibration_error": "favours reference",
            "brier": "favours reference",
        }
        others = dict.fromkeys(("away", "home_inactive", "sleeping"), "uncertain")
        assert substate_conclusion(verdicts, "favours model", others) == "trade-off"
        verdicts["calibration_error"] = "uncertain"
        assert substate_conclusion(verdicts, "favours model", others) == "success"

    def test_another_common_state_losing_recall_is_a_trade_off(self) -> None:
        verdicts = dict.fromkeys(
            ("balanced_accuracy", "log_loss", "calibration_error", "brier"),
            "negligible",
        )
        others = {
            "away": "negligible",
            "home_inactive": "favours reference",
            "sleeping": "insufficient",
        }
        assert substate_conclusion(verdicts, "favours model", others) == "trade-off"
        others["home_inactive"] = "uncertain"
        assert substate_conclusion(verdicts, "favours model", others) == "success"

    @pytest.mark.parametrize(
        ("readings", "expected"),
        [
            (("falls", "falls"), "falls"),
            (("falls", "uncertain"), "falls in part"),
            (("unchanged", "rises"), "does not fall"),
            (("unchanged", "uncertain"), "inconclusive"),
            (("insufficient", "insufficient"), "inconclusive"),
        ],
    )
    def test_the_mechanism_rule(self, readings: tuple[str, str], expected: str) -> None:
        assert mechanism_conclusion(dict(zip("ab", readings, strict=True))) == expected


# ----------------------------------------------------------------------------
# The experiment, on simulated homes
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def published(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    protocol = small_protocol()
    rates = run_fitted_rates(
        homes,
        protocol.base.base,
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("rates"),
    )
    assert rates.path is not None
    fitted_rates = load_record(rates.path)
    prior = run_combined_prior(
        homes,
        protocol.base,
        fitted_rates,
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("combined"),
    )
    checks = run_predictive_checks(
        homes,
        protocol.predictive,
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("predictive"),
    )
    assert prior.path is not None and checks.path is not None
    return {
        "fitted_rates": fitted_rates,
        "combined_prior": load_record(prior.path),
        "predictive": load_record(checks.path),
    }


@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording],
    published: dict[str, Any],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[SubstateResult]:
    yield run_substates(
        homes,
        small_protocol(),
        **published,
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("substates"),
    )


class TestExperiment:
    def test_the_record_validates_and_reports_the_check(
        self, result: SubstateResult
    ) -> None:
        assert result.path is not None
        results = load_record(result.path)["results"]
        assert results["result_schema"] == "activity-substates-results/1"
        check = results["check"]
        assert check["fits"]["reproduced"] and check["fits"]["identical_digests"]
        assert check["scores"] == {
            "model": REFERENCE,
            "households": 4,
            "reproduced": True,
            "identical": True,
        }
        assert check["runs"]["reproduced"] and check["runs"]["identical"]
        assert check["runs"]["cells"] > 0
        assert check["windows"] == {"households": 4, "same_windows": True}
        assert [c["model"] for c in results["cells"]] == [REFERENCE, MODEL]
        assert [e["key"] for e in results["estimands"]] == ["W1"]
        assert results["conclusion"] == results["estimands"][0]["conclusion"]
        assert set(results["fitted"]) == {"a", "b"}

    def test_the_reference_is_the_published_recursion(
        self, result: SubstateResult, published: dict[str, Any]
    ) -> None:
        mine = result.record.to_dict()["household_metrics"][f"{REFERENCE}@R"]
        theirs = published["combined_prior"]["household_metrics"][f"{REFERENCE}@R"]
        assert json.dumps(mine, sort_keys=True) == json.dumps(theirs, sort_keys=True)

    def test_the_primary_reads_home_active_recall(self, result: SubstateResult) -> None:
        entry = result.record.to_dict()["results"]["estimands"][0]
        assert entry["focus"]["state"] == "home_active"
        assert (
            entry["focus"]["verdict"]
            == entry["per_state_recall"]["home_active"]["verdict"]
        )
        assert entry["conclusion"] == substate_conclusion(
            entry["verdicts"], entry["focus"]["verdict"], entry["guarded_states"]
        )
        assert set(entry["guarded_states"]) == {"away", "home_inactive", "sleeping"}
        # Four homes are fewer than a per-state verdict needs.
        assert {e["verdict"] for e in entry["per_state_recall"].values()} == {
            "insufficient"
        }

    def test_the_mechanism_is_the_fall_in_misfit(self, result: SubstateResult) -> None:
        results = result.record.to_dict()["results"]
        for state, entry in results["mechanism"]["states"].items():
            for home, value in entry["values"].items():
                cells = [
                    c
                    for c in results["households"][home]["cells"]
                    if c["state"] == state
                    and "model" in c
                    and c["reference"].get("log_ratio") is not None
                ]
                expected = np.mean(
                    [
                        abs(c["reference"]["log_ratio"]) - abs(c["model"]["log_ratio"])
                        for c in cells
                    ]
                )
                assert value == pytest.approx(expected)
        for cells in results["households"].values():
            for cell in cells["cells"]:
                assert ("model" in cell) == (S(cell["state"]) in SPLIT)

    def test_different_windows_stop_the_run(self) -> None:
        def scored(truth: list[S]) -> Any:
            probabilities = np.full((len(truth), len(STATES)), 1.0 / len(STATES))
            return _score(truth, probabilities, STATES)

        same = {"h": scored([S.AWAY, S.SLEEPING])}
        assert check_windows({REFERENCE: same, MODEL: same})["same_windows"]
        with pytest.raises(ValueError, match="different windows"):
            check_windows({REFERENCE: same, MODEL: {"h": scored([S.AWAY, S.AWAY])}})
        with pytest.raises(ValueError, match="different windows"):
            check_windows({REFERENCE: same, MODEL: {}})

    def test_a_different_published_score_stops_the_run(
        self, homes: dict[str, CasasRecording], published: dict[str, Any]
    ) -> None:
        changed = copy.deepcopy(published)
        changed["combined_prior"]["household_metrics"][f"{REFERENCE}@R"]["sim3"][
            "log_loss"
        ] += 1e-4
        with pytest.raises(ValueError, match="reference scores differ"):
            run_substates(homes, small_protocol(), **changed, data_source="sim")

    def test_a_different_published_fit_stops_the_run(
        self, homes: dict[str, CasasRecording], published: dict[str, Any]
    ) -> None:
        changed = copy.deepcopy(published)
        fits = changed["fitted_rates"]["results"]["fitted"]
        fits["b"]["periodic_prior"]["population"][0][0] += 0.01
        with pytest.raises(ValueError, match="fold fits differ"):
            run_substates(homes, small_protocol(), **changed, data_source="sim")

    def test_a_different_quiet_run_statistic_stops_the_run(
        self, homes: dict[str, CasasRecording], published: dict[str, Any]
    ) -> None:
        changed = copy.deepcopy(published)
        cell = changed["predictive"]["results"]["households"]["sim1"]["cells"][0]
        cell["references"]["population"]["statistics"]["quiet_runs"]["predicted"] += 1.0
        with pytest.raises(ValueError, match="quiet-run statistics differ"):
            run_substates(homes, small_protocol(), **changed, data_source="sim")

    def test_the_results_page_is_generated_from_the_record(
        self, result: SubstateResult
    ) -> None:
        assert result.path is not None
        page = render_page(load_record(result.path))
        assert "## The check" in page
        assert "## The mechanism" in page
        for estimand in ESTIMANDS:
            assert f"| {estimand.key} |" in page
        assert "nan" not in page.lower()


# ----------------------------------------------------------------------------
# The protocol
# ----------------------------------------------------------------------------
class TestProtocol:
    def test_the_protocol_page_renders_from_a_written_protocol(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "protocol.json"
        write_protocol(small_protocol(), path)
        page = render_protocol(json.loads(path.read_text(encoding="utf-8")))
        assert "This page reports no result" in page
        assert "nan" not in page.lower()

    def test_any_change_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "protocol.json"
        write_protocol(small_protocol(), path)
        check_frozen_protocol(small_protocol(), path)
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(small_protocol(tolerance=1e-5), path)
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(
                small_protocol(substates=SubstateConfig(SPLIT, iterations=9)), path
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["estimands"][0]["model"] = "filter_periodic"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(small_protocol(), path)

    def test_the_freeze_records_the_sources_and_the_run_refuses_a_change(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "repository"
        for name in (
            "sensor_modeling/module.py",
            "pyproject.toml",
            "scripts/freeze_phase3_substates.py",
            "scripts/run_phase3_substates.py",
            ".github/workflows/run-phase3-substates.yml",
        ):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text("x = 1\n", encoding="utf-8")
        path = tmp_path / "protocol.json"
        write_protocol(small_protocol(), path, root)
        frozen = json.loads(path.read_text(encoding="utf-8"))
        assert "sensor_modeling/module.py" in frozen["at_freeze"]["sources"]
        check_frozen_protocol(small_protocol(), path)
        assert code_changes(path, root) == []
        (root / "sensor_modeling" / "module.py").write_text("x = 2\n", encoding="utf-8")
        (root / "sensor_modeling" / "added.py").write_text("", encoding="utf-8")
        assert code_changes(path, root) == [
            "sensor_modeling/added.py",
            "sensor_modeling/module.py",
        ]
        module = root / "sensor_modeling" / "module.py"
        amendment = {
            "reason": "a fix after a crash",
            "sources": {"sensor_modeling/module.py": file_sha256(module)},
        }
        assert code_changes(path, root, [amendment]) == ["sensor_modeling/added.py"]
        (root / "sensor_modeling" / "added.py").unlink()
        assert code_changes(path, root, [amendment]) == []
        module.write_text("x = 3\n", encoding="utf-8")
        assert code_changes(path, root, [amendment]) == ["sensor_modeling/module.py"]

    def test_an_amendment_needs_its_reason_and_sources(self, tmp_path: Path) -> None:
        path = tmp_path / "amendments.json"
        assert read_amendments(path) == []
        for amendment in (
            {"reason": "", "sources": {"a.py": "0" * 64}},
            {"reason": "a fix", "sources": {}},
        ):
            path.write_text(json.dumps({"amendments": [amendment]}), encoding="utf-8")
            with pytest.raises(ValueError, match="every amendment"):
                read_amendments(path)

    def test_a_changed_pinned_record_is_refused(self, tmp_path: Path) -> None:
        for name in PINNED_RECORDS:
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / name, tmp_path / name)
        check_pinned_records(tmp_path)
        with (tmp_path / PREDICTIVE_PROTOCOL).open("a", encoding="utf-8") as f:
            f.write(" ")
        with pytest.raises(ValueError, match="not the file"):
            check_pinned_records(tmp_path)

    def test_the_declared_protocol_reads_the_frozen_bases(self) -> None:
        protocol = declared_protocol(load_frozen_splits(SPLITS), ROOT)
        assert protocol.substates.to_dict()["split"] == [s.value for s in SPLIT]
        assert len(protocol.base.base.homes) == 20

    @pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan")])
    def test_invalid_tolerances_are_refused(self, tolerance: float) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            small_protocol(tolerance=tolerance)

    def test_home_active_must_be_split(self) -> None:
        with pytest.raises(ValueError, match="home_active"):
            small_protocol(substates=SubstateConfig((S.AWAY, S.HOME_INACTIVE)))


@pytest.mark.skipif(not (ROOT / PROTOCOL_FILE).exists(), reason="not yet frozen")
class TestFrozen:
    def test_the_committed_protocol_is_the_declared_one(self) -> None:
        protocol = declared_protocol(load_frozen_splits(SPLITS), ROOT)
        check_frozen_protocol(protocol, ROOT / PROTOCOL_FILE)
        frozen = json.loads((ROOT / PROTOCOL_FILE).read_text(encoding="utf-8"))
        assert frozen["protocol_sha256"] == protocol.sha256()

    def test_the_protocol_page_is_generated_from_the_frozen_file(self) -> None:
        frozen = json.loads((ROOT / PROTOCOL_FILE).read_text(encoding="utf-8"))
        page = (ROOT / "docs" / PROTOCOL_PAGE).read_text(encoding="utf-8")
        assert page == render_protocol(frozen)

    @pytest.mark.skipif((ROOT / RECORD_FILE).exists(), reason="already run")
    def test_no_pinned_source_has_changed_before_the_run(self) -> None:
        amendments = read_amendments(ROOT / AMENDMENTS_FILE)
        assert code_changes(ROOT / PROTOCOL_FILE, ROOT, amendments) == []


@pytest.mark.skipif(not (ROOT / RECORD_FILE).exists(), reason="not yet run")
class TestRecord:
    def test_the_results_page_is_generated_from_the_record(self) -> None:
        payload = load_record(ROOT / RECORD_FILE)
        page = (ROOT / "docs" / RESULTS_PAGE).read_text(encoding="utf-8")
        assert page == render_page(payload)

    def test_the_record_ran_the_frozen_protocol(self) -> None:
        payload = load_record(ROOT / RECORD_FILE)
        frozen = json.loads((ROOT / PROTOCOL_FILE).read_text(encoding="utf-8"))
        assert payload["configuration"]["protocol_sha256"] == frozen["protocol_sha256"]
        assert payload["environment"]["git_dirty"] == "false"
