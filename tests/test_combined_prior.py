"""Tests for the combined-prior experiment.

They guard what the result rests on:
- the recursion with the time prior is the production filter with the prior's
  circadian ontology, and without it is the fitted-rates recursion;
- the run reproduces the fitted-rates record before it reports anything, and
  refuses when it does not;
- the protocol is frozen, pins its base and the records it reads, and refuses
  any change;
- the pages are generated from the frozen file and the record.
"""

from __future__ import annotations

import copy
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
from sensor_modeling.datasets.channel_models import (
    filter_recursion,
    household_channel_counts,
    total_loglik,
)
from sensor_modeling.datasets.combined_prior_experiment import (
    ESTIMANDS,
    FITTED_RATES_PROTOCOL,
    MODELS,
    PINNED_RECORDS,
    CombinedPriorProtocol,
    CombinedPriorResult,
    check_frozen_protocol,
    check_pinned_records,
    declared_protocol,
    periodic_recursion,
    periodic_transitions,
    run_combined_prior,
    write_protocol,
)
from sensor_modeling.datasets.combined_prior_summary import (
    PROTOCOL_FILE,
    PROTOCOL_PAGE,
    RECORD_FILE,
    RESULTS_PAGE,
    render_page,
    render_protocol,
)
from sensor_modeling.datasets.matched_evaluation import _regular_moments
from sensor_modeling.datasets.rates_experiment import (
    FittedRatesProtocol,
    run_fitted_rates,
)
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.restricted_filter import channel_likelihoods
from sensor_modeling.evaluation import load_record
from sensor_modeling.fusion.defaults import default_emissions
from sensor_modeling.fusion.filter import MultimodalBayesFilter
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
ONTOLOGY = StateOntology()
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


def small_protocol() -> CombinedPriorProtocol:
    base = FittedRatesProtocol(folds=FOLDS, splits_sha256="0" * 64, resamples=200)
    return CombinedPriorProtocol(base)


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {f"sim{s}": simulated(s) for s in range(1, 5)}


# ----------------------------------------------------------------------------
# The recursion
# ----------------------------------------------------------------------------
class TestRecursion:
    def test_without_hourly_variation_it_is_the_homogeneous_recursion(self) -> None:
        rng = np.random.default_rng(0)
        loglik = rng.normal(size=(50, ONTOLOGY.size))
        transition = ONTOLOGY.transition(STEP)
        hours = [int(h) for h in rng.integers(0, 24, size=50)]
        stacked = np.repeat(transition[None], 24, axis=0)
        expected = filter_recursion(loglik, transition, ONTOLOGY.stationary())
        got = periodic_recursion(loglik, stacked, hours, ONTOLOGY.stationary())
        for a, b in zip(got, expected):
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_it_is_the_production_filter_with_the_prior(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        source, other = homes["sim1"], homes["sim2"]
        prior = fit_periodic_prior(
            {
                "other": hour_state_counts(
                    other, _regular_moments(other, STEP), step=STEP
                )
            }
        )
        counts, _, _, moments = household_channel_counts(
            source, RESOLUTION, ONTOLOGY, household="sim1"
        )
        terms, _ = channel_likelihoods(source.registry, RESOLUTION, ONTOLOGY)
        hours = [moment.hour for moment in moments]
        start = prior.probabilities([hours[0]])[0]
        _, posterior = periodic_recursion(
            total_loglik(terms, counts),
            periodic_transitions(prior, ONTOLOGY, STEP),
            hours,
            start,
        )

        routed = {sensor for term in terms.values() for sensor in term.sensors}
        circadian = prior.ontology(ONTOLOGY)
        emissions = [
            e
            for e in default_emissions(source.registry, circadian)
            if e.sensor_id in routed
        ]
        production = MultimodalBayesFilter(
            circadian, emissions, source.registry, prior=start
        )
        production.update(moments[0] - STEP)
        observations = sorted(source.observations, key=lambda o: o.timestamp)
        cursor = 0
        rows = min(len(moments), 600)
        for row in range(rows):
            end = moments[row]
            window = []
            while cursor < len(observations) and observations[cursor].timestamp <= end:
                if observations[cursor].timestamp > end - STEP:
                    window.append(observations[cursor])
                cursor += 1
            estimate = production.update(end, window)
            np.testing.assert_allclose(estimate.belief, posterior[row], atol=1e-10)
        assert len(set(hours[:rows])) == 24


# ----------------------------------------------------------------------------
# The experiment, on simulated homes
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def published(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    result = run_fitted_rates(
        homes,
        small_protocol().base,
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("rates"),
    )
    assert result.path is not None
    return load_record(result.path)


@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording],
    published: dict[str, Any],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[CombinedPriorResult]:
    yield run_combined_prior(
        homes,
        small_protocol(),
        published,
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("combined"),
    )


class TestExperiment:
    def test_the_record_validates_and_reports_the_check(
        self, result: CombinedPriorResult
    ) -> None:
        assert result.path is not None
        payload = load_record(result.path)
        results = payload["results"]
        assert results["result_schema"] == "combined-prior-results/1"
        assert results["check"]["fits"] == {
            "folds": ["a", "b"],
            "reproduced": True,
            "identical_digests": True,
        }
        assert results["check"]["scores"] == {
            "models": ["filter_declared", "filter_hurdle"],
            "households": 4,
            "reproduced": True,
            "identical": True,
        }
        assert {c["model"] for c in results["cells"]} == set(MODELS)
        assert [e["key"] for e in results["estimands"]] == ["K1", "K2", "K3"]
        assert results["conclusion"] == results["estimands"][0]["conclusion"]

    def test_the_reproduced_models_are_the_published_scores(
        self, result: CombinedPriorResult, published: dict[str, Any]
    ) -> None:
        mine = result.record.to_dict()["household_metrics"]
        for model in ("filter_declared@R", "filter_hurdle@R"):
            assert json.dumps(mine[model], sort_keys=True) == json.dumps(
                published["household_metrics"][model], sort_keys=True
            )

    def test_a_different_published_score_stops_the_run(
        self, homes: dict[str, CasasRecording], published: dict[str, Any]
    ) -> None:
        changed = copy.deepcopy(published)
        changed["household_metrics"]["filter_hurdle@R"]["sim3"]["log_loss"] += 1e-4
        with pytest.raises(ValueError, match="reproduced scores differ"):
            run_combined_prior(homes, small_protocol(), changed, data_source="sim")

    def test_a_different_published_fit_stops_the_run(
        self, homes: dict[str, CasasRecording], published: dict[str, Any]
    ) -> None:
        changed = copy.deepcopy(published)
        changed["results"]["fitted"]["b"]["periodic_prior"]["population"][0][0] += 0.01
        with pytest.raises(ValueError, match="fold fits differ"):
            run_combined_prior(homes, small_protocol(), changed, data_source="sim")

    def test_the_interaction_is_k1_minus_k2(self, result: CombinedPriorResult) -> None:
        results = result.record.to_dict()["results"]
        estimands = {e["key"]: e for e in results["estimands"]}
        for metric, entry in results["interaction"].items():
            k1 = estimands["K1"]["comparisons"][metric]["mean"]["estimate"]
            k2 = estimands["K2"]["comparisons"][metric]["mean"]["estimate"]
            assert entry["mean"]["estimate"] == pytest.approx(k1 - k2)

    def test_the_results_page_is_generated_from_the_record(
        self, result: CombinedPriorResult
    ) -> None:
        assert result.path is not None
        page = render_page(load_record(result.path))
        assert "## The check" in page
        for estimand in ESTIMANDS:
            assert f"| {estimand.key} |" in page
        assert "nan" not in page.lower()


# ----------------------------------------------------------------------------
# The protocol
# ----------------------------------------------------------------------------
class TestProtocol:
    def test_the_committed_protocol_is_the_declared_one(self) -> None:
        protocol = declared_protocol(load_frozen_splits(SPLITS), ROOT)
        check_frozen_protocol(protocol, ROOT / PROTOCOL_FILE)
        frozen = json.loads((ROOT / PROTOCOL_FILE).read_text(encoding="utf-8"))
        assert frozen["protocol_sha256"] == protocol.sha256()
        assert frozen["base_protocol"]["protocol_sha256"] == protocol.base.sha256()

    def test_the_protocol_page_is_generated_from_the_frozen_file(self) -> None:
        frozen = json.loads((ROOT / PROTOCOL_FILE).read_text(encoding="utf-8"))
        page = (ROOT / "docs" / PROTOCOL_PAGE).read_text(encoding="utf-8")
        assert page == render_protocol(frozen)

    def test_any_change_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "protocol.json"
        write_protocol(small_protocol(), path)
        check_frozen_protocol(small_protocol(), path)
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(
                CombinedPriorProtocol(small_protocol().base, tolerance=1e-5), path
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["estimands"][0]["model"] = "filter_periodic"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(small_protocol(), path)

    def test_a_changed_pinned_record_is_refused(self, tmp_path: Path) -> None:
        for name in PINNED_RECORDS:
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / name, tmp_path / name)
        check_pinned_records(tmp_path)
        with (tmp_path / FITTED_RATES_PROTOCOL).open("a", encoding="utf-8") as f:
            f.write(" ")
        with pytest.raises(ValueError, match="not the file"):
            check_pinned_records(tmp_path)

    @pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan")])
    def test_invalid_tolerances_are_refused(self, tolerance: float) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            CombinedPriorProtocol(small_protocol().base, tolerance=tolerance)


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
