"""Tests for Phase 4 on the reference with the time prior.

They guard what the result rests on:
- without hourly variation the time prior's ensemble trace is the Phase 4
  trace, and its reference is the scored recursion;
- the run reproduces the published fold fits and reference scores before it
  reports anything, and refuses when it does not;
- the abstention rule's signal follows the rule declared before scoring;
- the protocol is frozen with its bases and pinned records and refuses any
  change;
- the pages and figures are generated from the frozen file and the record.
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
)
from sensor_modeling.datasets.channel_models import (
    home_statistics,
    household_channel_counts,
)
from sensor_modeling.datasets.combined_prior_experiment import (
    CombinedPriorProtocol,
    run_combined_prior,
)
from sensor_modeling.datasets.prior_uncertainty_experiment import (
    AMENDMENTS_FILE,
    PHASE4_PROTOCOL,
    PINNED_RECORDS,
    PROTOCOL_FILE,
    RECORD_FILE,
    RUN_VERSIONS,
    WORKFLOW,
    PriorUncertaintyProtocol,
    PriorUncertaintyResult,
    check_frozen_protocol,
    check_pinned_records,
    code_changes,
    declared_protocol,
    library_changes,
    periodic_trace,
    read_amendments,
    rule_signal,
    run_prior_uncertainty,
    write_protocol,
)
from sensor_modeling.datasets.prior_uncertainty_summary import (
    FIGURE_PREFIX,
    PROTOCOL_PAGE,
    RESULTS_PAGE,
    render_page,
    render_protocol,
)
from sensor_modeling.datasets.rates_experiment import (
    FittedRatesProtocol,
    run_fitted_rates,
)
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.structural_models import fit_samples, household_trace
from sensor_modeling.datasets.uncertainty_experiment import (
    ENSEMBLE,
    STRUCTURAL_SIGNALS,
    UncertaintyProtocol,
)
from sensor_modeling.datasets.uncertainty_figures import draw_figures
from sensor_modeling.evaluation import load_record
from sensor_modeling.observations import SensorRegistry
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
ONTOLOGY = StateOntology()
STEP = timedelta(minutes=5)
FOLDS = (
    HouseholdSplit("a", train=("sim2", "sim4", "sim6"), test=("sim1", "sim3", "sim5")),
    HouseholdSplit("b", train=("sim1", "sim3", "sim5"), test=("sim2", "sim4", "sim6")),
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


def small_protocol(**changes: Any) -> PriorUncertaintyProtocol:
    base = UncertaintyProtocol(
        folds=FOLDS, splits_sha256="0" * 64, curve_resamples=200, resamples=1000
    )
    rates = FittedRatesProtocol(folds=FOLDS, splits_sha256="0" * 64, resamples=200)
    return PriorUncertaintyProtocol(base, CombinedPriorProtocol(rates), **changes)


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    out = {f"sim{s}": simulated(s) for s in range(1, 7)}
    out["sim6"] = without_kitchen_or_bed_awake(out["sim6"])
    return out


@pytest.fixture(scope="module")
def published(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    protocol = small_protocol()
    rates = run_fitted_rates(
        homes,
        protocol.combined.base,
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("rates"),
    )
    assert rates.path is not None
    fitted_rates = load_record(rates.path)
    prior = run_combined_prior(
        homes,
        protocol.combined,
        fitted_rates,
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("combined"),
    )
    assert prior.path is not None
    return {"fitted_rates": fitted_rates, "combined_prior": load_record(prior.path)}


@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording],
    published: dict[str, Any],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[PriorUncertaintyResult]:
    yield run_prior_uncertainty(
        homes,
        small_protocol(),
        **published,
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("prior-uncertainty"),
    )


# ----------------------------------------------------------------------------
# The trace
# ----------------------------------------------------------------------------
class TestTrace:
    def test_without_hourly_variation_it_is_the_phase4_trace(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        base = small_protocol().base
        space = tuple(ONTOLOGY.states)
        statistics = {
            h: home_statistics(homes[h], base.resolution, ONTOLOGY, household=h)
            for h in ("sim2", "sim4", "sim6")
        }
        samples = fit_samples(
            statistics, ("sim2", "sim4", "sim6"), states=space, pseudo_windows=12.0
        )
        counts, rows, labels, moments = household_channel_counts(
            homes["sim1"], base.resolution, ONTOLOGY, household="sim1"
        )
        transition = ONTOLOGY.transition(base.resolution.step)
        ours = periodic_trace(
            homes["sim1"],
            ENSEMBLE,
            samples,
            household="sim1",
            counts=counts,
            rows=rows,
            labels=labels,
            moments=moments,
            transitions=np.repeat(transition[None], 24, axis=0),
            start=ONTOLOGY.stationary(),
            protocol=base,
            ontology=ONTOLOGY,
        )
        theirs = household_trace(
            homes["sim1"],
            ENSEMBLE,
            samples,
            household="sim1",
            resolution=base.resolution,
            ontology=ONTOLOGY,
        )
        assert ours.models == theirs.models
        np.testing.assert_allclose(ours.posteriors, theirs.posteriors, atol=1e-12)

    def test_a_training_home_is_refused(self, homes: dict[str, CasasRecording]) -> None:
        base = small_protocol().base
        space = tuple(ONTOLOGY.states)
        statistics = {
            h: home_statistics(homes[h], base.resolution, ONTOLOGY, household=h)
            for h in ("sim2", "sim4")
        }
        samples = fit_samples(
            statistics, ("sim2", "sim4"), states=space, pseudo_windows=12.0
        )
        counts, rows, labels, moments = household_channel_counts(
            homes["sim2"], base.resolution, ONTOLOGY, household="sim2"
        )
        with pytest.raises(ValueError, match="used to fit"):
            periodic_trace(
                homes["sim2"],
                ENSEMBLE,
                samples,
                household="sim2",
                counts=counts,
                rows=rows,
                labels=labels,
                moments=moments,
                transitions=np.repeat(ONTOLOGY.transition(STEP)[None], 24, axis=0),
                start=ONTOLOGY.stationary(),
                protocol=base,
                ontology=ONTOLOGY,
            )


# ----------------------------------------------------------------------------
# The rule for the abstention rule's signal
# ----------------------------------------------------------------------------
def _comparisons(decisions: dict[str, tuple[str, float]]) -> dict[str, Any]:
    return {
        name: {
            "decision": decision,
            "aurc": {"comparison": {"mean": {"estimate": estimate}}},
        }
        for name, (decision, estimate) in decisions.items()
    }


class TestRuleSignal:
    def test_confidence_when_no_structural_signal_is_materially_better(self) -> None:
        chosen = rule_signal(
            _comparisons(
                {
                    "structural_disagreement": ("uncertain", 0.05),
                    "evidence_disagreement": ("worse", -0.1),
                    "predictive_mismatch": ("negligible difference", 0.0),
                }
            )
        )
        assert chosen["signal"] == "confidence"

    def test_the_largest_gain_among_the_materially_better(self) -> None:
        chosen = rule_signal(
            _comparisons(
                {
                    "structural_disagreement": ("materially better", 0.04),
                    "evidence_disagreement": ("materially better", 0.06),
                    "predictive_mismatch": (
                        "better but rejects difficult minority states",
                        0.2,
                    ),
                }
            )
        )
        assert chosen["signal"] == "evidence_disagreement"
        one = rule_signal(
            _comparisons(
                {
                    "structural_disagreement": ("materially better", 0.04),
                    "evidence_disagreement": ("worse", -0.1),
                    "predictive_mismatch": ("worse", -0.1),
                }
            )
        )
        assert one["signal"] == "structural_disagreement"


# ----------------------------------------------------------------------------
# The experiment, on simulated homes
# ----------------------------------------------------------------------------
class TestExperiment:
    def test_the_record_validates_and_reports_the_check(
        self, result: PriorUncertaintyResult
    ) -> None:
        assert result.path is not None
        results = load_record(result.path)["results"]
        check = results["check"]
        assert check["fits"]["reproduced"] and check["fits"]["identical_digests"]
        assert check["population"]["identical"]
        assert check["scores"] == {
            "model": "filter_periodic_hurdle",
            "households": 6,
            "reproduced": True,
            "identical": True,
        }
        assert set(results["decisions"]) == {
            "entropy",
            "structural_disagreement",
            "evidence_disagreement",
            "predictive_mismatch",
        }
        assert results["rule_signal"] == rule_signal(results["comparisons"])
        assert results["reproduction"]["published"] is None
        answer = results["key_question"]
        assert (answer["answer"] == "yes") == bool(answer["materially_better"])
        assert set(answer["materially_better"]) <= set(STRUCTURAL_SIGNALS)

    def test_a_different_published_score_stops_the_run(
        self, homes: dict[str, CasasRecording], published: dict[str, Any]
    ) -> None:
        changed = copy.deepcopy(published)
        metrics = changed["combined_prior"]["household_metrics"]
        metrics["filter_periodic_hurdle@R"]["sim3"]["log_loss"] += 1e-4
        with pytest.raises(ValueError, match="reference scores differ"):
            run_prior_uncertainty(homes, small_protocol(), **changed, data_source="s")

    def test_a_different_published_fit_stops_the_run(
        self, homes: dict[str, CasasRecording], published: dict[str, Any]
    ) -> None:
        changed = copy.deepcopy(published)
        fits = changed["fitted_rates"]["results"]["fitted"]
        fits["b"]["periodic_prior"]["population"][0][0] += 0.01
        with pytest.raises(ValueError, match="fold fits differ"):
            run_prior_uncertainty(homes, small_protocol(), **changed, data_source="s")

    def test_the_results_page_and_figures_are_generated_from_the_record(
        self, result: PriorUncertaintyResult, tmp_path: Path
    ) -> None:
        assert result.path is not None
        payload = load_record(result.path)
        page = render_page(payload)
        assert page.startswith("# Phase 4 for the time prior")
        assert "The 43 held-out CASAS homes are not scored." in page
        assert "frozen external cohort is not touched" not in page
        assert "### The abstention rule's signal" in page
        assert "## Checks" in page and "**Reproduction.**" in page
        assert "nan" not in page.lower()
        figures = draw_figures(payload, tmp_path, prefix=FIGURE_PREFIX)
        for name, path in figures.items():
            assert path.name == f"{FIGURE_PREFIX}-{name}.svg"
            assert f"figures/{path.name}" in page


# ----------------------------------------------------------------------------
# The protocol
# ----------------------------------------------------------------------------
class TestProtocol:
    def test_the_protocol_page_renders_from_a_written_protocol(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "protocol.json"
        write_protocol(small_protocol(), path)
        frozen = json.loads(path.read_text(encoding="utf-8"))
        assert "at_freeze" in frozen
        page = render_protocol(frozen)
        assert "This page reports no result" in page
        assert "nan" not in page.lower()

    def test_a_library_at_another_version_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "protocol.json"
        write_protocol(small_protocol(), path)
        assert library_changes(path) == []
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["at_freeze"]["distributions"]["numpy"] = "0.0.0"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert library_changes(path) == ["numpy"]

    def test_the_workflow_installs_the_versions_the_freeze_needs(self) -> None:
        workflow = (ROOT / WORKFLOW).read_text(encoding="utf-8")
        for name, version in RUN_VERSIONS.items():
            assert f'"{name}=={version}"' in workflow

    def test_any_change_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "protocol.json"
        write_protocol(small_protocol(), path)
        check_frozen_protocol(small_protocol(), path)
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(small_protocol(tolerance=1e-5), path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rule_signal"]["rule"] = "confidence"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(small_protocol(), path)

    def test_a_changed_pinned_record_is_refused(self, tmp_path: Path) -> None:
        for name in PINNED_RECORDS:
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / name, tmp_path / name)
        check_pinned_records(tmp_path)
        with (tmp_path / PHASE4_PROTOCOL).open("a", encoding="utf-8") as f:
            f.write(" ")
        with pytest.raises(ValueError, match="not the file"):
            check_pinned_records(tmp_path)

    def test_the_declared_protocol_reads_the_frozen_bases(self) -> None:
        protocol = declared_protocol(load_frozen_splits(SPLITS), ROOT)
        payload = protocol.to_dict()
        assert payload["predictions"]["model"] == "filter_periodic_hurdle"  # type: ignore[index]
        assert len(protocol.homes) == 20

    @pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan")])
    def test_invalid_tolerances_are_refused(self, tolerance: float) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            small_protocol(tolerance=tolerance)

    def test_bases_on_other_folds_are_refused(self) -> None:
        other = (
            HouseholdSplit("a", train=("sim2", "sim4"), test=("sim1", "sim3")),
            HouseholdSplit("b", train=("sim1", "sim3"), test=("sim2", "sim4")),
        )
        base = UncertaintyProtocol(folds=FOLDS, splits_sha256="0" * 64)
        rates = FittedRatesProtocol(folds=other, splits_sha256="0" * 64)
        with pytest.raises(ValueError, match="same folds"):
            PriorUncertaintyProtocol(base, CombinedPriorProtocol(rates))


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
