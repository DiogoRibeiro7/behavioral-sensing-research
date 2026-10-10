"""Tests for the held-out confirmation.

They guard what the confirmation rests on:
- everything is fitted on the training homes only, exactly as a fold of the
  fitted-rates experiment fits it;
- the recursions are the combined-prior experiment's, and the baselines are
  scored on the same windows;
- the protocol is frozen, names the frozen cohort, pins its records, and
  refuses any change;
- the pages are generated from the frozen file and the record.
"""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sensor_modeling.datasets import ActivityInterval, CasasRecording, HouseholdSplit
from sensor_modeling.datasets.combined_prior_experiment import MODELS
from sensor_modeling.datasets.heldout_experiment import (
    COHORT_MANIFEST,
    MIN_RECALL_HOMES,
    PINNED_RECORDS,
    HeldOutProtocol,
    HeldOutResult,
    check_frozen_protocol,
    check_pinned_records,
    code_changes,
    declared_protocol,
    family,
    inputs_report,
    load_cohort,
    run_heldout,
    training_fit,
    write_protocol,
)
from sensor_modeling.datasets.heldout_summary import (
    PROTOCOL_FILE,
    PROTOCOL_PAGE,
    RECORD_FILE,
    RESULTS_PAGE,
    render_page,
    render_protocol,
)
from sensor_modeling.datasets.rates_experiment import FittedRatesProtocol, fold_fits
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.evaluation import load_record
from sensor_modeling.observations import SensorRegistry
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
ONTOLOGY = StateOntology()
FOLDS = (
    HouseholdSplit("a", train=("sim2", "sim4"), test=("sim1", "sim3")),
    HouseholdSplit("b", train=("sim1", "sim3"), test=("sim2", "sim4")),
)
HELD_OUT = ("new5", "new6", "new7", "new8")
COHORT = tuple(
    {"id": home, "filename": f"{home}.csv", "sha256": "0" * 64} for home in HELD_OUT
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


def unlabelled(recording: CasasRecording) -> CasasRecording:
    return CasasRecording(recording.registry, recording.observations, ())


def small_protocol(inputs: dict[str, Any] | None = None) -> HeldOutProtocol:
    base = FittedRatesProtocol(folds=FOLDS, splits_sha256="0" * 64, resamples=200)
    return HeldOutProtocol(base, COHORT, inputs)


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    training = {f"sim{s}": simulated(s) for s in range(1, 5)}
    return {
        **training,
        "new5": simulated(5),
        "new6": simulated(6),
        "new7": without_kitchen_or_bed_awake(simulated(7)),
        "new8": unlabelled(simulated(8)),
    }


@pytest.fixture(scope="module")
def report(homes: dict[str, CasasRecording]) -> dict[str, Any]:
    return {**inputs_report(homes, small_protocol()), "sha256": "0" * 64}


@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[HeldOutResult]:
    yield run_heldout(
        homes,
        small_protocol(),
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("heldout"),
    )


class TestFitting:
    def test_it_is_a_fold_fit_on_the_training_homes(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        base = small_protocol().base
        fold = fold_fits(homes, base, ONTOLOGY)["a"]
        fit = training_fit(homes, FOLDS[0].train, base, ONTOLOGY)
        assert fit.channels.sha256() == fold.channels.sha256()
        assert fit.periodic.sha256() == fold.periodic.sha256()
        assert fit.shares == fold.shares

    def test_held_out_homes_change_nothing_that_is_fitted(
        self, result: HeldOutResult, homes: dict[str, CasasRecording]
    ) -> None:
        fit = training_fit(
            homes, small_protocol().training, small_protocol().base, ONTOLOGY
        )
        training = result.record.to_dict()["results"]["training"]
        assert training["channels_sha256"] == fit.channels.sha256()
        assert training["homes"] == ["sim1", "sim2", "sim3", "sim4"]


class TestExperiment:
    def test_the_record_validates(self, result: HeldOutResult) -> None:
        assert result.path is not None
        payload = load_record(result.path)
        results = payload["results"]
        assert results["result_schema"] == "heldout-results/1"
        assert results["check"] == {
            "windows": "identical",
            "households": 3,
            "unscored": ["new8"],
        }
        assert [e["key"] for e in results["estimands"]] == [
            "K1",
            "K2",
            "K3",
            "G1",
            "G2",
        ]
        assert results["conclusion"] == results["estimands"][0]["conclusion"]
        recall = results["estimands"][0]["per_state_recall"]
        assert all(  # three scored homes, fewer than the minimum
            entry["verdict"] == "too few homes" for entry in recall.values()
        )
        assert MIN_RECALL_HOMES > 3
        assert all("per_state_recall" not in e for e in results["estimands"][1:])

    def test_every_model_scores_every_held_out_home(
        self, result: HeldOutResult
    ) -> None:
        metrics = result.record.to_dict()["household_metrics"]
        expected = {f"{m}@R" for m in MODELS} | {
            f"{m}@I3"
            for m in (
                "state_frequency",
                "persistence",
                "logistic",
                "tree",
                "diagnostic",
            )
        }
        assert set(metrics) == expected
        assert all(
            set(per_home) == {"new5", "new6", "new7"} for per_home in metrics.values()
        )
        recall = metrics["filter_hurdle@R"]["new7"]["per_class_recall"]
        assert "bed_awake" not in recall or recall["bed_awake"] is None

    def test_gap_estimands_are_read_by_balanced_accuracy(
        self, result: HeldOutResult
    ) -> None:
        reading = {
            "favours model": "the baseline leads",
            "favours reference": "the combination leads",
            "negligible": "no gap",
            "uncertain": "uncertain",
        }
        for entry in result.record.to_dict()["results"]["estimands"][3:]:
            assert entry["model"].endswith("@I3") and entry["reference"].endswith("@R")
            assert (
                entry["conclusion"] == reading[entry["verdicts"]["balanced_accuracy"]]
            )

    def test_families_are_described(self, result: HeldOutResult) -> None:
        families = result.record.to_dict()["results"]["by_family"]
        assert list(families) == ["new"]
        entry = families["new"]
        assert entry["households"] == 3
        assert entry["k1"] == pytest.approx(
            entry["balanced_accuracy"]["filter_periodic_hurdle"]
            - entry["balanced_accuracy"]["filter_hurdle"]
        )

    def test_the_inputs_report(self, report: dict[str, Any]) -> None:
        homes = report["homes"]
        assert set(homes) == {"sim1", "sim2", "sim3", "sim4", *HELD_OUT}
        assert not {"kitchen_motion", "kitchen_contact"} & set(
            homes["new7"]["instrumented"]
        )
        assert "bed_awake" not in homes["new7"]["states"]
        assert homes["new8"]["labelled"] == 0
        assert all(homes[h]["recursions_finite"] for h in HELD_OUT)
        assert all("recursions_finite" not in homes[f"sim{s}"] for s in range(1, 5))

    def test_the_protocol_declares_what_is_missing(
        self, report: dict[str, Any]
    ) -> None:
        declared = small_protocol(report).to_dict()["instrumentation"]
        lacking = declared["held_out_homes_lacking_one"]  # type: ignore[index]
        assert set(lacking) == {"new7"}
        assert set(lacking["new7"]) <= {"kitchen_motion", "kitchen_contact"}

    def test_a_missing_home_stops_the_run(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        partial = {k: v for k, v in homes.items() if k != "new6"}
        with pytest.raises(ValueError, match="no recording"):
            run_heldout(partial, small_protocol(), data_source="sim")

    def test_the_results_page_is_generated_from_the_record(
        self, result: HeldOutResult
    ) -> None:
        assert result.path is not None
        page = render_page(load_record(result.path))
        for key in ("K1", "K2", "K3", "G1", "G2"):
            assert f"| {key} |" in page
        assert "nan" not in page.lower()


class TestProtocol:
    def test_families(self) -> None:
        assert [family(h) for h in ("hh104", "tm001", "rw101", "ihs07")] == [
            "hh",
            "tm",
            "rw",
            "ihs",
        ]
        with pytest.raises(ValueError):
            family("101")

    def test_the_cohort_is_the_frozen_one(self) -> None:
        cohort = load_cohort(ROOT / COHORT_MANIFEST)
        assert len(cohort) == 43
        manifest = json.loads((ROOT / COHORT_MANIFEST).read_text(encoding="utf-8"))
        assert {h["id"]: h["sha256"] for h in cohort} == {
            h["id"]: h["sha256"] for h in manifest["eligible_homes"]
        }

    def test_a_training_home_cannot_be_held_out(self) -> None:
        base = small_protocol().base
        with pytest.raises(ValueError, match="training home"):
            HeldOutProtocol(base, ({"id": "sim1", "filename": "x", "sha256": "0"},))

    @pytest.mark.skipif(not (ROOT / PROTOCOL_FILE).exists(), reason="not frozen")
    def test_the_committed_protocol_is_the_declared_one(self) -> None:
        protocol = declared_protocol(load_frozen_splits(SPLITS), ROOT)
        check_frozen_protocol(protocol, ROOT / PROTOCOL_FILE)
        frozen = json.loads((ROOT / PROTOCOL_FILE).read_text(encoding="utf-8"))
        assert frozen["protocol_sha256"] == protocol.sha256()
        assert len(frozen["households"]["test"]["homes"]) == 43
        assert frozen["households"]["test"]["families"] == {
            "hh": 8,
            "ihs": 2,
            "mn": 3,
            "rw": 5,
            "tm": 25,
        }

    @pytest.mark.skipif(not (ROOT / PROTOCOL_FILE).exists(), reason="not frozen")
    def test_the_protocol_page_is_generated_from_the_frozen_file(self) -> None:
        frozen = json.loads((ROOT / PROTOCOL_FILE).read_text(encoding="utf-8"))
        page = (ROOT / "docs" / PROTOCOL_PAGE).read_text(encoding="utf-8")
        assert page == render_protocol(frozen)

    def test_a_protocol_without_its_inputs_report_cannot_be_frozen(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="inputs report"):
            write_protocol(small_protocol(), tmp_path / "protocol.json")

    def test_a_changed_source_is_reported(
        self, tmp_path: Path, report: dict[str, Any]
    ) -> None:
        path = tmp_path / "protocol.json"
        write_protocol(small_protocol(report), path)
        assert code_changes(path) == []
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = "sensor_modeling/datasets/heldout_experiment.py"
        payload["at_freeze"]["sources"][name] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert code_changes(path) == [name]

    def test_any_change_is_refused(
        self, tmp_path: Path, report: dict[str, Any]
    ) -> None:
        path = tmp_path / "protocol.json"
        write_protocol(small_protocol(report), path)
        check_frozen_protocol(small_protocol(report), path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        changed = copy.deepcopy(payload)
        changed["households"]["test"]["homes"].pop()
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(small_protocol(report), path)

    def test_a_changed_pinned_record_is_refused(self, tmp_path: Path) -> None:
        for name in PINNED_RECORDS:
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / name, tmp_path / name)
        check_pinned_records(tmp_path)
        with (tmp_path / COHORT_MANIFEST).open("a", encoding="utf-8") as handle:
            handle.write(" ")
        with pytest.raises(ValueError, match="not the file"):
            check_pinned_records(tmp_path)


@pytest.mark.skipif(not (ROOT / RECORD_FILE).exists(), reason="not yet run")
class TestRecord:
    def test_the_results_page_is_generated_from_the_record(self) -> None:
        payload: dict[str, Any] = load_record(ROOT / RECORD_FILE)
        page = (ROOT / "docs" / RESULTS_PAGE).read_text(encoding="utf-8")
        assert page == render_page(payload)

    def test_the_record_ran_the_frozen_protocol(self) -> None:
        payload = load_record(ROOT / RECORD_FILE)
        frozen = json.loads((ROOT / PROTOCOL_FILE).read_text(encoding="utf-8"))
        assert payload["configuration"]["protocol_sha256"] == frozen["protocol_sha256"]
        assert payload["environment"]["git_dirty"] == "false"
