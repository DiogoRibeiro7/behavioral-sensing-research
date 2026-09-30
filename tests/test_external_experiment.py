"""Tests for the Phase 5 external evaluation, run as the frozen protocol declares.

The day-resampled metrics must equal the package's own at full sample, the
conclusions must follow the declared criteria, the populations must reproduce
their frozen digests, an ineligible home must be reported rather than hidden,
and the page and figures must be generated from the record alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import ActivityInterval, CasasRecording
from sensor_modeling.datasets.channel_models import home_statistics
from sensor_modeling.datasets.external_experiment import (
    CONDITIONS,
    _conclusion,
    _error_aurc,
    conclusions,
    metrics_from_sums,
    reproduce_populations,
    run_external,
    score_home,
    window_statistics,
)
from sensor_modeling.datasets.external_figures import (
    FIGURES,
    data_sha256,
    draw_figures,
    figure_data,
)
from sensor_modeling.datasets.external_protocol import declared_protocol
from sensor_modeling.datasets.external_summary import render_page
from sensor_modeling.datasets.information_sets import EvidenceResolution
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.structural_models import fit_samples
from sensor_modeling.evaluation import load_record
from sensor_modeling.evaluation.metrics import prediction_metrics
from sensor_modeling.evaluation.selective import (
    HIGHER_IS_RISKIER,
    SelectiveData,
    Signal,
    evaluate_signal,
)
from sensor_modeling.external import InMemoryAdapter
from sensor_modeling.external.ordonez import PROVENANCE, OrdonezAdapter, read_household
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
SPACE = tuple(StateOntology().states)


# ----------------------------------------------------------------------------
class TestMetrics:
    """The additive day statistics reproduce the package's metrics exactly."""

    def test_full_sample_equals_prediction_metrics(self) -> None:
        rng = np.random.default_rng(3)
        n, size = 600, len(SPACE)
        truth = rng.choice([0, 2, 3, 5], size=n)
        posterior = rng.dirichlet(np.ones(size) * 0.7, size=n)
        found = metrics_from_sums(window_statistics(truth, posterior).sum(axis=0), size)
        expected = prediction_metrics(
            [SPACE[t] for t in truth],
            [SPACE[int(i)] for i in posterior.argmax(axis=1)],
            posterior,
            states=SPACE,
        )
        for name in (
            "balanced_accuracy",
            "log_loss",
            "brier",
            "calibration_error",
            "macro_f1",
            "accuracy",
        ):
            assert found[name] == pytest.approx(
                getattr(expected, name), abs=1e-12
            ), name
        for state, recall in expected.per_class_recall.items():
            assert found["recall"][SPACE.index(state)] == pytest.approx(
                recall, abs=1e-12
            )
        assert found["chance"] == pytest.approx(1 / 4)

    def test_day_weights_resample_whole_days(self) -> None:
        rng = np.random.default_rng(4)
        truth = rng.choice([0, 3], size=40)
        posterior = rng.dirichlet(np.ones(len(SPACE)), size=40)
        days = np.repeat(np.arange(4), 10)
        x = window_statistics(truth, posterior)
        by_day = np.zeros((4, x.shape[1]))
        np.add.at(by_day, days, x)
        doubled = metrics_from_sums(np.array([2.0, 0.0, 1.0, 1.0]) @ by_day, len(SPACE))
        keep = np.concatenate([np.flatnonzero(days == d) for d in (0, 0, 2, 3)])
        direct = metrics_from_sums(x[keep].sum(axis=0), len(SPACE))
        for name in ("balanced_accuracy", "log_loss", "calibration_error"):
            assert doubled[name] == pytest.approx(direct[name], abs=1e-12)

    def test_the_aurc_is_the_selective_frameworks(self) -> None:
        rng = np.random.default_rng(5)
        n = 300
        truth = rng.integers(0, 3, n)
        predicted = np.where(rng.random(n) < 0.35, (truth + 1) % 3, truth)
        risk = rng.random(n) + (predicted != truth) * 0.4
        days = np.repeat(np.arange(3), 100)
        grid = np.array([round(0.05 * k, 2) for k in range(1, 21)])
        found = _error_aurc(risk, predicted != truth, days, np.ones((1, 3)), grid)[0]
        data = SelectiveData(
            np.full(n, "h"), truth, predicted, ("a", "b", "c"), np.full(n, 0.5)
        )
        expected = evaluate_signal(
            data, Signal("r", risk, HIGHER_IS_RISKIER), resamples=100
        )
        assert found == pytest.approx(expected.aurc, abs=1e-12)


# ----------------------------------------------------------------------------
class TestCriteria:
    """The declared criteria, over the two homes."""

    @pytest.mark.parametrize(
        ("estimand", "verdicts", "expected"),
        [
            ("T", ["favours zero_shot", "favours zero_shot"], "transfers"),
            ("T", ["negligible", "favours the other"], "does not transfer"),
            ("T", ["favours zero_shot", "uncertain"], "inconclusive"),
            ("A", ["favours adapted", "favours adapted"], "helps"),
            ("A", ["negligible", "negligible"], "does not help"),
            ("C", ["favours zero_shot", "favours zero_shot"], "above chance"),
            ("C", ["favours the other", "negligible"], "not above chance"),
            ("S", ["favours structural", "favours structural"], "survives"),
            ("S", ["favours the other", "favours the other"], "reverses"),
            ("S", ["favours the other", "negligible"], "inconclusive"),
        ],
    )
    def test_conclusions(
        self, estimand: str, verdicts: list[str], expected: str
    ) -> None:
        assert _conclusion(verdicts, estimand) == expected

    def test_log_loss_can_veto_transfer(self) -> None:
        def home(accuracy: str, loss: str) -> dict[str, Any]:
            items = {
                "balanced_accuracy": {"verdict": accuracy, "estimate": 0.1},
                "log_loss": {"verdict": loss, "estimate": -0.1},
                "calibration_error": {"verdict": "uncertain", "estimate": 0.0},
                "brier": {"verdict": "uncertain", "estimate": 0.0},
            }
            return {
                "eligible": True,
                "estimands": {
                    "T": items,
                    "A": items,
                    "C": {
                        "balanced_accuracy": {"verdict": "negligible", "estimate": 0.0}
                    },
                    "S": {"aurc": {"verdict": "uncertain", "estimate": 0.0}},
                },
            }

        found = conclusions(
            {
                "A": home("favours zero_shot", "favours the other"),
                "B": home("favours zero_shot", "negligible"),
            }
        )
        assert found["T"]["conclusion"] == "inconclusive"
        assert found["improved"]["T"]["balanced_accuracy"] == {
            "improved": 2,
            "worsened": 0,
        }
        assert found["improved"]["T"]["log_loss"] == {"improved": 0, "worsened": 2}


# ----------------------------------------------------------------------------
H_S = (
    "Start time          \tEnd time            \tLocation\tType\t\tPlace\n"
    "--------------------\t--------------------\t--------\t--------\t-----\n"
)
H_A = (
    "Start time          \tEnd time            \tActivity\t\n"
    "--------------------\t--------------------\t--------\n"
)


def _t(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def write_home(
    directory: Path, name: str, day0: datetime, days: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    sensors: list[str] = []
    adls: list[str] = []

    def act(start: datetime, minutes: float, where: tuple[str, str, str]) -> None:
        end = start + timedelta(minutes=minutes)
        sensors.append(
            f"{_t(start)}\t\t{_t(end)}\t\t{where[0]}\t\t{where[1]}\t{where[2]}\n"
        )

    def ann(start: datetime, minutes: float, label: str) -> None:
        adls.append(
            f"{_t(start)}\t\t{_t(start + timedelta(minutes=minutes))}\t\t{label}\t\n"
        )

    kitchen = (
        ("Cooktop", "PIR", "Kitchen")
        if name.endswith("A")
        else ("Door", "PIR", "Kitchen")
    )
    for d in range(days):
        b = day0 + timedelta(days=d, minutes=int(rng.integers(0, 30)))
        ann(b + timedelta(minutes=30), 420, "Sleeping")
        act(b + timedelta(minutes=30), 420, ("Bed", "Pressure", "Bedroom"))
        ann(b + timedelta(hours=7, minutes=40), 10, "Toileting")
        for k in range(int(rng.integers(1, 4))):
            act(
                b + timedelta(hours=7, minutes=41 + 2 * k),
                1,
                ("Basin", "PIR", "Bathroom"),
            )
        ann(b + timedelta(hours=8), 20, "Breakfast")
        act(b + timedelta(hours=8, minutes=1), 5, kitchen)
        ann(b + timedelta(hours=9), 180, "Leaving")
        act(b + timedelta(hours=9), 0.1, ("Maindoor", "Magnetic", "Entrance"))
        act(b + timedelta(hours=12), 0.1, ("Maindoor", "Magnetic", "Entrance"))
        ann(b + timedelta(hours=13), 240, "Spare_Time/TV")
        act(b + timedelta(hours=13), 240, ("Seat", "Pressure", "Living"))
        ann(b + timedelta(hours=17, minutes=30), 15, "Showering")
        act(b + timedelta(hours=17, minutes=31), 10, ("Shower", "PIR", "Bathroom"))
    (directory / f"{name}_Sensors.txt").write_text(
        H_S + "".join(sensors), encoding="latin-1"
    )
    (directory / f"{name}_ADLs.txt").write_text(H_A + "".join(adls), encoding="latin-1")


def simulated(seed: int) -> CasasRecording:
    sim = simulate(HouseholdConfig(days=2, seed=seed))
    return CasasRecording(
        sim.registry,
        sim.observations,
        tuple(
            ActivityInterval(e.state.value, e.start, e.end, e.state)
            for e in sim.truth.episodes
        ),
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    ontology, resolution = StateOntology(), EvidenceResolution()
    homes = {f"sim{s}": simulated(s) for s in range(1, 5)}
    statistics = {
        h: home_statistics(r, resolution, ontology, household=h)
        for h, r in homes.items()
    }
    samples = fit_samples(
        statistics, sorted(homes), states=tuple(ontology.states), pseudo_windows=12.0
    )
    data = tmp_path_factory.mktemp("ordonez")
    write_home(data, "OrdonezA", datetime(2011, 11, 28), 14, 1)
    write_home(data, "OrdonezB", datetime(2012, 11, 11), 21, 2)
    protocol = declared_protocol(load_frozen_splits(SPLITS))
    out = tmp_path_factory.mktemp("results")
    development = {
        "record": "test",
        "calibration": {
            "mean_confidence": 0.8,
            "accuracy": 0.5,
            "calibration_error": 0.3,
            "balanced_accuracy": 0.42,
        },
        "states": {s.value: {"share": 1 / 7} for s in ontology.states},
    }
    result = run_external(
        OrdonezAdapter(data),
        samples,
        protocol,
        protocol_sha256="0" * 64,
        development=development,
        data_source="synthetic",
        output_dir=out,
    )
    assert result.path is not None
    return {
        "samples": samples,
        "protocol": protocol,
        "data": data,
        "homes": homes,
        "path": result.path,
        "payload": load_record(result.path),
    }


class TestRun:
    """The whole evaluation on synthetic homes in the dataset's format."""

    def test_every_home_and_estimand_is_reported(self, world: dict[str, Any]) -> None:
        results = world["payload"]["results"]
        assert set(results["households"]) == {"OrdonezA", "OrdonezB"}
        for entry in results["households"].values():
            assert entry["eligible"]
            assert set(entry["estimands"]) == {"T", "A", "C", "S"}
            counts = {entry["conditions"][c]["n"] for c in CONDITIONS}
            assert counts == {entry["scored_windows"]}  # the same windows
            assert entry["scored_days"] and len(entry["scored_days"]) >= 7
        for key in ("T", "A", "C", "S"):
            assert results["conclusions"][key]["conclusion"]

    def test_conclusions_follow_the_recorded_verdicts(
        self, world: dict[str, Any]
    ) -> None:
        results = world["payload"]["results"]
        assert results["conclusions"] == conclusions(results["households"])

    def test_the_attribution_is_consistent(self, world: dict[str, Any]) -> None:
        for entry in world["payload"]["results"]["households"].values():
            a = entry["attribution"]
            zero = entry["conditions"]["zero_shot"]["balanced_accuracy"]["estimate"]
            oracle = entry["diagnostics"]["oracle"]["balanced_accuracy"]["estimate"]
            assert a["model_failure"]["zero_shot_below_oracle"] == pytest.approx(
                oracle - zero
            )
            assert a["sensing_limitation"]["oracle_shortfall"] == pytest.approx(
                1 - oracle
            )
            coverage = entry["diagnostics"]["mapping_coverage"]
            assert 0.0 <= coverage["scorable_fraction"] <= 1.0
            assert "ambiguous" in coverage["seconds"]
            obs = entry["diagnostics"]["unsupported_observations"]
            assert obs["unsupported"] == obs["activations"] - obs["by_reason"]["routed"]

    def test_the_record_is_strict_json(self, world: dict[str, Any]) -> None:
        json.loads(
            world["path"].read_text(encoding="utf-8"),
            parse_constant=lambda token: pytest.fail(token),
        )

    def test_the_page_and_figures_come_from_the_record(
        self, world: dict[str, Any], tmp_path: Path
    ) -> None:
        payload = world["payload"]
        page = render_page(payload)
        assert page == render_page(load_record(world["path"]))
        for home in ("OrdonezA", "OrdonezB"):
            assert f"### {home}" in page
        first = draw_figures(payload, tmp_path / "a")
        second = draw_figures(payload, tmp_path / "b")
        data = figure_data(payload)
        for name in FIGURES:
            assert first[name].read_bytes() == second[name].read_bytes()
            assert f"data sha256 {data_sha256(data[name])}" in first[name].read_text(
                encoding="utf-8"
            )

    def test_an_ineligible_home_is_reported_not_hidden(
        self, world: dict[str, Any]
    ) -> None:
        data = read_household(world["data"], "OrdonezA")
        broken = replace(data, timezone=None)
        entry = score_home(broken, world["samples"], world["protocol"])
        assert entry["eligible"] is False
        assert entry["validation"]["blocking"] == ["missing_timezone"]
        result = run_external(
            InMemoryAdapter(PROVENANCE, {"OrdonezA": broken}),
            world["samples"],
            world["protocol"],
            protocol_sha256="0" * 64,
            data_source="synthetic",
        )
        payload = result.record.to_dict()
        assert payload["results"]["households"]["OrdonezA"]["eligible"] is False
        assert "Not eligible" in render_page(json.loads(json.dumps(payload)))

    def test_populations_must_reproduce_their_digests(
        self, world: dict[str, Any]
    ) -> None:
        @dataclass
        class Protocol:
            development_homes: tuple[str, ...]
            resolution: EvidenceResolution
            pseudo_windows: float

        with pytest.raises(ValueError, match="does not reproduce"):
            reproduce_populations(
                world["homes"],
                Protocol(tuple(sorted(world["homes"])), EvidenceResolution(), 12.0),  # type: ignore[arg-type]
            )
