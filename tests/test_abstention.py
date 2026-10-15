"""Tests for the abstention-rule selection: the rule, the run, the protocol."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import ActivityInterval, CasasRecording, HouseholdSplit
from sensor_modeling.datasets.abstention_experiment import (
    PROTOCOL_FILE,
    RECORD_FILE,
    AbstentionProtocol,
    AbstentionResult,
    check_frozen_protocol,
    code_changes,
    declared_protocol,
    run_abstention,
    select_threshold,
    write_protocol,
)
from sensor_modeling.datasets.abstention_summary import (
    PROTOCOL_PAGE,
    RESULTS_PAGE,
    render_page,
    render_protocol,
)
from sensor_modeling.datasets.combined_prior_experiment import (
    CombinedPriorProtocol,
    run_combined_prior,
)
from sensor_modeling.datasets.prior_uncertainty_experiment import (
    PriorUncertaintyProtocol,
)
from sensor_modeling.datasets.rates_experiment import (
    FittedRatesProtocol,
    run_fitted_rates,
)
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.uncertainty_experiment import UncertaintyProtocol
from sensor_modeling.evaluation import load_record
from sensor_modeling.simulation import HouseholdConfig, simulate

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
STATES = [
    "away",
    "home_active",
    "home_inactive",
    "sleeping",
    "bed_awake",
    "bathroom_activity",
    "kitchen_activity",
]
FOLDS = (
    HouseholdSplit("a", train=("sim2", "sim4", "sim6"), test=("sim1", "sim3", "sim5")),
    HouseholdSplit("b", train=("sim1", "sim3", "sim5"), test=("sim2", "sim4", "sim6")),
)
STEP1 = {"results": {"rule_signal": {"signal": "structural_disagreement"}}}


def small_protocol(**changes: Any) -> AbstentionProtocol:
    base = UncertaintyProtocol(
        folds=FOLDS, splits_sha256="0" * 64, curve_resamples=200, resamples=1000
    )
    rates = FittedRatesProtocol(folds=FOLDS, splits_sha256="0" * 64, resamples=200)
    first = PriorUncertaintyProtocol(base, CombinedPriorProtocol(rates))
    return AbstentionProtocol(first, **changes)


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


# ----------------------------------------------------------------------------
# The selection
# ----------------------------------------------------------------------------
class TestSelection:
    def test_the_target_is_kept_when_the_guard_holds(self) -> None:
        signal = np.arange(100, dtype=float)
        truth = np.zeros(100, dtype=int)
        chosen = select_threshold(signal, truth, STATES, small_protocol())
        assert chosen["coverage"] == pytest.approx(0.8)
        assert chosen["abstains"] and chosen["tried"][-1]["coverage"] == 0.8

    def test_the_guard_raises_the_coverage(self) -> None:
        # Kitchen windows carry the highest signal: kept only near full coverage.
        signal = np.arange(100, dtype=float)
        truth = np.zeros(100, dtype=int)
        truth[90:] = STATES.index("kitchen_activity")
        chosen = select_threshold(signal, truth, STATES, small_protocol())
        assert [t["guard_holds"] for t in chosen["tried"]][-1]
        assert chosen["coverage"] > 0.9

    def test_never_abstain_when_the_guard_never_holds(self) -> None:
        signal = np.arange(100, dtype=float)
        truth = np.zeros(100, dtype=int)
        truth[97:] = STATES.index("bathroom_activity")
        chosen = select_threshold(signal, truth, STATES, small_protocol())
        assert chosen["coverage"] == 1.0
        assert chosen["threshold"] is None and not chosen["abstains"]

    def test_ties_are_kept_and_absent_states_not_judged(self) -> None:
        signal = np.array([0.0] * 90 + [1.0] * 10)
        truth = np.zeros(100, dtype=int)
        chosen = select_threshold(signal, truth, STATES, small_protocol())
        first = chosen["tried"][0]
        assert first["coverage"] == pytest.approx(0.9)
        assert first["ratios"]["kitchen_activity"] is None

    def test_an_undefined_signal_is_refused(self) -> None:
        with pytest.raises(ValueError, match="defined"):
            select_threshold(
                np.array([0.1, math.nan]),
                np.zeros(2, dtype=int),
                STATES,
                small_protocol(),
            )


# ----------------------------------------------------------------------------
# The run, on simulated homes
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {f"sim{s}": simulated(s) for s in range(1, 7)}


@pytest.fixture(scope="module")
def published(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    combined = small_protocol().step1.combined
    rates = run_fitted_rates(
        homes, combined.base, data_source="sim", output_dir=tmp_path_factory.mktemp("r")
    )
    assert rates.path is not None
    fitted = load_record(rates.path)
    prior = run_combined_prior(
        homes,
        combined,
        fitted,
        data_source="sim",
        output_dir=tmp_path_factory.mktemp("c"),
    )
    assert prior.path is not None
    return {
        "fitted_rates": fitted,
        "combined_prior": load_record(prior.path),
        "step1_record": STEP1,
    }


@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording],
    published: dict[str, Any],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[AbstentionResult]:
    yield run_abstention(
        homes,
        small_protocol(),
        **published,
        data_source="sim",
        output_dir=tmp_path_factory.mktemp("abstention"),
    )


class TestRun:
    def test_the_record_validates_and_holds_the_rule(
        self, result: AbstentionResult
    ) -> None:
        assert result.path is not None
        results = load_record(result.path)["results"]
        assert results["check"]["scores"]["identical"]
        assert results["check"]["fits"]["reproduced"]
        assert set(results["folds"]) == {"a", "b"}
        assert len(results["households"]) == 6
        assert results["decision"] in (
            "supported on development data",
            "not supported",
            "inconclusive",
        )
        assert results["rule"]["signal"] == "structural_disagreement"
        for home, entry in results["households"].items():
            assert 0.0 < entry["coverage"] <= 1.0, home

    def test_another_signal_from_step1_stops_the_run(
        self, homes: dict[str, CasasRecording], published: dict[str, Any]
    ) -> None:
        changed = {
            **published,
            "step1_record": {"results": {"rule_signal": {"signal": "confidence"}}},
        }
        with pytest.raises(ValueError, match="did not pick"):
            run_abstention(homes, small_protocol(), **changed, data_source="sim")

    def test_the_results_page_is_generated_from_the_record(
        self, result: AbstentionResult
    ) -> None:
        assert result.path is not None
        page = render_page(load_record(result.path))
        assert "## Selection" in page and "A1, error" in page
        assert "nan" not in page.lower()


# ----------------------------------------------------------------------------
# The protocol
# ----------------------------------------------------------------------------
class TestProtocol:
    def test_any_change_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "protocol.json"
        write_protocol(small_protocol(), path)
        check_frozen_protocol(small_protocol(), path)
        assert "This page reports no result" in render_protocol(
            json.loads(path.read_text(encoding="utf-8"))
        )
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(small_protocol(fraction=0.7), path)

    @pytest.mark.parametrize(
        "changes", [{"target": 0.82}, {"target": 1.0}, {"fraction": 1.5}]
    )
    def test_invalid_settings_are_refused(self, changes: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            small_protocol(**changes)

    def test_the_declared_protocol_reads_step1(self) -> None:
        protocol = declared_protocol(load_frozen_splits(SPLITS), ROOT)
        assert protocol.coverages == (0.8, 0.85, 0.9, 0.95, 1.0)


@pytest.mark.skipif(not (ROOT / PROTOCOL_FILE).exists(), reason="not yet frozen")
class TestFrozen:
    def test_the_committed_protocol_and_page(self) -> None:
        protocol = declared_protocol(load_frozen_splits(SPLITS), ROOT)
        check_frozen_protocol(protocol, ROOT / PROTOCOL_FILE)
        frozen = json.loads((ROOT / PROTOCOL_FILE).read_text(encoding="utf-8"))
        page = (ROOT / "docs" / PROTOCOL_PAGE).read_text(encoding="utf-8")
        assert page == render_protocol(frozen)

    @pytest.mark.skipif((ROOT / RECORD_FILE).exists(), reason="already run")
    def test_no_pinned_source_has_changed_before_the_run(self) -> None:
        assert code_changes(ROOT / PROTOCOL_FILE, ROOT) == []


@pytest.mark.skipif(not (ROOT / RECORD_FILE).exists(), reason="not yet run")
def test_the_results_page_is_generated_from_the_record() -> None:
    payload = load_record(ROOT / RECORD_FILE)
    page = (ROOT / "docs" / RESULTS_PAGE).read_text(encoding="utf-8")
    assert page == render_page(payload)
    assert payload["environment"]["git_dirty"] == "false"
