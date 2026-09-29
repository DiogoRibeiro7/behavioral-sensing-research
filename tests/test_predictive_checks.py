"""Tests for the posterior predictive checks of the fitted hurdle model.

They guard what makes the diagnostic interpretable:
- the protocol that ran is the one frozen before any household was examined;
- the checks accept a correctly specified Poisson model and detect a clearly
  over-dispersed, negative-binomial-like process, and clustering in time;
- the rules that read the result are the declared ones;
- households stay visible, and the record, its summary and its figures are
  reproducible from the artifact.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.stats import poisson

from sensor_modeling.datasets import ActivityInterval, CasasRecording, HouseholdSplit
from sensor_modeling.datasets.predictive_checks import (
    CRITERIA,
    OWN,
    POPULATION,
    STATISTICS,
    PredictiveProtocol,
    PredictiveResult,
    across_households,
    cell_verdict,
    check_cell,
    check_frozen_protocol,
    declared_protocol,
    hurdle_bins,
    observed_bins,
    pattern_conclusion,
    route,
    run_predictive_checks,
    run_windows,
    sample_hurdle,
    stretches_of,
    ztp_moments,
    ztp_quantile,
    ztp_tail,
)
from sensor_modeling.datasets.predictive_figures import (
    FIGURES,
    data_sha256,
    draw_figures,
    figure_data,
)
from sensor_modeling.datasets.predictive_summary import render_summary
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.evaluation import load_record
from sensor_modeling.simulation import HouseholdConfig, simulate

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
PROTOCOL = ROOT / "artifacts" / "phase3" / "predictive_protocol.json"
FOLDS = (
    HouseholdSplit("a", train=("sim2", "sim4", "sim6"), test=("sim1", "sim3", "sim5")),
    HouseholdSplit("b", train=("sim1", "sim3", "sim5"), test=("sim2", "sim4", "sim6")),
)


def small_protocol(**changes: Any) -> PredictiveProtocol:
    """The declared settings on simulated folds, with a lighter bootstrap."""
    settings: dict[str, Any] = {
        "folds": FOLDS,
        "splits_sha256": "0" * 64,
        "resamples": 200,
        "min_households": 3,
    }
    settings.update(changes)
    return PredictiveProtocol(**settings)


def numbers(value: Any) -> Iterator[float]:
    """Every number in a nested structure."""
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from numbers(item)


# ----------------------------------------------------------------------------
# The protocol
# ----------------------------------------------------------------------------
class TestProtocol:
    def test_the_committed_protocol_is_the_declared_one(self) -> None:
        declared = declared_protocol(load_frozen_splits(SPLITS))
        check_frozen_protocol(declared, PROTOCOL)
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert frozen["protocol_sha256"] == declared.sha256()

    def test_it_freezes_everything_the_diagnostic_depends_on(self) -> None:
        declared: dict[str, Any] = declared_protocol(
            load_frozen_splits(SPLITS)
        ).to_dict()
        assert set(declared["statistics"]) == set(STATISTICS)
        assert declared["pseudo_windows"] == 12.0
        assert declared["runs"]["quiet_run_windows"] == 12
        assert declared["runs"]["burst_windows"] == 3
        assert declared["replicates"]["count"] == 200
        assert declared["minimal_ratio"] == 1.25
        assert declared["common_states"] == [
            "away",
            "home_active",
            "home_inactive",
            "sleeping",
        ]
        assert declared["criteria"] == CRITERIA
        assert declared["bootstrap"]["resamples"] == 10_000
        assert set(declared["references"]) == {OWN, POPULATION}

    @pytest.mark.parametrize(
        "change",
        [
            {"replicates": 400},
            {"minimal_ratio": 1.5},
            {"quiet_run_windows": 6},
            {"min_active": 20},
            {"pseudo_windows": 24.0},
        ],
    )
    def test_any_change_to_the_protocol_is_refused(
        self, change: dict[str, Any]
    ) -> None:
        declared = declared_protocol(load_frozen_splits(SPLITS))
        with pytest.raises(ValueError, match="cannot change after the diagnostic"):
            check_frozen_protocol(replace(declared, **change), PROTOCOL)

    def test_a_tampered_protocol_file_is_refused(self, tmp_path: Path) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        frozen["minimal_ratio"] = 1.1
        tampered = tmp_path / "protocol.json"
        tampered.write_text(json.dumps(frozen), encoding="utf-8")
        declared = declared_protocol(load_frozen_splits(SPLITS))
        with pytest.raises(ValueError, match="cannot change after the diagnostic"):
            check_frozen_protocol(declared, tampered)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"folds": ()}, "at least one fold"),
            (
                {
                    "folds": (
                        HouseholdSplit("a", train=("x",), test=("y",)),
                        HouseholdSplit("b", train=("z",), test=("y",)),
                    )
                },
                "exactly once",
            ),
            ({"splits_sha256": "0"}, "splits_sha256"),
            ({"replicates": 10}, "at least 40"),
            ({"min_active": 0}, "min_active"),
            ({"minimal_ratio": 1.0}, "minimal_ratio"),
            ({"band": 1.0}, "band"),
            ({"common_states": ()}, "common_states"),
            ({"bin_edges": (1, 2, 3)}, "bin_edges"),
            ({"min_runs": 0.0}, "min_runs"),
        ],
    )
    def test_invalid_protocols_are_refused(
        self, change: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            small_protocol(**change)


# ----------------------------------------------------------------------------
# The zero-truncated Poisson and the hurdle
# ----------------------------------------------------------------------------
class TestHurdle:
    @pytest.mark.parametrize("rate", [0.05, 0.7, 3.0, 25.0])
    def test_moments_quantiles_and_tails_match_enumeration(self, rate: float) -> None:
        k = np.arange(1, 400)
        pmf = poisson.pmf(k, rate) / -math.expm1(-rate)
        mean, variance = ztp_moments(rate)
        assert float(mean) == pytest.approx(float((k * pmf).sum()), rel=1e-9)
        assert float(variance) == pytest.approx(
            float((k * k * pmf).sum() - (k * pmf).sum() ** 2), rel=1e-7
        )
        cdf = np.cumsum(pmf)
        for q in (0.5, 0.9, 0.99):
            assert ztp_quantile(rate, q) == int(k[np.argmax(cdf >= q - 1e-12)])
        top = ztp_quantile(rate, 0.99)
        assert ztp_tail(rate, top) == pytest.approx(
            float(pmf[k > top].sum()), abs=1e-12
        )

    def test_the_bins_are_the_hurdle_distribution(self) -> None:
        edges = (0, 1, 2, 4, 8)
        probabilities = hurdle_bins(0.6, 2.0, edges)
        assert sum(probabilities) == pytest.approx(1.0)
        assert probabilities[0] == 0.6
        pmf = 0.4 * poisson.pmf(np.arange(1, 400), 2.0) / -math.expm1(-2.0)
        assert probabilities[2] == pytest.approx(pmf[1] + pmf[2])
        assert observed_bins(np.array([0, 0, 1, 2, 3, 5, 9, 40]), edges) == [
            2,
            1,
            2,
            1,
            2,
        ]

    def test_the_sampler_draws_the_hurdle(self) -> None:
        draws = sample_hurdle(np.random.default_rng(0), 0.3, 2.5, (200_000,))
        active = draws[draws > 0]
        mean, variance = ztp_moments(2.5)
        assert np.mean(draws == 0) == pytest.approx(0.3, abs=0.005)
        assert active.min() >= 1
        assert active.mean() == pytest.approx(float(mean), rel=0.01)
        assert active.var() == pytest.approx(float(variance), rel=0.03)

    def test_windows_in_long_runs_respect_stretches(self) -> None:
        flags = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1], dtype=bool)
        one = np.zeros(12, dtype=int)
        assert run_windows(flags, one, 3).tolist() == [10]
        assert run_windows(flags, one, 4).tolist() == [4]
        split = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        assert run_windows(flags, split, 3).tolist() == [3 + 3]
        both = np.vstack([flags, ~flags])
        assert run_windows(both, one, 1).tolist() == [10, 2]

    def test_stretches_break_where_rows_are_not_consecutive(self) -> None:
        assert stretches_of(np.array([3, 4, 5, 9, 10, 20])).tolist() == [
            0,
            0,
            0,
            1,
            1,
            2,
        ]


# ----------------------------------------------------------------------------
# The checks, on processes whose answer is known
# ----------------------------------------------------------------------------
PROTOCOL_FOR_CELLS = small_protocol()
FLAGGED = {"above", "below"}


def poisson_counts(seed: int, size: int = 1500, rate: float = 1.2) -> np.ndarray:
    return np.random.default_rng(1000 + seed).poisson(rate, size).astype(float)


def negative_binomial_counts(seed: int, size: int = 1500) -> np.ndarray:
    """Mean 1.2 with variance 1.2 + 1.2^2 / 0.5: a gamma mixture of Poissons."""
    rng = np.random.default_rng(2000 + seed)
    return rng.negative_binomial(0.5, 0.5 / (0.5 + 1.2), size).astype(float)


def clustered_counts(seed: int, size: int = 1500) -> np.ndarray:
    """Quiet and busy spells that persist: dependence in time given the state."""
    rng = np.random.default_rng(3000 + seed)
    busy = np.zeros(size, dtype=bool)
    state = False
    for i in range(size):
        if rng.random() < 0.02:
            state = not state
        busy[i] = state
    return np.where(busy, rng.poisson(3.0, size), rng.poisson(0.05, size)).astype(float)


def verdicts(counts: np.ndarray, seed: int, reference: str = OWN, **kw: Any) -> dict:
    out = check_cell(
        counts,
        np.zeros(counts.size, dtype=int),
        reference,
        kw.get("parameters"),
        PROTOCOL_FOR_CELLS,
        np.random.default_rng(seed),
    )
    return {name: entry["verdict"] for name, entry in out["statistics"].items()}


class TestCalibration:
    def test_a_correctly_specified_poisson_is_accepted(self) -> None:
        outside = 0
        for seed in range(40):
            found = verdicts(poisson_counts(seed), seed)
            for name in ("variance", "active_dispersion", "q90", "q99"):
                assert found[name] not in FLAGGED, (seed, name)
            outside += found["active_dispersion"] != "consistent"
        assert outside <= 6  # a 95% band leaves about 2 of 40 outside
        assert {verdicts(poisson_counts(0), 0)[n] for n in ("silence", "mean")} == {
            "fitted"
        }

    def test_its_household_values_read_as_negligible(self) -> None:
        values = {}
        for seed in range(8):
            out = check_cell(
                poisson_counts(seed),
                np.zeros(1500, dtype=int),
                OWN,
                None,
                PROTOCOL_FOR_CELLS,
                np.random.default_rng(seed),
            )
            values[f"h{seed}"] = out["statistics"]["active_dispersion"]["log_ratio"]
        summary = across_households(values, PROTOCOL_FOR_CELLS)
        assert summary["verdict"] == "negligible"
        assert pattern_conclusion(dict.fromkeys("abcd", summary["verdict"])) == "absent"

    def test_an_over_dispersed_negative_binomial_is_detected(self) -> None:
        values = {}
        for seed in range(8):
            counts = negative_binomial_counts(seed)
            out = check_cell(
                counts,
                np.zeros(counts.size, dtype=int),
                OWN,
                None,
                PROTOCOL_FOR_CELLS,
                np.random.default_rng(seed),
            )
            stats = out["statistics"]
            assert stats["active_dispersion"]["verdict"] == "above"
            assert stats["active_dispersion"]["log_ratio"] > math.log(1.25)
            assert stats["q99"]["verdict"] == "above"
            assert stats["tail_excess"]["verdict"] == "above"
            values[f"h{seed}"] = stats["active_dispersion"]["log_ratio"]
        summary = across_households(values, PROTOCOL_FOR_CELLS)
        assert summary["verdict"] == "positive"
        assert summary["above"] == 8
        states = dict.fromkeys(("away", "home_active", "home_inactive", "sleeping"))
        found = pattern_conclusion({s: summary["verdict"] for s in states})
        assert found == "systematic"
        assert route(
            {"active_dispersion": found, "quiet_runs": "absent", "bursts": "absent"}
        )["family"] == ("dispersion")

    def test_clustering_in_time_is_detected_and_independence_is_not(self) -> None:
        for seed in range(6):
            clustered = verdicts(clustered_counts(seed), seed)
            assert clustered["quiet_runs"] == "above"
            assert clustered["bursts"] == "above"
            independent = verdicts(
                sample_hurdle(np.random.default_rng(seed), 0.6, 2.0, (1500,)), seed
            )
            assert independent["quiet_runs"] not in FLAGGED
            assert independent["bursts"] not in FLAGGED

    def test_the_population_reference_sees_a_different_household(self) -> None:
        counts = sample_hurdle(np.random.default_rng(7), 0.5, 2.0, (1500,))
        same = verdicts(counts, 7, POPULATION, parameters=(0.5, 2.0))
        assert same["silence"] in ("consistent", "small")
        assert same["active_dispersion"] in ("consistent", "small")
        quieter = verdicts(counts, 7, POPULATION, parameters=(0.8, 2.0))
        assert quieter["silence"] == "below"
        busier = verdicts(counts, 7, POPULATION, parameters=(0.5, 0.5))
        assert busier["active_mean"] == "above"

    def test_a_cell_without_silence_or_activity_has_no_own_fit(self) -> None:
        out = check_cell(
            np.ones(200),
            np.zeros(200, dtype=int),
            OWN,
            None,
            PROTOCOL_FOR_CELLS,
            np.random.default_rng(0),
        )
        assert out == {"parameters": None, "statistics": {}}


# ----------------------------------------------------------------------------
# The rules
# ----------------------------------------------------------------------------
class TestRules:
    @pytest.mark.parametrize(
        ("ratio", "band", "expected"),
        [
            (0.1, [-0.2, 0.2], "consistent"),
            (0.1, [-0.05, 0.05], "small"),
            (0.5, [-0.05, 0.05], "above"),
            (-0.5, [-0.05, 0.05], "below"),
            (None, [-0.05, 0.05], "not estimable"),
            (0.5, None, "not estimable"),
        ],
    )
    def test_cell_verdicts(
        self, ratio: float | None, band: list[float] | None, expected: str
    ) -> None:
        assert cell_verdict(ratio, band, math.log(1.25)) == expected

    @pytest.mark.parametrize(
        ("verdicts", "expected"),
        [
            (("positive",) * 4, "systematic"),
            (("positive", "positive", "positive", "uncertain"), "systematic"),
            (("positive", "positive", "positive", "negative"), "partial"),
            (("positive", "negligible", "negligible", "negligible"), "partial"),
            (("negligible", "negative", "negligible", "negligible"), "absent"),
            (("negligible", "uncertain", "negligible", "negligible"), "inconclusive"),
            (("insufficient",) * 4, "inconclusive"),
        ],
    )
    def test_pattern_conclusions(
        self, verdicts: tuple[str, ...], expected: str
    ) -> None:
        assert pattern_conclusion(dict(zip("abcd", verdicts))) == expected

    @pytest.mark.parametrize(
        ("conclusions", "family"),
        [
            (("systematic", "absent", "absent"), "dispersion"),
            (("systematic", "systematic", "absent"), "temporal"),
            (("absent", "absent", "systematic"), "temporal"),
            (("partial", "partial", "partial"), None),
            (("absent", "absent", "absent"), None),
        ],
    )
    def test_routing(
        self, conclusions: tuple[str, str, str], family: str | None
    ) -> None:
        found = route(
            dict(zip(("active_dispersion", "quiet_runs", "bursts"), conclusions))
        )
        assert found["family"] == family
        assert found["roadmap"] == (
            "unchanged" if family is None else "names the family"
        )

    def test_too_few_households_are_insufficient(self) -> None:
        summary = across_households({"a": 0.5, "b": 0.6}, PROTOCOL_FOR_CELLS)
        assert summary["verdict"] == "insufficient"
        assert summary["values"] == {"a": 0.5, "b": 0.6}


# ----------------------------------------------------------------------------
# A run on simulated households
# ----------------------------------------------------------------------------
def simulated(seed: int) -> CasasRecording:
    sim = simulate(HouseholdConfig(days=4, seed=seed))
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
    return {f"sim{s}": simulated(s) for s in range(1, 7)}


@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> PredictiveResult:
    return run_predictive_checks(
        homes,
        small_protocol(),
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("predictive"),
    )


class TestExperiment:
    def test_the_record_validates_and_estimates_no_state(
        self, result: PredictiveResult
    ) -> None:
        assert result.path is not None
        payload = load_record(result.path)
        assert payload["results"]["result_schema"] == "predictive-checks-results/1"
        assert payload["configuration"]["protocol_sha256"] == small_protocol().sha256()
        assert payload["inference"]["mode"] == "online_filter"
        assert payload["inference"]["evidence"]["enumerated"] is False

    def test_every_household_stays_visible(self, result: PredictiveResult) -> None:
        results = result.record.to_dict()["results"]
        assert sorted(results["households"]) == [f"sim{s}" for s in range(1, 7)]
        for entry in results["households"].values():
            assert entry["cells"]
            for cell in entry["cells"]:
                assert cell["windows"] >= 50
                assert cell["windows"] == cell["silent"] + cell["active"]
                assert sum(cell["observed_bins"]) == cell["windows"]
                own = cell["references"][OWN]
                if own["parameters"] is not None:
                    for name in ("silence", "mean", "active_mean"):
                        assert own["statistics"][name]["verdict"] in (
                            "fitted",
                            "not estimable",
                        )
        for summary in results["groups"]["state"][OWN]["variance"].values():
            assert set(summary["values"]) <= set(results["households"])

    def test_household_values_are_the_mean_of_their_cells(
        self, result: PredictiveResult
    ) -> None:
        results = result.record.to_dict()["results"]
        for state, summary in results["groups"]["state"][OWN]["variance"].items():
            for home, value in summary["values"].items():
                ratios = [
                    c["references"][OWN]["statistics"]["variance"]["log_ratio"]
                    for c in results["households"][home]["cells"]
                    if c["state"] == state
                    and c["references"][OWN]["statistics"]
                    .get("variance", {})
                    .get("verdict")
                    not in (None, "not estimable")
                ]
                assert value == pytest.approx(float(np.mean(ratios)))

    def test_the_population_is_fitted_on_training_homes_only(
        self, result: PredictiveResult
    ) -> None:
        fitted = result.record.to_dict()["results"]["fitted"]
        assert fitted["a"]["fitted_on"] == ["sim2", "sim4", "sim6"]
        assert fitted["b"]["fitted_on"] == ["sim1", "sim3", "sim5"]

    def test_conclusions_follow_the_rules(self, result: PredictiveResult) -> None:
        results = result.record.to_dict()["results"]
        for key, entry in results["conclusions"].items():
            by_state = results["groups"]["state"][OWN][key]
            for state, verdict in entry["verdicts"].items():
                assert (
                    verdict
                    == by_state.get(state, {"verdict": "insufficient"})["verdict"]
                )
            assert entry["conclusion"] == pattern_conclusion(entry["verdicts"])
        assert results["routing"] == route(
            {k: v["conclusion"] for k, v in results["conclusions"].items()}
        )

    def test_a_second_run_gives_the_same_record(
        self, homes: dict[str, CasasRecording], result: PredictiveResult
    ) -> None:
        again = run_predictive_checks(homes, small_protocol(), data_source="simulator")
        first, second = result.record.to_dict(), again.record.to_dict()
        first.pop("recorded_at")
        second.pop("recorded_at")
        assert first == second

    def test_missing_recordings_are_refused(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        partial = {h: r for h, r in homes.items() if h != "sim4"}
        with pytest.raises(ValueError, match="sim4"):
            run_predictive_checks(partial, small_protocol(), data_source="simulator")

    def test_the_summary_is_generated_from_the_written_record(
        self, result: PredictiveResult
    ) -> None:
        assert result.path is not None
        summary = render_summary(load_record(result.path))
        assert summary == render_summary(
            json.loads(json.dumps(result.record.to_dict()))
        )
        for heading in (
            "## Pre-specified conclusions",
            "## The shape of the active count, own reference",
            "## Runs in time, own reference",
            "## Against the population model",
            "## By channel, room and channel type",
            "## Variance and mean",
            "## Cells by verdict",
            "## Households",
        ):
            assert heading in summary
        payload = result.record.to_dict()
        payload["results"] = {"result_schema": "silence-results/1"}
        with pytest.raises(ValueError, match="predictive-checks-results"):
            render_summary(payload)

    def test_figures_are_drawn_from_the_record_and_reproducible(
        self, result: PredictiveResult, tmp_path: Path
    ) -> None:
        assert result.path is not None
        written = load_record(result.path)
        data = figure_data(written)
        assert set(data) == set(FIGURES)
        first = draw_figures(written, tmp_path / "first")
        second = draw_figures(written, tmp_path / "second")
        for name, path in first.items():
            text = path.read_text(encoding="utf-8")
            assert f"data sha256 {data_sha256(data[name])}" in text
            assert path.read_bytes() == second[name].read_bytes()

    def test_figure_data_is_copied_verbatim_from_the_record(
        self, result: PredictiveResult
    ) -> None:
        # A digest over computed values, such as an exponential, could differ
        # in the last bit between platforms, so every plotted number must be a
        # number in the record.
        assert result.path is not None
        written = load_record(result.path)
        recorded = set(numbers(written["results"])) | set(
            numbers(written["configuration"])
        )
        for name, data in figure_data(written).items():
            assert set(numbers(data)) <= recorded, name
