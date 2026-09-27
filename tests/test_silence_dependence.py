"""Tests for the pre-specified correlated-silence diagnostic.

They guard what makes its result interpretable:
- on synthetic households, it reads conditionally independent streams as
  independent and strongly correlated silence as dependent;
- its concentration slope is zero in expectation for a correctly specified
  model, and positive when correlated silence is counted as independent;
- the recursion it decomposes is the production filter's;
- the protocol that ran is the one frozen before any household was examined;
- the record, its summary and its figures are reproducible from the artifact.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import ActivityInterval, CasasRecording
from sensor_modeling.datasets.information_sets import (
    _route_sensors,
    build_feature_table,
    evidence_column,
)
from sensor_modeling.datasets.matched_evaluation import _regular_moments
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.restricted_filter import channel_likelihoods
from sensor_modeling.datasets.silence_dependence import (
    ESTIMANDS,
    MINIMAL_EFFECTS,
    SilenceProtocol,
    SilenceResult,
    against_independence,
    check_frozen_protocol,
    concentration,
    conclusion,
    declared_protocol,
    decompose_case,
    filter_trajectory,
    quiet_positions,
    read_verdict,
    representative_runs,
    run_silence_diagnostic,
    silence_inflation,
    state_dependence,
)
from sensor_modeling.datasets.silence_figures import (
    FIGURES,
    data_sha256,
    draw_figures,
    figure_data,
)
from sensor_modeling.datasets.silence_summary import render_summary
from sensor_modeling.evaluation import load_record
from sensor_modeling.fusion import MultimodalBayesFilter
from sensor_modeling.fusion.defaults import default_emissions
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
PROTOCOL = ROOT / "artifacts" / "phase3" / "silence_protocol.json"
PUBLISHED = ROOT / "artifacts" / "phase3" / "phase3-correlated-silence.json"
DOC = ROOT / "docs" / "PHASE3_CORRELATED_SILENCE.md"
FIGURE_DIR = ROOT / "docs" / "figures"

ONTOLOGY = StateOntology()
STATES = ONTOLOGY.size
STEP = timedelta(minutes=5)
TRANSITION = ONTOLOGY.transition(STEP)
PRIOR = ONTOLOGY.stationary()
CHANNELS = 5
QUIET = [
    ONTOLOGY.states.index(s) for s in SilenceProtocol(("a", "b"), "0" * 64).quiet_states
]
#: Expected activations per window in each state, before a per-channel factor.
LEVELS = np.array([0.1, 1.2, 0.5, 0.15, 0.6, 1.0, 1.0])
ANCHOR = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ----------------------------------------------------------------------------
# Synthetic households
# ----------------------------------------------------------------------------
def rates(seed: int) -> np.ndarray:
    """``(channels, states)`` expected activations per window."""
    factors = np.random.default_rng(seed).uniform(0.5, 1.5, size=(CHANNELS, STATES))
    return LEVELS[None, :] * factors


def synthetic(
    seed: int, *, correlated: bool, windows: int = 6000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """States from the ontology's chain, and counts given the states.

    Independent: each channel is Poisson with its own mean. Correlated: in
    half the windows every channel is silent together, whatever the state;
    otherwise each is Poisson with twice its mean, so the mean is unchanged.
    """
    rng = np.random.default_rng(seed)
    mean = rates(seed)
    states = np.empty(windows, dtype=int)
    state = rng.choice(STATES, p=PRIOR)
    for row in range(windows):
        state = rng.choice(STATES, p=TRANSITION[state])
        states[row] = state
    expected = mean[:, states].T
    if correlated:
        counts = rng.poisson(2.0 * expected)
        counts[rng.random(windows) < 0.5] = 0
    else:
        counts = rng.poisson(expected)
    return states, counts.astype(float), mean


def terms(mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Poisson terms for *mean*, centred like the filter's."""
    per_activation = np.log(mean) - np.log(mean).mean(axis=1, keepdims=True)
    per_window = -mean + mean.mean(axis=1, keepdims=True)
    return per_activation, per_window


def household_values(seed: int, *, correlated: bool) -> dict[str, float | None]:
    states, counts, mean = synthetic(seed, correlated=correlated)
    names = [f"c{c}" for c in range(CHANNELS)]
    by_state = {
        str(s): state_dependence(
            counts[states == s], names, mean[:, s], smoothing=0.5, min_expected=5.0
        )
        for s in range(STATES)
        if np.sum(states == s) >= 50
    }
    per_activation, per_window = terms(mean)
    inflation = silence_inflation(
        by_state, {s: float(per_window[:, int(s)].sum()) for s in by_state}
    )
    trajectory = filter_trajectory(
        counts, per_activation, per_window, TRANSITION, PRIOR
    )
    concentrated = concentration(
        trajectory.posterior,
        np.arange(states.size),
        states,
        (counts == 0).sum(axis=1),
        CHANNELS,
        accumulation_steps=36,
        steps_per_hour=12.0,
    )
    quiet = [str(s) for s in QUIET if str(s) in by_state]

    def quiet_mean(key: str) -> float:
        return float(np.mean([by_state[s][key] for s in quiet]))

    return {
        "log_inflation": inflation["log_inflation"],
        "log_joint_silence_quiet": quiet_mean("log_joint_silence"),
        "log_odds_silence_quiet": quiet_mean("log_odds_silence"),
        "log_dispersion_quiet": quiet_mean("log_dispersion"),
        "gap_slope": concentrated["gap_slope"],
    }


PANEL_PROTOCOL = SilenceProtocol(
    tuple(f"h{i}" for i in range(8)), "0" * 64, resamples=500
)


@pytest.fixture(scope="module")
def panels() -> dict[str, dict[str, dict[str, float | None]]]:
    """Per quantity, per household, for an independent and a correlated panel."""
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for kind in ("independent", "correlated"):
        homes = {
            f"h{i}": household_values(100 + i, correlated=kind == "correlated")
            for i in range(8)
        }
        out[kind] = {
            quantity: {home: values[quantity] for home, values in homes.items()}
            for quantity in next(iter(homes.values()))
        }
    return out


def reading(values: dict[str, float | None], scale: str) -> str:
    summary = against_independence(values, PANEL_PROTOCOL)
    return read_verdict(summary, MINIMAL_EFFECTS[scale])


class TestTheDiagnosticDistinguishesTheCases:
    @pytest.mark.parametrize(
        ("quantity", "scale"),
        [
            ("log_inflation", "log_ratio"),
            ("log_joint_silence_quiet", "log_ratio"),
            ("log_odds_silence_quiet", "log_ratio"),
            ("log_dispersion_quiet", "log_ratio"),
            ("gap_slope", "slope"),
        ],
    )
    def test_independent_streams_read_as_independent(
        self, panels: dict[str, Any], quantity: str, scale: str
    ) -> None:
        assert reading(panels["independent"][quantity], scale) == "negligible"

    @pytest.mark.parametrize(
        ("quantity", "scale"),
        [
            ("log_inflation", "log_ratio"),
            ("log_joint_silence_quiet", "log_ratio"),
            ("log_odds_silence_quiet", "log_ratio"),
            ("log_dispersion_quiet", "log_ratio"),
            ("gap_slope", "slope"),
        ],
    )
    def test_correlated_silence_reads_as_dependent(
        self, panels: dict[str, Any], quantity: str, scale: str
    ) -> None:
        assert reading(panels["correlated"][quantity], scale) == "positive"

    def test_the_declared_conclusion_follows(self, panels: dict[str, Any]) -> None:
        verdicts = {
            kind: {
                "D1": reading(panels[kind]["log_inflation"], "log_ratio"),
                "C1": reading(panels[kind]["gap_slope"], "slope"),
            }
            for kind in panels
        }
        assert conclusion(verdicts["independent"]) == "weakened"
        assert conclusion(verdicts["correlated"]) == "supported"


class TestStatistics:
    def test_independence_gives_a_joint_ratio_near_zero_and_duplication_does_not(
        self,
    ) -> None:
        rng = np.random.default_rng(0)
        one = rng.poisson(0.7, size=20_000).astype(float)
        independent = np.column_stack([one, rng.poisson(0.7, size=20_000)])
        duplicated = np.column_stack([one, one])
        expected = np.array([0.7, 0.7])
        low = state_dependence(
            independent, ["a", "b"], expected, smoothing=0.5, min_expected=5.0
        )
        high = state_dependence(
            duplicated, ["a", "b"], expected, smoothing=0.5, min_expected=5.0
        )
        assert abs(low["log_joint_silence"]) < 0.03
        # Two copies of one stream: joint silence is its silence, not its square.
        assert high["log_joint_silence"] == pytest.approx(0.7, abs=0.03)
        assert high["pairs"]["a|b"]["count_correlation"] == pytest.approx(1.0)
        assert high["log_dispersion"] > 0.6

    def test_rare_joint_silence_is_not_estimated(self) -> None:
        counts = np.random.default_rng(1).poisson(5.0, size=(200, 4)).astype(float)
        stats = state_dependence(
            counts, list("abcd"), np.full(4, 5.0), smoothing=0.5, min_expected=5.0
        )
        assert stats["joint_estimable"] is False
        assert stats["log_joint_silence"] is None
        assert stats["log_odds_silence"] is None

    def test_inflation_uses_only_estimable_states(self) -> None:
        estimable = {
            "joint_estimable": True,
            "joint_silence": 0.5,
            "silence": {"a": 0.5},
        }
        other = {"joint_estimable": True, "joint_silence": 0.25, "silence": {"a": 0.25}}
        floored = {
            "joint_estimable": False,
            "joint_silence": 0.1,
            "silence": {"a": 1e-9},
        }
        declared = {"x": 0.0, "y": -1.0, "z": -50.0}
        both = silence_inflation({"x": estimable, "y": other}, declared)
        with_floor = silence_inflation(
            {"x": estimable, "y": other, "z": floored}, declared
        )
        assert both == with_floor
        assert both["log_inflation"] == pytest.approx(0.0)

    def test_quiet_positions_count_consecutive_fully_silent_windows(self) -> None:
        assert quiet_positions(np.array([3, 3, 2, 3, 3, 3, 0]), 3).tolist() == [
            1,
            2,
            0,
            1,
            2,
            3,
            0,
        ]

    def test_the_representative_run_is_the_lower_median(self) -> None:
        silent = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
        truth = [0] * 14
        # Lengths 3, 2, 4 and 2: the lower median is 2, and the earliest such run
        # starts at 4.
        assert representative_runs(silent, truth, 0, min_steps=2) == (4, 2)
        assert representative_runs(silent, truth, 0, min_steps=3) == (0, 3)
        assert representative_runs(silent, truth, 0, min_steps=5) is None
        relabelled = truth[:]
        relabelled[1] = 2
        assert representative_runs(silent, relabelled, 0, min_steps=3) == (7, 4)

    @pytest.mark.parametrize(
        ("d1", "c1", "expected"),
        [
            ("positive", "positive", "supported"),
            ("positive", "negligible", "weakened"),
            ("negative", "positive", "weakened"),
            ("uncertain", "positive", "inconclusive"),
            ("positive", "uncertain", "inconclusive"),
        ],
    )
    def test_the_conclusion_rule(self, d1: str, c1: str, expected: str) -> None:
        assert conclusion({"D1": d1, "C1": c1}) == expected


class TestTheRecursionIsTheFilters:
    def test_it_equals_the_production_filter_fed_the_same_windows(self) -> None:
        recording = simulated(3)
        protocol = small_protocol()
        resolution = protocol.resolution
        moments = _regular_moments(recording, STEP)[:400]
        table = build_feature_table(
            recording, protocol.information_set, moments, household="sim3"
        )
        likelihoods, _ = channel_likelihoods(recording.registry, resolution, ONTOLOGY)
        channels = sorted(likelihoods)
        counts = np.column_stack(
            [table.column(evidence_column(c.name, 0)) for c in channels]
        )
        trajectory = filter_trajectory(
            counts,
            np.stack([likelihoods[c].per_activation for c in channels]),
            np.stack([likelihoods[c].per_window for c in channels]),
            TRANSITION,
            PRIOR,
        )

        routed, _ = _route_sensors(recording.registry, resolution.channels)
        emissions = [
            m
            for m in default_emissions(recording.registry, ONTOLOGY)
            if m.sensor_id in routed
        ]
        production = MultimodalBayesFilter(ONTOLOGY, emissions, recording.registry)
        production.update(moments[0] - STEP)
        observations = sorted(
            (o for o in recording.observations if o.sensor_id in routed),
            key=lambda o: o.timestamp,
        )
        position = 0
        for row, moment in enumerate(moments):
            window = []
            while (
                position < len(observations)
                and observations[position].timestamp <= moment
            ):
                if observations[position].timestamp > moment - STEP:
                    window.append(observations[position])
                position += 1
            before = production.belief
            estimate = production.update(moment, window)
            np.testing.assert_allclose(
                trajectory.predicted[row], before @ TRANSITION, atol=1e-12
            )
            np.testing.assert_allclose(
                trajectory.posterior[row], estimate.belief, atol=1e-9
            )

    def test_a_decomposed_window_adds_up_to_the_posterior(self) -> None:
        states, counts, mean = synthetic(7, correlated=False, windows=300)
        per_activation, per_window = terms(mean)
        trajectory = filter_trajectory(
            counts, per_activation, per_window, TRANSITION, PRIOR
        )
        silent = (counts == 0).all(axis=1)
        row = int(np.argmax(silent))
        assert silent[row]
        moments = [ANCHOR + STEP * k for k in range(300)]
        steps = decompose_case(
            trajectory,
            row,
            1,
            [f"c{c}" for c in range(CHANNELS)],
            tuple(ONTOLOGY.states),
            moments,
            counts,
        )
        step = steps[0]
        assert step["silent_channels"] == CHANNELS
        assert step["channels"][-1]["cumulative_confidence"] == pytest.approx(
            step["confidence"]
        )
        expected = trajectory.predicted[row] * np.exp(
            trajectory.channel_loglik[row].sum(axis=0)
        )
        np.testing.assert_allclose(step["posterior"], expected / expected.sum())


# ----------------------------------------------------------------------------
# The protocol
# ----------------------------------------------------------------------------
class TestFrozenProtocol:
    def test_the_committed_protocol_is_the_declared_one(self) -> None:
        splits = load_frozen_splits(SPLITS)
        declared = declared_protocol(splits)
        check_frozen_protocol(declared, PROTOCOL)
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert frozen["protocol_sha256"] == declared.sha256()
        assert frozen["households"]["households"] == sorted(splits.homes)

    def test_it_freezes_everything_the_diagnostic_depends_on(self) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert frozen["streams"]["information_set"]["step_seconds"] == 300.0
        assert [e["key"] for e in frozen["estimands"] if e["role"] == "primary"] == [
            "D1",
            "D2",
            "C1",
        ]
        assert frozen["minimal_effects"]["log_ratio"] == pytest.approx(math.log(1.25))
        assert frozen["minimal_effects"]["slope"] == 0.02
        assert frozen["min_expected"]["value"] == 5.0
        assert frozen["bootstrap"]["unit"] == "household"
        assert frozen["bootstrap"]["resamples"] == 10_000
        assert "unchanged" in frozen["inference"]
        assert frozen["states"]["quiet"] == ["sleeping", "away", "home_inactive"]

    @pytest.mark.parametrize(
        "change", [{"seed": 1}, {"min_windows": 40}, {"min_expected": 10.0}]
    )
    def test_any_change_to_the_protocol_is_refused(
        self, change: dict[str, Any]
    ) -> None:
        changed = replace(declared_protocol(load_frozen_splits(SPLITS)), **change)
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(changed, PROTOCOL)

    def test_a_tampered_protocol_file_is_refused(self, tmp_path: Path) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        frozen["criteria"]["supported"] = "anything"
        tampered = tmp_path / "protocol.json"
        tampered.write_text(json.dumps(frozen), encoding="utf-8")
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(
                declared_protocol(load_frozen_splits(SPLITS)), tampered
            )

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"households": ("a",)}, "two distinct"),
            ({"splits_sha256": "abc"}, "SHA-256"),
            ({"case_steps": 20}, "case_steps"),
            ({"min_expected": 0.0}, "min_expected"),
            ({"minimal_effects": {"slope": 0.02}}, "minimal effect"),
        ],
    )
    def test_invalid_protocols_are_refused(
        self, change: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            small_protocol(**change)


# ----------------------------------------------------------------------------
# End to end, on simulated homes
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


def small_protocol(**changes: Any) -> SilenceProtocol:
    settings: dict[str, Any] = {
        "households": ("sim1", "sim2", "sim3", "sim4"),
        "splits_sha256": "0" * 64,
        "resamples": 200,
    }
    settings.update(changes)
    return SilenceProtocol(**settings)


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {f"sim{s}": simulated(s) for s in range(1, 5)}


@pytest.fixture(scope="module")
def result(
    homes: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[SilenceResult]:
    yield run_silence_diagnostic(
        homes,
        small_protocol(),
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("silence"),
    )


class TestRecord:
    def test_the_record_validates(self, result: SilenceResult) -> None:
        assert result.path is not None
        payload = load_record(result.path)
        assert payload["results"]["result_schema"] == "silence-results/1"
        assert payload["configuration"]["protocol_sha256"] == small_protocol().sha256()

    def test_households_are_the_unit_of_replication(
        self, result: SilenceResult
    ) -> None:
        results = result.record.to_dict()["results"]
        homes = set(results["households"])
        for entry in results["estimands"]:
            summary = entry["summary"]
            assert set(summary.get("values", {})) <= homes
            assert summary["n"] == len(summary.get("values", {}))
            if summary["mean"] and summary["mean"]["interval"]:
                assert summary["n"] >= 2
        assert {e["key"] for e in results["estimands"]} == {e.key for e in ESTIMANDS}

    def test_every_quiet_case_is_decomposed(self, result: SilenceResult) -> None:
        results = result.record.to_dict()["results"]
        for entry in results["households"].values():
            for case in entry["cases"].values():
                assert len(case["steps"]) == small_protocol().case_steps
                for step in case["steps"]:
                    assert step["silent_channels"] == len(entry["channels"])
                    assert len(step["channels"]) == step["silent_channels"]
                    assert math.isclose(sum(step["posterior"]), 1.0)

    def test_a_second_run_gives_the_same_record(
        self, homes: dict[str, CasasRecording], result: SilenceResult
    ) -> None:
        again = run_silence_diagnostic(homes, small_protocol(), data_source="simulator")
        first, second = result.record.to_dict(), again.record.to_dict()
        first.pop("recorded_at")
        second.pop("recorded_at")
        assert first == second

    def test_the_summary_is_generated_from_the_written_record(
        self, result: SilenceResult
    ) -> None:
        assert result.path is not None
        written = load_record(result.path)
        summary = render_summary(written)
        assert summary == render_summary(
            json.loads(json.dumps(result.record.to_dict()))
        )
        for heading in (
            "## Pre-specified conclusion",
            "## Estimands",
            "## Joint silence by state",
            "## Pairwise dependence by state",
            "## Channel pairs in the quiet states",
            "## Households",
            "## Representative quiet periods",
        ):
            assert heading in summary
        payload = result.record.to_dict()
        payload["results"] = {"result_schema": "history-results/1"}
        with pytest.raises(ValueError, match="silence-results"):
            render_summary(payload)

    def test_figures_are_drawn_from_the_record_and_reproducible(
        self, result: SilenceResult, tmp_path: Path
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
