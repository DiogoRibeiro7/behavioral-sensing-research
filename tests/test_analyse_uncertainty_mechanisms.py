"""Tests for the Paper 1 exploratory mechanism runner.

Everything here runs on synthetic inputs. The declared plan allows the runner to
touch CASAS recordings only through its workflow, so no test reads a real archive.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from sensor_modeling.datasets import (
    evaluate_recording,
    read_casas,
    uncertainty_diagnostics,
)
from sensor_modeling.datasets.evaluate import _evidence_strength
from sensor_modeling.states import BehaviouralState, StateOntology

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyse_uncertainty_mechanisms.py"
WORKFLOW = ROOT / ".github" / "workflows" / "paper1-uncertainty-mechanisms.yml"

S = BehaviouralState
UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)
STEP = timedelta(minutes=10)
ROOMS = {"M004": "bedroom", "M001": "kitchen", "M002": "living", "M003": "bathroom"}


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyse_uncertainty_mechanisms", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through sys.modules while the class body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mechanisms = _load_script()


def _synthetic_day() -> list[str]:
    """A CASAS-format day whose event density depends on the activity.

    It is synthesised here and exercises the pipeline code path the runner
    reconstructs; it says nothing about how any real home behaves.
    """
    rng = random.Random(7)
    start = datetime(2010, 11, 4, 0, 0)
    lines: list[str] = []

    def stamp(moment: datetime) -> str:
        return moment.strftime("%Y-%m-%d %H:%M:%S.%f")

    def activity(sensor: str, label: str, begin: float, end: float, lo: int, hi: int):
        first = start + timedelta(hours=begin)
        last = start + timedelta(hours=end)
        lines.append(f"{stamp(first)}\t{sensor}\tON\t{label}\tbegin")
        moment = first
        while (moment := moment + timedelta(minutes=rng.randint(lo, hi))) < last:
            lines.append(f"{stamp(moment)}\t{sensor}\tON")
        lines.append(f"{stamp(last)}\t{sensor}\tON\t{label}\tend")

    activity("M004", "Sleeping", 0.0, 7.0, 40, 110)
    activity("M001", "Meal_Preparation", 7.1, 7.75, 1, 3)
    activity("M002", "Relax", 8.0, 11.0, 8, 20)
    activity("M001", "Meal_Preparation", 12.0, 12.6, 1, 3)
    activity("M002", "Relax", 13.0, 16.0, 8, 20)
    activity("M004", "Sleeping", 22.0, 24.0, 40, 110)
    lines.sort()
    return lines


@pytest.fixture(scope="module")
def synthetic():
    recording = read_casas(_synthetic_day(), timezone=UTC, rooms=ROOMS)
    steps, truth = mechanisms.run_recording(recording, step=STEP)
    return recording, steps, truth, mechanisms.extract_trace(steps, truth)


def test_the_synthetic_day_has_both_correct_and_incorrect_steps(synthetic) -> None:
    _, _, _, trace = synthetic

    assert (trace.scored & trace.correct).any()
    assert (trace.scored & ~trace.correct).any()


def test_run_recording_matches_evaluate_recording(synthetic) -> None:
    recording, steps, truth, _ = synthetic

    expected = evaluate_recording(recording, step=STEP).uncertainty.to_dict()

    assert uncertainty_diagnostics(truth, steps).to_dict() == expected


def test_trace_aucs_equal_the_library_diagnostics_exactly(synthetic) -> None:
    _, steps, truth, trace = synthetic

    differences = mechanisms.equivalence_differences(
        mechanisms.frozen_diagnostic_aucs(trace),
        uncertainty_diagnostics(truth, steps).to_dict(),
    )

    assert set(differences.values()) == {0.0}


def test_equivalence_differences_flag_mismatches_and_one_sided_gaps() -> None:
    keys = [f"{name}_correctness_auc" for name in mechanisms.FROZEN_DIAGNOSTICS]
    reference = {key: 0.5 for key in keys}
    recomputed = dict(reference)
    recomputed["confidence_correctness_auc"] = 0.5001
    recomputed["margin_correctness_auc"] = None

    differences = mechanisms.equivalence_differences(recomputed, reference)

    assert differences["confidence_correctness_auc"] > mechanisms.EQUIVALENCE_TOLERANCE
    assert differences["margin_correctness_auc"] == float("inf")
    assert differences["evidence_strength_correctness_auc"] == 0.0


def test_supporting_and_contradicting_magnitude_sum_to_evidence_strength(
    synthetic,
) -> None:
    _, steps, _, trace = synthetic

    for step, row in zip(steps, trace.supports):
        supporting = sum(max(s, 0.0) for s in row) / len(row) if row else 0.0
        contradicting = sum(max(-s, 0.0) for s in row) / len(row) if row else 0.0
        assert supporting + contradicting == pytest.approx(_evidence_strength(step))


def _hand_trace(supports: list[tuple[float, ...]], correct: list[bool]):
    n = len(supports)
    ontology = StateOntology()
    return mechanisms.Trace(
        ontology=ontology,
        at=[T0 + i * STEP for i in range(n)],
        labels=[S.SLEEPING] * n,
        scored=np.ones(n, dtype=bool),
        correct=np.array(correct, dtype=bool),
        beliefs=np.full((n, ontology.size), 1.0 / ontology.size),
        diagnostics={},
        recorded_information_gain=[None] * n,
        supports=supports,
        quiet=np.zeros(n, dtype=bool),
    )


def test_signed_support_scores_each_quantity_in_its_declared_direction() -> None:
    trace = _hand_trace(
        [(2.0, 1.0), (1.0, -0.5), (-3.0, 0.5), (0.5, -2.0)],
        [True, True, False, False],
    )

    result = mechanisms.signed_support(trace)

    assert result["supporting_magnitude_auc"] == pytest.approx(1.0)
    assert result["contradicting_magnitude_auc"] == pytest.approx(1.0)
    assert result["signed_support_auc"] == pytest.approx(1.0)


def test_rebuilt_predicted_belief_reproduces_recorded_information_gain(
    synthetic,
) -> None:
    _, _, _, trace = synthetic

    result = mechanisms.update_direction(trace)

    assert result["computed"] is True
    assert result["max_reconstruction_error"] < 1e-9
    scored_after_first = trace.scored.copy()
    scored_after_first[0] = False
    incorrect = int((scored_after_first & ~trace.correct).sum())
    assert result["all_incorrect"]["steps"] == incorrect
    assert result["top_quartile_incorrect"]["steps"] <= incorrect


def test_update_direction_refuses_a_trace_it_cannot_reconstruct(synthetic) -> None:
    _, _, _, trace = synthetic
    shifted = [
        None if gain is None else gain + 1e-3
        for gain in trace.recorded_information_gain
    ]

    result = mechanisms.update_direction(
        dataclasses.replace(trace, recorded_information_gain=shifted)
    )

    assert result["computed"] is False
    assert all(result[group] is None for group in mechanisms.M2_GROUPS)


def test_near_change_ignores_unlabelled_and_distant_steps() -> None:
    labels = [
        S.SLEEPING,
        S.SLEEPING,
        None,
        S.KITCHEN_ACTIVITY,
        None,
        None,
        None,
        S.KITCHEN_ACTIVITY,
        S.SLEEPING,
    ]

    marked = mechanisms.near_change(labels, window=3)

    assert marked.tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
    ]


def test_spearman_uses_average_ranks_for_ties() -> None:
    assert mechanisms.spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(
        4.5 / (22.5**0.5)
    )
    assert mechanisms.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert mechanisms.spearman([1, 1, 1], [1, 2, 3]) is None


def test_summary_is_complete_and_strict_json(synthetic) -> None:
    _, steps, truth, trace = synthetic
    library = uncertainty_diagnostics(truth, steps).to_dict()
    row = {
        "m1_signed_support": mechanisms.signed_support(trace),
        "m2_update_direction": mechanisms.update_direction(trace),
        "m3_change_proximity": mechanisms.change_proximity(trace),
        "m4_quiet_active": mechanisms.quiet_active(trace),
    }
    rows = [{"home": "a", **row}, {"home": "b", **row}]
    frozen = {
        "homes": [
            {"home": "a", "uncertainty": library},
            {"home": "b", "uncertainty": library},
        ],
        "summary": {
            f"median_{name}_correctness_auc": library[f"{name}_correctness_auc"]
            for name in mechanisms.FROZEN_DIAGNOSTICS
        },
    }

    summary = mechanisms.summarise(rows, frozen)
    json.dumps({"summary": summary, "homes": rows}, allow_nan=False)

    assert summary["m2_update_direction"]["homes_computed"] == 2
    assert set(summary["m4_quiet_active"]["aucs_quiet"]) == set(library) & {
        f"{name}_correctness_auc" for name in mechanisms.FROZEN_DIAGNOSTICS
    }


def test_workflow_runs_only_on_manual_dispatch() -> None:
    """The plan freezes the first successful run, so no event may start one early."""
    text = WORKFLOW.read_text(encoding="utf-8")
    header = text.split("\npermissions:")[0]
    triggers = "\n".join(
        line for line in header.splitlines() if not line.lstrip().startswith("#")
    )

    assert "workflow_dispatch" in triggers
    assert "pull_request" not in triggers
    assert "push" not in triggers
    assert "schedule" not in triggers
