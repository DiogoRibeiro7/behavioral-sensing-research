"""Run the declared Paper 1 exploratory mechanism analyses.

This implements
``papers/when-confidence-is-not-information/EXPLORATORY_MECHANISM_PLAN.md`` on the
22-home CASAS development panel. Every quantity in section 5 of that plan is
computed and written whatever it shows. Nothing here selects an abstention score,
threshold or policy, and nothing re-tests the frozen hypotheses H1-H3.

The run is gated twice. No mechanism quantity is computed until every home has
reproduced its frozen per-home AUCs from ``artifacts/paper1/uncertainty_panel.json``
(plan section 4). M2 is then computed for a home only if the predicted belief it
rebuilds reproduces the recorded information gain (plan section 5, M2).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from sensor_modeling.datasets import (
    read_casas_hh,
    truth_series,
    uncertainty_diagnostics,
)
from sensor_modeling.datasets.evaluate import _correctness_auc, _evidence_strength
from sensor_modeling.fusion.filter import _kl_divergence
from sensor_modeling.online import BehaviouralSensingPipeline, PipelineConfig
from sensor_modeling.online.pipeline import PipelineStep, scoring_steps
from sensor_modeling.states import BehaviouralState, StateOntology
from sensor_modeling.utils import text_file_sha256

PLAN_PATH = Path(
    "papers/when-confidence-is-not-information/EXPLORATORY_MECHANISM_PLAN.md"
)
FROZEN_PANEL_PATH = Path("artifacts/paper1/uncertainty_panel.json")
PANEL_SCRIPT = Path(__file__).with_name("analyse_uncertainty_panel.py")

EQUIVALENCE_TOLERANCE = 1e-5
RECONSTRUCTION_TOLERANCE = 1e-6
NEAR_CHANGE_STEPS = 3

#: The five frozen diagnostics, keyed as in the frozen artifact, each mapped to
#: whether a lower value is the favourable direction.
FROZEN_DIAGNOSTICS: dict[str, bool] = {
    "confidence": False,
    "margin": False,
    "normalised_entropy": True,
    "evidence_strength": False,
    "information_gain": False,
}

M1_QUANTITIES = (
    "supporting_magnitude_auc",
    "contradicting_magnitude_auc",
    "signed_support_auc",
)
M2_GROUPS = ("top_quartile_incorrect", "all_incorrect", "top_quartile_correct")


def _load_panel_script() -> ModuleType:
    """Load the frozen panel runner so its source verification is reused as is."""
    spec = importlib.util.spec_from_file_location(
        "analyse_uncertainty_panel", PANEL_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {PANEL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


panel = _load_panel_script()


@dataclass(frozen=True)
class Trace:
    """The per-step values one home's mechanism analyses need, in step order."""

    ontology: StateOntology
    at: list[datetime]
    labels: list[BehaviouralState | None]
    scored: np.ndarray
    correct: np.ndarray
    beliefs: np.ndarray
    diagnostics: dict[str, np.ndarray]
    recorded_information_gain: list[float | None]
    supports: list[tuple[float, ...]]
    quiet: np.ndarray


def run_recording(
    recording: Any, *, step: timedelta
) -> tuple[list[PipelineStep], list[BehaviouralState | None]]:
    """Run the pipeline exactly as ``evaluate_recording`` does, keeping the steps."""
    tz = recording.observations[0].timestamp.tzinfo
    pipeline = BehaviouralSensingPipeline(
        recording.registry, config=PipelineConfig(tz=tz, step=step)
    )
    steps = pipeline.run(recording.observations)
    steps.extend(pipeline.close(recording.observations[-1].timestamp))
    steps = scoring_steps(steps)
    return steps, truth_series(recording.activities, [s.at for s in steps])


def extract_trace(
    steps: Sequence[PipelineStep], truth: Sequence[BehaviouralState | None]
) -> Trace:
    """Keep what the analyses need from each step, so the steps can be freed.

    Correctness is the frozen panel's: the reported state, which is ``UNKNOWN``
    when the estimate abstains, must equal the label.
    """
    if len(steps) != len(truth):
        raise ValueError("steps and truth must have the same length")
    if not steps:
        raise ValueError("a trace needs at least one step")

    gains = [step.state.information_gain for step in steps]
    return Trace(
        ontology=steps[0].state.ontology,
        at=[step.at for step in steps],
        labels=list(truth),
        scored=np.array([label is not None for label in truth], dtype=bool),
        correct=np.array(
            [
                label is not None and step.state.state == label
                for step, label in zip(steps, truth)
            ],
            dtype=bool,
        ),
        beliefs=np.array([step.state.belief for step in steps], dtype=float),
        diagnostics={
            "confidence": np.array([s.state.confidence for s in steps]),
            "margin": np.array([s.state.margin for s in steps]),
            "normalised_entropy": np.array([s.state.normalised_entropy for s in steps]),
            "evidence_strength": np.array([_evidence_strength(s) for s in steps]),
            "information_gain": np.array(
                [np.nan if gain is None else gain for gain in gains], dtype=float
            ),
        },
        recorded_information_gain=gains,
        supports=[
            tuple(c.support for c in step.state.evidence if c.informative)
            for step in steps
        ],
        quiet=np.array(
            [all(c.observations == 0 for c in step.state.evidence) for step in steps],
            dtype=bool,
        ),
    )


def _share(part: int | np.integer, whole: int | np.integer) -> float | None:
    """Return ``part / whole``, or ``None`` when there is nothing to divide."""
    return float(part) / float(whole) if whole else None


def _auc(
    values: np.ndarray,
    trace: Trace,
    mask: np.ndarray,
    *,
    lower_is_better: bool = False,
) -> float | None:
    """Within-home correctness AUC over the scored steps selected by *mask*."""
    keep = mask & trace.scored & ~np.isnan(values)
    return _correctness_auc(
        values[keep & trace.correct].tolist(),
        values[keep & ~trace.correct].tolist(),
        lower_is_better=lower_is_better,
    )


def frozen_diagnostic_aucs(
    trace: Trace, mask: np.ndarray | None = None
) -> dict[str, float | None]:
    """AUCs of the five frozen diagnostics, keyed as in the frozen artifact."""
    selected = trace.scored if mask is None else mask
    return {
        f"{name}_correctness_auc": _auc(
            trace.diagnostics[name], trace, selected, lower_is_better=lower
        )
        for name, lower in FROZEN_DIAGNOSTICS.items()
    }


def equivalence_differences(
    recomputed: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, float]:
    """Absolute difference from *reference* for each frozen diagnostic's AUC.

    A value missing on one side only is an infinite difference, so it can never
    pass a gate.
    """
    differences: dict[str, float] = {}
    for name in FROZEN_DIAGNOSTICS:
        key = f"{name}_correctness_auc"
        mine, theirs = recomputed.get(key), reference.get(key)
        if mine is None or theirs is None:
            differences[key] = 0.0 if mine is None and theirs is None else math.inf
        else:
            differences[key] = abs(float(mine) - float(theirs))
    return differences


def signed_support(trace: Trace) -> dict[str, float | None]:
    """M1: AUCs of supporting magnitude, contradicting magnitude and signed support.

    Each is a mean over the sensors whose likelihood was applied, and 0 when
    there were none, so supporting plus contradicting magnitude is exactly the
    frozen evidence strength.
    """

    def mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    supporting = np.array([mean([max(s, 0.0) for s in row]) for row in trace.supports])
    contradicting = np.array(
        [mean([max(-s, 0.0) for s in row]) for row in trace.supports]
    )
    signed = np.array([mean(row) for row in trace.supports])
    return {
        "supporting_magnitude_auc": _auc(supporting, trace, trace.scored),
        "contradicting_magnitude_auc": _auc(
            contradicting, trace, trace.scored, lower_is_better=True
        ),
        "signed_support_auc": _auc(signed, trace, trace.scored),
    }


def update_direction(
    trace: Trace, *, tolerance: float = RECONSTRUCTION_TOLERANCE
) -> dict[str, Any]:
    """M2: whether updates move posterior mass toward the labelled state.

    The predicted belief is rebuilt from the preceding step's posterior and the
    ontology's transition over the elapsed interval. It is trusted only if it
    reproduces the recorded information gain at every scored step; otherwise no
    share is computed for the home. The first step has no predecessor and is
    excluded throughout, including from the information-gain quartile.
    """
    position = {state: index for index, state in enumerate(trace.ontology.states)}
    deltas: list[float] = []
    gains: list[float] = []
    correct: list[bool] = []
    worst = 0.0
    missing = 0

    for i in range(1, len(trace.at)):
        label = trace.labels[i]
        if label is None:
            continue
        recorded = trace.recorded_information_gain[i]
        if recorded is None:
            missing += 1
            continue
        transition = trace.ontology.transition(
            trace.at[i] - trace.at[i - 1], at=trace.at[i]
        )
        predicted = trace.beliefs[i - 1] @ transition
        error = abs(_kl_divergence(trace.beliefs[i], predicted) - float(recorded))
        worst = max(worst, error)
        target = position[label]
        deltas.append(float(trace.beliefs[i][target] - predicted[target]))
        gains.append(float(recorded))
        correct.append(bool(trace.correct[i]))

    result: dict[str, Any] = {
        "computed": missing == 0 and worst <= tolerance,
        "first_step_scored": bool(trace.scored[0]),
        "missing_information_gain": missing,
        "max_reconstruction_error": worst,
        "information_gain_q75": None,
        **{group: None for group in M2_GROUPS},
    }
    if not result["computed"] or not gains:
        return result

    q75 = float(np.quantile(gains, 0.75))

    def group(select: Callable[[float, bool], bool]) -> dict[str, Any]:
        chosen = [d for d, g, c in zip(deltas, gains, correct) if select(g, c)]
        positive = sum(d > 0.0 for d in chosen)
        return {
            "steps": len(chosen),
            "positive": positive,
            "zero": sum(d == 0.0 for d in chosen),
            "share_positive": _share(positive, len(chosen)),
        }

    result["information_gain_q75"] = q75
    result["top_quartile_incorrect"] = group(lambda g, c: not c and g >= q75)
    result["all_incorrect"] = group(lambda g, c: not c)
    result["top_quartile_correct"] = group(lambda g, c: c and g >= q75)
    return result


def near_change(
    labels: Sequence[BehaviouralState | None], window: int = NEAR_CHANGE_STEPS
) -> np.ndarray:
    """Mark labelled steps whose preceding *window* steps carry a different label.

    Unlabelled preceding steps do not create a change.
    """
    marked = np.zeros(len(labels), dtype=bool)
    for i, label in enumerate(labels):
        if label is None:
            continue
        marked[i] = any(
            labels[j] is not None and labels[j] != label
            for j in range(max(0, i - window), i)
        )
    return marked


def change_proximity(trace: Trace, window: int = NEAR_CHANGE_STEPS) -> dict[str, Any]:
    """M3: how much lies near a labelled change, and the AUCs without it."""
    near = near_change(trace.labels, window) & trace.scored
    incorrect = trace.scored & ~trace.correct
    return {
        "near_change_scored": int(near.sum()),
        "share_scored_near_change": _share(near.sum(), trace.scored.sum()),
        "near_change_incorrect": int((near & incorrect).sum()),
        "share_incorrect_near_change": _share(
            (near & incorrect).sum(), incorrect.sum()
        ),
        "aucs_excluding_near_change": frozen_diagnostic_aucs(
            trace, trace.scored & ~near
        ),
    }


def quiet_active(trace: Trace) -> dict[str, Any]:
    """M4: where errors fall between quiet and active steps, and AUCs within each."""
    quiet = trace.quiet & trace.scored
    active = ~trace.quiet & trace.scored
    incorrect = trace.scored & ~trace.correct
    return {
        "quiet_scored": int(quiet.sum()),
        "share_scored_quiet": _share(quiet.sum(), trace.scored.sum()),
        "accuracy_quiet": _share((quiet & trace.correct).sum(), quiet.sum()),
        "accuracy_active": _share((active & trace.correct).sum(), active.sum()),
        "share_incorrect_quiet": _share((quiet & incorrect).sum(), incorrect.sum()),
        "aucs_quiet": frozen_diagnostic_aucs(trace, quiet),
        "aucs_active": frozen_diagnostic_aucs(trace, active),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Rank from 1, giving tied values the mean of the ranks they span."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for k in range(start, end + 1):
            ranks[order[k]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation with average ranks, or ``None`` if undefined."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        return None
    rx, ry = _average_ranks(x), _average_ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0.0 or syy == 0.0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _panel(values: Sequence[float | None], *, reference: bool = True) -> dict[str, Any]:
    """Median over homes with a value, the count, and optionally those above 0.5."""
    present = [float(v) for v in values if v is not None]
    summary: dict[str, Any] = {
        "median": statistics.median(present) if present else None,
        "homes": len(present),
    }
    if reference:
        summary["homes_above_half"] = sum(v > 0.5 for v in present)
    return summary


def summarise(homes: Sequence[Mapping[str, Any]], frozen: Mapping[str, Any]) -> dict:
    """Panel medians and counts for every quantity the plan reports."""
    frozen_homes = {row["home"]: row["uncertainty"] for row in frozen["homes"]}
    frozen_summary = frozen["summary"]
    diagnostic_keys = [f"{name}_correctness_auc" for name in FROZEN_DIAGNOSTICS]

    m1 = {
        key: _panel([home["m1_signed_support"][key] for home in homes])
        for key in M1_QUANTITIES
    }

    computed = [
        home["m2_update_direction"]
        for home in homes
        if home["m2_update_direction"]["computed"]
    ]
    m2: dict[str, Any] = {
        "homes_computed": len(computed),
        "homes_not_computed": len(homes) - len(computed),
    }
    for group in M2_GROUPS:
        rows = [row[group] for row in computed if row[group] is not None]
        m2[group] = {
            **_panel([row["share_positive"] for row in rows]),
            "zero_change_steps": sum(row["zero"] for row in rows),
        }

    m3: dict[str, Any] = {
        key: _panel(
            [home["m3_change_proximity"][key] for home in homes], reference=False
        )
        for key in ("share_scored_near_change", "share_incorrect_near_change")
    }
    m3["aucs_excluding_near_change"] = {}
    for key in diagnostic_keys:
        excluded = _panel(
            [
                home["m3_change_proximity"]["aucs_excluding_near_change"][key]
                for home in homes
            ]
        )
        frozen_median = frozen_summary[f"median_{key}"]
        excluded["frozen_median"] = frozen_median
        excluded["distance_from_half_change"] = (
            None
            if excluded["median"] is None
            else abs(excluded["median"] - 0.5) - abs(frozen_median - 0.5)
        )
        m3["aucs_excluding_near_change"][key] = excluded

    m4: dict[str, Any] = {
        key: _panel([home["m4_quiet_active"][key] for home in homes], reference=False)
        for key in (
            "share_scored_quiet",
            "accuracy_quiet",
            "accuracy_active",
            "share_incorrect_quiet",
        )
    }
    for stratum in ("aucs_quiet", "aucs_active"):
        m4[stratum] = {
            key: _panel([home["m4_quiet_active"][stratum][key] for home in homes])
            for key in diagnostic_keys
        }
    paired = [
        (home["m4_quiet_active"]["share_incorrect_quiet"], frozen_homes[home["home"]])
        for home in homes
        if home["m4_quiet_active"]["share_incorrect_quiet"] is not None
    ]
    for key in ("confidence_correctness_auc", "information_gain_correctness_auc"):
        m4[f"spearman_share_incorrect_quiet_vs_frozen_{key}"] = {
            "rho": spearman(
                [share for share, _ in paired], [row[key] for _, row in paired]
            ),
            "homes": len(paired),
        }

    return {
        "m1_signed_support": m1,
        "m2_update_direction": m2,
        "m3_change_proximity": m3,
        "m4_quiet_active": m4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timezone", default=panel.DEFAULT_TIMEZONE)
    parser.add_argument("--step-minutes", type=int, default=panel.DEFAULT_STEP_MINUTES)
    args = parser.parse_args()

    if not PLAN_PATH.is_file():
        raise SystemExit(f"declared plan not found at {PLAN_PATH}")
    manifest = json.loads(panel.MANIFEST_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_PANEL_PATH.read_text(encoding="utf-8"))
    if (args.timezone, args.step_minutes) != (
        frozen["timezone"],
        frozen["step_minutes"],
    ):
        raise SystemExit("timezone and step must match the frozen panel")

    sources = panel.development_sources(manifest)
    frozen_homes = {row["home"]: row for row in frozen["homes"]}
    if set(sources) != set(frozen_homes):
        raise SystemExit("development panel differs from the frozen panel's homes")

    zone = ZoneInfo(args.timezone)
    step = timedelta(minutes=args.step_minutes)

    # Plan section 4: every home reproduces its frozen AUCs before any mechanism
    # quantity is computed.
    traces: dict[str, Trace] = {}
    gate: dict[str, float] = {}
    digests: dict[str, str] = {}
    for home, source in sources.items():
        path = panel.resolve_recording(args.archive_root, str(source["filename"]))
        digests[home] = panel.verify_recording(path, source)
        recording = read_casas_hh(path, timezone=zone)
        steps, truth = run_recording(recording, step=step)
        trace = extract_trace(steps, truth)

        from_trace = frozen_diagnostic_aucs(trace)
        library = uncertainty_diagnostics(truth, steps).to_dict()
        if max(equivalence_differences(from_trace, library).values()) != 0.0:
            raise SystemExit(
                f"{home}: trace AUCs differ from uncertainty_diagnostics; "
                "no mechanism output written"
            )
        differences = equivalence_differences(
            from_trace, frozen_homes[home]["uncertainty"]
        )
        worst = max(differences.values())
        if worst > EQUIVALENCE_TOLERANCE:
            raise SystemExit(
                f"{home}: equivalence gate failed {differences}; "
                "no mechanism output written"
            )
        gate[home] = worst
        traces[home] = trace
        print(f"{home:8} source=verified  equivalence=passed", flush=True)

    rows: list[dict[str, Any]] = []
    for home, trace in traces.items():
        rows.append(
            {
                "home": home,
                "source_bytes": int(sources[home]["bytes"]),
                "source_sha256": digests[home],
                "scored": int(trace.scored.sum()),
                "correct": int(trace.correct.sum()),
                "equivalence_max_difference": gate[home],
                "m1_signed_support": signed_support(trace),
                "m2_update_direction": update_direction(trace),
                "m3_change_proximity": change_proximity(trace),
                "m4_quiet_active": quiet_active(trace),
            }
        )
        print(f"{home:8} mechanisms computed", flush=True)

    payload = {
        "schema_version": 1,
        "analysis": "paper1-exploratory-mechanisms",
        "plan": PLAN_PATH.as_posix(),
        "plan_sha256": text_file_sha256(PLAN_PATH),
        "frozen_panel": FROZEN_PANEL_PATH.as_posix(),
        "frozen_panel_sha256": text_file_sha256(FROZEN_PANEL_PATH),
        "cohort": frozen["cohort"],
        "inference": frozen["inference"],
        "timezone": args.timezone,
        "step_minutes": args.step_minutes,
        "near_change_steps": NEAR_CHANGE_STEPS,
        "equivalence_gate": {
            "tolerance": EQUIVALENCE_TOLERANCE,
            "homes_passed": len(gate),
            "max_difference": max(gate.values()),
        },
        "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
        "summary": summarise(rows, frozen),
        "homes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
