"""End-to-end worked example: from raw sensor traffic to an explained alert.

Runs one reproducible experiment covering everything the platform claims to
handle at once:

.. code-block:: text

    a synthetic household over several months
    seven sensing modalities
    a carer and occasional visitors tripping the same ambient sensors
    a bed sensor that dies for three days
    a wearable left off for five days
    records lost, duplicated, delayed, and stamped by a drifting clock
    a genuine persistent change in the resident's sleep, on a known day
    probabilistic state inference with abstention
    an adaptive, weekday-aware personal baseline
    an explained alert, with its caveats

Everything is seeded, so two runs of the same command produce identical
numbers. The demonstration deliberately reports what the system got wrong as
well as what it got right: the point is a defensible account of the platform's
behaviour, not a favourable one.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ...alerts.alert import AlertKind
from ...baseline.adaptive import ChangeKind
from ...evaluation.metrics import (
    binary_metrics,
    detection_metrics,
    state_metrics,
    transition_timing,
)
from ...online.pipeline import (
    BehaviouralSensingPipeline,
    PipelineConfig,
    PipelineStep,
    collect_alerts,
    collect_changes,
    daily_summaries,
)
from ...simulation.faults import DegradationConfig, degrade, dropout, not_worn
from ...simulation.household import BehaviourShift, HouseholdConfig, simulate
from ...states.ontology import BehaviouralState

logger = logging.getLogger(__name__)

S = BehaviouralState

#: The demonstration's fixed defaults. Changing any of these changes every
#: number the demo prints, so they are stated in one place.
DEFAULT_DAYS = 90
DEFAULT_SEED = 20240304
CHANGE_DAY_INDEX = 60


def build_scenario(
    days: int = DEFAULT_DAYS, seed: int = DEFAULT_SEED
) -> tuple[Any, list, dict[str, Any]]:
    """Build the household, inject faults, and degrade the record.

    Returns the simulation result, the delivered observations, and a summary
    of exactly what was injected, so the demo can report the truth it is
    being scored against.
    """
    change_day = min(CHANGE_DAY_INDEX, max(days - 20, 5))
    household = HouseholdConfig(
        days=days,
        seed=seed,
        shift=BehaviourShift(
            start_day=change_day,
            sleep_delta_hours=1.6,
            night_bathroom_extra=1.4,
        ),
    )
    result = simulate(household)

    bed_fault_start = result.start + timedelta(days=max(days // 3, 2))
    wearable_fault_start = result.start + timedelta(days=max(days // 2, 3))
    faults = (
        dropout("bed_pressure", bed_fault_start, timedelta(days=3)),
        *not_worn(
            ["wearable_motion", "resident_beacon"],
            wearable_fault_start,
            timedelta(days=5),
        ),
    )
    degradation = DegradationConfig(
        missing_rate=0.03,
        duplication_rate=0.02,
        late_rate=0.05,
        late_delay=timedelta(minutes=4),
        clock_drift={"sim-radar": timedelta(seconds=45)},
        faults=faults,
        seed=seed + 1,
    )
    delivered, withheld = degrade(result.observations, degradation)

    injected = {
        "days": days,
        "seed": seed,
        "change_day": (result.config.start + timedelta(days=change_day)).isoformat(),
        "change_description": (
            "sleep shortened by ~1.6 h and night bathroom trips increased"
        ),
        "faults": [fault.to_dict() for fault in faults],
        "records_generated": len(result.observations),
        "records_delivered": len(delivered),
        "records_withheld": len(withheld),
    }
    return result, delivered, injected


def run_pipeline(result: Any, delivered: Sequence, step: timedelta) -> tuple:
    """Run the delivered record through the online pipeline."""
    pipeline = BehaviouralSensingPipeline(
        result.registry, config=PipelineConfig(tz=result.config.tz, step=step)
    )
    steps = pipeline.run(delivered)
    steps.extend(pipeline.close(result.end))
    return pipeline, steps


def _transitions(values: Sequence, times: Sequence) -> list:
    """Return the times at which a sequence of labels changes."""
    return [
        moment
        for previous, current, moment in zip(values, values[1:], times[1:])
        if previous is not current
    ]


def evaluate(result: Any, steps: Sequence[PipelineStep]) -> dict[str, Any]:
    """Score the run against the ground truth the simulator recorded."""
    moments = [step.at for step in steps]
    truth = result.truth.states_at(moments)

    states = state_metrics(truth, [step.state for step in steps])
    visitors = binary_metrics(
        [result.truth.visitor_at(moment) for moment in moments],
        [step.context.visitor_present for step in steps],
    )
    timing = transition_timing(
        _transitions(truth, moments),
        _transitions([step.state.state for step in steps], moments),
        tolerance=timedelta(minutes=45),
    )
    return {"state": states, "visitor": visitors, "timing": timing}


def _fault_recognition(steps: Sequence[PipelineStep]) -> dict[str, dict[str, object]]:
    """Return which sensors the health monitor judged faulty, and for how long.

    ``UNKNOWN`` is excluded: every event sensor starts there before its first
    report, and reporting that as a fault would make a healthy deployment
    look broken.
    """
    faults: dict[str, set[str]] = {}
    duration: dict[str, int] = {}
    for step in steps:
        for sensor_id, report in step.health.sensors.items():
            if report.is_faulty:
                faults.setdefault(sensor_id, set()).add(report.status.value)
                duration[sensor_id] = duration.get(sensor_id, 0) + 1
    return {
        sensor: {"statuses": sorted(statuses), "steps_faulty": duration[sensor]}
        for sensor, statuses in sorted(faults.items())
    }


def _change_report(
    steps: Sequence[PipelineStep], change_day: date, person_days: float
) -> dict[str, Any]:
    """Summarise what the baseline concluded and when it concluded it."""
    changes = collect_changes(steps)
    sleep_changes = [change for change in changes if change.feature == "sleeping_hours"]

    # Score the alerts a carer would actually receive, not the raw verdicts.
    # Deduplication and rate limiting sit between the two, and it is the
    # delivered burden that decides whether a system is usable.
    detected = [
        alert.at.date()
        for alert in collect_alerts(steps)
        if alert.kind is AlertKind.BEHAVIOURAL_CHANGE
        and "sleeping" in str(alert.subject)
    ]
    metrics = detection_metrics(
        detected, [change_day], person_days=person_days, max_delay_days=21.0
    )
    kinds: dict[str, int] = {}
    for change in changes:
        kinds[change.kind.value] = kinds.get(change.kind.value, 0) + 1
    return {
        "verdict_counts": kinds,
        "sleep_changes": len([c for c in sleep_changes if c.is_change]),
        "skipped_days": len(
            [c for c in sleep_changes if c.kind is ChangeKind.INSUFFICIENT_DATA]
        ),
        "detection": metrics,
    }


def _format_alerts(steps: Sequence[PipelineStep], limit: int = 4) -> list[str]:
    """Render the alerts raised, with their evidence and caveats."""
    lines: list[str] = []
    for alert in collect_alerts(steps)[:limit]:
        lines.append(
            f"  [{alert.severity.value:>11}] {alert.kind.value}: {alert.summary}"
        )
        lines.append(
            f"      confidence {alert.confidence:.2f}, score {alert.score:.2f}"
        )
        for caveat in alert.caveats:
            lines.append(f"      caveat: {caveat}")
    return lines


def report(
    result: Any,
    pipeline: BehaviouralSensingPipeline,
    steps: Sequence[PipelineStep],
    injected: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full structured result of the demonstration."""
    scores = evaluate(result, steps)
    summaries = daily_summaries(steps)
    change_day = date.fromisoformat(injected["change_day"])
    person_days = float(injected["days"])

    return {
        "scenario": injected,
        "ingestion": {
            **pipeline.ingestion.to_dict(),
            "arrived_too_late": pipeline.too_late,
        },
        "state_inference": scores["state"].to_dict(),
        "visitor_attribution": scores["visitor"].to_dict(),
        "transition_timing": scores["timing"].to_dict(),
        "sensor_health": _fault_recognition(steps),
        "days_summarised": len(summaries),
        "days_excluded_from_baseline": len([s for s in summaries if not s.is_usable()]),
        "behavioural_change": {
            **{
                key: value
                for key, value in _change_report(steps, change_day, person_days).items()
                if key != "detection"
            },
            "detection": _change_report(steps, change_day, person_days)[
                "detection"
            ].to_dict(),
        },
        "alerts": [alert.to_dict() for alert in collect_alerts(steps)],
    }


def render(payload: dict[str, Any], steps: Sequence[PipelineStep]) -> str:
    """Render the demonstration as readable text."""
    scenario = payload["scenario"]
    state = payload["state_inference"]
    visitor = payload["visitor_attribution"]
    timing = payload["transition_timing"]
    change = payload["behavioural_change"]
    detection = change["detection"]

    lines = [
        "=" * 72,
        "AMBIENT BEHAVIOURAL SENSING -- END-TO-END DEMONSTRATION",
        "=" * 72,
        "",
        "SCENARIO (ground truth, hidden from inference)",
        f"  {scenario['days']} days, seed {scenario['seed']}",
        f"  behavioural change on {scenario['change_day']}:",
        f"    {scenario['change_description']}",
        "  injected sensor faults:",
    ]
    for fault in scenario["faults"]:
        lines.append(
            f"    {fault['sensor_id']:<18} {fault['kind']:<14} "
            f"from {fault['start'][:16]} to {fault['end'][:16]}"
        )
    lines += [
        f"  records generated {scenario['records_generated']}, "
        f"delivered {scenario['records_delivered']}, "
        f"withheld {scenario['records_withheld']}",
        "",
        "INGESTION",
        f"  accepted {payload['ingestion']['accepted']}, "
        f"rejected {len(payload['ingestion']['rejected'])}, "
        f"out of order {payload['ingestion']['out_of_order']}, "
        f"late {payload['ingestion']['late_arrivals']}, "
        f"too late to use {payload['ingestion']['arrived_too_late']}",
        "",
        "STATE INFERENCE (against schedule-generated truth)",
        f"  balanced accuracy   {state['balanced_accuracy']:.3f}",
        f"  macro F1            {state['macro_f1']:.3f}",
        f"  accuracy            {state['accuracy']:.3f}"
        f"   (abstained on {state['abstention_rate']:.1%})",
        f"  log loss            {state['log_loss']:.3f}",
        f"  Brier score         {state['brier']:.3f}",
        f"  calibration error   {state['calibration_error']:.3f}",
        "  per-state recall:",
    ]
    for name, value in sorted(state["per_class_recall"].items()):
        lines.append(f"    {name:<20} {value:.3f}")

    lines += [
        "",
        "OCCUPANCY ATTRIBUTION (was it the resident, or a visitor?)",
        f"  visitor precision {visitor['precision']:.3f}, "
        f"recall {visitor['recall']:.3f}, F1 {visitor['f1']:.3f}",
        f"  calibration error {visitor['calibration_error']:.3f}"
        f"   (visitors present {visitor['positive_rate']:.1%} of the time)",
        "",
        "TRANSITION TIMING",
        f"  matched {timing['matched']} of {timing['true_transitions']} "
        f"true transitions ({timing['matched_fraction']:.1%})",
        f"  median offset {timing['median_error_seconds'] / 60:+.1f} min",
        "",
        "SENSOR HEALTH (independent of any behavioural conclusion)",
    ]
    if payload["sensor_health"]:
        for sensor, detail in payload["sensor_health"].items():
            lines.append(
                f"  {sensor:<18} {', '.join(detail['statuses']):<34}"
                f" over {detail['steps_faulty']} steps"
            )
    else:
        lines.append("  no sensor was ever judged faulty")

    lines += [
        "",
        "ADAPTIVE BASELINE",
        f"  {payload['days_summarised']} days summarised, "
        f"{payload['days_excluded_from_baseline']} excluded as poorly observed",
        f"  verdicts: {change['verdict_counts']}",
        "",
        "BEHAVIOURAL CHANGE DETECTION",
        f"  true change on {scenario['change_day']}",
        f"  behavioural alerts delivered: {detection['detected']} matched of "
        f"{detection['true_changes']} true change(s), "
        f"median delay {detection['median_delay_days']:.0f} days",
        f"  unmatched behavioural alerts {detection['false_positives']} "
        f"({detection['false_positives_per_person_day']:.3f} per person-day)",
        "",
        f"ALERTS RAISED ({len(payload['alerts'])})",
    ]
    lines.extend(_format_alerts(steps) or ["  none"])

    explained = next(
        (
            alert
            for step in steps
            for alert in step.alerts
            if alert.kind is AlertKind.BEHAVIOURAL_CHANGE
        ),
        None,
    )
    if explained is not None:
        evidence = explained.evidence.get("change", {})
        lines += [
            "",
            "WORKED EXPLANATION OF THE FIRST BEHAVIOURAL ALERT",
            f"  {explained.summary}",
            f"  feature       {evidence.get('feature')}",
            f"  observed      {evidence.get('value'):.2f} h",
            f"  reference     {evidence.get('reference', {}).get('centre'):.2f} h "
            f"(robust scale {evidence.get('reference', {}).get('scale'):.2f})",
            f"  deviation     {evidence.get('deviation'):+.2f} robust SD",
            f"  held for      {evidence.get('duration_days')} days",
            f"  weekday-aware {evidence.get('reference', {}).get('weekday_aware')}",
        ]

    sample = steps[len(steps) // 2]
    lines += [
        "",
        "A SINGLE INFERENCE, EXPLAINED",
        f"  {sample.state.explain()}",
        f"  occupancy: resident home {sample.context.resident_home:.2f}, "
        f"visitor {sample.context.visitor_present:.2f}, "
        f"attribution {sample.context.ambient_attribution():.2f}",
        "",
        "=" * 72,
        "This is a research toolkit. Nothing above is a clinical finding, and",
        "no part of this system is a medical device.",
        "=" * 72,
    ]
    return "\n".join(lines)


def run_demo(
    days: int = DEFAULT_DAYS,
    seed: int = DEFAULT_SEED,
    step: timedelta = timedelta(minutes=10),
    output: Path | None = None,
) -> dict[str, Any]:
    """Run the full demonstration and return its structured result."""
    result, delivered, injected = build_scenario(days=days, seed=seed)
    pipeline, steps = run_pipeline(result, delivered, step)
    payload = report(result, pipeline, steps, injected)
    print(render(payload, steps))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nStructured results written to {output}")
    return payload
