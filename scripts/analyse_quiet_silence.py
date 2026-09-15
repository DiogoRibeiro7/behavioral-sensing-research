#!/usr/bin/env python3
"""Characterise confidence accumulation under complementary sensor silence.

This is a controlled mechanism analysis for Paper 1. It does not alter the
filter, emissions, abstention rule, or thresholds. Each condition starts from a
fresh stationary filter and applies five-minute intervals with no activations.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from sensor_modeling.datasets import hh_sensor_specs
from sensor_modeling.fusion import MultimodalBayesFilter, default_emissions
from sensor_modeling.observations import SensorRegistry
from sensor_modeling.states import StateOntology

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
STEP_MINUTES = 5
DURATIONS = (5, 15, 30, 60)
MOTION_LOCATIONS = ("Kitchen", "Bathroom", "Bedroom", "LivingRoom", "Hall")
DEFAULT_LOCATIONS = (*MOTION_LOCATIONS, "FrontDoor")
CONDITIONS = (
    ("prior_only", DEFAULT_LOCATIONS, 0.0),
    ("kitchen", ("Kitchen",), 1.0),
    ("kitchen_bathroom", ("Kitchen", "Bathroom"), 1.0),
    (
        "kitchen_bathroom_bedroom",
        ("Kitchen", "Bathroom", "Bedroom"),
        1.0,
    ),
    ("all_motion", MOTION_LOCATIONS, 1.0),
    ("front_door", ("FrontDoor",), 1.0),
    ("bedroom_front_door", ("Bedroom", "FrontDoor"), 1.0),
    ("kitchen_front_door", ("Kitchen", "FrontDoor"), 1.0),
    ("all_motion_front_door", DEFAULT_LOCATIONS, 1.0),
)


def build_filter(locations: tuple[str, ...]) -> MultimodalBayesFilter:
    """Build the aggregate-room deployment used in the quiet-period tests."""
    specs, unmapped = hh_sensor_specs(locations)
    if unmapped:
        raise ValueError(f"unmapped quiet-period locations: {unmapped}")
    registry = SensorRegistry.from_specs(specs)
    ontology = StateOntology()
    return MultimodalBayesFilter(
        ontology,
        default_emissions(registry, ontology),
        registry=registry,
    )


def evidence_strength(estimate: object) -> float:
    """Return mean absolute per-sensor support for one update."""
    evidence = getattr(estimate, "evidence")
    if not evidence:
        return 0.0
    return float(np.mean([abs(item.support) for item in evidence]))


def run_condition(
    name: str,
    locations: tuple[str, ...],
    reliability: float,
) -> list[dict[str, object]]:
    """Run one silence configuration and return the requested snapshots."""
    model = build_filter(locations)
    rows: list[dict[str, object]] = []

    for minute in range(STEP_MINUTES, max(DURATIONS) + STEP_MINUTES, STEP_MINUTES):
        estimate = model.update(
            T0 + timedelta(minutes=minute),
            (),
            reliabilities=reliability,
        )
        if minute not in DURATIONS:
            continue

        rows.append(
            {
                "condition": name,
                "locations": list(locations),
                "reliability": reliability,
                "duration_minutes": minute,
                "most_likely": estimate.most_likely.value,
                "confidence": estimate.confidence,
                "margin": estimate.margin,
                "normalised_entropy": estimate.normalised_entropy,
                "information_gain": estimate.information_gain,
                "evidence_strength": evidence_strength(estimate),
                "abstained": estimate.abstained,
            }
        )

    return rows


def build_payload() -> dict[str, object]:
    """Build the complete deterministic quiet-silence mechanism grid."""
    rows = [
        row
        for name, locations, reliability in CONDITIONS
        for row in run_condition(name, locations, reliability)
    ]
    return {
        "schema_version": 1,
        "analysis": "paper1-quiet-silence-mechanism-grid",
        "step_minutes": STEP_MINUTES,
        "durations_minutes": list(DURATIONS),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/paper1/quiet_silence_grid.json"),
    )
    args = parser.parse_args()

    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
