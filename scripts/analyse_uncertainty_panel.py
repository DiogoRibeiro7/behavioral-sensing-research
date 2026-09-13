"""Analyse uncertainty separation on the 22-home CASAS development panel.

This is exploratory/mechanistic analysis for the post-v0.3 Paper 1 work. It
reuses the development-panel membership already recorded in the frozen v0.3
cohort manifest, runs the ordinary default inference path on each home, and
writes both household diagnostics and an equal-household panel summary.

It does not touch or rescore the frozen 43-home primary external cohort.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import (
    evaluate_recording,
    read_casas_hh,
    uncertainty_panel_summary,
)

MANIFEST_PATH = Path("artifacts/v03/external_cohort_manifest.json")
DEFAULT_TIMEZONE = "America/Los_Angeles"
DEFAULT_STEP_MINUTES = 5


def resolve_recording(root: Path, filename: str) -> Path:
    """Find one recording in the archive, refusing missing or ambiguous matches."""
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one archive member named {filename}, found "
            f"{len(matches)}"
        )
    return matches[0]


def build_payload(
    rows: list[dict[str, Any]],
    *,
    timezone: str,
    step_minutes: int,
) -> dict[str, Any]:
    """Build the machine-readable Paper 1 panel result."""
    diagnostics = [row.pop("_diagnostics") for row in rows]
    return {
        "schema_version": 1,
        "analysis": "paper1-uncertainty-correctness-separation",
        "cohort": "v03-development-panel",
        "inference": "default inference; circadian profile disabled",
        "timezone": timezone,
        "step_minutes": step_minutes,
        "summary": uncertainty_panel_summary(diagnostics),
        "homes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--step-minutes", type=int, default=DEFAULT_STEP_MINUTES)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    homes = list(manifest["development_panel"])
    if len(homes) != 22:
        raise SystemExit(
            f"expected the recorded 22-home development panel, found {len(homes)}"
        )

    zone = ZoneInfo(args.timezone)
    step = timedelta(minutes=args.step_minutes)
    rows: list[dict[str, Any]] = []

    for home in homes:
        path = resolve_recording(args.archive_root, f"{home}.csv")
        recording = read_casas_hh(path, timezone=zone)
        result = evaluate_recording(recording, step=step)
        rows.append(
            {
                "home": home,
                "scored": result.scored,
                "scored_fraction": result.scored_fraction,
                "labelled_fraction": result.labelled_fraction,
                "uncertainty": result.uncertainty.to_dict(),
                "_diagnostics": result.uncertainty,
            }
        )
        print(
            f"{home:8} scored={result.scored:6d}  "
            f"confidence_auc={result.uncertainty.confidence_correctness_auc!r}  "
            f"information_gain_auc="
            f"{result.uncertainty.information_gain_correctness_auc!r}",
            flush=True,
        )

    payload = build_payload(
        rows,
        timezone=args.timezone,
        step_minutes=args.step_minutes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    summary = payload["summary"]
    print()
    print(f"homes             {summary['homes']}")
    print(
        "confidence AUC    "
        f"{summary['median_confidence_correctness_auc']!r} "
        f"({summary['confidence_homes']} homes)"
    )
    print(
        "information AUC   "
        f"{summary['median_information_gain_correctness_auc']!r} "
        f"({summary['information_gain_homes']} homes)"
    )
    print(f"written to        {args.output}")


if __name__ == "__main__":
    main()
