"""Analyse uncertainty separation on the 22-home CASAS development panel.

This is exploratory/mechanistic analysis for the post-v0.3 Paper 1 work. It
reuses the development-panel membership already recorded in the frozen v0.3
cohort manifest, verifies every source file against that manifest, runs the
ordinary default inference path on each home, and writes both household
diagnostics and an equal-household panel summary.

It does not touch or rescore the frozen 43-home primary external cohort.
"""

from __future__ import annotations

import argparse
import hashlib
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


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source recording."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def development_sources(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the frozen source identity for every development-panel home."""
    homes = [str(home) for home in manifest["development_panel"]]
    if len(homes) != 22:
        raise SystemExit(
            f"expected the recorded 22-home development panel, found {len(homes)}"
        )

    screened = {str(row["id"]): row for row in manifest["screened"]}
    missing = [home for home in homes if home not in screened]
    if missing:
        raise SystemExit(
            "frozen manifest is missing screened source identities for: "
            + ", ".join(missing)
        )

    sources: dict[str, dict[str, Any]] = {}
    for home in homes:
        entry = screened[home]
        if entry.get("eligible") is not False or entry.get("reason") != (
            "development panel; outcomes inspected"
        ):
            raise SystemExit(
                f"{home}: frozen manifest no longer identifies this as a "
                "development-only home"
            )
        expected_filename = f"{home}.csv"
        if entry.get("filename") != expected_filename:
            raise SystemExit(
                f"{home}: frozen manifest filename is {entry.get('filename')!r}, "
                f"expected {expected_filename!r}"
            )
        sources[home] = entry
    return sources


def verify_recording(path: Path, entry: dict[str, Any]) -> str:
    """Refuse any source file that differs from the frozen manifest bytes."""
    expected_bytes = int(entry["bytes"])
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise SystemExit(
            f"{entry['id']}: source size {actual_bytes} does not match frozen "
            f"size {expected_bytes}"
        )

    actual_sha256 = sha256(path)
    expected_sha256 = str(entry["sha256"])
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"{entry['id']}: source SHA-256 {actual_sha256} does not match "
            f"frozen {expected_sha256}"
        )
    return actual_sha256


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
        "source_identity": "artifacts/v03/external_cohort_manifest.json screened",
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
    sources = development_sources(manifest)

    zone = ZoneInfo(args.timezone)
    step = timedelta(minutes=args.step_minutes)
    rows: list[dict[str, Any]] = []

    for home, source in sources.items():
        path = resolve_recording(args.archive_root, str(source["filename"]))
        source_sha256 = verify_recording(path, source)
        recording = read_casas_hh(path, timezone=zone)
        result = evaluate_recording(recording, step=step)
        rows.append(
            {
                "home": home,
                "source_bytes": int(source["bytes"]),
                "source_sha256": source_sha256,
                "scored": result.scored,
                "scored_fraction": result.scored_fraction,
                "labelled_fraction": result.labelled_fraction,
                "uncertainty": result.uncertainty.to_dict(),
                "_diagnostics": result.uncertainty,
            }
        )
        print(
            f"{home:8} source=verified  scored={result.scored:6d}  "
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
