"""Reconstruct the development cohort and freeze the v0.3 circadian profile.

This script is intentionally outcome-blind. It uses only archive metadata and
activity annotations from the already-declared development cohort. It never
runs the behavioural inference model and therefore cannot inspect primary
external-test outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import fit_circadian_profile, read_casas_hh

HH_NAME = re.compile(r"^hh\d+.*\.csv$", re.IGNORECASE)
EXPECTED_DEVELOPMENT_HOMES = 22
DECIMAL_LIMIT = 12_000_000
BINARY_LIMIT = 12 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidates(root: Path, limit: int) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*.csv")
            if HH_NAME.match(path.name) and path.stat().st_size < limit
        ),
        key=lambda path: path.name.lower(),
    )


def _agreeing_conventions(root: Path) -> list[str]:
    """Return every 12 MB convention that selects the documented cohort.

    More than one means the prose's decimal/binary ambiguity never mattered:
    they pick the same files, so the cohort is determined by the archive.
    """
    return [
        name
        for name, limit in (("decimal_MB", DECIMAL_LIMIT), ("binary_MiB", BINARY_LIMIT))
        if len(_candidates(root, limit)) == EXPECTED_DEVELOPMENT_HOMES
    ]


def _reconstruct(root: Path) -> tuple[list[Path], str, int]:
    """Recover the documented 22-home cohort from metadata only.

    Historical prose says "under 12 MB" but did not retain whether MB meant
    decimal or MiB. Both metadata conventions are evaluated. What matters is
    whether they disagree about *which* homes are selected, not how many
    conventions happen to yield the documented count: if every convention
    reaching 22 homes selects the same 22, the cohort is determined and the
    prose's ambiguity never mattered. Only a disagreement about membership is
    genuinely unresolvable from metadata.

    No labels, model predictions or scores participate in this choice.
    """
    options = [
        ("decimal_MB", DECIMAL_LIMIT, _candidates(root, DECIMAL_LIMIT)),
        ("binary_MiB", BINARY_LIMIT, _candidates(root, BINARY_LIMIT)),
    ]
    matches = [item for item in options if len(item[2]) == EXPECTED_DEVELOPMENT_HOMES]
    if not matches:
        counts = {name: len(paths) for name, _, paths in options}
        raise SystemExit(
            "development cohort reconstruction failed; no 12 MB convention "
            f"yields {EXPECTED_DEVELOPMENT_HOMES} homes, got {counts}"
        )

    selections = {tuple(path.name for path in paths) for _, _, paths in matches}
    if len(selections) != 1:
        raise SystemExit(
            "development cohort reconstruction is ambiguous; the 12 MB "
            "conventions each yield "
            f"{EXPECTED_DEVELOPMENT_HOMES} homes but disagree on which: "
            + "; ".join(
                f"{name}={[path.name for path in paths]}" for name, _, paths in matches
            )
        )

    # Record a single convention so the artefact stays within the validator's
    # vocabulary, and the strictest limit, since every matching convention
    # selected the same files anyway. The agreement itself is recorded
    # alongside it.
    name, limit, paths = min(matches, key=lambda item: item[1])
    return paths, name, limit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-record", default="15708568")
    parser.add_argument("--fitting-revision", required=True)
    args = parser.parse_args()

    paths, convention, byte_limit = _reconstruct(args.archive_root)
    zone = ZoneInfo("America/Los_Angeles")
    recordings = [read_casas_hh(path, timezone=zone) for path in paths]
    fit = fit_circadian_profile(recordings)

    homes = [
        {
            "id": path.stem,
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    payload = {
        "schema_version": 1,
        "status": "frozen-development-fit",
        "candidate": "v0.2 + circadian prior only",
        "source": {
            "provider": "CASAS / Zenodo",
            "record": str(args.source_record),
            "archive_rule": "single-resident hh CSV under documented 12 MB cutoff",
            "size_convention": convention,
            "equivalent_size_conventions": _agreeing_conventions(args.archive_root),
            "byte_limit_exclusive": byte_limit,
        },
        "development_home_count": len(homes),
        "development_homes": homes,
        "fitting_revision": args.fitting_revision,
        "fit": fit.to_dict(),
        "profile_sha256": fit.sha256(),
        "test_outcomes_inspected": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"homes": [row["id"] for row in homes], "profile_sha256": fit.sha256()}
        )
    )


if __name__ == "__main__":
    main()
