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
SIZE_RULES = (
    ("decimal_MB", DECIMAL_LIMIT),
    ("binary_MiB", BINARY_LIMIT),
)


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
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )


def _cohort_key(root: Path, paths: list[Path]) -> tuple[str, ...]:
    return tuple(path.relative_to(root).as_posix() for path in paths)


def _reconstruct(
    root: Path,
) -> tuple[list[Path], str, int, dict[str, dict[str, int]], bool]:
    """Recover the documented 22-home cohort from metadata only.

    Historical prose says "under 12 MB" but did not retain whether MB meant
    decimal or MiB. Both conventions are evaluated using file metadata only.

    If exactly one convention yields 22 homes, that convention is recorded. If
    several conventions yield 22 homes and select the identical files, the
    cohort is invariant to the ambiguity and the stricter byte limit is used as
    the effective bound while every equivalent convention is recorded. The
    function refuses to proceed only when no convention reproduces 22 homes or
    when equally-sized candidate cohorts genuinely differ.
    """
    options = [(name, limit, _candidates(root, limit)) for name, limit in SIZE_RULES]
    evaluation = {
        name: {
            "byte_limit_exclusive": limit,
            "development_home_count": len(paths),
        }
        for name, limit, paths in options
    }
    matches = [item for item in options if len(item[2]) == EXPECTED_DEVELOPMENT_HOMES]
    if not matches:
        counts = {name: len(paths) for name, _, paths in options}
        raise SystemExit(
            "development cohort reconstruction failed; expected at least one "
            f"12 MB convention to yield {EXPECTED_DEVELOPMENT_HOMES} homes, "
            f"got {counts}"
        )

    if len(matches) == 1:
        name, limit, paths = matches[0]
        return paths, name, limit, evaluation, False

    reference_key = _cohort_key(root, matches[0][2])
    if any(_cohort_key(root, paths) != reference_key for _, _, paths in matches[1:]):
        raise SystemExit(
            "development cohort reconstruction is genuinely ambiguous: multiple "
            "12 MB conventions yield 22 homes but select different files"
        )

    names = [name for name, _, _ in matches]
    effective_limit = min(limit for _, limit, _ in matches)
    convention = "equivalent_" + "_and_".join(names)
    return matches[0][2], convention, effective_limit, evaluation, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-record", default="15708568")
    parser.add_argument("--fitting-revision", required=True)
    args = parser.parse_args()

    paths, convention, byte_limit, rule_evaluation, invariant = _reconstruct(
        args.archive_root
    )
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
            "byte_limit_exclusive": byte_limit,
            "size_rule_evaluation": rule_evaluation,
            "cohort_invariant_across_size_conventions": invariant,
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
