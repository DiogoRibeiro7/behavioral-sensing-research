"""Reconstruct the development panel and freeze the v0.3 circadian profile.

This script is intentionally outcome-blind. It uses only archive metadata and
activity annotations from the already-declared development panel. It never runs
the behavioural inference model and therefore cannot inspect primary external-
test outcomes.
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
HH_ID = re.compile(r"^(hh\d+)", re.IGNORECASE)
EXPECTED_PANEL_HOMES = 22
EXPECTED_FITTING_HOMES = 20
MULTI_RESIDENT_IDS = frozenset({"hh107", "hh121"})
EXCLUSION_REASON = "two-resident CASAS metadata"
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


def _home_id(path: Path) -> str:
    match = HH_ID.match(path.stem)
    if match is None:
        raise ValueError(f"cannot derive hh identifier from {path.name}")
    return match.group(1).lower()


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
    """Recover the historical 22-recording panel from file metadata only."""
    options = [(name, limit, _candidates(root, limit)) for name, limit in SIZE_RULES]
    evaluation = {
        name: {
            "byte_limit_exclusive": limit,
            "development_panel_count": len(paths),
        }
        for name, limit, paths in options
    }
    matches = [item for item in options if len(item[2]) == EXPECTED_PANEL_HOMES]
    if not matches:
        counts = {name: len(paths) for name, _, paths in options}
        raise SystemExit(
            "development panel reconstruction failed; expected at least one "
            f"12 MB convention to yield {EXPECTED_PANEL_HOMES} recordings, got {counts}"
        )

    if len(matches) == 1:
        name, limit, paths = matches[0]
        return paths, name, limit, evaluation, False

    reference_key = _cohort_key(root, matches[0][2])
    if any(_cohort_key(root, paths) != reference_key for _, _, paths in matches[1:]):
        raise SystemExit(
            "development panel reconstruction is genuinely ambiguous: multiple "
            "12 MB conventions yield 22 recordings but select different files"
        )

    names = [name for name, _, _ in matches]
    effective_limit = min(limit for _, limit, _ in matches)
    convention = "equivalent_" + "_and_".join(names)
    return matches[0][2], convention, effective_limit, evaluation, True


def _select_fitting_paths(panel_paths: list[Path]) -> list[Path]:
    panel_ids = [_home_id(path) for path in panel_paths]
    if (
        len(panel_paths) != EXPECTED_PANEL_HOMES
        or len(set(panel_ids)) != EXPECTED_PANEL_HOMES
    ):
        raise SystemExit("development panel does not contain 22 unique hh identifiers")
    if not MULTI_RESIDENT_IDS <= set(panel_ids):
        missing = sorted(MULTI_RESIDENT_IDS - set(panel_ids))
        raise SystemExit(
            f"expected metadata exclusions are absent from panel: {missing}"
        )

    fitting_paths = [
        path for path in panel_paths if _home_id(path) not in MULTI_RESIDENT_IDS
    ]
    if len(fitting_paths) != EXPECTED_FITTING_HOMES:
        raise SystemExit(
            f"expected {EXPECTED_FITTING_HOMES} single-resident fitting homes, "
            f"got {len(fitting_paths)}"
        )
    return fitting_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-record", default="15708568")
    parser.add_argument("--fitting-revision", required=True)
    args = parser.parse_args()

    panel_paths, convention, byte_limit, rule_evaluation, invariant = _reconstruct(
        args.archive_root
    )
    fitting_paths = _select_fitting_paths(panel_paths)
    panel_ids = [_home_id(path) for path in panel_paths]

    zone = ZoneInfo("America/Los_Angeles")
    recordings = [read_casas_hh(path, timezone=zone) for path in fitting_paths]
    fit = fit_circadian_profile(recordings)

    panel_homes = [
        {
            "id": _home_id(path),
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in panel_paths
    ]
    fitting_ids = [_home_id(path) for path in fitting_paths]
    exclusions = [
        {"id": home_id, "reason": EXCLUSION_REASON}
        for home_id in sorted(MULTI_RESIDENT_IDS)
    ]
    payload = {
        "schema_version": 2,
        "status": "frozen-development-fit",
        "candidate": "v0.2 + circadian prior only",
        "source": {
            "provider": "CASAS / Zenodo",
            "record": str(args.source_record),
            "archive_rule": "historical hh CSV under documented 12 MB cutoff",
            "size_convention": convention,
            "byte_limit_exclusive": byte_limit,
            "size_rule_evaluation": rule_evaluation,
            "cohort_invariant_across_size_conventions": invariant,
        },
        "development_panel_count": len(panel_homes),
        "development_panel_homes": panel_homes,
        "fitting_home_count": len(fitting_ids),
        "fitting_home_ids": fitting_ids,
        "fitting_exclusions": exclusions,
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
            {
                "development_panel_ids": panel_ids,
                "fitting_home_ids": fitting_ids,
                "profile_sha256": fit.sha256(),
            }
        )
    )


if __name__ == "__main__":
    main()
