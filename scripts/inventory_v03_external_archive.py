"""Inventory the frozen CASAS external-validation archive without model inference.

This script is deliberately limited to archive/file metadata and raw text
semantics. It must not import the behavioural inference or evaluation stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

_TIMESTAMP = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})[\s,]+(\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
_MARKER = re.compile(r'([A-Za-z][A-Za-z0-9_]*)\s*=\s*"?(begin|end)"?', re.I)
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(value: str) -> str:
    return "".join(_TOKEN.findall(value.lower()))


def _home_matches(path: Path, known_ids: set[str]) -> list[str]:
    """Return resident-registry IDs explicitly represented in a path component."""
    components = [path.stem, *path.parts[:-1]]
    normalised = {_normalise(component) for component in components if component}
    return sorted(home_id for home_id in known_ids if _normalise(home_id) in normalised)


def _delimiter(line: str) -> str:
    if "," in line:
        return "comma"
    if "\t" in line:
        return "tab"
    return "whitespace"


def _inspect_text(path: Path, max_lines: int = 5000) -> dict[str, object]:
    """Inspect raw syntax only; never instantiate a dataset adapter or model."""
    delimiters: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    timestamp_rows = 0
    nonempty_rows = 0
    first_timestamp = None
    last_timestamp = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, raw in enumerate(handle):
            if index >= max_lines:
                break
            line = raw.strip()
            if not line:
                continue
            nonempty_rows += 1
            delimiters[_delimiter(raw)] += 1
            stamp = _TIMESTAMP.match(raw)
            if stamp is not None:
                timestamp_rows += 1
                value = f"{stamp.group(1)} {stamp.group(2)}"
                if first_timestamp is None:
                    first_timestamp = value
                last_timestamp = value
            for match in _MARKER.finditer(raw):
                labels[match.group(1)] += 1

    return {
        "sampled_nonempty_rows": nonempty_rows,
        "timestamp_parseable_rows": timestamp_rows,
        "timestamp_parseable_fraction": (
            timestamp_rows / nonempty_rows if nonempty_rows else 0.0
        ),
        "first_sampled_timestamp": first_timestamp,
        "last_sampled_timestamp": last_timestamp,
        "delimiter_counts": dict(sorted(delimiters.items())),
        "annotation_marker_labels": dict(sorted(labels.items())),
    }


def build_inventory(
    archive_root: Path, registry: dict[str, object]
) -> dict[str, object]:
    singles = set(registry["single_resident_home_ids"])
    development = set(registry["development_only_home_ids"])
    candidates = singles - development
    known = (
        singles
        | set(registry["two_resident_home_ids"])
        | set(registry["more_than_two_resident_home_ids"])
        | set(registry["unknown_resident_home_ids"])
    )

    by_home: dict[str, list[dict[str, object]]] = {
        home_id: [] for home_id in candidates
    }
    ambiguous_paths: list[dict[str, object]] = []

    for path in sorted(p for p in archive_root.rglob("*") if p.is_file()):
        matches = _home_matches(path.relative_to(archive_root), known)
        if len(matches) > 1:
            ambiguous_paths.append(
                {
                    "path": path.relative_to(archive_root).as_posix(),
                    "matches": matches,
                }
            )
            continue
        if len(matches) != 1 or matches[0] not in candidates:
            continue

        entry: dict[str, object] = {
            "path": path.relative_to(archive_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        try:
            entry["raw_text_audit"] = _inspect_text(path)
        except OSError as exc:
            entry["raw_text_audit_error"] = type(exc).__name__
        by_home[matches[0]].append(entry)

    homes = [
        {
            "home_id": home_id,
            "resident_count": 1,
            "development_only": False,
            "files": by_home[home_id],
            "archive_member_found": bool(by_home[home_id]),
        }
        for home_id in sorted(candidates)
    ]

    return {
        "schema_version": 1,
        "status": "prediction-free-archive-inventory",
        "source": registry["source"],
        "selection_stage": "inventory_only_not_final_eligibility",
        "model_inference_performed": False,
        "performance_metrics_computed": False,
        "candidate_registry_count": len(candidates),
        "homes": homes,
        "ambiguous_paths": ambiguous_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    payload = build_inventory(args.archive_root, registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_registry_count": payload["candidate_registry_count"],
                "archive_members_found": sum(
                    bool(home["archive_member_found"]) for home in payload["homes"]
                ),
                "ambiguous_paths": len(payload["ambiguous_paths"]),
            }
        )
    )


if __name__ == "__main__":
    main()
