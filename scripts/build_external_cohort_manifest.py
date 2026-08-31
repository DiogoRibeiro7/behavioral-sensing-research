"""Screen candidate homes for the external-validation cohort.

The contract requires the eligible test-home identifiers and raw-file checksums
to be frozen in a machine-readable manifest **before** primary scoring. This
builds that manifest.

The manifest records the revision of this script and the checksum of the
registry it read, so a frozen cohort can be traced to the code and the metadata
that produced it. The profile freeze records its fitting revision for the same
reason; a cohort without one would be the weaker half of the pair.

It is outcome-blind by construction. It reads resident-count metadata, file
sizes, checksums and activity annotations; it never constructs a pipeline, never
runs inference, and never computes a metric. Eligibility is decided by the
contract's six criteria and by nothing about how well a home would score.

Resident counts come from the Zenodo record description, which is the
authoritative published source and the same one identifying ``hh107`` and
``hh121`` as two-resident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import read_casas_hh

#: Machine-readable resident counts and development panel.
#:
#: Read rather than transcribed. An inline copy of the Zenodo table was a second
#: source of truth for the same facts, and a second source of truth drifts: the
#: transcription was already missing two single-resident homes when the registry
#: was compared against it.
REGISTRY_PATH = Path("artifacts/v03/casas_v1_resident_registry.json")


def load_registry(
    path: Path,
) -> tuple[frozenset[str], frozenset[str], dict[str, object]]:
    """Return the development panel, single-resident ids, and provenance."""
    if not path.exists():
        raise SystemExit(f"resident registry not found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    panel = frozenset(payload["development_only_home_ids"])
    single = frozenset(payload["single_resident_home_ids"])
    if not panel or not single:
        raise SystemExit("resident registry is missing required id lists")
    return panel, single, payload.get("source", {})


#: The contract requires at least five of the seven frozen states to be mappable.
MINIMUM_MAPPED_STATES = 5

NAME = re.compile(r"^([a-z]+\d+)\.csv$", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def screen(
    path: Path,
    zone: ZoneInfo,
    panel: frozenset[str],
    single_resident: frozenset[str],
) -> dict[str, object]:
    """Apply the contract's eligibility criteria to one recording.

    Returns the verdict and the evidence for it, including the reason for any
    rejection, so that a reader can audit the decision without rerunning this.
    """
    match = NAME.match(path.name)
    home = match.group(1).lower() if match else path.stem.lower()
    row: dict[str, object] = {
        "id": home,
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }

    if home in panel:
        row.update(eligible=False, reason="development panel; outcomes inspected")
        return row
    if home not in single_resident:
        row.update(eligible=False, reason="not single-resident by Zenodo metadata")
        return row

    try:
        recording = read_casas_hh(path, timezone=zone)
    except Exception as error:  # parser or provenance defect
        row.update(eligible=False, reason=f"unreadable: {type(error).__name__}")
        return row

    mapped = {
        interval.state
        for interval in recording.activities
        if interval.state is not None
    }
    row["mapped_states"] = sorted(state.value for state in mapped)
    row["labelled_fraction"] = recording.labelled_fraction
    row["observations"] = len(recording.observations)
    row["unmapped_locations"] = sorted(recording.unmapped_sensors)
    row["unmapped_activities"] = sorted(recording.unmapped_activities)

    if not recording.observations:
        row.update(eligible=False, reason="no representable observations")
    elif len(mapped) < MINIMUM_MAPPED_STATES:
        row.update(
            eligible=False,
            reason=f"only {len(mapped)} mapped states, {MINIMUM_MAPPED_STATES} required",
        )
    elif recording.labelled_fraction <= 0.0:
        row.update(eligible=False, reason="zero mapped labelled coverage")
    else:
        row.update(eligible=True, reason="meets all criteria")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timezone", default="America/Los_Angeles")
    parser.add_argument("--source-record", default="15708568")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument(
        "--screening-revision",
        required=True,
        help="Git revision of the screening code, recorded in the manifest",
    )
    args = parser.parse_args()

    panel, single_resident, registry_source = load_registry(args.registry)
    zone = ZoneInfo(args.timezone)
    paths = sorted(
        (p for p in args.archive_root.rglob("*.csv") if NAME.match(p.name)),
        key=lambda p: p.name.lower(),
    )
    if not paths:
        raise SystemExit(f"no candidate recordings found under {args.archive_root}")

    screened = [screen(path, zone, panel, single_resident) for path in paths]
    eligible = [row for row in screened if row["eligible"]]

    # Family composition decides how a result should be read. The candidate was
    # developed entirely on `hh`; a cohort drawn mostly from other families is a
    # harder test than the development panel implies, and a null result there
    # means something different from a null result within `hh`.
    families: dict[str, int] = {}
    for row in eligible:
        family = str(row["id"]).rstrip("0123456789")
        families[family] = families.get(family, 0) + 1

    manifest = {
        "schema_version": 1,
        "screening_revision": args.screening_revision,
        "status": "prospective-cohort-manifest",
        "purpose": "external-validation primary test cohort",
        "scored": False,
        "source": {
            "provider": "CASAS / Zenodo",
            "record": args.source_record,
            "resident_counts": "artifacts/v03/casas_v1_resident_registry.json",
            "registry_source": registry_source,
            "registry_sha256": _sha256(args.registry),
        },
        "criteria": {
            "outside_development_panel": True,
            "single_resident": True,
            "minimum_mapped_states": MINIMUM_MAPPED_STATES,
            "representable_observations": True,
            "non_zero_labelled_coverage": True,
            "size_cutoff": None,
        },
        "development_panel": sorted(panel),
        "eligible_count": len(eligible),
        "eligible_by_family": dict(sorted(families.items())),
        "development_panel_family": "hh",
        "eligible_homes": eligible,
        "screened": screened,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "screening_revision": args.screening_revision,
                "eligible": len(eligible),
                "screened": len(screened),
                "homes": [row["id"] for row in eligible],
            }
        )
    )


if __name__ == "__main__":
    main()
