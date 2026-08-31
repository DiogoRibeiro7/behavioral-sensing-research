"""Screen candidate homes for the external-validation cohort.

The contract requires the eligible test-home identifiers and raw-file checksums
to be frozen in a machine-readable manifest **before** primary scoring. This
builds that manifest.

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

#: Recordings the development panel already consumed. Permanently ineligible:
#: their outcomes were inspected, which is sufficient to make them development
#: data regardless of resident count.
DEVELOPMENT_PANEL = frozenset(
    f"hh{n}"
    for n in [
        101,
        102,
        103,
        105,
        106,
        107,
        108,
        110,
        111,
        114,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        129,
        130,
    ]
)

#: Single-resident homes, transcribed from the Zenodo record description for
#: record 15708568. Only families present in ``labeled_data.zip`` are listed.
SINGLE_RESIDENT = frozenset(
    [
        f"hh{n}"
        for n in list(range(101, 107)) + list(range(108, 121)) + list(range(122, 131))
    ]
    + ["rw101", "rw103", "rw105", "rw106", "rw107"]
    + ["mv101"]
    + [
        f"tm{n:03d}"
        for n in list(range(1, 4))
        + list(range(5, 12))
        + list(range(13, 23))
        + [26, 29, 32]
        + list(range(35, 44))
    ]
    + [
        "ihs07",
        "ihs11",
        "ihs12",
        "ihs21",
        "ihs28",
        "ihs35",
        "ihs37",
        "ihs38",
        "ihs40",
        "ihs58",
        "ihs59",
        "ihs68",
        "ihs70",
        "ihs75",
        "ihs80",
        "ihs84",
        "ihs85",
        "ihs95",
        "ihs96",
        "ihs107",
        "ihs108",
        "ihs114",
        "ihs118",
    ]
    + ["mn57", "mn77", "mn82", "mn85"]
)

#: The contract requires at least five of the seven frozen states to be mappable.
MINIMUM_MAPPED_STATES = 5

NAME = re.compile(r"^([a-z]+\d+)\.csv$", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def screen(path: Path, zone: ZoneInfo) -> dict[str, object]:
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

    if home in DEVELOPMENT_PANEL:
        row.update(eligible=False, reason="development panel; outcomes inspected")
        return row
    if home not in SINGLE_RESIDENT:
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
    args = parser.parse_args()

    zone = ZoneInfo(args.timezone)
    paths = sorted(
        (p for p in args.archive_root.rglob("*.csv") if NAME.match(p.name)),
        key=lambda p: p.name.lower(),
    )
    if not paths:
        raise SystemExit(f"no candidate recordings found under {args.archive_root}")

    screened = [screen(path, zone) for path in paths]
    eligible = [row for row in screened if row["eligible"]]

    manifest = {
        "schema_version": 1,
        "status": "prospective-cohort-manifest",
        "purpose": "external-validation primary test cohort",
        "scored": False,
        "source": {
            "provider": "CASAS / Zenodo",
            "record": args.source_record,
            "resident_counts": "Zenodo record description for the same record",
        },
        "criteria": {
            "outside_development_panel": True,
            "single_resident": True,
            "minimum_mapped_states": MINIMUM_MAPPED_STATES,
            "representable_observations": True,
            "non_zero_labelled_coverage": True,
            "size_cutoff": None,
        },
        "development_panel": sorted(DEVELOPMENT_PANEL),
        "eligible_count": len(eligible),
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
                "eligible": len(eligible),
                "screened": len(screened),
                "homes": [row["id"] for row in eligible],
            }
        )
    )


if __name__ == "__main__":
    main()
