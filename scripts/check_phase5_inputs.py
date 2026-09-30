"""Phase 5: check the external dataset against the frozen protocol, without scoring.

It verifies every file of the extracted UCI archive against the digests frozen
in ``artifacts/phase5/external_protocol.json``. It then validates each home's
adaptation period, its first seven days, against the frozen mapping. Of the
scored period it reads nothing, and it computes no model output.

Usage::

    python scripts/check_phase5_inputs.py <extracted_archive_root>
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

from sensor_modeling.datasets.external_protocol import (
    adaptation_end,
    check_frozen_protocol,
    declared_protocol,
    eligibility,
)
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.external import validate_household
from sensor_modeling.external.ordonez import (
    FILES,
    FOLDER,
    HOUSEHOLDS,
    ORDONEZ_MAPPING,
    read_household,
)

SPLITS_PATH = Path("artifacts/phase1/household_splits.json")
PROTOCOL_PATH = Path("artifacts/phase5/external_protocol.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("archive_root", type=Path)
    args = parser.parse_args()

    check_frozen_protocol(
        declared_protocol(load_frozen_splits(SPLITS_PATH)), PROTOCOL_PATH
    )
    folder = args.archive_root / FOLDER
    for name, digest in FILES.items():
        found = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if found != digest:
            raise SystemExit(f"{name} does not match its frozen digest")
    print(f"every file matches its frozen digest ({len(FILES)} files)")
    for home in HOUSEHOLDS:
        data = read_household(folder, home)
        cut = adaptation_end(data)
        early = replace(
            data,
            events=tuple(e for e in data.events if e.timestamp < cut),
            annotations=tuple(a for a in data.annotations if a.start < cut),
        )
        report = validate_household(early, ORDONEZ_MAPPING)
        eligible, blocking = eligibility(report)
        print(
            f"{home}: adaptation period until {cut.isoformat()}; eligible: {eligible}"
        )
        for issue in report.issues:
            print(f"  {issue.severity.value:7} {issue.code}: {issue.count}")
        if blocking:
            print(f"  blocking: {', '.join(blocking)}")


if __name__ == "__main__":
    main()
