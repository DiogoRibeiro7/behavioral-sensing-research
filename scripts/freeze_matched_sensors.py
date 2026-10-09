"""Freeze the matched-sensor protocol, and write its page.

Usage::

    python scripts/freeze_matched_sensors.py [--docs-dir docs]

Writes ``artifacts/matched_sensors/matched_sensors_protocol.json`` from the
protocol the code declares, with its digest and, beside it, the digest of every
source file of the package and of the scripts that freeze and run it, the
versions of Python and of the numerical libraries, and every default setting
as they are now. It refuses when the protocol does not pin the planning record
and the published records to the files in the repository, when its matched and
sensitivity profiles are not the planning record's nearest and second-nearest
grid points, and when a frozen file already exists that differs from what the
code declares.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sensor_modeling.datasets.matched_sensors_protocol import (
    HOLD_OFF_SECONDS,
    MATCHED,
    PLANNING_RECORD,
    THRESHOLD_PROTOCOL,
    THRESHOLD_PROTOCOL_SHA256,
    THRESHOLD_RECORD,
    THRESHOLD_RECORD_SHA256,
    TIHM_POST_HOC_RECORD,
    TIHM_POST_HOC_RECORD_SHA256,
    check_frozen_protocol,
    declared_protocol,
    file_sha256,
    planned_choice,
    write_protocol,
)
from sensor_modeling.datasets.matched_sensors_summary import (
    PROTOCOL_PAGE,
    render_protocol,
)

PROTOCOL_PATH = Path("artifacts/matched_sensors/matched_sensors_protocol.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    protocol = declared_protocol()
    pinned = {
        PLANNING_RECORD: protocol.planning_record_sha256,
        THRESHOLD_PROTOCOL: THRESHOLD_PROTOCOL_SHA256,
        THRESHOLD_RECORD: THRESHOLD_RECORD_SHA256,
        TIHM_POST_HOC_RECORD: TIHM_POST_HOC_RECORD_SHA256,
    }
    for name, digest in pinned.items():
        if file_sha256(Path(name)) != digest:
            raise SystemExit(f"{name} is not the file the protocol pins")
    planning = json.loads(Path(PLANNING_RECORD).read_text(encoding="utf-8"))
    chosen = planning["results"]["chosen"]
    if (
        chosen["hold_off_seconds"] != HOLD_OFF_SECONDS
        or chosen["at_the_edge_of_the_grid"]
    ):
        raise SystemExit("the hold-off or the grid is not what the protocol says")
    for name, (scale, spill) in planned_choice(planning).items():
        profile = protocol.profile(name)
        if (profile.presence_scale, profile.spill_rate) != (scale, spill):
            raise SystemExit(f"the {name} profile is not the planning record's")
    if planned_choice(planning)[MATCHED] != (
        chosen["presence_scale"],
        chosen["spill_rate"],
    ):
        raise SystemExit("the planning record's choice is not its nearest point")
    if PROTOCOL_PATH.exists():
        check_frozen_protocol(protocol, PROTOCOL_PATH)
        print(f"{PROTOCOL_PATH} is already frozen and is what the code declares")
    else:
        digest = write_protocol(protocol, PROTOCOL_PATH)
        print(f"frozen {PROTOCOL_PATH}, {digest}")
    frozen = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    page = args.docs_dir / PROTOCOL_PAGE
    page.write_text(render_protocol(frozen), encoding="utf-8", newline="\n")
    print(f"written {page}")


if __name__ == "__main__":
    main()
