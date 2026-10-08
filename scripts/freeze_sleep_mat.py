"""Freeze the sleep-mat protocol, and write its page.

Usage::

    python scripts/freeze_sleep_mat.py [--docs-dir docs]

Writes ``artifacts/sleep_mat/sleep_mat_protocol.json`` from the protocol the
code declares, with its digest and, beside it, the digest of every source file
of the package and of the scripts that freeze and run it, the versions of
Python and of the numerical libraries, and every default setting as they are
now. It refuses when the protocol does not pin the planning record and the
published TIHM records to the files in the repository, and when a frozen file
already exists that differs from what the code declares: a frozen protocol is
not written twice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sensor_modeling.datasets.sleep_mat_protocol import (
    PLANNING_RECORD,
    PUBLISHED_RECORD,
    PUBLISHED_RECORD_SHA256,
    SILENT_HOME_RECORD,
    SILENT_HOME_RECORD_SHA256,
    check_frozen_protocol,
    declared_protocol,
    file_sha256,
    write_protocol,
)
from sensor_modeling.datasets.sleep_mat_summary import PROTOCOL_PAGE, render_protocol

PROTOCOL_PATH = Path("artifacts/sleep_mat/sleep_mat_protocol.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    protocol = declared_protocol()
    pinned = {
        PLANNING_RECORD: protocol.planning_record_sha256,
        PUBLISHED_RECORD: PUBLISHED_RECORD_SHA256,
        SILENT_HOME_RECORD: SILENT_HOME_RECORD_SHA256,
    }
    for name, digest in pinned.items():
        if file_sha256(Path(name)) != digest:
            raise SystemExit(f"{name} is not the file the protocol pins")
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
