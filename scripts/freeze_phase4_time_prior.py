"""Freeze the Phase 4 time-prior protocol, and write its page.

Usage::

    python scripts/freeze_phase4_time_prior.py [--docs-dir docs]

Writes ``artifacts/phase4/prior_uncertainty_protocol.json`` from the protocol
the code declares, with the digest of every source the run requires unchanged.
It refuses when the tree has uncommitted changes or the package has untracked
files, when a pinned record is not the file in the repository, when a base
protocol is not the frozen Phase 4 or combined-prior protocol, and when a
frozen file already exists that differs from what the code declares.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from importlib import metadata
from pathlib import Path

from sensor_modeling.datasets.prior_uncertainty_experiment import (
    RUN_VERSIONS,
    check_frozen_protocol,
    declared_protocol,
    write_protocol,
)
from sensor_modeling.datasets.prior_uncertainty_summary import (
    PROTOCOL_FILE,
    PROTOCOL_PAGE,
    render_protocol,
)
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits

SPLITS_PATH = Path("artifacts/phase1/household_splits.json")
PROTOCOL_PATH = Path(PROTOCOL_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    # Untracked modules under the package would enter the code digest.
    unclean = [
        line
        for line in status
        if not line.startswith("??") or line[3:].startswith("sensor_modeling/")
    ]
    if unclean:
        raise SystemExit(f"the tree has uncommitted changes: {unclean}")
    # The run refuses other versions than the freeze records, and installs these.
    other = {
        name: metadata.version(name)
        for name, version in RUN_VERSIONS.items()
        if metadata.version(name) != version
    }
    if other:
        raise SystemExit(f"freeze with the workflow's versions {RUN_VERSIONS}: {other}")
    protocol = declared_protocol(load_frozen_splits(SPLITS_PATH))
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
