"""Freeze the held-out confirmation protocol, and write its page.

Usage::

    python scripts/freeze_heldout_confirmation.py [--docs-dir docs]

Writes ``artifacts/heldout/heldout_protocol.json`` from the protocol the code
declares. It refuses when a pinned record is not the file in the repository,
when the base protocol is not the frozen fitted-rates protocol, and when a
frozen file already exists that differs from what the code declares.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sensor_modeling.datasets.heldout_experiment import (
    check_frozen_protocol,
    declared_protocol,
    write_protocol,
)
from sensor_modeling.datasets.heldout_summary import (
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
