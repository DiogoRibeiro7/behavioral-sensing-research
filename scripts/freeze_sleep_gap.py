"""Freeze the sleep-gap plan, and write its page.

Usage::

    python scripts/freeze_sleep_gap.py [--docs-dir docs]

Writes ``artifacts/sleep_gap/sleep_gap_plan.json`` from the plan the code
declares, with its digest and, beside it, the digest of every source file of
the package and of the scripts that freeze and run it, the versions of Python
and of the numerical libraries, and every default setting as they are now. It
refuses when a pinned record is not the file in the repository, and when a
frozen file already exists that differs from what the code declares.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sensor_modeling.datasets.sleep_gap_plan import (
    PINNED_RECORDS,
    check_frozen_plan,
    declared_plan,
    file_sha256,
    write_plan,
)
from sensor_modeling.datasets.sleep_gap_summary import PLAN_PAGE, render_plan

PLAN_PATH = Path("artifacts/sleep_gap/sleep_gap_plan.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    plan = declared_plan()
    for name, digest in PINNED_RECORDS.items():
        if file_sha256(Path(name)) != digest:
            raise SystemExit(f"{name} is not the file the plan pins")
    if PLAN_PATH.exists():
        check_frozen_plan(plan, PLAN_PATH)
        print(f"{PLAN_PATH} is already frozen and is what the code declares")
    else:
        digest = write_plan(plan, PLAN_PATH)
        print(f"frozen {PLAN_PATH}, {digest}")
    frozen = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    page = args.docs_dir / PLAN_PAGE
    page.write_text(render_plan(frozen), encoding="utf-8", newline="\n")
    print(f"written {page}")


if __name__ == "__main__":
    main()
