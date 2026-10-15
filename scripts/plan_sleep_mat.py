"""Run the sleep-mat planning trials on synthetic data.

The trials choose the protocol's interval and say how often each criterion's
readings come out when the truth is known. They read nothing from TIHM. The
script refuses to run from a modified tree unless told to, so the record can be
traced to the code that made it.

Usage::

    python scripts/plan_sleep_mat.py <output_dir>

It writes ``<output_dir>/sleep-mat-planning.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sensor_modeling.datasets.sleep_mat_planning import run_planning
from sensor_modeling.evaluation.provenance import git_state


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run from a modified tree; the record will say so",
    )
    args = parser.parse_args()
    if git_state().get("git_dirty") != "false" and not args.allow_dirty:
        raise SystemExit(
            "the working tree is not a clean commit; commit first, or pass "
            "--allow-dirty"
        )
    _, path = run_planning(output_dir=args.output_dir)
    print(f"written {path}")


if __name__ == "__main__":
    main()
