"""Measure TIHM's event sensors and choose the matched simulator's sensor profile.

The planning reads TIHM's ``Activity.csv`` only, never a label, a sleep-mat
record or an output of the pipeline. It runs no pipeline. The script refuses to
run from a modified tree unless told to, so the record can be traced to the
code that made it.

Usage::

    python scripts/plan_matched_sensors.py <tihm_dataset_dir> <output_dir>

``tihm_dataset_dir`` is the ``Dataset`` folder of the extracted archive of
Zenodo record 7622128, which is not redistributed. The script writes
``<output_dir>/matched-sensors-planning.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sensor_modeling.datasets.matched_sensors_planning import run_planning
from sensor_modeling.evaluation.provenance import git_state


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("tihm_dataset_dir", type=Path)
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
    _, path = run_planning(args.tihm_dataset_dir, output_dir=args.output_dir)
    print(f"written {path}")


if __name__ == "__main__":
    main()
