"""Try the threshold-calibration protocol's criterion on curves written down by hand.

Outcomes are drawn from operating curves that
``sensor_modeling.datasets.threshold_calibration_planning`` writes down, and
the criterion is decided on each cohort drawn as the scoring would decide it.
No home is simulated and no reference is run. The script refuses to run from
a modified tree, so that the record it writes can be traced to one commit.

Usage::

    python scripts/plan_threshold_calibration.py <output_dir> [--jobs N]

The script writes the record,
``<output_dir>/threshold-calibration-planning.json``. The protocol quotes
ranges of it and names it by its digest. Each cell is kept in
``<output_dir>/cells-<commit>`` as soon as it is finished, so a run that is
interrupted can be started again without repeating the cells already done;
that directory is working storage and is not part of the result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sensor_modeling.datasets.threshold_calibration_planning import (
    Planning,
    run_planning,
)
from sensor_modeling.evaluation.provenance import git_state


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--jobs", type=int, default=1, help="cells run at once; the result is the same"
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run from a modified tree; the record will say so",
    )
    args = parser.parse_args()

    if git_state().get("git_dirty") != "false" and not args.allow_dirty:
        raise SystemExit(
            "the working tree is not a clean commit, so the record could not be "
            "traced to the code that made it; commit first, or pass --allow-dirty"
        )
    began = time.monotonic()

    def progress(done: int, cells: int) -> None:
        minutes = (time.monotonic() - began) / 60.0
        print(f"{done}/{cells} cells, {minutes:.1f} min", file=sys.stderr, flush=True)

    commit = str(git_state().get("git_commit", "unknown"))[:12]
    result = run_planning(
        Planning(),
        output_dir=args.output_dir,
        jobs=args.jobs,
        progress=progress,
        checkpoint=args.output_dir / f"cells-{commit}",
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    print(f"written {result.path}")


if __name__ == "__main__":
    main()
