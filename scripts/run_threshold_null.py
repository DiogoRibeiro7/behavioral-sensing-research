"""Measure the baseline's thresholds on synthetic days with nothing in them.

Independent Gaussian days are given to the personal baseline directly, with
the default reference and with the calibrated one. No home is simulated. The
script refuses to run from a modified tree, so that the record it writes can
be traced to one commit.

Usage::

    python scripts/run_threshold_null.py <output_dir> [--jobs N]

The script writes the record, ``<output_dir>/threshold-null.json``. The page
and its figures are built from the record by
``scripts/render_threshold_calibration.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sensor_modeling.datasets.threshold_null import ThresholdNull, run_threshold_null
from sensor_modeling.evaluation.provenance import git_state


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--jobs", type=int, default=1, help="blocks run at once; the result is the same"
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

    def progress(stage: str, done: int, blocks: int) -> None:
        if done % 5 == 0 or done == blocks:
            minutes = (time.monotonic() - began) / 60.0
            print(
                f"{stage}: {done}/{blocks} blocks, {minutes:.1f} min",
                file=sys.stderr,
                flush=True,
            )

    result = run_threshold_null(
        ThresholdNull(), output_dir=args.output_dir, jobs=args.jobs, progress=progress
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    print(f"written {result.path}")


if __name__ == "__main__":
    main()
