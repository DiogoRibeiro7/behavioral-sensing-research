"""Build the page of the measurement on synthetic days, and its figures.

Usage::

    python scripts/render_threshold_calibration.py <records_dir> <docs_dir>

``records_dir`` holds ``threshold-null.json``, the record of the measurement
on synthetic days, written by ``scripts/run_threshold_null.py``. The script
writes ``<docs_dir>/THRESHOLD_CALIBRATION_NULL.md`` and its figures under
``<docs_dir>/figures``. It runs nothing: everything on the page comes from the
record.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sensor_modeling.datasets.threshold_calibration_figures import draw_null_figures
from sensor_modeling.datasets.threshold_null import EXPERIMENT as NULL_EXPERIMENT
from sensor_modeling.datasets.threshold_null_summary import (
    FIGURE_DIR,
    NULL_PAGE,
    render_page,
)
from sensor_modeling.evaluation import load_record


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("docs_dir", type=Path)
    args = parser.parse_args()

    record = load_record(args.records_dir / f"{NULL_EXPERIMENT}.json")
    page = args.docs_dir / NULL_PAGE
    page.write_text(render_page(record), encoding="utf-8", newline="\n")
    figures = draw_null_figures(record, args.docs_dir / FIGURE_DIR)
    print(f"written {page} and {len(figures)} figures")


if __name__ == "__main__":
    main()
