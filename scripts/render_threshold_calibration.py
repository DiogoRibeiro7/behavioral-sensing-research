"""Build the threshold-calibration pages and their figures from the records.

Usage::

    python scripts/render_threshold_calibration.py <records_dir> <docs_dir>

``records_dir`` may hold ``threshold-null.json``, the record of the
measurement on synthetic days, written by ``scripts/run_threshold_null.py``;
``threshold-calibration.json``, the record of the protocol's test on simulated
homes, written by ``scripts/run_threshold_calibration.py``; and
``threshold-calibration-tihm.json``, the record of the description on TIHM,
written by ``scripts/run_threshold_calibration_tihm.py``. The script writes
the page of each record it finds, and its figures, under ``docs_dir``. It runs
nothing: everything on a page comes from its records.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sensor_modeling.datasets.threshold_calibration_figures import (
    draw_figures,
    draw_null_figures,
)
from sensor_modeling.datasets.threshold_calibration_protocol import (
    EXPERIMENT,
    TIHM_EXPERIMENT,
)
from sensor_modeling.datasets.threshold_calibration_summary import (
    RESULTS_PAGE,
    render_page,
)
from sensor_modeling.datasets.threshold_null import EXPERIMENT as NULL_EXPERIMENT
from sensor_modeling.datasets.threshold_null_summary import FIGURE_DIR, NULL_PAGE
from sensor_modeling.datasets.threshold_null_summary import (
    render_page as render_null_page,
)
from sensor_modeling.evaluation import load_record


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("docs_dir", type=Path)
    args = parser.parse_args()

    measured = args.records_dir / f"{NULL_EXPERIMENT}.json"
    if measured.exists():
        record = load_record(measured)
        page = args.docs_dir / NULL_PAGE
        page.write_text(render_null_page(record), encoding="utf-8", newline="\n")
        figures = draw_null_figures(record, args.docs_dir / FIGURE_DIR)
        print(f"written {page} and {len(figures)} figures")

    tested = args.records_dir / f"{EXPERIMENT}.json"
    if tested.exists():
        record = load_record(tested)
        described = args.records_dir / f"{TIHM_EXPERIMENT}.json"
        tihm = load_record(described) if described.exists() else None
        page = args.docs_dir / RESULTS_PAGE
        page.write_text(render_page(record, tihm), encoding="utf-8", newline="\n")
        figures = draw_figures(record, args.docs_dir / FIGURE_DIR, tihm)
        print(f"written {page} and {len(figures)} figures")


if __name__ == "__main__":
    main()
