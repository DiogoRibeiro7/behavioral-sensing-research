"""Render the sleep-mat results page and its figures from the records.

Usage::

    python scripts/render_sleep_mat.py <records_dir> <docs_dir>

``records_dir`` holds ``sleep-mat.json`` and ``sleep-mat-simulated.json``. The
page is written to ``<docs_dir>/SLEEP_MAT_RESULTS.md`` and the figures to
``<docs_dir>/figures``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sensor_modeling.datasets.sleep_mat_figures import draw_figures
from sensor_modeling.datasets.sleep_mat_summary import (
    FIGURE_DIR,
    RESULTS_PAGE,
    render_page,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("docs_dir", type=Path)
    args = parser.parse_args()
    record = json.loads((args.records_dir / "sleep-mat.json").read_text("utf-8"))
    simulated = json.loads(
        (args.records_dir / "sleep-mat-simulated.json").read_text("utf-8")
    )
    page = args.docs_dir / RESULTS_PAGE
    page.write_text(render_page(record, simulated), encoding="utf-8", newline="\n")
    paths = draw_figures(record, args.docs_dir / FIGURE_DIR)
    print(f"written {page} and {len(paths)} figures")


if __name__ == "__main__":
    main()
