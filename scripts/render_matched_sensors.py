"""Render the matched-sensor results page and its figures from the record.

Usage::

    python scripts/render_matched_sensors.py <records_dir> <docs_dir>

``records_dir`` holds ``matched-sensors.json``. The page is written to
``<docs_dir>/MATCHED_SENSORS_RESULTS.md`` and the figures to
``<docs_dir>/figures``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sensor_modeling.datasets.matched_sensors_figures import draw_figures
from sensor_modeling.datasets.matched_sensors_summary import (
    FIGURE_DIR,
    RESULTS_PAGE,
    render_page,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("docs_dir", type=Path)
    args = parser.parse_args()
    record = json.loads(
        (args.records_dir / "matched-sensors.json").read_text(encoding="utf-8")
    )
    page = args.docs_dir / RESULTS_PAGE
    page.write_text(render_page(record), encoding="utf-8", newline="\n")
    paths = draw_figures(record, args.docs_dir / FIGURE_DIR)
    print(f"written {page} and {len(paths)} figures")


if __name__ == "__main__":
    main()
