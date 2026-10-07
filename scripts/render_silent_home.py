"""Build the silent-home results page and its figures from the records.

Usage::

    python scripts/render_silent_home.py <records_dir> <docs_dir>

``records_dir`` holds ``silent-home.json``, the record of the protocol's test
on simulated homes, written by ``scripts/run_silent_home.py``, and, if the
description was made, ``silent-home-tihm.json``, written by
``scripts/run_silent_home_tihm.py``. The script writes
``<docs_dir>/SILENT_HOME_RESULTS.md`` and its figures under
``<docs_dir>/figures``. It runs no home: everything on the page comes from the
records, and from the frozen protocol for the windows their alerts fall in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sensor_modeling.datasets.silent_home_figures import draw_figures
from sensor_modeling.datasets.silent_home_protocol import (
    EXPERIMENT,
    TIHM_EXPERIMENT,
    check_frozen_protocol,
    declared_protocol,
)
from sensor_modeling.datasets.silent_home_summary import (
    FIGURE_DIR,
    RESULTS_PAGE,
    render_page,
)
from sensor_modeling.evaluation import load_record

PROTOCOL_PATH = Path("artifacts/silent_home/silent_home_protocol.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("docs_dir", type=Path)
    args = parser.parse_args()

    protocol = declared_protocol()
    check_frozen_protocol(protocol, PROTOCOL_PATH)
    record = load_record(args.records_dir / f"{EXPERIMENT}.json")
    described = args.records_dir / f"{TIHM_EXPERIMENT}.json"
    tihm = load_record(described) if described.exists() else None

    page = args.docs_dir / RESULTS_PAGE
    page.write_text(render_page(record, tihm, protocol), encoding="utf-8", newline="\n")
    figures = draw_figures(record, args.docs_dir / FIGURE_DIR, tihm)
    print(f"written {page} and {len(figures)} figures")


if __name__ == "__main__":
    main()
