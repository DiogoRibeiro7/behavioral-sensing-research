"""Draw the Phase 3.3 correlated-silence figures from a written record.

The figures read nothing but the record, so they can be regenerated from the
published artifact at any time. Each SVG carries the SHA-256 of the data it
plots.

Usage::

    python scripts/plot_phase3_silence.py <record.json> <output_dir>

For the published result::

    python scripts/plot_phase3_silence.py \\
        artifacts/phase3/phase3-correlated-silence.json docs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sensor_modeling.datasets.silence_figures import draw_figures
from sensor_modeling.evaluation import load_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("record", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for path in draw_figures(load_record(args.record), args.output_dir).values():
        print(f"written {path}")


if __name__ == "__main__":
    main()
