"""Phase 1: the recoverable-information gap on the development panel.

This analysis is exploratory. The development homes have been inspected
throughout earlier work, so nothing here is confirmatory evidence. It separates
what added information is worth to a fixed model from what a model's
formulation is worth on fixed information, with every comparison paired by
household.

Protocol, fixed in this file and in ``sensor_modeling.datasets.recoverable_gap``
before any household was scored:

- **Homes and folds.** ``artifacts/phase1/household_splits.json``: the 20
  single-resident development homes and the two frozen Phase 1 folds. Each
  recording is verified against its recorded SHA-256.
- **Grid.** The America/Los_Angeles time zone and a 5-minute step.
- **Sets.** The four nested information sets, ``I0`` to ``I3``.
- **Models.**
  - a state-frequency reference;
  - the gradient-boosted diagnostic;
  - regularised multinomial logistic regression;
  - the generative filter's model restricted to ``I0`` and ``I2``.

  All use fixed settings and seed 0.
- **Production filter.** It is run as deployed, as a reference matched to no
  set.
- **Statistics.** Balanced accuracy, log loss, Brier score and calibration
  error, each with 10,000 household resamples and 95% percentile intervals.

Usage::

    python scripts/run_phase1_recoverable_gap.py <archive_root> <output_dir>

``archive_root`` is the extracted ``labeled_data.zip`` of Zenodo record
15708568 (CC-BY-4.0). The script writes the record,
``<output_dir>/phase1-recoverable-information-gap.json``, and a Markdown summary
generated from the written record, with the same name and a ``.md`` suffix.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import read_casas_hh
from sensor_modeling.datasets.gap_summary import render_summary
from sensor_modeling.datasets.recoverable_gap import (
    GapProtocol,
    load_frozen_splits,
    run_recoverable_gap,
)
from sensor_modeling.evaluation import InputArtifact, load_record

SPLITS_PATH = Path("artifacts/phase1/household_splits.json")
TIMEZONE = "America/Los_Angeles"
SOURCE = "zenodo:15708568 labeled_data.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate(root: Path, homes: dict[str, dict[str, str]]) -> dict[str, Path]:
    """Find every recording under *root* and verify it against its frozen digest."""
    paths: dict[str, Path] = {}
    for home, entry in sorted(homes.items()):
        matches = sorted(p for p in root.rglob(entry["filename"]) if p.is_file())
        if len(matches) != 1:
            raise SystemExit(f"expected one {entry['filename']}, found {len(matches)}")
        if _sha256(matches[0]) != entry["sha256"]:
            raise SystemExit(f"{home}: recording does not match its frozen digest")
        paths[home] = matches[0]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    splits = load_frozen_splits(SPLITS_PATH)
    homes = {home: dict(entry) for home, entry in splits.homes.items()}
    paths = locate(args.archive_root, homes)
    zone = ZoneInfo(TIMEZONE)
    recordings = {
        home: read_casas_hh(path, timezone=zone) for home, path in paths.items()
    }
    inputs = [
        InputArtifact(
            SPLITS_PATH.name, splits.sha256, "frozen household splits", str(SPLITS_PATH)
        ),
        *(
            InputArtifact(
                homes[home]["filename"], homes[home]["sha256"], "recording", SOURCE
            )
            for home in sorted(paths)
        ),
    ]
    result = run_recoverable_gap(
        recordings,
        GapProtocol(splits.folds),
        data_source="casas-hh",
        inputs=inputs,
        output_dir=args.output_dir,
    )
    assert result.path is not None
    summary = result.path.with_suffix(".md")
    summary.write_text(
        render_summary(load_record(result.path)), encoding="utf-8", newline="\n"
    )
    print(f"written {result.path} and {summary}")


if __name__ == "__main__":
    main()
