"""Phase 3.3 follow-up: the pre-specified fitted-rates experiment.

The correlated-silence diagnostic concluded 'weakened', which routes the work
here: fit each evidence channel's silence probability and activity rate on
training households, keep the channels conditionally independent, and compare
with the declared rates on identical information.

Everything is fixed in ``artifacts/phase3/fitted_rates_protocol.json``,
committed before any household was scored:

- the folds;
- the information sets and the recursion;
- the channel model families and their fitting;
- the pseudo-windows of the shrinkage;
- the time prior;
- the metrics;
- the bootstrap;
- the seed;
- the estimands and mechanisms;
- the criteria.

The script refuses to run if the code's protocol differs from the frozen file,
and verifies every recording against its frozen SHA-256.

Usage::

    python scripts/run_phase3_fitted_rates.py <archive_root> <output_dir>

``archive_root`` is the extracted ``labeled_data.zip`` of Zenodo record
15708568 (CC-BY-4.0). The script writes the record,
``<output_dir>/phase3-fitted-rates.json``, and a Markdown summary generated
from the written record, with the same name and a ``.md`` suffix.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import read_casas_hh
from sensor_modeling.datasets.rates_experiment import (
    check_frozen_protocol,
    declared_protocol,
    run_fitted_rates,
)
from sensor_modeling.datasets.rates_summary import render_summary
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.evaluation import InputArtifact, load_record

SPLITS_PATH = Path("artifacts/phase1/household_splits.json")
PROTOCOL_PATH = Path("artifacts/phase3/fitted_rates_protocol.json")
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
    protocol = declared_protocol(splits)
    protocol_sha256 = check_frozen_protocol(protocol, PROTOCOL_PATH)
    homes = {home: dict(entry) for home, entry in splits.homes.items()}
    paths = locate(args.archive_root, homes)
    zone = ZoneInfo(TIMEZONE)
    recordings = {
        home: read_casas_hh(path, timezone=zone) for home, path in paths.items()
    }
    inputs = [
        InputArtifact(
            PROTOCOL_PATH.name, protocol_sha256, "frozen protocol", str(PROTOCOL_PATH)
        ),
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
    result = run_fitted_rates(
        recordings,
        protocol,
        data_source="casas-hh",
        inputs=inputs,
        output_dir=args.output_dir,
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    summary = result.path.with_suffix(".md")
    summary.write_text(
        render_summary(load_record(result.path)), encoding="utf-8", newline="\n"
    )
    print(f"written {result.path} and {summary}")


if __name__ == "__main__":
    main()
