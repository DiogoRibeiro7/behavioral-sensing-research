"""Phase 3.3 follow-up diagnostic: posterior predictive checks of the fitted hurdle model.

It locates where the fitted hurdle channel observation model is misspecified,
before another count distribution is introduced. It compares observed channel
counts with the model's predictive distribution for each household, state and
channel with enough data, against the household's own hurdle fit and against
the fold's population fit. It changes no model and runs no inference.

Everything is fixed in ``artifacts/phase3/predictive_protocol.json``,
committed before any household was examined:

- the folds;
- the references;
- the statistics and the runs;
- the replicates and their band;
- the sufficiency rules;
- the household bootstrap;
- the seed;
- the rules that read the result and route it to a next model family.

The script refuses to run if the code's protocol differs from the frozen file,
and verifies every recording against its frozen SHA-256.

Usage::

    python scripts/run_phase3_predictive_checks.py <archive_root> <output_dir>

``archive_root`` is the extracted ``labeled_data.zip`` of Zenodo record
15708568 (CC-BY-4.0). The script writes the record,
``<output_dir>/phase3-hurdle-predictive-checks.json``, a Markdown summary
generated from the written record, with the same name and a ``.md`` suffix,
and the figures drawn from it.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import read_casas_hh
from sensor_modeling.datasets.predictive_checks import (
    check_frozen_protocol,
    declared_protocol,
    run_predictive_checks,
)
from sensor_modeling.datasets.predictive_figures import draw_figures
from sensor_modeling.datasets.predictive_summary import render_summary
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.evaluation import InputArtifact, load_record

SPLITS_PATH = Path("artifacts/phase1/household_splits.json")
PROTOCOL_PATH = Path("artifacts/phase3/predictive_protocol.json")
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
    result = run_predictive_checks(
        recordings,
        protocol,
        data_source="casas-hh",
        inputs=inputs,
        output_dir=args.output_dir,
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    written = load_record(result.path)
    summary = result.path.with_suffix(".md")
    summary.write_text(render_summary(written), encoding="utf-8", newline="\n")
    figures = draw_figures(written, args.output_dir)
    print(f"written {result.path}, {summary} and {len(figures)} figures")


if __name__ == "__main__":
    main()
