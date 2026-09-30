"""Phase 4: the pre-specified comparison of richer uncertainty diagnostics.

It compares five signals for rejecting the same predictions on the development
households: posterior confidence, entropy, model-structure disagreement,
evidence-channel disagreement and predictive mismatch, as selective-prediction
curves with household-level paired differences against confidence.

Everything is fixed in ``artifacts/phase4/uncertainty_protocol.json``,
committed before any household was scored:

- the folds and the predictions every signal ranks;
- the signals, their directions and missing policies;
- the structural ensemble and the evidence groups;
- the coverage grid and the inspection levels;
- the estimands and their minimal differences;
- the rule for difficult minority states and the rejection-bias guard;
- the bootstrap and its seed;
- the question and the decision rule.

The script refuses to run if the code's protocol differs from the frozen file,
and verifies every recording against its frozen SHA-256. No threshold is
selected.

Usage::

    python scripts/run_phase4_uncertainty.py <archive_root> <output_dir>

``archive_root`` is the extracted ``labeled_data.zip`` of Zenodo record
15708568 (CC-BY-4.0). The script writes the record,
``<output_dir>/phase4-uncertainty-diagnostics.json``, the documentation page
generated from the written record, ``<output_dir>/PHASE4_UNCERTAINTY_DIAGNOSTICS.md``,
and its figures in ``<output_dir>/figures``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import read_casas_hh
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.uncertainty_experiment import (
    check_frozen_protocol,
    declared_protocol,
    run_uncertainty,
)
from sensor_modeling.datasets.uncertainty_figures import draw_figures
from sensor_modeling.datasets.uncertainty_summary import render_page
from sensor_modeling.evaluation import InputArtifact, load_record

SPLITS_PATH = Path("artifacts/phase1/household_splits.json")
PROTOCOL_PATH = Path("artifacts/phase4/uncertainty_protocol.json")
PUBLISHED_PATH = Path("artifacts/phase3/phase3-hurdle-negative-binomial.json")
TIMEZONE = "America/Los_Angeles"
SOURCE = "zenodo:15708568 labeled_data.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(path: Path) -> str:
    """SHA-256 of a text file with its line endings normalised to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


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
        InputArtifact(
            PUBLISHED_PATH.name,
            _text_sha256(PUBLISHED_PATH),
            "published record the reference predictions must reproduce",
            str(PUBLISHED_PATH),
        ),
        *(
            InputArtifact(
                homes[home]["filename"], homes[home]["sha256"], "recording", SOURCE
            )
            for home in sorted(paths)
        ),
    ]
    result = run_uncertainty(
        recordings,
        protocol,
        data_source="casas-hh",
        inputs=inputs,
        published=load_record(PUBLISHED_PATH),
        output_dir=args.output_dir,
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    written = load_record(result.path)
    page = args.output_dir / "PHASE4_UNCERTAINTY_DIAGNOSTICS.md"
    page.write_text(render_page(written), encoding="utf-8", newline="\n")
    figures = draw_figures(written, args.output_dir / "figures")
    print(f"written {result.path}, {page} and {len(figures)} figures")


if __name__ == "__main__":
    main()
