"""Phase 5: run the frozen external-generalisation protocol on the UCI ADL Binary data.

The protocol, ``artifacts/phase5/external_protocol.json``, was frozen in commit
``a863bff`` before any external scoring. The script refuses to run unless the
code's protocol equals the frozen file, the mapping file matches its frozen
digest, every external and CASAS recording matches its frozen digest, and the
CASAS populations reproduce the digests the protocol pins.

Usage::

    python scripts/run_phase5_external.py <casas_archive_root> <uci_archive_root> <output_dir>

``casas_archive_root`` is the extracted ``labeled_data.zip`` of Zenodo record
15708568; ``uci_archive_root`` the extracted UCI dataset 271 archive. The script
writes the record, ``<output_dir>/phase5-external-ordonez-results.json``, the
page generated from the written record, ``<output_dir>/PHASE5_EXTERNAL_RESULTS.md``,
and its figures in ``<output_dir>/figures``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import read_casas_hh
from sensor_modeling.datasets.external_experiment import (
    development_reference,
    reproduce_populations,
    run_external,
)
from sensor_modeling.datasets.external_figures import draw_figures
from sensor_modeling.datasets.external_protocol import (
    check_frozen_protocol,
    declared_protocol,
)
from sensor_modeling.datasets.external_summary import render_page
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.evaluation import InputArtifact, load_record
from sensor_modeling.external import OntologyMapping
from sensor_modeling.external.ordonez import FILES, FOLDER, OrdonezAdapter

SPLITS_PATH = Path("artifacts/phase1/household_splits.json")
PROTOCOL_PATH = Path("artifacts/phase5/external_protocol.json")
MAPPING_PATH = Path("artifacts/phase5/ordonez_mapping.json")
DEVELOPMENT_PATH = Path("artifacts/phase4/phase4-uncertainty-diagnostics.json")
CASAS_TIMEZONE = "America/Los_Angeles"
CASAS_SOURCE = "zenodo:15708568 labeled_data.zip"
UCI_SOURCE = "https://doi.org/10.24432/C5J02M"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("casas_archive_root", type=Path)
    parser.add_argument("uci_archive_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    splits = load_frozen_splits(SPLITS_PATH)
    protocol = declared_protocol(splits)
    protocol_sha256 = check_frozen_protocol(protocol, PROTOCOL_PATH)
    OntologyMapping.read(MAPPING_PATH, protocol.mapping.sha256())

    folder = args.uci_archive_root / FOLDER
    for name, digest in FILES.items():
        if _sha256(folder / name) != digest:
            raise SystemExit(f"{name} does not match its frozen digest")

    recordings = {}
    for home in protocol.development_homes:
        entry = splits.homes[home]
        matches = sorted(p for p in args.casas_archive_root.rglob(entry["filename"]))
        if len(matches) != 1 or _sha256(matches[0]) != entry["sha256"]:
            raise SystemExit(f"{home}: recording missing or not its frozen digest")
        recordings[home] = read_casas_hh(matches[0], timezone=ZoneInfo(CASAS_TIMEZONE))
    samples = reproduce_populations(recordings, protocol)

    development = load_record(DEVELOPMENT_PATH)
    inputs = [
        InputArtifact(
            PROTOCOL_PATH.name, protocol_sha256, "frozen protocol", str(PROTOCOL_PATH)
        ),
        InputArtifact(
            MAPPING_PATH.name,
            _text_sha256(MAPPING_PATH),
            "frozen mapping",
            str(MAPPING_PATH),
        ),
        InputArtifact(
            SPLITS_PATH.name,
            splits.sha256,
            "frozen development splits",
            str(SPLITS_PATH),
        ),
        InputArtifact(
            DEVELOPMENT_PATH.name,
            _text_sha256(DEVELOPMENT_PATH),
            "the same model on the development panel, for descriptive shifts",
            str(DEVELOPMENT_PATH),
        ),
        *(
            InputArtifact(name, digest, "external recording", UCI_SOURCE)
            for name, digest in FILES.items()
        ),
        *(
            InputArtifact(
                splits.homes[h]["filename"],
                splits.homes[h]["sha256"],
                "development recording",
                CASAS_SOURCE,
            )
            for h in protocol.development_homes
        ),
    ]
    result = run_external(
        OrdonezAdapter(folder),
        samples,
        protocol,
        protocol_sha256=protocol_sha256,
        development=development_reference(development),
        data_source="uci-adl-binary-ordonez",
        inputs=inputs,
        output_dir=args.output_dir,
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    written = load_record(result.path)
    page = args.output_dir / "PHASE5_EXTERNAL_RESULTS.md"
    page.write_text(render_page(written), encoding="utf-8", newline="\n")
    figures = draw_figures(written, args.output_dir / "figures")
    print(f"written {result.path}, {page} and {len(figures)} figures")


if __name__ == "__main__":
    main()
