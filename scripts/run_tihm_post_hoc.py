"""Describe the published TIHM alert-burden run, after its results were read.

Nothing here is part of the frozen protocol. The script runs every household
again, refuses to go on unless scoring those runs gives the results of the
published record, ``artifacts/tihm/tihm-alert-burden.json``, and then writes
the post hoc descriptions as a record of their own.

Usage::

    python scripts/run_tihm_post_hoc.py <tihm_dataset_dir> <output_dir>

``tihm_dataset_dir`` is the ``Dataset`` folder of the extracted archive of
Zenodo record 7622128. The script writes
``<output_dir>/tihm-alert-burden-post-hoc.json``, the results page built from
both records, ``<output_dir>/TIHM_ALERT_BURDEN_RESULTS.md``, and every figure
under ``<output_dir>/figures``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from sensor_modeling.datasets.tihm_experiment import read_physiology
from sensor_modeling.datasets.tihm_figures import draw_figures
from sensor_modeling.datasets.tihm_post_hoc import PostHocConfig, run_post_hoc
from sensor_modeling.datasets.tihm_protocol import (
    check_frozen_protocol,
    declared_protocol,
)
from sensor_modeling.datasets.tihm_summary import RESULTS_PAGE, render_page
from sensor_modeling.evaluation import InputArtifact, load_record
from sensor_modeling.external import OntologyMapping
from sensor_modeling.external.tihm import FILES, READ_FILES, TihmAdapter, check_files

PROTOCOL_PATH = Path("artifacts/tihm/alert_burden_protocol.json")
MAPPING_PATH = Path("artifacts/tihm/tihm_mapping.json")
RECORD_PATH = Path("artifacts/tihm/tihm-alert-burden.json")
SOURCE = "https://doi.org/10.5281/zenodo.7622128"


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tihm_dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    protocol = declared_protocol()
    protocol_sha256 = check_frozen_protocol(protocol, PROTOCOL_PATH)
    OntologyMapping.read(MAPPING_PATH, protocol.mapping.sha256())
    problems = check_files(args.tihm_dataset_dir)
    if problems:
        raise SystemExit(
            "; ".join(f"{name} {problem}" for name, problem in sorted(problems.items()))
        )
    published = load_record(RECORD_PATH)

    inputs = [
        InputArtifact(
            RECORD_PATH.name,
            _text_sha256(RECORD_PATH),
            "the record of the protocol's run, which this describes",
            str(RECORD_PATH),
        ),
        InputArtifact(
            PROTOCOL_PATH.name, protocol_sha256, "frozen protocol", str(PROTOCOL_PATH)
        ),
        InputArtifact(
            MAPPING_PATH.name,
            _text_sha256(MAPPING_PATH),
            "frozen mapping",
            str(MAPPING_PATH),
        ),
        *(
            InputArtifact(
                name,
                digest,
                (
                    "recording, read through the contract"
                    if name in READ_FILES
                    else (
                        "readings, read only to check the published results"
                        if name == "Physiology.csv"
                        else "pinned, not read"
                    )
                ),
                SOURCE,
            )
            for name, digest in FILES.items()
        ),
    ]
    result = run_post_hoc(
        TihmAdapter(args.tihm_dataset_dir),
        read_physiology(args.tihm_dataset_dir / "Physiology.csv"),
        protocol,
        published,
        config=PostHocConfig(),
        inputs=inputs,
        output_dir=args.output_dir,
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    written = load_record(result.path)
    page = args.output_dir / RESULTS_PAGE
    page.write_text(render_page(published, written), encoding="utf-8", newline="\n")
    figures = draw_figures(published, args.output_dir / "figures", written)
    print(f"written {result.path}, {page} and {len(figures)} figures")


if __name__ == "__main__":
    main()
