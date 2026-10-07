"""Run the frozen TIHM alert-burden protocol on the extracted TIHM dataset.

The protocol, ``artifacts/tihm/alert_burden_protocol.json``, is exploratory and
was committed before any pipeline output was compared with a label. The script
refuses to run unless the code's protocol equals the frozen file, the mapping
file matches its frozen digest, and every dataset file matches its pinned
digest.

Usage::

    python scripts/run_tihm_alert_burden.py <tihm_dataset_dir> <output_dir>

``tihm_dataset_dir`` is the ``Dataset`` folder of the extracted archive of
Zenodo record 7622128. The script writes the record,
``<output_dir>/tihm-alert-burden.json``, the results page built from it,
``<output_dir>/TIHM_ALERT_BURDEN_RESULTS.md``, and its figures under
``<output_dir>/figures``. The descriptions made after the run was read are a
separate step, ``scripts/run_tihm_post_hoc.py``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from sensor_modeling.datasets.tihm_experiment import read_physiology, run_tihm
from sensor_modeling.datasets.tihm_figures import draw_figures
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
        *(
            InputArtifact(
                name,
                digest,
                (
                    "recording, read through the contract"
                    if name in READ_FILES
                    else (
                        "readings, read only to describe the labels"
                        if name == "Physiology.csv"
                        else "pinned, not read"
                    )
                ),
                SOURCE,
            )
            for name, digest in FILES.items()
        ),
    ]
    result = run_tihm(
        TihmAdapter(args.tihm_dataset_dir),
        read_physiology(args.tihm_dataset_dir / "Physiology.csv"),
        protocol,
        protocol_sha256=protocol_sha256,
        inputs=inputs,
        output_dir=args.output_dir,
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    written = load_record(result.path)
    page = args.output_dir / RESULTS_PAGE
    page.write_text(render_page(written), encoding="utf-8", newline="\n")
    figures = draw_figures(written, args.output_dir / "figures")
    print(f"written {result.path}, {page} and {len(figures)} figures")


if __name__ == "__main__":
    main()
