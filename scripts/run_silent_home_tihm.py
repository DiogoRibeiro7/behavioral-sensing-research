"""Describe what the silent-home rule does on the TIHM dataset.

This is the descriptive part of ``artifacts/silent_home/silent_home_protocol.json``.
The rule was designed from the TIHM homes, so nothing here is a test. The
script refuses to run unless both frozen protocols equal what the code
declares, the mapping file matches its frozen digest, every dataset file
matches its pinned digest and the working tree is clean. It reports nothing
unless the run with the rule off gives, home by home, the monitored days and
the behavioural alerts of the published alert-burden record.

Usage::

    python scripts/run_silent_home_tihm.py <tihm_dataset_dir> <output_dir> [--jobs N]

``tihm_dataset_dir`` is the ``Dataset`` folder of the extracted archive of
Zenodo record 7622128, which is not redistributed. The script writes the
record, ``<output_dir>/silent-home-tihm.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

from sensor_modeling.datasets import tihm_protocol
from sensor_modeling.datasets.silent_home_experiment import describe_tihm
from sensor_modeling.datasets.silent_home_protocol import (
    check_frozen_protocol,
    declared_protocol,
)
from sensor_modeling.evaluation import InputArtifact, load_record
from sensor_modeling.evaluation.provenance import git_state
from sensor_modeling.external import OntologyMapping
from sensor_modeling.external.tihm import FILES, READ_FILES, TihmAdapter, check_files

PROTOCOL_PATH = Path("artifacts/silent_home/silent_home_protocol.json")
TIHM_PROTOCOL_PATH = Path("artifacts/tihm/alert_burden_protocol.json")
MAPPING_PATH = Path("artifacts/tihm/tihm_mapping.json")
PUBLISHED_PATH = Path("artifacts/tihm/tihm-alert-burden.json")
SOURCE = "https://doi.org/10.5281/zenodo.7622128"


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("tihm_dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--jobs", type=int, default=1, help="homes run at once; the result is the same"
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run from a modified tree; the record will say so",
    )
    args = parser.parse_args()

    protocol = declared_protocol()
    protocol_sha256 = check_frozen_protocol(protocol, PROTOCOL_PATH)
    alert_burden = tihm_protocol.declared_protocol()
    alert_burden_sha256 = tihm_protocol.check_frozen_protocol(
        alert_burden, TIHM_PROTOCOL_PATH
    )
    if alert_burden.step_minutes != protocol.tihm_step_minutes:
        raise SystemExit(
            "the two protocols do not agree on the pipeline's step on TIHM: "
            f"{alert_burden.step_minutes} and {protocol.tihm_step_minutes} minutes"
        )
    OntologyMapping.read(MAPPING_PATH, alert_burden.mapping.sha256())
    problems = check_files(args.tihm_dataset_dir)
    if problems:
        raise SystemExit(
            "; ".join(f"{name} {problem}" for name, problem in sorted(problems.items()))
        )
    if git_state().get("git_dirty") != "false" and not args.allow_dirty:
        raise SystemExit(
            "the working tree is not a clean commit, so the record could not be "
            "traced to the code that made it; commit first, or pass --allow-dirty"
        )
    published = load_record(PUBLISHED_PATH)

    inputs = [
        InputArtifact(
            PROTOCOL_PATH.name, protocol_sha256, "frozen protocol", str(PROTOCOL_PATH)
        ),
        InputArtifact(
            TIHM_PROTOCOL_PATH.name,
            alert_burden_sha256,
            "frozen alert-burden protocol, whose run the rule is added to",
            str(TIHM_PROTOCOL_PATH),
        ),
        InputArtifact(
            MAPPING_PATH.name,
            _text_sha256(MAPPING_PATH),
            "frozen mapping",
            str(MAPPING_PATH),
        ),
        InputArtifact(
            PUBLISHED_PATH.name,
            _text_sha256(PUBLISHED_PATH),
            "published alert-burden record, which the run with the rule off "
            "must reproduce home by home",
            str(PUBLISHED_PATH),
        ),
        *(
            InputArtifact(
                name,
                digest,
                (
                    "recording, read through the contract"
                    if name in READ_FILES
                    else "pinned, not read"
                ),
                SOURCE,
            )
            for name, digest in FILES.items()
        ),
    ]

    began = time.monotonic()

    def progress(done: int, runs: int) -> None:
        if done % 14 == 0 or done == runs:
            minutes = (time.monotonic() - began) / 60.0
            print(f"{done}/{runs} runs, {minutes:.1f} min", file=sys.stderr, flush=True)

    result = describe_tihm(
        TihmAdapter(args.tihm_dataset_dir),
        protocol,
        alert_burden,
        protocol_sha256=protocol_sha256,
        inputs=inputs,
        output_dir=args.output_dir,
        jobs=args.jobs,
        progress=progress,
        published_homes=published["results"]["households"],
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    print(f"written {result.path}")


if __name__ == "__main__":
    main()
