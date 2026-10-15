"""Describe what the calibrated reference does on the TIHM dataset.

This is the descriptive part of
``artifacts/threshold_calibration/threshold_calibration_protocol.json``. The
calibrated reference was built from what the TIHM homes showed, so nothing
here is a test. The script refuses to run unless both frozen protocols equal
what the code declares, the records the protocol names are the ones in the
repository, the mapping file matches its frozen digest, every dataset file
matches its pinned digest and the working tree is clean. It compares the
source files, the libraries' versions and the default settings the frozen
file recorded with those it is about to use, and refuses if any has changed
unless it is told why. It reports nothing unless the replay under the default
reference returns each run's verdicts and alerts, the replay under the
calibrated reference returns, in the homes the protocol names, what the
pipeline run with that reference concluded, and the runs are the published
record's homes with its counts, home by home with the silent-home rule off.

Usage::

    python scripts/run_threshold_calibration_tihm.py <tihm_dataset_dir> <output_dir> [--jobs N]

``tihm_dataset_dir`` is the ``Dataset`` folder of the extracted archive of
Zenodo record 7622128, which is not redistributed. The script writes the
record, ``<output_dir>/threshold-calibration-tihm.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sensor_modeling.datasets import tihm_protocol
from sensor_modeling.datasets.threshold_calibration_experiment import describe_tihm
from sensor_modeling.datasets.threshold_calibration_protocol import (
    TIHM_RECORDS,
    check_frozen_protocol,
    check_inputs,
    code_changes,
    declared_protocol,
    file_sha256,
)
from sensor_modeling.evaluation import InputArtifact, load_record
from sensor_modeling.evaluation.provenance import git_state
from sensor_modeling.external import OntologyMapping
from sensor_modeling.external.tihm import FILES, READ_FILES, TihmAdapter, check_files

PROTOCOL_PATH = Path(
    "artifacts/threshold_calibration/threshold_calibration_protocol.json"
)
TIHM_PROTOCOL_PATH = Path("artifacts/tihm/alert_burden_protocol.json")
MAPPING_PATH = Path("artifacts/tihm/tihm_mapping.json")
PUBLISHED_PATH = Path("artifacts/tihm/tihm-alert-burden.json")
SOURCE = "https://doi.org/10.5281/zenodo.7622128"


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
    parser.add_argument(
        "--code-changed",
        default="",
        metavar="WHY",
        help="run although code recorded at the freeze has changed, and say why; "
        "the record carries what changed and this reason",
    )
    args = parser.parse_args()

    protocol = declared_protocol()
    protocol_sha256 = check_frozen_protocol(protocol, PROTOCOL_PATH)
    check_inputs(protocol)
    changed = code_changes(PROTOCOL_PATH)
    if any(changed.values()) and not args.code_changed:
        raise SystemExit(
            "code the protocol recorded at its freeze has changed: "
            f"{changed}; run with --code-changed and the reason, which the "
            "record will carry"
        )
    alert_burden = tihm_protocol.declared_protocol()
    alert_burden_sha256 = tihm_protocol.check_frozen_protocol(
        alert_burden, TIHM_PROTOCOL_PATH
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
            "frozen alert-burden protocol, whose run the conditions are replayed over",
            str(TIHM_PROTOCOL_PATH),
        ),
        InputArtifact(
            MAPPING_PATH.name,
            file_sha256(MAPPING_PATH),
            "frozen mapping",
            str(MAPPING_PATH),
        ),
        *(
            InputArtifact(
                Path(name).name,
                file_sha256(Path(name)),
                "published record, which the runs must reproduce",
                name,
            )
            for name in TIHM_RECORDS
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
        code_changes=changed,
        code_note=args.code_changed,
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    print(f"written {result.path}")


if __name__ == "__main__":
    main()
