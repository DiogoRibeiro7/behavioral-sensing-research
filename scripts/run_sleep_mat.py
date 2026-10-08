"""Set the pipeline's hours of sleep beside the TIHM sleep mat.

This runs ``artifacts/sleep_mat/sleep_mat_protocol.json``. The script refuses
to run unless the frozen protocols equal what the code declares, the mapping
matches its frozen digest, every dataset file matches its pinned digest and the
working tree is clean. It reports nothing on TIHM unless the run with the
silent-home rule off gives, for every mat home, the monitored days and the
behavioural alerts of the published alert-burden record.

Usage::

    python scripts/run_sleep_mat.py <tihm_dataset_dir> <output_dir> [--jobs N]

``tihm_dataset_dir`` is the ``Dataset`` folder of the extracted archive of
Zenodo record 7622128, which is not redistributed. The script writes
``<output_dir>/sleep-mat.json`` and ``<output_dir>/sleep-mat-simulated.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

from sensor_modeling.datasets import tihm_protocol
from sensor_modeling.datasets.sleep_mat_experiment import (
    read_mat,
    run_mat_homes,
    run_simulated_homes,
    simulated_record,
    tihm_record,
)
from sensor_modeling.datasets.sleep_mat_protocol import (
    MAT_FILE,
    PLANNING_RECORD,
    check_frozen_protocol,
    declared_protocol,
)
from sensor_modeling.evaluation import InputArtifact, load_record
from sensor_modeling.evaluation.provenance import git_state
from sensor_modeling.external import OntologyMapping
from sensor_modeling.external.tihm import FILES, READ_FILES, TihmAdapter, check_files

PROTOCOL_PATH = Path("artifacts/sleep_mat/sleep_mat_protocol.json")
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
        help="run from a modified tree; the records will say so",
    )
    args = parser.parse_args()

    protocol = declared_protocol()
    protocol_sha256 = check_frozen_protocol(protocol, PROTOCOL_PATH)
    planning = Path(PLANNING_RECORD)
    if _text_sha256(planning) != protocol.planning_record_sha256:
        raise SystemExit(f"{planning} is not the planning record the protocol pins")
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
            "the working tree is not a clean commit, so the records could not be "
            "traced to the code that made them; commit first, or pass --allow-dirty"
        )

    published = load_record(PUBLISHED_PATH)["results"]["households"]
    inputs = [
        InputArtifact(
            PROTOCOL_PATH.name, protocol_sha256, "frozen protocol", str(PROTOCOL_PATH)
        ),
        InputArtifact(
            planning.name,
            protocol.planning_record_sha256,
            "planning record the protocol's interval was chosen from",
            str(planning),
        ),
        InputArtifact(
            TIHM_PROTOCOL_PATH.name,
            alert_burden_sha256,
            "frozen alert-burden protocol, whose run is set beside the mat",
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
            "published alert-burden record, which the run with the rule off must "
            "reproduce for every mat home",
            str(PUBLISHED_PATH),
        ),
        *(
            InputArtifact(
                name,
                digest,
                (
                    "recording, read through the contract"
                    if name in READ_FILES
                    else (
                        "the sleep mat, read as the reference"
                        if name == MAT_FILE
                        else "pinned, not read"
                    )
                ),
                SOURCE,
            )
            for name, digest in FILES.items()
        ),
    ]

    mat_path = args.tihm_dataset_dir / MAT_FILE
    mats = read_mat(mat_path)
    shifted = {hours: read_mat(mat_path, hours) for hours in protocol.shifts_hours}
    began = time.monotonic()

    def progress(done: int, runs: int) -> None:
        minutes = (time.monotonic() - began) / 60.0
        print(f"{done}/{runs} runs, {minutes:.1f} min", file=sys.stderr, flush=True)

    runs = run_mat_homes(
        TihmAdapter(args.tihm_dataset_dir),
        sorted(mats),
        protocol,
        alert_burden,
        jobs=args.jobs,
        progress=progress,
    )
    record = tihm_record(
        runs,
        mats,
        shifted,
        protocol,
        alert_burden,
        published,
        protocol_sha256=protocol_sha256,
        inputs=inputs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = record.write(args.output_dir / "sleep-mat.json")
    print(f"written {path}", file=sys.stderr)

    homes = run_simulated_homes(protocol, jobs=args.jobs, progress=progress)
    simulated = simulated_record(
        homes,
        protocol,
        protocol_sha256=protocol_sha256,
        inputs=inputs[:2],
    )
    path = simulated.write(args.output_dir / "sleep-mat-simulated.json")
    print(f"written {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
