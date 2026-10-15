"""Run the matched-sensor study: the simulator's homes with TIHM's sensors.

This runs ``artifacts/matched_sensors/matched_sensors_protocol.json``. The
script refuses to run unless the frozen protocol equals what the code declares,
the threshold-calibration protocol and the pinned records are the files in the
repository, and the working tree is clean. It reports nothing unless every run
of the standard profile raises the alerts the published threshold-calibration
record gives that home and arm.

Usage::

    python scripts/run_matched_sensors.py <output_dir> [--jobs N] [--checkpoint DIR]

It writes ``<output_dir>/matched-sensors.json``. No TIHM file is read.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

from sensor_modeling.datasets import threshold_calibration_protocol
from sensor_modeling.datasets.matched_sensors_experiment import (
    published_alerts,
    read_json,
    record_of,
    run_homes,
)
from sensor_modeling.datasets.matched_sensors_protocol import (
    PLANNING_RECORD,
    THRESHOLD_PROTOCOL,
    THRESHOLD_PROTOCOL_SHA256,
    THRESHOLD_RECORD,
    THRESHOLD_RECORD_SHA256,
    TIHM_POST_HOC_RECORD,
    TIHM_POST_HOC_RECORD_SHA256,
    check_frozen_protocol,
    code_changes,
    declared_protocol,
)
from sensor_modeling.evaluation import InputArtifact
from sensor_modeling.evaluation.provenance import git_state

PROTOCOL_PATH = Path("artifacts/matched_sensors/matched_sensors_protocol.json")


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--jobs", type=int, default=1, help="homes run at once; the result is the same"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="a directory each finished home is kept in, to resume a run",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run from a modified tree; the record will say so",
    )
    args = parser.parse_args()

    protocol = declared_protocol()
    protocol_sha256 = check_frozen_protocol(protocol, PROTOCOL_PATH)
    pinned = {
        Path(PLANNING_RECORD): protocol.planning_record_sha256,
        Path(THRESHOLD_PROTOCOL): THRESHOLD_PROTOCOL_SHA256,
        Path(THRESHOLD_RECORD): THRESHOLD_RECORD_SHA256,
        Path(TIHM_POST_HOC_RECORD): TIHM_POST_HOC_RECORD_SHA256,
    }
    for path, digest in pinned.items():
        if _text_sha256(path) != digest:
            raise SystemExit(f"{path} is not the file the protocol pins")
    threshold_calibration_protocol.check_frozen_protocol(
        threshold_calibration_protocol.declared_protocol(), Path(THRESHOLD_PROTOCOL)
    )
    changed = code_changes(PROTOCOL_PATH)
    if any(changed.values()):
        print(
            "the code differs from what the protocol was frozen against: "
            + "; ".join(f"{k}: {', '.join(v)}" for k, v in changed.items() if v),
            file=sys.stderr,
        )
    clean = git_state().get("git_dirty") == "false"
    if not clean and not args.allow_dirty:
        raise SystemExit(
            "the working tree is not a clean commit, so the record could not be "
            "traced to the code that made it; commit first, or pass --allow-dirty"
        )

    planning = read_json(Path(PLANNING_RECORD))
    published = published_alerts(read_json(Path(THRESHOLD_RECORD)))
    inputs = [
        InputArtifact(
            PROTOCOL_PATH.name, protocol_sha256, "frozen protocol", str(PROTOCOL_PATH)
        ),
        InputArtifact(
            Path(PLANNING_RECORD).name,
            protocol.planning_record_sha256,
            "planning record the matched profile was chosen from",
            PLANNING_RECORD,
        ),
        InputArtifact(
            Path(THRESHOLD_PROTOCOL).name,
            THRESHOLD_PROTOCOL_SHA256,
            "frozen threshold-calibration protocol, whose homes, arms and "
            "settings are used",
            THRESHOLD_PROTOCOL,
        ),
        InputArtifact(
            Path(THRESHOLD_RECORD).name,
            THRESHOLD_RECORD_SHA256,
            "published threshold-calibration record, whose alerts under the "
            "default the standard profile must reproduce",
            THRESHOLD_RECORD,
        ),
        InputArtifact(
            Path(TIHM_POST_HOC_RECORD).name,
            TIHM_POST_HOC_RECORD_SHA256,
            "TIHM post hoc record, whose pooled-day medians the ceilings of C3 "
            "are set beside",
            TIHM_POST_HOC_RECORD,
        ),
    ]
    began = time.monotonic()

    def progress(done: int, homes: int) -> None:
        minutes = (time.monotonic() - began) / 60.0
        print(f"{done}/{homes} homes, {minutes:.1f} min", file=sys.stderr, flush=True)

    homes = run_homes(
        protocol, jobs=args.jobs, progress=progress, checkpoint=args.checkpoint
    )
    record = record_of(
        homes,
        protocol,
        planning=planning,
        published=published,
        protocol_sha256=protocol_sha256,
        inputs=inputs,
        code_changes=changed,
        clean=clean,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = record.write(args.output_dir / "matched-sensors.json")
    print(f"written {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
