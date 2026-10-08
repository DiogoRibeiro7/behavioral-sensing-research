"""Run the frozen threshold-calibration protocol on its simulated homes.

The protocol, ``artifacts/threshold_calibration/threshold_calibration_protocol.json``,
was frozen before any simulated home had been run with the calibrated
reference. The script refuses to run unless the code's protocol equals the
frozen file, the records the protocol names are the ones in the repository,
and the working tree is clean, so that the record it writes can be traced to
one commit. It also compares the source files, the libraries' versions and
the default settings the frozen file recorded with those it is about to use,
and refuses if any has changed unless it is told why; the record then says
what changed and why.

Usage::

    python scripts/run_threshold_calibration.py <output_dir> [--jobs N]

The script writes the record, ``<output_dir>/threshold-calibration.json``.
Each home is kept in ``<output_dir>/homes-<protocol>-<commit>-<code>`` as soon
as it is finished, so a run that is interrupted can be started again without
repeating the homes already done; the last part of the name is a digest of
the source files, so homes run by other code are never read back. That
directory is working storage and is not part of the result, and the output
directory must be outside the repository or ignored by it, or the homes kept
there would make the tree a modified one. The evidence is simulated. The page
and its figures are built from the record by
``scripts/render_threshold_calibration.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from sensor_modeling.datasets.threshold_calibration_experiment import (
    run_threshold_calibration,
)
from sensor_modeling.datasets.threshold_calibration_protocol import (
    NULL_RECORD,
    check_frozen_protocol,
    check_inputs,
    code_changes,
    code_now,
    declared_protocol,
    file_sha256,
)
from sensor_modeling.evaluation import InputArtifact
from sensor_modeling.evaluation.provenance import git_state

PROTOCOL_PATH = Path(
    "artifacts/threshold_calibration/threshold_calibration_protocol.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
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
    state = git_state()
    if state.get("git_dirty") != "false" and not args.allow_dirty:
        raise SystemExit(
            "the working tree is not a clean commit, so the record could not be "
            "traced to the code that made it; commit first, or pass --allow-dirty"
        )
    inside = subprocess.run(  # noqa: S603 - a fixed command
        ["git", "check-ignore", "-q", str(args.output_dir)],  # noqa: S607
        check=False,
    )
    here = Path.cwd().resolve()
    if here in args.output_dir.resolve().parents and inside.returncode != 0:
        raise SystemExit(
            "the output directory is inside the repository and is not ignored, "
            "so the homes kept there would make the tree a modified one"
        )
    commit = str(state.get("git_commit", "unknown"))[:12]
    code = hashlib.sha256(
        json.dumps(code_now()["sources"], sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    checkpoint = args.output_dir / f"homes-{protocol.sha256()[:12]}-{commit}-{code}"

    began = time.monotonic()

    def progress(done: int, homes: int) -> None:
        if done % 5 == 0 or done == homes:
            minutes = (time.monotonic() - began) / 60.0
            print(
                f"{done}/{homes} homes, {minutes:.1f} min", file=sys.stderr, flush=True
            )

    result = run_threshold_calibration(
        protocol,
        protocol_sha256=protocol_sha256,
        inputs=[
            InputArtifact(
                PROTOCOL_PATH.name,
                protocol_sha256,
                "frozen protocol",
                str(PROTOCOL_PATH),
            ),
            InputArtifact(
                Path(NULL_RECORD).name,
                file_sha256(Path(NULL_RECORD)),
                "the measurement on synthetic days that the option was designed "
                "on; not read by the run",
                NULL_RECORD,
            ),
        ],
        output_dir=args.output_dir,
        jobs=args.jobs,
        progress=progress,
        checkpoint=checkpoint,
        code_changes=changed,
        code_note=args.code_changed,
        clean=state.get("git_dirty") == "false",
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    judged = result.record.results["criteria"]
    verdicts = {
        name.partition("_")[0]: entry.get("verdict", entry.get("verdicts"))
        for name, entry in judged.items()
        if name != "reading"
    }
    print(f"written {result.path}")
    print(", ".join(f"{name} {verdict}" for name, verdict in verdicts.items()))
    print(judged["reading"])


if __name__ == "__main__":
    main()
