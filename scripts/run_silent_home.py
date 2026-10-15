"""Run the frozen silent-home protocol on its simulated homes.

The protocol, ``artifacts/silent_home/silent_home_protocol.json``, was frozen
before any simulated home had been run with the rule on. The script refuses to
run unless the code's protocol equals the frozen file and the working tree is
clean, so that the record it writes can be traced to one commit.

Usage::

    python scripts/run_silent_home.py <output_dir> [--jobs N]

The script writes the record, ``<output_dir>/silent-home.json``. Each home is
kept in ``<output_dir>/homes-<protocol>-<commit>`` as soon as it is finished,
so a run that is interrupted can be started again without repeating the homes
already done; that directory is working storage and is not part of the result.
The evidence is simulated. The page and its figures are built from the record
by ``scripts/render_silent_home.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sensor_modeling.datasets.silent_home_experiment import run_silent_home
from sensor_modeling.datasets.silent_home_protocol import (
    check_frozen_protocol,
    declared_protocol,
)
from sensor_modeling.evaluation import InputArtifact
from sensor_modeling.evaluation.provenance import git_state

PROTOCOL_PATH = Path("artifacts/silent_home/silent_home_protocol.json")


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
    args = parser.parse_args()

    protocol = declared_protocol()
    protocol_sha256 = check_frozen_protocol(protocol, PROTOCOL_PATH)
    state = git_state()
    if state.get("git_dirty") != "false" and not args.allow_dirty:
        raise SystemExit(
            "the working tree is not a clean commit, so the record could not be "
            "traced to the code that made it; commit first, or pass --allow-dirty"
        )
    commit = str(state.get("git_commit", "unknown"))[:12]
    checkpoint = args.output_dir / f"homes-{protocol.sha256()[:12]}-{commit}"

    began = time.monotonic()

    def progress(done: int, homes: int) -> None:
        minutes = (time.monotonic() - began) / 60.0
        print(f"{done}/{homes} homes, {minutes:.1f} min", file=sys.stderr, flush=True)

    result = run_silent_home(
        protocol,
        protocol_sha256=protocol_sha256,
        inputs=[
            InputArtifact(
                PROTOCOL_PATH.name,
                protocol_sha256,
                "frozen protocol",
                str(PROTOCOL_PATH),
            )
        ],
        output_dir=args.output_dir,
        jobs=args.jobs,
        progress=progress,
        checkpoint=checkpoint,
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    verdicts = {
        name: entry["verdict"]
        for name, entry in result.record.results["criteria"].items()
    }
    print(f"written {result.path}")
    print(", ".join(f"{name} {verdict}" for name, verdict in verdicts.items()))


if __name__ == "__main__":
    main()
