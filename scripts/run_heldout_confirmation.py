"""The held-out confirmation: fit on the development homes, score the cohort once.

Usage::

    python scripts/run_heldout_confirmation.py <archive_root> <output_dir> --check-inputs
    python scripts/run_heldout_confirmation.py <archive_root> <output_dir>

``archive_root`` is the extracted ``labeled_data.zip`` of Zenodo record
15708568 (CC-BY-4.0). Every recording is verified against its frozen SHA-256.

``--check-inputs`` runs before the freeze. It writes the inputs report the
protocol will declare, ``<output_dir>/heldout_inputs.json``: what each home
instruments, which windows are labelled and which states occur, and whether
every recursion runs on each held-out home. It scores nothing.

Without it, the script runs the frozen confirmation once. It refuses if the
record exists in the repository, if the code's protocol differs from the frozen
file, or if any pinned source file differs from the freeze. It writes the
record, ``<output_dir>/casas-heldout-confirmation.json``, and the results page
generated from it, ``<output_dir>/HELDOUT_CONFIRMATION_RESULTS.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import read_casas_hh
from sensor_modeling.datasets.heldout_experiment import (
    INPUTS_FILE,
    PINNED_RECORDS,
    HeldOutProtocol,
    check_frozen_protocol,
    code_changes,
    declared_protocol,
    inputs_report,
    run_heldout,
)
from sensor_modeling.datasets.heldout_summary import (
    PROTOCOL_FILE,
    RECORD_FILE,
    RESULTS_PAGE,
    render_page,
)
from sensor_modeling.datasets.recoverable_gap import FrozenSplits, load_frozen_splits
from sensor_modeling.evaluation import InputArtifact, load_record

SPLITS_PATH = Path("artifacts/phase1/household_splits.json")
PROTOCOL_PATH = Path(PROTOCOL_FILE)
TIMEZONE = "America/Los_Angeles"
SOURCE = "zenodo:15708568 labeled_data.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _homes(
    splits: FrozenSplits, protocol: HeldOutProtocol
) -> dict[str, dict[str, str]]:
    homes = {home: dict(splits.homes[home]) for home in protocol.training}
    homes.update({entry["id"]: dict(entry) for entry in protocol.cohort})
    return homes


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    splits = load_frozen_splits(SPLITS_PATH)
    zone = ZoneInfo(TIMEZONE)

    if args.check_inputs:
        protocol = declared_protocol(splits, inputs=False)
        homes = _homes(splits, protocol)
        paths = locate(args.archive_root, homes)
        recordings = {h: read_casas_hh(p, timezone=zone) for h, p in paths.items()}
        target = args.output_dir / Path(INPUTS_FILE).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(inputs_report(recordings, protocol), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"written {target}")
        return

    if Path(RECORD_FILE).exists():
        raise SystemExit(f"{RECORD_FILE} exists: the confirmation has been run")
    protocol = declared_protocol(splits)
    protocol_sha256 = check_frozen_protocol(protocol, PROTOCOL_PATH)
    changed = code_changes(PROTOCOL_PATH)
    if changed:
        raise SystemExit(f"pinned sources differ from the freeze: {changed}")
    homes = _homes(splits, protocol)
    paths = locate(args.archive_root, homes)
    recordings = {h: read_casas_hh(p, timezone=zone) for h, p in paths.items()}
    roles = {home: "training recording" for home in protocol.training}
    roles.update({home: "held-out recording" for home in protocol.test})
    inputs = [
        InputArtifact(
            PROTOCOL_PATH.name, protocol_sha256, "frozen protocol", str(PROTOCOL_PATH)
        ),
        InputArtifact(
            SPLITS_PATH.name, splits.sha256, "frozen household splits", str(SPLITS_PATH)
        ),
        *(
            InputArtifact(Path(name).name, digest, "pinned record", name)
            for name, digest in {
                **PINNED_RECORDS,
                INPUTS_FILE: protocol.inputs["sha256"] if protocol.inputs else "",
            }.items()
        ),
        *(
            InputArtifact(
                homes[home]["filename"], homes[home]["sha256"], roles[home], SOURCE
            )
            for home in sorted(paths)
        ),
    ]
    result = run_heldout(
        recordings,
        protocol,
        data_source="casas",
        inputs=inputs,
        output_dir=args.output_dir,
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    page = args.output_dir / RESULTS_PAGE
    page.write_text(
        render_page(load_record(result.path)), encoding="utf-8", newline="\n"
    )
    print(f"written {result.path} and {page}")


if __name__ == "__main__":
    main()
