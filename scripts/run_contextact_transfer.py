"""Run the ContextAct transfer test once, or write its inputs report.

Usage::

    python scripts/run_contextact_transfer.py <CASAS archive> <ContextAct directory> \
        <output dir> [--check-inputs]

The CASAS archive is extracted ``labeled_data.zip`` from Zenodo record
15708568; the ContextAct directory holds the files of Mendeley Data record
``fcj2hmz5kb`` version 3, as published. Every file read is verified against its
digest first: the training recordings against the frozen splits, the ContextAct
files against the adapter's, and the mapping against its frozen file.

With ``--check-inputs`` it writes the inputs report the protocol declares and
scores nothing. Without it, it refuses unless the protocol is the frozen one, no
pinned source has changed since the freeze and no record exists, then writes the
record and its page. Either way it refuses unless Python and the libraries are
the versions the protocol declares.
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import read_casas_hh
from sensor_modeling.datasets.archive import locate, sha256
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.transfer_experiment import (
    ENVIRONMENT,
    INPUTS_FILE,
    PINNED_RECORDS,
    PROTOCOL_FILE,
    RECORD_FILE,
    check_frozen_protocol,
    code_changes,
    declared_protocol,
    inputs_report,
    read_periods,
    run_transfer,
)
from sensor_modeling.datasets.transfer_summary import RESULTS_PAGE, render_page
from sensor_modeling.evaluation import InputArtifact, load_record
from sensor_modeling.external import OntologyMapping
from sensor_modeling.external.contextact import (
    CONTEXTACT_MAPPING,
    FILES,
    MAPPING_FILE,
    RECORD,
)

SPLITS_PATH = Path("artifacts/phase1/household_splits.json")
PROTOCOL_PATH = Path(PROTOCOL_FILE)
TIMEZONE = "America/Los_Angeles"
CASAS = "zenodo:15708568 labeled_data.zip"


def _check_environment() -> None:
    found = {
        "python": ".".join(platform.python_version_tuple()[:2]),
        **{name: metadata.version(name) for name in ENVIRONMENT if name != "python"},
    }
    if found != ENVIRONMENT:
        raise SystemExit(f"the environment is {found}, not the declared {ENVIRONMENT}")


def _verify_contextact(directory: Path) -> None:
    for name, digest in FILES.items():
        path = directory / name
        if not path.is_file():
            raise SystemExit(f"ContextAct file {name} is missing")
        if sha256(path) != digest:
            raise SystemExit(f"ContextAct file {name} does not match its digest")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("casas_root", type=Path)
    parser.add_argument("contextact_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()

    _check_environment()
    splits = load_frozen_splits(SPLITS_PATH)
    _verify_contextact(args.contextact_dir)
    OntologyMapping.read(Path(MAPPING_FILE), CONTEXTACT_MAPPING.sha256())

    if args.check_inputs:
        protocol = declared_protocol(splits, inputs=False)
    else:
        if Path(RECORD_FILE).exists():
            raise SystemExit(f"{RECORD_FILE} exists: the transfer test has been run")
        protocol = declared_protocol(splits)
        check_frozen_protocol(protocol, PROTOCOL_PATH)
        changed = code_changes(PROTOCOL_PATH)
        if changed:
            raise SystemExit(f"pinned sources differ from the freeze: {changed}")

    homes = {home: dict(splits.homes[home]) for home in protocol.training}
    paths = locate(args.casas_root, homes)
    zone = ZoneInfo(TIMEZONE)
    training = {
        home: read_casas_hh(path, timezone=zone) for home, path in paths.items()
    }
    periods, validation = read_periods(args.contextact_dir, protocol.periods)

    if args.check_inputs:
        target = args.output_dir / Path(INPUTS_FILE).name
        target.parent.mkdir(parents=True, exist_ok=True)
        report = inputs_report(training, periods, validation, protocol)
        target.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"written {target}")
        return

    inputs = [
        InputArtifact(
            PROTOCOL_PATH.name,
            check_frozen_protocol(protocol, PROTOCOL_PATH),
            "frozen protocol",
            str(PROTOCOL_PATH),
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
                homes[home]["filename"],
                homes[home]["sha256"],
                "training recording",
                CASAS,
            )
            for home in sorted(paths)
        ),
        *(
            InputArtifact(
                Path(name).name, digest, "ContextAct file", f"{RECORD} {name}"
            )
            for name, digest in sorted(FILES.items())
        ),
    ]
    result = run_transfer(
        training,
        periods,
        validation,
        protocol,
        data_source="contextact-a4h",
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
