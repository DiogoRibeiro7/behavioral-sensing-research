"""Phase 3.3 follow-up: activity sub-states in the recursion.

Everything is fixed in ``artifacts/phase3/substates_protocol.json``, committed
before any household was scored. The script refuses to run if the code's
protocol differs from the frozen file, if a source differs from the one the
freeze recorded, if the record already exists, if a pinned record is not the
file in the repository, or if a recording does not match its frozen SHA-256.
The run itself reports nothing unless the fold fits, the reference's scores and
the quiet-run statistic give back the published records.

Usage::

    python scripts/run_phase3_substates.py <archive_root> <output_dir>

``archive_root`` is the extracted ``labeled_data.zip`` of Zenodo record
15708568 (CC-BY-4.0). The script writes the record,
``<output_dir>/phase3-activity-substates.json``, and the results page generated
from the written record, ``<output_dir>/PHASE3_ACTIVITY_SUBSTATES_RESULTS.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import read_casas_hh
from sensor_modeling.datasets.archive import locate
from sensor_modeling.datasets.combined_prior_experiment import FITTED_RATES_RECORD
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.substate_experiment import (
    AMENDMENTS_FILE,
    COMBINED_PRIOR_RECORD,
    PINNED_RECORDS,
    PREDICTIVE_RECORD,
    check_frozen_protocol,
    code_changes,
    declared_protocol,
    file_sha256,
    read_amendments,
    run_substates,
)
from sensor_modeling.datasets.substate_summary import (
    PROTOCOL_FILE,
    RECORD_FILE,
    RESULTS_PAGE,
    render_page,
)
from sensor_modeling.evaluation import InputArtifact, load_record

SPLITS_PATH = Path("artifacts/phase1/household_splits.json")
PROTOCOL_PATH = Path(PROTOCOL_FILE)
TIMEZONE = "America/Los_Angeles"
SOURCE = "zenodo:15708568 labeled_data.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    if Path(RECORD_FILE).exists():
        raise SystemExit(f"{RECORD_FILE} exists: the experiment has been run")
    splits = load_frozen_splits(SPLITS_PATH)
    protocol = declared_protocol(splits)
    protocol_sha256 = check_frozen_protocol(protocol, PROTOCOL_PATH)
    amendments = read_amendments(Path(AMENDMENTS_FILE))
    changed = code_changes(PROTOCOL_PATH, amendments=amendments)
    if changed:
        raise SystemExit(
            f"pinned sources differ from the freeze and its amendments: {changed}"
        )
    homes = {home: dict(entry) for home, entry in splits.homes.items()}
    try:
        paths = locate(args.archive_root, homes)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    zone = ZoneInfo(TIMEZONE)
    recordings = {
        home: read_casas_hh(path, timezone=zone) for home, path in paths.items()
    }
    inputs = [
        InputArtifact(
            PROTOCOL_PATH.name, protocol_sha256, "frozen protocol", str(PROTOCOL_PATH)
        ),
        InputArtifact(
            SPLITS_PATH.name, splits.sha256, "frozen household splits", str(SPLITS_PATH)
        ),
        *(
            InputArtifact(Path(name).name, digest, "pinned record", name)
            for name, digest in PINNED_RECORDS.items()
        ),
        *(
            [
                InputArtifact(
                    Path(AMENDMENTS_FILE).name,
                    file_sha256(Path(AMENDMENTS_FILE)),
                    f"amendments to the freeze: {len(amendments)}",
                    AMENDMENTS_FILE,
                )
            ]
            if amendments
            else []
        ),
        *(
            InputArtifact(
                homes[home]["filename"], homes[home]["sha256"], "recording", SOURCE
            )
            for home in sorted(paths)
        ),
    ]
    result = run_substates(
        recordings,
        protocol,
        fitted_rates=load_record(Path(FITTED_RATES_RECORD)),
        combined_prior=load_record(Path(COMBINED_PRIOR_RECORD)),
        predictive=load_record(Path(PREDICTIVE_RECORD)),
        data_source="casas-hh",
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
