"""Phase 4: select the abstention rule, once, as frozen.

Everything is fixed in ``artifacts/phase4/abstention_protocol.json``, committed
with the code before any household was scored. The script refuses outside its
workflow on develop, when the record exists, when the protocol or a pinned
source or library differs from the freeze, or when a recording does not match
its digest.

Usage::

    python scripts/run_abstention_rule.py <archive_root> <output_dir>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import read_casas_hh
from sensor_modeling.datasets.abstention_experiment import (
    AMENDMENTS_FILE,
    PINNED_DISTRIBUTIONS,
    PINNED_RECORDS,
    PROTOCOL_FILE,
    RECORD_FILE,
    STEP1_RECORD,
    check_frozen_protocol,
    code_changes,
    declared_protocol,
    run_abstention,
)
from sensor_modeling.datasets.abstention_summary import RESULTS_PAGE, render_page
from sensor_modeling.datasets.archive import locate
from sensor_modeling.datasets.combined_prior_experiment import FITTED_RATES_RECORD
from sensor_modeling.datasets.pinning import file_sha256, library_changes
from sensor_modeling.datasets.prior_uncertainty_experiment import (
    COMBINED_PRIOR_RECORD,
)
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.evaluation import InputArtifact, load_record

SPLITS_PATH = Path("artifacts/phase1/household_splits.json")
PROTOCOL_PATH = Path(PROTOCOL_FILE)
SOURCE = "zenodo:15708568 labeled_data.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REF") != (
        "refs/heads/develop"
    ):
        raise SystemExit("the one run is made only by its workflow, on develop")
    if Path(RECORD_FILE).exists():
        raise SystemExit(f"{RECORD_FILE} exists: the rule has been selected")
    splits = load_frozen_splits(SPLITS_PATH)
    protocol = declared_protocol(splits)
    protocol_sha256 = check_frozen_protocol(protocol, PROTOCOL_PATH)
    changed = code_changes(PROTOCOL_PATH) + library_changes(
        PROTOCOL_PATH, PINNED_DISTRIBUTIONS
    )
    if changed:
        raise SystemExit(
            f"pinned sources or libraries differ from the freeze: {changed}"
        )
    homes = {home: dict(entry) for home, entry in splits.homes.items()}
    try:
        paths = locate(args.archive_root, homes)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    zone = ZoneInfo("America/Los_Angeles")
    recordings = {h: read_casas_hh(p, timezone=zone) for h, p in paths.items()}
    inputs = [
        InputArtifact(
            PROTOCOL_PATH.name, protocol_sha256, "frozen protocol", PROTOCOL_FILE
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
                    "amendments to the freeze",
                    AMENDMENTS_FILE,
                )
            ]
            if Path(AMENDMENTS_FILE).exists()
            else []
        ),
        *(
            InputArtifact(homes[h]["filename"], homes[h]["sha256"], "recording", SOURCE)
            for h in sorted(paths)
        ),
    ]
    result = run_abstention(
        recordings,
        protocol,
        fitted_rates=load_record(Path(FITTED_RATES_RECORD)),
        combined_prior=load_record(Path(COMBINED_PRIOR_RECORD)),
        step1_record=load_record(Path(STEP1_RECORD)),
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
