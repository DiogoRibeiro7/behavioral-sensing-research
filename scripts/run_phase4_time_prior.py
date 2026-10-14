"""Phase 4 for the time prior: uncertainty diagnostics against confidence.

Everything is fixed in ``artifacts/phase4/prior_uncertainty_protocol.json``,
committed with the code it depends on before any household was scored. The
script refuses to run if the code's protocol differs from the frozen file, if a
source differs from the one the freeze recorded, if the record already exists,
if a pinned record is not the file in the repository, or if a recording does
not match its frozen SHA-256. The run itself reports nothing unless the fold
fits and the reference's scores give back the published records.

Usage::

    python scripts/run_phase4_time_prior.py <archive_root> <output_dir>

``archive_root`` is the extracted ``labeled_data.zip`` of Zenodo record
15708568 (CC-BY-4.0). The script writes the record,
``<output_dir>/phase4-uncertainty-time-prior.json``, the results page generated
from the written record, ``<output_dir>/PHASE4_TIME_PRIOR_RESULTS.md``, and its
figures in ``<output_dir>/figures``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import read_casas_hh
from sensor_modeling.datasets.archive import locate
from sensor_modeling.datasets.combined_prior_experiment import FITTED_RATES_RECORD
from sensor_modeling.datasets.prior_uncertainty_experiment import (
    AMENDMENTS_FILE,
    COMBINED_PRIOR_RECORD,
    EXPERIMENT,
    PINNED_RECORDS,
    PROTOCOL_FILE,
    RECORD_FILE,
    check_frozen_protocol,
    code_changes,
    declared_protocol,
    file_sha256,
    library_changes,
    read_amendments,
    run_prior_uncertainty,
)
from sensor_modeling.datasets.prior_uncertainty_summary import (
    FIGURE_PREFIX,
    RESULTS_PAGE,
    TITLE,
    render_page,
)
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.uncertainty_figures import draw_figures
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

    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REF") != (
        "refs/heads/develop"
    ):
        raise SystemExit("the one run is made only by its workflow, on develop")
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
    libraries = library_changes(PROTOCOL_PATH)
    if libraries:
        raise SystemExit(f"libraries differ from the freeze: {libraries}")
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
    result = run_prior_uncertainty(
        recordings,
        protocol,
        fitted_rates=load_record(Path(FITTED_RATES_RECORD)),
        combined_prior=load_record(Path(COMBINED_PRIOR_RECORD)),
        data_source="casas-hh",
        inputs=inputs,
        output_dir=args.output_dir,
    )
    if result.path is None:  # pragma: no cover - an output directory is given
        raise SystemExit("the record was not written")
    written = load_record(result.path)
    page = args.output_dir / RESULTS_PAGE
    page.write_text(render_page(written), encoding="utf-8", newline="\n")
    figures = draw_figures(
        written,
        args.output_dir / "figures",
        prefix=FIGURE_PREFIX,
        salt=EXPERIMENT,
        title=TITLE,
    )
    print(f"written {result.path}, {page} and {len(figures)} figures")


if __name__ == "__main__":
    main()
