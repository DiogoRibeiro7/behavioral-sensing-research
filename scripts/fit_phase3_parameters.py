"""Fit the phase3 formulation's parameters once, as the held-out confirmation did.

The fit is :func:`~sensor_modeling.datasets.heldout_experiment.training_fit` on
the training homes of the frozen held-out protocol: the 20 CASAS development
homes. Nothing is scored. The script refuses unless the fitted channels and
time prior have exactly the digests the held-out record gives for its training
fit, then writes the parameter file the package ships.

The channels are fitted with Python's own arithmetic and reproduced the
record's digest on a laptop. The time prior is fitted by Newton's method with
numpy, whose vectorised functions and linear algebra choose code paths by CPU
and can differ in their last bits between CPUs. On that laptop, under every
OpenBLAS kernel it could run, the prior's coefficients differed from one
another by up to 2e-13 and none had the record's digest. So the prior may
reproduce the record only on a CPU like the held-out run's. The script records
the CPU it ran on.

Usage::

    python scripts/fit_phase3_parameters.py <extracted CASAS archive> <output dir>
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import read_casas_hh
from sensor_modeling.datasets.archive import locate
from sensor_modeling.datasets.heldout_experiment import (
    check_frozen_protocol,
    declared_protocol,
    file_sha256,
    training_fit,
)
from sensor_modeling.datasets.heldout_summary import PROTOCOL_FILE, RECORD_FILE
from sensor_modeling.datasets.phase3_formulation import (
    CHANNELS_SHA256,
    HELDOUT_RECORD,
    PARAMETERS_FILE,
    PERIODIC_PRIOR_SHA256,
    parameters_payload,
    write_parameters,
)
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.states import StateOntology

SPLITS_PATH = Path("artifacts/phase1/household_splits.json")
TIMEZONE = "America/Los_Angeles"
SOURCE = "zenodo:15708568 labeled_data.zip"
DISTRIBUTIONS = ("numpy", "pandas", "scipy", "scikit-learn")


def _cpu() -> dict[str, object]:
    """The CPU's model and whether it has AVX-512, as Linux reports them."""
    try:
        info = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return {"model": platform.processor(), "avx512f": None}
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in info.splitlines()
            if "model name" in line
        ),
        platform.processor(),
    )
    flags = next((line for line in info.splitlines() if line.startswith("flags")), "")
    return {"model": model, "avx512f": "avx512f" in flags.split()}


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    if HELDOUT_RECORD != RECORD_FILE:
        raise SystemExit("the package names a record other than the held-out one")
    record = json.loads(Path(HELDOUT_RECORD).read_text(encoding="utf-8"))
    training = record["results"]["training"]
    if (training["channels_sha256"], training["periodic_prior_sha256"]) != (
        CHANNELS_SHA256,
        PERIODIC_PRIOR_SHA256,
    ):
        raise SystemExit("the package's expected digests are not the record's")

    splits = load_frozen_splits(SPLITS_PATH)
    protocol = declared_protocol(splits)
    protocol_sha256 = check_frozen_protocol(protocol, Path(PROTOCOL_FILE))
    if list(protocol.training) != training["homes"]:
        raise SystemExit("the protocol's training homes are not the record's")

    homes = {home: dict(splits.homes[home]) for home in protocol.training}
    paths = locate(args.archive_root, homes)
    zone = ZoneInfo(TIMEZONE)
    recordings = {
        home: read_casas_hh(path, timezone=zone) for home, path in paths.items()
    }
    fit = training_fit(recordings, protocol.training, protocol.base, StateOntology())
    if fit.channels.sha256() != CHANNELS_SHA256:
        raise SystemExit("the fitted channels differ from the held-out record's")
    if fit.periodic.sha256() != PERIODIC_PRIOR_SHA256:
        raise SystemExit(
            "the fitted time prior differs from the held-out record's. Its fit "
            "rounds differently on other CPUs; this one is "
            f"{_cpu()}. Dispatch the workflow again to run on another runner."
        )

    source = {
        "fit": "sensor_modeling.datasets.heldout_experiment.training_fit",
        "record": HELDOUT_RECORD,
        "record_sha256": file_sha256(Path(HELDOUT_RECORD)),
        "protocol": PROTOCOL_FILE,
        "protocol_sha256": protocol_sha256,
        "splits": SPLITS_PATH.as_posix(),
        "splits_sha256": splits.sha256,
        "data": SOURCE,
        "timezone": TIMEZONE,
        "recordings": {home: homes[home]["sha256"] for home in sorted(homes)},
        "git_commit": _commit(),
        "python": platform.python_version(),
        "cpu": _cpu(),
        "distributions": {name: metadata.version(name) for name in DISTRIBUTIONS},
    }
    target = write_parameters(
        parameters_payload(fit.channels, fit.periodic, source),
        args.output_dir / PARAMETERS_FILE,
    )
    print(f"written {target}")


if __name__ == "__main__":
    main()
