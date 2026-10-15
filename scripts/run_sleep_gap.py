"""Set the pipeline's belief beside the TIHM sleep mat, minute by minute.

This runs ``artifacts/sleep_gap/sleep_gap_plan.json``. The script refuses to
run unless the frozen plan and the protocols it rests on equal what the code
declares, every pinned record and dataset file matches its digest, and the
working tree is clean. It reports nothing unless both runs of the mat homes
reproduce the published records, the matched days reproduce the sleep-mat
record's, and the second run's beliefs reproduce the first run's days.

Usage::

    python scripts/run_sleep_gap.py <tihm_dataset_dir> <output_dir> [--jobs N]

``tihm_dataset_dir`` is the ``Dataset`` folder of the extracted archive of
Zenodo record 7622128, which is not redistributed. The script writes
``<output_dir>/sleep-gap.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import sleep_mat_protocol, tihm_protocol
from sensor_modeling.datasets.sleep_gap_experiment import (
    read_mat_minutes,
    run_belief_homes,
    run_checks,
    sleep_gap_record,
)
from sensor_modeling.datasets.sleep_gap_plan import (
    ALERT_BURDEN_PROTOCOL,
    ALERT_BURDEN_RECORD,
    AN_HOUR_LATER,
    CLOCKS,
    PINNED_RECORDS,
    PRIMARY_CLOCK,
    SILENT_HOME_RECORD,
    SLEEP_MAT_PROTOCOL,
    SLEEP_MAT_RECORD,
    check_frozen_plan,
    code_changes,
    declared_plan,
    file_sha256,
)
from sensor_modeling.datasets.sleep_mat_experiment import (
    check_published,
    check_rule,
    read_mat,
    run_mat_homes,
)
from sensor_modeling.datasets.sleep_mat_protocol import MAT_FILE, RUN_OFF, RUN_RULE
from sensor_modeling.evaluation import InputArtifact, load_record
from sensor_modeling.evaluation.provenance import git_state
from sensor_modeling.external import OntologyMapping
from sensor_modeling.external.tihm import (
    FILES,
    READ_FILES,
    TIMEZONE,
    TihmAdapter,
    check_files,
)

PLAN_PATH = Path("artifacts/sleep_gap/sleep_gap_plan.json")
MAPPING_PATH = Path("artifacts/tihm/tihm_mapping.json")
SOURCE = "https://doi.org/10.5281/zenodo.7622128"


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("tihm_dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--jobs", type=int, default=1, help="homes run at once; the result is the same"
    )
    parser.add_argument(
        "--checks-only",
        action="store_true",
        help="run the checks, print whether each held, and compute nothing else",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run from a modified tree; the record will say so",
    )
    args = parser.parse_args()

    plan = declared_plan()
    for name, digest in PINNED_RECORDS.items():
        if file_sha256(Path(name)) != digest:
            raise SystemExit(f"{name} is not the file the plan pins")
    if args.checks_only and not PLAN_PATH.exists():
        # The checks may be rehearsed before the plan is frozen.
        plan_sha256 = ""
    else:
        plan_sha256 = check_frozen_plan(plan, PLAN_PATH)
        changed = code_changes(PLAN_PATH)
        if any(changed.values()):
            print(
                "the code differs from what the plan was frozen against: "
                + "; ".join(f"{k}: {', '.join(v)}" for k, v in changed.items() if v),
                file=sys.stderr,
            )
    protocol = sleep_mat_protocol.declared_protocol()
    sleep_mat_protocol.check_frozen_protocol(protocol, Path(SLEEP_MAT_PROTOCOL))
    alert_burden = tihm_protocol.declared_protocol()
    tihm_protocol.check_frozen_protocol(alert_burden, Path(ALERT_BURDEN_PROTOCOL))
    if not (
        alert_burden.step_minutes == protocol.tihm_step_minutes == plan.step_minutes
    ):
        raise SystemExit("the plan and the protocols do not agree on the step")
    if protocol.min_matched_days != plan.min_matched_days:
        raise SystemExit("the plan and the sleep-mat protocol differ on matched days")
    OntologyMapping.read(MAPPING_PATH, alert_burden.mapping.sha256())
    problems = check_files(args.tihm_dataset_dir)
    if problems:
        raise SystemExit(
            "; ".join(f"{name} {problem}" for name, problem in sorted(problems.items()))
        )
    if git_state().get("git_dirty") != "false" and not args.allow_dirty:
        raise SystemExit(
            "the working tree is not a clean commit, so the record could not be "
            "traced to the code that made it; commit first, or pass --allow-dirty"
        )

    published = load_record(Path(ALERT_BURDEN_RECORD))["results"]["households"]
    silent_home = load_record(Path(SILENT_HOME_RECORD))["results"]
    sleep_mat = load_record(Path(SLEEP_MAT_RECORD))["results"]
    inputs = [
        InputArtifact(PLAN_PATH.name, plan_sha256, "frozen plan", str(PLAN_PATH)),
        *(
            InputArtifact(Path(name).name, digest, "pinned record", name)
            for name, digest in PINNED_RECORDS.items()
        ),
        InputArtifact(
            MAPPING_PATH.name,
            file_sha256(MAPPING_PATH),
            "frozen mapping",
            str(MAPPING_PATH),
        ),
        *(
            InputArtifact(
                name,
                digest,
                (
                    "recording, read through the contract"
                    if name in READ_FILES
                    else (
                        "the sleep mat, read minute by minute"
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
    mats = {clock: read_mat(mat_path, shift) for clock, shift in CLOCKS.items()}
    homes = sorted(mats[PRIMARY_CLOCK])
    minutes = read_mat_minutes(mat_path, homes)
    began = time.monotonic()

    def progress(done: int, runs: int) -> None:
        elapsed = (time.monotonic() - began) / 60.0
        print(f"{done}/{runs} runs, {elapsed:.1f} min", file=sys.stderr, flush=True)

    adapter = TihmAdapter(args.tihm_dataset_dir)
    runs = run_mat_homes(
        adapter, homes, protocol, alert_burden, jobs=args.jobs, progress=progress
    )
    beliefs = run_belief_homes(
        adapter, homes, alert_burden, jobs=args.jobs, progress=progress
    )
    if args.checks_only:
        checked = run_checks(
            runs,
            beliefs,
            minutes,
            mats,
            protocol,
            plan,
            ZoneInfo(TIMEZONE),
            load_record(Path(SLEEP_MAT_RECORD))["results"],
        )
        check_published(runs[RUN_OFF], published)
        check_rule(
            runs[RUN_RULE], runs[RUN_OFF], silent_home, f"h{protocol.rule_hours:g}"
        )
        later = mats[AN_HOUR_LATER]
        unobserved = sum(
            1
            for home in checked.homes
            for day in checked.pairs[home].days
            if day not in later[home].observed
        )
        print("every check held:")
        for name, entry in checked.check.items():
            print(f"  {name}: {entry.get('reproduced')}")
        print(
            "  largest difference, hours: "
            f"{checked.check['beliefs']['largest_difference_hours']:.1e}"
        )
        print(f"matched days the mat does not observe an hour later: {unobserved}")
        return
    record = sleep_gap_record(
        runs,
        beliefs,
        minutes,
        mats,
        protocol,
        alert_burden,
        plan,
        published,
        silent_home,
        sleep_mat,
        plan_sha256=plan_sha256,
        inputs=inputs,
        code_changed=code_changes(PLAN_PATH),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = record.write(args.output_dir / "sleep-gap.json")
    print(f"written {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
