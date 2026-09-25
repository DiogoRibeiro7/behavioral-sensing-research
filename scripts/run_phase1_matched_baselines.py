"""Phase 1: the four matched information sets and the Phase 2 baselines.

This analysis is exploratory. The development homes have been inspected
throughout earlier work, so nothing here is confirmatory evidence. It measures,
on real homes, how much each kind of information is worth to each baseline,
with every comparison matched and resampled by household.

Protocol, fixed in this file before any household was scored:

- homes: the 20 single-resident fitting homes of the frozen v0.3 development
  panel (``artifacts/v03/v03_circadian_profile.json``). Each recording is
  verified against its recorded SHA-256. The two two-resident homes stay
  excluded, as in the frozen fit;
- the America/Los_Angeles time zone and a 5-minute step, as in every
  development result;
- two cross-fitted folds: the sorted home identifiers alternate between them,
  and each fold is held out once, so every home is scored exactly once by
  models that never saw it;
- the four nested information sets at their default resolution;
- ``baseline_suite`` for each set, with seed 0;
- balanced accuracy, log loss and Brier score, each summarised across
  households and compared with 10,000 household resamples and 95% percentile
  intervals.

Usage::

    python scripts/run_phase1_matched_baselines.py <archive_root> <output_dir>

``archive_root`` is the extracted ``labeled_data.zip`` of Zenodo record
15708568 (CC-BY-4.0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import permutations
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import (
    HouseholdSplit,
    MatchedEvaluation,
    baseline_suite,
    compare_information_sets,
    held_out_metrics,
    nested_information_sets,
    read_casas_hh,
    run_matched_evaluation,
)
from sensor_modeling.datasets.matched_evaluation import MATCHED_METRIC_DEFINITIONS
from sensor_modeling.evaluation import (
    ExperimentRecord,
    compare_households,
    household_values,
    summarise_households,
)

PROFILE_PATH = Path("artifacts/v03/v03_circadian_profile.json")
TIMEZONE = "America/Los_Angeles"
SEED = 0
RESAMPLES = 10_000
CONFIDENCE = 0.95
METRICS = ("balanced_accuracy", "log_loss", "brier")
REFERENCE_MODEL = "state_frequency"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def development_homes(profile_path: Path = PROFILE_PATH) -> dict[str, dict[str, str]]:
    """The frozen single-resident development homes, with filename and digest."""
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    panel = {home["id"]: home for home in profile["development_panel_homes"]}
    homes = {
        home: {"filename": panel[home]["filename"], "sha256": panel[home]["sha256"]}
        for home in sorted(profile["fitting_home_ids"])
    }
    excluded = {item["id"] for item in profile["fitting_exclusions"]}
    if excluded & set(homes) or len(homes) != profile["fitting_home_count"]:
        raise SystemExit("frozen development homes are inconsistent")
    return homes


def folds(homes: list[str]) -> tuple[HouseholdSplit, HouseholdSplit]:
    """Two cross-fitted folds: sorted homes alternate, each fold held out once."""
    ordered = sorted(homes)
    first, second = tuple(ordered[0::2]), tuple(ordered[1::2])
    return (
        HouseholdSplit("phase1-fold-a", train=second, test=first),
        HouseholdSplit("phase1-fold-b", train=first, test=second),
    )


def locate(root: Path, homes: dict[str, dict[str, str]]) -> dict[str, Path]:
    """Find every recording under *root* and verify it against its frozen digest."""
    paths: dict[str, Path] = {}
    for home, entry in homes.items():
        matches = sorted(p for p in root.rglob(entry["filename"]) if p.is_file())
        if len(matches) != 1:
            raise SystemExit(f"expected one {entry['filename']}, found {len(matches)}")
        actual = _sha256(matches[0])
        if actual != entry["sha256"]:
            raise SystemExit(
                f"{home}: digest {actual} is not the frozen {entry['sha256']}"
            )
        paths[home] = matches[0]
    return paths


def _summaries(runs: dict[str, list[MatchedEvaluation]]) -> dict[str, Any]:
    """Each model's metrics across held-out households, per information set."""
    return {
        name: {
            spec.name: {
                metric: summarise_households(
                    household_values(held_out_metrics(folded, spec.name), metric)
                ).to_dict()
                for metric in METRICS
            }
            for spec in folded[0].models
        }
        for name, folded in runs.items()
    }


def _against_reference(runs: dict[str, list[MatchedEvaluation]]) -> dict[str, Any]:
    """Every baseline against state frequency within each set, pooled over folds."""
    results: dict[str, Any] = {}
    for name, folded in runs.items():
        reference = held_out_metrics(folded, REFERENCE_MODEL)
        results[name] = {
            spec.name: {
                metric: compare_households(
                    household_values(held_out_metrics(folded, spec.name), metric),
                    household_values(reference, metric),
                    higher_is_better=metric == "balanced_accuracy",
                    confidence=CONFIDENCE,
                    resamples=RESAMPLES,
                    seed=SEED,
                ).to_dict()
                for metric in METRICS
            }
            for spec in folded[0].models
            if spec.name != REFERENCE_MODEL
        }
    return results


def _information_gains(runs: dict[str, list[MatchedEvaluation]]) -> dict[str, Any]:
    """What each larger set adds, per model, for every strictly nested pair."""
    sets = {
        folded[0].information_set.name: folded[0].information_set
        for folded in runs.values()
    }
    results: dict[str, Any] = {}
    for small, large in permutations(sets, 2):
        if sets[small] == sets[large] or not sets[small].is_nested_in(sets[large]):
            continue
        shared = [
            spec.name
            for spec in runs[small][0].models
            if spec.name in {other.name for other in runs[large][0].models}
        ]
        results[f"{small} -> {large}"] = {
            model: {
                metric: compare_information_sets(
                    runs[small],
                    runs[large],
                    model=model,
                    metric=metric,
                    confidence=CONFIDENCE,
                    resamples=RESAMPLES,
                    seed=SEED,
                ).to_dict()
                for metric in METRICS
            }
            for model in shared
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    homes = development_homes()
    paths = locate(args.archive_root, homes)
    zone = ZoneInfo(TIMEZONE)
    recordings = {
        home: read_casas_hh(path, timezone=zone) for home, path in paths.items()
    }
    splits = folds(list(recordings))

    runs: dict[str, list[MatchedEvaluation]] = {}
    for information_set in nested_information_sets():
        runs[information_set.name] = [
            run_matched_evaluation(
                recordings,
                split=split,
                information_set=information_set,
                models=baseline_suite(information_set),
                seed=SEED,
                data_source="casas-hh",
                output_dir=args.output_dir,
                experiment=f"phase1-{information_set.name}-{split.name}",
                metrics=METRICS,
                resamples=RESAMPLES,
                confidence=CONFIDENCE,
            )
            for split in splits
        ]
        print(f"scored {information_set.name}", flush=True)

    record = ExperimentRecord(
        experiment="phase1-matched-baselines",
        configuration={
            "homes": homes,
            "folds": [split.to_dict() for split in splits],
            "timezone": TIMEZONE,
            "information_sets": [
                {**info.to_dict(), "sha256": info.sha256()}
                for info in nested_information_sets()
            ],
            "metrics": list(METRICS),
            "bootstrap": {
                "unit": "household",
                "resamples": RESAMPLES,
                "confidence": CONFIDENCE,
                "interval": "percentile",
            },
            "status": "exploratory, development panel",
        },
        seeds=[SEED],
        metric_definitions=MATCHED_METRIC_DEFINITIONS,
        results={
            "households": _summaries(runs),
            "against_state_frequency": _against_reference(runs),
            "information_gain": _information_gains(runs),
        },
        data_source="casas-hh",
    )
    path = record.write(args.output_dir / "phase1_matched_baselines.json")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
