"""Shard and merge the frozen confirmatory paper experiment.

This launcher keeps household replications independent by using the configured
seed stride and makes a long N=200 study safe to execute across multiple jobs.
Numerical summaries are produced only after every shard has been collected and
the exact frozen seed set has been verified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
RUNNER_PATH = HERE / "run_confirmatory.py"
HYPOTHESES = ("h1", "h2", "h3", "h4", "h5")


def _load_runner() -> ModuleType:
    """Load the paper runner without turning the paper directory into a package."""
    spec = importlib.util.spec_from_file_location(
        "paper_confirmatory_runner", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load run_confirmatory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _full_seeds(config: Mapping[str, Any], n: int | None = None) -> list[int]:
    """Return the frozen independent household seed sequence."""
    replications = config["replications"]
    minimum = int(replications["minimum"])
    maximum = int(replications["maximum"])
    count = minimum if n is None else int(n)
    if count < minimum or count > maximum:
        raise ValueError(f"replications must lie in [{minimum}, {maximum}]")
    start = int(replications["seed_start"])
    stride = int(replications.get("seed_stride", 1))
    if stride < 4:
        raise ValueError(
            "seed_stride must be at least 4 because H3 derives degradation seeds "
            "at offsets +1, +2 and +3"
        )
    return [start + stride * index for index in range(count)]


def _shard_seeds(seeds: Sequence[int], index: int, count: int) -> list[int]:
    """Partition seeds deterministically by position modulo shard count."""
    if count < 1:
        raise ValueError("shard_count must be positive")
    if index < 0 or index >= count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    selected = list(seeds[index::count])
    if not selected:
        raise ValueError("selected shard contains no household seeds")
    return selected


def _run_shard(
    runner: ModuleType,
    config_path: Path,
    output: Path,
    *,
    replications: int | None,
    shard_index: int,
    shard_count: int,
) -> None:
    """Execute one deterministic subset and write raw household-level results."""
    config = runner._load(config_path)
    full_seeds = _full_seeds(config, replications)
    seeds = _shard_seeds(full_seeds, shard_index, shard_count)
    provenance = runner._git_provenance()
    if provenance["commit"] == "unknown" or provenance["dirty"] is not False:
        raise RuntimeError(
            f"production shard requires clean Git provenance: {provenance}"
        )

    step = runner.timedelta(minutes=int(config["household"]["step_minutes"]))
    raw = {
        "h1": runner._run_h1(config, seeds, step),
        "h2": runner._run_h2(config, seeds, step),
        "h3": runner._run_h3(config, seeds, step),
        "h4": runner._run_h4(config, seeds, step),
        "h5": runner._run_h5(config, seeds, step),
    }
    artifact = {
        "artifact_type": "confirmatory-shard",
        "experiment_schema_version": int(config["schema_version"]),
        "git": provenance,
        "environment": runner._environment(),
        "config_sha256": runner._config_sha256(config_path),
        "resolved_config": config,
        "replications_total": len(full_seeds),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "seeds": seeds,
        "raw": raw,
        "interpretation_guardrail": (
            "Shard outputs have no standalone inferential status and must be merged "
            "before scientific interpretation."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"shard-{shard_index:03d}-of-{shard_count:03d}.json"
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {target} with {len(seeds)} household seeds")


def _read_shards(input_dir: Path) -> list[dict[str, Any]]:
    """Read every shard artifact from a directory."""
    paths = sorted(input_dir.glob("shard-*-of-*.json"))
    if not paths:
        raise ValueError(f"no shard artifacts found in {input_dir}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _assert_same(shards: Sequence[Mapping[str, Any]], key: str) -> Any:
    """Require identical provenance/configuration values across all shards."""
    values = [shard[key] for shard in shards]
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"shards disagree on {key}")
    return first


def _merge_raw(shards: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Combine household rows and sort them deterministically by seed."""
    merged: dict[str, list[dict[str, Any]]] = {name: [] for name in HYPOTHESES}
    for shard in shards:
        if shard.get("artifact_type") != "confirmatory-shard":
            raise ValueError("unexpected shard artifact type")
        for hypothesis in HYPOTHESES:
            merged[hypothesis].extend(shard["raw"][hypothesis])
    for rows in merged.values():
        rows.sort(key=lambda row: int(row["seed"]))
    return merged


def _merge_shards(
    runner: ModuleType,
    config_path: Path,
    input_dir: Path,
    output: Path,
    *,
    replications: int | None,
) -> None:
    """Validate shard completeness and create the only manuscript-facing artifact."""
    config = runner._load(config_path)
    expected_seeds = _full_seeds(config, replications)
    shards = _read_shards(input_dir)

    config_sha = _assert_same(shards, "config_sha256")
    if config_sha != runner._config_sha256(config_path):
        raise ValueError("shards were not produced from the supplied frozen config")
    git = _assert_same(shards, "git")
    environment = _assert_same(shards, "environment")
    shard_count = int(_assert_same(shards, "shard_count"))
    total = int(_assert_same(shards, "replications_total"))
    if total != len(expected_seeds):
        raise ValueError("shard replication total does not match frozen seed set")
    if len(shards) != shard_count:
        raise ValueError(f"expected {shard_count} shard files, found {len(shards)}")
    indices = sorted(int(shard["shard_index"]) for shard in shards)
    if indices != list(range(shard_count)):
        raise ValueError("shard indices are incomplete or duplicated")

    observed_seeds = [int(seed) for shard in shards for seed in shard["seeds"]]
    if len(observed_seeds) != len(set(observed_seeds)):
        raise ValueError("duplicate household seeds across shards")
    if sorted(observed_seeds) != sorted(expected_seeds):
        missing = sorted(set(expected_seeds) - set(observed_seeds))
        extra = sorted(set(observed_seeds) - set(expected_seeds))
        raise ValueError(f"shard seed union mismatch; missing={missing}, extra={extra}")

    raw = _merge_raw(shards)
    summaries = runner._summaries(config, raw)
    mcse_gate = runner._mcse_gate(config, summaries)
    results = {"raw": raw, "summaries": summaries, "mcse_gate": mcse_gate}
    status, reasons = runner._confirmatory_status(config, expected_seeds, results, git)
    artifact = {
        "experiment_schema_version": int(config["schema_version"]),
        "status": status,
        "non_confirmatory_reasons": reasons,
        "git": git,
        "environment": environment,
        "config_sha256": config_sha,
        "resolved_config": config,
        "seeds": expected_seeds,
        "shard_count": shard_count,
        "results": results,
        "interpretation_guardrail": (
            "All outcomes are simulator-derived and are not estimates of field "
            "performance."
        ),
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "confirmatory_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    for hypothesis in HYPOTHESES:
        runner._write_csv(output / f"{hypothesis}.csv", raw[hypothesis])
    (output / "generated_results.tex").write_text(
        runner._latex_macros(summaries, len(expected_seeds), status), encoding="utf-8"
    )
    print(f"Merged {len(shards)} shards covering {len(expected_seeds)} households")
    print(f"MCSE gate: {mcse_gate}")
    print(f"Status: {status}")
    for reason in reasons:
        print(f"- {reason}")


def main() -> None:
    """Run one production shard or merge a complete shard set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--replications", type=int, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    shard = subparsers.add_parser("shard")
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-count", type=int, required=True)
    shard.add_argument("--output", type=Path, required=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--input", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    runner = _load_runner()
    if args.command == "shard":
        _run_shard(
            runner,
            args.config,
            args.output,
            replications=args.replications,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    else:
        _merge_shards(
            runner,
            args.config,
            args.input,
            args.output,
            replications=args.replications,
        )


if __name__ == "__main__":
    main()
