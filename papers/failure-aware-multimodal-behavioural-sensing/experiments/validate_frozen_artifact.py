"""Validate a merged paper artifact against the pre-result freeze manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "FREEZE_MANIFEST.json"
HYPOTHESES = ("h1", "h2", "h3", "h4", "h5")


def _load(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _expected_seeds(manifest: Mapping[str, Any], n: int) -> list[int]:
    """Construct the exact frozen household seed sequence."""
    spec = manifest["replications"]
    initial_n = int(spec["initial_n"])
    maximum_n = int(spec["maximum_n"])
    if n < initial_n or n > maximum_n:
        raise ValueError(f"artifact N={n} lies outside [{initial_n}, {maximum_n}]")
    start = int(spec["seed_start"])
    stride = int(spec["seed_stride"])
    return [start + stride * index for index in range(n)]


def _seed_set(rows: Sequence[Mapping[str, Any]]) -> set[int]:
    """Return household seeds represented by one result arm."""
    return {int(row["seed"]) for row in rows}


def _group_seed_sets(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> dict[tuple[str, ...], set[int]]:
    """Group result rows and record the seed set in every arm."""
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        grouped[key].append(row)
    return {key: _seed_set(group) for key, group in grouped.items()}


def _assert_complete_coverage(artifact: Mapping[str, Any], expected: set[int]) -> None:
    """Require every pre-specified result arm to contain the frozen households."""
    raw = artifact["results"]["raw"]
    specs = {
        "h1": ("missing_rate",),
        "h2": ("configuration",),
        "h3": ("scenario",),
        "h4": ("arm",),
        "h5": ("first_sensor", "second_sensor"),
    }
    for hypothesis in HYPOTHESES:
        rows = raw[hypothesis]
        if not rows:
            raise ValueError(f"{hypothesis} contains no rows")
        groups = _group_seed_sets(rows, specs[hypothesis])
        if not groups:
            raise ValueError(f"{hypothesis} contains no result arms")
        for key, observed in groups.items():
            if observed != expected:
                missing = sorted(expected - observed)
                extra = sorted(observed - expected)
                raise ValueError(
                    f"{hypothesis} arm {key} seed mismatch; missing={missing}, extra={extra}"
                )


def _validate_h2_decision(
    manifest: Mapping[str, Any], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the frozen H2 non-inferiority rule without affecting run validity."""
    spec = manifest["h2_primary_noninferiority"]
    reduced = str(spec["reduced_configuration"])
    summary = artifact["results"]["summaries"]["h2"][reduced]["full_minus_subset"]
    metric = str(spec["metric"])
    estimate = summary[metric]
    margin = float(spec["absolute_margin"])
    ci_high = float(estimate["ci_high"])
    return {
        "reference_configuration": spec["reference_configuration"],
        "reduced_configuration": reduced,
        "metric": metric,
        "estimate": float(estimate["mean"]),
        "ci_low": float(estimate["ci_low"]),
        "ci_high": ci_high,
        "margin": margin,
        "noninferior": ci_high < margin,
    }


def validate(manifest: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless an artifact satisfies the frozen confirmatory contract."""
    frozen_revision = str(manifest["experiment_git_revision"])
    frozen_config_sha = str(manifest["frozen_files"]["config"]["sha256"])

    if artifact.get("status") != "confirmatory":
        raise ValueError(f"artifact status is not confirmatory: {artifact.get('status')}")
    if "shard_count" not in artifact or int(artifact["shard_count"]) < 1:
        raise ValueError("confirmatory artifact was not produced by the production shard merge")
    if artifact.get("git", {}).get("commit") != frozen_revision:
        raise ValueError("artifact Git revision does not match the frozen experiment revision")
    if artifact.get("git", {}).get("dirty") is not False:
        raise ValueError("artifact Git provenance is not clean")
    if artifact.get("config_sha256") != frozen_config_sha:
        raise ValueError("artifact config SHA-256 does not match the freeze manifest")

    seeds = [int(seed) for seed in artifact.get("seeds", [])]
    if len(seeds) != len(set(seeds)):
        raise ValueError("artifact contains duplicate household seeds")
    expected = _expected_seeds(manifest, len(seeds))
    if seeds != expected:
        raise ValueError("artifact seed sequence does not exactly match the frozen stride sequence")

    _assert_complete_coverage(artifact, set(expected))

    mcse = artifact["results"]["mcse_gate"]
    target = float(manifest["precision"]["mcse_target"])
    if not bool(mcse.get("passed")):
        raise ValueError("artifact Monte Carlo precision gate did not pass")
    if float(mcse.get("target")) != target:
        raise ValueError("artifact MCSE target differs from the frozen target")
    if float(mcse.get("maximum_observed_mcse")) > target:
        raise ValueError("artifact maximum observed MCSE exceeds the frozen target")

    guardrail = str(artifact.get("interpretation_guardrail", "")).lower()
    if "simulator" not in guardrail or "field performance" not in guardrail:
        raise ValueError("artifact is missing the simulator-only interpretation guardrail")

    environment = artifact.get("environment", {})
    required_environment = manifest["reference_environment"]
    for package, version in required_environment.items():
        if str(environment.get(package)) != str(version):
            raise ValueError(
                f"artifact environment mismatch for {package}: "
                f"{environment.get(package)!r} != {version!r}"
            )

    return {
        "valid_confirmatory_artifact": True,
        "experiment_git_revision": frozen_revision,
        "config_sha256": frozen_config_sha,
        "replications": len(seeds),
        "shard_count": int(artifact["shard_count"]),
        "h2_primary_noninferiority": _validate_h2_decision(manifest, artifact),
    }


def main() -> None:
    """Validate one merged confirmatory artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    result = validate(_load(args.manifest), _load(args.artifact))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
