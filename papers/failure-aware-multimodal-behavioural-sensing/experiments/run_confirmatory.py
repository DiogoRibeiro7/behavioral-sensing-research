"""Execute the pre-specified simulation study for the behavioural-sensing paper.

This module is intentionally paper-scoped. It orchestrates the public
``sensor_modeling`` APIs without adding manuscript-specific behaviour to the
library. Household trajectories, rather than time points, are the independent
replications and all comparative designs are paired by household seed.

A short run may be used as a smoke test. The output is labelled
``confirmatory`` only when the configured minimum replication count and Monte
Carlo precision criterion are satisfied on a clean, identifiable Git commit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from sensor_modeling.evaluation import (
    Scenario,
    compare_scenario,
    named_subsets,
    state_metrics,
)
from sensor_modeling.evaluation.attribution import standard_scenarios
from sensor_modeling.health.monitor import SensorHealthReport, SystemHealthReport
from sensor_modeling.online import BehaviouralSensingPipeline, PipelineConfig
from sensor_modeling.simulation import (
    DegradationConfig,
    HouseholdConfig,
    degrade,
    dropout,
    simulate,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_OUTPUT = HERE / "results"
SUPPORTED_METRICS = {
    "balanced_accuracy",
    "log_loss",
    "brier",
    "calibration_error",
    "abstention_rate",
}


class _UnitReliabilityHealthMonitor:
    """Preserve health statuses while forcing fusion reliability to one."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def observe_many(self, observations: Iterable[Any]) -> None:
        """Forward observations to the real health monitor."""
        self._wrapped.observe_many(observations)

    def report(self, moment: Any) -> SystemHealthReport:
        """Return real health statuses with unit reliability weights."""
        original = self._wrapped.report(moment)
        sensors = {
            sensor_id: SensorHealthReport(
                sensor_id=report.sensor_id,
                status=report.status,
                reliability=1.0,
                last_seen=report.last_seen,
                silence=report.silence,
                observations=report.observations,
                detail=report.detail,
            )
            for sensor_id, report in original.sensors.items()
        }
        return SystemHealthReport(at=original.at, sensors=sensors)

    def snapshot(self) -> dict[str, object]:
        """Return the underlying monitor snapshot."""
        return self._wrapped.snapshot()

    def restore(self, snapshot: Mapping[str, object]) -> None:
        """Restore the underlying monitor snapshot."""
        self._wrapped.restore(snapshot)


def _validate_config(config: Mapping[str, Any]) -> None:
    """Reject malformed designs before any simulation is executed."""
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("unsupported experiment schema_version")

    replications = config["replications"]
    minimum = int(replications["minimum"])
    maximum = int(replications["maximum"])
    target = float(replications["monte_carlo_se_target"])
    if minimum < 2 or maximum < minimum:
        raise ValueError("replication bounds must satisfy 2 <= minimum <= maximum")
    if not 0.0 < target < 1.0:
        raise ValueError("monte_carlo_se_target must lie in (0, 1)")

    metrics = set(config["primary_metrics"])
    unknown_metrics = metrics - SUPPORTED_METRICS
    if unknown_metrics:
        raise ValueError(f"unsupported primary metrics: {sorted(unknown_metrics)}")

    rates = [float(rate) for rate in config["h1_missingness"]["rates"]]
    if not rates or 0.0 not in rates:
        raise ValueError("H1 missingness rates must include the 0.0 reference")
    if any(rate < 0.0 or rate > 1.0 for rate in rates):
        raise ValueError("H1 missingness rates must lie in [0, 1]")
    if len(rates) != len(set(rates)):
        raise ValueError("H1 missingness rates must be unique")

    subsets = config["h2_sensor_subsets"]
    if "all_modalities" not in subsets:
        raise ValueError("H2 requires an all_modalities reference configuration")
    if any(not sensors for sensors in subsets.values()):
        raise ValueError("every H2 sensor configuration must contain a sensor")

    reporting = config["reporting"]
    required_guards = (
        "paired_by_seed",
        "time_points_are_not_replications",
        "field_performance_claims_forbidden",
    )
    if not all(bool(reporting.get(name)) for name in required_guards):
        raise ValueError("scientific reporting guardrails must remain enabled")


def _load(path: Path) -> dict[str, Any]:
    """Read and validate the experiment configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    _validate_config(config)
    return config


def _config_sha256(path: Path) -> str:
    """Return the SHA-256 digest of the exact experiment configuration."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    """Return one Git command result or ``unknown`` when Git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_provenance() -> dict[str, object]:
    """Return commit and pre-execution worktree state."""
    commit = _git_value("rev-parse", "HEAD")
    status = _git_value("status", "--porcelain")
    dirty: bool | str = "unknown" if status == "unknown" else bool(status)
    return {"commit": commit, "dirty": dirty}


def _environment() -> dict[str, str]:
    """Return the numerical software environment."""
    packages = ("sensor-modeling", "numpy", "scipy", "pandas")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **versions,
    }


def _seeds(config: Mapping[str, Any], override: int | None) -> list[int]:
    """Resolve the exact independent household seeds."""
    specification = config["replications"]
    minimum = int(specification["minimum"])
    maximum = int(specification["maximum"])
    n = minimum if override is None else int(override)
    if n < 2:
        raise ValueError("at least two replications are required")
    if n > maximum:
        raise ValueError(f"replications={n} exceeds the maximum of {maximum}")
    start = int(specification["seed_start"])
    return list(range(start, start + n))


def _pipeline_metrics(
    household_result: Any,
    observations: Sequence[Any],
    *,
    step: timedelta,
    health_aware: bool = True,
) -> dict[str, object]:
    """Run the online path and return state metrics against simulator truth."""
    pipeline = BehaviouralSensingPipeline(
        household_result.registry,
        config=PipelineConfig(tz=household_result.config.tz, step=step),
    )
    if not health_aware:
        pipeline.health = _UnitReliabilityHealthMonitor(pipeline.health)
    steps = pipeline.run(observations)
    steps.extend(pipeline.close(household_result.end))
    if not steps:
        raise ValueError("pipeline produced no scored steps")
    truth = household_result.truth.states_at([item.at for item in steps])
    return state_metrics(truth, [item.state for item in steps]).to_dict()


def _subset_metrics(
    household: Any, sensors: Sequence[str], *, step: timedelta
) -> dict[str, object]:
    """Score one sensor subset on one already-simulated household."""
    registry = household.registry.subset(sensors)
    observations = household.observations_for(sensors)
    pipeline = BehaviouralSensingPipeline(
        registry,
        config=PipelineConfig(tz=household.config.tz, step=step),
    )
    steps = pipeline.run(observations)
    steps.extend(pipeline.close(household.end))
    if not steps:
        raise ValueError("sensor subset produced no scored steps")
    truth = household.truth.states_at([item.at for item in steps])
    return state_metrics(truth, [item.state for item in steps]).to_dict()


def _run_h1(
    config: Mapping[str, Any], seeds: Sequence[int], step: timedelta
) -> list[dict[str, Any]]:
    """H1: quantify degradation as random observations are withheld."""
    rows: list[dict[str, Any]] = []
    days = int(config["household"]["days"])
    for seed in seeds:
        household = simulate(HouseholdConfig(days=days, seed=seed))
        for rate in config["h1_missingness"]["rates"]:
            degradation = DegradationConfig(missing_rate=float(rate), seed=seed)
            observations, dropped = degrade(household.observations, degradation)
            rows.append(
                {
                    "seed": seed,
                    "missing_rate": float(rate),
                    "dropped_records": len(dropped),
                    **_pipeline_metrics(household, observations, step=step),
                }
            )
    return rows


def _run_h2(
    config: Mapping[str, Any], seeds: Sequence[int], step: timedelta
) -> list[dict[str, Any]]:
    """H2: evaluate pre-specified sensor subsets on paired trajectories."""
    subsets = named_subsets(config["h2_sensor_subsets"])
    rows: list[dict[str, Any]] = []
    days = int(config["household"]["days"])
    for seed in seeds:
        household = simulate(HouseholdConfig(days=days, seed=seed))
        for subset in subsets:
            rows.append(
                {
                    "seed": seed,
                    "configuration": subset.name,
                    "n_sensors": len(subset.sensors),
                    **_subset_metrics(household, subset.sensors, step=step),
                }
            )
    return rows


def _scenarios_for_seed(config: Mapping[str, Any], seed: int) -> list[Scenario]:
    """Resolve the pre-specified H3 scenarios for one seed."""
    available = {
        scenario.name: scenario
        for scenario in standard_scenarios(
            days=int(config["household"]["days"]), seed=seed
        )
    }
    requested = list(config["h3_attribution_scenarios"])
    missing = set(requested) - set(available)
    if missing:
        raise ValueError(f"unknown H3 scenarios: {sorted(missing)}")
    return [available[name] for name in requested]


def _run_h3(
    config: Mapping[str, Any], seeds: Sequence[int], step: timedelta
) -> list[dict[str, Any]]:
    """H3: compare occupancy-aware and naive attribution within seed."""
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for scenario in _scenarios_for_seed(config, seed):
            comparison = compare_scenario(scenario, step=step)
            rows.append(
                {
                    "seed": seed,
                    "scenario": comparison.scenario,
                    "contaminated_fraction": comparison.contaminated_fraction,
                    "balanced_accuracy_gain": comparison.balanced_accuracy_gain,
                    "calibration_gain": comparison.calibration_gain,
                    "visitor_precision": comparison.visitor_detection.precision,
                    "visitor_recall": comparison.visitor_detection.recall,
                    "visitor_f1": comparison.visitor_detection.f1,
                }
            )
    return rows


def _run_h4(
    config: Mapping[str, Any], seeds: Sequence[int], step: timedelta
) -> list[dict[str, Any]]:
    """H4: compare failure-aware and health-naive inference within seed."""
    rows: list[dict[str, Any]] = []
    days = int(config["household"]["days"])
    specification = config["h4_failures"]
    for seed in seeds:
        household = simulate(HouseholdConfig(days=days, seed=seed))
        fault = dropout(
            str(specification["sensor_dropout"]),
            household.start + timedelta(days=int(specification["dropout_start_day"])),
            timedelta(days=int(specification["dropout_days"])),
        )
        degradation = DegradationConfig(
            missing_rate=float(specification["missing_rate"]),
            faults=(fault,),
            seed=seed,
        )
        observations, dropped = degrade(household.observations, degradation)
        arms = {
            "failure_aware": _pipeline_metrics(
                household, observations, step=step, health_aware=True
            ),
            "health_naive": _pipeline_metrics(
                household, observations, step=step, health_aware=False
            ),
        }
        for arm, metrics in arms.items():
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "dropped_records": len(dropped),
                    **metrics,
                }
            )
    return rows


def _run_h5(
    config: Mapping[str, Any], seeds: Sequence[int], step: timedelta
) -> list[dict[str, Any]]:
    """H5: estimate non-additive sensor interactions within seed."""
    rows: list[dict[str, Any]] = []
    days = int(config["household"]["days"])
    for seed in seeds:
        household = simulate(HouseholdConfig(days=days, seed=seed))
        all_sensors = tuple(household.registry.sensor_ids())
        for first, second in config["h5_interactions"]:
            unknown = {first, second} - set(all_sensors)
            if unknown:
                raise ValueError(f"unknown H5 sensors: {sorted(unknown)}")
            sensor_sets = {
                "full": all_sensors,
                "without_first": tuple(s for s in all_sensors if s != first),
                "without_second": tuple(s for s in all_sensors if s != second),
                "without_both": tuple(
                    s for s in all_sensors if s not in {first, second}
                ),
            }
            values = {
                name: float(
                    _subset_metrics(household, sensors, step=step)[
                        "balanced_accuracy"
                    ]
                )
                for name, sensors in sensor_sets.items()
            }
            first_value = values["full"] - values["without_first"]
            second_value = values["full"] - values["without_second"]
            joint = values["full"] - values["without_both"]
            rows.append(
                {
                    "seed": seed,
                    "first_sensor": first,
                    "second_sensor": second,
                    "interaction": joint - first_value - second_value,
                    "joint_contribution": joint,
                    "first_contribution": first_value,
                    "second_contribution": second_value,
                }
            )
    return rows


def _values(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    """Extract one scalar field as a numerical vector."""
    return np.asarray([float(row[field]) for row in rows], dtype=float)


def _sample_summary(
    values: Sequence[float], *, confidence: float, resamples: int, seed: int
) -> dict[str, float | int]:
    """Summarise household-level values with a bootstrap mean interval."""
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size < 2:
        raise ValueError("uncertainty summaries require at least two households")
    rng = np.random.default_rng(seed)
    draws = rng.choice(sample, size=(resamples, sample.size), replace=True)
    means = draws.mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    sd = float(sample.std(ddof=1))
    return {
        "n": int(sample.size),
        "mean": float(sample.mean()),
        "ci_low": float(np.quantile(means, tail)),
        "ci_high": float(np.quantile(means, 1.0 - tail)),
        "sd": sd,
        "mcse": sd / math.sqrt(sample.size),
    }


def _paired_summary(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
    contrast: str,
) -> dict[str, float | int | str]:
    """Summarise a household-paired treatment-minus-control contrast."""
    left = np.asarray(treatment, dtype=float)
    right = np.asarray(control, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired series must be one-dimensional and equal length")
    differences = left - right
    summary = _sample_summary(
        differences,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    return {
        **summary,
        "contrast": contrast,
        "wins": int((differences > 0).sum()),
        "losses": int((differences < 0).sum()),
        "ties": int((differences == 0).sum()),
    }


def _group_by(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, list[Mapping[str, Any]]]:
    """Group rows by one scalar field."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return grouped


def _summaries(
    config: Mapping[str, Any], results: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    """Build manuscript-facing household-level summaries."""
    confidence = float(config["reporting"]["confidence_level"])
    resamples = int(config["reporting"]["bootstrap_resamples"])
    metrics = list(config["primary_metrics"])
    output: dict[str, Any] = {}

    h1_groups = _group_by(results["h1"], "missing_rate")
    h1_reference = h1_groups["0.0"]
    output["h1"] = {}
    for rate_index, (rate, rows) in enumerate(h1_groups.items()):
        entry: dict[str, Any] = {
            metric: _sample_summary(
                _values(rows, metric),
                confidence=confidence,
                resamples=resamples,
                seed=100 + rate_index * 10 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }
        if rate != "0.0":
            entry["change_from_zero"] = {
                metric: _paired_summary(
                    _values(rows, metric),
                    _values(h1_reference, metric),
                    confidence=confidence,
                    resamples=resamples,
                    seed=200 + rate_index * 10 + metric_index,
                    contrast="missingness_minus_zero",
                )
                for metric_index, metric in enumerate(metrics)
            }
        output["h1"][rate] = entry

    h2_groups = _group_by(results["h2"], "configuration")
    full = h2_groups["all_modalities"]
    output["h2"] = {}
    for config_index, (name, rows) in enumerate(h2_groups.items()):
        entry = {
            "n_sensors": int(rows[0]["n_sensors"]),
            "metrics": {
                metric: _sample_summary(
                    _values(rows, metric),
                    confidence=confidence,
                    resamples=resamples,
                    seed=300 + config_index * 10 + metric_index,
                )
                for metric_index, metric in enumerate(metrics)
            },
        }
        if name != "all_modalities":
            entry["full_minus_subset"] = {
                metric: _paired_summary(
                    _values(full, metric),
                    _values(rows, metric),
                    confidence=confidence,
                    resamples=resamples,
                    seed=400 + config_index * 10 + metric_index,
                    contrast="full_minus_subset",
                )
                for metric_index, metric in enumerate(metrics)
            }
        output["h2"][name] = entry

    h3_groups = _group_by(results["h3"], "scenario")
    output["h3"] = {
        name: {
            field: _sample_summary(
                _values(rows, field),
                confidence=confidence,
                resamples=resamples,
                seed=500 + scenario_index * 10 + field_index,
            )
            for field_index, field in enumerate(
                ("balanced_accuracy_gain", "calibration_gain", "visitor_f1")
            )
        }
        for scenario_index, (name, rows) in enumerate(h3_groups.items())
    }

    h4_groups = _group_by(results["h4"], "arm")
    output["h4"] = {
        metric: _paired_summary(
            _values(h4_groups["failure_aware"], metric),
            _values(h4_groups["health_naive"], metric),
            confidence=confidence,
            resamples=resamples,
            seed=600 + metric_index,
            contrast="failure_aware_minus_health_naive",
        )
        for metric_index, metric in enumerate(metrics)
    }

    h5_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in results["h5"]:
        h5_groups[f"{row['first_sensor']} + {row['second_sensor']}"].append(row)
    output["h5"] = {
        pair: _sample_summary(
            _values(rows, "interaction"),
            confidence=confidence,
            resamples=resamples,
            seed=700 + pair_index,
        )
        for pair_index, (pair, rows) in enumerate(h5_groups.items())
    }
    return output


def _mcse_gate(
    config: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the pre-specified precision gate to H2 balanced-accuracy gaps."""
    target = float(config["replications"]["monte_carlo_se_target"])
    observed = [
        float(entry["full_minus_subset"]["balanced_accuracy"]["mcse"])
        for entry in summary["h2"].values()
        if "full_minus_subset" in entry
    ]
    maximum = max(observed) if observed else float("inf")
    return {
        "metric": "balanced_accuracy",
        "contrast": "full_minus_subset",
        "target": target,
        "maximum_observed_mcse": maximum,
        "passed": maximum <= target,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write household-level rows."""
    if not rows:
        raise ValueError(f"refusing to write empty result table: {path.name}")
    scalar_rows = [
        {
            key: json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def _macro_name(text: str) -> str:
    """Convert an identifier to a letters-only LaTeX command suffix."""
    cleaned = text.replace("+", " ").replace("_", " ")
    return "".join(part[:1].upper() + part[1:] for part in cleaned.split())


def _latex_macros(summary: Mapping[str, Any], n: int, status: str) -> str:
    """Generate the machine-written bridge from results to the manuscript."""
    lines = [
        "% Generated by experiments/run_confirmatory.py. Do not edit by hand.",
        f"\\newcommand{{\\ConfirmatoryN}}{{{n}}}",
        f"\\newcommand{{\\ExperimentStatus}}{{{status}}}",
    ]
    h1_40 = summary["h1"].get("0.4")
    if h1_40 is not None:
        value = h1_40["balanced_accuracy"]["mean"]
        lines.append(f"\\newcommand{{\\HOneFortyMissingBA}}{{{value:.3f}}}")
    for name, entry in summary["h2"].items():
        if "full_minus_subset" in entry:
            gap = entry["full_minus_subset"]["balanced_accuracy"]["mean"]
            lines.append(
                f"\\newcommand{{\\HTwo{_macro_name(name)}Gap}}{{{gap:.3f}}}"
            )
    carer = summary["h3"].get("carer_visits")
    if carer is not None:
        value = carer["balanced_accuracy_gain"]["mean"]
        lines.append(f"\\newcommand{{\\HThreeCarerGain}}{{{value:.3f}}}")
    h4 = summary["h4"]["balanced_accuracy"]
    lines.extend(
        [
            f"\\newcommand{{\\HFourMeanDifference}}{{{h4['mean']:.3f}}}",
            f"\\newcommand{{\\HFourCILow}}{{{h4['ci_low']:.3f}}}",
            f"\\newcommand{{\\HFourCIHigh}}{{{h4['ci_high']:.3f}}}",
        ]
    )
    for pair, entry in summary["h5"].items():
        lines.append(
            f"\\newcommand{{\\HFive{_macro_name(pair)}Interaction}}"
            f"{{{entry['mean']:.3f}}}"
        )
    return "\n".join(lines) + "\n"


def run(config: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    """Execute H1-H5 and build uncertainty summaries."""
    step = timedelta(minutes=int(config["household"]["step_minutes"]))
    raw: dict[str, Sequence[Mapping[str, Any]]] = {
        "h1": _run_h1(config, seeds, step),
        "h2": _run_h2(config, seeds, step),
        "h3": _run_h3(config, seeds, step),
        "h4": _run_h4(config, seeds, step),
        "h5": _run_h5(config, seeds, step),
    }
    summary = _summaries(config, raw)
    return {
        "raw": raw,
        "summaries": summary,
        "mcse_gate": _mcse_gate(config, summary),
    }


def _confirmatory_status(
    config: Mapping[str, Any],
    seeds: Sequence[int],
    results: Mapping[str, Any],
    provenance: Mapping[str, object],
) -> tuple[str, list[str]]:
    """Return status plus explicit reasons a run is not confirmatory."""
    reasons: list[str] = []
    minimum = int(config["replications"]["minimum"])
    if len(seeds) < minimum:
        reasons.append(f"replications {len(seeds)} < minimum {minimum}")
    if not bool(results["mcse_gate"]["passed"]):
        reasons.append("primary H2 Monte Carlo precision gate not met")
    if provenance["commit"] == "unknown":
        reasons.append("Git commit could not be identified")
    if provenance["dirty"] is not False:
        reasons.append("working tree was not clean before execution")
    return ("confirmatory" if not reasons else "pilot-or-incomplete", reasons)


def main() -> None:
    """Validate, smoke-test, or execute the confirmatory experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--replications",
        type=int,
        default=None,
        help="Override N for smoke/precision-extension runs; config.json is unchanged.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the design and provenance without running simulations.",
    )
    args = parser.parse_args()

    config = _load(args.config)
    seeds = _seeds(config, args.replications)
    provenance = _git_provenance()
    if args.validate_only:
        print("Configuration valid")
        print(f"Config SHA-256: {_config_sha256(args.config)}")
        print(f"Resolved replications: {len(seeds)}")
        print(f"Git provenance: {provenance}")
        return

    results = run(config, seeds)
    status, reasons = _confirmatory_status(config, seeds, results, provenance)
    artifact = {
        "experiment_schema_version": int(config["schema_version"]),
        "status": status,
        "non_confirmatory_reasons": reasons,
        "git": provenance,
        "environment": _environment(),
        "config_sha256": _config_sha256(args.config),
        "resolved_config": config,
        "seeds": list(seeds),
        "results": results,
        "interpretation_guardrail": (
            "All outcomes are simulator-derived and are not estimates of field performance."
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "confirmatory_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    raw = results["raw"]
    for hypothesis in ("h1", "h2", "h3", "h4", "h5"):
        _write_csv(args.output / f"{hypothesis}.csv", raw[hypothesis])
    (args.output / "generated_results.tex").write_text(
        _latex_macros(results["summaries"], len(seeds), status), encoding="utf-8"
    )

    print(f"Wrote paper experiment outputs to {args.output}")
    print(f"Replications: {len(seeds)}")
    print(f"MCSE gate: {results['mcse_gate']}")
    print(f"Status: {status}")
    for reason in reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    main()