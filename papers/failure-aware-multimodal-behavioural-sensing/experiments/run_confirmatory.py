"""Executable confirmatory experiment for the multimodal sensing paper.

This runner is intentionally paper-scoped. It orchestrates the public
``sensor_modeling`` APIs but does not add manuscript-specific logic to the core
library. Every comparison is paired by household seed. Time points within a
household are observations, not independent replications.

The default configuration is pre-specified in ``config.json``. The runner can
be smoke-tested with fewer replications, but a result is marked confirmatory
only when the configured minimum replication count and Monte Carlo precision
criteria are satisfied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sensor_modeling.evaluation import (
    Scenario,
    compare_scenario,
    named_subsets,
    state_metrics,
)
from sensor_modeling.health.monitor import SensorHealthReport, SystemHealthReport
from sensor_modeling.online import BehaviouralSensingPipeline, PipelineConfig
from sensor_modeling.simulation import DegradationConfig, HouseholdConfig, degrade, dropout, simulate

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_OUTPUT = HERE / "results"


class _UnitReliabilityHealthMonitor:
    """Wrap a real health monitor but make fusion ignore its reliability scores.

    The wrapped monitor still observes the stream and emits the same health
    statuses. Only the numerical reliability handed to fusion is replaced by
    one. This defines the H4 naive control: the apparatus can be known to be
    failing, but behavioural inference acts as though every sensor were fully
    trustworthy.
    """

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def observe_many(self, observations: Iterable[Any]) -> None:
        self._wrapped.observe_many(observations)

    def report(self, moment: Any) -> SystemHealthReport:
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
        return self._wrapped.snapshot()

    def restore(self, snapshot: Mapping[str, object]) -> None:
        self._wrapped.restore(snapshot)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool | str:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _seeds(config: Mapping[str, Any], override: int | None) -> list[int]:
    specification = config["replications"]
    n = override if override is not None else int(specification["minimum"])
    start = int(specification["seed_start"])
    return list(range(start, start + n))


def _pipeline_metrics(
    household_result: Any,
    observations: Sequence[Any],
    *,
    step: timedelta,
    health_aware: bool = True,
) -> dict[str, object]:
    pipeline = BehaviouralSensingPipeline(
        household_result.registry,
        config=PipelineConfig(tz=household_result.config.tz, step=step),
    )
    if not health_aware:
        pipeline.health = _UnitReliabilityHealthMonitor(pipeline.health)

    steps = pipeline.run(observations)
    steps.extend(pipeline.close(household_result.end))
    truth = household_result.truth.states_at([item.at for item in steps])
    return state_metrics(truth, [item.state for item in steps]).to_dict()


def _metric_values(rows: Sequence[Mapping[str, Any]], metric: str) -> np.ndarray:
    return np.asarray([float(row[metric]) for row in rows], dtype=float)


def _paired_summary(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    left = np.asarray(treatment, dtype=float)
    right = np.asarray(control, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired series must be one-dimensional and equal length")
    if left.size < 2:
        raise ValueError("paired inference requires at least two household seeds")

    differences = left - right
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(resamples, differences.size), replace=True)
    means = draws.mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    sd = float(differences.std(ddof=1))
    mean = float(differences.mean())
    mcse = sd / math.sqrt(differences.size)
    return {
        "n": int(differences.size),
        "mean_difference": mean,
        "ci_low": float(np.quantile(means, tail)),
        "ci_high": float(np.quantile(means, 1.0 - tail)),
        "sd_difference": sd,
        "mcse": mcse,
        "wins": int((differences > 0).sum()),
        "losses": int((differences < 0).sum()),
        "ties": int((differences == 0).sum()),
    }


def _run_h1(
    config: Mapping[str, Any], seeds: Sequence[int], step: timedelta
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    days = int(config["household"]["days"])
    for seed in seeds:
        household = simulate(HouseholdConfig(days=days, seed=seed))
        for rate in config["h1_missingness"]["rates"]:
            degradation = DegradationConfig(missing_rate=float(rate), seed=seed)
            observations, dropped = degrade(household.observations, degradation)
            metrics = _pipeline_metrics(household, observations, step=step)
            rows.append(
                {
                    "seed": seed,
                    "missing_rate": float(rate),
                    "dropped_records": len(dropped),
                    **metrics,
                }
            )
    return {"rows": rows}


def _run_h2(
    config: Mapping[str, Any], seeds: Sequence[int], step: timedelta
) -> dict[str, Any]:
    subsets = named_subsets(config["h2_sensor_subsets"])
    rows: list[dict[str, Any]] = []
    days = int(config["household"]["days"])
    for seed in seeds:
        household = simulate(HouseholdConfig(days=days, seed=seed))
        for subset in subsets:
            registry = subset.restrict(household.registry)
            observations = household.observations_for(subset.sensors)
            pipeline = BehaviouralSensingPipeline(
                registry,
                config=PipelineConfig(tz=household.config.tz, step=step),
            )
            steps = pipeline.run(observations)
            steps.extend(pipeline.close(household.end))
            truth = household.truth.states_at([item.at for item in steps])
            metrics = state_metrics(truth, [item.state for item in steps]).to_dict()
            rows.append(
                {
                    "seed": seed,
                    "configuration": subset.name,
                    "n_sensors": len(subset.sensors),
                    **metrics,
                }
            )
    return {"rows": rows}


def _scenarios_for_seed(
    config: Mapping[str, Any], seed: int
) -> list[Scenario]:
    from sensor_modeling.evaluation.attribution import standard_scenarios

    available = {
        scenario.name: scenario
        for scenario in standard_scenarios(
            days=int(config["household"]["days"]), seed=seed
        )
    }
    return [available[name] for name in config["h3_attribution_scenarios"]]


def _run_h3(
    config: Mapping[str, Any], seeds: Sequence[int], step: timedelta
) -> dict[str, Any]:
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
    return {"rows": rows}


def _run_h4(
    config: Mapping[str, Any], seeds: Sequence[int], step: timedelta
) -> dict[str, Any]:
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
        aware = _pipeline_metrics(household, observations, step=step, health_aware=True)
        naive = _pipeline_metrics(household, observations, step=step, health_aware=False)
        for arm, metrics in (("failure_aware", aware), ("health_naive", naive)):
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "dropped_records": len(dropped),
                    **metrics,
                }
            )
    return {"rows": rows}


def _run_h5(
    config: Mapping[str, Any], seeds: Sequence[int], step: timedelta
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    days = int(config["household"]["days"])
    for seed in seeds:
        household = simulate(HouseholdConfig(days=days, seed=seed))
        all_sensors = tuple(household.registry.sensor_ids())
        for first, second in config["h5_interactions"]:
            configurations = {
                "full": all_sensors,
                "without_first": tuple(s for s in all_sensors if s != first),
                "without_second": tuple(s for s in all_sensors if s != second),
                "without_both": tuple(s for s in all_sensors if s not in {first, second}),
            }
            values: dict[str, float] = {}
            for name, sensors in configurations.items():
                registry = household.registry.subset(sensors)
                observations = household.observations_for(sensors)
                pipeline = BehaviouralSensingPipeline(
                    registry,
                    config=PipelineConfig(tz=household.config.tz, step=step),
                )
                steps = pipeline.run(observations)
                steps.extend(pipeline.close(household.end))
                truth = household.truth.states_at([item.at for item in steps])
                metrics = state_metrics(truth, [item.state for item in steps])
                values[name] = metrics.balanced_accuracy
            first_value = values["full"] - values["without_first"]
            second_value = values["full"] - values["without_second"]
            joint = values["full"] - values["without_both"]
            rows.append(
                {
                    "seed": seed,
                    "first_sensor": first,
                    "second_sensor": second,
                    "interaction": joint - (first_value + second_value),
                    "joint_contribution": joint,
                    "first_contribution": first_value,
                    "second_contribution": second_value,
                }
            )
    return {"rows": rows}


def _summaries(
    config: Mapping[str, Any], results: Mapping[str, Any]
) -> dict[str, Any]:
    confidence = float(config["reporting"]["confidence_level"])
    resamples = int(config["reporting"]["bootstrap_resamples"])
    output: dict[str, Any] = {}

    h1 = results["h1"]["rows"]
    by_rate: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in h1:
        by_rate[float(row["missing_rate"])].append(row)
    output["h1"] = {
        str(rate): {
            metric: {
                "mean": float(_metric_values(rows, metric).mean()),
                "sd": float(_metric_values(rows, metric).std(ddof=1)),
            }
            for metric in config["primary_metrics"]
        }
        for rate, rows in sorted(by_rate.items())
    }

    h2 = results["h2"]["rows"]
    by_configuration: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in h2:
        by_configuration[str(row["configuration"])].append(row)
    reference = by_configuration["all_modalities"]
    output["h2"] = {}
    for name, rows in by_configuration.items():
        output["h2"][name] = {
            "n_sensors": int(rows[0]["n_sensors"]),
            "balanced_accuracy_mean": float(
                _metric_values(rows, "balanced_accuracy").mean()
            ),
        }
        if name != "all_modalities":
            output["h2"][name]["gap_from_full"] = _paired_summary(
                _metric_values(reference, "balanced_accuracy"),
                _metric_values(rows, "balanced_accuracy"),
                confidence=confidence,
                resamples=resamples,
                seed=101,
            )

    h3 = results["h3"]["rows"]
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in h3:
        by_scenario[str(row["scenario"])].append(row)
    output["h3"] = {
        name: {
            "balanced_accuracy_gain_mean": float(
                _metric_values(rows, "balanced_accuracy_gain").mean()
            ),
            "calibration_gain_mean": float(
                _metric_values(rows, "calibration_gain").mean()
            ),
            "visitor_f1_mean": float(_metric_values(rows, "visitor_f1").mean()),
        }
        for name, rows in by_scenario.items()
    }

    h4 = results["h4"]["rows"]
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in h4:
        by_arm[str(row["arm"])].append(row)
    output["h4"] = {
        metric: _paired_summary(
            _metric_values(by_arm["failure_aware"], metric),
            _metric_values(by_arm["health_naive"], metric),
            confidence=confidence,
            resamples=resamples,
            seed=202,
        )
        for metric in config["primary_metrics"]
    }

    h5 = results["h5"]["rows"]
    by_pair: dict[str, list[float]] = defaultdict(list)
    for row in h5:
        pair = f"{row['first_sensor']} + {row['second_sensor']}"
        by_pair[pair].append(float(row["interaction"]))
    output["h5"] = {
        pair: {
            "mean_interaction": float(np.mean(values)),
            "sd_interaction": float(np.std(values, ddof=1)),
        }
        for pair, values in by_pair.items()
    }
    return output


def _mcse_gate(config: Mapping[str, Any], results: Mapping[str, Any]) -> dict[str, Any]:
    target = float(config["replications"]["monte_carlo_se_target"])
    h2 = results["summaries"]["h2"]
    observed = [
        float(entry["gap_from_full"]["mcse"])
        for entry in h2.values()
        if "gap_from_full" in entry
    ]
    maximum = max(observed) if observed else float("inf")
    return {
        "target": target,
        "maximum_primary_h2_mcse": maximum,
        "passed": maximum <= target,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    columns = list(scalar_rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(scalar_rows)


def _latex_macros(summary: Mapping[str, Any]) -> str:
    h4 = summary["h4"]["balanced_accuracy"]
    lines = [
        "% Generated by experiments/run_confirmatory.py. Do not edit by hand.",
        f"\\newcommand{{\\HFourMeanDifference}}{{{h4['mean_difference']:.3f}}}",
        f"\\newcommand{{\\HFourCILow}}{{{h4['ci_low']:.3f}}}",
        f"\\newcommand{{\\HFourCIHigh}}{{{h4['ci_high']:.3f}}}",
        f"\\newcommand{{\\HFourMCSE}}{{{h4['mcse']:.4f}}}",
    ]
    return "\n".join(lines) + "\n"


def run(config: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    step = timedelta(minutes=int(config["household"]["step_minutes"]))
    results: dict[str, Any] = {
        "h1": _run_h1(config, seeds, step),
        "h2": _run_h2(config, seeds, step),
        "h3": _run_h3(config, seeds, step),
        "h4": _run_h4(config, seeds, step),
        "h5": _run_h5(config, seeds, step),
    }
    results["summaries"] = _summaries(config, results)
    results["mcse_gate"] = _mcse_gate(config, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--replications",
        type=int,
        default=None,
        help="Override replication count for smoke tests; does not alter config.json.",
    )
    args = parser.parse_args()

    config = _load(args.config)
    seeds = _seeds(config, args.replications)
    args.output.mkdir(parents=True, exist_ok=True)

    results = run(config, seeds)
    minimum = int(config["replications"]["minimum"])
    confirmatory = len(seeds) >= minimum and bool(results["mcse_gate"]["passed"])
    artifact = {
        "experiment_schema_version": int(config["schema_version"]),
        "status": "confirmatory" if confirmatory else "pilot-or-incomplete",
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "resolved_config": config,
        "seeds": list(seeds),
        "results": results,
        "interpretation_guardrail": (
            "All outcomes are simulator-derived and are not estimates of field performance."
        ),
    }
    (args.output / "confirmatory_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    for hypothesis in ("h1", "h2", "h3", "h4", "h5"):
        _write_csv(args.output / f"{hypothesis}.csv", results[hypothesis]["rows"])
    (args.output / "generated_results.tex").write_text(
        _latex_macros(results["summaries"]), encoding="utf-8"
    )

    print(f"Wrote paper experiment outputs to {args.output}")
    print(f"Replications: {len(seeds)}")
    print(f"MCSE gate: {results['mcse_gate']}")
    print(f"Status: {artifact['status']}")


if __name__ == "__main__":
    main()
