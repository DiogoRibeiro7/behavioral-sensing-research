"""Unified command-line interface for sensor modeling."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import timedelta
from pathlib import Path

from .models.bernoulli_ar.base_model import BernoulliAutoregressiveModel
from .models.nhpp_pelt.model import NHPPPELT, NHPPConfig
from .utils.data_io import SensorDataset
from .utils.logging_config import setup_logging


def _add_model_parsers(sub: argparse._SubParsersAction) -> None:
    """Register the single-model fitting commands."""
    ar_p = sub.add_parser("bernoulli-ar", help="Run Bernoulli autoregressive model")
    ar_p.add_argument("data", help="Path to CSV sensor data")
    ar_p.add_argument("target", help="Target sensor column")

    nhpp_p = sub.add_parser("nhpp-pelt", help="Run NHPP-PELT changepoint model")
    nhpp_p.add_argument("data", help="Path to CSV sensor data")
    nhpp_p.add_argument("sensor", help="Sensor column to use")


def _add_demo_parser(sub: argparse._SubParsersAction) -> None:
    """Register the reproducible end-to-end demonstration."""
    demo = sub.add_parser(
        "demo",
        help="Run the end-to-end ambient sensing demonstration",
        description=(
            "Simulate a synthetic household with visitors, sensor faults and a "
            "known behavioural change, run the full inference pipeline over the "
            "degraded record, and report what was and was not recovered."
        ),
    )
    demo.add_argument("--days", type=int, default=90, help="Days to simulate")
    demo.add_argument("--seed", type=int, default=20240304, help="Random seed")
    demo.add_argument(
        "--step-minutes", type=int, default=10, help="Inference step in minutes"
    )
    demo.add_argument(
        "--output", type=Path, default=None, help="Write structured results as JSON"
    )


def _add_ablation_parser(sub: argparse._SubParsersAction) -> None:
    """Register the paired sensor-ablation experiment."""
    ablate = sub.add_parser(
        "ablate",
        help="Run the paired sensor-ablation experiment",
        description=(
            "Evaluate several sensor configurations on identical simulated "
            "households and report how much each modality contributes."
        ),
    )
    ablate.add_argument("--days", type=int, default=14, help="Days per household")
    ablate.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33, 44],
        help="Seeds to pair over",
    )
    ablate.add_argument(
        "--step-minutes", type=int, default=10, help="Inference step in minutes"
    )
    ablate.add_argument(
        "--metric",
        default="balanced_accuracy",
        help="Metric to compare configurations on",
    )
    ablate.add_argument(
        "--output", type=Path, default=None, help="Write structured results as JSON"
    )


def _add_attribution_parser(sub: argparse._SubParsersAction) -> None:
    """Register the attribution comparison experiment."""
    study = sub.add_parser(
        "attribution",
        help="Compare naive against occupancy-aware activity attribution",
        description=(
            "Run the same simulated households twice -- once attributing all "
            "ambient activity to the resident, once discounting it by the "
            "probability the resident generated it -- and report the difference."
        ),
    )
    study.add_argument("--days", type=int, default=10, help="Days per scenario")
    study.add_argument("--seed", type=int, default=4242, help="Random seed")
    study.add_argument(
        "--step-minutes", type=int, default=15, help="Inference step in minutes"
    )
    study.add_argument(
        "--output", type=Path, default=None, help="Write structured results as JSON"
    )


def _run_attribution(args: argparse.Namespace) -> None:
    """Dispatch the attribution comparison experiment."""
    from .evaluation.attribution import run_attribution_study, standard_scenarios

    study = run_attribution_study(
        standard_scenarios(days=args.days, seed=args.seed),
        step=timedelta(minutes=args.step_minutes),
    )

    print("\nNaive against occupancy-aware attribution")
    print("Both arms see identical households, visitors and sensor records.\n")
    print(
        f"{'scenario':<28}{'contam':>8}{'naive':>8}{'aware':>8}"
        f"{'gain':>8}{'calib':>8}{'visF1':>7}"
    )
    for comparison in study.comparisons:
        print(
            f"{comparison.scenario:<28}"
            f"{comparison.contaminated_fraction:>8.3f}"
            f"{comparison.naive.states.balanced_accuracy:>8.3f}"
            f"{comparison.occupancy_aware.states.balanced_accuracy:>8.3f}"
            f"{comparison.balanced_accuracy_gain:>+8.3f}"
            f"{comparison.calibration_gain:>+8.3f}"
            f"{comparison.visitor_detection.f1:>7.2f}"
        )

    contaminated = study.contaminated
    if contaminated:
        mean_gain = sum(c.balanced_accuracy_gain for c in contaminated) / len(
            contaminated
        )
        print(
            "\nMean balanced-accuracy gain where another person was present: "
            f"{mean_gain:+.4f}"
        )
    print(
        "\nA gain of zero in an uncontaminated scenario is the expected result: "
        "attribution should be a no-op when nobody else is in the home."
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(study.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        print(f"Structured results written to {args.output}")


def _run_demo(args: argparse.Namespace) -> None:
    """Dispatch the end-to-end demonstration."""
    from .examples.demos.ambient_pipeline_demo import run_demo

    run_demo(
        days=args.days,
        seed=args.seed,
        step=timedelta(minutes=args.step_minutes),
        output=args.output,
    )


#: Sensor configurations compared by the ``ablate`` command. Each is a
#: plausible real deployment rather than an arbitrary subset, so the
#: comparison answers a question a designer would actually ask.
ABLATION_SUBSETS: dict[str, tuple[str, ...]] = {
    "all_modalities": (
        "front_door",
        "bedroom_motion",
        "bathroom_motion",
        "kitchen_motion",
        "living_motion",
        "fridge_contact",
        "bed_pressure",
        "living_radar",
        "wearable_motion",
        "resident_beacon",
    ),
    "object_sensors_only": (
        "front_door",
        "bedroom_motion",
        "bathroom_motion",
        "kitchen_motion",
        "living_motion",
        "fridge_contact",
    ),
    "objects_plus_wearable": (
        "front_door",
        "bedroom_motion",
        "bathroom_motion",
        "kitchen_motion",
        "living_motion",
        "fridge_contact",
        "wearable_motion",
        "resident_beacon",
    ),
    "radar_door_bed": ("front_door", "living_radar", "bed_pressure"),
    "radar_door_bed_wearable": (
        "front_door",
        "living_radar",
        "bed_pressure",
        "wearable_motion",
        "resident_beacon",
    ),
    "minimal_door_bed": ("front_door", "bed_pressure"),
}


def _run_ablation(args: argparse.Namespace) -> None:
    """Dispatch the paired sensor-ablation experiment."""
    from .evaluation.ablation import named_subsets, run_ablation
    from .simulation.household import HouseholdConfig

    report = run_ablation(
        named_subsets(ABLATION_SUBSETS),
        seeds=args.seeds,
        household=HouseholdConfig(days=args.days),
        step=timedelta(minutes=args.step_minutes),
    )

    summary = report.summary(args.metric)
    print(f"\nPaired sensor ablation over seeds {report.seeds}")
    print(f"Metric: {args.metric}\n")
    print(
        f"{'configuration':<26}{'sensors':>8}{'mean':>9}{'sd':>8}{'min':>8}{'max':>8}"
    )
    for name, stats in summary.items():
        print(
            f"{name:<26}{int(stats['n_sensors']):>8}{stats['mean']:>9.3f}"
            f"{stats['sd']:>8.3f}{stats['min']:>8.3f}{stats['max']:>8.3f}"
        )

    reference = next(iter(summary))
    print(f"\nPaired differences against '{reference}' (same households):")
    for name in report.configurations:
        if name == reference:
            continue
        difference = report.compare(reference, name, metric=args.metric)
        verdict = "clear" if difference.excludes_zero else "not distinguishable"
        print(
            f"  {reference} - {name:<26} {difference.mean_difference:+.3f}  "
            f"95% CI [{difference.ci_low:+.3f}, {difference.ci_high:+.3f}]  "
            f"dz={difference.effect_size:+.2f}  {verdict}"
        )

    print(
        "\nDifferences are paired: every configuration was evaluated on the same\n"
        "simulated households, so these compare sensing rather than residents."
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_dict(args.metric), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Structured results written to {args.output}")


def _run_model(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Dispatch the single-model fitting commands."""
    dataset = SensorDataset.from_csv(args.data)
    if args.model == "bernoulli-ar":
        model = BernoulliAutoregressiveModel(list(dataset.data.columns), args.target)
        model.fit(dataset)
    elif args.model == "nhpp-pelt":
        nhpp = NHPPPELT(NHPPConfig())
        nhpp.fit(dataset, sensor=args.sensor)
    else:  # pragma: no cover - argparse rejects unknown commands first
        parser.print_help()


def main() -> None:
    """Run the sensor modeling command-line interface."""
    parser = argparse.ArgumentParser(description="Sensor modeling CLI")
    sub = parser.add_subparsers(dest="model", required=True)
    _add_model_parsers(sub)
    _add_demo_parser(sub)
    _add_ablation_parser(sub)
    _add_attribution_parser(sub)

    args = parser.parse_args()
    setup_logging()
    logging.getLogger("sensor_modeling").setLevel(logging.WARNING)

    if args.model == "demo":
        _run_demo(args)
    elif args.model == "ablate":
        _run_ablation(args)
    elif args.model == "attribution":
        _run_attribution(args)
    else:
        _run_model(args, parser)


if __name__ == "__main__":
    main()
