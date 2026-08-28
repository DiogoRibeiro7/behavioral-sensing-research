"""Sensor-ablation experiments: what is each modality actually worth?

The research question this package exists to answer is whether useful
behavioural inference survives with fewer physical sensors. Answering it needs
more discipline than running a few configurations and comparing numbers.

Three things are built in rather than left to the user to remember.

*The design is paired.* Every configuration is evaluated on identical
simulated trajectories. Simulated households differ from each other far more
than two sensor configurations differ on one household, so an unpaired
comparison buries a real effect under between-household variance.

*Ablation removes sensors, not code.* A configuration is a subset of the
registry. The pipeline is constructed from that subset exactly as it would be
from a full deployment, so an ablated run exercises the same inference path a
real sparse deployment would.

*Marginal value is not assumed additive.* Two sensors that each look
worthless alone can be jointly essential -- a door tells you little without
something to say who walked through it. :func:`interaction` reports how far
the joint contribution departs from the sum of the individual ones, so that
departure is measured rather than assumed away.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from ..observations.registry import SensorRegistry
from ..online.pipeline import BehaviouralSensingPipeline, PipelineConfig
from ..simulation.faults import DegradationConfig, degrade
from ..simulation.household import HouseholdConfig, simulate
from .metrics import PairedDifference, StateMetrics, paired_difference, state_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensorConfiguration:
    """A named subset of a deployment to evaluate."""

    name: str
    sensors: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if not self.name.strip():
            raise ValueError("a configuration needs a name")
        if not self.sensors:
            raise ValueError(f"configuration '{self.name}' has no sensors")

    def restrict(self, registry: SensorRegistry) -> SensorRegistry:
        """Return the registry restricted to this configuration."""
        return registry.subset(self.sensors)


@dataclass(frozen=True)
class AblationRun:
    """The outcome of one configuration on one simulated trajectory."""

    configuration: str
    seed: int
    metrics: StateMetrics
    sensors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the run."""
        return {
            "configuration": self.configuration,
            "seed": self.seed,
            "sensors": list(self.sensors),
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class AblationReport:
    """Results of a paired ablation sweep."""

    runs: list[AblationRun] = field(default_factory=list)

    @property
    def configurations(self) -> list[str]:
        """Names of the configurations evaluated, in first-seen order."""
        seen: dict[str, None] = {}
        for run in self.runs:
            seen.setdefault(run.configuration, None)
        return list(seen)

    @property
    def seeds(self) -> list[int]:
        """Seeds evaluated, sorted."""
        return sorted({run.seed for run in self.runs})

    def series(
        self, configuration: str, metric: str = "balanced_accuracy"
    ) -> list[float]:
        """Return one configuration's scores, ordered by seed.

        Ordering by seed is what keeps the pairing intact: position ``i`` in
        two configurations' series refers to the same simulated household.
        """
        by_seed = {
            run.seed: getattr(run.metrics, metric)
            for run in self.runs
            if run.configuration == configuration
        }
        missing = set(self.seeds) - set(by_seed)
        if missing:
            raise ValueError(
                f"configuration '{configuration}' is missing seeds {sorted(missing)}; "
                "the paired design requires every configuration on every seed"
            )
        return [by_seed[seed] for seed in self.seeds]

    def compare(
        self,
        treatment: str,
        control: str,
        *,
        metric: str = "balanced_accuracy",
        seed: int = 0,
    ) -> PairedDifference:
        """Compare two configurations on the trajectories they share."""
        return paired_difference(
            self.series(treatment, metric),
            self.series(control, metric),
            seed=seed,
        )

    def marginal(
        self,
        full: str,
        without: str,
        *,
        metric: str = "balanced_accuracy",
    ) -> PairedDifference:
        """Return the value lost by removing a sensor from *full*.

        This is ``performance(S) - performance(S without j)``, evaluated on
        paired trajectories.
        """
        return self.compare(full, without, metric=metric)

    def interaction(
        self,
        full: str,
        without_first: str,
        without_second: str,
        without_both: str,
        *,
        metric: str = "balanced_accuracy",
    ) -> float:
        """Return how far two sensors' joint value departs from additivity.

        Positive means the pair is worth more together than separately --
        they are complementary. Negative means they are redundant, each
        covering for the other's absence.
        """
        scores = {
            name: float(np.mean(self.series(name, metric)))
            for name in (full, without_first, without_second, without_both)
        }
        first = scores[full] - scores[without_first]
        second = scores[full] - scores[without_second]
        joint = scores[full] - scores[without_both]
        return joint - (first + second)

    def summary(self, metric: str = "balanced_accuracy") -> dict[str, dict[str, float]]:
        """Return mean and spread of *metric* for each configuration."""
        result: dict[str, dict[str, float]] = {}
        for name in self.configurations:
            series = self.series(name, metric)
            result[name] = {
                "mean": float(np.mean(series)),
                "sd": float(np.std(series, ddof=1)) if len(series) > 1 else 0.0,
                "min": float(np.min(series)),
                "max": float(np.max(series)),
                "n_sensors": float(
                    len(
                        next(
                            run.sensors
                            for run in self.runs
                            if run.configuration == name
                        )
                    )
                ),
            }
        return result

    def to_dict(self, metric: str = "balanced_accuracy") -> dict[str, object]:
        """Return a serialisable form of the report."""
        return {
            "metric": metric,
            "seeds": self.seeds,
            "summary": self.summary(metric),
            "runs": [run.to_dict() for run in self.runs],
        }


def evaluate_configuration(
    configuration: SensorConfiguration,
    result: object,
    *,
    step: timedelta = timedelta(minutes=5),
    degradation: DegradationConfig | None = None,
) -> StateMetrics:
    """Run one sensor configuration over one simulated trajectory."""
    registry = configuration.restrict(result.registry)  # type: ignore[attr-defined]
    observations = result.observations_for(configuration.sensors)  # type: ignore[attr-defined]
    if degradation is not None:
        observations, _ = degrade(observations, degradation)

    pipeline = BehaviouralSensingPipeline(
        registry,
        config=PipelineConfig(tz=result.config.tz, step=step),  # type: ignore[attr-defined]
    )
    steps = pipeline.run(observations)
    steps.extend(pipeline.close(result.end))  # type: ignore[attr-defined]
    if not steps:
        raise ValueError(
            f"configuration '{configuration.name}' produced no pipeline steps"
        )

    truth = result.truth.states_at([step_.at for step_ in steps])  # type: ignore[attr-defined]
    return state_metrics(truth, [step_.state for step_ in steps])


def run_ablation(
    configurations: Sequence[SensorConfiguration],
    *,
    seeds: Iterable[int],
    household: HouseholdConfig | None = None,
    step: timedelta = timedelta(minutes=5),
    degradation: DegradationConfig | None = None,
) -> AblationReport:
    """Evaluate every configuration on every seeded trajectory.

    One household is simulated per seed and shared by all configurations, so
    differences between configurations are differences in sensing rather than
    differences in the person being sensed.
    """
    if not configurations:
        raise ValueError("at least one configuration is required")
    names = [configuration.name for configuration in configurations]
    if len(set(names)) != len(names):
        raise ValueError("configuration names must be unique")

    base = household or HouseholdConfig()
    report = AblationReport()
    for seed in seeds:
        from dataclasses import replace as _replace

        result = simulate(_replace(base, seed=seed))
        for configuration in configurations:
            metrics = evaluate_configuration(
                configuration, result, step=step, degradation=degradation
            )
            report.runs.append(
                AblationRun(
                    configuration=configuration.name,
                    seed=seed,
                    metrics=metrics,
                    sensors=configuration.sensors,
                )
            )
            logger.info(
                "seed %s | %-22s | balanced accuracy %.3f | abstention %.3f",
                seed,
                configuration.name,
                metrics.balanced_accuracy,
                metrics.abstention_rate,
            )
    return report


def leave_one_out(
    registry: SensorRegistry, sensors: Iterable[str] | None = None
) -> list[SensorConfiguration]:
    """Build the full configuration plus one with each sensor removed."""
    everything = tuple(registry.sensor_ids())
    candidates = tuple(sensors) if sensors is not None else everything
    configurations = [SensorConfiguration("all", everything)]
    for sensor_id in candidates:
        remaining = tuple(s for s in everything if s != sensor_id)
        if remaining:
            configurations.append(
                SensorConfiguration(f"without_{sensor_id}", remaining)
            )
    return configurations


def named_subsets(subsets: Mapping[str, Sequence[str]]) -> list[SensorConfiguration]:
    """Build configurations from a mapping of name to sensor identifiers."""
    return [
        SensorConfiguration(name, tuple(sensors)) for name, sensors in subsets.items()
    ]
