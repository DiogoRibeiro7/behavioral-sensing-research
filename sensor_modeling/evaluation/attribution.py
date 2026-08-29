"""Measuring what person attribution is worth.

Most ambient monitoring implicitly attributes every event to the monitored
resident. This module makes that assumption testable by running the same
simulated household twice on identical trajectories -- once attributing all
ambient activity to the resident, once discounting it by the probability that
the resident actually generated it -- and reporting the difference.

The comparison is paired by construction. Both arms see the same household,
the same visitors and the same sensor record, so any difference is
attributable to attribution and to nothing else.

A negative result is meaningful here. If occupancy-aware attribution changes
nothing, either contamination is negligible in this simulator or the occupancy
model is too weak to exploit it, and both are worth knowing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any

import numpy as np

from ..online.pipeline import BehaviouralSensingPipeline, PipelineConfig
from ..simulation.faults import DegradationConfig, degrade, dropout, not_worn
from ..simulation.household import HouseholdConfig, simulate
from .metrics import BinaryMetrics, StateMetrics, binary_metrics, state_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scenario:
    """A named occupancy situation to evaluate attribution against."""

    name: str
    household: HouseholdConfig
    degradation: DegradationConfig | None = None
    description: str = ""

    def build(self) -> Any:
        """Simulate the scenario and return the result plus its record."""
        result = simulate(self.household)
        observations: Sequence[Any] = result.observations
        if self.degradation is not None:
            observations, _ = degrade(observations, self.degradation)
        return result, observations


def standard_scenarios(days: int = 10, seed: int = 4242) -> list[Scenario]:
    """Build the occupancy situations attribution has to cope with.

    Each isolates one way in which ambient activity can fail to belong to the
    monitored resident, or one way the evidence for deciding that can be lost.
    """
    base = HouseholdConfig(days=days, seed=seed)
    start = simulate(HouseholdConfig(days=1, seed=seed)).start

    return [
        Scenario(
            "resident_alone",
            replace(base, visitor_probability=0.0, carer_weekday_visits=False),
            description="No other person ever enters. Attribution should be a no-op.",
        ),
        Scenario(
            "resident_goes_out",
            replace(
                base,
                visitor_probability=0.0,
                carer_weekday_visits=False,
                outing_probability=0.95,
            ),
            description="The resident is frequently away; ambient events then "
            "belong to nobody.",
        ),
        Scenario(
            "short_visitor",
            replace(base, visitor_probability=0.5, carer_weekday_visits=False),
            description="Occasional short social visits.",
        ),
        Scenario(
            "prolonged_visitor",
            replace(base, visitor_probability=1.0, carer_weekday_visits=False),
            description="A visitor present on every day of the record.",
        ),
        Scenario(
            "carer_visits",
            replace(base, visitor_probability=0.0, carer_weekday_visits=True),
            description="A regular weekday carer round, the case most likely "
            "to be mistaken for the resident rising early.",
        ),
        Scenario(
            "visitor_and_carer",
            replace(base, visitor_probability=0.6, carer_weekday_visits=True),
            description="Both, so ambient sensors across several rooms are "
            "contaminated.",
        ),
        Scenario(
            "resident_without_wearable",
            base,
            DegradationConfig(
                faults=not_worn(
                    ["wearable_motion", "resident_beacon"], start, timedelta(days=days)
                ),
                seed=seed + 1,
            ),
            description="The strongest attribution evidence is unavailable for "
            "the whole record.",
        ),
        Scenario(
            "no_radar",
            base,
            DegradationConfig(
                faults=(dropout("living_radar", start, timedelta(days=days)),),
                seed=seed + 2,
            ),
            description="No track count, so multi-person evidence is limited to "
            "concurrent room activity.",
        ),
        Scenario(
            "sparse_coverage",
            base,
            DegradationConfig(missing_rate=0.3, seed=seed + 3),
            description="Incomplete sensor coverage on top of contamination.",
        ),
    ]


@dataclass(frozen=True)
class ArmResult:
    """Outcome of one attribution setting on one scenario."""

    attributed: bool
    states: StateMetrics
    mean_attribution: float

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the arm."""
        return {
            "attributed": self.attributed,
            "mean_attribution": self.mean_attribution,
            "states": self.states.to_dict(),
        }


@dataclass(frozen=True)
class ScenarioComparison:
    """Naive against occupancy-aware attribution on one scenario."""

    scenario: str
    description: str
    naive: ArmResult
    occupancy_aware: ArmResult
    visitor_detection: BinaryMetrics
    contaminated_fraction: float

    @property
    def balanced_accuracy_gain(self) -> float:
        """Balanced accuracy gained by discounting unattributable evidence."""
        return (
            self.occupancy_aware.states.balanced_accuracy
            - self.naive.states.balanced_accuracy
        )

    @property
    def calibration_gain(self) -> float:
        """Reduction in calibration error. Positive means better calibrated."""
        return (
            self.naive.states.calibration_error
            - self.occupancy_aware.states.calibration_error
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the comparison."""
        return {
            "scenario": self.scenario,
            "description": self.description,
            "contaminated_fraction": self.contaminated_fraction,
            "balanced_accuracy_gain": self.balanced_accuracy_gain,
            "calibration_gain": self.calibration_gain,
            "visitor_detection": self.visitor_detection.to_dict(),
            "naive": self.naive.to_dict(),
            "occupancy_aware": self.occupancy_aware.to_dict(),
        }


def _run_arm(
    result: Any,
    observations: Sequence[Any],
    *,
    attribute: bool,
    step: timedelta,
) -> tuple[ArmResult, list[Any]]:
    """Run one attribution setting over a prepared record."""
    pipeline = BehaviouralSensingPipeline(
        result.registry,
        config=PipelineConfig(
            tz=result.config.tz, step=step, attribute_activity=attribute
        ),
    )
    steps = pipeline.run(observations)
    steps.extend(pipeline.close(result.end))
    if not steps:
        raise ValueError("scenario produced no pipeline steps")

    truth = result.truth.states_at([s.at for s in steps])
    shares = [
        contribution.attribution
        for s in steps
        for contribution in s.state.evidence
        if contribution.attribution < 1.0 or not attribute
    ]
    return (
        ArmResult(
            attributed=attribute,
            states=state_metrics(truth, [s.state for s in steps]),
            mean_attribution=float(np.mean(shares)) if shares else 1.0,
        ),
        steps,
    )


def compare_scenario(
    scenario: Scenario, *, step: timedelta = timedelta(minutes=10)
) -> ScenarioComparison:
    """Run both attribution arms over one scenario and report the difference."""
    result, observations = scenario.build()

    naive, _ = _run_arm(result, observations, attribute=False, step=step)
    aware, aware_steps = _run_arm(result, observations, attribute=True, step=step)

    moments = [s.at for s in aware_steps]
    truth_visitor = [result.truth.visitor_at(moment) for moment in moments]
    detection = binary_metrics(
        truth_visitor, [s.context.visitor_present for s in aware_steps]
    )

    logger.info(
        "%-26s balanced accuracy naive %.3f -> aware %.3f",
        scenario.name,
        naive.states.balanced_accuracy,
        aware.states.balanced_accuracy,
    )
    return ScenarioComparison(
        scenario=scenario.name,
        description=scenario.description,
        naive=naive,
        occupancy_aware=aware,
        visitor_detection=detection,
        contaminated_fraction=float(np.mean(truth_visitor)),
    )


@dataclass
class AttributionStudy:
    """Results of comparing attribution across several scenarios."""

    comparisons: list[ScenarioComparison] = field(default_factory=list)

    def by_name(self, name: str) -> ScenarioComparison:
        """Return the comparison for one scenario."""
        for comparison in self.comparisons:
            if comparison.scenario == name:
                return comparison
        raise KeyError(f"no scenario named '{name}'")

    @property
    def contaminated(self) -> list[ScenarioComparison]:
        """Scenarios in which another person was actually present."""
        return [c for c in self.comparisons if c.contaminated_fraction > 0.01]

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the study."""
        return {
            "scenarios": [c.to_dict() for c in self.comparisons],
            "mean_gain_when_contaminated": (
                float(np.mean([c.balanced_accuracy_gain for c in self.contaminated]))
                if self.contaminated
                else 0.0
            ),
        }


def run_attribution_study(
    scenarios: Iterable[Scenario] | None = None,
    *,
    step: timedelta = timedelta(minutes=10),
) -> AttributionStudy:
    """Compare naive and occupancy-aware attribution across scenarios."""
    selected = list(scenarios) if scenarios is not None else standard_scenarios()
    if not selected:
        raise ValueError("at least one scenario is required")
    return AttributionStudy(
        comparisons=[compare_scenario(s, step=step) for s in selected]
    )
