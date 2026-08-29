"""Evaluation metrics and sensor-ablation experiments.

Metrics are chosen per problem rather than defaulting to accuracy, which is
close to meaningless on states this imbalanced. Ablation studies are paired by
construction: every configuration sees identical simulated trajectories, so a
difference between configurations is a difference in sensing rather than in
the person being sensed.
"""

from .ablation import (
    AblationReport,
    AblationRun,
    SensorConfiguration,
    evaluate_configuration,
    leave_one_out,
    named_subsets,
    run_ablation,
)
from .attribution import (
    ArmResult,
    AttributionStudy,
    Scenario,
    ScenarioComparison,
    compare_scenario,
    run_attribution_study,
    standard_scenarios,
)
from .metrics import (
    BinaryMetrics,
    DetectionMetrics,
    PairedDifference,
    StateMetrics,
    TimingMetrics,
    binary_metrics,
    detection_metrics,
    paired_difference,
    state_metrics,
    summarise,
    transition_timing,
)

__all__ = [
    "AblationReport",
    "ArmResult",
    "AttributionStudy",
    "AblationRun",
    "BinaryMetrics",
    "DetectionMetrics",
    "PairedDifference",
    "Scenario",
    "ScenarioComparison",
    "SensorConfiguration",
    "StateMetrics",
    "TimingMetrics",
    "binary_metrics",
    "compare_scenario",
    "detection_metrics",
    "evaluate_configuration",
    "leave_one_out",
    "named_subsets",
    "paired_difference",
    "run_ablation",
    "run_attribution_study",
    "standard_scenarios",
    "state_metrics",
    "summarise",
    "transition_timing",
]
