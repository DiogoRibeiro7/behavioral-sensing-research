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
from .detection import (
    ArmOutcome,
    ChangeArm,
    DetectionStudy,
    run_detection_study,
    standard_arms,
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
from .provenance import (
    METRIC_DEFINITIONS,
    RESULTS_DIR,
    ExperimentRecord,
    environment,
    load_record,
)

__all__ = [
    "METRIC_DEFINITIONS",
    "RESULTS_DIR",
    "AblationReport",
    "ArmOutcome",
    "ArmResult",
    "AttributionStudy",
    "AblationRun",
    "BinaryMetrics",
    "ChangeArm",
    "DetectionMetrics",
    "DetectionStudy",
    "ExperimentRecord",
    "PairedDifference",
    "Scenario",
    "ScenarioComparison",
    "SensorConfiguration",
    "StateMetrics",
    "TimingMetrics",
    "binary_metrics",
    "compare_scenario",
    "detection_metrics",
    "environment",
    "evaluate_configuration",
    "leave_one_out",
    "load_record",
    "named_subsets",
    "paired_difference",
    "run_ablation",
    "run_attribution_study",
    "run_detection_study",
    "standard_arms",
    "standard_scenarios",
    "state_metrics",
    "summarise",
    "transition_timing",
]
