"""Evaluation metrics and sensor-ablation experiments.

Metrics are chosen per problem rather than defaulting to accuracy, which is
close to meaningless on states this imbalanced. Ablation studies are paired by
construction: every configuration sees identical simulated trajectories, so a
difference between configurations is a difference in sensing rather than in
the person being sensed.

Model comparisons keep three layers apart: timestamps are scored within one
household (:func:`prediction_metrics`), households are summarised with each
counted once (:func:`summarise_households`), and models are compared by
resampling households (:func:`compare_households`), never timestamps.
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
from .households import (
    Estimate,
    HouseholdComparison,
    HouseholdSummary,
    compare_households,
    household_values,
    recall_of,
    score_households,
    summarise_households,
)
from .metrics import (
    BinaryMetrics,
    ConfusionMatrix,
    DetectionMetrics,
    PairedDifference,
    PredictionMetrics,
    StateMetrics,
    TimingMetrics,
    binary_metrics,
    confusion_matrix,
    detection_metrics,
    paired_difference,
    prediction_metrics,
    state_metrics,
    summarise,
    transition_timing,
)
from .provenance import (
    METRIC_DEFINITIONS,
    RESULTS_DIR,
    SCHEMA_VERSION,
    ArtifactError,
    ExperimentRecord,
    InputArtifact,
    ModelRecord,
    ReportedInterval,
    environment,
    load_record,
    validate_record,
)
from .resampling import Interval, monte_carlo_standard_error

__all__ = [
    "METRIC_DEFINITIONS",
    "RESULTS_DIR",
    "SCHEMA_VERSION",
    "AblationReport",
    "ArmOutcome",
    "ArmResult",
    "ArtifactError",
    "AttributionStudy",
    "AblationRun",
    "BinaryMetrics",
    "ChangeArm",
    "ConfusionMatrix",
    "DetectionMetrics",
    "DetectionStudy",
    "Estimate",
    "ExperimentRecord",
    "HouseholdComparison",
    "HouseholdSummary",
    "InputArtifact",
    "Interval",
    "ModelRecord",
    "PairedDifference",
    "PredictionMetrics",
    "ReportedInterval",
    "Scenario",
    "ScenarioComparison",
    "SensorConfiguration",
    "StateMetrics",
    "TimingMetrics",
    "binary_metrics",
    "compare_scenario",
    "compare_households",
    "confusion_matrix",
    "detection_metrics",
    "environment",
    "household_values",
    "evaluate_configuration",
    "leave_one_out",
    "load_record",
    "monte_carlo_standard_error",
    "named_subsets",
    "paired_difference",
    "prediction_metrics",
    "recall_of",
    "run_ablation",
    "run_attribution_study",
    "run_detection_study",
    "score_households",
    "standard_arms",
    "standard_scenarios",
    "state_metrics",
    "summarise",
    "summarise_households",
    "transition_timing",
    "validate_record",
]
