"""Analysis routines for sensor data."""

from .granger_causality import GrangerCausalityTest
from .dependency_network import SensorDependencyNetwork
from .behavioral_metrics import calculate_behavioral_metrics
from .pipeline import AnalysisPipeline
from .comparison import (
    cross_validate,
    significance_test,
    standardize_metrics,
    visualize_comparison,
)
from .behavioral_analysis import (
    recognize_activity_patterns,
    score_anomalies,
    detect_trends,
    health_indicators,
)
from .reporting import (
    generate_latex_report,
    create_html_dashboard,
    export_to_fhir,
    render_template,
)

__all__ = [
    "GrangerCausalityTest",
    "SensorDependencyNetwork",
    "calculate_behavioral_metrics",
    "AnalysisPipeline",
    "cross_validate",
    "significance_test",
    "standardize_metrics",
    "visualize_comparison",
    "recognize_activity_patterns",
    "score_anomalies",
    "detect_trends",
    "health_indicators",
    "generate_latex_report",
    "create_html_dashboard",
    "export_to_fhir",
    "render_template",
]
