"""Analysis routines for sensor data."""

from .behavioral_analysis import (
    detect_trends,
    health_indicators,
    recognize_activity_patterns,
    score_anomalies,
)
from .behavioral_metrics import calculate_behavioral_metrics
from .comparison import (
    cross_validate,
    significance_test,
    standardize_metrics,
    visualize_comparison,
)
from .dependency_network import SensorDependencyNetwork
from .granger_causality import GrangerCausalityTest
from .pipeline import AnalysisPipeline
from .reporting import (
    create_html_dashboard,
    export_to_fhir,
    generate_latex_report,
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
