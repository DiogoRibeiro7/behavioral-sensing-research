"""Reporting utilities for analysis results."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def generate_latex_report(results: dict[str, Any], path: str) -> None:
    """Generate a minimal LaTeX report summarizing *results*."""
    content = [
        r"\documentclass{article}",
        r"\begin{document}",
        r"\section*{Sensor Modeling Report}",
        r"\begin{verbatim}",
        json.dumps(results, indent=2, default=str),
        r"\end{verbatim}",
        r"\end{document}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(content))
    logger.info("LaTeX report written to %s", path)


# ---------------------------------------------------------------------------
def create_html_dashboard(results: dict[str, Any], path: str) -> None:
    """Generate a simple HTML dashboard for *results*."""
    html = [
        "<html><body><h1>Sensor Modeling Dashboard</h1><pre>",
        json.dumps(results, indent=2, default=str),
        "</pre></body></html>",
    ]
    with open(path, "w") as f:
        f.write("\n".join(html))
    logger.info("HTML dashboard written to %s", path)


# ---------------------------------------------------------------------------
def export_to_fhir(results: dict[str, Any], path: str) -> None:
    """Export *results* as a minimal FHIR-like Observation resource.

    The export preserves each top-level analysis result as an Observation
    component. It is intended as an interoperable starting point, not a full
    clinical profile implementation.
    """
    fhir = {
        "resourceType": "Observation",
        "id": "sensor-analysis",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/CodeSystem/"
                            "observation-category"
                        ),
                        "code": "activity",
                        "display": "Activity",
                    }
                ]
            }
        ],
        "code": {
            "text": "Sensor modeling analysis summary",
        },
        "effectiveDateTime": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "component": [
            {
                "code": {"text": name},
                "valueString": json.dumps(value, default=str),
            }
            for name, value in results.items()
        ],
    }
    with open(path, "w") as f:
        json.dump(fhir, f, indent=2)
    logger.info("FHIR export written to %s", path)


# ---------------------------------------------------------------------------
def render_template(template: str, context: dict[str, Any]) -> str:
    """Render *template* using ``str.format`` with the provided *context*."""
    try:
        return template.format(**context)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Template rendering failed: %s", exc)
        return template
