"""Reporting utilities for analysis results."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def generate_latex_report(results: Dict[str, Any], path: str) -> None:
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
def create_html_dashboard(results: Dict[str, Any], path: str) -> None:
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
def export_to_fhir(results: Dict[str, Any], path: str) -> None:
    """Export *results* in a minimal HL7 FHIR Observation JSON format."""
    fhir = {
        "resourceType": "Observation",
        "id": "sensor-analysis",
        "status": "final",
        "valueString": json.dumps(results, default=str),
    }
    with open(path, "w") as f:
        json.dump(fhir, f, indent=2)
    logger.info("FHIR export written to %s", path)


# ---------------------------------------------------------------------------
def render_template(template: str, context: Dict[str, Any]) -> str:
    """Render *template* using ``str.format`` with the provided *context*."""
    try:
        return template.format(**context)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Template rendering failed: %s", exc)
        return template
