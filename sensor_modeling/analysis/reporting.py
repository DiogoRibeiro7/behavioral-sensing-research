"""Reporting utilities for analysis results."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ReportPath = str | PathLike[str]


def _prepare_output_path(path: ReportPath) -> Path:
    """Return an output path with its parent directory created."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


# ---------------------------------------------------------------------------
def generate_latex_report(results: dict[str, Any], path: ReportPath) -> Path:
    """Generate a minimal LaTeX report summarizing *results*."""
    output_path = _prepare_output_path(path)
    content = [
        r"\documentclass{article}",
        r"\begin{document}",
        r"\section*{Sensor Modeling Report}",
        r"\begin{verbatim}",
        json.dumps(results, indent=2, default=str),
        r"\end{verbatim}",
        r"\end{document}",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")
    logger.info("LaTeX report written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
def create_html_dashboard(results: dict[str, Any], path: ReportPath) -> Path:
    """Generate a simple HTML dashboard for *results*."""
    output_path = _prepare_output_path(path)
    html = [
        "<html><body><h1>Sensor Modeling Dashboard</h1><pre>",
        json.dumps(results, indent=2, default=str),
        "</pre></body></html>",
    ]
    output_path.write_text("\n".join(html), encoding="utf-8")
    logger.info("HTML dashboard written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
def export_to_fhir(results: dict[str, Any], path: ReportPath) -> Path:
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
    output_path = _prepare_output_path(path)
    output_path.write_text(json.dumps(fhir, indent=2), encoding="utf-8")
    logger.info("FHIR export written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
def render_template(template: str, context: dict[str, Any]) -> str:
    """Render *template* using ``str.format`` with the provided *context*."""
    try:
        return template.format(**context)
    except (KeyError, IndexError, ValueError, AttributeError) as exc:
        logger.error("Template rendering failed: %s", exc)
        return template
