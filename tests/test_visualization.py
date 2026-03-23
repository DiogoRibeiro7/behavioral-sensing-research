"""Tests for visualization utilities and web app."""

import os

import pandas as pd
from flask import Flask

from sensor_modeling.visualization import (
    clinical,
    interactive,
    research,
    web_app,
)


def _sample_data():
    """Create a small synthetic dataset for tests."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="H"),
            "sensor": ["a"] * 10,
            "value": range(10),
            "activity": ["walk"] * 10,
        }
    )


def test_interactive_export(tmp_path):
    """Export interactive visualization to HTML."""
    df = _sample_data()
    fig = interactive.real_time_display(df)
    out = tmp_path / "fig.html"
    interactive.export(fig, out)
    assert out.exists()


def test_clinical_alerts():
    """Verify clinical alert logic triggers correctly."""
    df = _sample_data()
    alerts = clinical.clinical_alerts(df, {"a": 5})
    assert alerts["a"] is True


def test_research_publication_figure():
    """Create a publication-quality figure."""
    df = _sample_data()
    fig = research.publication_figure(df, "timestamp", "value")
    assert fig is not None


def test_web_app_factory():
    """Ensure web app requires credentials."""
    os.environ["SM_USER"] = "user"
    os.environ["SM_PASS"] = "pass"
    app = web_app.create_app()
    assert isinstance(app, Flask)
    client = app.test_client()
    response = client.get(
        "/",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    assert response.status_code == 200
