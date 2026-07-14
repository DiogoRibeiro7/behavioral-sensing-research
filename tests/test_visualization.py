"""Tests for visualization utilities and web app."""

import os
from unittest.mock import MagicMock

import pandas as pd
import pytest
from bokeh.models import Div, Slider
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
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
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


class _DummyTuningModel:
    """Small model with a deterministic parameter score."""

    def __init__(self):
        self.penalty = 1.0

    def score_parameter(self, param, value):
        assert param == "penalty"
        return -((value - 2.0) ** 2) + 4.0


def test_parameter_tuning_returns_visible_diagnostic_layout():
    """Parameter tuning should render a real plot and summary, not a stub."""
    layout = interactive.parameter_tuning(
        _DummyTuningModel(), "penalty", [0.5, 1.0, 2.0, 3.0]
    )
    assert isinstance(layout.children[0], Slider)
    assert isinstance(layout.children[1], Div)
    assert layout.children[2].height > 0
    assert layout.children[2].title.text == "penalty diagnostic sweep"


def test_parameter_tuning_export_supports_bokeh_layout(tmp_path):
    """Bokeh parameter tuning layouts should export to standalone HTML."""
    layout = interactive.parameter_tuning(
        _DummyTuningModel(), "penalty", [0.5, 1.0, 2.0, 3.0]
    )
    out = tmp_path / "tuning.html"
    interactive.export(layout, out)
    assert out.exists()
    assert "penalty diagnostic sweep" in out.read_text(encoding="utf-8")


def test_clinical_alerts():
    """Verify clinical alert logic triggers correctly."""
    df = _sample_data()
    alerts = clinical.clinical_alerts(df, {"a": 5})
    assert alerts["a"] is True


def test_clinical_visualizations_validate_inputs():
    """Clinical helpers should report missing columns and invalid windows clearly."""
    df = _sample_data()
    normative = df.copy()
    activity_fig = clinical.activity_summary(df)
    trend_fig = clinical.trend_monitor(df, window=2)
    comparison_fig = clinical.compare_norms(df, normative)

    assert activity_fig is not None
    assert trend_fig is not None
    assert comparison_fig is not None

    with pytest.raises(ValueError, match="activity_summary requires columns"):
        clinical.activity_summary(df.drop(columns=["activity"]))

    with pytest.raises(ValueError, match="clinical_alerts requires columns"):
        clinical.clinical_alerts(df.drop(columns=["value"]), {"a": 5})

    with pytest.raises(ValueError, match="window must be at least 1"):
        clinical.trend_monitor(df, window=0)

    with pytest.raises(ValueError, match="normative data requires columns"):
        clinical.compare_norms(df, normative.drop(columns=["value"]))


def test_research_publication_figure():
    """Create a publication-quality figure."""
    df = _sample_data()
    fig = research.publication_figure(df, "timestamp", "value")
    assert fig is not None


def test_research_visualizations_validate_inputs():
    """Research plotting helpers should validate required schema and residuals."""
    df = _sample_data()
    scores = pd.DataFrame(
        {"model": ["ar", "hmm"], "metric": ["auc", "auc"], "score": [0.7, 0.8]}
    )
    results = pd.DataFrame(
        {"model": ["ar", "hmm"], "test": ["t-test", "t-test"], "pvalue": [0.2, 0.01]}
    )

    assert research.model_diagnostics([0.1, -0.2, 0.0]) is not None
    assert research.performance_comparison(scores) is not None
    assert research.statistical_tests(results) is not None

    with pytest.raises(ValueError, match="publication_figure requires columns"):
        research.publication_figure(df.drop(columns=["value"]), "timestamp", "value")

    with pytest.raises(ValueError, match="at least one value"):
        research.model_diagnostics([])

    with pytest.raises(ValueError, match="finite values"):
        research.model_diagnostics([0.1, float("nan")])

    with pytest.raises(ValueError, match="performance_comparison requires columns"):
        research.performance_comparison(scores.drop(columns=["metric"]))

    with pytest.raises(ValueError, match="statistical_tests requires columns"):
        research.statistical_tests(results.drop(columns=["pvalue"]))


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


def test_web_app_main(monkeypatch):
    """The web app CLI entry point starts the Flask app."""
    fake_run = MagicMock()
    monkeypatch.setattr(web_app.app, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["sensor-viz"])

    web_app.main()

    fake_run.assert_called_once_with(host="127.0.0.1", port=5000, debug=False)
