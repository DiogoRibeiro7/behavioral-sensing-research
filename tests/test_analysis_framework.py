import json

import numpy as np
import pandas as pd
import pytest

from sensor_modeling.analysis import behavioral_analysis, comparison, reporting
from sensor_modeling.analysis.pipeline import AnalysisPipeline
from sensor_modeling.hmm.base import BaseHMM
from sensor_modeling.utils.data_io import SensorDataset


def _sample_df():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h")
    data = {"sensor_0": [0, 1] * 5, "sensor_1": [1, 0] * 5}
    return pd.DataFrame(data, index=idx)


def test_pipeline_and_reporting(tmp_path):
    ds = SensorDataset(_sample_df())
    pipe = AnalysisPipeline()
    results = pipe.run(ds)
    assert {"ar", "hmm", "cpd", "nhpp"} <= results.keys()

    tex = tmp_path / "direct" / "report.tex"
    html = tmp_path / "direct" / "dashboard.html"
    fhir = tmp_path / "direct" / "fhir.json"
    assert reporting.generate_latex_report(results, tex) == tex
    assert reporting.create_html_dashboard(results, html) == html
    assert reporting.export_to_fhir(results, fhir) == fhir
    assert tex.exists()
    assert html.exists()
    assert fhir.exists()
    fhir_payload = json.loads(fhir.read_text(encoding="utf-8"))
    assert fhir_payload["resourceType"] == "Observation"
    assert fhir_payload["code"]["text"] == "Sensor modeling analysis summary"
    assert "valueString" not in fhir_payload
    assert {item["code"]["text"] for item in fhir_payload["component"]} == set(results)

    nested_output = tmp_path / "nested" / "reports"
    report_paths = pipe.generate_report(results, nested_output)
    assert report_paths == {
        "latex": nested_output / "analysis.tex",
        "html": nested_output / "dashboard.html",
        "fhir": nested_output / "analysis_fhir.json",
    }
    assert (nested_output / "analysis.tex").exists()
    assert (nested_output / "dashboard.html").exists()
    assert (nested_output / "analysis_fhir.json").exists()


class _ArrayProbabilityModel:
    def fit(self, data):
        return self

    def predict_probabilities(self, data):
        return np.arange(10, dtype=float)


class _FailingPipelineModel:
    def fit(self, data):
        raise ValueError("model cannot fit")


def test_pipeline_formats_numpy_probability_outputs():
    pipe = AnalysisPipeline(models={"ar": _ArrayProbabilityModel()})

    results = pipe.run(SensorDataset(_sample_df()))

    assert results == {"ar": {"probabilities": [0.0, 1.0, 2.0, 3.0, 4.0]}}


def test_pipeline_records_expected_model_failures():
    pipe = AnalysisPipeline(models={"hmm": _FailingPipelineModel()})

    results = pipe.run(SensorDataset(_sample_df()))

    assert results == {"hmm": {"error": "model cannot fit"}}


def test_pipeline_validates_input_frame():
    pipe = AnalysisPipeline(models={"unknown": object()})

    with pytest.raises(ValueError, match="at least one row"):
        pipe.run(pd.DataFrame(columns=["sensor_0"]))

    with pytest.raises(ValueError, match="sensor column"):
        pipe.run(pd.DataFrame(index=[pd.Timestamp("2024-01-01")]))


def test_template_rendering_returns_original_on_format_errors():
    assert reporting.render_template("Hello {name}", {"name": "Ada"}) == "Hello Ada"
    assert reporting.render_template("Hello {missing}", {}) == "Hello {missing}"
    assert (
        reporting.render_template("Value {name!z}", {"name": "Ada"}) == "Value {name!z}"
    )


def test_comparison_and_behavioral():
    ds = SensorDataset(_sample_df())
    for train_idx, test_idx in comparison.time_series_splits(ds, n_splits=3):
        assert train_idx.max() < test_idx.min()

    models = {"hmm": BaseHMM()}
    scores = comparison.cross_validate(models, ds)
    assert "hmm" in scores
    standardized = comparison.standardize_metrics(scores)
    assert "hmm" in standardized
    pval = comparison.significance_test([0.1, 0.2, 0.3], [0.1, 0.2, 0.25])
    assert 0.0 <= pval <= 1.0

    patterns = behavioral_analysis.recognize_activity_patterns(ds.to_dataframe())
    anomalies = behavioral_analysis.score_anomalies(ds.to_dataframe())
    trends = behavioral_analysis.detect_trends(ds.to_dataframe(), window=2)
    health = behavioral_analysis.health_indicators(ds.to_dataframe())
    assert "peak_hours" in patterns
    assert len(anomalies) == len(ds.to_dataframe())
    assert trends.shape[0] == ds.to_dataframe().shape[0]
    assert "overall_activity" in health


def test_standardize_metrics_handles_empty_and_missing_values():
    assert comparison.standardize_metrics({}) == {}
    assert comparison.standardize_metrics({"a": 2.0, "b": 2.0}) == {
        "a": 0.0,
        "b": 0.0,
    }

    standardized = comparison.standardize_metrics(
        {"low": 2.0, "missing": float("nan"), "high": 4.0}
    )

    assert standardized["low"] == 0.0
    assert np.isnan(standardized["missing"])
    assert standardized["high"] == 1.0


def test_significance_test_validates_paired_scores():
    assert comparison.significance_test([0.2, 0.3], [0.2, 0.3]) == 1.0

    with pytest.raises(ValueError, match="same length"):
        comparison.significance_test([0.1, 0.2], [0.1])

    with pytest.raises(ValueError, match="at least two"):
        comparison.significance_test([0.1, float("nan")], [0.1, 0.2])


class _PredictOnlyModel:
    def fit(self, data):
        return self

    def predict(self, data):
        return data[:, 0]


def test_cross_validate_requires_explicit_scoring_contract():
    ds = SensorDataset(_sample_df())

    with pytest.raises(TypeError, match="explicit scorer"):
        comparison.cross_validate({"predict_only": _PredictOnlyModel()}, ds)


def test_cross_validate_accepts_explicit_model_scorer():
    ds = SensorDataset(_sample_df())

    def scorer(model, train_df, test_df):
        assert isinstance(model, _PredictOnlyModel)
        assert train_df.index.max() < test_df.index.min()
        return 0.5

    scores = comparison.cross_validate(
        {"predict_only": _PredictOnlyModel()},
        ds,
        scorers={"predict_only": scorer},
    )

    assert scores["predict_only"] == 0.5


class _FailingScoreModel:
    def fit(self, data):
        return self

    def score(self, data):
        raise ValueError("cannot score this fold")


def test_cross_validate_records_nan_for_expected_fold_failures():
    ds = SensorDataset(_sample_df())

    scores = comparison.cross_validate({"failing": _FailingScoreModel()}, ds)

    assert np.isnan(scores["failing"])


def test_time_series_splits_requires_positive_split_count():
    with pytest.raises(ValueError, match="n_splits"):
        comparison.time_series_splits(_sample_df(), n_splits=0)
