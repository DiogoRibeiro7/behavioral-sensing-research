import pandas as pd

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

    tex = tmp_path / "report.tex"
    html = tmp_path / "dashboard.html"
    fhir = tmp_path / "fhir.json"
    reporting.generate_latex_report(results, tex)
    reporting.create_html_dashboard(results, html)
    reporting.export_to_fhir(results, fhir)
    assert tex.exists()
    assert html.exists()
    assert fhir.exists()

    nested_output = tmp_path / "nested" / "reports"
    pipe.generate_report(results, nested_output)
    assert (nested_output / "analysis.tex").exists()
    assert (nested_output / "dashboard.html").exists()
    assert (nested_output / "analysis_fhir.json").exists()


def test_comparison_and_behavioral():
    ds = SensorDataset(_sample_df())
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
