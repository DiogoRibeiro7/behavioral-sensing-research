"""Tests for behavioral analysis helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from sensor_modeling.analysis.behavioral_analysis import (
    detect_trends,
    health_indicators,
    recognize_activity_patterns,
    score_anomalies,
)
from sensor_modeling.analysis.behavioral_metrics import calculate_behavioral_metrics


def _sensor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor_0": [0, 1, 1, 0],
            "sensor_1": [1, 1, 0, 0],
            "label": ["a", "b", "c", "d"],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="1h"),
    )


def test_behavioral_helpers_ignore_non_numeric_columns():
    df = _sensor_frame()

    patterns = recognize_activity_patterns(df)
    trends = detect_trends(df, window=2)
    health = health_indicators(df)
    metrics = calculate_behavioral_metrics(df)

    assert patterns == {"peak_hours": 1, "quiet_hours": 3}
    assert list(trends.columns) == ["sensor_0", "sensor_1"]
    assert health["overall_activity"] == 1.0
    assert metrics["total_activations"] == 4


def test_score_anomalies_handles_constant_sensor_without_nan():
    df = pd.DataFrame(
        {"sensor_0": [1, 1, 1], "sensor_1": [0, 1, 0]},
        index=pd.date_range("2024-01-01", periods=3, freq="1h"),
    )

    anomalies = score_anomalies(df)

    assert not anomalies.isna().any()
    assert anomalies.index.equals(df.index)


def test_behavioral_metrics_use_zero_variance_for_single_day():
    metrics = calculate_behavioral_metrics(_sensor_frame())
    sensor_metrics = metrics["sensor_metrics"]

    assert sensor_metrics["sensor_0"]["daily_variance"] == 0.0
    assert sensor_metrics["sensor_1"]["daily_variance"] == 0.0


def test_behavioral_helpers_reject_invalid_frames():
    plain_index = pd.DataFrame({"sensor_0": [1, 0]})
    non_numeric = pd.DataFrame(
        {"label": ["a", "b"]},
        index=pd.date_range("2024-01-01", periods=2, freq="1h"),
    )

    with pytest.raises(TypeError, match="DatetimeIndex"):
        recognize_activity_patterns(plain_index)
    with pytest.raises(ValueError, match="numeric sensor column"):
        calculate_behavioral_metrics(non_numeric)
    with pytest.raises(ValueError, match="window must be at least 1"):
        detect_trends(_sensor_frame(), window=0)
    with pytest.raises(ValueError, match="at least one row"):
        health_indicators(pd.DataFrame(index=pd.DatetimeIndex([])))
