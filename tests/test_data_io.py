"""Tests for shared data IO utilities."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sensor_modeling.utils.data_io import (
    SensorDataset,
    export_analysis_results,
    read_sensor_csv,
    simulate_sensor_data,
)


def test_read_sensor_csv_accepts_pathlike_timestamp_column(tmp_path):
    path = tmp_path / "sensor.csv"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1min"),
            "sensor_0": [0, 1, 0],
        }
    )
    df.to_csv(path, index=False)

    loaded = read_sensor_csv(path)

    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert list(loaded.columns) == ["sensor_0"]
    assert loaded.index.is_monotonic_increasing


def test_sensor_dataset_event_sequences_require_known_sensor():
    dataset = SensorDataset(
        pd.DataFrame(
            {"sensor_0": [0, 1]},
            index=pd.date_range("2024-01-01", periods=2, freq="1h"),
        )
    )

    with pytest.raises(KeyError, match="Sensor 'missing' not found"):
        dataset.to_event_sequences("missing")


def test_simulate_sensor_data_is_reproducible_without_global_rng_side_effects():
    np.random.seed(123)
    before = np.random.random()

    first = simulate_sensor_data(n_days=2, n_sensors=2, seed=7)
    second = simulate_sensor_data(n_days=2, n_sensors=2, seed=7)

    after = np.random.random()
    np.random.seed(123)
    assert before == np.random.random()
    assert after == np.random.random()
    pd.testing.assert_frame_equal(first.to_dataframe(), second.to_dataframe())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_days": 0}, "n_days must be at least 1"),
        ({"n_sensors": 0}, "n_sensors must be at least 1"),
        ({"interaction_strength": -0.1}, "interaction_strength must be non-negative"),
    ],
)
def test_simulate_sensor_data_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        simulate_sensor_data(**kwargs)


def test_export_analysis_results_writes_json_and_dataframe_csv(tmp_path):
    output_base = tmp_path / "analysis"
    causality = pd.DataFrame({"source": ["a"], "target": ["b"], "score": [0.7]})

    export_analysis_results(
        {"summary": {"ok": True}, "causality_results": causality},
        filename=output_base,
    )

    payload = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    exported_csv = pd.read_csv(tmp_path / "analysis_causality.csv")

    assert payload["summary"] == {"ok": True}
    pd.testing.assert_frame_equal(exported_csv, causality)
