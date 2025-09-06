import json
import numpy as np
import pandas as pd
import pytest

from sensor_modeling.data import loaders, preprocessing, validation, synthetic
from sensor_modeling.utils.data_io import SensorDataset


def _sample_df():
    idx = pd.date_range("2024-01-01", periods=5, freq="1min")
    df = pd.DataFrame({"sensor_0": [0, 1, np.nan, 0, 1], "sensor_1": [1, 1, 1, 0, 0]}, index=idx)
    df.index.name = "timestamp"
    return df


def test_loaders(tmp_path):
    df = _sample_df()
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path)
    ds_csv = loaders.load_csv(csv_path)
    assert ds_csv.to_dataframe().shape == df.shape

    json_path = tmp_path / "data.json"
    records = df.reset_index()
    records["timestamp"] = records["timestamp"].astype(str)
    with open(json_path, "w") as f:
        json.dump(records.to_dict(orient="records"), f)
    ds_json = loaders.load_json(json_path)
    assert ds_json.to_dataframe().shape == df.shape

    try:
        import h5py
    except Exception:
        pytest.skip("h5py not installed")
    h5_path = tmp_path / "data.h5"
    with h5py.File(h5_path, "w") as h5:
        dset = h5.create_dataset("data", data=df.values)
        dset.attrs["timestamp"] = df.index.astype(str).to_list()
    ds_h5 = loaders.load_hdf5(h5_path, key="data")
    assert ds_h5.to_dataframe().shape == df.shape

    stream = [
        {"timestamp": str(df.index[0]), "sensor_0": 0, "sensor_1": 1},
        {"timestamp": str(df.index[1]), "sensor_0": 1, "sensor_1": 1},
    ]
    first = next(loaders.stream_data(stream))
    assert list(first.to_dataframe().columns) == ["sensor_0", "sensor_1"]


def test_preprocessing_and_validation():
    ds = SensorDataset(_sample_df())
    miss = preprocessing.detect_missing(ds)
    assert miss["sensor_0"] > 0
    ds_imputed = preprocessing.impute_missing(ds)
    assert preprocessing.detect_missing(ds_imputed).sum() == 0
    outliers = preprocessing.detect_outliers(ds_imputed)
    assert outliers.shape == ds_imputed.to_dataframe().shape
    aligned = preprocessing.align_sensors([ds_imputed, ds_imputed], freq="1min")
    assert len(aligned) == 2
    report = preprocessing.data_quality_report(ds)
    assert "missing_ratio" in report

    assert validation.check_temporal_consistency(ds)
    assert validation.validate_sensor_ranges(ds_imputed)
    failures = validation.detect_sensor_failures(ds_imputed)
    assert set(failures.keys()) == set(ds_imputed.to_dataframe().columns)
    corr = validation.cross_sensor_correlation(ds_imputed)
    assert corr.shape[0] == len(ds_imputed.to_dataframe().columns)


def test_synthetic_generation_and_export(tmp_path):
    cfg = synthetic.SyntheticConfig(n_steps=50, n_sensors=2, change_points=[20], failure_rate=0.0)
    ds, meta = synthetic.generate(cfg)
    assert 20 in meta["change_points"]
    out_csv = tmp_path / "syn.csv"
    synthetic.export(ds, meta, out_csv, fmt="csv")
    assert out_csv.exists()
