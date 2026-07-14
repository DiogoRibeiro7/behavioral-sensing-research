import json
import sys

import numpy as np
import pandas as pd
import pytest

from sensor_modeling.data import loaders, preprocessing, synthetic, validation
from sensor_modeling.utils.data_io import SensorDataset


def _sample_df():
    idx = pd.date_range("2024-01-01", periods=5, freq="1min")
    df = pd.DataFrame(
        {"sensor_0": [0, 1, np.nan, 0, 1], "sensor_1": [1, 1, 1, 0, 0]}, index=idx
    )
    df.index.name = "timestamp"
    return df


def test_loaders(tmp_path):
    df = _sample_df()
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path)
    ds_csv = loaders.load_csv(csv_path)
    assert ds_csv.to_dataframe().shape == df.shape
    ds_from_csv = SensorDataset.from_csv(csv_path)
    pd.testing.assert_frame_equal(ds_from_csv.to_dataframe(), ds_csv.to_dataframe())

    unnamed_index_path = tmp_path / "unnamed-index.csv"
    df.rename_axis(None).to_csv(unnamed_index_path)
    ds_unnamed = loaders.load_csv(unnamed_index_path)
    assert ds_unnamed.to_dataframe().index.equals(df.index)
    assert ds_unnamed.to_dataframe().shape == df.shape

    plain_path = tmp_path / "plain.csv"
    plain_df = df.reset_index(drop=True)
    plain_df.to_csv(plain_path, index=False)
    ds_plain = loaders.load_csv(plain_path)
    assert isinstance(ds_plain.to_dataframe().index, pd.RangeIndex)
    assert ds_plain.to_dataframe().shape == plain_df.shape

    json_path = tmp_path / "data.json"
    records = df.reset_index()
    records["timestamp"] = records["timestamp"].astype(str)
    with open(json_path, "w") as f:
        json.dump(records.to_dict(orient="records"), f)
    ds_json = loaders.load_json(json_path)
    assert ds_json.to_dataframe().shape == df.shape

    try:
        import h5py
    except ImportError:
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


def test_load_json_reports_invalid_payloads(tmp_path):
    invalid_cases = [
        ("object.json", {"timestamp": "2024-01-01", "sensor_0": 1}, "list"),
        ("missing-timestamp.json", [{"sensor_0": 1}], "Timestamp field"),
        (
            "invalid-timestamp.json",
            [{"timestamp": "not-a-date", "sensor_0": 1}],
            "invalid timestamps",
        ),
        ("scalar-list.json", [1, 2, 3], "Invalid JSON structure"),
    ]

    for filename, payload, message in invalid_cases:
        path = tmp_path / filename
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            loaders.load_json(path)


def test_loaders_report_unreadable_files(tmp_path):
    with pytest.raises(ValueError, match="Unable to read CSV file"):
        loaders.load_csv(tmp_path / "missing.csv")

    malformed_json = tmp_path / "malformed.json"
    malformed_json.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to read JSON file"):
        loaders.load_json(malformed_json)


def test_stream_data_skips_malformed_items():
    stream = [
        {"timestamp": "2024-01-01 00:00:00", "sensor_0": 1},
        {"sensor_0": 0},
        {"timestamp": "not-a-date", "sensor_0": 1},
        ["not", "a", "mapping"],
        {"timestamp": "2024-01-01 00:01:00", "sensor_0": 0},
    ]

    datasets = list(loaders.stream_data(stream))

    assert [dataset.to_dataframe()["sensor_0"].iloc[0] for dataset in datasets] == [
        1,
        0,
    ]
    assert [dataset.to_dataframe().index[0] for dataset in datasets] == [
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 00:01:00"),
    ]


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
    cfg = synthetic.SyntheticConfig(
        n_steps=50, n_sensors=2, change_points=[20], failure_rate=0.0
    )
    ds, meta = synthetic.generate(cfg)
    assert 20 in meta["change_points"]
    out_csv = tmp_path / "nested" / "syn.csv"
    csv_paths = synthetic.export(ds, meta, out_csv, fmt="csv")
    assert csv_paths == {
        "data": tmp_path / "nested" / "syn.csv",
        "metadata": tmp_path / "nested" / "syn.csv.meta.json",
    }
    assert csv_paths["data"].exists()
    assert csv_paths["metadata"].exists()

    out_json = tmp_path / "nested" / "syn.json"
    json_paths = synthetic.export(ds, meta, out_json, fmt="json")
    assert json_paths == {"data": tmp_path / "nested" / "syn.json"}
    payload = json.loads(json_paths["data"].read_text(encoding="utf-8"))
    assert payload["meta"]["change_points"] == [20]
    assert len(payload["data"]) == 50

    unsupported_path = tmp_path / "unsupported" / "syn.unsupported"
    with pytest.raises(ValueError, match="Unsupported export format"):
        synthetic.export(ds, meta, unsupported_path, fmt="parquet")
    assert not unsupported_path.parent.exists()


def test_synthetic_generation_validates_config():
    invalid_configs = [
        (synthetic.SyntheticConfig(n_steps=0), "n_steps"),
        (synthetic.SyntheticConfig(n_sensors=0), "n_sensors"),
        (synthetic.SyntheticConfig(failure_rate=-0.1), "failure_rate"),
        (synthetic.SyntheticConfig(failure_rate=1.1), "failure_rate"),
        (
            synthetic.SyntheticConfig(n_steps=10, change_points=[10]),
            "change_points",
        ),
        (synthetic.SyntheticConfig(change_points=[True]), "change_points"),
    ]

    for config, message in invalid_configs:
        with pytest.raises(ValueError, match=message):
            synthetic.generate(config)


def test_synthetic_generation_handles_single_step_failure_config():
    dataset, meta = synthetic.generate(
        synthetic.SyntheticConfig(n_steps=1, n_sensors=1, failure_rate=1.0)
    )

    assert dataset.to_dataframe().shape == (1, 1)
    assert meta == {"change_points": [0]}


def test_synthetic_generation_bounds_multiple_change_points():
    dataset, meta = synthetic.generate(
        synthetic.SyntheticConfig(n_steps=5, n_sensors=2, change_points=[0, 1, 2])
    )

    assert dataset.to_dataframe().shape == (5, 2)
    assert meta == {"change_points": [0, 1, 2]}


def test_synthetic_hdf5_export_returns_path(tmp_path):
    h5py = pytest.importorskip("h5py")
    dataset, meta = synthetic.generate(synthetic.SyntheticConfig(n_steps=5))

    output_paths = synthetic.export(
        dataset, meta, tmp_path / "nested" / "syn.h5", "hdf5"
    )

    assert output_paths == {"data": tmp_path / "nested" / "syn.h5"}
    with h5py.File(output_paths["data"], "r") as h5:
        assert "data" in h5
        assert "meta" in h5


def test_hdf5_loader_reports_missing_dataset(tmp_path):
    h5py = pytest.importorskip("h5py")

    h5_path = tmp_path / "data.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("other", data=np.array([[1, 0]]))

    with pytest.raises(ValueError, match="Dataset 'data' not found"):
        loaders.load_hdf5(h5_path)


def test_hdf5_loader_requires_dependency(monkeypatch, tmp_path):
    h5_path = tmp_path / "data.h5"

    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match="h5py is required"):
        loaders.load_hdf5(h5_path)


def test_detect_sensor_failures_flags_constant_stretches():
    idx = pd.date_range("2024-01-01", periods=6, freq="1min")
    df = pd.DataFrame(
        {"stuck": [1, 1, 1, 1, 1, 1], "active": [0, 1, 0, 1, 0, 1]},
        index=idx,
    )

    failures = validation.detect_sensor_failures(SensorDataset(df), window=3)

    assert failures == {"stuck": True, "active": False}


def test_temporal_consistency_rejects_invalid_indexes():
    assert not validation.check_temporal_consistency(
        SensorDataset(pd.DataFrame({"sensor_0": [0, 1]}))
    )

    duplicate_index = pd.to_datetime(["2024-01-01", "2024-01-01"])
    assert not validation.check_temporal_consistency(
        SensorDataset(pd.DataFrame({"sensor_0": [0, 1]}, index=duplicate_index))
    )

    irregular_index = pd.to_datetime(
        ["2024-01-01 00:00:00", "2024-01-01 00:01:00", "2024-01-01 00:03:00"]
    )
    assert not validation.check_temporal_consistency(
        SensorDataset(pd.DataFrame({"sensor_0": [0, 1, 0]}, index=irregular_index))
    )


def test_validate_sensor_ranges_handles_invalid_values():
    valid_index = pd.date_range("2024-01-01", periods=2, freq="1min")

    assert not validation.validate_sensor_ranges(
        SensorDataset(pd.DataFrame({"sensor_0": ["off", "on"]}, index=valid_index))
    )
    assert not validation.validate_sensor_ranges(
        SensorDataset(pd.DataFrame({"sensor_0": [0, 2]}, index=valid_index))
    )

    with pytest.raises(ValueError, match="min_val"):
        validation.validate_sensor_ranges(
            SensorDataset(pd.DataFrame({"sensor_0": [0, 1]}, index=valid_index)),
            min_val=1,
            max_val=0,
        )


def test_detect_sensor_failures_validates_window():
    with pytest.raises(ValueError, match="window"):
        validation.detect_sensor_failures(SensorDataset(_sample_df()), window=0)


def test_preprocessing_handles_mixed_dtype_outliers_and_mean_imputation():
    idx = pd.date_range("2024-01-01", periods=4, freq="1min")
    df = pd.DataFrame(
        {
            "sensor_0": [1.0, 1.0, 1.0, 1.0],
            "sensor_1": [0.0, 0.0, 0.0, 10.0],
            "label": ["ok", None, "ok", "alert"],
        },
        index=idx,
    )

    outliers = preprocessing.detect_outliers(SensorDataset(df), z_thresh=1.0)
    imputed = preprocessing.impute_missing(SensorDataset(df), strategy="mean")

    assert list(outliers.columns) == ["sensor_0", "sensor_1", "label"]
    assert not outliers["sensor_0"].any()
    assert not outliers["label"].any()
    assert bool(outliers.loc[idx[-1], "sensor_1"])
    assert pd.isna(imputed.to_dataframe().loc[idx[1], "label"])


def test_preprocessing_validates_outlier_threshold():
    with pytest.raises(ValueError, match="z_thresh"):
        preprocessing.detect_outliers(SensorDataset(_sample_df()), z_thresh=0)
