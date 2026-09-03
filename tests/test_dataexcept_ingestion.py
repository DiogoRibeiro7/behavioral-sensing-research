"""Regression tests for the structured sensor-ingestion exception contract."""

from __future__ import annotations

import json
import sys

import pytest
from dataexcept import (
    DataFormatError,
    DataLoadingError,
    DataValidationError,
    DependencyError,
    MissingDataError,
)

from sensor_modeling.data import loaders
from sensor_modeling.data.exceptions import (
    SensorDataFormatError,
    SensorDataLoadingError,
    SensorDataValidationError,
    SensorDependencyError,
    SensorMissingDataError,
)
from sensor_modeling.utils.data_io import SensorDataset


def test_unreadable_csv_uses_structured_loading_error(tmp_path) -> None:
    """Missing CSV files should expose DataExcept without breaking ValueError catches."""
    with pytest.raises(SensorDataLoadingError) as caught:
        loaders.load_csv(tmp_path / "missing.csv")

    assert isinstance(caught.value, DataLoadingError)
    assert isinstance(caught.value, ValueError)
    assert "Unable to read CSV file" in str(caught.value)


def test_sensor_dataset_from_csv_preserves_loading_compatibility(tmp_path) -> None:
    """The direct CSV constructor should expose the same compatibility contract."""
    with pytest.raises(SensorDataLoadingError) as caught:
        SensorDataset.from_csv(tmp_path / "missing.csv")

    assert isinstance(caught.value, DataLoadingError)
    assert isinstance(caught.value, ValueError)
    assert "Unable to read CSV file" in str(caught.value)


def test_sensor_dataset_from_csv_preserves_timestamp_compatibility(tmp_path) -> None:
    """CSV timestamp validation should remain catchable as legacy ValueError."""
    path = tmp_path / "invalid-timestamp.csv"
    path.write_text("timestamp,sensor_0\nnot-a-date,1\n", encoding="utf-8")

    with pytest.raises(SensorDataValidationError) as caught:
        SensorDataset.from_csv(path)

    assert isinstance(caught.value, DataValidationError)
    assert isinstance(caught.value, ValueError)


def test_malformed_json_uses_structured_loading_error(tmp_path) -> None:
    """JSON decoding failures should be classified as data-loading failures."""
    path = tmp_path / "malformed.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(SensorDataLoadingError) as caught:
        loaders.load_json(path)

    assert isinstance(caught.value, DataLoadingError)
    assert isinstance(caught.value, ValueError)


def test_json_shape_and_required_fields_use_specific_data_errors(tmp_path) -> None:
    """Payload shape and missing fields should have distinct DataExcept classes."""
    object_path = tmp_path / "object.json"
    object_path.write_text(json.dumps({"timestamp": "2024-01-01"}), encoding="utf-8")
    with pytest.raises(SensorDataFormatError) as malformed:
        loaders.load_json(object_path)
    assert isinstance(malformed.value, DataFormatError)
    assert isinstance(malformed.value, ValueError)

    missing_path = tmp_path / "missing.json"
    missing_path.write_text(json.dumps([{"sensor_0": 1}]), encoding="utf-8")
    with pytest.raises(SensorMissingDataError) as missing:
        loaders.load_json(missing_path)
    assert isinstance(missing.value, MissingDataError)
    assert isinstance(missing.value, ValueError)


def test_invalid_timestamp_uses_data_validation_error(tmp_path) -> None:
    """Invalid source timestamps should be data-validation failures."""
    path = tmp_path / "invalid-timestamp.json"
    path.write_text(
        json.dumps([{"timestamp": "not-a-date", "sensor_0": 1}]),
        encoding="utf-8",
    )

    with pytest.raises(SensorDataValidationError) as caught:
        loaders.load_json(path)

    assert isinstance(caught.value, DataValidationError)
    assert isinstance(caught.value, ValueError)


def test_hdf5_missing_dataset_uses_missing_data_error(tmp_path) -> None:
    """Absent HDF5 datasets should be classified separately from file-read failures."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("other", data=[[1, 0]])

    with pytest.raises(SensorMissingDataError) as caught:
        loaders.load_hdf5(path)

    assert isinstance(caught.value, MissingDataError)
    assert isinstance(caught.value, ValueError)


def test_hdf5_missing_dependency_uses_dependency_error(monkeypatch, tmp_path) -> None:
    """Optional dependency failures should remain compatible with ImportError handlers."""
    monkeypatch.setitem(sys.modules, "h5py", None)

    with pytest.raises(SensorDependencyError) as caught:
        loaders.load_hdf5(tmp_path / "data.h5")

    assert isinstance(caught.value, DependencyError)
    assert isinstance(caught.value, ImportError)


def test_streaming_bad_records_are_still_skipped() -> None:
    """Structured record errors should preserve the stream's skip-invalid contract."""
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
