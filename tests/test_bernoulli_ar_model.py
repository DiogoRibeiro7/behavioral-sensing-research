"""Tests for Bernoulli autoregressive model contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sensor_modeling.models.bernoulli_ar.base_model import (
    BernoulliAutoregressiveModel,
)
from sensor_modeling.utils.data_io import SensorDataset


def _binary_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=12, freq="15min")
    return pd.DataFrame(
        {
            "sensor_0": [0, 1] * 6,
            "sensor_1": [1, 0] * 6,
        },
        index=index,
    )


def test_bernoulli_ar_fit_accepts_sensor_dataset() -> None:
    dataset = SensorDataset(_binary_frame())
    model = BernoulliAutoregressiveModel(["sensor_0", "sensor_1"], "sensor_0")

    result = model.fit(dataset, perform_selection=False)

    assert "convergence" in result


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            pd.DataFrame(columns=["sensor_0", "sensor_1"]),
            "at least one observation",
        ),
        (_binary_frame().drop(columns=["sensor_1"]), "Missing sensor columns"),
        (_binary_frame().assign(sensor_1=["on", "off"] * 6), "must be numeric"),
        (_binary_frame().assign(sensor_1=[np.nan, 0] * 6), "must not contain NaN"),
        (_binary_frame().assign(sensor_1=[2, 0] * 6), "must be binary"),
    ],
)
def test_bernoulli_ar_fit_validates_training_frame(
    frame: pd.DataFrame, message: str
) -> None:
    model = BernoulliAutoregressiveModel(["sensor_0", "sensor_1"], "sensor_0")

    with pytest.raises(ValueError, match=message):
        model.fit(frame, perform_selection=False)


def test_bernoulli_ar_fit_requires_target_in_sensor_names() -> None:
    model = BernoulliAutoregressiveModel(["sensor_0"], "sensor_1")

    with pytest.raises(ValueError, match="target_sensor"):
        model.fit(_binary_frame()[["sensor_0"]], perform_selection=False)


def test_fit_model_subset_returns_infinity_on_optimizer_error(monkeypatch) -> None:
    model = BernoulliAutoregressiveModel(["sensor_0", "sensor_1"], "sensor_0")
    model._set_training_arrays(_binary_frame())

    def raise_optimizer_error(*args, **kwargs):
        raise ValueError("bad optimizer inputs")

    monkeypatch.setattr(
        "sensor_modeling.models.bernoulli_ar.base_model.minimize",
        raise_optimizer_error,
    )

    assert model._fit_model_subset([], include_seasonal=False) == float("inf")
