"""Tests for multivariate Bernoulli autoregressive wrappers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pytest
from matplotlib.figure import Figure

import sensor_modeling.models.bernoulli_ar.multivariate_model as multivariate_module
from sensor_modeling.models.bernoulli_ar.multivariate_model import (
    MultivariateAutoregressiveModel,
)


def _binary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor_0": [0, 1] * 6,
            "sensor_1": [1, 0] * 6,
        },
        index=pd.date_range("2024-01-01", periods=12, freq="15min"),
    )


def test_multivariate_model_requires_sensors() -> None:
    with pytest.raises(ValueError, match="sensor_names"):
        MultivariateAutoregressiveModel([])


def test_compare_model_approaches_fits_independent_single_sensor_frames(
    monkeypatch,
) -> None:
    fitted_columns: list[list[str]] = []

    class _FakeBernoulliModel:
        def __init__(self, sensor_names, target_sensor):
            self.sensor_names = sensor_names
            self.target_sensor = target_sensor

        def fit(self, data, perform_selection=True):
            fitted_columns.append(list(data.columns))
            return {"bic": 1.0, "convergence": True}

    def fake_fit_joint_model(self, data, use_network_structure=True):
        if use_network_structure:
            return {"sensor_0": {"bic": 0.75}, "sensor_1": {"bic": 0.75}}
        return {"sensor_0": {"bic": 0.5}, "sensor_1": {"bic": 0.5}}

    monkeypatch.setattr(
        multivariate_module,
        "BernoulliAutoregressiveModel",
        _FakeBernoulliModel,
    )
    monkeypatch.setattr(
        MultivariateAutoregressiveModel,
        "fit_joint_model",
        fake_fit_joint_model,
    )

    comparison = MultivariateAutoregressiveModel(
        ["sensor_0", "sensor_1"]
    ).compare_model_approaches(_binary_frame())

    assert fitted_columns == [["sensor_0"], ["sensor_1"]]
    assert comparison["independent_models"]["total_bic"] == 2.0
    assert comparison["network_informed_models"]["total_bic"] == 1.5
    assert comparison["full_multivariate_models"]["total_bic"] == 1.0


def test_plot_interaction_summary_returns_figure_without_show(monkeypatch) -> None:
    model = MultivariateAutoregressiveModel(["sensor_0", "sensor_1"])
    model.network = nx.DiGraph()
    model.network.add_nodes_from(["sensor_0", "sensor_1"])

    def fake_analysis(data):
        return {
            "network_statistics": {
                "num_nodes": 2,
                "num_edges": 0,
                "density": 0.0,
                "num_components": 2,
            },
            "sensor_roles": {
                "triggers": [],
                "responders": [],
                "hubs": [],
                "isolated": ["sensor_0", "sensor_1"],
            },
            "mutual_information": pd.DataFrame(
                [[0.0, 0.1], [0.1, 0.0]],
                index=["sensor_0", "sensor_1"],
                columns=["sensor_0", "sensor_1"],
            ),
            "communities": [["sensor_0"], ["sensor_1"]],
            "critical_sensors": {"most_critical": "sensor_0"},
            "causality_results": None,
        }

    monkeypatch.setattr(model, "analyze_sensor_interactions", fake_analysis)

    fig = model.plot_interaction_summary(_binary_frame(), show=False)

    assert isinstance(fig, Figure)
    plt.close(fig)
