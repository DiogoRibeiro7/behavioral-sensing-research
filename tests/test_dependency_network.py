"""Tests for sensor dependency network utilities."""

from __future__ import annotations

import json

import networkx as nx
import pandas as pd
import pytest
from matplotlib.figure import Figure

from sensor_modeling.analysis.dependency_network import SensorDependencyNetwork


def _sample_binary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor_0": [0, 1] * 6,
            "sensor_1": [1, 0] * 6,
            "label": ["room"] * 12,
        }
    )


def test_build_network_uses_numeric_binary_sensor_columns():
    network = SensorDependencyNetwork().build_network(_sample_binary_frame())

    assert set(network.nodes) == {"sensor_0", "sensor_1"}
    assert "label" not in network.nodes


def test_dependency_network_validates_inputs():
    with pytest.raises(ValueError, match="between 0 and 1"):
        SensorDependencyNetwork(significance_level=1.0)

    tester = SensorDependencyNetwork()
    with pytest.raises(ValueError, match="numeric sensor column"):
        tester.build_network(pd.DataFrame({"label": ["a", "b"]}))
    with pytest.raises(ValueError, match="binary sensor columns"):
        tester.build_network(pd.DataFrame({"sensor_0": [0, 2, 1]}))


def test_edgeless_network_statistics_roles_and_communities():
    tester = SensorDependencyNetwork()
    tester.network = nx.DiGraph()
    tester.network.add_nodes_from(["sensor_0", "sensor_1"])

    stats = tester.get_network_statistics()
    roles = tester.identify_sensor_roles()
    communities = tester.detect_communities()
    critical = tester.find_critical_sensors()

    assert stats["num_nodes"] == 2
    assert stats["num_edges"] == 0
    assert roles["isolated"] == ["sensor_0", "sensor_1"]
    assert communities == [["sensor_0"], ["sensor_1"]]
    assert critical["most_critical"] in {"sensor_0", "sensor_1"}


def test_mutual_information_returns_named_square_matrix():
    matrix = SensorDependencyNetwork().calculate_mutual_information(
        _sample_binary_frame()
    )

    assert list(matrix.index) == ["sensor_0", "sensor_1"]
    assert list(matrix.columns) == ["sensor_0", "sensor_1"]
    assert matrix.loc["sensor_0", "sensor_0"] == 0.0
    assert matrix.loc["sensor_0", "sensor_1"] > 0.0


def test_network_plots_and_export_return_artifacts(tmp_path):
    tester = SensorDependencyNetwork()
    tester.build_network(_sample_binary_frame())

    network_fig = tester.plot_network(show=False)
    matrix_fig = tester.plot_causality_matrix(show=False)
    output = tmp_path / "network.json"
    export_data = tester.export_network_data(output)

    assert isinstance(network_fig, Figure)
    assert isinstance(matrix_fig, Figure)
    assert output.exists()
    assert json.loads(output.read_text())["nodes"] == export_data["nodes"]
