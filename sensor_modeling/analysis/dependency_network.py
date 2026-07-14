"""
Sensor Dependency Network Analysis

This module builds and analyzes cross-sensor dependency networks
using Granger causality and network analysis techniques.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mutual_info_score

from .granger_causality import GrangerCausalityTest

logger = logging.getLogger(__name__)


class SensorDependencyNetwork:
    """
    Build and analyze cross-sensor dependency networks.
    Creates directed graphs showing causal relationships between sensors.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize dependency network builder.

        Args:
            significance_level: P-value threshold for including edges
        """
        if not 0 < significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")
        self.significance_level = significance_level
        self.granger_test = GrangerCausalityTest()
        self.network: nx.DiGraph | None = None
        self.causality_results: pd.DataFrame | None = None

    def build_network(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Build sensor dependency network using Granger causality.

        Args:
            data: DataFrame with sensor data

        Returns:
            NetworkX directed graph
        """
        logger.info("Building sensor dependency network...")
        sensor_data = self._prepare_sensor_data(data)

        # Test all pairs for Granger causality
        self.causality_results = self.granger_test.test_all_pairs(sensor_data)

        # Create directed graph
        self.network = nx.DiGraph()

        # Add all sensors as nodes
        sensors = sensor_data.columns.tolist()
        self.network.add_nodes_from(sensors)

        # Add edges for significant causal relationships
        significant_results = self.causality_results[
            self.causality_results["causality_detected"]
        ]

        for _, row in significant_results.iterrows():
            self.network.add_edge(
                row["cause"],
                row["effect"],
                weight=row["test_statistic"],
                p_value=row["p_value"],
                lags=row["lags_used"],
            )

        logger.info(
            "Network created with %d nodes and %d edges",
            len(self.network.nodes),
            len(self.network.edges),
        )
        return self.network

    def _prepare_sensor_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a numeric binary sensor frame for network analysis."""
        if data.empty:
            raise ValueError("dependency network analysis requires at least one row")

        sensor_data = data.select_dtypes(include="number")
        if sensor_data.empty:
            raise ValueError(
                "dependency network analysis requires at least one numeric sensor column"
            )

        if sensor_data.isna().any().any():
            raise ValueError("dependency network analysis does not accept NaN values")

        non_binary = [
            column
            for column in sensor_data.columns
            if not set(sensor_data[column].unique()) <= {0, 1}
        ]
        if non_binary:
            raise ValueError(
                "dependency network analysis requires binary sensor columns: "
                + ", ".join(map(str, non_binary))
            )

        return sensor_data.copy()

    def get_network_statistics(self) -> dict[str, Any]:
        """Get network topology statistics."""
        if self.network is None:
            raise ValueError("Network must be built first")

        is_connected = (
            nx.is_weakly_connected(self.network)
            if self.network.number_of_nodes()
            else False
        )

        return {
            "num_nodes": len(self.network.nodes),
            "num_edges": len(self.network.edges),
            "density": nx.density(self.network),
            "is_connected": is_connected,
            "num_components": nx.number_weakly_connected_components(self.network),
            "avg_clustering": nx.average_clustering(self.network.to_undirected()),
            "in_degree_centrality": nx.in_degree_centrality(self.network),
            "out_degree_centrality": nx.out_degree_centrality(self.network),
        }

    def identify_sensor_roles(self) -> dict[str, list[str]]:
        """
        Identify different roles of sensors in the network.

        Returns:
            Dictionary categorizing sensors by their network roles
        """
        if self.network is None:
            raise ValueError("Network must be built first")

        # Calculate centrality measures
        in_degree = dict(self.network.in_degree())
        out_degree = dict(self.network.out_degree())

        roles: dict[str, list[str]] = {
            "triggers": [],  # High out-degree, low in-degree
            "responders": [],  # High in-degree, low out-degree
            "hubs": [],  # High both in and out degree
            "isolated": [],  # Low both in and out degree
        }

        if self.network.number_of_nodes() == 0:
            return roles
        if all(
            degree == 0
            for degree in list(in_degree.values()) + list(out_degree.values())
        ):
            roles["isolated"] = list(self.network.nodes())
            return roles

        # Define thresholds (can be adjusted)
        high_threshold = np.percentile(
            list(in_degree.values()) + list(out_degree.values()), 75
        )
        low_threshold = np.percentile(
            list(in_degree.values()) + list(out_degree.values()), 25
        )

        for node in self.network.nodes():
            in_deg = in_degree[node]
            out_deg = out_degree[node]

            if out_deg >= high_threshold and in_deg <= low_threshold:
                roles["triggers"].append(node)
            elif in_deg >= high_threshold and out_deg <= low_threshold:
                roles["responders"].append(node)
            elif in_deg >= high_threshold and out_deg >= high_threshold:
                roles["hubs"].append(node)
            else:
                roles["isolated"].append(node)

        return roles

    def detect_communities(self) -> list[list[str]]:
        """Detect communities/clusters in the sensor network."""
        if self.network is None:
            raise ValueError("Network must be built first")

        if self.network.number_of_nodes() == 0:
            return []
        if self.network.number_of_edges() == 0:
            return [[node] for node in self.network.nodes()]

        try:
            # Convert to undirected for community detection
            undirected_network = self.network.to_undirected()

            # Use greedy modularity optimization
            communities = nx.community.greedy_modularity_communities(undirected_network)
            return [list(community) for community in communities]

        except (nx.NetworkXException, ValueError) as exc:
            logger.warning("Community detection failed: %s", exc)
            return []

    def calculate_mutual_information(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mutual information between all sensor pairs."""
        sensor_data = self._prepare_sensor_data(data)
        sensors = sensor_data.columns.tolist()
        n_sensors = len(sensors)
        mi_matrix = np.zeros((n_sensors, n_sensors))

        for i, sensor1 in enumerate(sensors):
            for j, sensor2 in enumerate(sensors):
                if i != j:
                    mi_matrix[i, j] = mutual_info_score(
                        sensor_data[sensor1].values, sensor_data[sensor2].values
                    )

        return pd.DataFrame(mi_matrix, index=sensors, columns=sensors)

    def find_critical_sensors(self) -> dict[str, Any]:
        """
        Identify critical sensors whose removal would significantly affect network connectivity.

        Returns:
            Dictionary with criticality analysis
        """
        if self.network is None:
            raise ValueError("Network must be built first")

        original_components = nx.number_weakly_connected_components(self.network)

        criticality_scores: dict[str, dict[str, float]] = {}

        for node in self.network.nodes():
            # Create network without this node
            temp_network = self.network.copy()
            temp_network.remove_node(node)

            # Calculate impact on connectivity
            new_components = nx.number_weakly_connected_components(temp_network)
            component_change = new_components - original_components

            # Calculate impact on edges
            edges_lost = len(self.network.edges()) - len(temp_network.edges())

            # Criticality score combines connectivity and edge impact
            criticality_scores[node] = {
                "component_change": component_change,
                "edges_lost": edges_lost,
                "criticality_score": component_change + 0.1 * edges_lost,
            }

        # Sort by criticality
        sorted_sensors = sorted(
            criticality_scores.items(),
            key=lambda x: x[1]["criticality_score"],
            reverse=True,
        )

        return {
            "criticality_rankings": sorted_sensors,
            "most_critical": sorted_sensors[0][0] if sorted_sensors else None,
            "original_components": original_components,
        }

    def plot_network(
        self,
        figsize: tuple[int, int] = (12, 8),
        node_size_factor: int = 500,
        *,
        show: bool = True,
    ):
        """
        Plot the sensor dependency network.

        Args:
            figsize: Figure size
            node_size_factor: Factor to scale node sizes
            show: Whether to display the figure immediately
        """
        if self.network is None:
            raise ValueError("Network must be built first")

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate node sizes based on degree centrality
        centrality = nx.degree_centrality(self.network.to_undirected())
        node_sizes = [
            centrality[node] * node_size_factor + 100 for node in self.network.nodes()
        ]

        # Calculate edge weights for visualization
        edge_weights = [
            self.network[u][v]["weight"] / 10 for u, v in self.network.edges()
        ]

        # Use spring layout for positioning
        pos = nx.spring_layout(self.network, k=1, iterations=50)

        # Draw network
        nx.draw_networkx_nodes(
            self.network,
            pos,
            node_size=node_sizes,
            node_color="lightblue",
            alpha=0.7,
            ax=ax,
        )

        nx.draw_networkx_edges(
            self.network,
            pos,
            width=edge_weights,
            edge_color="gray",
            arrows=True,
            arrowsize=20,
            alpha=0.6,
            ax=ax,
        )

        nx.draw_networkx_labels(self.network, pos, font_size=10, ax=ax)

        ax.set_title("Sensor Dependency Network\n(Arrows show causal direction)")
        ax.axis("off")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_causality_matrix(
        self, figsize: tuple[int, int] = (10, 8), *, show: bool = True
    ):
        """
        Plot heatmap of causality test results.

        Args:
            figsize: Figure size
            show: Whether to display the figure immediately
        """
        if self.causality_results is None:
            raise ValueError("Network must be built first")

        # Create pivot table for heatmap
        pivot_data = self.causality_results.pivot(
            index="effect", columns="cause", values="test_statistic"
        )

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        mask = pivot_data.isna()
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".2f",
            mask=mask,
            cmap="Reds",
            cbar_kws={"label": "Granger Causality Test Statistic"},
            ax=ax,
        )

        ax.set_title("Sensor Causality Matrix\n(Rows: Effects, Columns: Causes)")
        ax.set_xlabel("Potential Causes")
        ax.set_ylabel("Effects")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def export_network_data(self, filename: str | Path | None = None) -> dict[str, Any]:
        """
        Export network data for external analysis.

        Args:
            filename: Optional filename to save data

        Returns:
            Dictionary with all network data
        """
        if self.network is None:
            raise ValueError("Network must be built first")

        export_data = {
            "nodes": list(self.network.nodes()),
            "edges": [(u, v, self.network[u][v]) for u, v in self.network.edges()],
            "network_statistics": self.get_network_statistics(),
            "sensor_roles": self.identify_sensor_roles(),
            "communities": self.detect_communities(),
            "causality_results": self.causality_results.to_dict("records"),
            "critical_sensors": self.find_critical_sensors(),
        }

        if filename:
            import json

            path = Path(filename)
            path.write_text(json.dumps(export_data, indent=2, default=str))
            logger.info("Network data exported to %s", path)

        return export_data
