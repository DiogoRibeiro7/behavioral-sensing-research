"""
Sensor Dependency Network Analysis

This module builds and analyzes cross-sensor dependency networks 
using Granger causality and network analysis techniques.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from typing import Dict, List, Tuple
import logging

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
        self.significance_level = significance_level
        self.granger_test = GrangerCausalityTest()
        self.network = None
        self.causality_results = None

    def build_network(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Build sensor dependency network using Granger causality.

        Args:
            data: DataFrame with sensor data

        Returns:
            NetworkX directed graph
        """
        logger.info("Building sensor dependency network...")

        # Test all pairs for Granger causality
        self.causality_results = self.granger_test.test_all_pairs(data)

        # Create directed graph
        self.network = nx.DiGraph()

        # Add all sensors as nodes
        sensors = data.columns.tolist()
        self.network.add_nodes_from(sensors)

        # Add edges for significant causal relationships
        significant_results = self.causality_results[
            self.causality_results["causality_detected"] == True
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
            f"Network created with {len(self.network.nodes)} nodes and {len(self.network.edges)} edges"
        )
        return self.network

    def get_network_statistics(self) -> Dict:
        """Get network topology statistics."""
        if self.network is None:
            raise ValueError("Network must be built first")

        return {
            "num_nodes": len(self.network.nodes),
            "num_edges": len(self.network.edges),
            "density": nx.density(self.network),
            "is_connected": nx.is_weakly_connected(self.network),
            "num_components": nx.number_weakly_connected_components(self.network),
            "avg_clustering": nx.average_clustering(self.network.to_undirected()),
            "in_degree_centrality": nx.in_degree_centrality(self.network),
            "out_degree_centrality": nx.out_degree_centrality(self.network),
        }

    def identify_sensor_roles(self) -> Dict:
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

        roles = {
            "triggers": [],  # High out-degree, low in-degree
            "responders": [],  # High in-degree, low out-degree
            "hubs": [],  # High both in and out degree
            "isolated": [],  # Low both in and out degree
        }

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

    def detect_communities(self) -> List[List[str]]:
        """Detect communities/clusters in the sensor network."""
        if self.network is None:
            raise ValueError("Network must be built first")

        try:
            # Convert to undirected for community detection
            undirected_network = self.network.to_undirected()

            # Use greedy modularity optimization
            communities = nx.community.greedy_modularity_communities(undirected_network)
            return [list(community) for community in communities]

        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return []

    def calculate_mutual_information(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mutual information between all sensor pairs."""
        sensors = data.columns.tolist()
        n_sensors = len(sensors)
        mi_matrix = np.zeros((n_sensors, n_sensors))

        for i, sensor1 in enumerate(sensors):
            for j, sensor2 in enumerate(sensors):
                if i != j:
                    mi_matrix[i, j] = mutual_info_score(
                        data[sensor1].values, data[sensor2].values
                    )

        return pd.DataFrame(mi_matrix, index=sensors, columns=sensors)

    def find_critical_sensors(self) -> Dict:
        """
        Identify critical sensors whose removal would significantly affect network connectivity.

        Returns:
            Dictionary with criticality analysis
        """
        if self.network is None:
            raise ValueError("Network must be built first")

        original_connectivity = nx.is_weakly_connected(self.network)
        original_components = nx.number_weakly_connected_components(self.network)

        criticality_scores = {}

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
        self, figsize: Tuple[int, int] = (12, 8), node_size_factor: int = 500
    ):
        """
        Plot the sensor dependency network.

        Args:
            figsize: Figure size
            node_size_factor: Factor to scale node sizes
        """
        if self.network is None:
            raise ValueError("Network must be built first")

        plt.figure(figsize=figsize)

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
            self.network, pos, node_size=node_sizes, node_color="lightblue", alpha=0.7
        )

        nx.draw_networkx_edges(
            self.network,
            pos,
            width=edge_weights,
            edge_color="gray",
            arrows=True,
            arrowsize=20,
            alpha=0.6,
        )

        nx.draw_networkx_labels(self.network, pos, font_size=10)

        plt.title("Sensor Dependency Network\n(Arrows show causal direction)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def plot_causality_matrix(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot heatmap of causality test results.

        Args:
            figsize: Figure size
        """
        if self.causality_results is None:
            raise ValueError("Network must be built first")

        # Create pivot table for heatmap
        pivot_data = self.causality_results.pivot(
            index="effect", columns="cause", values="test_statistic"
        )

        plt.figure(figsize=figsize)

        # Create heatmap
        mask = pivot_data.isna()
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".2f",
            mask=mask,
            cmap="Reds",
            cbar_kws={"label": "Granger Causality Test Statistic"},
        )

        plt.title("Sensor Causality Matrix\n(Rows: Effects, Columns: Causes)")
        plt.xlabel("Potential Causes")
        plt.ylabel("Effects")
        plt.tight_layout()
        plt.show()

    def export_network_data(self, filename: str = None) -> Dict:
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

            with open(filename, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(f"Network data exported to {filename}")

        return export_data
