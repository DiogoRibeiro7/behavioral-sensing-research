"""
Multivariate Autoregressive Model for Sensor Data

This module extends the single-sensor model to predict multiple
sensors simultaneously while modeling their interactions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

from .base_model import BernoulliAutoregressiveModel
from ...analysis.dependency_network import SensorDependencyNetwork

logger = logging.getLogger(__name__)


class MultivariateAutoregressiveModel:
    """
    Multivariate extension of the Bernoulli autoregressive model.
    Predicts multiple sensors simultaneously while modeling their interactions.
    """

    def __init__(self, sensor_names: List[str]):
        """
        Initialize multivariate model.

        Args:
            sensor_names: List of sensor names to model jointly
        """
        self.sensor_names = sensor_names
        self.n_sensors = len(sensor_names)

        # Model parameters for each sensor
        self.models = {}
        for sensor in sensor_names:
            self.models[sensor] = BernoulliAutoregressiveModel(sensor_names, sensor)

        # Cross-sensor dependency network
        self.dependency_network = SensorDependencyNetwork()
        self.network = None

    def fit_joint_model(
        self, data: pd.DataFrame, use_network_structure: bool = True
    ) -> Dict:
        """
        Fit multivariate model with optional network-informed structure.

        Args:
            data: DataFrame with sensor data
            use_network_structure: Whether to use dependency network for model selection

        Returns:
            Dictionary with fitting results for all sensors
        """
        logger.info("Fitting multivariate sensor model...")

        # Build dependency network first
        if use_network_structure:
            self.network = self.dependency_network.build_network(data)
            network_roles = self.dependency_network.identify_sensor_roles()
            logger.info(f"Network roles identified: {network_roles}")

        results = {}

        # Fit individual models for each sensor
        for sensor in self.sensor_names:
            logger.info(f"Fitting model for {sensor}...")

            # If using network structure, pre-select relevant sensors
            if use_network_structure and self.network is not None:
                # Get sensors that have causal influence on current sensor
                predecessors = list(self.network.predecessors(sensor))
                if predecessors:
                    # Pre-select sensors based on network structure
                    self.models[sensor].selected_sensors = predecessors
                    result = self.models[sensor].fit(data, perform_selection=False)
                else:
                    # No network predecessors, use regular selection
                    result = self.models[sensor].fit(data, perform_selection=True)
            else:
                # Standard fitting with automatic selection
                result = self.models[sensor].fit(data, perform_selection=True)

            results[sensor] = result

        return results

    def predict_joint_probabilities(
        self, data: pd.DataFrame, start_idx: int = None
    ) -> pd.DataFrame:
        """
        Predict probabilities for all sensors jointly.

        Args:
            data: DataFrame with sensor data
            start_idx: Starting index for prediction

        Returns:
            DataFrame with predicted probabilities for each sensor
        """
        predictions = {}

        for sensor in self.sensor_names:
            if self.models[sensor].params:
                probs = self.models[sensor].predict_probabilities(data, start_idx)
                predictions[f"{sensor}_prob"] = probs
            else:
                logger.warning(f"Model for {sensor} not fitted")

        return pd.DataFrame(predictions)

    def analyze_sensor_interactions(self, data: pd.DataFrame) -> Dict:
        """
        Comprehensive analysis of sensor interactions.

        Args:
            data: DataFrame with sensor data

        Returns:
            Dictionary with various interaction analyses
        """
        # Build network if not already built
        if self.network is None:
            self.network = self.dependency_network.build_network(data)

        # Get network statistics
        network_stats = self.dependency_network.get_network_statistics()
        sensor_roles = self.dependency_network.identify_sensor_roles()

        # Calculate mutual information between sensors
        mutual_info = self.dependency_network.calculate_mutual_information(data)

        # Find sensor clusters/communities
        communities = self.dependency_network.detect_communities()

        # Critical sensor analysis
        critical_sensors = self.dependency_network.find_critical_sensors()

        return {
            "network_statistics": network_stats,
            "sensor_roles": sensor_roles,
            "mutual_information": mutual_info,
            "communities": communities,
            "critical_sensors": critical_sensors,
            "causality_results": self.dependency_network.causality_results,
        }

    def compare_model_approaches(self, data: pd.DataFrame) -> Dict:
        """
        Compare different modeling approaches.

        Args:
            data: DataFrame with sensor data

        Returns:
            Dictionary comparing different approaches
        """
        comparison_results = {}

        # 1. Independent models (no cross-sensor effects)
        logger.info("Fitting independent models...")
        independent_bics = {}
        for sensor in self.sensor_names:
            temp_model = BernoulliAutoregressiveModel([sensor], sensor)
            result = temp_model.fit(data[sensor : sensor + 1])  # Only use target sensor
            independent_bics[sensor] = result.get("bic", float("inf"))

        comparison_results["independent_models"] = {
            "total_bic": sum(independent_bics.values()),
            "individual_bics": independent_bics,
        }

        # 2. Network-informed models
        logger.info("Fitting network-informed models...")
        network_results = self.fit_joint_model(data, use_network_structure=True)
        network_bics = {
            k: v.get("bic", float("inf")) for k, v in network_results.items()
        }

        comparison_results["network_informed_models"] = {
            "total_bic": sum(network_bics.values()),
            "individual_bics": network_bics,
        }

        # 3. Full multivariate models (all sensors as predictors)
        logger.info("Fitting full multivariate models...")
        full_results = self.fit_joint_model(data, use_network_structure=False)
        full_bics = {k: v.get("bic", float("inf")) for k, v in full_results.items()}

        comparison_results["full_multivariate_models"] = {
            "total_bic": sum(full_bics.values()),
            "individual_bics": full_bics,
        }

        # Calculate improvements
        baseline_bic = comparison_results["independent_models"]["total_bic"]
        network_bic = comparison_results["network_informed_models"]["total_bic"]
        full_bic = comparison_results["full_multivariate_models"]["total_bic"]

        comparison_results["improvements"] = {
            "network_vs_independent": baseline_bic - network_bic,
            "full_vs_independent": baseline_bic - full_bic,
            "network_vs_full": full_bic - network_bic,
        }

        return comparison_results

    def generate_interaction_report(self, data: pd.DataFrame) -> str:
        """
        Generate a comprehensive text report of sensor interactions.

        Args:
            data: DataFrame with sensor data

        Returns:
            Formatted text report
        """
        analysis = self.analyze_sensor_interactions(data)
        model_comparison = self.compare_model_approaches(data)

        report = []
        report.append("SENSOR INTERACTION ANALYSIS REPORT")
        report.append("=" * 50)

        # Network overview
        net_stats = analysis["network_statistics"]
        report.append(f"\nNETWORK OVERVIEW:")
        report.append(f"Sensors analyzed: {net_stats['num_nodes']}")
        report.append(f"Causal relationships: {net_stats['num_edges']}")
        report.append(f"Network density: {net_stats['density']:.3f}")
        report.append(f"Connected components: {net_stats['num_components']}")

        # Sensor roles
        roles = analysis["sensor_roles"]
        report.append(f"\nSENSOR ROLES:")
        for role, sensors in roles.items():
            if sensors:
                report.append(f"{role.capitalize()}: {', '.join(sensors)}")

        # Critical sensors
        critical = analysis["critical_sensors"]
        if critical["most_critical"]:
            report.append(f"\nMost critical sensor: {critical['most_critical']}")

        # Communities
        communities = analysis["communities"]
        if communities:
            report.append(f"\nSENSOR COMMUNITIES:")
            for i, community in enumerate(communities, 1):
                report.append(f"Community {i}: {', '.join(community)}")

        # Model comparison
        improvements = model_comparison["improvements"]
        report.append(f"\nMODEL COMPARISON (BIC improvements):")
        report.append(
            f"Network-informed vs Independent: {improvements['network_vs_independent']:.2f}"
        )
        report.append(
            f"Full multivariate vs Independent: {improvements['full_vs_independent']:.2f}"
        )
        report.append(
            f"Network-informed vs Full: {improvements['network_vs_full']:.2f}"
        )

        # Significant relationships
        significant = analysis["causality_results"][
            analysis["causality_results"]["causality_detected"]
        ]
        if len(significant) > 0:
            report.append(f"\nSIGNIFICANT CAUSAL RELATIONSHIPS:")
            for _, row in significant.head(10).iterrows():  # Show top 10
                report.append(
                    f"{row['cause']} → {row['effect']} (p={row['p_value']:.4f})"
                )

        return "\n".join(report)

    def plot_interaction_summary(self, data: pd.DataFrame):
        """
        Create comprehensive visualization of sensor interactions.

        Args:
            data: DataFrame with sensor data
        """
        analysis = self.analyze_sensor_interactions(data)

        # Create subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Sensor Interaction Analysis Summary", fontsize=16, y=0.98)

        # Plot 1: Dependency Network
        plt.subplot(2, 3, 1)
        try:
            if self.network and len(self.network.edges()) > 0:
                pos = nx.spring_layout(self.network, k=1, iterations=50)
                nx.draw(
                    self.network,
                    pos,
                    with_labels=True,
                    node_color="lightblue",
                    node_size=500,
                    arrows=True,
                    arrowsize=20,
                    font_size=8,
                )
                plt.title("Dependency Network")
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No significant\ndependencies found",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
                plt.title("Dependency Network")
        except Exception:
            plt.text(
                0.5,
                0.5,
                "Network visualization\nnot available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Dependency Network")
        plt.axis("off")

        # Plot 2: Causality Matrix
        plt.subplot(2, 3, 2)
        if self.dependency_network.causality_results is not None:
            pivot_data = self.dependency_network.causality_results.pivot(
                index="effect", columns="cause", values="test_statistic"
            )
            sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="Reds", cbar=False)
            plt.title("Causality Test Statistics")
        else:
            plt.text(0.5, 0.5, "No causality data", ha="center", va="center")
            plt.title("Causality Matrix")

        # Plot 3: Mutual Information
        plt.subplot(2, 3, 3)
        mi_matrix = analysis["mutual_information"]
        sns.heatmap(mi_matrix, annot=True, fmt=".3f", cmap="Blues", cbar=False)
        plt.title("Mutual Information")

        # Plot 4: Sensor Roles
        plt.subplot(2, 3, 4)
        roles = analysis["sensor_roles"]
        role_counts = {role: len(sensors) for role, sensors in roles.items()}
        if any(role_counts.values()):
            plt.bar(role_counts.keys(), role_counts.values())
            plt.title("Sensor Role Distribution")
            plt.xticks(rotation=45)
        else:
            plt.text(
                0.5,
                0.5,
                "No distinct roles\nidentified",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Sensor Roles")

        # Plot 5: Network Metrics
        plt.subplot(2, 3, 5)
        net_stats = analysis["network_statistics"]
        metrics = ["num_edges", "density", "num_components"]
        values = [net_stats[metric] for metric in metrics]
        labels = ["Edges", "Density", "Components"]
        plt.bar(labels, values)
        plt.title("Network Metrics")
        plt.xticks(rotation=45)

        # Plot 6: Communities
        plt.subplot(2, 3, 6)
        communities = analysis["communities"]
        if communities:
            comm_sizes = [len(comm) for comm in communities]
            plt.bar(range(len(comm_sizes)), comm_sizes)
            plt.xlabel("Community")
            plt.ylabel("Size")
            plt.title(f"Communities ({len(communities)} found)")
        else:
            plt.text(
                0.5,
                0.5,
                "No communities\ndetected",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Community Structure")

        plt.tight_layout()
        plt.show()
