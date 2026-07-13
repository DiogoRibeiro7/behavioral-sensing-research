"""
Main Example and Testing Module

This module demonstrates how to use all the sensor modeling components
together and provides comprehensive testing examples.
"""

import logging

from sensor_modeling.analysis import (
    GrangerCausalityTest,
    SensorDependencyNetwork,
    calculate_behavioral_metrics,
)

# Import our modules
from sensor_modeling.models.bernoulli_ar import (
    BernoulliAutoregressiveModel,
    MultivariateAutoregressiveModel,
)
from sensor_modeling.utils import (
    create_model_comparison_report,
    export_analysis_results,
    plot_quantile_intervals,
    plot_sensor_activity_patterns,
    simulate_sensor_data,
    validate_model_predictions,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_single_sensor_example():
    """Run example with single sensor modeling (original functionality)."""
    print("\n" + "=" * 60)
    print("SINGLE SENSOR MODEL EXAMPLE")
    print("=" * 60)

    # Simulate data
    print("Simulating sensor data...")
    data = simulate_sensor_data(n_days=30, n_sensors=4)

    # Split into training and testing
    split_idx = len(data) // 2
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    print(f"Training data: {len(train_data)} intervals")
    print(f"Testing data: {len(test_data)} intervals")

    # Initialize and fit model
    target_sensor = "sensor_0"
    model = BernoulliAutoregressiveModel(data.columns.tolist(), target_sensor)

    print(f"\nFitting model for {target_sensor}...")
    result = model.fit(train_data)

    if result["convergence"]:
        print("\nModel Parameters:")
        for param, value in result["parameters"].items():
            print(f"  {param}: {value:.4f}")

        print(f"\nSelected sensors: {result['selected_sensors']}")
        print(f"Include seasonal: {result['include_seasonal']}")
        print(f"BIC: {result['bic']:.2f}")

        # Validate model
        validation = validate_model_predictions(model, test_data, target_sensor)

        if validation["validation_successful"]:
            print("\nValidation Results:")
            print(
                f"Points outside 95% interval: {validation['outside_interval_count']}/96"
            )
            print(
                f"Outside percentage: {validation['outside_interval_percentage']:.1f}%"
            )
            print("Expected: ~5.0%")
            print(f"Well calibrated: {validation['is_well_calibrated']}")

            # Plot results
            try:
                plot_quantile_intervals(
                    validation["actual_counts"],
                    validation["quantile_info"],
                    "Single Sensor Model Validation",
                    target_sensor,
                )
            except Exception as e:
                print(f"Plotting skipped: {e}")

    return {
        "model": model,
        "result": result,
        "validation": validation if "validation" in locals() else None,
    }


def run_granger_causality_example():
    """Run example with Granger causality testing."""
    print("\n" + "=" * 60)
    print("GRANGER CAUSALITY ANALYSIS EXAMPLE")
    print("=" * 60)

    # Simulate data with stronger interactions
    data = simulate_sensor_data(n_days=45, n_sensors=5, interaction_strength=0.5)

    # Initialize Granger test
    granger_test = GrangerCausalityTest(max_lags=3)

    # Test all sensor pairs
    print("Testing causality for all sensor pairs...")
    causality_results = granger_test.test_all_pairs(data)

    # Show significant relationships
    significant = causality_results[causality_results["causality_detected"]]

    print(f"\nFound {len(significant)} significant causal relationships:")
    for _, row in significant.iterrows():
        print(
            f"  {row['cause']} → {row['effect']} "
            + f"(test stat: {row['test_statistic']:.3f}, p-value: {row['p_value']:.4f})"
        )

    # Create summary
    summary = granger_test.create_causality_summary(causality_results)

    print("\nCausality Summary:")
    print(f"  Total pairs tested: {summary['total_pairs_tested']}")
    print(f"  Significant relationships: {summary['significant_relationships']}")
    print(f"  Causality rate: {summary['causality_rate']:.3f}")

    if summary["top_causes"]:
        print(f"  Top causal sensors: {list(summary['top_causes'].keys())[:3]}")

    if summary["bidirectional_relationships"]:
        print(
            f"  Bidirectional relationships: {summary['bidirectional_relationships']}"
        )

    return {"causality_results": causality_results, "summary": summary}


def run_dependency_network_example():
    """Run example with dependency network analysis."""
    print("\n" + "=" * 60)
    print("SENSOR DEPENDENCY NETWORK EXAMPLE")
    print("=" * 60)

    # Use data with interactions
    data = simulate_sensor_data(n_days=40, n_sensors=6, interaction_strength=0.4)

    # Build network
    network_builder = SensorDependencyNetwork(
        significance_level=0.1
    )  # Lenient for demo
    network = network_builder.build_network(data)

    # Get network statistics
    net_stats = network_builder.get_network_statistics()
    print("\nNetwork Statistics:")
    print(f"  Nodes: {net_stats['num_nodes']}")
    print(f"  Edges: {net_stats['num_edges']}")
    print(f"  Density: {net_stats['density']:.4f}")
    print(f"  Connected: {net_stats['is_connected']}")
    print(f"  Components: {net_stats['num_components']}")

    # Identify sensor roles
    roles = network_builder.identify_sensor_roles()
    print("\nSensor Roles:")
    for role, sensors in roles.items():
        if sensors:
            print(f"  {role.capitalize()}: {sensors}")

    # Find communities
    communities = network_builder.detect_communities()
    if communities:
        print("\nCommunities detected:")
        for i, community in enumerate(communities, 1):
            print(f"  Community {i}: {community}")

    # Critical sensor analysis
    critical_analysis = network_builder.find_critical_sensors()
    if critical_analysis["most_critical"]:
        print(f"\nMost critical sensor: {critical_analysis['most_critical']}")

    # Visualization (skip if running headless)
    try:
        print("\nGenerating network visualizations...")
        network_builder.plot_network()
        network_builder.plot_causality_matrix()
    except Exception as e:
        print(f"Network visualization skipped: {e}")

    return {"network": network, "network_builder": network_builder}


def run_multivariate_model_example():
    """Run example with multivariate modeling."""
    print("\n" + "=" * 60)
    print("MULTIVARIATE MODEL EXAMPLE")
    print("=" * 60)

    # Generate data with clear interaction patterns
    data = simulate_sensor_data(n_days=50, n_sensors=4, interaction_strength=0.6)

    # Split data
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    print(f"Training: {len(train_data)} intervals")
    print(f"Testing: {len(test_data)} intervals")

    # Initialize multivariate model
    multivariate_model = MultivariateAutoregressiveModel(data.columns.tolist())

    # Fit with network-informed structure
    print("\nFitting multivariate model with network structure...")
    results = multivariate_model.fit_joint_model(train_data, use_network_structure=True)

    # Show results for each sensor
    print("\nMultivariate Model Results:")
    converged_models = 0
    for sensor, result in results.items():
        if result["convergence"]:
            converged_models += 1
            print(f"\n{sensor}:")
            print(f"  Selected sensors: {result['selected_sensors']}")
            print(f"  Include seasonal: {result['include_seasonal']}")
            print(f"  BIC: {result['bic']:.2f}")

    print(f"\nSuccessfully fitted {converged_models}/{len(results)} sensor models")

    # Generate joint predictions
    print("\nGenerating joint predictions...")
    joint_predictions = multivariate_model.predict_joint_probabilities(test_data)
    print(f"Joint predictions shape: {joint_predictions.shape}")

    # Comprehensive interaction analysis
    print("\nPerforming comprehensive interaction analysis...")
    interaction_analysis = multivariate_model.analyze_sensor_interactions(train_data)

    # Generate interaction report
    report = multivariate_model.generate_interaction_report(train_data)
    print(f"\n{report}")

    # Model comparison
    print("\nComparing different modeling approaches...")
    model_comparison = multivariate_model.compare_model_approaches(train_data)
    comparison_report = create_model_comparison_report(model_comparison)
    print(f"\n{comparison_report}")

    # Comprehensive visualization
    try:
        print("\nGenerating comprehensive interaction visualization...")
        multivariate_model.plot_interaction_summary(train_data)
    except Exception as e:
        print(f"Comprehensive visualization skipped: {e}")

    return {
        "multivariate_model": multivariate_model,
        "results": results,
        "joint_predictions": joint_predictions,
        "interaction_analysis": interaction_analysis,
        "model_comparison": model_comparison,
    }


def run_behavioral_analysis_example():
    """Run example with behavioral pattern analysis."""
    print("\n" + "=" * 60)
    print("BEHAVIORAL PATTERN ANALYSIS EXAMPLE")
    print("=" * 60)

    # Generate data
    data = simulate_sensor_data(n_days=30, n_sensors=5)

    # Calculate behavioral metrics
    print("Calculating behavioral metrics...")
    behavioral_metrics = calculate_behavioral_metrics(data)

    print("\nOverall Behavioral Patterns:")
    print(f"  Activity rate: {behavioral_metrics['overall_activity_rate']:.3f}")
    print(f"  Total activations: {behavioral_metrics['total_activations']}")
    print(f"  Peak activity hour: {behavioral_metrics['peak_activity_hour']}:00")
    print(f"  Quietest hour: {behavioral_metrics['quietest_hour']}:00")
    print(f"  Most active day: {behavioral_metrics['most_active_day']}")
    print(f"  Least active day: {behavioral_metrics['least_active_day']}")

    print("\nSensor-Specific Patterns:")
    for sensor, metrics in behavioral_metrics["sensor_metrics"].items():
        print(f"  {sensor}:")
        print(f"    Activation rate: {metrics['activation_rate']:.3f}")
        print(
            f"    Longest inactive period: {metrics['longest_inactive_period']} intervals"
        )
        print(f"    Daily variance: {metrics['daily_variance']:.2f}")

    # Plot activity patterns
    try:
        print("\nGenerating activity pattern plots...")
        plot_sensor_activity_patterns(data)
    except Exception as e:
        print(f"Activity pattern plotting skipped: {e}")

    return {"behavioral_metrics": behavioral_metrics}


def run_comprehensive_demo():
    """Run all examples in sequence for comprehensive demonstration."""
    print("COMPREHENSIVE SENSOR MODELING DEMONSTRATION")
    print("=" * 80)

    all_results = {}

    try:
        # Run all examples
        all_results["single_sensor"] = run_single_sensor_example()
        all_results["granger_causality"] = run_granger_causality_example()
        all_results["dependency_network"] = run_dependency_network_example()
        all_results["multivariate_model"] = run_multivariate_model_example()
        all_results["behavioral_analysis"] = run_behavioral_analysis_example()

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)

        print("\nSUMMARY:")
        print("- Single sensor modeling: Baseline approach from Gillam et al. (2022)")
        print("- Granger causality: Statistical testing for sensor interactions")
        print("- Dependency networks: Graph-based analysis of sensor relationships")
        print("- Multivariate modeling: Joint prediction of multiple sensors")
        print("- Behavioral analysis: Pattern extraction and characterization")

        print("\nAll components successfully demonstrated!")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"Error during demonstration: {e}")

    return all_results


if __name__ == "__main__":
    # Run comprehensive demonstration
    results = run_comprehensive_demo()

    # Optional: Export results
    try:
        export_analysis_results(results, "comprehensive_demo_results")
        print("\nResults exported to files.")
    except Exception as e:
        print(f"Export failed: {e}")
