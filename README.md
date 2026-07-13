# Sensor Modeling Research Toolkit

A comprehensive, research-grade Python library for analyzing, modeling, and visualizing behavioral sensor data in smart environments. The toolkit unifies multiple modeling paradigms including Bernoulli autoregressive models, hidden Markov models, change-point detection, and non-homogeneous Poisson processes into a single cohesive framework designed for ambient assisted living (AAL), digital health, and smart home research.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/DiogoRibeiro7/behavioral-sensing-research/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/behavioral-sensing-research/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/sensor-modeling/badge/?version=latest)](https://sensor-modeling.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17070041.svg)](https://doi.org/10.5281/zenodo.17070041)

## 🎯 Overview

The **Sensor Modeling Research Toolkit** addresses the growing need for reproducible, interpretable analysis of behavioral sensor streams in smart environments. Unlike general-purpose machine learning libraries, this toolkit provides domain-specific implementations optimized for the unique characteristics of ambient sensor data: irregular sampling, frequent missingness, binary activations, and the need for transparent, clinically interpretable models.

### Key Differentiators

- **Research-Grade Implementation**: Clean, documented, and tested implementations of established algorithms from recent literature
- **Unified Interface**: Consistent API across different modeling approaches for easy comparison and ensemble methods
- **Clinical Focus**: Visualization and reporting utilities designed for healthcare stakeholders and non-technical users
- **Lightweight Deployment**: Minimal dependencies and efficient implementations suitable for edge computing and real-time applications
- **Extensible Architecture**: Modular design allows researchers to easily add new algorithms and extend existing functionality

## ✨ Features

### 🔧 **Comprehensive Data Pipeline**

- **Multi-format Loaders**: Support for CSV, JSON, HDF5, and real-time streaming data
- **Robust Preprocessing**: Missing value imputation, outlier detection, temporal alignment, and data validation
- **Synthetic Data Generation**: Configurable simulation of sensor networks with ground truth for benchmarking
- **Quality Assessment**: Automated data quality reporting and sensor failure detection

### 🧠 **Advanced Modeling Capabilities**

#### **Bernoulli Autoregressive Models**

- Implementation of Gillam et al. (2022) approach for activity prediction
- Automatic sensor selection using stepwise BIC optimization
- Seasonal pattern detection and multivariate extensions
- Uncertainty quantification through prediction intervals

#### **Hidden Markov Models (HMMs)**

- Hierarchical HMMs for multi-level activity modeling ([Asghari & Nazerfard, 2019](https://arxiv.org/abs/1903.04820))
- Scaled Dirichlet HMMs with variational inference
- Heterogeneous HMMs for multi-source data integration
- Adaptive HMMs incorporating personal experience
- Circadian HMMs for rhythm monitoring applications

#### **Change-Point Detection**

- Embedding-based real-time detection ([Dadi et al., 2021](https://doi.org/10.1016/j.eswa.2021.115217))
- Energy-efficient CPAM algorithm ([Cook et al., 2020](https://doi.org/10.3390/s20010310))
- Adaptive normalization for non-stationary data
- Genetic algorithm optimization for parameter tuning
- Univariate PELT-based segmentation with configurable penalty and L1/L2 costs

#### **Non-Homogeneous Poisson Processes (NHPP)**

- B-spline intensity estimation with PELT segmentation
- Automatic model selection via AIC/BIC
- P-spline regularization for smooth intensity curves
- Time-rescaling diagnostics for model validation
- Lewis-Shedler thinning for simulation and testing

### 📊 **Advanced Analysis & Interpretation**

#### **Causal Analysis**

- Granger causality testing adapted for binary time series
- Sensor dependency network construction and analysis
- Community detection in sensor interaction graphs
- Critical sensor identification for system robustness

#### **Behavioral Metrics**

- Activity pattern recognition (peak/quiet hours, routine detection)
- Anomaly scoring using statistical and network-based approaches
- Trend detection with configurable temporal windows
- Health indicators derived from activity levels and variability

#### **Cross-Model Comparison**

- Standardized evaluation metrics across different modeling paradigms
- Statistical significance testing for model performance
- Automated hyperparameter sweeps and elbow plot generation
- Cross-validation frameworks adapted for time series data

### 🎨 **Rich Visualization & Reporting**

#### **Interactive Dashboards**

- Real-time data exploration using Plotly and Bokeh
- Parameter tuning interfaces with immediate visual feedback
- Drill-down capabilities for detected changes and anomalies
- Export functionality for presentations and publications

#### **Clinical Visualizations**

- Patient-friendly activity summaries and trend monitors
- Alert generation based on configurable clinical thresholds
- Comparison against normative population statistics
- Minimal FHIR-style observation export for clinical workflow prototyping

#### **Research Tools**

- Publication-quality figures with customizable styling
- Model diagnostic plots (residuals, QQ plots, time-rescaling)
- Performance comparison visualizations across multiple models
- Statistical test result visualization and interpretation

### 🌐 **Deployment & Integration**

#### **Command-Line Interface**

- Batch processing capabilities for large-scale experiments
- Configurable analysis pipelines with JSON/YAML configuration
- Automated report generation in multiple formats (LaTeX, HTML, minimal FHIR-style JSON)
- Integration with cluster computing environments

#### **Web Application**

- Lightweight Flask-based interface for non-technical users
- Secure file upload with authentication and validation
- Real-time analysis results and interactive visualizations
- RESTful API for integration with existing systems

## 🚀 Installation

```bash
# Basic installation
pip install -e .[dev]

# For development with all tools
pip install -e .[dev]
pre-commit install
```

## 📖 Quick Start

### Basic Usage Example

```python
from sensor_modeling.models import BernoulliAutoregressiveModel
from sensor_modeling.utils import simulate_sensor_data
import pandas as pd

# Load or simulate sensor data
data = simulate_sensor_data(n_days=30, n_sensors=4)
print(f"Generated {len(data.data)} 15-minute intervals")

# Fit Bernoulli autoregressive model
model = BernoulliAutoregressiveModel(
    sensor_names=data.data.columns.tolist(),
    target_sensor="sensor_0"
)
result = model.fit(data)

if result["convergence"]:
    print(f"Model converged with BIC: {result['bic']:.2f}")
    print(f"Selected sensors: {result['selected_sensors']}")

    # Generate predictions
    probabilities = model.predict_probabilities(data)
    print(f"Predicted activation probabilities: {probabilities[:5]}")
```

### Advanced Multi-Model Analysis

```python
from sensor_modeling.analysis import AnalysisPipeline
from sensor_modeling.models import BernoulliAutoregressiveModel
from sensor_modeling.hmm import HierarchicalHMM
from sensor_modeling.change_point import EmbeddingCPD

# Set up comprehensive analysis pipeline
pipeline = AnalysisPipeline()

# Run all available models
results = pipeline.run(data)

# Generate comprehensive reports
pipeline.generate_report(results, output_dir="analysis_output")
print("Analysis complete! Check analysis_output/ for results.")
```

### Causal Network Analysis

```python
from sensor_modeling.analysis import SensorDependencyNetwork

# Build causal dependency network
network_builder = SensorDependencyNetwork(significance_level=0.05)
network = network_builder.build_network(data.data)

# Analyze network structure
stats = network_builder.get_network_statistics()
roles = network_builder.identify_sensor_roles()
critical = network_builder.find_critical_sensors()

print(f"Network has {stats['num_edges']} causal relationships")
print(f"Most critical sensor: {critical['most_critical']}")

# Visualize network
network_builder.plot_network()
```

### Command-Line Usage

```bash
# Fit Bernoulli autoregressive model
sensor-modeling bernoulli-ar data/sensor_readings.csv kitchen_motion

# Run NHPP-PELT change-point detection  
sensor-modeling nhpp-pelt data/sensor_readings.csv motion_sensor

# Get help on available options
sensor-modeling --help
```

## 🏗️ Architecture Overview

The toolkit is organized into four primary layers designed for modularity and extensibility:

### Core Models (`sensor_modeling.models`)

- **Bernoulli Autoregressive**: Single and multivariate models for activity prediction
- **NHPP-PELT**: Non-homogeneous Poisson process with change-point segmentation
- **Change-Point Detection**: Multiple algorithms for detecting behavioral changes
- **Hidden Markov Models**: Various HMM variants for state-based modeling

### Analysis Framework (`sensor_modeling.analysis`)

- **Preprocessing**: Data cleaning, validation, and feature engineering pipelines
- **Causal Analysis**: Granger causality testing and network analysis
- **Behavioral Metrics**: Activity pattern recognition and health indicators
- **Model Comparison**: Cross-validation and statistical testing frameworks

### Visualization Suite (`sensor_modeling.visualization`)

- **Interactive**: Real-time dashboards and parameter tuning interfaces
- **Clinical**: Patient-friendly summaries and alert systems
- **Research**: Publication-quality plots and diagnostic visualizations
- **Web Application**: Browser-based interface for non-technical users

### Utilities (`sensor_modeling.utils`)

- **Data I/O**: Multi-format loaders and synthetic data generation
- **Validation**: Model performance assessment and calibration testing
- **Plotting**: Specialized plotting functions for sensor data
- **Missing Data**: Robust handling of incomplete observations

## 📈 Roadmap Progress

The high-level status table below summarizes current capabilities. See
[`ROADMAP.md`](ROADMAP.md) for release milestones, quality gates, and
longer-term priorities.

Feature                             | Status     | Implementation
----------------------------------- | ---------- | -----------------------------------------
**Bernoulli Autoregressive Models** | ✅ Complete | Single/multivariate, automatic selection
**Hidden Markov Models**            | ✅ Complete | 5 variants with different emission models
**Change Point Detection**          | 🟡 Partial | 4 algorithms, expanding to deep learning
**NHPP-PELT**                       | ✅ Complete | B-spline intensities, diagnostics
**Causal Network Analysis**         | ✅ Complete | Granger tests, network metrics
**Missing Data Handling**           | 🟡 Partial | Gap-aware fill/interpolate/drop/flag workflows with masks
**Deep Learning CPD**               | 🔵 Planned | Transformer and CNN-based approaches
**Real-time Processing**            | 🔵 Planned | Streaming algorithms and online learning
**Clinical Integration**            | 🟡 Partial | Minimal FHIR-style export, expanding toward validated HL7 profiles

## 📚 Research Foundation

This toolkit implements and extends algorithms from recent peer-reviewed research:

### Core Publications

- **Gillam et al. (2022)**: "Modeling and forecasting of at home activity in older adults using passive sensor technology" - _Computers in Biology and Medicine_
- **Asghari & Nazerfard (2019)**: "Online Human Activity Recognition Employing Hierarchical Hidden Markov Models" - _arXiv:1903.04820_
- **Dadi et al. (2021)**: "Embedding-based real-time change point detection" - _Expert Systems with Applications_
- **Cook et al. (2020)**: "Easing Power Consumption of Wearable Activity Monitoring with Change Point Detection" - _Sensors_

### Additional References

The toolkit incorporates methodologies from 20+ research papers in ambient assisted living, change-point detection, and time series analysis. See [`paper.bib`](paper.bib) for complete references.

## 🔬 Example Applications

### Smart Home Monitoring

```python
# Detect changes in daily routines
from sensor_modeling.change_point import EmbeddingCPD

cpd = EmbeddingCPD(window=7)
cpd.fit(daily_activity_data)
change_points = cpd.predict(plot=True)
print(f"Detected {len(change_points)} routine changes")
```

### Clinical Decision Support

```python
# Generate clinical alerts
from sensor_modeling.visualization.clinical import clinical_alerts

thresholds = {
    "bathroom_visits": 8,  # per day
    "sleep_duration": 4,   # hours minimum
    "activity_level": 0.1  # baseline activity
}

alerts = clinical_alerts(patient_data, thresholds)
active_alerts = [sensor for sensor, triggered in alerts.items() if triggered]
print(f"Active clinical alerts: {active_alerts}")
```

### Research Studies

```python
# Cross-model comparison for publication
from sensor_modeling.analysis.comparison import cross_validate

models = {
    "Bernoulli AR": BernoulliAutoregressiveModel(sensors, target),
    "Hierarchical HMM": HierarchicalHMM(n_states=4),
    "NHPP-PELT": NHPPPELT(NHPPConfig(n_basis=5))
}

cv_scores = cross_validate(models, dataset, n_splits=5)
print("Cross-validation results:", cv_scores)
```

## 🤝 Contributing

We welcome contributions from researchers and practitioners! The toolkit is designed to be easily extensible:

### Getting Started

1. **Fork the repository** and create your feature branch:

  ```bash
  git checkout -b feature/my-new-algorithm
  ```

2. **Install development dependencies**:

  ```bash
  pip install -e .[dev]
  pre-commit install
  ```

3. **Add your implementation** following the existing patterns:

  ```python
  # Example: New change-point detector
  from sensor_modeling.change_point.base import BaseCPD

  class MyNewCPD(BaseCPD):
      def fit(self, series):
          # Your algorithm here
          return self

      def predict(self):
          # Return change points
          return self.change_points_
  ```

4. **Write tests and documentation**:

  ```bash
  pytest tests/test_my_new_algorithm.py
  sphinx-build -b html docs docs/_build/html
  ```

5. **Submit a pull request** with:

  - Clear description of the algorithm and its benefits
  - Tests demonstrating correctness and performance
  - Documentation updates including usage examples
  - Reference to relevant publications

### Contribution Guidelines

- **Code Style**: Follow PEP 8, use type hints, write comprehensive docstrings
- **Testing**: Maintain >90% test coverage, include property-based tests for core algorithms
- **Documentation**: Update API docs and add tutorial notebooks for new features
- **Performance**: Include benchmarks for computationally intensive algorithms
- **Reproducibility**: Use fixed random seeds and provide example datasets

See <CONTRIBUTING.md> for detailed guidelines and our [Code of Conduct](CODE_OF_CONDUCT.md).
Maintainers should use [`RELEASE.md`](RELEASE.md) for the main-only release
checklist.

## 📄 License

Distributed under the [MIT License](LICENSE). This allows for both academic and commercial use while maintaining attribution to the original authors.

## 📞 Contact & Support

- **Primary Author**: Diogo Ribeiro (<dfr@esmad.ipp.pt>)
- **Institution**: ESMAD - Instituto Politécnico do Porto
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and community support
- **Security**: Follow [`SECURITY.md`](SECURITY.md) for private vulnerability reports
- **Support**: See [`SUPPORT.md`](SUPPORT.md) for the right support channel

## 📖 Documentation

- **Online Documentation**: [sensor-modeling.readthedocs.io](https://sensor-modeling.readthedocs.io)
- **API Reference**: Complete documentation of all classes and functions
- **Tutorials**: Step-by-step guides for common use cases
- **Examples**: Jupyter notebooks demonstrating advanced workflows

## 🏆 Citation

If you use this software in your research, please cite it as:

```bibtex
@software{ribeiro2025sensor,
  title={Sensor Modeling Research Toolkit},
  author={Ribeiro, Diogo},
  year={2026},
  url={https://github.com/DiogoRibeiro7/behavioral-sensing-research},
  version={0.1.3},
  doi={10.5281/zenodo.17070041}
}
```

For the underlying methodology, please also cite relevant papers listed in [`CITATION.cff`](CITATION.cff).
