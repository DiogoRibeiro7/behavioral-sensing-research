# Sensor Modeling Research Toolkit

Research-grade Python library for analyzing, modeling, and visualizing behavioral sensor data. It unifies data loaders, preprocessing, multiple modeling paradigms (Bernoulli autoregressive, NHPP-PELT, HMM variants, change-point detectors), analysis pipelines, and interactive visualizations into a single cohesive package.

## Features

- **Data pipeline**: loaders for CSV/JSON/HDF5 and streaming inputs, cleaning, validation, and synthetic data generation.
- **Modeling**: autoregressive, non-homogeneous Poisson, hidden Markov, and change-point detection models.
- **Analysis**: cross-model comparison, behavioral metrics, and reporting utilities.
- **Visualization**: clinical dashboards, research plots, interactive web applications, and benchmarking charts.
- **CLI and API**: run analyses from the command line or import modules directly.

## Installation

```bash
pip install -e .[dev]
```

## Quick Start

```bash
sensor-modeling --help
```

See the [online documentation](https://sensor-modeling.readthedocs.io) for detailed tutorials and API references, or build the docs locally from the `docs/` directory.

## Architecture Overview

The toolkit comprises four primary layers:

- **Models** (`sensor_modeling.models`) implement Bernoulli autoregressive and hidden Markov approaches.
- **Analysis** (`sensor_modeling.analysis`) provides preprocessing and evaluation pipelines.
- **Visualization** (`sensor_modeling.visualization`) offers dashboards and a Flask web app.
- **CLI** (`sensor_modeling.cli`) exposes the functionality for batch experiments.

## Usage Example

```python
from sensor_modeling.models import BernoulliAutoregressiveModel
import pandas as pd

df = pd.DataFrame({"a": [0, 1, 0], "b": [1, 0, 1]})
model = BernoulliAutoregressiveModel(df.columns.tolist(), "a")
model.fit(df)
print(model.predict(df))
```

## Roadmap Progress

| Feature | Status |
| --- | --- |
| Change Point Detection | Partially implemented |
| Deep-learning CPD | Planned |
| Missing-data handling | Planned |
| HMM variants | Implemented |
| Sensor failure detection | Basic |

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines. By participating, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md). Changes are documented in the
[Changelog](CHANGELOG.md).

## License

Distributed under the [MIT License](LICENSE).

## Research Roadmap: Sensor-Based Behavioral Modeling Extensions

This document outlines promising research directions and implementable extensions to the Bernoulli autoregressive sensor modeling framework, with specific papers, authors, and implementation details.

## Overview

The current repository implements the Gillam et al. (2022) Bernoulli autoregressive model with multivariate extensions. The following areas represent natural progressions and complementary approaches that address practical deployment challenges and extend modeling capabilities.

--------------------------------------------------------------------------------

## 1\. Change Point Detection for Activity Segmentation

### Research Foundation

Change point detection methods identify transitions between activities in sensor data streams, enabling automated activity segmentation and real-time monitoring.

### Key Papers to Implement:

**1.1 Embedding-based Real-time Change Point Detection**

- **Title:** "Embedding-based real-time change point detection with application to activity segmentation in smart home time series data"
- **Authors:** Kamalaker Dadi, et al.
- **Journal:** Expert Systems with Applications, Volume 182, 2021
- **DOI:** <https://doi.org/10.1016/j.eswa.2021.115217>
- **Key Innovation:** Neural embeddings for sensor event representation with superior performance

**1.2 Energy-Efficient Activity Monitoring**

- **Title:** "Easing Power Consumption of Wearable Activity Monitoring with Change Point Detection"
- **Authors:** Diane Cook, et al.
- **Journal:** Sensors, Volume 20, Issue 1, 2020
- **DOI:** <https://doi.org/10.3390/s20010310>
- **Key Innovation:** CPAM algorithm reducing energy consumption by 74.64%

**1.3 Deep Neural Network CPD**

- **Title:** "Real-time Change-Point Detection: A deep neural network-based adaptive approach for detecting changes in multivariate time series data"
- **Authors:** Muktesh Gupta, Rajesh Wadhvani, Akhtar Rasool
- **Journal:** Expert Systems with Applications, Volume 209, 2022
- **DOI:** <https://doi.org/10.1016/j.eswa.2022.118310>
- **Key Innovation:** End-to-end adaptive method with automatic normalization

**1.4 Meta Change Point Detection**

- **Title:** "Leveraging change point detection to discover natural experiments in data"
- **Authors:** Yolanda Hagar, Kristina Lerman, et al.
- **Journal:** EPJ Data Science, Volume 11, 2022
- **DOI:** <https://doi.org/10.1140/epjds/s13688-022-00361-7>
- **Key Innovation:** Self-supervised method for high-dimensional data

**1.5 Genetic Algorithm Optimization**

- **Title:** "Optimal Parameter Exploration for Online Change-Point Detection in Activity Monitoring Using Genetic Algorithms"
- **Authors:** Muhammad Awais, et al.
- **Journal:** Sensors, PMC5134443, 2016
- **Key Innovation:** MEWMA with genetic algorithm parameter optimization

### Implementation Components:

```python
# Proposed modules
change_point_detection.py     # Core CPD algorithms (SEP, MEWMA)
embedding_cpd.py             # Neural embedding approaches
energy_optimization.py       # CPAM energy-efficient sampling
adaptive_normalization.py    # Real-time data preprocessing
genetic_optimization.py      # GA-based parameter tuning
```

--------------------------------------------------------------------------------

## 2\. Hidden Markov Models for Sequential Activity Modeling

### Research Foundation

HMMs capture sequential patterns in activity data and handle temporal dependencies differently than autoregressive approaches.

### Key Papers to Implement:

**2.1 Hierarchical Hidden Markov Models**

- **Title:** "Online Human Activity Recognition Employing Hierarchical Hidden Markov Models"
- **Authors:** Parviz Asghari, Ehsan Nazerfard
- **arXiv:** 1903.04820, 2019
- **Key Innovation:** Online activity recognition with interrupted activity detection

**2.2 Scaled Dirichlet-based HMMs**

- **Title:** "Human Activity Recognition with an HMM-Based Generative Model"
- **Authors:** Nastaran Manouchehri, Nizar Bouguila
- **Journal:** Sensors, Volume 23, Issue 3, 2023
- **DOI:** <https://doi.org/10.3390/s23031390>
- **Key Innovation:** Scaled Dirichlet distribution with variational inference

**2.3 Heterogeneous HMMs**

- **Title:** "Heterogeneous Hidden Markov Models for Sleep Activity Recognition from Multi-Source Passively Sensed Data"
- **Authors:** Fernando Moreno-Pino, et al.
- **arXiv:** 2211.10371, 2022
- **Key Innovation:** Multi-source passive sensing integration

**2.4 Adaptive HMMs with Personal Experience**

- **Title:** "An Adaptive Hidden Markov Model for Activity Recognition Based on a Wearable Multi-Sensor Device"
- **Authors:** Multiple authors
- **Journal:** PMC5729042, 2017
- **Key Innovation:** Personal experience incorporation as prior knowledge

**2.5 Circadian Rhythm Monitoring**

- **Title:** "Hidden Markov models for monitoring circadian rhythmicity in telemetric activity data"
- **Authors:** Multiple authors
- **Journal:** Journal of The Royal Society Interface, 2017
- **DOI:** <https://doi.org/10.1098/rsif.2017.0885>
- **Key Innovation:** Circadian parameter estimation from activity transitions

**2.6 Smart Annotation for Cyclic Activities**

- **Title:** "Hidden Markov Model-Based Smart Annotation for Benchmark Cyclic Activity Recognition Database Using Wearables"
- **Authors:** Multiple authors
- **Journal:** Sensors, Volume 19, Issue 8, 2019
- **DOI:** <https://doi.org/10.3390/s19081820>
- **Key Innovation:** Hierarchical HMM for automated activity labeling

### Implementation Components:

```python
# Proposed modules
hierarchical_hmm.py          # Multi-level activity modeling
scaled_dirichlet_hmm.py      # Variational inference implementation
heterogeneous_hmm.py         # Multi-source data integration
adaptive_hmm.py              # Personal experience incorporation
circadian_hmm.py             # Rhythm monitoring applications
smart_annotation.py          # Automated labeling pipeline
```

--------------------------------------------------------------------------------

## 3\. Sensor Failure Detection and System Reliability

### Research Foundation

Practical AAL deployments require robust handling of sensor failures, distinguishable from legitimate behavioral changes.

### Key Papers to Implement:

**3.1 Association Rule Mining for Sensor Failures**

- **Title:** "Sensor Failure Detection in Ambient Assisted Living Using Association Rule Mining"
- **Authors:** Multiple authors
- **Journal:** Sensors, Volume 20, Issue 23, 2020
- **DOI:** <https://doi.org/10.3390/s20236760>
- **Key Innovation:** Rule mining for fault-free sensor correlations

### Implementation Components:

```python
# Proposed modules
sensor_failure_detection.py  # Association rule-based failure detection
correlation_mining.py        # Sensor relationship extraction
health_monitoring.py         # Real-time sensor status tracking
fault_isolation.py           # Failed sensor identification and isolation
```

--------------------------------------------------------------------------------

## 4\. Missing Data Handling and Robustness

### Research Foundation

Real-world sensor deployments face data loss from connectivity issues, failures, or environmental interference.

### Key Papers to Implement:

**4.1 Activity Recognition with Missing Data**

- **Title:** "A Method for Sensor-Based Activity Recognition in Missing Data Scenario"
- **Authors:** Multiple authors
- **Journal:** Sensors, Volume 20, Issue 14, 2020
- **DOI:** <https://doi.org/10.3390/s20143811>
- **Key Innovation:** Recognition without data imputation using statistical features

### Implementation Components:

```python
# Proposed modules
missing_data_handler.py      # Robust inference with missing observations
robust_features.py           # Missing-data-aware feature extraction
adaptive_modeling.py         # Dynamic model complexity adjustment
robustness_evaluation.py     # Performance assessment under data loss
```

--------------------------------------------------------------------------------

## 5\. Multi-Modal Sensor Integration and AAL Systems

### Research Foundation

Modern AAL systems incorporate diverse sensor types requiring sophisticated integration strategies.

### Key Papers to Implement:

**5.1 Comprehensive AAL Review**

- **Title:** "Ambient Assisted Living: A Review of Technologies, Methodologies and Future Perspectives for Healthy Aging of Population"
- **Authors:** Grazia Cicirelli, Roberto Marani, Antonio Petitti, Annalisa Milella, Tiziana D'Orazio
- **Journal:** Sensors, Volume 21, Issue 10, 2021
- **DOI:** <https://doi.org/10.3390/s21103549>
- **Key Innovation:** Comprehensive analysis of AAL system integration approaches

**5.2 AAL Scoping Review**

- **Title:** "Ambient Assisted Living: Scoping Review of Artificial Intelligence Models, Domains, Technology, and Concerns"
- **Authors:** Multiple authors
- **Journal:** Journal of Medical Internet Research, 2022
- **DOI:** <https://doi.org/10.2196/36553>
- **Key Innovation:** AI model categorization for AAL applications

**5.3 Ontology-Based AAL Systems**

- **Title:** "OntoDomus: A Semantic Model for Ambient Assisted Living System Based on Smart Homes"
- **Authors:** Multiple authors
- **Journal:** Electronics, Volume 11, Issue 7, 2022
- **DOI:** <https://doi.org/10.3390/electronics11071143>
- **Key Innovation:** Semantic modeling for AAL system integration

### Implementation Components:

```python
# Proposed modules
multimodal_fusion.py         # Cross-sensor data integration
aal_framework.py             # Complete AAL system architecture
semantic_modeling.py         # Ontology-based sensor integration
domain_adaptation.py         # Cross-domain model transfer
```

--------------------------------------------------------------------------------

## 6\. General Change Point Detection Methods

### Research Foundation

Foundational change point detection methods applicable across domains.

### Key Papers to Implement:

**6.1 Comprehensive CPD Survey**

- **Title:** "A Survey of Methods for Time Series Change Point Detection"
- **Authors:** Multiple authors
- **Journal:** PMC5464762, 2017
- **Key Innovation:** Comprehensive review of CPD methods for time series

**6.2 Physical Activity Change Detection**

- **Title:** "Unsupervised detection and analysis of changes in everyday physical activity data"
- **Authors:** Multiple authors
- **Journal:** Journal of Biomedical Informatics, Volume 63, 2016
- **DOI:** <https://doi.org/10.1016/j.jbi.2016.07.020>
- **Key Innovation:** PACD framework for physical activity monitoring

**6.3 Real-Time Smart Home CPD**

- **Title:** "Real-Time Change Point Detection with application to Smart Home Time Series Data"
- **Authors:** Multiple authors
- **Journal:** PMC9328027, 2022
- **Key Innovation:** SEP algorithm for smart home sensor data

### Implementation Components:

```python
# Proposed modules
cpd_survey_methods.py        # Implementation of surveyed CPD methods
physical_activity_cpd.py     # PACD framework implementation
sep_algorithm.py             # SEParation distance-based CPD
smart_home_cpd.py           # Smart home specific optimizations
```

--------------------------------------------------------------------------------

## Implementation Priority and Timeline

### Phase 1 (Months 1-3): High Impact, Moderate Complexity

1. **Change Point Detection Module**

  - Start with: Gupta et al. (2022) adaptive deep neural network approach
  - Add: Cook et al. (2020) CPAM energy optimization

2. **Basic Hidden Markov Models**

  - Start with: Manouchehri & Bouguila (2023) scaled Dirichlet HMMs

3. **Sensor Failure Detection**

  - Implement: Association rule mining approach from Sensors 2020 paper

### Phase 2 (Months 4-6): Medium Impact, Higher Complexity

1. **Advanced HMM Variants**

  - Implement: Asghari & Nazerfard (2019) hierarchical HMMs
  - Add: Moreno-Pino et al. (2022) heterogeneous HMMs

2. **Missing Data Handling**

  - Implement: Sensors 2020 missing data methodology

3. **Multi-Modal Integration**

  - Build framework based on: Cicirelli et al. (2021) comprehensive review

### Phase 3 (Months 7-9): Research-Oriented Applications

1. **Comprehensive CPD Framework**

  - Integrate methods from: PMC5464762 survey
  - Add: Real-time smart home applications

2. **Advanced AAL Features**

  - Implement: OntoDomus semantic modeling
  - Add: Clinical integration capabilities

--------------------------------------------------------------------------------

## Expected Research Impact

This roadmap transforms the repository from a single-method implementation into a comprehensive sensor modeling toolkit. Each paper provides specific algorithmic contributions that address real deployment challenges while advancing the theoretical understanding of temporal behavioral modeling.

The combination of multiple modeling paradigms (autoregressive, HMM, change point detection) with practical considerations (sensor failures, missing data, energy efficiency) creates a robust foundation for both research advancement and practical AAL system deployment.
