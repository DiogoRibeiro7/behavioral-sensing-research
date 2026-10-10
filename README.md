<p align="center">
  <img src="assets/project-logo.png" alt="behavioral-sensing-research project logo" width="160" height="160">
</p>

# Sensor Modeling Research Toolkit

A research-grade Python toolkit for **interpretable, probabilistic,
privacy-preserving** analysis of behavioural sensor data, and for multimodal
ambient sensing in ambient assisted living (AAL), digital health and smart-home
research.

It provides an end-to-end pipeline from heterogeneous sensor observations to
explained alerts, alongside an established modelling core of Bernoulli
autoregressive models, hidden Markov models, change-point detection and
non-homogeneous Poisson processes.

> **This is a research toolkit, not a medical device.** Nothing it produces is
> a diagnosis, and no claim of clinical effectiveness is made or supported.
> Simulator results are not estimates of field performance; the pipeline has
> now been evaluated on real CASAS recordings, externally tested on a frozen
> 43-home CASAS cohort, and tested on two independently collected homes, to
> which it does not transfer. The evidence boundaries are described below.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11--3.14-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/DiogoRibeiro7/behavioral-sensing-research/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/behavioral-sensing-research/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/sensor-modeling/badge/?version=latest)](https://sensor-modeling.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21337272.svg)](https://doi.org/10.5281/zenodo.21337272)
[![Version](https://img.shields.io/badge/version-0.9.0-informational.svg)](CHANGELOG.md)

## 🎯 Overview

The **Sensor Modeling Research Toolkit** addresses the growing need for reproducible, interpretable analysis of behavioral sensor streams in smart environments. Unlike general-purpose machine learning libraries, this toolkit provides domain-specific implementations optimized for the unique characteristics of ambient sensor data: irregular sampling, frequent missingness, binary activations, and the need for transparent, clinically interpretable models.

### What the numbers mean

Most quantitative results in this repository come from the bundled simulator.
They have since been checked against real recordings, and the comparison
matters when reading them:

| | Balanced accuracy |
| --- | --- |
| Simulator | 0.816 |
| 22 real CASAS homes | **0.420** |
| Recoverable from those sensors by any method | 0.607 |

The simulator's figure sits **above** what that instrumentation supports even
for a supervised classifier with access to the labels. Simulator results are
therefore not an estimate of real-world performance, and should not be read as
one. The gap between the pipeline and that ceiling is not only missing
information. In a matched comparison on 20 single-resident development homes, a
supervised classifier given strictly less information than the online filter,
the current window and three previous ones, still scored 0.081 higher balanced
accuracy, in 19 of 20 homes; part of the gap is the formulation. See
[the recoverable-information gap](docs/PHASE1_RECOVERABLE_GAP.md). Pre-specified
changes to the generative model have since been measured on the same homes; see
[`ROADMAP.md`](ROADMAP.md). None of them is a held-out claim.

The two that succeeded, a hurdle model fitted for each evidence channel and a
hierarchical time-of-day prior, were then run together in the filter's
recursion, under a protocol frozen before any household was scored. Together
they score 0.517 balanced accuracy on those 20 homes, against 0.456 for the
fitted channels alone and 0.417 for the declared model. The prior adds +0.061
[+0.044, +0.077], most in the recall of `away` and `sleeping`, and lowers
calibration error from 0.284 to 0.213. The earlier versions of the same two
parts, in the production filter on 11 held-out homes of the panel, scored 0.434
together against 0.460 for the circadian term alone. The production pipeline
does not yet offer the combination, and a held-out confirmation is next. See
[the combined-prior results](docs/PHASE3_COMBINED_PRIOR_RESULTS.md).

The frozen v0.3 candidate was also tested once on 43 single-resident CASAS homes
outside the development panel. The optional circadian prior improved the median
paired household balanced accuracy by **+0.0091**, with a 95% household bootstrap
interval of **[+0.0054, +0.0117]**; 37 of 43 homes improved. The effect is small
and does not establish clinical effectiveness or general smart-home performance.

Under a protocol frozen before any external scoring, the CASAS-trained model was
then tested on the two homes of the independently collected UCI ADL Binary
dataset, which has a different sensing layout. **It does not transfer**: zero-shot
balanced accuracy is 0.246 and 0.244, against chance at 0.25, and a week of
limited adaptation leaves balanced accuracy unchanged. See
[Phase 5 external results](docs/PHASE5_EXTERNAL_RESULTS.md).

The online pipeline was also run, with its declared emissions, baseline and
alert policy, the ten-minute step its protocol declares and nothing fitted, on
the 56 homes of the TIHM dataset of people living with dementia. Its
labels are alerts a clinical team verified, not behavioural states, so no state
is scored; the run is exploratory and describes what the pipeline raises. It
raises 0.064 behavioural alerts per person-day, with a 95% household bootstrap
interval of [0.046, 0.084], and **the alerts are not more frequent on the days
the team confirmed**: alert days are 5.3% of agitation label days and 8.5% of
other days. Looked at afterwards, a quarter of the alert days are days on which
no sensor in the home reported anything, which the pipeline reads as sleep. See
[TIHM alert-burden results](docs/TIHM_ALERT_BURDEN_RESULTS.md).

An opt-in rule, off by default, now treats a home in which no sensor has
reported for a declared time as not observed, and says so in an alert of its
own. It was built from the TIHM finding, so it was tested elsewhere: on 100
paired simulated homes reduced to their event sensors, under a protocol frozen
before any of them had been run with the rule on. **The evidence is
simulated.**

- **An outage raises alerts, and the rule removes them.** A 60-hour outage
  raises 2.76 behavioural alerts per home more than the same home without it,
  with a 95% interval over homes of [2.36, 3.15] and a Monte Carlo standard
  error of 0.20. With the rule on at 12 hours the excess is −0.28
  [−0.44, −0.13], standard error 0.08. That is below zero: the outage arm
  raises fewer alerts than the same homes left alone, 30 against 58. Whether
  that is a loss of sensitivity after an outage is not settled.
- **Every outage is reported**, a median 12.0 hours after it began.
- **Nothing changes where no home is silent.** No alert of a stable home
  changes, and the rule raises none of its own there.
- **After an outage, fewer homes meet the detection definition with the rule
  on.** 76 of 100 against 91: −0.15 [−0.23, −0.07], standard error 0.04,
  against a margin of −0.10. The interval excludes zero and lies on both sides
  of the margin, so the pre-specified verdict is inconclusive. With the rule
  off, 38 of 100 homes with an outage and no change meet the same definition,
  and with no outage at all 75 are detected, so whether changes are missed or
  alerts the outage raised are gone was not tested.
- **The eight criteria.** The problem was reproduced, four criteria came out
  as success, detection with no outage was non-inferior, detection after an
  outage was inconclusive, and a stated limit of the rule was not confirmed in
  one home in 100.

Run on the TIHM homes as a description and not as a test, the rule shows a
cost the simulator could not: those homes are silent for twelve hours often.
At 12 hours 147 of the 183 behavioural alerts are no longer raised and 13 new
ones are, 322 of the 2,850 monitored days are refused, and 324 alerts about
silence are raised, 145 of them in the two stretches in which most monitored
homes were silent together. See
[the silent-home results](docs/SILENT_HOME_RESULTS.md).

The personal baseline's deviation threshold of three standard deviations is
passed far more often than it states when a weekday reference rests on a few
days. On synthetic Gaussian days the default reference passes it on 6.2% of
days, 23 times the 0.27% it states. An opt-in reference, off by default,
pools the scale over every retained day and maps the deviation through a
Student t, and passes it on 0.28%; see
[the baseline's thresholds on synthetic days](docs/THRESHOLD_CALIBRATION_NULL.md).
It was built from the TIHM finding, so it was tested on 400 paired simulated
homes, under a protocol frozen before any of them had been run with it. **The
evidence is simulated.**

- **At the same false alerts it finds more of a step change.** Where its
  operating curve has the default's 0.855 false alerts a home in 84 days, it
  finds the step net of what it raises with nothing injected in 0.72 of homes,
  against 0.63: +0.09, with a 95% interval over homes of [+0.03, +0.16] and a
  Monte Carlo standard error of 0.03. The match is placed again in every
  resample.
- **Its threshold means what it says for hours of sleep on these homes.**
  0.35% of stable days of sleep pass 3, the share a Gaussian value passes
  2.92 [2.85, 2.99]; the default passes 3 on 7.9% of them. On hours away,
  where no criterion was stated, it passes 3 on 1.66% of days, as often as a
  Gaussian value passes 2.40.
- **For a smaller step and a gradual change no difference is shown**: −0.02
  [−0.09, +0.04] and −0.01 [−0.10, +0.09].
- **It is not a drop-in replacement.** At the declared thresholds it raises 3
  false alerts in the 400 homes against 342, and finds the step net of chance
  in 0.04 of them against 0.63.

On TIHM, as a description: with the silent-home rule off it passes 3 on 8.1%
of evaluable days of sleep, against the default's 17.4%. Real days are far
from what its score assumes. See
[the threshold-calibration results](docs/THRESHOLD_CALIBRATION_RESULTS.md).

Every alerting study here counts its detections on `sleeping_hours`, the
hours of a day the pipeline gives to sleep. In 17 TIHM homes a mat under the
mattress recorded each minute a person was in bed, so the two were set side
by side, under a protocol frozen before any value of the pipeline was set
beside any record of the mat. On the days the sensors reported throughout, in
the 14 homes with enough of them:

- **Within a home, the pipeline's hours of sleep barely follow the mat's.** A
  mean within-home Spearman correlation of 0.11 [0.03, 0.19], against a
  declared margin of 0.5. On 100 simulated homes, against the simulator's true
  hours of sleep, the same correlation is 0.70 [0.68, 0.71].
- **It counts more sleep than the mat does, and about two hours more than the
  time in bed.** 3.95 hours a day more than the mat's sleep [+1.54, +6.37],
  and 1.97 more than its hours in bed [−0.01, +3.94], an interval that reaches
  zero.
- **A deviation in one is seldom a deviation in the other.** The deviations a
  personal baseline gives the two correlate at 0.03 [−0.04, 0.11], over 12
  homes; on 4 of the 46 days the pipeline's reached 3, the mat's did too, in
  the same direction.

The mat's stages are the device's own and are not validated here. See
[the sleep-mat results](docs/SLEEP_MAT_RESULTS.md).

The simulator's event sensors are cleaner than TIHM's: no hold-off between
activations, no hallway sensor, a contact logged once, and 7% of motion
activations after another room's against 63%. The 400 threshold-calibration
homes were run again with their sensors drawn from the same days under a
profile matched to TIHM's sensor records: a 61-second hold-off, a hallway
sensor, contacts logged twice, and 2 activations an hour from each motion
sensor of a room the resident is not in, while the resident is at home and
awake. The protocol was frozen before any of the study's homes was run with
it. **The evidence is simulated.**

- **The detection of the step change survives.** Net of false detections it
  finds the step in 0.70 of homes against 0.63: +0.07 [+0.01, +0.13], against
  a margin of −0.10. False alerts rise from 0.86 to 1.02 a home in 84 days.
- **The hours of sleep still follow the truth.** A mean within-home
  correlation of 0.71 [0.70, 0.71], against a margin of 0.5, and 0.67 on the
  same homes with the simulator's own sensors. In the simulator, the sensor
  properties the profile matches do not by themselves bring the correlation
  anywhere near the 0.11 against the sleep mat.
- **The bathroom collapses; the kitchen falls to about a third.** Pooled-day
  medians of 0.01 hours of bathroom activity against a true 0.28, and 0.68 of
  kitchen activity against a true 1.96; TIHM's are 0.00 and 0.06. The
  pre-specified criterion, which needed both under their ceilings of 0.05 and
  0.25 hours, reads not reproduced: the kitchen's is over its ceiling. Their
  correlations with the truth fall from 0.72 to 0.51 and from 0.70 to 0.40.

See [the matched-sensor results](docs/MATCHED_SENSORS_RESULTS.md).

Where, then, do the pipeline's sleep and the mat's part? In the same 14 TIHM
homes the pipeline's belief was set beside the mat minute by minute, under a
plan frozen before any of it was computed. It is a description, not a test:
the comparison it breaks down had been read. With the mat's clock as
recorded:

- **The extra sleep falls when the mat has no record, mostly by day.** Of the
  3.95 hours a day more sleep than the mat's, 4.10 are counted when the mat
  has no record, 3.20 of them between 07:00 and 22:00; 1.34 are counted while
  the mat says awake in bed; and 1.48 hours of the mat's sleep are not
  counted. The median home's excess is 2.38 hours; one home's is 17.20.
  Without the three homes the mat stages mostly awake, 3.36 hours are counted
  with no record and the excess is 2.55.
- **The longer the home is quiet, the surer the pipeline is of sleep.** With
  no mat record, its belief in sleep is 0.39 between ten minutes and an hour
  after the last activation, 0.90 between one and three hours, and 0.99
  after three; 2.29 of the 4.10 hours fall within the hour, 1.80 after it.
  By the sensor that last reported, 1.40 hours follow the bedroom's and 1.07
  an exit door's, 0.79 of them an hour or more after it. A door's record does
  not say which way anyone went.
- **That part moves the daily number.** It carries 0.68 [0.54, 0.83] of the
  day-to-day variance of the pipeline's hours of sleep, and it follows the
  mat no better than any other day's beliefs set beside the same mat would.
  The sleep counted while the mat has a record does follow the mat's sleep
  beyond that, by +0.10 [+0.05, +0.15], above zero in all 14 homes. The daily
  hours of sleep correlate with the day's activations at −0.71, which the way
  the pipeline infers sleep partly builds in.
- **Part of it sits at the edges of the mat's nights.** Counting each mat
  record an hour later, the sleep counted off the mat within 30 minutes of a
  record falls from 0.73 to 0.32 hours a day, and the on-mat part follows the
  mat by +0.18 [+0.09, +0.26] beyond the reference, above zero in 12 of 14
  homes. That is consistent with a mat clock an hour behind, or with the
  pipeline's timing at bedtime and waking.

The mat's stages are the device's own, and a minute with no record may be an
empty bed, a person asleep elsewhere or a mat that stopped. See
[the sleep-gap results](docs/SLEEP_GAP_RESULTS.md).

The TIHM dataset is by Palermo et al., *Scientific Data* 10, 606 (2023),
under CC BY 4.0. Surrey and Borders Partnership NHS Foundation Trust and Howz
are acknowledged, as the dataset asks. It is not redistributed here.

See [Real-data validation](docs/real_data.md) and
[Known limitations](docs/limitations.md).

### Key Differentiators

- **Research-Grade Implementation**: Clean, documented, and tested implementations of established algorithms from recent literature
- **Unified Interface**: Consistent API across different modeling approaches for easy comparison and ensemble methods
- **Clinical Focus**: Visualization and reporting utilities designed for healthcare stakeholders and non-technical users
- **Lightweight Deployment**: Minimal dependencies and efficient implementations suitable for edge computing and real-time applications
- **Extensible Architecture**: Modular design allows researchers to easily add new algorithms and extend existing functionality

## 🧭 Observation, state, change, alert

The platform keeps five kinds of thing strictly distinct, and most of its
design follows from refusing to collapse them:

| Kind | What it is | Example |
| --- | --- | --- |
| **Measured observation** | A sensor reported a value at an instant | The fridge contact closed at 08:14 |
| **Derived feature** | A value an upstream device computed, carrying its own confidence | The radar reports 2 tracked people |
| **Inferred state** | A posterior over what the resident was probably doing | `P(kitchen_activity) = 0.81` |
| **Behavioural change** | A shift against the resident's own history | Sleep has trended down for three weeks |
| **Alert** | A judgement that a person should look at something | An `attention` alert, with its caveats |

A sensor event is not a behaviour:

```text
fridge opening      != eating
tap activation      != drinking
toilet event        != confirmed toileting
chair activity      != sedentary behaviour
door event          != resident movement
missing observation != inactivity
```

The state ontology therefore stops at `kitchen_activity` and makes no claim
about food intake. Two rules are enforced mechanically rather than by
convention:

- **A missing observation is missing evidence, never negative evidence.**
  Sensor reliability enters the fusion likelihood as a tempering weight, so a
  failed sensor contributes a flat likelihood and cannot look like a quiet
  resident.
- **Ambient activity is not automatically the resident's.** Occupancy
  estimation produces `P(activity was the resident's)`, which discounts
  evidence while a visitor or carer may be present.

The system can also return `unknown`. Abstention is a first-class output, not
a failure. Current real-data diagnostics show that the implemented confidence
threshold is not yet a reliable safety mechanism: confidence weakly separates
correct from incorrect predictions and becomes less reliable in the highest
confidence band. Version 0.9.0 adds structural uncertainty diagnostics and a
selective-prediction evaluation. On the development homes, disagreement between
fitted model specifications ranks errors better than confidence, but no
abstention rule has been selected, and abstention is not solved.

### Supported and unsupported claims

The distinction the platform is built to hold. The left column is what the
evidence supports; the right is what it does **not**, however tempting the
inference.

| Supported | Not supported |
| --- | --- |
| Evidence of kitchen activity | Food consumption |
| Evidence of bathroom activity | Confirmed toileting |
| Bed occupancy with sustained low movement | Clinically defined sleep, or a sleep disorder |
| A door was crossed | The resident left the house |
| A sustained change against the resident's own history | A cause, a prognosis, or a diagnosis |
| Reduced room-to-room transitions | Deterioration in mobility as a clinical finding |
| Sensor coverage has fallen | The resident has become less active |
| `P(resident generated this activity) = 0.5` | Identification of who did it |

Two of these deserve spelling out.

**`kitchen_activity` is not eating.** A fridge contact records a door opening.
Turning that into a meal requires evidence the sensor cannot supply, so the
ontology stops where the evidence stops.

**`sleeping` is not sleep.** It is bed occupancy accompanied by sustained low
movement. It has no relationship to polysomnography, and mapping it to a
clinical sleep concept is a further inferential step this platform does not
take.

## 🏠 Multimodal ambient sensing pipeline

```text
heterogeneous observations -> validation -> sensor health -> occupancy context
    -> multimodal fusion -> behavioural state -> adaptive baseline
    -> change detection -> restrained alerts -> evaluation
```

| Package | Responsibility |
| --- | --- |
| `sensor_modeling.observations` | Canonical hardware-neutral observation model, sensor registry, boundary validation, clock-drift correction |
| `sensor_modeling.health` | Online per-sensor reliability, emitted as an evidence weight |
| `sensor_modeling.context` | Occupancy contexts and uncertainty-aware attribution, from anonymous evidence only |
| `sensor_modeling.states` / `sensor_modeling.fusion` | Continuous-time state ontology and the recursive multimodal filter |
| `sensor_modeling.baseline` | Adaptive, weekday-aware, non-stationary personal baselines |
| `sensor_modeling.alerts` | Restrained, explained alerting with deduplication and rate limiting |
| `sensor_modeling.simulation` | Synthetic households with controlled ground truth |
| `sensor_modeling.evaluation` | Problem-appropriate metrics and paired sensor-ablation studies |
| `sensor_modeling.online` | Incremental, snapshot-able orchestration |

### Reproducible end-to-end example

```bash
sensor-modeling demo --days 90 --seed 20240304 --step-minutes 10
```

Simulates a household with a carer and visitors, injects a three-day bed-sensor
dropout and five days of wearable non-adherence, loses, duplicates, delays and
clock-skews the record, introduces a genuine change in sleep on a known day,
then runs the whole pipeline and reports what it did and did not recover —
including its own false-alert burden. Two runs produce identical numbers.

### Sensor-ablation experiment

```bash
sensor-modeling ablate --days 14 --seeds 11 22 33 44
```

The CLI supports reproducible paired ablations, but reported conclusions should
use study-scale replications rather than the four-seed smoke-test example above.
In the 100-seed study, the eight-sensor configuration was 0.0073 balanced
accuracy below the full deployment (95% CI [+0.0063, +0.0083]), while the
five-sensor configuration was 0.171 lower. In the later frozen confirmatory
study, the pre-specified five-sensor deployment failed the 0.02 non-inferiority
margin with a gap of 0.1548 (95% CI [0.1531, 0.1564]), whereas an eight-sensor
configuration remained within 0.00529 of the full ten-sensor system.

> These numbers describe behaviour on the bundled simulator under its default
> parameters. They are not estimates of field performance. Real-data and
> external-validation results are reported separately in
> [`docs/real_data.md`](docs/real_data.md) and the v0.3 release notes.

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

Python 3.11, 3.12, 3.13 and 3.14 are supported.

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
**Change Point Detection**          | ✅ Complete | 4 algorithms
**NHPP-PELT**                       | ✅ Complete | B-spline intensities, diagnostics
**Causal Network Analysis**         | ✅ Complete | Granger tests, network metrics
**Missing Data Handling**           | ✅ Complete | Gap-aware workflows plus reliability-tempered fusion
**Multimodal Fusion**               | ✅ Complete | Continuous-time filter over asynchronous modalities
**Sensor Health Modelling**         | ✅ Complete | Online reliability feeding the inference layer
**Occupancy & Attribution**         | ✅ Complete | Probabilistic visitor/resident attribution
**Adaptive Baselines**              | ✅ Complete | Robust, weekday-aware, non-stationary
**Sensor Ablation Studies**         | ✅ Complete | Paired designs with effect sizes
**Real-time Processing**            | ✅ Complete | Incremental pipeline, bounded memory, snapshot/restore
**Clinical Integration**            | 🟡 Partial | Minimal FHIR-style export; no validated clinical profile

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
  mkdocs build --strict
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
- **Institution**: Faculty of Media Arts and Design, Technical University of Porto
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
@software{ribeiro2026sensor,
  title={Sensor Modeling Research Toolkit},
  author={Ribeiro, Diogo},
  year={2026},
  url={https://github.com/DiogoRibeiro7/behavioral-sensing-research},
  version={0.9.0},
  doi={10.5281/zenodo.21337272}
}
```

For the underlying methodology, please also cite relevant papers listed in [`CITATION.cff`](CITATION.cff).