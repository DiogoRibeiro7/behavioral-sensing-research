# Project Roadmap

This roadmap describes the planned direction for the Sensor Modeling Research
Toolkit. It is release-oriented: items are grouped by the outcome they should
unlock, not by implementation preference. The roadmap can change as research
needs, user feedback, and maintenance constraints evolve.

## Guiding Principles

The project should remain useful for reproducible behavioral sensing research.
New work should prioritize:

- Stable, documented APIs over one-off scripts.
- Interpretable methods before opaque models, unless the use case clearly
  benefits from additional complexity.
- Explicit validation for data assumptions, missingness, and model calibration.
- Tests and examples that make research workflows reproducible.
- Lightweight deployment paths for local, clinical, and edge environments.

## Current Baseline

The current `0.1.x` series provides the foundation for data loading,
preprocessing, simulation, model fitting, change-point detection, analysis,
visualization, and report generation.

Implemented or substantially available:

- CSV, JSON, HDF5, and streaming-oriented data loading.
- Gap-aware missing-data handling with masks and summaries.
- Synthetic behavioral sensor data generation with reproducible simulation.
- Bernoulli autoregressive models, including multivariate variants.
- Multiple HMM variants for activity-state modeling.
- NHPP-PELT segmentation with B-spline intensities and diagnostics.
- Several change-point detection implementations.
- Analysis pipeline reports in LaTeX, HTML, and minimal FHIR-style JSON.
- Flask-based visualization app with authenticated upload workflow.
- Zenodo, citation, and release metadata.

## Capability Status by Theme

Work is grouped by the outcome it unlocks. Each theme states what exists now
and what is outstanding.

### Existing capability (pre-multimodal core)

Retained unchanged and still supported: CSV/JSON/HDF5 loading, gap-aware
missing-data handling, Bernoulli autoregressive models, HMM variants,
NHPP-PELT segmentation, change-point detectors, Granger and dependency-network
analysis, LaTeX/HTML reporting, the Flask visualisation app, and the original
CLI subcommands.

### Foundation work

Delivered:

- A canonical, hardware-neutral observation model with boundary validation,
  unit conversion, duplicate collapse, out-of-order handling, late-arrival
  flagging, and per-source clock-drift correction.
- A sensor registry that declares each sensor's modality, semantics, room,
  expected cadence, and whether its activations are attributable to a person.
  Inference reads declarations rather than sensor names.
- Online sensor health estimation emitting a per-sensor evidence weight, with
  silence only treated as failure where a sensor promised to report.

### Multimodal sensing

Delivered:

- A configurable continuous-time behavioural state ontology and a recursive
  multimodal Bayes filter over asynchronous, partially missing evidence, with
  explicit abstention.
- Probabilistic occupancy estimation and uncertainty-aware attribution of
  ambient activity, using anonymous evidence only.

### State inference

Delivered: a configurable state ontology whose semantic claims stop where the
evidence stops, with explicit abstention when confidence or sensor coverage is
insufficient.

### Personalisation

Delivered:

- Adaptive, robust, weekday-aware personal baselines distinguishing ordinary
  variability, weekly rhythm, temporary disturbance, persistent change,
  gradual drift, abrupt change, and insufficient data.
- Restrained alerting with deduplication, rate limiting, explicit caveats, and
  a strict separation between system-health and behavioural findings.

### Evaluation

Delivered:

- A synthetic household simulator with controlled ground truth, generatively
  independent of the inference model, plus separate fault injection.
- Problem-appropriate evaluation metrics and a paired sensor-ablation
  framework reporting effect sizes and bootstrap intervals.

### Online and edge processing

Delivered: an incremental, bounded-memory, snapshot-able pipeline with a
lateness buffer for stream reordering, independent of any UI or storage.

### Clinical interoperability

Delivered: a FHIR-style export that keeps measurements, derived features,
inferred states and algorithmic alerts distinguishable through explicit
provenance. Not a validated profile; see `docs/limitations.rst`.

### Reproducible research

Delivered:

- A reproducible end-to-end command-line demonstration and ablation
  experiment.

User documentation: `docs/ambient_architecture.rst`, `docs/inference.rst`,
`docs/evaluation.rst`, `docs/limitations.rst`.

Design record: `docs/MULTIMODAL_ARCHITECTURE.md`, `docs/RESEARCH_QUESTIONS.md`,
`docs/SENSOR_DATA_MODEL.md`, `docs/UNCERTAINTY_MODEL.md`,
`docs/EVALUATION_DESIGN.md`.

### Longer-term research

Outstanding, in priority order

The single most important outstanding item is **validation against real
annotated sensor data**. Every quantitative result currently comes from the
bundled simulator, and until that changes the numbers demonstrate that the
framework behaves sensibly rather than saying anything about real deployments.

In priority order:

1. Evaluate against a public annotated smart-home dataset (CASAS, ARAS,
   MARBLE) using the same metrics.
2. Fit emission and dwell parameters from data rather than declaring them, and
   compare the fitted values against the documented defaults.
3. Model the dependency between the occupancy and state layers, which
   currently share radar and beacon evidence and therefore have correlated
   errors that the reported uncertainty does not reflect.
4. Improve visitor recall, which is conservative by construction and currently
   misses most short visits.
5. Assess calibration across households rather than only within one.
6. Reduce the pre-existing type-checking debt in the older modules and widen
   the CI `mypy` gate beyond its current two files.

## Near-Term Roadmap

### 0.2.0: API Stabilization and Documentation

Goal: make the toolkit easier to adopt by researchers who are not already
familiar with the internals.

Planned work:

- Define public API boundaries for `data`, `utils`, `analysis`, `models`,
  `change_point`, `hmm`, and `visualization` modules.
- Add docstring coverage for public classes and functions.
- Expand quickstart examples into complete workflows: load data, validate data,
  fit model, evaluate model, export report.
- Document supported input data schemas and timestamp handling.
- Add migration notes for renamed or clarified APIs.
- Add a release checklist covering `develop` to `main` merge, tag creation,
  GitHub release, Zenodo metadata, and documentation build.

Quality gates:

- Full test suite passes.
- Pre-commit passes.
- Sphinx documentation builds without warnings for changed pages.
- Public API changes are documented in the changelog.

### 0.3.0: Data Quality and Validation

Goal: make data problems visible before they affect model results.

Planned work:

- Add a formal data validation report object for temporal consistency, missing
  values, duplicate timestamps, irregular sampling, and sensor range checks.
- Improve sensor failure detection with configurable stuck-sensor, dropout, and
  drift heuristics.
- Add dataset summary exports for reproducibility appendices.
- Add tests for malformed CSV, JSON, HDF5, and streaming records.
- Add example notebooks for data cleaning and quality assessment.

Quality gates:

- Data validation outputs are deterministic and serializable.
- Failure modes raise actionable exceptions or warnings.
- Validation utilities have focused unit tests and integration examples.

### 0.4.0: Model Evaluation and Comparison

Goal: provide consistent evaluation across modeling approaches.

Planned work:

- Standardize model scoring interfaces for Bernoulli AR, HMM, NHPP, and CPD
  components.
- Expand time-series cross-validation utilities.
- Add calibration metrics and uncertainty diagnostics for probability models.
- Add model comparison reports with structured machine-readable output.
- Add benchmark datasets or synthetic benchmark recipes with fixed seeds.

Quality gates:

- Comparison utilities work with at least two model families.
- Metrics are documented with expected input and output shapes.
- Benchmarks are reproducible from the command line.

### 0.5.0: Clinical and Interoperability Workflows

Goal: improve reporting and integration paths for clinical and applied research
prototypes.

Planned work:

- Expand the current minimal FHIR-style export toward documented Observation
  and Bundle structures.
- Add configurable clinical threshold profiles.
- Improve patient-friendly summaries and trend visualizations.
- Add anonymization and redaction helpers for exported reports.
- Document limitations clearly: research prototype, not a medical device.

Quality gates:

- Clinical exports have schema-oriented tests.
- Report examples avoid exposing sensitive identifiers.
- Clinical documentation distinguishes supported behavior from future work.

## Longer-Term Roadmap

These items are valuable but should follow the stabilization work above.

### Deep Learning Change-Point Detection

- Add transformer, CNN, or autoencoder-based CPD only after baseline evaluation
  utilities are stable.
- Provide simple training examples and clear dataset requirements.
- Compare deep approaches against existing interpretable baselines.

### Online and Real-Time Processing

- Add incremental preprocessing and validation utilities.
- Add online change-point detection interfaces.
- Support streaming report updates and bounded-memory operation.

### Packaging and Distribution

- Publish stable package artifacts when the public API is ready.
- Add compatibility checks for supported Python versions.
- Document optional dependency groups by workflow.

### Performance and Scalability

- Profile slow model-fitting and analysis paths.
- Add benchmark tracking for core algorithms.
- Evaluate vectorization and optional acceleration where it reduces real runtime
  without making the code harder to maintain.

## Maintenance Backlog

The following work can be handled continuously across releases:

- Replace remaining legacy typing aliases with modern annotations.
- Reduce broad exception handling where errors can be handled specifically.
- Improve coverage for analysis and visualization modules.
- Keep examples synchronized with public APIs.
- Expand changelog entries for every release after `0.1.0`.
- Keep Zenodo, citation, and package metadata aligned before each release.

## Release Policy

Development work happens on `develop`. Releases are cut from `main` only.

Release flow:

1. Finish and validate work on `develop`.
2. Merge `develop` into `main`.
3. Tag the release on the `main` commit.
4. Publish the GitHub release from that tag.
5. Confirm Zenodo metadata and DOI linkage.
6. Update the changelog and documentation as needed.

## Out of Scope for Now

The following are not immediate priorities:

- Medical-device claims or regulated clinical decision support.
- Large-scale cloud platform features.
- Opaque deep-learning models without benchmark comparisons.
- Breaking public APIs without a documented migration path.
