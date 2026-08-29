# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added `sensor_modeling.observations`: a canonical, hardware-neutral observation model with timezone-aware validation, unit conversion with dimension checking, a declarative sensor registry, boundary ingestion with duplicate collapse, out-of-order and late-arrival flagging, and minimum-latency clock-drift correction.
- Added `sensor_modeling.health`: online per-sensor reliability estimation emitting an evidence weight, with silence treated as failure only for sensors that declared a reporting cadence.
- Added `sensor_modeling.states` and `sensor_modeling.fusion`: a configurable continuous-time behavioural state ontology and a recursive multimodal Bayes filter computing `P(Z_t | O_1:t)` over asynchronous, heterogeneous, partially missing evidence, with per-sensor supporting and contradicting evidence and explicit abstention.
- Added `sensor_modeling.context`: probabilistic occupancy estimation over four household contexts and uncertainty-aware attribution of ambient activity, using anonymous evidence only.
- Added `sensor_modeling.baseline`: adaptive, robust, weekday-aware personal baselines over non-stationary behaviour, distinguishing ordinary variability, weekly rhythm, temporary disturbance, persistent change, gradual drift, abrupt change, and insufficient data.
- Added `sensor_modeling.alerts`: restrained alerting with joint magnitude/duration grading, coverage and attribution gates, explicit caveats, deduplication, rate limiting, and a strict separation between system-health and behavioural findings.
- Added `sensor_modeling.simulation`: synthetic households with schedule-driven ground truth generatively independent of the inference model, plus separate injection of dropout, stuck sensors, random loss, wearable non-adherence, late arrival, duplication, and clock drift.
- Added `sensor_modeling.evaluation`: problem-appropriate metrics (balanced accuracy, macro F1, log loss, Brier, calibration error, transition timing, detection delay, false positives per person-day) and a paired sensor-ablation framework reporting bootstrap intervals and effect sizes.
- Added `sensor_modeling.online`: an incremental, bounded-memory, snapshot-able pipeline orchestrating the full chain with a lateness buffer for stream reordering.
- Added `sensor-modeling demo` and `sensor-modeling ablate` commands, both reproducible from a fixed seed.
- Added `sensor_modeling.interop`: a FHIR-style export that keeps measurements, derived features, inferred states and algorithmic alerts distinguishable, with explicit provenance on every resource, inferences marked `preliminary` with their full posterior and method, abstentions exported as `dataAbsentReason`, and alerts exported as `DetectedIssue` rather than `Observation`.
- Added `docs/ambient_architecture.rst`, `docs/inference.rst`, `docs/evaluation.rst`, and `docs/limitations.rst`.
- Added 327 tests covering the new packages, including DST transitions in both directions, out-of-order and duplicate delivery, sensor dropout, wearable non-adherence, visitor contamination, snapshot/restore, and end-to-end recovery against ground truth.
- Added a top-level `ROADMAP.md` with release milestones, quality gates, longer-term priorities, maintenance backlog, and release policy.
- Added `RELEASE.md` with the main-only release checklist, tag verification steps, and Zenodo release verification.
- Added GitHub issue templates, a pull request template, and CI coverage for pushes to `develop`.
- Added CI jobs for Sphinx documentation builds and package artifact validation.
- Added security and support policy documents.
- Added focused tests for shared data IO, synthetic exports, HDF5 loading, sensor failure detection, plotting helpers, and model validation utilities.
- Added release metadata consistency tests for package, citation, Zenodo, and README metadata.
- Added a non-interactive Matplotlib backend guard for the pytest suite.

### Changed
- Refactored data-layer typing, HDF5 import handling, synthetic data export behavior, and validation internals.
- Refactored behavioral analysis helpers to validate sensor frames, ignore non-numeric columns, and avoid NaN metrics for constant or single-day data.
- Refactored Granger causality summaries to validate lag configuration, keep empty result schemas stable, and ignore non-finite summary statistics.
- Refactored Granger causality testing to validate binary inputs and align lagged design matrices correctly.
- Refactored dependency network analysis to validate binary sensor frames, handle edgeless graphs, and return plot figures.
- Refactored cross-validation helpers to validate split counts and narrow fold failure handling.
- Refactored analysis result exports to create parent directories, return written paths, and propagate write failures.
- Refactored JSON and streaming data loaders to validate timestamps explicitly and avoid broad malformed-record handling.
- Refactored data validation helpers to reject non-timestamp indexes, duplicate timestamps, invalid range bounds, and invalid failure windows.
- Refactored preprocessing outlier detection and mean imputation to handle mixed dtypes and constant numeric columns.
- Refactored model comparison statistics to validate paired tests and scale finite metrics without NaN warnings.
- Refactored clinical visualization helpers to validate required columns and rolling-window inputs.
- Refactored research visualization helpers to validate plotting schemas and residual diagnostics.
- Refactored interactive visualization helpers to validate plotting schemas, finite parameter sweeps, and export paths.
- Refactored the visualization web app upload endpoint to return explicit client errors for malformed uploads.
- Refactored lightweight change-point detectors to share validation for configuration, input series, and thresholds.
- Refactored the PELT change-point detector to validate configuration and custom cost outputs explicitly.
- Refactored analysis report writers to create output directories, return written paths, and narrow template formatting errors.
- Refactored analysis pipeline model dispatch to validate input frames, format NumPy probability outputs, and narrow model failure handling.
- Refactored Bernoulli autoregressive fitting to validate training frames and narrow optimizer failure handling.
- Refactored multivariate Bernoulli AR comparison and plotting helpers to use explicit column selection and return figures.
- Refactored synthetic data generation to validate configs, bound generated probabilities, and report export paths.
- Refactored sensor simulation to use local NumPy random generators instead of mutating global random state.
- Refactored plotting helpers to return Matplotlib figures and support non-interactive `show=False` workflows.
- Simplified legacy `setup.py` so package metadata is sourced from `pyproject.toml`.
- Updated package metadata and source distribution exclusions for cleaner release artifacts.
- Updated README and Sphinx roadmap documentation to point to the canonical roadmap.

### Fixed
- Fixed default emission rates conflating a state's location with its activity level, which made a bedroom motion sensor's silence argue against `sleeping` as hard as the bed sensor argued for it. End-to-end state accuracy rose from 0.585 to 0.918 and sleep recall from 0.00 to 0.96.
- Fixed gradual drift being permanently unalertable: it has no deviation streak by construction, so grading it on magnitude and duration scored every slow decline at zero.
- Fixed a drift's reported direction being read from the day rather than the trend, which allowed a verdict to announce a decrease while reporting a rising slope.
- Fixed the default trend threshold firing on noise; across a four-week window the Theil-Sen slope of a stable series already accumulates more than one robust standard deviation of apparent movement.
- Fixed synthetic JSON export by serializing timestamp values before writing JSON.
- Fixed model validation calibration output to return plain Python booleans.

## [0.1.3] - 2026-07-13
### Added
- Added minimal FHIR-style report export to the analysis pipeline.
- Added configurable web upload directory support for the Flask application.
- Added time-series cross-validation helpers for model comparison workflows.

### Changed
- Aligned CSV loading semantics across `SensorDataset.from_csv` and data loaders.
- Updated comparison scoring to require explicit scoring behavior when models do not expose `score`.
- Restored and aligned the lint/pre-commit baseline for the current codebase.
- Removed duplicated README sections.

### Fixed
- Fixed analysis report generation so output directories are created before writing reports.
- Fixed README examples to use Bernoulli probability prediction APIs.

## [0.1.2] - 2026-07-13
### Added
- Added Zenodo release metadata for archive publication.
- Added `.zenodo.json` metadata for Zenodo integration.

## [0.1.1] - 2026-07-13
### Added
- Added repository-to-Zenodo linkage documentation.
- Prepared release metadata for Zenodo DOI archival.

## [0.1.0] - 2025-08-28
### Added
- Initial release with unified sensor modeling framework, analysis utilities, and visualization tools.
