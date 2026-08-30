# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed `Leave_Home` being mapped to `AWAY` in the CASAS `hh` reader. That label annotates the act of crossing the threshold, with a median duration of about twelve seconds, so the truth series marked a burst of motion inside the house as absence and the inference was scored wrong for correctly reporting activity, while the hours actually spent out carried no label and were never scored. Intervals labelled `AWAY` showed 130 activations per hour, which is what exposed it. Away is now derived from the gap between a departure and the next arrival. Across 22 homes this raises `away` recall from 0.00 to 0.36, labelled coverage from 64% to 90%, and median balanced accuracy from 0.364 to 0.420. The previously published `away` figure was an adapter artefact, not a property of the pipeline, and `docs/real_data.md` carries the correction.

### Added
- Added `measure_event_rates` and `pooled_rate_report`, measuring the activations per hour a real annotated recording produces for each state so the declared emission rates can be checked against it. Pooled over six homes, real in-room sensors fire at 299/h during bathroom activity and 580/h during cooking against a declared 40/h, and at 5.2/h during sleep against a declared 0.8/h.
- Added `read_casas_hh` for the CASAS `hh` CSV export, which is the form currently distributed on Zenodo and which the classic reader cannot parse: fields are comma-separated, the third field is a location rather than a sensor identifier, markers are quoted, and the activity vocabulary shares only six labels with the classic one, none of them frequent. The fifth column mixes two annotation styles, interval markers and bare per-event labels, and treating the latter as unparsable discarded hundreds of genuinely annotated events.
- Added the first evaluation against real recordings. Across 22 CASAS `hh` homes, with nothing refitted and every location and activity label mapped so no evidence is discarded, median balanced accuracy was 0.364 against 0.816 on the simulator and median calibration error 0.285 against 0.084, with no home exceeding 0.468. States with a distinctive room-and-rate signature held up, while those needing evidence that the resident is present but still collapsed: `away` recall was 0.00 in the median home, and abstention reached at most 5.2% while the model was wrong more often than right. Three alternative explanations were tested and rejected: an incomplete location map (median moved only 0.356 to 0.364), absent presence-confirming sensors (the five homes with a chair occupancy sensor are no better), and modelling motion as occupancy state rather than events (which reaches 1.00 `away` recall by reporting `away` almost always, collapsing balanced accuracy to 0.16-0.23). The failure is established; its mechanism is not. Recorded in `docs/real_data.md` and `docs/limitations.md`.
- Added `sensor_modeling.datasets`, bringing published annotated recordings into the canonical observation model so the pipeline can be run over data this project did not generate. The CASAS adapter refuses to guess a timezone, refuses to label unannotated time, and refuses to force unrecognised sensors or activity labels into the ontology, reporting each as discarded rather than admitting it as evidence. `evaluate_recording` scores the unmodified pipeline and reports coverage beside the metrics, and refuses to score a recording in which nothing carried a mapped label.
- Added `docs/real_data.md` describing what the adapter will not do and why, and stating plainly that the shipped integration test uses a CASAS-format fixture written by this repository rather than a real recording.

### Changed
- Moved every citation target to concept DOI `10.5281/zenodo.21337272`. Zenodo minted a new concept lineage when `0.2.0` was archived through the GitHub integration rather than adding a version to the existing one, so `10.5281/zenodo.17070041` now resolves to `0.1.0` alone and no longer stands for all versions. The `0.2.0` record links back with `isNewVersionOf`, and `ZENODO.md` documents both lineages.

## [0.2.0] - 2026-08-30

Extends the toolkit into a multimodal ambient-sensing research platform. The
work is additive: no public symbol was removed or changed, and a user of
`0.1.3` can upgrade and ignore the new packages entirely.

Every quantitative result quoted below comes from the bundled simulator and
has not been validated against real sensor data. See `docs/limitations.md`.

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
- Added legacy adapters converting wide frames, datasets and long-format records into canonical observations, plus property-based tests for observation and stream invariants.
- Added structured `Explanation` output alongside the one-line form, and calibration tests for state inference.
- Added an attribution comparison measuring naive against occupancy-aware activity attribution across nine occupancy situations, with a `sensor-modeling attribution` command.
- Added a change-detection study measuring delay against alert burden, and a ramped behaviour shift so gradual decline can be injected.
- Added experiment provenance recording configuration, seeds, library versions and written metric definitions with every result.
- Added online pipeline benchmarks for throughput, latency, snapshot size and bounded retained state.
- Added pseudonymisation and export redaction with salt-keyed, study-scoped identifiers.
- Added `sensor_modeling.observations.SensorSpec.redundancy_group` so correlated sensors are not counted as independent evidence.
- Added detection of sensors that keep reporting while delivering below their declared cadence.
- Added `sensor-modeling attribution --seeds`, replicating the attribution comparison across independent paired households and reporting bootstrap intervals for balanced-accuracy gain, calibration gain and visitor detection. The replicated study refuses fewer than two seeds, and refuses repeated ones, so a demonstration cannot be presented as an estimate.
- Added a minimum spacing check on replicated attribution seeds. Scenarios derive degradation seeds from neighbouring values, so consecutive study seeds would have shared simulated faults between replications that were reported as independent.
- Added Monte Carlo standard error to `PairedDifference`, so a narrow interval from few replications can be told apart from a narrow interval from many.
- Ran the attribution comparison at study scale, 100 paired seeds per scenario against the previous one. The result contradicts the single-seed demonstration: a carer round gives +0.0065 balanced accuracy rather than the reported +0.032, the claim that attribution is a no-op in an uncontaminated home is false (both empty-home scenarios show a small but clear loss), and the largest effect attribution has anywhere is a harm of -0.0118 when the resident is not wearing the wearable, where ambient activity that was genuinely theirs is discounted as possibly a visitor's. Calibration improves in all nine scenarios, including those where accuracy falls.
- Ran the sensor ablation at study scale, 100 paired seeds against the previous four. The eight-sensor gap against the full deployment is 0.0073 balanced accuracy (95% CI [+0.0063, +0.0083], MCSE 0.0005), superseding the four-seed pilot's 0.012 (CI [+0.004, +0.020]), whose interval does not contain the study estimate. Calibration was found not to follow accuracy: the five-sensor configuration is the best calibrated of any tested, at 0.0332 expected calibration error against 0.0839 for ten sensors, while scoring 0.171 lower in balanced accuracy.
- Added `docs/SIMULATION_PROTOCOLS.md`, specifying replication counts derived from Monte Carlo standard error and separating the shipped smoke-test seeds from the counts a reported result needs.
- Added guard tests importing every module and rejecting shared container defaults on dataclass fields, so a version-specific failure of this kind fails on any interpreter.
- Added `git_commit`, `git_dirty`, `schema_version` and a snapshot of resolved algorithm defaults to experiment provenance, so a record identifies the exact code and model specification behind it rather than only a package version.
- Added a `gate` job aggregating lint, test, docs and package into a single required check.
- Changed the lint job to check every file on pushes to a shared branch, keeping the fast changed-files scope for pull requests. The narrow scope could not see breakage that arrived in an earlier merge, so a red lint merged anyway stayed red in the tree while every later run reported success after checking unrelated files.
- Added `docs/MULTIMODAL_ARCHITECTURE.md`, `docs/RESEARCH_QUESTIONS.md`, `docs/SENSOR_DATA_MODEL.md`, `docs/UNCERTAINTY_MODEL.md`, `docs/EVALUATION_DESIGN.md`, `docs/ADVERSARIAL_REVIEW.md`, `docs/RELEASE_READINESS.md` and `docs/multimodal_ingestion.md`.
- Added backwards-compatibility tests exercising the original models, data layer and public surface.
- Added `docs/ambient_architecture.md`, `docs/inference.md`, `docs/evaluation.md`, and `docs/limitations.md`.
- Moved documentation from Sphinx to MkDocs with the Material theme, mkdocstrings API reference covering every package, and a `--strict` build in CI.
- Added 327 tests covering the new packages, including DST transitions in both directions, out-of-order and duplicate delivery, sensor dropout, wearable non-adherence, visitor contamination, snapshot/restore, and end-to-end recovery against ground truth.
- Added a top-level `ROADMAP.md` with release milestones, quality gates, longer-term priorities, maintenance backlog, and release policy.
- Added `RELEASE.md` with the main-only release checklist, tag verification steps, and Zenodo release verification.
- Added GitHub issue templates, a pull request template, and CI coverage for pushes to `develop`.
- Added CI jobs for documentation builds and package artifact validation.
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
- Updated README and roadmap documentation to point to the canonical roadmap.

### Fixed
- Fixed `Observation` using a `mappingproxy` as a dataclass field default, which Python 3.11 rejects at class-definition time. Python 3.10 accepts it under its older `isinstance` check and 3.12 onwards accepts it because `mappingproxy` became hashable, so the failure was confined to 3.11, where 21 test files could not be collected and every subsystem depending on `Observation` failed to import.
- Fixed `DetectionStudy.summary()` reporting the mean of each seed's median delay while documenting the value as a median of delays. Delays are now pooled across seeds before the median is taken, `mean_seed_median_delay_days` reports the per-seed view under its own name, and `detected_changes` records the sample size behind both.
- Fixed `research_identifier()` scoping a study by its visible prefix only. The HMAC ignored the study, so two studies sharing a salt produced identical digests for the same subject and their records stayed trivially linkable. The study now derives a subkey.
- Fixed `pytest.ini` overriding the stricter configuration in `pyproject.toml`, which silently disabled `--strict-markers`, `--strict-config` and the warning policy the project appeared to enforce.
- Fixed the analysis pipeline defaulting to `NHPPConfig(n_basis=3)`, which violates the model's own `n_basis >= degree+1` constraint for the cubic default. Every NHPP fit raised and the pipeline recorded an error in place of a result, so that arm had never worked while appearing present in the output.
- Fixed redundant sensors being counted as independent evidence, which drove the posterior toward certainty it had not earned.
- Fixed a sensor delivering only part of its promised record being rated healthy, which biases inference toward inactivity when loss correlates with activity.
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
