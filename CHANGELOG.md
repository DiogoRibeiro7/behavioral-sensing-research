# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added interpretable history summaries as an optional information component, `InformationComponent.HISTORY_SUMMARY`, for measuring how much longer history explains the gap between the filter and the diagnostic ceiling. Over each summary window, the component gives per-channel activation counts and room changes between active steps. Over the longest window, it gives each channel's quiet minutes and the rooms active in the most recent active step. The windows are `EvidenceResolution.summary_windows`, 60 and 180 minutes by default, chosen from measured bout durations and quiet spells on the development homes. Every summary is a function of whole-step per-channel counts, the evidence the filter receives, so none uses timing or order inside a step. Column names are stable, built by `summary_column` and read back by `parse_summary_column`. Missing, censored and silent history are distinguished, as documented in `docs/INFORMATION_SETS.md`. The four Phase 1 sets, their columns and their digests are unchanged, and no model or result is changed.
- Added a cyclic time-of-day representation for matched-information experiments, in `sensor_modeling.datasets.time_features`:
  - `local_hour` defines the local wall-clock hour once, for the feature builder, with the same reading as the circadian prior;
  - `cyclic_hour_features` places each hour's midpoint on the 24-hour circle with 1 to 11 sine-cosine harmonics, so 23:00 and 00:00 are neighbours;
  - `peak_hour` reads a fitted daily cycle's peak;
  - `LogisticBaseline(hour_encoding="cyclic", harmonics=K)` uses the encoding. The default stays one-hot, and the baseline suite is unchanged.

  It re-encodes the existing hour, so it adds no information and stays within the same information set. Daylight-saving behaviour is documented and tested. Day of week was evaluated on the development homes and not added: whole-day state shares barely differ between weekdays and weekends, and the one visible effect is a morning-only weekend lie-in, which an additive term cannot represent. The production filter and its priors are unchanged.
- Experiment records now follow a versioned, validated schema, 1.1, documented in `docs/EXPERIMENT_ARTIFACTS.md`. `ExperimentRecord` gains typed fields for input provenance (`InputArtifact`, identified by SHA-256), household split, information set, preprocessing, models and their hyperparameters (`ModelRecord`), household-level metrics, quotable intervals (`ReportedInterval`) and simulation MCSE. `write()` validates before writing; `load_record()` and `ExperimentRecord.load()` validate on reading; `validate_record()` reports every problem at once through `ArtifactError`. Files are strict, deterministic JSON: sorted keys, LF endings, the time and environment fixed when the record is created, non-finite numbers written as `null`, and unknown objects refused. Schema 1.0 files are migrated on load with their results unchanged; a newer or foreign version is refused.

### Changed
- The matched evaluation runner fills the new record fields and accepts `inputs`. Per-household metrics, models, split and information set move out of `results` and `configuration` into the record's own fields, and every paired interval is listed in `intervals`. Its result layout is now `matched-evaluation/3`.

### Fixed
- `CITATION.cff` is now valid Citation File Format 1.2.0. Since its first version it had carried three keys the schema forbids (`programming-languages`, `operating-systems`, `subjects`). GitHub therefore never showed a "Cite this repository" button, and tools reading the file, Zenodo among them, could not rely on it. The subject headings not already present are kept as keywords. A new test fails if a non-CFF top-level key returns.
- `ZENODO.md` now matches the archive. The table of archived releases lists every version Zenodo holds, 0.1.0 to 0.6.0, including both 0.1.3 records. The account of the two concept DOIs is corrected: the current concept began with `v0.1.1` on 2026-07-13, not with `0.2.0` as this page and the 0.3.0 changelog entry stated, and the `0.2.0` record declares `isVersionOf 10.5281/zenodo.17070041`, not `isNewVersionOf 10.5281/zenodo.17070042`. The page also no longer claims that release results are simulator-only; real CASAS results have been documented since 0.3.0.

## [0.6.0] - 2026-09-25

Adds the matched-information evaluation needed for Phases 1 and 2 of the roadmap, and records its first real-data result. It provides:

- information sets that declare what a model may condition on;
- a runner that fits and scores models on identical rows under one information set and household split;
- four pre-declared baselines that plug into the runner;
- household-level statistics that compare models by resampling households, never timestamps;
- a first exploratory run on 20 real development homes, measuring what time of day and recent history are worth to the baselines.

It does not change inference, abstention thresholds, transition dynamics, emissions, the behavioural ontology, or the frozen external-validation result.

### Added
- Added matched information sets for the Phase 1 recoverable-information study (`sensor_modeling.datasets.information_sets`). An `InformationSet` declares what a model may condition on: current per-channel activation counts, the local hour, and recent per-channel history, at a shared step, channel vocabulary and history depth. `nested_information_sets()` returns the four Phase 1 sets. `build_feature_table` and `build_panel_features` build per-household features at given prediction moments. Every window closes at or before its moment, annotations are never read, and households are never pooled. An uninstrumented channel or a window before the recording starts is reported as NaN, never as zero. Nothing here changes inference, the ontology, the evaluation splits or the frozen validation results. The contract is documented in `docs/INFORMATION_SETS.md`.
- Added a matched-information evaluation runner, `run_matched_evaluation` in `sensor_modeling.datasets.matched_evaluation`. It fits and scores several models under one information set on a declared training, development and held-out household split. Every model receives identical feature rows in seeded random order, and fitting never sees a held-out household. A model whose prediction for a row depends on other rows is refused. Results are per held-out household: balanced accuracy, per-state recall, confusion matrix, and log loss and Brier score where probabilities exist. Paired household-level differences between models come with bootstrap intervals over households. The output is an `ExperimentRecord` carrying the git commit, package version, information-set and split declarations with digests, model configurations, seed, timestamp and metric definitions. It is documented in `docs/MATCHED_EVALUATION.md`, with an executable example that the test suite runs.
- Added `prediction_metrics` and `confusion_matrix` for predictions from any model. They share their code with `state_metrics`, which now delegates to the same label- and probability-based helpers; its outputs are unchanged, verified identical to the previous implementation.
- Added four pre-declared Phase 2 baselines in `sensor_modeling.datasets.baseline_models`, behind one `Baseline` interface that plugs into the matched evaluation runner:
  - state frequency / majority;
  - persistence, which carries forward the state inferred from the previous window's evidence and reports the training frequencies at a recording's first timestamp;
  - L2-regularised multinomial logistic regression;
  - one depth-limited decision tree.

  `baseline_suite(information_set)` returns them with fixed settings; nothing is tuned and no baseline reads the development rows. Every baseline reports probabilities in label-space order and encodes missing evidence as a value plus an indicator rather than as zero. No baseline reports a probability of zero: counted probabilities (state frequencies and tree leaves) use Laplace's add-one rule over the label space, and a state the logistic model cannot represent, because it has no training rows, gets the same add-one probability, `1 / (N + K)`. The log loss of a state absent from training is therefore set by the training data, not by the metric's `1e-12` floor. Reported states are unchanged by this smoothing. A fitted baseline refuses rows from a different information set. Documented in `docs/BASELINES.md`, including which metrics are meaningful for each model; no model is claimed to perform better. Inference is unchanged.
- Added `evidence_column` and `parse_evidence_column`, which name and parse the information-set feature columns from one definition.
- Added household-level statistics in `sensor_modeling.evaluation.households`. They keep three layers apart:
  1. timestamps are scored within one household (`prediction_metrics`);
  2. households are summarised with each counted once, whatever their length (`score_households`, `household_values`, `summarise_households`);
  3. models are compared by resampling households, never timestamps (`compare_households`).

  `compare_households` reports the mean and median paired difference with bootstrap intervals and standard errors, the number and share of households favouring each model or tied, Cohen's dz, and each household's difference. It gives no p-value. Intervals are percentile or, optionally, BCa, which falls back to percentile where undefined and says so. Missing values are excluded and listed, never imputed. A single household gets an estimate without an interval, and a constant difference gets a zero-width interval. On synthetic panels at nominal 90% coverage, household intervals cover the true effect about 90% of the time; intervals treating timestamps as independent cover it about 14% of the time. The rationale is documented in `docs/EVALUATION_DESIGN.md`.
- Added `sensor_modeling.evaluation.resampling`, the single seeded resampling engine behind every comparison, with `monte_carlo_standard_error` for simulation summaries.
- Added `compare_information_sets`, the deliberate way to measure what extra information is worth to one model. It is refused unless the smaller information set is strictly nested in the larger, and both runs share the split, the scored moments, the label space and the model specification. It accepts one run per cross-fitted fold. `held_out_metrics` pools a model's held-out households across folds and refuses a household held out twice. `MatchedEvaluation` now carries the information set, split, models, label space and moment digests needed for these checks.
- Added the first Phase 1 run, `scripts/run_phase1_matched_baselines.py`, with results in `docs/PHASE1_MATCHED_BASELINES.md`. The protocol was committed before any household was scored: the 20 frozen single-resident development homes, two cross-fitted folds, the four information sets, and the pre-declared baselines. It is exploratory. For linear and tree baselines, time of day is worth +0.09 to +0.13 household balanced accuracy in every home, and recent history +0.02 to +0.04. That does not reproduce the +0.140 an earlier gradient-boosted diagnostic attributed to history. Two runs from a clean commit gave identical results.

### Changed
- Experiment records now store the scikit-learn version, since it changes the fitted baselines.
- `paired_difference` now draws its resamples through the shared engine. Its output is identical to the previous implementation, verified byte for byte, so simulation results are unchanged.
- The matched evaluation runner compares models with `compare_households` and summarises with `summarise_households`. Comparisons now report the median difference and the share of households favouring each model, take an `interval` option (`"percentile"` or `"bca"`), and report a single held-out household instead of skipping it. The result schema is now `matched-evaluation/2`.
- `ExperimentRecord` takes a `data_source` and optional `metric_definitions`. Only `"simulator"` records, the default, carry the note that they were not validated on real data, so a real-data record no longer claims to be simulated.

## [0.5.0] - 2026-09-16

A platform and support-policy release. It modernises the supported Python range,
reduces pull-request CI latency, automates guarded GitHub releases, and separates
manuscript-specific research assets from the software package repository. It
does not change inference, abstention thresholds, transition dynamics,
emissions, the behavioural ontology, or the frozen external-validation result.

### Added
- Added Python 3.13 and Python 3.14 to the supported and tested interpreter matrix. The fast regression suite now runs on Python 3.11, 3.12, 3.13 and 3.14, while the expensive research/performance regression runs once on Python 3.14 after pushes to shared branches.
- Added a guarded manual GitHub Actions release workflow. It runs from `main` only, verifies semantic version input, release metadata, a dated changelog section and tag target, creates or reuses the exact annotated tag, and publishes the GitHub Release from the matching `CHANGELOG.md` section.
- Added a metadata-only pull-request fast path that runs `tests/test_project_metadata.py` once on Python 3.11 without loading the repository-wide pytest conftest or installing the full development environment.

### Changed
- Changed the supported Python contract from 3.10-3.12 to 3.11-3.14. `requires-python` is now `>=3.11,<3.15`; Python 3.10 is no longer supported, and package classifiers now advertise 3.11, 3.12, 3.13 and 3.14.
- Split ordinary pull-request testing from heavy research/performance regression. Pull requests keep broad interpreter coverage without rerunning the slow studies on every Python version; shared-branch pushes retain the slow backstop.
- Updated Black to 26.5.1 and aligned formatting with the minimum supported syntax, Python 3.11, while runtime compatibility remains tested through Python 3.14.
- Made the GitHub Actions release workflow the canonical release mechanism, with manual Git/CLI commands retained only as a fallback.
- Moved both manuscripts (`papers/`), the Paper 1 analysis scripts, their tests, `artifacts/paper1/` and the five paper workflows to the private `research-articles` repository. They were copied byte for byte from commit `012f22b` and are pinned there to `sensor-modeling` `v0.4.0`. No package code changes resulted from the move, and the repository history remains intact. `artifacts/v03/` stays here because the documentation cites it.

### Fixed
- Fixed the metadata fast path so the self-contained release-metadata test does not load `tests/conftest.py` and therefore does not require Matplotlib merely to validate citation/version files.
- Fixed the release workflow to use the same isolated metadata-test invocation, preventing the first automated release from failing for the same conftest dependency reason.
- Fixed shared-branch lint after the Python support expansion: the old Black hook did not recognise Python 3.13/3.14 targets, and Black's safety check cannot validate Python 3.14-targeted formatting while itself running under Python 3.11. The formatter now targets the minimum supported syntax and the one newly formatted benchmark file is committed.
- Documentation now gives the scope of three real-data figures. The 60,948-step confidence analysis covers five homes, not the 22-home panel (`docs/limitations.md`, `docs/RESEARCH_QUESTIONS.md`, `docs/UNCERTAINTY_MODEL.md`). The unscoped 2.2% abstention figure is replaced by the 22-home median of 2.5%, at most 3.9% in any home, from the results table in `docs/real_data.md` (`docs/limitations.md`, `docs/real_data.md`). `docs/real_data.md` no longer says all 22 homes are single-resident; `hh107` and `hh121` are two-occupant recordings, as the same page already records further down.

## [0.4.0] - 2026-09-15

Turns the uncertainty diagnostics introduced in 0.3.0 into a reproducible,
threshold-free correctness-separation workflow for real-data research. This
release is diagnostic: it does not change inference, abstention thresholds,
transition dynamics, emissions, the ontology, or the frozen v0.3 external
validation result.

### Added
- Added correct/incorrect median summaries for per-update information gain to `UncertaintyDiagnostics`. Missing information-gain values remain missing rather than being coerced to zero, while a genuine measured zero remains a valid observation.
- Added a common-language/AUC-style correctness-separation statistic for posterior confidence, posterior margin, normalised entropy, interval-level evidence strength, and per-update information gain. The statistic gives half credit for ties, uses the favourable direction for each metric, and has the common interpretation `0.5` = no separation, `>0.5` = correct-favouring separation, `<0.5` = inversion.
- Added `uncertainty_panel_summary(...)`, aggregating the five separation diagnostics across households with one value per home so long recordings do not dominate short ones. Missing information-gain diagnostics reduce only the information-gain coverage count.
- Added `scripts/analyse_uncertainty_panel.py`, a reproducible Paper 1 analysis entry point for the recorded 22-home CASAS development panel. It runs the existing default inference path, records household uncertainty diagnostics, and writes a machine-readable equal-household panel summary without touching the frozen 43-home primary external cohort.
- Added byte-exact provenance checks to the Paper 1 panel runner. Every development recording must match the frozen manifest filename, development-only status, byte size, and SHA-256 digest before it can be analysed; the verified source identity is written into the result artifact. Focused tests lock the exact 22-home membership and rejection of missing identities, size mismatches, and digest mismatches.
- Added a GitHub Actions execution path for the Paper 1 panel analysis. The workflow downloads the frozen CASAS Zenodo archive, verifies its recorded size and MD5, relies on the runner to verify all 22 household files again, and uploads only the resulting JSON artifact.
- Executed the frozen 22-home development-panel analysis and committed the machine-readable result at `artifacts/paper1/uncertainty_panel.json`. Median within-home correctness AUC is `0.5193` for posterior confidence, `0.5184` for posterior margin, `0.5135` for normalised entropy, `0.3719` for evidence strength, and `0.3875` for per-update information gain; all 22 homes contribute to every diagnostic. All three directional hypotheses fixed before execution fail.
- Completed the first post-v0.3 follow-up manuscript, `papers/when-confidence-is-not-information/`, with the executed result and its pre-specified negative interpretation: information gain does not rescue confidence-based abstention, evidence strength is also inverted, and no new threshold or deployment rule is selected.
- Added `scripts/plot_uncertainty_panel.py` and the corresponding paper figure `papers/when-confidence-is-not-information/uncertainty_panel_auc.pdf`, showing one household point per diagnostic with the fixed `AUC = 0.5` reference and no fitted trend, threshold line, post-hoc filtering, or weighting.

### Changed
- Refocused `ROADMAP.md` on three post-v0.3 research papers: confidence/information failure, recoverable-information versus formulation limits, and the sensor-information frontier. Software work is now explicitly subordinate to those scientific questions rather than to a pre-declared next package release.
- Simplified release documentation so the matching `CHANGELOG.md` version section is the single source of GitHub Release notes. Separate per-version release-note files are no longer maintained, and ancestry-only `main` to `develop` merges are not required when the released file tree is unchanged.

## [0.3.0] - 2026-09-12

Moves the ambient-sensing work from simulator-only claims to measured real-data
and frozen external validation, while keeping the new modelling components
optional and the negative abstention result explicit.

### Fixed
- Fixed `Leave_Home` being mapped to `AWAY` in the CASAS `hh` reader. That label annotates the act of crossing the threshold, with a median duration of about twelve seconds, so the truth series marked a burst of motion inside the house as absence and the inference was scored wrong for correctly reporting activity, while the hours actually spent out carried no label and were never scored. Intervals labelled `AWAY` showed 130 activations per hour, which is what exposed it. Away is now derived from the gap between a departure and the next arrival. Across 22 homes this raises `away` recall from 0.00 to 0.36, labelled coverage from 64% to 90%, and median balanced accuracy from 0.364 to 0.420. The previously published `away` figure was an adapter artefact, not a property of the pipeline, and `docs/real_data.md` carries the correction.

### Added
- Executed the frozen external validation once on 43 single-resident CASAS homes outside the development panel. The optional circadian v0.3 candidate improved the median paired household balanced accuracy by **+0.0091**, with a 95% household bootstrap interval of **[+0.0054, +0.0117]**; 37 homes improved, 6 worsened, and none were unchanged or unscoreable. Secondary metrics moved in the same direction: balanced accuracy 0.496 to 0.504, calibration error 0.343 to 0.328, Brier 0.842 to 0.812, and log loss 4.302 to 4.168. The effect is small and does not establish clinical effectiveness or general smart-home performance.
- Added real-data uncertainty diagnostics separating correct from incorrect estimates by posterior confidence, margin, normalised entropy and interval-level evidence strength. Controlled quiet-period regressions show that the transition prior alone remains near its stationary confidence, while complementary Poisson silence likelihoods from working room-motion and entrance-door streams can drive `sleeping` confidence above 0.95. `StateEstimate` also now exposes optional per-update information gain, `D_KL(posterior || prediction)`, as a passive diagnostic; it does not change abstention or inference.
- Added `sensor_modeling.utils.text_file_sha256`, hashing a text file with CRLF newlines normalised to LF. Frozen artifacts record the digest of the file they were derived from, and a later run recomputes it to decide whether the input still is what was frozen. Hashing raw bytes made that describe the checkout rather than the content: git rewrites line endings on platforms that ask for CRLF, so the same manifest verified on Linux and was rejected on Windows, which stopped an external scoring run against a cohort that had not changed. The script that records these digests and the script that verifies them now share this one function. Archive data and result files are still hashed byte-exact, where a normalising digest would hide real corruption.
- Sensor ingestion now raises typed errors from the `DataExcept` taxonomy: `SensorDataLoadingError`, `SensorDataFormatError`, `SensorDataValidationError`, `SensorMissingDataError` and `SensorDependencyError` in `sensor_modeling.data.exceptions`. Each subclasses both its `DataExcept` base and the exception the loaders raised before it, `ValueError` for the first four and `ImportError` for the last, so existing `except ValueError:` handlers around CSV, JSON and HDF5 loading keep working unchanged. Callers that want to distinguish a malformed payload from an out-of-range value can now do so without parsing message strings. This adds a runtime dependency on `DataExcept>=1.3,<2.0`.
- Measured the three real-data components together rather than only in isolation. They interfere: all three score 0.435 balanced accuracy on held-out homes against 0.463 for smoothing alone, with calibration error worse than the baseline. Fitted rates trade accuracy for calibration, smoothing trades calibration for accuracy, and only the circadian prior improves both. `docs/real_data.md` now recommends one component per use rather than leaving a reader to stack three features each documented as an improvement.
- Added `smooth_estimates` and `smooth_beliefs`, revising past estimates with evidence that arrived after them. This is the only way a recursive filter can use more context without double-counting, since it conditions on later observations rather than re-reading absorbed ones. On 11 held-out CASAS homes a lag of one step raises median balanced accuracy from 0.449 to 0.463, and longer lags add nothing: four hours scores the same as five minutes. Intended for evaluation, baseline and reporting paths, and explicitly not for alerting, where the delay is the whole cost.
- Tested and rejected dwell-time miscalibration as a cause of the real-data accuracy gap. Declared dwell times are 3 to 9 times longer than measured state durations, but fitting them on 11 homes and scoring on 11 held out lowered median balanced accuracy from 0.449 to 0.429 and left `bathroom_activity` recall essentially unchanged despite halving its prior. The long dwells appear to be earning their keep as regularisation, so no dwell-fitting feature is shipped.
- Added an optional `circadian` profile to `StateOntology`, giving the continuous-time chain a time-of-day term: 24 stickiness multipliers per state divide that state's exit rates for the current hour, so a state the resident usually occupies then becomes harder to leave. Fitted on 11 CASAS homes and scored on 11 held out, it raises median balanced accuracy from 0.449 to 0.460 and cuts calibration error from 0.312 to 0.296, improving 10 of 11 homes. That is roughly a tenth of the +0.105 the feature ablation attributed to time of day: modulating transition rates is a weaker lever than conditioning on the hour directly, and most of the circadian signal remains unexploited. Off by default and identical to earlier releases when unset.
- Located the real-data accuracy shortfall by ablating the diagnostic classifier's features. Given only instantaneous per-room event counts the achievable ceiling is 0.397 and the pipeline scores 0.420, so on the evidence it uses it is already at the ceiling. The missing accuracy is in an explicit time-of-day term, worth about +0.105, and several steps of recent room-resolved counts, worth about +0.140. The continuous-time Markov prior models state duration but not circadian plausibility, so a resident motionless at 02:00 and at 14:00 are indistinguishable to it.
- Measured how much of the state ontology is recoverable from real motion and door sensors at all. A gradient-boosted classifier given the same per-room event counts, lagged steps and time of day reaches 0.607 balanced accuracy on 11 held-out homes against a 0.143 majority-class baseline, so the seven states are identifiable from this instrumentation and the pipeline's 0.420 is a deficiency in the inference rather than a limit of the deployment. The simulator's 0.816 sits above that ceiling, meaning its figures exceed what real instrumentation supports rather than merely being optimistic. Recorded in `docs/real_data.md`; the classifier is a diagnostic bound and is not part of the package or the inference path.
- Added `fit_emission_defaults`, deriving emission rate constants from annotated recordings so declared rates can be compared against measured ones on held-out homes. Fitting on 11 CASAS homes and scoring on 11 others cut median calibration error from 0.312 to 0.202 while leaving balanced accuracy unchanged at roughly 0.42: rate miscalibration explains the pipeline's overconfidence on real data and not its accuracy gap.
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
- Added `sensor_modeling.evaluation`: problem-appropriate evaluation metrics (balanced accuracy, macro F1, log loss, Brier, calibration error, transition timing, detection delay, false positives per person-day) and a paired sensor-ablation framework reporting bootstrap intervals and effect sizes.
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