# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Ran the frozen Phase 5 external-generalisation protocol on the UCI ADL Binary dataset, `sensor_modeling.datasets.external_experiment`, without changing preprocessing, mapping, hyperparameters or eligibility.
  - **Run.** `scripts/run_phase5_external.py` refuses to run unless:
    - the code's protocol equals the frozen file;
    - the mapping and every external and CASAS recording match their frozen digests;
    - the CASAS populations reproduce their pinned fit digests.

    The record, `artifacts/phase5/phase5-external-ordonez-results.json`, was made from a clean commit.
  - **Reported per home.**
    - Balanced accuracy, per-state recall, log loss, Brier score, calibration error, macro F1 and accuracy for the declared rates, zero-shot and adapted conditions, with day-block bootstrap intervals.
    - The unsupported-observation fraction and the ontology-mapping coverage.
  - **Per estimand.** Paired differences, intervals and verdicts, and homes improved and worsened.
  - **Descriptive diagnostics.** Missing sensor semantics, ontology mismatch, room structure, event rates, calibration shift and state-prior shift. An in-sample oracle separates dataset incompatibility, the sensing-information limitation and model failure.
  - **Output.** `external_summary.render_page` and `external_figures.draw_figures` generate `docs/PHASE5_EXTERNAL_RESULTS.md` and its figures from the record alone. An ineligible home would be reported, not dropped.
  - **Result.**
    - The CASAS-trained model does not transfer to either home.
    - Zero-shot balanced accuracy is at chance: 0.246 and 0.244 against 0.25. It predicts `sleeping` or the unscorable `bed_awake` for every window.
    - Limited adaptation improves log loss in both homes but leaves balanced accuracy unchanged.
    - Structural disagreement beats confidence in one home and is uncertain in the other: inconclusive.
    - The in-sample oracle reaches 0.437 and 0.559.
- Froze the first Phase 5 external-generalisation protocol, `artifacts/phase5/external_protocol.json`, before any external scoring. `docs/PHASE5_EXTERNAL_PROTOCOL.md` is generated from it. No model performance on external data is computed or reported.
  - **Dataset.** The UCI ADL Binary dataset of Ordóñez et al. (DOI 10.24432/C5J02M, CC BY 4.0): two single-resident homes, independent of CASAS, with every file's SHA-256 and the archive's. `sensor_modeling.external.ordonez` reads it into the external-dataset contract.
  - **Mapping.** `artifacts/phase5/ordonez_mapping.json`, written from the dataset's documentation, with its SHA-256. Meal labels are ambiguous and unscored, `Leaving` maps approximately to `away`, and each sensor's type, room and model channel, or why it is unsupported, is recorded per home.
  - **Protocol.** `sensor_modeling.datasets.external_protocol` declares every item:
    - households, eligibility and a seven-day adaptation period per home;
    - preprocessing, and the unsupported states and sensors;
    - model versions pinned by the CASAS population's fit digests;
    - zero-shot and limited-adaptation conditions, with their adaptation rules;
    - metrics, a day-block bootstrap within each home, and minimal differences;
    - the success and failure criteria.
  - **Checks.** `scripts/check_phase5_inputs.py` verifies a downloaded archive against the frozen digests, and validates each home's adaptation period only.
- Added the Phase 5 external-dataset contract, `sensor_modeling.external`, described in `docs/EXTERNAL_DATASET_CONTRACT.md`. It is infrastructure: no model is scored on external data.
  - **Contract.** `HouseholdData`, `SensorDescription`, `RawEvent`, `Annotation` with interval or point semantics, `OccupancyPeriod`, and `DatasetProvenance`, in the dataset's own terms, behind a `DatasetAdapter` protocol. `InMemoryAdapter` and a column-declared long-format `CsvAdapter` are the reference adapters.
  - **Mapping.** `OntologyMapping` declares every native label, sensor type and location as exact, approximate, unmappable or ambiguous, with a rationale for all but exact. Undeclared values are reported, never dropped. A mapping is written and read with its SHA-256, so it can be frozen before external results.
  - **Validation.** `validate_household` and `validate_dataset` report missing or invalid timezones, daylight-saving gaps and folds, timestamp order, duplicate events, unknown and unused sensors, impossible intervals, overlapping labels, unsupported multi-resident periods, undeclared labels and sensors, and sensor-semantic mismatches, with label coverage by mapping outcome.
  - **Conversion.** `to_canonical` produces the repository's recording type: sensor registry, ordered observations, and non-overlapping labelled segments. Conflicting, multi-resident and unmappable spans are unscored. Every event, sensor and annotated second it does not carry is counted by reason.
- Added the pre-specified Phase 4 comparison of richer uncertainty diagnostics, `sensor_modeling.datasets.uncertainty_experiment`, with its frozen protocol, `artifacts/phase4/uncertainty_protocol.json`, committed before any household was scored.
  - **Signals.** Posterior confidence, entropy, model-structure disagreement over a population-only ensemble, evidence-channel disagreement, and predictive mismatch. Each ranks the same predictions: the Phase 3.3 follow-up's hurdle recursion on the 20 development homes, cross-fitted on the frozen folds.
  - **Evaluation.** Selective-prediction curves of error, balanced accuracy, calibration and per-state retained coverage, pooled and per household. Each signal is compared with confidence by paired household differences with household bootstrap intervals.
  - **Rule.** Declared in advance: a structural diagnostic is materially better when its error AURC beats confidence's by at least 0.01, with the interval above 0, and no difficult minority state is retained less than under confidence. No threshold is selected.
  - **Output.** `uncertainty_summary.render_page` and `uncertainty_figures.draw_figures` generate the documentation page and its figures entirely from the record. `scripts/run_phase4_uncertainty.py` refuses to run unless the code's protocol equals the frozen file.
  - **Result.** The run is published in `artifacts/phase4/phase4-uncertainty-diagnostics.json`, made from the protocol commit on a clean tree. The page, `docs/PHASE4_UNCERTAINTY_DIAGNOSTICS.md`, and its figures are generated from it. It is on the 20 development homes, so it is not a held-out claim. Its predictions reproduce the Phase 3.3 follow-up's hurdle recursion to 3e-15.
    - Model-structure disagreement is materially better than confidence by the declared rule. Household error AURC is 0.059 [0.031, 0.087] lower, and the guard holds.
    - Its gain is at low coverage: at 50% to 90% coverage its error and balanced-accuracy differences are uncertain or negligible.
    - It retains less of kitchen and bathroom activity than confidence does at 50% and 70%, and the guard did not cover these states. The frozen rule admitted only `bed_awake`, with 68 windows.
    - Confidence barely orders the predictions (pooled gain 0.04), and entropy is indistinguishable from it.
    - Evidence-channel disagreement and predictive mismatch are worse than confidence, and reject activity states and `bed_awake` heavily.
- Added a selective-prediction evaluation framework for ROADMAP Phase 4, `sensor_modeling.evaluation.selective`, described in `docs/SELECTIVE_PREDICTION.md`. It evaluates any candidate risk signal without selecting a threshold or an abstention rule.
  - **Signals.** A `Signal` states its direction, `higher_is_riskier` or `higher_is_safer`, which is never assumed. A missing value is refused unless a policy ranks it for rejection or retention.
  - **Curves.** Over a grid of coverage levels, pooled over the panel and per household:
    - selective risk under 0-1 or a stated loss, and the error rate;
    - selective balanced accuracy and the states still scored;
    - per-state retained coverage;
    - calibration among retained predictions;
    - the rejected windows' state composition and error rate;
    - each household's coverage under a pooled selection.
  - **Ties.** Ties at the boundary are retained in part, the expectation of random tie-breaking, so a constant signal is exactly random rejection.
  - **References and summaries.** Random rejection, whose ratio metrics stay at their full-coverage values, and the oracle, which rejects the costliest predictions first. They give the AURC, the excess AURC over the oracle, and the gain over random: 1 for the oracle, 0 for random.
  - **Uncertainty.** Timestamps are selected and aggregated. Households, not timestamps, are resampled, with selection repeated in every resample. Household curves are summarised with each household counted once.
- Added observation-model mismatch diagnostics for ROADMAP Phase 4, described in `docs/OBSERVATION_MISMATCH.md`. They ask how surprising a window's evidence is under every state, which confidence cannot show. They are diagnostic infrastructure: no threshold, no abstention, no evaluation, and no decision changed.
  - **Count laws.** `sensor_modeling.datasets.observation_mismatch.CountLaw` gives the hurdle, hurdle negative-binomial and Poisson channel models normalised laws, keeping the `log k!` their likelihoods drop. The laws have exact log-space upper and lower tails, a direct sum near underflow, and the entropy and varentropy of each state.
  - **Measures.** `sensor_modeling.evaluation.MismatchTrace` keeps each window's per-channel, per-state log-probabilities and tails. From them it derives:
    - the best achievable state;
    - two-sided tails, which are valid p-values, and each channel's tail under its most lenient state;
    - the standardised surprise per state, with its exact mean and variance;
    - each channel's excess surprise, which sums to the window's;
    - the posterior predictive and its gap below the best state;
    - the activity pattern's exact tail and training support;
    - counts beyond the training support.
  - **Missing evidence.** Channels are available, missing or failed, a known sensor failure given as intervals. Only available channels are scored, so a failure is never read as novelty, and a window with none has no evidence.
  - **Summaries.** Distributions, and shares at the tail levels 10⁻² to 10⁻⁹, with no single threshold.
  - **Traces and households.** Traces are canonical gzip JSON with a SHA-256. `household_mismatch` scores a CASAS household against its channel models, with the recursion's prediction, the population's training support and the training households' activity patterns.
- Added evidence-group disagreement diagnostics for ROADMAP Phase 4, `sensor_modeling.evaluation.evidence_groups`, described in `docs/EVIDENCE_GROUP_DISAGREEMENT.md`. They are diagnostic infrastructure: no abstention rule, no evaluation, and every filter output unchanged.
  - **Groups.** From each sensor's modality in the registry: `motion` (motion, radar), `contact` (door, contact, vibration), `bed`, `wearable` (wearable motion and physiology, proximity) and `other`. The `context` group is the filter's prediction before the window's evidence.
  - **Per window.** Each identifiable group's posterior from its own tempered evidence, the full posterior, whether the groups favour disjoint states, the vote disagreement, pairwise and generalised Jensen-Shannon divergence, each group's divergence from the full posterior, descriptive conflicts at a stated likelihood ratio, and each group's exact term in the decision's log odds, with the group that dominates it.
  - **Missing evidence.** Sensor statuses keep the estimate's semantics: `unavailable` is its `missing`, and `silent` and `no data` together its `silent`. Absent, unavailable and uninformative groups are never compared, so missing evidence is never disagreement.
  - **Results.** `GroupDisagreementRecorder` observes a filter and returns a JSON-safe results block, layout `evidence-group-disagreement/1`, with a canonical gzip trace referenced by SHA-256.
- Added model-structure disagreement diagnostics for ROADMAP Phase 4, described in `docs/STRUCTURAL_DISAGREEMENT.md`. They are diagnostic infrastructure: no threshold, no abstention, and the deployed decision rule unchanged.
  - **Measures.** `sensor_modeling.evaluation.disagreement` compares the posteriors of several fitted specifications over the same windows. For every window it gives each specification's posterior and most probable state, the pairwise Jensen-Shannon divergence in bits, the vote disagreement, each state's probability spread, the mean posterior, and the generalised Jensen-Shannon divergence. Each measure is symmetric, bounded and zero for identical specifications.
  - **Traces.** `DisagreementTrace` holds one household's windows. It writes them as canonical gzip-compressed JSON with a SHA-256, and summarises them.
  - **Specifications.** `sensor_modeling.datasets.structural_models` allows only variants the published evidence supports: the hurdle or hurdle negative-binomial observation model, population or partially pooled parameters, and the population fitted on all training households or on either half of them. Each cites its supporting record and finding. The time prior as a transition, fixed-lag smoothing and the declared rates are refused with reasons, and an ensemble holds at most six specifications.
  - **Decision.** The reference, the Phase 3.3 follow-up's recursion, keeps the decision, bit for bit.
- Added the pre-specified evaluation of the hurdle negative binomial, `sensor_modeling.datasets.dispersion_experiment`, with its frozen protocol, `artifacts/phase3/dispersion_protocol.json`, committed before any household was scored. The design is described in `docs/PHASE3_NEGATIVE_BINOMIAL.md`.
  - **Models.** The declared rates, the fitted hurdle-Poisson and the fitted hurdle negative binomial, with the same prior, transition, channels, windows and inference, scored with current windows and in the filter's recursion.
  - **Questions.** Against the hurdle-Poisson, whether the negative binomial recovers `home_active` recall, preserves the calibration improvement, improves log loss and avoids degrading balanced accuracy. A rule fixed in advance adopts it only with a log-loss gain and every guard met, so extra complexity is never accepted for one improving metric.
  - **Also reported.** Per-state recall, Brier score, the quiet-run overconfidence slope, and every fold's dispersion estimate per channel and state, classed by whether an extreme estimate rests on little data or a flat likelihood.
  - **Output.** `dispersion_summary.render_summary` generates the Markdown summary from the record. `scripts/run_phase3_negative_binomial.py` refuses to run unless the code's protocol equals the frozen file.
  - **Result.** The run is published in `artifacts/phase3/phase3-hurdle-negative-binomial.json`, made from the protocol commit on a clean tree. It is on the 20 development homes, so it is not a held-out claim. Its declared and hurdle cells reproduce the Phase 3.3 follow-up exactly.
    - Against the hurdle-Poisson, log loss improves by 0.484 with current windows and 0.784 in the recursion, in all 20 homes. Calibration error improves by 0.048 and 0.104, and balanced accuracy by 0.044 in both.
    - `home_active` recall is not recovered: it falls a further 0.055 with current windows, and the change in the recursion is uncertain.
    - By the rule declared in advance, the model is adopted in the recursion and a trade-off with current windows, so it is not adopted.
    - Of 9 extreme dispersion estimates, 2 rest on little data; 6 are in `sleeping`, where motion counts approach the logarithmic-series limit.
- Added a hurdle model with a zero-truncated negative-binomial active count, the `hurdle_nb` family in `sensor_modeling.datasets.channel_models`, described in `docs/HURDLE_NEGATIVE_BINOMIAL.md`. The posterior predictive checks found the zero-truncated Poisson active count systematically under-dispersed, with a variance growing with the square of the mean. No comparison with the hurdle-Poisson model is run here.
  - **Model.** `HurdleNBChannel` keeps the hurdle's silence probability and replaces the active count with a negative binomial of mean `mu` and dispersion `alpha`, truncated at zero. `alpha = 0` is exactly the hurdle-Poisson model, which remains available and unchanged.
  - **Fitting.** On training households only. Silence and the active mean are the hurdle's shrunk estimates. The dispersion maximises the profile likelihood of the active-count table plus 12 Poisson-shaped pseudo-windows at the same mean, which shrink it toward the Poisson. It is bounded at 100, where the family approaches the logarithmic series, and a fit at the bound is flagged.
  - **Pooling.** `HouseholdChannels.models(dispersion=...)` builds a household's models from its pooled silence and active mean and the population's dispersion.
  - **Numerics.** The coefficients use `log B(k, r)`, the truncation term a stable `log(1 - exp(x))`, and the Poisson limit its own branch; counts of a million stay finite.
- Added pre-specified posterior predictive checks of the fitted hurdle channel model, `sensor_modeling.datasets.predictive_checks`, with its frozen protocol, `artifacts/phase3/predictive_protocol.json`, committed before any household was examined. The design is described in `docs/PHASE3_PREDICTIVE_CHECKS.md`. It is diagnostic only: no model is changed and no inference is run.
  - **Cells.** Each household's labelled windows of one state on one channel, with at least 50 windows, are checked against two references: the hurdle fitted to the cell alone, which tests the family, and the fold's population hurdle, which inference uses.
  - **Statistics.** Silence, mean, variance, the active mean and dispersion, the 90th and 99th percentiles, the tail above the 99th, windows in long quiet runs, and windows in bursts. Each is compared with 200 replicates of the cell's own windows, refitted for the cell's own fit.
  - **Households.** Households are the unit. Every summary by state, channel, room and channel type is a household bootstrap of household means, and every household's values stay in the record.
  - **Rules.** Declared in advance, they judge whether the zero-truncated Poisson active count is under-dispersed and whether quiet runs or bursts are in excess, and route the result to a next model family, if any.
  - **Output.** `predictive_summary.render_summary` and `predictive_figures.draw_figures` generate the Markdown summary and the figures from the record. `scripts/run_phase3_predictive_checks.py` refuses to run unless the code's protocol equals the frozen file.
  - **Tests.** Synthetic tests check that the checks accept a correctly specified Poisson model and detect a negative-binomial-like process and clustering in time.
  - **Result.** The run is published in `artifacts/phase3/phase3-hurdle-predictive-checks.json`, made from the protocol commit on a clean tree, with its figures in `docs/figures/phase3-predictive-*.svg`. It is on the 20 development homes. Its population fit is the Phase 3.3 follow-up's, by digest.
    - The zero-truncated Poisson active count is systematically under-dispersed: the observed active variance is 4.3 to 7.4 times the predicted in the common states, in every household, and 472 of 493 cells are flagged. The excess grows with the square of the mean.
    - Long quiet runs are in systematic excess, 1.3 to 5.2 times in `away`, `home_active` and `home_inactive`. Bursts are in excess only in `sleeping` and `home_inactive`.
    - The declared routing names within-state temporal dependence, activity sub-states or a Markov-modulated emission, as the next model family, and the roadmap records it.
- Added the pre-specified Phase 3.5 evaluation of fixed-lag smoothing, `sensor_modeling.datasets.smoothing_experiment`, with its frozen protocol, `artifacts/phase3/smoothing_protocol.json`, committed before any household was scored. The design is described in `docs/PHASE3_SMOOTHING.md`.
  - **Formulation.** The Phase 3.3 follow-up's fitted-hurdle recursion, the recursion over every window with a pre-specified success.
  - **Regimes.** The online filter, and fixed-lag smoothers with lags of 1, 6 and 12 windows (5, 30 and 60 minutes), each with a declared operational use. Every regime is scored on the same labelled windows, and every smoothed estimate reads its full lag.
  - **Measures.** Balanced accuracy, per-state recall, log loss, Brier score and calibration error. Also the share of states changed, corrected and made wrong relative to the online filter, accuracy near transitions, and each regime's decision delay after a transition, including its reporting delay.
  - **Estimands.** Each smoother against the online filter, as a labelled smoothing gain, judged by a rule fixed in advance: gain, trade-off, probability gain, no gain or inconclusive.
  - **Output.** `smoothing_summary.render_summary` generates the Markdown summary from the record. `scripts/run_phase3_smoothing.py` refuses to run unless the code's protocol equals the frozen file.
  - **The record.** It is the first to compare regimes. Its `inference` field states the longest lag, which bounds every estimate in it, and its results label every cell and comparison with its own regime.
  - **Result.** The run is published in `artifacts/phase3/phase3-fixed-lag-smoothing.json`, made from the protocol commit on a clean tree. It is on the 20 development homes, so it is not a held-out claim. Its online filter reproduces the Phase 3.3 follow-up's `filter_hurdle@R` exactly. Every figure is a smoothing gain, available only after the smoother's delay.
    - Five minutes of lag is a pre-specified gain. Balanced accuracy rises by 0.031 [0.021, 0.043], in all 20 homes. No probability metric favours the online filter: log loss worsens by 0.044, short of its minimal difference, an uncertain verdict.
    - Thirty and sixty minutes are pre-specified trade-offs. Balanced accuracy rises by 0.043 and 0.049, and log loss worsens by 0.101 and 0.119.
    - Only 37% of the states smoothing changes at five minutes are corrections. `home_active` recall is unchanged at every lag, and `away` recall gains 0.060 and 0.083 at the longer lags.
    - No smoother reports a new state sooner than the online filter.
- Added the pre-specified Phase 3.4 evaluation of partial pooling, `sensor_modeling.datasets.pooling_experiment`, with its frozen protocol, `artifacts/phase3/pooling_protocol.json`, committed before any household was scored. The design is described in `docs/PHASE3_PARTIAL_POOLING.md`.
  - **Models.** The hurdle channel parameters come from one of: the population, fitted per fold on training homes; each held-out home pooled toward it with the declared strength, 288 windows; the same with a strength selected by leave-one-household-out on training homes only; or unconstrained per-home estimates. The declared rates give context.
  - **Arms and settings.** A 7-day and a 1-day adaptation arm, with every model in an arm scored on the same windows after the cut-off. Each model is scored with current windows and in the filter's recursion.
  - **Estimands.** Pooled against population is the primary pooling question. Pooled against unconstrained after one day is the primary overfitting question. Log loss is the primary metric, and balanced accuracy and calibration error are guards. Results are also stratified by each home's amount of adaptation data.
  - **Output.** `pooling_summary.render_summary` generates the Markdown summary from the record. `scripts/run_phase3_pooling.py` refuses to run unless the code's protocol equals the frozen file.
  - **Result.** The run is published in `artifacts/phase3/phase3-partial-pooling.json`, made from the protocol commit on a clean tree. It is on the 20 development homes, so it is not a held-out claim. The population is identical to the Phase 3.3 follow-up's.
    - Pooling a week of household data improves log loss with current windows by 0.080 [0.039, 0.118], in 17 of 20 homes, a pre-specified success. In the recursion it is inconclusive: one home carries the mean, and four worsen by more than 0.4.
    - Unconstrained per-home fitting overfits small homes. After one day it is worse than pooling by 0.175 [0.055, 0.302], and worse than the population alone by 0.161.
    - The strength selected on training homes was the grid's smallest, 24 windows, in both folds. With it, pooling improves log loss by 0.132. The declared 288 pools more than a week of data needs.

- Added the inference-regime contract of ROADMAP 3.5, so that a smoothing gain cannot be reported as an online one. It is described in `docs/INFERENCE_REGIMES.md`. No inference algorithm changes.
  - **Regimes.** `InferenceRegime` gains `is_causal` and `delay`, the effective reporting delay.
  - **Estimates.** `ReportedEstimate` records an estimate's prediction timestamp, the latest evidence it read, and when it can first be reported. It refuses evidence beyond its regime and an availability the regime cannot have. `regime_estimates` builds them from filtered beliefs.
  - **Records.** `EvidenceSummary` summarises the scored estimates' timestamps and their largest lead of evidence over prediction. `NotEnumerated` gives the reason a causal result does not list them; a smoother may not use it.
  - **Evaluation.** `sensor_modeling.evaluation.RegimeResult` keeps one value per household with its regime and evidence.
    - `compare_results` refuses unlabelled results, and results of different regimes unless a smoothing gain is requested. A smoothing gain is then labelled with its delay.
    - `RegimeComparison.online_gain` refuses anything but two online results.
    - `pool_results` refuses to mix regimes or count a household twice.
  - **Tests.** The strict leakage tests stream a synthetic sequence whose later evidence changes an earlier posterior, and a simulated day through the online pipeline.

### Changed
- The experiment-record schema is 1.6. It adds an optional, validated `selective_prediction` section with each signal's direction, curves, intervals and summaries, the random and oracle references, and the coverage grid and bootstrap settings. Records written at 1.0 to 1.5 are migrated with the section `null`, and are otherwise unchanged.
- The experiment-record schema is 1.5. It adds an optional, validated `observation_mismatch` section with the model's family and training households, the tail levels, and each household's summary and trace file with its digest. Records written at 1.0 to 1.4 are migrated with the section `null`, and are otherwise unchanged.
- `MultimodalBayesFilter.update` is split into `evidence_terms`, which computes an update's prediction and each sensor's tempered log-likelihood without changing the filter, and the fold of those terms. Observers appended to the new `observers` list are called after every update with its `WindowTerms` and estimate. Every belief, information gain, completeness and contribution is unchanged to the last bit.
- The experiment-record schema is 1.4. It adds an optional, validated `structural_disagreement` section with the specifications and their support, the reference, and each household's summary and trace file with its digest. Records written at 1.0 to 1.3 are migrated with the section `null`, and are otherwise unchanged.
- `ChannelStatistics` gains an optional `active_counts` table of active-window counts, built by `home_statistics` and merged by `combine_statistics`. It is excluded from `FittedChannels.to_dict`, so every fit's SHA-256, including the published populations', is unchanged.
- `sensor_modeling.datasets.channel_models` gains `home_statistics`, `combine_statistics` and `pool_channels`, so that one household's statistics can be combined or pooled at several strengths without being counted again. `fit_channel_models` and `adapt_channels` now use them, and give identical results.
- The experiment-record schema is 1.3. The `inference` block adds `causal`, `delay_seconds` and `evidence`, and `ExperimentRecord` requires `evidence`.
  - A record is refused if its causality or delay disagrees with its regime, or if its estimates read past it.
  - Every experiment in `sensor_modeling.datasets` lists its scored moments. The CLI's simulation studies give a reason instead.
  - 1.1 and 1.2 records are migrated on load, with their evidence marked not listed and the reason given. A smoothed 1.2 record is refused, and none was published.
  - The published records and their generated summaries are unchanged.
- A `StateEstimate`'s belief is read-only, so a reported estimate cannot be revised in place. `smooth_estimates` already returned new estimates.
- `compare_households` refuses a `RegimeResult` and points to `compare_results`.

## [0.8.0] - 2026-09-27

Records the first Phase 3 inference-redesign results. Each experiment was pre-specified, with its protocol frozen before any household was scored, and run on the 20 development homes, so none is a held-out claim. It provides:

- a hierarchical periodic state prior and its evaluation. The hour is worth +0.131 balanced accuracy to the generative model, a pre-specified success;
- an explicit recent-history state and its evaluation. On identical information it lowers balanced accuracy by 0.042, a pre-specified failure;
- a correlated-silence diagnostic. Silence is strongly dependent between channels, but the pre-specified consequence is not observed, so the hypothesis is weakened;
- fitted hurdle observation models for the evidence channels and their evaluation. Calibration improves in both primaries without losing balanced accuracy, a pre-specified success;
- a partial-pooling framework for household parameters, applied to the fitted channel parameters, with no evaluation yet;
- explicit online-filter and fixed-lag-smoother inference regimes. Every experiment record carries its regime, and every generated report states it.

It does not change the online pipeline's defaults, abstention thresholds, transition dynamics, the declared emissions, the behavioural ontology, or the frozen external-validation result.

### Added
- Added an explicit, bounded recent-history state to the generative model, `sensor_modeling.fusion.history`, to test whether the model needs one rather than relying on the filtered posterior. The model, its assumptions and the diagnostics are documented in `docs/HISTORY_STATE.md`.
  - **The model.** Each channel's activations over the three previous windows, the Phase 1 history depth, condition its sensors' current Poisson rates by `exp(β (log(1 + h) − E_s[log(1 + H)]))`. The expectation is the memoryless model's own, so observed silence is informative and missing history is neutral. There is one coefficient per state and room relation, eleven for the default ontology. A window's counts still enter the likelihood once, so no evidence is counted twice. With every coefficient zero the filter is exactly the original one, which is unchanged and remains the comparator.
  - **The filter.** `HistoryAwareBayesFilter` keeps a ring buffer of `k` windows. It treats cold starts, gaps, irregular updates and unreliable sensors as missing history, never as silence, and snapshots and restores its history with the belief.
  - **Diagnostics.** `explain()` returns a `PosteriorDecomposition` that splits each prediction exactly into prior and transitions, the current window, and recent history, channel by channel.
  - **Fitting and matched sets.** `fit_history_model` fits the coefficients from labelled training households only. `restricted_posteriors(history_model=...)` scores the model in `I2` and `I3`, exactly equal to the online filter fed the declared windows. `PoissonEventEmission.rates_per_second` exposes the rates its likelihood uses.

  No evaluation is reported, and inference and abstention defaults are unchanged.
- Added a hierarchical periodic state prior, `sensor_modeling.datasets.periodic_prior`, so the generative model can take part in matched comparisons that declare time of day. It follows the roadmap's hierarchical time-structure hypothesis (3.1). The model and its assumptions are documented in `docs/PERIODIC_STATE_PRIOR.md`.
  - **The model.** The prior probability of each state at each local hour is a softmax of state-specific Fourier log-probabilities: a population effect plus a household deviation. The deviation is the posterior mode under a Gaussian prior, so it is shrunk toward the population with a configurable, serialised precision. A household with no labelled time, or one never seen, gets exactly the population prior.
  - **Fitting.** `fit_periodic_prior` fits the population from the given households only. `PeriodicStatePrior.adapt` fits a household's deviation. `hour_state_counts` counts labelled time with the information sets' own reading of the local hour; its `until` argument keeps later labels out of an adaptation. Fitting is deterministic.
  - **In the filter.** The prior enters through the existing circadian term, with stickiness `π_h(s) / π(s)`. At every hour the chain's equilibrium is then exactly the prior, and its mean exit rate is unchanged. `StateOntology` gains `generator_at_hour` and `transition_at_hour`; `transition` behaves exactly as before.
  - **In the matched evaluation.** `restricted_posteriors` accepts a periodic prior in sets with time of day. It starts from the prior at the declared hour and reads nothing finer than that hour. `GapProtocol(periodic_prior=...)` adds the model `generative_periodic` in `I1` and `I3`, with each fold's prior fitted on that fold's training households only.

  The original generative model, abstention, the default experiment protocol and the published Phase 1 result are unchanged. No development-panel or held-out result is reported for the new model.
- Added the pre-specified Phase 3.1 experiment for the hierarchical time-of-day prior, `sensor_modeling.datasets.time_prior_experiment`, with its frozen protocol, `artifacts/phase3/time_prior_protocol.json`, committed before any household was scored. The design is described in `docs/PHASE3_TIME_PRIOR.md`.
  - **Models.** The original generative model and the hierarchical model with time disabled are scored in `I0` and `I2`. The hierarchical model with the hour, the diagnostic and logistic regression are scored in `I1` and `I3`.
  - **Estimands and criteria.** The estimands are fixed, primary and secondary. Their verdicts compare effect sizes and household bootstrap intervals with declared minimal important differences; they are not significance tests. A secondary arm adapts each held-out home's prior on its first 7 days and scores it only afterwards.
  - **Output.** The run writes a versioned record, and `time_prior_summary.render_summary` generates the Markdown summary from it. `scripts/run_phase3_time_prior.py` refuses to run unless the code's protocol equals the frozen file.
  - **Result.** The run is published in `artifacts/phase3/phase3-hierarchical-time-prior.json`, made from the protocol commit on a clean tree. It is on the 20 development homes, so it is not a held-out claim.
    - The hour is worth +0.131 [+0.120, +0.142] household balanced accuracy to the generative model, in all 20 homes: pre-specified success. Almost all of it comes from `away`, whose median recall rises from 0.006 to 0.824.
    - At `I1` the model is within +0.006 [−0.022, +0.030] of the diagnostic, but its log loss stays above the no-information reference.
    - Recent history still adds nothing to it.
    - Household adaptation improves log loss and calibration, but its balanced-accuracy gain is below the declared minimal difference: pre-specified inconclusive.
- Added the pre-specified Phase 3.2 experiment for the explicit history state, `sensor_modeling.datasets.history_experiment`, with its frozen protocol, `artifacts/phase3/history_protocol.json`, committed before any household was scored. The design is described in `docs/PHASE3_HISTORY_STATE.md`.
  - **Models.** The original generative model is scored in `I0` and `I2`, and the history-state model in `I2`. The Phase 3.1 time-prior model is scored in `I1` and `I3`, and the time-prior model with the history state, declared before any result, in `I3`. The diagnostic, logistic regression and the no-information reference are scored in every set.
  - **Estimands.** Formulation estimands compare two models on one set. Information estimands compare one model family across nested sets. No estimand changes both. H1, the history state against the original model in `I2`, is exactly the difference between the two models' history gains, so the earlier +0.021 is re-measured on the same homes rather than used as a threshold. A time-by-history interaction is reported for each model family.
  - **Output.** The run writes a versioned record, and `history_summary.render_summary` generates the Markdown summary from it. `scripts/run_phase3_history.py` refuses to run unless the code's protocol equals the frozen file.
  - **Result.** The run is published in `artifacts/phase3/phase3-explicit-history.json`, made from the protocol commit on a clean tree. It is on the 20 development homes, so it is not a held-out claim.
    - On identical information the history state lowers balanced accuracy by 0.042 [0.026, 0.060] in `I2`, and by 0.044 [0.028, 0.060] with the hour in `I3`. Both are pre-specified failures.
    - Recent history is worth −0.021 to the model with the history state, against +0.021 to the original model re-measured in the same run.
    - The loss is mostly `home_active` recall, down 0.164. Log loss improves by 0.726, but Brier score and calibration error do not.
    - The history gain is 0.013 to 0.021 smaller when the hour is known, in every model family.
- Added the pre-specified Phase 3.3 correlated-silence diagnostic, `sensor_modeling.datasets.silence_dependence`, with its frozen protocol, `artifacts/phase3/silence_protocol.json`, committed before any household was examined. It tests whether independent room-level Poisson silence likelihoods cause posterior over-concentration. It is diagnostic only: no inference, emission model or default changes. The design is described in `docs/PHASE3_CORRELATED_SILENCE.md`.
  - **Dependence.** Within each household and state, it compares joint silence on every channel with three independence baselines: observed marginals, Poisson streams and the declared rates. It also reports pairwise log odds ratios of silence, count correlations and the dispersion of the number of silent channels.
  - **Silence-evidence inflation.** Only the variation of joint silence across states can move a posterior. The inflation factor measures how much independence overstates that variation, from dependence alone and with the filter's declared terms.
  - **Concentration.** It measures the slope of overconfidence on the number of silent channels and along quiet runs, within the predicted state, where a calibrated model's slope is zero. Representative quiet periods are decomposed channel by channel.
  - **Rules.** Households are the unit of replication, with household bootstrap intervals. Joint-silence statistics are computed only where they are estimable. The conclusion rule is declared in advance.
  - **Checks.** On synthetic households, the tests show that the diagnostic reads conditionally independent streams as independent and strongly correlated silence as dependent. The decomposed recursion equals `MultimodalBayesFilter`.
  - **Output.** `scripts/run_phase3_silence.py` writes the record, a summary generated from it, and figures drawn from it. `scripts/plot_phase3_silence.py` redraws the figures from any record, and each SVG carries the SHA-256 of the data it plots.
  - **Result.** The run is published in `artifacts/phase3/phase3-correlated-silence.json`, made from the protocol commit on a clean tree, with its figures in `docs/figures/`. It is on the 20 development homes, so it is not a held-out claim. By the declared rule the hypothesis is weakened.
    - Silences are strongly dependent given the state. Independence overstates the spread of joint-silence evidence by 1.55 [1.38, 1.75], in 19 of 20 homes: material.
    - The filter's declared silence terms overstate it by 3.38, and the declared rates account for more of that than dependence.
    - Overconfidence falls by 0.059 per additional silent channel, in 16 of 20 homes, so the pre-specified consequence is not observed. Overconfidence rises by 0.175 per hour along quiet runs, and after one silent hour the model reports `sleeping` in 37 of 38 away and `home_inactive` runs.
- Added fitted observation models for the generative filter's evidence channels, `sensor_modeling.datasets.channel_models`, and their pre-specified experiment, `sensor_modeling.datasets.rates_experiment`. The frozen protocol, `artifacts/phase3/fitted_rates_protocol.json`, was committed before any household was scored. This is the route the Phase 3.3 diagnostic's 'weakened' result selects; the two-stage count-and-allocation model is not built. The likelihood and protocol are in `docs/PHASE3_FITTED_RATES.md`.
  - **The hurdle model.** Each channel gets a silence probability and a zero-truncated Poisson rate for active windows, per state. When the silence probability equals the Poisson's, it reduces exactly to a Poisson. `HurdleChannel.decompose` splits a window's log-likelihood into its silence and activity parts.
  - **The fitted Poisson.** The declared family with fitted means, kept to test whether silence needs its own parameter.
  - **Fitting.** Per fold, per channel and state, from the training households' labelled windows only. Each parameter is shrunk toward the declared model with 12 pseudo-windows. There is no household adaptation.
  - **The declared model is unchanged.** It stays available and is selected by family. `restricted_posteriors` now accepts any channel model with a `loglik` method; its results for the declared terms are identical.
  - **Estimands.** Fitted against declared channels on identical information, in `I0` to `I3` and in the filter's recursion over every window. There are mechanism checks on silence-evidence inflation and quiet-run accumulation. Calibration error is the primary outcome, and balanced accuracy is a guard, with a declared trade-off verdict.
  - **Result.** The run is published in `artifacts/phase3/phase3-fitted-rates.json`, made from the protocol commit on a clean tree. It is on the 20 development homes, so it is not a held-out claim.
    - Fitted hurdle channels are a pre-specified success in both primaries. With current windows, calibration error improves by 0.058 [0.030, 0.087] and log loss by 1.645, and balanced accuracy is unchanged. In the recursion, calibration error improves by 0.119 [0.087, 0.153] and balanced accuracy by 0.039 [0.015, 0.062].
    - With the time prior, median log loss is 1.448: the first generative model below the no-information 1.510.
    - Silence-evidence inflation falls from 3.38 to 1.62, near the dependence-only 1.55. Overconfidence per quiet hour falls from 0.175 to 0.023.
    - `home_active` recall falls by 0.23. A Poisson fitted to the mean count improves balanced accuracy but not calibration, and it makes silence inflation four times worse.
- Added a partial-pooling framework for household-specific parameters, `sensor_modeling.datasets.partial_pooling`, for roadmap item 3.4. The mathematics is in `docs/PARTIAL_POOLING.md`.
  - **The estimator.** A household's estimate is the population parameter plus its own deviation shrunk toward zero: `θ_pop + w (θ_raw − θ_pop)`, with `w = n / (n + κ)`. It is the posterior mean under a Beta or Gamma prior centred on the population, with `κ` pseudo-observations. The limits of `κ` are population-only and unconstrained per-household fitting.
  - **Its properties.** With no data it gives the population exactly. It is closed-form and deterministic. `PooledEstimate` reports the raw, pooled and population estimates and the effective shrinkage `κ / (n + κ)`, and round-trips through JSON.
  - **First application.** The fitted hurdle channel parameters, the silence probability and the active-window mean, whose silence differs widely between homes. `adapt_channels` pools a household toward a fitted population from its own labelled windows up to a declared moment only, and refuses a household the population was fitted on. An unseen household gets the population exactly. `HouseholdChannels` records its provenance (population digest, training households, strength, cut-off) and reports diagnostics per channel, state and parameter.
  - **Not changed.** No evaluation is reported, and no default or published result changes. The periodic state prior keeps its own household shrinkage. The default strength, 288 windows, is 24 labelled hours, as that prior declares.
- Added explicit inference regimes, `sensor_modeling.fusion.regime`, so online filtering and fixed-lag smoothing cannot be blurred by accident (roadmap 3.5). They are described in `docs/INFERENCE_REGIMES.md`.
  - **The regimes.** `InferenceMode` is `ONLINE_FILTER` or `FIXED_LAG_SMOOTHER`. `InferenceRegime` also carries a smoother's lag and window width, and derives a label. `regime_beliefs` turns filtered beliefs into what a regime reports, using the existing `smooth_beliefs`; neither inference algorithm changes. Zero-lag smoothing reports exactly the filtered estimate.
  - **Loud failures.** `require_online` refuses a smoother presented as online, and `require_same_regime` refuses to combine results from different regimes. `check_evidence_access` and `assert_respects_horizon` refuse evidence read beyond a regime's horizon.
  - **Experiment records, schema 1.2.** Every record carries `inference`: the mode, a smoother's lag and window, the label and its provenance. `ExperimentRecord` requires it as a keyword with no default, and a record whose label does not describe its regime is refused. 1.0 and 1.1 records migrate to the online filter, with provenance saying this was attested on migration: no experiment code ever called the smoother.
  - **Matched evaluation.** It is online by construction. It records and labels its regime, and refuses a smoothing regime or a model declaring one. Pooling folds and comparing information sets refuse runs from different regimes.
  - **Reports.** Every generated summary states its regime. The five published pages were regenerated from their unchanged records, adding only that line. The table in `docs/real_data.md` that set smoothed results beside online ones now labels each row's regime.

### Changed
- `ExperimentRecord` now requires an `inference` keyword, an `InferenceRegime`, and writes experiment-record schema 1.2. Code that builds records must declare the regime of its estimates. Records written at schema 1.0 and 1.1 are migrated when read, and their files are not changed.
- `restricted_posteriors` accepts any per-channel model with a `loglik` method. With the declared terms its results are identical.

## [0.7.0] - 2026-09-26

Completes the first Phase 1 measurement of the recoverable-information gap. It provides:

- a versioned, validated schema for experiment records, which migrates earlier records;
- a cyclic time-of-day encoding and interpretable recent-history summaries for matched experiments;
- a declared gradient-boosted supervised diagnostic;
- the generative filter's model restricted to the information sets it can consume, so it can be compared with supervised models on identical information;
- an exploratory run on the 20 development homes under frozen folds. It separates what added information is worth to a fixed model from what a formulation is worth on fixed information, and it records every comparison it could not make.

It does not change inference, abstention thresholds, transition dynamics, emissions, the behavioural ontology, or the frozen external-validation result.

### Added
- Added the Phase 1 recoverable-information-gap experiment, which separates what added information is worth to a fixed model from what a formulation is worth on fixed information. The protocol and design are in `docs/PHASE1_RECOVERABLE_GAP.md`.
  - `sensor_modeling.datasets.recoverable_gap` runs the four nested sets under frozen folds, with every setting fixed and no household used for tuning. It reports household-level metrics, per-state recall and calibration summaries with bootstrap intervals, and paired information gains, formulation gaps, interactions and comparisons with the production filter. The result is one experiment record. `gap_summary.render_summary` generates a Markdown summary from the written record alone.
  - `sensor_modeling.datasets.restricted_filter` restricts the generative filter's model to a declared set: its own stationary prior, transitions and emission models, fed only the set's windows. A test shows it equals the production `MultimodalBayesFilter` fed those windows. Sets with time of day are recorded as unsupported, because the current generative model has no time-of-day input. The production filter is scored as an unmatched reference.
  - `GradientBoostingBaseline` is the supervised diagnostic: gradient-boosted trees with fixed settings, kept out of `baseline_suite`.
  - `artifacts/phase1/household_splits.json` freezes the two Phase 1 folds, with each home's recording digest. `scripts/run_phase1_recoverable_gap.py` runs the experiment on the development panel.
  - The first run is published in `artifacts/phase1/phase1-recoverable-information-gap.json` and summarised in the documentation. It is exploratory, on the 20 development homes.
    - Given the same current and three previous windows, the diagnostic leads the generative model by +0.116 household balanced accuracy, in all 20 homes.
    - The generative model gains only +0.021 from that history; the diagnostic gains +0.088.
    - Information gains and formulation gaps interact, so the observed gap has no unique additive split.

  Inference, abstention and the existing baselines are unchanged.
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
- The Phase 1 artifacts now verify on any platform. On a Windows checkout, git rewrote `artifacts/phase1/household_splits.json` with CRLF line endings. Its raw-byte digest then no longer matched the one recorded in the published record, and a test failed there, while it passed in Linux CI. As for the v0.3 artifacts, `.gitattributes` now keeps `artifacts/phase1/*.json` at LF, and `load_frozen_splits` hashes the file after normalising its line endings. The recorded digest and the published result are unchanged.
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