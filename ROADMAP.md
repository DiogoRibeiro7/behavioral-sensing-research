# Project Roadmap

This roadmap describes the post-`0.5.0` direction of the Sensor Modeling
Research Toolkit.

The project is **research-first rather than release-first**. New software work
must be justified by a scientific question, a reproducible evaluation need, or
a stable public API requirement. A new numbered release is not a milestone by
itself.

The central objective for the next phase is stronger and more demanding than
incremental feature growth:

> Build and validate an ambient behavioural-sensing system that can outperform
> strong reproducible baselines under matched information, while remaining
> interpretable, calibrated, robust to missing or failed sensors, and credible
> across independently collected datasets.

Any claim of improvement must be earned by pre-specified comparisons and
held-out evidence. Headline accuracy alone is not sufficient.

## Current Stable Baseline

`0.5.0` was released on 2026-09-16. It is a platform and support-policy release:
Python 3.11--3.14 are supported, release automation is guarded, CI is split into
fast and slow paths, and manuscript-specific assets live outside the package
repository.

`0.5.0` does **not** change inference, abstention thresholds, transition
dynamics, emissions, the behavioural ontology, or the frozen external
validation result.

The current scientific evidence base is:

| Result | Current evidence |
| --- | ---: |
| Simulator balanced accuracy | 0.816 |
| 22-home real CASAS development panel | 0.420 |
| Supervised diagnostic ceiling on the same sensing problem | 0.607 |
| Frozen external candidate gain over `0.2.0` | +0.0091 |
| 95% household bootstrap interval | [+0.0054, +0.0117] |
| External homes improved | 37 / 43 |
| Frozen five-sensor non-inferiority gap | 0.1548 |
| Frozen eight-sensor gap to full deployment | 0.00529 |

The main lessons are now established:

1. The simulator is substantially easier than the corresponding real sensing
   problem and must not be treated as a field-performance estimate.
2. Time of day and recent room-resolved history contain useful information that
   the current online generative filter does not fully exploit.
3. Confidence is not a reliable proxy for correctness or information gain.
4. Quiet-period posterior saturation can be driven by compounded silence
   likelihoods even without activations.
5. Sensor reduction is a multi-objective frontier problem, not a search for one
   universal minimal kit.
6. Improvements must be judged under matched information sets; a model that
   sees more information has not demonstrated a modelling advantage merely by
   scoring higher.

## Completed Research Milestone — Paper 1

The uncertainty study is now frozen. Its pre-specified hypotheses all fail:

- confidence AUC: 0.5193;
- evidence-strength AUC: 0.3719;
- information-gain AUC: 0.3875;
- posterior-margin AUC: 0.5184;
- entropy-derived discrimination: 0.5135;
- 12/22 homes have confidence AUC above 0.5;
- information gain exceeds confidence in only 5/22 homes.

The conclusion is negative but useful: replacing one scalar uncertainty score
with another is not a credible repair. The next uncertainty work must change
the observation/inference formulation or use richer decision information.

No further experiments belong to Paper 1 unless they are required by peer
review.

## Phase 1 — Recoverable-Information Gap

### Scientific question

How much of the real-data performance gap is caused by sensing limitations, and
how much by the current probabilistic formulation?

### Required comparisons

Construct nested, matched information sets and evaluate all candidate models on
identical household splits. At minimum separate:

1. current instantaneous evidence;
2. current evidence + time of day;
3. current evidence + recent room-resolved history;
4. current evidence + time of day + recent history;
5. any richer observation representation proposed later.

The current generative filter and diagnostic supervised models must be compared
**within the same information set** wherever possible.

### Success criterion

Produce a reproducible decomposition of:

\[
\text{observed performance gap}
=
\text{information gap}
+
\text{formulation gap}
+
\text{residual uncertainty}.
\]

The exact decomposition need not be additive in a strict causal sense, but the
experimental design must make clear which information each model receives.

### First result

An exploratory first run on the 20 single-resident development homes is
recorded in `docs/PHASE1_MATCHED_BASELINES.md`. For the pre-declared linear
and tree baselines, time of day is worth +0.09 to +0.13 household balanced
accuracy and recent history +0.02 to +0.04. The filter is not yet in the
comparison, so the formulation gap remains to be measured.

### Boundary

The supervised diagnostic remains a measurement instrument, not a production
replacement and not evidence of clinical effectiveness.

## Phase 2 — Strong Baseline Benchmark

Before inventing a substantially new model, establish a benchmark suite that is
hard to beat for legitimate reasons.

### Baseline families

The benchmark should include, where scientifically appropriate:

- current probabilistic filter;
- simple persistence and majority/state-frequency baselines;
- regularised multinomial/logistic models;
- tree-based supervised baselines;
- sequence models only when they consume exactly the same permitted
  information;
- calibrated versions of discriminative baselines when calibration is being
  compared.

Deep learning should not be added merely because it is fashionable. It belongs
in the benchmark only if sample size, information structure, and evaluation
protocol make the comparison meaningful.

### Metrics

No single metric determines success. Report at least:

- balanced accuracy;
- per-state recall and confusion structure;
- Brier score;
- log loss;
- calibration error / calibration curves;
- abstention or selective-risk curves where applicable;
- household-level paired differences with uncertainty intervals;
- compute and latency when models are plausible for deployment.

### Evidence rule

Model selection and final evaluation must be separated. Candidate architectures,
hyperparameters and feature sets are chosen on development households only.
Final claims use frozen held-out households or an external dataset.

## Phase 3 — Inference Redesign

Only after Phases 1 and 2 should the core inference model be changed.

Priority hypotheses are:

### 3.1 Hierarchical time structure

Introduce time-of-day effects in a way that remains probabilistically explicit
and household-adaptable rather than hard-coding one global circadian schedule.
Candidate approaches include hierarchical periodic priors and partial pooling
across homes.

### 3.2 Explicit recent-history state

Represent recent event history directly rather than relying on the current
filter state to absorb all temporal structure. The representation must remain
interpretable enough to audit which historical evidence changed a posterior.

### 3.3 Correlated silence model

Replace the product of many independent room-level silence likelihoods only if
the data support the need. The leading hypothesis is a two-stage model:

\[
N_{\text{total}}(t)
\rightarrow
\text{conditional room allocation},
\]

rather than independent Poisson silence processes for strongly correlated room
streams.

This is intended to address over-concentration without arbitrary posterior caps.

### 3.4 Household adaptation

Separate population-level parameters from household-specific effects. Evaluate
partial pooling before introducing unconstrained per-home fitting.

### 3.5 Smoothing versus online inference

Keep fixed-lag smoothing and online filtering as separate operational regimes.
A gain obtained with future evidence must never be reported as an online gain.

## Phase 4 — Uncertainty and Selective Prediction Redesign

Paper 1 rules out a simple scalar-score swap as the main solution.

Future abstention work should therefore evaluate decision rules based on richer
structure, for example:

- posterior instability across plausible model specifications;
- disagreement between independent evidence channels;
- predictive checks for observation-model mismatch;
- out-of-distribution or low-support household states;
- expected loss under an explicit decision cost;
- ensemble/model uncertainty where it can be estimated honestly.

### Required evaluation

Selective prediction must be presented as a risk--coverage curve, not a single
hand-picked threshold. Thresholds used for final reporting must be chosen on
training/development data and frozen before held-out evaluation.

## Phase 5 — External Generalisation

CASAS provides multiple homes, but it remains one instrumentation ecosystem.
The next major credibility gate is an independently collected annotated dataset
with a materially different sensing layout.

### Required tests

1. Freeze preprocessing, ontology mapping and evaluation rules before seeing
   final external results.
2. Report what can and cannot be mapped across datasets.
3. Distinguish zero-shot transfer from any dataset-specific refitting.
4. Compare household-level performance distributions rather than only pooled
   event-level scores.
5. Document failure modes caused by different sensor semantics or coverage.

### Success criterion

A modelling improvement is substantially more convincing if its direction and
practical value survive outside the original CASAS development ecosystem.

## Phase 6 — Sensor-Information Frontier

The deployment study should move from isolated subset comparisons to a formal
multi-objective frontier.

For each pre-specified candidate sensor configuration, quantify:

- discrimination;
- calibration;
- robustness to missingness/failure;
- marginal information contribution;
- redundancy;
- cost or sensor count;
- computational consequences when relevant.

Candidate configurations must be selected before confirmatory scoring. Do not
mine the final evaluation set for the best subset.

## Phase 7 — Reliability and Failure Robustness

The current failure-aware weighting result shows that improving one metric can
harm others. Future work should therefore treat sensor reliability as a
statistical modelling problem rather than a heuristic multiplier.

Priorities:

1. estimate failure/missingness processes separately from behavioural state;
2. distinguish sensor absence, communication failure and genuine zero-event
   periods;
3. test informative missingness explicitly;
4. evaluate recovery after transient failures;
5. propagate reliability uncertainty into state uncertainty where feasible.

## Phase 8 — Real-Time System Hardening

Only after the inference improvements survive held-out and external evaluation
should the project expand its production surface.

Potential work includes:

- bounded-latency streaming inference;
- deterministic state/checkpoint recovery;
- schema-versioned event ingestion;
- drift and sensor-health monitoring;
- reproducible model/configuration snapshots;
- household-level audit trails explaining posterior updates;
- resource and latency benchmarks on realistic edge/server hardware.

Operational sophistication must not outrun scientific validity.

## Reproducibility Standard

Every major empirical result should satisfy the following whenever applicable:

1. pre-specified analysis contract;
2. frozen train/development/test or household split;
3. paired comparisons when predictions share the same observations;
4. uncertainty intervals at the household level;
5. Monte Carlo precision reporting for simulation studies;
6. deterministic seeds or recorded seed schedules;
7. machine-readable result artifacts;
8. exact package version / commit provenance;
9. explicit distinction between exploratory and confirmatory analyses;
10. negative results retained rather than silently discarded.

## Maintenance Backlog

Maintenance remains important but should not displace the scientific programme
unless it blocks reproducibility or supported users.

- Maintain Python 3.11--3.14 compatibility as the current supported range.
- Reduce pre-existing type-checking debt in older modules.
- Keep package, citation and Zenodo metadata synchronized.
- Keep release automation self-contained and tested.
- Keep documentation aligned with measured real-data limitations.
- Remove stale terminology that describes trusted event-stream silence as
  "absence of evidence"; under the current Poisson model, zero counts from
  working sensors are themselves evidence.
- Keep CI proportional to change scope.
- Keep manuscript-specific experiments and publication artifacts outside the
  public software package repository.

## Release Policy

`0.5.0` is the current stable release.

Future versions are created only when the research programme produces a
coherent user-facing software increment. Paper milestones do not automatically
require package releases.

Release notes come from the matching `CHANGELOG.md` section. The changelog is
the single source of truth for release notes.

## Immediate Order of Work

1. **Recoverable-information gap:** freeze matched information sets and quantify
   how much current performance is information-limited versus model-limited.
2. **Strong baseline benchmark:** establish pre-specified classical,
   probabilistic and selected sequence baselines under identical information
   and household splits.
3. **Inference redesign:** test hierarchical time structure, explicit recent
   history and a correlated-silence formulation one mechanism at a time.
4. **External generalisation:** test the frozen comparison protocol on an
   independently collected annotated sensing dataset.
5. **Uncertainty redesign:** evaluate structural/model-mismatch uncertainty and
   selective-risk curves rather than another scalar threshold swap.
6. **Sensor-information frontier:** formalise multi-objective deployment
   trade-offs and confirm selected configurations without post-hoc search.
7. **Reliability modelling:** separate failure processes from behavioural state
   and quantify robustness under realistic missingness.
8. **System hardening:** expand real-time operational capabilities only after
   the scientific improvements survive held-out and external evaluation.
