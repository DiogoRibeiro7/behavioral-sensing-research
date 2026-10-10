# Project Roadmap

This roadmap describes the research programme of the Sensor Modeling Research
Toolkit as it stands at `0.9.0`: what has been tested, what the tests showed,
and which questions remain open.

The project is **research-first rather than release-first**. New software work
must be justified by a scientific question, a reproducible evaluation need, or
a stable public API requirement. A new numbered release is not a milestone by
itself.

The central objective is stronger and more demanding than incremental feature
growth:

> Build and validate an ambient behavioural-sensing system that can outperform
> strong reproducible baselines under matched information, while remaining
> interpretable, calibrated, robust to missing or failed sensors, and credible
> across independently collected datasets.

Any claim of improvement must be earned by pre-specified comparisons and
held-out evidence. Headline accuracy alone is not sufficient.

## Where the Programme Stands

Five kinds of evidence appear below, and they are never merged:

- **Development panel.** The 20 single-resident CASAS homes, cross-fitted on
  two frozen household folds. Every Phase 1 to 4 result uses them. They have
  been inspected many times, so none of those results is a held-out claim.
- **Held-out CASAS.** The frozen v0.3 external test on 43 CASAS homes outside
  the development panel. It evaluated an older model, before Phase 1.
- **Independent external.** The Phase 5 evaluation on the two homes of the UCI
  ADL Binary dataset, collected independently of CASAS with a different
  sensing layout.
- **Clinical cohort, exploratory.** The online pipeline, with its declared
  emissions, baseline and alert policy, on the 56 homes of the TIHM dataset.
  Its labels are alerts a clinical team verified, not behavioural states, so
  it scores no state and describes only what the pipeline raises. The labels
  had been analysed before the protocol was written.
- **Simulated homes, pre-specified.** The silent-home rule on 100 paired
  homes from this repository's simulator, under a protocol frozen before any
  of them had been run with the rule on. A result on them is a statement
  about the simulator.

| Phase | Status | Evidence | Headline |
| --- | --- | --- | --- |
| 1. Recoverable-information gap | Evaluated, exploratory | Development panel | Part of the gap is the formulation, not only missing information |
| 2. Baseline benchmark | Partly complete | Development panel | Baselines and a supervised diagnostic are in place; no held-out benchmark |
| 3. Inference redesign | All five hypotheses evaluated | Development panel | The time prior and fitted channels succeed; the history state fails |
| 4. Uncertainty and selective prediction | First comparison evaluated | Development panel; external check inconclusive | Structural disagreement beats confidence; no abstention rule |
| 5. External generalisation | First evaluation complete | Independent external, 2 homes | The CASAS-trained model does not transfer |
| 6. Sensor-information frontier | Not started | — | — |
| 7. Reliability and failure robustness | One rule evaluated | Simulated homes, pre-specified; clinical cohort, described | An opt-in rule treats a home silent past a declared horizon as not observed; the TIHM homes are silent that long often |
| 8. Real-time system hardening | Deferred by its own gate | — | — |

In one sentence: on the development panel the generative model has improved
where a mechanism was tested one at a time, but no improvement has yet been
confirmed on held-out homes, and on the first independent dataset the model is
at chance.

## Current Stable Baseline

`0.9.0` completes the pre-specified Phase 3 evaluations and records the first
Phase 4 comparison of uncertainty diagnostics and the first Phase 5 external
evaluation. Phase 3 and 4 results are on the development panel:

- partial pooling of household channel parameters: a success with current
  windows, inconclusive in the filter's recursion;
- fixed-lag smoothing: a five-minute smoothing gain, available only after the
  smoother's delay and never an online gain;
- posterior predictive checks of the fitted hurdle model: the active count is
  under-dispersed, and silence comes in long runs;
- a hurdle negative-binomial channel model: better probabilities, not adopted
  by its declared rule;
- richer uncertainty diagnostics: model-structure disagreement ranks errors
  better than confidence by the declared rule, and no abstention rule is
  selected;
- the first external evaluation: the CASAS-trained model does not transfer to
  two independently collected homes.

It also enforces the inference-regime contract and adds the external-dataset
contract. It does not change the online pipeline's defaults, abstention, the
behavioural ontology or the frozen v0.3 external-validation result.

Earlier releases:

- `0.8.0` recorded the first Phase 3 results: a hierarchical time-of-day prior
  (a success), an explicit history state (a failure), a correlated-silence
  diagnostic (the hypothesis weakened) and fitted silence and activity rates (a
  success). It added the partial-pooling framework and explicit inference
  regimes.
- `0.7.0` recorded the first complete Phase 1 recoverable-information-gap
  result, with a versioned experiment-record schema, a cyclic time-of-day
  encoding, interpretable history summaries and the generative model restricted
  to the information sets it can consume.
- `0.6.0` added matched-information evaluation: information sets, the matched
  runner, four pre-declared baselines, household-level comparison and the first
  exploratory Phase 1 run.
- `0.5.0`, released on 2026-09-16, was a platform and support-policy release:
  Python 3.11--3.14, guarded release automation, fast and slow CI paths, and
  manuscript assets outside the package repository.

None of `0.5.0` to `0.9.0` changes the online pipeline's inference defaults,
abstention thresholds, transition dynamics, declared emissions, the behavioural
ontology or the frozen v0.3 external-validation result.

## The Evidence Base

Household balanced accuracy. The 22-home figures are medians; the 20-home
figures are household means with 95% household bootstrap intervals.

| Result | Value | Evidence |
| --- | ---: | --- |
| Simulator balanced accuracy | 0.816 | Simulator |
| Online pipeline with declared defaults, 22-home CASAS panel | 0.420 | Development |
| Supervised diagnostic on that panel | 0.607 | Development |
| Generative recursion, declared rates | 0.417 [0.392, 0.442] | Development, 20 homes |
| Generative recursion, fitted hurdle channels, the current reference | 0.456 [0.425, 0.484] | Development, 20 homes |
| Generative recursion, hurdle negative binomial, not adopted | 0.499 [0.470, 0.529] | Development, 20 homes |
| Generative model with the time prior, current window and hour (`I1`) | 0.509 [0.486, 0.531] | Development, 20 homes |
| Supervised diagnostic, current window, hour and history (`I3`) | 0.590 [0.550, 0.624] | Development, 20 homes |
| Frozen v0.3 candidate gain over `0.2.0` | +0.0091 [+0.0054, +0.0117], 37 / 43 homes | Held-out CASAS |
| Zero-shot transfer of the fitted hurdle recursion | 0.246 and 0.244, chance 0.25 | Independent external, 2 homes |
| In-sample oracle on those homes | 0.437 and 0.559 | Independent external, descriptive |
| Frozen five-sensor non-inferiority gap | 0.1548 | Simulator |
| Frozen eight-sensor gap to full deployment | 0.00529 | Simulator |

The rows use different information and different models, so they rank nothing
against each other. Matched comparisons are in the phase sections.

## Established Lessons

1. **The simulator is easier than the real problem.** It must not be treated as
   a field-performance estimate.
2. **Part of the real-data gap is the formulation.** On the development panel,
   a supervised classifier given strictly less information than the online
   filter, the current window and three previous ones, scores 0.081 higher
   balanced accuracy, in 19 of 20 homes.
3. **Time of day can be given to the generative model; recent history has not
   been.**
   - **The hour.** A hierarchical prior is worth +0.131 balanced accuracy.
   - **History.** An explicit history state lowered balanced accuracy by 0.042
     on identical information. With the hour known, the diagnostic still leads
     by 0.079 when it also sees history.
4. **Overstated silence evidence came mostly from the declared rates.** Fitting
   the channels cut silence-evidence inflation from 3.38 to 1.62, near the
   dependence-only 1.55. It cut overconfidence per quiet hour from 0.175 to
   0.023.
5. **Confidence is not a reliable proxy for correctness.** Model-structure
   disagreement ranks errors better on the development panel, but it retains
   less of the activity states.
6. **Comparisons must hold the information and the regime fixed.** A model
   that sees more has not shown a modelling advantage by scoring higher, and a
   smoothing gain is not an online gain.
7. **The development-panel model did not transfer.** On two independently
   collected homes it is at chance.
   - **Labels.** Label incompatibility is small: 96% to 98% of annotated time
     is scorable.
   - **Parameters.** The transferred parameters lose 0.19 and 0.32 of balanced
     accuracy against an in-sample oracle.
   - **The oracle.** Even the oracle reaches only 0.44 and 0.56.
8. **Sensor reduction is a multi-objective frontier problem.** It is not a
   search for one universal minimal kit.

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
with another is not a credible repair. The next uncertainty work had to change
the observation or inference formulation, or use richer decision information;
Phase 4 records it.

No further experiments belong to Paper 1 unless they are required by peer
review.

## Phase 1 — Recoverable-Information Gap

**Status: evaluated, exploratory, on the development panel.**

How much of the real-data performance gap is caused by sensing limitations, and
how much by the probabilistic formulation?

Every model is scored within nested, matched information sets on identical
household folds:

- `I0`: the current window;
- `I1`: the current window and the hour;
- `I2`: the current window and the three previous windows;
- `I3`: all three.

The runs are in `docs/PHASE1_MATCHED_BASELINES.md` and
`docs/PHASE1_RECOVERABLE_GAP.md`.

Measured:

- **Information.** For the pre-declared linear and tree baselines, time of day
  is worth +0.09 to +0.13 household balanced accuracy, and recent history +0.02
  to +0.04.
- **Formulation.** Given the same current window and three previous windows,
  the supervised diagnostic leads the generative model by +0.116, in all 20
  homes. The generative model gains +0.021 from that history; the diagnostic
  gains +0.088.
- **Against the production filter.** The diagnostic at `I2`, strictly less
  information than the filter, scores +0.081 [+0.052, +0.111] higher, in 19 of
  20 homes.
- **Not additive.** Information gains and formulation gaps interact, so the
  observed gap has no unique additive split.

Time of day was unsupported for the generative model in Phase 1. Phase 3.1
supplied it: at `I1` the gap closes to +0.006 [−0.022, +0.030].

**What remains.**

- **The history part of the formulation gap.** At `I3` the diagnostic still
  leads the generative model with the time prior by +0.079, and the Phase 3.2
  history state did not close it.
- **Confirmation.** No part of the decomposition is confirmed on held-out homes.

The supervised diagnostic remains a measurement instrument, not a production
replacement and not evidence of clinical effectiveness.

## Phase 2 — Strong Baseline Benchmark

**Status: partly complete.**

What exists: four pre-declared baselines with fixed settings, described in
`docs/BASELINES.md`:

- state frequency;
- persistence;
- regularised multinomial logistic regression;
- a depth-limited tree.

They sit alongside the gradient-boosted supervised diagnostic and the
generative model. All are scored by the matched runner, `docs/MATCHED_EVALUATION.md`,
with the following metrics:

- balanced accuracy and per-state recall;
- Brier score, log loss and calibration error;
- household-level paired differences with bootstrap intervals;
- selective-risk curves, through the Phase 4 framework.

Measured, on the development panel:

- **Balanced accuracy.** The diagnostic has the highest, 0.590 at `I3`.
- **Probabilities.** Logistic regression has the lowest log loss of any model
  scored.
  - **Identical information.** With current windows only, its median log loss
    is 1.242, against 1.351 for the best generative channel model, the hurdle
    negative binomial.
  - **With the hour.** At `I1` and `I3` its median log loss is 0.925 and 0.909.

**What remains.**

- **Held-out benchmark.** No benchmark comparison has been run on held-out
  households.
- **Sequence models.** None has been added. One belongs in the benchmark only
  if it consumes exactly the permitted information and the sample size and
  protocol make the comparison meaningful. Deep learning is not added because
  it is fashionable.
- **Compute and latency.** Not measured.

The evidence rule stands. Model selection and final evaluation are separated.
Candidate architectures, hyperparameters and feature sets are chosen on
development households only, and final claims use frozen held-out households or
an external dataset.

## Phase 3 — Inference Redesign

**Status: all five hypotheses evaluated on the development panel.** Each
evaluation was pre-specified, with its protocol frozen before any household was
scored. None is a held-out claim.

| Hypothesis | Result | Consequence |
| --- | --- | --- |
| 3.1 Hierarchical time structure | Success | The time prior is the generative model's hour term; it has not been evaluated in the recursion |
| 3.2 Explicit recent-history state | Failure | Not adopted |
| 3.3 Correlated silence | Weakened | Routed to fitted channel models, a success; the negative binomial is not adopted |
| 3.4 Household adaptation | Success with current windows; inconclusive in the recursion | The declared pooling strength is too strong |
| 3.5 Smoothing versus online inference | Contract enforced; smoothing gain measured | No smoothing result is reported as online |

The current reference formulation is the online filter's recursion with fitted
hurdle-Poisson channels and population parameters. Phases 4 and 5 use it.

### 3.1 Hierarchical time structure

A hierarchical periodic state prior is implemented; see
`docs/PERIODIC_STATE_PRIOR.md`. Its evaluation is in `docs/PHASE3_TIME_PRIOR.md`.

- **The hour.** It is worth +0.131 [+0.120, +0.142] household balanced
  accuracy to the generative model, in all 20 homes: a pre-specified success.
- **Where the gain comes from.** Almost entirely `away`: median recall rises
  from 0.006 to 0.824.
- **Against the diagnostic at `I1`.** Within +0.006 [−0.022, +0.030], but its
  probabilities stay poor: median log loss 2.96, against 1.51 with no
  information.
- **Recent history.** It adds nothing (+0.002). At `I3` the diagnostic still
  leads by +0.079.
- **Household adaptation.** On a 7-day window it improves log loss and
  calibration. The balanced-accuracy gain, +0.011, is below the declared
  minimal difference: inconclusive.

### 3.2 Explicit recent-history state

An explicit history state is implemented; see `docs/HISTORY_STATE.md`. Each
channel's activations over the three previous windows condition its current
rate. Its evaluation is in `docs/PHASE3_HISTORY_STATE.md`.

- **On identical information.** It lowers balanced accuracy by 0.042
  [0.026, 0.060] in `I2`, and by 0.044 [0.028, 0.060] in `I3` with the hour. At
  most 3 of 20 homes improve. Both are pre-specified failures.
- **Recent history.** It is worth −0.021 to the model with the history state,
  against +0.021 to the original model re-measured in the same run.
- **Where the loss is.** Mostly `home_active`, whose recall falls by 0.164.
- **Probabilities.** Log loss improves by 0.726, but Brier score and
  calibration error do not.

Consequence: the history state is not adopted, and the history part of the
formulation gap remains open. Whether its coefficients also absorb errors in
the declared rates was not separated.

### 3.3 Correlated silence and the channel observation model

The pre-specified diagnostic, `docs/PHASE3_CORRELATED_SILENCE.md`, **weakened**
the correlated-silence hypothesis.

- **Dependence is real.** Independence overstates the spread of joint-silence
  evidence across states by 1.55 [1.38, 1.75], in 19 of 20 homes.
- **The declared rates matter more.** The filter's own silence terms overstate
  it by 3.38. On the log scale dependence accounts for 0.435 of that, and the
  declared rates for 0.784.
- **The predicted consequence is absent.** Overconfidence falls, by 0.059 per
  silent channel. It grows along quiet runs, by 0.175 per hour.

The result routed the work to fitting each channel's marginal observation
model, keeping channels independent: a hurdle model per channel and state,
`docs/PHASE3_FITTED_RATES.md`.

- **Pre-specified success in both primaries.**
  - **Current windows.** Calibration error improves by 0.058, and balanced
    accuracy is unchanged.
  - **The recursion.** Calibration error improves by 0.119 and balanced
    accuracy by 0.039.
- **The mechanism.** Silence-evidence inflation falls from 3.38 to 1.62.
  Overconfidence per quiet hour falls from 0.175 to 0.023.
- **The cost.** `home_active` recall falls by 0.23.

Posterior predictive checks located the misspecification;
see `docs/PHASE3_PREDICTIVE_CHECKS.md`.

- **The active count is under-dispersed.** Observed variance is 4.3 to 7.4
  times the model's in the common states, in every household.
- **Silence comes in long runs.** They are 1.3 to 5.2 times as common as the
  model allows in `away`, `home_active` and `home_inactive`.
- **The declared routing.** It names within-state temporal dependence as the
  next model family: activity sub-states or a Markov-modulated emission.

A zero-truncated negative-binomial active count was then evaluated against the
hurdle-Poisson; see `docs/PHASE3_NEGATIVE_BINOMIAL.md`.

- **Probabilities.** Log loss improves by 0.48 with current windows and 0.78 in
  the recursion, in all 20 homes.
- **Balanced accuracy.** It improves by 0.044.
- **`home_active` recall is not recovered.** With current windows it falls a
  further 0.055.
- **Not adopted.** The rule declared in advance needed both settings, and with
  current windows the result is a trade-off.

Consequence: the fitted hurdle-Poisson channels are the reference. Within-state
temporal dependence is named but not built. The negative binomial addresses the
dispersion but not the runs.

### 3.4 Household adaptation

A partial-pooling framework is implemented; see `docs/PARTIAL_POOLING.md`. Its
evaluation is in `docs/PHASE3_PARTIAL_POOLING.md`.

- **Pooling helps with current windows.** After a week of household data it
  improves log loss by 0.080 [0.039, 0.118], in 17 of 20 homes: a
  pre-specified success. Balanced accuracy and calibration are unchanged.
- **In the recursion it is inconclusive.** One home improves greatly. Without
  it the mean change is −0.032, and four homes worsen by more than 0.4.
- **Unconstrained per-home fitting overfits.** After one day it is worse than
  pooling by 0.175.
- **The strength.** Selected on training homes only, it was the grid's
  smallest, 24 windows, in both folds. The declared 288 pools more than a week
  of data needs.

Consequence: pooling is supported with current windows only, and a strength
selected on training homes has not been evaluated. The Phase 5 adaptation used
the declared 288, as its protocol froze.

### 3.5 Smoothing versus online inference

The regime contract is enforced in code; see `docs/INFERENCE_REGIMES.md`.

- **Every experiment record states its regime.** It also records whether the
  regime is causal, its reporting delay, and the timestamps of its estimates. A
  record whose estimates read past its regime is refused.
- **Reported online estimates are never revised.** Streaming tests show it.
- **Regimes are never mixed silently.** A smoother against the online filter
  must be requested as a smoothing gain, labelled with its delay.

The smoothing evaluation is in `docs/PHASE3_SMOOTHING.md`. Every figure is a
smoothing gain, available only after the smoother's delay; none is an online
improvement.

- **Five minutes of lag.** Balanced accuracy rises by 0.031 [0.021, 0.043], in
  all 20 homes: a pre-specified gain.
- **Thirty and sixty minutes.** Balanced accuracy rises by 0.043 and 0.049,
  and log loss worsens by 0.101 and 0.119: trade-offs.
- **Most changes are not corrections.** Only 37% of the states smoothing
  changes at five minutes are corrections.
- **No smoother reports a new state sooner** than the online filter. Smoothing
  is no help to alerting or to any decision made when a state changes.

### What remains in Phase 3

- **Combining the two successes.** The time prior and the fitted channels have
  not been evaluated together in the recursion.
- **Within-state temporal dependence.** The family the predictive checks route
  to has not been built.
- **`home_active` recall.** The fitted channels lose 0.23 of it, and the
  negative binomial does not recover it.
- **Confirmation.** Every Phase 3 conclusion is on the development panel.

## Phase 4 — Uncertainty and Selective Prediction Redesign

**Status: diagnostics implemented; the first comparison evaluated on the
development panel; no abstention rule selected.**

Paper 1 ruled out a scalar-score swap. Four pieces of infrastructure are
implemented. Each is diagnostic only and changes no decision:

- **Model-structure disagreement.** The posteriors of fitted specifications
  that published results support, compared window by window;
  `docs/STRUCTURAL_DISAGREEMENT.md`.
- **Evidence-group disagreement.** What the motion, contact, bed and wearable
  groups each support within a filter update. Missing groups are never counted
  as disagreement; `docs/EVIDENCE_GROUP_DISAGREEMENT.md`.
- **Observation-model mismatch.** How surprising a window's evidence is under
  every state. Known sensor failures are excluded, never scored as novelty;
  `docs/OBSERVATION_MISMATCH.md`.
- **Selective-prediction evaluation.** Risk-coverage curves for any signal
  with its direction stated, against random and oracle rejection. Households,
  never timestamps, are resampled; `docs/SELECTIVE_PREDICTION.md`.

### Measured: the first comparison of the diagnostics

The protocol was frozen in `ad459f1`; see
`docs/PHASE4_UNCERTAINTY_DIAGNOSTICS.md`. Every signal ranks the same
cross-fitted hurdle-recursion predictions on the development panel, whose
full-coverage error is 0.487.

- **Confidence barely orders the predictions.**
  - **Pooled gain.** Over random rejection it is 0.04 [−0.22, 0.31].
  - **Its most confident windows.** The most confident 10% have an error of
    0.66.
  - **Entropy.** It is indistinguishable from confidence.
- **Model-structure disagreement is materially better by the declared rule.**
  - **AURC.** Household error AURC is 0.059 [0.031, 0.087] lower than
    confidence's, lower in 16 of 20 homes.
  - **Calibration.** Its retained predictions are better calibrated.
- **Its gain is at low coverage.** At 50%, 70% and 90% coverage the paired
  differences are uncertain or negligible.
- **The minority-state guard was weak.**
  - **What it admitted.** The frozen rule admitted only `bed_awake`.
  - **Activity states.** At 50% coverage structural disagreement keeps 0.85
    and 0.87 of the proportional share of `kitchen_activity` and
    `bathroom_activity`, where confidence keeps 1.46 and 1.16.
- **Evidence-channel disagreement and predictive mismatch are worse than
  confidence.** Their household AURC is 0.149 and 0.040 higher, and both reject
  activity states heavily.
- **No threshold was selected**, and no abstention rule.

The external check, the Phase 5 estimand S, is **inconclusive**. Structural
disagreement beats confidence in one external home, by 0.249 [0.163, 0.340],
and is uncertain in the other.

### What remains in Phase 4

- **An abstention rule.** None has been selected. A rule must be chosen on
  development data with a guard that covers the activity states and `away`,
  and frozen before held-out evaluation.
- **Other decision information.** Expected loss under an explicit decision cost
  has not been evaluated, and neither have out-of-distribution or low-support
  states as abstention signals.

Selective prediction is presented as a risk--coverage curve, not a single
hand-picked threshold. Thresholds used for final reporting are chosen on
development data and frozen before held-out evaluation.

## Phase 5 — External Generalisation

**Status: the infrastructure and the first external evaluation are complete.
The result is negative.**

CASAS is one instrumentation ecosystem. Phase 5 tests the model on an
independently collected annotated dataset with a different sensing layout.

- **The contract.** `docs/EXTERNAL_DATASET_CONTRACT.md`. Adapters expose a
  dataset in its own terms. Labels, sensor types and locations map to the
  ontology only through declared entries: exact, approximate, unmappable or
  ambiguous. Anything undeclared is an error, and conversion counts everything
  it cannot carry.
- **The protocol.** `docs/PHASE5_EXTERNAL_PROTOCOL.md`, frozen in `a863bff`
  before any external scoring.
  - **Dataset.** The UCI ADL Binary dataset (DOI 10.24432/C5J02M, CC BY 4.0):
    two single-resident homes in Spain.
  - **Mapping.** Meal labels are ambiguous and unscored, so `home_active`,
    `kitchen_activity` and `bed_awake` cannot be scored.
  - **Conditions.** Zero-shot transfer of the fitted hurdle recursion, and
    limited adaptation by partial pooling on each home's first seven days.
- **The run.** `docs/PHASE5_EXTERNAL_RESULTS.md`, made from the clean commit
  `2d495e4`. Every CASAS population reproduced its frozen digest, and both
  homes are reported.

### Measured: the first external evaluation

- **Zero-shot is at chance.** Balanced accuracy is 0.246 [0.244, 0.249] and
  0.244 [0.242, 0.246], against 0.25. Every window is predicted `sleeping`
  (92.3% and 89.1%) or the unscorable `bed_awake`.
- **Transfer: it does not transfer.**
  - **OrdonezA.** The balanced-accuracy difference from the declared rates is
    negligible.
  - **OrdonezB.** The declared rates are better on every metric, by 0.098
    [0.086, 0.113] in balanced accuracy.
- **Adaptation: it does not help.** Balanced accuracy is unchanged, −0.001 and
  +0.000.
  - **Log loss.** It improves by 0.396 and 0.422.
  - **Recall.** No state's recall changes by more than 0.01.
- **The Phase 4 direction: inconclusive.** Structural disagreement beats
  confidence in OrdonezB and is uncertain in OrdonezA.
- **The scored periods.** They cover 8 and 17 local dates, where the
  protocol's text expected 7 and 14 days; its definition is what was run.

Descriptive diagnostics, which change no conclusion:

- **Dataset incompatibility.**
  - **Labels.** It is small for labels: 97.6% and 96.0% of annotated time is
    scorable.
  - **Sensors.** 65.7% and 15.7% of sensor activations come from sensors that
    feed no model channel.
- **Event rates differ.**
  - **Active windows.** They carry 1.0 to 2.1 events against 2.1 to 14.0 in
    CASAS.
  - **Silence.** It is at least as frequent in every channel and state.
- **Calibration shift.** Confidence exceeds accuracy by 0.410 and 0.453,
  against 0.299 on the development panel.
- **Separating the losses.** An in-sample oracle, the same recursion with each
  home's own parameters, reaches 0.437 and 0.559.
  - **The transferred parameters.** They lose 0.191 and 0.315 of balanced
    accuracy against it.
  - **The oracle's shortfall.** It mixes the sensing limitation with the model
    family's.

Consequence: no Phase 3 or Phase 4 result is supported beyond CASAS. The
CASAS-fitted observation model is not portable to this sensing layout, and a
week of adaptation at the declared strength does not change a decision.

### Measured: the alert layer on a clinical cohort

**Status: exploratory. The result is negative.**

The TIHM dataset (DOI 10.5281/zenodo.7622128, CC BY 4.0) holds 56 homes of
people living with dementia, with alerts a clinical monitoring team verified.
Through the contract every label is unmappable, so no state is scored. The
online pipeline was run with its declared emissions, baseline and alert
policy, and with nothing fitted. Its step is the protocol's ten minutes, the
one the CASAS evaluation used; the configuration's own default is five.

- **The protocol.** `docs/TIHM_ALERT_BURDEN_PROTOCOL.md`, committed in
  `b15126f` before any pipeline output was compared with a label. The labels
  had been analysed before it was written, and it says so.
- **The run.** `docs/TIHM_ALERT_BURDEN_RESULTS.md`, made from that clean
  commit. Every household is reported.
- **Alert burden.** 0.064 [0.046, 0.084] behavioural alerts per monitored
  person-day; for `sleeping_hours` alone 0.055 [0.038, 0.073], against the
  simulator's 0.010 in its stable arm and 0.033 in its changed arm.
- **No positive relation to the verified labels.** Alert days are 5.3% of
  agitation label days and 8.5% of other evaluable days, a difference of −3.2
  points [−7.9, +1.9]. The deviation score's within-household concordance with
  label days is 0.466 [0.410, 0.530]. Both intervals include no relation; for
  labels of any type the differences are below zero.
- **Two rules without a model do better.** At the pipeline's 507 flagged days
  it catches 18 of 94 label days, where 23 are expected at random. The
  household's own label history catches 63, and its sensor event count against
  earlier days 53.

Descriptions made after the run was read, in a record of their own made from
the clean commit `bc2ea4e`. They change no result:

- **Silent days are read as sleep.** On 128 of 2,850 monitored days no sensor
  of the home reported an event. The pipeline counts all 128 as usable and
  infers a median 23.8 hours of sleep on them. On 16 June 2019 all 47
  monitored homes are silent. The dataset paper reports a failure of the data
  collection server and dates the drop to 14 June; in the released file the
  drop is on 15 to 17 June. That they are one event is an inference. The next
  day holds 33 of the 183 alerts.
- **Leaving them out.** Replaying the baseline with silent days left out of
  the history gives 78 change verdicts, where the run gave 188. 44 of the 188
  are on silent days themselves.
- **The deviation threshold is not three standard deviations.** A
  same-weekday reference is used from four days. Against a median and MAD of
  four others, a Gaussian value reaches the threshold about 17.5% of the time,
  where a known mean and standard deviation give 0.27%. Stationary Gaussian
  values with each household's own centre and spread make 14.9% of days
  deviate; the run has 24.4%.
- **Opposite directions.** The pipeline's deviating days are quiet days, with
  a mean event-count score of −1.17. Label days are busy ones, at +0.86.

Consequence: the simulator's alert burden is not an estimate for a real
cohort. On this one, more than half of the change verdicts go when days
without data are left out, the share of deviating days is mostly what the
reference's estimation gives on values with no change over time, and nothing
the pipeline raises is more frequent on the days a clinical team confirmed.
The dataset carries no signal that a sensor or its gateway was working, which
is the case Phase 7 exists for.

### Measured: a threshold that means what it says

`BaselineConfig.calibrated` is an opt-in reference for the second failure the
TIHM run showed, the threshold that a reference of a few days passes far more
often than it states. Its centre stays weekday-aware; its scale is pooled over
every retained day, from how far each fell from the centre the other days of
its weekday gave it; the deviation is mapped through a Student t; and its
trend is fitted to those same distances, so a weekly rhythm is not read as a
drift. It is off by default.

It was built from the TIHM run, so that run cannot be its evidence.

- **On synthetic days.** `docs/THRESHOLD_CALIBRATION_NULL.md`, independent
  Gaussian days given to the baseline directly. The default passes its
  threshold of 3 on 6.24% of days, 23 times what it states; the calibrated
  reference on 0.28%.
- **The protocol.** `docs/THRESHOLD_CALIBRATION_PROTOCOL.md`, frozen and
  pushed before any simulated home had been run with the calibrated
  reference, after three reviews. The second found that reading two
  multiples off a quarter of the homes would call a reference with the
  default's own curve better, or worse, in about one run in ten; the third
  found a grid of 13 multiples too coarse to draw a straight line across,
  and the grid was made three times finer. It fixes one comparison: where the calibrated
  reference's operating curve, over 42 multiples of the thresholds, has the
  false alerts of the default as it ships, how much of a step change it
  finds net of chance. The match is placed again in every resample of homes.
- **The run.** `docs/THRESHOLD_CALIBRATION_RESULTS.md`, from a clean commit,
  on 400 paired simulated homes. **The evidence is simulated.**
- **At the same false alerts it finds more of the step.** 0.72 of homes net
  of false detections against 0.63, +0.09 [+0.03, +0.16], Monte Carlo
  standard error 0.03, at the default's 0.855 false alerts a home in 84 days.
  For the smaller step and the gradual change no difference is shown, −0.02
  [−0.09, +0.04] and −0.01 [−0.10, +0.09].
- **Its threshold means what it says for hours of sleep on these homes.**
  0.35% of stable days of sleep pass 3, 4.79% pass 2 and 12.75% pass 1.5,
  each within a quarter of a standard deviation of the threshold it states;
  the default passes 3 on 7.91%. On hours away, where no criterion was
  stated, it passes 3 on 1.66% of days, as often as a Gaussian value passes
  2.40.
- **It is not a drop-in replacement.** At the declared thresholds it raises 3
  false alerts in the 400 homes against the default's 342, and finds the step
  net of chance in 0.04 of them against 0.63. Its thresholds have to be set
  lower, about 0.59 times the declared ones on these homes, and where that is
  on a real home is not known.

On TIHM, as a description and not a test: with the silent-home rule off the
calibrated reference passes 3 on 8.1% of evaluable days of sleep, against the
default's 17.4%, and with the rule on at 12 hours on 1.7%, against 10.9%. A
real day is far from the independent Gaussian draw its score assumes, so its
threshold says less there than on the simulated homes.

Consequence: in simulation the calibrated reference keeps its threshold's
meaning and, at the default's false alerts, finds more of a step change; it
shows no gain for a smaller or a slower change. The default stays the default.
A change of default needs evidence from real homes on where to set its
thresholds, which this run cannot give.

### Measured: the pipeline's hours of sleep beside a sleep mat

`sleeping_hours`, the feature every alerting study counts its detections on,
had never been set beside a measurement of sleep. In 17 TIHM homes a mat under
the mattress recorded each minute in bed, with a sleep stage computed by the
device.

- **The protocol.** `docs/SLEEP_MAT_PROTOCOL.md`, frozen and pushed before any
  value of the pipeline was set beside any record of the mat, after two
  reviews, which moved every day touched by a silence of the whole home of
  12 hours or more out of the primary analysis: 16 June, when every monitored home was silent, is a
  mat-observed day in 11 of the 14 homes, and the partial days around it are
  too.
- **The run.** From clean commit `c42d69a`, with both runs reproducing the
  published records home by home; `docs/SLEEP_MAT_RESULTS.md` is generated from
  its records. A measurement on a real cohort; nothing was fitted.
- **Within a home it does not follow the mat.** A mean Spearman correlation of
  0.11 [0.03, 0.19] over 14 homes, against a margin of 0.5; against the hours
  in bed, 0.14 [0.01, 0.27]. The deviations a personal baseline gives the two
  correlate at 0.03 [−0.04, 0.11], over 12 homes.
- **In level it does not agree.** It gives 3.95 hours a day more sleep than the
  mat [+1.54, +6.37], and 1.97 more than its hours in bed [−0.01, +3.94].
- **Leaving out the mat's odd homes, or moving its clock an hour, does not
  change the reading.** Without the three homes in which the mat stages less
  than 60% of the time in bed as asleep, 0.07 [−0.01, 0.16]. With the mat's
  clock an hour earlier, 0.10 [0.02, 0.17]; an hour later, 0.13 [0.04, 0.21],
  over 13 homes. Hourly activity is most opposed to time in bed with the
  mat's clock an hour later, which would fit a mat clock in UTC.
- **In simulation it does follow.** Against the simulator's true hours of
  sleep, 0.70 [0.68, 0.71], with a bias of −0.23 hours. **That evidence is
  simulated.**

Consequence: on this evidence, `sleeping_hours`, which follows sleep in the
simulator, does not follow a sleep mat in these homes; the disagreement may
be the pipeline's or the mat's. A detection counted on that feature in
simulation is a statement about the simulator's sleep. The mat is not
validated and these are homes of people living with dementia, so what is
missing is a sleep reference in homes where the pipeline is meant to run,
before an alert about sleep is read as one about a person's sleep. TIHM is by
Palermo et al., *Scientific Data* 10, 606 (2023), under CC BY 4.0; Surrey and
Borders Partnership NHS Foundation Trust and Howz are acknowledged, as the
dataset asks.

### Measured: the simulator's homes with TIHM's sensors

Until this study every detection in simulation came from event sensors
cleaner than TIHM's. Two activations of one of TIHM's motion sensors are never
less than 61 seconds apart, 54 of its 56 homes have hallway records, its
contacts log an opening twice, and 63% of its motion activations follow
another room's, against 7% in the simulator.

- **The protocol.** `docs/MATCHED_SENSORS_PROTOCOL.md`, frozen before any of
  the study's homes was run with the matched profile, after two reviews. The profile's
  hold-off is TIHM's; its presence scale and spill-over rate are the grid
  point whose eight moments of the sensor records were nearest TIHM's, in a
  planning record that runs no pipeline. The second-nearest point is run on
  100 homes as a sensitivity profile.
- **The run.** From clean commit `e7d41db`, with the standard profile's runs
  reproducing the published threshold-calibration alerts home by home;
  `docs/MATCHED_SENSORS_RESULTS.md` is generated from its record. Nothing in
  the pipeline was changed or fitted.
- **The detection survives (C1).** Excess detection 0.70 against 0.63,
  +0.07 [+0.01, +0.13], against a margin of −0.10. False alerts 1.02 a home
  against 0.86. The sensitivity profile gives +0.14 [+0.02, +0.26].
- **The hours of sleep still follow the truth (C2).** 0.71 [0.70, 0.71],
  against 0.67 with the simulator's own sensors and a margin of 0.5.
- **The kitchen and bathroom collapse is not reproduced (C3).** The bathroom's
  pooled-day median falls to 0.01 hours against a true 0.28, under its
  ceiling of 0.05; the kitchen's to 0.68 against a true 1.96, over its
  ceiling of 0.25. After steps with only the kitchen's motion activations,
  belief in kitchen activity is 0.28 against 0.71.
- **The profile is nearer TIHM's records than the simulator's own, not on
  them.** 158 living-room activations a day against TIHM's 77, 50 hallway
  activations against 67, and 57% of motion activations after another room's
  against 63%. **That evidence is
  simulated.**

Consequence: in simulation the detection does not rest on the sensors
lacking the properties the profile matches. In the simulator those properties
do not make the hours of sleep stop following the truth, so they do not by
themselves bring it near TIHM's 0.11 against the sleep mat; the gap may lie in
what the profile does not model, in the homes' days or in the mat. In the
simulator they do take the bathroom's hours to TIHM's level, and the
kitchen's to about a third of the truth.

### Described: where the pipeline's sleep parts from the sleep mat

The sleep-mat comparison said that `sleeping_hours` does not follow the mat
within a home; the matched-sensor study said the sensor properties its matched
profile reproduces do not explain it in the simulator. Neither said where in a
day the two part.

- **The plan.** `docs/SLEEP_GAP_PLAN.md`, frozen and pushed before any value
  was computed, after an independent review of the draft, which the plan
  records, and a second of its revision, which it does not. It sets each
  step's belief beside the mat's class of each minute, under the mat's clock
  as recorded and an hour later, on the sleep-mat comparison's primary days
  and homes. Exploratory: no criterion, no margin.
- **The run.** From clean commit `1e46865`; both runs and the matched days
  reproduced the published records, the repeat the first run's daily hours,
  and the mat's minutes its daily hours. `docs/SLEEP_GAP_RESULTS.md` is
  generated from its record.
- **The extra sleep.** Of the 3.95 hours a day, 4.10 are counted when the mat
  has no record, 3.20 of them by day; 1.34 while the mat says awake in bed;
  and 1.48 hours of the mat's sleep are not counted.
- **A quiet home is read as a sleeping one.** With no mat record the belief in
  sleep is 0.90 one to three hours after the last activation and 0.99 after
  three, and 0.39 between ten minutes and an hour; 2.29 of the 4.10 hours fall
  within the hour. 1.40 hours a day follow the bedroom's sensor and 1.07 an
  exit door.
- **That part moves the daily number.** It carries 0.68 [0.54, 0.83] of the
  daily variance and follows the mat no better than a swapped-mask reference;
  the sleep counted on the mat follows it by +0.10 [+0.05, +0.15] beyond the
  reference, above zero in all 14 homes, and by +0.18 [+0.09, +0.26], above
  zero in 12, with the mat's clock an hour later.

Consequence: in these homes the pipeline's daily hours of sleep are consistent
with a measure of how quiet the day was, more than of the night the mat
records: the part counted with no mat record moves them and does not follow
the mat. They correlate at −0.71 with the day's activations, which the way the
pipeline infers sleep partly builds in. Telling rest by day, absence and sleep
apart may need what these event sensors do not give, such as a door's
direction or a bed sensor, or a model that gives more weight to the resident
being out after a silence that follows an exit door; that is a conjecture, and
any such change would need its own frozen test on homes other than these. TIHM
is by Palermo et al., *Scientific Data* 10, 606 (2023), under CC BY 4.0;
Surrey and Borders Partnership NHS Foundation Trust and Howz are acknowledged,
as the dataset asks.

### What remains in Phase 5

- **More independent evidence.** Two homes are two case studies. A
  confirmatory claim needs an independent dataset with more homes.
- **The Ordóñez homes are now inspected.** Any further analysis of them,
  including another adaptation strength, is exploratory.
- **A reference for sleep.** `sleeping_hours` does not follow a sleep mat in
  the 14 TIHM homes with enough days on one. The alerting studies' detections are counted
  on it, so a real-home sleep reference is needed before they say anything
  about a person.

## Phase 6 — Sensor-Information Frontier

**Status: not started.** The simulator ablation framework and the frozen five-
and eight-sensor simulator results predate the programme.

The deployment study should move from isolated subset comparisons to a formal
multi-objective frontier. For each pre-specified candidate sensor configuration,
quantify:

- discrimination;
- calibration;
- robustness to missingness/failure;
- marginal information contribution;
- redundancy;
- cost or sensor count;
- computational consequences when relevant.

Candidate configurations must be selected before confirmatory scoring. Do not
mine the final evaluation set for the best subset. A frontier depends on the
model scored, so it follows a confirmed reference formulation.

## Phase 7 — Reliability and Failure Robustness

**Status: one rule evaluated; not started as statistical modelling.** Related
infrastructure exists:

- online sensor health tempers each sensor's likelihood;
- the observation-mismatch diagnostics exclude known failures rather than
  scoring them as novelty;
- evidence-group disagreement treats unavailable sensors as missing.

The failure-aware weighting result showed that improving one metric can harm
others, so sensor reliability should be treated as a statistical modelling
problem rather than a heuristic multiplier.

The TIHM run is a measured case. Its event-only streams give the health layer
nothing to judge: it raised no system-health alert in 2,850 monitored days,
and counted as usable every one of the 128 days on which no sensor reported,
a day on which no home reports among them. See the Phase 5 section. Priorities:

1. estimate failure/missingness processes separately from behavioural state;
2. distinguish sensor absence, communication failure and genuine zero-event
   periods;
3. test informative missingness explicitly;
4. evaluate recovery after transient failures;
5. propagate reliability uncertainty into state uncertainty where feasible.

### Measured: a home that reports nothing

`HealthConfig.home_silence_horizon` is an opt-in rule for a deployment with no
sensor that promises to report. When no sensor of a home has reported for the
horizon, the home is treated as not observed from its last observation on, the
days that lost any time are refused by the baseline, and an alert about the
silence is raised and repeated once per cooldown, which does not say why.
`sensor_modeling.health.fleet` recognises a silence that most homes share. The
rule is off by default, so by default a silent home is read as before.

It was designed from the TIHM run, so that run cannot be its evidence.

- **The protocol.** `docs/SILENT_HOME_PROTOCOL.md`, frozen and pushed before
  any simulated home had been run with the rule on. It fixes eleven estimands
  and eight criteria on 100 paired simulated homes reduced to their event
  sensors, and declares the run on TIHM a description.
- **The run.** `docs/SILENT_HOME_RESULTS.md`, from a clean commit. **The
  evidence is simulated.**
- **An outage raises alerts, and the rule removes them.** With the rule off a
  60-hour outage raises 2.76 behavioural alerts per home more than the same
  home without it, [2.36, 3.15], Monte Carlo standard error 0.20. At 12 hours
  the excess is −0.28 [−0.44, −0.13], standard error 0.08, and the rule
  removes 3.04 [2.68, 3.39], standard error 0.18.
- **With the rule on, the outage arm raises fewer alerts than the home left
  alone.** 30 against 58 in the window, in 30 homes fewer and in 9 more. The
  criterion asks only that the excess lie below its margin and counts this as
  success. One alert against 16 is raised up to a day after the outage ends,
  when most of those days are refused and a refused day raises nothing; 29
  against 42 in the rest of the window. Whether the second is a loss of
  sensitivity after an outage is not settled. The nearest evidence is the two
  detection arms set side by side after the run: with the rule on, 76 of 100
  homes are detected after an outage and 75 with none, at a median 11.0
  against 10.0 days.
- **Nothing changes where no home is silent.** In 8,400 person-days of stable
  homes the rule changes no alert and raises none. A real change is detected
  in 75 of 100 homes with and without it; in that arm no home was silent for
  the horizon, so the rule never acted and the two are one finding.
- **After an outage, fewer homes meet the detection definition with the rule
  on.** 76 of 100 against 91: −0.15 [−0.23, −0.07], standard error 0.04. The
  interval
  excludes zero and lies on both sides of the −0.10 margin, so the
  pre-specified verdict is inconclusive: the loss is shown neither to be
  within the margin nor to exceed it. Beside it, and not as a criterion: with
  the rule off 38 of 100 homes with an outage and no change also meet the
  definition, 8 with it on, and with no outage 75 of the 100 homes are
  detected. A reading, not tested, is that part of the 91 are alerts the
  outage itself raises; one home outside the protocol had suggested it before
  the run, which the record says. The 76 are not the 75: with the rule on, 67
  homes are detected both with and without an outage, 8 only without and 9
  only after.
- **The silence is reported.** Every outage, a median 12.0 hours after it
  began, in three alerts, with three or four days refused. A shared outage is
  one stretch for the fleet check, 12.0 hours after it began, and outages each
  home has on its own are none.
- **The rule saw more than its stated limits in three places.** A silence is
  dated from the home's last observation, which precedes an outage by minutes
  to hours, and a day is closed by the first step after midnight. So the day
  an outage began on was refused in 4 of the 43 homes whose outage began at
  noon or later; at 24 hours it was refused in 8 of 100, where the protocol
  expected none; and an 8-hour outage was reported in one home in 100. The
  homes whose outage began at noon or later show no excess, −0.19
  [−0.47, 0.07].

On TIHM, as a description and not a test:

- **Fewer behavioural alerts, some of them new.** At 12 hours 147 of the 183
  alerts are no longer raised and 13 are raised that were not, leaving 49;
  the day that held 33 has none. The record does not say which of the 147
  were on days with no report; the alert-burden description counted 44 alerts
  on such days.
- **The TIHM homes are silent often.** The rule refuses 322 of the 2,850
  monitored days, in 48 of the 56 homes, and raises 324 alerts about silence,
  which repeat while a silence lasts. 145 of them fall inside the two
  stretches the fleet check calls common: the first days of the dataset, when
  three homes were monitored, and 15 to 17 June, in which all 47 monitored
  homes were silent at once. At 24 hours it refuses 205 days and raises 205.

Consequence: in simulation the rule removes the alerts it was built to
remove; whether that costs detection after an outage the test left undecided.
On a real cohort it exchanges alerts about behaviour for more alerts about
silence: 373 alerts of both kinds where there were 183. A simulated home is
not silent for twelve hours unless a fault makes it so, so the simulator could
not have shown that. Which horizon a real home needs, and what an alert about
silence is worth to the person who receives it, are open. Priorities 1 and 2
above are what would answer them.

TIHM is by Palermo et al., *Scientific Data* 10, 606 (2023), under CC BY 4.0;
Surrey and Borders Partnership NHS Foundation Trust and Howz are acknowledged,
as the dataset asks.

## Phase 8 — Real-Time System Hardening

**Status: deferred.** Its gate is not met: no inference improvement has
survived held-out or external evaluation, and the first external evaluation was
negative.

Only after the inference improvements survive held-out and external evaluation
should the project expand its production surface. Potential work includes:

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

`0.9.0` is the current stable release.

Future versions are created only when the research programme produces a
coherent user-facing software increment. Paper milestones do not automatically
require package releases.

Release notes come from the matching `CHANGELOG.md` section. The changelog is
the single source of truth for release notes.

## Immediate Order of Work

Each item is an open question that a measured result left, in the order that
lets each answer inform the next.

1. **Evaluate the reference formulation with the time prior, then confirm it
   on held-out homes.**
   - **The combination.** The time prior (3.1) and the fitted hurdle channels
     (3.3) are the two Phase 3 successes, and they have never been evaluated
     together in the recursion. Evaluate the combination on the development
     panel under a frozen protocol. *Protocol frozen
     ([PHASE3_COMBINED_PRIOR_PROTOCOL](docs/PHASE3_COMBINED_PRIOR_PROTOCOL.md));
     the run on the CASAS archive is pending.*
   - **The confirmation.** Freeze the resulting formulation, with the Phase 2
     baselines under matched information. Score it on CASAS households outside
     the development panel, under a protocol frozen before scoring.

   Until then, every Phase 3 and 4 conclusion is a development-panel result.
2. **Explain the external transfer failure (Phase 5).**
   - **The questions.** The transferred parameters lose 0.19 to 0.32 of
     balanced accuracy against an in-sample oracle, and adaptation at the
     declared strength changes no decision. Whether adaptation permitted before
     scoring can recover part of that loss is open. Phase 3.4 found the
     declared strength stronger than the data need. Whether the event-rate
     mismatch is a property of the sensing layout is also open.
   - **The data.** The Ordóñez homes are now exploratory only. A confirmatory
     answer needs an independent dataset not yet inspected.
3. **Within-state temporal dependence (Phase 3.3).**
   - **The model family.** The predictive checks route to activity sub-states
     or a Markov-modulated emission. These would produce the long quiet runs
     that no marginal count distribution can.
   - **What it must recover.** `home_active` recall, lost to the fitted
     channels, is the test it must pass.
4. **An abstention rule (Phase 4).** Select a selective-prediction rule on
   development data, with a guard covering the activity states and `away`
   that the first comparison under-retained. Freeze it before any held-out
   evaluation.
5. **Sensor-information frontier (Phase 6).** Formalise multi-objective
   deployment trade-offs for the confirmed formulation, with configurations
   selected before confirmatory scoring.
6. **Reliability modelling (Phase 7).** Separate failure processes from
   behavioural state, and quantify robustness under realistic missingness.
   The silent-home rule is a declared rule with a declared horizon. How long
   a real home is silent when nothing is wrong is the measurement it lacks.
   The calibrated baseline reference is opt-in for a similar reason: where
   its thresholds belong on a real home, which the simulated homes put at
   about 0.59 times the declared ones, is not known.
7. **System hardening (Phase 8).** Only after an improvement survives held-out
   and external evaluation.
