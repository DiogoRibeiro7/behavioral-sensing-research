# Phase 3.1: the hierarchical time-of-day prior

Phase 1 found time of day worth +0.084 to +0.133 household balanced accuracy to
the discriminative models, but the generative model had no way to use it. The
[periodic state prior](PERIODIC_STATE_PRIOR.md) gives it one. This
pre-specified experiment measures three things on the development panel:

- whether the generative model can use the hour through the prior;
- how its result compares with the matched discriminative models;
- what the prior costs or gains in probability quality.

**Status: pre-specified, development panel.** Every setting, comparison and
criterion was frozen before any household was scored. The development homes
have been inspected in earlier work, so this is not a final held-out claim, and
the frozen external cohort is not touched.

## Protocol

The protocol is `artifacts/phase3/time_prior_protocol.json`. It was committed
before scoring, and the run refuses to start if the code's protocol differs
from it by a single value. It freezes:

- **Households.** The 20 single-resident development homes and the two frozen
  Phase 1 folds (`artifacts/phase1/household_splits.json`, recorded by digest).
  Each home is scored once, by models never fitted on it.
- **Sets.** The four nested sets, `I0` to `I3`.
- **Models.** Each is scored only where it can consume the set exactly:

  | Model | Scored in | What it is |
  | --- | --- | --- |
  | `generative` | `I0`, `I2` | the original generative model, restricted to the set |
  | `generative_untimed` | `I0`, `I2` | the hierarchical model with time disabled (see below) |
  | `generative_periodic` | `I1`, `I3` | the hierarchical model with the hour |
  | `diagnostic` | `I1`, `I3` | the supervised diagnostic, gradient-boosted trees |
  | `logistic` | `I1`, `I3` | regularised multinomial logistic regression |
  | `state_frequency` | `I1`, `I3` | the no-information reference |

- **Hyperparameters.**
  - The prior has `K = 2` harmonics and ridge `λ₀ = 1`.
  - The pooling strength `λ = 24` labelled hours is declared, not selected on
    any data.
  - The diagnostic and logistic regression keep their declared settings.
  - Nothing is tuned.
- **Fitting.** Each fold's priors are fitted on that fold's training homes
  only. Held-out homes get the population prior.
- **Metrics.** Household balanced accuracy, log loss, Brier score, expected
  calibration error over 10 bins, and per-state recall.
- **Bootstrap.** 10,000 household resamples and 95% percentile intervals for
  the mean paired difference; the median is recorded too. All seeds are 0.

### The time-disabled model

`generative_untimed` is the hierarchical model with its periodic terms removed.
Its prior is fitted exactly as the periodic one is, but on counts spread evenly
over the hours. Every periodic coefficient is then zero, and the prior is one
distribution `π̄` at every hour. A prior that ignores the hour needs no hour, so
it is expressed as a time-homogeneous ontology: the base dwell times scaled by
`π̄(s) / π(s)`, which is the same generator as constant circadian stickiness.
A test checks that it gives the same posteriors, to 10⁻⁸, as the periodic
machinery given the hour with that flat prior.

`generative_untimed` in `I0` and `generative_periodic` in `I1` therefore differ
only in the hour. `generative_untimed` against `generative` in `I0` differs only
in a fitted prior against the declared one.

### Estimands

Each is a paired household difference. A positive value favours the first
model.

| Key | Role | First | Second | Question |
| --- | --- | --- | --- | --- |
| P1 | primary | `generative_periodic` `I1` | `generative_untimed` `I0` | what the hour is worth to the hierarchical model |
| P2 | primary | `generative_periodic` `I1` | `generative` `I0` | the new model with the hour against the original without it |
| P3 | primary | `diagnostic` `I1` | `generative_periodic` `I1` | formulation gap at `I1` |
| P4 | primary | `logistic` `I1` | `generative_periodic` `I1` | formulation gap at `I1` |
| S1 | secondary | `generative_untimed` `I0` | `generative` `I0` | a fitted prior against the declared one |
| S2 | secondary | `generative_periodic` `I3` | `generative_untimed` `I2` | what the hour is worth with recent history |
| S3 | secondary | `generative_periodic` `I3` | `generative` `I2` | the new model against the original, both with history |
| S4 | secondary | `diagnostic` `I3` | `generative_periodic` `I3` | formulation gap at `I3` |
| S5 | secondary | `logistic` `I3` | `generative_periodic` `I3` | formulation gap at `I3` |
| S6 | secondary | `generative_periodic` `I3` | `generative_periodic` `I1` | what recent history is worth to the hierarchical model |

**Per-state recall.** It is reported for P1 and P2. A state is rare if its
share of labelled time in either fold's training homes is below 5%.

**Adaptation arm (secondary).** This is the one place the hierarchy's
shrinkage is used:

- **Window.** Each held-out home's deviation is fitted from its own labels in
  the first 7 days after its first observation.
- **Scoring.** Only its labelled moments after those 7 days are scored, with the
  adapted prior and with the population prior.
- **Pooling strengths.** `λ = 24` is declared. `λ = 2.4` and `λ = 240` are
  reported as sensitivity checks, and none is selected.
- **Matching.** It uses the home's own earlier labels, which no other model
  receives, so it is compared only with the population prior.

### Criteria

The criteria compare effect sizes, with their uncertainty, against declared
minimal important differences `δ`. They are not significance tests.

| Metric | `δ` |
| --- | ---: |
| balanced accuracy | 0.02, the smallest Phase 1 information gain reported as real |
| log loss | 0.05 |
| Brier score | 0.01 |
| calibration error | 0.02 |
| per-state recall | 0.05 |

Each comparison gets one verdict:

| Verdict | When |
| --- | --- |
| favours model | mean ≥ `δ` and the interval's lower bound > 0 |
| favours reference | mean ≤ `−δ` and the interval's upper bound < 0 |
| negligible | the whole interval lies within `(−δ, δ)` |
| uncertain | anything else |

The overall rules are fixed in advance:

- **Success.** P1 balanced accuracy favours the model, and neither P1 log loss
  nor P1 calibration error favours the reference.
- **Failure.** P1 balanced accuracy is negligible or favours the reference.
- **Inconclusive.** Anything else.
- **P2.** It is judged by the same rule, as the practical improvement over the
  original model.
- **Household adaptation.** It helps if the declared arm's `I1` balanced accuracy
  favours the adapted prior, and its log loss does not favour the population
  prior.

## Results

The summary below is generated from the published record,
`artifacts/phase3/phase3-hierarchical-time-prior.json`, by `render_summary`.
A test checks that it still matches the record.

<!-- generated-summary:start -->

### Hierarchical time-of-day prior: summary

Generated from the record `phase3-hierarchical-time-prior`: protocol `50c92f687181`, commit `5dd81777279e`. Status: pre-specified; development panel, which earlier work has inspected.

- Inference regime: online filter (attested on migration from schema 1.1).
- 20 households in 2 cross-fitted folds, each scored once by models never fitted on it.
- Differences are means of paired household differences, with 95% household bootstrap intervals from 10,000 resamples. A positive value favours the first model.
- Minimal important differences: balanced_accuracy 0.02, brier 0.01, calibration_error 0.02, log_loss 0.05, recall 0.05.

#### Pre-specified conclusions

| Question | Conclusion | Rule |
| --- | ---: | ---: |
| Can the hierarchical model use the hour? (P1) | success | P1 balanced accuracy favours the model, and P1 log loss and P1 calibration error do not favour the reference |
| Does it improve on the original model? (P2) | success | P2 is judged by the same rule, as the improvement over the original model |
| Does household adaptation help? (A-I1) | inconclusive | household adaptation helps if A-I1 balanced accuracy favours the adapted prior and A-I1 log loss does not favour the population prior |

Verdicts: favours model: mean >= delta and lower bound > 0; favours reference: mean <= -delta and upper bound < 0; negligible: the whole interval within (-delta, delta); uncertain: anything else.

#### Every cell

Balanced accuracy as the median, then the mean with its interval. The other metrics are medians, and lower is better:

| Model | Set | Balanced accuracy | Log loss | Brier | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: |
| generative | I0 | 0.362 (0.370 [0.347, 0.391]) | 3.403 | 0.800 | 0.233 |
| generative | I2 | 0.383 (0.390 [0.367, 0.414]) | 3.756 | 0.876 | 0.372 |
| generative_untimed | I0 | 0.382 (0.378 [0.358, 0.398]) | 3.246 | 0.744 | 0.201 |
| generative_untimed | I2 | 0.407 (0.404 [0.382, 0.425]) | 3.565 | 0.770 | 0.315 |
| generative_periodic | I1 | 0.520 (0.509 [0.486, 0.531]) | 2.955 | 0.555 | 0.174 |
| generative_periodic | I3 | 0.521 (0.511 [0.487, 0.534]) | 3.204 | 0.574 | 0.192 |
| diagnostic | I1 | 0.519 (0.515 [0.477, 0.548]) | 4.636 | 0.624 | 0.190 |
| diagnostic | I3 | 0.596 (0.590 [0.550, 0.624]) | 2.605 | 0.492 | 0.135 |
| logistic | I1 | 0.488 (0.477 [0.434, 0.516]) | 0.925 | 0.470 | 0.098 |
| logistic | I3 | 0.500 (0.498 [0.450, 0.539]) | 0.909 | 0.448 | 0.098 |
| state_frequency | I1 | 0.167 (0.161 [0.156, 0.164]) | 1.510 | 0.750 | 0.095 |
| state_frequency | I3 | 0.167 (0.161 [0.156, 0.164]) | 1.510 | 0.750 | 0.095 |

#### Estimands: balanced accuracy

| Key | Comparison | Mean difference | Homes favouring the first | Verdict |
| --- | ---: | ---: | ---: | ---: |
| P1 | generative_periodic@I1 vs generative_untimed@I0 | +0.131 [+0.120, +0.142] | 20/20 | favours model |
| P2 | generative_periodic@I1 vs generative@I0 | +0.140 [+0.125, +0.154] | 20/20 | favours model |
| P3 | diagnostic@I1 vs generative_periodic@I1 | +0.006 [−0.022, +0.030] | 12/20 | uncertain |
| P4 | logistic@I1 vs generative_periodic@I1 | −0.032 [−0.064, −0.004] | 6/20 | favours reference |
| S1 | generative_untimed@I0 vs generative@I0 | +0.009 [−0.001, +0.016] | 18/20 | negligible |
| S2 | generative_periodic@I3 vs generative_untimed@I2 | +0.106 [+0.099, +0.114] | 20/20 | favours model |
| S3 | generative_periodic@I3 vs generative@I2 | +0.120 [+0.106, +0.134] | 20/20 | favours model |
| S4 | diagnostic@I3 vs generative_periodic@I3 | +0.079 [+0.051, +0.105] | 19/20 | favours model |
| S5 | logistic@I3 vs generative_periodic@I3 | −0.013 [−0.046, +0.017] | 8/20 | uncertain |
| S6 | generative_periodic@I3 vs generative_periodic@I1 | +0.002 [−0.005, +0.008] | 12/20 | negligible |

#### Estimands: probability quality

Positive values favour the first model, that is, its loss or calibration error is lower:

| Key | Log loss | Brier | Calibration error |
| --- | ---: | ---: | ---: |
| P1 | +0.278 [+0.212, +0.332], favours model | +0.171 [+0.145, +0.195], favours model | +0.028 [−0.009, +0.069], uncertain |
| P2 | +0.456 [+0.369, +0.535], favours model | +0.208 [+0.174, +0.239], favours model | +0.048 [+0.015, +0.084], favours model |
| P3 | −0.630 [−1.641, +0.344], uncertain | −0.058 [−0.157, +0.023], uncertain | −0.021 [−0.089, +0.035], uncertain |
| P4 | +1.880 [+1.524, +2.180], favours model | +0.022 [−0.101, +0.106], uncertain | +0.043 [−0.035, +0.095], uncertain |
| S1 | +0.178 [+0.127, +0.232], favours model | +0.036 [+0.020, +0.052], favours model | +0.021 [+0.009, +0.032], favours model |
| S2 | +0.264 [+0.214, +0.310], favours model | +0.200 [+0.162, +0.236], favours model | +0.109 [+0.072, +0.149], favours model |
| S3 | +0.532 [+0.418, +0.648], favours model | +0.276 [+0.226, +0.320], favours model | +0.160 [+0.120, +0.199], favours model |
| S4 | +0.741 [−0.224, +1.577], uncertain | +0.076 [−0.019, +0.148], uncertain | +0.066 [+0.013, +0.104], favours model |
| S5 | +1.882 [+1.214, +2.369], favours model | +0.047 [−0.088, +0.135], uncertain | +0.062 [−0.025, +0.121], uncertain |
| S6 | −0.172 [−0.240, −0.113], favours reference | −0.010 [−0.026, +0.003], uncertain | −0.022 [−0.037, −0.006], favours reference |

#### Per-state recall

Rare states have under the declared share of labelled time in either fold's training homes. A state is left out of a household where it never occurs:

| State | Rare | P1 (generative_periodic@I1 vs generative_untimed@I0) | P2 (generative_periodic@I1 vs generative@I0) |
| --- | ---: | ---: | ---: |
| away | no | +0.807 [+0.760, +0.852], 20/20, favours model | +0.807 [+0.760, +0.852], 20/20, favours model |
| home_active | no | +0.007 [+0.001, +0.013], 12/20, negligible | +0.023 [+0.010, +0.045], 17/20, negligible |
| home_inactive | no | +0.020 [+0.007, +0.038], 15/20, negligible | +0.043 [+0.016, +0.075], 16/20, uncertain |
| sleeping | no | −0.039 [−0.070, −0.013], 5/20, uncertain | +0.004 [−0.032, +0.035], 13/20, negligible |
| bed_awake | yes | +0.067 [+0.000, +0.200], 1/5, uncertain | −0.074 [−0.207, +0.000], 0/5, uncertain |
| bathroom_activity | yes | +0.005 [+0.002, +0.008], 11/20, negligible | +0.007 [+0.003, +0.010], 12/20, negligible |
| kitchen_activity | yes | +0.001 [−0.001, +0.002], 5/20, negligible | +0.000 [−0.001, +0.001], 3/20, negligible |

#### Household adaptation

Each held-out home's deviation is fitted from its own labels in the first 7 days, and it is scored only after them. The adapted prior is compared with the population prior on those moments. The declared pooling strength is marked; the others are sensitivity checks:

| Key | Set | Declared | Balanced accuracy | Log loss | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: |
| A-I1 | I1 | yes | +0.011 [+0.002, +0.021], 13/20, uncertain | +0.142 [+0.070, +0.233], favours model | +0.032 [+0.017, +0.049], favours model |
| A-I3 | I3 | yes | +0.008 [+0.002, +0.015], 16/20, negligible | +0.106 [+0.025, +0.213], favours model | +0.013 [−0.001, +0.029], uncertain |
| A-I1 (adapted@2.4) | I1 | no | +0.010 [−0.007, +0.027], 12/20, uncertain | +0.184 [+0.077, +0.306], favours model | +0.033 [+0.009, +0.059], favours model |
| A-I3 (adapted@2.4) | I3 | no | +0.009 [−0.005, +0.022], 16/20, uncertain | +0.145 [+0.022, +0.293], favours model | +0.021 [−0.003, +0.051], uncertain |
| A-I1 (adapted@240) | I1 | no | +0.001 [+0.000, +0.003], 17/20, negligible | +0.036 [+0.018, +0.061], uncertain | +0.014 [+0.006, +0.022], uncertain |
| A-I3 (adapted@240) | I3 | no | +0.001 [+0.000, +0.001], 15/20, negligible | +0.026 [+0.006, +0.055], uncertain | −0.002 [−0.008, +0.003], negligible |

<!-- generated-summary:end -->

### What this shows and does not show

Balanced-accuracy figures are mean paired differences, with their 95% household
intervals and the number of the 20 homes that improved.

- **The run agrees with Phase 1.** The original generative model in `I0` and
  `I2`, and the diagnostic and logistic regression in `I1` and `I3`, reproduce
  the [recoverable-gap](PHASE1_RECOVERABLE_GAP.md) values exactly.
- **The hierarchical model can use the hour: pre-specified success.**
  - It gains +0.131 [+0.120, +0.142] from the hour, in all 20 homes (P1).
  - Its log loss and Brier score improve too. The change in calibration error,
    +0.028 [−0.009, +0.069], is uncertain.
  - Against the original model without the hour, the gain is +0.140
    [+0.125, +0.154], in all 20 homes, and calibration error also improves
    (P2).
- **The gain is almost entirely `away`.**
  - Without the hour, the generative model almost never reports `away`: its
    median recall is 0.006, for a state that is about 29% of labelled time.
    With the hour it is 0.824, level with the diagnostic (0.859) and logistic
    regression (0.797).
  - Every other state's recall changes by less than 0.05. Sleeping moves by
    −0.039 [−0.070, −0.013], an uncertain verdict.
  - The rare states, `bathroom_activity`, `kitchen_activity` and `bed_awake`,
    show no meaningful change. `bathroom_activity` stays far below the
    diagnostic (median recall 0.267 against 0.554).
  - Read this way, the hour mostly settles whether a quiet home is empty.
- **At `I1` the formulation gap to the diagnostic is small but not settled.**
  The diagnostic's lead is +0.006 [−0.022, +0.030]. The interval reaches
  beyond the minimal difference on both sides, so equivalence is not shown.
  The generative model is ahead of logistic regression by 0.032
  [0.004, 0.064] in balanced accuracy.
- **Probability quality is still the weak point.**
  - Logistic regression's log loss is better by 1.88 [1.52, 2.18].
  - The generative model's median log loss at `I1`, 2.955, is above the
    no-information 1.510.
  - Its median calibration error is 0.174, against 0.098 for logistic
    regression.
  - The prior improves probabilities relative to the original model, but does
    not make them good.
- **Recent history still does nothing for the generative model.**
  - With the hour, adding three previous windows is worth +0.002 [−0.005,
    +0.008], a negligible verdict (S6).
  - It makes log loss worse by 0.172 and calibration error by 0.022.
  - At `I3` the diagnostic leads by +0.079 [+0.051, +0.105], in 19 of 20 homes.
  - The Phase 1 finding stands: the history part of the gap is a formulation
    problem this prior does not touch.
- **A fitted prior without the hour changes little.** Its balanced accuracy
  effect is negligible, +0.009 [−0.001, +0.016], though log loss and
  calibration error improve (S1). What matters is the hour, not the fitted
  marginal.
- **Household adaptation mostly improves probabilities: pre-specified
  inconclusive.**
  - Each home had a median of 154 labelled hours in its 7-day window. The
    adapted prior gains +0.011 [+0.002, +0.021] balanced accuracy in 13 of 20
    homes, below the minimal difference.
  - It improves log loss by 0.142 [0.070, 0.233] and calibration error by
    0.032 [0.017, 0.049].
  - Weaker pooling (`λ = 2.4`) behaves alike. Stronger pooling (`λ = 240`)
    makes the effect negligible.

**Not shown:**

- This is the development panel, which earlier work has inspected, so it is
  not a held-out or external claim.
- There is one seed.
- No model abstains, so the figures are not comparable with the pipeline's
  abstaining output.
- The adaptation arm scores a later period than the population arm, so the two
  are compared only within themselves.

## Reproducing it

```text
python scripts/run_phase3_time_prior.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes the record and a Markdown
summary generated from the written record.
