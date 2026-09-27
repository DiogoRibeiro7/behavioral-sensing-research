# Phase 3.2: the explicit history state

Phase 1 found recent history worth +0.021 household balanced accuracy to the
generative filter, against +0.088 to the supervised diagnostic. With the
hierarchical time prior, [Phase 3.1](PHASE3_TIME_PRIOR.md) measured +0.002.
The [explicit history state](HISTORY_STATE.md) lets each channel's recent
activations condition its current rate, rather than leaving the past to the
filtered posterior alone. This pre-specified experiment measures on the
development panel:

- whether the history state improves the generative model on identical
  information;
- whether the model now recovers materially more from recent history than the
  original model did;
- how it compares with the matched discriminative models;
- whether time and history information are complementary.

**Status: pre-specified, development panel.** Every setting, comparison and
criterion was frozen before any household was scored. The development homes
have been inspected in earlier work, so this is not a final held-out claim, and
the frozen external cohort is not touched.

## Protocol

The protocol is `artifacts/phase3/history_protocol.json`. It was committed
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
  | `generative_history` | `I2` | the original model with the explicit history state |
  | `generative_periodic` | `I1`, `I3` | the hierarchical time-prior model of Phase 3.1 |
  | `generative_periodic_history` | `I3` | the hierarchical time-prior model with the history state |
  | `diagnostic` | `I0` to `I3` | the supervised diagnostic, gradient-boosted trees |
  | `logistic` | `I0` to `I3` | regularised multinomial logistic regression |
  | `state_frequency` | `I0` to `I3` | the no-information reference |

- **Why each model is included.**
  - `generative_periodic` is included because Phase 3.1 met its pre-specified
    success rule.
  - `generative_periodic_history` is declared here, before any result of the
    history state was observed.
  - The history state needs the three lagged windows, so the scorer refuses it
    in `I0` and `I1`. There, the models without it stand in for the family.
- **Hyperparameters.**
  - The history window is `k = 3` lagged windows: the lags of `I2` and `I3`.
  - The coefficients' Gaussian prior has precision 1.
  - The time prior keeps its Phase 3.1 settings: `K = 2`, `λ₀ = 1`, `λ = 24`.
  - The diagnostic and logistic regression keep their declared settings.
  - Nothing is tuned.
- **Fitting.** Each fold's history coefficients and time prior are fitted on
  that fold's training homes only. Held-out homes get the population prior and
  the fold's coefficients.
- **Metrics.** Household balanced accuracy, log loss, Brier score, expected
  calibration error over 10 bins, and per-state recall.
- **Bootstrap.** 10,000 household resamples, and 95% percentile intervals for
  both the mean and the median paired difference. All seeds are 0.

### Estimands

Each is a paired household difference. A positive value favours the first
model or the larger set. There are two kinds:

- a **formulation** estimand compares two models on the same set;
- an **information** estimand compares one model family across nested sets
  that differ only in recent history.

No estimand changes the model and the information at once.

| Key | Role | Kind | First | Second | Question |
| --- | --- | --- | --- | --- | --- |
| H1 | primary | formulation | `generative_history` `I2` | `generative` `I2` | what the history state adds on identical information |
| H2 | primary | information | `generative_history` `I2` | `generative` `I0` | what recent history is worth to the history-state model |
| H3 | primary | formulation | `diagnostic` `I2` | `generative_history` `I2` | formulation gap at `I2` |
| H4 | primary | formulation | `logistic` `I2` | `generative_history` `I2` | formulation gap at `I2` |
| R1 | reference | information | `generative` `I2` | `generative` `I0` | what recent history is worth to the original model |
| S1 | secondary | formulation | `generative_periodic_history` `I3` | `generative_periodic` `I3` | what the history state adds when the hour is known |
| S2 | secondary | information | `generative_periodic_history` `I3` | `generative_periodic` `I1` | what recent history is worth to the model with both |
| S3 | secondary | formulation | `diagnostic` `I3` | `generative_periodic_history` `I3` | formulation gap at `I3` |
| S4 | secondary | formulation | `logistic` `I3` | `generative_periodic_history` `I3` | formulation gap at `I3` |
| R2 | reference | information | `generative_periodic` `I3` | `generative_periodic` `I1` | what recent history is worth to the time-prior model |

H2 and R1 are information estimands for two model families. Their difference,
on the same homes, is exactly H1, because both start from the same `I0` model.

**Per-state recall.** It is reported for H1, H2 and S1. A state is rare if its
share of labelled time in either fold's training homes is below 5%.

**Time and history interaction.** For each model family and household:

```text
interaction = (score in I3 − score in I1) − (score in I2 − score in I0)
```

That is, the history gain with the hour minus the gain without it. A positive
value means time and history are complementary. The families are:

| Family | `I0` | `I1` | `I2` | `I3` |
| --- | --- | --- | --- | --- |
| generative with the history state | `generative` | `generative_periodic` | `generative_history` | `generative_periodic_history` |
| generative without the history state | `generative` | `generative_periodic` | `generative` | `generative_periodic` |
| diagnostic | `diagnostic` | `diagnostic` | `diagnostic` | `diagnostic` |
| logistic | `logistic` | `logistic` | `logistic` | `logistic` |

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

- **Success.** H1 balanced accuracy favours the history-state model, and
  neither H1 log loss nor H1 calibration error favours the original model.
- **Failure.** H1 balanced accuracy is negligible or favours the original
  model.
- **Inconclusive.** Anything else.
- **S1.** It is judged by the same rule, as the result when the hour is known.
- **Materially more than +0.021.** The history-state model recovers materially
  more from recent history than the original model if H1 favours it. H1 is
  exactly H2 − R1 on the same homes.

The earlier +0.021 is not a threshold. It is re-measured in this run as R1,
and the comparison is paired against that. A fixed threshold would ignore both
its uncertainty and the pairing, and the minimal important difference already
states what size of change matters.

## Results

The summary below is generated from the published record,
`artifacts/phase3/phase3-explicit-history.json`, by `render_summary`.
A test checks that it still matches the record.

<!-- generated-summary:start -->

### Explicit history state: summary

Generated from the record `phase3-explicit-history`: protocol `3e3f42fc5c32`, commit `36892ce1c782`. Status: pre-specified; development panel, which earlier work has inspected.

- Inference regime: online filter (attested on migration from schema 1.1).
- 20 households in 2 cross-fitted folds, each scored once by models never fitted on it.
- Differences are paired by household, with the mean and the median and their 95% household bootstrap intervals from 10,000 resamples. A positive value favours the first model or the larger set.
- Formulation estimands compare two models on one set. Information estimands compare one model family across nested sets.
- Minimal important differences: balanced_accuracy 0.02, brier 0.01, calibration_error 0.02, log_loss 0.05, recall 0.05.

#### Pre-specified conclusions

| Question | Conclusion | Rule |
| --- | ---: | ---: |
| Does the explicit history state add to the original model on identical information? (H1) | failure | H1 balanced accuracy favours the explicit-history model, and H1 log loss and H1 calibration error do not favour the original model |
| Does it add when the hour is known? (S1) | failure | S1 is judged by the same rule |

Recent history is worth −0.021 [−0.038, −0.005] to the explicit-history model (H2) and +0.021 [+0.016, +0.026] to the original model (R1); H1, their difference on the same homes, is −0.042 [−0.060, −0.026]. H1 equals the explicit-history model's history gain (H2) minus the original model's (R1) on the same homes. Materially more means H1 favours the model: at least the declared minimal difference, with its interval above zero. The earlier +0.021 is re-measured as R1 and is not a threshold.

Verdicts: favours model: mean >= delta and lower bound > 0; favours reference: mean <= -delta and upper bound < 0; negligible: the whole interval within (-delta, delta); uncertain: anything else.

#### Every cell

Balanced accuracy as the median, then the mean with its interval. The other metrics are medians, and lower is better:

| Model | Set | Balanced accuracy | Log loss | Brier | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: |
| generative | I0 | 0.362 (0.370 [0.347, 0.391]) | 3.403 | 0.800 | 0.233 |
| generative | I2 | 0.383 (0.390 [0.367, 0.414]) | 3.756 | 0.876 | 0.372 |
| generative_history | I2 | 0.339 (0.348 [0.334, 0.363]) | 3.120 | 0.888 | 0.365 |
| generative_periodic | I1 | 0.520 (0.509 [0.486, 0.531]) | 2.955 | 0.555 | 0.174 |
| generative_periodic | I3 | 0.521 (0.511 [0.487, 0.534]) | 3.204 | 0.574 | 0.192 |
| generative_periodic_history | I3 | 0.466 (0.467 [0.448, 0.486]) | 2.611 | 0.599 | 0.212 |
| diagnostic | I0 | 0.418 (0.418 [0.397, 0.436]) | 1.450 | 0.648 | 0.113 |
| diagnostic | I1 | 0.519 (0.515 [0.477, 0.548]) | 4.636 | 0.624 | 0.190 |
| diagnostic | I2 | 0.505 (0.506 [0.477, 0.532]) | 1.370 | 0.565 | 0.116 |
| diagnostic | I3 | 0.596 (0.590 [0.550, 0.624]) | 2.605 | 0.492 | 0.135 |
| logistic | I0 | 0.354 (0.344 [0.317, 0.370]) | 1.242 | 0.665 | 0.109 |
| logistic | I1 | 0.488 (0.477 [0.434, 0.516]) | 0.925 | 0.470 | 0.098 |
| logistic | I2 | 0.392 (0.383 [0.348, 0.414]) | 1.176 | 0.631 | 0.107 |
| logistic | I3 | 0.500 (0.498 [0.450, 0.539]) | 0.909 | 0.448 | 0.098 |
| state_frequency | I0 | 0.167 (0.161 [0.156, 0.164]) | 1.510 | 0.750 | 0.095 |
| state_frequency | I1 | 0.167 (0.161 [0.156, 0.164]) | 1.510 | 0.750 | 0.095 |
| state_frequency | I2 | 0.167 (0.161 [0.156, 0.164]) | 1.510 | 0.750 | 0.095 |
| state_frequency | I3 | 0.167 (0.161 [0.156, 0.164]) | 1.510 | 0.750 | 0.095 |

#### Estimands: balanced accuracy

| Key | Kind | Comparison | Mean difference | Median difference | Homes improved / worsened | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 | formulation | generative_history@I2 vs generative@I2 | −0.042 [−0.060, −0.026] | −0.035 [−0.060, −0.022] | 3 / 17 | favours reference |
| H2 | information | generative_history@I2 vs generative@I0 | −0.021 [−0.038, −0.005] | −0.019 [−0.044, −0.003] | 5 / 15 | favours reference |
| H3 | formulation | diagnostic@I2 vs generative_history@I2 | +0.158 [+0.132, +0.182] | +0.155 [+0.139, +0.185] | 19 / 1 | favours model |
| H4 | formulation | logistic@I2 vs generative_history@I2 | +0.035 [+0.004, +0.063] | +0.034 [+0.005, +0.073] | 14 / 6 | favours model |
| R1 | information | generative@I2 vs generative@I0 | +0.021 [+0.016, +0.026] | +0.022 [+0.014, +0.026] | 19 / 1 | favours model |
| S1 | formulation | generative_periodic_history@I3 vs generative_periodic@I3 | −0.044 [−0.060, −0.028] | −0.045 [−0.063, −0.019] | 2 / 18 | favours reference |
| S2 | information | generative_periodic_history@I3 vs generative_periodic@I1 | −0.042 [−0.057, −0.027] | −0.048 [−0.064, −0.017] | 1 / 19 | favours reference |
| S3 | formulation | diagnostic@I3 vs generative_periodic_history@I3 | +0.123 [+0.091, +0.148] | +0.141 [+0.122, +0.155] | 19 / 1 | favours model |
| S4 | formulation | logistic@I3 vs generative_periodic_history@I3 | +0.031 [−0.009, +0.065] | +0.032 [+0.009, +0.076] | 15 / 5 | uncertain |
| R2 | information | generative_periodic@I3 vs generative_periodic@I1 | +0.002 [−0.005, +0.008] | +0.003 [−0.005, +0.008] | 12 / 8 | negligible |

#### Estimands: probability quality

Mean differences. Positive values favour the first model, that is, its loss or calibration error is lower:

| Key | Log loss | Brier | Calibration error |
| --- | ---: | ---: | ---: |
| H1 | +0.726 [+0.537, +0.929], favours model | −0.012 [−0.032, +0.007], uncertain | +0.007 [−0.003, +0.018], negligible |
| H2 | +0.478 [+0.284, +0.688], favours model | −0.090 [−0.134, −0.050], favours reference | −0.126 [−0.160, −0.088], favours reference |
| H3 | +1.536 [+1.233, +1.803], favours model | +0.273 [+0.158, +0.361], favours model | +0.194 [+0.118, +0.254], favours model |
| H4 | +1.407 [+0.572, +1.931], favours model | +0.169 [+0.025, +0.256], favours model | +0.196 [+0.094, +0.266], favours model |
| R1 | −0.249 [−0.310, −0.189], favours reference | −0.078 [−0.127, −0.038], favours reference | −0.134 [−0.163, −0.099], favours reference |
| S1 | +0.650 [+0.465, +0.854], favours model | −0.037 [−0.055, −0.018], favours reference | −0.008 [−0.020, +0.005], negligible |
| S2 | +0.477 [+0.294, +0.681], favours model | −0.047 [−0.065, −0.027], favours reference | −0.030 [−0.050, −0.009], favours reference |
| S3 | +0.092 [−0.808, +0.866], uncertain | +0.113 [+0.012, +0.185], favours model | +0.073 [+0.013, +0.115], favours model |
| S4 | +1.233 [+0.472, +1.715], favours model | +0.083 [−0.058, +0.172], uncertain | +0.070 [−0.026, +0.133], uncertain |
| R2 | −0.172 [−0.240, −0.113], favours reference | −0.010 [−0.026, +0.003], uncertain | −0.022 [−0.037, −0.006], favours reference |

#### Per-state recall

Mean differences, homes improved and worsened, and verdicts. A state is left out of a household where it never occurs:

| State | Rare | H1 (formulation) | H2 (information) | S1 (formulation) |
| --- | ---: | ---: | ---: | ---: |
| away | no | −0.024 [−0.034, −0.013], 3 / 17, negligible | +0.089 [+0.071, +0.109], 20 / 0, favours model | +0.000 [−0.018, +0.019], 7 / 13, negligible |
| home_active | no | −0.164 [−0.214, −0.114], 1 / 19, favours reference | −0.135 [−0.180, −0.092], 1 / 19, favours reference | −0.173 [−0.220, −0.125], 1 / 19, favours reference |
| home_inactive | no | −0.023 [−0.036, −0.010], 6 / 13, negligible | +0.008 [−0.004, +0.022], 12 / 7, negligible | −0.028 [−0.040, −0.016], 3 / 17, negligible |
| sleeping | no | +0.011 [+0.003, +0.021], 13 / 6, negligible | −0.043 [−0.056, −0.031], 1 / 19, uncertain | −0.018 [−0.030, −0.007], 3 / 17, negligible |
| bed_awake | yes | −0.120 [−0.250, −0.004], 0 / 3, favours reference | −0.117 [−0.250, +0.000], 0 / 2, uncertain | −0.117 [−0.250, +0.000], 0 / 2, uncertain |
| bathroom_activity | yes | +0.016 [−0.041, +0.063], 13 / 7, uncertain | +0.021 [−0.036, +0.067], 14 / 6, uncertain | +0.021 [−0.035, +0.066], 13 / 7, uncertain |
| kitchen_activity | yes | −0.058 [−0.104, −0.009], 5 / 15, favours reference | −0.051 [−0.094, −0.003], 4 / 15, favours reference | −0.052 [−0.098, −0.001], 5 / 14, favours reference |

#### Time and history interaction

For each family, the history gain with the hour (I1 to I3) minus the gain without it (I0 to I2), per household; positive means time and history are complementary:

| Family | Balanced accuracy | Homes improved / worsened | Log loss | Verdict |
| --- | ---: | ---: | ---: | ---: |
| generative with the history state | −0.021 [−0.027, −0.015] | 0 / 20 | −0.000 [−0.022, +0.022] | favours reference |
| generative without the history state | −0.019 [−0.024, −0.014] | 1 / 19 | +0.076 [+0.027, +0.127] | uncertain |
| diagnostic | −0.013 [−0.028, +0.002] | 6 / 14 | +1.126 [+0.596, +1.714] | uncertain |
| logistic | −0.018 [−0.024, −0.012] | 2 / 18 | −0.006 [−0.040, +0.046] | uncertain |

<!-- generated-summary:end -->

### What this shows and does not show

Balanced-accuracy figures are mean paired differences, with their 95% household
intervals and the number of the 20 homes that improved.

- **The run agrees with earlier work.** Every model without the history state
  reproduces the [recoverable-gap](PHASE1_RECOVERABLE_GAP.md) and
  [Phase 3.1](PHASE3_TIME_PRIOR.md) household values exactly. The original
  model's history gain is re-measured as +0.021 [+0.016, +0.026] (R1), and the
  time-prior model's as +0.002 (R2).
- **The history state makes the generative model worse: pre-specified
  failure.**
  - On identical information in `I2`, it loses 0.042 [0.026, 0.060] balanced
    accuracy against the original model. Only 3 of 20 homes improve (H1).
  - When the hour is known, in `I3`, it loses 0.044 [0.028, 0.060]. Only 2 of
    20 homes improve (S1).
- **It does not recover more from recent history; it recovers less.**
  - With the history state, adding the three previous windows is worth −0.021
    [−0.038, −0.005], and only 5 of 20 homes improve (H2).
  - The original model gains +0.021 from them (R1).
  - H1, the difference, favours the original model.
- **The loss is mostly `home_active`.**
  - Its recall falls by 0.164 [0.114, 0.214], in 19 of 20 homes. The median
    falls from 0.503 to 0.333.
  - Pooled over homes, the share of true `home_active` moments reported as
    `away` rises from 0.099 to 0.222.
  - `kitchen_activity` recall also falls, by 0.058 [0.009, 0.104].
  - `bed_awake` occurs in only 5 homes. Its recall falls in 3 and is unchanged
    in 2.
  - Every other state's mean change is below the minimal difference.
    `bathroom_activity` is uncertain.
- **Log loss improves; the other probability metrics do not.**
  - The history state lowers H1 log loss by 0.726 [0.537, 0.929]. The median
    falls from 3.756 to 3.120.
  - The original model's log loss worsens by 0.249 when history is added (R1).
    With the history state, it improves by 0.478 (H2).
  - Brier score and calibration error do not improve: on H1 they are uncertain
    and negligible. Against `I0` both still worsen (H2).
  - Median log loss stays about twice the no-information 1.510.
- **The fitted coefficients are positive almost everywhere.**
  - In both folds, nine of the eleven groups lie between 0.76 and 1.80.
  - `bathroom_activity` in its own room is near zero (0.24 and −0.16).
  - `bed_awake` in its own room, with 7 and 61 windows of support, changes
    sign between folds.
  - Of the three states that read every room, `away` has the largest
    coefficient (1.34 and 1.41) and `home_active` the smallest (0.87 and 0.76).
  - So recent activity raises the expected current rate in nearly every state.
    It explains activity more than it separates states.
- **One reading, not tested here.**
  - After recent activity, `away`'s expected rate rises more than
    `home_active`'s. Current activity then separates the two less, which fits
    the shift from `home_active` to `away`.
  - The coefficients are fitted against the declared rates, so they may also
    absorb errors in those rates.
  - This experiment cannot separate the two explanations.
- **The formulation gap widens.**
  - At `I2` the diagnostic leads the history-state model by +0.158
    [+0.132, +0.182], in 19 of 20 homes (H3).
  - Logistic regression's mean at `I2`, 0.383, is close to the original
    model's, 0.390. It leads the history-state model by +0.035
    [+0.004, +0.063] (H4).
  - At `I3` the diagnostic leads by +0.123 [+0.091, +0.148] (S3). Logistic
    regression's lead, +0.031 [−0.009, +0.065], is uncertain (S4).
- **Time and history overlap slightly in every family.**
  - The history gain is smaller when the hour is known in all four families:
    by 0.013 to 0.021.
  - Only the family with the history state reaches the minimal difference
    (−0.021 [−0.027, −0.015]). The other three are uncertain.
  - Logistic regression shows the same overlap, and the diagnostic's estimate
    points the same way. So the overlap is not specific to the generative
    formulation.

**Not shown:**

- This is the development panel, which earlier work has inspected, so it is
  not a held-out or external claim.
- There is one seed, one declared history window (`k = 3`) and one declared
  coefficient precision. No sensitivity analysis was pre-specified, and none is
  run after seeing the result.
- Whether coefficients fitted after the state rates themselves are fitted
  would behave differently is a separate hypothesis. It would need its own
  protocol.
- No model abstains, so the figures are not comparable with the pipeline's
  abstaining output.

## Reproducing it

```text
python scripts/run_phase3_history.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes the record and a Markdown
summary generated from the written record.
