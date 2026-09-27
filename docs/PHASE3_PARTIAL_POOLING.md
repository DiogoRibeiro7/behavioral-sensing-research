# Phase 3.4: partial pooling of household parameters

The [fitted hurdle channel models](PHASE3_FITTED_RATES.md) give every household
its fold's population parameters. Households differ a lot in how often each
channel is silent. The [partial-pooling framework](PARTIAL_POOLING.md) lets a
household move toward its own values as it accumulates labelled data:

```text
θ̂ = θ_pop + w · (θ_raw − θ_pop),     w = n / (n + κ)
```

This pre-specified experiment measures two things on households that none of
the parameters were fitted on:

- whether pooling improves on the population;
- whether fitting each household on its own overfits small homes.

**Status: pre-specified, development panel.** Every setting, comparison and
criterion was frozen before any household was scored. The development homes
have been inspected in earlier work, so this is not a final held-out claim. The
frozen external cohort is not touched.

## Protocol

The protocol is `artifacts/phase3/pooling_protocol.json`. The script refuses to
run if the code's protocol differs from it by a single value.

- **Households.** The 20 development homes and the two frozen Phase 1 folds.
  Each home is held out once. Its fold's training homes fit the population and
  select the pooling strength.
- **Models.** They share the prior, transition and channels, and differ only in
  the hurdle channel parameters:

  | Model | Channel parameters |
  | --- | --- |
  | `declared` | the declared rates, for context |
  | `population` | fitted per fold on the training homes, as in the Phase 3.3 follow-up |
  | `pooled` | the household pooled toward the population, `κ = 288` windows (24 labelled hours) |
  | `pooled_selected` | the same, with `κ` selected on the training homes |
  | `unconstrained` | the household's own estimates, with `κ = 0.5` |

  - **Pooled parameters.** Pooling is applied to each channel's silence
    probability and active-window mean in every state.
  - **Unconstrained.** Half a window keeps a probability off 0 and 1. A cell
    without household data takes the population value.
- **Settings.** Each model is scored in `I0`, the current windows, and in `R`,
  the filter's recursion over every window. Both are online.
- **Adaptation arms.**
  - Each held-out household adapts on its labelled windows in the first days
    of its recording, and is scored on its labelled windows after them.
  - Every model in an arm is scored on exactly those windows.
  - `week` (7 days) is the Phase 3.1 window.
  - `day` (1 day) leaves most cells with little data, which is where
    overfitting would show.
  - Nothing is compared across arms.
- **Strength selection.**
  - **Procedure.** For each fold, each training home is left out in turn. The
    population is fitted on the other training homes, and the left-out home is
    pooled with its week-arm window. Its labelled windows after the window are
    scored in `I0`.
  - **Choice.** The strength with the smallest mean log loss over the grid
    24, 72, 288, 1152 and 4608 windows is selected, with ties going to the
    larger strength.
  - **No leakage.** Held-out homes are never used.
- **Eligibility.**
  - A home enters an arm with at least 12 labelled windows in its adaptation
    window, and at least 288 after it.
  - Otherwise it is reported and left out of every model in that arm.
  - A run with fewer than two eligible homes in an arm is refused.
- **Stratification.** Within each arm, the eligible homes are sorted by
  labelled adaptation windows and split at the median into `less data` and
  `more data`. Every estimand is also reported within each half.
- **Metrics.** Household balanced accuracy, log loss, Brier score, expected
  calibration error over 10 bins, and per-state recall.
- **Bootstrap.** 10,000 household resamples, with 95% percentile intervals for
  the mean and the median paired difference. The seed is 0.

### Estimands

Each is a paired household difference within one setting and arm. A positive
value favours the first model.

| Key | Role | First | Second | Setting, arm |
| --- | --- | --- | --- | --- |
| P1 | primary | `pooled` | `population` | `I0`, week |
| P2 | primary | `pooled` | `population` | `R`, week |
| O1 | primary | `pooled` | `unconstrained` | `I0`, day |
| O2 | secondary | `pooled` | `unconstrained` | `I0`, week |
| O3 | secondary | `pooled` | `unconstrained` | `R`, day |
| S1 | secondary | `pooled_selected` | `population` | `I0`, week |
| S2 | secondary | `pooled_selected` | `population` | `R`, week |
| S3 | secondary | `pooled` | `population` | `I0`, day |
| S4 | secondary | `unconstrained` | `population` | `I0`, week |
| S5 | secondary | `unconstrained` | `population` | `I0`, day |
| C1 | context | `population` | `declared` | `I0`, week |
| C2 | context | `population` | `declared` | `R`, week |

Per-state recall is reported for P1 and O1.

For each household, the record keeps the pooled parameters of both arms. For
every channel, state and parameter, that is:

- the household's sample size;
- its raw estimate;
- the population estimate;
- the pooled estimate;
- the effective shrinkage.

It also keeps the household's performance change in every estimand.

### Criteria

Verdicts compare effect sizes and household intervals with the Phase 3
minimal important differences. They are not significance tests.

| Metric | `δ` |
| --- | ---: |
| balanced accuracy | 0.02 |
| log loss | 0.05 |
| Brier score | 0.01 |
| calibration error | 0.02 |
| per-state recall | 0.05 |

Log loss is the primary metric. Adaptation changes each household's
probability model, so a proper scoring rule is the target. Balanced accuracy
and calibration error are guards.

- **Pooling rule, for P1, P2 and the other pooling estimands:**
  - **Success.** Log loss favours the adapted model, and neither balanced
    accuracy nor calibration error favours the reference.
  - **Trade-off.** Log loss favours the adapted model, and balanced accuracy
    favours the reference.
  - **Failure.** Log loss is negligible, or favours the reference.
  - **Inconclusive.** Anything else.
- **Overfitting rule, for O1 to O3:**
  - **Overfits.** Log loss favours the pooled model over the unconstrained one.
  - **Does not overfit.** Log loss is negligible, or favours the unconstrained
    model.
  - **Inconclusive.** Anything else.

## Results

The summary below is generated from the published record,
`artifacts/phase3/phase3-partial-pooling.json`, by `render_summary`.
A test checks that it still matches the record.

<!-- generated-summary:start -->

### Partial pooling of household parameters: summary

Generated from the record `phase3-partial-pooling`: protocol `ac77fa9716ec`, commit `e497cdccf9f7`. Status: pre-specified; development panel, which earlier work has inspected.

- Inference regime: online filter (declared by the experiment that wrote the record).
- 20 held-out households in 2 cross-fitted folds. Each is scored once, by a population and a selected strength fitted without it.
- Two adaptation arms: 7 days (`week`) and 1 day (`day`). Within an arm, every model is scored on the same labelled windows, after the household's adaptation window. Nothing is compared across arms.
- The declared pooling strength is 288 windows; the unconstrained model uses 0.5.
- Differences are paired by household, with the mean and the median and their 95% household bootstrap intervals from 10,000 resamples. A positive value favours the first model.
- Minimal important differences: balanced_accuracy 0.02, brier 0.01, calibration_error 0.02, log_loss 0.05, recall 0.05.

#### Pre-specified conclusions

| Question | Conclusion |
| --- | ---: |
| Does partial pooling improve on the population, current windows? (P1) | success |
| And in the recursion? (P2) | inconclusive |
| Does unconstrained per-home fitting overfit small homes? (O1) | overfits |

Pooling success: log loss favours the adapted model, and neither balanced accuracy nor calibration error favours the reference. Trade-off: log loss favours the adapted model and balanced accuracy favours the reference. Failure: log loss is negligible or favours the reference.

Overfits: log loss favours the pooled model over the unconstrained one. Does not overfit: log loss is negligible or favours the unconstrained model.

Verdicts: favours model: mean >= delta and lower bound > 0; favours reference: mean <= -delta and upper bound < 0; negligible: the whole interval within (-delta, delta); uncertain: anything else.

#### Strength selection

Mean log loss of the left-out training households at each strength, and the strength selected. Held-out households are never used:

| Fold | Training households | 24 | 72 | 288 | 1152 | 4608 | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1-fold-a | 10 | 1.756 | 1.782 | 1.826 | 1.868 | 1.893 | 24 |
| phase1-fold-b | 10 | 1.640 | 1.641 | 1.666 | 1.708 | 1.742 | 24 |

#### Every cell

Balanced accuracy as the median, then the mean with its interval. The other metrics are medians, and lower is better:

| Model | Setting | Arm | Households | Balanced accuracy | Log loss | Brier | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| declared | I0 | week | 20 | 0.364 (0.367 [0.342, 0.391]) | 3.362 | 0.801 | 0.231 |
| declared | R | week | 20 | 0.423 (0.413 [0.383, 0.442]) | 4.624 | 0.877 | 0.377 |
| population | I0 | week | 20 | 0.387 (0.380 [0.359, 0.401]) | 1.830 | 0.725 | 0.166 |
| population | R | week | 20 | 0.468 (0.452 [0.413, 0.484]) | 2.090 | 0.702 | 0.252 |
| pooled | I0 | week | 20 | 0.392 (0.390 [0.368, 0.411]) | 1.815 | 0.709 | 0.167 |
| pooled | R | week | 20 | 0.480 (0.469 [0.444, 0.493]) | 1.968 | 0.682 | 0.249 |
| pooled_selected | I0 | week | 20 | 0.406 (0.393 [0.364, 0.416]) | 1.730 | 0.702 | 0.162 |
| pooled_selected | R | week | 20 | 0.485 (0.488 [0.469, 0.507]) | 1.852 | 0.638 | 0.221 |
| unconstrained | I0 | week | 20 | 0.402 (0.391 [0.366, 0.412]) | 1.739 | 0.706 | 0.167 |
| unconstrained | R | week | 20 | 0.507 (0.491 [0.468, 0.512]) | 1.950 | 0.614 | 0.219 |
| declared | I0 | day | 20 | 0.362 (0.368 [0.346, 0.391]) | 3.412 | 0.801 | 0.231 |
| declared | R | day | 20 | 0.422 (0.416 [0.390, 0.442]) | 4.625 | 0.882 | 0.379 |
| population | I0 | day | 20 | 0.392 (0.378 [0.354, 0.400]) | 1.849 | 0.729 | 0.158 |
| population | R | day | 20 | 0.470 (0.453 [0.419, 0.484]) | 2.112 | 0.707 | 0.248 |
| pooled | I0 | day | 20 | 0.394 (0.380 [0.357, 0.402]) | 1.831 | 0.725 | 0.166 |
| pooled | R | day | 20 | 0.476 (0.454 [0.421, 0.483]) | 2.184 | 0.703 | 0.245 |
| unconstrained | I0 | day | 20 | 0.389 (0.379 [0.360, 0.398]) | 2.009 | 0.734 | 0.183 |
| unconstrained | R | day | 20 | 0.472 (0.469 [0.448, 0.489]) | 2.376 | 0.685 | 0.258 |

#### Estimands: log loss

The primary metric. Positive values favour the first model, whose log loss is lower:

| Key | Comparison | Mean difference | Median difference | Homes improved / worsened | Verdict | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | pooled@I0/week vs population@I0/week | +0.080 [+0.039, +0.118] | +0.085 [+0.050, +0.119] | 17 / 3 | favours model | success |
| P2 | pooled@R/week vs population@R/week | +0.314 [−0.192, +1.104] | +0.168 [−0.077, +0.279] | 14 / 6 | uncertain | inconclusive |
| O1 | pooled@I0/day vs unconstrained@I0/day | +0.175 [+0.055, +0.302] | +0.160 [+0.030, +0.274] | 14 / 6 | favours model | overfits |
| O2 | pooled@I0/week vs unconstrained@I0/week | −0.027 [−0.106, +0.050] | −0.006 [−0.112, +0.095] | 10 / 10 | uncertain | inconclusive |
| O3 | pooled@R/day vs unconstrained@R/day | +0.376 [+0.129, +0.627] | +0.309 [+0.035, +0.719] | 14 / 6 | favours model | overfits |
| S1 | pooled_selected@I0/week vs population@I0/week | +0.132 [+0.035, +0.232] | +0.109 [−0.003, +0.223] | 14 / 6 | favours model | success |
| S2 | pooled_selected@R/week vs population@R/week | +0.361 [−0.220, +1.185] | +0.204 [−0.225, +0.509] | 14 / 6 | uncertain | inconclusive |
| S3 | pooled@I0/day vs population@I0/day | +0.014 [+0.002, +0.024] | +0.015 [+0.006, +0.034] | 16 / 4 | negligible | failure |
| S4 | unconstrained@I0/week vs population@I0/week | +0.107 [−0.004, +0.221] | +0.116 [−0.034, +0.240] | 12 / 8 | uncertain | inconclusive |
| S5 | unconstrained@I0/day vs population@I0/day | −0.161 [−0.293, −0.035] | −0.157 [−0.255, −0.008] | 6 / 14 | favours reference | failure |
| C1 | population@I0/week vs declared@I0/week | +1.624 [+1.361, +1.892] | +1.520 [+1.270, +1.838] | 20 / 0 | favours model | success |
| C2 | population@R/week vs declared@R/week | +2.410 [+2.153, +2.666] | +2.538 [+2.113, +2.776] | 20 / 0 | favours model | success |

#### Estimands: the guards

Mean differences, homes improved and worsened, and verdicts:

| Key | Balanced accuracy | Brier | Calibration error |
| --- | ---: | ---: | ---: |
| P1 | +0.009 [+0.003, +0.017], 13 / 7, negligible | +0.020 [+0.011, +0.031], 18 / 2, favours model | +0.002 [−0.006, +0.011], 11 / 9, negligible |
| P2 | +0.017 [−0.012, +0.060], 10 / 10, uncertain | +0.080 [−0.063, +0.301], 14 / 6, uncertain | +0.034 [−0.043, +0.147], 14 / 6, uncertain |
| O1 | +0.001 [−0.021, +0.022], 13 / 7, uncertain | +0.009 [−0.009, +0.028], 11 / 9, uncertain | +0.008 [−0.012, +0.027], 13 / 7, uncertain |
| O2 | −0.001 [−0.017, +0.017], 9 / 11, negligible | −0.015 [−0.032, −0.000], 7 / 13, favours reference | −0.011 [−0.029, +0.005], 9 / 11, uncertain |
| O3 | −0.015 [−0.047, +0.012], 9 / 11, uncertain | +0.009 [−0.043, +0.056], 10 / 10, uncertain | +0.017 [−0.013, +0.044], 12 / 8, uncertain |
| S1 | +0.012 [−0.008, +0.029], 13 / 7, uncertain | +0.032 [+0.016, +0.049], 17 / 3, favours model | +0.012 [−0.006, +0.032], 11 / 9, uncertain |
| S2 | +0.036 [+0.004, +0.077], 12 / 8, favours model | +0.108 [−0.046, +0.330], 14 / 6, uncertain | +0.048 [−0.035, +0.162], 13 / 7, uncertain |
| S3 | +0.002 [−0.001, +0.005], 13 / 7, negligible | +0.005 [+0.002, +0.009], 17 / 3, negligible | −0.003 [−0.008, +0.002], 11 / 9, negligible |
| S4 | +0.010 [−0.009, +0.029], 12 / 8, uncertain | +0.036 [+0.014, +0.057], 16 / 4, favours model | +0.014 [−0.007, +0.037], 12 / 8, uncertain |
| S5 | +0.001 [−0.021, +0.024], 8 / 12, uncertain | −0.003 [−0.024, +0.015], 10 / 10, uncertain | −0.011 [−0.032, +0.012], 7 / 13, uncertain |
| C1 | +0.014 [−0.005, +0.034], 9 / 11, uncertain | +0.064 [+0.037, +0.093], 17 / 3, favours model | +0.060 [+0.031, +0.089], 16 / 4, favours model |
| C2 | +0.039 [+0.014, +0.064], 15 / 5, favours model | +0.184 [+0.128, +0.246], 19 / 1, favours model | +0.124 [+0.091, +0.160], 20 / 0, favours model |

#### By amount of adaptation data

Within each arm, the eligible households are split at the median number of labelled adaptation windows. Log loss mean difference, homes improved and worsened, and conclusion:

| Key | Stratum | Households | Log loss | Homes | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1 | less data | 10 | +0.102 [+0.049, +0.159] | 8 / 2 | success |
| P1 | more data | 10 | +0.058 [−0.002, +0.099] | 9 / 1 | inconclusive |
| P2 | less data | 10 | −0.245 [−0.614, +0.076] | 5 / 5 | inconclusive |
| P2 | more data | 10 | +0.872 [+0.124, +2.243] | 9 / 1 | success |
| O1 | less data | 10 | +0.048 [−0.106, +0.176] | 5 / 5 | inconclusive |
| O1 | more data | 10 | +0.302 [+0.147, +0.480] | 9 / 1 | overfits |

#### Per-state recall

Mean differences, homes improved and worsened, and verdicts:

| State | Rare | P1 | O1 |
| --- | ---: | ---: | ---: |
| away | no | +0.002 [+0.000, +0.004], 14 / 3, negligible | −0.012 [−0.020, −0.005], 4 / 16, negligible |
| home_active | no | +0.004 [−0.008, +0.015], 13 / 7, negligible | +0.058 [+0.028, +0.088], 15 / 5, favours model |
| home_inactive | no | +0.043 [+0.021, +0.066], 15 / 3, uncertain | −0.016 [−0.046, +0.015], 8 / 11, negligible |
| sleeping | no | −0.008 [−0.016, −0.001], 9 / 9, negligible | −0.013 [−0.029, +0.002], 8 / 11, negligible |
| bed_awake | yes | +0.067 [+0.000, +0.200], 1 / 0, uncertain | +0.011 [−0.189, +0.200], 2 / 1, uncertain |
| bathroom_activity | yes | −0.000 [−0.020, +0.014], 12 / 5, negligible | +0.067 [−0.024, +0.165], 10 / 10, uncertain |
| kitchen_activity | yes | +0.002 [−0.014, +0.024], 6 / 9, negligible | −0.079 [−0.162, −0.001], 10 / 10, favours reference |

#### Pooled silence by state

For the silence probability pooled with the declared strength, across households and channels: the median number of household windows, the median effective shrinkage, and the median distance of the raw and the pooled estimate from the population's. Week arm, then day arm:

| State (week) | Windows | Shrinkage | Raw from population | Pooled from population |
| --- | ---: | ---: | ---: | ---: |
| away | 494.000 | 0.368 | 0.013 | 0.006 |
| home_active | 182.000 | 0.613 | 0.080 | 0.027 |
| home_inactive | 267.000 | 0.519 | 0.051 | 0.025 |
| sleeping | 693.000 | 0.294 | 0.005 | 0.003 |
| bed_awake | 0.000 | 1.000 | 0.197 | 0.001 |
| bathroom_activity | 58.000 | 0.832 | 0.106 | 0.017 |
| kitchen_activity | 48.000 | 0.857 | 0.082 | 0.010 |

| State (day) | Windows | Shrinkage | Raw from population | Pooled from population |
| --- | ---: | ---: | ---: | ---: |
| away | 52.000 | 0.847 | 0.024 | 0.004 |
| home_active | 21.000 | 0.932 | 0.109 | 0.007 |
| home_inactive | 37.000 | 0.886 | 0.071 | 0.006 |
| sleeping | 100.000 | 0.742 | 0.009 | 0.002 |
| bed_awake | 0.000 | 1.000 | n/a | n/a |
| bathroom_activity | 8.000 | 0.973 | 0.129 | 0.004 |
| kitchen_activity | 5.000 | 0.983 | 0.162 | 0.002 |

#### Households

Each household's labelled adaptation windows per arm (an asterisk marks a household left out of that arm) and its log loss change in the primary estimands:

| Household | Fold | Week windows | Day windows | P1 | P2 | O1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hh101 | phase1-fold-a | 1853 | 177 | +0.060 | +0.045 | −0.027 |
| hh102 | phase1-fold-b | 1824 | 245 | −0.011 | −0.727 | +0.208 |
| hh103 | phase1-fold-a | 1752 | 273 | +0.105 | +0.262 | +0.065 |
| hh105 | phase1-fold-b | 1792 | 242 | −0.002 | −1.541 | +0.164 |
| hh106 | phase1-fold-a | 1785 | 229 | +0.175 | −0.438 | −0.074 |
| hh108 | phase1-fold-b | 1792 | 246 | +0.011 | −0.355 | +0.280 |
| hh110 | phase1-fold-a | 1887 | 259 | +0.125 | +0.216 | +0.157 |
| hh111 | phase1-fold-b | 1935 | 279 | +0.136 | +0.411 | +0.310 |
| hh114 | phase1-fold-a | 1855 | 268 | +0.108 | +0.408 | +0.111 |
| hh118 | phase1-fold-b | 1970 | 282 | −0.186 | −0.166 | +0.528 |
| hh119 | phase1-fold-a | 1808 | 244 | +0.169 | +0.275 | −0.017 |
| hh120 | phase1-fold-b | 1930 | 273 | +0.074 | +0.282 | +0.549 |
| hh122 | phase1-fold-a | 1877 | 270 | +0.067 | +0.300 | +0.090 |
| hh123 | phase1-fold-b | 1943 | 274 | +0.063 | +0.149 | +0.306 |
| hh124 | phase1-fold-a | 1917 | 230 | +0.095 | +6.887 | +0.268 |
| hh125 | phase1-fold-b | 1765 | 238 | +0.040 | +0.012 | +0.262 |
| hh126 | phase1-fold-a | 1336 | 181 | +0.141 | +0.138 | −0.071 |
| hh127 | phase1-fold-b | 996 | 26 | +0.281 | +0.344 | −0.510 |
| hh129 | phase1-fold-a | 1801 | 262 | +0.113 | −0.417 | −0.006 |
| hh130 | phase1-fold-b | 1876 | 258 | +0.034 | +0.188 | +0.912 |

<!-- generated-summary:end -->

### What this shows and does not show

Figures are mean paired log-loss differences across the 20 homes, with 95%
household intervals and the number of homes improved, unless stated otherwise.
Every home was eligible in both arms.

- **The run builds on the published fit.** In both folds, the population is
  identical, by digest, to the fitted population of the
  [Phase 3.3 follow-up](PHASE3_FITTED_RATES.md). It again improves on the
  declared rates by a wide margin (C1, C2).
- **Pooling a week of household data helps with current windows: pre-specified
  success (P1).**
  - It improves log loss by 0.080 [0.039, 0.118], in 17 of 20 homes, and the
    Brier score by 0.020.
  - Balanced accuracy and calibration error do not move beyond their minimal
    differences.
- **In the recursion the effect is inconclusive (P2).**
  - The mean is +0.314 [−0.192, +1.104], and the median +0.168
    [−0.077, +0.279], in 14 of 20 homes.
  - One home, `hh124`, improves by 6.9 and carries the mean. Without it the
    mean change is −0.032.
  - Four homes worsen by more than 0.4.
  - In the recursion, adapting a home's silence probabilities changes how its
    quiet periods accumulate, and that cuts both ways.
- **Unconstrained per-home fitting overfits small homes: pre-specified
  "overfits" (O1).**
  - After one day, the unconstrained model is worse than the pooled one by
    0.175 [0.055, 0.302], in 14 of 20 homes. It is worse in the recursion too
    (O3).
  - It is worse than using the population alone, by 0.161 [0.035, 0.293]
    (S5).
  - After a week the two are close, −0.027 [−0.106, +0.050], an uncertain
    verdict (O2).
- **After a day, the declared pooling strength barely moves a home.**
  - Pooled against the population after one day is negligible, +0.014 (S3).
  - The median effective shrinkage is 0.74 to 0.98, depending on the state.
- **The selected strength is the smallest in the grid.**
  - On the training homes, mean log loss rose with the strength in both folds,
    so leave-one-household-out chose 24 windows (two labelled hours) both
    times.
  - With it, pooling improves log loss by 0.132 [0.035, 0.232] with current
    windows (S1).
  - A week of data therefore supports weaker pooling than the declared 288.
    The best strength may lie below the grid.
- **Shrinkage in the week arm.** For the silence probability, the median
  effective shrinkage ranges by state:
  - 0.29 for `sleeping`, whose household estimates are close to the
    population's anyway;
  - 0.61 for `home_active`;
  - 0.83 to 0.86 for the rare bathroom and kitchen states.
- **By amount of adaptation data.**
  - With current windows, pooling helps in both halves: +0.102 in the half
    with less data, and +0.058 [−0.002, +0.099] in the half with more.
  - After a day, the overfitting shows in the half with more data (+0.302).
    In the half with less data it is inconclusive, and that half includes one
    home with only 26 labelled windows.
  - Each half has 10 homes, so these intervals are wide.
- **Per-state recall.** Pooling changes it little. `home_inactive` gains
  0.043, an uncertain verdict.

**Not shown:**

- This is the development panel, which earlier work has inspected, so it is
  not a held-out or external claim.
- There is one seed.
- The strength was selected on the week arm only. Whether 24 windows also
  suits a single day was not tested.
- The declared grid stopped at 24. Weaker pooling than that was not tested,
  except the unconstrained limit.
- The online production filter is not changed.
- The recursion here has full reliability and no health or attribution layer.

## Reproducing it

```text
python scripts/run_phase3_pooling.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes the record and a Markdown
summary generated from the written record.
