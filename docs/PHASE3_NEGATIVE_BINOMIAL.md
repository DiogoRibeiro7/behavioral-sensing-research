# Phase 3.3 follow-up: the hurdle negative binomial against the hurdle-Poisson

The [posterior predictive checks](PHASE3_PREDICTIVE_CHECKS.md) found the fitted
hurdle model's zero-truncated Poisson active count systematically
under-dispersed. The [hurdle negative binomial](HURDLE_NEGATIVE_BINOMIAL.md)
keeps the hurdle's silence and gives the active count a zero-truncated negative
binomial. This pre-specified experiment measures whether that helps on
households none of the parameters were fitted on.

Those checks also found long quiet runs in systematic excess. Their
pre-declared routing named within-state temporal dependence as the next model
family, which a count distribution cannot address. This experiment measures
what the over-dispersed count does on its own.

**Status: pre-specified, development panel.** Every setting, comparison and
criterion was frozen before any household was scored. The development homes
have been inspected in earlier work, so this is not a final held-out claim. The
frozen external cohort is not touched.

## Protocol

The protocol is `artifacts/phase3/dispersion_protocol.json`. The script refuses
to run if the code's protocol differs from it by a single value.

- **Households.** The 20 development homes and the two frozen Phase 1 folds.
  Each home is held out once, and scored by channel models fitted on its fold's
  training homes.
- **Models.** Three channel observation models:

  | Model | Channels |
  | --- | --- |
  | `declared` | the declared Poisson rates |
  | `hurdle` | the fitted hurdle-Poisson, the Phase 3.3 follow-up's model |
  | `hurdle_nb` | the fitted hurdle negative binomial |

  They have the same prior, transition, channels, windows and inference, and
  differ only in each channel's observation model. The hurdle and the negative
  binomial share the silence probability and the active mean. The negative
  binomial adds a dispersion per channel and state, fitted on the training
  homes and shrunk toward the Poisson with 12 pseudo-windows.
- **Settings.**
  - **`I0`.** The current windows.
  - **`R`.** The filter's recursion over every window of the recording. Every
    model receives the same windows.
- **Metrics.**
  - Household balanced accuracy, per-state recall, log loss, Brier score, and
    expected calibration error over 10 bins.
  - **Quiet-run overconfidence.** Phase 3.3's accumulation slope: the slope per
    hour of confidence minus correctness along runs of fully silent windows,
    within the predicted state. It is measured in the recursion only. A fully
    silent window has the same likelihood under both hurdle families, so with
    current windows their slopes coincide.
- **Bootstrap.** 10,000 household resamples, with 95% percentile intervals for
  the mean and the median paired difference. The seed is 0.

### Estimands

Each is a paired household difference within one setting. A positive value
favours the first model.

| Key | Role | First | Second | Setting |
| --- | --- | --- | --- | --- |
| N0 | primary | `hurdle_nb` | `hurdle` | `I0` |
| NR | primary | `hurdle_nb` | `hurdle` | `R` |
| D0 | secondary | `hurdle_nb` | `declared` | `I0` |
| DR | secondary | `hurdle_nb` | `declared` | `R` |
| H0 | context | `hurdle` | `declared` | `I0` |
| HR | context | `hurdle` | `declared` | `R` |

- **The comparisons.**
  - N0 and NR answer the questions.
  - D0 and DR show whether the calibration improvement over the declared rates
    survives.
  - H0 and HR reproduce the Phase 3.3 follow-up's F0 and FR.
- **What each reports.** Per-state recall for every estimand, and the
  quiet-run slope for the recursion estimands.

### The questions and the decision

In each setting, from its primary estimand (and, for calibration, the one
against the declared rates):

| Question | Answer |
| --- | --- |
| 1. Does the negative binomial recover some of the `home_active` recall the hurdle-Poisson lost? | recovers, no change, loses more or uncertain, from the `home_active` recall verdict |
| 2. Does it preserve the calibration improvement? | preserved, lost or uncertain |
| 3. Does it improve log loss? | improves, no change, worsens or uncertain |
| 4. Does it avoid degrading balanced accuracy? | not degraded, degraded or uncertain |

**Extra complexity is not accepted because one metric improves.** The decision
in each setting is fixed in advance:

- **Adopt.** Log loss favours the negative binomial, calibration is preserved,
  balanced accuracy is not degraded, and `home_active` recall does not favour
  the hurdle-Poisson.
- **Trade-off.** Log loss favours the negative binomial, but calibration is
  lost, or balanced accuracy or `home_active` recall favours the hurdle-Poisson.
- **Reject.** Log loss is negligible or favours the hurdle-Poisson. The extra
  parameter is not accepted, whatever else improves.
- **Inconclusive.** Anything else.

Overall, the model is adopted only if adopted in both settings, and rejected
only if rejected in both. Otherwise it is not adopted, and each setting's
decision is reported.

Verdicts compare effect sizes and household intervals with the Phase 3 minimal
important differences. They are not significance tests.

| Quantity | `δ` |
| --- | ---: |
| balanced accuracy | 0.02 |
| log loss | 0.05 |
| Brier score | 0.01 |
| calibration error | 0.02 |
| per-state recall | 0.05 |
| quiet-run slope | 0.02 per hour |

### The dispersion estimates

For each fold, channel and state, the record keeps:

- the dispersion `α`, the size `1/α`, `μ` and the active mean;
- the active training windows the estimate rests on;
- its log-likelihood gain over the Poisson on the observed table;
- its **flatness**: the observed table's log-likelihood at `α` minus at `α/2`.

Each estimate is classed as:

| Class | Rule |
| --- | --- |
| poisson | `α = 0` |
| moderate | `0 < α < 10` |
| extreme, low data | `α ≥ 10`, fewer than 100 active training windows |
| extreme, flat | `α ≥ 10`, flatness below 1 nat: weakly identified |
| extreme, supported | anything else with `α ≥ 10` |

Extreme estimates *indicate insufficient data* if more than half of them are
low data. Otherwise they are not a data shortage.

## Results

The summary below is generated from the published record,
`artifacts/phase3/phase3-hurdle-negative-binomial.json`, by `render_summary`.
A test checks that it still matches the record. The run was made from the
protocol commit, `af458fa`, on a clean tree.

<!-- generated-summary:start -->

### The hurdle negative binomial against the hurdle-Poisson: summary

Generated from the record `phase3-hurdle-negative-binomial`: protocol `e85c21744e3d`, commit `af458fa2b01f`. Status: pre-specified; development panel, which earlier work has inspected.

- Inference regime: online filter; with current windows (`I0`) and the filter's recursion over every window (`R`).
- 20 held-out households in 2 cross-fitted folds, each scored once by channel models fitted without it.
- Three channel observation models, `declared`, `hurdle` and `hurdle_nb`, with the same prior, transition, channels, windows and inference.
- Differences are paired by household, with the mean and the median and their 95% household bootstrap intervals from 10,000 resamples. A positive value favours the first model.
- Minimal important differences: accumulation_slope 0.02, balanced_accuracy 0.02, brier 0.01, calibration_error 0.02, log_loss 0.05, recall 0.05.

#### Pre-specified questions

| Question | I0 | R |
| --- | ---: | ---: |
| 1. Recovers home_active recall? | loses more | uncertain |
| 2. Preserves the calibration improvement? | preserved | preserved |
| 3. Improves log loss? | improves | improves |
| 4. Avoids degrading balanced accuracy? | not degraded | not degraded |
| Decision | trade-off | adopt |

**Overall: not adopted.**

Adopt: log loss favours the negative binomial, calibration is preserved, balanced accuracy is not degraded, and home_active recall does not favour the hurdle-Poisson. Trade-off: log loss favours the negative binomial, but calibration is lost, or balanced accuracy or home_active recall favours the hurdle-Poisson. Reject: log loss is negligible or favours the hurdle-Poisson: the extra parameter is not accepted, whatever else improves. Overall: adopt if adopted in both settings, reject if rejected in both, and otherwise not adopted, with each setting's decision reported.

Verdicts: favours model: mean >= delta and lower bound > 0; favours reference: mean <= -delta and upper bound < 0; negligible: the whole interval within (-delta, delta); uncertain: anything else.

#### Every cell

Balanced accuracy as the median, then the mean with its interval. The other metrics are medians, and lower is better:

| Model | Setting | Balanced accuracy | Log loss | Brier | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: |
| declared | I0 | 0.362 (0.370 [0.347, 0.391]) | 3.403 | 0.800 | 0.233 |
| hurdle | I0 | 0.393 (0.379 [0.354, 0.402]) | 1.857 | 0.730 | 0.159 |
| hurdle_nb | I0 | 0.419 (0.422 [0.399, 0.445]) | 1.351 | 0.662 | 0.128 |
| declared | R | 0.422 (0.417 [0.392, 0.442]) | 4.624 | 0.879 | 0.377 |
| hurdle | R | 0.469 (0.456 [0.425, 0.484]) | 2.124 | 0.709 | 0.248 |
| hurdle_nb | R | 0.501 (0.499 [0.470, 0.529]) | 1.315 | 0.541 | 0.148 |

#### Estimands

Mean differences, households improved and worsened, and verdicts. Positive values favour the first model, that is, higher accuracy or lower loss and calibration error:

| Key | Comparison | balanced_accuracy | log_loss | brier | calibration_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0 | hurdle_nb@I0 vs hurdle@I0 | +0.044 [+0.028, +0.059], 18 / 2, favours model | +0.484 [+0.402, +0.562], 20 / 0, favours model | +0.059 [+0.047, +0.070], 20 / 0, favours model | +0.048 [+0.032, +0.064], 19 / 1, favours model |
| NR | hurdle_nb@R vs hurdle@R | +0.044 [+0.028, +0.060], 17 / 3, favours model | +0.784 [+0.649, +0.949], 20 / 0, favours model | +0.150 [+0.099, +0.209], 18 / 2, favours model | +0.104 [+0.073, +0.139], 19 / 1, favours model |
| D0 | hurdle_nb@I0 vs declared@I0 | +0.053 [+0.027, +0.077], 16 / 4, favours model | +2.129 [+1.838, +2.402], 20 / 0, favours model | +0.122 [+0.094, +0.153], 20 / 0, favours model | +0.106 [+0.079, +0.134], 20 / 0, favours model |
| DR | hurdle_nb@R vs declared@R | +0.082 [+0.055, +0.111], 19 / 1, favours model | +3.159 [+2.901, +3.409], 20 / 0, favours model | +0.325 [+0.250, +0.402], 20 / 0, favours model | +0.223 [+0.181, +0.265], 20 / 0, favours model |
| H0 | hurdle@I0 vs declared@I0 | +0.009 [−0.010, +0.029], 9 / 11, uncertain | +1.645 [+1.392, +1.903], 20 / 0, favours model | +0.063 [+0.038, +0.092], 18 / 2, favours model | +0.058 [+0.030, +0.087], 16 / 4, favours model |
| HR | hurdle@R vs declared@R | +0.039 [+0.015, +0.062], 16 / 4, favours model | +2.374 [+2.134, +2.613], 20 / 0, favours model | +0.175 [+0.124, +0.232], 19 / 1, favours model | +0.119 [+0.087, +0.153], 20 / 0, favours model |

#### Per-state recall

Mean differences, households improved and worsened, and verdicts:

| State | Rare | N0 | NR | D0 | DR | H0 | HR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| away | no | +0.000 [−0.001, +0.001], 7 / 9, negligible | +0.210 [+0.153, +0.273], 20 / 0, favours model | −0.002 [−0.005, −0.000], 6 / 13, negligible | +0.354 [+0.250, +0.449], 19 / 1, favours model | −0.002 [−0.005, −0.001], 4 / 14, negligible | +0.145 [+0.048, +0.234], 17 / 3, favours model |
| home_active | no | −0.055 [−0.087, −0.024], 6 / 14, favours reference | +0.029 [−0.004, +0.063], 12 / 8, uncertain | −0.284 [−0.352, −0.217], 0 / 20, favours reference | −0.202 [−0.271, −0.135], 1 / 19, favours reference | −0.229 [−0.277, −0.180], 0 / 20, favours reference | −0.231 [−0.290, −0.172], 1 / 19, favours reference |
| home_inactive | no | +0.052 [+0.017, +0.085], 14 / 6, favours model | −0.028 [−0.066, +0.006], 9 / 11, uncertain | +0.128 [+0.083, +0.176], 20 / 0, favours model | +0.191 [+0.150, +0.231], 20 / 0, favours model | +0.076 [+0.043, +0.111], 17 / 3, favours model | +0.219 [+0.169, +0.272], 20 / 0, favours model |
| sleeping | no | +0.025 [+0.015, +0.034], 19 / 0, negligible | +0.035 [−0.026, +0.091], 14 / 6, uncertain | +0.072 [+0.047, +0.102], 20 / 0, favours model | +0.017 [−0.051, +0.075], 12 / 8, uncertain | +0.047 [+0.021, +0.079], 19 / 0, uncertain | −0.018 [−0.059, +0.020], 10 / 10, uncertain |
| bed_awake | yes | −0.004 [−0.200, +0.196], 1 / 2, uncertain | −0.070 [−0.204, +0.000], 0 / 2, uncertain | −0.163 [−0.446, +0.100], 2 / 3, uncertain | −0.289 [−0.500, −0.083], 1 / 4, favours reference | −0.159 [−0.283, −0.035], 1 / 3, favours reference | −0.219 [−0.317, −0.072], 1 / 4, favours reference |
| bathroom_activity | yes | +0.144 [+0.055, +0.234], 15 / 5, favours model | +0.095 [+0.054, +0.138], 17 / 3, favours model | +0.325 [+0.240, +0.410], 19 / 1, favours model | +0.272 [+0.208, +0.338], 19 / 1, favours model | +0.181 [+0.130, +0.234], 18 / 2, favours model | +0.177 [+0.124, +0.228], 18 / 2, favours model |
| kitchen_activity | yes | +0.108 [+0.070, +0.151], 19 / 0, favours model | −0.054 [−0.099, −0.007], 6 / 13, favours reference | +0.122 [+0.009, +0.228], 14 / 6, favours model | −0.056 [−0.155, +0.034], 10 / 8, uncertain | +0.014 [−0.084, +0.108], 11 / 9, uncertain | −0.002 [−0.115, +0.102], 11 / 9, uncertain |

#### Overconfidence along quiet runs

In the recursion, the slope per hour of confidence minus correctness along runs of fully silent windows, within the predicted state. Lower is better, and a calibrated model has zero. Each model's household median, then mean with its interval:

| Model | Slope per hour |
| --- | ---: |
| declared | 0.172 (0.175 [0.118, 0.231]) |
| hurdle | 0.025 (0.023 [−0.014, 0.060]) |
| hurdle_nb | −0.003 (0.003 [−0.018, 0.022]) |

Paired differences in the recursion; a positive value means the first model's slope is lower:

| Key | Comparison | Slope |
| --- | ---: | ---: |
| NR | hurdle_nb@R vs hurdle@R | +0.021 [−0.005, +0.045], 15 / 5, uncertain |
| DR | hurdle_nb@R vs declared@R | +0.172 [+0.121, +0.222], 19 / 1, favours model |
| HR | hurdle@R vs declared@R | +0.152 [+0.114, +0.192], 20 / 0, favours model |

#### Estimated dispersion

The negative binomial's dispersion `alpha` per channel and state, fitted on each fold's training households (phase1-fold-a / phase1-fold-b). Zero is the Poisson; the bound is 100:

| Channel | away | home_active | home_inactive | sleeping | bed_awake | bathroom_activity | kitchen_activity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bathroom_motion | 1.31 / 1.45 | 1.21 / 1.41 | 1.48 / 0.92 | 0.973 / 0.786 | 0.773 / 1.51 | 0.94 / 1.51 | 0.936 / 1.26 |
| bedroom_motion | 1.36 / 2.58 | 1.14 / 1.57 | 2.26 / 1.86 | 100 / 100 | 0.317 / 2.39 | 0.882 / 2.19 | 0.859 / 1.28 |
| hall_door | 0.219 / 0.107 | 1.31 / 0.702 | 0.608 / 0.699 | 1.99 / 100 | 0 / 0 | 3.43 / 0.858 | 0.791 / 0.799 |
| hall_motion | 0 / 1.65 | 0.0892 / 1.1 | 1.33 / 0.0956 | 0 / 0 | n/a / n/a | 0.521 / 100 | 12.1 / 0 |
| kitchen_motion | 2.12 / 10.3 | 1.5 / 1.3 | 1.23 / 1.53 | 100 / 100 | 0 / 0 | 1.57 / 2.34 | 0.728 / 0.455 |
| living_motion | 1.04 / 1.68 | 1.26 / 1.32 | 2.22 / 1.11 | 100 / 2.48 | 0.495 / 0 | 0.809 / 0.555 | 0.702 / 0.707 |

Each estimate's class. Extreme is alpha of at least 10; low data is fewer than 100 active training windows; flat means the observed table's log-likelihood at alpha exceeds that at alpha / 2 by less than 1 nat: weakly identified. Channel and state estimates over both folds:

| State | poisson | moderate | extreme, low data | extreme, flat | extreme, supported |
| --- | ---: | ---: | ---: | ---: | ---: |
| away | 1 | 10 | 0 | 0 | 1 |
| home_active | 0 | 12 | 0 | 0 | 0 |
| home_inactive | 0 | 12 | 0 | 0 | 0 |
| sleeping | 2 | 4 | 0 | 4 | 2 |
| bed_awake | 5 | 5 | 0 | 0 | 0 |
| bathroom_activity | 0 | 11 | 1 | 0 | 0 |
| kitchen_activity | 1 | 10 | 1 | 0 | 0 |

Extreme estimates: 9, of which 2 low data, 4 flat and 3 supported. **Do extreme estimates indicate insufficient data? not a data shortage.**

| Fold | Channel | State | Active windows | alpha | Gain over Poisson | Flatness | Class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1-fold-a | bedroom_motion | sleeping | 5311 | 100 | 11380.038 | 1.651 | extreme, supported |
| phase1-fold-a | hall_motion | kitchen_activity | 91 | 12.1 | 7.899 | 0.066 | extreme, low data |
| phase1-fold-a | kitchen_motion | sleeping | 200 | 100 | 218.574 | 0.107 | extreme, flat |
| phase1-fold-a | living_motion | sleeping | 482 | 100 | 535.172 | 0.607 | extreme, flat |
| phase1-fold-b | bedroom_motion | sleeping | 3844 | 100 | 9699.222 | 2.509 | extreme, supported |
| phase1-fold-b | hall_door | sleeping | 101 | 100 | 54.082 | 0.116 | extreme, flat |
| phase1-fold-b | hall_motion | bathroom_activity | 5 | 100 | 1.238 | 0.006 | extreme, low data |
| phase1-fold-b | kitchen_motion | away | 769 | 10.3 | 2842.711 | 2.812 | extreme, supported |
| phase1-fold-b | kitchen_motion | sleeping | 300 | 100 | 575.823 | 0.377 | extreme, flat |

#### Households

Each household's change with the negative binomial against the hurdle-Poisson: balanced accuracy and log loss in each setting, and the quiet-run slope in the recursion. Positive favours the negative binomial:

| Household | Fold | N0 balanced accuracy | N0 log loss | NR balanced accuracy | NR log loss | NR slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hh101 | phase1-fold-a | +0.048 | +0.484 | +0.031 | +0.636 | +0.009 |
| hh102 | phase1-fold-b | −0.006 | +0.319 | +0.039 | +0.558 | +0.052 |
| hh103 | phase1-fold-a | +0.112 | +0.379 | +0.023 | +0.548 | +0.063 |
| hh105 | phase1-fold-b | −0.014 | +0.278 | −0.025 | +0.457 | +0.002 |
| hh106 | phase1-fold-a | +0.065 | +0.668 | +0.080 | +1.787 | +0.109 |
| hh108 | phase1-fold-b | +0.065 | +0.356 | +0.039 | +0.506 | +0.012 |
| hh110 | phase1-fold-a | +0.073 | +0.732 | +0.077 | +1.055 | −0.004 |
| hh111 | phase1-fold-b | +0.033 | +0.396 | +0.078 | +0.728 | +0.079 |
| hh114 | phase1-fold-a | +0.022 | +0.585 | +0.027 | +0.789 | −0.054 |
| hh118 | phase1-fold-b | +0.034 | +0.429 | +0.060 | +0.723 | +0.055 |
| hh119 | phase1-fold-a | +0.024 | +0.591 | +0.060 | +0.899 | +0.025 |
| hh120 | phase1-fold-b | +0.013 | +0.656 | +0.062 | +0.998 | +0.029 |
| hh122 | phase1-fold-a | +0.067 | +0.718 | +0.131 | +1.577 | +0.125 |
| hh123 | phase1-fold-b | +0.040 | +0.479 | +0.075 | +0.716 | +0.032 |
| hh124 | phase1-fold-a | +0.131 | +0.067 | +0.076 | +0.683 | +0.003 |
| hh125 | phase1-fold-b | +0.000 | +0.393 | −0.008 | +0.602 | −0.085 |
| hh126 | phase1-fold-a | +0.052 | +0.606 | −0.001 | +0.580 | −0.079 |
| hh127 | phase1-fold-b | +0.024 | +0.726 | +0.028 | +0.650 | −0.074 |
| hh129 | phase1-fold-a | +0.046 | +0.584 | +0.009 | +0.826 | +0.054 |
| hh130 | phase1-fold-b | +0.043 | +0.230 | +0.009 | +0.370 | +0.057 |

<!-- generated-summary:end -->

### What this shows and does not show

Figures are mean paired differences across the 20 homes, with 95% household
intervals and the number of homes improved, unless stated otherwise.

- **The run reproduces the Phase 3.3 follow-up.** The fold populations are
  identical by digest. The declared and hurdle cells, with current windows and
  in the recursion, reproduce that record's household metrics exactly. Only
  the negative binomial is new.
- **By the pre-specified rule, the negative binomial is not adopted.**
  - **The recursion: adopt.** Every guard is met.
  - **Current windows: trade-off.** `home_active` recall favours the
    hurdle-Poisson.
  - **Overall.** Adoption was declared to need both settings.
- **Log loss improves substantially: question 3, yes.**
  - With current windows, by 0.484 [0.402, 0.562], and in the recursion by
    0.784 [0.649, 0.949], in all 20 homes both times.
  - The median log loss in the recursion falls from 2.124 to 1.315. With
    current windows it falls from 1.857 to 1.351.
- **Calibration improves further: question 2, preserved.**
  - Calibration error improves over the hurdle-Poisson by 0.048 with current
    windows and 0.104 in the recursion, in 19 of 20 homes each time.
  - Against the declared rates it improves by 0.106 and 0.223. The median in
    the recursion is 0.148, against 0.248 for the hurdle-Poisson and 0.377
    for the declared rates.
- **Balanced accuracy improves: question 4, not degraded.** It improves by
  0.044 in both settings, in 18 and 17 of 20 homes. The mean in the recursion
  is 0.499, against 0.456 for the hurdle-Poisson.
- **It does not recover `home_active` recall: question 1, no.**
  - **Current windows.** Recall falls a further 0.055 [0.024, 0.087], in 14
    of 20 homes. The mean falls from 0.238 with the hurdle-Poisson to 0.183.
  - **The recursion.** The change is uncertain, +0.029 [−0.004, +0.063].
  - **Against the declared rates.** `home_active` recall remains 0.28 lower
    with current windows and 0.20 lower in the recursion.
- **Where the accuracy comes from.** In the recursion:
  - `away` recall rises by 0.210, in all 20 homes, and `bathroom_activity` by
    0.095.
  - `kitchen_activity` falls by 0.054.

  With current windows:
  - `home_inactive` gains 0.052, `bathroom_activity` 0.144 and
    `kitchen_activity` 0.108;
  - `home_active` loses 0.055.
- **Quiet runs.** The negative binomial's overconfidence slope along quiet runs
  is 0.003 per hour, against 0.023 for the hurdle-Poisson and 0.175 for the
  declared rates. The paired difference with the hurdle-Poisson, +0.021
  [−0.005, +0.045], is uncertain.
- **The dispersion estimates.**
  - **Typical values.** Most estimates are moderate: 64 of 82 channel and
    state estimates over both folds. Their median is 1.23. In `away`,
    `home_active` and `home_inactive` they run from 0.09 to 2.6.
  - **Extreme values are not a data shortage.** Of the 9 extreme estimates, 2
    rest on little data: `hall_motion`, with 91 and 5 active windows. The
    other 7 rest on 101 to 5,311 windows. 4 are flat and 3 supported.
  - **`sleeping` holds 6 of the 9.** Motion counts there have many single
    activations and a long tail, and the likelihood rises toward the
    logarithmic-series limit. For `bedroom_motion` in `sleeping` the gain over
    the Poisson is about 10,000 nats in each fold.
- **What the result means.** The negative binomial is a large improvement in
  probability quality in both settings, and a gain in balanced accuracy.
  - **What it does not do.** It does not undo the `home_active` loss. The two
    hurdle families share the silence parameter, and changing the active count
    alone does not reach that loss.
  - **Why it is not adopted.** With current windows it deepens that loss, so
    the rule declared in advance does not adopt it.
  - **A possible explanation.** The loss is consistent with the long quiet runs
    within `home_active` found by the predictive checks, a temporal pattern no
    count distribution addresses. That is not tested here.

**Not shown:**

- This is the development panel, which earlier work has inspected, so it is
  not a held-out or external claim.
- There is one seed and one declared shrinkage, 12 pseudo-windows.
- The negative binomial is not combined with the Phase 3.1 time prior or with
  household pooling.
- The online production filter is not changed. The recursion has full
  reliability and no health or attribution layer.
- Dependence in time within a state is not modelled.

## Reproducing it

```text
python scripts/run_phase3_negative_binomial.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes the record and a Markdown
summary generated from the written record.
