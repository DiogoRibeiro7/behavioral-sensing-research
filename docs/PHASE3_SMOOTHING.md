# Phase 3.5: fixed-lag smoothing against online filtering

A fixed-lag smoother revises each window's estimate with up to `lag` later
windows. It reads evidence a live system does not yet have, and its estimate
for a window exists only `lag` windows after it
([inference regimes](INFERENCE_REGIMES.md)). This pre-specified experiment
measures how much performance smoothing recovers over the online filter, on
households none of the parameters were fitted on.

**Every difference it reports is a smoothing gain.** A smoothing gain is
available only after the smoother's delay. It is never an improvement of the
online filter, and it cannot be had by a live system at the moment it is
about.

**Status: pre-specified, development panel.** Every setting, comparison and
criterion was frozen before any household was scored. The development homes
have been inspected in earlier work, so this is not a final held-out claim. The
frozen external cohort is not touched.

An earlier, unregistered measurement on the production pipeline found most of
the smoothing gain at one window of lag ([real data](real_data.md)). That run
used an 11/11 split that was never recorded, and the declared rates. This one
uses the frozen folds and the current best recursion.

## The formulation

Smoothing needs a recursion over consecutive windows. The one on `develop` with
a pre-specified success in the recursion is the
[Phase 3.3 follow-up](PHASE3_FITTED_RATES.md)'s `filter_hurdle`:

- population hurdle channel models, fitted per fold on the training households
  with 12 pseudo-windows;
- the default ontology's transition over one 5-minute step;
- the stationary distribution one step before the first window;
- every window of the recording, with full reliability and attribution.

The other candidates are not used:

- Partial pooling in the recursion was inconclusive ([Phase 3.4](PHASE3_PARTIAL_POOLING.md), P2).
- The Phase 3.1 time prior was evaluated only on declared information sets,
  never as a recursion.

The online filter and every smoother share one filter pass. Each smoother only
revises it with the declared lag, through `regime_beliefs`. No model is changed
or refitted.

## Protocol

The protocol is `artifacts/phase3/smoothing_protocol.json`. The script refuses
to run if the code's protocol differs from it by a single value.

- **Households.** The 20 development homes and the two frozen Phase 1 folds.
  Each home is held out once, and scored by channel models fitted without it.
- **Regimes.** A small set, declared and never searched:

  | Regime | Lag | Reporting delay | Operational use |
  | --- | ---: | ---: | --- |
  | `online` | 0 | 0 min | the reference: what a live system reports at each window |
  | `lag_1` | 1 window | 5 min | a display that may run one window behind |
  | `lag_6` | 6 windows | 30 min | a non-urgent check within the hour, such as whether the resident has got up |
  | `lag_12` | 12 windows | 60 min | retrospective hourly or daily reporting |

  One hour is the longest delay treated as plausible for a decision made the
  same day. Longer lags serve only offline analysis and are not scored.
- **Scored windows.** Each held-out household's labelled windows, except the
  last 12 windows of its recording. Every regime is scored on exactly those
  windows, so every smoothed estimate reads its full lag.
- **Reproduction.** The online filter is also scored on every labelled window,
  as `filter_hurdle@R`. That reproduces the Phase 3.3 follow-up's cell, and a
  test checks it household by household.
- **Metrics.** Household balanced accuracy, per-state recall, log loss, Brier
  score, and expected calibration error over 10 bins.
- **Operational measures.** Each is a share of a household's scored windows,
  against the online filter's most probable state on the same windows:
  - **changed:** the smoother reports a different state;
  - **corrected:** among the windows the filter gets wrong, those the smoother
    gets right;
  - **made wrong:** among the windows the filter gets right, those the smoother
    gets wrong;
  - **corrections among changes:** among changed windows, those that are
    corrections.

  They are also reported within the windows of each focus state: `away`,
  `home_active` and `sleeping`. Each regime's reporting delay is its lag.
- **Transitions.**
  - A transition is a scored window whose labelled state differs from that of
    the window before it, both labelled.
  - **Accuracy near transitions** is measured within 6 windows of a transition,
    before or after it, and compared with accuracy elsewhere.
  - **Decision delay** is how long after a transition a regime can first report
    the new state. It counts the windows until the regime's most probable state
    first equals the new state within the new episode, plus the regime's lag.
    Each household contributes the median over its detected transitions.
  - **Detection rate** is the share of transitions whose new state the regime
    reports at some window of the new episode.
  - Each is reported for all transitions, and for the transitions into each
    focus state.
- **Bootstrap.** 10,000 household resamples, with 95% percentile intervals for
  the mean and the median paired difference. The seed is 0.

### Estimands

Each compares one smoother with the online filter on the same windows, paired
by household. A positive value favours the smoother. Every one is a smoothing
gain, and the record labels it with the smoother's delay.

| Key | Role | Smoother | Against |
| --- | --- | --- | --- |
| G1 | primary | `lag_1`, 5 min | `online` |
| G6 | primary | `lag_6`, 30 min | `online` |
| G12 | primary | `lag_12`, 60 min | `online` |

Per-state recall, the operational measures and the transition measures are
reported for each.

### Criteria

Verdicts compare effect sizes and household intervals with the Phase 3
minimal important differences. They are not significance tests.

| Quantity | `δ` |
| --- | ---: |
| balanced accuracy | 0.02 |
| log loss | 0.05 |
| Brier score | 0.01 |
| calibration error | 0.02 |
| per-state recall | 0.05 |
| accuracy near transitions | 0.02 |
| detection rate | 0.05 |
| decision delay | 5 min |

Each estimand is judged by a rule fixed in advance:

- **Gain.** Balanced accuracy favours the smoother, and neither log loss nor
  calibration error favours the online filter.
- **Trade-off.** Balanced accuracy favours the smoother, and log loss or
  calibration error favours the online filter.
- **Probability gain.** Balanced accuracy does not favour the smoother, log loss
  does, and neither balanced accuracy nor calibration error favours the online
  filter.
- **No gain.** Balanced accuracy is negligible or favours the online filter, and
  log loss is negligible or favours the online filter.
- **Inconclusive.** Anything else.

The trade-off verdict exists because the earlier, unregistered measurement
found that smoothing raised accuracy and worsened calibration.

### The record

The record compares regimes, so its `inference` field states the longest lag,
60 minutes, which bounds every estimate in it. Its results label every cell and
comparison with its own regime, and list each regime's prediction and
latest-evidence timestamps ([experiment artifacts](EXPERIMENT_ARTIFACTS.md#the-inference-regime)).

## Results

The summary below is generated from the published record,
`artifacts/phase3/phase3-fixed-lag-smoothing.json`, by `render_summary`.
A test checks that it still matches the record. The run was made from the
protocol commit, `923a1f1`, on a clean tree.

<!-- generated-summary:start -->

### Fixed-lag smoothing against online filtering: summary

Generated from the record `phase3-fixed-lag-smoothing`: protocol `f68d2733fb8f`, commit `923a1f15cef8`. Status: pre-specified; development panel, which earlier work has inspected.

- Inference regimes compared: online filter; fixed-lag smoother, lag 1 window (5 min); fixed-lag smoother, lag 6 windows (30 min); fixed-lag smoother, lag 12 windows (60 min). The record states the longest, fixed-lag smoother, lag 12 windows (60 min), which bounds every estimate in it. Every cell and comparison carries its own regime.
- **Every difference below is a smoothing gain.** The smoother reads up to its lag after each window and can report only that long after it. No difference is an improvement of the online filter.
- 20 held-out households in 2 cross-fitted folds, each scored once by channel models fitted without it.
- Formulation: the Phase 3.3 follow-up's filter_hurdle recursion.
- Every regime is scored on the same labelled windows. The last 12 windows of each recording are not scored, so every smoothed estimate reads its full lag.
- Differences are paired by household, with the mean and the median and their 95% household bootstrap intervals from 10,000 resamples. A positive value favours the smoother.
- Minimal important differences: balanced_accuracy 0.02, boundary_accuracy 0.02, brier 0.01, calibration_error 0.02, decision_delay_minutes 5, detection_rate 0.05, log_loss 0.05, recall 0.05.

#### Pre-specified conclusions

| Question | Reporting delay | Conclusion |
| --- | ---: | ---: |
| How much does the fixed-lag smoother, lag 1 window (5 min) recover over the online filter? (G1) | 5 min | gain |
| How much does the fixed-lag smoother, lag 6 windows (30 min) recover over the online filter? (G6) | 30 min | trade-off |
| How much does the fixed-lag smoother, lag 12 windows (60 min) recover over the online filter? (G12) | 60 min | trade-off |

Gain: balanced accuracy favours the smoother, and neither log loss nor calibration error favours the online filter. Trade-off: balanced accuracy favours the smoother, and log loss or calibration error favours the online filter. Probability gain: balanced accuracy does not favour the smoother, log loss does, and neither balanced accuracy nor calibration error favours the online filter. No gain: balanced accuracy is negligible or favours the online filter, and log loss is negligible or favours the online filter.

Verdicts: favours model: mean >= delta and lower bound > 0; favours reference: mean <= -delta and upper bound < 0; negligible: the whole interval within (-delta, delta); uncertain: anything else.

#### Every regime

Balanced accuracy as the median, then the mean with its interval. The other metrics are medians, and lower is better:

| Regime | Reporting delay | Balanced accuracy | Log loss | Brier | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: |
| online | 0 min | 0.469 (0.456 [0.425, 0.484]) | 2.124 | 0.709 | 0.249 |
| lag_1 | 5 min | 0.500 (0.487 [0.454, 0.518]) | 2.175 | 0.711 | 0.255 |
| lag_6 | 30 min | 0.510 (0.499 [0.465, 0.530]) | 2.177 | 0.678 | 0.258 |
| lag_12 | 60 min | 0.520 (0.505 [0.470, 0.538]) | 2.171 | 0.679 | 0.254 |

#### Smoothing gains

Each smoother against the online filter on the same windows: mean difference, homes improved and worsened, and verdict. Positive values favour the smoother, that is, higher accuracy or lower loss and calibration error. Each is available only after the smoother's delay:

| Key | Smoother | Balanced accuracy | Log loss | Brier | Calibration error | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 | lag_1 | +0.031 [+0.021, +0.043], 20 / 0, favours model | −0.044 [−0.073, −0.017], 7 / 13, uncertain | +0.005 [+0.001, +0.009], 14 / 6, negligible | −0.006 [−0.009, −0.002], 5 / 15, negligible | gain |
| G6 | lag_6 | +0.043 [+0.032, +0.055], 20 / 0, favours model | −0.101 [−0.173, −0.034], 4 / 16, favours reference | +0.016 [+0.003, +0.028], 12 / 8, favours model | −0.009 [−0.017, −0.000], 8 / 12, negligible | trade-off |
| G12 | lag_12 | +0.049 [+0.037, +0.062], 20 / 0, favours model | −0.119 [−0.239, −0.027], 7 / 13, favours reference | +0.026 [+0.010, +0.040], 14 / 6, favours model | −0.006 [−0.016, +0.003], 8 / 12, negligible | trade-off |

#### Per-state recall

Smoothing gain in each state's recall: mean difference, homes improved and worsened, and verdict:

| State | Rare | Focus | G1 | G6 | G12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| away | no | yes | +0.016 [+0.012, +0.022], 20 / 0, negligible | +0.060 [+0.045, +0.078], 19 / 1, favours model | +0.083 [+0.057, +0.111], 18 / 2, favours model |
| home_active | no | yes | +0.011 [+0.003, +0.019], 16 / 4, negligible | +0.014 [+0.004, +0.025], 13 / 4, negligible | +0.015 [+0.004, +0.026], 14 / 4, negligible |
| home_inactive | no | no | +0.003 [−0.003, +0.009], 10 / 9, negligible | +0.005 [−0.012, +0.022], 12 / 8, negligible | +0.011 [−0.011, +0.033], 13 / 7, negligible |
| sleeping | no | yes | +0.007 [+0.002, +0.012], 13 / 7, negligible | +0.025 [+0.011, +0.040], 17 / 3, negligible | +0.038 [+0.019, +0.056], 17 / 3, uncertain |
| bed_awake | yes | no | +0.239 [+0.028, +0.467], 3 / 1, favours model | +0.235 [+0.020, +0.467], 3 / 1, favours model | +0.235 [+0.020, +0.467], 3 / 1, favours model |
| bathroom_activity | yes | no | +0.089 [+0.072, +0.105], 20 / 0, favours model | +0.094 [+0.076, +0.114], 20 / 0, favours model | +0.094 [+0.075, +0.114], 20 / 0, favours model |
| kitchen_activity | yes | no | +0.014 [+0.006, +0.023], 13 / 4, negligible | +0.015 [+0.006, +0.024], 12 / 5, negligible | +0.015 [+0.006, +0.024], 12 / 5, negligible |

#### What smoothing changes

Against the online filter's most probable state on the same windows: the share of windows changed, the share of the filter's errors corrected, the share of its correct windows made wrong, and the share of changes that are corrections. Household means with their intervals:

| Key | Reporting delay | Changed | Corrected | Made wrong | Corrections among changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| G1 | 5 min | 0.083 [0.073, 0.092] | 0.069 [0.060, 0.079] | 0.039 [0.034, 0.045] | 0.372 [0.347, 0.396] |
| G6 | 30 min | 0.169 [0.147, 0.188] | 0.157 [0.135, 0.176] | 0.085 [0.071, 0.100] | 0.415 [0.384, 0.445] |
| G12 | 60 min | 0.193 [0.169, 0.214] | 0.192 [0.165, 0.215] | 0.101 [0.082, 0.122] | 0.443 [0.410, 0.475] |

Within the windows whose labelled state is each focus state:

| State | Key | Changed | Corrected | Made wrong |
| --- | ---: | ---: | ---: | ---: |
| away | G1 | 0.066 [0.052, 0.081] | 0.052 [0.039, 0.065] | 0.022 [0.016, 0.030] |
| away | G6 | 0.197 [0.158, 0.239] | 0.192 [0.155, 0.229] | 0.090 [0.066, 0.120] |
| away | G12 | 0.246 [0.200, 0.295] | 0.263 [0.212, 0.311] | 0.126 [0.091, 0.168] |
| home_active | G1 | 0.145 [0.133, 0.159] | 0.063 [0.044, 0.084] | 0.143 [0.112, 0.179] |
| home_active | G6 | 0.210 [0.186, 0.237] | 0.076 [0.051, 0.105] | 0.162 [0.128, 0.199] |
| home_active | G12 | 0.209 [0.184, 0.236] | 0.076 [0.051, 0.105] | 0.161 [0.127, 0.203] |
| sleeping | G1 | 0.062 [0.051, 0.074] | 0.131 [0.103, 0.165] | 0.030 [0.022, 0.039] |
| sleeping | G6 | 0.138 [0.116, 0.159] | 0.311 [0.273, 0.352] | 0.065 [0.046, 0.085] |
| sleeping | G12 | 0.166 [0.140, 0.191] | 0.392 [0.348, 0.436] | 0.076 [0.054, 0.100] |

#### Transitions

Accuracy within 6 windows of a true transition, before or after it, and further from every transition. Household means with their intervals:

| Regime | Near transitions | Elsewhere |
| --- | ---: | ---: |
| online | 0.413 [0.388, 0.440] | 0.603 [0.530, 0.659] |
| lag_1 | 0.435 [0.407, 0.464] | 0.607 [0.534, 0.664] |
| lag_6 | 0.460 [0.427, 0.493] | 0.620 [0.545, 0.677] |
| lag_12 | 0.464 [0.430, 0.497] | 0.633 [0.556, 0.691] |

Detection rate, and decision delay in minutes, including the regime's reporting delay. Household means of each household's rate and median delay:

| Transitions into | Regime | Detection rate | Decision delay |
| --- | ---: | ---: | ---: |
| all | online | 0.519 [0.482, 0.554] | 2.375 [1.375, 3.500] |
| all | lag_1 | 0.558 [0.521, 0.594] | 5.500 [5.000, 6.250] |
| all | lag_6 | 0.542 [0.506, 0.576] | 30.250 [30.000, 30.750] |
| all | lag_12 | 0.533 [0.496, 0.567] | 60.375 [60.000, 61.000] |
| away | online | 0.444 [0.360, 0.524] | 16.750 [13.500, 20.500] |
| away | lag_1 | 0.526 [0.437, 0.611] | 24.500 [19.750, 30.000] |
| away | lag_6 | 0.505 [0.425, 0.583] | 39.500 [36.622, 43.250] |
| away | lag_12 | 0.473 [0.392, 0.552] | 66.750 [65.000, 69.750] |
| home_active | online | 0.418 [0.327, 0.506] | 0.000 [0.000, 0.000] |
| home_active | lag_1 | 0.447 [0.356, 0.536] | 5.000 [5.000, 5.000] |
| home_active | lag_6 | 0.443 [0.352, 0.533] | 30.000 [30.000, 30.000] |
| home_active | lag_12 | 0.442 [0.350, 0.533] | 60.000 [60.000, 60.000] |
| sleeping | online | 0.929 [0.900, 0.955] | 23.875 [20.375, 27.500] |
| sleeping | lag_1 | 0.931 [0.906, 0.954] | 24.750 [21.250, 28.250] |
| sleeping | lag_6 | 0.905 [0.874, 0.935] | 43.875 [41.250, 46.625] |
| sleeping | lag_12 | 0.884 [0.850, 0.917] | 71.375 [68.875, 74.250] |

Smoothing gains near transitions: mean difference, homes improved and worsened, and verdict. For decision delay, a positive value means the smoother reports the new state sooner:

| Key | Accuracy near transitions | detection (all) | delay (all) | detection (away) | delay (away) | detection (home_active) | delay (home_active) | detection (sleeping) | delay (sleeping) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 | +0.022 [+0.017, +0.027], 20 / 0, favours model | +0.040 [+0.033, +0.046], 20 / 0, negligible | −3.125 [−4.125, −2.000], 0 / 13, negligible | +0.082 [+0.051, +0.113], 17 / 2, favours model | −7.750 [−11.625, −4.875], 0 / 17, favours reference | +0.029 [+0.013, +0.047], 16 / 4, negligible | −5.000 [−5.000, −5.000], 0 / 20, favours reference | +0.002 [−0.005, +0.010], 5 / 3, negligible | −0.875 [−1.750, −0.125], 0 / 4, negligible |
| G6 | +0.047 [+0.035, +0.059], 19 / 1, favours model | +0.023 [+0.016, +0.031], 16 / 4, negligible | −27.875 [−28.875, −26.875], 0 / 20, favours reference | +0.062 [+0.026, +0.096], 17 / 3, favours model | −22.750 [−26.000, −19.750], 0 / 20, favours reference | +0.025 [+0.005, +0.046], 15 / 4, negligible | −30.000 [−30.000, −30.000], 0 / 20, favours reference | −0.024 [−0.038, −0.009], 2 / 12, negligible | −20.000 [−21.500, −18.250], 0 / 20, favours reference |
| G12 | +0.051 [+0.037, +0.064], 20 / 0, favours model | +0.014 [+0.006, +0.023], 15 / 5, negligible | −58.000 [−59.000, −57.000], 0 / 20, favours reference | +0.030 [−0.006, +0.065], 12 / 8, uncertain | −50.000 [−52.875, −46.875], 0 / 20, favours reference | +0.024 [+0.005, +0.045], 15 / 3, negligible | −60.000 [−60.000, −60.000], 0 / 20, favours reference | −0.045 [−0.067, −0.024], 2 / 14, uncertain | −47.500 [−49.375, −45.375], 0 / 20, favours reference |

#### Evidence read

Each regime's scored estimates: how many, the first and last prediction timestamps, the latest evidence any of them read, and the most future information any of them used:

| Regime | Causal | Predictions | First prediction | Last prediction | Latest evidence | Largest lead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| online | yes | 232,819 | 2011-06-15T00:06:32.834414-07:00 | 2014-08-20T15:29:12.589833-07:00 | 2014-08-20T15:29:12.589833-07:00 | 0 min |
| lag_1 | no | 232,819 | 2011-06-15T00:06:32.834414-07:00 | 2014-08-20T15:29:12.589833-07:00 | 2014-08-20T15:34:12.589833-07:00 | 5 min |
| lag_6 | no | 232,819 | 2011-06-15T00:06:32.834414-07:00 | 2014-08-20T15:29:12.589833-07:00 | 2014-08-20T15:59:12.589833-07:00 | 30 min |
| lag_12 | no | 232,819 | 2011-06-15T00:06:32.834414-07:00 | 2014-08-20T15:29:12.589833-07:00 | 2014-08-20T16:29:12.589833-07:00 | 60 min |

#### Households

Each household's scored windows and transitions, and its balanced accuracy smoothing gain per smoother:

| Household | Fold | Scored windows | Transitions | G1 | G6 | G12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hh101 | phase1-fold-a | 16371 | 1267 | +0.028 | +0.041 | +0.051 |
| hh102 | phase1-fold-b | 16574 | 1028 | +0.058 | +0.070 | +0.077 |
| hh103 | phase1-fold-a | 14096 | 988 | +0.110 | +0.118 | +0.128 |
| hh105 | phase1-fold-b | 11820 | 632 | +0.083 | +0.092 | +0.096 |
| hh106 | phase1-fold-a | 15944 | 1119 | +0.002 | +0.009 | +0.011 |
| hh108 | phase1-fold-b | 15795 | 1166 | +0.017 | +0.024 | +0.026 |
| hh110 | phase1-fold-a | 6837 | 495 | +0.022 | +0.043 | +0.050 |
| hh111 | phase1-fold-b | 16802 | 913 | +0.012 | +0.014 | +0.020 |
| hh114 | phase1-fold-a | 7938 | 535 | +0.021 | +0.028 | +0.033 |
| hh118 | phase1-fold-b | 8381 | 627 | +0.019 | +0.041 | +0.050 |
| hh119 | phase1-fold-a | 7978 | 515 | +0.025 | +0.040 | +0.051 |
| hh120 | phase1-fold-b | 17526 | 1170 | +0.023 | +0.035 | +0.043 |
| hh122 | phase1-fold-a | 8036 | 476 | +0.027 | +0.052 | +0.066 |
| hh123 | phase1-fold-b | 8509 | 652 | +0.023 | +0.027 | +0.032 |
| hh124 | phase1-fold-a | 17064 | 60 | +0.032 | +0.048 | +0.049 |
| hh125 | phase1-fold-b | 15458 | 968 | +0.016 | +0.021 | +0.032 |
| hh126 | phase1-fold-a | 7137 | 532 | +0.034 | +0.062 | +0.087 |
| hh127 | phase1-fold-b | 4613 | 373 | +0.015 | +0.025 | +0.018 |
| hh129 | phase1-fold-a | 7846 | 417 | +0.033 | +0.046 | +0.045 |
| hh130 | phase1-fold-b | 8094 | 638 | +0.019 | +0.018 | +0.018 |

<!-- generated-summary:end -->

### What this shows and does not show

Figures are mean paired differences across the 20 homes, with 95% household
intervals and the number of homes that improved, unless stated otherwise.
**Every one is a smoothing gain.** It is available only after the smoother's
delay, and none is an improvement of the online filter.

- **The run agrees with earlier work.**
  - In both folds, the population is identical, by digest, to the Phase 3.3
    follow-up's fit.
  - On every labelled window, the online filter reproduces that record's
    `filter_hurdle@R` metrics household by household, exactly.
- **Five minutes of lag is a pre-specified gain (G1).**
  - Balanced accuracy rises by 0.031 [0.021, 0.043], in all 20 homes.
  - Log loss worsens by 0.044 [0.017, 0.073], short of its minimal
    difference, so the verdict is uncertain.
  - The Brier score and calibration error are negligible.
- **Thirty and sixty minutes are pre-specified trade-offs (G6, G12).**
  - Balanced accuracy rises by 0.043 [0.032, 0.055] and 0.049 [0.037, 0.062],
    in all 20 homes each time.
  - Log loss worsens by 0.101 [0.034, 0.173] and 0.119 [0.027, 0.239].
  - The Brier score improves, by 0.016 and 0.026. Calibration error is
    negligible.
  - Log loss worsening while the Brier score improves is consistent with
    sharper beliefs that are more confidently wrong where they are wrong. That
    mechanism was not tested.
- **The gain grows with the lag, with diminishing returns.**
  - Five minutes gives 63% of the sixty-minute gain in balanced accuracy, and
    thirty minutes 87%.
  - The earlier, unregistered measurement found nothing beyond one window. It
    used a different filter and split.
- **Most changes are not corrections.**
  - At five minutes, the smoother changes 8.3% of reported states.
  - It corrects 6.9% of the filter's errors, and makes 3.9% of its correct
    windows wrong.
  - Only 37% of changed windows are corrections. At sixty minutes it is 44%,
    with 19.3% of states changed.
  - Plain accuracy rises by only 0.010 at five minutes, a third of the
    balanced-accuracy gain. So the net corrections fall mostly in the less
    common states, which balanced accuracy weights equally.
- **The focus states.**
  - **`away`.** Recall gains grow with the lag.
    - At five minutes the gain is negligible, +0.016.
    - At thirty and sixty minutes it favours the smoother: +0.060 and +0.083.
    - At sixty minutes the smoother corrects 26.3% of the filter's `away`
      errors and makes 12.6% of its correct `away` windows wrong.
  - **`home_active`.** The recall gain is negligible at every lag, +0.011 to
    +0.015. Within `home_active` windows, the smoother makes 14% to 16% of the
    filter's correct windows wrong, and corrects 6% to 8% of its errors.
  - **`sleeping`.** The recall gain is negligible at five and thirty minutes.
    At sixty minutes it is uncertain, +0.038. The smoother corrects 13% to 39%
    of the filter's `sleeping` errors.
  - **Rare states.** The largest recall gains are there. `bathroom_activity`
    gains 0.089 at five minutes, in all 20 homes.
- **Transitions.**
  - **Accuracy.** Within six windows of a true transition, accuracy rises by
    0.022, 0.047 and 0.051, favouring the smoother at every lag.
  - **Timing.** No smoother reports a new state sooner than the online filter.
    Including its lag, the household mean of each home's median decision
    delay over all transitions is:
    - 2.4 minutes online;
    - 5.5 minutes at a five-minute lag;
    - 30.3 minutes at thirty minutes;
    - 60.4 minutes at sixty minutes.
  - **Leaving home.** At a five-minute lag, the smoother detects more
    transitions into `away`, by 0.082. It reports them later: 24.5 minutes
    against 16.8.
  - **Falling asleep.** At a five-minute lag, the cost for transitions into
    `sleeping` is under a minute, a negligible verdict.
  - **Longer lags.** At thirty and sixty minutes, a smoother places the
    boundary earlier, but that recovers only part of its wait:
    - about 2 minutes over all transitions;
    - none for transitions into `home_active`;
    - 7 to 10 minutes for transitions into `away`;
    - 10 to 12.5 minutes for transitions into `sleeping`.

    At sixty minutes it also detects fewer transitions into `sleeping`,
    −0.045, an uncertain verdict.

**What it means operationally.**

- **Retrospective displays and reports.** Where a five-minute delay is
  acceptable, the one-window smoother adds balanced accuracy. By the
  pre-specified rule, it has no probability cost. Log loss worsens by 0.044,
  an uncertain verdict. Longer lags add more accuracy, and cost log loss.
- **Alerting and anything decided at the moment.** No lag helps a decision
  that must be made when a state changes. Every smoother learns of a new state
  later than the online filter.

**Not shown:**

- This is the development panel, which earlier work has inspected, so it is
  not a held-out or external claim.
- **One formulation.** It is the fitted-hurdle recursion, without the time
  prior or household pooling. Smoothing either of those was not tested.
- **Lags.** Only the three declared lags were scored.
- **Seeds.** There is one seed.
- **The recursion.** It has full reliability and no health or attribution
  layer. The online production filter is not changed.
- **Decision delay.** It is a median over each regime's detected transitions,
  so regimes that detect different transitions are compared on different sets.
- **Unscored windows.** The last hour of each recording is not scored in any
  regime.

## Reproducing it

```text
python scripts/run_phase3_smoothing.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes the record and a Markdown
summary generated from the written record.
