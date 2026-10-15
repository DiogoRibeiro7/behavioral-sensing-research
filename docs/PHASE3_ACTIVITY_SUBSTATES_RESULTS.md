# Phase 3.3 follow-up: activity sub-states in the recursion: results

The frozen [protocol](PHASE3_ACTIVITY_SUBSTATES_PROTOCOL.md), run as declared. Generated entirely from `artifacts/phase3/phase3-activity-substates.json` by `sensor_modeling.datasets.substate_summary.render_page`.

**Status: pre-specified; development panel, which earlier work has inspected.**

- **Protocol digest.** `0c02781a58f8d96c4d69948bf0313c71da52dc369b2c520adb88c99b60028e01`.
- **Run.** Commit `5588830`, recorded 2026-10-14T15:23:44.126836+00:00, on Linux-6.17.0-1022-azure-x86_64-with-glibc2.39.
- **Conclusion of W1.** Inconclusive.
- **M1, whether the excess of long quiet runs falls.** Falls.

## The check

The fold fits of phase1-fold-a, phase1-fold-b equal the fitted-rates record's; `filter_periodic_hurdle` gives back the combined-prior record's scores for all 20 households; and the quiet-run statistic against the population hurdle gives back the predictive-check record's in all 594 cells; each to within 1e-06. Both models scored the same labelled windows in all 20 households. They are not all bit-identical, which the tolerance allows for: the published records were made on other platforms.

## Each model in the recursion

The mean over households, with its interval.

| Model | Balanced accuracy | Log loss | Brier score | Calibration error |
| --- | --- | --- | --- | --- |
| `filter_periodic_hurdle` | 0.517 [0.486, 0.547] | 1.728 [1.554, 1.895] | 0.567 [0.507, 0.628] | 0.213 [0.177, 0.249] |
| `filter_periodic_substates` | 0.515 [0.487, 0.545] | 1.280 [1.072, 1.483] | 0.432 [0.365, 0.494] | 0.172 [0.143, 0.199] |

Recall of each state, the mean over households with its interval:

| State | `filter_periodic_hurdle` | `filter_periodic_substates` |
| --- | --- | --- |
| `away` | 0.626 [0.556, 0.687] | 0.839 [0.793, 0.881] |
| `home_active` | 0.268 [0.200, 0.338] | 0.316 [0.259, 0.374] |
| `home_inactive` | 0.440 [0.361, 0.519] | 0.446 [0.373, 0.514] |
| `sleeping` | 0.863 [0.823, 0.900] | 0.916 [0.885, 0.944] |
| `bed_awake` | 0.011 [0.000, 0.033] | 0.000 [0.000, 0.000] |
| `bathroom_activity` | 0.511 [0.419, 0.601] | 0.344 [0.258, 0.434] |
| `kitchen_activity` | 0.513 [0.394, 0.624] | 0.356 [0.269, 0.438] |

## The primary estimand

Mean paired differences, oriented so that a positive value favours the model, with their intervals and verdicts.

|  | Model against reference | Balanced accuracy | Log loss | Brier score | Calibration error | `home_active` recall | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W1 | `filter_periodic_substates@R` against `filter_periodic_hurdle@R` | -0.001 [-0.018, +0.017]; negligible | +0.448 [+0.267, +0.686]; favours model | +0.135 [+0.060, +0.233]; favours model | +0.041 [+0.003, +0.091]; favours model | +0.048 [-0.009, +0.099]; uncertain | inconclusive |

The other common states' recall, which the rule also guards: `away` favours model; `home_inactive` uncertain; `sleeping` favours model.

**W1, recall by state.**

| State | Difference | Verdict | Households |
| --- | --- | --- | --- |
| `away` | +0.213 [+0.153, +0.281] | favours model | 20 |
| `bathroom_activity` | -0.168 [-0.212, -0.125] | favours reference | 20 |
| `bed_awake` | -0.011 [-0.033, +0.000] | negligible | 5 |
| `home_active` | +0.048 [-0.009, +0.099] | uncertain | 20 |
| `home_inactive` | +0.005 [-0.049, +0.060] | uncertain | 20 |
| `kitchen_activity` | -0.157 [-0.233, -0.088] | favours reference | 20 |
| `sleeping` | +0.053 [+0.012, +0.095] | favours model | 20 |

## The mechanism

Windows in quiet runs of at least an hour, against the predictive distribution. Each household's value is the mean over its cells of the log ratio, shown as a ratio: the geometric mean over households, its interval, households above and below one, and the verdict.

**M1.** How much nearer the observed long quiet runs the model's prediction is than the reference's, whichever side of them each lies: the exponential of the mean fall in the absolute log ratio. Above one means the excess falls.

| State | Fall in misfit | Reading |
| --- | --- | --- |
| `home_active` | 1.809 [1.695, 1.921], 15 / 0, positive | falls |
| `home_inactive` | 1.346 [1.279, 1.415], 20 / 0, positive | falls |

Conclusion of M1: falls.

The ratio that remains in each split state, observed over predicted, under each model, described; below one means the model predicts more long quiet runs than occur:

| State | Reference | Model |
| --- | --- | --- |
| `away` | 1.109 [1.086, 1.133], 20 / 0, negligible | 0.998 [0.983, 1.008], 11 / 9, negligible |
| `home_active` | 6.173 [4.530, 8.292], 15 / 0, positive | 3.368 [2.485, 4.603], 15 / 0, positive |
| `home_inactive` | 2.584 [2.235, 2.961], 20 / 0, positive | 1.804 [1.598, 2.027], 20 / 0, positive |
| `sleeping` | 1.084 [1.045, 1.120], 18 / 2, negligible | 1.037 [0.999, 1.071], 14 / 6, negligible |

## The fitted sub-states

Per fold and split state: the training windows and stretches EM used; the quiet sub-state's share of the windows, counted over channels; each sub-state's silence probability, averaged over the channels; the probability of entering the state quiet, of staying quiet and of staying busy over one step; and the training log-likelihood at the last E-step, with its change over the last iteration. The quiet sub-state is the one whose silence probability is higher.

| Fold | State | Windows | Stretches | Quiet share | Silence, quiet / busy | Enter quiet | Stay quiet | Stay busy | Log-likelihood | Last change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase1-fold-a | `away` | 35,461 | 1,574 | 0.954 | 0.997 / 0.407 | 0.017 | 0.998 | 0.028 | 36,003.4 | -0.000000 |
| phase1-fold-a | `home_active` | 14,717 | 3,492 | 0.636 | 0.839 / 0.518 | 0.333 | 0.871 | 0.564 | 391,777.0 | +0.000000 |
| phase1-fold-a | `home_inactive` | 22,123 | 2,931 | 0.818 | 0.922 / 0.629 | 0.222 | 0.958 | 0.340 | 105,652.0 | +0.000000 |
| phase1-fold-a | `sleeping` | 43,422 | 903 | 0.970 | 0.983 / 0.601 | 0.081 | 0.990 | 0.064 | 18,809.8 | +0.000000 |
| phase1-fold-b | `away` | 40,206 | 1,142 | 0.972 | 0.998 / 0.380 | 0.020 | 0.999 | 0.040 | 23,696.1 | +0.000000 |
| phase1-fold-b | `home_active` | 10,120 | 2,575 | 0.712 | 0.857 / 0.504 | 0.375 | 0.921 | 0.454 | 185,146.2 | -0.000000 |
| phase1-fold-b | `home_inactive` | 18,612 | 2,046 | 0.810 | 0.930 / 0.650 | 0.188 | 0.952 | 0.410 | 80,780.2 | +0.000000 |
| phase1-fold-b | `sleeping` | 34,331 | 858 | 0.966 | 0.984 / 0.665 | 0.180 | 0.989 | 0.092 | 18,724.9 | -0.000000 |

## Notes

- Every setting, estimand and criterion was declared in the protocol before any household was scored.
- The fold fits are the fitted-rates protocol's, on training households only; the sub-states are fitted on the same households, and the reference and the quiet-run statistic reproduce the published records.
- Verdicts compare effect sizes and household bootstrap intervals with declared minimal differences; they are not significance tests.
