# The Phase 3 formulation on held-out CASAS homes: results

The frozen [protocol](HELDOUT_CONFIRMATION_PROTOCOL.md), run once as declared. Generated entirely from `artifacts/heldout/casas-heldout-confirmation.json` by `sensor_modeling.datasets.heldout_summary.render_page`.

**Status: pre-specified, one-shot; CASAS homes outside the development panel, scored once before by the production filter.**

- **Protocol digest.** `ffe1d932a477ef68d3d81bf797017c3d26eeab35158c344002d31379f0054bef`.
- **Run.** Commit `d056b86`, recorded 2026-10-11T09:14:28.121301+00:00, on Linux-6.17.0-1022-azure-x86_64-with-glibc2.39.
- **Conclusion of K1.** success.
- **Households.** 20 training, 43 held-out homes scored, unscored: none. In every scored home every model scored the same windows in each true state.

## Each model on the held-out homes

The mean over households, with its interval.

| Model | Information | Balanced accuracy | Log loss | Brier score | Calibration error |
| --- | --- | --- | --- | --- | --- |
| `filter_declared` | R | 0.482 [0.459, 0.506] | 5.683 [5.266, 6.101] | 0.890 [0.827, 0.950] | 0.401 [0.370, 0.432] |
| `filter_hurdle` | R | 0.490 [0.467, 0.513] | 2.693 [2.444, 2.970] | 0.763 [0.713, 0.816] | 0.296 [0.268, 0.327] |
| `filter_periodic` | R | 0.525 [0.505, 0.545] | 4.950 [4.548, 5.360] | 0.750 [0.699, 0.801] | 0.322 [0.296, 0.348] |
| `filter_periodic_hurdle` | R | 0.567 [0.544, 0.590] | 2.262 [2.025, 2.525] | 0.605 [0.558, 0.658] | 0.233 [0.209, 0.261] |
| `state_frequency` | I3 | 0.166 [0.164, 0.168] | 1.660 [1.596, 1.729] | 0.789 [0.771, 0.807] | 0.101 [0.079, 0.125] |
| `persistence` | I3 | 0.260 [0.246, 0.274] | 2.938 [2.671, 3.189] | 1.193 [1.097, 1.286] | 0.542 [0.471, 0.608] |
| `logistic` | I3 | 0.343 [0.314, 0.374] | 2.494 [2.211, 2.764] | 1.076 [0.968, 1.179] | 0.507 [0.437, 0.570] |
| `tree` | I3 | 0.440 [0.418, 0.462] | 1.777 [1.618, 1.935] | 0.810 [0.744, 0.873] | 0.282 [0.247, 0.315] |
| `diagnostic` | I3 | 0.557 [0.532, 0.582] | 1.577 [1.396, 1.760] | 0.713 [0.641, 0.783] | 0.267 [0.227, 0.307] |

## The estimands

Mean paired differences, oriented so that a positive value favours the model, with their intervals and verdicts.

|  | Model against reference | Balanced accuracy | Log loss | Brier score | Calibration error | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| K1 | `filter_periodic_hurdle@R` against `filter_hurdle@R` | +0.077 [+0.070, +0.084]; favours model | +0.431 [+0.372, +0.496]; favours model | +0.158 [+0.136, +0.182]; favours model | +0.063 [+0.051, +0.077]; favours model | success |
| K2 | `filter_periodic@R` against `filter_declared@R` | +0.043 [+0.035, +0.051]; favours model | +0.733 [+0.575, +0.907]; favours model | +0.140 [+0.114, +0.168]; favours model | +0.079 [+0.065, +0.095]; favours model | success |
| K3 | `filter_periodic_hurdle@R` against `filter_periodic@R` | +0.042 [+0.024, +0.060]; favours model | +2.689 [+2.407, +2.967]; favours model | +0.145 [+0.096, +0.191]; favours model | +0.089 [+0.063, +0.114]; favours model | success |
| G1 | `diagnostic@I3` against `filter_periodic_hurdle@R` | -0.010 [-0.033, +0.014]; uncertain | +0.685 [+0.420, +0.973]; favours model | -0.109 [-0.187, -0.027]; favours reference | -0.034 [-0.078, +0.012]; uncertain | uncertain |
| G2 | `logistic@I3` against `filter_periodic_hurdle@R` | -0.224 [-0.250, -0.196]; favours reference | -0.232 [-0.532, +0.076]; uncertain | -0.472 [-0.576, -0.362]; favours reference | -0.274 [-0.336, -0.206]; favours reference | the combination leads |

**K1, recall by state.**

| State | Difference | Verdict | Households |
| --- | --- | --- | --- |
| `away` | +0.194 [+0.162, +0.228] | favours model | 43 |
| `bathroom_activity` | +0.043 [+0.036, +0.050] | uncertain | 43 |
| `bed_awake` | -0.123 [-0.260, -0.048] | too few homes | 3 |
| `home_active` | -0.000 [-0.003, +0.002] | negligible | 43 |
| `home_inactive` | +0.106 [+0.087, +0.126] | favours model | 43 |
| `kitchen_activity` | -0.006 [-0.017, +0.000] | negligible | 41 |
| `sleeping` | +0.132 [+0.112, +0.154] | favours model | 43 |

## Interaction

Per household, the prior's gain with the fitted channels minus its gain with the declared channels; positive means the two combine better than additively. Described, not judged.

| Metric | K1 minus K2 |
| --- | --- |
| Balanced accuracy | +0.034 [+0.024, +0.044] |
| Log loss | -0.301 [-0.469, -0.149] |
| Calibration error | -0.016 [-0.034, +0.002] |

## By family

Mean household balanced accuracy of each recursion within each CASAS family, and the mean of K1. Described, not judged: some families have two or three homes.

| Family | Homes | `filter_declared` | `filter_hurdle` | `filter_periodic` | `filter_periodic_hurdle` | K1 |
| --- | --- | --- | --- | --- | --- | --- |
| `hh` | 8 | 0.448 | 0.485 | 0.505 | 0.556 | +0.071 |
| `ihs` | 2 | 0.600 | 0.566 | 0.617 | 0.645 | +0.078 |
| `mn` | 3 | 0.446 | 0.553 | 0.499 | 0.635 | +0.082 |
| `rw` | 5 | 0.469 | 0.447 | 0.510 | 0.535 | +0.087 |
| `tm` | 25 | 0.491 | 0.487 | 0.530 | 0.563 | +0.076 |

## Notes

- Every setting, estimand and criterion was declared in the protocol before any held-out home was scored.
- Everything was fitted once, on the 20 development homes only.
- Verdicts compare effect sizes and household bootstrap intervals with declared minimal differences; they are not significance tests.
