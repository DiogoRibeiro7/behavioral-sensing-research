# Phase 3: the time prior with the fitted channels, in the recursion: results

The frozen [protocol](PHASE3_COMBINED_PRIOR_PROTOCOL.md), run as declared. Generated entirely from `artifacts/phase3/phase3-combined-prior.json` by `sensor_modeling.datasets.combined_prior_summary.render_page`.

**Status: pre-specified; development panel, which earlier work has inspected.**

- **Protocol digest.** `8570de52f23ef2a30c5d66876895888bad3dc422d67fc2266440ed3ddc471429`.
- **Run.** Commit `4b8f7af`, recorded 2026-10-10T13:03:59.488130+00:00, on Linux-6.17.0-1022-azure-x86_64-with-glibc2.39.
- **Conclusion of K1.** success.

## The check

The fold fits of phase1-fold-a, phase1-fold-b equal the fitted-rates record's, and the recursions without the time prior give back its scores for all 20 households, to within 1e-06. They are not bit-identical, which the tolerance allows for: the record was made on another platform.

## Each model in the recursion

The mean over households, with its interval.

| Model | Balanced accuracy | Log loss | Brier score | Calibration error |
| --- | --- | --- | --- | --- |
| `filter_declared` | 0.417 [0.392, 0.442] | 4.680 [4.337, 5.149] | 0.922 [0.834, 1.042] | 0.403 [0.356, 0.469] |
| `filter_hurdle` | 0.456 [0.425, 0.484] | 2.306 [1.937, 2.845] | 0.747 [0.651, 0.883] | 0.284 [0.226, 0.364] |
| `filter_periodic` | 0.473 [0.444, 0.499] | 3.633 [3.443, 3.828] | 0.723 [0.658, 0.801] | 0.296 [0.259, 0.339] |
| `filter_periodic_hurdle` | 0.517 [0.486, 0.547] | 1.728 [1.554, 1.895] | 0.567 [0.507, 0.628] | 0.213 [0.177, 0.249] |

## The estimands

Mean paired differences, oriented so that a positive value favours the model, with their intervals and verdicts.

|  | Model against reference | Balanced accuracy | Log loss | Brier score | Calibration error | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| K1 | `filter_periodic_hurdle@R` against `filter_hurdle@R` | +0.061 [+0.044, +0.077]; favours model | +0.578 [+0.327, +1.007]; favours model | +0.180 [+0.116, +0.273]; favours model | +0.071 [+0.034, +0.123]; favours model | success |
| K2 | `filter_periodic@R` against `filter_declared@R` | +0.055 [+0.044, +0.067]; favours model | +1.047 [+0.729, +1.476]; favours model | +0.199 [+0.149, +0.257]; favours model | +0.107 [+0.082, +0.136]; favours model | success |
| K3 | `filter_periodic_hurdle@R` against `filter_periodic@R` | +0.044 [+0.016, +0.071]; favours model | +1.905 [+1.693, +2.120]; favours model | +0.156 [+0.102, +0.213]; favours model | +0.083 [+0.052, +0.115]; favours model | success |

**K1, recall by state.**

| State | Difference | Verdict | Households |
| --- | --- | --- | --- |
| `away` | +0.167 [+0.122, +0.216] | favours model | 20 |
| `bathroom_activity` | +0.065 [+0.047, +0.086] | favours model | 20 |
| `bed_awake` | -0.148 [-0.281, -0.015] | favours reference | 5 |
| `home_active` | +0.002 [-0.002, +0.007] | negligible | 20 |
| `home_inactive` | +0.064 [+0.044, +0.087] | favours model | 20 |
| `kitchen_activity` | -0.001 [-0.005, +0.002] | negligible | 20 |
| `sleeping` | +0.110 [+0.075, +0.142] | favours model | 20 |

**K2, recall by state.**

| State | Difference | Verdict | Households |
| --- | --- | --- | --- |
| `away` | +0.178 [+0.124, +0.227] | favours model | 20 |
| `bathroom_activity` | +0.005 [+0.001, +0.011] | negligible | 20 |
| `bed_awake` | -0.011 [-0.033, +0.000] | negligible | 5 |
| `home_active` | +0.032 [+0.016, +0.058] | uncertain | 20 |
| `home_inactive` | +0.012 [-0.001, +0.027] | negligible | 20 |
| `kitchen_activity` | -0.000 [-0.001, +0.001] | negligible | 20 |
| `sleeping` | +0.121 [+0.102, +0.140] | favours model | 20 |

## Interaction

Per household, the prior's gain with the fitted channels minus its gain with the declared channels; positive means the two combine better than additively. Described, not judged.

| Metric | K1 minus K2 |
| --- | --- |
| Balanced accuracy | +0.005 [-0.009, +0.020] |
| Log loss | -0.469 [-0.625, -0.314] |
| Calibration error | -0.036 [-0.058, -0.009] |

## Notes

- Every setting, estimand and criterion was declared in the protocol before any household was scored.
- The fold fits are the fitted-rates protocol's, on training households only, and the recursions without the time prior reproduce that record's scores.
- Verdicts compare effect sizes and household bootstrap intervals with declared minimal differences; they are not significance tests.
