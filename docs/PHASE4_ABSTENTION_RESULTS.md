# Phase 4: an abstention rule for the fitted channels with the time prior: results

The frozen [protocol](PHASE4_ABSTENTION_PROTOCOL.md), run once. Generated from `artifacts/phase4/phase4-abstention-rule.json` by `sensor_modeling.datasets.abstention_summary.render_page`.

**Status: pre-specified selection; development panel, which earlier work has inspected.** Protocol `0ba0228a7f31e8dcf197cf8e9553cb64bb8d04d8744f0f53dffdfab34ee88887`; commit `d9224f3`.

- **Decision.** Not supported.
- **The rule frozen.** Abstain when `structural_disagreement` exceeds 0.2385, which keeps 0.950 of the windows it was chosen on.

## Selection

Each coverage tried from the target up, with each guarded state's kept share over the coverage kept, pooled over the homes the threshold was chosen on:

| Chosen on | Target | Coverage kept | `away` | `bathroom_activity` | `kitchen_activity` | Guard |
| --- | --- | --- | --- | --- | --- | --- |
| phase1-fold-a | 0.80 | 0.800 | 1.11 | 0.63 | 0.60 | fails |
| phase1-fold-a | 0.85 | 0.850 | 1.09 | 0.67 | 0.65 | fails |
| phase1-fold-a | 0.90 | 0.900 | 1.07 | 0.73 | 0.74 | fails |
| phase1-fold-a | 0.95 | 0.950 | 1.03 | 0.81 | 0.85 | holds |
| phase1-fold-b | 0.80 | 0.800 | 1.11 | 0.64 | 0.64 | fails |
| phase1-fold-b | 0.85 | 0.850 | 1.08 | 0.71 | 0.72 | fails |
| phase1-fold-b | 0.90 | 0.900 | 1.05 | 0.76 | 0.81 | fails |
| phase1-fold-b | 0.95 | 0.950 | 1.02 | 0.85 | 0.91 | holds |
| all 20 homes | 0.80 | 0.800 | 1.08 | 0.61 | 0.60 | fails |
| all 20 homes | 0.85 | 0.850 | 1.08 | 0.65 | 0.64 | fails |
| all 20 homes | 0.90 | 0.900 | 1.06 | 0.71 | 0.72 | fails |
| all 20 homes | 0.95 | 0.950 | 1.04 | 0.81 | 0.84 | holds |

## On the held-out homes

Each fold's threshold applied to its held-out homes. Differences are per home, with household intervals; positive favours the rule.

| Estimand | Difference |
| --- | --- |
| A1, error: full coverage minus kept | +0.016 [+0.012, +0.019]; negligible |
| A2, balanced accuracy: kept minus full coverage | +0.018 [+0.010, +0.027]; uncertain |

Pooled over the held-out homes the rule keeps 0.954 of the windows, and the guarded states keep `away` 1.02, `bathroom_activity` 0.87, `kitchen_activity` 0.88 of their proportional share: the guard holds.

| Home | Coverage | Error kept | Error full | BA kept | BA full |
| --- | --- | --- | --- | --- | --- |
| hh101 | 0.938 | 0.290 | 0.310 | 0.541 | 0.523 |
| hh102 | 0.976 | 0.265 | 0.269 | 0.576 | 0.568 |
| hh103 | 0.969 | 0.349 | 0.363 | 0.404 | 0.388 |
| hh105 | 0.951 | 0.240 | 0.244 | 0.555 | 0.547 |
| hh106 | 0.925 | 0.353 | 0.390 | 0.451 | 0.442 |
| hh108 | 0.955 | 0.255 | 0.266 | 0.620 | 0.609 |
| hh110 | 0.952 | 0.366 | 0.386 | 0.536 | 0.499 |
| hh111 | 0.949 | 0.325 | 0.336 | 0.572 | 0.561 |
| hh114 | 0.950 | 0.378 | 0.398 | 0.570 | 0.550 |
| hh118 | 0.980 | 0.296 | 0.301 | 0.653 | 0.643 |
| hh119 | 0.956 | 0.453 | 0.467 | 0.502 | 0.480 |
| hh120 | 0.960 | 0.415 | 0.433 | 0.452 | 0.440 |
| hh122 | 0.954 | 0.334 | 0.354 | 0.632 | 0.604 |
| hh123 | 0.980 | 0.404 | 0.410 | 0.509 | 0.504 |
| hh124 | 0.976 | 0.502 | 0.512 | 0.480 | 0.407 |
| hh125 | 0.913 | 0.508 | 0.532 | 0.532 | 0.518 |
| hh126 | 0.947 | 0.286 | 0.305 | 0.545 | 0.514 |
| hh127 | 0.946 | 0.351 | 0.377 | 0.414 | 0.404 |
| hh129 | 0.948 | 0.257 | 0.282 | 0.616 | 0.566 |
| hh130 | 0.981 | 0.260 | 0.264 | 0.536 | 0.568 |

## What this cannot show

- A confirmation: the 20 development homes have been inspected, the 43 held-out CASAS homes are used up for this formulation, and ContextAct's per-day scores of it are published too. A confirmation needs an independent dataset this formulation has not been scored on, with the rule frozen here applied once, unchanged.
- Independence of the selection: step 1 saw these homes' curves before the target and fraction were declared.
- The signal's choice: it was picked on these homes, and is not confirmed.
- Uncertainty in the fits and the selection: the homes of a fold share one fit and one threshold, and the household bootstrap resamples the scored homes without selecting the threshold again.
- The frozen threshold's own performance: A1 judges the procedure, with thresholds chosen on fits to nine homes and applied to fits to ten; the frozen threshold is chosen on fits to 19 and has no held-out estimate.
