# The Phase 3 formulation on ContextAct@A4H: results

The frozen [protocol](CONTEXTACT_TRANSFER_PROTOCOL.md), run once as declared. Generated entirely from `artifacts/contextact/contextact-transfer.json` by `sensor_modeling.datasets.transfer_summary.render_page`.

**Status: pre-specified, one-shot; the one resident of ContextAct@A4H, collected independently of CASAS.**

- **Protocol digest.** `cea13b32c66f082335610e59786bca38122e861e40889bab631116e16b6db8cf`.
- **Run.** Commit `e397687`, recorded 2026-10-14T11:48:53.647336+00:00, on Linux-6.17.0-1022-azure-x86_64-with-glibc2.39.
- **Conclusion of K1.** success.
- **Days.** 29 scored: 7 in A4H-2016-07, 22 in A4H-2016-11. Unscored, with too few labelled windows: 0. Excluded periods: none. On every scored day every model scored the same windows in each true state.
- **Reproduction.** The refitted channels have the held-out record's digest, `15a63d668737ede83e3c8b10d7c97917b60849e2e253285c84a27a775b352ed6`; the refitted time prior was 0.0e+00 from the shipped one, which was used. The online pipeline's opt-in was at most 2.0e-15 from the recursion in A4H-2016-07, 1.7e-15 from the recursion in A4H-2016-11.

## Each model on ContextAct

The mean over days, with its interval.

| Model | Information | Balanced accuracy | Log loss | Brier score | Calibration error |
| --- | --- | --- | --- | --- | --- |
| `filter_declared` | R | 0.414 [0.357, 0.470] | 6.328 [5.411, 7.321] | 1.134 [0.980, 1.280] | 0.543 [0.464, 0.619] |
| `filter_hurdle` | R | 0.336 [0.270, 0.405] | 2.943 [2.140, 3.833] | 0.728 [0.554, 0.911] | 0.330 [0.238, 0.430] |
| `filter_periodic` | R | 0.534 [0.488, 0.580] | 4.249 [3.147, 5.499] | 0.696 [0.569, 0.829] | 0.331 [0.267, 0.399] |
| `filter_periodic_hurdle` | R | 0.385 [0.319, 0.451] | 2.664 [1.838, 3.571] | 0.595 [0.425, 0.783] | 0.286 [0.201, 0.380] |
| `state_frequency` | I3 | 0.206 [0.164, 0.246] | 1.611 [1.441, 1.807] | 0.750 [0.700, 0.805] | 0.213 [0.151, 0.280] |
| `persistence` | I3 | 0.241 [0.193, 0.290] | 1.523 [1.306, 1.789] | 0.714 [0.654, 0.781] | 0.276 [0.220, 0.331] |
| `logistic` | I3 | 0.426 [0.372, 0.490] | 1.171 [0.851, 1.558] | 0.486 [0.366, 0.622] | 0.274 [0.231, 0.327] |
| `tree` | I3 | 0.528 [0.473, 0.585] | 1.135 [0.747, 1.644] | 0.428 [0.308, 0.564] | 0.267 [0.219, 0.325] |
| `diagnostic` | I3 | 0.566 [0.513, 0.619] | 1.237 [0.774, 1.853] | 0.429 [0.302, 0.573] | 0.212 [0.158, 0.279] |

## The estimands

Mean paired differences over days, oriented so that a positive value favours the model, with their intervals and verdicts.

|  | Model against reference | Metric | Difference; verdict | Conclusion |
| --- | --- | --- | --- | --- |
| K1 | `filter_periodic_hurdle@R` against `filter_hurdle@R` | Balanced accuracy | +0.049 [+0.013, +0.087]; favours model | success |
| K1 | `filter_periodic_hurdle@R` against `filter_hurdle@R` | Brier score | +0.133 [+0.060, +0.220]; favours model |  |
| K1 | `filter_periodic_hurdle@R` against `filter_hurdle@R` | Calibration error | +0.044 [-0.001, +0.099]; uncertain |  |
| K1 | `filter_periodic_hurdle@R` against `filter_hurdle@R` | Log loss | +0.279 [+0.172, +0.403]; favours model |  |
| K2 | `filter_periodic@R` against `filter_declared@R` | Balanced accuracy | +0.120 [+0.088, +0.157]; favours model | success |
| K2 | `filter_periodic@R` against `filter_declared@R` | Brier score | +0.438 [+0.329, +0.548]; favours model |  |
| K2 | `filter_periodic@R` against `filter_declared@R` | Calibration error | +0.212 [+0.155, +0.271]; favours model |  |
| K2 | `filter_periodic@R` against `filter_declared@R` | Log loss | +2.079 [+1.427, +2.753]; favours model |  |
| K3 | `filter_periodic_hurdle@R` against `filter_periodic@R` | Balanced accuracy | -0.149 [-0.221, -0.076]; favours reference | inconclusive |
| K3 | `filter_periodic_hurdle@R` against `filter_periodic@R` | Brier score | +0.101 [-0.040, +0.234]; uncertain |  |
| K3 | `filter_periodic_hurdle@R` against `filter_periodic@R` | Calibration error | +0.045 [-0.028, +0.112]; uncertain |  |
| K3 | `filter_periodic_hurdle@R` against `filter_periodic@R` | Log loss | +1.585 [+0.933, +2.362]; favours model |  |
| G1 | `diagnostic@I3` against `filter_periodic_hurdle@R` | Balanced accuracy | +0.181 [+0.124, +0.239]; favours model | the baseline leads |
| G1 | `diagnostic@I3` against `filter_periodic_hurdle@R` | Brier score | +0.167 [+0.074, +0.268]; favours model |  |
| G1 | `diagnostic@I3` against `filter_periodic_hurdle@R` | Calibration error | +0.074 [+0.023, +0.130]; favours model |  |
| G1 | `diagnostic@I3` against `filter_periodic_hurdle@R` | Log loss | +1.427 [+0.906, +1.996]; favours model |  |
| G2 | `logistic@I3` against `filter_periodic_hurdle@R` | Balanced accuracy | +0.041 [-0.001, +0.081]; uncertain | uncertain |
| G2 | `logistic@I3` against `filter_periodic_hurdle@R` | Brier score | +0.109 [+0.021, +0.203]; favours model |  |
| G2 | `logistic@I3` against `filter_periodic_hurdle@R` | Calibration error | +0.013 [-0.055, +0.083]; uncertain |  |
| G2 | `logistic@I3` against `filter_periodic_hurdle@R` | Log loss | +1.492 [+0.938, +2.091]; favours model |  |
| L1 | `filter_periodic_hurdle@R` against `filter_declared@R` | Balanced accuracy | -0.029 [-0.125, +0.073]; uncertain | uncertain |
| L1 | `filter_periodic_hurdle@R` against `filter_declared@R` | Brier score | +0.538 [+0.311, +0.758]; favours model |  |
| L1 | `filter_periodic_hurdle@R` against `filter_declared@R` | Calibration error | +0.257 [+0.140, +0.370]; favours model |  |
| L1 | `filter_periodic_hurdle@R` against `filter_declared@R` | Log loss | +3.664 [+2.737, +4.559]; favours model |  |
| C1 | `filter_periodic_hurdle@R` against `chance@day` | Balanced accuracy | +0.113 [+0.049, +0.175]; favours model | above chance |

- **C1.** Days left out with fewer than two labelled states: 1.

**K1, recall by state.**

| State | Difference | Verdict | Days |
| --- | --- | --- | --- |
| `away` | +0.193 [+0.073, +0.329] | favours model | 19 |
| `bathroom_activity` | -0.019 [-0.054, +0.001] | uncertain | 20 |
| `bed_awake` | — | never labelled | 0 |
| `home_active` | -0.012 [-0.060, +0.018] | uncertain | 24 |
| `home_inactive` | +0.052 [+0.000, +0.100] | too few days | 3 |
| `kitchen_activity` | +0.000 [+0.000, +0.000] | negligible | 27 |
| `sleeping` | +0.075 [+0.036, +0.128] | favours model | 24 |

## Interaction

Per day, the prior's gain with the fitted channels minus its gain with the declared channels. Described, not judged.

| Metric | K1 minus K2 |
| --- | --- |
| Balanced accuracy | -0.071 [-0.113, -0.033] |
| Log loss | -1.800 [-2.428, -1.196] |
| Calibration error | -0.168 [-0.229, -0.111] |

## By period

Mean daily balanced accuracy of each recursion within each period, and the mean of K1. Described, not judged.

| Period | Days | `filter_declared` | `filter_hurdle` | `filter_periodic` | `filter_periodic_hurdle` | K1 |
| --- | --- | --- | --- | --- | --- | --- |
| A4H-2016-07 | 7 | 0.431 | 0.230 | 0.477 | 0.256 | +0.026 |
| A4H-2016-11 | 22 | 0.409 | 0.370 | 0.553 | 0.426 | +0.056 |

## Notes

- Every setting, estimand and criterion was declared in the protocol before ContextAct was scored.
- Everything was fitted once, on the 20 CASAS development homes only; nothing was fitted on ContextAct.
- The unit is a day of one resident. Verdicts compare effect sizes and day bootstrap intervals with declared minimal differences; they are not significance tests, and they describe this resident only.
