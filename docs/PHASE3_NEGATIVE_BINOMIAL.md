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

Not yet run. The protocol above was committed before any household was scored.

## Reproducing it

```text
python scripts/run_phase3_negative_binomial.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes the record and a Markdown
summary generated from the written record.
