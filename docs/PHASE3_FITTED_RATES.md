# Phase 3.3 follow-up: fitted silence and activity rates

The [correlated-silence diagnostic](PHASE3_CORRELATED_SILENCE.md) concluded
*weakened* under its frozen rule. Dependence between channels is real, but the
filter's declared rates overstate the evidence of silence more than dependence
does: 0.784 against 0.435 on the log scale. Overconfidence also builds along
quiet runs.

The routing rule, declared before this experiment, maps that result to this
route:

- fit each channel's observation model on training households;
- keep the channels conditionally independent given the state;
- compare with the declared rates on identical information.

The two-stage count-and-allocation model is the route for *supported*, and it
is not built.

**Status: pre-specified, development panel.** Every setting, comparison and
criterion was frozen before any household was scored. The development homes
have been inspected in earlier work, so this is not a final held-out claim. The
frozen external cohort is not touched.

## Why silence needs its own parameter

The obvious fit, a Poisson rate per channel and state set to the observed mean
count, makes silence worse, not better. Real counts are bursty. In Phase 3.3,
windows in which every channel was silent were far more common than
independent Poisson streams with the observed means allow: by a log ratio of
+15.6 in `home_active` and +0.85 in `sleeping`. A mean-matched Poisson would
call a silent `home_active` window almost impossible.

So the primary model gives silence and activity separate parameters: a
**hurdle** model. A fitted Poisson is kept as a secondary arm, to test whether
the separate silence parameter is needed.

## The likelihood

For channel `c`, state `s` and a window's pooled count `n`:

```text
P(n = 0     | s) = π
P(n = k ≥ 1 | s) = (1 − π) · μ^k e^{−μ} / (k! (1 − e^{−μ}))
```

where:

- `π = π_c(s)` is the probability that the channel is silent in a window;
- `μ = μ_c(s)` is the rate of the zero-truncated Poisson that says how much
  activity there is once there is any.

Its log-likelihood is:

```text
log P(n | s) = log π                                          if n = 0
             = log(1 − π) − log(e^μ − 1) + n log μ − log n!    if n ≥ 1
```

- **The shared term.** `log n!` is shared by every state and dropped.
- **The window.** The window's log-likelihood is the sum over the household's
  instrumented channels. The channels are independent given the state, as in
  the declared model.
- **Reducing to the Poisson.** When `π = e^{−μ}`, the model is exactly a
  Poisson with rate `μ`; a test checks this.
- **Decomposition.** Each window's contribution splits into a silence part
  and an activity part:
  - silence: `log π` if silent, `log(1 − π)` otherwise;
  - activity: `n log μ − log(e^μ − 1)`, and zero if silent.

  `HurdleChannel.decompose` reports both.

The declared model, `P(n | s) = Poisson(λ_c(s))`, is unchanged and remains
available. A model is chosen by its family, `declared`, `hurdle` or `poisson`,
in `FittedChannels.models`.

Four cases need care:

- **An absent channel.** An uninstrumented channel has no model and no
  column. It is never treated as silent.
- **A missing window.** A window that closes before the recording starts
  carries no evidence and is skipped.
- **No clipping.** No probability is clipped. `π` is fitted strictly inside
  `(0, 1)` and `μ > 0`.
- **Numerical stability.** `log(e^μ − 1)` is evaluated stably for both small
  and large `μ`. A count of 10⁶ stays finite, and a test checks this.

## Fitting

Parameters are per channel and state, pooled over the labelled windows of the
fold's training households only. Each is a posterior mean shrunk toward the
declared model with `κ = 12` pseudo-windows, one hour:

```text
π̂ = (Z + κ π₀) / (W + κ)
m̂ = (S + κ m₀) / (P + κ),      μ̂ solves μ / (1 − e^{−μ}) = m̂
λ̂ = (S + κ λ₀) / (W + κ)       (the fitted Poisson)
```

- `W` is the windows, `Z` the silent ones, `P = W − Z` the active ones, and `S`
  the total count.
- `λ₀` is the declared expected count, `π₀ = e^{−λ₀}` the declared silence
  probability, and `m₀ = λ₀ / (1 − e^{−λ₀})` the declared mean count of an
  active window. Each is averaged over the training windows.
- With the thousands of training windows most channels have, the prior barely
  matters. With none, the fitted model is the declared one.
- A channel without training windows in a state takes the held-out household's
  own declared value. That uses its sensor registry but none of its labels.
- There is no household adaptation. Every held-out household gets its fold's
  population parameters.

`living_motion` pools 1 to 4 sensors, depending on the home. The other
channels pool one. Channel-level parameters average over those installations.

## Protocol

The protocol is `artifacts/phase3/fitted_rates_protocol.json`. The script
refuses to run if the code's protocol differs from it by a single value.

- **Households.** The 20 development homes and the two frozen Phase 1 folds.
  Each home is scored once, by models fitted without it.
- **Models.** Every model has the same prior, transition and channels, and
  differs only in its channel observation model:

  | Model | Channels | Scored in |
  | --- | --- | --- |
  | `generative` | declared | `I0`, `I2` |
  | `generative_hurdle` | fitted hurdle | `I0`, `I2` |
  | `generative_fitted_poisson` | fitted Poisson | `I0` |
  | `generative_periodic` | declared, with the Phase 3.1 time prior | `I1`, `I3` |
  | `generative_periodic_hurdle` | fitted hurdle, with the time prior | `I1`, `I3` |
  | `filter_declared`, `filter_hurdle`, `filter_fitted_poisson` | each family | `R` |

  `R` is the filter's recursion fed every window of the recording, where
  Phase 3.3 measured quiet-run accumulation. Both sides of a comparison in `R`
  receive the same windows. The time prior is fitted per fold on training
  homes, as in Phase 3.2.
- **Metrics.** Household balanced accuracy, log loss, Brier score, expected
  calibration error over 10 bins, and per-state recall.
- **Bootstrap.** 10,000 household resamples, with 95% percentile intervals for
  the mean and the median. The seed is 0.

### Estimands

Every estimand compares two channel models on identical information. A
positive value favours the fitted model.

| Key | Role | First | Second |
| --- | --- | --- | --- |
| F0 | primary | `generative_hurdle` `I0` | `generative` `I0` |
| FR | primary | `filter_hurdle` `R` | `filter_declared` `R` |
| F1 | secondary | `generative_periodic_hurdle` `I1` | `generative_periodic` `I1` |
| F2 | secondary | `generative_hurdle` `I2` | `generative` `I2` |
| F3 | secondary | `generative_periodic_hurdle` `I3` | `generative_periodic` `I3` |
| P0 | secondary | `generative_fitted_poisson` `I0` | `generative` `I0` |
| PR | secondary | `filter_fitted_poisson` `R` | `filter_declared` `R` |

Per-state recall is reported for F0 and FR.

**Mechanisms.** Each is one value per household, tested against zero on the
Phase 3.3 scales:

- **M1**: how much the fitted hurdle channels reduce the silence-evidence
  inflation. This is log inflation with the declared terms minus log inflation
  with the fitted ones, using Phase 3.3's definition and settings. M1P is the
  same for the fitted Poisson.
- **M2**: how much the fitted hurdle channels slow the growth of
  overconfidence along quiet runs. This is Phase 3.3's accumulation slope, per
  hour, declared minus fitted. M2P is the same for the fitted Poisson.

**Descriptive.**

- In each household's Phase 3.3 representative quiet runs: the state each
  recursion reports after the recorded windows, and its confidence.
- Per fold, channel and state: the fitted parameters, and an active window's
  count variance against its mean.

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

The mechanisms use Phase 3.3's `δ`: log 1.25 for log ratios, and 0.02 for
slopes.

F0 and FR are each judged by a rule fixed in advance:

- **Success.** Calibration error favours the fitted model, and neither log
  loss nor balanced accuracy favours the declared rates.
- **Trade-off.** Calibration error favours the fitted model, and balanced
  accuracy favours the declared rates.
- **Failure.** Calibration error is negligible or favours the declared rates.
- **Inconclusive.** Anything else.

Calibration is the target because the diagnostic found overstated evidence,
which shows up as overconfidence. Balanced accuracy is a guard. An earlier,
unregistered fit on an 11/11 split that was never recorded moved calibration
error from 0.312 to 0.202 and cost 0.031 balanced accuracy. The trade-off
verdict exists so that such a result is named as what it is.

## Results

The summary below is generated from the published record,
`artifacts/phase3/phase3-fitted-rates.json`, by `render_summary`.
A test checks that it still matches the record.

<!-- generated-summary:start -->

### Fitted silence and activity rates: summary

Generated from the record `phase3-fitted-rates`: protocol `4f5db165af7e`, commit `4b6a1aca94b2`. Status: pre-specified; development panel, which earlier work has inspected.

- Inference regime: online filter (attested on migration from schema 1.1).
- 20 households in 2 cross-fitted folds, each scored once by channel models fitted without it.
- Every comparison is between two channel observation models on identical information: one set, or the recursion over every window (`R`).
- Differences are paired by household, with the mean and the median and their 95% household bootstrap intervals from 10,000 resamples. A positive value favours the fitted model.
- Minimal important differences: balanced_accuracy 0.02, brier 0.01, calibration_error 0.02, log_loss 0.05, recall 0.05.

#### Pre-specified conclusions

| Question | Conclusion |
| --- | ---: |
| Do fitted hurdle channels improve on the declared rates, current windows? (F0) | success |
| And in the filter's recursion over every window? (FR) | success |

Success: calibration error favours the fitted model, and neither log loss nor balanced accuracy favours the declared rates. Trade-off: calibration error favours the fitted model and balanced accuracy favours the declared rates. Failure: calibration error is negligible or favours the declared rates.

Verdicts: favours model: mean >= delta and lower bound > 0; favours reference: mean <= -delta and upper bound < 0; negligible: the whole interval within (-delta, delta); uncertain: anything else.

#### Every cell

Balanced accuracy as the median, then the mean with its interval. The other metrics are medians, and lower is better:

| Model | Set | Balanced accuracy | Log loss | Brier | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: |
| generative | I0 | 0.362 (0.370 [0.347, 0.391]) | 3.403 | 0.800 | 0.233 |
| generative | I2 | 0.383 (0.390 [0.367, 0.414]) | 3.756 | 0.876 | 0.372 |
| generative_hurdle | I0 | 0.393 (0.379 [0.354, 0.402]) | 1.857 | 0.730 | 0.159 |
| generative_hurdle | I2 | 0.415 (0.396 [0.369, 0.422]) | 2.002 | 0.725 | 0.217 |
| generative_fitted_poisson | I0 | 0.425 (0.419 [0.399, 0.439]) | 3.423 | 0.793 | 0.230 |
| generative_periodic | I1 | 0.520 (0.509 [0.486, 0.531]) | 2.955 | 0.555 | 0.174 |
| generative_periodic | I3 | 0.521 (0.511 [0.487, 0.534]) | 3.204 | 0.574 | 0.192 |
| generative_periodic_hurdle | I1 | 0.529 (0.522 [0.493, 0.551]) | 1.448 | 0.520 | 0.119 |
| generative_periodic_hurdle | I3 | 0.535 (0.528 [0.500, 0.558]) | 1.606 | 0.547 | 0.177 |
| filter_declared | R | 0.422 (0.417 [0.392, 0.442]) | 4.624 | 0.879 | 0.377 |
| filter_hurdle | R | 0.469 (0.456 [0.425, 0.484]) | 2.124 | 0.709 | 0.248 |
| filter_fitted_poisson | R | 0.476 (0.477 [0.449, 0.503]) | 4.642 | 0.828 | 0.382 |

#### Estimands

Mean differences, with households improved and worsened. Positive values favour the fitted model, that is, higher accuracy or lower loss and calibration error:

| Key | Comparison | Balanced accuracy | Log loss | Brier | Calibration error | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F0 | generative_hurdle@I0 vs generative@I0 | +0.009 [−0.010, +0.029], 9 / 11, uncertain | +1.645 [+1.392, +1.903], 20 / 0, favours model | +0.063 [+0.038, +0.092], 18 / 2, favours model | +0.058 [+0.030, +0.087], 16 / 4, favours model | success |
| FR | filter_hurdle@R vs filter_declared@R | +0.039 [+0.015, +0.062], 16 / 4, favours model | +2.374 [+2.134, +2.613], 20 / 0, favours model | +0.175 [+0.124, +0.232], 19 / 1, favours model | +0.119 [+0.087, +0.153], 20 / 0, favours model | success |
| F1 | generative_periodic_hurdle@I1 vs generative_periodic@I1 | +0.013 [−0.010, +0.036], 10 / 10, uncertain | +1.592 [+1.342, +1.839], 20 / 0, favours model | +0.056 [+0.026, +0.089], 16 / 4, favours model | +0.054 [+0.030, +0.078], 18 / 2, favours model | success |
| F2 | generative_hurdle@I2 vs generative@I2 | +0.006 [−0.017, +0.028], 10 / 10, uncertain | +1.729 [+1.485, +1.985], 20 / 0, favours model | +0.133 [+0.091, +0.176], 18 / 2, favours model | +0.136 [+0.105, +0.163], 19 / 1, favours model | success |
| F3 | generative_periodic_hurdle@I3 vs generative_periodic@I3 | +0.018 [−0.009, +0.043], 10 / 10, uncertain | +1.563 [+1.309, +1.828], 20 / 0, favours model | +0.045 [+0.007, +0.087], 14 / 6, favours model | +0.035 [+0.012, +0.060], 16 / 4, favours model | success |
| P0 | generative_fitted_poisson@I0 vs generative@I0 | +0.049 [+0.032, +0.068], 19 / 1, favours model | −0.031 [−0.388, +0.320], 10 / 10, uncertain | +0.006 [−0.023, +0.037], 10 / 10, uncertain | −0.013 [−0.042, +0.018], 9 / 11, uncertain | inconclusive |
| PR | filter_fitted_poisson@R vs filter_declared@R | +0.060 [+0.033, +0.087], 17 / 3, favours model | −0.181 [−0.692, +0.330], 9 / 11, uncertain | +0.048 [−0.017, +0.120], 12 / 8, uncertain | −0.000 [−0.035, +0.038], 6 / 14, uncertain | inconclusive |

#### Per-state recall

Mean differences, homes improved and worsened, and verdicts:

| State | Rare | F0 | FR |
| --- | ---: | ---: | ---: |
| away | no | −0.002 [−0.005, −0.001], 4 / 14, negligible | +0.145 [+0.048, +0.234], 17 / 3, favours model |
| home_active | no | −0.229 [−0.277, −0.180], 0 / 20, favours reference | −0.231 [−0.290, −0.172], 1 / 19, favours reference |
| home_inactive | no | +0.076 [+0.043, +0.111], 17 / 3, favours model | +0.219 [+0.169, +0.272], 20 / 0, favours model |
| sleeping | no | +0.047 [+0.021, +0.079], 19 / 0, uncertain | −0.018 [−0.059, +0.020], 10 / 10, uncertain |
| bed_awake | yes | −0.159 [−0.283, −0.035], 1 / 3, favours reference | −0.219 [−0.317, −0.072], 1 / 4, favours reference |
| bathroom_activity | yes | +0.181 [+0.130, +0.234], 18 / 2, favours model | +0.177 [+0.124, +0.228], 18 / 2, favours model |
| kitchen_activity | yes | +0.014 [−0.084, +0.108], 11 / 9, uncertain | −0.002 [−0.115, +0.102], 11 / 9, uncertain |

#### Mechanisms

One value per household, against zero. Inflation reductions are log ratios, also shown as ratios; accumulation reductions are slopes of overconfidence per quiet hour:

| Key | Quantity | Mean | As a ratio | Households above / below | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 | inflation_reduction_hurdle | +0.736 [+0.505, +0.950] | 2.09 [1.66, 2.58] | 17 / 3 of 20 | positive |
| M1P | inflation_reduction_poisson | −1.378 [−1.576, −1.198] | 0.25 [0.21, 0.30] | 0 / 20 of 20 | negative |
| M2 | accumulation_reduction_hurdle | +0.152 [+0.114, +0.192] |  | 20 / 0 of 20 | positive |
| M2P | accumulation_reduction_poisson | +0.160 [+0.093, +0.238] |  | 19 / 1 of 20 | positive |

#### Quiet runs

In each household's Phase 3.3 representative quiet runs, how many end in `sleeping` after the recorded windows, and the median confidence then:

| Channels | sleeping | away | home_inactive |
| --- | ---: | ---: | ---: |
| declared | 20 of 20, 0.986 | 19 of 20, 0.658 | 18 of 18, 0.684 |
| hurdle | 16 of 20, 0.908 | 13 of 20, 0.770 | 10 of 18, 0.764 |
| poisson | 14 of 20, 0.996 | 9 of 20, 0.977 | 7 of 18, 0.891 |

#### Fitted silence

Mean over channels and folds of each family's silence probability in a window, and the median over channels and folds of an active window's count variance over its mean, from the training households:

| State | Declared | Hurdle | Fitted Poisson | Active dispersion |
| --- | ---: | ---: | ---: | ---: |
| away | 0.939 | 0.971 | 0.847 | 7.239 |
| home_active | 0.429 | 0.734 | 0.230 | 11.829 |
| home_inactive | 0.808 | 0.871 | 0.561 | 6.995 |
| sleeping | 0.976 | 0.972 | 0.881 | 4.970 |
| bed_awake | 0.846 | 0.838 | 0.646 | 1.570 |
| bathroom_activity | 0.825 | 0.591 | 0.254 | 9.568 |
| kitchen_activity | 0.826 | 0.612 | 0.286 | 6.821 |

<!-- generated-summary:end -->

### What this shows and does not show

Figures are mean paired differences across the 20 homes, with 95% household
intervals and the number of homes that improved.

- **The run agrees with earlier work.**
  - The declared-rate models reproduce the Phase 3.1 and 3.2 household values
    exactly.
  - Each home's declared silence inflation equals Phase 3.3's S1 exactly.
  - What changes is the fitted channels alone.
- **Fitted hurdle channels are a pre-specified success in both primaries.**
  - **Current windows (F0).**
    - Calibration error improves by 0.058 [0.030, 0.087], in 16 of 20 homes.
    - Log loss improves by 1.645 [1.392, 1.903], in all 20.
    - Balanced accuracy is unchanged, +0.009 [−0.010, +0.029].
  - **The recursion over every window (FR).**
    - Calibration error improves by 0.119 [0.087, 0.153], and log loss by
      2.374, in all 20 homes.
    - Balanced accuracy improves too, by 0.039 [0.015, 0.062], in 16 of 20.
  - F1, F2 and F3, with the time prior or lagged windows, are successes by the
    same rule.
- **Probabilities improve most, but are not yet good.**
  - With the time prior (F1), median log loss is 1.448. That is the first
    generative cell in Phases 1 to 3 below the no-information reference's
    1.510.
  - Without the prior, the hurdle model's median log loss is 1.857 in `I0`.
  - For orientation only, not a pre-specified comparison: the Phase 1 record
    has logistic regression at `I1` with median log loss 0.925 and calibration
    error 0.098. The hurdle model with the prior has 1.448 and 0.119.
- **It removes the part of the silence overstatement the declared rates
  caused (M1).**
  - The inflation factor falls by 2.09 [1.66, 2.58] times, in 17 of 20 homes:
    from a mean log inflation of 1.219 (3.38) to 0.483 (1.62).
  - What remains is close to Phase 3.3's dependence-only 0.435 (1.55). The
    channels are still independent, so that part is untouched, as designed.
- **It nearly stops the build-up along quiet runs (M2).**
  - Overconfidence per quiet hour falls from 0.175 to 0.023, a reduction of
    0.152 [0.114, 0.192], in all 20 homes.
  - After one silent hour, the model reports `sleeping` in 13 of 20 away runs
    instead of 19. In `home_inactive` runs it is 10 of 18 instead of 18.
  - It also calls fewer sleeping runs `sleeping`: 16 of 20 instead of 20. Its
    confidence there is 0.908 instead of 0.986.
- **The cost is `home_active` recall.**
  - It falls by 0.229 [0.180, 0.277] in `I0`, in all 20 homes, and by 0.231 in
    the recursion.
  - The rare `bed_awake`, which occurs in only a few homes, falls too.
  - The gains are in `home_inactive` (+0.076 in `I0`, +0.219 in the recursion)
    and in `bathroom_activity` (+0.181). In the recursion there is also a gain
    in `away` (+0.145).
- **Silence needs its own parameter.**
  - A Poisson fitted to the mean count raises balanced accuracy by 0.049 in
    `I0` and 0.060 in the recursion. It does not improve calibration error:
    both verdicts are uncertain, so both are inconclusive (P0, PR).
  - It makes the silence overstatement four times worse (M1P): mean log
    inflation 2.597, a factor of 13.4. That is what the diagnostic's zero
    excess predicted.
  - It too slows the build-up along quiet runs (M2P). Slowing the build-up is
    therefore not specific to the hurdle.
- **The activity part is over-dispersed.** In an active window, the count's
  variance is 5 to 12 times its mean in the common states. The zero-truncated
  Poisson does not model that. It is one candidate cause of the
  `home_active` loss, not tested here.

**Not shown:**

- This is the development panel, which earlier work has inspected, so it is
  not a held-out or external claim.
- There is one seed and one declared shrinkage, 12 pseudo-windows. With
  thousands of training windows per cell, the shrinkage barely matters.
- There is no household adaptation.
- The online production filter is not changed. It models each sensor, and a
  channel-level hurdle would need a channel-level emission there. The
  recursion here has full reliability and no health or attribution layer.
- Channels remain conditionally independent, and the remaining inflation is
  the dependence Phase 3.3 measured.

## Reproducing it

```text
python scripts/run_phase3_fitted_rates.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes the record and a Markdown
summary generated from the written record.
