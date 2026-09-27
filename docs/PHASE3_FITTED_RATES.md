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

The results will be published here from the record
`artifacts/phase3/phase3-fitted-rates.json`.

## Reproducing it

```text
python scripts/run_phase3_fitted_rates.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes the record and a Markdown
summary generated from the written record.
