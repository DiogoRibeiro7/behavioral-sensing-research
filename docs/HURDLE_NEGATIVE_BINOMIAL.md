# The hurdle negative-binomial channel model

The [posterior predictive checks](PHASE3_PREDICTIVE_CHECKS.md) found the fitted
hurdle model's active count **systematically under-dispersed**.

- **How much.** The observed active variance is 4 to 7 times the zero-truncated
  Poisson's in every common state, in every development home.
- **How it grows.** The excess grows with the square of the mean: the
  household slope of log variance on log mean is 1.94, where the model gives
  1.15. That is the negative binomial's variance function, `μ + α μ²`.

This page documents an over-dispersed alternative for the active count, the
`hurdle_nb` family in `sensor_modeling.datasets.channel_models`. It keeps the
hurdle's separation of silence from activity, and replaces only the active
count's distribution.

**Status: implemented, not evaluated.** No comparison with the hurdle-Poisson
model has been run. The existing `hurdle` family is unchanged and remains the
default everywhere it was used.

**What it does not address.** The same checks found long quiet runs in
systematic excess, and their pre-declared routing named within-state temporal
dependence as the next model family. A different marginal count distribution
cannot produce runs. This family addresses the dispersion only.

## The likelihood

For channel `c`, state `s` and a window's pooled count `n`:

```text
P(n = 0     | s) = π
P(n = k ≥ 1 | s) = (1 − π) · NB(k; μ, α) / (1 − NB(0; μ, α))
```

`NB(k; μ, α)` is the negative binomial with mean `μ`, variance `μ + α μ²` and
size `r = 1/α`:

```text
NB(k; μ, α) = Γ(k + r) / (Γ(r) k!) · (r / (r + μ))^r · (μ / (r + μ))^k
NB(0; μ, α) = (1 + α μ)^(−1/α)
```

At `α = 0`, `NB` is the Poisson with rate `μ`, so the model is then exactly the
hurdle-Poisson model with the same `π` and `μ`. The log-likelihood of an active
window is:

```text
log P(n = k | s) = log(1 − π) + log NB(k; μ, α) − log(1 − NB(0; μ, α))
```

As with the hurdle-Poisson model, `log k!` is shared by every state and is
dropped from `loglik`. `log_pmf` keeps it, and its probabilities sum to one.

## The parameters

| Symbol | Meaning |
| --- | --- |
| `π` | probability that the channel is silent in a window of the state |
| `μ` | mean of the untruncated negative binomial; not the mean of an active window |
| `α ≥ 0` | dispersion: how much faster than its mean the count's variance grows |
| `r = 1/α` | size; infinite at the Poisson limit |
| `m = μ / (1 − NB(0))` | mean count of an active window |
| `v = (μ + (1 + α) μ²) / (1 − NB(0)) − m²` | variance of an active window's count |

- **Silence and activity.** `π` alone decides silence, as in the
  hurdle-Poisson model. `μ` and `α` together decide how much activity there is,
  once there is any.
- **Dispersion.**
  - At `α = 0` an active window's variance is the zero-truncated Poisson's.
  - At `α = 1` the untruncated variance is `μ + μ²`, twice the mean at a mean
    of one and eleven times at a mean of ten.
  - At a fixed active mean, a larger `α` puts more windows at a single
    activation and more in the far tail. That is the shape the predictive
    checks found.
- **The other limit.** As `α` grows with `m` held fixed, the zero-truncated
  negative binomial tends to the logarithmic series. A count table with many
  ones and a long tail can then have its likelihood rise toward that limit
  without a maximum. The fit stops at `α = 100`, where the likelihood is flat,
  and marks the cell `at_bound`.

## Fitting

Everything is fitted per channel and state on the fold's training households
only, as for the hurdle-Poisson model. `W` is the windows, `Z` the silent ones,
`P = W − Z` the active ones, `S` the total count and `κ` the pseudo-windows
(12, one hour):

1. **Silence** `π̂ = (Z + κ π₀) / (W + κ)`, unchanged from the hurdle-Poisson
   model.
2. **Active mean** `m̂ = (S + κ m₀) / (P + κ)`, unchanged. `π₀` and `m₀` are
   the declared model's.
3. **Dispersion** `α̂`: it maximises a penalised profile likelihood.
   - For each `α`, `μ` is solved so that the active mean is `m̂`. With `α`
     fixed, the zero-truncated negative binomial is an exponential family in
     `k`, so this is the maximum-likelihood `μ` for that `α`.
   - The likelihood is that of the training households' active-count table,
     plus `κ` pseudo-windows distributed as the zero-truncated Poisson with
     mean `m̂`.
   - Those pseudo-windows carry no information about the mean, only the
     Poisson's shape. By Gibbs' inequality their log-likelihood is largest at
     `α = 0`, so they shrink `α̂` toward the Poisson limit with the weight of
     `κ` windows.
   - `α` is searched over `{0} ∪ [10⁻⁴, 100]`: a grid of 41 points in `log α`,
     refined by bounded Brent search around the best one.
   - A positive `α` is preferred to 0 only if it gains more than `10⁻⁶` nats
     per window. The likelihood is flat to second order near 0, and a smaller
     gain is below the formula's rounding.
4. **Rate** `μ̂` solves `μ / (1 − NB(0; μ, α̂)) = m̂`, which has one root for any
   `m̂ > 1`.

### Little data

- **A few active windows.** The pseudo-windows dominate, and `α̂` is near 0: the
  fit is close to the hurdle-Poisson model.
- **None.** No training household has a window of the state. The declared prior
  of the household the model is for supplies `m₀`, and the pseudo-windows alone
  give `α̂ = 0`, the declared model exactly.
- **Nothing to fit.** A channel with no active window in a state has only
  pseudo-windows, and `α̂ = 0`.
- **Shrinkage is second order.** Near 0, the penalty grows with `α²`. One
  extreme count can therefore keep `α̂` slightly above 0 even under a very
  strong prior. It falls toward 0 as `κ` grows.

### What the fit needs

The dispersion is not determined by a count's sum and sum of squares, so the
fit needs the whole table of active counts.

- **Where the table lives.** `ChannelStatistics` carries it as `active_counts`,
  an `ActiveCounts`: the distinct counts and their frequency in each state.
  `home_statistics` builds it, and `combine_statistics` merges it across
  households.
- **Statistics without it.** Statistics built from sums alone still fit the
  `hurdle` and `poisson` families. `dispersion` refuses them.

## Partial pooling

`HouseholdChannels.models(dispersion=...)` builds a household's hurdle
negative-binomial models from its pooled silence and active mean, as for the
hurdle-Poisson model, and the population's dispersion. A household's own
dispersion is not pooled: a few active windows cannot estimate it stably.
Without `dispersion`, `models()` returns the hurdle-Poisson models as before.

```python
population = fit_channel_models(training, resolution=resolution, pseudo_windows=12.0)
household = adapt_channels(population, recording, household=home, resolution=resolution,
                           config=PoolingConfig(288.0), until=cutoff)
dispersion = {
    channel: population.dispersion(channel, terms.expected)
    for channel, terms in channel_likelihoods(recording.registry, resolution)[0].items()
}
models = household.models(dispersion=dispersion)
```

## Numerical stability

- **The negative binomial's coefficients.** `log Γ(k + r) − log Γ(r) − log k!`
  is evaluated as `−log k − log B(k, r)`. It stays accurate as `r` grows, and
  matches SciPy's negative binomial to better than `10⁻¹²` relative, even at a
  count of a million.
- **The zero-truncation term.** `log(1 − NB(0))` uses `log(−expm1(x))` near 0
  and `log1p(−exp(x))` below it.
- **The Poisson limit.** `α = 0` has its own branch, the Poisson formulas.
  `α = 10⁻¹²` agrees with it to `10⁻⁶` at counts up to 500.
- **Solving for `μ`.** `μ` is found by Brent's method on
  `μ + m · expm1(log NB(0))`, over `[10⁻⁹, m]`.
- **Clipping.** No probability is clipped. Counts of a million give finite
  log-likelihoods, which fall with the count.

## Serialisation

- **The model.** `HurdleNBChannel.to_dict` and `from_dict` round-trip a model
  exactly.
- **The table.** `ActiveCounts.to_dict` and `from_dict` do the same for a table.
- **The fit.** `FittedChannels.dispersion_to_dict` gives each channel and state
  its active windows, `α`, whether it reached the bound, the size, `μ` and the
  active mean.
- **Digests.** `FittedChannels.to_dict` and its SHA-256 are unchanged: the
  active-count table is never part of them. A fit's digest therefore still
  identifies it, and every published population digest is reproduced by the
  current code.

## Using it

```python
from sensor_modeling.datasets.channel_models import HURDLE_NB, fit_channel_models

population = fit_channel_models(training, resolution=resolution, pseudo_windows=12.0)
models = population.models(recording.registry, resolution, HURDLE_NB)
loglik = total_loglik(models, counts)
```

`HurdleNBChannel` satisfies the same `ChannelModel` protocol as
`HurdleChannel`, so `total_loglik`, `filter_recursion` and
`restricted_posteriors` take it unchanged.

## What this does not do

- **No evaluation.** It does not show whether the family improves prediction
  or calibration. That is the next pre-specified comparison, not run here.
- **The production pipeline.** The online filter still models each sensor
  with its declared emissions.
- **Time.** It does not model dependence in time within a state, which the
  predictive checks also found.
