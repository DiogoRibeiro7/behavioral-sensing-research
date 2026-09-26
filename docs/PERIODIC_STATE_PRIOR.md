# Periodic state prior

Phase 1 found time of day worth +0.084 to +0.133 household balanced accuracy to
the discriminative models ([recoverable-information gap](PHASE1_RECOVERABLE_GAP.md)).
The generative model could not take part in those comparisons. Its optional
circadian term rescales how quickly each state is left at each hour, but
nothing in the model states how probable a state is at a given hour.

`sensor_modeling.datasets.periodic_prior` adds that statement. It is an
explicit probability model with three properties:

- it has population structure;
- it adapts to each household, with shrinkage toward the population;
- its time representation is cyclic.

This is the "hierarchical time structure" hypothesis of the roadmap (3.1). The
original generative model is unchanged and remains the baseline.

## The model

For local wall-clock hour `h`, read exactly as the
[information sets](INFORMATION_SETS.md#time-of-day) read it:

```text
θ(h) = 2π (h + ½) / 24                                   angle of the hour's midpoint
φ(h) = (1, sin θ, cos θ, …, sin Kθ, cos Kθ)               Fourier basis, K harmonics

η_{g,s}(h) = (w_s + d_{g,s})ᵀ φ(h)                        periodic log-probability
π_{g,h}(s) = exp η_{g,s}(h) / Σ_{s'} exp η_{g,s'}(h)      prior of state s at hour h
```

- `w` is the population effect, one row of `2K + 1` coefficients per state.
- `d_g` is household `g`'s deviation from it.
- The intercept of a row sets how common the state is.
- Each sine-cosine pair adds one daily cycle with its own amplitude and peak.
  For the first harmonic, `peak_hour(β, γ)` from `time_features` reads the peak
  of the state's log-probability. The softmax couples the states, so the peak of
  the probability itself can shift slightly.

Being periodic, the log-probabilities are continuous across midnight, with
23:00 as close to 00:00 as any two neighbouring hours are. The prior is a
proper distribution at every hour: every probability is positive, and they sum
to one.

## Estimation

`n_g(h, s)` is the labelled time, in hours, that household `g` spent in state
`s` during hour `h`. `hour_state_counts` counts it at the evaluation's
prediction moments, with the same local-hour reading as the information sets.

```text
population:  ŵ   = argmax_w  Σ_g Σ_{h,s} n_g(h,s) log π_h(s; w)       − (λ₀/2) ‖w‖²
household:   d̂_g = argmax_d  Σ_{h,s} n_g(h,s) log π_h(s; ŵ + d)       − (λ/2)  ‖d‖²
```

- **Population.** The population effect is fitted only from households whose
  labels the evaluation permits for fitting.
- **Shrinkage.** The household deviation is the posterior mode under the prior
  `d_g ~ N(0, λ⁻¹ I)`. It is shrunk toward the population with precision `λ`,
  the pooling strength.
  - A household with no labelled time gets `d = 0`, exactly the population.
  - A household the prior has no deviation for, an unseen household, also gets
    the population.
  - As a household's labelled time grows, it moves from the population toward
    its own fit. A larger `λ` pools more. A deviation coefficient informed by
    `I` hours' worth of Fisher information is shrunk by roughly `λ / (λ + I)`.
- **Uniqueness.** `λ₀` is a small ridge that makes the population fit unique:
  adding the same function of the hour to every state leaves the softmax
  unchanged. Both objectives are strictly concave, so each fit is unique.
- **Determinism.** Newton's method from zero, with a backtracking line search,
  finds each fit. It stops once the Newton decrement is below rounding error.
  Fitting is deterministic and takes milliseconds, because the data reduce to a
  24 × states table.

The settings, `PeriodicPriorConfig`, are declared rather than tuned, and are
recorded with every fit:

| Setting | Default | Why |
| --- | ---: | --- |
| `harmonics` `K` | 2 | enough for a morning and an evening structure; `K` is limited to 1–11, as for the cyclic hour encoding |
| `population_precision` `λ₀` | 1 labelled hour | only makes the fit unique; the population is fitted from thousands of hours |
| `household_precision` `λ` | 24 labelled hours | about a day, so a household's own pattern dominates only once it has several days of labels |

A fitted prior serialises to JSON with its settings, coefficients, household
deviations and the households it was fitted on. `sha256()` gives its digest.

## In the generative model

The prior enters the filter through the ontology's existing circadian term,
not through a second time-of-day mechanism. `PeriodicStatePrior.ontology` sets
the circadian stickiness to

```text
c_s(h) = π_h(s) / π(s)                  π: the base ontology's stationary distribution
Q_h    = diag(1 / c(h)) Q               the generator in force during hour h
```

Two properties follow and are tested:

- **Equilibrium.** `(π ⊙ c(h)) Q_h = π Q = 0`, so at every hour the chain's
  equilibrium is exactly the prior `π_h`.
- **Speed.** The `π_h`-weighted mean exit rate equals the base chain's, so the
  prior changes where the chain settles, not how fast it moves on average.

The frozen v0.3 candidate uses the same term, with a different estimate:

| | v0.3 circadian profile | Periodic state prior |
| --- | --- | --- |
| Estimate | per-hour occupancy lift: a state's share at the hour over its overall share, each hour separately, clipped | smooth Fourier log-probabilities |
| Households | one pooled profile | population plus shrunk household deviations |
| Statement about the state at an hour | none; it only rescales exit rates | `π_h(s)` |
| Fitted on | every development home, frozen | the permitted training households |

The v0.3 profile and candidate are unchanged.

### Restricted to an information set

In a set with time of day, the [restricted generative model](PHASE1_RECOVERABLE_GAP.md#the-generative-model-inside-a-declared-set)
takes a periodic prior:

- **Start.** It starts from `π_h`, the model's own prior at hour `h`, instead of
  the stationary `π`.
- **Transitions.** It steps with `Q_h`.
- **Hour.** `h` is the row's `hour_of_day`, the hour the set declares. The set
  declares nothing finer, so that hour also stands for every declared window.
  Near an hour boundary, the 20 minutes of `I3` can span two clock hours; the
  model uses the prediction moment's.
- **Exactness.** A test checks that the result equals the production filter,
  with the prior's ontology, started from `π_h` and fed the declared windows,
  wherever those windows share one hour.

The prior needs the hour, so it is supported in `I1` and `I3` and refused in
`I0` and `I2`. There, the original generative model is scored.

## In the matched evaluation

`GapProtocol(folds, periodic_prior=PeriodicPriorConfig())` adds the model
`generative_periodic` to the
[recoverable-information-gap experiment](PHASE1_RECOVERABLE_GAP.md):

- **Fitting.** Each fold's population prior is fitted on that fold's training
  households only (`fold_priors`). A test changes every label of a fold's
  held-out households and checks that the fold's prior does not change. The
  record stores each fold's prior with its digest and the households it was
  fitted on.
- **Held-out households.** They have no labels the protocol permits, so they get
  the population prior. Household adaptation is not used in the evaluation.
- **Comparisons.** The model is scored in `I1` and `I3`, against the diagnostic
  and logistic regression. Its information gain from `I1` to `I3`, and the
  matching interactions, are reported. Its cells in `I0` and `I2` are recorded
  as unsupported, with the reason.
- **Default unchanged.** Without the option, the protocol, its digest and the
  published result are unchanged.

This change runs no development-panel or held-out comparison, and makes no
claim about how the model performs.

## Assumptions and limits

- **Hour resolution.** Time of day is the local clock hour. Variation within an
  hour is not modelled.
- **Daylight saving.** Time of day follows the clock, as in the information
  sets. At fall back the repeated hour counts twice; at spring forward a grid
  moment named in the skipped hour is read as written.
- **Stationary patterns.** A household's pattern is assumed stable over its
  recording. Day of week and seasonal drift are not modelled. The weekend
  lie-in found in the development homes would need its own term.
- **Pooled counts.** The population fit weighs households by their labelled
  time, as the v0.3 fit does.
- **Declared pooling.** The pooling strength is declared, not estimated from the
  spread between households.
- **Adaptation needs labels.** A household deviation needs labelled time from
  that household, recorded before the moments it is used for.
  `hour_state_counts(until=...)` enforces the cut-off.
- **An equilibrium, not an exact marginal.** The chain relaxes toward `π_h` at
  the rate the declared dwell times set. `π_h` is the equilibrium at each
  hour, not the exact marginal of a chain that has run for a finite time.
- **Rare states.** No probability is zero, but a state seldom labelled at an
  hour can receive a very small one.

## Example

```python
from datetime import timedelta

from sensor_modeling.datasets import (
    PeriodicPriorConfig,
    fit_periodic_prior,
    hour_state_counts,
)
from sensor_modeling.datasets.matched_evaluation import _regular_moments

step = timedelta(minutes=5)
counts = {
    home: hour_state_counts(recording, _regular_moments(recording, step), step=step)
    for home, recording in training_recordings.items()
}
prior = fit_periodic_prior(counts, config=PeriodicPriorConfig())

prior.hourly()        # (24, states) population prior
ontology = prior.ontology()  # default ontology with the prior as its circadian term
adapted = prior.adapt("hh999", early_counts)  # labels recorded before use only
```
