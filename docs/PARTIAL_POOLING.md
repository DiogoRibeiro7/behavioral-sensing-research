# Partial pooling of household parameters

Roadmap item 3.4 asks to separate population-level parameters from
household-specific effects, and to evaluate partial pooling before any
unconstrained per-home fitting. `sensor_modeling.datasets.partial_pooling`
provides a small, explicit framework for that. Its first application is the
fitted hurdle channel parameters of the
[fitted-rates experiment](PHASE3_FITTED_RATES.md).

This is an implementation. No evaluation is reported, and no default or
published result changes.

## Three levels on one scale

For a household `h` and a parameter estimated as a mean:

| Level | The household gets | Setting |
| --- | --- | --- |
| population structure | `θ_pop`, fitted on other households | `κ → ∞` |
| household adaptation | `θ_pop` plus its own deviation, shrunk toward zero | `0 < κ < ∞` |
| unconstrained per-household fitting | its own raw estimate `θ_raw` | `κ → 0` |

The framework is the middle level. Given the household's `n` observations with
total `s`:

```text
θ_raw = s / n
w     = n / (n + κ)
θ̂     = θ_pop + w · (θ_raw − θ_pop)      which equals (s + κ θ_pop) / (n + κ)
```

- **Population plus shrunk deviation.** The estimate is the population
  parameter plus the household deviation `δ = w · (θ_raw − θ_pop)`, shrunk
  toward zero by `w`.
- **The pooling strength `κ`.** It is how many observations the population is
  worth, and it is the one setting, in `PoolingConfig`.
- **Effective shrinkage.** `1 − w = κ / (n + κ)` is the share of the estimate
  that comes from the population.
- **Bayesian reading.** For a probability, `θ̂` is the posterior mean under a
  Beta prior with mean `θ_pop` and `κ` pseudo-observations. For a Poisson
  mean, it is the posterior mean under a Gamma prior with mean `θ_pop` and `κ`
  pseudo-windows.

Its behaviour follows from the formula:

- **No data.** `w = 0`, so the household gets `θ_pop` exactly. It is
  computed in the first form so that this holds bit for bit.
- **Little data.** The estimate stays close to the population, since the
  deviation is at most `w` times the raw one.
- **Much data.** `w → 1`, and the estimate approaches the household's own.
- **Always bounded.** The estimate lies between the raw and the population
  value, so a probability stays a probability.
- **Deterministic.** The estimate is closed-form.

## The default strength

`κ = 288` five-minute windows, which is 24 labelled hours. That is the pooling
strength the [periodic state prior](PERIODIC_STATE_PRIOR.md) declares for its
household deviations. The two priors act on different scales, so the number
is shared, not the prior. It is a declared default, not a tuned one.

## First application: hurdle channel parameters

**Why these parameters.** The Phase 3.3 record shows large differences
between households in how often each channel is silent:

- In `home_inactive`, the bedroom channel is silent in 22% to 98% of windows,
  depending on the home.
- In `home_active`, the living-room channel is silent in 17% to 84%.
- `living_motion` pools one to four sensors, depending on the installation.

The fitted-rates experiment gave every home its fold's population parameters.
Pooling lets a home with enough labelled data move toward its own values.

For each instrumented channel and state:

- **Silence.** The silence probability `π` is pooled from the household's
  silent windows among its windows.
- **Activity.** The mean count of an active window, `m`, is pooled from its
  total count among its active windows. The activity rate `μ` is then solved
  from the pooled `m`, as the population's is.
- **Validity.** Both pooled values lie between the household's own and the
  population's. So `π` stays in `(0, 1)` and `m ≥ 1`, and the hurdle model
  stays valid.

```python
from sensor_modeling.datasets.channel_models import adapt_channels, fit_channel_models
from sensor_modeling.datasets.partial_pooling import PoolingConfig

population = fit_channel_models(training, resolution=resolution, pseudo_windows=12.0)
household = adapt_channels(
    population,
    recording,
    household="hh101",
    resolution=resolution,
    config=PoolingConfig(strength=288.0),
    until=adaptation_end,      # None: an unseen household, population only
)
models = household.models()          # HurdleChannel per channel
rows = household.diagnostics()       # raw, pooled, population, shrinkage
```

### No leakage

- **No double counting.** A household the population was fitted on is
  refused. Its data would count twice, and a held-out score would no longer be
  held out.
- **Only the household's own windows up to `until`.** A test checks that
  changing any later annotation leaves the adaptation unchanged. The household
  can then be scored on the windows after `until`.
- **An unseen household.** `until=None` gives exactly the population models.

### Diagnostics and provenance

`HouseholdChannels.diagnostics()` returns one row per channel, state and
parameter (`silence` or `active_mean`). Each row has:

- the household's observations;
- the raw household estimate, or `None` without data;
- the pooled estimate;
- the population estimate;
- the effective shrinkage.

`HouseholdChannels.to_dict()` records the household, the pooling strength, the
population's SHA-256 and the households it was fitted on, the `until` moment,
and every estimate with its inputs. `from_dict` rebuilds it exactly from JSON,
and refuses a payload whose derived values disagree with its inputs.

## What is not done

- **No evaluation.** Whether pooled parameters improve on population ones
  needs its own pre-specified experiment, with an adaptation window before
  each household's scored period.
- **One parameter family.** Only the hurdle channel parameters are pooled.
  The periodic state prior already has a population plus a shrunk household
  deviation, a Gaussian prior on its Fourier coefficients, and it is
  unchanged.
- **The activity mean is only approximately Bayesian.** It is pooled as the
  same weighted average. For the zero-truncated Poisson that is a pragmatic
  estimator, not an exact posterior mean.
