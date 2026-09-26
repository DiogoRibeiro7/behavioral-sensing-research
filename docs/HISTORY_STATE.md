# Recent-history state

Phase 1 found that recent history is worth +0.088 household balanced accuracy
to the supervised diagnostic, and +0.021 to the generative filter
([recoverable-information gap](PHASE1_RECOVERABLE_GAP.md)). With the
[periodic state prior](PERIODIC_STATE_PRIOR.md) in place, it is worth +0.002
([Phase 3.1](PHASE3_TIME_PRIOR.md)).

The filter's posterior carries the past, but its emission is memoryless: a
window's activations are assumed to depend on the current state alone. Real
activity is bursty and has refractory periods. A bathroom visit, for example,
often starts with a burst and then falls quiet while the person is still
there. This component tests the roadmap's hypothesis 3.2: the generative model
needs an explicit history state, instead of expecting the filtered posterior
to carry all the useful temporal information.

`sensor_modeling.fusion.history` implements the component. It is small and
auditable. It is not a recurrent network, a transformer or an embedding. The
original `MultimodalBayesFilter` is unchanged and remains the comparator.

## The history state

For channel `c`, a room and a modality as in the
[information sets](INFORMATION_SETS.md), and step windows of length `Δ`:

```text
n_{c,t}   activations of channel c in window t
h_{c,t} = n_{c,t−1} + n_{c,t−2} + … + n_{c,t−k}         k = 3
```

- **The window.** `k = 3` is the history depth Phase 1 justified: the lagged
  windows of `I2` and `I3`. It is the only window, declared in advance.
- **Observed or missing.** `h_{c,t}` is observed only if every one of the `k`
  windows was observed for the channel. Otherwise it is missing, never zero:
  - before `k` complete windows have passed since the filter started, resumed
    or had a gap;
  - when an update covers a gap or a partial step;
  - when a sensor of the channel was unreliable.
- **Bounded memory.** The filter keeps `k` windows of per-channel counts in a
  ring buffer.

## The model

The emission rate of each Poisson event sensor `i` on channel `c` is
conditioned on the channel's history:

```text
λ_i(s | h) = λ_i(s) · exp(η_c(s, h))
η_c(s, h)  = β_{s, r(c,s)} · ( log(1 + h) − E_s[ log(1 + H) ] )     if h is observed
η_c(s, h)  = 0                                                       if h is missing

H ~ Poisson(k Δ Λ_c(s)),   Λ_c(s) = Σ_{i ∈ c} λ_i(s)
```

- **The reference.** `E_s[log(1 + H)]` is what the memoryless model itself
  expects of the history if the state had been `s` throughout.
  - A history that matches that expectation leaves the rate unchanged.
  - Observed silence is informative, because it lies below the expectation.
  - A missing history is exactly neutral.
- **Coefficients.** `β > 0` means that recent activity beyond what the state
  predicts makes more activity now likely, as in a burst. `β < 0` means it
  makes activity now less likely, as in a refractory period.
- **Groups.** `r(c, s)` relates the channel's room to the state's: `own room`
  or `other room` for a state that names a room, and `any room` otherwise.
  The default ontology has eleven coefficients, not dozens of lag parameters.

Per sensor, conditioning adds to the tempered log-likelihood

```text
w_i · ( n_i η − λ_i(s) Δ (e^η − 1) )
```

where `w_i` is the sensor's evidence weight times its reliability and
attribution, as in the original filter.

### Why this is consistent

- **No double counting.** Each window's counts enter the likelihood exactly
  once, as that window's current observation. The history only changes how
  likely they are. The filter is the exact forward recursion of a
  Markov-switching autoregressive model, and no evidence is counted twice.
  Adding the past windows again as separate evidence of the current state
  would count them twice, which is why that design was not used.
- **The comparator is exact.** With every `β = 0` the filter is the original
  one. A test checks this to 10⁻¹².
- **The ontology is unchanged.** The same states, transitions and declared
  rates are used. The history changes the emission only.

### Fitting

`fit_history_model` fits each group's coefficient from labelled training
households only. The rows are the labelled windows in state `s` and the
channels with relation `r`, where the current window and every history window
were observed:

```text
β̂ = argmax_β  Σ_rows ( y β z − μ e^{β z} )  −  (τ/2) β²

y  activations in the current window
μ  the memoryless expected activations in state s
z  log(1 + h) − E_s[log(1 + H)]
```

- **Uniqueness.** The objective is strictly concave, and Newton's method from
  zero finds its maximum deterministically.
- **The rates stay declared.** The declared rates are not refitted, so `β`
  measures only what the history adds.
- **Ridge.** `τ = 1` keeps a group with few windows near zero, the memoryless
  model.
- **No other households.** No household outside the ones given is read.

## Diagnostics

`HistoryAwareBayesFilter.explain()` returns a `PosteriorDecomposition` for the
last prediction:

```text
log p_t(s) = log p̂_t(s) + C_t(s) + R_t(s) − log Z_t
```

| Term | Field | Meaning |
| --- | --- | --- |
| `log p̂_t` | `prior_transition` | the prior, or the previous posterior carried by the transition model |
| `C_t` | `current` | the current window's memoryless log-likelihood, over every sensor |
| `R_t` | `history` | what conditioning on the recent history added |

- **`log_odds(state, versus)`** gives each term's share of the log posterior
  odds between two states; the shares sum to the posterior log odds.
- **`explain()`** reports the reported state against the runner-up, term by
  term. For each channel it adds the history count, why the history is missing
  if it is, and that channel's share of the history term.
- **Exactness.** A test checks, at every step of a run, that the three terms
  reconstruct the posterior exactly and that the channel shares sum to the
  history term.

## Persistence

`snapshot()` stores the belief, the clock, the history buffer, the history
model's digest and the step. `restore()` refuses a snapshot from another
history model, another step or another set of channels, such as another
household's. A run that is snapshotted and restored mid-way continues exactly
as an uninterrupted one. A new filter, for example for another household,
always starts cold.

## Matched information sets

`restricted_posteriors(..., history_model=...)` restricts the model to a
declared set.

- **The current window.** Its history is the row's three lagged windows, so
  the set must declare recent history with at least `k` lags. That makes
  `I2` and `I3` supported, and `I0` and `I1` not.
- **Earlier windows.** Each lagged window's own history lies outside the set,
  so it is treated as missing.
- **Exactness.** A test checks that the result equals the online filter fed
  only the declared windows.
- **With the periodic prior.** In `I3` it combines with the periodic state
  prior.

No development-panel or held-out evaluation is part of this change.

## Assumptions and limits

- **One feature per channel.** Each channel's history is its own recent
  intensity. Cross-room patterns, such as activity moving from the kitchen to
  the bathroom, reach the model only through the posterior. The time since
  the last activity and the recent room distribution are candidates for later
  work.
- **Step-aligned updates.** History is defined on consecutive full steps, and
  an irregular update resets it to missing.
- **Declared rates.** The history multiplies declared rates that are known to
  be misspecified, and `β` absorbs some of that misspecification.
- **Labels.** Fitting needs labelled training households, as the periodic
  prior does.
- **Event channels only.** Only Poisson event sensors with a room have a
  history. Other modalities keep the memoryless emission.
