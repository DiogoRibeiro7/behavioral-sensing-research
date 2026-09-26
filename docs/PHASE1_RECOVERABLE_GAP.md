# Phase 1: the recoverable-information gap

The [roadmap](roadmap.md) asks how much of the real-data gap comes from what
the sensors support and how much from the current probabilistic formulation.
This experiment answers with two kinds of paired comparison:

- **Information gain.** The model is held fixed and information is added, from
  one [information set](INFORMATION_SETS.md) to a larger nested one.
- **Formulation gap.** The information is held fixed and the model changes.

The [first Phase 1 run](PHASE1_MATCHED_BASELINES.md) measured information gains
for the baselines only. This run adds three things:

- the supervised diagnostic;
- the generative model, wherever it can consume a set exactly;
- the interactions that show whether the two kinds of difference add up.

**Status: exploratory.** The development homes have been inspected throughout
earlier work, so nothing here is confirmatory evidence.

## Protocol

The protocol is fixed by the defaults of `GapProtocol` in
`sensor_modeling.datasets.recoverable_gap` and by
`scripts/run_phase1_recoverable_gap.py`. It was committed before any household
was scored.

- **Homes and folds.** The folds come from
  `artifacts/phase1/household_splits.json`. It freezes the two cross-fitted
  folds of the first Phase 1 run over the 20 single-resident development
  homes. Each fold is held out once, so every home is scored exactly once, by
  models never fitted on it. Each recording is verified against its SHA-256.
- **Grid.** The America/Los_Angeles time zone and a 5-minute step. Every model
  is scored at the same moments of every home.
- **Sets.** The four nested sets at their default resolution:

  | Label | Set | Information |
  | --- | --- | --- |
  | `I0` | `current` | per-channel activations in the current window |
  | `I1` | `current+time_of_day` | plus the local hour |
  | `I2` | `current+history` | plus the three previous windows |
  | `I3` | `current+time_of_day+history` | plus both |

- **Models.**

  | Name | Model | Fitted on |
  | --- | --- | --- |
  | `diagnostic` | `GradientBoostingBaseline`, the [supervised diagnostic](BASELINES.md#the-supervised-diagnostic) | the fold's training homes |
  | `logistic` | `LogisticBaseline`, L2-regularised multinomial logistic regression | the fold's training homes |
  | `generative` | the generative filter's model restricted to the set, described below | nothing; every parameter is the declared default |
  | `state_frequency` | smoothed training state frequencies, the no-information reference | the fold's training homes |
  | `filter` | the production pipeline, unrestricted, as a reference outside every set | nothing |

- **Metrics.** For each household:
  - balanced accuracy;
  - log loss;
  - Brier score;
  - expected calibration error over 10 confidence bins;
  - recall of each state.
- **Statistics.** Every cell is summarised across households by its median and
  mean. The mean has a 95% household bootstrap interval. Every comparison is a
  paired household difference with the same kind of interval. There are
  10,000 resamples and the seed is 0.
- **No tuning.** Every setting is declared in advance, and no fold has
  development households, so no household is used to choose anything. The
  frozen external cohort is not touched.

## What each model receives

| Model | `I0` | `I1` | `I2` | `I3` |
| --- | --- | --- | --- | --- |
| `diagnostic` | matched | matched | matched | matched |
| `logistic` | matched | matched | matched | matched |
| `generative` | matched | unsupported | matched | unsupported |
| `filter` | reference | reference | reference | reference |

### The generative model inside a declared set

The online filter is recursive, so it cannot be placed in a declared set. Its
generative model can. Under the filter's own prior, transition and emission
models, the posterior given only the windows a set declares is

```text
p(s_t | x_{t-k}, ..., x_t) ∝ [ ... ((π ⊙ ℓ_{t-k}) T ⊙ ℓ_{t-k+1}) T ... ] ⊙ ℓ_t

π     stationary distribution of the default ontology, the filter's own prior
T     transition over one step
ℓ_j   likelihood of window j under the filter's default emission models
k     0 in I0; 3, the set's history depth, in I2
```

`sensor_modeling.datasets.restricted_filter` computes it.

- **Exactness.** A test checks that the result equals the production
  `MultimodalBayesFilter` started one step before the first declared window and
  fed those windows alone.
- **Input.** It reads the same per-channel counts as every other model. The
  Poisson emission is linear in a window's count, and sensors on one channel
  share one rate, so the pooled count gives exactly the likelihood of the
  separate sensors. A registry where that would not hold is refused.
- **Deployment metadata.** Through the default emission models it knows how many
  sensors pool into each channel. The filter always has this metadata. The
  discriminative models see only whether a channel is instrumented.
- **Reliability and attribution.** Every sensor has full reliability and
  attribution. The production pipeline derives both from the whole observation
  history, which no declared set includes.
- **Prediction.** It reports the posterior's most probable state. It never
  abstains, and neither does any other model here.

### What is not compared

- **The generative model in `I1` and `I3`.** The current generative model has
  no time-of-day input. Its default ontology is time-homogeneous. Its optional
  circadian term rescales transition rates by hour. That acts only through
  unbounded recursion and says nothing about how probable a state is at a
  given hour. Placing it in `I1` would either ignore the hour or need a term
  the model does not have.
- **The production filter in any set.** It conditions on every earlier window
  and on health and attribution layers built from the whole history. Its
  information strictly contains `I0` and `I2`. It neither contains nor is
  contained in `I1` or `I3`, which have the hour but only 20 minutes of
  history. It is scored as a reference, and each comparison with it states
  that relation.
- **The circadian candidate.** Its v0.3 profile was fitted on every development
  home, so no household of this panel is held out from it.

The artifact records every one of these, with its reason.

## Reading the comparisons

- **An information gain** holds the model fixed. It is what the added
  information is worth to that model.
- **A formulation gap** holds the information fixed. It is what one model gains
  over another from how it uses the same inputs, which includes whether it
  learns from labels.
- **An interaction** is the difference between what the same added information
  is worth to two models. Equivalently, it is how much a formulation gap
  changes when information is added. When it is not zero, information and
  formulation do not add up, and no split of an overall gap into an
  information part and a formulation part is unique.
- **A comparison with the filter** is not matched. Where the filter's
  information contains the set, a model that beats it cannot owe that to
  seeing more. The difference still mixes formulation with how the filter uses
  its extra history, so it is not a formulation gap.

The roadmap writes the decomposition as a sum:
`observed gap = information gap + formulation gap + residual`. This experiment
reports its terms as separate paired comparisons, with their interactions. It
does not claim that they add up.

## Results

The summary below is generated from the published record,
`artifacts/phase1/phase1-recoverable-information-gap.json`, by
`render_summary`. A test checks that it still matches the record.

<!-- generated-summary:start -->

### Recoverable-information gap: summary

Generated from the record `phase1-recoverable-information-gap`: protocol `e1082697cfbf`, commit `9431b9a2f0e3`. Status: exploratory: the development homes have been inspected in earlier work.

- 20 households in 2 cross-fitted folds. Each is scored once, by models never fitted on it.
- Intervals are 95% household bootstrap intervals from 10,000 resamples.
- Differences are paired by household. A positive value favours the first model, or the larger set.
- No setting was selected on any data.

#### What each model receives

| Model | I0 | I1 | I2 | I3 |
| --- | ---: | ---: | ---: | ---: |
| diagnostic | matched | matched | matched | matched |
| logistic | matched | matched | matched | matched |
| generative | matched | unsupported | matched | unsupported |

The production `filter` is matched to no set. It conditions on every earlier window, so its information strictly contains `I0` and `I2`, and neither contains nor is contained in `I1` or `I3`.

#### Balanced accuracy

Median over households, then the mean with its 95% interval:

| Model | I0 | I1 | I2 | I3 |
| --- | ---: | ---: | ---: | ---: |
| diagnostic | 0.418 (0.418 [0.397, 0.436]) | 0.519 (0.515 [0.477, 0.548]) | 0.505 (0.506 [0.477, 0.532]) | 0.596 (0.590 [0.550, 0.624]) |
| logistic | 0.354 (0.344 [0.317, 0.370]) | 0.488 (0.477 [0.434, 0.516]) | 0.392 (0.383 [0.348, 0.414]) | 0.500 (0.498 [0.450, 0.539]) |
| generative | 0.362 (0.370 [0.347, 0.391]) | unsupported | 0.383 (0.390 [0.367, 0.414]) | unsupported |
| state_frequency | 0.167 (0.161 [0.156, 0.164]) | 0.167 (0.161 [0.156, 0.164]) | 0.167 (0.161 [0.156, 0.164]) | 0.167 (0.161 [0.156, 0.164]) |

Production filter, unrestricted: 0.429 (0.426 [0.399, 0.451]).

#### Probability quality

Median over households. Lower is better:

| Model | Set | log_loss | brier | calibration_error |
| --- | ---: | ---: | ---: | ---: |
| diagnostic | I0 | 1.450 | 0.648 | 0.113 |
| diagnostic | I1 | 4.636 | 0.624 | 0.190 |
| diagnostic | I2 | 1.370 | 0.565 | 0.116 |
| diagnostic | I3 | 2.605 | 0.492 | 0.135 |
| logistic | I0 | 1.242 | 0.665 | 0.109 |
| logistic | I1 | 0.925 | 0.470 | 0.098 |
| logistic | I2 | 1.176 | 0.631 | 0.107 |
| logistic | I3 | 0.909 | 0.448 | 0.098 |
| generative | I0 | 3.403 | 0.800 | 0.233 |
| generative | I2 | 3.756 | 0.876 | 0.372 |
| state_frequency | I0 | 1.510 | 0.750 | 0.095 |
| state_frequency | I1 | 1.510 | 0.750 | 0.095 |
| state_frequency | I2 | 1.510 | 0.750 | 0.095 |
| state_frequency | I3 | 1.510 | 0.750 | 0.095 |
| filter | unbounded | 3.620 | 0.812 | 0.314 |

#### What added information is worth

The model is held fixed. Each cell gives the mean paired difference in balanced accuracy, its interval, and how many households improved:

| Added | Sets | diagnostic | logistic | generative |
| --- | ---: | ---: | ---: | ---: |
| time of day | I0 to I1 | +0.097 [+0.067, +0.123], 18/20 | +0.133 [+0.112, +0.152], 20/20 | unsupported |
| history | I0 to I2 | +0.088 [+0.075, +0.101], 20/20 | +0.039 [+0.029, +0.047], 19/20 | +0.021 [+0.016, +0.026], 19/20 |
| history | I1 to I3 | +0.075 [+0.064, +0.087], 20/20 | +0.021 [+0.013, +0.028], 18/20 | unsupported |
| time of day | I2 to I3 | +0.084 [+0.057, +0.107], 18/20 | +0.115 [+0.097, +0.132], 20/20 | unsupported |
| both | I0 to I3 | +0.172 [+0.141, +0.199], 19/20 | +0.154 [+0.129, +0.175], 19/20 | unsupported |

#### What the formulation is worth

The information is held fixed. Each cell gives the first model minus the second, in balanced accuracy:

| Set | diagnostic − logistic | diagnostic − generative | logistic − generative |
| --- | ---: | ---: | ---: |
| I0 | +0.074 [+0.057, +0.091], 20/20 | +0.048 [+0.027, +0.071], 16/20 | −0.025 [−0.047, −0.004], 7/20 |
| I1 | +0.038 [+0.016, +0.059], 13/20 | unsupported | unsupported |
| I2 | +0.123 [+0.101, +0.147], 20/20 | +0.116 [+0.088, +0.144], 20/20 | −0.007 [−0.035, +0.019], 9/20 |
| I3 | +0.092 [+0.074, +0.109], 20/20 | unsupported | unsupported |

#### Interactions

How much more the added information is worth to the first model than to the second, in balanced accuracy. A value away from zero means that information gains and formulation gaps do not add up:

| Sets | diagnostic vs logistic | diagnostic vs generative | logistic vs generative |
| --- | ---: | ---: | ---: |
| I0 to I1 | −0.036 [−0.054, −0.019], 3/20 | not estimable | not estimable |
| I0 to I2 | +0.050 [+0.039, +0.061], 20/20 | +0.067 [+0.054, +0.082], 20/20 | +0.018 [+0.008, +0.028], 17/20 |
| I1 to I3 | +0.054 [+0.041, +0.068], 19/20 | not estimable | not estimable |
| I2 to I3 | −0.031 [−0.052, −0.014], 3/20 | not estimable | not estimable |
| I0 to I3 | +0.018 [+0.002, +0.031], 16/20 | not estimable | not estimable |

#### Against the production filter

These comparisons are not matched. Where the filter's information contains the set, a model that scores higher cannot owe it to seeing more:

| Model | Set | Relation | Balanced accuracy difference |
| --- | ---: | ---: | ---: |
| diagnostic | I0 | the filter's information strictly contains the set | −0.008 [−0.031, +0.019], 10/20 |
| diagnostic | I1 | neither set contains the other | +0.089 [+0.056, +0.120], 18/20 |
| diagnostic | I2 | the filter's information strictly contains the set | +0.081 [+0.052, +0.111], 19/20 |
| diagnostic | I3 | neither set contains the other | +0.164 [+0.132, +0.194], 19/20 |
| logistic | I0 | the filter's information strictly contains the set | −0.081 [−0.103, −0.059], 1/20 |
| logistic | I1 | neither set contains the other | +0.051 [+0.018, +0.083], 15/20 |
| logistic | I2 | the filter's information strictly contains the set | −0.043 [−0.069, −0.016], 6/20 |
| logistic | I3 | neither set contains the other | +0.072 [+0.035, +0.107], 18/20 |
| generative | I0 | the filter's information strictly contains the set | −0.056 [−0.069, −0.043], 0/20 |
| generative | I2 | the filter's information strictly contains the set | −0.035 [−0.044, −0.025], 1/20 |

#### Per-state recall

Median over the households where the state occurs, at `I2` (the richest set every model can consume) and for the filter. The record holds every cell:

| State | diagnostic | logistic | generative | filter |
| --- | ---: | ---: | ---: | ---: |
| away | 0.764 | 0.114 | 0.116 | 0.379 |
| home_active | 0.293 | 0.250 | 0.503 | 0.561 |
| home_inactive | 0.481 | 0.251 | 0.136 | 0.189 |
| sleeping | 0.388 | 0.966 | 0.832 | 0.755 |
| bed_awake | 0.000 | 0.000 | 0.250 | 0.250 |
| bathroom_activity | 0.767 | 0.369 | 0.262 | 0.249 |
| kitchen_activity | 0.637 | 0.551 | 0.567 | 0.572 |

#### Not compared

- `generative` in I1, I3: the current generative model has no time-of-day input: its default ontology is time-homogeneous, and its optional circadian term rescales transition rates, so it acts only through unbounded recursion, not as a statement about the state at an hour; the one fitted profile (v0.3) was fitted on every development home and has no held-out score there.
- `filter` in I0, I1, I2, I3: the production filter is recursive: its belief conditions on every earlier window and on health and attribution layers built from the whole history, so it belongs to no declared set; its information strictly contains I0 and I2 and neither contains nor is contained in I1 or I3.
- `circadian filter` in any: the v0.3 circadian candidate's profile was fitted on every development home, so no household of this panel is held out from it.

<!-- generated-summary:end -->

### What this shows and does not show

Balanced-accuracy figures are mean paired differences, with their intervals
and the number of the 20 homes that improved.

- **The run reproduces the first one.** `logistic` uses the same folds, seed
  and settings as in the [first Phase 1 run](PHASE1_MATCHED_BASELINES.md). Its
  scores are identical in every set.
- **History is worth more to the diagnostic than to the linear model.**
  - The diagnostic gains +0.088 [+0.075, +0.101] from `I0` to `I2`, and +0.075
    once time of day is present.
  - `logistic` gains +0.039 and +0.021.
  - The interaction, +0.050 [+0.039, +0.061], favours the diagnostic in all 20
    homes.

  How much recent history is worth depends on the model that uses it. That
  accounts for part of the distance between the first run's +0.02 to +0.04
  and the +0.140 of the lost gradient-boosted diagnostic.
- **The diagnostic in `I3` has a median of 0.596**, with a mean of 0.590
  [0.550, 0.624]. The lost diagnostic reported 0.607 on an unrecorded split.
  The two are not paired, so this is not a reproduction.
- **On the current window alone, the generative formulation is competitive.**
  In `I0`, the generative model's median is 0.362 against 0.354 for
  `logistic`, which trails it by −0.025 [−0.047, −0.004]. The diagnostic leads
  it by +0.048 [+0.027, +0.071], in 16 of 20 homes.
- **The generative model gains little from recent history.**
  - It gains +0.021 [+0.016, +0.026] from `I0` to `I2`, against +0.088 for the
    diagnostic.
  - Its gap to the diagnostic therefore grows from +0.048 in `I0` to +0.116
    [+0.088, +0.144] in `I2`, in all 20 homes.
  - Both models receive the same information there, so this is a formulation
    gap. It is the largest matched one measured here.
- **The production filter's extra history adds a little.** The filter's median
  is 0.429. It beats its own model restricted to `I2` by 0.035 [0.025, 0.044],
  in 19 of 20 homes. Its extra is unbounded history plus its health and
  attribution layers.
- **The diagnostic beats the filter while seeing strictly less.** In `I2` it
  leads by +0.081 [+0.052, +0.111], in 19 of 20 homes. That difference cannot
  come from information. It lies in formulation, which includes training on
  labels. The observed gap, the diagnostic in `I3` against the filter, is
  +0.164 [+0.132, +0.194]. It is not matched: `I3` has the hour, while the
  filter has longer history.
- **The generative model cannot take time of day.** The hour is worth +0.084 to
  +0.133 to the discriminative models, and the current generative model has no
  input for it. That part of the gap cannot be measured within the generative
  formulation, and closing it needs a model change, which is Phase 3.
- **Balanced accuracy and probability quality disagree.**
  - The diagnostic's probabilities are poor once the hour is added. Its median
    log loss is 4.64 in `I1` and 2.61 in `I3`, above the no-information 1.51,
    with calibration error 0.19 and 0.14.
  - The generative model and the filter are overconfident too: median log loss
    3.4 to 3.8, calibration error 0.23 to 0.37.
  - Only `logistic` improves on the reference's log loss in every set. The
    diagnostic does so only without the hour, at 1.45 in `I0` and 1.37 in
    `I2`.

  A formulation gap in balanced accuracy is therefore not a gap in probability
  quality.
- **The errors differ in kind.**
  - In `I2` the diagnostic's lead comes from `away` (median recall 0.76,
    against 0.12 for the generative model and 0.38 for the filter),
    `bathroom_activity` (0.77 against 0.26) and `home_inactive` (0.48 against
    0.14).
  - It does worse on `sleeping` (0.39 against 0.83) and `home_active` (0.29
    against 0.50).
- **The gaps do not add up.** Every estimable interaction is away from zero.
  Time of day is worth less to the diagnostic than to `logistic` (−0.036 from
  `I0`, −0.031 from `I2`). History is worth more (+0.050 from `I0`, +0.054 from
  `I1`). No single split of the observed gap into an information part and a
  formulation part is supported.

Every figure above is scored by the most probable state, without abstention.
The filter's 0.429 is therefore not comparable with the published 0.420, which
counts abstentions as errors and comes from a different split.

## The artifact

`run_recoverable_gap` writes one [experiment record](EXPERIMENT_ARTIFACTS.md),
schema 1.1, and `gap_summary.render_summary` generates the Markdown summary
from the written record alone.

| Field | Contents |
| --- | --- |
| `configuration` | the protocol and its SHA-256 |
| `inputs` | the frozen split file and every recording, by SHA-256 |
| `split`, `information_set` | the folds and the four sets, each with its digest |
| `models` | every model's full configuration |
| `household_metrics` | every metric per household, keyed `model@set`, plus `filter` |
| `results.cells` | each model and set: summaries, per-state recall, or why it is unsupported |
| `results.information_gains`, `formulation_gaps`, `interactions`, `against_filter` | paired comparisons, with per-household differences |
| `results.unsupported` | every comparison not made, with its reason |
| `intervals` | every mean paired difference with its interval, labelled for quoting |

## Reproducing it

```text
python scripts/run_phase1_recoverable_gap.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes
`phase1-recoverable-information-gap.json`, and a Markdown summary generated
from it with the same name and a `.md` suffix.
