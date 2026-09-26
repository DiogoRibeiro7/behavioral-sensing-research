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
