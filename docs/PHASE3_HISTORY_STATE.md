# Phase 3.2: the explicit history state

Phase 1 found recent history worth +0.021 household balanced accuracy to the
generative filter, against +0.088 to the supervised diagnostic. With the
hierarchical time prior, [Phase 3.1](PHASE3_TIME_PRIOR.md) measured +0.002.
The [explicit history state](HISTORY_STATE.md) lets each channel's recent
activations condition its current rate, rather than leaving the past to the
filtered posterior alone. This pre-specified experiment measures on the
development panel:

- whether the history state improves the generative model on identical
  information;
- whether the model now recovers materially more from recent history than the
  original model did;
- how it compares with the matched discriminative models;
- whether time and history information are complementary.

**Status: pre-specified, development panel.** Every setting, comparison and
criterion was frozen before any household was scored. The development homes
have been inspected in earlier work, so this is not a final held-out claim, and
the frozen external cohort is not touched.

## Protocol

The protocol is `artifacts/phase3/history_protocol.json`. It was committed
before scoring, and the run refuses to start if the code's protocol differs
from it by a single value. It freezes:

- **Households.** The 20 single-resident development homes and the two frozen
  Phase 1 folds (`artifacts/phase1/household_splits.json`, recorded by digest).
  Each home is scored once, by models never fitted on it.
- **Sets.** The four nested sets, `I0` to `I3`.
- **Models.** Each is scored only where it can consume the set exactly:

  | Model | Scored in | What it is |
  | --- | --- | --- |
  | `generative` | `I0`, `I2` | the original generative model, restricted to the set |
  | `generative_history` | `I2` | the original model with the explicit history state |
  | `generative_periodic` | `I1`, `I3` | the hierarchical time-prior model of Phase 3.1 |
  | `generative_periodic_history` | `I3` | the hierarchical time-prior model with the history state |
  | `diagnostic` | `I0` to `I3` | the supervised diagnostic, gradient-boosted trees |
  | `logistic` | `I0` to `I3` | regularised multinomial logistic regression |
  | `state_frequency` | `I0` to `I3` | the no-information reference |

- **Why each model is included.**
  - `generative_periodic` is included because Phase 3.1 met its pre-specified
    success rule.
  - `generative_periodic_history` is declared here, before any result of the
    history state was observed.
  - The history state needs the three lagged windows, so the scorer refuses it
    in `I0` and `I1`. There, the models without it stand in for the family.
- **Hyperparameters.**
  - The history window is `k = 3` lagged windows: the lags of `I2` and `I3`.
  - The coefficients' Gaussian prior has precision 1.
  - The time prior keeps its Phase 3.1 settings: `K = 2`, `λ₀ = 1`, `λ = 24`.
  - The diagnostic and logistic regression keep their declared settings.
  - Nothing is tuned.
- **Fitting.** Each fold's history coefficients and time prior are fitted on
  that fold's training homes only. Held-out homes get the population prior and
  the fold's coefficients.
- **Metrics.** Household balanced accuracy, log loss, Brier score, expected
  calibration error over 10 bins, and per-state recall.
- **Bootstrap.** 10,000 household resamples, and 95% percentile intervals for
  both the mean and the median paired difference. All seeds are 0.

### Estimands

Each is a paired household difference. A positive value favours the first
model or the larger set. There are two kinds:

- a **formulation** estimand compares two models on the same set;
- an **information** estimand compares one model family across nested sets
  that differ only in recent history.

No estimand changes the model and the information at once.

| Key | Role | Kind | First | Second | Question |
| --- | --- | --- | --- | --- | --- |
| H1 | primary | formulation | `generative_history` `I2` | `generative` `I2` | what the history state adds on identical information |
| H2 | primary | information | `generative_history` `I2` | `generative` `I0` | what recent history is worth to the history-state model |
| H3 | primary | formulation | `diagnostic` `I2` | `generative_history` `I2` | formulation gap at `I2` |
| H4 | primary | formulation | `logistic` `I2` | `generative_history` `I2` | formulation gap at `I2` |
| R1 | reference | information | `generative` `I2` | `generative` `I0` | what recent history is worth to the original model |
| S1 | secondary | formulation | `generative_periodic_history` `I3` | `generative_periodic` `I3` | what the history state adds when the hour is known |
| S2 | secondary | information | `generative_periodic_history` `I3` | `generative_periodic` `I1` | what recent history is worth to the model with both |
| S3 | secondary | formulation | `diagnostic` `I3` | `generative_periodic_history` `I3` | formulation gap at `I3` |
| S4 | secondary | formulation | `logistic` `I3` | `generative_periodic_history` `I3` | formulation gap at `I3` |
| R2 | reference | information | `generative_periodic` `I3` | `generative_periodic` `I1` | what recent history is worth to the time-prior model |

H2 and R1 are information estimands for two model families. Their difference,
on the same homes, is exactly H1, because both start from the same `I0` model.

**Per-state recall.** It is reported for H1, H2 and S1. A state is rare if its
share of labelled time in either fold's training homes is below 5%.

**Time and history interaction.** For each model family and household:

```text
interaction = (score in I3 − score in I1) − (score in I2 − score in I0)
```

That is, the history gain with the hour minus the gain without it. A positive
value means time and history are complementary. The families are:

| Family | `I0` | `I1` | `I2` | `I3` |
| --- | --- | --- | --- | --- |
| generative with the history state | `generative` | `generative_periodic` | `generative_history` | `generative_periodic_history` |
| generative without the history state | `generative` | `generative_periodic` | `generative` | `generative_periodic` |
| diagnostic | `diagnostic` | `diagnostic` | `diagnostic` | `diagnostic` |
| logistic | `logistic` | `logistic` | `logistic` | `logistic` |

### Criteria

The criteria compare effect sizes, with their uncertainty, against declared
minimal important differences `δ`. They are not significance tests.

| Metric | `δ` |
| --- | ---: |
| balanced accuracy | 0.02, the smallest Phase 1 information gain reported as real |
| log loss | 0.05 |
| Brier score | 0.01 |
| calibration error | 0.02 |
| per-state recall | 0.05 |

Each comparison gets one verdict:

| Verdict | When |
| --- | --- |
| favours model | mean ≥ `δ` and the interval's lower bound > 0 |
| favours reference | mean ≤ `−δ` and the interval's upper bound < 0 |
| negligible | the whole interval lies within `(−δ, δ)` |
| uncertain | anything else |

The overall rules are fixed in advance:

- **Success.** H1 balanced accuracy favours the history-state model, and
  neither H1 log loss nor H1 calibration error favours the original model.
- **Failure.** H1 balanced accuracy is negligible or favours the original
  model.
- **Inconclusive.** Anything else.
- **S1.** It is judged by the same rule, as the result when the hour is known.
- **Materially more than +0.021.** The history-state model recovers materially
  more from recent history than the original model if H1 favours it. H1 is
  exactly H2 − R1 on the same homes.

The earlier +0.021 is not a threshold. It is re-measured in this run as R1,
and the comparison is paired against that. A fixed threshold would ignore both
its uncertainty and the pairing, and the minimal important difference already
states what size of change matters.

## Results

The results will be published here from the record
`artifacts/phase3/phase3-explicit-history.json`.

## Reproducing it

```text
python scripts/run_phase3_history.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes the record and a Markdown
summary generated from the written record.
