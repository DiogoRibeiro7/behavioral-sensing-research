# Phase 3.1: the hierarchical time-of-day prior

Phase 1 found time of day worth +0.084 to +0.133 household balanced accuracy to
the discriminative models, but the generative model had no way to use it. The
[periodic state prior](PERIODIC_STATE_PRIOR.md) gives it one. This
pre-specified experiment measures three things on the development panel:

- whether the generative model can use the hour through the prior;
- how its result compares with the matched discriminative models;
- what the prior costs or gains in probability quality.

**Status: pre-specified, development panel.** Every setting, comparison and
criterion was frozen before any household was scored. The development homes
have been inspected in earlier work, so this is not a final held-out claim, and
the frozen external cohort is not touched.

## Protocol

The protocol is `artifacts/phase3/time_prior_protocol.json`. It was committed
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
  | `generative_untimed` | `I0`, `I2` | the hierarchical model with time disabled (see below) |
  | `generative_periodic` | `I1`, `I3` | the hierarchical model with the hour |
  | `diagnostic` | `I1`, `I3` | the supervised diagnostic, gradient-boosted trees |
  | `logistic` | `I1`, `I3` | regularised multinomial logistic regression |
  | `state_frequency` | `I1`, `I3` | the no-information reference |

- **Hyperparameters.**
  - The prior has `K = 2` harmonics and ridge `λ₀ = 1`.
  - The pooling strength `λ = 24` labelled hours is declared, not selected on
    any data.
  - The diagnostic and logistic regression keep their declared settings.
  - Nothing is tuned.
- **Fitting.** Each fold's priors are fitted on that fold's training homes
  only. Held-out homes get the population prior.
- **Metrics.** Household balanced accuracy, log loss, Brier score, expected
  calibration error over 10 bins, and per-state recall.
- **Bootstrap.** 10,000 household resamples and 95% percentile intervals for
  the mean paired difference; the median is recorded too. All seeds are 0.

### The time-disabled model

`generative_untimed` is the hierarchical model with its periodic terms removed.
Its prior is fitted exactly as the periodic one is, but on counts spread evenly
over the hours. Every periodic coefficient is then zero, and the prior is one
distribution `π̄` at every hour. A prior that ignores the hour needs no hour, so
it is expressed as a time-homogeneous ontology: the base dwell times scaled by
`π̄(s) / π(s)`, which is the same generator as constant circadian stickiness.
A test checks that it gives the same posteriors, to 10⁻⁸, as the periodic
machinery given the hour with that flat prior.

`generative_untimed` in `I0` and `generative_periodic` in `I1` therefore differ
only in the hour. `generative_untimed` against `generative` in `I0` differs only
in a fitted prior against the declared one.

### Estimands

Each is a paired household difference. A positive value favours the first
model.

| Key | Role | First | Second | Question |
| --- | --- | --- | --- | --- |
| P1 | primary | `generative_periodic` `I1` | `generative_untimed` `I0` | what the hour is worth to the hierarchical model |
| P2 | primary | `generative_periodic` `I1` | `generative` `I0` | the new model with the hour against the original without it |
| P3 | primary | `diagnostic` `I1` | `generative_periodic` `I1` | formulation gap at `I1` |
| P4 | primary | `logistic` `I1` | `generative_periodic` `I1` | formulation gap at `I1` |
| S1 | secondary | `generative_untimed` `I0` | `generative` `I0` | a fitted prior against the declared one |
| S2 | secondary | `generative_periodic` `I3` | `generative_untimed` `I2` | what the hour is worth with recent history |
| S3 | secondary | `generative_periodic` `I3` | `generative` `I2` | the new model against the original, both with history |
| S4 | secondary | `diagnostic` `I3` | `generative_periodic` `I3` | formulation gap at `I3` |
| S5 | secondary | `logistic` `I3` | `generative_periodic` `I3` | formulation gap at `I3` |
| S6 | secondary | `generative_periodic` `I3` | `generative_periodic` `I1` | what recent history is worth to the hierarchical model |

**Per-state recall.** It is reported for P1 and P2. A state is rare if its
share of labelled time in either fold's training homes is below 5%.

**Adaptation arm (secondary).** This is the one place the hierarchy's
shrinkage is used:

- **Window.** Each held-out home's deviation is fitted from its own labels in
  the first 7 days after its first observation.
- **Scoring.** Only its labelled moments after those 7 days are scored, with the
  adapted prior and with the population prior.
- **Pooling strengths.** `λ = 24` is declared. `λ = 2.4` and `λ = 240` are
  reported as sensitivity checks, and none is selected.
- **Matching.** It uses the home's own earlier labels, which no other model
  receives, so it is compared only with the population prior.

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

- **Success.** P1 balanced accuracy favours the model, and neither P1 log loss
  nor P1 calibration error favours the reference.
- **Failure.** P1 balanced accuracy is negligible or favours the reference.
- **Inconclusive.** Anything else.
- **P2.** It is judged by the same rule, as the practical improvement over the
  original model.
- **Household adaptation.** It helps if the declared arm's `I1` balanced accuracy
  favours the adapted prior, and its log loss does not favour the population
  prior.

## Reproducing it

```text
python scripts/run_phase3_time_prior.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes the record and a Markdown
summary generated from the written record.
