# Observation-model mismatch

A posterior is normalised over the states. Evidence that every state explains
poorly can still leave one state far ahead of the others, so the prediction
looks confident while the observation model fits none of them. Confidence
answers "which state?". This diagnostic answers a different question:

> How surprising is the evidence under every plausible behavioural state?

ROADMAP Phase 4 lists predictive checks for observation-model mismatch among
the structures a better abstention rule could use. This page describes the
diagnostic that measures it, window by window, against the fitted channel
models.

**Status: diagnostic infrastructure.**

- **What it changes.** It sets no threshold, makes no abstention and changes no
  decision. It runs no evaluation.
- **The open question.** Whether mismatch predicts errors is for a later,
  pre-specified Phase 4 experiment, reported as a risk–coverage curve.

## The count laws

Every channel family the filter's recursion uses is a hurdle
(`sensor_modeling.datasets.observation_mismatch.CountLaw`). Under state `s`, a
channel is silent with probability `π`. Otherwise its count follows a
zero-truncated negative binomial with untruncated mean `μ` and dispersion `α`:

```text
P(X = 0 | s) = π
P(X = k | s) = (1 − π) · NB(k; μ, α) / (1 − NB(0; μ, α)),    k ≥ 1
```

| Channel model | Law |
| --- | --- |
| hurdle negative binomial | `π`, `μ`, `α` as fitted |
| hurdle | `α = 0`: a zero-truncated Poisson |
| fitted or declared Poisson with mean `λ` | `π = e^{−λ}`, `μ = λ`, `α = 0`: exactly the Poisson |

The models' `loglik` drops the `log k!` every state shares, since a posterior
does not depend on it. A surprise must be absolute to be compared with what the
state itself produces, so these laws are normalised over the counts. The
channels are independent given the state, as the model assumes.

## What is measured

For each window, from its available channels (`MismatchTrace`):

| Quantity | Definition | Range |
| --- | --- | --- |
| channel log-probability | `log P(X_c = x_c \| s)`, per channel and state | `≤ 0` |
| window log-likelihood | their sum over the channels, per state | `≤ 0` |
| best state, best achievable | the state under which the window is most probable, and its log-likelihood | — |
| upper, lower tail | `P(X_c ≥ x_c \| s)` and `P(X_c ≤ x_c \| s)`, exact | `(0, 1]` |
| two-sided tail | twice the smaller tail, at most 1: a valid p-value | `(0, 1]` |
| most lenient tail | each channel's tail under the state that makes it largest | `(0, 1]` |
| state-conditional surprise | `−log p(x \| s)` | `≥ 0` |
| standardised surprise | the surprise minus its exact mean under `s`, over its exact standard deviation: how unusual the window is for evidence `s` itself produces | real |
| least standardised surprise | its smallest value over the states | real |
| channel excess | each channel's surprise minus its mean under `s`; the excesses sum to the window's | real |
| predictive log-probability | `log p(x \| past)` under the filter's prediction before the window, per channel and for the window | `≤ 0` |
| predictive two-sided tail | each channel's two-sided tail under the prediction's mixture | `(0, 1]` |
| predictive gap | the window's predictive log-probability minus the best achievable | `≤ 0` |
| activity pattern | which available channels fired: its log-probability, exact tail and standardised surprise per state | — |
| pattern training support | how many training windows of each state showed the pattern, out of those that could | counts |
| beyond training | a count above every count the training windows showed on the channel | — |

The mean and variance of the surprise are the laws' entropy and varentropy.
Under the model, the standardised surprise at the true state has mean 0 and
variance 1, and each channel's two-sided tail is below `L` with probability at
most `L`. The tests check both by simulation.

### What each measure identifies

| Pattern | Seen in |
| --- | --- |
| **out-of-support observation** | a count beyond the training support; a channel whose most lenient two-sided tail is extreme |
| **unseen combination of channels** | an activity pattern no comparable training window showed; a small most lenient pattern tail; each channel typical of some state, but no state typical of all, so the least standardised surprise is large |
| **abnormal event burst** | a small most lenient upper tail: the count is extreme even for the state that expects the most activity |
| **impossible or nearly impossible evidence** | a very low best achievable log-likelihood; a large least standardised surprise; tails far below the reported levels under every state |
| **evidence the prediction did not expect** | a large predictive gap with a typical best state: the dynamics, not the observation model, missed |

The last row separates a surprise the transition caused from a mismatch of
the observation model. When some state explains the evidence well, but the
prediction gave it little weight, the gap is large and the least standardised
surprise small.

### No single threshold

No measure is collapsed to one decision.

- **Per window.** The trace keeps every raw term, and derives the rest.
- **Distributions.** A household's summary gives each measure's distribution
  over its windows.
- **Reported levels.** The summary gives the share of windows beyond each of
  the tail levels 10⁻², 10⁻⁴, 10⁻⁶ and 10⁻⁹. The levels describe the tails;
  none is a decision rule.
- **Per channel.** How often each channel's count is beyond training, or its
  upper tail beyond each level, and its mean excess at the best state.

With many windows and channels, some small tails are expected under the model
itself. The per-level shares describe this; they do not test it.

## Missing evidence and sensor failure

Each channel in each window is:

- **`available`**, and scored;
- **`missing`**, with no count recorded;
- **`failed`**, a known sensor failure, given as intervals per channel. A
  window overlapping one is failed on that channel.

Only available channels enter any measure. A failed sensor's stuck counts or
dead silence are not behaviour, so they are never scored as novelty. The
household helper also leaves failed channels out of the recursion that makes
the prediction, as a filter aware of the failure would.

- **No evidence.** A window with no available channel has no evidence. Every
  measure of it is `null`, and it is counted separately. It is neither typical
  nor surprising.
- **Unknown support.** A channel without training counts is never beyond
  training. A pattern no training window could show, because none instrumented
  every available channel, is unknown rather than unseen.

## Numerical method

Everything is computed in log space.

- **Upper tails** come from the negative binomial's or Poisson's log survival
  function. Near double precision's underflow, below about 10⁻²⁹⁰, they are
  summed directly over their first 64 terms, with a geometric bound on the
  rest.
- **Lower tails** sum the probabilities of every count up to the observed one,
  with a running log-sum-exp.
- **The mean and variance** of the surprise sum every count up to the one
  whose tail is below 10⁻¹⁵, found by doubling on the log survival function,
  or at most a million counts.
- **Pattern tails** are exact: every pattern of up to 20 available channels is
  enumerated.
- **Tests.** The tests check the tails against brute-force sums to 10⁻¹⁰ in
  log, and to a relative 10⁻¹² for tails below 10⁻⁴³⁰. Counts in the millions
  stay finite.

## In the experiment record

Schema 1.5 adds an optional `observation_mismatch` section to every experiment
record ([experiment artifacts](EXPERIMENT_ARTIFACTS.md)). It holds:

- the trace format and the reported tail levels;
- the model: its family, and the households it was fitted on;
- the state order;
- per household, the summary of its trace, and the trace's file with its
  SHA-256, or `null` if the trace is not written.

`validate_record` checks the section:

- no scored household among those the model was fitted on;
- window counts that add up, and every share in `[0, 1]`;
- every tail level reported;
- every trace reference a file with a digest.

Records written before 1.5 are migrated with the section `null`.

- **Trace files.** `MismatchTrace.write` writes canonical gzip-compressed JSON,
  the same bytes for the same trace, and returns the digest the record keeps.
  `MismatchTrace.read(path, sha256)` refuses a file that does not match it.
- **What a trace holds.** Only the raw terms: statuses, counts, each channel's
  log-probability and tails per state, the laws' moments and silence
  probabilities, the prediction, and the training support. Every other
  quantity is recomputed from them.

## Using it

```python
from sensor_modeling.datasets.observation_mismatch import (
    household_mismatch, model_description, pattern_reference,
)
from sensor_modeling.evaluation import MismatchReport

models = population.models(recording.registry, resolution, "hurdle_nb", ontology)
patterns = pattern_reference(recordings, population.fitted_on, resolution=resolution)
trace = household_mismatch(recording, models, household=home, resolution=resolution,
                           fitted=population, patterns=patterns,
                           failures={channel: [(start, end)]})
digest = trace.write(output_dir / f"{home}.json.gz")
section = MismatchReport.from_traces(
    [trace], model_description("hurdle_nb", population),
    {home: {"file": f"{home}.json.gz", "sha256": digest}},
)
record = ExperimentRecord(..., observation_mismatch=section)
```

`mismatch_trace(laws, counts, ...)` scores any counts against any count laws,
without a recording.

## What this does not do

- **No thresholds.** It sets no threshold and makes no abstention.
- **No evaluation.** It does not show whether mismatch signals errors. That is
  Phase 4's pre-specified question.
- **The model's independence.** The window's surprise assumes the channels are
  independent given the state, as the model does. Dependence the model misses
  shows up as unusual patterns and contradictions; it is not modelled.
- **Known failures only.** CASAS records no sensor health. Failures must be
  given as intervals; an unrecorded failure looks like the evidence it
  produces.
- **Count channels only.** The online pipeline's continuous and binary
  emission models are not covered.
