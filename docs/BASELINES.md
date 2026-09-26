# Baseline models

Phase 2 of the [roadmap](roadmap.md) sets up reference models before any new
model is proposed. `sensor_modeling.datasets.baseline_models` provides four.
Each consumes exactly the rows that the
[matched evaluation runner](MATCHED_EVALUATION.md) builds from one
[information set](INFORMATION_SETS.md), so each is scored on the same
information as every other model. None of them is a proposal, and this page
makes no claim about which performs better.

## The four baselines

| Name | Class | Uses | Probabilities |
| --- | --- | --- | --- |
| `state_frequency` | `StateFrequencyBaseline` | training labels only | add-one smoothed training state frequencies, identical for every row |
| `persistence` | `PersistenceBaseline` | the previous window's evidence | its inner model's, applied to the previous window |
| `logistic` | `LogisticBaseline` | every permitted column | L2-regularised multinomial logistic regression |
| `tree` | `TreeBaseline` | every permitted column | add-one smoothed leaf frequencies of one depth-limited tree |

All four implement one interface, `Baseline`:

- `predict_proba(rows)` returns a probability for every state in the label
  space, in the label space's order;
- `predict(rows)` reports the most probable state with those probabilities.

A fitted baseline refuses rows whose columns or label space differ from what
it was fitted on. A model fitted under one information set therefore cannot be
scored under another.

### Persistence

The protocol never shows a model the true state of a held-out household, so
persistence cannot copy the previous true state. The only previous state
available is one inferred from the previous window's evidence. Persistence
fits an inner model, `LogisticBaseline` by default, to map one window's
per-channel counts to that window's state. At prediction it applies the inner
model to the previous window (lag 1) and reports the result for the current
moment. The one added assumption is that the state has not changed since the
previous window. Time of day and older history are not used.

**At the first timestamp** of a recording, the previous window precedes the
recording and every lag-1 column is NaN. There is no previous state, so
persistence reports the training state frequencies. It does the same for any
row with no previous-window evidence at all.

Persistence needs recent history. It refuses to fit on an information set
without it, and `baseline_suite` leaves it out of such sets.

## Fixed settings

Every setting is fixed before any household is scored. None is searched or
tuned, and no baseline reads the development rows.

| Baseline | Settings |
| --- | --- |
| `state_frequency` | add-one smoothing |
| `persistence` | inner `logistic` with the settings below |
| `logistic` | `C = 1.0`, L-BFGS, `max_iter = 1000`, features standardised on training rows, hour one-hot encoded, add-one probability for states without training rows. `hour_encoding="cyclic"` offers the [cyclic hour](INFORMATION_SETS.md#time-of-day) instead; the suite keeps one-hot |
| `tree` | `max_depth = 6`, `min_samples_leaf = 20`, hour as an ordinal column, add-one smoothed leaves |

`logistic`, `tree` and the inner model of `persistence` take the run's seed as
their random state.

Missing evidence is encoded without imputation. Each evidence column is
represented twice: once as its value with NaN replaced by zero, and once as a
separate missing indicator. An uninstrumented channel, or history from before
the recording starts, never looks like a silent sensor. The encoding uses no
statistics from any data.

## Which metrics mean what

All four baselines report probabilities, so the runner computes every metric
for each. Not every metric says the same thing for every model.

| Metric | `state_frequency` | `persistence` | `logistic` | `tree` |
| --- | --- | --- | --- | --- |
| Balanced accuracy, per-state recall | a floor by construction: recall is 1 for the majority training state and 0 for every other | meaningful | meaningful | meaningful |
| Log loss, Brier score | the no-information reference: what a model scores without using any row | meaningful; first-timestamp rows carry the training frequencies | meaningful | meaningful |
| Calibration error | reflects only how the majority state's training frequency differs from its held-out accuracy; says nothing about discrimination | meaningful | meaningful | coarse: confidences take only as many values as there are leaves |

## No probability is zero

A reported probability of zero means certainty that a state cannot occur. If
the state then occurs, log loss for that row is set by the metric's numerical
floor, `-log(1e-12) ≈ 27.6`, rather than by anything the model learned. A few
such rows can dominate a household's log loss.

No baseline reports zero. Wherever a baseline estimates a probability by
counting, it adds one pseudo-observation of every state in the label space.
This is Laplace's rule of succession: the posterior mean under a uniform prior,
the conventional parameter-free choice. It is fixed, not tuned.

| Baseline | Where the pseudo-count enters |
| --- | --- |
| `state_frequency` | training state counts: `(n_s + 1) / (N + K)` for `N` training rows and `K` states |
| `tree` | each leaf's counts: `(n_leaf,s + 1) / (n_leaf + K)`, the Laplace correction for probability estimation trees |
| `logistic` | a state with no training rows cannot be represented by the fitted model, so it gets `1 / (N + K)`; the represented states keep their relative odds and are scaled to make the row sum to one |
| `persistence` | through its inner model, and through the training frequencies it reports at the first timestamp |

A state with no training rows is therefore never predicted, and never ruled
out. Its held-out log loss per row is set by the training data, `log(N + K)`
for `state_frequency`, `logistic` and `persistence`, not by a numerical floor.
Adding the same pseudo-count to every state never changes which state is most
probable, so the smoothing leaves reported states, balanced accuracy and
per-state recall unchanged.

## Example

This example runs the suite on six simulated homes, with no downloaded data.

```python
from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    HouseholdSplit,
    baseline_suite,
    nested_information_sets,
    run_matched_evaluation,
)
from sensor_modeling.simulation import HouseholdConfig, simulate


def household(seed):
    """A simulated home, packaged like an annotated recording."""
    sim = simulate(HouseholdConfig(days=3, seed=seed))
    labels = tuple(
        ActivityInterval(e.state.value, e.start, e.end, e.state)
        for e in sim.truth.episodes
    )
    return CasasRecording(sim.registry, sim.observations, labels)


information_set = nested_information_sets()[-1]
result = run_matched_evaluation(
    {f"sim{seed}": household(seed) for seed in range(1, 7)},
    split=HouseholdSplit(
        "example", train=("sim1", "sim2", "sim3"), test=("sim4", "sim5", "sim6")
    ),
    information_set=information_set,
    models=baseline_suite(information_set),
    seed=0,
    data_source="simulator",
)
for name, model in result.record.results["models"].items():
    print(name, model["summary"]["balanced_accuracy"]["median"])
```

Simulated homes are easier than real ones (see
[Real-data validation](real_data.md)), so figures from this example say
nothing about field performance.
