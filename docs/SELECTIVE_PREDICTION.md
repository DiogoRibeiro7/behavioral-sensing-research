# Selective prediction

Phase 4 asks which signal, if any, should decide when the model abstains. Paper
1 showed that confidence, entropy, margin, evidence strength and information
gain fail as standalone uncertainty signals. The candidates since then include
structural disagreement, evidence-group disagreement, observation mismatch, and
expected loss. This page describes the framework that evaluates all of them the
same way.

**Status: evaluation infrastructure.**

- **What it does.** It reports each signal's selective-prediction curves over a
  grid of coverage levels, with household-level uncertainty.
- **What it does not do.** It selects no threshold and fixes no abstention
  rule. Final thresholds are chosen on development data and frozen before
  held-out evaluation, in a later, pre-specified experiment (ROADMAP Phase 4).

## Signals and their direction

A signal is any scalar or ordered value per scored window
(`sensor_modeling.evaluation.Signal`). Its direction is always stated. Nothing
assumes that a higher value means a worse prediction.

| Direction | Rejected first | Examples |
| --- | --- | --- |
| `higher_is_riskier` | the highest values | entropy, disagreement, surprise, expected loss |
| `higher_is_safer` | the lowest values | confidence, margin |

A missing value is refused unless the signal states a policy:

- **`reject_first`** ranks missing windows before every other window for
  rejection.
- **`retain_first`** ranks them before every other window for retention.

The number of missing windows is recorded.

## Selection

At coverage `c`, the fraction `c` of windows the signal ranks safest is
retained. Windows tied at the boundary are each retained in part, with the
fraction that makes the coverage exact. That is the expectation of breaking
ties at random. It keeps a signal from gaining information from the order
its ties happen to be in. A constant signal therefore performs exactly as
random rejection does.

- **Pooled.** The panel curve selects over every household's windows together,
  as one threshold on the signal would, and aggregates the retained timestamps.
  Each household's share of the retained windows then follows from the
  signal, and is reported.
- **Per household.** Each household's curve retains the fraction `c` of its own
  windows.

## What is reported

For each coverage level on the grid (5% to 100% in steps of 5% by default):

| Quantity | Definition |
| --- | --- |
| coverage | the share of windows retained |
| selective risk | the mean loss of the retained predictions; with the default 0-1 loss, the error rate |
| selective error | the error rate of the retained predictions |
| selective balanced accuracy | the mean recall over the states among the retained windows |
| states scored | how many states still have retained windows |
| per-state retained coverage | each true state's share of its windows retained |
| calibration among retained predictions | the expected calibration error (10 equal-width bins, as elsewhere), the mean stated confidence, and the confidence minus the accuracy |
| rejected-state composition | the true states of the rejected windows, as shares |
| rejected error | the error rate among the rejected windows |
| household coverage | under the pooled selection, the smallest, median and largest household coverage |
| excess risk | the selective risk minus the full-coverage risk: the change from random rejection |

- **Balanced accuracy when a state is rejected.** When a state's windows are
  all rejected, balanced accuracy averages over the remaining states. Read it
  with the states scored and the per-state coverage, so that a signal cannot
  look good by rejecting a hard state entirely without that being seen.
- **Loss matrices.** A loss matrix may replace 0-1 loss. The risk is then its
  mean over the retained predictions, and the error rate is reported beside it.

### References

- **Random rejection.** Retaining a uniformly random fraction `c` keeps, in
  expectation, the fraction `c` of every count. The selective risk is then the
  full-coverage risk, exactly. The other ratio metrics are their full-coverage
  values to first order, every state keeps `c` of its windows, and the
  rejected windows have the panel's composition. The random reference is these
  values at every coverage level.
- **The oracle.** It rejects the windows with the greatest loss first. It is
  the best curve any signal could reach on the same predictions.

### Curve summaries

| Summary | Definition | Random | Oracle |
| --- | --- | --- | --- |
| AURC | the area under the risk-coverage curve, by the trapezoid rule over the grid, divided by the grid's span: the mean selective risk over the levels | the full-coverage risk | its own |
| excess AURC | the AURC minus the oracle's | — | 0 |
| gain | `(AURC_random − AURC) / (AURC_random − AURC_oracle)` | 0 | 1 |

A negative gain is a signal worse than random. The gain is undefined, and
`null`, when the predictions have no loss to remove.

The AURC is justified as a summary of how well a signal orders predictions,
over every operating point at once. It is not a deployed risk: it averages
over coverage levels no deployment would use together. Comparisons at a stated
coverage level stay on the curves.

## Uncertainty across households

Timestamps within a household share a resident, a routine, a layout and an
annotator, and consecutive states persist, so they are never resampled as if
independent ([evaluation design](EVALUATION_DESIGN.md)). Households are.

- **The pooled curve.** Each bootstrap resample draws households with
  replacement, selects again at every coverage level over the resampled panel,
  and recomputes every metric. The percentile intervals of the risk, error,
  balanced accuracy, excess risk, AURC, excess AURC and gain come from those
  resamples. Every signal in one evaluation uses the same resamples, so their
  intervals are comparable.
- **Household curves.** Each household's curve is one value per level, and
  the curves are described with each household counted once: the mean,
  median, range and a bootstrap interval of the mean over households. Each
  household's AURC, excess AURC and gain are described the same way.
- **One household.** With one household, no between-household variability can
  be estimated, and no interval is reported.

The pooled estimate weights households by their windows, and the household
summaries weight them equally. When they differ, a few large households are
driving the pooled curve.

## In the experiment record

Schema 1.6 adds an optional `selective_prediction` section to every experiment
record ([experiment artifacts](EXPERIMENT_ARTIFACTS.md)). It holds:

- the coverage grid, the states, the loss matrix and the calibration bins;
- the bootstrap settings, with households as the resampled unit;
- each household's windows by true state;
- the random reference and the oracle's curve;
- per signal:
  - its direction and missing policy;
  - the pooled curve with its intervals;
  - the curve summaries;
  - each household's curve;
  - the across-household descriptions.

`validate_record` checks the section:

- a valid coverage grid ending at full coverage;
- stated directions and missing policies;
- shares within `[0, 1]`;
- intervals whose lower end does not exceed their upper end;
- every household present for every signal;
- households as the resampled unit.

Records written before 1.6 are migrated with the section `null`.

## Using it

```python
from sensor_modeling.evaluation import (
    HIGHER_IS_RISKIER, HIGHER_IS_SAFER, SelectiveData, Signal, evaluate_signals,
)

data = SelectiveData.from_households(states, truth, predicted, confidence)
report = evaluate_signals(
    data,
    [
        Signal("confidence", data.align(confidence), HIGHER_IS_SAFER),
        Signal("discordance", data.align(discordance), HIGHER_IS_RISKIER),
        Signal("surprise", data.align(surprise), HIGHER_IS_RISKIER,
               missing="reject_first"),
    ],
    resamples=2000,
    seed=0,
)
record = ExperimentRecord(..., selective_prediction=report)
```

`evaluate_signal(data, signal)` returns one signal's curves as arrays, for
analysis.

## What this does not do

- **No thresholds.** It selects no threshold and no abstention rule.
- **No evaluation run.** Which signal orders predictions best is the
  pre-specified Phase 4 experiment's question.
- **No combined signals.** Each signal is evaluated as given. A combination is
  evaluated once it is itself a signal.
