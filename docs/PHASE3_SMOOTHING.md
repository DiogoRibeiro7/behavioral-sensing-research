# Phase 3.5: fixed-lag smoothing against online filtering

A fixed-lag smoother revises each window's estimate with up to `lag` later
windows. It reads evidence a live system does not yet have, and its estimate
for a window exists only `lag` windows after it
([inference regimes](INFERENCE_REGIMES.md)). This pre-specified experiment
measures how much performance smoothing recovers over the online filter, on
households none of the parameters were fitted on.

**Every difference it reports is a smoothing gain.** A smoothing gain is
available only after the smoother's delay. It is never an improvement of the
online filter, and it cannot be had by a live system at the moment it is
about.

**Status: pre-specified, development panel.** Every setting, comparison and
criterion was frozen before any household was scored. The development homes
have been inspected in earlier work, so this is not a final held-out claim. The
frozen external cohort is not touched.

An earlier, unregistered measurement on the production pipeline found most of
the smoothing gain at one window of lag ([real data](real_data.md)). That run
used an 11/11 split that was never recorded, and the declared rates. This one
uses the frozen folds and the current best recursion.

## The formulation

Smoothing needs a recursion over consecutive windows. The one on `develop` with
a pre-specified success in the recursion is the
[Phase 3.3 follow-up](PHASE3_FITTED_RATES.md)'s `filter_hurdle`:

- population hurdle channel models, fitted per fold on the training households
  with 12 pseudo-windows;
- the default ontology's transition over one 5-minute step;
- the stationary distribution one step before the first window;
- every window of the recording, with full reliability and attribution.

The other candidates are not used:

- Partial pooling in the recursion was inconclusive ([Phase 3.4](PHASE3_PARTIAL_POOLING.md), P2).
- The Phase 3.1 time prior was evaluated only on declared information sets,
  never as a recursion.

The online filter and every smoother share one filter pass. Each smoother only
revises it with the declared lag, through `regime_beliefs`. No model is changed
or refitted.

## Protocol

The protocol is `artifacts/phase3/smoothing_protocol.json`. The script refuses
to run if the code's protocol differs from it by a single value.

- **Households.** The 20 development homes and the two frozen Phase 1 folds.
  Each home is held out once, and scored by channel models fitted without it.
- **Regimes.** A small set, declared and never searched:

  | Regime | Lag | Reporting delay | Operational use |
  | --- | ---: | ---: | --- |
  | `online` | 0 | 0 min | the reference: what a live system reports at each window |
  | `lag_1` | 1 window | 5 min | a display that may run one window behind |
  | `lag_6` | 6 windows | 30 min | a non-urgent check within the hour, such as whether the resident has got up |
  | `lag_12` | 12 windows | 60 min | retrospective hourly or daily reporting |

  One hour is the longest delay treated as plausible for a decision made the
  same day. Longer lags serve only offline analysis and are not scored.
- **Scored windows.** Each held-out household's labelled windows, except the
  last 12 windows of its recording. Every regime is scored on exactly those
  windows, so every smoothed estimate reads its full lag.
- **Reproduction.** The online filter is also scored on every labelled window,
  as `filter_hurdle@R`. That reproduces the Phase 3.3 follow-up's cell, and a
  test checks it household by household.
- **Metrics.** Household balanced accuracy, per-state recall, log loss, Brier
  score, and expected calibration error over 10 bins.
- **Operational measures.** Each is a share of a household's scored windows,
  against the online filter's most probable state on the same windows:
  - **changed:** the smoother reports a different state;
  - **corrected:** among the windows the filter gets wrong, those the smoother
    gets right;
  - **made wrong:** among the windows the filter gets right, those the smoother
    gets wrong;
  - **corrections among changes:** among changed windows, those that are
    corrections.

  They are also reported within the windows of each focus state: `away`,
  `home_active` and `sleeping`. Each regime's reporting delay is its lag.
- **Transitions.**
  - A transition is a scored window whose labelled state differs from that of
    the window before it, both labelled.
  - **Accuracy near transitions** is measured within 6 windows of a transition,
    before or after it, and compared with accuracy elsewhere.
  - **Decision delay** is how long after a transition a regime can first report
    the new state. It counts the windows until the regime's most probable state
    first equals the new state within the new episode, plus the regime's lag.
    Each household contributes the median over its detected transitions.
  - **Detection rate** is the share of transitions whose new state the regime
    reports at some window of the new episode.
  - Each is reported for all transitions, and for the transitions into each
    focus state.
- **Bootstrap.** 10,000 household resamples, with 95% percentile intervals for
  the mean and the median paired difference. The seed is 0.

### Estimands

Each compares one smoother with the online filter on the same windows, paired
by household. A positive value favours the smoother. Every one is a smoothing
gain, and the record labels it with the smoother's delay.

| Key | Role | Smoother | Against |
| --- | --- | --- | --- |
| G1 | primary | `lag_1`, 5 min | `online` |
| G6 | primary | `lag_6`, 30 min | `online` |
| G12 | primary | `lag_12`, 60 min | `online` |

Per-state recall, the operational measures and the transition measures are
reported for each.

### Criteria

Verdicts compare effect sizes and household intervals with the Phase 3
minimal important differences. They are not significance tests.

| Quantity | `δ` |
| --- | ---: |
| balanced accuracy | 0.02 |
| log loss | 0.05 |
| Brier score | 0.01 |
| calibration error | 0.02 |
| per-state recall | 0.05 |
| accuracy near transitions | 0.02 |
| detection rate | 0.05 |
| decision delay | 5 min |

Each estimand is judged by a rule fixed in advance:

- **Gain.** Balanced accuracy favours the smoother, and neither log loss nor
  calibration error favours the online filter.
- **Trade-off.** Balanced accuracy favours the smoother, and log loss or
  calibration error favours the online filter.
- **Probability gain.** Balanced accuracy does not favour the smoother, log loss
  does, and neither balanced accuracy nor calibration error favours the online
  filter.
- **No gain.** Balanced accuracy is negligible or favours the online filter, and
  log loss is negligible or favours the online filter.
- **Inconclusive.** Anything else.

The trade-off verdict exists because the earlier, unregistered measurement
found that smoothing raised accuracy and worsened calibration.

### The record

The record compares regimes, so its `inference` field states the longest lag,
60 minutes, which bounds every estimate in it. Its results label every cell and
comparison with its own regime, and list each regime's prediction and
latest-evidence timestamps ([experiment artifacts](EXPERIMENT_ARTIFACTS.md#the-inference-regime)).

## Results

Not yet run. The protocol above was committed before any household was scored.

## Reproducing it

```text
python scripts/run_phase3_smoothing.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes the record and a Markdown
summary generated from the written record.
