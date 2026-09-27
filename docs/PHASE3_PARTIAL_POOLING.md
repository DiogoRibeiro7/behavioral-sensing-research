# Phase 3.4: partial pooling of household parameters

The [fitted hurdle channel models](PHASE3_FITTED_RATES.md) give every household
its fold's population parameters. Households differ a lot in how often each
channel is silent. The [partial-pooling framework](PARTIAL_POOLING.md) lets a
household move toward its own values as it accumulates labelled data:

```text
θ̂ = θ_pop + w · (θ_raw − θ_pop),     w = n / (n + κ)
```

This pre-specified experiment measures two things on households that none of
the parameters were fitted on:

- whether pooling improves on the population;
- whether fitting each household on its own overfits small homes.

**Status: pre-specified, development panel.** Every setting, comparison and
criterion was frozen before any household was scored. The development homes
have been inspected in earlier work, so this is not a final held-out claim. The
frozen external cohort is not touched.

## Protocol

The protocol is `artifacts/phase3/pooling_protocol.json`. The script refuses to
run if the code's protocol differs from it by a single value.

- **Households.** The 20 development homes and the two frozen Phase 1 folds.
  Each home is held out once. Its fold's training homes fit the population and
  select the pooling strength.
- **Models.** They share the prior, transition and channels, and differ only in
  the hurdle channel parameters:

  | Model | Channel parameters |
  | --- | --- |
  | `declared` | the declared rates, for context |
  | `population` | fitted per fold on the training homes, as in the Phase 3.3 follow-up |
  | `pooled` | the household pooled toward the population, `κ = 288` windows (24 labelled hours) |
  | `pooled_selected` | the same, with `κ` selected on the training homes |
  | `unconstrained` | the household's own estimates, with `κ = 0.5` |

  - **Pooled parameters.** Pooling is applied to each channel's silence
    probability and active-window mean in every state.
  - **Unconstrained.** Half a window keeps a probability off 0 and 1. A cell
    without household data takes the population value.
- **Settings.** Each model is scored in `I0`, the current windows, and in `R`,
  the filter's recursion over every window. Both are online.
- **Adaptation arms.**
  - Each held-out household adapts on its labelled windows in the first days
    of its recording, and is scored on its labelled windows after them.
  - Every model in an arm is scored on exactly those windows.
  - `week` (7 days) is the Phase 3.1 window.
  - `day` (1 day) leaves most cells with little data, which is where
    overfitting would show.
  - Nothing is compared across arms.
- **Strength selection.**
  - **Procedure.** For each fold, each training home is left out in turn. The
    population is fitted on the other training homes, and the left-out home is
    pooled with its week-arm window. Its labelled windows after the window are
    scored in `I0`.
  - **Choice.** The strength with the smallest mean log loss over the grid
    24, 72, 288, 1152 and 4608 windows is selected, with ties going to the
    larger strength.
  - **No leakage.** Held-out homes are never used.
- **Eligibility.**
  - A home enters an arm with at least 12 labelled windows in its adaptation
    window, and at least 288 after it.
  - Otherwise it is reported and left out of every model in that arm.
  - A run with fewer than two eligible homes in an arm is refused.
- **Stratification.** Within each arm, the eligible homes are sorted by
  labelled adaptation windows and split at the median into `less data` and
  `more data`. Every estimand is also reported within each half.
- **Metrics.** Household balanced accuracy, log loss, Brier score, expected
  calibration error over 10 bins, and per-state recall.
- **Bootstrap.** 10,000 household resamples, with 95% percentile intervals for
  the mean and the median paired difference. The seed is 0.

### Estimands

Each is a paired household difference within one setting and arm. A positive
value favours the first model.

| Key | Role | First | Second | Setting, arm |
| --- | --- | --- | --- | --- |
| P1 | primary | `pooled` | `population` | `I0`, week |
| P2 | primary | `pooled` | `population` | `R`, week |
| O1 | primary | `pooled` | `unconstrained` | `I0`, day |
| O2 | secondary | `pooled` | `unconstrained` | `I0`, week |
| O3 | secondary | `pooled` | `unconstrained` | `R`, day |
| S1 | secondary | `pooled_selected` | `population` | `I0`, week |
| S2 | secondary | `pooled_selected` | `population` | `R`, week |
| S3 | secondary | `pooled` | `population` | `I0`, day |
| S4 | secondary | `unconstrained` | `population` | `I0`, week |
| S5 | secondary | `unconstrained` | `population` | `I0`, day |
| C1 | context | `population` | `declared` | `I0`, week |
| C2 | context | `population` | `declared` | `R`, week |

Per-state recall is reported for P1 and O1.

For each household, the record keeps the pooled parameters of both arms. For
every channel, state and parameter, that is:

- the household's sample size;
- its raw estimate;
- the population estimate;
- the pooled estimate;
- the effective shrinkage.

It also keeps the household's performance change in every estimand.

### Criteria

Verdicts compare effect sizes and household intervals with the Phase 3
minimal important differences. They are not significance tests.

| Metric | `δ` |
| --- | ---: |
| balanced accuracy | 0.02 |
| log loss | 0.05 |
| Brier score | 0.01 |
| calibration error | 0.02 |
| per-state recall | 0.05 |

Log loss is the primary metric. Adaptation changes each household's
probability model, so a proper scoring rule is the target. Balanced accuracy
and calibration error are guards.

- **Pooling rule, for P1, P2 and the other pooling estimands:**
  - **Success.** Log loss favours the adapted model, and neither balanced
    accuracy nor calibration error favours the reference.
  - **Trade-off.** Log loss favours the adapted model, and balanced accuracy
    favours the reference.
  - **Failure.** Log loss is negligible, or favours the reference.
  - **Inconclusive.** Anything else.
- **Overfitting rule, for O1 to O3:**
  - **Overfits.** Log loss favours the pooled model over the unconstrained one.
  - **Does not overfit.** Log loss is negligible, or favours the unconstrained
    model.
  - **Inconclusive.** Anything else.

## Results

The results will be published here from the record
`artifacts/phase3/phase3-partial-pooling.json`.

## Reproducing it

```text
python scripts/run_phase3_pooling.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes the record and a Markdown
summary generated from the written record.
