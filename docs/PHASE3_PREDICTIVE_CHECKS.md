# Phase 3.3 follow-up diagnostic: posterior predictive checks of the hurdle model

The [fitted hurdle channel models](PHASE3_FITTED_RATES.md) improved calibration
substantially. Two problems remained:

- An active window's count is over-dispersed, with variance 5 to 12 times its
  mean in the common states.
- `home_active` recall fell by 0.23.

Before another distribution is introduced, this diagnostic locates where the
fitted observation model is misspecified. It compares observed channel counts
with the model's predictive distribution for each household, state and channel
with enough data.

**Status: pre-specified diagnostic, development panel.** Every setting and the
rule that reads the result were frozen before any household was examined.

- **Diagnostic only.** No model is changed, no inference is run, and no state
  is estimated.
- **Data.** Only the 20 development homes are used, which earlier work has
  inspected. The frozen external cohort is not touched.

## Two references

A **cell** is one household's labelled windows of one state on one
instrumented channel, in time order. Each cell with at least 50 windows is
checked against two hurdle models:

| Reference | Fitted on | What its misses mean |
| --- | --- | --- |
| `own` | the cell alone: `π` its share of silent windows, `μ` solved from its active windows' mean | the hurdle family itself is wrong, whatever the household |
| `population` | the fold's training households, with 12 pseudo-windows, exactly as inference uses it | the family's misses and the differences between households |

The `own` fit reproduces a cell's silence, mean and active mean exactly, so
those three are not judged against it.

## The statistics

Each statistic's observed value is compared with its predictive distribution,
from 200 replicates of the cell's own windows drawn from the reference:

| Statistic | Observed against predicted |
| --- | --- |
| `silence` | share of silent windows, as a log odds ratio |
| `mean`, `variance` | of the count over every window |
| `active_mean` | mean count of an active window |
| `active_dispersion` | variance of an active window's count over the zero-truncated Poisson's; above one means the model is under-dispersed |
| `q90`, `q99` | upper quantiles of an active window's count |
| `tail_excess` | active windows above the model's 99th percentile |
| `quiet_runs` | windows in runs of at least 12 silent windows, one hour |
| `bursts` | windows in runs of at least 3 active windows, 15 minutes |

- **Runs.** Runs are counted within stretches of consecutive labelled windows
  of the state. The model draws each window independently given the state. An
  excess of either run is therefore dependence in time, which no count
  distribution can produce.
- **Count distribution.** The observed windows in each count bin are recorded
  against each reference's expected windows. The bins are 0, 1, 2, 3, 4–5,
  6–7, 8–10, 11–15, 16–22, 23–31, 32–44, 45–63 and 64 or more.
- **Replicates.** `own` replicates are refitted before their statistics are
  computed: a parametric bootstrap. With a correctly specified model, the
  observed value then falls inside the replicates' central band as often as the
  band's coverage.
- **Sufficiency.**
  - Active-window statistics need at least 30 active windows.
  - A run statistic needs its observed or predicted windows in runs to reach
    five runs' worth.

### Verdicts

- **For a cell.** The comparison uses the log ratio of observed over predicted
  and a 95% band of the replicates' log ratios:
  - **consistent:** inside the band;
  - **above** or **below:** outside it by at least a ratio of 1.25 in that
    direction;
  - **small:** outside it by less.
- **Across households.** Households are the unit, and windows are never pooled
  across households.
  - Each household's value for a group, such as a state, a channel, a room or a
    channel type, is the mean of its cells' log ratios.
  - The household mean has a 95% household bootstrap interval from 10,000
    resamples.
  - The verdict is **positive** if the mean is at least log 1.25 and the lower
    bound is above 0, and **negative** in the mirror case.
  - It is **negligible** if the whole interval lies within ±log 1.25, and
    **uncertain** otherwise.
  - It is **insufficient** with fewer than five households.
  - Every household's values stay in the record and in the summary.

### Conclusions and routing

Three questions are answered with the `own` reference, across the four common
states: `away`, `home_active`, `home_inactive` and `sleeping`.

- Is the zero-truncated Poisson active count under-dispersed?
- Are long quiet runs in excess?
- Are bursts in excess?

Each answer is one of:

- **systematic:** positive in at least three of the four common states and
  negative in none;
- **partial:** positive in one or two;
- **absent:** negligible or negative in all four;
- **inconclusive:** anything else.

The routing to a next model family was declared before any household was
examined:

1. **Temporal dependence.** A systematic excess of quiet runs or bursts points
   to within-state temporal dependence: activity sub-states, or a
   Markov-modulated emission within each state. No marginal count family
   produces runs beyond independence, and a latent intensity that varies over
   time also over-disperses the counts.
2. **Dispersion.** Otherwise, systematic active-count under-dispersion points to
   an over-dispersed zero-truncated count model for the active part, such as
   the zero-truncated negative binomial. It would keep the hurdle's silence
   parameter.
3. **Nothing indicated.** Otherwise, no new family is indicated.

The roadmap names a next model family only if the routing indicates one.

### Also reported

- Every statistic by state, channel, room and channel type, against both
  references.
- The median across households for each state and channel.
- Each household's slope of log active variance on log active mean.
- How many cells fall in each verdict.
- Figures drawn from the record alone.

## The synthetic checks

The tests run the checks on processes whose answer is known:

- **Poisson.** A correctly specified Poisson is accepted. Across 40 seeds, no
  cell is flagged for variance, dispersion or the upper quantiles, and
  household values read as negligible.
- **Negative binomial.** A gamma-mixed Poisson with size 0.5 is detected in
  every seed, in active dispersion, the 99th percentile and the tail. Its
  household values read as positive, and the pattern as systematic.
- **Clustering in time.** Quiet and busy spells that persist are detected in
  both quiet runs and bursts. Independent windows are not flagged.
- **Population reference.** A household quieter or busier than the population
  is flagged by the population reference.

## Results

Not yet run. The protocol above was committed before any household was
examined.

## Reproducing it

```text
python scripts/run_phase3_predictive_checks.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The script writes the record, a Markdown
summary generated from it, and the figures.
