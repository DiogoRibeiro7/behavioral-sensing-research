# Phase 1: matched baselines on the development panel

This is the first run of the [roadmap](roadmap.md)'s Phase 1 comparison. It
runs the four nested [information sets](INFORMATION_SETS.md) and the
pre-declared [baselines](BASELINES.md) on real homes. Every comparison is
matched on information and resampled by household.

**Status: exploratory.** The development homes have been inspected throughout
earlier work, so nothing here is confirmatory evidence.

## Protocol

The protocol was fixed in `scripts/run_phase1_matched_baselines.py` and
committed (`1b4d705`) before any household was scored.

- **Homes:** the 20 single-resident homes of the frozen v0.3 development panel,
  each verified against its recorded SHA-256. The two two-resident homes stay
  excluded, as in the frozen fit.
- **Grid:** America/Los_Angeles time and a 5-minute step, as in every
  development result.
- **Folds:** two cross-fitted folds that alternate the sorted home
  identifiers, each held out once. Every home is scored exactly once, by
  models never fitted on it, so every figure below covers 20 held-out homes.
- **Models:** the four nested information sets at their default resolution,
  with `baseline_suite` for each and seed 0.
- **Statistics:** household summaries, and paired household differences with
  10,000 household resamples and 95% percentile intervals.

## Balanced accuracy

The median over 20 held-out homes, with the mean in parentheses:

| Information set | `state_frequency` | `persistence` | `logistic` | `tree` |
| --- | ---: | ---: | ---: | ---: |
| current | 0.167 (0.161) | | 0.354 (0.344) | 0.438 (0.424) |
| current + time of day | 0.167 (0.161) | | 0.488 (0.477) | 0.518 (0.517) |
| current + history | 0.167 (0.161) | 0.268 (0.269) | 0.392 (0.383) | 0.452 (0.445) |
| current + time of day + history | 0.167 (0.161) | 0.268 (0.269) | 0.500 (0.498) | 0.511 (0.504) |

`persistence` needs history, so it is absent from the first two rows.

## What each kind of information is worth

Each cell is the mean paired household difference in balanced accuracy when
the information is added, with its 95% interval and the number of the 20 homes
that improved. The model is held fixed and only its information changes.

| Added | To | `logistic` | `tree` |
| --- | --- | --- | --- |
| time of day | current | +0.133 [+0.112, +0.152], 20/20 | +0.093 [+0.078, +0.109], 20/20 |
| time of day | current + history | +0.115 [+0.097, +0.132], 20/20 | +0.059 [+0.045, +0.074], 20/20 |
| history | current | +0.039 [+0.029, +0.047], 19/20 | +0.022 [+0.007, +0.036], 15/20 |
| history | current + time of day | +0.021 [+0.013, +0.028], 18/20 | −0.012 [−0.022, −0.003], 8/20 |

Two rows serve as checks on the machinery. `state_frequency` gains exactly zero
from any added information, and `persistence` gains exactly zero from time of
day. Neither reads those columns, and the comparisons confirm it household by
household.

## Probability quality

The median household log loss, where lower is better:

| Information set | `state_frequency` | `persistence` | `logistic` | `tree` |
| --- | ---: | ---: | ---: | ---: |
| current | 1.510 | | 1.242 | 1.140 |
| current + time of day | 1.510 | | 0.925 | 0.911 |
| current + history | 1.510 | 1.411 | 1.176 | 1.061 |
| current + time of day + history | 1.510 | 1.411 | 0.909 | 0.905 |

The medians and the means disagree for `logistic`. With history it has a
median log loss of 1.176 but a mean of 1.595, above the no-information
reference of 1.500. A few households receive confidently wrong probabilities,
and they dominate the mean. Its mean paired change in log loss when history is
added is −0.164 [−0.618, +0.069], even though 18 of the 20 homes improve.
Report household medians and counts alongside means for this model. The Brier
score shows the same ordering without the instability.

## What this does and does not show

- **Time of day is the largest single increment** for both baselines, in every
  home. That agrees with the earlier diagnostic ablation, which attributed
  about +0.105 to it ([Real-data validation](real_data.md)).
- **Recent history adds little for these baselines:** +0.02 to +0.04 on
  current evidence alone. Once time of day is present it adds +0.021 for
  `logistic` and slightly lowers the tree, by −0.012. The earlier ablation attributed about +0.140 to history, using a
  gradient-boosted classifier whose code was not retained. The model, the
  feature resolution (rooms, not room-and-modality channels) and the split all
  differ, so this run does not explain the discrepancy. It does show that the
  +0.140 is not reproduced by pre-declared linear or depth-limited models.
- **The previous window says less than the current one.** `persistence`
  applies a logistic model to the previous window and scores 0.268. The same
  model family on the current window alone, `logistic` in the `current` set,
  scores 0.354. States change within minutes, so the current window carries
  information the previous one lacks. The two figures come from different
  information sets and are descriptive, not a paired comparison.
- **These are baselines, not a ceiling.** The best medians here, about 0.50 to
  0.52, are below the 0.607 the earlier diagnostic reported. That figure came
  from a more flexible classifier on a different split, so the two are not a
  matched comparison.
- **The filter is not in this comparison.** The generative filter is recursive
  and cannot be restricted to a declared information set, so the formulation
  part of the Phase 1 decomposition is still to be measured. Its published
  development median of 0.420 is not matched to any row above. The
  [recoverable-information gap](PHASE1_RECOVERABLE_GAP.md) run measures that
  formulation part. It restricts the filter's generative model to the sets it
  can consume.

## Reproducing it

```text
python scripts/run_phase1_matched_baselines.py <archive_root> <output_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The run takes about 13 minutes on one
machine. It writes one record per fold and information set, and a combined
`phase1_matched_baselines.json`.

The figures above come from commit `fe251d4` on a clean working tree, run on
Windows with Python 3.13.5, NumPy 2.3.5, SciPy 1.15.3 and scikit-learn 1.6.1.
Two runs from the same commit gave identical results. The fitted models depend
on the scikit-learn version, so the record stores it.
