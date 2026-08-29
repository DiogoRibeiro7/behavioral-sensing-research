# Simulation protocols

The experiments shipped with this repository run four paired seeds, or in one
case a single seed. Those settings exist so the command-line examples and the
continuous-integration smoke tests finish quickly. They are **not** the settings
a reported result should use.

This page separates the two, and specifies how many replications a claim needs.

## Why the shipped defaults are not a study

`sensor-modeling ablate` defaults to seeds `11 22 33 44`. `paired_difference()`
then produces a bootstrap 95% interval from four paired differences.

A bootstrap over four numbers resamples those four numbers. It gives an
interval, and the interval looks precise, but its width reflects the arithmetic
of that particular sample rather than the uncertainty in the underlying effect.
With `n = 4` the interval is also badly behaved: the bootstrap cannot see tail
behaviour it never sampled, and a single unusual household moves the whole
interval.

Attribution is more exploratory still. It runs one seed. A statement such as
"largest gain +0.032 during a carer round" describes one generated household.
It is a demonstration that the mechanism behaves as designed, and it supports
no estimate of expected benefit.

| Purpose | Replications | Appropriate claim |
| --- | --- | --- |
| CI smoke test | 3–4 seeds | "the command runs and the pipeline is wired up" |
| Pilot | 4–20 seeds | "the direction is consistent; magnitude unresolved" |
| Simulation study | see below | "the effect is estimated at D, with MCSE e" |

## Choosing the replication count

Fix the replication count from the Monte Carlo standard error you are willing
to report, not from what finishes quickly. For a paired difference over `n`
independent trajectories:

```text
MCSE = s_D / sqrt(n)             the precision achieved
n    >= (s_D / epsilon) ** 2     the n needed for a target precision
```

`s_D` is the standard deviation of the paired differences and `epsilon` is the
largest MCSE you would accept.

Pairing is what makes this affordable: both arms see the same household, so
`s_D` reflects the effect's variability rather than between-household variance,
and is far smaller than the spread of either arm alone.

### Applying it to the ablation

The four-seed pilot reported a mean difference of 0.012 balanced accuracy with
a nominal 95% interval of [+0.004, +0.020]. Inverting the half-width
`h = 0.008` through `h = t * s_D / sqrt(n)`, with `t(0.975, 3) = 3.182`, gives
`s_D` of roughly 0.005.

That estimate is itself based on four points, so treat it as an order of
magnitude and round up rather than down:

| Target MCSE | Required `n` | Interpretation |
| --- | --- | --- |
| 0.002 | 7 | too coarse: comparable to the effect itself |
| 0.001 | 25 | effect distinguishable from zero, magnitude loose |
| 0.0005 | 100 | magnitude resolved to about a tenth of the effect |

**A study should therefore use at least 100 paired trajectories**, not four. The
simulator is cheap and the arms are independent, so this is a matter of
scheduling rather than feasibility: the run below takes roughly 45 minutes on
one machine.

```bash
python -c "import numpy as np;   s = np.random.SeedSequence(20260829).generate_state(100, dtype=np.uint32) % 1000000;   print(' '.join(map(str, sorted(set(int(x) for x in s)))))" > seeds.txt

sensor-modeling ablate --days 14 --step-minutes 10   --seeds $(cat seeds.txt) --output results/ablation_study_n100.json
```

Seeds are derived from a recorded root rather than typed, so the set is
reproducible from the root alone. The artefact records the root's seeds, the
git commit and the resolved defaults.

This run has been done. It found the four-seed pilot had overstated the
headline effect by roughly 60%, with the study estimate falling outside the
pilot's interval; see [Release readiness](RELEASE_READINESS.md). Four seeds
were not simply imprecise, they were wrong, which is the practical case for
this page.

Re-estimate `s_D` from the first 20–30 trajectories of the real run and adjust
`n` upwards if it exceeds the pilot value. Report the achieved MCSE alongside
the effect; an interval without one is not interpretable.

`PairedDifference.mcse` carries it, so every paired comparison in this package
reports the precision its replication count bought.

### Applying it to attribution

Attribution needs a scenario factor as well as replications, because its whole
purpose is to behave differently when someone else is present:

```text
scenario x paired seeds
```

`sensor-modeling attribution --seeds ...` runs this form; a single `--seed`
still runs the one-household demonstration, and the replicated study refuses
fewer than two seeds so a demonstration cannot be mistaken for an estimate.

Use at least 100 paired seeds **per scenario**, and report intervals for each
of:

- balanced-accuracy gain;
- calibration gain;
- visitor precision, recall and F1;
- per-state consequences, since a gain concentrated in one state is a different
  finding from a uniform one.

Visitor recall is about 0.48 in the current synthetic evaluation. Any claim
about attribution's value has to be read against that: the component whose
benefit is being measured is itself only half-detecting the thing it keys on.

### Applying it to change detection

Detection delay is heavily skewed, and undetected changes are censored rather
than missing. Report:

- recall, with a binomial interval;
- the **pooled** median delay over detections, the number of detections behind
  it, and a bootstrap interval over trajectories;
- false alerts per person-day, from the stable arms;
- the delay distribution, not only its centre.

Do not average per-seed medians. A mean of medians is not a median: it weights a
seed that detected one change as heavily as a seed that detected twenty. The
implementation now pools, and reports `mean_seed_median_delay_days` separately
where the per-seed view is wanted.

Because delay is skewed, 100 trajectories is a floor rather than a target.

## Seed management

- Seeds must be independent and disjoint across arms of a factorial design.
  Reusing a seed across scenarios reintroduces the correlation pairing removes.
- **Space them.** The attribution scenarios derive degradation seeds from
  `seed + 1` to `seed + 3`, so consecutive study seeds would give two
  supposedly independent replications identical record loss and identical
  stuck sensors. `11 12 13` is the obvious thing to type and is wrong; the
  replicated study refuses seeds closer than four apart rather than leaving it
  to the caller to know. Seeds drawn from a `SeedSequence` are spread widely
  enough that this never binds.
- Record every seed in the artefact. `ExperimentRecord` does this.
- Derive study seeds from a single recorded root rather than typing them, so the
  set is reproducible and auditable.
- Pair by seed: within a comparison, both arms must see the identical
  trajectory. `series()` refuses unpaired input, which is a guard worth keeping.

## What a reported result must carry

Every artefact already records configuration, resolved algorithm defaults,
seeds, metric definitions, the environment, the package version, the git commit
and whether the tree was dirty. A result intended for publication additionally
needs:

- the achieved MCSE for each reported effect;
- the replication count, stated in the text and not only in the artefact;
- the pilot that motivated the replication count;
- an explicit statement that the data are simulated.

## The limitation none of this addresses

Increasing `n` reduces Monte Carlo error. It does nothing about the fact that
every trajectory comes from one simulator written by this project. A tighter
interval around a simulator-derived effect is a more precise statement about the
simulator.

Real-data validation is the next milestone, not more seeds. See
[Limitations](limitations.md).
