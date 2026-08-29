# Evaluation Methodology

This page documents how the ambient pipeline is evaluated, why the design
avoids circular validation, and what the sensor-ablation experiments measure.

## Avoiding circular validation

The simulator exists so that inference can be scored against something known.
That only works if the generative process differs from the inference model.

`simulation` therefore generates a **schedule**: a person
who wakes at roughly the same time each morning, makes breakfast, goes out,
comes back and goes to bed, with stochastic timings and durations. Inference
recovers that schedule through a continuous-time Markov model that knows
nothing about schedules.

Had the two shared a generator, good results would prove only that the code
can invert its own assumptions.

Ground truth records resident state and room minute by minute, sleep and
night-time bathroom trips, visitor arrivals and their own movements, and which
activation came from whom. It is available only to the evaluation code and is
never passed to the pipeline.

### Realistic imperfection

The clean record already contains background false activations, a radar that
occasionally loses a still person or splits one into two, and visitors
tripping the same ambient sensors as the resident.

`faults` then degrades it separately, so a
robustness study can hold behaviour fixed and vary only what went wrong with
the apparatus. That separation is what makes paired comparisons possible: the
same simulated fortnight, once with a working wearable and once without,
differs in exactly one thing.

Available degradations: dropout, stuck sensors, random loss, wearable
non-adherence, late arrival, duplication, and per-source clock drift.
`degrade` returns the withheld records, so
an evaluation can report how much evidence was actually missing rather than
infer it.

## Metrics

Accuracy alone is close to meaningless here. The states are heavily
imbalanced -- a resident is asleep or quietly at home for most of the day --
so a model that predicts `home_inactive` forever scores about 0.9 and knows
nothing. Scoring only the argmax also discards the part of every output that
matters: whether a stated confidence means anything.

### State inference

`state_metrics` reports balanced accuracy,
macro F1, log loss, multiclass Brier score, expected calibration error, and
per-class recall.

The **abstention convention** is stated once and applied consistently: a
reported `UNKNOWN` is never correct, because `UNKNOWN` is never a true
label, so it costs recall. `selective_accuracy` and `abstention_rate` are
reported alongside, so a model that declines usefully is distinguishable from
one that is simply wrong.

`transition_timing` matches each true
transition to the nearest inferred one within a tolerance, consuming each
inferred transition at most once so a flickering model cannot claim credit
twice.

### Attribution

`binary_metrics` reports precision, recall,
F1 and calibration for probabilistic binary judgements such as visitor
presence. Calibration is scored on the stated probability of the *predicted*
class, so a confident "no visitor" is judged as strictly as a confident
"visitor".

### Change detection

`detection_metrics` reports detection delay,
recall, precision, and **false positives per person-day** -- the number that
decides whether a system is usable in practice.

A detection counts only if it falls at or after the true change and within
`max_delay_days`. An alert raised *before* the change is a false positive,
not early detection: a system that alarms before anything happened has alarmed
at noise.

Detection should be scored on the alerts actually delivered rather than on raw
baseline verdicts. Deduplication and rate limiting sit between the two, and it
is the delivered burden a carer experiences.

### Comparison

`paired_difference` reports the mean paired
difference, a bootstrap confidence interval, a standardised effect size
(Cohen's *dz*), and a win/loss count.

It deliberately does not report a p-value. With simulations, any effect can be
made "significant" simply by running more seeds, so the size of the difference
and its uncertainty are the informative quantities.

## Sensor ablation

The research question is whether useful behavioural inference survives with
fewer physical sensors. `ablation` answers it
with three properties built in rather than left to the user to remember.

**The design is paired.** One household is simulated per seed and shared by
every configuration.
`series` refuses to return a
series when a configuration is missing a seed, so an accidentally unpaired
comparison fails loudly instead of quietly.

Pairing matters more than it might seem: simulated households differ from each
other far more than two sensor configurations differ on one household, so an
unpaired comparison buries a real effect under between-household variance.

**Ablation removes sensors, not code.** A configuration is a subset of the
registry, and the pipeline is constructed from it exactly as from a full
deployment, so an ablated run exercises the same inference path a genuinely
sparse deployment would.

**Marginal value is not assumed additive.**
`interaction` reports how far
two sensors' joint contribution departs from the sum of their individual
contributions. Two sensors that each look worthless alone can be jointly
essential -- a door tells you little without something to say who walked
through it.

### Running it

```bash
sensor-modeling ablate --days 14 --seeds 11 22 33 44 --step-minutes 10
```

### Example results

From a four-seed paired sweep over eight-day households at a ten-minute
inference step, scoring balanced accuracy:

| Configuration | Sensors | Bal. acc. | Macro F1 | Log loss | Calib. error |
| --- | --- | --- | --- | --- | --- |
| all modalities | 10 | 0.814 | 0.789 | 0.988 | 0.084 |
| objects + wearable | 8 | 0.802 | 0.790 | 1.026 | 0.081 |
| radar + door + bed + wearable | 5 | 0.651 | 0.645 | 0.676 | 0.034 |
| object sensors only | 6 | 0.641 | 0.646 | 1.054 | 0.114 |
| radar + door + bed | 3 | 0.323 | 0.302 | 0.973 | 0.134 |
| door + bed only | 2 | 0.289 | 0.257 | 0.992 | 0.169 |

Paired differences against the full deployment:

```text
all - objects_plus_wearable    +0.012  95% CI [+0.004, +0.020]  clear
all - radar_door_bed_wearable  +0.163  95% CI [+0.142, +0.181]  clear
all - object_sensors_only      +0.173  95% CI [+0.140, +0.201]  clear
all - radar_door_bed           +0.491  95% CI [+0.476, +0.512]  clear
all - minimal_door_bed         +0.526  95% CI [+0.513, +0.538]  clear
```

Four findings, stated with their caveats:

1. **Adding a person-bound wearable to six object sensors recovers most, but
   not quite all, of the full deployment's accuracy.** The eight-sensor
   configuration loses 0.012 balanced accuracy, and the interval excludes
   zero, so the gap is real rather than noise. Whether roughly one point of
   balanced accuracy justifies two further sensors is a deployment decision,
   not a statistical one.

   This is also a worked illustration of the caveat in
   [limitations](limitations.md): pairing removes between-household variance, which
   makes small differences detectable that an unpaired field study of the
   same size would not resolve. A detectable difference is not automatically
   an important one.

2. **The wearable is the single most valuable addition to a sparse
   deployment**, roughly doubling balanced accuracy over radar + door + bed
   alone (0.323 to 0.651) for two extra person-bound sensors.

3. **Fewer, better-attributed sensors can be better calibrated even when less
   accurate.** The five-sensor configuration has the best log loss (0.676)
   and by far the best calibration error (0.034) of any configuration,
   despite balanced accuracy well below the full deployment. An
   accuracy-only evaluation would hide this entirely, and for a system whose
   output is meant to be a probability it is arguably the more important
   property.

4. **Very sparse ambient-only configurations are genuinely poor.** Door and
   bed sensors alone reach 0.289 balanced accuracy with the worst
   calibration in the sweep. This is reported because it is true, not
   because it flatters the framework.

These numbers are properties of *this simulator* under *these defaults*. They
are a demonstration that the framework produces interpretable comparisons, not
a claim about real deployments. See [limitations](limitations.md).

## The worked example

```bash
sensor-modeling demo --days 90 --seed 20240304 --step-minutes 10
```

Runs one seeded experiment covering a synthetic household over months, seven
modalities, a carer and occasional visitors, a bed sensor that dies for three
days, a wearable left off for five, records lost, duplicated, delayed and
stamped by a drifting clock, and a genuine persistent change in sleep on a
known day.

It reports what the system got wrong alongside what it got right: state
inference metrics, attribution quality, transition timing, which sensors were
judged faulty and for how long, the baseline's verdicts, detection delay,
unmatched alert burden per person-day, and a worked explanation of one alert
and one individual inference.

`--output results.json` writes the same content as structured JSON. Two runs
of the same command produce byte-identical results.
