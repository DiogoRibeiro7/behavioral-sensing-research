# Evaluation Design

The experimental design, why each choice was made, and the threats to validity
that remain. Implemented in `sensor_modeling.evaluation` and
`sensor_modeling.simulation`.

## Avoiding circular validation

The simulator exists so inference can be scored against something known. That
only works if the generative process differs from the inference model.

| | Generative process | Inference model |
| --- | --- | --- |
| Structure | Stochastic daily **schedule** | Continuous-time **Markov chain** |
| State duration | Drawn from per-activity distributions | Implicitly exponential |
| Day structure | Wake, breakfast, outing, lunch, dinner, bed | None; the chain has no notion of a day |
| Sensor firing | Per-room Poisson at activity-scaled rates | Poisson likelihood with declared rates |

The estimator has to recover a schedule through a model that knows nothing
about schedules. Had the two shared a generator, good results would prove only
that the code can invert its own assumptions.

Two guard tests enforce this and will fail if a future edit couples them:

- `sensor_modeling.simulation.household` must not import `StateOntology`,
  `transition_matrix` or `MultimodalBayesFilter`.
- `sensor_modeling.online.pipeline` must not import `GroundTruth`, `Episode`
  or anything from `simulation`.

Ground truth is available only to evaluation code and is never passed to the
pipeline.

## Separating behaviour from degradation

Fault injection is a separate module from behaviour generation. This is what
makes paired comparison possible: the same simulated fortnight, once with a
working wearable and once without, differs in exactly one thing.

Degradations available: dropout, stuck sensors, random loss, wearable
non-adherence, late arrival, duplication, per-source clock drift.
`degrade()` returns the withheld records, so an evaluation reports how much
evidence was actually missing rather than inferring it.

## Metric selection

Accuracy alone is close to meaningless here. The states are heavily
imbalanced — a resident is asleep or quietly at home for most of the day — so
a model that predicts `home_inactive` forever scores about 0.9 and knows
nothing. Scoring only the argmax also discards the part of every output that
matters.

### State inference

| Metric | Why |
| --- | --- |
| Balanced accuracy | Immune to the class imbalance that makes raw accuracy misleading |
| Macro F1 | Penalises a model that achieves recall by over-predicting a class |
| Log loss | Punishes confident errors far more than hedged ones |
| Brier score | Proper scoring rule over the whole posterior |
| Expected calibration error | Whether a stated confidence of 0.9 means anything |
| Per-class recall | Exposes which states are being lost |
| Transition timing | Whether changes are detected at the right moment, not merely counted |

Transition matching is one-to-one within a tolerance: each inferred transition
is consumed at most once, so a flickering model cannot match every truth by
accident.

### Abstention convention

Stated once and applied everywhere: **a reported `UNKNOWN` is never correct**,
because `UNKNOWN` is never a true label. It therefore costs recall.

`selective_accuracy` (accuracy among committed estimates) and
`abstention_rate` are reported alongside, so a model that declines usefully is
distinguishable from one that is simply wrong. Without both numbers, either
convention alone would mislead.

### Attribution

Precision, recall, F1 and calibration. Calibration is scored on the stated
probability of the *predicted* class, so a confident "no visitor" is judged as
strictly as a confident "visitor".

### Change detection

Detection delay, precision, recall, and **false positives per person-day** —
the number that decides whether a system is usable in practice.

A detection counts only if it falls at or after the true change and within
`max_delay_days`. An alert raised *before* the change is a false positive, not
early detection: a system that alarms before anything happened has alarmed at
noise.

Detection is scored on **alerts actually delivered**, not raw baseline
verdicts. Deduplication and rate limiting sit between the two, and it is the
delivered burden a carer experiences.

### Reliability

Performance is swept across 5%, 10%, 20% and 40% missingness, and separately
under single-sensor failure, multi-sensor failure, wearable non-adherence,
visitor contamination and total blackout. Calibration is reported at every
point, because the failure mode of interest is confidence staying high while
accuracy collapses.

## Statistical design

### Pairing is structural, not procedural

One household is simulated per seed and shared by every configuration.
`AblationReport.series()` **refuses** to return a series when a configuration
is missing a seed, so an accidentally unpaired comparison fails loudly rather
than quietly producing a plausible number.

Why it matters: simulated households differ from each other far more than two
sensor configurations differ on one household. An unpaired comparison buries a
real effect under between-household variance.

### Effect sizes and intervals, not p-values

`paired_difference()` reports the mean paired difference, a bootstrap
confidence interval, Cohen's *dz*, and a win/loss count. It deliberately does
not report a p-value.

With simulations, any effect can be made "significant" by running more seeds.
The size of the difference and its uncertainty are the informative quantities;
significance is a statement about how long the computer ran.

**A caveat that cuts against the design.** Pairing removes between-household
variance, which makes small differences detectable that an unpaired field
study of the same size would not resolve. A detectable difference is not
automatically an important one. Both the raw mean difference and the interval
are reported so magnitude stays visible.

## Ablation protocol

Configurations are **registry subsets**, not code variants. The pipeline is
constructed from the subset exactly as from a full deployment, so an ablated
run exercises the same inference path a genuinely sparse deployment would.

Marginal contribution of sensor `j`:

```text
Delta_j = performance(S) - performance(S \ {j})
```

evaluated on paired trajectories.

Interaction between two sensors:

```text
interaction = joint_contribution - (individual_a + individual_b)
```

Positive means complementary, negative means redundant. Marginal sensor value
is **not** assumed additive: a door tells you little without something to say
who walked through it, so a sensor that looks worthless alone can be
conditionally essential.

## Reproducibility requirements

Every experimental result records configuration, resolved algorithm defaults,
seed, sensor subset, metric definitions, the software version, and the git
commit the code came from together with whether the working tree was dirty.

The guarantee is:

> the same seed, the same commit and the same resolved configuration produce
> the same scientific results.

Not byte-identical files. Each artefact carries a `recorded_at` stamp and the
environment it ran in, so two writes of the same record differ in that envelope
while the `results` they contain are identical. The results equality is what
reproducibility needs and is what the tests assert.

A run made from a dirty working tree is marked `git_dirty: true` and is not
reproducible from its commit alone.

Generated datasets are not committed. Experiments regenerate from their seed.

## Threats to validity

Ordered by how much they should worry a reader.

| Threat | Status |
| --- | --- |
| **All results come from one simulator, written by this project** | Unmitigated. The dominant limitation. |
| Simulator encodes the project's beliefs about behaviour | Partially mitigated by structural independence and guard tests |
| Seeds vary noise, not structure | Unmitigated. Intervals quantify Monte-Carlo variability under one model, not model uncertainty |
| Occupancy and state layers share evidence | Unmitigated. Errors are correlated; uncertainty does not reflect it |
| Pairing inflates standardised effect sizes | Mitigated by reporting raw differences alongside |
| Metric choice could still flatter | Mitigated by reporting calibration and log loss alongside accuracy, where they disagree with it |
| Cherry-picked sensor subsets | Mitigated: the CLI subsets are fixed, named, plausible deployments rather than a search |

## What would make the evaluation trustworthy

1. A public annotated smart-home dataset (CASAS, ARAS, MARBLE) scored with the
   same metrics.
2. Parameters fitted from data rather than declared.
3. A second, independently written simulator with different structural
   assumptions.
4. Calibration assessed across households, not only within one.
5. Prospective assessment of alert burden with people who would act on the
   alerts, since false-positive tolerance is a human judgement no metric
   supplies.
