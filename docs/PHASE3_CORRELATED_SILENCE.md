# Phase 3.3: correlated silence, a diagnostic

The generative model multiplies one Poisson likelihood per sensor. A channel
that records nothing in a window adds `−λ_c(s) Δ` to the log-evidence for
state `s`, and a quiet window's evidence is the sum of those terms over every
silent channel. The [uncertainty model](UNCERTAINTY_MODEL.md) showed that
silent working streams, not the transition prior, drive quiet-period
confidence above 0.95.

The roadmap's hypothesis 3.3 is that this is over-concentration: channels fall
silent together, given the state, more often than independence implies, so the
sum overstates the evidence. The proposed remedy, a two-stage model with a
total count and a room allocation, is only worth building if that is
materially true. This diagnostic measures it.

**Status: pre-specified diagnostic, development panel.**

- **Diagnostic only.** No inference, emission model or default changes. The
  diagnostic reads the filter's own emission terms and transition.
- **Frozen first.** Every setting and rule was committed before any household
  was examined.
- **Not held out.** The development homes have been inspected in earlier work.
  The frozen external cohort is not touched.

## Streams

A stream is an evidence channel of the [information sets](INFORMATION_SETS.md):
one room and one modality, pooling that room's event sensors, counted in the
same 5-minute windows as Phases 1, 3.1 and 3.2. There are six:

- bathroom, bedroom, hall, kitchen and living-room motion;
- the hall door.

A household's uninstrumented channels are left out.

Under the model, a silent channel's likelihood is the same whether its sensors
are pooled or multiplied one by one, `exp(−Σ_i λ_i(s) Δ)`. So pooling loses
nothing for the silence terms.

## Protocol

The protocol is `artifacts/phase3/silence_protocol.json`. The script refuses
to run if the code's protocol differs from it by a single value. It freezes:

- **Households.** All 20 single-resident development homes of
  `artifacts/phase1/household_splits.json`, recorded by digest. Nothing is
  fitted, so the folds are not used.
- **Eligibility.** A state enters a household's dependence statistics with at
  least 50 labelled windows there.
- **Estimability.** A joint-silence statistic is computed only where at least
  5 windows would be jointly silent if the channels fell silent independently.
  - Below that, the smoothed estimate sits at its floor, and a ratio to
    independence would measure the floor.
  - The rule uses only the channels' own silence, so it does not select on
    dependence.
  - A log odds ratio needs all four cells of its table to reach 5.
  - This is the classical expected-count rule for contingency tables.
- **Smoothing.** Probabilities are `(k + 0.5) / (n + 1)`.
- **Model.** The filter's generative model: default emissions pooled per
  channel, the default ontology, and full reliability. It is fed every window
  of the recording from the stationary distribution, which is the production
  filter's recursion. A test checks this against `MultimodalBayesFilter`.
- **Bootstrap.** 10,000 household resamples, with 95% percentile intervals for
  the mean and the median. The seed is 0.

### What is measured

Within each household, conditional on the annotated state:

- **Joint silence.** The observed probability that every instrumented channel
  is silent is compared with three independence baselines:
  - the product of the channels' observed silence probabilities, which tests
    dependence alone;
  - independent Poisson streams with each channel's observed mean count, which
    adds each channel's departure from a Poisson count;
  - the filter's declared rates, which are what the filter assumes.
- **Pairwise dependence.** For every pair of channels:
  - the log odds ratio of silence;
  - the log ratio of joint to independent silence;
  - the correlation and covariance of counts.
- **Dispersion.** The variance of the number of silent channels, against its
  variance if channels fell silent independently.
- **Silence-evidence inflation.**
  - A constant excess of joint silence in every state shifts each state's
    evidence equally, and cannot move a posterior. Only its variation across
    states can.
  - The inflation factor is therefore the spread across states of the
    independent joint-silence log-evidence, over the spread of the observed
    one: `SD_s[Σ_c log p_c(s)] / SD_s[log p_joint(s)]`.
  - It is 1 under independence, and above 1 when independence overstates the
    evidence.
  - It is computed with the channels' observed silence probabilities
    (dependence alone, D1) and with the filter's declared terms (S1). Their
    difference, S2, is what the declared rates add.
- **Concentration.**
  - Each labelled window's overconfidence is its confidence minus its
    correctness.
  - C1 is its least-squares slope on the number of silent channels, within
    the predicted state.
  - A calibrated model has zero expected overconfidence given anything
    computed from the data, so every such slope is zero in expectation.
  - Grouping by the annotated state would condition on the truth, and a
    calibrated model would then show a slope. Every labelled window enters
    for the same reason.
  - S8 is the same slope along consecutive fully silent windows, per hour,
    over the first three hours of each quiet run.
- **Representative quiet periods.**
  - For each household and quiet state, the run is chosen by a fixed rule.
    Candidates are the maximal runs of fully silent windows labelled with
    that state, at least 12 windows (one hour) long. The case is the earliest
    run of the lower-median length.
  - Its first 12 windows are decomposed: the prediction from the transition,
    each silent channel's contribution and the confidence after adding it, the
    posterior, the number of silent channels and the final confidence.

The quiet states are `sleeping`, `away` and `home_inactive`, where the
saturation was observed.

### Estimands

Each is one value per household, tested against zero: independence for
dependence measures, or a calibrated model for slopes.

| Key | Role | Quantity | Question |
| --- | --- | --- | --- |
| D1 | primary | log inflation factor, observed marginals | does independence overstate silence evidence, from dependence alone? |
| D2 | primary | log joint-silence ratio, quiet states | do channels fall silent together more often than independence implies? |
| C1 | primary | overconfidence slope per silent channel | does overconfidence grow with the number of silent channels? |
| S1 | secondary | log inflation factor, declared terms | how much do the filter's own silence terms overstate it? |
| S2 | secondary | S1 − D1 | how much of that comes from the declared rates? |
| S3 | secondary | median pairwise log odds ratio of silence, quiet states | pairwise dependence of silence |
| S4 | secondary | mean pairwise count correlation, quiet states | dependence of counts |
| S5 | secondary | log dispersion of the number of silent channels, quiet states | dependence across all channels |
| S6 | secondary | confidence slope per silent channel | the confidence part of C1 |
| S7 | secondary | accuracy slope per silent channel | the accuracy part of C1 |
| S8 | secondary | overconfidence slope per quiet hour | accumulation along quiet runs |

Every per-state statistic is also reported across households for every
state. Per channel pair, both pairwise silence measures are reported, averaged
over each household's quiet states. The Spearman correlation of D1 and C1
across households is reported with a household bootstrap interval.

### Criteria

The criteria compare effect sizes, with their uncertainty, against declared
minimal effects `δ`. They are not significance tests.

| Scale | `δ` | Why |
| --- | ---: | --- |
| log ratio | log 1.25 | At an inflation of 1.25, a posterior of 0.95 reached on silence alone would be 0.91 with the evidence corrected. That leaves the uncertainty study's 0.95–1.00 confidence band. |
| slope | 0.02 per silent channel or per hour | The calibration-error minimal difference of Phases 3.1 and 3.2. |
| correlation | 0.1 | The conventional smallest correlation of note. |

Each estimand gets one verdict:

| Verdict | When |
| --- | --- |
| positive | mean ≥ `δ` and the interval's lower bound > 0 |
| negative | mean ≤ `−δ` and the interval's upper bound < 0 |
| negligible | the whole interval lies within `(−δ, δ)` |
| uncertain | anything else |

The overall rule is fixed in advance:

- **Supported.** D1 and C1 are both positive. Independence materially inflates
  silence evidence, and overconfidence grows with the number of silent
  channels.
- **Weakened.** D1 or C1 is negligible or negative.
- **Inconclusive.** Anything else.

`ROADMAP.md` is updated only if the conclusion is supported or weakened.

D1, not D2, enters the rule. A constant excess of joint silence cannot move a
posterior, so D2 can be large while the posterior is unaffected. C1 alone
cannot attribute over-concentration to dependence, because wrong declared
rates can produce it too. D1 separates the two.

### Checked on synthetic households

The tests build panels of synthetic households whose states follow the
ontology's own chain:

- **Conditionally independent streams**, scored by a correctly specified
  model. D1, D2, the pairwise log odds, the dispersion and C1 are all
  negligible, and the conclusion is weakened.
- **Strongly correlated silence.** Half the windows are silent on every
  channel together, and each channel keeps its mean. D1, D2, the pairwise log
  odds, the dispersion and C1 are all positive, and the conclusion is
  supported.

They also check that:

- the recursion equals `MultimodalBayesFilter` fed the same windows;
- a decomposed quiet window adds up to its posterior.

## Results

The results will be published here from the record
`artifacts/phase3/phase3-correlated-silence.json`.

## Reproducing it

```text
python scripts/run_phase3_silence.py <archive_root> <output_dir>
python scripts/plot_phase3_silence.py <record.json> <figure_dir>
```

`archive_root` is the extracted `labeled_data.zip` of Zenodo record 15708568
(CC-BY-4.0, not redistributed here). The first script writes the record, a
Markdown summary generated from the written record, and the figures. The
second redraws the figures from any written record. Each SVG carries the
SHA-256 of the data it plots, and a test checks the committed figures against
the published record.
