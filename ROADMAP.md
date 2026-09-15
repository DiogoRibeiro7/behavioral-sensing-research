# Project Roadmap

This roadmap describes the post-`0.3.0` direction of the Sensor Modeling
Research Toolkit.

The project is now **research-first rather than release-first**. New software
work should be driven by a clearly stated scientific question, a reproducible
analysis requirement, or a stable public API need. A new numbered release is
not a milestone by itself.

## Current Baseline

`0.3.0` was released on 2026-09-12. It is the first release whose central
behavioural-sensing claims are informed by real annotated recordings rather
than the simulator alone.

The current evidence base is:

| Result | Current evidence |
| --- | ---: |
| Simulator balanced accuracy | 0.816 |
| 22-home real CASAS development panel | 0.420 |
| Supervised diagnostic ceiling on the same sensing problem | 0.607 |
| Frozen external candidate gain over `0.2.0` | +0.0091 |
| 95% household bootstrap interval | [+0.0054, +0.0117] |
| External homes improved | 37 / 43 |
| Frozen five-sensor non-inferiority gap | 0.1548 |
| Frozen eight-sensor gap to full deployment | 0.00529 |

The practical conclusions are now clearer than they were before external
validation:

1. The simulator is substantially easier than the corresponding real sensing
   problem and must not be read as an estimate of field performance.
2. The current formulation leaves useful temporal information on the table.
   Time of day and recent room-resolved history explain much of the gap between
   the present filter and the diagnostic ceiling.
3. The current abstention mechanism is not reliable. Confidence weakly
   separates correct from incorrect real-data predictions, and confirmatory
   simulation shows that abstention remains almost silent under large amounts
   of missingness.
4. Quiet-period confidence saturation is not produced by transition dynamics
   alone. Working event streams contribute state-dependent Poisson silence
   likelihoods; complementary room-motion and entrance-door silence can jointly
   drive extreme posterior concentration even with no activations.
5. Sensor reduction should be treated as a frontier problem, not as a search
   for one universally optimal kit.

## Publication Program

The immediate objective is to develop **three additional papers** from the
research questions that `0.3.0` made precise. They should remain separate
papers because they ask different questions, require different evidence, and
have different failure conditions.

### Paper 1 — When Confidence Is Not Information

**Working title:** *When Confidence Is Not Information: Failure Modes of
Abstention in Ambient Behavioural Sensing*

**Question:** Why can the model become highly confident when its prediction is
wrong or when little genuinely new information arrived in the current update?

**Existing evidence:**

- real-data confidence is only weakly related to correctness;
- the highest-confidence real-data band performs worse than the band below it;
- simulated abstention is almost silent even as missingness rises to 40%;
- transition dynamics alone remain near the stationary posterior during a
  quiet period;
- complementary room-motion and entrance-door silence jointly drive posterior
  confidence above 0.95;
- `StateEstimate.information_gain` now exposes
  \(D_{\mathrm{KL}}(p_t\|p_{t|t-1})\) without changing inference or abstention.

**Next empirical work:**

1. Extend the existing uncertainty diagnostics with threshold-free summaries
   of information gain for correct and incorrect predictions.
2. Compare confidence, margin, entropy, evidence strength and information gain
   on real recordings.
3. Characterise quiet-period trajectories as the number and type of silent
   event streams vary.
4. Test whether extreme confidence is associated with genuinely informative
   updates or mainly with accumulated model structure.

**Boundary:** This paper should diagnose the failure before proposing a new
abstention rule. No arbitrary confidence, entropy or information-gain threshold
should be introduced just to manufacture abstention.

**Completion criterion:** A reproducible empirical account of when confidence
and information diverge, with a clearly falsifiable explanation of quiet-period
posterior saturation.

### Paper 2 — The Recoverable-Information Gap

**Working title:** *How Much Behaviour Is Recoverable from Passive Motion and
Door Sensors? Evidence Limits and Model Limits in Smart-Home Inference*

**Question:** How much of the observed performance gap is caused by the sensing
instrumentation, and how much is caused by the current probabilistic
formulation?

**Existing evidence:**

- the current pipeline reaches about 0.420 balanced accuracy on the corrected
  22-home real-data panel;
- an instantaneous-count diagnostic is about 0.397, showing that the current
  filter is already close to the ceiling of the evidence it explicitly uses;
- a supervised diagnostic using the same sensors plus time of day and recent
  history reaches about 0.607;
- the simulator reaches about 0.816 and therefore exceeds what the real
  instrumentation supports even for the diagnostic classifier;
- the circadian prior, fixed-lag smoothing and fitted emissions each improve a
  different part of the problem, but combining them does not produce additive
  gains.

**Next empirical work:**

1. Formalise the sequence of information sets used by the diagnostic ceiling
   analysis.
2. Measure the marginal value of time of day, recent history and room-resolved
   event structure under the same household splits.
3. Compare the generative filter and supervised diagnostic under matched
   information sets rather than comparing model families with different input
   information.
4. Quantify how much of the simulator-real gap is attributable to easier
   synthetic instrumentation versus inference error.

**Boundary:** The supervised model is a diagnostic upper-bound construction,
not a replacement production model and not evidence for a clinical claim.

**Completion criterion:** A decomposition of the observed gap into
instrumentation/information limitations and formulation limitations, with no
claim that one algorithm is universally optimal.

### Paper 3 — The Sensor-Information Frontier

**Working title:** *The Sensor-Information Frontier: Non-Inferiority,
Redundancy and Reliability Trade-offs in Ambient Behavioural Sensing*

**Question:** What sensing reductions preserve useful information, and how
should deployment choices be compared when accuracy, calibration and
robustness move in different directions?

**Existing evidence:**

- the frozen five-sensor deployment fails the 0.02 non-inferiority margin by a
  large amount: gap 0.1548, 95% CI [0.1531, 0.1564];
- an eight-sensor configuration sits within 0.00529 of the full ten-sensor
  system in the confirmatory simulation;
- lower-cardinality configurations lose substantially more information;
- all four pre-specified sensor interactions in the confirmatory study are
  positive;
- failure-aware reliability weighting improves log loss while worsening
  balanced accuracy, Brier score and calibration error.

**Next empirical work:**

1. Define the frontier formally using multiple metrics rather than balanced
   accuracy alone.
2. Compare candidate sensor sets by incremental information contribution,
   redundancy and robustness to failure.
3. Separate configuration selection from confirmatory scoring: candidate sets
   must be chosen before the final comparison.
4. Where the CASAS mappings support it, test whether the simulation frontier
   ordering survives on real recordings.

**Boundary:** Do not promote the current eight-sensor result into a universal
recommended deployment. It is one point on one experimentally defined
frontier.

**Completion criterion:** A reproducible sensor-information frontier with
pre-specified comparison rules and explicit trade-offs among discrimination,
calibration and failure robustness.

## Existing Paper

The current manuscript,
`papers/failure-aware-multimodal-behavioural-sensing/`, remains the primary
system paper. It should describe the platform, frozen confirmatory simulation,
real-data validation and the limits established by `0.3.0`.

The three papers above should not duplicate that manuscript. They are intended
as focused follow-ups:

1. uncertainty and abstention failure;
2. recoverable information and formulation limits;
3. sensor-information frontier and deployment trade-offs.

## Software Work Supporting the Papers

Software changes should stay subordinate to the analyses above.

### Priority A — uncertainty diagnostics

- Summarise per-update information gain in the real-data uncertainty report.
- Preserve `None` when information gain is unavailable; do not silently coerce
  missing diagnostics to zero.
- Add only diagnostics that are required by Paper 1.

### Priority B — matched-information evaluation

- Keep the diagnostic classifier isolated from production inference.
- Make information-set comparisons reproducible across household splits.
- Add machine-readable outputs required for Paper 2 figures and tables.

### Priority C — frontier analysis

- Generalise sensor-subset comparison only as far as Paper 3 requires.
- Retain paired designs, fixed seeds where applicable, uncertainty intervals
  and Monte Carlo precision reporting.
- Avoid a combinatorial search framework until the scientific comparison is
  specified.

## Deferred Model Changes

The following are scientifically plausible, but should not be implemented
before the relevant paper establishes the need:

- a new abstention rule based on information gain, entropy or evidence strength;
- arbitrary caps on accumulated silence evidence;
- changing transition persistence merely to reduce confidence;
- replacing independent room-silence likelihoods without first measuring the
  effect of correlated silence streams;
- introducing deep learning solely to improve headline accuracy;
- selecting a reduced sensor kit post hoc from the confirmatory results.

A potentially principled future direction for the silence model is to separate
a global total-event process from conditional room allocation, rather than
multiplying many correlated room-level silence likelihoods independently. This
remains a research hypothesis, not a planned implementation.

## External Validation Beyond CASAS

All current real-data evidence comes from CASAS instrumentation. The number of
homes is not the number of independent studies.

A later validation phase should therefore use at least one independently
collected annotated smart-home dataset with a materially different sensing
layout. This becomes a priority after the three papers above have frozen the
questions and metrics that should transfer.

## Maintenance Backlog

Maintenance work remains worthwhile but should not displace the publication
program unless it blocks reproducibility or supported users.

- Reduce pre-existing type-checking debt in older modules.
- Keep package metadata, citation metadata and Zenodo records synchronized.
- Preserve Python 3.10–3.12 compatibility until a deliberate support change.
- Keep documentation aligned with measured real-data limitations.
- Remove stale terminology that describes trusted event-stream silence as
  "absence of evidence"; under the current Poisson event model, zero counts
  from working sensors are themselves evidence.
- Keep CI proportional to change scope so documentation changes do not trigger
  expensive scientific test matrices unnecessarily.

## Release Policy

`0.3.0` is the current stable release.

No `0.4.0` scope is frozen. The next version should be created only when the
publication work produces a coherent, user-facing software increment. A paper
milestone does not require a package release, and a package release should not
be created merely because enough commits have accumulated.

Release notes come from the corresponding `CHANGELOG.md` section. The changelog
is the single source of truth for release notes.

## Immediate Order of Work

1. **Paper 1:** extend real-data uncertainty diagnostics with information gain
   and quantify confidence-versus-information failure.
2. **Paper 2:** freeze the matched-information ceiling analysis and gap
   decomposition.
3. **Paper 3:** formalise and estimate the sensor-information frontier.
4. Validate beyond CASAS once the three analysis contracts are stable.
5. Consider the next package release only after those results identify a
   coherent software change worth releasing.
