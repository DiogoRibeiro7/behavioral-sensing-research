# Project Roadmap

This roadmap describes the planned direction for the Sensor Modeling Research
Toolkit. It is release-oriented: items are grouped by the outcome they should
unlock, not by implementation preference. The roadmap can change as research
needs, user feedback, and maintenance constraints evolve.

## Guiding Principles

The project should remain useful for reproducible behavioral sensing research.
New work should prioritize:

- Stable, documented APIs over one-off scripts.
- Interpretable methods before opaque models, unless the use case clearly
  benefits from additional complexity.
- Explicit validation for data assumptions, missingness, and model calibration.
- Tests and examples that make research workflows reproducible.
- Lightweight deployment paths for local, clinical, and edge environments.

## Current Baseline

`0.2.0` is released, on PyPI and archived on Zenodo. It extends the original
modelling core into a multimodal ambient-sensing platform: canonical
observations, sensor health, occupancy and attribution, continuous-time fusion
with abstention, adaptive baselines, restrained alerting, simulation,
evaluation, an incremental pipeline, and interoperability export.

Since release, the platform has been measured against real recordings for the
first time. **That measurement is now the most important fact about the
project's status**, and the rest of this roadmap is organised around it.

| | Balanced accuracy |
| --- | --- |
| Simulator | 0.816 |
| 22 real CASAS homes, nothing refitted | **0.420** |
| Recoverable from those sensors by any method | 0.607 |

Three consequences, each documented in
[Real-data validation](docs/real_data.md):

- **The simulator is easier than reality even for an optimal method.** Its
  0.816 sits above the 0.607 a supervised classifier reaches on the same
  sensors, so simulator figures are not an estimate of real-world performance
  and every number produced by it should be read with that in mind.
- **The inference is not the bottleneck for the evidence it uses.** Given only
  instantaneous per-room counts the ceiling is 0.397 and the pipeline scores
  0.420. The shortfall is in time-of-day and recent history, neither of which
  the formulation carries.
- **Abstention does not work on real data, and cannot be fixed by tuning.**
  Stated confidence separates right from wrong by 0.073 and inverts above 0.95,
  so no threshold setting repairs it. This is the project's central safety
  claim and it does not currently hold outside the simulator.

The frozen confirmatory simulation also completed since release, at 200 paired
household trajectories across 40 shards, with its Monte Carlo precision gate met
(maximum observed MCSE 0.001125 against a 0.002 target). Its headline result is
negative. The frozen five-sensor deployment is **not** non-inferior to the
ten-sensor deployment: it loses 0.1548 balanced accuracy, 95% CI
[0.1531, 0.1564], against a 0.02 margin — nearly eight times the margin. A
secondary eight-sensor configuration sits within 0.00529 of the full system
(0.7405 against 0.7458). These are the frozen study's figures; the 100-seed
`sensor-modeling ablate` study reported in
[research questions](docs/RESEARCH_QUESTIONS.md) puts the same two contrasts at
0.0073 and 0.171 under a different protocol and different seeds, and the two
sets are not interchangeable. The useful object is a sensor-information frontier
rather than a winning kit, and only that one kit is settled. All four
pre-specified sensor interactions are positive.

Two of its other results bear on the sections below. Failure-aware reliability
weighting is a scoring trade-off rather than a win: it improves log loss by
0.110 while worsening balanced accuracy by 0.0145, Brier by 0.0156 and
calibration error by 0.0039. And abstention barely moved — mean rates rose only
from 9.7e-06 to 5.3e-05 as random missingness went from 0% to 40%, so the
mechanism is effectively silent and hardly responds to losing two fifths of the
evidence.

**That last result changes the standing of the abstention failure.** It is no
longer one observation on one kind of data. The mechanism fails in the
simulator and on real recordings by different routes — near-silence in the
first, uninformative confidence in the second — which removes the most
comfortable reading, that it is an artefact of unfamiliar real-world data. All
confirmatory outcomes remain simulator-derived and are not estimates of field
performance.

Retained and unchanged: CSV/JSON/HDF5 loading, missing-data handling, Bernoulli
autoregressive models, HMM variants, NHPP-PELT segmentation, change-point
detectors, dependency analysis, reporting, the Flask app, and the original CLI.

## Capability Status by Theme

Work is grouped by the outcome it unlocks. Each theme states what exists now
and what is outstanding.

### Existing capability (pre-multimodal core)

Retained unchanged and still supported: CSV/JSON/HDF5 loading, gap-aware
missing-data handling, Bernoulli autoregressive models, HMM variants,
NHPP-PELT segmentation, change-point detectors, Granger and dependency-network
analysis, LaTeX/HTML reporting, the Flask visualisation app, and the original
CLI subcommands.

### Foundation work

Delivered:

- A canonical, hardware-neutral observation model with boundary validation,
  unit conversion, duplicate collapse, out-of-order handling, late-arrival
  flagging, and per-source clock-drift correction.
- A sensor registry that declares each sensor's modality, semantics, room,
  expected cadence, and whether its activations are attributable to a person.
  Inference reads declarations rather than sensor names.
- Online sensor health estimation emitting a per-sensor evidence weight, with
  silence only treated as failure where a sensor promised to report.

### Multimodal sensing

Delivered:

- A configurable continuous-time behavioural state ontology and a recursive
  multimodal Bayes filter over asynchronous, partially missing evidence, with
  explicit abstention.
- Probabilistic occupancy estimation and uncertainty-aware attribution of
  ambient activity, using anonymous evidence only.

### State inference

Delivered: a configurable state ontology whose semantic claims stop where the
evidence stops, with explicit abstention when confidence or sensor coverage is
insufficient.

### Personalisation

Delivered:

- Adaptive, robust, weekday-aware personal baselines distinguishing ordinary
  variability, weekly rhythm, temporary disturbance, persistent change,
  gradual drift, abrupt change, and insufficient data.
- Restrained alerting with deduplication, rate limiting, explicit caveats, and
  a strict separation between system-health and behavioural findings.

### Evaluation

Delivered:

- A synthetic household simulator with controlled ground truth, generatively
  independent of the inference model, plus separate fault injection.
- Problem-appropriate evaluation metrics and a paired sensor-ablation
  framework reporting effect sizes and bootstrap intervals.

### Online and edge processing

Delivered: an incremental, bounded-memory, snapshot-able pipeline with a
lateness buffer for stream reordering, independent of any UI or storage.

### Clinical interoperability

Delivered: a FHIR-style export that keeps measurements, derived features,
inferred states and algorithmic alerts distinguishable through explicit
provenance. Not a validated profile; see `docs/limitations.md`.

### Reproducible research

Delivered:

- A reproducible end-to-end command-line demonstration and ablation
  experiment.

User documentation: `docs/ambient_architecture.md`, `docs/inference.md`,
`docs/evaluation.md`, `docs/limitations.md`.

Design record: `docs/MULTIMODAL_ARCHITECTURE.md`, `docs/RESEARCH_QUESTIONS.md`,
`docs/SENSOR_DATA_MODEL.md`, `docs/UNCERTAINTY_MODEL.md`,
`docs/EVALUATION_DESIGN.md`.

### Longer-term research

The item that dominated this list — validation against real annotated sensor
data — has been done, and its results reshaped the rest. What follows is what
the measurement left open, in priority order.

1. **Reconnect abstention to something informative.** Confidence currently
   inverts above 0.95, most likely because a sticky chain saturates the belief
   during quiet periods: the posterior approaches certainty because no evidence
   arrived, not because the evidence was strong. That mechanism remains a
   hypothesis, but the failure itself is now measured twice over, in the
   confirmatory simulation and on real recordings, so it cannot be attributed to
   real-world data alone. RQ2 — can a system fail safely rather than silently —
   is accordingly answered negatively, its own falsification condition met; see
   [research questions](docs/RESEARCH_QUESTIONS.md). Until this is addressed the
   project cannot claim a model that knows when it does not know.
2. **Close the accuracy gap, or establish that this formulation cannot.**
   0.420 against a 0.607 ceiling. Both principled additions tried so far, a
   circadian prior and a fixed-lag smoother, recovered about a tenth of what a
   discriminative model extracts from the same inputs, and the three components
   that help individually do not combine. That consistency suggests a
   formulation limit rather than a missing term.
3. **Resolve the dwell-time tension.** Declared dwells are 3 to 9 times longer
   than measured state durations, and correcting them *lowers* accuracy: they
   are doing work as regularisation. The prior that stabilises the filter is
   also the prior that saturates its confidence, and those cannot both be
   right.
4. **Fit emissions from data with held-out validation.** Fitting rates cut
   calibration error from 0.312 to 0.202 on unseen homes without moving
   accuracy, so the parameters explain the overconfidence and not the
   discrimination.
5. **Re-open sensor selection against the frontier rather than the frozen kit.**
   The confirmatory study found the specific five-sensor deployment decisively
   short of the non-inferiority criterion, but an eight-sensor configuration came within 0.00529 of the full
   system while six, three and two sensors lost far more. The reduction question
   is therefore still open and still worth answering; only the one frozen kit is
   settled. Any successor kit should be chosen against the frontier and then
   frozen before scoring, exactly as this one was.
6. **Model the dependency between the occupancy and state layers**, which share
   evidence and therefore have correlated errors the reported uncertainty does
   not reflect.
7. **Validate beyond CASAS.** Everything real measured so far comes from one
   research group's instrumentation, so it is not 22 or 43 independent studies.
8. **Reduce the pre-existing type-checking debt** in the older modules and
   widen the CI `mypy` gate.

## Near-Term Roadmap

### 0.2.0: released

Shipped, and not what this section originally planned. The release delivered
the multimodal ambient-sensing platform rather than an API-stabilisation pass:
canonical observations, sensor health, occupancy and attribution, continuous-time
fusion, adaptive baselines, alerting, simulation, evaluation, the incremental
pipeline, and FHIR-style export. Documentation moved to MkDocs and the release
checklist now requires reading CI before tagging.

The API-stabilisation items that motivated the original plan remain worth doing
and are folded into the maintenance backlog rather than a numbered release.

**Caveat carried forward.** The quantitative results distributed with `0.2.0`
come from the simulator, which measurement has since shown sits above the
ceiling real instrumentation supports. The repository documentation says so; the
published PyPI page and Zenodo record for `0.2.0` carry the earlier wording and
cannot be amended from here.

### 0.3.0: External validation

Goal: find out whether the platform transfers to homes it has never seen, under
a design that cannot be adjusted after the result is known.

The candidate is **v0.2 inference plus the optional circadian state-dynamics
profile**, with every other v0.2 choice retained. It was selected on development
evidence alone: the circadian prior raised held-out balanced accuracy from 0.449
to 0.460 across 11 homes, improving 10 of them, while fitted emission rates and
fitted dwell times were considered and excluded because they worsen the
pre-specified primary metric.

Frozen before any scoring:

- the candidate, in [the specification](docs/V03_CANDIDATE_SPECIFICATION.md);
- the circadian profile `5f03753f...`, fitted on 20 single-resident development
  homes, reproduced independently across implementations and machines;
- the primary cohort of 43 single-resident homes outside the development panel,
  with per-home checksums and a screening revision;
- the interpretation boundaries, in
  [cohort composition](docs/EXTERNAL_COHORT_COMPOSITION.md).

**The scoring has been executed.** It ran once, against the frozen cohort and
grid, and the result stands: median paired difference in household balanced
accuracy **+0.0091**, 95% bootstrap interval [+0.0054, +0.0117], with 37 of 43
homes improved and none unscoreable. The interval excludes zero, so the
circadian prior transfers to homes and instrumentation the candidate was not
developed on. Secondary outcomes moved with it: balanced accuracy 0.496 to
0.504, calibration error 0.343 to 0.328, Brier 0.842 to 0.812, log loss 4.302 to
4.168. Recorded in `artifacts/v03/external_primary_result.json`, with the
one-shot marker in `artifacts/v03/external_primary_scored.json`.

Two things temper it. The effect is small — nine thousandths against the 0.607
ceiling real instrumentation supports — so it closes a sliver of the accuracy
gap rather than the gap. And abstention did not improve, moving 0.0009 to
0.0013 and continuing not to fire, which leaves item 1 below exactly where it
was.

**Reading the result will require care.** Only 19% of the cohort comes from the
`hh` family the candidate was developed on, so a null result is ambiguous
between the prior failing to transfer and the adapter losing information on
unfamiliar sensor vocabularies. And because v0.3 changes the transition prior,
which is the mechanism implicated in the abstention saturation, the abstention
secondary outcome may move for reasons unrelated to whether the circadian prior
is a good idea.

The data-quality and validation work that previously occupied this slot is not
abandoned; it moves to the maintenance backlog.

### 0.4.0: Model Evaluation and Comparison

Goal: provide consistent evaluation across modeling approaches.

Planned work:

- Standardize model scoring interfaces for Bernoulli AR, HMM, NHPP, and CPD
  components.
- Expand time-series cross-validation utilities.
- Add calibration metrics and uncertainty diagnostics for probability models.
  This is now the highest-value item in the section rather than a routine one:
  real-data measurement showed the pipeline's stated confidence separates
  correct from incorrect answers by only 0.073 and inverts above 0.95, which no
  aggregate calibration score would have revealed. A diagnostic that reports
  accuracy *within* confidence bands would have caught it.
- Add model comparison reports with structured machine-readable output. The
  confirmatory H4 result is the argument for reporting several scores together
  rather than one: failure-aware weighting improved log loss by 0.110 while
  worsening balanced accuracy, Brier and calibration error, so any single-metric
  comparison would have declared it a straight win or a straight loss depending
  only on which metric was chosen.
- Add benchmark datasets or synthetic benchmark recipes with fixed seeds.

Quality gates:

- Comparison utilities work with at least two model families.
- Metrics are documented with expected input and output shapes.
- Benchmarks are reproducible from the command line.

### 0.5.0: Clinical and Interoperability Workflows

Goal: improve reporting and integration paths for clinical and applied research
prototypes.

Planned work:

- Expand the current minimal FHIR-style export toward documented Observation
  and Bundle structures.
- Add configurable clinical threshold profiles.
- Improve patient-friendly summaries and trend visualizations.
- Add anonymization and redaction helpers for exported reports.
- Document limitations clearly: research prototype, not a medical device.

Quality gates:

- Clinical exports have schema-oriented tests.
- Report examples avoid exposing sensitive identifiers.
- Clinical documentation distinguishes supported behavior from future work.

## Longer-Term Roadmap

These items are valuable but should follow the stabilization work above.

### Deep Learning Change-Point Detection

- Add transformer, CNN, or autoencoder-based CPD only after baseline evaluation
  utilities are stable.
- Provide simple training examples and clear dataset requirements.
- Compare deep approaches against existing interpretable baselines.

### Online and Real-Time Processing

- Add incremental preprocessing and validation utilities.
- Add online change-point detection interfaces.
- Support streaming report updates and bounded-memory operation.

### Packaging and Distribution

- Publish stable package artifacts when the public API is ready.
- Add compatibility checks for supported Python versions.
- Document optional dependency groups by workflow.

### Performance and Scalability

- Profile slow model-fitting and analysis paths.
- Add benchmark tracking for core algorithms.
- Evaluate vectorization and optional acceleration where it reduces real runtime
  without making the code harder to maintain.

## Maintenance Backlog

The following work can be handled continuously across releases:

- Replace remaining legacy typing aliases with modern annotations.
- Reduce broad exception handling where errors can be handled specifically.
- Improve coverage for analysis and visualization modules.
- Keep examples synchronized with public APIs.
- Expand changelog entries for every release after `0.1.0`.
- Keep Zenodo, citation, and package metadata aligned before each release.

## Release Policy

Development work happens on `develop`. Releases are cut from `main` only.

Release flow:

1. Finish and validate work on `develop`.
2. Merge `develop` into `main`.
3. Tag the release on the `main` commit.
4. Publish the GitHub release from that tag.
5. Confirm Zenodo metadata and DOI linkage.
6. Update the changelog and documentation as needed.

## Out of Scope for Now

The following are not immediate priorities:

- Medical-device claims or regulated clinical decision support.
- Large-scale cloud platform features.
- Opaque deep-learning models without benchmark comparisons.
- Breaking public APIs without a documented migration path.
