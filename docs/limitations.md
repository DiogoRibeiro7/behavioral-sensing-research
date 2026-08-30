# Known Limitations

This page states what the ambient-sensing pipeline does **not** establish. It
is deliberately blunt. A monitoring system whose limitations are not written
down will have them discovered by whoever trusts it first.

## Measured on real data

Twenty-two real CASAS homes have now been scored with nothing refitted, and
with every location and activity label mapped so that no evidence is discarded.

| | Simulator | Real homes (median) |
| --- | --- | --- |
| Balanced accuracy | 0.816 | **0.364** |
| Calibration error | 0.084 | **0.285** |

No home exceeded 0.468. The failure takes the same shape in all of them. States
with a distinctive room-and-rate signature held up — sleeping 0.74, activity
0.57, cooking 0.57 — while states needing evidence that the resident is
*present but still* collapsed: `home_inactive` 0.16, and **`away` 0.00 in the
median home**.

Why remains open. The obvious explanation, that these deployments lack the
sensors confirming presence, was tested twice and did not hold. The five homes
carrying a chair occupancy sensor are not better, with `home_inactive` recall of
0.10 against 0.17 without it. Re-reading motion as a persistent occupancy state
so that an OFF asserts absence made things worse: `away` recall reaches 1.00
while balanced accuracy falls to 0.16-0.23, because it then reports `away`
almost always. A motion sensor's OFF means "no motion just now", not "nobody
home", and the event-kind reading is the correct one.

So the failure is established and its mechanism is not.

Abstention reached at most 5.2% in any home, median 2.2%, while the model was
wrong more often than right. The mechanism presented throughout as the safety
valve did not open.

An obvious alternative explanation was checked and rejected: an incomplete
location map was discarding evidence in 14 homes, and fixing it moved the median
from 0.356 to 0.364.

These are single-resident homes from one research group's instrumentation, so
they are not 22 independent studies. But they are 22 real homes, and they do not
support the simulator's numbers.

## The single most important limitation

**Every quantitative result in this repository comes from a simulator.**

No component has been validated against real sensor data from a real home.
The simulator was written by the same project as the inference code. Although
its generative process is deliberately different -- a stochastic daily
schedule rather than a Markov chain -- it still encodes this project's beliefs
about how people and sensors behave.

Consequences:

* The reported accuracies, calibration errors and ablation differences
  describe behaviour *on this simulator*. They are not estimates of field
  performance and must not be quoted as such.
* The ablation finding that a wearable substitutes for four ambient sensors
  holds under the simulator's assumed activity levels and sensor rates.
  A different household layout, a different resident, or a different sensor
  product could change it.
* Any real deployment must re-derive its emission parameters from its own
  data. The defaults in `defaults` are documented
  starting points, not fitted values.

Until the pipeline has been run against a public annotated smart-home dataset,
its numbers should be read as evidence that the *framework* behaves sensibly,
not as evidence about behavioural sensing in general.

## Statistical and methodological limitations

Shared evidence between layers
:   The occupancy layer and the state layer both consume the radar and beacon
    signals. They answer different questions -- who is present, versus what
    the resident is doing -- but their errors are consequently **not
    independent**. A radar fault degrades attribution and state inference
    together, and the pipeline's uncertainty estimates do not model that
    correlation.

Correlated samples treated as independent
:   Presence samples carry an explicit `sample_weight` discount because
    successive readings mostly re-observe an unchanged situation. The discount
    is a deliberate, inspectable correction, **not** a claim that the samples
    are independent, and its value is chosen rather than estimated.

    The same issue applies to the Gaussian emission for wearable data, which
    sums independent log-likelihoods across a fast-sampling stream and
    compensates with a fixed `weight` of 0.25.

Filtering approximation in daily aggregation
:   Daily features attribute each estimate's posterior to the interval
    *preceding* it. This is the standard filtering approximation and it lags
    slightly at transitions. A fixed-interval smoother would be more accurate
    and is not implemented.

Seeds are not independent replicates of reality
:   Repeated seeds vary the simulator's noise, not its structure. Confidence
    intervals across seeds quantify Monte-Carlo variability under one model;
    they do not quantify uncertainty about whether that model is right.

Effect sizes can be inflated by pairing
:   Paired comparisons remove between-household variance, which is correct,
    but it makes standardised effect sizes (*dz*) large in a way that would
    not survive an unpaired field study. Report the raw mean difference
    alongside.

Calibration is measured, not enforced
:   Nothing in the pipeline post-hoc calibrates its probabilities. The
    reported expected calibration error is a diagnostic; no temperature
    scaling or isotonic correction is applied.

## Modelling limitations

Single monitored resident
:   The state ontology models one person. Occupancy estimation recognises that
    others are present and discounts ambient evidence accordingly, but the
    platform cannot track two residents' states simultaneously. A genuinely
    two-resident household is out of scope.

Attribution is a weight, not an identity
:   `P(activity was the resident's)` is a marginal probability applied
    uniformly to all ambient sensors at a given moment. It cannot say that
    *this particular* kitchen event was the carer's while *that one* was the
    resident's. Per-event attribution would need evidence the platform
    deliberately does not collect.

Visitor recall is moderate
:   On the 90-day demonstration the occupancy layer reaches precision 0.71 and
    recall 0.48 for visitor presence, with a calibration error of 0.012. Around
    half of visit time is still missed, mostly short visits. The model is
    conservative by construction -- the correlation discount on presence
    samples and a prior favouring living alone both pull against declaring a
    visitor -- so it under-detects rather than over-detects. That is the safer
    direction for attribution, since a false "visitor present" would wrongly
    discount the resident's own activity, but it means visitor contamination
    is only partially removed.

Fixed, hand-specified parameters
:   Dwell times, jump structure, emission rates, occupancy priors and alert
    thresholds are all declared rather than learned. No expectation-maximisation
    or Bayesian parameter estimation is implemented for the fusion layer. This
    is a deliberate interpretability trade-off, but it means the model is only
    as good as its declarations.

Drift detection is noisy
:   The trend detector fires on a minority of stable periods. On the 90-day
    demonstration it produced three unmatched behavioural alerts -- roughly
    0.033 per person-day, about one per month -- alongside correctly detecting
    the injected change with a six-day delay. That burden is low but not zero,
    and the threshold trades directly against detection delay. Whether one
    spurious alert per month is acceptable is a question for the people who
    would receive them, not one the metrics can settle.

Sensor drift versus environmental change
:   The health monitor cannot distinguish a drifting temperature sensor from a
    genuinely warming room without redundant sensing. It reports the shift and
    leaves the judgement to the analyst.

Ontology states are not clinical states
:   `kitchen_activity` is not eating. `bathroom_activity` is not
    toileting. `sleeping` is bed occupancy with sustained low movement, not
    polysomnographically defined sleep. Any mapping from these states to
    clinical concepts is an additional inferential step this platform does not
    take.

## Engineering limitations

Late records beyond tolerance are dropped
:   The pipeline reorders within `lateness_tolerance` and counts anything
    later in `pipeline.too_late`. Those records are discarded rather than
    triggering a replay. A deployment with long uplink outages needs a replay
    strategy the pipeline does not provide.

No incremental smoothing or backfill
:   Once a step is emitted it is never revised. Evidence that arrives late
    cannot correct an earlier conclusion.

Snapshots are not versioned
:   `snapshot` output is checked against the current state ontology and
    context set, but there is no schema version field. A future change to the
    state set will invalidate stored snapshots with a clear error rather than
    a migration.

Performance is adequate, not optimised
:   The pipeline processes roughly 530,000 observations over 90 simulated days
    in a few minutes on a laptop. Nothing has been profiled or vectorised for
    scale, and the ablation sweep is serial.

Pre-existing type-checking debt
:   The ten packages added for ambient sensing pass `mypy` under the
    repository's strict settings. The older modules do not: 164 pre-existing
    errors remain across 31 files, and CI gates `mypy` on only two files.

## Scope and safety

**This is a research toolkit. It is not a medical device.**

* Nothing it produces is a diagnosis. Alert text is deliberately phrased as an
  observation about sensor-derived behaviour.
* No claim of clinical effectiveness is made or supported anywhere in this
  repository.
* The platform has not been evaluated for safety, and it should not be used as
  the sole means of detecting a person coming to harm.
* Its outputs are appropriate for research, method development and
  hypothesis generation. They are not appropriate for unsupervised clinical
  decision-making.

## Privacy posture

The design deliberately excludes cameras, microphones, facial recognition,
voice recognition and any other biometric identification. Attribution is
achieved from anonymous evidence and is expressed as a probability.

This is a property of the *design*, not a guarantee about a deployment. A
deployment that adds a camera and feeds derived features through the
`Observation` interface would defeat it.
The privacy posture depends on what a deployment chooses to install.

## What would make this scientifically trustworthy

In rough priority order:

1. **Validation on a public annotated dataset** (for example CASAS, ARAS or
   MARBLE), reporting the same metrics against real annotations. Until this
   exists, every number here is conditional on the simulator.
2. **Parameter estimation from data** rather than declaration, with the fitted
   values compared against the hand-specified defaults.
3. **A second, independently written simulator** with different structural
   assumptions, to test how much of the performance depends on this one.
4. **Calibration assessment across households**, not only within one, since a
   model calibrated on average may be badly calibrated per person.
5. **A prospective evaluation of alert burden** with people who would act on
   the alerts, since false-positive tolerance is a human judgement the metrics
   cannot supply.
6. **Explicit modelling of the dependency** between the occupancy and state
   layers, so that shared-evidence correlation is reflected in the reported
   uncertainty.
