# Known Limitations

This page states what the ambient-sensing pipeline does **not** establish. It
is deliberately blunt. A monitoring system whose limitations are not written
down will have them discovered by whoever trusts it first.

## Measured on real data

Twenty-two real CASAS homes have been scored with nothing refitted and nothing
discarded, over a median 90% of each recording.

| | Simulator | Real homes (median) |
| --- | --- | --- |
| Balanced accuracy | 0.816 | **0.420** |
| Calibration error | 0.084 | **0.314** |

No home exceeded 0.514. Sleeping (0.74), general activity (0.58) and cooking
(0.57) hold up. **`home_inactive` is the genuine failure at 0.16** — a resident
sitting still is the state the pipeline is worst at recognising.

Declared event rates are measurably wrong: real in-room sensors fire at 299/h
during bathroom activity and 580/h during cooking against a declared 40/h. That
is a plausible mechanism, since `HOME_INACTIVE` is declared to emit roughly 10
activations an hour and a still resident produces far fewer. Substituting
measured rates doubles to triples `home_inactive` recall but costs sleeping
recall and calibration, so it points the work somewhere specific without
resolving it.

**Abstention cannot be repaired by raising its threshold.** It fired on 2.2% of
steps while the model was wrong more often than right, and the obvious fix —
thresholds tuned for a simulator where the model is right 82% of the time — does
not work. Over 60,948 scored steps, stated confidence separates right from wrong
by only 0.073, and the relationship inverts where it matters: the 0.95-1.00
band, covering 39% of all steps, is *less* accurate (0.561) than the 0.85-0.95
band (0.653). Raising the threshold discards the pipeline's best band and keeps
its saturated one. The likely cause is that during quiet periods the belief
approaches certainty because no evidence arrived, not because the evidence was
strong. This is a safety limitation, not a tuning parameter, and the v0.3
candidate does not touch it.

An earlier revision of this page reported `away` recall of 0.00 and treated it
as a finding. **That was wrong.** `Leave_Home` annotates twelve seconds of
crossing the threshold, not the hours spent out, so the truth series labelled
motion inside the house as absence. With away derived from the gap between
departure and return, recall is 0.36 and coverage rises from 64% to 90%. Three
further explanations for the remaining gap were tested and rejected: an
incomplete location map, absent presence-confirming sensors, and occupancy-state
modelling.

A supervised classifier given the same per-room event counts, three lagged
steps and time of day reaches **0.607** balanced accuracy on held-out homes,
against a 0.143 majority-class baseline. Two things follow. The seven-state
ontology *is* recoverable from motion and door sensors, so the gap is a
deficiency in the inference rather than a limit of the deployment. And the
simulator's 0.816 sits **above** the ceiling measured on real homes, so its
figures exceed what this instrumentation supports at all rather than merely
being optimistic.

The pipeline recovers about two thirds of what is available, losing most ground
on `bathroom_activity` (0.25 against 0.80), `away` (0.36 against 0.82) and
`home_inactive` (0.16 against 0.57), while beating the classifier on
`home_active`.

Ablating the classifier's features locates the shortfall precisely. Given only
instantaneous event counts the ceiling is 0.397, and the pipeline scores 0.420:
**on the evidence it uses, it is already at the ceiling.** The missing accuracy
is in two things it does not have — an explicit time-of-day term, worth about
+0.105, and several steps of recent room-resolved counts, worth about +0.140.
The continuous-time Markov prior models how long a state lasts but not when in
the day it is plausible, so a resident motionless at 02:00 and at 14:00 look
alike to it. A circadian prior recovers about a tenth of that.

The three components added in response — fitted emission rates, the circadian
prior and smoothing — **do not combine.** All three together score 0.435
balanced accuracy against 0.463 for smoothing alone, with calibration worse than
the baseline. Fitted rates trade accuracy for calibration and smoothing trades
calibration for accuracy, so stacking them gives up both. See
[Real-data validation](real_data.md) for which to use when.

Two further attempts failed. Declared dwell times are 3 to 9 times longer than
real state durations, but fitting them to measurement lowered balanced accuracy
from 0.449 to 0.429: the long dwells are doing useful work as regularisation,
and being empirically accurate is not the same as being a useful prior. And the
history term the ablation valued at +0.140 is not straightforwardly available to
a recursive filter, which already carries history in its belief and would
double-count evidence if given lagged observations as well.

Two of the 22, `hh107` and `hh121`, are two-occupant recordings by CASAS
metadata, and every figure here was computed before that was noticed. Excluding
them changes the medians by less than a thousandth, but the ontology models one
resident, and `hh107`'s anomalous behaviour was a clue that went unexamined.

The remaining homes are from one research group's instrumentation, so they are
not 22 independent studies, and 0.607 is a ceiling for this
instrumentation rather than for ambient sensing generally. The gap between 0.420
and what is recoverable is real nonetheless.

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
