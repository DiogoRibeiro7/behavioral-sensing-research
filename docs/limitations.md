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

**Abstention cannot be repaired by raising its threshold.** Across the 22 homes
it fired on a median 2.5% of steps, and on at most 3.9% in any home, while the
model was wrong more often than right, and the obvious fix — thresholds tuned
for a simulator where the model is right 82% of the time — does not work. Over
60,948 scored steps from five homes, stated confidence separates right from wrong
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

**That conclusion did not survive a matched comparison.** The Phase 1 run used
the 20 single-resident development homes and frozen folds. With current windows
only, the diagnostic and the production filter are level, −0.008 [−0.031,
+0.019]. But the filter also conditions on every earlier window. Given strictly
less than that, the current window and three previous ones, the diagnostic
scores 0.081 [0.052, 0.111] higher balanced accuracy, in 19 of 20 homes. Part of
the gap is the formulation, not only the missing information; see
[the recoverable-information gap](PHASE1_RECOVERABLE_GAP.md). Of the two missing
pieces, time of day has since been given to the generative model as a
hierarchical prior, worth +0.131 balanced accuracy on the development homes. An
explicit recent-history state lowered balanced accuracy by 0.042 on identical
information. See [the time prior](PHASE3_TIME_PRIOR.md) and
[the history state](PHASE3_HISTORY_STATE.md).

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

### On a clinical cohort

The online pipeline was run on the 56 homes of the TIHM dataset of people
living with dementia, with its declared emissions, baseline and alert policy
and the ten-minute step its protocol declares. The run is exploratory, and the
dataset's labels are alerts a clinical team verified, so no state is scored.
See [the results](TIHM_ALERT_BURDEN_RESULTS.md).

**Its alerts are not more frequent on the days the clinical team confirmed.**
Alert days are 5.3% of agitation label days and 8.5% of other evaluable days;
the interval of the difference includes zero. At the same number of flagged
days, the household's own label history catches 63 of 94 label days and its
sensor event count against earlier days 53, where the pipeline's deviating
days catch 18 and chance expects 23.

**A home that reports nothing is read as a home asleep.** On 128 of 2,850
monitored days no sensor reported an event. Every one passed the pipeline's
test of a usable day, and the filter inferred a median 23.8 hours of sleep on
them. On 16 June 2019 all 47 monitored homes are silent, and the next day
holds 33 of the run's 183 behavioural alerts. The dataset paper reports a
failure of the data collection server and dates the drop to 14 June; in the
released file the drop is on 15 to 17 June, so treating them as one event is
an inference. The dataset holds activations only, so nothing in it says a
sensor or its gateway was working, and the health layer raised no alert in the
whole run. A likely reading, not a tested one, is that this is the mechanism
behind the saturated confidence above: with no evidence the belief goes to the
quietest state, and the pipeline abstained on 0.06% of usable time.

**The pipeline's hours of sleep do not follow a sleep mat in the TIHM homes.** In
the 14 TIHM homes with enough days on a mat under the mattress, the mean
within-home Spearman correlation between the pipeline's daily
`sleeping_hours` and the mat's hours asleep is 0.11 [0.03, 0.19], on the days
the sensors reported throughout, and the pipeline gives 3.95 hours a day more
sleep than the mat. On simulated homes, against the simulator's true hours of
sleep, the same correlation is 0.70. Every alerting study here counts its
detections on that feature, so a detection in simulation is a statement about
the simulator's sleep, not a person's. The mat's stages are the device's own
and are not validated; see [the sleep-mat results](SLEEP_MAT_RESULTS.md).

**An opt-in baseline reference now keeps its threshold's meaning, and where
its thresholds belong on a real home is not known.** On synthetic Gaussian
days the default reference passes its deviation threshold of 3 on 6.24% of
days, where 0.27% is stated, because a weekday reference of a few days has
too small a spread; the calibrated reference, `BaselineConfig.calibrated`,
passes it on 0.28%. On 400 paired simulated homes it keeps that meaning for
hours of sleep, the feature the criterion was stated for; on hours away it
passes 3 on 1.66% of days, as often as a Gaussian value passes 2.40. At the
default's false alerts it finds more of a step change, +0.09 of homes
net of chance [+0.03, +0.16]; see
[the threshold-calibration results](THRESHOLD_CALIBRATION_RESULTS.md). That
evidence is simulated, and it shows no gain for a smaller or a slower change.
At the declared thresholds the reference reports far less of everything, so
its thresholds must be set lower, and the simulated homes put them at about
0.59 times the declared ones. On TIHM, described and not tested, it still
passes 3 on 8.1% of evaluable days of sleep with the silent-home rule off,
and on 1.7% with it on at 12 hours: real days are far from the independent
Gaussian draws its score assumes. It is off by default.

**An opt-in rule now treats a silent home as not observed, and it has a cost
the simulator cannot show.** `HealthConfig.home_silence_horizon` is off by
default, so by default a silent home is still read as above. On 100 paired
simulated homes the rule removes, on average, the alerts an outage raises, and
changes nothing in a stable home; see
[the silent-home results](SILENT_HOME_RESULTS.md). That evidence is simulated,
and a simulated home is not silent for twelve hours unless a fault makes it
so. The TIHM homes often are. Run on them as a description and not as a test,
at twelve hours the rule refuses 322 of the 2,850 monitored days in 48 of the
56 homes and raises 324 alerts about silence, 145 of them in the two stretches
in which most monitored homes were silent together. Of the 183 behavioural
alerts of the run without it, 147 are no longer raised and 13 new ones are. It
does not say why a home is silent: a resident who is away, a resident who
needs help and a gateway that has stopped are reported in the same words. Four
things remain open:

- which horizon a real home needs, which depends on its sensors and its
  resident and was declared here, not estimated;
- what a person receiving an alert about silence does with it, and whether
  one about a home is worth more than one about a behaviour;
- whether the rule costs detection of a real change after an outage. In
  simulation 76 of 100 homes met the detection definition with the rule on
  against 91 with it off, −0.15 [−0.23, −0.07], Monte Carlo standard error
  0.04: fewer, and inconclusive against the −0.10 margin. Whether changes
  were missed or alerts the outage itself raised were gone was not tested;
- whether the baseline is less sensitive for a while after the days the rule
  refuses. In simulation the outage arm raised fewer alerts in its window
  than the same homes with nothing injected, 30 against 58, an excess of
  −0.28 [−0.44, −0.13] with a standard error of 0.08, and no arm of the
  protocol can say why.

The TIHM dataset is by Palermo et al., *Scientific Data* 10, 606 (2023),
under CC BY 4.0. Surrey and Borders Partnership NHS Foundation Trust and Howz
are acknowledged, as the dataset asks. It is not redistributed here.

**The deviation threshold does not mean three standard deviations.** The
baseline compares a day with the same weekday's earlier days once four exist.
A median and MAD of four values put a Gaussian value past the threshold about 17.5%
of the time, where a known mean and standard deviation give 0.27%. On
stationary Gaussian values with each household's own centre and spread, 14.9%
of days deviate, and the run has 24.4%. Most of what the threshold flags is
therefore present with no change over time at all.

**Two of the four baseline features are close to inert.** The filter infers a
median 0.06 hours of kitchen activity and 0.00 hours of bathroom activity a
day, so the reference sits at the scale floor for 87% of kitchen feature-days
and for every bathroom one. Why the declared emissions read so little from
these homes has not been examined. What the room sensors are is declared, not
known: the activity table names a location and no sensor type, and the dataset
paper is not of one voice. Its methods place passive infrared sensors in the
hallway and living room and movement sensors on the kitchen, bedroom and
bathroom doors; its first figure's caption says passive infrared and door
sensors are included in each room. The frozen mapping takes the five room
locations as motion in that room, with the sensor type declared approximate.

Each of the last three was found after the run had been read. They are
diagnoses, kept apart from the protocol's results, and no default has been
changed because of them.

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

In rough priority order, with what has been done since this list was written:

1. **Validation on public annotated datasets**, reporting the same metrics
   against real annotations.
   - **CASAS.** Done: a 22-home development panel and a frozen 43-home external
     cohort; see [Real-data validation](real_data.md).
   - **An independent dataset.** The two homes of UCI ADL Binary were evaluated
     under a frozen protocol, and the CASAS-trained model does not transfer;
     see [Phase 5 external results](PHASE5_EXTERNAL_RESULTS.md).
   - **Still needed.** Independent datasets with more homes.
2. **Parameter estimation from data** rather than declaration.
   - **Done for the evidence channels.** Fitted channel models improve
     calibration on the development homes; see
     [fitted rates](PHASE3_FITTED_RATES.md).
   - **Partial pooling.** Household parameters with partial pooling are
     evaluated; see [partial pooling](PHASE3_PARTIAL_POOLING.md).
   - **Transfer.** The fitted parameters did not transfer to the external
     homes.
3. **A second, independently written simulator** with different structural
   assumptions, to test how much of the performance depends on this one. Not
   done.
4. **Calibration assessment across households**, not only within one, since a
   model calibrated on average may be badly calibrated per person.
   - **Per household.** The matched evaluations since Phase 1 report
     calibration error per household.
   - **Across datasets.** The calibration gap widens: zero-shot confidence
     exceeds accuracy by 0.41 and 0.45 in the external homes, against 0.30 on
     the development homes.
5. **A prospective evaluation of alert burden** with people who would act on
   the alerts, since false-positive tolerance is a human judgement the metrics
   cannot supply. Not done.
6. **Explicit modelling of the dependency** between the occupancy and state
   layers, so that shared-evidence correlation is reflected in the reported
   uncertainty. Not done.
