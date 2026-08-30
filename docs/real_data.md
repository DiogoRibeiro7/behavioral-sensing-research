# Real-data validation

Every quantitative result elsewhere in this documentation comes from the
simulator bundled with this repository. The simulator and the inference model
are deliberately different in structure, and guard tests check that ground truth
does not leak between them, so those results are not circular. They still only
establish properties of the inference model, not of any real home.

`sensor_modeling.datasets` exists to close that gap: it brings published,
annotated recordings into the same canonical observation model, so the same
pipeline can be run over data this project did not generate.

## What is implemented

An adapter for [CASAS](https://casas.wsu.edu/datasets/) recordings, and a runner
that scores the unmodified pipeline against their annotations.

```python
from datetime import timedelta
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import read_casas, evaluate_recording

recording = read_casas(
    "aruba/data",
    timezone=ZoneInfo("America/Los_Angeles"),
    rooms={"M003": "kitchen", "M004": "bedroom"},
)
print(recording.summary())

result = evaluate_recording(recording, step=timedelta(minutes=10))
print(result.to_dict())
```

Recordings are not redistributed with this package. Download them from CASAS
under the terms stated there.

## What the adapter refuses to do

Three conveniences that would each produce a better-looking number and a worse
result.

**It will not guess a timezone.** CASAS timestamps are naive local wall-clock.
Reading them as UTC displaces every event by hours, which is invisible in
aggregate counts and fatal to anything involving daily rhythm. `timezone` is a
required argument.

**It will not label unlabelled time.** Much of a recording carries no
annotation. `truth_series` returns `None` there and the metrics skip those
positions. Filling gaps with a plausible state would manufacture agreement.

**It will not force sensors or activities into the ontology.** A light switch or
power meter has no unambiguous canonical modality, so it is excluded and named
in `unmapped_sensors` rather than admitted as `OTHER`, where it would push the
posterior around under a modality nobody reasoned about. Activity labels with no
counterpart are reported in `unmapped_activities` rather than collapsed into
`UNKNOWN`, which is a claim about abstention and not a place to put vocabulary
gaps.

## Choices that affect the result

**Rooms must be supplied.** Raw CASAS files carry no machine-readable
sensor-to-room map, and the occupancy and fusion layers key on rooms. Inferring
one from sensor numbering would be fabrication, so `rooms` is optional and unset
by default — and the pipeline will do poorly without it. Build the map from the
apartment's floor plan in the dataset documentation.

**Only activations become observations.** Motion and door sensors report
ON/OFF and OPEN/CLOSE pairs. The pipeline models these as event-kind sensors
whose rate carries the information, so admitting both halves would double every
count. The discarded readings are counted in `deactivations`. Treating them as
state-kind evidence instead is a legitimate alternative this adapter does not
take.

**`Eating` maps to `HOME_ACTIVE`, not `KITCHEN_ACTIVITY`.** The ontology's
kitchen state means being active in the kitchen, and meals are often eaten
elsewhere. Mapping it to the kitchen would manufacture agreement with an
inference that keys on kitchen sensors. Pass `activity_states` to decide
differently; the point is that the decision is visible.

## Read coverage before reading accuracy

`DatasetEvaluation` reports `scored`, `scored_fraction` and `labelled_fraction`
alongside the metrics, deliberately in the same structure. A balanced accuracy
computed over a tenth of a recording is a different claim from one computed over
most of it, and separating the two invites a reader to forget the difference.

Scoring is refused outright when nothing carried a mapped label, because a
metric over an empty sample still prints as a number.

## Two export formats

CASAS publishes recordings in more than one shape, and a reader written for one
does not read the other.

| | Classic (`read_casas`) | `hh` CSV (`read_casas_hh`) |
| --- | --- | --- |
| Separator | whitespace | comma, date and time split |
| Third field | sensor id, `M004` | **location**, `Bedroom` |
| Marker | `Sleeping begin` | `Sleep="begin"` |
| Vocabulary | `Sleeping`, `Meal_Preparation` | `Sleep`, `Cook_Dinner` |

Only six labels are common to both vocabularies, and none of the frequent ones
are. The archives currently on Zenodo use the `hh` form, so that is the reader
to start from.

The `hh` export also carries the room in the data, which removes the need to
reconstruct one from a floor plan — at the cost of having already aggregated
each room's sensors into a single stream, so within-room redundancy is not
observable.

## Results on 22 real homes

Every CASAS `hh` recording under 12 MB was scored: 22 homes, one resident each,
motion and door sensors, **nothing refitted**. Only annotated time was scored.
No location and no activity label goes unmapped, so nothing is silently
discarded.

| | Simulator | Real homes (median) | Real homes (range) |
| --- | --- | --- | --- |
| Balanced accuracy | 0.816 | **0.420** | 0.329 – 0.514 |
| Calibration error | 0.084 | **0.314** | 0.221 – 0.921 |
| Abstention rate | — | 0.025 | max 0.039 |
| Labelled coverage | — | 89.9% | — |

No home reaches 0.52. **The declared defaults do not transfer**, and the result
is consistent across 22 independent homes.

| State | Median recall | Range | Homes |
| --- | --- | --- | --- |
| sleeping | 0.74 | 0.39 – 0.87 | 22 |
| home_active | 0.58 | 0.10 – 0.82 | 22 |
| kitchen_activity | 0.57 | 0.00 – 0.76 | 22 |
| away | 0.36 | 0.03 – 0.62 | 22 |
| bed_awake | 0.25 | 0.06 – 0.67 | 5 |
| bathroom_activity | 0.25 | 0.00 – 0.57 | 22 |
| home_inactive | 0.16 | 0.00 – 0.39 | 22 |

### Correction: an earlier version of this page was wrong about `away`

A previous revision reported `away` recall of **0.00 in the median home** and
built an explanation on it. That was an artefact of this adapter, not a property
of the pipeline.

`Leave_Home` and `Enter_Home` annotate the *act of crossing the threshold*, with
a median duration of about **twelve seconds**. Mapping `Leave_Home` to `AWAY`
labelled a burst of motion inside the house as absence, so the inference was
scored wrong for correctly reporting activity. Meanwhile the hours actually
spent out carried no label and were never scored at all.

The measurement that exposed it: intervals labelled `AWAY` showed **130
activations per hour** across the deployment. Absence does not produce 130
events an hour.

`read_casas_hh` now derives the away period from the gap between a departure and
the next arrival, which is the only annotation of absence the dataset supports.
With that fix `away` recall is 0.36 and labelled coverage rises from 64% to 90%,
because the time spent out is now scored rather than skipped. Median balanced
accuracy rises from 0.364 to 0.420.

The lesson generalises beyond this dataset. An annotation vocabulary can look
like it names states when it actually names events, and the failure is silent:
every number downstream is computed correctly from a truth series that means
something other than what it claims.

### What is left after the correction

**The gap is real.** 0.420 against 0.816 is roughly half, across 22 homes, with
none above 0.514.

**`home_inactive` is the genuine failure**, at 0.16 median and unaffected by the
away correction. A resident sitting still is the state the pipeline is worst at
recognising, and that is not an artefact.

**The declared event rates are measurably wrong.** Using
`measure_event_rates`, pooled over six homes:

| State, sensor in-room | Declared | Observed median | Ratio |
| --- | --- | --- | --- |
| kitchen_activity | 40/h | 580 | 14× |
| bathroom_activity | 40/h | 299 | 7.5× |
| sleeping | 0.8/h | 5.2 | 6.5× |
| any sensor, elsewhere | 0.15/h | 0.0 | — |

That is a plausible mechanism for the `home_inactive` failure. `HOME_INACTIVE`
is declared at activity level 0.25, implying roughly 10 activations an hour from
the in-room sensor, and a resident sitting still produces far fewer.

### Fitting the rates on held-out homes

The rates were fitted on 11 homes with `fit_emission_defaults` and scored on the
other 11, which never contributed to the fit. Nothing but the rate constants
changed: ontology, fusion, occupancy and alerting are untouched.

| | Declared | Fitted | |
| --- | --- | --- | --- |
| Balanced accuracy | 0.449 | 0.418 | slightly worse |
| Calibration error | 0.312 | **0.202** | **35% better** |

**Fitting the rates fixes the overconfidence and not the accuracy.** That is the
cleanest result available here, and it splits the failure in two.

The declared rates were making the pipeline confidently wrong; measured rates
make it appropriately uncertain, cutting calibration error by a third on homes
that had no part in the fit. But discrimination does not move. Whatever limits
balanced accuracy to roughly 0.42 is **not** the emission constants, and no
amount of rate-fitting will reach the simulator's 0.816.

So the two failures have different causes, and only one of them is now
explained. Candidates for the accuracy gap that remain untested: the dwell-time
priors, the occupancy layer's assumptions, and the possibility that a
seven-state ontology is simply not identifiable from motion and door sensors
alone.

An earlier probe using seven constants eyeballed from six homes, with no
held-out set, appeared to improve `home_inactive` recall while degrading
everything else. That was parameter-fiddling and its result should be
disregarded in favour of the held-out numbers above.

### How much is recoverable at all

The remaining question was whether roughly 0.42 is a poor result or close to
what motion and door sensors support. That is measurable: fit a gradient-boosted
classifier on the same information the pipeline receives — per-room event counts
for the current step, three lagged steps, and time of day — training on 11 homes
and scoring on the 11 held out.

This is a diagnostic bound, not a proposed model, and nothing about it goes near
the inference path.

| | Balanced accuracy |
| --- | --- |
| Majority-class baseline | 0.143 |
| **Pipeline, declared defaults** | **0.420** |
| **Supervised ceiling** | **0.607** |
| Simulator | 0.816 |

**The ontology is identifiable from these sensors.** A classifier reaches 0.607,
far above chance, so the states are genuinely separable from motion and door
data. The worry that a seven-state ontology might be unrecoverable in principle
from this instrumentation is answered, and answered favourably.

**The pipeline recovers about two thirds of what is available.** 0.420 against a
0.607 ceiling. The accuracy gap is therefore a deficiency in the inference, not
a limit of the deployment, and it is worth closing.

Per state, where the pipeline loses ground:

| State | Pipeline | Ceiling |
| --- | --- | --- |
| bathroom_activity | 0.25 | 0.80 |
| away | 0.36 | 0.82 |
| home_inactive | 0.16 | 0.57 |
| kitchen_activity | 0.57 | 0.82 |
| sleeping | 0.74 | 0.87 |
| home_active | **0.58** | 0.36 |

The losses are concentrated in `bathroom_activity`, `away` and `home_inactive`.
Notably the pipeline is *better* than the classifier at `home_active`, so the
two have different error profiles rather than one dominating throughout.

**The simulator is easier than reality even for an optimal method.** Its 0.816
sits above the 0.607 ceiling measured on real homes. That is a fact about the
simulator, not about the pipeline: results from it are not merely optimistic in
degree, they exceed what the real instrumentation supports at all. Any figure
quoted from the simulator should be read with that in mind.

Two caveats that matter. The classifier sees labels and the pipeline does not,
so this bounds the information present rather than what an unsupervised filter
ought to find. And it is the same 22 homes from one research group, so 0.607 is
a ceiling for this instrumentation, not for ambient sensing generally.

### Where the missing accuracy actually lives

Ablating the classifier's features says which information carries the signal,
and therefore what the pipeline is not using.

| Features given to the classifier | Balanced accuracy |
| --- | --- |
| Event counts, lags and time of day | 0.607 |
| Without time of day | 0.537 |
| Counts and time of day, no lags | 0.502 |
| **Event counts only** | **0.397** |
| Time of day only, no sensors | 0.262 |

**The pipeline is already at the ceiling for the evidence it uses.** Given only
instantaneous per-room event counts, the best achievable is 0.397. The pipeline
scores 0.420. It is not squandering the evidence in front of it — on that
evidence it is slightly *ahead* of a supervised classifier, which is a credit to
the fusion layer rather than a criticism of it.

The missing 0.19 is not hidden in the counts. It is in two things the pipeline
does not have:

- **Time of day is worth about +0.105** over counts alone. The pipeline has no
  circadian term at all. Its continuous-time Markov prior models how long a
  state persists, not when in the day that state is plausible, so it cannot
  distinguish someone motionless at 02:00 from someone motionless at 14:00.
- **Recent history is worth about +0.140** over counts alone. The pipeline is
  recursive and carries a posterior forward, but that is not the same as having
  the last few steps of raw room-resolved counts available as evidence.

Together they account for +0.210, which is the whole of the gap and slightly
more, so the two overlap.

Time of day alone, with no sensor information whatsoever, reaches 0.262 against
a 0.143 baseline. Daily rhythm is genuinely informative and the pipeline
currently ignores it, but it is not a shortcut: a clock alone lands far below
the 0.420 the pipeline achieves, so the sensors are doing most of the work.

**This turns the gap into two specific, principled changes.** A time-inhomogeneous
generator gives the CTMC a circadian prior, and a longer evidence window gives
the emission model recent history. Both are interpretable, both are
probabilistic, and neither requires anything the project's constraints exclude.
### A circadian prior, and what it actually bought

The ablation said time of day is worth about +0.105. `StateOntology` now takes
an optional `circadian` profile — 24 stickiness multipliers per state — which
divides that state's exit rates by its multiplier for the current hour, making
a state the resident usually occupies at this time correspondingly harder to
leave.

Profiles fitted on 11 homes as a lift ratio, scored on the 11 held out:

| | Plain | Circadian |
| --- | --- | --- |
| Balanced accuracy | 0.449 | **0.460** |
| Calibration error | 0.312 | **0.296** |

Better in 10 of the 11 homes, and better calibrated in 9.

**It delivers about a tenth of what the ablation suggested was there.** +0.011
against +0.105. That gap is the interesting part: modulating transition rates is
a much weaker lever than letting a model condition on the hour directly. The
prior only changes how reluctant the chain is to leave a state; it never enters
the evidence, so it cannot say "these counts look like 03:00 rather than 14:00".

So most of the circadian signal remains unexploited, and the transition-rate
route is not how to reach it. A time-varying term on the *emission* side, or a
time-conditioned prior over states rather than over dwell, would be the next
thing to measure. The feature is worth keeping — the improvement is real,
consistent and free when unused — but it should not be mistaken for having
closed that part of the gap.

### Explanations tested and rejected

- **An incomplete location map.** `DiningRoom` was unmapped in 14 homes.
  Completing it moved the median from 0.356 to 0.364.
- **Absent presence-confirming sensors.** Five homes carry a `LoungeChair`
  occupancy sensor; they are not better, with `home_inactive` recall of 0.10
  against 0.17 without.
- **Modelling motion as occupancy state.** Reading ON/OFF pairs as a persistent
  `PROXIMITY` state so that OFF asserts absence reaches 1.00 `away` recall by
  reporting `away` almost always, collapsing balanced accuracy to 0.16 – 0.23.
  A motion sensor's OFF means "no motion just now", not "nobody home", and the
  event-kind reading is the correct one.

### The calibration result is the one to worry about

Median calibration error 0.285 with median abstention 0.022 means the pipeline
was **confidently wrong**: incorrect more often than not, and almost never
willing to say it did not know. Abstention never exceeded 5.2% in any home.

The abstention mechanism is presented throughout this project as the safety
valve that makes the rest defensible. On real data it did not open.

A monitoring system that reports "resident is out" with high stated confidence
while they sit quietly in a chair is worse than one that reports nothing.

### A check that mattered

The first pass left `DiningRoom` unmapped in 14 of the 22 homes, along with
`Office`, `Hall` and four activity labels, so real evidence was being discarded.
That could have accounted for the poor scores. It did not: completing the maps
moved median balanced accuracy from 0.356 to 0.364. The result is robust to the
most obvious explanation that would have let the architecture off.

### Reproducing it

```python
from datetime import timedelta
from zoneinfo import ZoneInfo

from sensor_modeling.datasets import evaluate_recording, read_casas_hh

recording = read_casas_hh("labeled/hh103.csv", timezone=ZoneInfo("America/Los_Angeles"))
print(recording.summary())
print(evaluate_recording(recording, step=timedelta(minutes=5)).to_dict())
```

Data from <https://zenodo.org/records/15708568> (CC-BY-4.0), file
`labeled_data.zip`. Not redistributed here.

Results move with step size — on `hh103`, balanced accuracy is 0.349 at five
minutes, 0.252 at ten and 0.248 at thirty — so quote the step alongside any
number from this. All figures above use a five-minute step, the most favourable
of the three.

## Status

The adapters are implemented and tested, and 22 real homes have been scored.
What has **not** been done: any dataset other than CASAS, any fitting of
parameters to real data, and any paired comparison of the kind the simulator
experiments use. These are all single-resident homes from one research group's
instrumentation, so they are not independent of each other in the way 22
households from different studies would be.

The obvious next steps are refitting emissions from a subset of homes and
scoring on held-out ones, and checking whether the `home_inactive`/`away`
confusion closes when a presence-confirming sensor is present in the
deployment.
