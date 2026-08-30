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

## First result on a real recording

`hh103`, 57 days, one resident, seven locations. Nothing was tuned: no emission
rate, dwell time or threshold was refitted. 65.5% of the span carried a mapped
annotation and only that part was scored.

| Metric | Simulator | **hh103** |
| --- | --- | --- |
| Balanced accuracy | 0.816 | **0.349** |
| Calibration error | 0.084 | **0.299** |
| Abstention rate | — | 0.021 |

| State | Recall |
| --- | --- |
| kitchen_activity | 0.74 |
| sleeping | 0.72 |
| bed_awake | 0.67 |
| bathroom_activity | 0.19 |
| home_active | 0.08 |
| away | 0.02 |
| home_inactive | 0.02 |

**The architecture does not transfer on declared defaults.** Balanced accuracy
falls by more than half, and calibration error more than triples. This is one
home in one dataset, so it is a data point rather than a verdict, but it is a
real one and it points the same way the design documents feared.

### What fails, and why it is the interesting part

The split is not random. States with a distinctive room-and-rate signature
survive: cooking, sleeping and being awake in bed all score around 0.7. States
that require knowing the resident is *present but still* collapse.

The confusion is specific. When the resident was genuinely `home_inactive`, the
pipeline said `away` 53% of the time. When they were genuinely `away`, it said
`home_active` essentially always.

This home has seven motion and door streams and nothing else. The simulator's
deployment includes a bed pressure sensor, a wearable and a resident beacon —
precisely the modalities that distinguish a quiet resident from an empty house.
Strip them out and motion silence becomes ambiguous between "sitting still" and
"gone out", and the model resolves that ambiguity badly.

That bears directly on the project's central question. Modality does substitute
for sensor count, as the ablation found — but the substitution has a floor, and
the presence-confirming modality appears to be the one that cannot be dropped.
Reading motion silence as absence is exactly the failure the design set out to
avoid, and on real data with a reduced sensor suite it happens anyway.

### The calibration result is the one to worry about

Calibration error of 0.299 with an abstention rate of 0.021 means the pipeline
was **confidently wrong**: mostly incorrect, and almost never willing to say it
did not know. The abstention mechanism is presented throughout this project as
the safety valve that makes the rest defensible. On this recording it did not
open.

A monitoring system that reports "resident is out" while they sit quietly in a
chair, with high stated confidence, is worse than one that reports nothing.

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

Results move with step size — balanced accuracy is 0.349 at five minutes, 0.252
at ten, 0.248 at thirty — so quote the step alongside any number from this.

## Status

The adapters are implemented and tested, and one real recording has been
scored. What has **not** been done: more than one home, any other dataset, any
fitting of parameters to real data, and any paired comparison of the kind the
simulator experiments use. The single result above should not be generalised
into a claim about the architecture, in either direction.

The obvious next steps are refitting emissions from a subset of homes and
scoring on held-out ones, and checking whether the `home_inactive`/`away`
confusion closes when a presence-confirming sensor is present in the
deployment.
