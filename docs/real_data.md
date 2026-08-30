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

## Status

The adapter is implemented and tested end to end, including an integration test
that runs a CASAS-format recording through the unmodified pipeline. **That
fixture is synthesised by this repository in CASAS format.** It establishes that
the plumbing is correct and that the declared emission defaults separate states
without being refitted. It says nothing about performance on a real apartment.

Running this against downloaded CASAS recordings, and reporting what happens, is
the outstanding work — and the answer is genuinely unknown. Nothing in the
package is tuned for it: no emission rate, dwell time or threshold has been
refitted. Whatever comes out is a lower bound on the approach with fitted
parameters, and an honest measure of how far the defaults transfer.
