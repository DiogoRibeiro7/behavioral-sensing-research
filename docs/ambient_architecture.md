# Ambient Sensing Architecture

This page describes the multimodal ambient-sensing pipeline added on top of the
toolkit's original modelling layers. It covers the system architecture, the
canonical sensor data model, and the online execution model.

## The pipeline

```text
heterogeneous sensor observations
        v
sensor_modeling.observations      validation, units, ordering, provenance
        v
sensor_modeling.health            per-sensor reliability
        v
sensor_modeling.context           who is present, and whose activity it was
        v
sensor_modeling.fusion            P(Z_t | O_1:t) over latent states
        v
sensor_modeling.baseline          adaptive personal normal, change verdicts
        v
sensor_modeling.alerts            restrained, explained alerts
        v
sensor_modeling.evaluation        metrics and paired ablation studies
```

`sensor_modeling.online` orchestrates the chain incrementally.
`sensor_modeling.simulation` supplies synthetic households with ground truth
for evaluating it.

## Five kinds of thing, kept distinct

The most important structural property of the platform is that these five are
never conflated:

| Kind | What it is | Where it lives |
| --- | --- | --- |
| Measured observation | A sensor reported a value at an instant. | `Observation` |
| Derived feature | A value an upstream device computed, such as a radar track count or presence probability. Carries a `confidence` below one. | `Observation` with reduced `confidence` |
| Inferred state | A posterior over what the resident was probably doing. | `StateEstimate` |
| Behavioural change | A shift against the resident's own history. | `BehaviouralChange` |
| Alert | A judgement that a person should look at something. | `Alert` |

A sensor event is not a behaviour. A fridge door opening is a contact
observation; it is not a record of eating. The ontology therefore stops at
`kitchen_activity` and makes no claim about food intake.

## The canonical observation model

Every adapter converts device-specific payloads into
`Observation`, and every later stage
consumes only that type. This is what keeps the scientific code independent of
any manufacturer or protocol.

An observation carries:

`timestamp`
:   Timezone-aware instant. Naive timestamps are **rejected**, not assumed to
    be UTC or local: that assumption silently shifts every event by hours and
    breaks daily-rhythm analysis across DST boundaries.

`sensor_id`, `modality`, `kind`
:   Identity, what kind of physical evidence, and temporal semantics.

`value`, `unit`
:   The measurement and the unit it was taken in. Units are converted at the
    boundary against the declared `SensorSpec`;
    an incompatible dimension raises rather than passing through.

`quality`, `confidence`
:   Device-reported measurement quality, and confidence that the value is
    correct. Derived features set `confidence` below one.

`source`, `received_at`, `flags`, `context`
:   Provenance: which gateway, when it arrived, what was repaired on the way
    in, and free-form metadata.

### Temporal semantics

`ObservationKind` is the reason event
streams are never forward-filled:

`EVENT`
:   An instantaneous occurrence. **Absence of an event is not a zero value.**

`STATE`
:   A level that persists until the next reported change.

`SAMPLE`
:   A measurement of a continuously existing quantity at a point in time.

`event_counts` counts
activations and leaves empty bins at zero, meaning *no event was recorded*.
`observed_mask` is its
companion and separates "the sensor reported nothing" from "the sensor
reported no activity". `sample_frame` leaves unsampled bins `NaN`.
`state_frame` carries a state forward only for a bounded `max_hold`,
after which the state is unknown rather than unchanged.

### Declaring a deployment

`SensorRegistry` holds one
`SensorSpec` per sensor: its modality,
kind, unit, room, expected reporting interval, plausible value range, prior
reliability, and crucially whether it is `attributable` -- whether an
activation identifies *who* generated it.

Inference reads the registry rather than pattern-matching on sensor names.
Adding a device to a deployment means adding a spec, not editing inference
code, and `subset` lets an
ablation study drop a modality without touching anything else.

### Ingestion

`ObservationIngestor` performs
*structural* validation only: contract checks, unit normalisation,
out-of-order and late-arrival flagging, and optional per-source clock-offset
correction by minimum-latency filtering. Judgements about whether a sensor is
*behaving* belong to `health`. Keeping the two apart
means a malfunctioning sensor never looks like a malformed message.

Rejections are counted in an
`IngestionReport`, never raised: one
malformed record must not stop a live stream.

## Clocks and time zones

* Every timestamp is timezone-aware and ordering is done on absolute UTC
  instants.
* Tabular framing bins on UTC while labelling in local time, so DST
  transitions stay complete in both directions.
* Daily aggregation is bounded by *local* midnight, so a DST day is correctly
  23 or 25 hours rather than assumed to be 24.
* Wall-clock arithmetic is confined to a single localisation helper in the
  simulator; everything else works in absolute time.

## Online execution

`BehaviouralSensingPipeline` runs the chain
incrementally:

```text
push(observation)      validate, buffer behind the lateness watermark
advance(now)           release in timestamp order, then per step:
                         update sensor health
                         update occupancy context and attribution
                         update the latent state posterior
                         on a day boundary: summarise, update baselines,
                                            evaluate change, consider alerts
                       emit a PipelineStep
```

### Bounded memory

Every stage keeps bounded state: short deques per sensor in the health
monitor, a single belief vector in the filter and context estimator, a capped
history in each baseline, and one day of estimates in the pipeline. The
pipeline runs indefinitely without growing.

### Reordering

The filter is causal and raises
`NonMonotonicUpdateError` rather than folding a
stale record into an already-advanced belief. Reordering is therefore the
pipeline's job: records are held behind a watermark
(`lateness_tolerance`), released in timestamp order, and anything later than
the tolerance is counted in `pipeline.too_late` rather than silently
absorbed.

### Snapshot and restore

`snapshot` returns
JSON-serialisable state for every stage, and `restore` rebuilds it. A
snapshot taken under a different state ontology is rejected rather than
reinterpreted position by position. The lateness buffer is deliberately
excluded: those records have not been folded into any belief yet, so
re-delivering them after a restart is correct, whereas persisting them would
risk counting them twice.

## Separation of concerns

* `sensor_modeling.online` performs no inference of its own and knows
  nothing about files, HTTP, dashboards, or storage.
* Scientific layers are usable without the pipeline.
* The simulator's generative process is deliberately different from the
  inference model; see [evaluation](evaluation.md).
