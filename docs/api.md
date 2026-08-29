# API Reference

Generated from the source. Private members are omitted.

The packages are listed in the order data flows through them, which is also
the order in which they are easiest to read.

## Observations

The canonical, hardware-neutral data model every other stage consumes.

::: sensor_modeling.observations

## Sensor health

Per-sensor reliability, emitted as the evidence weight fusion applies.

::: sensor_modeling.health

## Behavioural states

The configurable ontology and its continuous-time dynamics.

::: sensor_modeling.states

## Multimodal fusion

Recursive estimation of `P(Z_t | O_1:t)` over asynchronous evidence.

::: sensor_modeling.fusion

## Occupancy and attribution

Who is present, and whose activity the sensors saw.

::: sensor_modeling.context

## Adaptive baseline

Non-stationary personal normal, and the verdicts derived from it.

::: sensor_modeling.baseline

## Alerts

The last filter between a behavioural change and somebody's attention.

::: sensor_modeling.alerts

## Online pipeline

Incremental orchestration, snapshotting and benchmarks.

::: sensor_modeling.online

## Simulation

Synthetic households with controlled ground truth, and fault injection.

::: sensor_modeling.simulation

## Evaluation

Metrics, paired ablation, attribution and detection studies, provenance.

::: sensor_modeling.evaluation

## Interoperability

Provenance-preserving export, pseudonymisation and redaction.

::: sensor_modeling.interop

## Modelling core

The original toolkit, unchanged and still supported.

::: sensor_modeling.models

::: sensor_modeling.hmm

::: sensor_modeling.change_point

::: sensor_modeling.analysis

::: sensor_modeling.utils
