# Ingesting Heterogeneous Sensors

How four physically unlike sensors reach the same inference pipeline without
any hardware-specific logic leaking downstream.

## The principle

Every adapter produces
`Observation`. Every stage above the
observation layer reads a
`SensorRegistry`, never a sensor name.
Adding a device to a deployment therefore means adding a declaration, not
editing inference code.

## Declaring the deployment

```python
from datetime import timedelta

from sensor_modeling.observations import (
    Modality, ObservationKind, SensorRegistry, SensorSpec, Unit,
)

registry = SensorRegistry.from_specs([
    # An event sensor. It makes no promise to report, so its silence is
    # never treated as a fault.
    SensorSpec(
        "front_door", Modality.DOOR, room="hall",
        description="Contact on the entrance door; fires on any crossing.",
    ),

    # A state sensor: a level that persists until it changes.
    SensorSpec(
        "bed_pressure", Modality.BED_PRESSURE,
        kind=ObservationKind.STATE, room="bedroom",
        expected_interval=timedelta(minutes=5),
        value_range=(0.0, 1.0),
        description="Load cell reporting bed occupancy as a level.",
    ),

    # A person-bound sampled sensor. `attributable` is what distinguishes
    # evidence about *this resident* from evidence about the home.
    SensorSpec(
        "wearable_motion", Modality.WEARABLE_MOTION,
        kind=ObservationKind.SAMPLE,
        expected_interval=timedelta(minutes=1),
        value_range=(0.0, 10.0), attributable=True,
        description="Accelerometer activity magnitude from a worn device.",
    ),

    # A derived mmWave feature. Not raw radar, and not a measurement:
    # observations from it carry a confidence below one.
    SensorSpec(
        "living_radar", Modality.RADAR,
        kind=ObservationKind.SAMPLE, unit=Unit.COUNT,
        room="living", expected_interval=timedelta(minutes=1),
        value_range=(0.0, 6.0),
        description="Derived mmWave feature: number of tracked people.",
    ),
])
```

## Four adapters, one observation type

```python
from datetime import datetime, timezone
from sensor_modeling.observations import Observation

now = datetime(2024, 5, 1, 8, 0, tzinfo=timezone.utc)

door = Observation(
    timestamp=now, sensor_id="front_door",
    modality=Modality.DOOR, kind=ObservationKind.EVENT, value=1.0,
)

bed = Observation(
    timestamp=now, sensor_id="bed_pressure",
    modality=Modality.BED_PRESSURE, kind=ObservationKind.STATE, value=1.0,
)

wearable = Observation(
    timestamp=now, sensor_id="wearable_motion",
    modality=Modality.WEARABLE_MOTION, kind=ObservationKind.SAMPLE,
    value=0.04,
)

# A derived feature states its own uncertainty. The device computed this
# number; it did not measure it.
radar = Observation(
    timestamp=now, sensor_id="living_radar",
    modality=Modality.RADAR, kind=ObservationKind.SAMPLE,
    value=2.0, unit=Unit.COUNT, confidence=0.75,
)
```

## Running the pipeline

The pipeline is constructed from the registry alone. It contains no branch on
sensor identity, and derives its observation models from the declarations.

```python
from sensor_modeling.online import BehaviouralSensingPipeline, PipelineConfig

pipeline = BehaviouralSensingPipeline(
    registry, config=PipelineConfig(step=timedelta(minutes=5))
)

for observation in (door, bed, wearable, radar):
    pipeline.push(observation)

for step in pipeline.advance(now + timedelta(minutes=20)):
    print(step.state.explain())
```

Swapping a modality out is a registry change:

```python
without_radar = registry.subset(
    ["front_door", "bed_pressure", "wearable_motion"]
)
sparse = BehaviouralSensingPipeline(without_radar)
```

This is the same mechanism the ablation framework uses, which is why an
ablated run exercises the same inference path a genuinely sparse deployment
would.

## Converting legacy tabular data

The original toolkit represents sensor data as a timestamp-indexed frame with
one column per sensor. `adapters` converts
it, but requires a registry, because the frame carries no modality, no unit
and no temporal semantics.

```python
from sensor_modeling.observations import observations_from_frame

observations = list(observations_from_frame(legacy_frame, registry))
```

Two behaviours are worth knowing about.

**Zeros in event columns are dropped.** A zero in a motion column is the
absence of a record, not an observation that nothing happened. Emitting it
would fabricate exactly the evidence the canonical model exists to withhold.
Zeros in state and sample columns are kept, because there a zero is a genuine
measurement.

**A naive index is refused.** Assuming UTC or local time would silently shift
every record by hours and break daily-rhythm analysis across DST boundaries,
so the timezone must be supplied:

```python
observations_from_frame(legacy_frame, registry, tz=ZoneInfo("Europe/Lisbon"))
```

Long-format records -- one row per reading -- are also supported through
`observations_from_records`, and carry
event semantics correctly by construction, since an absent row is simply an
absent row.

## Going back to tables

Conversion runs one way for inference, but
`ObservationStream` frames observations
back into tables so the original models keep working:

```python
stream.event_counts("15min")    # activations per bin; zeros mean "not recorded"
stream.observed_mask("15min")   # did anything arrive at all?
stream.sample_frame("15min")    # mean per bin; NaN where unsampled
stream.state_frame("15min", max_hold=timedelta(hours=2))
```

There are three methods rather than one general `to_frame()` so that
choosing the wrong framing for a modality requires calling the wrong method by
name.
