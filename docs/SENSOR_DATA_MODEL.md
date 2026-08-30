# Sensor Data Model

The contract every sensor record satisfies before any inference sees it, and
the precise semantics of each field. Implemented in
`sensor_modeling.observations`.

## The canonical observation

```python
Observation(
    timestamp,            # timezone-aware datetime, required
    sensor_id,            # registered identifier
    modality,             # what kind of physical evidence
    kind,                 # EVENT | STATE | SAMPLE
    value,                # finite float in `unit`
    unit=Unit.NONE,
    quality=1.0,          # device-reported measurement quality, [0, 1]
    confidence=1.0,       # confidence the value is correct, [0, 1]
    source="",            # gateway / hub / adapter identity
    sampling_interval=None,
    received_at=None,     # arrival time, for lateness and clock drift
    flags=frozenset(),    # what ingestion repaired
    context={},           # free-form string metadata
)
```

Frozen and validated on construction. Rejected at the boundary: naive
timestamps, non-finite values, empty sensor ids, probabilities outside
`[0, 1]`, non-integer counts, non-positive sampling intervals.

### quality versus confidence

These are separate because they fail separately.

- `quality` describes the **hardware**: a device reporting a low signal-to-noise
  ratio, a weak battery, a marginal radio link.
- `confidence` describes the **value**: an upstream estimator's own belief that
  the number is right.

A radar reporting a presence probability sets `confidence` below one even when
the hardware is perfect. `evidence_weight()` multiplies them, because both must
hold for the record to count as strong evidence.

## Temporal semantics: the `kind` field

This field exists to make one class of error impossible by construction.

| Kind | Meaning | What a gap means | Permitted framing |
| --- | --- | --- | --- |
| `EVENT` | Instantaneous occurrence | **No event was recorded.** Not a zero. | Count per bin |
| `STATE` | A level persisting until the next reported change | The level probably held | Carry forward, bounded by `max_hold` |
| `SAMPLE` | Measurement of a continuously existing quantity | The quantity existed but was not measured | Mean per bin, `NaN` when absent |

`ObservationStream` offers `event_counts()`, `state_frame()` and
`sample_frame()` rather than one general `to_frame()`, so choosing the wrong
framing requires calling the wrong method by name.

`observed_mask()` is the companion to `event_counts()`: it separates *the
sensor reported nothing* from *the sensor reported no activity*.

## Modalities

| Modality | Physical evidence | Default kind |
| --- | --- | --- |
| `CONTACT` | Open/close on an object (cupboard, fridge, drawer) | EVENT |
| `DOOR` | Entrance or room door crossing | EVENT |
| `MOTION` | PIR or equivalent binary movement | EVENT |
| `VIBRATION` | Accelerometer-derived vibration on furniture | EVENT |
| `ENVIRONMENTAL` | Ambient scalar (temperature, humidity, light, CO2) | SAMPLE |
| `BED_PRESSURE` | Bed or chair occupancy from pressure or load cells | SAMPLE |
| `WEARABLE_MOTION` | Accelerometer activity counts or magnitude | SAMPLE |
| `WEARABLE_PHYSIOLOGY` | Heart rate, skin temperature | SAMPLE |
| `ROOM_OCCUPANCY` | Room-level occupancy from an upstream device | SAMPLE |
| `RADAR` | Derived mmWave feature. **Never raw radar cubes.** | SAMPLE |
| `PROXIMITY` | Short-range presence beacon bound to an identity | SAMPLE |
| `OTHER` | Anything else; adapters must document the semantics | SAMPLE |

Defaults are conventions, overridable per sensor. They exist so the dangerous
case — treating an event stream as a sampled signal — does not happen by
omission.

### Radar and mmWave

The platform consumes **derived features**, not raw radar. Raw DSP is out of
scope: it would couple the core to a specific chipset and add a large signal
processing surface for no inferential gain the deployment cannot get from the
device's own firmware.

Supported derived features and their required semantics:

| Feature | Unit | Meaning |
| --- | --- | --- |
| `presence_probability` | `PROBABILITY` | Device's belief that at least one person is in range |
| `track_count` | `COUNT` | Number of simultaneously tracked people |
| `range` | `METRE` | Distance to the nearest track |
| `radial_velocity` | `METRE_PER_SECOND` | Signed velocity along the beam |
| `motion_energy` | `NONE` | Device-defined movement magnitude; document its scale |
| `posture_probability` | `PROBABILITY` | Device's belief in a named posture |
| `fall_probability` | `PROBABILITY` | Device's belief a fall occurred |
| `respiration_rate` | `BPM` | Estimated breaths per minute |

Each must be registered as a separate `sensor_id` with its own `SensorSpec`.
A device emitting several features is several sensors. Every one of these is a
*derived feature*: `confidence` below one, and exported with derived
provenance.

## Units

Units are explicit and converted at the boundary. Mismatched dimensions raise
rather than passing through, because silently mixing Celsius and Fahrenheit
corrupts every downstream statistic without any visible symptom.

Supported: `NONE`, `PROBABILITY`, `COUNT`, `degC`, `degF`, `percent`, `lux`,
`ppm`, `m`, `cm`, `m/s`, `cm/s`, `g`, `mg`, `bpm`, `s`, `min`.

Affine conversion is handled for temperature; the rest are multiplicative.
`PROBABILITY` is bounds-checked; `COUNT` is checked for non-negative
integrality.

## Declaring a deployment

`SensorSpec` is how a deployment tells inference what a sensor is:

```python
SensorSpec(
    sensor_id,
    modality,
    kind=None,               # defaults per modality
    unit=Unit.NONE,
    room=None,               # spatial claim, or None
    expected_interval=None,  # a promise to report on a cadence
    value_range=None,        # plausible bounds
    prior_reliability=0.99,
    attributable=False,      # does an activation identify *who*?
    description="",          # documented semantics of the value
)
```

Two fields carry more weight than their size suggests.

**`expected_interval`** is a *promise*. Declaring it means prolonged silence
is diagnostic of a fault. Leaving it `None` means silence is ambiguous between
"broken" and "nobody used it", and the health monitor will decline to call a
failure. Declaring it falsely makes a quiet cupboard look broken; omitting it
falsely makes a dead sensor look quiet.

**`attributable`** is true only for person-bound sensing — a worn device, a
personal beacon. It is the difference between evidence about *this resident*
and evidence about *the home*.

## Temporal handling

| Situation | Handling |
| --- | --- |
| Timezone-naive timestamp | Rejected. Never assumed UTC or local. |
| Irregular sampling | Native. The continuous-time filter needs no grid. |
| Duplicate delivery | Collapsed on `(utc_instant, sensor_id, value)`. |
| Out-of-order arrival | Flagged, then reordered by the pipeline's lateness buffer. |
| Late beyond tolerance | Counted in `pipeline.too_late`, not folded into an advanced belief. |
| Clock drift | Estimated per source by minimum-latency filtering; corrections are flagged. |
| Gaps | Reported by `stream.gaps()`. Interior only: silence before the first record cannot be distinguished from a sensor not yet installed. |
| DST transitions | Framing bins on absolute UTC and labels in local time. Daily aggregation is bounded by local midnight, so a DST day is 23 or 25 hours. |

Wall-clock arithmetic is confined to a single helper in the simulator.
Everything else works in absolute time.

## Sensor health vocabulary

Estimated by `sensor_modeling.health`, consumed by fusion as an evidence
weight.

| Status | Weight | Meaning |
| --- | --- | --- |
| `HEALTHY` | 1.00 | Reporting as declared, plausible values |
| `DEGRADED` | 0.60 | Reporting with reduced quality |
| `DRIFTING` | 0.50 | Calibration shifted against its own history |
| `UNKNOWN` | 0.50 | Not enough evidence to judge. The honest default. |
| `OUT_OF_RANGE` | 0.20 | Implausible values persisting past tolerance |
| `STUCK` | 0.10 | Unchanging where variation is expected |
| `DROPOUT` | 0.05 | Silent long enough to suspect a fault |
| `MISSING` | 0.00 | Supplies no evidence at all |

`MISSING` is exactly zero by design. A sensor that is not reporting must
contribute a flat likelihood, so the fusion layer falls back on other
modalities rather than reading silence as inactivity.

## Interop with the tabular models

`ObservationStream` converts into the `DataFrame` form the original models
expect, via `event_counts()`, `sample_frame()` and `state_frame()`. This is a
deliberate one-way bridge: canonical observations can become tables, and the
tabular models continue to work unchanged, but nothing above the observation
layer consumes tables.
