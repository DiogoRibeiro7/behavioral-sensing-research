# External-validation dataset contract

ROADMAP Phase 5 asks whether results survive outside the CASAS ecosystem, on an
independently collected, annotated smart-home dataset with a different sensing
layout. Such a dataset names its sensors, sensor types, rooms and activities its
own way. This page describes the contract such a dataset must be read through,
in `sensor_modeling.external`, before any model is scored on it.

**Status: infrastructure.**

- **No scoring.** No model is scored on any external dataset here.
- **The existing CASAS external cohort.** The cohort of
  [43 further CASAS homes](EXTERNAL_COHORT_COMPOSITION.md) is untouched, and
  its frozen v0.3 evaluation is unaffected.

## What an adapter exposes

An adapter delivers each household in the dataset's own terms
(`sensor_modeling.external.contract`):

| Field | Where | Notes |
| --- | --- | --- |
| household identifier | `HouseholdData.household` | the dataset's own |
| timestamp | `RawEvent.timestamp` | naive or aware; placed only through the declared timezone |
| sensor identifier | `RawEvent.sensor_id`, `SensorDescription.sensor_id` | the dataset's own |
| sensor type | `SensorDescription.sensor_type` | the dataset's own, such as `pir` or `reed_switch` |
| location or room | `SensorDescription.location` | the dataset's own, or `None` |
| raw event or value | `RawEvent.value` | exactly as recorded: a token such as `ACTIVE`, or a number |
| behavioural annotation | `Annotation.label` | the dataset's own label |
| annotation semantics | `Annotation.semantics` | `interval` (a span, end-exclusive) or `point` (a moment) |
| timezone | `HouseholdData.timezone` | an IANA name such as `Europe/Lisbon` |
| provenance | `DatasetProvenance` | name, version, source, licence, citation, retrieval date and each file's SHA-256 |

A household can also carry:

- **`occupancy`.** Periods with a known number of residents, such as a
  visitor's stay.
- **`residents`.** Its nominal number of residents.
- **`Annotation.resident`.** Whom an annotation describes.

Nothing in the contract assumes CASAS conventions: no sensor-identifier
prefixes, begin and end markers, room names or activity labels. A test checks
that the contract, mapping and validation modules import nothing from the
CASAS readers.

### Adapters

`DatasetAdapter` is a protocol: a `provenance`, the sorted `households()`, and
`load(household)`. Two reference adapters are provided.

- **`InMemoryAdapter`.** Wraps households already built in memory.
- **`CsvAdapter`.** Reads a long-format dataset of four files: events,
  annotations, sensors and households, one row each. Its column names are
  declared in a `CsvLayout`, never guessed. It records every file's SHA-256 in
  the provenance.

A new dataset needs its own adapter only when it cannot be exported to that
layout. Two datasets have one:

- **`OrdonezAdapter`**, in `sensor_modeling.external.ordonez`. The UCI ADL
  Binary dataset, used by the [Phase 5 external
  evaluation](PHASE5_EXTERNAL_PROTOCOL.md).
- **`TihmAdapter`**, in `sensor_modeling.external.tihm`. The TIHM dataset of
  56 homes of people living with dementia. Its labels are alerts a clinical
  team verified, stamped at a moment, so they are point annotations and every
  one is declared unmappable: the contract converts the sensor events and
  scores nothing against a state. It is used by the [TIHM alert-burden
  protocol](TIHM_ALERT_BURDEN_PROTOCOL.md), and by the description of the
  silent-home rule on the same homes in the [silent-home
  results](SILENT_HOME_RESULTS.md), and of the calibrated baseline reference
  in the [threshold-calibration results](THRESHOLD_CALIBRATION_RESULTS.md).
  The [sleep-mat comparison](SLEEP_MAT_RESULTS.md) reads the dataset's
  `Sleep.csv` itself, beside the adapter, which still does not. The planning record of the
  [matched-sensor protocol](MATCHED_SENSORS_PROTOCOL.md) reads `Activity.csv`
  itself for its sensors alone, the gaps between activations and eight
  moments of each home's records, and maps no record to a state. The
  [sleep-gap description](SLEEP_GAP_RESULTS.md) reads `Sleep.csv` minute by
  minute, beside the adapter, and records per-home summaries only.

## The mapping layer

Correspondence with the repository's ontology is declared, never inferred, in
an `OntologyMapping` (`sensor_modeling.external.mapping`). It has three
tables:

- **Labels** map to `BehaviouralState`: a latent state, never `unknown`.
- **Sensor types** map to `SensorSemantics`: a canonical modality and
  temporal kind (`event`, `state` or `sample`), with the raw tokens that
  activate and deactivate, or numeric values and their unit.
- **Locations** map to a canonical room: `bathroom`, `bedroom`, `hall`,
  `kitchen` or `living`. These are the rooms the ontology's states and the
  evidence channels use.

Every entry has one of four outcomes:

| Outcome | Targets | Rationale | Used for scoring |
| --- | --- | --- | --- |
| `exact` | one | optional | yes |
| `approximate` | one | required: what is lost or added | yes, and reported as approximate |
| `unmappable` | none | required: why nothing fits | no |
| `ambiguous` | two or more | required: why none can be chosen | no: choosing one would invent an annotation |

A native value the mapping does not declare resolves as `undeclared`, and a
sensor without a location as `missing`. An undeclared label, sensor type or
location is a validation error: it is never silently dropped and never
guessed. Every native value must therefore be declared, even as unmappable.

```python
from sensor_modeling.external import (
    OntologyMapping, SensorSemantics, exact, approximate, ambiguous, unmappable,
)

mapping = OntologyMapping(
    dataset="aurora-living-lab", version="2.1",
    labels=(
        exact("sleep", BehaviouralState.SLEEPING),
        approximate("tv", BehaviouralState.HOME_INACTIVE, "watching television, assumed seated"),
        ambiguous("nap", (BehaviouralState.SLEEPING, BehaviouralState.HOME_INACTIVE),
                  "a nap in bed or on the sofa"),
        unmappable("medication", "a brief action, not a behavioural state"),
    ),
    sensor_types=(exact("pir", SensorSemantics(Modality.MOTION, ObservationKind.EVENT,
                                               frozenset({"ACTIVE"}), frozenset({"CLEAR"}))),),
    locations=(exact("Kitchen", "kitchen"), approximate("Lounge", "living", "a lounge-diner")),
)
digest = mapping.write(Path("mapping.json"))
```

### Freezing a mapping

A mapping is data.

- **`write`** stores it as JSON with its SHA-256.
- **`read(path, sha256)`** refuses a file whose content does not match its
  recorded digest, or the digest a protocol expects.

Phase 5 requires the ontology mapping to be frozen before any external result
is seen. That means committing the mapping file and citing its digest in the
protocol, as the Phase 1 to 4 protocols cite theirs.

## Validation

`validate_household(data, mapping)` checks one household, and
`validate_dataset(adapter, mapping)` every household. They repair nothing. Each
problem is an `Issue` with a code, a severity, a count and a few examples:

| Code | Severity | Meaning |
| --- | --- | --- |
| `missing_timezone` | error | no timezone, so local times and time-of-day features cannot be placed |
| `invalid_timezone` | error | not an IANA timezone |
| `mixed_awareness` | error | some timestamps carry an offset and some do not |
| `dst_transition` | warning | naive local times in a daylight-saving gap or fold, where they do not name one instant |
| `empty_household` | error | no events at all |
| `timestamp_order` | warning | events not in time order as delivered |
| `duplicate_event` | warning | the same sensor, time and value more than once |
| `unknown_sensor` | error | events from sensors the dataset does not describe |
| `unused_sensor` | info | described sensors that never report |
| `impossible_interval` | error | an interval ending at or before its start, an interval without an end, a point with an end, or an occupancy period that cannot exist |
| `outside_recording` | warning | annotations beyond the span of the events |
| `overlapping_labels` | warning | annotations that overlap in time |
| `multi_resident` | warning | periods with more than one resident, or overlapping annotations of different residents: unsupported, and left unscored |
| `multi_resident_home` | error | a household declared multi-resident throughout |
| `point_annotations` | info | annotations of a moment, which carry no state span |
| `undeclared_label` | error | a label the mapping does not declare |
| `undeclared_sensor` | error | a sensor type or location the mapping does not declare |
| `sensor_semantics` | warning | a sensor whose type or location maps other than exactly, which has no location, or whose raw values do not fit its declared semantics |

A report also carries three tables:

- **Each label.** Its mapping status, number of annotations and annotated
  seconds.
- **Each sensor.** Its type's and location's status, its room, and how many
  of its values fit none of its declared tokens.
- **Coverage.** The annotated time by mapping status, which answers "what can
  and cannot be mapped".

A `DatasetReport` adds totals over every household, and serialises to JSON
with the provenance and the mapping's digest.

## Conversion to the repository's form

`to_canonical(data, mapping)` turns a household into the recording type the
rest of the repository consumes: a sensor registry, canonical observations in
time order, and labelled intervals.

- **Refusal.** It refuses a household with any validation error, unless
  `strict=False`.
- **Counting.** Everything it does not carry is counted in `dispositions`, by
  reason.

The rules:

- **Timestamps.** Naive timestamps are placed in the declared timezone. In a
  daylight-saving fold, the earlier reading is used.
- **Order and duplicates.** Events are sorted, and exact duplicates are kept
  once and counted.
- **Sensors.** A sensor whose type is unmappable, ambiguous or undeclared is
  left out with its events. A sensor's room is its location's target.
- **Values.**
  - **Events.** An event sensor emits 1 for an activation token, and counts
    deactivations without emitting them.
  - **States.** A state sensor emits 1 and 0.
  - **Numbers.** Numeric values are emitted as they are.
  - **Other values.** A value outside the declared semantics is counted and
    left out.
- **Annotations.** Interval annotations are split at every boundary, including
  those of multi-resident periods. A segment gets a state only when its
  exact or approximate annotations agree on one.
  - **Conflict.** A segment whose annotations name different states is left
    unscored.
  - **Multi-resident.** A segment in a multi-resident period is left unscored.
  - **No usable label.** A segment whose only labels are unmappable, ambiguous
    or undeclared is left unscored.
  - **Points.** Point annotations are counted and carry no span.

Every unscored segment carries `state=None`. The evaluation code already skips
such positions rather than scoring a guess. The annotated seconds by reason add
up to the labelled time, and the event dispositions add up to the events
delivered, so nothing is lost uncounted.

## What the contract serves in Phase 5

The ROADMAP's Phase 5 requirements, and where the contract meets them:

1. **Freeze preprocessing, the ontology mapping and the evaluation rules before
   final external results.** The mapping and the conversion rules are data and
   code with digests. The evaluation rules belong to the Phase 5 protocol,
   which is not written here.
2. **Report what can and cannot be mapped.** Label coverage by outcome, sensor
   statuses, and the dispositions of conversion.
3. **Distinguish zero-shot transfer from dataset-specific refitting.** The
   contract fits nothing. Refitting, if any, is a separate, declared step of
   the Phase 5 protocol.
4. **Compare household-level distributions.** Every report is per household
   first.
5. **Document failure modes caused by different sensor semantics or coverage.**
   `sensor_semantics` issues, per-sensor statuses and unfit values, and the
   coverage tables.

## What this does not do

- **No scoring.** No model is scored on external data.
- **No multi-resident support.** Multi-resident periods are left unscored;
  modelling them is outside the single-resident ontology.
- **No inference of meaning.** Undeclared labels, types and locations are
  errors, never guesses.
- **No spans from points.** Point annotations are not turned into state spans.
