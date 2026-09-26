# Information-set contract

Phase 1 of the [roadmap](roadmap.md) asks how much of the real-data gap comes
from what the sensors can support and how much from the current probabilistic
formulation. There is no answer to that unless every model is compared on an
explicitly declared information set. A model that sees more has not shown a
modelling advantage just because it scores higher.

`sensor_modeling.datasets.information_sets` declares those sets and builds the
matching feature tables from a real recording. It does not fit, score or
select any model.

## The four Phase 1 sets

`nested_information_sets()` returns them in this order. All four share one
`EvidenceResolution`.

| Set | Components | Columns at the default resolution |
| --- | --- | ---: |
| `current` | current evidence | 6 |
| `current+time_of_day` | + time of day | 7 |
| `current+history` | + recent history | 24 |
| `current+time_of_day+history` | + both | 25 |

`current+time_of_day` and `current+history` are not nested in each other. Both
lie between the first and last sets. `InformationSet.is_nested_in` reports this.
It returns false for any two sets whose resolutions differ, because those sets
differ in more than their declared components.

Columns always appear in the same order: current evidence, then `hour_of_day`,
then history from the most recent window backwards, then history summaries. A
nested set's columns are therefore an ordered subsequence of the larger set's
columns, with identical values.

History summaries are an optional fifth component, outside these four sets. See
[History summaries](#history-summaries).

## What each component means

For a prediction at moment `t`:

| Component | Definition | Matches |
| --- | --- | --- |
| Current evidence | activations per channel in `(t - step, t]` | the window the pipeline batches at `t`; an activation is a non-zero event observation, as the Poisson emission counts it |
| Time of day | local wall-clock hour of `t`, 0–23 | the hour the optional circadian prior reads |
| Recent history | activations per channel in each of the `history_steps` windows before the current one | the lagged steps given to the supervised diagnostic |
| History summaries | counts, room changes, quiet time and last active room over longer lookbacks | a condensed form of the step history the filter carries forward |

The hour is a raw integer. How a model encodes it is the model's choice, and
no encoding adds information; see [Time of day](#time-of-day).

## Time of day

### Definition

The time of day of a prediction moment `t` is its hour on the household's
local clock:

```text
h(t) = hour of t on the household's local clock,   h in {0, 1, ..., 23}
```

The household's zone is the one its recording was read in, for example
`read_casas_hh(timezone=...)`. A moment given in another zone is converted.
`local_hour(moment, zone)` computes it. This is the same reading the optional
circadian prior makes with `at.hour`, so an information set with time of day
gives every model what the filter can use.

### Encodings

| Encoding | Columns | Suits |
| --- | --- | --- |
| ordinal | `h` | trees, which can split anywhere |
| one-hot | 24 indicators | models that need a free parameter per hour |
| cyclic | `sin(k θ)`, `cos(k θ)` for `k = 1 … K` | linear models; one smooth daily cycle per harmonic |

The cyclic encoding places each hour at the angle of its midpoint on the
24-hour clock:

```text
θ(h) = 2π (h + ½) / 24
```

- **Placement.** The hours sit evenly on a circle, so 23:00 is exactly as close
  to 00:00 as 00:00 is to 01:00.
- **Harmonics.** Harmonic `k` repeats `k` times a day, so `K = 2` can describe
  a morning and an evening peak.
- **Limit.** `K` is limited to 1–11. At `K = 11` the encoding already has 22 of
  the 24 degrees of freedom that one-hot has.
- **Using it.** `cyclic_hour_features(hours, K)` computes the columns, and
  `LogisticBaseline(hour_encoding="cyclic", harmonics=K)` uses them. The
  pre-declared baseline suite keeps one-hot for the logistic model and the
  ordinal hour for the tree.

Every encoding carries the same information, the hour. So an information set
with time of day is the same set whichever encoding a model uses, and a
comparison between encodings is a comparison of models, not of information.

### Reading a fitted daily cycle

With one harmonic, a linear score for a state is

```text
β sin θ + γ cos θ = A cos(θ − φ)
A = √(β² + γ²)
φ = atan2(β, γ)
peak hour = 24 φ / 2π − ½   (mod 24)
```

That is one daily cycle of amplitude `A`, peaking at the clock hour in the last
line. `peak_hour(beta, gamma)` returns it.

The logistic baseline standardises its features. Convert its coefficients back
to the unstandardised sine and cosine columns before reading a phase, because
the two columns are scaled separately.

### Daylight-saving time

Time of day follows the clock, not the time elapsed since midnight. A day with
a daylight-saving change still runs from 00:00 to 23:59 on the clock, even
though it lasts 23 or 25 hours.

- **Spring forward.** The local times from 02:00 to 02:59 do not exist, and no
  absolute instant reads that hour. The evaluation grid, however, steps by
  wall-clock arithmetic in the household's zone, as the online pipeline does,
  so it can name 02:00 or 02:30. Such a moment is read as written, as hour 2.
  The circadian prior makes the same reading.
- **Fall back.** The local times from 01:00 to 01:59 occur twice. Both
  occurrences read the same clock time, so they get the same hour and the same
  encoding.

**Known limitation.** Evidence windows compare wall-clock times, as the online
pipeline does, so the two occurrences of a repeated fall-back hour are not
ordered by instant. CASAS timestamps are naive local times, and the reader
treats every one as the first occurrence, so current data cannot trigger this.
Timestamps that distinguish the occurrences, such as ones converted from UTC,
can. On that one night, a window may then count up to an hour of later
evidence. A strict expected-failure test,
`test_a_repeated_fall_back_hour_is_ordered_by_instant`, pins the behaviour.
Fixing it means ordering by absolute time in both the pipeline and this
builder, which is a change to production inference.

### Day of week: evaluated, not added

Before adding day of week, it was checked whether the development data show
weekly structure. The comparison used the 20 single-resident homes, reading
each home's weekend and weekday share of labelled time in each state as a
paired difference across homes, with a 95% household bootstrap interval. The
analysis is descriptive: no model was fitted or scored.

| Weekend minus weekday | Mean over homes | 95% interval | Homes higher at weekends |
| --- | ---: | --- | ---: |
| Whole day, every state | within ±0.015 | | |
| Whole day, `home_active` | −0.015 | [−0.024, −0.004] | 7 of 20 |
| 06:00–12:00, `sleeping` | +0.043 | [+0.009, +0.074] | 15 of 20 |
| 06:00–12:00, `home_active` | −0.030 | [−0.046, −0.016] | 5 of 20 |

Over whole days, the states barely differ between weekdays and weekends. The
visible structure is a modest weekend lie-in, confined to the morning. A day of
week term added alongside time of day, cyclic or not, cannot represent a shift
that happens only in the morning. Day of week would also be a new information
component, not a new encoding. It is therefore not added. If it is pursued,
pre-specify a weekend-by-time-of-day term as its own information component.

## History summaries

The supervised diagnostic behind the ceiling sees 20 minutes of history: the
current step and three lagged steps. The filter conditions on its whole past.
In the median development home, 52% of moments have no activation at all in
those 20 minutes. There, every lagged count is zero, so 20 quiet minutes look
the same as 3 quiet hours.

History summaries give a model a longer past in a few interpretable columns.
With them, a matched comparison can measure how much of the gap between the
filter and the ceiling that longer history explains. They are not a sequence
model: each column is a count, a duration or an indicator that can be read
directly.

### Features

For a prediction at moment `t`, each summary window `W` is a lookback
`(t − W, t]`. The horizon `H` is the longest window.

| Feature | Column | Value |
| --- | --- | --- |
| Count | `history_count_{channel}_{W}m` | activations on the channel in `(t − W, t]` |
| Room changes | `history_room_changes_{W}m` | how often the set of active rooms differs from one active step to the next in `(t − W, t]` |
| Quiet minutes | `history_quiet_minutes_{channel}_{H}m` | minutes of whole silent steps since the channel last fired; `H` if it did not fire in `(t − H, t]` |
| Last room | `history_last_room_{room}_{H}m` | 1 if the room was active in the most recent active step in `(t − H, t]`, otherwise 0 |

- **Active step.** A step is active when at least one instrumented channel
  fired in it.
- **Active room.** A room is active in a step when any of its channels fired.
  The front door counts as its room, the hall.
- **Room changes.** Silent steps are skipped when looking for the next active
  step.

Two examples with 5-minute steps and `t = 12:00`:

- **Quiet minutes.** Kitchen motion at 11:53 falls in the step
  `(11:50, 11:55]`, so the kitchen has been silent for one whole step. Its
  quiet minutes are 5.
- **Last room.** If the kitchen and then the living room fire, both inside
  `(11:50, 11:55]`, and nothing fires afterwards, both rooms are marked as last
  room. The step does not say which came first.

`summary_column` builds these names. `parse_summary_column` reads one back as
`(feature, subject, minutes)`.

### Matched to the filter's steps

Every summary is a function of the per-channel counts in whole step windows,
`(t − (j + 1) step, t − j step]`. These are the windows the filter batches. No
summary uses a time or an order inside a step, which the filter never
receives. That has three consequences:

- Room changes are counted between steps, not between events. An event-level
  count would need the order of activations within a step.
- Quiet minutes are whole multiples of the step.
- If two rooms are active in the most recent active step, both are marked.

A set with summaries over a horizon `H` therefore carries no information beyond
current evidence and `H / step − 1` lagged steps, and the filter conditions on
all of that. The summaries condense that history; they add no new information.
The tests check this by recomputing every summary from the lagged counts.

### Why these windows

There are two windows, both declared before any model was fitted on them and
both whole numbers of steps:

| Window | Reason |
| --- | --- |
| 60 minutes | spans the median `away` bout (53 minutes) and `home_inactive` bout (19.6 minutes), two of the three states where the filter falls furthest below the ceiling |
| 180 minutes | spans the median `sleeping` bout (155 minutes) and is the horizon |

In the median development home, 27.8% of moments have had no activation for
at least an hour, and 6.8% for at least three hours. With a three-hour
horizon, quiet minutes are therefore censored for few moments.

- **Sources.** The bout medians come from the dwell table in
  [Real-data validation](real_data.md). The shares were measured on the 20
  development homes, at every step from each home's first observation to its
  last, without labels.
- **Only at the horizon.** Quiet minutes and last room are reported only at
  the horizon. At a shorter window `W` they are exact functions of the horizon
  columns; for example, quiet minutes at `W` equal `min(quiet minutes at H, W)`.
  Repeating them would add columns without adding information.
- **At every window.** Counts and room changes cannot be recovered from one
  window to another, so each window has its own.

At the default resolution the component adds 25 columns:

| Feature | Columns |
| --- | ---: |
| counts: 6 channels × 2 windows | 12 |
| room changes: 1 per window | 2 |
| quiet minutes: 1 per channel | 6 |
| last room: 1 per room | 5 |

### Missing and sparse history

| Situation | Value |
| --- | --- |
| Lookback inside the recording with no activations | counts `0`, room changes `0`, quiet minutes `H`, every last-room column `0` |
| Lookback that reaches a step closing before the first observation | counts and room changes `NaN`. Quiet minutes and last room are known once something has fired inside the recording, and `NaN` until then |
| Channel with no event sensor in this household | its counts and quiet minutes `NaN` |
| Room with no instrumented channel | its last-room column `NaN`; room changes use the instrumented rooms |
| Household with no instrumented channel | every summary `NaN` |
| After the last observation | as inside the recording; nothing is missing |

A step counts as observed on the same terms as a lagged window: it must close
at or after the first observation. On the development homes, summaries are
missing in at most 0.5% of any home's rows, all within three hours of its first
observation.

### Using them

```python
from sensor_modeling.datasets import (
    InformationComponent,
    InformationSet,
    nested_information_sets,
)

diagnostic = nested_information_sets()[-1]  # current, time of day, 3 lagged steps
extended = InformationSet(
    "current+time_of_day+history+history_summary",
    diagnostic.components | {InformationComponent.HISTORY_SUMMARY},
    diagnostic.resolution,
)
```

`diagnostic.is_nested_in(extended)` is true, so `compare_information_sets` can
report what the summaries add for each model under the
[matched evaluation runner](MATCHED_EVALUATION.md). That measurement is a
separate, pre-declared run. This component makes no claim about its outcome.

The summary windows enter a set's declaration and digest only when the set
includes summaries. The digests of the four Phase 1 sets are unchanged.

## Resolution

Every set in a comparison shares these four settings:

| Setting | Default | Source of the default |
| --- | --- | --- |
| `step` | 5 minutes | the step used for every figure in [Real-data validation](real_data.md) |
| `channels` | `HH_EVIDENCE_CHANNELS` | every event channel the CASAS `hh` adapter can produce |
| `history_steps` | 3 | the three lagged steps given to the supervised diagnostic |
| `summary_windows` | 60 and 180 minutes | see [Why these windows](#why-these-windows) |

When the filter is one of the models compared, `step` must equal its pipeline
step.

A channel is a room and a modality, not just a room. The filter's default
observation model gives an entrance door its own rates, separate from motion
in the same hall, so pooling the two would give the filter information that
its comparators lack. Sensors on the same channel are pooled. The default
model gives them identical rates, so the pooled count carries the same
likelihood information as the separate counts.

[Real-data validation](real_data.md) describes the supervised diagnostic as
using per-room counts. This contract separates door from motion in the hall,
the only room where the `hh` vocabulary has both, so tables built here will not
reproduce the diagnostic's figures exactly.

## Guarantees

- **No future information.** Every window closes at or before `t`. The tests
  remove every observation after each moment and check that the row does not
  change.
- **No labels.** Annotations are never read. Labels come from `truth_series`
  at the same moments.
- **Household isolation.** Each table is built from one recording.
  `build_panel_features` never pools observations, so history cannot cross
  from one home to another. Splits are made on household keys and remain as
  they are fixed elsewhere: the development panel and the frozen external
  cohort do not change.
- **Determinism.** Channels are stored in sorted order, results do not depend
  on observation order, and a declaration serialises through `to_dict()` with
  a stable `sha256()` that can be recorded in a pre-specified analysis.

## Absence is not zero

| Situation | Value |
| --- | --- |
| Instrumented channel, no activations in the window | `0`: silence from a working sensor is evidence |
| Declared channel with no event sensor in this household | `NaN` at every lag; listed in `uninstrumented` |
| Window that closes before the recording's first observation | `NaN` |
| Window after the last observation | `0`: at `t` nobody can know the recording is about to end |
| Sensor that is not event-kind, has no room, or sits on an undeclared channel | excluded; listed in `excluded_sensors` |

History summaries follow the same rules; their cases are listed under
[Missing and sparse history](#missing-and-sparse-history).

The builder treats which sensors are installed as deployment metadata. The
filter receives the same registry from its first step.

The builder emits raw counts. Any scaling or imputation must be fitted on
development households only.

## Using it against the filter

To compare fitted models under one set, use the
[matched evaluation runner](MATCHED_EVALUATION.md), which builds these tables
itself. To align features with the filter's own estimates:

```python
from datetime import timedelta

from sensor_modeling.datasets import (
    build_feature_table,
    nested_information_sets,
    truth_series,
)
from sensor_modeling.online import BehaviouralSensingPipeline, PipelineConfig
from sensor_modeling.online.pipeline import scoring_steps

config = PipelineConfig(step=timedelta(minutes=5))  # the resolution's step
pipeline = BehaviouralSensingPipeline(recording.registry, config=config)
steps = pipeline.run(recording.observations)
steps.extend(pipeline.close(recording.observations[-1].timestamp))
moments = [s.at for s in scoring_steps(steps)]

truth = truth_series(recording.activities, moments)
tables = {
    info.name: build_feature_table(recording, info, moments, household="hh101")
    for info in nested_information_sets()
}
```

## What the filter cannot be restricted to

The filter is recursive. Its belief at `t` conditions on every earlier window,
so its information set always includes current evidence and unbounded history
at this resolution. The circadian profile can switch time of day on or off,
but history cannot be removed. A comparison must report this and must not
place the filter in the `current` set. The only models that can be compared
within all four sets are non-recursive ones.

History summaries do not change this. They are functions of step counts over a
bounded horizon, so a set that includes them still lies within what the filter
conditions on.
