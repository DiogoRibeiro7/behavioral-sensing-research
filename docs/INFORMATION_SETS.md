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
then history from the most recent window backwards. A nested set's columns are
therefore an ordered subsequence of the larger set's columns, with identical
values.

## What each component means

For a prediction at moment `t`:

| Component | Definition | Matches |
| --- | --- | --- |
| Current evidence | activations per channel in `(t - step, t]` | the window the pipeline batches at `t`; an activation is a non-zero event observation, as the Poisson emission counts it |
| Time of day | local wall-clock hour of `t`, 0–23 | the hour the optional circadian prior reads |
| Recent history | activations per channel in each of the `history_steps` windows before the current one | the lagged steps given to the supervised diagnostic |

The hour is a raw integer. How a model encodes it (one-hot, cyclic or ordinal)
is the model's choice, and it adds no information.

## Resolution

Every set in a comparison shares these three settings:

| Setting | Default | Source of the default |
| --- | --- | --- |
| `step` | 5 minutes | the step used for every figure in [Real-data validation](real_data.md) |
| `channels` | `HH_EVIDENCE_CHANNELS` | every event channel the CASAS `hh` adapter can produce |
| `history_steps` | 3 | the three lagged steps given to the supervised diagnostic |

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

The builder treats which sensors are installed as deployment metadata. The
filter receives the same registry from its first step.

The builder emits raw counts. Any scaling or imputation must be fitted on
development households only.

## Using it against the filter

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
