# Phase 5: the external-generalisation protocol

ROADMAP Phase 5 asks whether results survive outside the CASAS ecosystem. This page is the protocol for the first external evaluation, generated entirely from the frozen file `artifacts/phase5/external_protocol.json` by `sensor_modeling.datasets.external_protocol_summary.render_protocol`. A test checks that the committed page is exactly that rendering.

**Status: frozen before any external scoring; no model performance on the external dataset was computed or viewed.**

- **Protocol digest.** `620f3fdc55e0fe9da6ebcc0737ccf3c2be584f9573c490ce30388e02a218b334`.
- **The question.** Do the fitted channel models, and the Phase 4 structural signal, keep their direction and practical value on an independently collected dataset with a different sensing layout?
- **This page reports no performance.** The scoring run must check the frozen file, and reproduce its digests, before it scores anything.

## The dataset

| Field | Value |
| --- | --- |
| name | UCI Activities of Daily Living (ADLs) Recognition Using Binary Sensors |
| version | 1.0 (README, November 2013); UCI dataset 271, donated 2013-10-27 |
| source | https://doi.org/10.24432/C5J02M; archive https://archive.ics.uci.edu/static/public/271/activities+of+daily+living+adls+recognition+using+binary+sensors.zip |
| licence | CC BY 4.0 (UCI Machine Learning Repository); the README also asks for the citation below and prohibits commercial use |
| citation | Ordóñez, F.J.; de Toledo, P.; Sanchis, A. Activity Recognition Using Hybrid Generative/Discriminative Models on Home Environments Using Binary Sensors. Sensors 2013, 13, 5460-5477 |
| retrieved | 2026-09-30 |
| archive SHA-256 | `a03060857f2e9f9de2d0c7489e63bfd89b83c7d092e56e414b31f00c209964f5` |

Every file's SHA-256:

| File | SHA-256 |
| --- | --- |
| OrdonezA_ADLs.txt | `b018601885e1c62a9769f6b2ac7b47fcd51497d8880e272c1b3ff678eb63e15d` |
| OrdonezA_Description.txt | `4ce4b8e4dbbaa9208069ac184e5686c2b8fdcf97a446a8357112f3ea2bc7b3dc` |
| OrdonezA_Sensors.txt | `3ec41f2b1d15c90aec0753d02a21c9e7add0b6f761ef76f3d8d428fb2dd0fa9d` |
| OrdonezB_ADLs.txt | `d2c59c9ada5da5dbebe26c6f02f02e331fc61a5d4ec425e19ef5a9de98852d58` |
| OrdonezB_Description.txt | `727f35c52afca460dee311eff8f3897065dfd75bc755b3dbb9de6b616df4d43d` |
| OrdonezB_Sensors.txt | `b0c77812b48f3b7d81afa446c80faba35b33bd99075350e929fa34a463c49a00` |
| README.txt | `c80e86f00bbc279d09a7154ae2d0a1f7d3d72a84ec20d4d694ed4641a3341014` |

- Archive sha256 a03060857f2e9f9de2d0c7489e63bfd89b83c7d092e56e414b31f00c209964f5.
- Two single-resident homes in Spain; no timezone is given, and Europe/Madrid is declared.
- OrdonezB_Description.txt lists door PIR sensors in the kitchen, bathroom and bedroom; the recorded sensors place them in the kitchen, living room and bedroom, and the recorded rooms are used.
- **Why this dataset.** It is independently collected, single-resident by its documentation, annotated with intervals, and legally and reproducibly obtainable: a stated licence, a DOI and a stable archive; datasets without a stated licence, or with more than one resident per home, were not considered.
- **Redistribution.** None: the scoring run downloads the archive and verifies every file's digest.

## Households and periods

- **Inclusion.** Every candidate home, single-resident by the dataset's documentation.
- **Exclusion.** A home whose validation under the frozen mapping has an error other than impossible_interval, undeclared_label, undeclared_sensor, unknown_sensor; tolerated errors leave their items unscored and reported.
- **Eligibility.** At least 7 labelled days after the adaptation period, a declared timezone, and at least two channels the model can use.
- **Timezone.** `Europe/Madrid`.
- **Adaptation period.** Local midnight of the home's first annotated day, for 7 days.
- **Scored period.** From the end of the adaptation period to the end of the recording.
- **Use.** The adaptation period may inform the mapping and the adapted condition; the scored period informs nothing.

| Home | Labelled days | Scored days |
| --- | --- | --- |
| OrdonezA | 14 | 7 |
| OrdonezB | 21 | 14 |

## Ontology mapping

The mapping is `artifacts/phase5/ordonez_mapping.json`, SHA-256 `ac82ba9a9d6bf70392b14e67eea8e22ca3dd00081b7b0b372ccbc5dd2728fbb5`. Written from the README and the two description files, which list every label and sensor; the rooms and the meaning of Leaving were checked on the first seven days of each home only. Frozen before any scoring.

| Label | Outcome | Targets |
| --- | --- | --- |
| `Breakfast` | ambiguous | kitchen_activity, home_active |
| `Dinner` | ambiguous | kitchen_activity, home_active |
| `Grooming` | approximate | bathroom_activity |
| `Leaving` | approximate | away |
| `Lunch` | ambiguous | kitchen_activity, home_active |
| `Showering` | approximate | bathroom_activity |
| `Sleeping` | exact | sleeping |
| `Snack` | ambiguous | kitchen_activity, home_active, home_inactive |
| `Spare_Time/TV` | approximate | home_inactive |
| `Toileting` | approximate | bathroom_activity |

- **New values.** A label, sensor type or room first seen in the scored period is left unscored and reported; the frozen mapping is never edited.
- **Scored states.** `away`, `home_inactive`, `sleeping`, `bathroom_activity`.

### Unsupported states

| State | Why |
| --- | --- |
| `bed_awake` | no label names it |
| `home_active` | a candidate of the ambiguous labels Breakfast, Dinner, Lunch, Snack, which are not scored |
| `kitchen_activity` | a candidate of the ambiguous labels Breakfast, Dinner, Lunch, Snack, which are not scored |

## Sensor mapping

### OrdonezA

| Sensor | Native type | Type | Modality | Room | Model channel |
| --- | --- | --- | --- | --- | --- |
| `Basin.PIR.Bathroom` | PIR/Basin | approximate | motion (event) | bathroom | bathroom_motion |
| `Bed.Pressure.Bedroom` | Pressure/Bed | exact | bed_pressure (state) | bedroom | unsupported: a state sensor; the channels count events |
| `Cabinet.Magnetic.Bathroom` | Magnetic/Cabinet | exact | contact (event) | bathroom | unsupported: no bathroom_contact channel in the CASAS-trained model |
| `Cooktop.PIR.Kitchen` | PIR/Cooktop | approximate | motion (event) | kitchen | kitchen_motion |
| `Cupboard.Magnetic.Kitchen` | Magnetic/Cupboard | exact | contact (event) | kitchen | unsupported: no kitchen_contact channel in the CASAS-trained model |
| `Fridge.Magnetic.Kitchen` | Magnetic/Fridge | exact | contact (event) | kitchen | unsupported: no kitchen_contact channel in the CASAS-trained model |
| `Maindoor.Magnetic.Entrance` | Magnetic/Maindoor | exact | door (event) | hall | hall_door |
| `Microwave.Electric.Kitchen` | Electric/Microwave | unmappable | — | kitchen | unsupported: its type is unmappable: appliance power use has no modality in the observation model |
| `Seat.Pressure.Living` | Pressure/Seat | exact | bed_pressure (state) | living | unsupported: a state sensor; the channels count events |
| `Shower.PIR.Bathroom` | PIR/Shower | approximate | motion (event) | bathroom | bathroom_motion |
| `Toaster.Electric.Kitchen` | Electric/Toaster | unmappable | — | kitchen | unsupported: its type is unmappable: appliance power use has no modality in the observation model |
| `Toilet.Flush.Bathroom` | Flush/Toilet | approximate | contact (event) | bathroom | unsupported: no bathroom_contact channel in the CASAS-trained model |

Model channels with no sensor here: `bedroom_motion`, `hall_motion`, `living_motion`.

### OrdonezB

| Sensor | Native type | Type | Modality | Room | Model channel |
| --- | --- | --- | --- | --- | --- |
| `Basin.PIR.Bathroom` | PIR/Basin | approximate | motion (event) | bathroom | bathroom_motion |
| `Bed.Pressure.Bedroom` | Pressure/Bed | exact | bed_pressure (state) | bedroom | unsupported: a state sensor; the channels count events |
| `Cupboard.Magnetic.Kitchen` | Magnetic/Cupboard | exact | contact (event) | kitchen | unsupported: no kitchen_contact channel in the CASAS-trained model |
| `Door.PIR.Bedroom` | PIR/Door | approximate | motion (event) | bedroom | bedroom_motion |
| `Door.PIR.Kitchen` | PIR/Door | approximate | motion (event) | kitchen | kitchen_motion |
| `Door.PIR.Living` | PIR/Door | approximate | motion (event) | living | living_motion |
| `Fridge.Magnetic.Kitchen` | Magnetic/Fridge | exact | contact (event) | kitchen | unsupported: no kitchen_contact channel in the CASAS-trained model |
| `Maindoor.Magnetic.Entrance` | Magnetic/Maindoor | exact | door (event) | hall | hall_door |
| `Microwave.Electric.Kitchen` | Electric/Microwave | unmappable | — | kitchen | unsupported: its type is unmappable: appliance power use has no modality in the observation model |
| `Seat.Pressure.Living` | Pressure/Seat | exact | bed_pressure (state) | living | unsupported: a state sensor; the channels count events |
| `Shower.PIR.Bathroom` | PIR/Shower | approximate | motion (event) | bathroom | bathroom_motion |
| `Toilet.Flush.Bathroom` | Flush/Toilet | approximate | contact (event) | bathroom | unsupported: no bathroom_contact channel in the CASAS-trained model |

Model channels with no sensor here: `hall_motion`.

## Preprocessing

- **Adapter.** sensor_modeling.external.ordonez.read_household: each activation row an ON event at its start and an OFF event at its end; each annotation row an interval.
- **Conversion.** sensor_modeling.external.to_canonical with strict=False: events sorted, exact duplicates kept once, naive times placed in Europe/Madrid, overlapping annotations split, conflicting spans unscored, everything left out counted.
- **Windows.** The evidence resolution's 5-minute steps from each recording's first observation to its last.
- **Channels.** `bathroom_motion`, `bedroom_motion`, `hall_door`, `hall_motion`, `kitchen_motion`, `living_motion`.
- **Labels.** truth_series of the canonical segments at each step; unlabelled, unmappable, ambiguous and conflicting steps are not scored.
- **Scored windows.** Labelled steps in the scored period whose state is one of the scored states.

## Models and conditions

- **Declared.** `declared`: the filter's declared Poisson channel rates, from the external registry's default emissions, in the same recursion; nothing fitted; the reference for the transfer estimand.
- **Zero-shot.** `hurdle/population/all`: fit_samples(...)['all'] of every development home's labelled windows, 12 pseudo-windows, population SHA-256 `50e00c3da80c2ae814575b69b0423b75584c0278307cdd54554c8b3ca91da7a4`. filter_recursion over every step of the external recording from the stationary distribution, with the ontology's default transition.
- **Adapted.** pool_channels of the home's own statistics toward the population, with household_statistics(until=the end of the adaptation period), at strength 288 (the Phase 3.4 declared strength: 288 five-minute windows, 24 labelled hours).
- **Structural ensemble.** `hurdle/population/all`, `hurdle_nb/population/all`, `hurdle/population/half_a`, `hurdle/population/half_b`: the Phase 4 population-only ensemble, fitted on the 20 development homes and their fixed halves.
- **Code.** The scoring run's code must check this frozen file and reproduce every population digest before it scores.

| Population | SHA-256 |
| --- | --- |
| all | `50e00c3da80c2ae814575b69b0423b75584c0278307cdd54554c8b3ca91da7a4` |
| half_a | `dd9381bb3388e2b9ee3f2584c59f70d56f1eac12b37cbde2bdfbfba45121b0a6` |
| half_b | `f6f3b9f1e1271edf6c698e88ef8afb4aca8a542bbd6547746231e7043a6de611` |

The development homes: hh101, hh102, hh103, hh105, hh106, hh108, hh110, hh111, hh114, hh118, hh119, hh120, hh122, hh123, hh124, hh125, hh126, hh127, hh129, hh130. Splits SHA-256 `d9d52fae4415b306206717ccd942ff07a1faa039126561054d3b3f81f91feb21`.

- **Zero shot.** No parameter is fitted on external data; the adaptation period's events only run the recursion.
- **Adapted.** Each home's channel parameters are pooled toward the population on its adaptation period's labelled windows.
- **Same windows.** Both conditions, and the declared model, are scored on the same scored windows.

### Adaptation rules

- **Permitted.** The adaptation period's events and mapped labels.
- **Forbidden.** Any event or label of the scored period, and any setting chosen by looking at external results.
- **Adapted.** Each routed channel's silence probability and active mean, per state, by partial pooling.
- **Not adapted.** The transition, the prior, the mapping, the channel set, the dispersion and the ensemble.

## Evaluation

- **Primary metrics.** `balanced_accuracy`, `log_loss`, `calibration_error`.
- **Secondary metrics.** `brier`, `macro_f1`, `per_class_recall`.
- **Selective.** Error AURC of confidence and of the structural ensemble's normalised discordance over the default coverage grid, within the home.
- **Chance.** One over the number of scored states present in the home's scored windows.
- **Scoring.** Per home, over the scored windows; predictions over every latent state; a prediction of an unsupported state is an error.

- **Bootstrap.** Local calendar days of the scored period, within a home: 10,000 resamples, 95% percentile intervals, seed 0. Every model and condition is resampled on the same days. Two homes support no between-household inference; days keep within-day dependence intact.

| Quantity | Minimal difference |
| --- | --- |
| `aurc` | 0.01 |
| `balanced_accuracy` | 0.02 |
| `brier` | 0.01 |
| `calibration_error` | 0.02 |
| `chance` | 0.05 |
| `log_loss` | 0.05 |

### Estimands and criteria

| Estimand | Definition | Criterion |
| --- | --- | --- |
| T | Transfer: zero-shot minus declared, per home | Transfers when balanced accuracy favours zero-shot in both homes and log loss favours the declared model in neither; does not transfer when balanced accuracy is negligible or favours the declared model in both; inconclusive otherwise |
| A | Adaptation: adapted minus zero-shot, per home | Helps when balanced accuracy favours adapted in both homes and log loss favours zero-shot in neither; does not help when balanced accuracy is negligible or favours zero-shot in both; inconclusive otherwise |
| C | Above chance: zero-shot balanced accuracy minus chance, per home | Above chance when the difference favours zero-shot in both homes; not above chance when it is negligible or favours chance in both; inconclusive otherwise |
| S | Phase 4 direction: confidence's error AURC minus the structural signal's, per home, zero-shot | The direction survives when the AURC favours the structural signal in both homes; reverses when it favours confidence in both; inconclusive otherwise |

- **Verdicts.** Per home and metric, oriented so that positive favours the first model: favours it when the mean is at least the minimal difference and the day-bootstrap interval lies above 0; favours the second when at most minus it with the interval below 0; negligible when the interval lies within plus or minus it; uncertain otherwise.
- **Claims.** Two homes: every conclusion is about these two homes, not a population of homes.
- **Reporting.** Every home's results, every estimand and every negative result; household-level first, pooled only as description; no threshold, no retuning, no added model or metric.
