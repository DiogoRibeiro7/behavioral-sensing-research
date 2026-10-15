# The Phase 3 formulation on held-out CASAS homes: the protocol

Generated entirely from the frozen file `artifacts/heldout/heldout_protocol.json` by `sensor_modeling.datasets.heldout_summary.render_protocol`.

**Status: pre-specified, one-shot, on CASAS homes outside the development panel; the homes were scored once before, by the v0.2 and v0.3 production filter.**

- **Protocol digest.** `ffe1d932a477ef68d3d81bf797017c3d26eeab35158c344002d31379f0054bef`.
- **The question.** Whether the time prior's gain over the fitted hurdle channels in the recursion, a success on the development panel, holds on CASAS homes outside it.
- **This page reports no result.**

## What had been seen

- The 43 homes were scored once, in the frozen v0.3 external test, by the v0.2 production filter and the v0.3 candidate with its circadian profile: a median paired gain of +0.0091 [+0.0054, +0.0117] balanced accuracy, 37 of 43 homes improved; pooled balanced accuracy 0.496 and 0.504, log loss 4.30 and 4.17. Every per-home metric of those two models was seen, in artifacts/v03/external_primary_result.json.
- The adapter learned to read the sensor names and activities of the tm, rw, mn and ihs families while the cohort was screened, before that test; see docs/EXTERNAL_COHORT_COMPOSITION.md.
- Neither Phase 3 part, the fitted hurdle channels or the hierarchical time prior, nor any baseline or the diagnostic, has been scored on any of the 43 homes; the v0.3 candidate's time-of-day term was its circadian profile.
- Before this protocol was frozen, every recording was read once to write the inputs report it declares: each home's instrumented channels, windows, labelled windows and the states that occur. The recursions were run on the held-out homes only to check that every posterior is finite; no prediction was compared with a label.
- The development result this confirms: on the 20 development homes the time prior adds +0.061 [+0.044, +0.077] balanced accuracy to the fitted hurdle channels in the recursion, 0.517 against 0.456 (K1, a success); K2 and K3 are successes too.
- On the development homes, cross-fitted, the diagnostic at I3 scored a mean household balanced accuracy of 0.590 and a mean log loss of 2.455, logistic regression at I3 0.498 and 1.314, and the combination 0.517 and 1.728.

## The base protocol

- **File.** `artifacts/phase3/fitted_rates_protocol.json`, protocol digest `4f5db165af7eb9178b6154b3f01db806094955571cec1b07a92623f1e8cfc8d6`.
- **Used.** Its channel families and shrinkage, its time prior, its information sets, metrics, minimal differences and household bootstrap; not its folds.

## Households

- **Training.** 20 development homes: hh101, hh102, hh103, hh105, hh106, hh108, hh110, hh111, hh114, hh118, hh119, hh120, hh122, hh123, hh124, hh125, hh126, hh127, hh129, hh130. The channels, the time prior and every baseline are fitted once, on all 20 development homes.
- **Held out.** 43 homes of the frozen cohort, `artifacts/v03/external_cohort_manifest.json`. Every recording is verified against its frozen digest and read with the America/Los_Angeles zone, as every CASAS result here was; the hour the time prior reads is the recorded wall-clock hour.

| Family | Homes | Identifiers |
| --- | --- | --- |
| `hh` | 8 | hh104, hh109, hh112, hh113, hh115, hh116, hh117, hh128 |
| `ihs` | 2 | ihs07, ihs11 |
| `mn` | 3 | mn57, mn77, mn82 |
| `rw` | 5 | rw101, rw103, rw105, rw106, rw107 |
| `tm` | 25 | tm001, tm002, tm003, tm005, tm006, tm007, tm008, tm009, tm010, tm011, tm013, tm014, tm015, tm016, tm017, tm018, tm019, tm020, tm021, tm022, tm026, tm029, tm035, tm037, tm038 |

## Models

In the recursion, as in the combined-prior experiment:

| Model | Channels | Time prior |
| --- | --- | --- |
| `filter_declared` | declared | no |
| `filter_hurdle` | hurdle | no |
| `filter_periodic` | declared | yes |
| `filter_periodic_hurdle` | hurdle | yes |

Baselines at `I3`, fitted by the matched runner: `state_frequency`, `persistence`, `logistic`, `tree`, `diagnostic`. The richest declared information set, the hour and the last three windows, is strictly less than the recursion reads.

**Missing channels.** A channel a home does not instrument has no term in the recursion; the baselines see its columns as missing, encoded as zero beside a missing indicator, and where every training home instruments the channel that indicator never varied in training, so the baselines read the absent channel as silence. The homes this affects are listed under instrumentation.

**Windows.** Every window at the base protocol's step of 300 seconds; only labelled windows are scored; in every home every model must score the same number of windows in each true state, or nothing is reported.

**Unscored homes.** A held-out home with no labelled window is listed as unscored and enters no estimand; any other home without a score from every model stops the run.

## What each held-out home instruments

From the inputs report `artifacts/heldout/heldout_inputs.json`, SHA-256 `8865f8998c4f336861a63565203ec4610e5be2a0b3da92f687a33d78ced30b4d`. Every training home instruments `bedroom_motion`, `hall_door`, `kitchen_motion`, `living_motion`.

| Home | Channels | Lacks | Windows | Labelled | States |
| --- | --- | --- | --- | --- | --- |
| hh104 | 5 | — | 17,566 | 11,788 | 7 |
| hh109 | 5 | — | 17,562 | 15,924 | 6 |
| hh112 | 5 | — | 31,104 | 28,968 | 6 |
| hh113 | 5 | — | 146,696 | 128,512 | 6 |
| hh115 | 5 | — | 92,147 | 83,246 | 6 |
| hh116 | 5 | — | 17,677 | 14,806 | 6 |
| hh117 | 5 | — | 99,534 | 77,216 | 6 |
| hh128 | 6 | — | 17,563 | 11,643 | 6 |
| ihs07 | 6 | — | 20,368 | 14,325 | 6 |
| ihs11 | 5 | `hall_door` | 15,091 | 9,477 | 6 |
| mn57 | 6 | — | 7,599 | 6,293 | 6 |
| mn77 | 6 | — | 9,198 | 5,699 | 6 |
| mn82 | 6 | — | 4,189 | 3,684 | 6 |
| rw101 | 6 | — | 8,634 | 7,312 | 6 |
| rw103 | 6 | — | 8,619 | 6,525 | 6 |
| rw105 | 6 | — | 8,910 | 6,725 | 6 |
| rw106 | 6 | — | 7,339 | 4,296 | 6 |
| rw107 | 6 | — | 8,898 | 7,722 | 6 |
| tm001 | 6 | — | 17,560 | 13,340 | 6 |
| tm002 | 6 | — | 46,506 | 31,678 | 6 |
| tm003 | 6 | — | 20,249 | 12,245 | 6 |
| tm005 | 6 | — | 8,894 | 5,010 | 6 |
| tm006 | 6 | — | 8,631 | 6,486 | 6 |
| tm007 | 6 | — | 8,615 | 6,660 | 6 |
| tm008 | 6 | — | 8,634 | 7,125 | 6 |
| tm009 | 6 | — | 17,533 | 16,970 | 6 |
| tm010 | 6 | — | 17,554 | 15,049 | 6 |
| tm011 | 6 | — | 17,542 | 12,278 | 6 |
| tm013 | 6 | — | 17,557 | 16,950 | 7 |
| tm014 | 6 | — | 17,841 | 9,311 | 6 |
| tm015 | 6 | — | 8,919 | 6,332 | 5 |
| tm016 | 6 | — | 9,216 | 1,380 | 6 |
| tm017 | 6 | — | 9,206 | 1,775 | 6 |
| tm018 | 6 | — | 9,154 | 6,208 | 6 |
| tm019 | 6 | — | 8,877 | 8,051 | 6 |
| tm020 | 6 | — | 8,208 | 5,261 | 6 |
| tm021 | 6 | — | 9,215 | 7,187 | 6 |
| tm022 | 6 | — | 9,210 | 7,013 | 6 |
| tm026 | 6 | — | 9,214 | 7,174 | 6 |
| tm029 | 5 | `hall_door` | 9,213 | 4,626 | 6 |
| tm035 | 6 | — | 8,923 | 4,622 | 6 |
| tm037 | 6 | — | 9,213 | 8,336 | 6 |
| tm038 | 6 | — | 9,209 | 6,281 | 6 |

| Channel | Training homes | Held-out homes |
| --- | --- | --- |
| `bathroom_motion` | 19 | 43 |
| `bedroom_motion` | 20 | 43 |
| `hall_door` | 20 | 41 |
| `hall_motion` | 2 | 36 |
| `kitchen_motion` | 20 | 43 |
| `living_motion` | 20 | 43 |

**Shifted channels.** `hall_motion`: instrumented by fewer than half of the training homes and at least half of the held-out ones: the recursion's fitted channel rests on the few training homes that have it, and the baselines saw its columns mostly missing in training.

Held-out homes in which each state occurs: `away` 43; `bathroom_activity` 43; `bed_awake` 3; `home_active` 43; `home_inactive` 43; `kitchen_activity` 41; `sleeping` 43.

## Estimands

|  | Role | Question | Model | Reference | Rule |
| --- | --- | --- | --- | --- | --- |
| K1 | primary | What the time prior adds to the reference formulation, the fitted hurdle channels, in the recursion. | `filter_periodic_hurdle@R` | `filter_hurdle@R` | time prior |
| K2 | secondary | What the time prior adds to the declared channels in the recursion. | `filter_periodic@R` | `filter_declared@R` | time prior |
| K3 | secondary | What the fitted channels add with the time prior in the recursion. | `filter_periodic_hurdle@R` | `filter_periodic@R` | fitted channels |
| G1 | secondary | How far the supervised diagnostic, given the hour and the last three windows only, is from the combination in the recursion. | `diagnostic@I3` | `filter_periodic_hurdle@R` | gap |
| G2 | secondary | How far regularised logistic regression, given the hour and the last three windows only, is from the combination in the recursion. | `logistic@I3` | `filter_periodic_hurdle@R` | gap |

## Criteria

- **Fitted channels.** Success when calibration error favours the model and neither log loss nor balanced accuracy favours the reference; trade-off when calibration error favours the model and balanced accuracy the reference; failure when calibration error is negligible or favours the reference; inconclusive otherwise.
- **Gap.** Read by balanced accuracy: the baseline leads, the combination leads, no gap, or uncertain; every metric's verdict is reported.
- **Multiplicity.** K2, K3, G1 and G2 are secondary; nothing is adjusted.
- **Primary.** K1 alone decides: success confirms that the time prior adds to the fitted hurdle channels on held-out CASAS homes; failure does not; inconclusive is reported as such.
- **Time prior.** Success when balanced accuracy favours the model and neither log loss nor calibration error favours the reference; failure when balanced accuracy is negligible or favours the reference; inconclusive otherwise.
- **Verdicts.** Favours model: mean >= delta and lower bound > 0; favours reference: mean <= -delta and upper bound < 0; negligible: the whole interval within (-delta, delta); uncertain: anything else.

## Descriptions

- **Recall by state.** For K1: the same verdicts with the recall minimal difference, for a state that occurs in at least 5 held-out homes; for fewer the difference is described, not judged.
- **Interaction.** K1 minus K2 per household, described as in the combined-prior protocol.
- **By family.** Each family's mean of every recursion's balanced accuracy and of K1, described, not judged.

## Metrics and minimal differences

| Metric | Minimal difference |
| --- | --- |
| Balanced accuracy | 0.02 |
| Log loss | 0.05 |
| Brier score | 0.01 |
| Calibration error | 0.02 |
| Recall of one state | 0.05 |

## Bootstrap

- **Confidence.** 0.95.
- **Interval.** Percentile.
- **Resamples.** 10000.
- **Seed.** 0.
- **Statistics.** Mean and median paired differences.
- **Unit.** Household.

## One shot

The run refuses if its record exists, in the repository or on its results branch, and if any pinned source file differs from the freeze; whatever it reports stands, and nothing may be changed in response and rescored as the same confirmation. A run that stops before writing its record reports nothing; a fix that leaves this declaration unchanged, made without seeing any outcome, is disclosed beside the record and the run repeated.

## Pinned records

| File | SHA-256 |
| --- | --- |
| `artifacts/phase3/combined_prior_protocol.json` | `3a0c5cad58187609601181ae841099b9144707ba2327af4173c9442812dc52a5` |
| `artifacts/phase3/fitted_rates_protocol.json` | `f695f972fcfcd11266a8a1a720932442e8b7d455bdec7dd6304ecab41843075d` |
| `artifacts/phase3/phase3-combined-prior.json` | `57db137f57f5b298054d7a4cfe8c837d854ca9a8793329857d7803401477db59` |
| `artifacts/v03/external_cohort_manifest.json` | `6e3e157327c6eb40edfae498c2c4cc679a9e7c5b999899cf31683ed8161bc332` |
| `artifacts/v03/external_primary_result.json` | `66501d48099f48ce7c834839988830090812414b24ce76b193f84ed70783f23f` |

## What this cannot show

- Independence from CASAS: the homes share its sensors, labelling and annotation, and 8 of them its hh family.
- Why a failure happened: training is all hh, and 35 of the 43 homes come from families with other instrumentation and sensor names the adapter learned while screening them, so non-transfer and the mapping cannot be separated.
- A first look at these homes: they were scored once before, by the production filter.
- Intervals that allow for families: the bootstrap resamples homes, and 25 of the 43 are of the tm family.
- Anything about the online pipeline as shipped: it offers neither the fitted channels nor the time prior.
