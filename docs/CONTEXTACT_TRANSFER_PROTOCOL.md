# The Phase 3 formulation on ContextAct@A4H: the protocol

Generated entirely from the frozen file `artifacts/contextact/transfer_protocol.json` by `sensor_modeling.datasets.transfer_summary.render_protocol`.

**Status: pre-specified, one-shot, on a dataset collected independently of CASAS and never scored by any model in this repository.**

- **Protocol digest.** `cea13b32c66f082335610e59786bca38122e861e40889bab631116e16b6db8cf`.
- **The question.** Whether the time prior's gain over the fitted hurdle channels in the recursion, a success on the development panel and on 43 held-out CASAS homes, holds on the one resident of ContextAct@A4H.
- **This page reports no result.**

## What had been seen

- Nobody on this project had read ContextAct@A4H before the search for an independent dataset for this test. Its README, description file, apartment plan and paper were then read in full.
- Before the inputs check, from its change logs were read: each log's time span, the sensor identifiers present, and the value tokens of the motion, presence, door and bed sensors; of the annotations, only the distinct activity labels and markers, to declare each in the mapping. No annotation's time, duration or count was read then, nor any self-annotated position or number of people.
- The adapter was run on both periods to check that the mapping declares every label, sensor type and location, and to list which sensors feed which channel. No model was run.
- Before this protocol was frozen, the inputs check, a run of the workflow on develop named in the report it writes, read both periods to write that report: each period's validation issue codes and counts, the conversion's dispositions, including annotated seconds left unscored by reason, its channels, windows, labelled windows on each day and the states that occur. It recorded the digest of the protocol as declared without the report, so every setting here is shown to precede it. It ran the recursions to check their posteriors are finite and the online pipeline to check it gives the recursion's beliefs, and ran the whole experiment once on surrogate labels assigned by the hour, keeping only that it completed; no prediction was compared with a true label.
- The development result: on the 20 development homes, cross-fitted, the time prior adds +0.061 [+0.044, +0.077] balanced accuracy to the fitted hurdle channels in the recursion, 0.517 against 0.456 (K1, a success).
- The held-out confirmation: fitted on the 20 development homes and scored once on 43 held-out CASAS homes, +0.077 [+0.070, +0.084], 0.567 against 0.490 (K1, a success); the declared model scored 0.482 and the diagnostic at I3 0.557, not distinguishable from the combination, -0.010 [-0.033, +0.014].
- On the first independent dataset, the two homes of the UCI ADL Binary dataset, the CASAS-trained model of Phase 5 did not transfer, and its balanced accuracy was not above chance in either home.

## The base protocol

- **File.** `artifacts/phase3/fitted_rates_protocol.json`, protocol digest `4f5db165af7eb9178b6154b3f01db806094955571cec1b07a92623f1e8cfc8d6`.
- **Used.** Its channel families and shrinkage, its time prior, its information sets, metrics, minimal differences and bootstrap settings; not its folds.

## Training, and its reproduction

- **Homes.** 20 development homes: hh101, hh102, hh103, hh105, hh106, hh108, hh110, hh111, hh114, hh118, hh119, hh120, hh122, hh123, hh124, hh125, hh126, hh127, hh129, hh130. The channels, the time prior and every baseline are fitted once, on all 20 development homes, exactly as the held-out confirmation fitted them; nothing is fitted on ContextAct.
- **Channels.** The refitted channels must have this digest, the held-out record's, or nothing is reported. Digest `15a63d668737ede83e3c8b10d7c97917b60849e2e253285c84a27a775b352ed6`.
- **Time prior.** The refitted prior's coefficients must be within 1e-09 of those in sensor_modeling/data/phase3_parameters.json, or nothing is reported; the shipped prior is then the one used.
- **The opt-in.** In each period the online pipeline with formulation phase3 must give the combination's beliefs to within 1e-09 at every window, or nothing is reported.

## The dataset

- **Periods.** A4H-2016-07, A4H-2016-11.
- **Mapping.** `artifacts/contextact/contextact_mapping.json`, mapping digest `d292c3a3368f7cb8aef0d37e4b4c1fc3c90e2fa409f59c92f8354bc12364de58`.
- **Reading.** Each period is read by sensor_modeling.external.contextact, validated under the frozen mapping and converted to the canonical form; a period whose validation has an error other than unpaired annotations or undeclared labels is excluded and listed. Local times are read in the Europe/Paris zone, and the hour the time prior reads is the local hour; a period whose offset from UTC changes, across a daylight-saving change, stops the run. If only one period is eligible, the estimands use its days alone.
- **Adaptation.** None: no parameter is fitted or adapted on ContextAct.

| File | SHA-256 |
| --- | --- |
| `README.txt` | `a0b91ca101a9f61628d897c7cf80c89abcac37d279c801cd84ec6c4411b25163` |
| `Variable Description.xlsx` | `66e5e89fac90ae5398f09086a3d9e4cb3388c857aebfa20c9afafc32af82b282` |
| `apartment plan.pdf` | `4e3ebcff4df06329127e6fcab2dea58a846973ee5ec33f388962a6b5af3a5be6` |
| `indoor location measurements/bsp-info.txt` | `518d2b7c7a77b7744fa41b1b42e5d0128f979c1b7ea5843f5da491fc6639a165` |
| `indoor location measurements/bsp.csv.log` | `c22735593e21ee968a2e9cec6a84224445b62871c87147011efabc75eb6f8c55` |
| `sensor measurements/July_All_Change.csv.log` | `328022acc248259c71dd432324829519ba6ce2914b62d94902577d0410c23bad` |
| `sensor measurements/July_All_Update.csv.log` | `ee7bef1619f5adb25d2ac1d016579a531266ec093c371b7050d02880e0e549ec` |
| `sensor measurements/November_All_Change.csv.log` | `e985c6254ad7ecc2dd367acfbff80b5039342543eac49209c1b3986874636afb` |
| `sensor measurements/November_All_Update.csv.log` | `8628ab9f90ca51014d91a712b9058b906119144a2b81d7985bd199461a8b90f3` |

## What the inputs report found

From `artifacts/contextact/transfer_inputs.json`, SHA-256 `de34964606b6a5f80f703b4cc3e87c08b30924f1da8e4333bce8391dc9b537ea`.

- **Eligible periods.** A4H-2016-07, A4H-2016-11.
- **Channels in A4H-2016-07.** `bathroom_motion`, `bedroom_motion`, `hall_door`, `kitchen_motion`, `living_motion`.
- **Channels in A4H-2016-11.** `bathroom_motion`, `bedroom_motion`, `hall_door`, `kitchen_motion`, `living_motion`.
- **Days with enough labelled windows.** 29.
- **Periods in which each state occurs.** `away` 2; `bathroom_activity` 2; `home_active` 2; `home_inactive` 1; `kitchen_activity` 2; `sleeping` 2.

## Models

In the recursion, as in the combined-prior experiment:

| Model | Channels | Time prior |
| --- | --- | --- |
| `filter_declared` | declared | no |
| `filter_hurdle` | hurdle | no |
| `filter_periodic` | declared | yes |
| `filter_periodic_hurdle` | hurdle | yes |

Baselines at `I3`, fitted by the matched runner: `state_frequency`, `persistence`, `logistic`, `tree`, `diagnostic`. The richest declared information set, the hour and the last three windows, is strictly less than the recursion reads.

**Missing channels.** A channel the apartment does not instrument, hall motion, has no term in the recursion; the baselines see its columns as missing, encoded as zero beside a missing indicator, and where every training home instruments it that indicator never varied in training, so they read the absent channel as a silent hall.

## Units

- **Baselines.** The matched runner scores each day as a unit, reading the hour and the last three windows from the whole period.
- **Identical windows.** On every day every model must score the same number of windows in each true state, or nothing is reported.
- **Minimum.** A day with fewer than 12 labelled windows is listed as unscored and enters no estimand.
- **Recursions.** Each period's recursion runs over the whole period from its first window, and is scored day by day.
- **Scored.** A day's labelled windows, at the base protocol's step of 300 seconds.
- **Unit.** A local calendar day of a period.

## Estimands

|  | Role | Question | Model | Reference | Rule |
| --- | --- | --- | --- | --- | --- |
| K1 | primary | What the time prior adds to the reference formulation, the fitted hurdle channels, in the recursion. | `filter_periodic_hurdle@R` | `filter_hurdle@R` | time prior |
| K2 | secondary | What the time prior adds to the declared channels in the recursion. | `filter_periodic@R` | `filter_declared@R` | time prior |
| K3 | secondary | What the fitted channels add with the time prior in the recursion. | `filter_periodic_hurdle@R` | `filter_periodic@R` | fitted channels |
| G1 | secondary | How far the supervised diagnostic, given the hour and the last three windows only, is from the combination in the recursion. | `diagnostic@I3` | `filter_periodic_hurdle@R` | gap |
| G2 | secondary | How far regularised logistic regression, given the hour and the last three windows only, is from the combination in the recursion. | `logistic@I3` | `filter_periodic_hurdle@R` | gap |
| L1 | secondary | How far the combination is from the declared model, the production filter's own emissions without the time prior. | `filter_periodic_hurdle@R` | `filter_declared@R` | level |
| C1 | secondary | Whether the combination's balanced accuracy is above chance, one over the number of states labelled that day, on days with at least two. | `filter_periodic_hurdle@R` | `chance@day` | chance |

## Criteria

- **Chance.** The combination's balanced accuracy minus chance per day, on the days with at least 2 labelled states, with a minimal difference of 0.05: above chance, below chance, not above chance when negligible, or uncertain; the days left out are counted.
- **Fitted channels.** Success when calibration error favours the model and neither log loss nor balanced accuracy favours the reference; trade-off when calibration error favours the model and balanced accuracy the reference; failure when calibration error is negligible or favours the reference; inconclusive otherwise.
- **Gap.** Read by balanced accuracy: the baseline leads, the combination leads, no gap, or uncertain; every metric's verdict is reported.
- **Level.** Read by balanced accuracy: the combination leads, the declared model leads, no gap, or uncertain; every metric's verdict is reported.
- **Multiplicity.** K2, K3, G1, G2, L1 and C1 are secondary; nothing is adjusted.
- **Primary.** K1 alone decides: success means the time prior adds to the fitted hurdle channels for this resident of an independent dataset; failure that it does not; inconclusive is reported as such. With fewer than 10 scored days, K1 is inconclusive whatever its verdicts.
- **Time prior.** Success when balanced accuracy favours the model and neither log loss nor calibration error favours the reference; failure when balanced accuracy is negligible or favours the reference; inconclusive otherwise.
- **Verdicts.** Favours model: mean >= delta and lower bound > 0; favours reference: mean <= -delta and upper bound < 0; negligible: the whole interval within (-delta, delta); uncertain: anything else.

## Descriptions

- **Recall by state.** For K1: the same verdicts with the recall minimal difference, for a state that occurs on at least 5 scored days; for fewer the difference is described, not judged, and a state labelled on no scored day is listed as never labelled.
- **Interaction.** K1 minus K2 per day, described as in the combined-prior protocol.
- **By period.** Each period's mean of every recursion's balanced accuracy and of K1, described, not judged.

## Metrics and minimal differences

| Metric | Minimal difference |
| --- | --- |
| Balanced accuracy | 0.02 |
| Log loss | 0.05 |
| Brier score | 0.01 |
| Calibration error | 0.02 |
| Recall of one state | 0.05 |
| Balanced accuracy against chance | 0.05 |

## Bootstrap

- **Confidence.** 0.95.
- **Interval.** Percentile.
- **Reason.** One resident supports no between-household inference; days keep within-day dependence intact.
- **Resamples.** 10000.
- **Seed.** 0.
- **Statistics.** Mean and median paired differences over days.
- **Unit.** Day.

## One shot

Only a run of the workflow .github/workflows/run-contextact-transfer.yml on develop counts. The workflow refuses if its results branch exists, and the run refuses if its record exists in the repository, if any pinned source file differs from the freeze, or if the installed Python or libraries are not the declared ones. Whatever it reports stands, and nothing may be changed in response and rescored as the same test. A run that stops before writing its record reports nothing; a fix made without seeing any outcome is frozen as a new protocol file that names this one and the fix, and the run repeated.

## Pinned records

| File | SHA-256 |
| --- | --- |
| `artifacts/contextact/contextact_mapping.json` | `f837a24f2b305d21914d628c0d9c0534fe39a13f7cbd9113949d9465a1db5d58` |
| `artifacts/heldout/casas-heldout-confirmation.json` | `785f6757ed9e9890dc7dfbf471660403779cc7dcc6a246586a87fd2323e2c97a` |
| `artifacts/heldout/heldout_protocol.json` | `a1196911c9f07ded4e06ddad4e2df56aff344ef8223695e456bdb657ae77b23a` |
| `artifacts/phase3/fitted_rates_protocol.json` | `f695f972fcfcd11266a8a1a720932442e8b7d455bdec7dd6304ecab41843075d` |

## What this cannot show

- Anything about a population of homes or residents: ContextAct@A4H is one resident, in one smart apartment, who was part of the research group that ran the experiment.
- Intervals for other residents: the bootstrap resamples days of one person, which are not independent of one another, so the intervals are narrower than the uncertainty about this resident, and say nothing about anyone else.
- Independence of the two periods: they are the same person and apartment, months apart, and their days are pooled.
- The states no label names: bed_awake has no label, and the ambiguous labels, Musique and Loisir, are not scored. The resident annotated part of each day, reported missing some annotations, and labelled her own activities, not the CASAS annotators' states.
- Why a failure happened: the apartment's open ground floor, its pooled living-room channel, the missing hall motion, the different annotation conventions and the resident's own routine cannot be separated from non-transfer.
- A first look at the data: the documentation, the sensor encodings and the label vocabulary were read, and the inputs report was written, before the freeze.
- A fair test of the baselines: they read the missing hall motion as a silent hall, a shift that penalises G1 and G2 alone.
- Days of equal information: every scored day counts equally, from 12 labelled windows to a whole day.
- The clock: the logs give no timezone; local time in Europe/Paris is assumed, and if they were in UTC the time prior would read hours one or two hours off.
