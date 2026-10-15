# Threshold calibration: the protocol

The personal baseline's deviation threshold is passed far more often than it states. The [TIHM run](TIHM_ALERT_BURDEN_RESULTS.md) showed it, and an opt-in reference, `BaselineConfig.calibrated`, was built from what it showed and designed on [synthetic days](THRESHOLD_CALIBRATION_NULL.md). This page is the protocol for finding out whether that reference is better in a simulated home, generated entirely from the frozen file `artifacts/threshold_calibration/threshold_calibration_protocol.json` by `sensor_modeling.datasets.threshold_calibration_summary.render_protocol`.

**Status: pre-specified on simulated homes; descriptive on TIHM. The simulator's homes, estimands, criteria and matching rule are fixed before any simulated home is run with the calibrated reference. Nothing on TIHM is a test.**

- **Protocol digest.** `c452b592ff664a5dec6eafa600f38119ce6cc1e105dcfd02ab6cf5ca52ba145a`.
- **The question.** Whether a reference whose deviation threshold means what it says does better than the default one in a simulated home, when the two are compared at thresholds that make them report falsely as often.
- **This page reports no result.** The scoring run checks the frozen file before it runs.

## What had been seen

The reference was built from TIHM and designed on synthetic days, so neither can be the evidence that it helps. The test is made on simulated homes that had not been run with it.

- The TIHM alert-burden run and the descriptions made after it had been read, among them how often a Gaussian value passes the threshold against a median and MAD of a few others. The calibrated reference was built from that, which is why nothing on TIHM here is a test.
- The silent-home record had been read, in its totals and then home by home. Its homes are of the same design as these, from another root, and it gives what the default reference does on them: 99 behavioural alerts in 8,400 person-days of stable homes, 92 of them about a gradual drift and 49 of them raised when a day of the fifth or the sixth week closed; the step change detected in 75 of 100 homes at a median of 10 days, first by a drift verdict in 69 of the 75; and 12 of 100 stable homes meeting the detection definition in the step's window. So what the default will show here was known in advance, and what the calibrated reference will show was not.
- The alerts of simulated homes that are in neither protocol had been read when the silent-home scoring was tested and reviewed, under the default reference, as that record lists.
- The calibrated reference had been run on synthetic Gaussian days: in exploratory runs that are not recorded, in one full measurement that is not published, and then in the measurement recorded in `artifacts/threshold_calibration/threshold-null.json`. It was changed twice because of what such days showed. Its degrees of freedom went from half a degree for each residual to 0.368, because the first passed a threshold of three more often than it states. And its trend, which was fitted to the days as they are, was fitted instead to each day's distance from the centre of its own weekday, because a weekly rhythm made it report drifts.
- The calibrated reference had been exercised in unit tests on hand-built series and on one hand-built home of three event sensors. No simulated household had been run with it.
- One simulated household, seed 999, which is not a home of this protocol, had been run with the default reference, with its event sensors alone and with all its sensors, to measure run time; its numbers of observations and its run times were read, and none of its alerts, verdicts or summaries.
- The simulator's source had been read. It gives the resident a longer morning at the weekend, so the hours of sleep it plans differ by weekday.
- A draft of this protocol was reviewed before it was frozen. The reviewer read the silent-home record home by home and ran synthetic Gaussian series, and no simulated home. The number of homes, the detection net of chance, the tolerance of the criterion on what a threshold means and the window of the gradual change were set after that review.
- The scoring code had been run end to end before the freeze on six simulated homes that are not the protocol's, drawn from root 31, with the default reference alone: the default at other multiples of its own thresholds stood in for the calibrated conditions. What the default reference does on those six homes was read, and so were their days: the hours of sleep the pipeline infers there have a standard deviation of about 1.3 hours and are more than an hour longer at the weekend. The calibrated reference was not run on them.
- In a first draft the calibrated reference was to be tested at two multiples chosen on synthetic days. The measurement on synthetic days, and the weekly rhythm of the homes above, made it likely that such multiples would not match the default in a simulated home, and that the comparison would then decide nothing.
- In a second draft the two multiples were read off a quarter of the protocol's homes and tested on the rest, with a margin for how much detection could be lost. A second review, before the freeze, drew outcomes from operating curves written down by hand and found that a calibrated reference with the default's own curve would then be called better at the same detection in about one run in ten, and worse at the same false alerts as often: a multiple of the grid lies to one side of the match, and the margin let a reference slide along its own curve. The match is therefore placed on the curve itself, between two multiples, and found again in every resample. That reviewer read the record and the page of the six homes above and the silent-home record's alerts home by home, ran the calibrated reference on synthetic Gaussian series, and ran no simulated home.
- The scoring code, once the match was placed on the curve, was run once more on the six homes above, from the replays kept from the first time, in which the default at other multiples of its own thresholds stands in for the calibrated conditions. No home was run again. The stand-in has the default's own curve, and the match came out where it is known to be, with a difference of zero.
- A third review, before the freeze, read the protocol, the scoring, the trials on hand-written curves, the pages, the scripts and the tests, and both records of the measurement on synthetic days. It ran the declaration and both pages on made-up values and on twelve homes built by hand, drew 72,000 cohorts from operating curves of its own and decided the criterion on them with the scoring's own functions, and tested the unit tests against altered copies of the code. It ran no simulated home. It found that the straight line between two multiples of the grid, which then had 13, moved the criterion well past its stated error where neighbouring multiples were far apart. The grid was made three times as fine after that, from a scan of smooth curves written down by hand, in which no draw was made and no home run, and the trials were widened over where the match can fall.
- One home of the six above, seed 140752, was simulated once more with the default reference, to time a replay against a run of the pipeline. Only the times were read.
- The criterion as it now stands was then tried, before the freeze, on outcomes drawn from operating curves written down by hand, as recorded in `artifacts/threshold_calibration/threshold-calibration-planning.json`: counts of false alerts and detections with the sizes the silent-home record gives the default, for a default that finds most steps, nearly all of them or few, for a calibrated reference with the same curve, a better one and a worse one, and with the match placed from 0.45 to 1.1 times the declared thresholds. No home was simulated for it and no reference was run. Two matches were tried, and no estimand is stated at the same detection, for the reason the matching rule gives. At the same false alerts, a reference with the default's own curve was called better in 2 to 4 runs in 100 and worse in 2 to 4, where 2.5 of each are expected, and its estimate was on average from -0.003 to +0.001, where the truth is zero. The estimate's standard error was from 0.022 to 0.037.

## The option

`BaselineConfig.calibrated` is off by default. With it on, the centre stays weekday-aware; the scale is pooled over every retained day, from how far each fell from the centre the other days of its weekday would have given it; the deviation is mapped through a Student t with 0.368 degrees of freedom for each day behind the scale; the trend is fitted to those same distances and not to the days as they are, and its movement is measured against the same scale and is not mapped. See [the inference contract](inference.md).

- **Designed on.** Synthetic Gaussian days, which are not evidence about a home.
- **The measurement it rests on.** `artifacts/threshold_calibration/threshold-null.json`, SHA-256 `513c48884b69f956a1f842dbb3c7a695cd4400556bd406f2ee44843fb3c4711c`.

## Conditions

`reference@scale`: the reference, and the multiple of the declared thresholds it is run at. Both thresholds are multiplied together: at scale s the deviation threshold is 3 s and the trend threshold 3.5 s.

| Role | Condition | What it is |
| --- | --- | --- |
| default | `default@1` | The default reference at its declared thresholds: the pipeline as it ships. |
| declared | `calibrated@1` | The calibrated reference at the same thresholds. |
| same false alerts | `calibrated@` the multiple nearest the match | The calibrated reference at the multiple of the grid nearest its match at the same false alerts. |

- **Why not at the declared thresholds.** At the declared thresholds the calibrated reference reports far less of everything, so a comparison there would show fewer false alerts and fewer detections and decide nothing. Two rules can be compared when they report falsely as often.
- **Described as well.** Both references at 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 1.8, 2 times the declared thresholds: 84 conditions, those above among them.

## Where the two references are compared

- **The operating curve.** A reference's operating curve is the points its multiples give, in the order of the multiples: mean false alerts per home, and mean excess detection in each change arm, over the homes. Between two neighbouring multiples the curve is the straight line between their two points, and a place on that line has the multiple the same share of the way between the two.
- **The same false alerts.** Take the smallest multiple of the calibrated reference, which is its most sensitive thresholds, whose mean false alerts per home are at most those of `default@1`. The match is on the line from that multiple to the next smaller one, where the false alerts equal the default's. The calibrated reference's excess detection is read there, in each change arm.
- **When there is no match.** The match is bracketed when some multiple has no more false alerts than the default and the smallest multiple of the grid has more. When no multiple has so few the match is not reached, and when the smallest multiple already has so few it lies at the end of the grid. In both cases nothing is read.
- **What the rule uses.** The rule is applied to the means over the homes in hand. In the bootstrap those are the resampled homes, so the match moves from one resample to the next and the interval carries that.
- **Why the match is placed on the curve.** Where the two references match depends on how far a home's days are from Gaussian and on its weekly rhythm, which synthetic days do not settle, and no multiple of a grid falls exactly on a match. A multiple chosen on some homes and tested on others lies to one side of it, and that side decides the comparison.
- **Why not the same detection.** The other way to match two rules is where they find as much, and to read the false alerts there. No estimand does. Excess detection falls away at both ends of the grid: at strict thresholds little is found, and at loose ones the stable record is alerted on as often. So a curve can have the default's detection at two places, and where the default ships near the most its own curve finds, the same detection is found again at stricter thresholds with fewer false alerts, on the same curve. In the draws from curves written down by hand, where the default finds nearly every step, a calibrated reference with the default's own curve was called better there in 5 to 6 runs in 100, where 2.5 are expected. One with half as many false alerts again had no match there in 92 to 96 runs in 100, and where it had one its estimate was on average from -0.507 to -0.460 alerts a home, where the truth is more. False alerts are expected to fall as the thresholds rise, so the match at the same false alerts should have one place; where they do not, the rule takes the crossing at the most sensitive thresholds.
- **The condition described in full.** The tables that describe a condition in full need a multiple that was run. That is the multiple of the grid nearest the match on all the homes, the smaller of two equally near; the smallest multiple when the match lies at the end of the grid, and the largest when it is not reached. This condition is chosen from the data and describes; no criterion reads it.

## What the outcomes turn on

Under the default reference most alerts come from the trend and not from a run of deviating days: in the silent-home record 92 of 99 false alerts and 69 of 75 first detections are gradual-drift verdicts. So C1 mostly compares two trend rules: the days' own trend over a scale of a few same-weekday days, against the trend of the days' distance from their weekday over the pooled scale, at a lower threshold. The Student t, which is what makes the deviation threshold mean what it says, does not touch the trend. C2 is the criterion about the deviation threshold itself. The alert policy stands between a verdict and an alert and grades a deviation against its threshold, so a calibrated score, which is smaller than the raw one, is graded lower; how many verdicts become alerts is reported for every condition described in full. Whether a run of deviating days is called persistent or abrupt is decided, under both references, on the days as they are, so under the calibrated one that label, and no alert, can still turn on a weekly rhythm.

## The simulated homes

**Simulated: every home comes from this repository's simulator, so every result is a statement about the simulator.**

| Field | Value |
| --- | --- |
| homes | 400 |
| seed root | `20261008` |
| seeds | the first distinct values of `numpy.random.SeedSequence(seed_root).generate_state(4 * homes, uint32)` modulo 1,000,000, sorted. Two of them can be neighbouring numbers. Each is given whole to the simulator's generator and no arithmetic is done on a seed, so neighbours are as unrelated as any other two |
| days in each record | 84 |
| first day | 2024-03-04, day 0 |
| timezone | `Europe/Lisbon` |
| household | `HouseholdConfig` defaults apart from days, seed and the injected change |
| sensors | `front_door`, `bedroom_motion`, `bathroom_motion`, `kitchen_motion`, `living_motion`, `fridge_contact` |
| sensors left out | `bed_pressure`, `living_radar`, `wearable_motion`, `resident_beacon` |
| replications | 400 paired homes. `docs/SIMULATION_PROTOCOLS.md` sets a floor of 100 and asks that the number follow from the error that can be accepted |

- **How many homes.** No simulated home has been run with the calibrated reference, so the number of homes was planned on outcomes drawn from operating curves written down by hand, as recorded in `artifacts/threshold_calibration/threshold-calibration-planning.json`. C1 is planned with a standard error of 0.030 for E1, which is a standard deviation of 0.6 for a home over 400 homes. It would then be a success in 80 runs of 100 if the calibrated reference found 0.084 more of the homes than the default at the same false alerts, and in 38 of 100 if it found 0.05 more. In those draws a reference with half the default's false alerts at every detection was called better in 99 to 100 where the default finds most steps, 67 to 71 where it finds nearly all and 98 to 99 where it finds few. One with a quarter fewer was called better in 46 to 50 where the default finds most steps, 18 to 20 where it finds nearly all and 38 to 42 where it finds few. One with half as many false alerts again was called worse in 77 to 80 where the default finds most steps, 58 to 62 where it finds nearly all and 60 to 63 where it finds few. All are runs in 100. Where the default already finds nearly every step there is little more to be found, and a better reference shows itself more in false alerts, which C1 does not read.
- **The trials it was planned on.** `artifacts/threshold_calibration/threshold-calibration-planning.json`, SHA-256 `8ae86baca7d3ff90c76f70f02996b637d9a74b941936260e6f559be9d8fc970a`: trials of the criterion on outcomes drawn from operating curves written down by hand. No home is simulated in them and no reference is run.
- **Why sensors are left out.** As in the silent-home protocol, so that the homes are streams of activations, as in TIHM, and what the default reference does on them is already on record.
- **Delivery.** Every arm passes through `sensor_modeling.simulation.faults.degrade` with no loss, lateness, duplication or fault.
- **Pairing.** Every condition is replayed over the same run, so two conditions differ in the reference and its thresholds alone. A home with a change shares its seed with the same home without it and not its days: the simulator draws the sensor events after it has planned every day, and from the change on the days themselves are drawn again. So no estimand compares the two arms day by day; a detection is set against the same window of the stable record as a rate, home by home.
- **Calendar.** Every home has the same calendar. Day 0 is a Monday, both changes are declared from a Monday, and the day the clocks go forward, which is an hour short, is day 27, before either change and inside every reference.

### The pipeline

`BehaviouralSensingPipeline` over the event sensors' registry, with default emissions derived from it and every default setting; nothing is fitted.

| Setting | Value |
| --- | --- |
| step | 15 minutes |
| baseline features | `sleeping_hours`, `kitchen_activity_hours`, `bathroom_activity_hours`, `away_hours` |
| minimum day coverage | 0.5 |
| minimum day observed | 0.6 |
| baseline change point penalty | 8.0 |
| baseline deviation threshold | 3.0 |
| baseline history days | 120 |
| baseline min samples | 14 |
| baseline min scale | 0.25 |
| baseline persistence days | 3 |
| baseline trend threshold | 3.5 |
| baseline trend window | 28 |
| baseline weekday min samples | 4 |
| alert policy cooldown hours | 20.0 |
| alert policy max per window | 6 |
| alert policy min confidence | 0.4 |
| alert policy min score | 0.25 |
| alert policy storm window hours | 24.0 |

Every other setting: the defaults, which the frozen file records in full beside the digests of the source files they live in.

### One run, many thresholds

- **What.** Each arm of each home is run through the pipeline once, with the default reference. Every condition is then `sensor_modeling.online.replay_days` over the steps of that run: the same days, at the same moments, through fresh baselines and a fresh alert engine under the condition's thresholds.
- **Why.** Nothing upstream of a day's summary depends on the baseline's configuration, so the conditions differ in the baseline alone, and 84 of them cost little more than one.
- **Check.** In every home and arm the replay under `default@1` must return the verdicts and alerts of the pipeline's own run. In the 10 homes with the smallest seeds the pipeline is also run with the calibrated reference at 1, 0.6, 0.5 times the declared thresholds, in every arm, and the replay must return those runs' verdicts and alerts. If any of this fails nothing is reported.
- **Homes checked against the pipeline.** `2739`, `5651`, `5853`, `6355`, `8702`, `8848`, `9671`, `9997`, `11546`, `12179`.

### Arms

| Arm | What is injected |
| --- | --- |
| `stable` | Nothing is injected. Every behavioural alert is a false one. |
| `change` | The detection study's step change from day 56: the resident wakes 1.6 hours earlier, with 1.2 more night-time bathroom trips expected a night. |
| `small_change` | A step from day 56 of 0.5 of that size: 0.8 hours and 0.6 trips. |
| `gradual_change` | The step change's size reached gradually, declared from day 35 over 28 days. The simulator leaves day 35 itself unchanged, so the first changed day is day 36, by one part in 28. |

## Definitions

- **Behavioural alert.** An alert of kind behavioural_change, dated by the moment it was raised.
- **False alerts.** A home's behavioural alerts in the whole of its `stable` record, about any feature.
- **First changed day.** Day 56 for the two steps, and day 36 for the gradual change.
- **Window.** From local midnight at the end of the first changed day, for 21 days, or 42 days for the gradual change; closed on the left and open on the right. The pipeline raises behavioural alerts when a day closes, so an alert raised at the close of a day that holds none of the change is outside it.
- **Detected.** A home with a behavioural alert about `sleeping_hours` raised in its arm's window. The direction of the alert is not asked.
- **False detection.** The same definition met in the same home's `stable` record, in the same window, under the same condition.
- **Excess detection.** For a home, an arm and a condition: one if detected and zero if not, minus one if a false detection and zero if not. Its mean over homes is the share detected minus the share of false detections, which is what a condition detects beyond what it would have raised anyway.
- **Same false alerts.** The place on the calibrated reference's operating curve that has the false alerts per home of `default@1`, as the matching rule places it.
- **Delay.** Days from local midnight at the start of the first changed day to the first alert that counts.
- **Evaluable feature day.** A day and a feature on which the baseline gave a verdict other than insufficient data.
- **Deviating.** An evaluable feature-day whose deviation is at or past a threshold in absolute value.
- **Stated.** The share of Gaussian values at or past a threshold in absolute value: 0.1336 at 1.5, 0.0455 at 2, 0.0027 at 3.
- **Equivalent threshold.** The threshold at which Gaussian values would be deviating as often as a share that was observed: the inverse of the line above.
- **Lies below and above.** An interval lies below a value when its upper bound is strictly below it, and above a value when its lower bound is strictly above it.
- **Person days.** Homes times days.

## Estimands

| Estimand | What it is |
| --- | --- |
| E1 | The calibrated reference's mean excess detection in `change` at the same false alerts, minus that of `default@1`. |
| E2 | For `calibrated@1`, mean false alerts per home and mean excess detection in `change`, each minus the same under `default@1`, paired by home: what the calibrated reference costs and saves when the thresholds are left where they are. No criterion is stated on them. |
| E3 | In the `stable` records, the share of evaluable feature-days of `sleeping_hours` whose deviation under the calibrated reference is at or past each of 1.5, 2, 3, as a ratio of sums over homes, and the threshold it is equivalent to. Described beside it, with no criterion: the same under the default reference; at each of 1.5, 2, 3, the same before and after the reference becomes weekday-aware, and the same for each other feature; and the share at or past each of 1.5, 1.8, 2, 2.5, 3 for both references. |
| E4 | In `small_change` and `gradual_change`: the calibrated reference's mean excess detection at the same false alerts, read at the same place on its curve as E1, minus that of `default@1`. No criterion is stated on them. |
| E5 | Descriptions under each condition described in full, with no criterion: false alerts per home and per person-day, by feature, by the verdict that raised them and by the week of the day whose close raised them; in each change arm the share of homes detected, the share of false detections, the mean excess detection and their paired differences from the default, the pooled median delay, and the verdict and the direction of each home's first alert that counts; the same shares when the alert must also be of a decrease; detections by week of the window; and in every arm the change verdicts by kind, how many raised an alert, and how many did not because the day was not attributable enough. |
| E6 | Under every described condition: mean false alerts per home and, in each change arm, the share detected and the mean excess detection. These are the two references' operating curves, and no criterion is stated on them. Beside E1 they are read one more way, fixed here, for each change arm on its own: for each multiple the default reference is run at, whether some multiple of the calibrated one has no more false alerts and no less excess detection; and the same question the other way round. That reading compares estimates on the same homes, takes the best of the multiples and carries no uncertainty, so it describes the curves and tests nothing. |

## Criteria

Fixed before any simulated home was run with the calibrated reference. Both are decided on all the homes.

| Criterion | Claim | How it is decided |
| --- | --- | --- |
| C1 | More is detected at the same false alerts | Uninformative when the mean excess detection in `change` under `default@1` is below 0.2; otherwise not bracketed when the match is not bracketed on the homes as they are; otherwise success when E1's interval lies above zero, failure when it lies below zero, and inconclusive otherwise. |
| C2 | The threshold means what it says | Decided for the calibrated reference at each of 1.5, 2, 3, on the interval of the threshold E3's share is equivalent to: it holds when the interval lies within 0.25 of the threshold on both sides; it does not hold when the interval lies wholly outside that band; inconclusive otherwise. The verdicts are points on one curve, since a day's deviation does not depend on the threshold it is read against. |

The margins, and where each comes from:

- **Informative detection.** 0.2 of excess detection: below it the default finds too little beyond chance for a comparison to mean anything.
- **Calibration tolerance.** 0.25 on the scale of the threshold. At a threshold of 3 that is a share between 0.0012 and 0.0060, and at 1.5 between 0.080 and 0.211.

How the criteria are read:

- The calibrated reference is better when C1 is a success: at the false alerts of the default as it ships, it finds more of the step change.
- It is worse when C1 is a failure.
- Anything else is not shown, which is not the same as no difference.
- A match that is not bracketed is reported as such, with whether it was not reached or lies at the end of the grid, and no claim follows from it.
- C2 is decided threshold by threshold. The option's claim holds on these homes when it holds at all three, does not hold when it does not hold at one of them, and is not shown otherwise.
- C2 is about the option's claim and not about its usefulness: a threshold can be miscalibrated on a simulated home and still be the better rule, and the reverse.
- The intervals are 95% intervals, one for each estimand, and are not adjusted for there being two criteria. Only C1 and C2 are confirmatory; everything else is description.
- C1 is stated at one point of the default's curve, the one that ships. A reference that is better there can be worse at another level of false alerts, and E6 is where that would be seen.
- False alerts are counted over a record of 84 days, and half of the default's come in the fifth and sixth weeks, when its weekday reference rests on four or five days. Over a longer record the default would raise fewer a day, and the match would lie elsewhere on the calibrated reference's curve.

## What synthetic days lead one to expect

Where the record of the measurement on synthetic Gaussian days puts the calibrated reference at the default's false reports, as a multiple of its grid and by its own rule, on days with no weekly rhythm and on days with one, and how often each reference then reports a change in the three weeks from a step: with no step added, and with a step of two standard deviations. A simulated home's days are not Gaussian, so these are what the match found here can be read against, and they decide nothing.

**Days with no weekly rhythm.**

| Reference | Multiple | A change verdict with no step added | With a step of two standard deviations |
| --- | --- | --- | --- |
| default, as declared | 1 | 8.5% | 84.2% |
| calibrated, at the same false alerts | 0.55 | 5.3% | 96.9% |

**Days with a weekly rhythm.**

| Reference | Multiple | A change verdict with no step added | With a step of two standard deviations |
| --- | --- | --- | --- |
| default, as declared | 1 | 14.5% | 86.0% |
| calibrated, at the same false alerts | 0.5 | 9.3% | 98.9% |

## Uncertainty

- **Unit.** Homes.
- **Resamples.** 5000.
- **Confidence.** 0.95.
- **Seed.** 0.
- **Interval.** Percentile.
- **Shares.** A share of homes has a Wilson interval, since homes are independent; a mean of excess detections and every paired difference are resampled over homes.
- **Matched.** E1 and E4 are recomputed from the start in every resample: the calibrated reference's curve and the default's point are means over the resampled homes, and the match is placed on that curve. A resample in which the match is not bracketed has no value and counts against whichever claim a bound could support: above every value when the upper bound is taken, and below every value when the lower one is. The bounds are resampled values, with no interpolation between neighbours, that leave 2.5% of the resamples outside on each side. So a bound does not exist when more than that share of the resamples have no match, and no claim can rest on it. How many resamples were bracketed, not reached and at the end of the grid is reported.
- **Ratio.** A share of feature-days is a ratio of sums over homes, and each resample recomputes both sums. The interval of its equivalent threshold is the interval of the share, carried through the inverse.
- **Median delay.** Pooled over the homes that detected; a resample in which no home detected has no median and is left out, and how many remained is reported.
- **Monte carlo error.** For a mean paired difference, the standard deviation of the paired differences over the square root of the number of homes. For E1 and E4, the standard deviation of the resampled values over the resamples in which the match was bracketed.

## What the simulator cannot show

- The simulated resident's days are drawn from distributions this project wrote down. How far a real day's hours are from them, and so how a threshold behaves on a real home, is not something a simulated home can say.
- One size of step, half of it and one gradual change are injected, in one feature's direction, from one weekday. A reference that finds these may not find another kind of change.
- The homes have event sensors alone. With sensors that report on a cadence the day's hours are inferred differently.
- Every home has the same calendar, so nothing here varies the weekday a change begins on or where the short day of the clock change falls.
- The match is placed on the same homes it is read on. The interval carries that, and what it does not say is how a multiple chosen on one population of homes would do on another: these homes come from one simulator with one set of settings.
- Between two multiples the curve is taken to be a straight line, which can lie to either side of the true curve. The grid is fine where a match is likely, so that the gap is small beside the interval, and the trials on curves written down by hand say how often it moved a verdict there. Where the grid is coarse, above 1.5 times the declared thresholds, it can move one more often.
- No estimate is given of how many fewer false alerts the calibrated reference raises at the same detection, for the reason the matching rule gives.

## TIHM

**A description, not a test: the calibrated reference was built from what these homes showed.**

| Field | Value |
| --- | --- |
| dataset | TIHM: An Open Dataset for Remote Healthcare Monitoring in Dementia |
| citation | Palermo, F.; Chen, Y.; Capstick, A.; Fletcher-Lloyd, N.; Walsh, C.; Kouchaki, S.; True, J.; Balazikova, O.; Soreq, E.; Scott, G.; Rostill, H.; Nilforooshan, R.; Barnaghi, P. TIHM: An open dataset for remote healthcare monitoring in dementia. Scientific Data 10, 606 (2023). https://doi.org/10.1038/s41597-023-02519-y |
| licence | CC BY 4.0; the dataset's repository asks that Surrey and Borders Partnership NHS Foundation Trust and Howz be acknowledged in any publication or use |
| archive SHA-256 | `368d642b4cdc680d0706abfd3c8b2e5387824d9737ad33f50531bafa6a4bf5a1` |
| step | 10 minutes |
| conditions | all 84 described conditions |
| silent-home rule | `off`, `h12` |
| homes checked with the calibrated pipeline | the first 5 by identifier |
| everything else | as frozen in `artifacts/tihm/alert_burden_protocol.json` |
| published record `artifacts/silent_home/silent-home-tihm.json` | SHA-256 `96798aa2e573f281477faad624ee7369ff2508d5a8631c7003f7f75218bae6a9` |
| published record `artifacts/tihm/tihm-alert-burden.json` | SHA-256 `5a5bb24167c8f089de3138f9c35ec6e07cd902ba385999f172458e40b40a64ff` |

- **Replay.** Each home is run through the pipeline with the default reference, once with the silent-home rule off and once with it on at 12 hours, and every described condition is replayed over each run.
- **Check.** The replay under the default reference must return each run's verdicts and alerts. The homes run must be the homes of the published alert-burden record, each once, and with the rule off every one of them must have the monitored days and the behavioural alerts that record gives it. In the 5 homes that come first by identifier, with the rule off, the pipeline is also run with the calibrated reference at the declared thresholds, and the replay under `calibrated@1` must return that run's verdicts and alerts. Under both settings of the rule the totals must be those of the published records: 2,850 monitored days and 183 behavioural alerts with the rule off; 2,850 monitored days and 49 behavioural alerts with the rule on. Otherwise nothing is reported.
- **Reported.** Monitored, usable and evaluable days; the share of evaluable feature-days that are deviating, for each feature and each reference, beside what the threshold states, and the threshold it is equivalent to; under every described condition: change verdicts by kind, and behavioural alerts in all and by the verdict that raised them; and per home under `default@1` and `calibrated@1`.
- **Not reported.** Any relation to the dataset's labels. Whether what the pipeline raises relates to what a clinical team verified is the alert-burden protocol's question. Those labels have been read against the pipeline's output before, and more conditions would be more looks at them.
- **What it cannot say.** Fewer alerts on these homes are not better alerts. Nothing here says whether an alert that is no longer raised was a false one.
- **Redistribution.** None: the run reads an extracted download and verifies every file's digest.
- **Acknowledgement.** Surrey and Borders Partnership NHS Foundation Trust and Howz, as the dataset asks.

## Reporting

Every estimand, criterion, arm, condition and home, whatever it shows; no multiple, margin, window or rule is changed after scoring. The first run of this protocol that completes is its result. A run that stops before it completes, refused by a check or interrupted and not resumed with the same code, is not read, and the record of the run that completes lists every such attempt and why it stopped.

## The code the protocol was frozen against

The frozen file records the content of 193 source files: every module of the package, 190 of them, since a home's days pass through most of it before a threshold is asked anything, and the scripts that freeze and run the protocol, `scripts/freeze_threshold_calibration.py`, `scripts/run_threshold_calibration.py`, `scripts/run_threshold_calibration_tihm.py`. It also records the versions of the numerical libraries and every default setting of the pipeline, the baseline, the alert policy, the health monitor, the context estimator and the simulated household. None of this is part of the protocol's digest. The run compares all of it with what it uses, and refuses to go on when anything differs unless it is told why; the record then says what differed and why.

| Library | Version at the freeze |
| --- | --- |
| `numpy` | 2.5.3 |
| `pandas` | 3.0.6 |
| `python` | 3.13.16 |
| `scipy` | 1.18.1 |

## The homes

The seeds of the homes, in order.

`2739`, `5651`, `5853`, `6355`, `8702`, `8848`, `9671`, `9997`, `11546`, `12179`, `19303`, `21304`, `21825`, `27183`, `27748`, `31682`, `37447`, `39822`, `42093`, `43617`, `47220`, `47479`, `47989`, `48813`, `51820`, `52915`, `55096`, `56751`, `57152`, `58502`, `62107`, `65886`, `67745`, `68996`, `76079`, `76696`, `83524`, `85809`, `88439`, `90454`, `90820`, `91388`, `92395`, `94990`, `100084`, `104628`, `105773`, `106241`, `106474`, `107709`, `110364`, `118138`, `121283`, `121411`, `122227`, `126648`, `128131`, `134755`, `136867`, `139894`, `140274`, `140982`, `141813`, `146411`, `149993`, `154189`, `155695`, `157670`, `158716`, `160877`, `169741`, `169855`, `171201`, `173555`, `173612`, `173933`, `174894`, `179079`, `180015`, `181543`, `183567`, `184994`, `188530`, `188540`, `199972`, `200339`, `204645`, `205981`, `208031`, `214386`, `215367`, `215913`, `220349`, `220622`, `221133`, `227237`, `228566`, `230657`, `231026`, `232950`, `235188`, `236653`, `237102`, `244321`, `244358`, `246776`, `255440`, `257095`, `259724`, `264853`, `267045`, `268609`, `270589`, `270749`, `270977`, `279584`, `283827`, `285926`, `289741`, `291482`, `292467`, `293252`, `295886`, `298680`, `301585`, `301694`, `303808`, `304438`, `307587`, `310193`, `314515`, `315485`, `324638`, `326842`, `331938`, `341463`, `341980`, `352266`, `355613`, `355688`, `356369`, `357486`, `363649`, `364021`, `365284`, `367337`, `368874`, `369368`, `370642`, `372611`, `378228`, `381584`, `382283`, `382410`, `385901`, `386477`, `389958`, `396462`, `396891`, `401263`, `401857`, `404881`, `406218`, `408289`, `414522`, `415097`, `416100`, `416626`, `417700`, `419212`, `419875`, `423143`, `428010`, `437041`, `437093`, `439788`, `443509`, `446620`, `447129`, `448316`, `450255`, `450464`, `454328`, `454917`, `455628`, `457867`, `459048`, `462399`, `462581`, `463618`, `465597`, `466061`, `466593`, `469797`, `470845`, `476259`, `479947`, `485263`, `487813`, `488256`, `489339`, `489765`, `491676`, `493469`, `495693`, `503108`, `505093`, `510121`, `510775`, `514117`, `515263`, `517859`, `521856`, `523513`, `526254`, `529420`, `529717`, `530800`, `535820`, `536540`, `536965`, `537910`, `539247`, `539710`, `540733`, `546343`, `547511`, `548271`, `552367`, `552449`, `553937`, `556711`, `561690`, `569533`, `574217`, `577096`, `581655`, `587244`, `588205`, `592444`, `592877`, `593058`, `594021`, `605955`, `606447`, `610200`, `610412`, `611650`, `614080`, `615335`, `615428`, `616592`, `617539`, `622882`, `627111`, `629824`, `630171`, `630388`, `634567`, `634958`, `637267`, `639115`, `639592`, `640147`, `640588`, `642609`, `644266`, `652543`, `655381`, `656768`, `659143`, `664767`, `665280`, `667809`, `675715`, `676881`, `677838`, `681227`, `687862`, `688129`, `689250`, `691751`, `691960`, `692832`, `696521`, `700995`, `702375`, `703352`, `703985`, `706934`, `715601`, `718217`, `718323`, `724660`, `724847`, `725162`, `726585`, `726586`, `729734`, `730606`, `735536`, `736869`, `738771`, `739904`, `740233`, `740988`, `741223`, `741621`, `741794`, `746280`, `747005`, `750981`, `751216`, `753215`, `755661`, `756070`, `759492`, `761017`, `763505`, `765154`, `765924`, `766678`, `771497`, `771901`, `775450`, `775494`, `777285`, `779792`, `784245`, `785512`, `786233`, `792351`, `793294`, `793690`, `797406`, `802139`, `803132`, `806372`, `807699`, `808343`, `809398`, `814926`, `828390`, `828917`, `833822`, `838156`, `848994`, `855516`, `856818`, `859554`, `860088`, `864996`, `866489`, `866976`, `868131`, `871256`, `875112`, `876384`, `878702`, `879288`, `880327`, `880620`, `883658`, `885945`, `889690`, `895871`, `897500`, `898379`, `907581`, `909593`, `911149`, `913820`, `914904`, `916324`, `917510`, `917558`, `918948`, `918954`, `918997`, `922196`, `925373`, `928669`, `936488`, `937213`, `940363`, `944686`, `945003`, `947254`, `947749`, `948358`, `953717`, `957877`, `958206`, `975601`, `985185`, `988419`, `988559`, `991264`, `992280`, `993400`.
