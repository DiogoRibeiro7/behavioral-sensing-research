# The pipeline's hours of sleep beside a sleep mat: the protocol

Every alerting study in this repository counts its detections on `sleeping_hours`, the hours of a day the pipeline's state filter gives to sleeping. In 17 of the 56 homes of the [TIHM dataset](TIHM_ALERT_BURDEN_RESULTS.md) a mat under the mattress recorded each minute a person was in bed. This page is the protocol for setting the two side by side, generated entirely from the frozen file `artifacts/sleep_mat/sleep_mat_protocol.json` by `sensor_modeling.datasets.sleep_mat_summary.render_protocol`.

**Status: pre-specified measurement on a real cohort. The days, the homes, the estimands and the criteria are fixed before any value of the pipeline is set beside any record of the mat. Nothing in the pipeline is changed or fitted.**

- **Protocol digest.** `aceb56b37cc7dfa90650a1ac21b7795e08dfbf00f6be091f8a99f88d9ae72aae`.
- **The question.** Whether the pipeline's `sleeping_hours`, the hours of a day its state filter gives to sleeping, follows the sleep a mat under the mattress records, on the days the home's sensors reported: from one day to the next within a home, and in level.
- **This page reports no result.** The run checks the frozen file, the pinned records and the dataset's files before it runs.
- **The code it was frozen against.** The digests of 197 source files, every default setting, and numpy 2.5.3, pandas 3.0.6, python 3.13.16, scipy 1.18.1.

## What had been seen

- The mat's file alone, for its structure: its columns, its four stages, its 461,423 records in 17 homes from 2019-04-01 to 2019-06-30, one record a minute with no repeated minute, and no missing value.
- The mat's coverage alone: the days each home has records on, the days whose neighbouring days also have records, and each home's median hours asleep and in bed on a day. 835 home-days have a record, 771 of them with records on the day before and the day after, and 14 homes have at least 14 such days. Four homes have a median under five hours asleep on the days they have records: 0f352 (3.7 asleep, 5.3 in bed), 16f4b (4.2, 9.4), d7a46 (2.6, 6.1) and f220c (1.8, 7.9).
- The share of each home's minutes in bed on its mat-observed days that the mat stages as asleep: under 0.5 in 16f4b, d7a46 and f220c, and 0.74 or more in every other home. E12's cut was set from it.
- Beside those day counts, each mat home's monitored and usable day counts in the published alert-burden record. That compares how many days each source has, not what either says about them.
- The pipeline's output on TIHM had been read in the earlier studies: its alerts, its verdicts, its calendar, the silent days and the median 23.8 hours of sleep it infers on them. On 16 June 2019 all 47 monitored homes are silent. No value of the pipeline had been set beside any record of the mat, for any home or day.
- The dataset paper's description of the mat: per-minute heart rate, breathing rate and sleep state while a person is in bed. It does not say whether any file's timestamps are local or UTC.
- The interval was chosen on synthetic data, in the planning record, before the protocol was frozen. The planning was run twice: the second run added the outlying homes after the review below.
- The code was rehearsed end to end on data that is not the comparison's: one TIHM home without a mat, run with the rule off and on and paired with a mat of random minutes generated for it, and one simulated home of ten days from seed 999, which is not a seed of the reference. Their run times and that every step completed were read, and none of their values.
- An independent review of the draft read the mat's file alone, the published TIHM records and pages, and the first and last activity timestamp of each mat home; it ran no pipeline on TIHM. It found that the outage of 16 June falls on mat-observed days in 11 of the 14 homes, and the draft's primary was changed to leave days with no activity record out.
- A second review of the revision read the mat's file alone and, from the activity file, each mat home's activity timestamps, to find gaps of 12 hours or more on mat-observed days; it ran no pipeline on TIHM and read no value of the pipeline. Counted again from the same inputs: in the 14 homes with at least 14 mat-observed days, 737 mat-observed days have an activity record and are neither a home's first nor its last activity day, and 37 of them hold such a gap, 18 on 15 or 17 June, the edges of the server failure, in 11 homes. The primary was changed again to leave out every day the run with the rule on flags as touched by a home-wide silence. The review also ran simulated homes from seeds 1 to 14, which are not seeds of the reference, without the pipeline.
- The two checks of the runs were rehearsed on six TIHM homes without a mat before the freeze: with the rule off they gave every count of their rows in the published record, and with it on the silent-home record's refused days and silence alerts. Nothing else of those runs was read.

## The data

TIHM: An Open Dataset for Remote Healthcare Monitoring in Dementia. Palermo, F.; Chen, Y.; Capstick, A.; Fletcher-Lloyd, N.; Walsh, C.; Kouchaki, S.; True, J.; Balazikova, O.; Soreq, E.; Scott, G.; Rostill, H.; Nilforooshan, R.; Barnaghi, P. TIHM: An open dataset for remote healthcare monitoring in dementia. Scientific Data 10, 606 (2023). https://doi.org/10.1038/s41597-023-02519-y Licence: CC BY 4.0; the dataset's repository asks that Surrey and Borders Partnership NHS Foundation Trust and Howz be acknowledged in any publication or use.

| File | SHA-256 | Read |
| --- | --- | --- |
| Activity.csv | `ead7b1e3c2a91fbd679a909bff3a309a6465ae1ad18528dae57a4cf8599cafca` | yes |
| Demographics.csv | `15b13a680da6ee3790dfadc25d3a6102070f6e8fe1a47ea6da8f64b6c4b757d4` | yes |
| Labels.csv | `39495a08fdf1ff8f9b9b88bc98483a565c4433f58713ba3e4d0846e4449b846b` | yes |
| Physiology.csv | `377b3170b4b650795648e18258f30f9b985c214937823e9c9ea405c66b4ba696` | pinned, not read |
| Sleep.csv | `f1d5a6980263a55a8b69841e13bb349ce6e79152c15bdde0beffeea15293b9fe` | yes |

Redistributed: no: the records hold per-home summaries, never a day's values.

## The mat

An under-the-mattress sleep mat. The dataset paper says it records heart rate, breathing rate and a sleep state each minute while a person is in bed. Its file is `Sleep.csv`, with the columns `patient_id`, `date`, `state`, `heart_rate`, `respiratory_rate`, `snoring` and timestamps as `%Y-%m-%d %H:%M:%S`.

- **Stages.** `AWAKE`, `LIGHT`, `DEEP`, `REM`; asleep: `LIGHT`, `DEEP`, `REM`.
- **Clock.** Its timestamps are read as local time in Europe/London, as the adapter reads the activity records. The dataset does not say which clock either file is in. Every day of the release is in British Summer Time, so no day changes its offset.
- **Standing.** The device's own stages, not validated here. The mat is the reference because the pipeline never sees it, not because it is right.

## The pipeline

`sensor_modeling.datasets.tihm_experiment.run_household`, as the alert-burden run calls it: the contract, then `BehaviouralSensingPipeline` with default emissions derived from each home's registry; nothing is fitted. Its step is 10 minutes; everything else is as frozen in `artifacts/tihm/alert_burden_protocol.json`.

| Run | What it is |
| --- | --- |
| `off` | The silent-home rule off: the pipeline as it ships, and the run the alert-burden record was made from. |
| `rule_12h` | `HealthConfig.home_silence_horizon` at 12 hours, as in the silent-home results. |

| Setting | Value |
| --- | --- |
| day coverage a usable day needs | 0.5 |
| day observed a usable day needs | 0.6 |
| baseline: deviation threshold | 3.0 |
| baseline: min samples | 14 |
| baseline: persistence days | 3 |
| baseline: trend threshold | 3.5 |
| baseline: trend window | 28 |
| alert policy: cooldown hours | 20.0 |
| alert policy: min confidence | 0.4 |
| alert policy: min score | 0.25 |

**The check.** Nothing is reported unless both runs reproduce the published records:

- **`off`.** Every mat home must give every count of its row in the published record, `artifacts/tihm/tihm-alert-burden.json`: monitored, usable and evaluable days, events, behavioural alerts and those about the hours of sleep, alert days, deviating days, label days and alerts about the system and the data.
- **`rule_12h`.** Every mat home must give the days the baseline refused because of the rule and its silence alerts as the silent-home record has them at 12 hours, `artifacts/silent_home/silent-home-tihm.json`, and the same monitored days as the run with the rule off.
- `artifacts/silent_home/silent-home-tihm.json`: `96798aa2e573f281477faad624ee7369ff2508d5a8631c7003f7f75218bae6a9`.
- `artifacts/tihm/tihm-alert-burden.json`: `5a5bb24167c8f089de3138f9c35ec6e07cd902ba385999f172458e40b40a64ff`.

## Definitions

- **Constant values.** A home whose values on either source are all equal on the days an estimand reads has no correlation. It counts as 0 in E1, E5, E6 and E11, as not following, and its difference counts in E2 as any other.
- **Day.** A local calendar day in Europe/London, as the pipeline closes them.
- **Difference.** The pipeline's value minus the mat's, in hours.
- **Included home.** A mat home with at least 14 matched days in the analysis. Inclusion is decided in each analysis, and again at each shift of E10. E6 counts days with a verdict from both baselines instead; E7 and E8 read every mat home.
- **Mat in bed.** Every record of the day, in hours.
- **Mat observed day.** A day with at least one record of the mat, whose day before and day after also have one. A day on the edge of a stretch of records is left out because part of its night may be outside the stretch. A day with no record cannot be told from a day the mat did not work, so it is left out too.
- **Mat sleep.** The day's records whose state is one of LIGHT, DEEP, REM, in hours: a record is a minute.
- **Matched day.** A usable day that is mat-observed and is neither the home's first nor its last monitored day, which are partial. In an analysis that leaves silenced days out, it is also not a silenced day.
- **Pipeline value.** The expected hours of `sleeping` in the day's summary: the sum over the day's steps of the filter's probability of sleeping, times the step. The summary is made from the day's own estimates, so the step that spans midnight is in neither day, and a day covers at most 24 hours less one step.
- **Silenced day.** A silent day, or a day that the run with the rule on at 12 hours flags as having lost time to a silence of the whole home at least that long. Only the flag is taken from that run: the rule changes how far the sensors are trusted after a silence, so its values can differ.
- **Silent day.** A monitored day on which no activity record of the home falls.
- **Usable day.** A day whose summary passes `DailySummary.is_usable` with the pipeline's defaults; these are the days the baseline is given.

## The analyses

| Analysis | What it is |
| --- | --- |
| `off` | The rule off, silenced days left out. The primary analysis: it asks the question on the days the sensors reported throughout. A silent day is read as nearly a whole day asleep, which the alert-burden descriptions measured and the silent-home rule exists for, and a day partly lost to a silence, as on the edges of the server failure of 15 to 17 June, is read with too much sleep the same way; kept, they would measure that again. A long night with no movement anywhere in the home is left out with them. |
| `off_with_silent_days` | The rule off, every usable day: the pipeline as it ships and as its baseline sees the days. A description. |
| `rule_12h` | The rule on at 12 hours, silenced days left out: the pipeline's own values with the rule on, which refuses those days itself. A description. |

## Estimands

| Estimand | Definition |
| --- | --- |
| E1 | The mean over included homes of the within-home Spearman correlation between the pipeline's value and the mat's sleep on matched days, in the primary analysis. It is the rank of a day among the home's own days that a personal baseline reads, so the correlation is within homes and every home counts once. It is on the days' values, so a trend or a weekly rhythm both sources share counts as following. |
| E10 | E1 and E2 with the mat's clock shifted by -1 and by +1 hours, every definition otherwise unchanged. +1 stands for a mat clock in UTC while the activity clock is local, -1 for the reverse. A shift of an hour moves little of a night across midnight, so E10 says how much the result depends on the clock, and cannot find which clock is right. |
| E11 | An alignment of the two clocks from the inputs, which reads no value of the pipeline, only the days the primary analysis matched: at each lag of -3, -2, -1, +0, +1, +2, +3 hours, the mean over included homes of the within-home Spearman correlation between the home's activity records in each local hour of its matched days and the mat's minutes in bed in the same hour with its clock shifted by the lag. A person in bed moves little, so the correlation is expected to be most negative where the clocks agree. Whatever E11 shows, every other estimand keeps the mat's clock as read. |
| E12 | E1 to E5 in the primary analysis without the homes in which the mat stages less than 0.6 of the minutes in bed on its observed days as asleep. From the mat's file alone these are 16f4b, d7a46, f220c; the run refuses to go on if the file gives others. |
| E2 | The mean over included homes of the home's mean difference from the mat's sleep, in the primary analysis. |
| E3 | E1 and E2 against the mat's hours in bed. |
| E4 | Limits of agreement with the mat's sleep and with its hours in bed: E2 plus and minus 1.96 times the square root of the variance of the homes' mean differences plus the mean over homes of the variance of their differences about their mean. |
| E5 | The mean over included homes of the within-home Pearson correlation, against each reference. |
| E6 | The deviations a personal baseline gives each source, in the primary analysis. The mat's sleep on its observed days is passed in order through an `AdaptiveBaseline` with the default configuration. The pipeline's baseline was passed its own usable days, silent days among them, so the two histories differ. On matched days on which both have a verdict other than insufficient data: the mean, over homes with at least 14 such days, of the within-home Spearman correlation between the two deviations; and, pooled, of the days on which the pipeline's deviation is at or past 3 in absolute value, the number on which the mat's deviation has the same sign, and the number on which it is also at or past that value. |
| E7 | The behavioural alerts about `sleeping_hours` raised in the mat homes with the rule off, counted by the verdict they were raised for. An alert for an abrupt or a persistent change on a matched day of the primary analysis is judged two ways: whether the day's mat sleep is on the same side of the home's median mat sleep over its matched days as the pipeline's deviation, and whether the mat baseline's deviation that day, where E6 gives it a verdict, has the same sign; a mat deviation of exactly zero is neither. An alert for a drift, or on a day that is not matched, is counted and not judged. |
| E8 | The silent days of the mat homes with the rule off: how many there are, how many are mat-observed, and on those, the median pipeline value and the median hours of mat sleep and in bed. |
| E9 | E1 to E5 in the analyses `off_with_silent_days` and `rule_12h`. |
| S1 | The simulated reference: E1, E2, E4 and E5 against the simulator's true hours of sleep on simulated homes. |

## Criteria

| Criterion | Claim | Rule |
| --- | --- | --- |
| C1 | The pipeline follows the mat | The primary criterion, on E1. Follows when the interval lies above 0.5; does not follow when it lies below 0.5; inconclusive otherwise. |
| C2 | The pipeline agrees in level | Secondary, on E2, and read whatever C1 reads. Agrees when the interval lies inside (-1, +1) hours; does not agree when it lies wholly outside [-1, +1]; inconclusive otherwise. |

- **Estimable.** A criterion is not estimable when fewer than three homes are included.
- **Multiplicity.** C1 is the only primary criterion. The two criteria are not adjusted for each other. E3 to E12 and S1 are descriptions: those that are means over homes have intervals, and the pooled counts of E6, E7 and E8 have none.

### The margins

- **Level.** 1 hour, a declared convention of the order of the changes the detection studies inject: the step change has the resident wake 1.6 hours earlier, with 1.2 more night-time bathroom trips. A bias that size changes what a day's hours say about a person. It does not touch a personal baseline, which reads each home against itself, so C2 is secondary.
- **Tracking.** 0.5, a declared convention. If within a home the two sources' days were jointly Gaussian and stationary with correlation r, a day the pipeline put z standard deviations from the home's centre would be expected r times z from it on the mat, and at 0.5 a deviation at the baseline's threshold of 3 would stand for one of 1.5 in sleep. The baseline reads each day against a rolling, weekday-aware reference and not the whole period, so this is a guide, not a derivation. The margin is on the mean over homes of a finite-sample Spearman correlation: in the planning trials, when every home had a Gaussian correlation of 0.5 its true value was 0.47, and when homes were spread about 0.5 by 0.6 on Fisher's scale 0.39.

### What the planning says to expect

- **C1.** With the three homes the mat stages as mostly awake not following at all and the rest at a Gaussian correlation of 0.6, the true value of E1 was 0.45 in the planning: they alone can hold C1 below its margin, which is why E12 is declared.
- **C2.** The homes' mean differences are expected to spread by two hours or more: from the mat alone, the median hours asleep of the 14 homes with enough mat-observed days have a standard deviation of 2.5 hours between homes, and the published per-household median of the pipeline's hours of sleep runs from 5.6 to 20.2. With no bias and that spread, the planning read agrees in 5% of studies, and with the three homes the mat stages as mostly awake four hours further off, almost never; with a spread of one hour it would have read agrees in 86%. So C2 can say the pipeline does not agree in level, and can hardly say it agrees.

## Intervals

For a mean over homes, a Student t interval with one fewer degrees of freedom than homes, at 95%: the mean plus and minus the quantile times the standard deviation of the home values over the square root of their number.

- **Unit.** Homes; the interval chosen is `student_t`.
- **Why this interval.** In the planning trials, synthetic studies with the mat homes' numbers of mat-observed days, a percentile bootstrap over homes covered 89.7% to 92.9% of the time and the t interval 93.2% to 96.0%. With the three outlying homes both covered more, 92.3% to 100.0% and 94.8% to 100.0%, since their difference was fixed and not drawn. Homes that are outlying at random, rather than these three, can make a t interval over so few homes cover less than its level; C2, which they would move most, is secondary.
- **The planning's days.** The numbers of mat-observed days with both neighbours, from the mat alone: upper bounds on the matched days.
- **Counts.** A pooled count of days or alerts in E6 or E7 is given with no interval, since the days of one home are not independent.
- **Planning record.** `artifacts/sleep_mat/sleep-mat-planning.json`, `098876fe75003a8fe6af8e0933a0c31f6e16c5a9bf5936cd00d7b3ed7b8565e3`.

## The simulated reference

**A reference, not a test: what the estimands give when the observation model is the one that made the data.** Simulated: every home comes from this repository's simulator.

| Field | Value |
| --- | --- |
| homes | 100 |
| seed root | `20261008` |
| seeds | the first distinct values of `numpy.random.SeedSequence(seed_root).generate_state(4 * homes, uint32)` modulo 1,000,000, sorted |
| days in each record | 84 |
| first day | 2024-03-04 |
| timezone | `Europe/Lisbon` |
| household | `HouseholdConfig` defaults apart from days and seed; nothing is injected |
| sensors | `front_door`, `bedroom_motion`, `bathroom_motion`, `kitchen_motion`, `living_motion`, `fridge_contact` |
| delivery | `sensor_modeling.simulation.faults.degrade` with its default configuration |
| step, minutes | 10 |

- **Why these sensors.** The sensors that report on no cadence, as in TIHM.
- **Truth.** The hours of each local day covered by the simulator's episodes of `sleeping`, measured in elapsed time, so the night the clocks change is not given an hour it did not have.
- **Matched day.** A usable day other than the record's first and last, which are partial.

## Reporting

Every estimand, criterion, analysis and home, whatever it shows; no margin, rule or day is changed after the comparison is made. What was run and read between the freeze and the run is listed in `artifacts/sleep_mat/between_the_freeze_and_the_run.json`.

What this cannot show:

- Whether the mat's stages are right. A disagreement between the two may be the mat's.
- Anything about the 39 homes without a mat.
- How a day with no mat record went: the person may have slept out of bed, been away, or the mat may not have worked.

TIHM is by Palermo et al., Scientific Data 10, 606 (2023), under CC BY 4.0. Surrey and Borders Partnership NHS Foundation Trust and Howz are acknowledged, as the dataset asks.
