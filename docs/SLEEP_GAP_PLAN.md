# Where the pipeline's sleep parts from the sleep mat: the plan

The sleep-mat comparison found that, within a TIHM home, the pipeline's daily hours of sleep barely follow a sleep mat's. This page is the plan for setting the pipeline's belief beside the mat minute by minute, written before any of it was computed, and generated entirely from the frozen file `artifacts/sleep_gap/sleep_gap_plan.json` by `sensor_modeling.datasets.sleep_gap_summary.render_plan`.

**Status: exploratory description, planned after the sleep-mat comparison had been read and before any value of it was computed. No criterion and no margin is declared. Nothing in the pipeline is changed or fitted.**

- **Plan digest.** `dc2fae37de19128313e0c5def2c5efa367977aae697286b5dfcecb253e19e726`.
- **The question.** Where in the day, and on which of the mat's minutes, the pipeline puts its sleep, and which part of it moves the daily hours that do not follow the mat.
- **This page reports no result.**
- **The code it was frozen against.** The digests of 207 source files, every default setting, and numpy 2.5.3, pandas 3.0.6, python 3.13.16, scipy 1.18.1.

## What had been seen

- The sleep-mat comparison's record and pages, every estimand: within a home the pipeline's daily hours of sleep follow the mat's sleep at a mean Spearman correlation of 0.11 [0.03, 0.19] over 14 homes and its hours in bed at 0.14; the pipeline gives 3.95 hours a day more sleep than the mat and 1.97 more than its hours in bed; in one home its median is 19.9 hours of sleep a day; on the 19 silent days the mat observed its median is 23.8 hours; hourly activity is most opposed to time in bed with the mat's clock an hour later, at -0.58 against -0.54 as recorded, with overlapping intervals; with the mat an hour later the daily correlation is 0.13 over 13 homes; and the per-home summaries the record holds.
- The alert-burden run and its description after the run: days with no record read as sleep, the pipeline abstaining on 0.06% of usable time, and its deviating days quiet ones.
- The matched-sensor study: in three TIHM homes, 0d5ef, 30a32 and 55cd4, the default pipeline's step-level beliefs after 10-minute windows that held only kitchen, or only bathroom, activations, 68% to 89% on `home_inactive`; those homes' hours a day in each state under the default and five changed emission declarations; and, in the simulator, hours of sleep that still follow the truth at 0.71 under sensors matched to TIHM's records.
- For this plan: the activity file's location names and their counts, the date range of the activity and mat files, 1 April to 30 June 2019, so every timestamp falls in British Summer Time and no clock change lies inside the data, and the pipeline's code: its steps start at a home's first observation and follow every 10 minutes, each step's belief describes the time since the step before, and a day's summary counts only the intervals between two steps of that day, so the interval across midnight is counted by neither day. No pipeline output had been aggregated by hour or set beside the mat at a finer grain than a day.
- An independent review of the draft read the code and the mat's file on its own: only minute records, sessions that start AWAKE 3,183 times in 3,304 and end asleep about 71% of the time; 75 stretches without records of 3 hours or more beginning between 22:00 and 05:00, 61 of them in home 55cd4; and, over every home's observed days, 5 with under 2 hours of records and 15 with under 4. It found that the draft's correlations of the mat with the pipeline's sleep on and off the mat are partly mechanical, that a minute without a record was called an empty bed, and that a minute's context was taken before the step its belief came from had seen all it saw. The plan was revised for each.
- The run's checks were rehearsed on the mat homes before the freeze, on the draft and again on the revision: both runs, the included homes and their matched days, the clock-change guard, the repeat's daily hours of every state and its observed time, and the mat's minutes under each clock, the last time with the run script's checks-only mode. Only whether each check held, the largest difference between a day's minutes and its hours, 1.4e-13 hours, and that three matched days are not observed by the mat an hour later were read.

## The homes and days

- **Days.** That analysis's matched days: usable days of the run with the silent-home rule off that the mat observed, other than a home's first and last, leaving out every day touched by a silence of the whole home of 12 hours or more. The same days are kept under both of the mat's clocks, although three of them would not count as observed by the mat an hour later.
- **Homes.** The homes the sleep-mat comparison's primary analysis includes, those with at least 14 matched days.
- **Runs.** Both runs of the sleep-mat comparison, with the alert-burden protocol's frozen settings and a step of 10 minutes, the rule at 12 hours read only for the days it flags; the beliefs come from a repeat of the run with the rule off.
- **Timezone.** Europe/London.

## The minutes

- **Grid.** Every minute of each matched local day, from 00:00 to 24:00; a mat record stamped at a minute covers that minute.
- **The pipeline.** The belief of the step whose interval, from the step before to the step, covers the minute, weighted by the seconds it covers; only intervals between two steps of the same day, and no longer than 3 steps, count, as the daily summary counts them. A minute no interval covers is unwatched: the interval across midnight is counted by neither day, so about ten minutes around midnight are unwatched every day, the mat's sleep in them is sleep the pipeline did not count, and the pipeline's hours at 00 and 23 are lower by construction.

| What the mat says of a minute | When |
| --- | --- |
| the mat says asleep | A mat record staged LIGHT, DEEP or REM. |
| the mat says awake in bed | A mat record staged AWAKE. |
| no mat record | No mat record. The bed may be empty, the person may be asleep elsewhere, or the mat may have stopped recording; the file cannot tell these apart. |

- **An hour later.** Each mat record counted an hour after its timestamp, the lag at which the sleep-mat comparison found hourly activity most opposed to time in bed, narrowly, and where a mat clock in UTC would put it; every estimand is computed under it as well, on the same days.
- **Mat clock as recorded.** The mat's timestamps as the file writes them, as the sleep-mat comparison read them; the primary clock.
- **The context's moment.** For a watched minute, the moment of the step whose interval covers the minute's middle, which is what that step's belief had seen; for an unwatched minute, its middle.
- **The sensor that last reported.** The sensor of the home's latest activation at or before that moment, by its location in the activity file; the front and back doors are the exits.
- **The time since it.** 10 to 60 minutes: from 10 minutes to under 60; 1 to 3 hours: from 60 minutes to under 180; 3 hours or more: from 180 minutes, and on; measured: up to the context's moment; nothing has reported yet: no activation before it; under 10 minutes: from 0 minutes to under 10.
- **Period.** Day from 07:00 to 22:00, night from 22:00 to 07:00.
- **The time to the nearest mat record.** 2 to 6 hours: from 120 minutes to under 360; 30 minutes to 2 hours: from 30 minutes to under 120; 6 hours or more: from 360 minutes, and on; under 30 minutes: from 0 minutes to under 30.
- **Quiet hours.** A day's minutes whose middle is at least 60 minutes after the latest activation at or before it, in hours.

## Estimands

|  | Estimand |
| --- | --- |
| D1 | For each clock, the mean over homes, with its interval, and the median over homes of each home's mean over its matched days of: the pipeline's hours of sleep; its hours of sleep on the mat's asleep minutes, on its awake minutes and on minutes with no record; the mat's hours asleep and in bed; the mat's sleep and time in bed the pipeline did not count as sleep, and the mat's asleep hours on unwatched minutes; and the two differences, pipeline minus the mat's sleep, which equals sleep counted while awake on the mat plus sleep counted with no record minus the mat's sleep not counted, and pipeline minus the mat's time in bed, which equals sleep counted with no record minus the time in bed not counted. Each home's values are reported. |
| D2 | For each clock and each local hour, the mean over homes of each home's mean hours a day in that hour of the pipeline's sleep on each class of the mat's minutes, and of the mat's sleep and time in bed. |
| D3 | For each clock and each class of the mat's minutes, the mean over homes of each home's mean belief in each of the seven states over its watched minutes of that class. |
| D4 | For each clock, over the minutes with no mat record: the pipeline's hours of sleep and the hours of such minutes, each a mean over homes of a home's mean a day, and the mean over homes of the pipeline's belief in sleeping over them, by the sensor that last reported, by the time since it, by whether that sensor was an exit, by period of the day, and by sensor, time since and period together. |
| D5 | For each clock, the mean over homes of the within-home Spearman correlation over matched days of the mat's sleep, and of its time in bed, with: the pipeline's hours of sleep, its hours of sleep while the mat has a record, and its hours of sleep with no record; beside each correlation with a part, the same correlation with every day's beliefs set beside every other day's mat, the mean over all cyclic shifts of the days, which is what the mat's own hours give with no day-specific tracking, and each home's correlation minus its reference, as a mean over homes with the number of homes above zero; the share of the variance of the pipeline's daily hours of sleep that moves with each part, its covariance with the part over its variance, which sum to one; the correlation of the pipeline's hours of sleep with the day's quiet hours and with the day's activations; and the mean over homes of the within-home standard deviation of the pipeline's hours of sleep and of its two parts. |
| D6 | D1 to D5 and D7 without the three homes the sleep-mat comparison found the mat staging mostly awake. |
| D7 | For each clock, over the minutes with no mat record: the pipeline's hours of sleep, the hours of such minutes and the belief in sleeping, as in D4, by the time to the nearest mat record, earlier or later, in the home's whole file, and by that time and period of the day together; and the matched days with fewer than 4 hours of mat records, how many there are and the mean over homes of their share of the home's sleep counted with no record, a home with none of them counting as 0 and a home with no such sleep left out. |

## Definitions

- **Correlation.** As the sleep-mat protocol has it: Spearman's, over a home's matched days; a home whose values on either side are constant has no correlation and counts as 0, and how many did so is reported.
- **Pipeline on the mat.** The pipeline's hours of sleep on the mat's asleep and awake minutes.
- **Variance share.** A home's covariance between the pipeline's daily hours of sleep and a part, over the variance of the former; a home with constant hours has none and is left out.

## Intervals

- **Method.** A Student t interval at 95% for a mean over homes, as the sleep-mat comparison's.
- **Unit.** Homes.

## The check

- With the rule off, every mat home's run gives the published alert-burden record's counts, and with the rule at 12 hours the silent-home record's, as the sleep-mat run checked.
- Every home's number of matched days, correlations and mean differences against both of the mat's references equal the sleep-mat record's, as do the included homes.
- The beliefs read minute by minute come from a repeat of the run with the rule off whose daily summaries equal the first run's; on every matched day the minutes sum, for every state, to the day's hours, and the watched minutes to the day's observed time, to within 1e-09 hours.
- On every matched day the mat's minutes sum to the day's hours asleep and in bed as the sleep-mat comparison reads them, under each clock.
- With the mat's clock as recorded, D1's mean difference from the mat's sleep equals the sleep-mat record's.
- No matched day has a clock change.
- **Otherwise.** Nothing is reported, not even per home.

## What each pattern would mean

Written down before anything was computed.

- If the pipeline's sleep while the mat has a record follows the mat's sleep within a home by more than the swapped-mask reference gives, judged by each home's correlation minus its reference, and the whole day's does not, the daily number is diluted by sleep counted with no mat record; the covariance shares say which part moves the daily hours. If it follows no better than the reference, the pipeline also misses the night's own variation.
- If the sleep counted with no mat record falls mostly by day and after the lounge's sensor with a long quiet, it is consistent with sitting still read as sleep; after an exit door, front or back, with a long quiet, with absence read as sleep; within two hours of the mat's records, with the edges of the night, the mat's clock or the pipeline's timing of going to bed and getting up; far from any record at night, possibly with a mat that stopped recording.
- If the mat's clock an hour later moves much of the sleep counted near the mat's records onto them, the as-recorded clock puts that part off the mat. That the daily totals change little is known already from the sleep-mat comparison's shifted analysis.
- If, while the mat says asleep, the pipeline's belief sits on states other than sleeping, the pipeline misses sleep as well as adding it.
- None of these patterns says which of the pipeline and the mat is right about a minute: the mat's stages are the device's own and not validated, a minute with no record may be a person asleep elsewhere, out, or a mat that stopped, and a few homes dominate a mean over 14.

## Reporting

Every estimand under both clocks, whatever it shows; per-home summaries, never a day's or a minute's values.

## Pinned records

| File | SHA-256 |
| --- | --- |
| `artifacts/silent_home/silent-home-tihm.json` | `96798aa2e573f281477faad624ee7369ff2508d5a8631c7003f7f75218bae6a9` |
| `artifacts/sleep_mat/sleep-mat.json` | `5c003cda0d83dfc32336a6603774d8d32a26e4170154172d4f50b3d33e5304d4` |
| `artifacts/sleep_mat/sleep_mat_protocol.json` | `3e0f26fc82c4fde7ed241f469bd7c445afbdbe3e39e97169c8e0444ba5279f98` |
| `artifacts/tihm/alert_burden_protocol.json` | `ed04c9885f427f055e099e066d518188dd665cfbaaa012c459136e98378d8a2e` |
| `artifacts/tihm/tihm-alert-burden.json` | `5a5bb24167c8f089de3138f9c35ec6e07cd902ba385999f172458e40b40a64ff` |

## What this cannot show

- Whether the pipeline or the mat is right about a minute: the mat's stages are the device's own and are not validated.
- Why there is no mat record: a mat cannot tell sleep in a chair from a person who is out or from a mat that stopped.
- Anything beyond these 14 homes of people living with dementia, all in one study, or beyond the pipeline's default settings.
- A tested claim: the comparison it decomposes had been read, and nothing here is held to a margin.

## Data

TIHM is by Palermo et al., Scientific Data 10, 606 (2023), under CC BY 4.0. Surrey and Borders Partnership NHS Foundation Trust and Howz are acknowledged, as the dataset asks. Nothing of the dataset is redistributed.
