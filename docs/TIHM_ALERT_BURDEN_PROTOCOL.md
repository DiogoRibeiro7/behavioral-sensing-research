# TIHM: the alert-burden protocol

The TIHM dataset holds 56 homes of people living with dementia, with alerts a clinical monitoring team verified. This page is the protocol for running the online pipeline on it, generated entirely from the frozen file `artifacts/tihm/alert_burden_protocol.json` by `sensor_modeling.datasets.tihm_summary.render_protocol`.

**Status: exploratory: descriptive of these homes; no threshold, mapping or model setting is chosen from them, and nothing here is a held-out or confirmatory claim.**

- **Protocol digest.** `7f6c0eff69ab91dbbe51aeef432b941cc512c096298f3d7294c96b3eb2fc304a`.
- **The question.** At its declared defaults, how many alerts does the online pipeline raise per person-day on a clinical cohort, how much of the record will it not use, and does what it flags relate to the alerts a clinical monitoring team verified.
- **This page reports no result.** The scoring run checks the frozen file, the mapping and every dataset file's digest before it runs.

## What had been seen

The protocol was written after the labels had been analysed. It fixes every definition before any pipeline output is compared with a label; it does not make the result a held-out claim.

- Labels.csv, Activity.csv and Physiology.csv had been analysed outside this repository: label counts by type and participant, label timestamps, hourly activity around agitation labels, the stated limits against the readings, and a rebuild of the published baseline. The label and reference definitions below were written with that knowledge.
- One household, c55f8, had been run through the pipeline to measure run time; its step count and its days' usability were read, and none of its verdicts or alerts.
- No pipeline alert, verdict or deviation had been compared with any label.

## The dataset

| Field | Value |
| --- | --- |
| name | TIHM: An Open Dataset for Remote Healthcare Monitoring in Dementia |
| version | 1.0 (Zenodo record 7622128, published 2023-02-10) |
| source | https://doi.org/10.5281/zenodo.7622128; archive https://zenodo.org/records/7622128/files/TIHM_Dataset.zip |
| licence | CC BY 4.0; the dataset's repository asks that Surrey and Borders Partnership NHS Foundation Trust and Howz be acknowledged in any publication or use |
| citation | Palermo, F.; Chen, Y.; Capstick, A.; Fletcher-Lloyd, N.; Walsh, C.; Kouchaki, S.; True, J.; Balazikova, O.; Soreq, E.; Scott, G.; Rostill, H.; Nilforooshan, R.; Barnaghi, P. TIHM: An open dataset for remote healthcare monitoring in dementia. Scientific Data 10, 606 (2023). https://doi.org/10.1038/s41597-023-02519-y |
| archive SHA-256 | `368d642b4cdc680d0706abfd3c8b2e5387824d9737ad33f50531bafa6a4bf5a1` |
| timezone | `Europe/London`, declared |

Every file's SHA-256:

| File | SHA-256 | Use |
| --- | --- | --- |
| Activity.csv | `ead7b1e3c2a91fbd679a909bff3a309a6465ae1ad18528dae57a4cf8599cafca` | read through the contract |
| Demographics.csv | `15b13a680da6ee3790dfadc25d3a6102070f6e8fe1a47ea6da8f64b6c4b757d4` | read through the contract |
| Labels.csv | `39495a08fdf1ff8f9b9b88bc98483a565c4433f58713ba3e4d0846e4449b846b` | read through the contract |
| Physiology.csv | `377b3170b4b650795648e18258f30f9b985c214937823e9c9ea405c66b4ba696` | read outside the contract |
| Sleep.csv | `f1d5a6980263a55a8b69841e13bb349ce6e79152c15bdde0beffeea15293b9fe` | pinned, not read |

- **Physiology.csv.** Only to describe the labels: which days have a reading, and whether a day's readings meet the limits the dataset paper states. No pipeline input.
- **Redistribution.** None: the run reads an extracted download and verifies every file's digest.
- **Households.** Every participant in Demographics.csv; a household that fails the contract is reported and not run.

## Ontology mapping

The mapping is `artifacts/tihm/tihm_mapping.json`, SHA-256 `0a6e393e354e1053d70d308ba53f88ad32aec67bd25a476e7112312f40beccfb`.

| Label | Outcome |
| --- | --- |
| `Agitation` | unmappable |
| `Blood pressure` | unmappable |
| `Body temperature` | unmappable |
| `Body water` | unmappable |
| `Pulse` | unmappable |
| `Weight` | unmappable |

| Location name | Native sensor type |
| --- | --- |
| `Back Door` | `door` |
| `Bathroom` | `pir` |
| `Bedroom` | `pir` |
| `Fridge Door` | `fridge_door` |
| `Front Door` | `door` |
| `Hallway` | `pir` |
| `Kitchen` | `pir` |
| `Lounge` | `pir` |

- **Consequence.** No annotated time maps to a behavioural state, so state inference is not scored on this dataset.

## The pipeline

BehaviouralSensingPipeline with the registry the canonical conversion builds, default emissions derived from that registry, and default configuration throughout. Fitted on TIHM: nothing.

| Setting | Value |
| --- | --- |
| step | 10 minutes |
| baseline features | sleeping_hours, kitchen_activity_hours, bathroom_activity_hours, away_hours |
| minimum day coverage | 0.5 |
| minimum day observed | 0.6 |
| attribute activity to the resident | True |
| baseline deviation_threshold | 3.0 |
| baseline min_samples | 14 |
| baseline persistence_days | 3 |
| baseline trend_threshold | 3.5 |
| baseline trend_window | 28 |
| alert policy cooldown_hours | 20.0 |
| alert policy health_coverage_floor | 0.5 |
| alert policy max_per_window | 6 |
| alert policy min_confidence | 0.4 |
| alert policy min_score | 0.25 |

## Definitions

- **Alert day.** A monitored day with at least one behavioural alert.
- **Any label day.** The same for a label of any type.
- **Behavioural alert.** An alert of kind behavioural_change, attributed to the day its verdict is about.
- **Deviating day.** An evaluable day on which at least one feature's robust deviation reached the baseline's deviation threshold.
- **Deviation score.** The largest absolute robust deviation among the day's features with a verdict; zero where there is none.
- **Evaluable day.** A monitored day on which at least one baseline feature received a verdict other than insufficient_data.
- **Label day.** A local calendar day with at least one Agitation label; a label's day is the date of its timestamp.
- **Monitored day.** A local calendar day the pipeline closed with a daily summary.
- **Usable day.** A monitored day whose summary passes the pipeline's coverage and observed-fraction tests.

## Estimands

| Estimand | What it is |
| --- | --- |
| A1 | On evaluable days: the share of label days that are alert days, minus the share of other days that are. |
| A2 | The same for deviating days. |
| A3 | The probability that a label day has a higher deviation score than another evaluable day of the same household, ties counted as half. |
| B1 | Behavioural alerts per monitored person-day, all features. |
| B2 | The same for sleeping_hours alone, beside the simulator's figures, which are a reference and not a test. |
| B3 | Behavioural alerts per evaluable person-day. |
| H1 | System-health and data-quality alerts per monitored person-day, and the share of monitored days that are usable and evaluable. |
| P1 | Label days caught by each reference on the published protocol's five test weeks, with the published model's alert count in each week, beside the published count. |
| R1 | Label days caught by each reference when it flags as many evaluable days as the pipeline has deviating days. |

## References

Two rules that use no model, and chance. They say what a flag has to beat.

- **Event count.** The day's sensor event count as a z-score against the household's previous monitored days, zero until 5 exist.
- **Label history.** The share of the household's earlier monitored days that were label days, shrunk by 7 pseudo-days towards the cohort's overall share on evaluable days; it uses no sensor.
- **Random.** The expected catch of flagging the same number of days at random.
- **Ties.** Days tied at the cut share the remaining flags equally.

## The published baseline

The dataset is published with a notebook, `example_code/example_baseline_classfication.ipynb`, SHA-256 `49d22859216685014bc0000877a56c4d010be75cad335f5a6b69f5582380d735`, that trains a logistic regression to recognise label days and prints its confusion matrices. They are quoted here; the model is not retrained.

| Test week, newest first | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| 1 | 271 | 76 | 0 | 10 |
| 2 | 268 | 38 | 1 | 7 |
| 3 | 248 | 49 | 4 | 9 |
| 4 | 233 | 57 | 6 | 11 |
| 5 | 215 | 66 | 5 | 18 |

- **Protocol.** 5 test periods of 7 days stepping back from the last Agitation day; training on every earlier day; a person-day is a day with any activity row or non-zero physiology reading.
- **Check.** The run refuses to report this section unless each test period reproduces the published number of person-days and label days.
- **Label history here.** The household's share of label days in the training period, shrunk the same way towards the training period's overall share; a household with no training day gets that overall share.
- **Not done.** The published model is not retrained here; its counts are quoted from the notebook's printed output.

## Describing the labels

- **Slots.** A label is on a slot when it is stamped less than 180 seconds after 08:00, 12:00, 18:00.
- **Hourly profile.** Sensor events per hour as a z-score against the same household and hour over its full monitored days, averaged over label days grouped by the slot of the day's first label.
- **Blocks.** 00-06, 06-12, 12-18, 18-24.
- **Full day.** A monitored day that is neither the household's first nor its last.

The limits the dataset paper states for the daily measurements:

| Label | Device | Below | Above |
| --- | --- | --- | --- |
| Blood pressure | Systolic blood pressure | 80.0 | 190.0 |
| Blood pressure | Diastolic blood pressure | 50.0 | 110.0 |
| Body temperature | Body Temperature | 35.0 | 37.6 |
| Body water | Total body water | 40.0 | 70.0 |
| Pulse | Heart rate | 55.0 | 100.0 |

## Uncertainty

- **Confidence.** 0.95.
- **Interval.** Percentile.
- **Reason.** Days within a home are dependent; see docs/EVALUATION_DESIGN.md.
- **Resamples.** 5000.
- **Seed.** 0.
- **Statistic.** Every estimand is a ratio, or a difference of ratios, of sums over households, recomputed on each resample.
- **Unit.** Households.

## The simulator's figures

Beside B2, as a reference and not a test.

| Field | Value |
| --- | --- |
| changed arm | `0.033` |
| feature | `sleeping_hours` |
| source | `docs/ADVERSARIAL_REVIEW.md, item i4; docs/RESEARCH_QUESTIONS.md, RQ4` |
| stable arm | `0.01` |

## Reporting

Every estimand and every household, whatever it shows; no threshold, no retuning and no added reference after scoring.
