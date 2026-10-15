# The simulator's homes with TIHM's sensors: the protocol

Every detection this repository reports in simulation comes from homes whose event sensors fire at rates written into the simulator. This page is the protocol for drawing those homes' event sensors again under a profile matched to TIHM's sensor records, generated entirely from the frozen file `artifacts/matched_sensors/matched_sensors_protocol.json` by `sensor_modeling.datasets.matched_sensors_summary.render_protocol`.

**Status: pre-specified simulation study. The profile's hold-off is TIHM's, and two of its settings were chosen to match eight moments of TIHM's sensor records, in the planning record; its form is this project's, written after a pilot. No pipeline output under spill-over or under the chosen profile has been read for any home. Nothing in the pipeline is changed or fitted. Every result is a statement about the simulator.**

- **Protocol digest.** `c2910ff274040cacb4726df7bb7bd770602b9b8d82ddc7313ebdbc8f48b34921`.
- **The question.** When the simulator's event sensors are drawn again from the same plans under a profile matched to TIHM's sensor records, how much of what the simulator said about the pipeline still holds: its detection of the step change, its hours of sleep, and its hours in the kitchen and the bathroom.
- **This page reports no result.** The run checks the frozen file and the pinned records before it runs.
- **The code it was frozen against.** The digests of 203 source files, every default setting, and numpy 2.5.3, pandas 3.0.6, python 3.13.16, scipy 1.18.1.

## What had been seen

- TIHM's activity file, read for its sensors alone: over every home, the gaps between consecutive activations of each location's sensor. No two activations of one motion sensor are less than 61 seconds apart. The fridge and door contacts have 44% to 59% of their gaps under a minute, with a median of 7 to 9 seconds between two such rows.
- In three TIHM homes, 0d5ef, 30a32 and 55cd4, the activations a day of each location and the runs of consecutive activations of one location; and the default pipeline's beliefs in those homes after the 10-minute windows that held only kitchen, or only bathroom, activations: 68% to 89% on `home_inactive` and 0% to 5% on kitchen activity after the kitchen windows.
- The same three homes' median hours a day in each state under the default emissions and under five changed declarations: the fridge left out, counts read as fired or not in each step, the active rate at 15 an hour, and two combinations of these. None put kitchen activity above 0.42 hours a day or bathroom activity above 0.21. The pipeline's output on TIHM had been read in the earlier studies as well. No label and no sleep-mat record was read for this protocol.
- A pilot on four simulated homes from seeds 900001 to 900004, which belong to no protocol, over 42 days. With the simulator's motion sensors given a 61-second hold-off, the pipeline's median hours of kitchen activity fell from 1.59 to 0.96 against a true 1.93, their mean within-home correlation with the truth from 0.70 to 0.45, and that of sleep from 0.68 to 0.65. With a hallway sensor reporting at each change of room as well, kitchen activity was at 0.95 hours and 0.51, and sleep at 0.63. These read the pipeline's output on simulated homes before the protocol was written, and shaped it: they are why the profile is changed as a whole, why it has spill-over, and why C3 is declared. Neither pilot had spill-over.
- A coarse trial of the moments on six planning homes, and the planning record; neither runs the pipeline. The simulator's own records are far from TIHM's: 8% of its motion activations follow another room's against 63%, a median retrigger gap of 60 seconds against 153, no hallway, and 242 living-room activations a day against 77. The planning was run again after the review below corrected the profile's hallway, visitors, pair gap and streams, and the protocol pins the second record. It chose a spill-over of 2 an hour, at a distance of 0.8946, with 2.5 second at 0.8952; the first record had chosen 2.5.
- The published threshold-calibration results on the 400 homes under the default reference: 293 homes detected, 42 falsely detected and 342 false alerts; and TIHM's pooled-day medians of 0.06 hours of kitchen activity and 0.00 of bathroom activity.
- The standard profile's runs of the study's first two homes, 2739 and 5651, in both arms, to check that they raise the published alerts: they do.
- The code was rehearsed end to end on two homes outside the study, seeds 999 and 1001, under the standard profile and a draft matched profile, and the unit tests run one short home outside the study the same way. Their run times and that every step completed were read, and none of their values. No pipeline output under spill-over, or under the chosen profile, has been read for any home.
- An independent review of the draft read the code, the draft protocol and the first planning record, and drew sensor records on seeds outside the study without running the pipeline. It found the hallway reporting at each record's first instant, visitors in the hallway not firing it, the two arms of a home sharing the matched profile's sensor noise, an undeclared minimum of matched days, per-home outcomes written when the check fails, and a flat grid minimum whose nearest point changes between fresh draws of planning homes. The profile was corrected, the planning run again, the sensitivity profile and E7 to E9 declared, and the mechanisms below written down.
- A second review of the revision read the code and the second planning record. To check the code it ran the pipeline under the matched profile on seeds 999, 1001 and 4242 for 24 days, and under the sensitivity profile on seed 999, and built a results page and figures from those runs with every digit masked; it passed on no value. It found that under the standard profile the two arms of a home can share sensor draws, a mechanism wrongly stated for sleep, and the render script failing on a record whose check failed. All three were corrected, with other wording, before the freeze.

## The homes

- **Arms.** Change: from 2024-04-29, day 56 counting the first day as day 0, the resident wakes 1.6 hours earlier, with 1.2 more night-time bathroom trips expected a night; stable: nothing is injected; every behavioural alert is a false one.
- **Days.** 84.
- **From.** The threshold-calibration protocol, `artifacts/threshold_calibration/threshold_calibration_protocol.json`, whose homes, days, change, detection window and pipeline settings are used unchanged.
- **Homes.** 400.
- **Seed root.** 20261008.
- **Start.** 2024-03-04.
- **Step minutes.** 15.
- **Timezone.** Europe/Lisbon.
- **Window.** From 2024-04-29T23:00:00+00:00 to 2024-05-20T23:00:00+00:00: from local midnight at the end of the first changed day, for 21 days, closed on the left and open on the right.

## The profiles

- **Standard.** The simulator's own event record, as `simulate` draws it: the threshold-calibration study's homes exactly. Sensors: `front_door`, `bedroom_motion`, `bathroom_motion`, `kitchen_motion`, `living_motion`, `fridge_contact`.
- **Matched.** The event sensors drawn again from the same plan by `sensor_modeling.simulation.sensor_profile.profile_observations`, with the home's seed, the arm's stream and this profile. Sensors: the same and `hall_motion`.
- **Sensitivity.** The matched profile at the planning record's second-nearest grid point, on the study's first 100 homes, as a description.
- **Chosen by.** `artifacts/matched_sensors/matched-sensors-planning.json`. The hold-off is the shortest gap between two activations of one motion sensor in TIHM's activity file. TIHM's records have a hallway sensor and log a contact's opening as two rows; the profile imitates both with rules of this project's. The presence scale and the spill-over rate are the grid point whose eight moments of the sensor records were nearest TIHM's, summed squared log ratios of medians over homes. The minimum is flat, and fresh draws of planning homes order the nearest points differently, so the second-nearest is run as a sensitivity profile.
- **Pairing.** The profiles of one home and arm share the plan, so they differ in the sensors and not in the resident. The stable and changed arms of one home share the seed and not the days after the change. Under the standard profile the simulator draws the plan and the sensors from one generator, and two arms whose plans differ can fall back into step, so they share a part of their sensor draws before the change that depends on the home, from none of the weeks to all of them. Re-drawn sensors come from the arm's own stream, 0 for the stable arm and 1 for the changed one, so under the matched and sensitivity profiles the two arms share only the rows the plan fixes: the hallway's reports at changes of room and the door's crossings. The matched and sensitivity profiles of one arm draw from the same stream. This changes how a home's detection and its false detection move together, not what either is expected to be.
- **Delivery.** Every record passes through `sensor_modeling.simulation.faults.degrade` with its default configuration: no loss, lateness, duplication or fault.

| Setting | Meaning | Standard | Matched | Sensitivity |
| --- | --- | --- | --- | --- |
| Hallway | A hallway motion sensor that reports at each change of the resident's or a visitor's room, fires at the visitor rate while a visitor is in the hallway, and otherwise spills over. | no | yes | yes |
| Hold off seconds | The shortest gap, in seconds, between two activations one motion sensor reports. | 0 | 61 | 61 |
| Paired contacts | The fridge and the entrance door record each activation twice, the second row 1 to 17 seconds after the first, uniformly. | no | yes | yes |
| Presence scale | The multiple of the simulator's in-room rates of 45 an hour active, 6 still and 0.6 asleep, and of a visitor's 31.5. | 1 | 1 | 1 |
| Spill rate | Activations an hour of every motion sensor of a room the resident is not in, the bathroom's and the hallway's among them, while the resident is at home and awake, moving or still; asleep, out or with no resident at home, 0.12 an hour. | 0.12 | 2 | 2.5 |

## The pipeline

- **Alert policy.** Cooldown hours: 20; min confidence: 0.4; min score: 0.25.
- **Baseline.** Deviation threshold: 3; min samples: 14; persistence days: 3; trend threshold: 3.5; trend window: 28; weekday min samples: 4.
- **Features.** `sleeping_hours`, `kitchen_activity_hours`, `bathroom_activity_hours`, `away_hours`.
- **Min day coverage.** 0.5.
- **Min day observed.** 0.6.
- **Step minutes.** 15.
- **What.** `BehaviouralSensingPipeline` over each profile's registry, with default emissions derived from it and every default setting; nothing is fitted. The matched registry has the hallway's motion sensor, whose room no state names, as TIHM's has.

## What the outcomes turn on

Written down before any home's pipeline output under spill-over was read.

- The default emissions expect 0.15 activations an hour from the motion sensor of a room other than the one a state names, so while the resident is in the kitchen, the bathroom or in bed awake each spill-over activation is strong evidence against that state. Spill-over at 2 an hour therefore pushes belief away from kitchen and bathroom activity whenever the resident is awake.
- The living room's and the hallway's sensors name no state's room, so the default emissions read them as evidence for `home_active` (10 an hour) and `home_inactive` (2.5 an hour).
- The hold-off of 61 seconds lowers the rate a room's sensor reports while the resident is active in it from 45 an hour to about 25, against the 40 the emissions expect of the room's state; this is what the pilot showed.
- Paired contact rows double the evidence a fridge opening gives for an active state and a door crossing gives for `away` and `home_active`, which works against C3.
- The occupancy layer reads two rooms firing within 60 seconds of each other as evidence of a visitor. Spill-over makes such pairs common while the resident is alone, and the hallway's report at a change of room, with the next room's first activation, makes one at most changes of room even without spill-over. That lowers the attribution of activity to the resident, can hold an alert back at the confidence gate of 0.4, and discounts the evidence. E7 describes it.
- C3 reproduced would show that this sensor model suffices to give the pipeline TIHM's near-zero kitchen and bathroom hours in the simulator, not that it is why TIHM's are near zero. E8 shows whether the kitchen's own activations lose to `home_inactive` as they did in TIHM.

## The check

- **Otherwise.** Nothing is reported, not even per home.
- **Record.** `artifacts/threshold_calibration/threshold-calibration.json`.
- **What.** In every home, the standard profile's run of each arm must raise exactly the behavioural alerts the published record gives that home and arm under `default@1`: the same days, subjects, verdicts and directions.

## Definitions

- **Correlation.** The within-home Spearman correlation between the pipeline's value and the truth on the profile's matched days, for a home with at least 14 of them. A home whose values on either side span no more than 1e-09 hours has no correlation and counts as 0.
- **Detected.** An alert about `sleeping_hours` in the changed arm's window, in either direction, as the published study counts it.
- **Excess detection.** A home's detection minus its false detection: 1, 0 or -1.
- **False alerts.** The behavioural alerts of the stable arm, about any feature.
- **False detection.** An alert about the same feature in the same window of the stable arm.
- **Matched day.** A usable day of the stable arm other than the record's first and last days, as the sleep-mat protocol's simulated reference has them. Each profile's matched days are its own usable days.
- **Pipeline value.** A state's expected hours in the day's summary.
- **Pooled day median.** The median over every matched day of every home, as TIHM's medians are pooled over its usable days.
- **Truth.** The hours of the local day the simulator's episodes of the state cover, in elapsed time.

## Estimands

|  | Estimand |
| --- | --- |
| E1 | The mean over homes of the matched profile's excess detection minus the standard profile's. |
| E2 | For each profile, the share of homes detected, the share falsely detected, the mean excess detection, the same three counting only alerts of a decrease, and the median delay of a detection in days from the start of the first changed day; and the differences between the profiles in the shares detected and falsely detected. |
| E3 | For each profile, the mean over homes of the false alerts, and the stable arm's alerts about each feature per person-day. |
| E4 | For each profile and each of `sleeping_hours`, `kitchen_activity_hours`, `bathroom_activity_hours`, the mean over homes of the correlation; and the mean over homes in both of the matched profile's correlation minus the standard profile's, each on its own matched days. |
| E5 | For each profile and each of the seven states, the pooled-day median of the pipeline's value and of the truth; for the three features, the mean over homes of the home's mean difference, pipeline minus truth; and for kitchen and bathroom activity, a home-bootstrap interval for the pooled-day median of the pipeline's value. |
| E6 | The eight moments of the planning, on each profile's stable record of the study's homes, beside TIHM's from the planning record. |
| E7 | For each profile, in the stable arm: the change verdicts, those that raised an alert, and those that did not because the day's coverage times attribution was under the confidence gate or for another reason; the notices that alerts were held back in a burst; and the mean over homes of each home's mean ambient attribution of activity to the resident at its days' closes. |
| E8 | For each profile, in the stable arm: the mean belief in each state at the end of the steps whose window held motion activations of the kitchen's sensor and of no other motion sensor, and likewise of the bathroom's, with how many such steps there were. The study's steps are of 15 minutes; TIHM's beliefs were read after 10-minute windows, TIHM's step. |
| E9 | E1, E2 and E4 for the sensitivity profile against the standard profile, on the study's first 100 homes. |

## Criteria

- **C1 The detection survives the matched profile.** The primary criterion, on E1. Survives when the interval lies above -0.1; does not survive when it lies below -0.1; inconclusive otherwise.
- **C2 The hours of sleep still follow the truth.** Secondary, on E4's mean correlation of `sleeping_hours` under the matched profile. Follows when the interval lies above 0.5; does not follow when it lies below; inconclusive otherwise.
- **C3 The kitchen and bathroom collapse is reproduced.** Secondary, on E5 under the matched profile. Reproduced when the pooled-day median of `kitchen_activity_hours` is under 0.25 hours and that of `bathroom_activity_hours` under 0.05 hours; not reproduced otherwise. The medians are read as points; their home-bootstrap intervals are reported beside them.
- **Multiplicity.** C1 is the only primary criterion. The criteria are not adjusted for each other. E2 to E9 are descriptions.

## Margins

- **Ceilings.** 0.25 and 0.05 hours, declared conventions. TIHM's pooled-day medians are 0.06 and 0.00 hours in `artifacts/tihm/tihm-alert-burden-post-hoc.json`, over 2,793 usable days of which 128 had no record; the simulator's truth gives about two hours a day in the kitchen and a quarter of an hour in the bathroom.
- **Survival.** 0.1, a declared convention: a tenth of the homes. The standard profile's excess detection in the published record is 0.63. Assuming a standard deviation of about 0.6 for a home's paired difference, which the threshold-calibration protocol assumed for its own, 400 homes would give an interval about 0.06 either side.
- **Tracking.** 0.5, the sleep-mat protocol's margin. At 0.5 a deviation at the baseline's threshold of 3 would stand for one of 1.5 in sleep, under a Gaussian reading. Against the simulator's exact truth it is a lower bar than against a sleep mat.

## Pinned records

| File | SHA-256 |
| --- | --- |
| `artifacts/matched_sensors/matched-sensors-planning.json` | `c713dc27dd0308d098da0e0dad623da81cc9ac84ea770219df2e9a16f0ae569a` |
| `artifacts/threshold_calibration/threshold-calibration.json` | `1f0b22b7c329f30f97f16686ec5563cd6eafa4071079c691387fa5508da078d0` |
| `artifacts/threshold_calibration/threshold_calibration_protocol.json` | `459f90f8bb84752af537d34ac51f376c6815f83e593ed6360bc343c15ece7403` |
| `artifacts/tihm/tihm-alert-burden-post-hoc.json` | `2701e81efe7d126e753478e2c0187e8e7568716380ad359ac4a92f1f255723ac` |

## Intervals

- **Counts.** A pooled count has none.
- **Method.** A percentile bootstrap over homes at 95%, with 5,000 resamples from seed 0, as the published study's; a share of homes has a Wilson interval.
- **Unit.** Homes.

## Reporting

Every estimand, criterion, profile and home, whatever it shows; no margin, profile or day is changed after a home is run with the matched profile. What was run and read between the freeze and the run is listed in `artifacts/matched_sensors/between_the_freeze_and_the_run.json`.

## What this cannot show

- How the pipeline does on TIHM: the study runs no TIHM home.
- Whether the profile matches TIHM's sensors in anything but the eight moments. The residents' behaviour is the simulator's, and TIHM's carers, back doors and the behaviour of people living with dementia are not in it.
- Why TIHM's motion activations so often follow another room's. The profile puts it down to sensors that see beyond their room; more movement between rooms, or a second person at home, would fit the moments as well.
- Which part of the profile does what: it is changed as a whole.
- That the chosen point is the best: the grid's minimum is flat, and E9 runs the next point.
- The TIHM moments count every day with a record, partial days among them, which lowers TIHM's counts a day by a few percent.

## Data

TIHM: An Open Dataset for Remote Healthcare Monitoring in Dementia: TIHM's Activity.csv, by the planning only; the study reads no TIHM file. TIHM is by Palermo et al., Scientific Data 10, 606 (2023), under CC BY 4.0. Surrey and Borders Partnership NHS Foundation Trust and Howz are acknowledged, as the dataset's repository asks.
