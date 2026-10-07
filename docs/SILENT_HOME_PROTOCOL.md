# The silent-home rule: the protocol

A home of event sensors that stops reporting is read as a home asleep. The [TIHM run](TIHM_ALERT_BURDEN_RESULTS.md) showed it, and an opt-in rule, `HealthConfig.home_silence_horizon`, was built from what it showed. This page is the protocol for finding out whether the rule does what it was built for, generated entirely from the frozen file `artifacts/silent_home/silent_home_protocol.json` by `sensor_modeling.datasets.silent_home_summary.render_protocol`.

**Status: pre-specified on simulated homes; descriptive on TIHM. The simulator's homes, estimands and criteria are fixed before any simulated home is run with the rule on. Nothing on TIHM is a test.**

- **Protocol digest.** `52d6e6804657f6aaf0c3f921309f33f2d48b88598f38c83251944bb03a660fa6`.
- **The question.** Whether, when a home of event sensors stops reporting, the opt-in rule keeps the silence out of the personal baseline and reports it, without changing anything when nothing is wrong and without costing the detection of a real change.
- **This page reports no result.** The scoring run checks the frozen file before it runs.

## What had been seen

The rule was designed from TIHM, so the homes that showed the problem cannot also be the evidence that the rule solves it. The test is made on simulated homes that had not been run with the rule on.

- The TIHM alert-burden run and the descriptions made after it had been read: the silent days, the calendar, and the replay of the baseline with silent days left out. The rule was designed from them, which is why nothing on TIHM here is a test.
- The rule had been exercised in unit tests on a hand-built home of three event sensors and on hand-built fleets. No simulated household had been run with the rule on.
- The rule was changed once before this protocol was frozen, from reading the code and not from any run: a day that lost any time to a silence is refused by the baseline, where at first only a mostly silent day was.
- One simulated household, seed 999, which is not a home of this protocol, had been run with the rule off and with its event sensors alone to measure run time; its numbers of observations and of steps were read, and none of its alerts, verdicts or summaries.
- The simulator's source had been read, including the rates at which its sensors fire. How long a simulated or a TIHM home goes without an event had not been measured; the horizons were declared, not estimated.

## The rule

`HealthConfig.home_silence_horizon` is off by default. When it is set and no sensor of a home has reported for that long, the home is treated as not observed from its last observation on, the days that lost any time that way are refused by the baseline, and one data-quality alert is raised. See [the inference contract](inference.md).

| Condition | What it is |
| --- | --- |
| `h12`, primary | The rule on at 12 hours. |
| `h24` | The rule on at 24 hours. |
| `off` | The rule off: the pipeline's default. |

The horizons are declared, not estimated from any home.

## The simulated homes

**Simulated: every home comes from this repository's simulator, so every result is a statement about the simulator.**

| Field | Value |
| --- | --- |
| homes | 100 |
| seed root | `20261007` |
| seeds | the first distinct values of `numpy.random.SeedSequence(seed_root).generate_state(4 * homes, uint32)` modulo 1,000,000, sorted |
| days in each record | 84 |
| first day | 2024-03-04, day 0 |
| timezone | `Europe/Lisbon` |
| household | `HouseholdConfig` defaults apart from days, seed and the injected change |
| sensors | `front_door`, `bedroom_motion`, `bathroom_motion`, `kitchen_motion`, `living_motion`, `fridge_contact` |
| sensors left out | `bed_pressure`, `living_radar`, `wearable_motion`, `resident_beacon` |
| replications | 100 paired homes; `docs/SIMULATION_PROTOCOLS.md` sets a floor of 100 for a simulation study |

- **Why sensors are left out.** They report on a cadence, so their silence is already evidence of a failure. What is left is a stream of activations, as in TIHM.
- **Delivery.** Every arm passes through `sensor_modeling.simulation.faults.degrade` with no loss, lateness or duplication. An outage is a dropout fault on every sensor over the same window.
- **Pairing.** Every arm and condition is run on every home, so a home with a fault differs from the same home without it in the fault alone, and the rule on differs from the rule off in the rule alone.

### The pipeline

`BehaviouralSensingPipeline` over the event sensors' registry, with default emissions derived from it; nothing is fitted.

| Setting | Value |
| --- | --- |
| step | 15 minutes |
| baseline features | `sleeping_hours`, `kitchen_activity_hours`, `bathroom_activity_hours`, `away_hours` |
| minimum day coverage | 0.5 |
| minimum day observed | 0.6 |
| baseline deviation threshold | 3.0 |
| baseline min samples | 14 |
| baseline persistence days | 3 |
| baseline trend threshold | 3.5 |
| baseline trend window | 28 |
| alert policy cooldown hours | 20.0 |
| alert policy min confidence | 0.4 |
| alert policy min score | 0.25 |

### Arms

| Arm | What is injected |
| --- | --- |
| `change` | The detection study's step change from day 56: the resident wakes 1.6 hours earlier, with 1.2 more night-time bathroom trips expected a night. |
| `change_after_outage` | The home's own long outage and then the change, which begins at least a week after the outage has ended. |
| `outage` | No sensor of the home reports for 60 hours. Each home has its own outage: the day, between day 28 and day 45, and the local hour it begins at, from 0 to 23, are drawn from `numpy.random.default_rng([seed_root, 1])`, the days first. |
| `outage_common` | No sensor of any home reports from 2024-04-15T03:00:00+00:00 to 2024-04-17T15:00:00+00:00: 60 hours from 04:00 local on day 42. Only the fleet check reads this arm. To one home it is the same as an outage of its own, so it is not run through the pipeline. |
| `outage_short` | No sensor of the home reports for 8 hours from the same moment, which is shorter than every horizon. |
| `stable` | Nothing is injected. |

### Runs

Every home is run through the pipeline once for each mark.

| Arm | `h12` | `h24` | `off` |
| --- | --- | --- | --- |
| `change` | run |  | run |
| `change_after_outage` | run |  | run |
| `outage` | run | run | run |
| `outage_short` | run |  | run |
| `stable` | run | run | run |

That is 12 runs of the pipeline for each home.

### When an outage begins

The rule sees a silence once it has lasted the horizon. If the day on which an outage begins is closed before that, the day is closed as it would be without the rule, with the silent hours read as observed. The hour is drawn so that both cases are in the homes.

| Group | The outage | Homes |
| --- | --- | --- |
| `seen_after_the_day_closes` | It begins later than that. | 43 |
| `seen_before_the_day_closes` | The outage begins more than the primary horizon before local midnight. | 57 |

The outages drawn begin between day 28 and day 45.

## Definitions

- **Behavioural alert.** An alert of kind behavioural_change, dated by the moment it was raised.
- **Delay.** Days from local midnight at the start of day 56 to the first such alert.
- **Detected.** A home with a behavioural alert about `sleeping_hours` raised once the change day has closed and less than 21 days after that: from local midnight at the end of day 56. The pipeline raises behavioural alerts when a day closes, so these are the alerts that 21 days of changed behaviour can raise, and the alert raised at the close of the day before the change is not among them.
- **Excess alerts.** A home's behavioural alerts in its outage window in an outage arm, minus the same home's in the same window in the stable arm, under the same condition.
- **False detection.** The same definition met in the arm that has no change: `stable` for `change`, and `outage` for `change_after_outage`.
- **Outage window.** From the moment a home's outage begins until 28 days after it ends, which covers the baseline's trend window.
- **Person days.** Homes times days.
- **Reported.** A home with a silence alert dated between its outage's beginning and its end.
- **Silence alert.** An alert of kind data_quality about `home_silence`.
- **The rule changes an alert.** The behavioural alerts of a home with the rule on are not the same alerts, by subject and moment, as with it off.

## Estimands

| Estimand | What it is |
| --- | --- |
| E1 | Mean excess alerts per home in `outage` with the rule off. |
| E2 | The same with the rule on at the primary horizon. |
| E2 by group | E1, E2 and E3 within each outage group. No criterion is stated on them: they measure what the rule leaves when it sees a silence only after a day has closed. |
| E3 | E1 minus E2, paired by home. |
| E4 | In `stable`: behavioural alerts per person-day with the rule off and on, the homes in which the rule changes an alert, and silence alerts per person-day with the rule on. |
| E5 | In `change`: the share of homes detected with the rule off and on, their paired difference, the pooled median delay, and the share of false detections. |
| E6 | The same in `change_after_outage`. |
| E7 | In `outage` with the rule on: the share of homes reported, the hours from the outage's beginning to the first silence alert, the silence alerts per home, and the days the baseline refused in each home. |
| E8 | The stretches of common silence the fleet check finds in `stable`, `outage` and `outage_common`, and the largest share of homes silent at one assessment in each; for `outage_common`, the hours from the outage's beginning to the first assessment that calls it common. |
| E9 | Mean excess alerts per home in `outage_short` with the rule off and on, the homes in which the rule changes an alert, and the homes with a silence alert. |
| S1 | E2, E3 and E7 at each other horizon, with E4's counts there. At a horizon of a whole day no outage is seen before the day it begins on has closed. |

## Criteria

Fixed before any simulated home was run with the rule on.

| Criterion | Claim | How it is decided |
| --- | --- | --- |
| C1 | The outage raises alerts | Reproduced when E1's interval lies above zero; otherwise not reproduced in the simulator, and C2 is not testable. |
| C2 | The rule removes them | Success when E3's interval lies above zero and E2's interval lies below 0.25 alerts per home; failure when E3's interval does not lie above zero; partial otherwise. |
| C3 | No harm when nothing is wrong | Success when, in `stable`, the rule changes no behavioural alert in any home and raises no silence alert; failure otherwise. |
| C4 | Detection is kept | In `change`: uninformative when the share detected with the rule off is below 0.2; otherwise non-inferior when the interval of the paired difference, on minus off, lies above -0.1, inferior when it lies below it, and inconclusive otherwise. |
| C5 | Detection after an outage | The same rule applied to `change_after_outage`. |
| C6 | The silence is reported | Success when at least 0.95 of the homes are reported; failure otherwise. |
| C7 | A shared silence is told from scattered ones | Success when exactly one stretch is found in `outage_common` and it overlaps the outage, and none in `stable` or `outage`; failure otherwise. |
| C8 | A short silence is not seen | The rule cannot see a silence shorter than its horizon. Confirmed when, in `outage_short`, the rule changes no behavioural alert and raises no silence alert. Otherwise the outage joined a quiet stretch of the home and passed the horizon, and the homes in which it did are counted. Either way this is a limit of the rule and not a success. |

## The fleet check

`sensor_modeling.health.fleet.common_silences` over the times of each home's delivered observations.

| Setting | Value |
| --- | --- |
| horizon in hours | 12 |
| fraction of monitored homes | 0.6 |
| fewest homes | 3 |
| hours between assessments | 1 |

- **What it does not test.** In `outage` the homes' outages are spread over 18 days, so few homes are silent at once and the fraction is far away. C7 says the check is not tripped by outages that merely overlap, and nothing about where the fraction should be set.

## Uncertainty

- **Confidence.** 0.95.
- **Interval.** Percentile.
- **Monte carlo error.** The standard deviation of the paired differences over the square root of the number of homes, reported for every mean paired difference.
- **Resamples.** 5000.
- **Seed.** 0.
- **Share detected.** A Wilson interval, since homes are independent; its paired difference is resampled over homes.
- **Unit.** Homes.

## What the simulator cannot show

- A simulated home is not expected to be silent for a horizon unless a fault makes it so, so the simulator says nothing about how often a real home is silent for a harmless reason, or about which horizon a real home needs.
- A resident who is away for days, or who needs help, is not simulated. The rule reports such a silence in the same words as an outage, and the simulator cannot say what that costs.
- The simulated sensors fire at rates this project wrote down.

## TIHM

**A description, not a test: the rule was designed from these homes.**

| Field | Value |
| --- | --- |
| dataset | TIHM: An Open Dataset for Remote Healthcare Monitoring in Dementia |
| citation | Palermo, F.; Chen, Y.; Capstick, A.; Fletcher-Lloyd, N.; Walsh, C.; Kouchaki, S.; True, J.; Balazikova, O.; Soreq, E.; Scott, G.; Rostill, H.; Nilforooshan, R.; Barnaghi, P. TIHM: An open dataset for remote healthcare monitoring in dementia. Scientific Data 10, 606 (2023). https://doi.org/10.1038/s41597-023-02519-y |
| licence | CC BY 4.0; the dataset's repository asks that Surrey and Borders Partnership NHS Foundation Trust and Howz be acknowledged in any publication or use |
| archive SHA-256 | `368d642b4cdc680d0706abfd3c8b2e5387824d9737ad33f50531bafa6a4bf5a1` |
| step | 10 minutes |
| conditions | `off`, `h12`, `h24` |
| everything else | as frozen in `artifacts/tihm/alert_burden_protocol.json` |
| fleet check | the fleet check with the settings above, over the times of each home's activity records |

- **Check.** With the rule off the run must give the published record's 2,850 monitored days and 183 behavioural alerts, or nothing is reported.
- **Reported.** Monitored, usable and evaluable days; the days the baseline refused because of the rule, in all and per home; behavioural alerts, in all and by date; silence alerts, in all and per home; the stretches of common silence the fleet check finds, and the silence alerts dated inside one.
- **Not reported.** Any relation to the dataset's labels. Whether the alerts that remain relate to what a clinical team verified is the alert-burden protocol's question and is not reopened here.
- **Redistribution.** None: the run reads an extracted download and verifies every file's digest.

## Reporting

Every estimand, criterion, arm and home, whatever it shows; no horizon, margin or window is changed after scoring.

## The homes

Each home's seed, and the day and local hour its own outage begins at.

| Seed | Day | Hour | Group |
| --- | --- | --- | --- |
| `13526` | 42 | 20:00 | `seen_after_the_day_closes` |
| `20039` | 40 | 16:00 | `seen_after_the_day_closes` |
| `28983` | 40 | 03:00 | `seen_before_the_day_closes` |
| `30333` | 45 | 04:00 | `seen_before_the_day_closes` |
| `40041` | 36 | 01:00 | `seen_before_the_day_closes` |
| `40360` | 30 | 06:00 | `seen_before_the_day_closes` |
| `44530` | 30 | 07:00 | `seen_before_the_day_closes` |
| `46280` | 35 | 11:00 | `seen_before_the_day_closes` |
| `78794` | 45 | 07:00 | `seen_before_the_day_closes` |
| `83654` | 39 | 01:00 | `seen_before_the_day_closes` |
| `91149` | 37 | 00:00 | `seen_before_the_day_closes` |
| `94759` | 29 | 12:00 | `seen_after_the_day_closes` |
| `121658` | 41 | 13:00 | `seen_after_the_day_closes` |
| `128148` | 42 | 03:00 | `seen_before_the_day_closes` |
| `133269` | 42 | 09:00 | `seen_before_the_day_closes` |
| `143550` | 43 | 04:00 | `seen_before_the_day_closes` |
| `152418` | 29 | 21:00 | `seen_after_the_day_closes` |
| `155015` | 37 | 07:00 | `seen_before_the_day_closes` |
| `171009` | 42 | 00:00 | `seen_before_the_day_closes` |
| `173407` | 30 | 20:00 | `seen_after_the_day_closes` |
| `188647` | 43 | 21:00 | `seen_after_the_day_closes` |
| `214903` | 31 | 16:00 | `seen_after_the_day_closes` |
| `217565` | 43 | 02:00 | `seen_before_the_day_closes` |
| `220206` | 40 | 21:00 | `seen_after_the_day_closes` |
| `246969` | 29 | 15:00 | `seen_after_the_day_closes` |
| `259594` | 44 | 20:00 | `seen_after_the_day_closes` |
| `263821` | 41 | 21:00 | `seen_after_the_day_closes` |
| `273073` | 42 | 03:00 | `seen_before_the_day_closes` |
| `276573` | 37 | 08:00 | `seen_before_the_day_closes` |
| `278446` | 42 | 10:00 | `seen_before_the_day_closes` |
| `310941` | 37 | 22:00 | `seen_after_the_day_closes` |
| `316264` | 41 | 13:00 | `seen_after_the_day_closes` |
| `320700` | 43 | 09:00 | `seen_before_the_day_closes` |
| `333164` | 33 | 01:00 | `seen_before_the_day_closes` |
| `346595` | 45 | 13:00 | `seen_after_the_day_closes` |
| `347479` | 28 | 09:00 | `seen_before_the_day_closes` |
| `376909` | 36 | 02:00 | `seen_before_the_day_closes` |
| `389460` | 43 | 19:00 | `seen_after_the_day_closes` |
| `400728` | 43 | 05:00 | `seen_before_the_day_closes` |
| `419532` | 38 | 11:00 | `seen_before_the_day_closes` |
| `420985` | 31 | 09:00 | `seen_before_the_day_closes` |
| `424842` | 40 | 21:00 | `seen_after_the_day_closes` |
| `426479` | 36 | 09:00 | `seen_before_the_day_closes` |
| `442394` | 37 | 17:00 | `seen_after_the_day_closes` |
| `442747` | 40 | 06:00 | `seen_before_the_day_closes` |
| `447916` | 33 | 02:00 | `seen_before_the_day_closes` |
| `457526` | 34 | 07:00 | `seen_before_the_day_closes` |
| `463575` | 35 | 13:00 | `seen_after_the_day_closes` |
| `475808` | 41 | 01:00 | `seen_before_the_day_closes` |
| `486028` | 44 | 09:00 | `seen_before_the_day_closes` |
| `492826` | 44 | 22:00 | `seen_after_the_day_closes` |
| `496673` | 41 | 10:00 | `seen_before_the_day_closes` |
| `507993` | 45 | 10:00 | `seen_before_the_day_closes` |
| `508959` | 44 | 15:00 | `seen_after_the_day_closes` |
| `511611` | 43 | 19:00 | `seen_after_the_day_closes` |
| `540332` | 39 | 23:00 | `seen_after_the_day_closes` |
| `541550` | 30 | 11:00 | `seen_before_the_day_closes` |
| `543583` | 38 | 13:00 | `seen_after_the_day_closes` |
| `546733` | 38 | 16:00 | `seen_after_the_day_closes` |
| `566782` | 32 | 03:00 | `seen_before_the_day_closes` |
| `580416` | 35 | 06:00 | `seen_before_the_day_closes` |
| `582268` | 35 | 02:00 | `seen_before_the_day_closes` |
| `622795` | 36 | 08:00 | `seen_before_the_day_closes` |
| `636584` | 32 | 19:00 | `seen_after_the_day_closes` |
| `638168` | 30 | 13:00 | `seen_after_the_day_closes` |
| `643046` | 30 | 16:00 | `seen_after_the_day_closes` |
| `652792` | 37 | 20:00 | `seen_after_the_day_closes` |
| `660800` | 44 | 19:00 | `seen_after_the_day_closes` |
| `669761` | 37 | 12:00 | `seen_after_the_day_closes` |
| `671038` | 40 | 22:00 | `seen_after_the_day_closes` |
| `688693` | 37 | 13:00 | `seen_after_the_day_closes` |
| `690795` | 32 | 08:00 | `seen_before_the_day_closes` |
| `718017` | 36 | 02:00 | `seen_before_the_day_closes` |
| `735148` | 44 | 10:00 | `seen_before_the_day_closes` |
| `761663` | 33 | 04:00 | `seen_before_the_day_closes` |
| `777270` | 34 | 12:00 | `seen_after_the_day_closes` |
| `784723` | 44 | 22:00 | `seen_after_the_day_closes` |
| `795734` | 35 | 22:00 | `seen_after_the_day_closes` |
| `800441` | 44 | 04:00 | `seen_before_the_day_closes` |
| `812236` | 43 | 02:00 | `seen_before_the_day_closes` |
| `824245` | 35 | 09:00 | `seen_before_the_day_closes` |
| `829927` | 41 | 21:00 | `seen_after_the_day_closes` |
| `832781` | 28 | 20:00 | `seen_after_the_day_closes` |
| `833416` | 35 | 08:00 | `seen_before_the_day_closes` |
| `838462` | 32 | 08:00 | `seen_before_the_day_closes` |
| `869948` | 29 | 19:00 | `seen_after_the_day_closes` |
| `875901` | 33 | 11:00 | `seen_before_the_day_closes` |
| `879234` | 32 | 09:00 | `seen_before_the_day_closes` |
| `879321` | 34 | 16:00 | `seen_after_the_day_closes` |
| `881942` | 29 | 11:00 | `seen_before_the_day_closes` |
| `900784` | 39 | 00:00 | `seen_before_the_day_closes` |
| `902483` | 43 | 03:00 | `seen_before_the_day_closes` |
| `902607` | 28 | 21:00 | `seen_after_the_day_closes` |
| `933983` | 37 | 10:00 | `seen_before_the_day_closes` |
| `938034` | 29 | 00:00 | `seen_before_the_day_closes` |
| `940923` | 34 | 00:00 | `seen_before_the_day_closes` |
| `976067` | 41 | 05:00 | `seen_before_the_day_closes` |
| `976826` | 37 | 13:00 | `seen_after_the_day_closes` |
| `982391` | 29 | 21:00 | `seen_after_the_day_closes` |
| `988242` | 38 | 10:00 | `seen_before_the_day_closes` |
