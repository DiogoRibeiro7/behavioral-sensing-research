# Research Questions

What this platform is built to investigate, what would answer each question,
and what would falsify the answer. Questions are stated so that a negative
result is publishable and recognisable.

## Central question

> Can multimodal ambient sensing recover useful behavioural state using fewer
> physical sensors, while explicitly representing uncertainty, sensor failure,
> missingness, visitors, and person attribution?

"Useful" is deliberately not "accurate". A system that is 90% accurate and
badly calibrated is less useful for longitudinal monitoring than one that is
80% accurate and knows when it is guessing, because the second can abstain and
the first cannot.

## RQ1 — Does modality substitute for count?

**Question.** Does adding a small number of information-rich, person-bound
sensors recover the behavioural information lost by removing several ambient
sensors?

**How it is answered.** `sensor-modeling ablate` evaluates named sensor
subsets on identical simulated households and reports paired differences with
bootstrap intervals.

**What would answer it affirmatively.** A configuration with materially fewer
sensors whose paired difference from the full deployment is small in
*magnitude*, not merely statistically indistinguishable.

**Current state.** Adding a wearable and beacon to six object sensors closes
most of the gap. Over 100 paired seeds the remaining difference is 0.0073
balanced accuracy, 95% CI [+0.0063, +0.0083], MCSE 0.0005: eight sensors
recover all but three quarters of a point from ten. An earlier four-seed pilot
put this at 0.012 with an interval six times wider, and the study estimate
falls outside that interval, so the pilot is superseded rather than refined.
See [Simulation protocols](SIMULATION_PROTOCOLS.md).

**What would falsify it.** A configuration that looks equivalent on balanced
accuracy but is materially worse calibrated, or that degrades sharply under
sensor failure. Both are measured, precisely because accuracy alone would hide
them.

So far the evidence runs the other way. The five-sensor configuration is the
*best* calibrated of any tested, at 0.0332 expected calibration error against
0.0839 for the full ten-sensor deployment, while giving up 0.171 balanced
accuracy. Fewer sensors cost accuracy and bought honesty about uncertainty.
Which of those matters more depends on what the output is used for, and that
is a question this repository can pose but not settle from simulation.

**Known threat.** The result depends on the simulator's assumed activity
levels and sensor rates. See `limitations.md`.

## RQ2 — Can a system fail safely rather than fail silently?

**Question.** When sensors break, does behavioural inference degrade visibly
and gracefully, or does it produce confident nonsense?

**How it is answered.** Reliability sweeps at 5/10/20/40% missingness,
injected dropouts, stuck sensors, wearable non-adherence, and total blackout,
all scored with calibration alongside accuracy.

**What would answer it affirmatively.** Accuracy declining smoothly while
calibration error stays bounded, and abstention rising when evidence
genuinely disappears.

**Current state.** On the simulator, balanced accuracy 0.858 → 0.748 across
0–40% loss with calibration error 0.046 → 0.079, and under total blackout the
system abstains.

**What would falsify it.** Any regime where confidence stays high while
accuracy collapses. This is the failure mode the platform exists to prevent,
and it is tested adversarially rather than assumed.

**This falsification condition has been met.** On real recordings, over 60,948
scored steps, stated confidence separates correct from incorrect answers by only
+0.073, and the most confident band (0.95–1.00) is *less* accurate at 0.561 than
the band beneath it at 0.653 while covering 39% of all steps. That is confidence
staying high while accuracy collapses, measured rather than hypothesised, and no
threshold repairs it: raising the bar discards the pipeline's best band and keeps
the saturated one. The frozen confirmatory simulation agrees by a different
route — mean abstention rose only from 9.7e-06 to 5.3e-05 as missingness went
from 0% to 40%, so the mechanism barely responds to evidence disappearing.

RQ2 is therefore answered negatively on the evidence available. The architecture
still prevents the specific failure it was designed against — a failed sensor
contributes no state preference and never votes for inactivity — but that is a
weaker property than RQ2 asks about, and the system does not currently fail
safely in the sense defined here. See [real_data.md](real_data.md) for the
per-band figures and [limitations.md](limitations.md).

## RQ3 — How much does person attribution matter?

**Question.** How much does visitor and carer activity distort behavioural
conclusions if ambient events are attributed to the resident by default?

**How it is answered.** Synthetic households include a weekday carer and
occasional visitors who trip the same ambient sensors. Naive attribution can
be compared against occupancy-aware attribution on identical trajectories.

**What would answer it affirmatively.** A measurable difference in state
inference or in the behavioural-change verdicts between the two, in the
direction of fewer spurious findings under occupancy-aware attribution.

**What would falsify it.** Attribution making no measurable difference, which
would mean either that contamination is negligible in this simulator or that
the occupancy model is too weak to exploit it. Both are worth knowing, and
visitor recall of roughly 0.48 means the second is a live possibility.

## RQ4 — Can a personal baseline be adaptive without being amnesic?

**Question.** Can a baseline track genuine long-run change in a person's
routine while still detecting a decline, rather than absorbing the decline as
the new normal?

**How it is answered.** Injecting a known persistent change on a known day and
measuring detection delay, alongside the unmatched alert burden per
person-day during stable periods.

**What would answer it affirmatively.** Detection within a clinically
plausible window at an alert burden a carer would tolerate.

**Current state.** Detection at six days, roughly 0.033 unmatched behavioural
alerts per person-day.

**What would falsify it.** A regime where lowering the threshold enough to
detect real change floods the recipient. The threshold trades directly against
delay, and that trade-off is a property of the method, not a tuning accident.

## RQ5 — What is the cost of honesty?

**Question.** What does a system give up by abstaining, by discounting
unattributable evidence, and by refusing to fill gaps?

**How it is answered.** Selective accuracy and abstention rate are reported
alongside accuracy, so the accuracy sacrificed to abstention is visible.

**Why it matters.** Every safety property in this platform has a cost in
apparent performance. Reporting only the headline number would hide the price
and make the conservative system look worse than a reckless one.

## Explicit non-questions

These are out of scope, and results here should not be read as bearing on
them.

- **Clinical effectiveness.** Nothing in this repository supports a claim that
  monitoring improves any health outcome.
- **Diagnosis.** No state maps to a clinical condition. `sleeping` is bed
  occupancy with low movement, not polysomnographic sleep.
- **Identity.** Attribution is a probability derived from anonymous evidence,
  never a biometric identification.
- **Multi-resident state tracking.** The ontology models one person; others
  are detected but not tracked.
- **Real-world performance.** Every quantitative result here comes from a
  simulator written by this project.

## What would make these answers trustworthy

In priority order, and until the first is done, all answers above are
conditional on the simulator:

1. Evaluation on a public annotated smart-home dataset (CASAS, ARAS, MARBLE).
2. Emission and dwell parameters fitted from data rather than declared.
3. A second, independently written simulator with different structural
   assumptions.
4. Calibration assessed across households, not only within one.
5. Prospective assessment of alert burden with people who would act on the
   alerts.
