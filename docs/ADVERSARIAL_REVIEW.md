# Adversarial Review

A deliberate attempt to invalidate the platform, conducted as if the authors
intended to publish research based on it. Findings are classified by how much
they should stop that.

| Severity | Meaning |
| --- | --- |
| **BLOCKING** | Would make a published result wrong. Must be fixed before use. |
| **MAJOR** | Materially distorts results under conditions that will occur. |
| **MINOR** | Real but bounded; worth knowing when interpreting output. |
| **INFORMATIONAL** | Correct behaviour that looks like a defect, recorded so it is not "fixed" into a bug. |

Findings marked *fixed* have a regression test named alongside them.

---

## BLOCKING

**None outstanding.**

Two findings were blocking when discovered during development and are fixed:

### B1. A silent deployment was read as observed inactivity — *fixed*

Event sensors make no promise to report, so the health monitor rightly refused
to call their silence a failure. But when a gateway died and *every* sensor
went quiet at once, the filter read the resulting silence as a genuinely
inactive resident, at full confidence.

Sensors declaring an `expected_interval` now act as canaries for the delivery
path. When at least two of them, and enough of them, fall silent, event sensors
silent over the same period are downgraded so their silence stops carrying
evidential weight. Only *silence* counts as a canary signal; a stuck sensor is
still delivering and says nothing about whether other records are arriving.

Regression: `test_adversarial.py::test_a_total_outage_produces_abstention_not_confident_inactivity`

### B2. A sleeping resident made the bed sensor look broken — *fixed*

Stuck detection counted consecutive identical readings for every sensor kind.
A bed sensor correctly reporting "occupied" through eight hours of sleep was
therefore flagged `STUCK` and discounted to reliability 0.10 — removing the
strongest evidence for sleep at exactly the moment it mattered.

Consecutive identical readings now indicate a fault only for *sampled* sensors,
which are supposed to track a varying quantity. A state sensor is judged stuck
only once its level has persisted beyond any plausible real duration.

Fixing this raised clean-record balanced accuracy from 0.805 to 0.858.

Regression: `test_adversarial.py::test_a_night_of_bed_occupancy_is_not_a_stuck_sensor`

---

## MAJOR

### M1. Redundant sensors drove the posterior to false certainty — *fixed*

The filter combines sensors as conditionally independent given the state. Two
sensors watching the same doorway are not independent, and nothing prevented
their agreement being counted twice.

Measured on identical copies of a single physical event, using deliberately
weak evidence -- a rate of 2/hour for `kitchen_activity` against 1/hour
elsewhere, and one activation -- so that the posterior was not already
saturated and the compounding was visible:

| Redundant copies | P(kitchen_activity) |
| --- | --- |
| 1 | 0.056 |
| 2 | 0.098 |
| 4 | 0.270 |
| 8 | 0.809 |

Eight views of one event turned a 5.6% belief into 81%. A deployment with
overlapping coverage would have been systematically overconfident, and nothing
in the output would have revealed it.

`SensorSpec` now takes a `redundancy_group`, and `default_emissions` divides
the evidence weight across the group. With the group declared, the posterior is
identical (0.041) for one copy or eight.

This does not solve correlation in general — two sensors can be strongly
dependent without being redundant — but it addresses the case that actually
occurs in deployments and makes the assumption visible at the point of
declaration.

Regression: `test_adversarial_fixes.py::TestRedundancy`

### M2. Partial record loss biases inference toward inactivity — *partly fixed*

The platform guards hard against a sensor that *stops* reporting. It did not
guard against one that keeps reporting while dropping a share of its records,
because the health monitor saw ongoing traffic and rated it healthy.

This matters because the loss is rarely at random. A radio contended by
movement, or a battery sagging under load, drops records precisely when the
resident is active. Measured over eight simulated days:

| Condition | Balanced accuracy | ECE | Recall(kitchen) | Recall(inactive) |
| --- | --- | --- | --- | --- |
| Complete | 0.748 | 0.136 | 0.705 | 0.850 |
| 30% loss, at random | 0.680 | 0.133 | 0.639 | 0.912 |
| 60% loss *while active* | 0.593 | 0.214 | **0.180** | **0.985** |

Kitchen recall collapses to 0.18 while inactivity recall rises to 0.985: the
system reports a quiet resident because the evidence of activity is what went
missing. This is the exact conclusion the platform exists to avoid reaching by
accident.

**Fixed part.** A sensor that declared an `expected_interval` promised a
cadence, so a sustained shortfall against it is detectable even when no
individual gap is long. Delivering below `delivery_floor` (default 60%) of the
promised rate is now `DEGRADED`, which discounts the sensor instead of
weighting it as healthy.

**Unfixed part.** For a purely event-driven sensor there is no promised rate,
and a drop in activations is indistinguishable from the resident being less
active — that is the same ambiguity that makes silence uninformative for those
sensors. Detection there is not possible without redundancy or a heartbeat.

Recommendation for deployments: give every sensor a heartbeat where the
hardware allows it. It converts an undetectable bias into a detectable one.

Regression: `test_adversarial_fixes.py::TestUnderDelivery`

### M3. Occupancy and state layers share evidence — *open*

Both consume the radar and beacon signals. They answer different questions, but
their errors are consequently dependent: a radar fault degrades attribution and
state inference together, and the reported uncertainty does not model that
correlation.

Not fixed. A correct treatment needs a joint or hierarchical formulation, which
is a research change rather than a repair. Recorded in
`docs/limitations.rst` and in the roadmap.

---

## MINOR

### m1. Presence samples are correlated but combined as independent

Successive radar or beacon readings mostly re-observe an unchanged situation.
Treating them as independent drives the occupancy posterior to certainty within
minutes. A `sample_weight` discount (default 0.2) is applied.

The discount is chosen, not estimated. It is an inspectable correction rather
than a claim of independence, and it is why occupancy posteriors are
deliberately conservative.

### m2. Filtering, not smoothing

Daily features attribute each estimate's posterior to the interval *preceding*
it. This lags slightly at transitions. A fixed-interval smoother would be more
accurate; none is implemented, and no estimate is ever revised once emitted.

### m3. Late records beyond tolerance are discarded

Reordering succeeds within `lateness_tolerance`; anything later is counted in
`pipeline.too_late` and dropped rather than replayed. A deployment with long
uplink outages needs a replay strategy the pipeline does not provide.

### m4. Pairing inflates standardised effect sizes

Paired comparison removes between-household variance, which is correct, but it
makes Cohen's *dz* large in a way an unpaired field study would not reproduce.
Raw mean differences are reported alongside, and the caveat is stated where the
ablation results are presented.

### m5. Snapshots are unversioned

State is checked against the current ontology and context set, so a mismatched
snapshot is rejected with a clear error rather than misread. There is no schema
version, so a future change to the state set invalidates stored snapshots
rather than migrating them.

---

## INFORMATIONAL

Behaviour that looks wrong at first inspection and is not. Recorded so it does
not get "fixed" into a defect.

### i1. Attribution is not exactly a no-op when the resident is alone

Balanced accuracy is identical, but probabilities differ by order 1e-4. The
occupancy model never becomes *certain* the resident is alone, and that
residual uncertainty propagates. A model that reached certainty here would be
overconfident.

### i2. A sensor that never reported is `UNKNOWN`, not `MISSING`

It may not be installed. Only a sensor that stops after establishing a cadence
can be judged missing.

### i3. Abstentions count as errors

`UNKNOWN` is never a true label, so a reported `UNKNOWN` is scored wrong. This
makes the conservative system look worse on raw accuracy, which is why
`selective_accuracy` and `abstention_rate` are always reported beside it.

### i4. A detected change produces follow-on alerts

After a real change is reported, the trend detector keeps firing on the same
sustained shift, so the changed arm shows a higher unmatched-alert rate than
the stable arm (0.033 against 0.010 per person-day). Deduplication bounds it,
but a genuine change does cost more than one alert.

### i5. Spring-forward duplicates are collapsed, not lost

Two records whose local wall-clock times straddle a DST gap can resolve to the
same absolute instant. If they share a sensor and value they are indistinguishable
and are collapsed. Advancing time in UTC avoids constructing them.

---

## Areas probed with no finding

Checked deliberately, nothing to report:

- **Probability integrity.** Every belief remains finite, non-negative and
  normalised under combined loss, duplication, lateness and clock drift.
- **Time handling.** DST transitions in both directions, reordered events,
  duplicate timestamps, late arrival, per-source clock drift.
- **Missingness sweep.** 5/10/20/40% loss degrades accuracy smoothly with
  bounded calibration error and no cliff.
- **Alert storms.** A wholly broken deployment produces fewer than 15 alerts
  over ten days and no behavioural findings at all.
- **Circular validation.** Guard tests assert the simulator does not import the
  inference transition model, and the pipeline does not import ground truth.
- **Unpaired comparison.** `AblationReport.series()` refuses to return a series
  when a configuration is missing a seed.
- **Calibration metric.** Expected calibration error rises with overconfidence
  and falls with honest uncertainty on constructed cases.
- **Public API.** No symbol was removed or changed; the original models, CLI
  subcommands and `SensorDataset` behave as before.

---

## Standing recommendation

Every finding above was found against a simulator. The most likely place for an
undiscovered BLOCKING defect is the gap between that simulator and a real home,
and no amount of adversarial testing inside the simulator will close it.
Validation against a public annotated dataset remains the highest-value next
step.
