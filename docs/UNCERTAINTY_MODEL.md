# Uncertainty Model

How uncertainty is represented at each layer, how it propagates, where it is
deliberately approximated, and what is not modelled at all.

## The rule everything else follows from

> A missing observation is missing evidence. It is never negative evidence
> about the resident.

Stated formally: for a sensor `s` with reliability `r_s = 0`, its contribution
to the log-posterior is identically zero for every state. It does not favour
low-activity states. It does not favour anything.

This is enforced by the arithmetic rather than by a special case, which is
why it cannot be forgotten in a future edit.

## Layer by layer

### 1. Observation

Two independent scalars, both in `[0, 1]`:

- `quality` — the device's report on its own hardware.
- `confidence` — an upstream estimator's belief in the value.

`evidence_weight() = quality x confidence`. Multiplicative because both must
hold.

Provenance flags record what ingestion repaired: `CLOCK_ADJUSTED`,
`UNIT_CONVERTED`, `LATE_ARRIVAL`, `OUT_OF_ORDER`, `IMPUTED`. A downstream
reader can always tell a measured timestamp from a corrected one.

### 2. Sensor health

Produces `reliability ∈ [0, 1]` per sensor:

```text
reliability = prior_reliability x smoothed_quality x status_weight
```

The status weight is a fixed, documented table (see `SENSOR_DATA_MODEL.md`).
`MISSING` is zero; `UNKNOWN` is 0.5, because absence of a verdict is not a
verdict of failure.

### 3. Occupancy context

A posterior over four contexts, maintained by a continuous-time chain.
Marginalised into scalars:

```text
P(resident_home)         = P(alone) + P(with_visitor)
P(visitor_present)       = P(with_visitor) + P(visitor_only)
P(multiple_people)       = P(with_visitor)
attribution              = sum_c P(c) x resident_share(c)
```

`resident_share` is a declared per-context constant: 1.0 when alone, 0.5 in a
shared household, 0.0 when the resident is out. The 0.5 encodes genuine
ignorance about who did what, which is the honest value when two people are
present and nothing distinguishes them.

### 4. Fusion

The core quantity is `P(Z_t | O_1:t)`, maintained by forward filtering.

**Prediction.** `b <- b · exp(Q·Δt)`, where `Q` is the ontology generator.
Uncertainty grows with elapsed time automatically: with no evidence the belief
relaxes toward the stationary distribution, so confidence decays rather than
persisting.

**Update.** Each sensor contributes a tempered, centred log-likelihood:

```text
loglik_s = w_s · r_s · a_s · (raw_s − max(raw_s))
```

- `w_s` — modality evidence weight, correcting for sampling-rate imbalance.
- `r_s` — reliability from health.
- `a_s` — attribution from context.

**Why tempering rather than gating.** Multiplying the log-likelihood by a
weight in `[0, 1]` is a power-likelihood update in the sense of generalised
Bayesian updating. It degrades smoothly, it has a defined meaning at every
intermediate value, and at zero it is exactly a flat likelihood. A hard gate
would be discontinuous and would need a threshold nobody can justify.

**Why centring.** State-independent constants cancel under normalisation, so
subtracting the maximum changes nothing about the posterior. It makes each
model's output directly readable as "how much worse is this state than the
best one, according to this sensor", which is what the explanation layer
reports.

### 5. What each estimate exposes

`StateEstimate` never reports only an argmax:

| Quantity | Meaning |
| --- | --- |
| `probabilities` | The full posterior |
| `confidence` | Mass on the leading state |
| `margin` | Gap between the top two |
| `normalised_entropy` | 1.0 means completely undecided |
| `completeness` | Mean sensor reliability behind the estimate |
| `supporting` / `contradicting` | Per-sensor log-likelihood margin for the reported state |
| `silent` | Trusted sensors that reported nothing |
| `missing` | Sensors discounted to zero reliability |

The `silent` / `missing` distinction matters. A working-but-quiet event sensor
is *informative*: under a Poisson model its silence actively penalises the
states that would have triggered it. A dead sensor is not. Collapsing the two
would be the same error as filling event gaps with zeros.

### 6. Abstention

```text
state = UNKNOWN if confidence < min_confidence
                 or completeness < min_completeness
```

Two independent reasons, because they are different failures. A flat posterior
means the evidence is ambiguous; low completeness means there was barely any
evidence to be ambiguous about.

`UNKNOWN` is excluded from the ontology's state vector by construction. It is
an abstention, not an eighth behaviour, and `StateOntology` rejects any attempt
to include it.

### 7. Baseline

Daily features are accumulated from the posterior:

```text
hours(state) = sum_t P(state | t) · Δt
```

not by counting argmax wins. A day of 60%-confident guesses and a day of
99%-confident conclusions are genuinely different, and collapsing to argmax
first would erase that difference before the baseline saw it.

Deviation is a **robust** z-score against a median/MAD reference, weekday-aware
where enough same-weekday history exists. `min_scale` floors the denominator
so an extremely regular person does not have every ordinary hour of variation
reported as an enormous deviation.

Poorly observed days are excluded from the history entirely, not entered as
low values.

### 8. Alerts

Confidence is `coverage x attribution`. Below `min_confidence` nothing is
raised regardless of magnitude: a large change seen through a broken
deployment, or during a visit, is not actionable.

Above it, uncertainty is preserved as explicit **caveats** on the alert rather
than folded into the score, so the recipient sees the reservation rather than
a number that silently absorbed it.

## Deliberate approximations

Each of these is a known departure from exactness, chosen for interpretability
or tractability, and recorded rather than hidden.

| Approximation | Where | Consequence |
| --- | --- | --- |
| Sensors are conditionally independent given the state | Fusion | Log-likelihoods add. Correlated sensors are over-counted. Partially compensated by per-modality `weight`. |
| Presence samples treated as independent, then discounted | Context | `sample_weight` is chosen, not estimated. Without it the posterior reaches certainty within minutes. |
| Fast wearable sampling discounted by a fixed weight | Fusion defaults | `weight=0.25` is a judgement, not a fit. |
| Filtering, not smoothing | Baseline | Each estimate is attributed to the interval preceding it, which lags slightly at transitions. |
| Markov dynamics | Ontology | State duration is implicitly exponential. Real dwell times are not. |
| Bootstrap intervals across seeds | Evaluation | Quantify Monte-Carlo variability under one model, not uncertainty about the model. |

## Not modelled

- **Correlation between the occupancy and state layers.** They share radar and
  beacon evidence, so their errors are dependent, and the reported uncertainty
  does not reflect that. This is the largest known gap.
- **Parameter uncertainty.** Dwell times, rates and priors are point values.
  There is no posterior over them.
- **Calibration correction.** Calibration is measured and reported; no
  temperature scaling or isotonic correction is applied.
- **Uncertainty in the ontology itself.** The state set is assumed correct.

## How to check these claims

The properties above are tested rather than asserted:

- `tests/test_fusion.py` — reliability zero contributes nothing; partial
  reliability scales linearly; posteriors normalise; abstention triggers.
- `tests/test_adversarial.py` — total blackout abstains; higher missingness is
  never rewarded; every belief remains a valid distribution under combined
  loss, duplication, lateness and drift.
- `tests/test_evaluation.py` — calibration error rises with overconfidence;
  log loss punishes confident errors more than hedged ones.
