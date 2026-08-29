# Sensor Modeling Research Toolkit

A research-grade Python toolkit for **interpretable, probabilistic,
privacy-preserving** analysis of behavioural sensor data, and for multimodal
ambient sensing in assisted-living and digital-health research.

!!! warning "This is a research toolkit, not a medical device"

    Nothing it produces is a diagnosis, and no claim of clinical
    effectiveness is made or supported anywhere in this project. Every
    quantitative result in this documentation comes from the bundled
    simulator and has **not** been validated against real sensor data. See
    [Limitations](limitations.md).

## Five kinds of thing, kept distinct

Most of the design follows from refusing to conflate these:

| Kind | What it is | Example |
| --- | --- | --- |
| **Measured observation** | A sensor reported a value at an instant | The fridge contact closed at 08:14 |
| **Derived feature** | A value an upstream device computed, with its own confidence | The radar reports 2 tracked people |
| **Inferred state** | A posterior over what the resident was probably doing | `P(kitchen_activity) = 0.81` |
| **Behavioural change** | A shift against the resident's own history | Sleep has trended down for three weeks |
| **Alert** | A judgement that a person should look at something | An `attention` alert, with its caveats |

A sensor event is not a behaviour:

```text
fridge opening      != eating
tap activation      != drinking
toilet event        != confirmed toileting
door event          != resident movement
missing observation != inactivity
```

Two rules are enforced by the arithmetic rather than by convention:

- **A missing observation is missing evidence, never negative evidence.**
  Sensor reliability enters the fusion likelihood as a tempering weight, so a
  failed sensor contributes a flat likelihood and cannot look like a quiet
  resident.
- **Ambient activity is not automatically the resident's.** Occupancy
  estimation produces `P(activity was the resident's)`, which discounts
  evidence while a visitor or carer may be present.

The system can also answer `unknown`. Abstention is a first-class output.

## The pipeline

```text
heterogeneous observations
        ↓
observations/   validation, units, ordering, provenance
        ↓
health/         per-sensor reliability
        ↓
context/        who is present, and whose activity it was
        ↓
fusion/         P(Z_t | O_1:t) over latent behavioural states
        ↓
baseline/       adaptive personal normal, change verdicts
        ↓
alerts/         restrained, explained alerts
        ↓
interop/        provenance-preserving export
```

`online/` orchestrates this incrementally with bounded memory;
`simulation/` supplies ground truth; `evaluation/` measures it.

## Try it

```bash
pip install -e ".[dev]"

# One seeded experiment covering visitors, sensor failure, record loss
# and a known behavioural change.
sensor-modeling demo --days 90 --seed 20240304

# Does modality substitute for sensor count?
sensor-modeling ablate --days 14 --seeds 11 22 33 44

# What is person attribution actually worth?
sensor-modeling attribution --days 10
```

Every command is deterministic for a fixed seed.

## Where to go next

<div class="grid cards" markdown>

- **[Usage examples](usage.md)** — the shortest path to a fitted model.
- **[Ingesting sensors](multimodal_ingestion.md)** — how four unlike sensors
  reach one pipeline with no hardware-specific logic downstream.
- **[Inference](inference.md)** — what each layer claims and where it stops.
- **[Evaluation](evaluation.md)** — how it is measured, and why circular
  validation is avoided.
- **[Limitations](limitations.md)** — what the platform does *not* establish.
- **[Adversarial review](ADVERSARIAL_REVIEW.md)** — a deliberate attempt to
  invalidate it.

</div>

## Design principles

- Interpretable statistical models before opaque ones. No neural component is
  on the inference path, and no large language model is anywhere near it.
- No cameras, microphones, or biometric identification. Attribution is derived
  from anonymous evidence and expressed as a probability.
- Sensor failure is part of the model, not an operational afterthought.
- Scientific algorithms are separable from I/O, storage and presentation.
- A model that does not know must be able to say so.
