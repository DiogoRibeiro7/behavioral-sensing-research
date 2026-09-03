# Multimodal Ambient Sensing Architecture

Design record for the multimodal layer built on top of the original modelling
core. This document states what was audited, what the target architecture is,
what the contracts between layers are, and what would count as each stage
being finished.

User-facing documentation lives in `ambient_architecture.md`, `inference.md`,
`evaluation.md` and `limitations.md`. This file records the *decisions*.

## 1. Audit of the pre-existing repository

### What was already there and is reused

| Component | Status | How it is used now |
| --- | --- | --- |
| `models/bernoulli_ar` | Working, tested | Unchanged. Still reachable via CLI and the analysis pipeline. |
| `models/nhpp_pelt` | Working, tested | Unchanged. |
| `models/change_point_detection/pelt.py` | Working, well validated | **Reused** by the adaptive baseline to locate step changes. |
| `hmm/` | Working, lightweight | Retained. The fusion layer needed continuous-time dynamics and a per-modality likelihood, which `BaseHMM` does not provide, so it was not extended in place. |
| `change_point/` | Working | Retained unchanged. |
| `utils/missing.py` | Gap-aware masks and summaries | Retained. Complements, rather than duplicates, reliability tempering. |
| `utils/data_io.SensorDataset` | Tabular container | Retained as the interop point for the tabular models. |
| `analysis/` | Granger, dependency networks, metrics | Retained unchanged. |
| `visualization/`, `cli.py` | Working | CLI extended with two subcommands; visualisation untouched. |
| CI, packaging, release metadata | Working | Unchanged approach. |

### Gaps that blocked multimodal fusion

1. **No canonical observation.** Every model consumed a `DataFrame` on a
   regular grid. There was nowhere to record modality, unit, device quality,
   estimator confidence, or provenance, and no way to express that two
   sensors report on different time bases.
2. **No temporal semantics.** Nothing distinguished an event stream from a
   sampled signal, so gap-filling could not be made safe by construction.
3. **Sensor health was a detached utility.** `data/validation.detect_sensor_failures`
   returned booleans that no model consumed, so a failed sensor could not be
   discounted during inference.
4. **No notion of who generated an activation.** Ambient events were
   implicitly the resident's.
5. **Fixed-step models only.** Asynchronous, irregular arrival could only be
   handled by resampling, which fabricates data for event streams.
6. **Ground truth and inference were not separated** in the simulator, so
   evaluation risked being circular.

### Technical debt noted but deliberately not addressed

Fixing these was out of scope; addressing them would have mixed unrelated
changes into scientific work.

- 164 `mypy` errors across 31 older modules; CI gates `mypy` on two files.
- `utils/data_io.simulate_sensor_data` and `analysis/pipeline` have high
  cyclomatic complexity.
- Several demo scripts under `examples/demos/` call APIs with wrong argument
  names and would fail if run.
- `pyproject.toml` declares heavyweight optional extras (`tensorflow`,
  `torch`, `transformers`) that nothing in the repository uses.

## 2. Target architecture

```text
heterogeneous observations
        v
observations/   canonical model, registry, boundary validation
        v
health/         per-sensor reliability as an evidence weight
        v
context/        occupancy and attribution
        v
states/ fusion/ latent behavioural state, P(Z_t | O_1:t)
        v
baseline/       adaptive personal normal, change verdicts
        v
alerts/         restrained, explained alerts
        v
interop/        provenance-preserving export

online/         incremental orchestration of the above
simulation/     synthetic households with ground truth
evaluation/     metrics and paired ablation
```

### Why these boundaries

- **`observations` is the only place hardware appears.** Everything above it
  reads a `SensorRegistry`, never a sensor name. This is what makes ablation a
  registry subset rather than a code change.
- **`health` is separate from `observations`** so a malfunctioning sensor
  never looks like a malformed message, and vice versa.
- **`context` is separate from `fusion`** because "who is present" and "what
  is the resident doing" are different questions with different state spaces.
  Coupling them would have forced a joint state space that grows
  multiplicatively.
- **`states` holds the ontology, `fusion` holds the inference.** The ontology
  is a deployment configuration; the filter is an algorithm.
- **`online` performs no inference.** It owns buffering, day boundaries and
  snapshotting only, so every scientific layer stays usable without it.
- **`simulation` must not import from `fusion` or `states`.** Enforced by a
  test.

### Modules considered and rejected

| Proposed | Decision |
| --- | --- |
| `modalities/` | Rejected. Modality is a field on an observation and a dispatch key in `fusion.defaults`; a package would have been an empty wrapper. |
| `quality/` | Renamed to `health/`, which is what it actually estimates. |
| `occupancy/` | Renamed to `context/`, since it produces attribution as well as occupancy. |
| `preprocessing/` | Rejected. Modality-specific preprocessing belongs in the emission models, where the likelihood already encodes what the values mean. |

## 3. Data contracts between layers

Each contract is enforced at runtime, not merely documented.

| Boundary | Contract |
| --- | --- |
| ingestion → everything | `Observation`: timezone-aware timestamp, registered `sensor_id`, declared modality, value finite and in the declared unit. Violations are rejected and counted, never raised into the stream. |
| `health` → `fusion` | `dict[sensor_id, float]` in `[0, 1]`. Zero means *contributes nothing*, not *observed nothing*. |
| `context` → `fusion` | `dict[sensor_id, float]` in `[0, 1]`: probability the resident generated what this sensor saw. Attributable sensors are always 1.0. |
| `fusion` → `baseline` | `StateEstimate`: a normalised posterior over the ontology, plus completeness and per-sensor evidence. |
| `baseline` → `alerts` | `BehaviouralChange`: a verdict with its kind, magnitude, duration, trend strength and the reference it was judged against. |
| `alerts` → `interop` | `Alert`: severity, score, confidence and explicit caveats. |

Two invariants hold everywhere and are tested:

1. Every belief vector is finite, non-negative and sums to one.
2. No layer may turn absence of evidence into evidence of absence.

## 4. Backwards compatibility

The multimodal layer is **purely additive**. No public symbol was removed or
changed.

- Existing model APIs, `SensorDataset`, the analysis pipeline and the
  `bernoulli-ar` / `nhpp-pelt` CLI subcommands behave identically.
- New CLI subcommands were added; none were altered.
- `ObservationStream` provides `event_counts()`, `sample_frame()` and
  `state_frame()` as the bridge from the canonical model into the tabular form
  the older models expect.
- `sensor_modeling.__all__` gained entries; it lost none.

No migration is required. A user of the previous release can upgrade and
ignore the new packages entirely.

## 5. Staged plan and acceptance criteria

| Stage | Delivered | Acceptance criterion | Status |
| --- | --- | --- | --- |
| 1 | Canonical observations, registry, ingestion, stream | Naive timestamps rejected; DST complete in both directions; event streams never forward-filled | Met |
| 2 | Sensor health | A quiet event sensor is never called broken; a missing sensor yields reliability 0 | Met |
| 3 | Ontology and fusion | Posterior normalised; reliability 0 contributes a flat likelihood; abstention available | Met |
| 4 | Occupancy and attribution | Attribution falls when a visitor is present; attributable sensors stay at 1.0; no biometric evidence used | Met |
| 5 | Adaptive baseline | Weekly rhythm not reported as change; poorly observed days excluded from history | Met |
| 6 | Alerts | An unusual day is not an alert; bursts are capped; health alerts disclaim behavioural reading | Met |
| 7 | Simulation | Generative process independent of the estimator, enforced by test | Met |
| 8 | Evaluation and ablation | Paired design structural; unpaired series refused | Met |
| 9 | Online pipeline | Bounded memory; snapshot/restore round-trips through JSON | Met |

These are delivery conditions, not empirical validation. Stage 3's criterion is
that abstention is *available*, and it is implemented and wired in, so the stage
is met as specified. Whether abstention is *informative* is a separate question,
and on the evidence available the answer is no — see the measured behaviour in
[uncertainty model](UNCERTAINTY_MODEL.md) and [real data](real_data.md).
| 10 | Interoperability | Measurements and inferences distinguishable by provenance | Met |

## 6. Architectural risks

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Occupancy and state layers share radar/beacon evidence, so their errors are correlated | **High** | Documented in `limitations.md`. Not modelled. Would require a joint or hierarchical formulation. |
| Emission parameters are declared, not fitted | **High** | Defaults derived from registry declarations and documented as starting points. Fitting is the top roadmap item after real-data validation. |
| Simulator encodes the project's own beliefs about behaviour | **High** | Generative process deliberately differs from the estimator; two guard tests prevent convergence. Does not eliminate the risk. |
| Presence samples are correlated but combined as independent | Medium | Explicit, inspectable `sample_weight` discount rather than an independence claim. |
| Ontology is single-resident | Medium | Documented scope limit. Occupancy detects others but cannot track their states. |
| No smoothing or backfill; late records beyond tolerance are dropped | Medium | Counted in `pipeline.too_late` rather than silently absorbed. |
| Snapshots are unversioned | Low | Ontology mismatch is rejected with a clear error rather than misread. |

## 7. Principles that constrained the design

- Interpretable models before opaque ones. No neural component is on the
  inference path, and none is planned.
- No large language model anywhere in inference or alerting.
- No cameras, microphones, or biometric identification. Attribution is
  achieved from anonymous evidence and expressed as a probability.
- Scientific algorithms are separable from I/O, storage and presentation.
- A model that does not know must be able to say so.
