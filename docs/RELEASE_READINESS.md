# Release Readiness Review

Final integration review of the multimodal ambient-sensing work. Written for a
human reviewer deciding whether the diff is ready for a version bump.

**Recommendation: `0.2.0`.** The work is additive and backwards-compatible, so
a minor bump is correct; the scope is far too large for a patch.

Nothing has been merged to `main`, tagged, or published.

---

## 1. Implemented capabilities

The pipeline runs end to end:

```text
heterogeneous sensors -> canonical observations -> sensor health
    -> occupancy and attribution -> probabilistic behavioural state
    -> adaptive baseline -> change detection -> explained alerts
    -> evaluation and provenance-preserving export
```

| Package | Responsibility |
| --- | --- |
| `observations` | Canonical hardware-neutral model, registry, boundary validation, unit conversion, clock-drift correction, legacy adapters |
| `health` | Online per-sensor reliability as an evidence weight |
| `context` | Occupancy contexts and uncertainty-aware attribution |
| `states` / `fusion` | Continuous-time ontology and recursive multimodal filter |
| `baseline` | Adaptive, weekday-aware, non-stationary personal baselines |
| `alerts` | Restrained alerting with deduplication and rate limiting |
| `simulation` | Synthetic households with controlled ground truth |
| `evaluation` | Metrics, paired ablation, attribution and detection studies, provenance |
| `online` | Incremental orchestration, snapshotting, benchmarks |
| `interop` | FHIR-style export, pseudonymisation, redaction |

Four reproducible commands: `demo`, `ablate`, `attribution`, plus the original
`bernoulli-ar` and `nhpp-pelt`.

## 2. Quality gates

| Gate | Status |
| --- | --- |
| Test suite | **654 passing**, 0 failing |
| Coverage | 86% overall; 95% across the eleven new packages |
| `mypy` (strict) | **0 errors in all new packages**; 168 pre-existing in 32 older files |
| `ruff` | Clean |
| `black` | Clean on every file touched |
| `bandit` | No issues identified across 14,969 lines |
| Sphinx | Builds with no warnings |
| `python -m build` + `twine check` | Passing; all packages ship in the wheel |
| Worked example | Runs from the command line, deterministic for a fixed seed |

## 3. Backwards compatibility

**No public symbol was removed or changed.** `tests/test_backwards_compatibility.py`
exercises the original APIs in the way they were used before: Bernoulli AR,
NHPP-PELT, the HMM variants, PELT, the lightweight change-point detectors, the
analysis pipeline, behavioural metrics, `SensorDataset`, and the original CLI
subcommands.

A user of the previous release can upgrade and ignore the new packages
entirely. `ObservationStream` bridges the two layers in both directions.

**One pre-existing defect was fixed.** `AnalysisPipeline` defaulted to
`NHPPConfig(n_basis=3)`, which violates the model's own `n_basis >= degree+1`
constraint for the cubic default. Every fit raised, and the pipeline recorded
`{"error": ...}` in place of a result — so the NHPP arm had never once worked,
while appearing present in the output. Now `n_basis=4`, with a regression test.

### One regression found by this review

Wrapping experiment results in a provenance record nested the findings under
`results`, which broke the CLI integration test asserting the old flat shape.
It reached `develop` because that change was verified with targeted tests
rather than the full suite. Fixed, with the attribution artefact now covered
too.

The lesson is procedural rather than technical: a change to an output *shape*
needs the whole suite run, because the tests that break are rarely the ones
near the change.

## 4. Dependencies

**No new dependencies were added.** Every one of the eleven new packages
imports only `numpy`, `pandas` and `scipy`, all already declared as core
requirements.

One observation for the maintainer, not acted on: `pyproject.toml` declares an
`ml` extra pulling `tensorflow`, `torch` and `transformers`. Nothing in the
repository uses them, and the platform's stated position is that no neural
component belongs on the inference path. Removing the extra would be a
breaking change for anyone installing it, so it is flagged rather than changed.

## 5. Evaluation results

All figures come from the bundled simulator. See section 7.

**State inference**, 90-day demonstration with visitors, a bed-sensor dropout,
five days of wearable non-adherence, 3% loss, duplicates, late arrival and
clock drift:

| Metric | Value |
| --- | --- |
| Balanced accuracy | 0.824 |
| Macro F1 | 0.806 |
| Calibration error | 0.083 |
| Recall (sleeping / away) | 0.954 / 0.959 |
| Transition timing | 98.3% matched, median offset 0 min |

**Robustness**, 0–40% record loss: balanced accuracy 0.858 → 0.748, calibration
error 0.046 → 0.079. No cliff.

**Sensor ablation**, four paired seeds: adding a wearable to six object sensors
leaves a gap of 0.012 balanced accuracy against the full ten-sensor deployment
(95% CI [+0.004, +0.020]) — real but small. A five-sensor configuration was the
best calibrated of all despite lower accuracy.

**Attribution**: no effect when nobody else is present, largest gain (+0.032)
during a carer round.

**Change detection**, three seeds: a step change detected at 5.0 days with
0.010 false alerts per person-day on a stable record. Losing 30% of records did
not raise the false-alert rate; detection delay rose to 7.7 days. Gradual
change is harder — recall 0.67, delay 12.5 days.

**Performance**: 19,345 observations/second, step latency p95 3.8 ms, snapshot
6.5 KB. Once past the baseline history cap, retained state grows 1.01× for a 3×
longer run.

## 6. Scientific assumptions

Stated so a reviewer can disagree with them:

1. Sensors are conditionally independent given the state. Mitigated for the
   redundant case by declared redundancy groups; not solved in general.
2. State durations are exponentially distributed, as the Markov assumption
   implies. Real dwell times are not.
3. Successive presence samples are treated as independent and then discounted
   by a chosen weight, rather than modelled as correlated.
4. Emission rates, dwell times, occupancy priors and alert thresholds are
   declared, not fitted.
5. The resident is one person. Others are detected but not tracked.
6. Daily features use a filtering, not smoothing, approximation.

## 7. Known limitations

Full list in `docs/limitations.rst`. The three that matter most:

**Everything comes from a simulator.** No component has been validated against
real sensor data. The simulator was written by this project; its generative
process is deliberately different from the estimator and two guard tests keep
them apart, but that reduces circularity rather than eliminating it. Every
number above describes behaviour on this simulator and must not be quoted as an
estimate of field performance.

**Occupancy and state layers share evidence.** Both consume radar and beacon
signals, so their errors are dependent and the reported uncertainty does not
model that.

**Partial record loss remains partly undetectable.** For a sensor with a
declared cadence, under-delivery is now caught. For a purely event-driven
sensor there is no promised rate, and a drop in activations is
indistinguishable from a quieter resident. Where loss correlates with activity
this biases inference toward inactivity — measured at kitchen recall 0.705 →
0.180 under 60% activity-correlated loss.

## 8. Remaining risks

| Risk | Severity |
| --- | --- |
| Simulator-only validation | **High** |
| Declared rather than fitted parameters | **High** |
| Shared evidence between occupancy and state layers | High |
| Undetectable partial loss on event sensors | Medium |
| Visitor recall of 0.48; short visits missed | Medium |
| 168 pre-existing type errors; CI gates `mypy` on two files | Medium |
| No smoothing or backfill; late records dropped past tolerance | Medium |
| Unversioned snapshots | Low |

## 9. Recommended future work

In priority order. The first blocks any research claim:

1. **Validate against a public annotated dataset** (CASAS, ARAS, MARBLE) with
   the same metrics.
2. **Fit emission and dwell parameters from data**, comparing against the
   documented defaults.
3. **Model the occupancy–state dependency** with a joint or hierarchical
   formulation.
4. **Improve visitor recall**, currently conservative by construction.
5. **Assess calibration across households**, not only within one.
6. **Prospective assessment of alert burden** with people who would act on the
   alerts, since false-positive tolerance is a human judgement no metric
   supplies.
7. Reduce pre-existing type debt and widen the CI `mypy` gate.

## 10. What this remains

Interpretable, probabilistic, privacy-preserving, sensor-agnostic,
failure-aware, person-context-aware, reproducible, testable, edge-capable.

No large language model is anywhere on the inference or alerting path, and none
is planned. No neural component is on the inference path. There are no cameras,
microphones or biometric identification anywhere in the design.

**This is a research toolkit. It is not a medical device, and no claim of
clinical effectiveness is made or supported anywhere in this repository.**
