# Release Readiness Review

Final integration review of the multimodal ambient-sensing work. Written for a
human reviewer deciding whether the diff is ready for a version bump.

**Recommendation: `0.2.0` as a research-platform release, after the correction
pass below.** The work is additive and backwards-compatible, so a minor bump is
correct; the scope is far too large for a patch.

Nothing has been merged to `main`, tagged, or published.

> **How to read the gate table.** An earlier revision of this document asserted
> that all gates passed. They did not. The suite was green on one developer
> machine running Python 3.13, while the CI matrix had been failing on Python
> 3.11 for every merge in the series: `Observation` used a `mappingproxy` as a
> dataclass field default, which 3.10 accepts under its older `isinstance`
> check and 3.12+ accepts because `mappingproxy` became hashable, but which
> 3.11 rejects at class-definition time. Twenty-one test files could not be
> collected. Nothing in the local workflow could see it.
>
> The lesson is procedural, not technical: **a gate status written by hand is a
> claim, not a measurement.** Read the status of the latest run on `develop`
> in GitHub Actions rather than trusting the table below, and treat any
> hand-written figure here as provenance for when it was taken, not as current.

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

Figures below were taken at the commit named in the footer. **Verify against
Actions before releasing.**

| Gate | Status |
| --- | --- |
| CI matrix (3.10, 3.11, 3.12) | Must be green on `develop`; check Actions, not this table |
| Test suite | Local run, single interpreter — necessary, never sufficient |
| Coverage | 86% overall; 95% across the eleven new packages |
| `mypy` (strict) | **0 errors in all new packages**; 168 pre-existing in 32 older files |
| `ruff` | Clean |
| `black` | Clean on every file touched |
| `bandit` | No issues identified across 14,969 lines |
| `mkdocs build --strict` | Builds with no warnings |
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
clock drift. Re-run at the head of this branch and reproduces exactly, so
unlike the ablation figures these did not drift:

| Metric | Value |
| --- | --- |
| Balanced accuracy | 0.824 |
| Macro F1 | 0.806 |
| Calibration error | 0.083 |
| Recall (sleeping / away) | 0.954 / 0.959 |
| Transition timing | 98.3% matched, median offset 0 min |

**Robustness**, 0–40% record loss: balanced accuracy 0.858 → 0.748, calibration
error 0.046 → 0.079. No cliff.

> **Replication counts differ by experiment.** The ablation has been run at
> study scale. Attribution and change detection have not, and are marked as
> pilots below. Counts adequate for inference are derived in
> [Simulation protocols](SIMULATION_PROTOCOLS.md).

**Sensor ablation** (study, n = 100 paired seeds, 14 days, seeds derived from
root `20260829`):

| Comparison against all ten sensors | Difference | MCSE | 95% CI | dz |
| --- | --- | --- | --- | --- |
| objects_plus_wearable (8) | +0.0073 | 0.0005 | [+0.0063, +0.0083] | 1.44 |
| radar_door_bed_wearable (5) | +0.1707 | 0.0025 | [+0.1657, +0.1755] | 6.70 |
| object_sensors_only (6) | +0.1878 | 0.0025 | [+0.1828, +0.1928] | 7.41 |
| radar_door_bed (3) | +0.4775 | 0.0025 | [+0.4727, +0.4824] | 19.35 |
| minimal_door_bed (2) | +0.5264 | 0.0019 | [+0.5226, +0.5303] | 27.30 |

The achieved MCSE of 0.0005 on the headline comparison is the precision the
protocol specified for n = 100, so the replication count did what it was chosen
to do.

The artefact for this run is stamped `git_dirty: true`, because it was produced
while the working tree still held the changes being reviewed. The numbers are
sound and the seeds reproduce them, but a run intended to be cited should be
made from a clean checkout so the commit alone identifies the code. That the
stamp is there, and says so, is the provenance working.

**This supersedes the four-seed pilot, which overstated the effect.** The pilot
reported the eight-sensor gap as 0.012 with a nominal interval of
[+0.004, +0.020]. At n = 100 it is 0.0073 with an interval roughly six times
narrower, and the pilot's point estimate falls outside it. Four seeds were not
merely imprecise; they were misleading, which is the concrete argument for the
protocol.

**Calibration does not follow accuracy.** The five-sensor configuration is by
some distance the best calibrated, at 0.0332 expected calibration error
(MCSE 0.0008) against 0.0839 (MCSE 0.0005) for the full ten-sensor deployment,
while scoring 0.171 lower in balanced accuracy:

| Configuration | Sensors | Calibration error | MCSE |
| --- | --- | --- | --- |
| radar_door_bed_wearable | 5 | 0.0332 | 0.0008 |
| objects_plus_wearable | 8 | 0.0829 | 0.0006 |
| all_modalities | 10 | 0.0839 | 0.0005 |
| object_sensors_only | 6 | 0.1240 | 0.0013 |
| radar_door_bed | 3 | 0.1309 | 0.0009 |
| minimal_door_bed | 2 | 0.1792 | 0.0012 |

At four seeds this looked like a curiosity worth following up. At a hundred it
is a large, well-resolved effect, and it bears directly on the project's
central question: a sparser deployment that knows when it does not know may be
preferable to a denser one that is confidently wrong. Note the caveat that
applies to all of it -- the states being scored come from the same simulator,
so this establishes a property of the inference model, not of any home.

**Attribution** (study, n = 100 paired seeds per scenario, 10 days, seed root
`20260830`). **This supersedes the one-seed demonstration and contradicts it.**

| Scenario | Contam. | Accuracy gain | 95% CI | Calibration gain | Visitor recall |
| --- | --- | --- | --- | --- | --- |
| resident_alone | 0.00 | -0.0004 | [-0.0007, -0.0001] | +0.0006 | 0.00 |
| resident_goes_out | 0.00 | -0.0006 | [-0.0011, -0.0003] | +0.0010 | 0.00 |
| short_visitor | 0.04 | +0.0003 | [-0.0003, +0.0008] | +0.0014 | 0.61 |
| prolonged_visitor | 0.08 | +0.0007 | [+0.0000, +0.0014] | +0.0027 | 0.60 |
| carer_visits | 0.03 | +0.0065 | [+0.0055, +0.0074] | +0.0034 | 0.28 |
| visitor_and_carer | 0.08 | +0.0063 | [+0.0053, +0.0074] | +0.0040 | 0.49 |
| **resident_without_wearable** | 0.04 | **-0.0118** | [-0.0136, -0.0102] | +0.0043 | 0.31 |
| no_radar | 0.04 | +0.0064 | [+0.0053, +0.0075] | +0.0037 | 0.15 |
| sparse_coverage | 0.04 | +0.0048 | [+0.0037, +0.0060] | +0.0032 | 0.28 |

Three claims in the previous revision do not survive replication.

**"No effect when nobody else is present" is false.** Both uncontaminated
scenarios show a small but statistically clear *loss*, with intervals excluding
zero. The effect is tiny -- four to six ten-thousandths of balanced accuracy --
but it is a cost, not a no-op, and the command said otherwise.

**"Largest gain +0.032 during a carer round" is out by a factor of five.** At
100 seeds a carer round gives +0.0065.

**The largest effect attribution has is a harm.** When the resident is not
wearing the wearable, attribution costs 0.0118 balanced accuracy, an order of
magnitude larger than any gain it produces anywhere. The mechanism is
consistent with the design: with no wearable to place the resident, ambient
activity that really was theirs gets discounted as possibly a visitor's. The
guard against attributing a visitor's activity to the resident becomes, in the
absence of the evidence it depends on, a way of discarding the resident's own.

**Calibration improves in every scenario, including those where accuracy
falls.** All nine intervals exclude zero. This is the coherent reading of the
whole table: discounting ambient evidence makes the estimate less confident.
Where the discount is correct, accuracy rises too. Where it is wrong, accuracy
falls but the model hedges rather than committing confidently to the wrong
state. Whether that trade is worth making depends on what consumes the output,
which is not a question simulation can settle.

**Visitor detection is weaker than the single seed suggested.** The quoted 0.48
recall was approximately the `visitor_and_carer` case. Across scenarios recall
runs from 0.15 (`no_radar`) to 0.61 (`short_visitor`), with precision from 0.37
to 0.77. Attribution's benefit is bounded by this: the component whose value is
being measured detects a minority of the visits it keys on.

**Change detection** (pilot, n = 3 seeds): a step change detected at 5.0 days
with 0.010 false alerts per person-day on a stable record. Losing 30% of
records did not raise the false-alert rate; detection delay rose to 7.7 days.
Gradual change is harder — recall 0.67, delay 12.5 days. Delay figures are now
pooled medians over detections; an earlier revision averaged each seed's
median, which is not a median and weighted a seed detecting one change equally
with a seed detecting twenty.

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

Full list in `docs/limitations.md`. The three that matter most:

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
| Declared defaults do not transfer: 22 real homes, median balanced accuracy 0.420 against 0.816 | **High** |
| `home_inactive` recall 0.16 on real homes; declared event rates 7-14x below measured | **High** |
| Abstention does not fire when wrong on real data (median 2.5%, max 3.9%) | **High** |
| Simulator reports 0.816, above the 0.607 recoverable from real instrumentation | **High** |
| Pipeline recovers two thirds of what real sensors support (0.420 of 0.607) | **High** |
| Small-sample pilots reported as though they were studies | High, addressed |
| Attribution's value overstated by a single-seed demonstration | High, addressed |
| Declared rather than fitted parameters | **High** |
| Shared evidence between occupancy and state layers | High |
| Undetectable partial loss on event sensors | Medium |
| Attribution costs accuracy when the wearable is absent (-0.0118) | High |
| Visitor recall 0.15-0.61 by scenario, weakest without radar | Medium |
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

---

## 11. Pre-release correction pass

Applied in response to an external review of this branch. Each item was a
defect in the work as it stood, not a refinement.

### Blockers

| Was | Now |
| --- | --- |
| `Observation` used `MappingProxyType({})` as a dataclass default, which Python 3.11 rejects at class-definition time. 21 test files could not be collected; the CI matrix had been red on 3.11 for the entire series | `default_factory=dict`. `__post_init__` rebuilt the mapping regardless, so the default was always discarded. Verified by importing under 3.11 directly |
| Nothing caught it, because the local suite ran on one interpreter | Two guards: every module is imported as a test, and every dataclass field default is checked against an allowlist of genuinely immutable types. The second fails on any interpreter, including ones where dataclasses would accept the value |
| `pytest.ini` contained only coverage flags and silently overrode the far stricter `[tool.pytest.ini_options]`, so `--strict-markers`, `--strict-config` and the warning policy were never applied | `pytest.ini` deleted; the pyproject configuration is now the only one |
| This document asserted that all gates passed | The assertion is replaced by a pointer to Actions, and a `gate` job aggregates every required job into one check that branch protection can require |
| `main` held documentation infrastructure `develop` lacked | `main` merged in. The Sphinx workflow it carried was dropped rather than restored, see below |

### Scientific and reporting

| Was | Now |
| --- | --- |
| `DetectionStudy.summary()` reported `float(np.mean(delays))` over per-seed medians while the provenance definition promised a median of delays | Delays are pooled before the median. `mean_seed_median_delay_days` is reported separately and named for what it is, and `detected_changes` records the sample size behind it |
| Four-seed and one-seed results were described as "real but small" | Labelled pilots and demonstrations, with replication counts derived from Monte Carlo standard error in [Simulation protocols](SIMULATION_PROTOCOLS.md). A study needs at least 100 paired trajectories; these ran 4 and 1 |
| "Two runs of the same command produce byte-identical output", contradicted by the `recorded_at` stamp the implementation deliberately adds | The guarantee is stated over results, not files: same seed, same commit, same resolved configuration produce the same scientific results |
| Provenance recorded the package version, which stays fixed across many commits | Adds `git_commit`, `git_dirty`, `schema_version`, and a snapshot of the resolved algorithm defaults, so a record cannot be confused with one produced after a default changed |
| `research_identifier()` claimed studies were unlinkable, but scoped only the visible prefix; the HMAC stayed `HMAC(salt, subject)`, leaving identical digests across studies | The study derives a subkey, so digests are unrelated. The test that should have caught this compared whole identifiers, which differ by prefix alone; it now compares digests |

### On the documentation stack

The review recommended standardising on Sphinx, on the basis that no MkDocs
configuration existed. That was true of the snapshot reviewed but is no longer
true of `develop`: the migration completed in #71. There is no `docs/conf.py`,
no `.rst` source, and the documentation extra installs MkDocs only.

Reverting would mean recreating a Sphinx tree to replace a working one, and
reinstating the Pandoc dependency whose absence was failing the old
documentation job. The requirement behind the recommendation — one stack,
consistently configured, enforced in CI — is met by keeping MkDocs:

- `mkdocs build --strict` is the gate, warnings included;
- the built site is uploaded as an artifact;
- link checking runs as a separate, non-blocking step, so an external site
  going down does not fail a release;
- `.readthedocs.yaml` points at `mkdocs.yml`, and the Sphinx workflow inherited
  from `main` is removed rather than left to fail.

### Still outstanding

- **Branch protection is a repository setting, not a file.** The `gate` job
  exists, but requiring it is a change to repository configuration that has not
  been made here.
- **The production-scale runs have not been executed.** The protocols are
  specified; the numbers in section 5 are still the pilots.
- Real-data validation remains the next milestone, and is unaffected by any of
  the above.
