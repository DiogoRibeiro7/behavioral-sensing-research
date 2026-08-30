# Confirmatory experiments

This directory contains the executable experiment protocol for the paper
**Failure-Aware Multimodal Behavioural Sensing**.

The manuscript's exploratory values are development diagnostics. They are not the
confirmatory experiment and must not be silently promoted into the final results.

## Current status

The frozen confirmatory simulator study has now been executed and accepted at
`N=200`. The accepted repository snapshot is in
`results/confirmatory_n200/`.

The frozen scientific revision remains
`bd5a9b10068b69fc75cc6f806904430633ce7408`. Later documentation, validation,
workflow and model-development commits do not move that experimental target.

## Files

- `config.json` — frozen, machine-readable design and reporting rules.
- `CONFIRMATORY_CONTRACT.md` — scientific decision contract frozen before results.
- `FREEZE_MANIFEST.json` — exact pre-results acceptance contract.
- `run_confirmatory.py` — shared implementation and development/smoke runner;
  direct output from this script is **not** eligible as the manuscript confirmatory
  artifact.
- `run_production.py` — authoritative frozen shard/merge route for confirmatory
  execution.
- `validate_frozen_artifact.py` — post-freeze acceptance validator.
- `results/confirmatory_n200/` — accepted compact result snapshot and provenance.

## Hypotheses

The runner maps directly to the manuscript hypotheses:

- **H1** — graceful degradation across 0%, 5%, 10%, 20%, and 40% random missingness.
- **H2** — sensor-count versus modality using fixed, named deployment subsets, with
  the five-sensor `radar_door_bed_wearable` configuration as the frozen primary
  non-inferiority comparison.
- **H3** — occupancy-aware versus naive resident attribution across visitor/carer
  regimes.
- **H4** — failure-aware inference versus a paired health-naive control that forces
  sensor reliability weights to one while retaining identical observations and
  health status calculation.
- **H5** — non-additive interactions for pre-specified sensor pairs.

Every comparison is paired by household seed. Individual time points are observations,
not independent replications.

## Smoke test

A short direct run verifies execution only:

```bash
python papers/failure-aware-multimodal-behavioural-sensing/experiments/run_confirmatory.py \
  --replications 2 \
  --output /tmp/behavioural-sensing-paper-smoke
```

A smoke or direct-run artifact cannot be used as the paper's confirmatory result.

## Confirmatory production route

The accepted study used the frozen `run_production.py` shard/merge route with the
stride-4 household seed sequence. Forty deterministic shards covered 200 independent
households, after which the exact shard union was merged and passed the frozen
artifact validator.

The pre-specified precision rule used the maximum Monte Carlo standard error across
all pre-specified H2 reduced-configuration balanced-accuracy gaps. The accepted
`N=200` artifact attained a maximum MCSE of `0.001124983614711242`, below the
`0.002` target, so no prospective extension beyond 200 households is required.

The frozen primary H2 decision remains the five-sensor comparison only. Its negative
non-inferiority result does not invalidate confirmatory status and cannot be replaced
post hoc by a better-performing secondary subset.

## Accepted output

The full accepted GitHub Actions bundle was produced by recovery run `33332161062`
from the immutable shards of production run `33319406297`. The recovery reran no
simulation. It merged the original shards with the frozen production code and
validated them using the original acceptance revision.

`results/confirmatory_n200/RESULTS_MANIFEST.json` records the workflow IDs, Git
revisions, artifact fingerprint, file SHA-256 values and interpretation guardrails.
`accepted_summary.json` contains the compact accepted summaries used for manuscript
integration.

## Interpretation rules

- These experiments estimate performance under the specified simulator, not field
  performance.
- Do not treat repeated time points as the replication unit.
- Do not change hypotheses, sensor subsets, missingness levels, margins, primary
  metrics or acceptance rules after inspecting confirmatory results.
- The primary H2 five-sensor non-inferiority result is negative and must remain
  negative in the manuscript.
- Secondary H2 configurations may be reported as pre-specified secondary evidence,
  but cannot rescue or replace the primary comparison.
- If a genuine scientific software defect is later found in the frozen experiment,
  affected results must be invalidated transparently and a new versioned experiment
  designed rather than silently rewriting this accepted snapshot.
- External validation remains a separate stage and should not be tuned to reproduce
  the simulator conclusions.
