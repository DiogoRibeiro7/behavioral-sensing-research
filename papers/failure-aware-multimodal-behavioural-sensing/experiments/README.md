# Confirmatory experiments

This directory contains the executable experiment protocol for the paper
**Failure-Aware Multimodal Behavioural Sensing**.

The manuscript's exploratory values are development diagnostics. They are not the
confirmatory experiment and must not be silently promoted into the final results.

## Files

- `config.json` — frozen, machine-readable design and reporting rules.
- `run_confirmatory.py` — executes H1–H5 on paired simulated household trajectories.
- `results/` — generated outputs; not source-of-truth and should not be hand edited.

## Hypotheses

The runner maps directly to the manuscript hypotheses:

- **H1** — graceful degradation across 0%, 5%, 10%, 20%, and 40% random missingness.
- **H2** — sensor-count versus modality using fixed, named deployment subsets.
- **H3** — occupancy-aware versus naive resident attribution across visitor/carer regimes.
- **H4** — failure-aware inference versus a paired health-naive control that forces sensor reliability weights to one while retaining identical observations and health status calculation.
- **H5** — non-additive interactions for pre-specified sensor pairs.

Every comparison is paired by household seed. Individual time points are observations,
not independent replications.

## Smoke test

A short run verifies execution only:

```bash
python papers/failure-aware-multimodal-behavioural-sensing/experiments/run_confirmatory.py \
  --replications 2 \
  --output /tmp/behavioural-sensing-paper-smoke
```

A smoke run is marked `pilot-or-incomplete` and cannot be used as the paper's
confirmatory result.

## Confirmatory run

After the experimental software revision has passed the repository release gates and
is frozen:

```bash
python papers/failure-aware-multimodal-behavioural-sensing/experiments/run_confirmatory.py
```

The default configuration starts with 200 independent household trajectories. The
result is labelled `confirmatory` only when both conditions hold:

1. the configured minimum replication count is reached; and
2. the maximum primary H2 Monte Carlo standard error is at or below 0.002.

If the MCSE gate fails, increase the replication count without changing the model or
hypotheses, up to the pre-specified maximum of 1000.

## Outputs

The runner writes:

- `confirmatory_results.json` — complete provenance, resolved configuration, seeds,
  hypothesis-level results, summaries, and the MCSE gate;
- `h1.csv` ... `h5.csv` — household-level analysis tables;
- `generated_results.tex` — generated LaTeX macros for manuscript integration.

The result artefact records the Git commit and dirty-worktree state. A final paper run
should have `git_dirty: false`.

## Interpretation rules

- These experiments estimate performance under the specified simulator, not field
  performance.
- Do not treat repeated time points as the replication unit.
- Do not change hypotheses, sensor subsets, missingness levels, or primary metrics
  after inspecting confirmatory results.
- If software defects are found after the freeze, fix them transparently, invalidate
  affected results, increment the experiment schema/config as needed, and rerun all
  paired arms from the same seed set.
- External validation remains a separate stage and should not be tuned to reproduce
  the simulator conclusions.
