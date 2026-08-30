# Accepted confirmatory N=200 results

This directory is the compact, repository-retained snapshot of the validator-passing
confirmatory simulation for **Failure-Aware Multimodal Behavioural Sensing**.

## Acceptance chain

- frozen scientific revision: `bd5a9b10068b69fc75cc6f806904430633ce7408`;
- frozen config SHA-256: `7b48ec89155f78ae7965ac59c9b17a6bf164889ed7f38194c35ba1576aedfae4`;
- original 40-shard production run: `33319406297`;
- merge-only recovery run: `33332161062`;
- original acceptance revision: `311b3a375638fdbe27825da8d190f9e1582f56b6`;
- accepted GitHub Actions artifact: `paper-confirmatory-n200-recovered` (`9737957755`);
- artifact SHA-256: `05f73ed90c7c89a92b580f9ff64d06b1654799f3484dbde6b6068a95e2d36ae6`.

The recovery reran **no household simulation**. It reused the 40 immutable shard
artifacts from the original production run, verified their non-outcome provenance,
merged them with the frozen production runner, and applied the original frozen
validator.

## Files

- `RESULTS_MANIFEST.json` — immutable provenance, artifact fingerprints and
  interpretation rules.
- `accepted_summary.json` — compact copy of the accepted hypothesis summaries,
  precision gate and H2 primary decision.
- `freeze_validation.json` — exact frozen-validator output.
- `generated_results.tex` — exact LaTeX macros emitted by the frozen production
  runner.

The complete accepted bundle also contains the household-level `h1.csv` through
`h5.csv` tables and `confirmatory_results.json`. Their SHA-256 digests are frozen in
`RESULTS_MANIFEST.json`; they are reproducible from the frozen code, environment and
seed sequence.

## Scientific status

The artifact is a valid confirmatory simulator result at `N=200`. The maximum
pre-specified H2 balanced-accuracy-gap MCSE is `0.001124983614711242`, below the
`0.002` threshold, so the prospective replication extension stops at `N=200`.

The frozen primary H2 five-sensor non-inferiority claim is **not supported**:
the full-minus-reduced balanced-accuracy gap is `0.15478368245435956` with a 95%
paired bootstrap interval `[0.15310025955167178, 0.15641314849498592]`, far above
the `0.02` margin. Secondary sensor subsets remain secondary and cannot replace the
registered primary comparison.

All results in this directory are simulator-derived. They are not estimates of
field performance.
