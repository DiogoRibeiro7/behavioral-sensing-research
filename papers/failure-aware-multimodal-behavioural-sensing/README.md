# Failure-Aware Multimodal Behavioural Sensing

LaTeX source for the methodological paper:

> **Failure-Aware Multimodal Behavioural Sensing: Probabilistic State Inference, Person Attribution, and Sensor Reduction in Smart Homes**

## Manuscript sources

- `manuscript_confirmatory.tex` — current results-bearing publication candidate.
- `confirmatory_results.tex` — confirmatory H1–H5 result section, traceable to the accepted frozen result snapshot.
- `main.tex` — pre-results manuscript retained for provenance; it still contains exploratory/pre-execution wording and is no longer the publication candidate.
- `references.bib` — bibliography used by both manuscript versions.
- `experiments/` — frozen experiment specification, executable production route, validator, and repository-retained accepted N=200 result snapshot.

All confirmatory numerical results are simulator-derived and must not be interpreted as estimates of real-home performance.

## Build the confirmatory manuscript

The paper uses `biblatex` with the `biber` backend. Run from this directory:

```bash
pdflatex manuscript_confirmatory.tex
biber manuscript_confirmatory
pdflatex manuscript_confirmatory.tex
pdflatex manuscript_confirmatory.tex
```

The confirmatory manuscript inputs `confirmatory_results.tex`, whose values are copied from the accepted repository snapshot at:

```text
experiments/results/confirmatory_n200/accepted_summary.json
```

That snapshot records the frozen experiment revision, config digest, accepted workflow provenance, MCSE gate, primary H2 decision, and H1–H5 summaries.

## Confirmatory status

The frozen experiment was executed at N=200 paired household trajectories. The maximum observed H2 balanced-accuracy MCSE was `0.0011249836`, below the pre-specified `0.002` target, so no prospective replication extension was required.

The frozen primary H2 result is negative and must remain negative in downstream writing: the five-sensor `radar_door_bed_wearable` deployment had a full-minus-reduced balanced-accuracy gap of `0.1547836825` with 95% paired interval `[0.1531002596, 0.1564131485]`, compared with the non-inferiority margin `0.02`. The much smaller gap for the eight-sensor `objects_plus_wearable` configuration is secondary evidence about the sensor-information frontier and cannot replace the primary comparison.

H1, H3, and H4 are deliberately reported as mixed where their metrics or scenarios disagree. H5 reports all four pre-specified interactions. No result should be simplified to a stronger claim than the frozen summaries support.

## Remaining scientific boundary

The confirmatory simulator study is complete. The next scientific stage is external validation on held-out real smart-home data without tuning the architecture to reproduce the simulator conclusions. In particular, external validation should preserve the negative H2 result and the observed metric disagreements as hypotheses to test rather than problems to optimise away.
