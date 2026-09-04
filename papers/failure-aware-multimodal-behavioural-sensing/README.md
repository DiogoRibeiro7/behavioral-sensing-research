# Failure-Aware Multimodal Behavioural Sensing

LaTeX source for the methodological paper:

> **Failure-Aware Multimodal Behavioural Sensing: Probabilistic State Inference, Person Attribution, and Sensor Reduction in Smart Homes**

## Manuscript sources

- `main.tex` — the manuscript. One document, carrying every level of evidence the
  project has: exploratory simulator results, the frozen confirmatory simulation,
  development results on 22 real annotated homes, and the one-shot external
  validation.
- `confirmatory_results.tex` — confirmatory H1-H5 result section, `\input` by
  `main.tex` and traceable to the accepted frozen result snapshot.
- `references.bib` — bibliography.
- `experiments/` — frozen experiment specification, executable production route,
  validator, and repository-retained accepted N=200 result snapshot.

There was previously a second manuscript, `manuscript_confirmatory.tex`, holding
the confirmatory results while `main.tex` was meant to stay frozen as a
pre-results snapshot. That split did not survive contact with the work: `main.tex`
was maintained rather than frozen, so it stopped being a record of what the
project claimed before the results existed, and the two documents drifted — a
correction applied to one could sit unapplied in the other, which happened. The
manuscripts are now consolidated into `main.tex`. Use git history for what the
paper said at any earlier point; that is what history is for.

Confirmatory results are simulator-derived. The external-validation result is
confirmatory on real data. Development results on real recordings are neither,
and are labelled as such in the text.

## Build

The paper uses `biblatex` with the `biber` backend. Run from this directory:

```bash
latexmk -pdf main.tex
```

or equivalently:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

`main.tex` inputs `confirmatory_results.tex`, whose values are copied from the
accepted repository snapshot at:

```text
experiments/results/confirmatory_n200/accepted_summary.json
```

That snapshot records the frozen experiment revision, config digest, accepted
workflow provenance, MCSE gate, primary H2 decision, and H1-H5 summaries.

## Confirmatory status

The frozen experiment was executed at N=200 paired household trajectories. The maximum observed H2 balanced-accuracy MCSE was `0.0011249836`, below the pre-specified `0.002` target, so no prospective replication extension was required.

The frozen primary H2 result is negative and must remain negative in downstream writing: the five-sensor `radar_door_bed_wearable` deployment had a full-minus-reduced balanced-accuracy gap of `0.1547836825` with 95% paired interval `[0.1531002596, 0.1564131485]`, compared with the non-inferiority margin `0.02`. The much smaller gap for the eight-sensor `objects_plus_wearable` configuration is secondary evidence about the sensor-information frontier and cannot replace the primary comparison.

H1, H3, and H4 are deliberately reported as mixed where their metrics or scenarios disagree. H5 reports all four pre-specified interactions. No result should be simplified to a stronger claim than the frozen summaries support.

## Remaining scientific boundary

The confirmatory simulator study is complete. The next scientific stage is external validation on held-out real smart-home data without tuning the architecture to reproduce the simulator conclusions. In particular, external validation should preserve the negative H2 result and the observed metric disagreements as hypotheses to test rather than problems to optimise away.
