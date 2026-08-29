# Failure-Aware Multimodal Behavioural Sensing

LaTeX source for the methodological paper:

> **Failure-Aware Multimodal Behavioural Sensing: Probabilistic State Inference, Person Attribution, and Sensor Reduction in Smart Homes**

## Contents

- `main.tex` — manuscript source.
- `references.bib` — bibliography used by the manuscript.

The manuscript deliberately separates exploratory simulator-only results from the pre-specified confirmatory study. Current numerical values should not be interpreted as estimates of real-home performance.

## Build

The paper uses `biblatex` with the `biber` backend.

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Run the commands from this directory so that `references.bib` is resolved locally.

## Research status

The paper is a working manuscript. Before submission:

1. fix the repository release blockers and freeze the experimental software revision;
2. execute the confirmatory paired simulation study with the pre-specified replication rule;
3. replace or clearly separate exploratory pilot results from confirmatory results;
4. record the exact Git commit, resolved configuration, seeds, metric definitions, and environment for all reported experiments;
5. validate the frozen framework against at least one annotated external smart-home dataset;
6. confirm author affiliation and submission metadata.

Generated LaTeX build artefacts and submission-specific files should not be treated as source-of-truth unless explicitly added for a release or archive.
