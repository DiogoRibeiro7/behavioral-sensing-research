# Failure-Aware Multimodal Behavioural Sensing

LaTeX source for the methodological paper:

> **Failure-Aware Multimodal Behavioural Sensing: Probabilistic State Inference, Person Attribution, and Sensor Reduction in Smart Homes**

## Contents

- `main.tex` — manuscript source.
- `references.bib` — bibliography used by the manuscript.
- `experiments/` — frozen confirmatory experiment specification and executable H1–H5 runner.

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

## Confirmatory experiment

The machine-readable design is in `experiments/config.json`. A short execution can be used only as a smoke test:

```bash
python experiments/run_confirmatory.py --replications 2 --output /tmp/behavioural-sensing-paper-smoke
```

After the experimental software revision has passed the repository release gates and is frozen, run the pre-specified experiment with:

```bash
python experiments/run_confirmatory.py
```

The default protocol starts at 200 paired household trajectories and requires the primary Monte Carlo standard-error gate before an artefact can be marked `confirmatory`. See `experiments/README.md` for the full rules and outputs.

## Research status

The paper is a working manuscript. Before submission:

1. fix the repository release blockers and freeze the experimental software revision;
2. execute the confirmatory paired simulation study with the pre-specified replication and MCSE rules;
3. replace or clearly separate exploratory pilot results from confirmatory results;
4. use the generated result artefacts and LaTeX macros rather than hand-entering confirmatory numbers;
5. validate the frozen framework against at least one annotated external smart-home dataset;
6. confirm author affiliation and submission metadata.

Generated LaTeX build artefacts and submission-specific files should not be treated as source-of-truth unless explicitly added for a release or archive.
