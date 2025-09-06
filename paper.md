---
title: "Sensor Modeling Research Toolkit"
tags:
  - Python
  - Time series
  - Change point detection
  - Hidden Markov models
  - Non-homogeneous Poisson processes
  - Ambient assisted living
  - Behavioral sensing
authors:
  - name: Diogo Ribeiro
    orcid: 0009-0001-2022-7072
    email: dfr@esmad.ipp.pt
    affiliation: 1
affiliations:
  - name: ESMAD — Instituto Politécnico do Porto
    index: 1
date: 2025-08-28
bibliography: paper.bib
---

# Summary

The **Sensor Modeling Research Toolkit** is a Python library for end‑to‑end analysis of behavioral sensor streams. It couples typed, tested implementations for loading events, detecting change points, modeling intensities and discrete states, and generating compact visual summaries. The focus is on clarity and reproducibility: a minimal API that composes cleanly and favors transparent defaults over algorithmic breadth. The toolkit serves researchers and practitioners in ambient assisted living (AAL), digital health, and smart environments who need reliable baselines rather than black‑box systems. Typical applications include activity segmentation, routine profiling, anomaly detection, and longitudinal monitoring of home sensors. In prior deployments, similar modeling pipelines have powered online human‑activity recognition and energy‑aware sensing in smart homes [@asghari2019hmm; @cook2020cpam], demonstrating the practical value of lightweight CPD and HMM components.

# Statement of need

Projects in AAL and behavioral sensing gather multi‑modal events (motion, door, appliance, chair/bed, etc.) with irregular sampling and frequent missingness. Workflows face real‑world constraints: limited compute, strict privacy, short iterations with clinical partners, and the need to explain results to non‑technical stakeholders. Existing open‑source tools either target general machine‑learning pipelines or provide bespoke scripts for single datasets. Reproducible comparisons across change‑point detectors, intensity models for counts, and state‑space models remain uncommon, especially when small‑data defaults and lightweight deployment are required. Recent studies show that hierarchical HMMs and embedding‑based change detectors can segment daily routines and support real‑time activity recognition [@asghari2019hmm; @dadi2021embedding], yet each effort relies on custom implementations that are hard to reproduce or extend.

This toolkit addresses that gap. It packages established modeling ideas behind a consistent, explicit interface and includes helpers to go from raw events to counts, from counts to segments, and from segments to states. The surface area is intentionally compact so new researchers can audit and extend the code with minimal friction. In short, it offers a pragmatic path from raw data to interpretable artifacts suitable for monitoring and clinical review.

# State of the field

Change‑point detection (CPD) has matured substantially, with efficient exact solvers such as PELT for piecewise‑constant signals [@killick2012optimal] and broad surveys covering likelihood‑based and non‑parametric approaches [@troung2020review]. Online detectors based on Bayesian hazard modeling enable responsive monitoring in non‑stationary settings [@adams2007bocpd]. For multivariate, distribution‑free segmentation, energy‑based methods like E‑Divisive detect changes in dependence structure without parametric assumptions [@matteson2014edivisive]. Activity‑recognition systems increasingly pair CPD with representation learning to segment long smart‑home time series [@dadi2021embedding] or reduce energy consumption through adaptive sampling [@cook2020cpam].

For sequential modeling, Hidden Markov Models remain a compact and interpretable framework for phase structure [@rabiner1989hmm], while explicit‑duration variants (HSMMs) address dwell‑time biases and better capture routine segments [@johnson2013hsmm]. Event counts are naturally modeled by non‑homogeneous Poisson processes (NHPPs) with piecewise‑constant or smooth intensities, estimated by maximum likelihood; the Lewis–Shedler thinning procedure remains a standard tool for simulation and validation [@lewis1979thinning].

Several high‑quality libraries implement components of these ideas (e.g., general ML toolkits or CPD packages). However, there is persistent friction in AAL workflows: researchers must assemble data cleaning, segmentation, intensity estimation, and phase decoding by hand, and re‑implement plotting/reporting for each study. Prior work on hierarchical HMMs or embedding‑based CPD often ships as isolated scripts [@asghari2019hmm; @dadi2021embedding], making it difficult to compare approaches or reuse code. This project consolidates those building blocks into a concise, readable codebase geared towards behavioral sensor streams.

# Software description

## Design goals

1. **Small and explicit API.** Prefer a few well‑named classes and functions with typed inputs/outputs to extensive hierarchies.
2. **Reproducibility.** Deterministic defaults, minimal randomness, and example scripts/notebooks that run quickly end‑to‑end.
3. **Interpretability.** Outputs geared towards communication: segments as intervals, intensities as step functions, states as short alphabets.
4. **Extensibility.** Modular components so that alternative solvers (e.g., robust CPD penalties) or emissions (e.g., negative binomial counts) can be added locally.

## Architecture

- **Data layer (`datasets`, `preprocess`).** Event loaders (CSV/JSON) with schema validation; resampling to counts with missing‑value policies; synthetic generators for regression tests (step intensities, daily routines).
- **Models (`models`).** 
  - **CPD:** piecewise‑constant mean/intensity with PELT; optional BOCPD for online use; energy‑based CPD for distribution shifts.
  - **Counts:** NHPP with piecewise‑constant intensity; optional spline smoothing.
  - **States:** HMMs with discrete or Poisson emissions; Viterbi decoding and posterior state marginals; optional HSMM decoding when explicit durations are modeled.
- **Evaluation (`metrics`, `evaluation`).** Segmentation precision/recall over intervals; calibration curves for intensities; per‑segment distribution summaries; simple ablations.
- **Visualization (`plots`).** Compact figures: stepwise intensities with detected changepoints; state ribbons over time; count histograms per segment.

## Minimal usage example

```python
import pandas as pd
from sensor_modeling import datasets, models

# 1) Load events and aggregate to per-minute counts
events = datasets.load_demo_events()          # timestamp, sensor_id
X = datasets.to_counts(events, freq="1min")   # wide dataframe of counts

# 2) Segment a single channel with CPD (piecewise-constant intensity)
series = X["kitchen_motion"].to_numpy()
cpd = models.CPD(method="pelt", penalty="bic")
cpd.fit(series)
segments = cpd.segments_                      # [(start, end), ...]

# 3) Fit a small HMM on multivariate counts
hmm = models.HMM(n_states=4, emissions="poisson", random_state=0)
hmm.fit(X.to_numpy())
states = hmm.predict(X.to_numpy())            # array of length len(X)

# 4) Plot helpers (segments, intensities, states)
# plots.plot_segments(series, segments)
# plots.plot_states(states)
```

## Implementation notes

- **CPD.** For piecewise‑constant models we expose a PELT solver with BIC/MBIC penalties and maximum segments safeguards [@killick2012optimal]. The BOCPD variant follows a standard hazard‑based recursion [@adams2007bocpd]. The non‑parametric alternative uses energy statistics for multivariate change detection [@matteson2014edivisive].
- **NHPP counts.** Intensity estimation is performed by maximizing the Poisson likelihood over a partition (either fixed or selected by CPD), returning step functions that are easy to audit. Simulation uses Lewis–Shedler thinning for sanity checks [@lewis1979thinning].
- **HMM/HSMM.** We include discrete and Poisson emissions with EM training, Viterbi decoding, and posterior summaries; HSMM support adds explicit‑duration decoding to mitigate geometric dwell‑time bias [@rabiner1989hmm; @johnson2013hsmm].
- **Interfaces.** Public classes expose `.fit`, `.predict`/`.transform`, and `.plot_*` helpers. All arrays are NumPy‑compatible, and pandas is supported at the edges for IO and indexing convenience.

# Quality control

The repository includes unit tests for core components (CPD solvers, NHPP likelihoods, HMM forward–backward/Viterbi) and property‑based tests for invariants such as non‑decreasing segment endpoints, likelihood monotonicity under EM, and valid probability normalization. Synthetic generators create small canonical scenarios: single and multiple intensity steps, periodic routines with controllable dwell‑times, and missingness bursts. These scenarios are used for regression tests and visual baselines.

Continuous integration runs the test suite on Linux, macOS, and Windows for supported Python versions. Coverage is reported, and style checks enforce type hints and docstring consistency. Example notebooks are exercised (smoke tests) to ensure that end‑to‑end workflows keep running as the library evolves.

# Statement of functionality and comparison

The emphasis of this library is a **readable, end‑to‑end** pipeline tuned for behavioral sensors. In contrast to general‑purpose ML libraries, we provide opinionated defaults and small helpers that reduce friction between steps: converting raw events to counts, selecting partitions by CPD, estimating intensities, decoding phases, and producing figures that clinicians can read. Compared to specialized CPD packages, we trade algorithmic breadth for integrated usage and interpretation (plots, summaries, sanity‑check simulation). Where earlier studies present bespoke pipelines for hierarchical HMMs or representation‑learning‑based changepoints [@asghari2019hmm; @dadi2021embedding], this toolkit offers a unified, documented implementation that others can audit, benchmark, and build upon. The codebase is intentionally compact so that users can audit and extend it quickly.

# Availability

- **Source code:** public Git repository with an issue tracker and continuous integration.
- **Documentation:** API references and examples in `docs/` and online.
- **Community guidelines:** contributions welcomed via `CONTRIBUTING.md` and governed by a code of conduct.
- **License:** MIT.
- **Supported platforms:** Linux, macOS, Windows; Python ≥ 3.9.
- **Dependencies:** NumPy [@harris2020array], SciPy [@virtanen2020scipy], pandas [@mckinney2010data], scikit‑learn [@pedregosa2011scikit], Matplotlib [@hunter2007matplotlib].

# Acknowledgements

I thank colleagues and students at ESMAD for early feedback and discussions on sensor modeling in smart‑home deployments.

# References
