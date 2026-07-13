Project Roadmap
===============

The canonical GitHub-facing roadmap is maintained in ``ROADMAP.md`` at the
repository root. This documentation page mirrors that roadmap for the Sphinx
site.

This roadmap describes the planned direction for the Sensor Modeling Research
Toolkit. It is intentionally release-oriented: items are grouped by the outcome
they should unlock, not by implementation preference. The roadmap can change as
research needs, user feedback, and maintenance constraints evolve.

Guiding Principles
------------------

The project should remain useful for reproducible behavioral sensing research.
New work should therefore prioritize:

* stable, documented APIs over one-off scripts;
* interpretable methods before opaque models, unless the use case clearly
  benefits from additional complexity;
* explicit validation for data assumptions, missingness, and model calibration;
* tests and examples that make research workflows reproducible;
* lightweight deployment paths for local, clinical, and edge environments.

Current Baseline
----------------

The current ``0.1.x`` series provides the foundation for data loading,
preprocessing, simulation, model fitting, change-point detection, analysis,
visualization, and report generation.

Implemented or substantially available:

* CSV, JSON, HDF5, and streaming-oriented data loading.
* Gap-aware missing-data handling with masks and summaries.
* Synthetic behavioral sensor data generation with reproducible simulation.
* Bernoulli autoregressive models, including multivariate variants.
* Multiple HMM variants for activity-state modeling.
* NHPP-PELT segmentation with B-spline intensities and diagnostics.
* Several change-point detection implementations.
* Analysis pipeline reports in LaTeX, HTML, and minimal FHIR-style JSON.
* Flask-based visualization app with authenticated upload workflow.
* Zenodo, citation, and release metadata.

Near-Term Roadmap
-----------------

0.2.0: API Stabilization and Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal:
  Make the toolkit easier to adopt by researchers who are not already familiar
  with the internals.

Planned work:

* Define public API boundaries for ``data``, ``utils``, ``analysis``,
  ``models``, ``change_point``, ``hmm``, and ``visualization`` modules.
* Add docstring coverage for public classes and functions.
* Expand quickstart examples into complete workflows: load data, validate data,
  fit model, evaluate model, export report.
* Document supported input data schemas and timestamp handling.
* Add migration notes for any renamed or clarified APIs.
* Add a release checklist covering ``develop`` to ``main`` merge, tag creation,
  GitHub release, Zenodo metadata, and documentation build.

Quality gates:

* Full test suite passes.
* Pre-commit passes.
* Sphinx documentation builds without warnings for changed pages.
* Public API changes are documented in the changelog.

0.3.0: Data Quality and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal:
  Make data problems visible before they affect model results.

Planned work:

* Add a formal data validation report object for temporal consistency, missing
  values, duplicate timestamps, irregular sampling, and sensor range checks.
* Improve sensor failure detection with configurable stuck-sensor, dropout, and
  drift heuristics.
* Add dataset summary exports for reproducibility appendices.
* Add tests for malformed CSV, JSON, HDF5, and streaming records.
* Add example notebooks for data cleaning and quality assessment.

Quality gates:

* Data validation outputs are deterministic and serializable.
* Failure modes raise actionable exceptions or warnings.
* Validation utilities have focused unit tests and integration examples.

0.4.0: Model Evaluation and Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal:
  Provide consistent evaluation across modeling approaches.

Planned work:

* Standardize model scoring interfaces for Bernoulli AR, HMM, NHPP, and CPD
  components.
* Expand time-series cross-validation utilities.
* Add calibration metrics and uncertainty diagnostics for probability models.
* Add model comparison reports with structured machine-readable output.
* Add benchmark datasets or synthetic benchmark recipes with fixed seeds.

Quality gates:

* Comparison utilities work with at least two model families.
* Metrics are documented with expected input and output shapes.
* Benchmarks are reproducible from the command line.

0.5.0: Clinical and Interoperability Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal:
  Improve reporting and integration paths for clinical and applied research
  prototypes.

Planned work:

* Expand the current minimal FHIR-style export toward documented Observation
  and Bundle structures.
* Add configurable clinical threshold profiles.
* Improve patient-friendly summaries and trend visualizations.
* Add anonymization and redaction helpers for exported reports.
* Document limitations clearly: research prototype, not a medical device.

Quality gates:

* Clinical exports have schema-oriented tests.
* Report examples avoid exposing sensitive identifiers.
* Clinical documentation distinguishes supported behavior from future work.

Longer-Term Roadmap
-------------------

These items are valuable but should follow the stabilization work above.

Deep Learning Change-Point Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Add transformer, CNN, or autoencoder-based CPD only after baseline evaluation
  utilities are stable.
* Provide simple training examples and clear dataset requirements.
* Compare deep approaches against existing interpretable baselines.

Online and Real-Time Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Add incremental preprocessing and validation utilities.
* Add online change-point detection interfaces.
* Support streaming report updates and bounded-memory operation.

Packaging and Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Publish stable package artifacts when the public API is ready.
* Add compatibility checks for supported Python versions.
* Document optional dependency groups by workflow.

Performance and Scalability
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Profile slow model-fitting and analysis paths.
* Add benchmark tracking for core algorithms.
* Evaluate vectorization and optional acceleration where it reduces real
  runtime without making the code harder to maintain.

Maintenance Backlog
-------------------

The following work can be handled continuously across releases:

* Replace remaining legacy typing aliases with modern annotations.
* Reduce broad exception handling where errors can be handled specifically.
* Improve coverage for analysis and visualization modules.
* Keep examples synchronized with public APIs.
* Expand changelog entries for every release after ``0.1.0``.
* Keep Zenodo, citation, and package metadata aligned before each release.

Release Policy
--------------

Development work happens on ``develop``. Releases are cut from ``main`` only.
The release flow is:

1. Finish and validate work on ``develop``.
2. Merge ``develop`` into ``main``.
3. Tag the release on the ``main`` commit.
4. Publish the GitHub release from that tag.
5. Confirm Zenodo metadata and DOI linkage.
6. Update the changelog and documentation as needed.

Out of Scope for Now
--------------------

The following are not immediate priorities:

* medical-device claims or regulated clinical decision support;
* large-scale cloud platform features;
* opaque deep-learning models without benchmark comparisons;
* breaking public APIs without a documented migration path.
