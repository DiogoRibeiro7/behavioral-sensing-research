# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added a top-level `ROADMAP.md` with release milestones, quality gates, longer-term priorities, maintenance backlog, and release policy.
- Added focused tests for shared data IO, synthetic exports, HDF5 loading, sensor failure detection, plotting helpers, and model validation utilities.

### Changed
- Refactored data-layer typing, HDF5 import handling, synthetic data export behavior, and validation internals.
- Refactored sensor simulation to use local NumPy random generators instead of mutating global random state.
- Refactored plotting helpers to return Matplotlib figures and support non-interactive `show=False` workflows.
- Updated README and Sphinx roadmap documentation to point to the canonical roadmap.

### Fixed
- Fixed synthetic JSON export by serializing timestamp values before writing JSON.
- Fixed model validation calibration output to return plain Python booleans.

## [0.1.3] - 2026-07-13
### Added
- Added minimal FHIR-style report export to the analysis pipeline.
- Added configurable web upload directory support for the Flask application.
- Added time-series cross-validation helpers for model comparison workflows.

### Changed
- Aligned CSV loading semantics across `SensorDataset.from_csv` and data loaders.
- Updated comparison scoring to require explicit scoring behavior when models do not expose `score`.
- Restored and aligned the lint/pre-commit baseline for the current codebase.
- Removed duplicated README sections.

### Fixed
- Fixed analysis report generation so output directories are created before writing reports.
- Fixed README examples to use Bernoulli probability prediction APIs.

## [0.1.2] - 2026-07-13
### Added
- Added Zenodo release metadata for archive publication.
- Added `.zenodo.json` metadata for Zenodo integration.

## [0.1.1] - 2026-07-13
### Added
- Added repository-to-Zenodo linkage documentation.
- Prepared release metadata for Zenodo DOI archival.

## [0.1.0] - 2025-08-28
### Added
- Initial release with unified sensor modeling framework, analysis utilities, and visualization tools.
