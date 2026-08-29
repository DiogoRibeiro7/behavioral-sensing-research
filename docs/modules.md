# Module Guides

## NHPP-PELT Utilities

The `utils` module provides helpers
for exporting fitted models and sweeping penalty values. See
`sweep_beta` for automated
elbow plots and `save_results_json`
for JSON output.

## Deep Learning Change-Point Detection

The `deep` module offers a
compact neural network that learns to flag changes from sliding windows.
It relies on scikit-learn's `MLPClassifier` for a minimal dependency
footprint.

## Missing-Data Handling

The `missing` module now exposes a higher-level
`handle_missing_data()` API for common sensor-stream failure modes. It
supports forward-fill, interpolation, gap-aware imputation that preserves long
outages, row-drop policies, and flag-only workflows that append missingness
indicators for downstream models. The returned `MissingDataResult` includes
imputation masks and per-column gap summaries so uncertainty can be tracked
explicitly.

Use forward-fill for short stateful sensor dropouts, interpolation for smooth
numeric channels, gap-aware handling when long outages should stay visible, and
flag/drop policies when downstream models must reason about or reject missing
observations explicitly.
