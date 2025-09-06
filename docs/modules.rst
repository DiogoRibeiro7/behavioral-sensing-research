Module Guides
=============

NHPP-PELT Utilities
-------------------
The :mod:`sensor_modeling.models.nhpp_pelt.utils` module provides helpers
for exporting fitted models and sweeping penalty values. See
:func:`sensor_modeling.models.nhpp_pelt.utils.sweep_beta` for automated
elbow plots and :func:`sensor_modeling.models.nhpp_pelt.utils.save_results_json`
for JSON output.

Deep Learning Change-Point Detection
------------------------------------
The :mod:`sensor_modeling.models.change_point_detection.deep` module offers a
compact neural network that learns to flag changes from sliding windows.
It relies on scikit-learn's ``MLPClassifier`` for a minimal dependency
footprint.

Missing-Data Handling
---------------------
Utility functions in :mod:`sensor_modeling.utils.missing` support forward-fill
and linear interpolation strategies to cope with sensor dropouts.
