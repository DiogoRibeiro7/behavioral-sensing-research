"""Tests for the PELT change-point detector."""

import numpy as np

from sensor_modeling.models.change_point_detection import PELTChangePointDetector


def test_pelt_returns_no_change_points_for_constant_signal():
    """Constant sequences should not be over-segmented."""
    signal = np.zeros(40)
    detector = PELTChangePointDetector(penalty=2.0, min_segment_length=5)
    assert detector.detect(signal) == []


def test_pelt_detects_single_mean_shift():
    """A single clean breakpoint should be recovered exactly."""
    signal = np.concatenate([np.zeros(20), np.ones(20) * 5.0])
    detector = PELTChangePointDetector(penalty=1.0, min_segment_length=5)
    assert detector.detect(signal) == [20]


def test_pelt_detects_multiple_mean_shifts():
    """Multiple piecewise-constant segments should be recovered."""
    signal = np.concatenate(
        [np.zeros(15), np.ones(20) * 4.0, np.ones(18) * -3.0]
    )
    detector = PELTChangePointDetector(penalty=1.0, min_segment_length=5)
    assert detector.detect(signal) == [15, 35]


def test_pelt_penalty_controls_segmentation_strength():
    """Higher penalties should suppress weak change points."""
    signal = np.concatenate([np.zeros(25), np.ones(25) * 0.5])
    low_penalty = PELTChangePointDetector(penalty=0.2, min_segment_length=5)
    high_penalty = PELTChangePointDetector(penalty=10.0, min_segment_length=5)
    assert low_penalty.detect(signal) == [25]
    assert high_penalty.detect(signal) == []


def test_pelt_supports_l1_cost_for_robust_segmentation():
    """The alternate L1 cost path should also detect canonical changes."""
    signal = np.concatenate([np.zeros(12), np.ones(12) * 3.0, np.zeros(12)])
    detector = PELTChangePointDetector(
        penalty=0.5, min_segment_length=4, cost="l1"
    )
    assert detector.detect(signal) == [12, 24]
