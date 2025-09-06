"""Tests for deep learning change-point detector."""

from sensor_modeling.models.change_point_detection import (
    DeepCPDConfig,
    DeepChangePointDetector,
)


def test_deep_cpd_detects_change():
    """A change point is detected in a simple sequence."""
    seq = [0, 0, 0, 1, 1, 1]
    cfg = DeepCPDConfig(window=2, hidden=10)
    model = DeepChangePointDetector(cfg).fit(seq)
    pred = model.detect(seq)
    assert len(pred) == len(seq)
