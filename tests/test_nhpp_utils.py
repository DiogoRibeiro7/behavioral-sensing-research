"""Tests for NHPP-PELT utilities."""
import json
import numpy as np
import pytest

from sensor_modeling.models.nhpp_pelt import NHPPConfig
from sensor_modeling.models.nhpp_pelt import utils


def test_normal_ppf_symmetry():
    """Φ⁻¹(0.5) ≈ 0."""
    assert abs(utils.normal_ppf(0.5)) < 1e-6


def test_to_jsonable_and_save(tmp_path):
    """Model export utilities produce JSON output."""

    class Stub:
        delta_ = 0.1
        degree_ = 3
        P_ = 2
        beta_ = 1.0
        knots_ = np.array([0.0, 1.0])
        changepoints_ = [1]
        segments_ = [(1, 2)]
        weights_ = [np.array([1.0, 2.0])]

    stub = Stub()
    data = utils.to_jsonable(stub)
    assert data["delta"] == 0.1
    out = tmp_path / "model.json"
    utils.save_results_json(stub, out)
    loaded = json.loads(out.read_text())
    assert loaded["segments"] == [[1, 2]]


def test_sweep_beta_runs():
    """Penalty sweep returns results for each beta."""
    days = [np.array([0.1, 0.2])]
    cfg = NHPPConfig(n_basis=4, min_seg_len=1)
    results = utils.sweep_beta(days, cfg, [0.1])
    assert len(results) == 1
    beta, ncp, cost = results[0]
    assert beta == pytest.approx(0.1)
    assert isinstance(ncp, int)
    assert isinstance(cost, float)
