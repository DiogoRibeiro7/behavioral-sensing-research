import numpy as np

from sensor_modeling.change_point import (
    EmbeddingCPD,
    EnergyEfficientCPD,
    AdaptiveNormalizer,
    GeneticOptimizationCPD,
)


def _series():
    rng = np.random.default_rng(0)
    return np.concatenate([np.zeros(50), np.ones(50)]) + rng.normal(0, 0.1, 100)


def _assert_near_middle(cps):
    assert any(abs(cp - 50) <= 5 for cp in cps)


def test_embedding_cpd():
    series = _series()
    model = EmbeddingCPD().fit(series)
    cps = model.predict()
    _assert_near_middle(cps)


def test_energy_efficient_cpd():
    series = _series()
    model = EnergyEfficientCPD().fit(series)
    cps = model.predict()
    _assert_near_middle(cps)


def test_adaptive_normalizer():
    series = _series()
    model = AdaptiveNormalizer().fit(series)
    cps = model.predict()
    _assert_near_middle(cps)


def test_genetic_optimization():
    series = _series()
    model = GeneticOptimizationCPD().fit(series)
    cps = model.predict()
    _assert_near_middle(cps)
