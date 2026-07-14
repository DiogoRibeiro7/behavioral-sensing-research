import numpy as np
import pytest

from sensor_modeling.change_point import (
    AdaptiveNormalizer,
    EmbeddingCPD,
    EnergyEfficientCPD,
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


def test_change_point_detectors_validate_configuration():
    with pytest.raises(ValueError, match="window"):
        EmbeddingCPD(window=0)

    with pytest.raises(ValueError, match="window"):
        EnergyEfficientCPD(window=True)

    with pytest.raises(ValueError, match="population"):
        GeneticOptimizationCPD(population=0)

    with pytest.raises(ValueError, match="generations"):
        GeneticOptimizationCPD(generations=0)


def test_change_point_detectors_validate_series_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        EmbeddingCPD().fit(np.zeros((2, 2)))

    with pytest.raises(ValueError, match="at least one"):
        EnergyEfficientCPD().fit([])

    with pytest.raises(ValueError, match="finite values"):
        AdaptiveNormalizer().fit([0.0, np.nan, 1.0])

    with pytest.raises(ValueError, match="finite values"):
        GeneticOptimizationCPD().fit([0.0, np.inf, 1.0])


def test_change_point_detectors_validate_prediction_contracts():
    with pytest.raises(ValueError, match="fitted"):
        EmbeddingCPD().predict()

    with pytest.raises(ValueError, match="positive and finite"):
        EmbeddingCPD().fit(_series()).predict(threshold=0)

    with pytest.raises(ValueError, match="positive and finite"):
        EnergyEfficientCPD().fit(_series()).predict(threshold=np.inf)

    with pytest.raises(ValueError, match="positive and finite"):
        AdaptiveNormalizer().fit(_series()).predict(threshold=np.nan)
