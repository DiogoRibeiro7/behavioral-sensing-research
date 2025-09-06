import numpy as np
from sensor_modeling.hmm import (
    HierarchicalHMM,
    ScaledDirichletHMM,
    HeterogeneousHMM,
    AdaptiveHMM,
    CircadianHMM,
)


def _generate_data():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 2)
    X[rng.rand(*X.shape) < 0.1] = np.nan
    return X


def _check_model(model):
    X = _generate_data()
    model.fit(X)
    states = model.predict(X[:10])
    assert len(states) == 10
    aic = model.compute_aic(X)
    bic = model.compute_bic(X)
    assert np.isfinite(aic)
    assert np.isfinite(bic)
    model.partial_fit(X)


def test_hmm_variants():
    models = [
        HierarchicalHMM(n_states=3, random_state=0),
        ScaledDirichletHMM(n_states=3, random_state=0),
        HeterogeneousHMM(n_states=3, random_state=0),
        AdaptiveHMM(n_states=3, random_state=0),
        CircadianHMM(n_states=3, random_state=0),
    ]
    for m in models:
        _check_model(m)
