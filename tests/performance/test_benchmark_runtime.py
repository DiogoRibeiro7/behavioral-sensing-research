"""Performance benchmarks for core models."""

import time
import tracemalloc

from sensor_modeling.models.bernoulli_ar.base_model import (
    BernoulliAutoregressiveModel,
)
from sensor_modeling.utils.data_io import simulate_sensor_data


def test_runtime_and_memory() -> None:
    """Model fits within time and memory constraints."""
    dataset = simulate_sensor_data(n_days=1, n_sensors=2)
    model = BernoulliAutoregressiveModel(
        list(dataset.data.columns), dataset.data.columns[0]
    )
    tracemalloc.start()
    start = time.perf_counter()
    model.fit(dataset)
    runtime = time.perf_counter() - start
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert runtime < 20
    assert peak < 50_000_000
