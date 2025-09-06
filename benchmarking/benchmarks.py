"""Runtime and memory benchmarks for sensor modeling."""
from __future__ import annotations

import time
import tracemalloc
from typing import Dict

from sensor_modeling.utils.data_io import simulate_sensor_data
from sensor_modeling.models.bernoulli_ar.base_model import (
    BernoulliAutoregressiveModel,
)


def run_benchmark(n_days: int = 3, n_sensors: int = 3) -> Dict[str, float]:
    """Run a simple runtime and memory benchmark."""
    dataset = simulate_sensor_data(n_days=n_days, n_sensors=n_sensors)
    model = BernoulliAutoregressiveModel(
        list(dataset.data.columns), dataset.data.columns[0]
    )
    tracemalloc.start()
    start = time.perf_counter()
    model.fit(dataset)
    runtime = time.perf_counter() - start
    peak_mem = tracemalloc.get_traced_memory()[1] / 1_000_000
    tracemalloc.stop()
    return {"runtime_sec": runtime, "peak_mem_mb": peak_mem}


if __name__ == "__main__":
    print(run_benchmark())
