"""Benchmark different change point detection approaches."""

from __future__ import annotations

import time

import numpy as np

from sensor_modeling.change_point import (
    AdaptiveNormalizer,
    EmbeddingCPD,
    EnergyEfficientCPD,
    GeneticOptimizationCPD,
)
from sensor_modeling.utils import plot_benchmark_results


def _generate_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.concatenate([np.zeros(50), np.ones(50)]) + rng.normal(0, 0.1, 100)


def main() -> None:
    series = _generate_data()
    algorithms = {
        "embedding": EmbeddingCPD(),
        "energy": EnergyEfficientCPD(),
        "adaptive": AdaptiveNormalizer(),
        "genetic": GeneticOptimizationCPD(),
    }
    results = {}
    for name, alg in algorithms.items():
        start = time.perf_counter()
        alg.fit(series)
        cps = alg.predict()
        results[name] = time.perf_counter() - start
        print(f"{name}: {len(cps)} change points")
    plot_benchmark_results(results)


if __name__ == "__main__":
    main()
