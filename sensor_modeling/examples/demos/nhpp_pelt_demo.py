from __future__ import annotations

from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt

from sensor_modeling.models.nhpp_pelt import NHPPConfig, NHPPPELT
from sensor_modeling.models.nhpp_pelt.plotting import PlotConfig, HistConfig, plot_segments_and_intensities_with_histograms


Array1D = np.ndarray


def simulate_ihpp(rng: np.random.Generator, lam: Callable[[Array1D], Array1D], delta: float, lam_max: float) -> Array1D:
    """Ogata thinning with a supplied upper bound lam_max >= sup_t λ(t)."""
    t = 0.0
    events: List[float] = []
    while True:
        t += rng.exponential(1.0 / lam_max)
        if t > delta:
            break
        if rng.uniform() < float(lam(np.array([t])) / lam_max):
            events.append(t)
    return np.array(events, dtype=float)


def main() -> None:
    rng = np.random.default_rng(7)
    delta = 24.0

    def shape_peak(mu_hour: float):
        def lam(tt: Array1D) -> Array1D:
            g1 = np.exp(-0.5 * ((tt - mu_hour) / 2.0) ** 2) / (2.0 * np.sqrt(2 * np.pi))
            g2 = np.exp(-0.5 * ((tt - (mu_hour + 8.0)) / np.sqrt(8.0)) ** 2) / (np.sqrt(8.0) * np.sqrt(2 * np.pi))
            return 20.0 * (g1 + g2)
        return lam

    lam1 = shape_peak(6.0)
    lam2 = shape_peak(9.0)

    n1, n2 = 20, 20
    lam_max = 5.0
    days: List[Array1D] = [simulate_ihpp(rng, lam1, delta, lam_max) for _ in range(n1)]
    days += [simulate_ihpp(rng, lam2, delta, lam_max) for _ in range(n2)]

    cfg = NHPPConfig(
        delta=delta,
        degree=3,
        n_basis=5,
        knot_strategy="quantile",
        min_seg_len=2,  # avoid 1-day blips
    )
    model = NHPPPELT(cfg).fit(days)
    print("Changepoints:", model.changepoints_)
    print("Segments:", model.segments_)

    fig = plot_segments_and_intensities_with_histograms(
        days,
        model,
        config=PlotConfig(grid_points=500),
        hist=HistConfig(bins="fd", density=True, alpha=0.4),
        save_path=None,
    )
    plt.show()


if __name__ == "__main__":
    main()
