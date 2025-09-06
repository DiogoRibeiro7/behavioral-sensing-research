from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from .utils import type_check

Array1D = np.ndarray


@dataclass(frozen=True)
class PlotConfig:
    """Styling for the raster (top) and λ̂ curves (middle)."""

    figsize: Tuple[float, float] = (12.0, 8.5)
    dpi: int = 110
    raster_markersize: float = 4.0
    raster_alpha: float = 0.9
    band_alpha: float = 0.08
    grid_points: int = 400
    title_top: str = "Daily event raster with segment bands"
    title_bottom: str = "Estimated segment intensities λ̂(t)"


@dataclass(frozen=True)
class HistConfig:
    """Config for per-segment average histograms (bottom row)."""

    bins: Union[int, str] = "fd"  # 'fd' | 'sqrt' | 'sturges' | int
    density: bool = True  # True → average rate (events/day/unit-time)
    alpha: float = 0.5
    title: str = "Per-segment average histograms of event times"


def _common_bin_edges(
    all_events: Array1D, delta: float, bins: Union[int, str]
) -> Array1D:
    type_check(delta > 0, "delta must be positive.")
    all_events = np.asarray(all_events, float)
    all_events = all_events[(all_events >= 0.0) & (all_events <= delta)]

    if isinstance(bins, int):
        type_check(bins >= 1, "bins must be >= 1.")
        return np.linspace(0.0, delta, bins + 1)

    if all_events.size == 0:
        return np.linspace(0.0, delta, 21)

    if bins == "fd":
        q25, q75 = np.quantile(all_events, [0.25, 0.75])
        iqr = max(q75 - q25, 1e-12)
        h = 2.0 * iqr / np.cbrt(all_events.size)
        k = 20 if (not np.isfinite(h) or h <= 0) else max(1, int(np.ceil(delta / h)))
        return np.linspace(0.0, delta, k + 1)
    if bins == "sqrt":
        k = max(1, int(np.ceil(np.sqrt(all_events.size))))
        return np.linspace(0.0, delta, k + 1)
    if bins == "sturges":
        k = max(1, int(np.ceil(np.log2(all_events.size) + 1.0)))
        return np.linspace(0.0, delta, k + 1)
    raise ValueError("bins must be int or one of {'fd','sqrt','sturges'}.")


def _avg_hist_per_segment(
    days: Sequence[Array1D],
    segments: Sequence[Tuple[int, int]],
    edges: Array1D,
    density: bool,
) -> List[Array1D]:
    """
    Average per-day histogram for each segment, using common bin edges.
    If `density` is True, convert counts/bin/day to rate by dividing by bin width.
    """
    edges = np.asarray(edges, float)
    type_check(np.all(np.diff(edges) > 0), "edges must be strictly increasing.")

    widths = np.diff(edges)
    avg_counts: List[Array1D] = []

    for i_start, i_end in segments:
        type_check(1 <= i_start <= i_end <= len(days), "segment indices out of range.")
        L = i_end - i_start + 1

        acc = np.zeros(edges.size - 1, dtype=float)
        for d in range(i_start - 1, i_end):
            ev = np.asarray(days[d], float)
            if ev.size == 0:
                continue
            ev = ev[(ev >= edges[0]) & (ev <= edges[-1])]
            counts, _ = np.histogram(ev, bins=edges)
            acc += counts.astype(float)

        mean_per_day = acc / max(L, 1)
        if density:
            mean_per_day = mean_per_day / widths
        avg_counts.append(mean_per_day)

    return avg_counts


def plot_segments_and_intensities_with_histograms(
    days: Sequence[Array1D],
    model: Any,
    config: PlotConfig = PlotConfig(),
    hist: HistConfig = HistConfig(),
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Three-row figure:
      (1) event raster with segment bands,
      (2) segment λ̂(t) curves (optionally add CI bands outside),
      (3) per-segment average histograms (small multiples).
    """
    type_check(len(days) >= 1, "days must contain at least one array.")
    need = ["segments_", "knots_", "degree_", "delta_", "P_", "weights_"]
    missing = [a for a in need if not hasattr(model, a)]
    type_check(len(missing) == 0, f"Model is missing attributes: {missing}.")
    type_check(
        hasattr(model, "intensity_on_grid"), "Model must implement intensity_on_grid()."
    )

    n_days = len(days)
    segments: List[Tuple[int, int]] = list(getattr(model, "segments_"))
    delta: float = float(getattr(model, "delta_"))
    n_segments = len(segments)
    type_check(n_segments >= 1, "Model has no segments.")

    all_events = (
        np.concatenate([np.asarray(d, float) for d in days])
        if n_days
        else np.array([], float)
    )
    edges = _common_bin_edges(all_events, delta, hist.bins)
    widths = np.diff(edges)
    centers = edges[:-1] + 0.5 * widths
    avg_counts = _avg_hist_per_segment(days, segments, edges, density=hist.density)

    fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.4, 1.0, 1.0], hspace=0.35)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[1, 0])

    # -------- Top: raster with segment bands --------
    for k, (i_start, i_end) in enumerate(segments):
        y0, y1 = i_start - 0.5, i_end + 0.5
        if k % 2 == 0:
            ax_top.axhspan(y0, y1, alpha=config.band_alpha)
        ax_top.text(
            delta * 1.002,
            (y0 + y1) / 2.0,
            f"Seg {k+1}\n({i_start}–{i_end})",
            va="center",
            ha="left",
            fontsize=9,
        )

    for i, ev in enumerate(days, start=1):
        ev = np.asarray(ev, float)
        if ev.size:
            ax_top.plot(
                ev,
                np.full_like(ev, i, float),
                linestyle="None",
                marker="|",
                markersize=config.raster_markersize,
                alpha=config.raster_alpha,
            )

    ax_top.set_xlim(0.0, delta)
    ax_top.set_ylim(0.5, n_days + 0.5)
    ax_top.set_xlabel("Time within day")
    ax_top.set_ylabel("Day index")
    ax_top.set_title(config.title_top)
    ax_top.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # -------- Middle: intensity curves (no explicit colors) --------
    grid = np.linspace(0.0, delta, config.grid_points)
    for k, (i_start, i_end) in enumerate(segments):
        lam = np.asarray(model.intensity_on_grid(k, grid), float)
        type_check(lam.shape == grid.shape, "intensity_on_grid must match grid shape.")
        label = f"Seg {k+1} (days {i_start}–{i_end})"
        ax_mid.plot(grid, lam, label=label)

    ax_mid.set_xlim(0.0, delta)
    ax_mid.set_xlabel("Time within day")
    ax_mid.set_ylabel("Intensity λ(t)")
    ax_mid.set_title(config.title_bottom)
    ax_mid.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax_mid.legend(loc="best", fontsize=9, frameon=False)

    # -------- Bottom: per-segment average histograms --------
    subgs = gs[2, 0].subgridspec(1, n_segments, wspace=0.15)
    for k in range(n_segments):
        ax = fig.add_subplot(subgs[0, k])
        ax.bar(centers, avg_counts[k], width=widths, align="center", alpha=hist.alpha)
        ax.set_xlim(0.0, delta)
        if k == 0:
            ax.set_ylabel(
                "Avg rate\n(events/day/unit-time)"
                if hist.density
                else "Avg count/day/bin"
            )
        ax.set_xlabel("Time within day")
        ax.set_title(f"Seg {k+1}: days {segments[k][0]}–{segments[k][1]}", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.text(0.5, 0.04 + 0.03, hist.title, ha="center", va="center", fontsize=11)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=config.dpi, bbox_inches="tight")

    return fig
