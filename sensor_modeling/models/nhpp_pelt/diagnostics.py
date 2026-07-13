from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

Array1D = np.ndarray


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass(frozen=True)
class DiagConfig:
    """
    Configuration for time-rescaling diagnostics.

    Attributes
    ----------
    grid_points : int
        Uniform grid resolution for λ(t) and Λ(t) approximation.
    figsize : tuple[float, float]
        Matplotlib figure size.
    dpi : int
        Figure DPI.
    hist_bins : int
        Number of bins for U-histogram (Uniform(0,1) check).
    alpha : float
        Transparency for histograms and fills.
    """

    grid_points: int = 2000
    figsize: Tuple[float, float] = (12.0, 7.0)
    dpi: int = 110
    hist_bins: int = 20
    alpha: float = 0.5


def _lambda_grid(
    model: Any, seg_index: int, grid_points: int
) -> Tuple[Array1D, Array1D]:
    """Compute λ̂_k(t) on a uniform grid for segment k."""
    delta: float = float(model.delta_)
    grid = np.linspace(0.0, delta, grid_points)
    lam = np.asarray(model.intensity_on_grid(seg_index, grid), float)
    _check(lam.shape == grid.shape, "intensity_on_grid must match grid shape.")
    return grid, lam


def _cumulative_int(grid: Array1D, lam: Array1D) -> Array1D:
    """
    Cumulative integral Λ(t) via the trapezoidal rule on a uniform grid.
    Returns array of same length as grid with Λ(0)=0.
    """
    # Guard
    _check(
        grid.ndim == 1 and lam.ndim == 1 and grid.size == lam.size,
        "grid/lam shape mismatch.",
    )
    d = np.diff(grid)
    # trapezoid increments: ∫ lam ~ 0.5*(lam_i + lam_{i+1})*Δ
    incr = 0.5 * (lam[:-1] + lam[1:]) * d
    return np.r_[0.0, np.cumsum(incr)]


def _interp_cumint(t: Array1D, grid: Array1D, Lambda: Array1D) -> Array1D:
    """
    Interpolate Λ(t) for arbitrary event times using linear interpolation
    over the precomputed (grid, Λ(grid)).
    """
    t = np.asarray(t, float)
    t = np.clip(t, grid[0], grid[-1])
    return np.interp(t, grid, Lambda)


def time_rescale_events_for_segment(
    model: Any,
    seg_index: int,
    events_per_day: Sequence[Array1D],
    grid_points: int = 2000,
) -> Tuple[List[Array1D], List[Array1D]]:
    """
    Apply the time-rescaling theorem *within a fitted segment*:
      ΔΛ_j  should be i.i.d. Exp(1);
      U_j = 1 - exp(-ΔΛ_j) should be i.i.d. Uniform(0,1).

    Parameters
    ----------
    model : fitted NHPPPELT-like object
    seg_index : int
        Segment index in 0..K-1.
    events_per_day : list of 1-D arrays
        Only the days that belong to this segment.
    grid_points : int
        Resolution for Λ approximation.

    Returns
    -------
    delta_L_list : list[np.ndarray]
        For each day, the Exp(1)-should-be inter-arrival ΔΛ sequence.
    U_list : list[np.ndarray]
        Uniform(0,1)-should-be values 1 - exp(-ΔΛ).
    """
    grid, lam = _lambda_grid(model, seg_index, grid_points)
    Lambda = _cumulative_int(grid, lam)

    delta_L_list: List[Array1D] = []
    U_list: List[Array1D] = []

    for ev in events_per_day:
        ev = np.asarray(ev, float)
        if ev.size == 0:
            delta_L_list.append(np.array([], float))
            U_list.append(np.array([], float))
            continue
        # Rescaled event times Λ(t_j)
        L_times = _interp_cumint(ev, grid, Lambda)
        # Inter-arrival increments; include leading gap from 0 to first event
        L_with0 = np.r_[0.0, L_times]
        dL = np.diff(L_with0)
        U = 1.0 - np.exp(-dL)
        delta_L_list.append(dL)
        U_list.append(U)

    return delta_L_list, U_list


def ks_stat_uniform(u: Array1D) -> float:
    """
    Kolmogorov-Smirnov statistic for Uniform(0,1).
    (Asymptotic p-value can be added if needed; we return only D.)
    """
    u = np.sort(np.asarray(u, float))
    n = u.size
    if n == 0:
        return np.nan
    # D+ = max_i i/n - u_i ; D- = max_i u_i - (i-1)/n
    i = np.arange(1, n + 1, dtype=float)
    d_plus = np.max(i / n - u)
    d_minus = np.max(u - (i - 1.0) / n)
    return float(max(d_plus, d_minus))


def plot_time_rescaling_diagnostics(
    days: Sequence[Array1D],
    model: Any,
    config: DiagConfig = DiagConfig(),
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Build a 2-row diagnostics figure per segment:
      Row 1 (left-to-right over segments): histogram of U with Uniform(0,1) reference line.
      Row 2: exponential QQ plot for ΔΛ vs Exp(1) quantiles.

    Notes
    -----
    - No external stats deps; uniform line and exponential quantiles are computed analytically.
    - Uses Matplotlib defaults for colors.
    """
    _check(
        hasattr(model, "segments_") and hasattr(model, "delta_"),
        "Model missing required attributes.",
    )
    segments: List[Tuple[int, int]] = list(model.segments_)
    n_segments = len(segments)
    _check(n_segments >= 1, "No segments found.")

    # Group events per segment
    seg_days: List[List[Array1D]] = []
    for i_start, i_end in segments:
        seg_days.append([np.asarray(d, float) for d in days[i_start - 1 : i_end]])

    # Precompute transformed values
    all_U: List[Array1D] = []
    all_dL: List[Array1D] = []
    for k, evs in enumerate(seg_days):
        dL_list, U_list = time_rescale_events_for_segment(
            model, k, evs, grid_points=config.grid_points
        )
        all_U.append(np.concatenate(U_list) if U_list else np.array([], float))
        all_dL.append(np.concatenate(dL_list) if dL_list else np.array([], float))

    # Figure layout: 2 rows × K columns
    fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
    gs = fig.add_gridspec(nrows=2, ncols=n_segments, hspace=0.35, wspace=0.2)

    # Row 1: U histograms
    for k in range(n_segments):
        ax = fig.add_subplot(gs[0, k])
        u = all_U[k]
        # Histogram (counts normalized to density); no explicit color
        ax.hist(
            u, bins=config.hist_bins, range=(0.0, 1.0), density=True, alpha=config.alpha
        )
        # Uniform(0,1) reference density = 1 on [0,1]
        ax.plot([0.0, 1.0], [1.0, 1.0])  # default color/style
        D = ks_stat_uniform(u)
        ax.set_title(f"Seg {k+1} — U histogram (KS D={D:.3f})", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("U = 1 - exp(-ΔΛ)")
        if k == 0:
            ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # Row 2: Exponential QQ (ΔΛ vs Exp(1))
    for k in range(n_segments):
        ax = fig.add_subplot(gs[1, k])
        dL = np.sort(all_dL[k])
        n = dL.size
        if n >= 2:
            # Theoretical Exp(1) quantiles for p_i = (i-0.5)/n
            p = (np.arange(1, n + 1) - 0.5) / n
            q = -np.log(1.0 - p)
            ax.plot(q, dL, linestyle="None", marker="o", markersize=3, alpha=0.8)
            # y=x reference
            lim = float(max(q[-1], dL[-1])) if n > 0 else 1.0
            ax.plot([0.0, lim], [0.0, lim])
        ax.set_title(f"Seg {k+1} — QQ: ΔΛ vs Exp(1)", fontsize=10)
        ax.set_xlabel("Theoretical Exp(1) quantiles")
        if k == 0:
            ax.set_ylabel("Observed ΔΛ")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle("Time-rescaling diagnostics per segment", y=0.99, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
    return fig
