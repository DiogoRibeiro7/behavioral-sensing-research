from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np

Array1D = np.ndarray
Array2D = np.ndarray


def difference_matrix(p: int, order: int = 2) -> Array2D:
    """
    Finite-difference operator D^(order) for P-splines.

    For p parameters and order=2, returns a (p-2)×p matrix whose rows each
    realize [1, -2, 1] at consecutive positions. More generally:
        D^(k) = diff(I_p, n=k, axis=0).

    Returns
    -------
    D : np.ndarray, shape (p - order, p)
    """
    if p <= order:
        raise ValueError("p must be > order.")
    D = np.eye(p, dtype=float)
    for _ in range(order):
        D = np.diff(D, n=1, axis=0)
    return D


def p_spline_RtR(p: int, order: int = 2, gamma: float = 0.0) -> Array2D:
    """
    Build γ * (D^T D), where D is the 'order'-th difference operator.

    Parameters
    ----------
    p : int
        Number of spline coefficients (P).
    order : int
        Difference order (2 for the classic curvature penalty).
    gamma : float
        Penalty strength γ ≥ 0.

    Returns
    -------
    RtR : np.ndarray, shape (p, p)
        γ * (Dᵀ D). If gamma==0 returns a zero matrix for convenience.
    """
    if gamma <= 0.0:
        return np.zeros((p, p), dtype=float)
    D = difference_matrix(p, order=order)
    return float(gamma) * (D.T @ D)


def sweep_gamma(
    days: Sequence[np.ndarray],
    base_cfg: NHPPConfig,
    gammas: Iterable[float],
    *,
    order: int = 2,
) -> List[Tuple[float, int, float]]:
    """
    Fit across a grid of γ values (P-spline penalty strength) and return:
        (gamma, n_changepoints, total_penalized_cost)

    Notes
    -----
    - Uses the model's existing penalty_beta (SIC if None).
    - The reported total cost matches the objective: sum segment costs (incl. γ term)
      + β per segment.
    """
    # local imports to avoid circular deps
    from .bspline import bspline_design_matrix
    from .model import NHPPPELT, NHPPConfig
    from .optimizer import SegmentOptimizer
    from .quad import QuadratureConfig

    out: List[Tuple[float, int, float]] = []

    for g in gammas:
        cfg = NHPPConfig(
            **{
                **base_cfg.__dict__,
                "pspline_gamma": float(g),
                "pspline_order": int(order),
            }
        )
        model = NHPPPELT(cfg).fit(days)

        # re-accumulate the exact objective with the fitted weights
        total = 0.0
        quad = QuadratureConfig(n_points=cfg.quad.n_points, ridge=cfg.hessian_ridge)
        opt = SegmentOptimizer(
            delta=model.delta_,
            degree=model.degree_,
            knots=model.knots_,
            quad=quad,
            newton=cfg.newton,
            pspline_gamma=cfg.pspline_gamma,
            pspline_order=cfg.pspline_order,
        )

        for (i, j), w in zip(model.segments_, model.weights_):
            # sufficient stats s for [i..j]
            s = np.zeros(cfg.n_basis, dtype=float)
            for d in range(i - 1, j):
                ev = np.asarray(days[d], float)
                if ev.size:
                    s += bspline_design_matrix(ev, cfg.degree, model.knots_).sum(axis=0)
            # minimize with warm-start = fitted w to recover cost term
            w_star, c = opt.minimize(L=j - i + 1, s=s, w0=w)
            total += float(c)

        total += len(model.segments_) * float(model.beta_)
        out.append((float(g), len(model.changepoints_), total))

    return out
