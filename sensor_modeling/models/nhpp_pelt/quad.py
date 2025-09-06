# nhpp/quad.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .utils import type_check

Array1D = np.ndarray


@dataclass(frozen=True)
class QuadratureConfig:
    """
    Settings for Gauss–Legendre quadrature used to integrate terms like
    ∫ exp(wᵀψ(t)) {1, ψ, ψψᵀ} dt.

    Attributes
    ----------
    n_points : int
        Order of the Gauss–Legendre rule (≥ 2). Larger → more accurate, slower.
    ridge : float
        Small diagonal ridge added to the Hessian to ensure positive definiteness.
    """
    n_points: int = 256
    ridge: float = 1e-8


def leggauss_on_interval(a: float, b: float, n: int) -> Tuple[Array1D, Array1D]:
    """
    Compute Gauss–Legendre nodes and weights on a closed interval [a, b].

    Parameters
    ----------
    a : float
        Interval start (can equal b for a degenerate interval; not useful here).
    b : float
        Interval end; must satisfy b > a in practice.
    n : int
        Number of quadrature points (order). Must be ≥ 2.

    Returns
    -------
    nodes : np.ndarray, shape (n,)
        Quadrature abscissae in [a, b].
    weights : np.ndarray, shape (n,)
        Corresponding quadrature weights that integrate polynomials of degree 2n-1 exactly.

    Notes
    -----
    We obtain nodes/weights on [-1, 1] via `numpy.polynomial.legendre.leggauss`
    and map them affinely to [a, b].
    """
    type_check(n >= 2, "Quadrature order n must be >= 2.")
    type_check(np.isfinite(a) and np.isfinite(b), "Interval bounds must be finite.")
    type_check(b >= a, "Require b >= a.")

    # Nodes/weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n)

    # Affine map to [a, b]
    xm = 0.5 * (b + a)
    xr = 0.5 * (b - a)
    nodes = xm + xr * x
    weights = xr * w
    return nodes.astype(float, copy=False), weights.astype(float, copy=False)


__all__ = ["QuadratureConfig", "leggauss_on_interval"]
