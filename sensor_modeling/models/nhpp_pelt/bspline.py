from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .utils import type_check

Array1D = np.ndarray
Array2D = np.ndarray


def open_uniform_knots(
    delta: float, degree: int, n_basis: int, internal_knots: Optional[Array1D]
) -> Array1D:
    """
    Construct an open (clamped) knot vector on [0, delta].

    If internal_knots is provided, then:
        n_basis == len(internal_knots) + degree + 1
    and we place degree+1 repeats at boundaries.

    Returns
    -------
    knots : np.ndarray, shape (n_basis + degree + 1,)
    """
    type_check(delta > 0, "delta must be positive.")
    type_check(degree >= 0, "degree must be non-negative.")

    if internal_knots is not None:
        internal_knots = np.asarray(internal_knots, dtype=float)
        type_check(
            np.all((internal_knots > 0) & (internal_knots < delta)),
            "internal_knots must be in (0, delta).",
        )
        expected = internal_knots.size + degree + 1
        type_check(
            n_basis == expected,
            f"n_basis must equal len(internal_knots)+degree+1 = {expected}.",
        )
        breaks = np.r_[0.0, internal_knots, delta]
    else:
        n_internal = n_basis - degree - 1
        type_check(n_internal >= 0, "n_basis must be at least degree+1.")
        if n_internal > 0:
            internal = np.linspace(0.0, delta, n_internal + 2)[1:-1]
            breaks = np.r_[0.0, internal, delta]
        else:
            breaks = np.array([0.0, delta], dtype=float)

    t0 = np.repeat(breaks[0], degree + 1)
    t1 = np.repeat(breaks[-1], degree + 1)
    return np.r_[t0, breaks[1:-1], t1]


def _basis_at_scalar(t: float, degree: int, knots: Array1D) -> Array1D:
    """Cox–de Boor recursion for all basis functions at a scalar t."""
    M = knots.size
    n_basis = M - degree - 1
    N = np.zeros((degree + 1, n_basis), dtype=float)

    # degree 0
    for i in range(n_basis):
        if knots[i] <= t < knots[i + 1] or (
            t == knots[-1] and knots[i] <= t <= knots[i + 1]
        ):
            N[0, i] = 1.0

    # recursion
    for k in range(1, degree + 1):
        for i in range(n_basis):
            left_den = knots[i + k] - knots[i]
            right_den = knots[i + k + 1] - knots[i + 1]
            left = 0.0 if left_den <= 0 else (t - knots[i]) / left_den * N[k - 1, i]
            right = (
                0.0
                if right_den <= 0
                else (knots[i + k + 1] - t) / right_den * N[k - 1, i + 1]
                if i + 1 < n_basis
                else 0.0
            )
            N[k, i] = left + right

    return N[degree, :]


def bspline_design_matrix(ts: Array1D, degree: int, knots: Array1D) -> Array2D:
    """
    Evaluate B-spline basis at vector ts (shape (m,)),
    returning matrix Psi with shape (m, n_basis).
    """
    ts = np.asarray(ts, dtype=float)
    out = np.empty((ts.size, knots.size - degree - 1), dtype=float)
    for r, t in enumerate(ts):
        out[r, :] = _basis_at_scalar(float(t), degree, knots)
    return out
