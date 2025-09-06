from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .bspline import bspline_design_matrix
from .quad import QuadratureConfig, leggauss_on_interval
from .utils import type_check
from .regularization import p_spline_RtR   # <-- NEW

Array1D = np.ndarray
Array2D = np.ndarray


@dataclass(frozen=True)
class NewtonConfig:
    """Tuning for Newton backtracking solver."""
    max_iter: int = 60
    tol: float = 1e-7
    step_shrink: float = 0.5
    min_step: float = 1e-6


class SegmentOptimizer:
    """
    Optimize the segment negative log-likelihood with optional P-spline penalty:

        cost(w) = L * ∫ exp(wᵀψ(t)) dt  -  sᵀ w  +  wᵀ( RtR )w

    where s = ∑_events ψ(t_j), L = number of days in the segment, and RtR = γ·DᵀD.
    """

    def __init__(
        self,
        delta: float,
        degree: int,
        knots: Array1D,
        quad: QuadratureConfig = QuadratureConfig(),
        newton: NewtonConfig = NewtonConfig(),
        *,
        pspline_gamma: float = 0.0,     # <-- NEW (optional)
        pspline_order: int = 2          # <-- NEW (optional)
    ) -> None:
        type_check(delta > 0.0, "delta must be positive.")
        self.delta = float(delta)
        self.degree = int(degree)
        self.knots = np.asarray(knots, dtype=float)
        self.quad = quad
        self.newton = newton

        # Precompute quadrature nodes and basis at nodes
        nodes, weights = leggauss_on_interval(0.0, self.delta, quad.n_points)
        self._q_nodes = nodes
        self._q_w = weights
        self._q_psi = bspline_design_matrix(nodes, self.degree, self.knots)  # (Q,P)
        self.P = self._q_psi.shape[1]

        # Cache RtR = γ·DᵀD (zero matrix if gamma==0)
        self._RtR = p_spline_RtR(self.P, order=pspline_order, gamma=pspline_gamma)

    # ---------------------- Stable integral block ----------------------

    def _integrals(self, w: Array1D) -> Tuple[float, Array1D, Array2D]:
        psi = self._q_psi  # (Q,P)
        u = psi @ w        # (Q,)
        m = float(np.max(u))
        e_scaled = np.exp(u - m)
        wq = self._q_w

        I0_s = float(np.sum(e_scaled * wq))
        I1_s = psi.T @ (e_scaled * wq)
        I2_s = psi.T @ (psi * (e_scaled * wq)[:, None])

        scale = math.exp(m)
        return I0_s * scale, I1_s * scale, I2_s * scale

    # ------------------------- Newton optimizer -------------------------

    def minimize(self, L: int, s: Array1D, w0: Optional[Array1D] = None) -> Tuple[Array1D, float]:
        type_check(L >= 1, "Segment length L must be >= 1.")
        s = np.asarray(s, dtype=float)
        w = np.zeros(self.P, dtype=float) if w0 is None else np.asarray(w0, dtype=float).copy()
        type_check(w.shape == (self.P,), "Initial w has wrong shape.")

        ridgeI = self.quad.ridge * np.eye(self.P, dtype=float)

        for _ in range(self.newton.max_iter):
            I0, I1, I2 = self._integrals(w)
            cost = L * I0 - float(s @ w)
            grad = L * I1 - s
            H = L * I2 + ridgeI

            # ======== P-spline penalty (γ‖D²w‖²) — 3 math lines ========
            if np.any(self._RtR):                 # skip if gamma == 0
                cost += float(w @ (self._RtR @ w))
                grad += 2.0 * (self._RtR @ w)
                H += 2.0 * self._RtR
            # ===========================================================

            # Convergence
            if np.linalg.norm(grad, ord=np.inf) <= self.newton.tol * max(1.0, np.linalg.norm(s, ord=np.inf)):
                return w, cost

            # Newton step
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(H) @ grad

            # Backtracking
            t = 1.0
            while t >= self.newton.min_step:
                w_new = w - t * step
                I0_new, _, _ = self._integrals(w_new)
                cost_new = L * I0_new - float(s @ w_new)
                # penalty contribution at trial point
                if np.any(self._RtR):
                    cost_new += float(w_new @ (self._RtR @ w_new))
                if cost_new <= cost - 1e-8 * t * float(grad @ step):
                    w, cost = w_new, cost_new
                    break
                t *= self.newton.step_shrink
            else:
                w = w - self.newton.min_step * step

        I0, _, _ = self._integrals(w)
        final_cost = L * I0 - float(s @ w)
        if np.any(self._RtR):
            final_cost += float(w @ (self._RtR @ w))
        return w, final_cost
