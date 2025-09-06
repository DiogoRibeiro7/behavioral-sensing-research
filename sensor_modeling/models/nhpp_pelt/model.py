from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
from ...utils.data_io import SensorDataset

from .bspline import bspline_design_matrix, open_uniform_knots
from .optimizer import SegmentOptimizer, NewtonConfig
from .quad import QuadratureConfig
from .utils import type_check, normal_ppf

Array1D = np.ndarray
Array2D = np.ndarray


@dataclass
class NHPPConfig:
    """
    Configuration for NHPP segmentation with B-spline intensities.

    Attributes
    ----------
    delta : float
        Length of the time window for each day (e.g., 24.0 for hours).
    degree : int
        B-spline degree (3 = cubic).
    n_basis : int
        Number of basis functions P.
    knot_strategy : {'quantile', 'equispaced'}
        How to place internal knots.
    quad : QuadratureConfig
        Quadrature setup.
    newton : NewtonConfig
        Newton solver setup.
    penalty_beta : float | None
        If None, use SIC: (P+1) * log(n_days).
    hessian_ridge : float
        Ridge added to Hessian (and for covariance inversion).
    min_seg_len : int
        Minimum number of days per segment (>=1). Use 1 to disable.
    """
    delta: float = 24.0
    degree: int = 3
    n_basis: int = 5
    knot_strategy: str = "quantile"
    quad: QuadratureConfig = QuadratureConfig()
    newton: NewtonConfig = NewtonConfig()
    penalty_beta: Optional[float] = None
    hessian_ridge: float = 1e-8
    min_seg_len: int = 2
    pspline_gamma: float = 0.0
    pspline_order: int = 2


class NHPPPELT:
    """
    Changepoint detection over a sequence of daily inhomogeneous Poisson processes (IHPPs)
    where each segment k shares a B-spline log-intensity:

        λ_k(t) = exp( w_kᵀ ψ(t) ),  ψ: B-spline basis

    Segment cost = minimized negative log-likelihood:
        C(i..j) = min_w [ L*∫exp(wᵀψ)dt - sᵀw ]
      with L = j-i+1 and s = Σ_{events in i..j} ψ(t).

    Penalized objective (PELT):
        min over partitions  Σ_k [ C(segment_k) + β ].
    """

    # ----------------------------- Basics -----------------------------

    def __init__(self, config: NHPPConfig):
        self.cfg = config
        self._fitted = False

    # -------------------------- Knot builder --------------------------

    def _build_knots(self, all_event_times: Array1D) -> Array1D:
        deg = self.cfg.degree
        P = self.cfg.n_basis
        type_check(P >= deg + 1, f"n_basis must be >= degree+1 (= {deg + 1}).")

        n_internal = P - deg - 1
        if n_internal <= 0:
            internal = None
        else:
            if self.cfg.knot_strategy == "quantile":
                if all_event_times.size >= n_internal + 1:
                    qs = np.linspace(0, 1, n_internal + 2)[1:-1]
                    internal = np.quantile(all_event_times, qs)
                else:
                    internal = np.linspace(0.0, self.cfg.delta, n_internal + 2)[1:-1]
            elif self.cfg.knot_strategy == "equispaced":
                internal = np.linspace(0.0, self.cfg.delta, n_internal + 2)[1:-1]
            else:
                raise ValueError("knot_strategy must be 'quantile' or 'equispaced'.")

        return open_uniform_knots(self.cfg.delta, deg, P, internal)

    # ------------------------------ Fit ------------------------------

    def fit(self, data: Sequence[Array1D] | SensorDataset, sensor: str | None = None) -> "NHPPPELT":
        """
        Fit the model. After fitting, attributes include:
          - changepoints_ : List[int] in 1..n-1
          - segments_     : List[(start,end)] 1-based inclusive
          - weights_      : List[(P,)] B-spline weights per segment
          - degree_, delta_, P_, knots_, beta_
        """
        if isinstance(data, SensorDataset):
            if sensor is None:
                raise ValueError("sensor must be provided when using SensorDataset")
            days = data.to_event_sequences(sensor)
        else:
            days = data

        type_check(len(days) >= 1, "At least one day is required.")
        n = len(days)
        delta = float(self.cfg.delta)

        all_ts = np.concatenate([np.asarray(d, dtype=float) for d in days]) if n > 0 else np.array([], float)
        if all_ts.size > 0:
            type_check(np.all((all_ts >= 0.0) & (all_ts <= delta)), "Event times must lie in [0, delta].")

        knots = self._build_knots(all_ts)
        degree = self.cfg.degree
        P = self.cfg.n_basis

        # Per-day sufficient stats s_i = Σ ψ(t)
        s_list: List[Array1D] = []
        for d in days:
            d = np.asarray(d, dtype=float)
            if d.size == 0:
                s_list.append(np.zeros(P, dtype=float))
            else:
                Psi = bspline_design_matrix(d, degree, knots)
                s_list.append(Psi.sum(axis=0))
        S = np.vstack(s_list)                     # (n,P)
        S_cum = np.vstack([np.zeros((1, P)), np.cumsum(S, axis=0)])  # (n+1,P)

        # Penalty
        beta = self.cfg.penalty_beta if self.cfg.penalty_beta is not None else (P + 1) * math.log(n)

        # Optimizer
        quad = QuadratureConfig(n_points=self.cfg.quad.n_points, ridge=self.cfg.hessian_ridge)
        opt = SegmentOptimizer(delta=delta, degree=degree, knots=knots, quad=quad, newton=self.cfg.newton)

        # Segment caches
        cost_cache: Dict[Tuple[int, int], float] = {}
        w_cache: Dict[Tuple[int, int], Array1D] = {}

        def seg_stats(i: int, j: int) -> Tuple[int, Array1D]:
            L = j - i + 1
            s = S_cum[j, :] - S_cum[i - 1, :]
            return L, s

        def seg_cost(i: int, j: int, warm: Optional[Array1D] = None) -> Tuple[float, Array1D]:
            key = (i, j)
            if key in cost_cache:
                return cost_cache[key], w_cache[key]
            L, s = seg_stats(i, j)
            w0 = warm
            if w0 is None and (i, j - 1) in w_cache:
                w0 = w_cache[(i, j - 1)]
            if w0 is None and (i + 1, j) in w_cache:
                w0 = w_cache[(i + 1, j)]
            w_star, c_star = opt.minimize(L=L, s=s, w0=w0)
            cost_cache[key] = c_star
            w_cache[key] = w_star
            return c_star, w_star

        # ------------------------------ PELT ------------------------------
        msl = max(1, int(self.cfg.min_seg_len))
        F = np.full(n + 1, np.inf, dtype=float)  # DP table
        F[0] = -beta
        last_change = np.full(n + 1, -1, dtype=int)
        R: List[int] = [0]

        for q in range(1, n + 1):
            best_val = np.inf
            best_p = -1
            for p in R:
                if (q - p) < msl:
                    continue
                c, _ = seg_cost(p + 1, q)
                val = F[p] + c + beta
                if val < best_val:
                    best_val, best_p = val, p
            F[q] = best_val
            last_change[q] = best_p

            # pruning
            new_R = []
            for p in R:
                if (q - p) < msl:
                    continue
                c, _ = seg_cost(p + 1, q)
                if F[p] + c < F[q]:
                    new_R.append(p)
            R = new_R + [q]

        # backtrack
        cps: List[int] = []
        q = n
        while q > 0:
            p = last_change[q]
            if p <= 0:
                if p == 0:
                    cps.append(0)
                break
            cps.append(p)
            q = p
        cps = sorted([c for c in cps if 0 < c < n])

        # segments and weights
        segments: List[Tuple[int, int]] = []
        weights: List[Array1D] = []
        start = 1
        for cp in cps + [n]:
            c, w = seg_cost(start, cp if cp > 0 else n)
            segments.append((start, cp if cp > 0 else n))
            weights.append(w)
            start = cp + 1 if cp > 0 else n + 1

        # store
        self.changepoints_ = cps
        self.segments_ = segments
        self.weights_ = weights
        self.knots_ = knots
        self.delta_ = delta
        self.degree_ = degree
        self.P_ = P
        self.beta_ = beta
        self._fitted = True
        return self

    # ----------------------------- Helpers -----------------------------

    def intensity_on_grid(self, seg_index: int, grid: Array1D) -> Array1D:
        """Evaluate λ̂_k(t) on a grid for segment k."""
        type_check(self._fitted, "Call .fit() first.")
        type_check(0 <= seg_index < len(self.weights_), "Invalid segment index.")
        Psi = bspline_design_matrix(np.asarray(grid, float), self.degree_, self.knots_)
        w = self.weights_[seg_index]
        return np.exp(Psi @ w)

    def segment_covariance(self, seg_index: int) -> Array2D:
        """
        Approximate Cov(ŵ_k) via observed Hessian inverse at optimum:
            H(ŵ) = L * ∫ exp(ψᵀŵ) ψψᵀ dt + ridge*I + 2·(γ DᵀD)
        so  Σ ≈ H^{-1}.
        """
        from .regularization import p_spline_RtR  # local import to avoid cycles

        type_check(self._fitted, "Call .fit() first.")
        type_check(0 <= seg_index < len(self.weights_), "Invalid segment index.")
        w = self.weights_[seg_index]
        i_start, i_end = self.segments_[seg_index]
        L = i_end - i_start + 1

        quad = QuadratureConfig(n_points=self.cfg.quad.n_points, ridge=self.cfg.hessian_ridge)
        opt = SegmentOptimizer(self.delta_, self.degree_, self.knots_, quad=quad, newton=self.cfg.newton,
                            pspline_gamma=self.cfg.pspline_gamma, pspline_order=self.cfg.pspline_order)
        _, _, I2 = opt._integrals(w)

        RtR = p_spline_RtR(self.P_, order=self.cfg.pspline_order, gamma=self.cfg.pspline_gamma)
        H = L * I2 + (self.cfg.hessian_ridge * np.eye(I2.shape[0], dtype=float)) + 2.0 * RtR

        try:
            Sigma = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            Sigma = np.linalg.pinv(H)
        return Sigma


    def intensity_with_bands(
        self,
        seg_index: int,
        grid: Array1D,
        ci: float = 0.95,
    ) -> Tuple[Array1D, Array1D, Array1D]:
        """
        Return λ̂, lower, upper using a delta-method CI on the log-scale:

            Var[log λ(t)] = ψ(t)ᵀ Σ ψ(t),
            Var[λ(t)] ≈ λ(t)^2 * Var[log λ(t)].

        No SciPy required (uses a built-in normal PPF).
        """
        type_check(self._fitted, "Call .fit() first.")
        type_check(0.0 < ci < 1.0, "ci must be in (0,1).")
        z = float(normal_ppf(0.5 + 0.5 * ci))

        Psi = bspline_design_matrix(np.asarray(grid, float), self.degree_, self.knots_)
        w = self.weights_[seg_index]
        lam = np.exp(Psi @ w)

        Sigma = self.segment_covariance(seg_index)
        var_log = np.einsum("ij,jk,ik->i", Psi, Sigma, Psi)
        se_log = np.sqrt(np.maximum(var_log, 0.0))
        log_l = np.log(lam) - z * se_log
        log_u = np.log(lam) + z * se_log
        return lam, np.exp(log_l), np.exp(log_u)

    # ---------------------------- Model choice ----------------------------

    @staticmethod
    def select_P_via_AIC(
        days: Sequence[Array1D],
        delta: float,
        degree: int,
        P_grid: Iterable[int],
        knot_strategy: str = "quantile",
        quad: QuadratureConfig = QuadratureConfig(),
        newton: NewtonConfig = NewtonConfig(),
    ) -> int:
        """
        Pick P by minimizing average AIC across days fitted individually.
        AIC_i(P) = 2P - 2*loglik_i(P),   loglik = - min cost with L=1.
        """
        type_check(len(days) > 0, "Need at least one day.")
        all_ts = np.concatenate([np.asarray(d, float) for d in days]) if len(days) else np.array([], float)

        def build_knots(P: int) -> Array1D:
            n_internal = P - degree - 1
            if n_internal <= 0:
                internal = None
            else:
                if knot_strategy == "quantile":
                    if all_ts.size >= n_internal + 1:
                        qs = np.linspace(0, 1, n_internal + 2)[1:-1]
                        internal = np.quantile(all_ts, qs)
                    else:
                        internal = np.linspace(0.0, delta, n_internal + 2)[1:-1]
                elif knot_strategy == "equispaced":
                    internal = np.linspace(0.0, delta, n_internal + 2)[1:-1]
                else:
                    raise ValueError("knot_strategy must be 'quantile' or 'equispaced'.")
            return open_uniform_knots(delta, degree, P, internal)

        best_P, best_avg = None, np.inf
        for P in P_grid:
            type_check(P >= degree + 1, f"P must be >= degree+1 (= {degree + 1}).")
            knots = build_knots(P)
            opt = SegmentOptimizer(delta, degree, knots, quad=quad, newton=newton)

            aics = []
            for d in days:
                d = np.asarray(d, float)
                if d.size == 0:
                    # discourage degenerate zero-event days
                    aics.append(1e6)
                    continue
                Psi = bspline_design_matrix(d, degree, knots)
                s = Psi.sum(axis=0)
                _, cost = opt.minimize(L=1, s=s)
                loglik = -cost
                aics.append(2 * P - 2 * loglik)

            avg = float(np.mean(aics))
            if avg < best_avg:
                best_avg, best_P = avg, P

        return int(best_P)
