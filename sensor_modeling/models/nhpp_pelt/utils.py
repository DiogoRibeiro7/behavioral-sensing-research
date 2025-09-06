from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# ------------------------------- Types -------------------------------

Array1D = np.ndarray
Array2D = np.ndarray


# ---------------------------- Validators ----------------------------


def type_check(condition: bool, msg: str) -> None:
    """Runtime guard with a concise error message."""
    if not condition:
        raise ValueError(msg)


# -------------------------- Normal PPF (Φ⁻¹) -------------------------


def normal_ppf(p: float) -> float:
    """
    Approximate the standard-normal inverse CDF Φ⁻¹(p).
    Accuracy ~ 4e-4 absolute in typical ranges. SciPy-free.

    Parameters
    ----------
    p : float
        Probability in (0, 1).

    Returns
    -------
    float
        Approximate quantile.

    Notes
    -----
    Combines a Beasley–Springer/Wichura-style central approximation
    with a simple tail refinement. Good enough for CI bands.
    """
    type_check(0.0 < p < 1.0, "p must be in (0,1).")
    # exploit symmetry
    if p > 0.5:
        return -normal_ppf(1.0 - p)

    # coefficients (central region)
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]

    # auxiliary for mild tails
    c = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ]

    # central transform
    t = np.sqrt(-2.0 * np.log(p))
    x = t - (((a[3] * t + a[2]) * t + a[1]) * t + a[0]) / (
        (((b[3] * t + b[2]) * t + b[1]) * t + b[0]) * t + 1.0
    )

    # refine for not-too-small p
    if p >= 0.02425:
        q = p - 0.5
        r = q * q
        poly = c[8]
        for k in range(7, -1, -1):
            poly = poly * r + c[k]
        return float(q * poly)
    return float(x)


# --------------------------- JSON exporters --------------------------


def to_jsonable(model: "NHPPPELT") -> dict:
    """
    Convert a fitted model into a JSON-serializable dictionary.
    """
    return {
        "delta": float(model.delta_),
        "degree": int(model.degree_),
        "n_basis": int(model.P_),
        "beta": float(model.beta_),
        "knots": model.knots_.tolist(),
        "changepoints": [int(c) for c in model.changepoints_],
        "segments": [(int(i), int(j)) for (i, j) in model.segments_],
        "weights": [w.astype(float).tolist() for w in model.weights_],
    }


def save_results_json(model: "NHPPPELT", path: str) -> None:
    """
    Save fitted model parameters/results to a JSON file.

    Parameters
    ----------
    model : NHPPPELT
        Fitted model.
    path : str
        Destination path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(model), f, ensure_ascii=False, indent=2)


# ------------------------------ Sweep β ------------------------------


def sweep_beta(
    days: Sequence[Array1D],
    base_cfg: "NHPPConfig",
    betas: Iterable[float],
) -> List[Tuple[float, int, float]]:
    """
    Fit across candidate penalties and return tuples of:
        (beta, n_changepoints, total_penalized_cost)

    Useful to visualize an elbow/stability curve before fixing β.

    Notes
    -----
    - Reuses each solution’s segment weights as warm starts to
      recompute exact segment costs for the reported total.
    """
    from .bspline import bspline_design_matrix
    from .optimizer import SegmentOptimizer, QuadratureConfig
    from .model import NHPPConfig, NHPPPELT

    results: List[Tuple[float, int, float]] = []

    for b in betas:
        cfg = NHPPConfig(**{**base_cfg.__dict__, "penalty_beta": float(b)})
        model = NHPPPELT(cfg).fit(days)

        total_cost = 0.0
        for (i, j), w in zip(model.segments_, model.weights_):
            quad = QuadratureConfig(n_points=cfg.quad.n_points, ridge=cfg.hessian_ridge)
            opt = SegmentOptimizer(
                delta=model.delta_,
                degree=model.degree_,
                knots=model.knots_,
                quad=quad,
                newton=cfg.newton,
            )
            # build sufficient statistics s for days i..j
            s = np.zeros(cfg.n_basis, dtype=float)
            for d in range(i - 1, j):
                ev = np.asarray(days[d], dtype=float)
                if ev.size:
                    s += bspline_design_matrix(ev, cfg.degree, model.knots_).sum(axis=0)
            _, c = opt.minimize(L=j - i + 1, s=s, w0=w)  # warm start
            total_cost += float(c)

        total_cost += len(model.segments_) * float(b)
        results.append((float(b), len(model.changepoints_), total_cost))

    return results
