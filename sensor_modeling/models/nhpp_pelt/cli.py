from __future__ import annotations

import argparse
import json
import os
from typing import List, Sequence

import numpy as np
import matplotlib.pyplot as plt

from .model import NHPPConfig, NHPPPELT
from .plotting import PlotConfig, HistConfig, plot_segments_and_intensities_with_histograms
from .utils import save_results_json
from .diagnostics import DiagConfig, plot_time_rescaling_diagnostics


Array1D = np.ndarray


# --------------------------- I/O helpers ---------------------------

def _parse_grid(expr: str) -> List[float]:
    """
    Parse a numeric grid:
      - comma list: 0,0.001,0.01
      - range: start:stop:step  (inclusive end within tolerance)
    """
    if ":" in expr:
        a, b, c = expr.split(":")
        start, stop, step = float(a), float(b), float(c)
        if step <= 0:
            raise ValueError("step must be > 0")
        vals, x = [], start
        while x <= stop + 1e-12:
            vals.append(x)
            x += step
        return vals
    return [float(x) for x in expr.split(",") if x.strip()]


def _load_npz_days(path: str) -> List[Array1D]:
    """
    Load days from a .npz created as:
        np.savez(path, days=np.array(list_of_arrays, dtype=object))
    """
    with np.load(path, allow_pickle=True) as data:
        if "days" not in data:
            raise ValueError("NPZ must contain array 'days' (dtype=object).")
        arr = data["days"]
    if arr.dtype != object:
        raise ValueError("NPZ 'days' must be an object array of 1-D float arrays.")
    return [np.asarray(x, float) for x in arr.tolist()]


def _parse_betas(expr: str) -> List[float]:
    """
    Parse betas:
      - Comma list: 10,20,40
      - Range: start:stop:step  (inclusive-ish, step > 0)
    """
    if ":" in expr:
        a, b, c = expr.split(":")
        start, stop, step = float(a), float(b), float(c)
        if step <= 0:
            raise ValueError("step must be > 0")
        vals = []
        x = start
        # include stop within tolerance
        while x <= stop + 1e-12:
            vals.append(x)
            x += step
        return vals
    return [float(x) for x in expr.split(",") if x.strip()]


# --------------------------- Subcommands ---------------------------

def cmd_fit(args: argparse.Namespace) -> None:
    days = _load_npz_days(args.input)
    cfg = NHPPConfig(
        delta=args.delta,
        degree=args.degree,
        n_basis=args.n_basis,
        knot_strategy=args.knot_strategy,
        min_seg_len=args.min_seg_len,
        penalty_beta=(None if args.beta is None else float(args.beta)),
    )
    model = NHPPPELT(cfg).fit(days)
    print("Changepoints:", model.changepoints_)
    print("Segments:", model.segments_)

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        save_results_json(model, args.json_out)
        print(f"Saved JSON to {args.json_out}")

    if args.plot_out:
        fig = plot_segments_and_intensities_with_histograms(
            days,
            model,
            config=PlotConfig(grid_points=args.grid_points),
            hist=HistConfig(bins="fd", density=True, alpha=0.4),
            save_path=args.plot_out,
        )
        plt.close(fig)
        print(f"Saved plot to {args.plot_out}")

    if args.diag_out:
        fig = plot_time_rescaling_diagnostics(
            days,
            model,
            config=DiagConfig(grid_points=args.grid_points),
            save_path=args.diag_out,
        )
        plt.close(fig)
        print(f"Saved diagnostics to {args.diag_out}")


def cmd_sweep_beta(args: argparse.Namespace) -> None:
    from .utils import sweep_beta

    days = _load_npz_days(args.input)
    base_cfg = NHPPConfig(
        delta=args.delta,
        degree=args.degree,
        n_basis=args.n_basis,
        knot_strategy=args.knot_strategy,
        min_seg_len=args.min_seg_len,
        penalty_beta=None,
    )
    betas = _parse_betas(args.betas)
    results = sweep_beta(days, base_cfg, betas)
    # Print as JSON lines for easy piping
    for beta, n_cp, total in results:
        print(json.dumps({"beta": beta, "n_changepoints": n_cp, "total_cost": total}))
        

def cmd_sweep_gamma(args: argparse.Namespace) -> None:
    from .regularization import sweep_gamma
    from .model import NHPPConfig

    days = _load_npz_days(args.input)
    base_cfg = NHPPConfig(
        delta=args.delta,
        degree=args.degree,
        n_basis=args.n_basis,
        knot_strategy=args.knot_strategy,
        min_seg_len=args.min_seg_len,
        penalty_beta=(None if args.beta is None else float(args.beta)),
    )
    gammas = _parse_grid(args.gammas)
    results = sweep_gamma(days, base_cfg, gammas, order=args.order)
    for gamma, n_cp, total in results:
        print(json.dumps({"gamma": gamma, "n_changepoints": n_cp, "total_cost": total}))


# --------------------------- CLI wiring ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nhpp",
        description="NHPP segmentation with B-splines & PELT (fit/plot/diagnostics).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # fit
    pf = sub.add_parser("fit", help="Fit model to days and optionally save plots/JSON.")
    pf.add_argument("--input", required=True, help="Path to .npz with object array 'days'.")
    pf.add_argument("--delta", type=float, default=24.0)
    pf.add_argument("--degree", type=int, default=3)
    pf.add_argument("--n-basis", type=int, default=5, dest="n_basis")
    pf.add_argument("--knot-strategy", choices=["quantile", "equispaced"], default="quantile")
    pf.add_argument("--min-seg-len", type=int, default=2, dest="min_seg_len")
    pf.add_argument("--beta", type=float, default=None, help="If omitted, uses SIC.")
    pf.add_argument("--grid-points", type=int, default=500)
    pf.add_argument("--json-out", default=None, help="Save fitted state JSON here.")
    pf.add_argument("--plot-out", default=None, help="Save segmentation plot PNG here.")
    pf.add_argument("--diag-out", default=None, help="Save time-rescaling diagnostics PNG here.")
    pf.set_defaults(func=cmd_fit)

    # sweep-beta
    ps = sub.add_parser("sweep-beta", help="Evaluate different penalties (β).")
    ps.add_argument("--input", required=True, help="Path to .npz with object array 'days'.")
    ps.add_argument("--delta", type=float, default=24.0)
    ps.add_argument("--degree", type=int, default=3)
    ps.add_argument("--n-basis", type=int, default=5, dest="n_basis")
    ps.add_argument("--knot-strategy", choices=["quantile", "equispaced"], default="quantile")
    ps.add_argument("--min-seg-len", type=int, default=2, dest="min_seg_len")
    ps.add_argument("--betas", required=True, help="Comma list or start:stop:step.")
    ps.set_defaults(func=cmd_sweep_beta)

    # sweep-gamma
    pg = sub.add_parser("sweep-gamma", help="Evaluate different P-spline strengths (γ).")
    pg.add_argument("--input", required=True, help="Path to .npz with object array 'days'.")
    pg.add_argument("--delta", type=float, default=24.0)
    pg.add_argument("--degree", type=int, default=3)
    pg.add_argument("--n-basis", type=int, default=5, dest="n_basis")
    pg.add_argument("--knot-strategy", choices=["quantile", "equispaced"], default="quantile")
    pg.add_argument("--min-seg-len", type=int, default=2, dest="min_seg_len")
    pg.add_argument("--beta", type=float, default=None, help="If omitted, uses SIC.")
    pg.add_argument("--gammas", required=True, help="Comma list or start:stop:step, e.g. 0:0.1:0.005")
    pg.add_argument("--order", type=int, default=2, help="Difference order (usually 2).")
    pg.set_defaults(func=cmd_sweep_gamma)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
