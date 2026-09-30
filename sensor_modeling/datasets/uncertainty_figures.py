"""Figures for the Phase 4 uncertainty-diagnostics comparison, drawn from the record alone.

:func:`figure_data` copies what each figure plots from the record verbatim, and
:func:`draw_figures` draws it. Each SVG carries the SHA-256 of the data it
plots in its metadata, so a committed figure can be checked against the
published record without comparing rendered pixels.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .uncertainty_experiment import RESULT_SCHEMA, SIGNALS

FIGURES = ("error", "balanced-accuracy", "calibration", "retention")

#: Okabe-Ito colours, distinguishable with the common colour-vision deficiencies.
_COLOURS = {
    "confidence": "#000000",
    "entropy": "#999999",
    "structural_disagreement": "#0072B2",
    "evidence_disagreement": "#D55E00",
    "predictive_mismatch": "#009E73",
}


def figure_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What each figure plots, copied verbatim from the record."""
    results = payload["results"]
    if results.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    section = payload["selective_prediction"]
    grid = section["coverages"]
    signals = section["signals"]

    def band(name: str, metric: str) -> dict[str, Any]:
        pooled = signals[name]["pooled"]
        intervals = pooled["intervals"].get(metric)
        return {
            "estimate": pooled[metric],
            "low": intervals["low"] if intervals is not None else None,
            "high": intervals["high"] if intervals is not None else None,
        }

    random = section["reference"]["random"]
    share = payload["configuration"]["difficult_minority_states"]["minority_share"]
    difficult = results["difficult_minority_states"]
    minority = [
        s for s, e in results["states"].items() if e["share"] < share or s in difficult
    ]
    return {
        "error": {
            "coverages": grid,
            "signals": {name: band(name, "error") for name in SIGNALS},
            "random": random["error"],
            "oracle": section["reference"]["oracle"]["risk"],
        },
        "balanced-accuracy": {
            "coverages": grid,
            "signals": {name: band(name, "balanced_accuracy") for name in SIGNALS},
            "random": random["balanced_accuracy"],
        },
        "calibration": {
            "coverages": grid,
            "signals": {name: band(name, "calibration_error") for name in SIGNALS},
            "random": random["calibration_error"],
        },
        "retention": {
            "coverages": grid,
            "difficult": difficult,
            "states": {
                state: {
                    name: signals[name]["pooled"]["state_coverage"][state]
                    for name in SIGNALS
                }
                for state in minority
            },
        },
    }


def data_sha256(data: Any) -> str:
    """SHA-256 of a figure's data, canonically serialised."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _curves(plt: Any, data: Mapping[str, Any], ylabel: str, reference: str) -> Any:
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    grid = data["coverages"]
    for name, band in data["signals"].items():
        colour = _COLOURS[name]
        values = [float("nan") if v is None else v for v in band["estimate"]]
        axis.plot(grid, values, color=colour, label=name.replace("_", " "), lw=1.6)
        if band["low"] is not None and band["high"] is not None:
            low = [float("nan") if v is None else v for v in band["low"]]
            high = [float("nan") if v is None else v for v in band["high"]]
            axis.fill_between(grid, low, high, color=colour, alpha=0.12, lw=0)
    if data.get("random") is not None:
        axis.axhline(data["random"], color="#777777", ls=":", lw=1.0, label=reference)
    if "oracle" in data:
        axis.plot(
            grid, data["oracle"], color="#777777", ls="--", lw=1.0, label="oracle"
        )
    axis.set_xlabel("coverage")
    axis.set_ylabel(ylabel)
    axis.set_xlim(0.0, 1.0)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return figure


def _retention(plt: Any, data: Mapping[str, Any]) -> Any:
    states = list(data["states"])
    count = max(len(states), 1)
    figure, axes = plt.subplots(1, count, figsize=(3.2 * count, 3.2), squeeze=False)
    grid = data["coverages"]
    for axis, state in zip(axes[0], states):
        for name, values in data["states"][state].items():
            ratios = [
                float("nan") if v is None else v / c for v, c in zip(values, grid)
            ]
            axis.plot(grid, ratios, color=_COLOURS[name], label=name.replace("_", " "))
        axis.axhline(1.0, color="#777777", ls=":", lw=1.0)
        marker = " (difficult)" if state in data["difficult"] else ""
        axis.set_title(state.replace("_", " ") + marker)
        axis.set_xlabel("coverage")
        axis.set_xlim(0.0, 1.0)
    axes[0][0].set_ylabel("retained share / coverage")
    if not states:
        axes[0][0].text(0.5, 0.5, "no minority state", ha="center")
    else:
        axes[0][-1].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return figure


def draw_figures(payload: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    """Draw every figure as SVG into *out_dir*; returns each figure's path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = figure_data(payload)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = f"record {payload['experiment']}, recorded {payload['recorded_at']}"
    drawers = {
        "error": lambda d: _curves(plt, d, "selective error", "random rejection"),
        "balanced-accuracy": lambda d: _curves(
            plt, d, "selective balanced accuracy", "random rejection"
        ),
        "calibration": lambda d: _curves(
            plt, d, "calibration error of retained", "random rejection"
        ),
        "retention": lambda d: _retention(plt, d),
    }
    paths: dict[str, Path] = {}
    with plt.rc_context(
        {
            "svg.hashsalt": "phase4-uncertainty-diagnostics",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        for name in FIGURES:
            figure = drawers[name](data[name])
            path = out_dir / f"phase4-uncertainty-{name}.svg"
            figure.savefig(
                path,
                format="svg",
                metadata={
                    "Date": None,
                    "Creator": None,
                    "Title": f"Phase 4 uncertainty diagnostics: {name}",
                    "Source": source,
                    "Description": f"data sha256 {data_sha256(data[name])}",
                },
            )
            plt.close(figure)
            paths[name] = path
    return paths
