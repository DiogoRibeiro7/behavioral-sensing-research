"""Figures for the threshold-calibration pages, drawn from the records alone.

:func:`null_figure_data` copies what each figure of the measurement on
synthetic days plots from its record, and :func:`draw_null_figures` draws
them. Each SVG carries the SHA-256 of the data it plots in its metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .external_figures import data_sha256
from .threshold_null import (
    CALIBRATED,
    DEFAULT,
    NONE,
    RAMP2,
    REFERENCES,
    STEP2,
)
from .threshold_null import RESULT_SCHEMA as NULL_SCHEMA

NULL_FIGURES = ("sizes", "operating")
NULL_PREFIX = "threshold-null"

_COLOURS = {DEFAULT: "#D55E00", CALIBRATED: "#0072B2", "stated": "#555555"}
_PANELS = {STEP2: "a step of 2 SD", RAMP2: "a ramp of 2 SD"}


def null_figure_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What each figure of the measurement plots, copied from its record."""
    results, configuration = payload["results"], payload["configuration"]
    if results.get("result_schema") != NULL_SCHEMA:
        raise ValueError(f"not a {NULL_SCHEMA} record")
    baseline = configuration["baseline"]
    threshold = f"{baseline['deviation_threshold']:g}"
    cells = {ref: results["rates"][NONE][ref] for ref in REFERENCES}
    sizes = sorted(cells[DEFAULT]["by_weekday_size"], key=int)
    scales = [f"{scale:g}" for scale in configuration["scales"]]
    operating = results["operating"]
    return {
        "sizes": {
            "threshold": float(threshold),
            "stated": results["nominal"][threshold],
            "sizes": [int(size) for size in sizes],
            "share": {
                ref: [
                    cells[ref]["by_weekday_size"][size][threshold]["rate"]
                    for size in sizes
                ]
                for ref in REFERENCES
            },
            "pooled": {
                ref: cells[ref]["phase"]["pooled"][threshold]["rate"]
                for ref in REFERENCES
            },
        },
        "operating": {
            "scales": [float(scale) for scale in scales],
            "matched": sorted(
                {
                    float(results["matching"]["plain"][which])
                    for which in ("same_detection", "same_false_reports")
                }
            ),
            "false": {
                ref: [operating[NONE][ref][s]["found"]["share"] for s in scales]
                for ref in REFERENCES
            },
            "found": {
                shape: {
                    ref: [operating[shape][ref][s]["found"]["share"] for s in scales]
                    for ref in REFERENCES
                }
                for shape in _PANELS
            },
        },
    }


def _sizes(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axis = plt.subplots(figsize=(6.4, 3.2))
    for reference in REFERENCES:
        axis.plot(
            data["sizes"],
            [100.0 * share for share in data["share"][reference]],
            marker="o",
            ms=3,
            lw=1.2,
            color=_COLOURS[reference],
            label=f"{reference} reference",
        )
    axis.axhline(100.0 * data["stated"], color=_COLOURS["stated"], ls=":", lw=1.0)
    axis.annotate(
        f"what a threshold of {data['threshold']:g} states",
        xy=(0.99, 100.0 * data["stated"]),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        fontsize=7,
        color=_COLOURS["stated"],
    )
    axis.set_yscale("log")
    axis.set_yticks([0.1, 0.3, 1, 3, 10, 30])
    axis.set_yticklabels(["0.1%", "0.3%", "1%", "3%", "10%", "30%"])
    axis.set_ylim(0.08, 40)
    axis.set_xticks(data["sizes"])
    axis.set_xlabel("days of the same weekday behind the reference")
    axis.set_ylabel("days at or past the threshold")
    axis.legend(frameon=False, fontsize=7, loc="center right")
    figure.tight_layout()
    return figure


def _operating(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axes = plt.subplots(1, len(_PANELS), figsize=(7.2, 3.3), sharey=True)
    for axis, (shape, title) in zip(axes, _PANELS.items()):
        for reference in REFERENCES:
            false = [100.0 * share for share in data["false"][reference]]
            found = [100.0 * share for share in data["found"][shape][reference]]
            axis.plot(
                false,
                found,
                marker="o",
                ms=3,
                lw=1.2,
                color=_COLOURS[reference],
                label=f"{reference} reference",
            )
            marked = [1.0] if reference == DEFAULT else data["matched"]
            for scale in marked:
                place = data["scales"].index(scale)
                axis.plot(
                    [false[place]],
                    [found[place]],
                    marker="o",
                    ms=7,
                    mfc="none",
                    mec=_COLOURS[reference],
                    lw=0,
                )
                axis.annotate(
                    f"{scale:g}",
                    xy=(false[place], found[place]),
                    xytext=(5, -9),
                    textcoords="offset points",
                    fontsize=7,
                    color=_COLOURS[reference],
                )
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("series with a false report, nothing added (%)")
        axis.set_xlim(-1, 30)
        axis.set_ylim(0, 102)
    axes[0].set_ylabel("series in which the change is found (%)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    figure.tight_layout()
    return figure


def draw_null_figures(payload: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    """Draw the measurement's figures as SVG into *out_dir*; returns their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = null_figure_data(payload)
    drawers = {"sizes": _sizes, "operating": _operating}
    series = payload["configuration"]
    paths: dict[str, Path] = {}
    with plt.rc_context(
        {
            "svg.hashsalt": NULL_PREFIX,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        for name in NULL_FIGURES:
            figure = drawers[name](plt, data[name])
            count = series["rate_series"] if name == "sizes" else series["series"]
            figure.text(
                0.995,
                0.005,
                f"{count:,} synthetic Gaussian series; no home",
                ha="right",
                va="bottom",
                fontsize=7,
                color="#555555",
            )
            path = out_dir / f"{NULL_PREFIX}-{name}.svg"
            figure.savefig(
                path,
                format="svg",
                metadata={
                    "Date": None,
                    "Creator": None,
                    "Title": f"The baseline's thresholds on synthetic days: {name}",
                    "Source": f"record {payload['experiment']}, "
                    f"recorded {payload['recorded_at']}",
                    "Description": f"data sha256 {data_sha256(data[name])}",
                },
            )
            plt.close(figure)
            paths[name] = path
    return paths
