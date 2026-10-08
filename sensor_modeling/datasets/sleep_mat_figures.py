"""The sleep-mat figures, drawn from the record of the comparison on TIHM.

Each figure reads per-home summaries from the record and nothing else, so it
can be drawn again from the committed record alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .sleep_mat_protocol import IN_BED, PRIMARY, SLEEP
from .uncertainty_figures import data_sha256

PREFIX = "sleep-mat"
FIGURES = ("homes", "differences", "clocks")

_COLOURS = {SLEEP: "#0072B2", IN_BED: "#D55E00"}
_LABELS = {SLEEP: "against the mat's sleep", IN_BED: "against its hours in bed"}


def figure_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What each figure plots, from the record."""
    results = payload["results"]
    primary = results["analyses"][PRIMARY]
    homes = primary["homes"]
    rows = {home: results["homes"][home][PRIMARY] for home in homes}
    criteria = results["criteria"]
    lags = sorted(results["E11_clocks"]["lags"].items(), key=lambda item: int(item[0]))
    return {
        "homes": {
            "homes": homes,
            "values": {
                reference: [rows[h][reference]["spearman"] for h in homes]
                for reference in (SLEEP, IN_BED)
            },
            "estimate": primary[SLEEP]["spearman"],
            "margin": criteria["C1_the_pipeline_follows_the_mat"]["margin"],
        },
        "differences": {
            "homes": homes,
            "values": {
                reference: [rows[h][reference]["mean_difference"] for h in homes]
                for reference in (SLEEP, IN_BED)
            },
            "estimate": primary[SLEEP]["mean_difference"],
            "margin": criteria["C2_the_pipeline_agrees_in_level"]["margin"],
        },
        "clocks": {
            "lags": [int(lag) for lag, _ in lags],
            "estimates": [entry["spearman"] for _, entry in lags],
        },
    }


def _homes(plt: Any, data: Mapping[str, Any], key: str, axis_label: str) -> Any:
    homes = data["homes"]
    figure, axis = plt.subplots(figsize=(6.4, 0.32 * len(homes) + 1.4))
    order = sorted(range(len(homes)), key=lambda k: data["values"][SLEEP][k])
    places = list(range(len(homes)))
    for offset, reference in ((0.12, SLEEP), (-0.12, IN_BED)):
        axis.plot(
            [data["values"][reference][k] for k in order],
            [p + offset for p in places],
            "o",
            color=_COLOURS[reference],
            label=_LABELS[reference],
            markersize=4,
        )
    estimate = data["estimate"]
    if estimate.get("interval"):
        low, high = estimate["interval"]["low"], estimate["interval"]["high"]
        axis.axvspan(low, high, color=_COLOURS[SLEEP], alpha=0.12, lw=0)
    axis.axvline(estimate["estimate"], color=_COLOURS[SLEEP], lw=1)
    margin = data["margin"]
    if key == "differences":
        axis.axvspan(-margin, margin, color="#BBBBBB", alpha=0.25, lw=0)
        axis.axvline(0.0, color="#555555", lw=0.6)
    else:
        axis.axvline(margin, color="#555555", lw=0.8, ls="--")
        axis.axvline(0.0, color="#888888", lw=0.6)
    axis.set_yticks(places, [homes[k] for k in order], fontsize=7)
    axis.set_xlabel(axis_label)
    axis.legend(frameon=False, fontsize=7, loc="lower right")
    figure.tight_layout()
    return figure


def _clocks(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axis = plt.subplots(figsize=(4.8, 3.0))
    lags = data["lags"]
    centre = [entry["estimate"] for entry in data["estimates"]]
    low = [entry["interval"]["low"] for entry in data["estimates"]]
    high = [entry["interval"]["high"] for entry in data["estimates"]]
    axis.fill_between(lags, low, high, color=_COLOURS[IN_BED], alpha=0.15, lw=0)
    axis.plot(lags, centre, "o-", color=_COLOURS[IN_BED], markersize=4)
    axis.axvline(0, color="#888888", lw=0.6)
    axis.set_xticks(lags)
    axis.set_xlabel("lag of the mat's clock, hours")
    axis.set_ylabel("hourly activity against\nminutes in bed (Spearman)")
    figure.tight_layout()
    return figure


def draw_figures(payload: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    """Draw every figure as SVG into *out_dir*; returns each figure's path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = figure_data(payload)
    drawers = {
        "homes": lambda p, d: _homes(
            p, d, "homes", "within-home Spearman correlation with the mat"
        ),
        "differences": lambda p, d: _homes(
            p, d, "differences", "pipeline minus mat, hours a day"
        ),
        "clocks": _clocks,
    }
    paths: dict[str, Path] = {}
    with plt.rc_context(
        {
            "svg.hashsalt": PREFIX,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        for name in FIGURES:
            figure = drawers[name](plt, data[name])
            figure.text(
                0.995,
                0.005,
                "TIHM homes with a sleep mat",
                ha="right",
                va="bottom",
                fontsize=7,
                color="#555555",
            )
            path = out_dir / f"{PREFIX}-{name}.svg"
            figure.savefig(
                path,
                format="svg",
                metadata={
                    "Date": None,
                    "Creator": None,
                    "Title": f"The pipeline's hours of sleep beside a sleep mat: {name}",
                    "Source": f"record {payload['experiment']}, "
                    f"recorded {payload['recorded_at']}",
                    "Description": f"data sha256 {data_sha256(data[name])}",
                },
            )
            plt.close(figure)
            paths[name] = path
    return paths
