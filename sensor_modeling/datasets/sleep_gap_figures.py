"""The sleep-gap figures, drawn from the description's record.

Each figure reads summaries from the record and nothing else, so it can be
drawn again from the committed record alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sleep_gap_plan import (
    AN_HOUR_LATER,
    AS_RECORDED,
    ASLEEP,
    AWAKE_ON_THE_MAT,
    CLOCKS,
    DISTANCE_BINS,
    MAT_IN_BED,
    MAT_SLEEP,
    NOTHING_YET,
    OFF_THE_MAT,
    SINCE_BINS,
)
from .uncertainty_figures import data_sha256

PREFIX = "sleep-gap"
FIGURES = ("composition", "hourly", "context")

_CLOCK_LABELS = {AS_RECORDED: "mat clock as recorded", AN_HOUR_LATER: "an hour later"}
_CLOCK_COLOURS = {AS_RECORDED: "#0072B2", AN_HOUR_LATER: "#56B4E9"}
_ON_ASLEEP, _ON_AWAKE, _OFF = "#0072B2", "#56B4E9", "#D55E00"
_COMPONENTS = (
    (f"pipeline_{OFF_THE_MAT}", "counted with no mat record", 1.0),
    (f"pipeline_{AWAKE_ON_THE_MAT}", "counted while awake in bed", 1.0),
    ("mat_sleep_not_counted", "mat's sleep not counted", -1.0),
    ("difference", "pipeline minus mat", 1.0),
)
_SINCE = (*(name for name, _, _ in SINCE_BINS), NOTHING_YET)
_DISTANCE = tuple(name for name, _, _ in DISTANCE_BINS)
_DISTANCE_LABELS = {
    "under_30_minutes": "under\n30 min",
    "30_minutes_to_2_hours": "30 min\nto 2 h",
    "2_to_6_hours": "2 to\n6 h",
    "6_hours_or_more": "6 h or\nmore",
}
_SINCE_LABELS = {
    "under_10_minutes": "under\n10 min",
    "10_to_60_minutes": "10 to\n60 min",
    "1_to_3_hours": "1 to\n3 h",
    "3_hours_or_more": "3 h or\nmore",
    NOTHING_YET: "nothing\nyet",
}


def _estimate(entry: Mapping[str, Any]) -> tuple[float, float, float]:
    estimate = float(entry["estimate"])
    interval = entry.get("interval") or {}
    return (
        estimate,
        float(interval.get("low", estimate)),
        float(interval.get("high", estimate)),
    )


def figure_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What each figure plots, from the record."""
    clocks = payload["results"]["clocks"]
    return {
        "composition": {
            clock: {
                name: _estimate(clocks[clock]["D1"][name]) for name, _, _ in _COMPONENTS
            }
            for clock in CLOCKS
        },
        "hourly": {
            clock: {
                name: list(clocks[clock]["D2"][name])
                for name in (
                    f"pipeline_{ASLEEP}",
                    f"pipeline_{AWAKE_ON_THE_MAT}",
                    f"pipeline_{OFF_THE_MAT}",
                    MAT_SLEEP,
                    MAT_IN_BED,
                )
            }
            for clock in CLOCKS
        },
        "context": {
            group: {
                clock: {
                    name: cells.get(name, {}).get("pipeline_sleep_hours", 0.0)
                    for name in names
                }
                for clock in CLOCKS
                for cells in (clocks[clock][estimand][group],)
            }
            for group, estimand, names in (
                ("since", "D4", _SINCE),
                ("distance", "D7", _DISTANCE),
            )
        },
    }


def _composition(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axis = plt.subplots(figsize=(6.0, 3.1))
    height = 0.36
    for offset, clock in ((-height / 2, AS_RECORDED), (height / 2, AN_HOUR_LATER)):
        values, lows, highs = [], [], []
        for name, _, sign in _COMPONENTS:
            estimate, low, high = data[clock][name]
            value = sign * estimate
            values.append(value)
            bounds = sorted((sign * low, sign * high))
            lows.append(value - bounds[0])
            highs.append(bounds[1] - value)
        axis.barh(
            [k + offset for k in range(len(_COMPONENTS))],
            values,
            height=height,
            color=_CLOCK_COLOURS[clock],
            label=_CLOCK_LABELS[clock],
            xerr=[lows, highs],
            capsize=2,
            error_kw={"lw": 0.8},
        )
    axis.axvline(0.0, color="#555555", lw=0.6)
    axis.set_yticks(range(len(_COMPONENTS)), [label for _, label, _ in _COMPONENTS])
    axis.invert_yaxis()
    axis.set_xlabel("hours a day; the mat's sleep not counted is subtracted")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=7, loc="lower left", ncol=2)
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    return figure


def _hourly(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    hours = list(range(25))

    def closed(values: Sequence[float]) -> list[float]:
        # The last hour is drawn to 24:00 by repeating its value.
        return [*values, values[-1]]

    for axis, clock in zip(axes, CLOCKS):
        series = data[clock]
        axis.stackplot(
            hours,
            closed(series[f"pipeline_{ASLEEP}"]),
            closed(series[f"pipeline_{AWAKE_ON_THE_MAT}"]),
            closed(series[f"pipeline_{OFF_THE_MAT}"]),
            colors=(_ON_ASLEEP, _ON_AWAKE, _OFF),
            labels=(
                "pipeline's sleep, mat asleep",
                "pipeline's sleep, mat awake",
                "pipeline's sleep, no mat record",
            ),
            step="post",
            alpha=0.85,
        )
        axis.step(
            hours,
            closed(series[MAT_IN_BED]),
            where="post",
            color="#222222",
            lw=1.2,
            label="mat: in bed",
        )
        axis.step(
            hours,
            closed(series[MAT_SLEEP]),
            where="post",
            color="#222222",
            lw=1.0,
            ls="--",
            label="mat: asleep",
        )
        axis.set_title(_CLOCK_LABELS[clock], fontsize=8)
        axis.set_xticks(range(0, 25, 3), [f"{h:02d}" for h in range(0, 25, 3)])
        axis.set_xlim(0, 24)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("local hour")
    axes[0].set_ylabel("hours of each hour, a day")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, frameon=False, fontsize=6.5, loc="lower left", ncol=3
    )
    figure.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    return figure


def _context(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    width = 0.36
    panels = (
        (
            "since",
            _SINCE,
            _SINCE_LABELS,
            "time since the last activation, as of the step",
        ),
        ("distance", _DISTANCE, _DISTANCE_LABELS, "time to the nearest mat record"),
    )
    for axis, (group, names, labels, title) in zip(axes, panels):
        for offset, clock in ((-width / 2, AS_RECORDED), (width / 2, AN_HOUR_LATER)):
            axis.bar(
                [k + offset for k in range(len(names))],
                [data[group][clock][name] for name in names],
                width=width,
                color=_CLOCK_COLOURS[clock],
                label=_CLOCK_LABELS[clock],
            )
        axis.set_xticks(
            range(len(names)), [labels[name] for name in names], fontsize=6.5
        )
        axis.set_xlabel(title)
    axes[0].set_ylabel("pipeline's sleep, no mat record,\nhours a day")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=7, loc="lower left", ncol=2)
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    return figure


def draw_figures(payload: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    """Draw every figure as SVG into *out_dir*; returns each figure's path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = figure_data(payload)
    homes = len(payload["results"]["clocks"][AS_RECORDED]["homes"])
    drawers = {"composition": _composition, "hourly": _hourly, "context": _context}
    paths: dict[str, Path] = {}
    with plt.rc_context(
        {
            "svg.hashsalt": PREFIX,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        for name in FIGURES:
            figure = drawers[name](plt, data[name])
            figure.text(
                0.995,
                0.005,
                f"{homes} TIHM homes; TIHM, Palermo et al. 2023, CC BY 4.0",
                ha="right",
                va="bottom",
                fontsize=6,
                color="#555555",
            )
            path = out_dir / f"{PREFIX}-{name}.svg"
            figure.savefig(
                path,
                format="svg",
                metadata={
                    "Date": None,
                    "Creator": None,
                    "Title": f"Where the pipeline's sleep parts from the mat: {name}",
                    "Source": f"record {payload['experiment']}, "
                    f"recorded {payload['recorded_at']}",
                    "Description": f"data sha256 {data_sha256(data[name])}",
                },
            )
            plt.close(figure)
            paths[name] = path
    return paths
