"""Figures for the correlated-silence diagnostic, drawn from its record alone.

:func:`figure_data` extracts what each figure plots from the record as written,
and :func:`draw_figures` draws it. Each SVG carries the SHA-256 of the data it
plots in its metadata, so a committed figure can be checked against the
published record without comparing rendered pixels, which differ between
Matplotlib versions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .silence_dependence import RESULT_SCHEMA

#: Okabe-Ito colours, distinguishable with the common colour-vision deficiencies.
_COLOURS = {
    "away": "#E69F00",
    "home_active": "#D55E00",
    "home_inactive": "#56B4E9",
    "sleeping": "#0072B2",
    "bed_awake": "#CC79A7",
    "bathroom_activity": "#009E73",
    "kitchen_activity": "#F0E442",
}

FIGURES = ("joint-silence", "inflation", "overconfidence", "quiet-runs")


def figure_data(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """What each figure plots, read from the record."""
    results = payload["results"]
    if results.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    households = results["households"]
    points = [
        {
            "household": home,
            "state": state,
            "independent": math.log(stats["independent_joint_silence"]),
            "observed": math.log(stats["joint_silence"]),
        }
        for home, entry in households.items()
        for state in results["states"]
        if (stats := entry["states"].get(state)) is not None
        if stats["joint_estimable"]
    ]
    inflation = {
        "households": list(households),
        "dependence": [e["values"]["log_inflation"] for e in households.values()],
        "declared": [
            e["values"]["log_inflation_declared"] for e in households.values()
        ],
    }
    overconfidence: dict[str, Any] = {"households": {}}
    for home, entry in households.items():
        by_count = entry["concentration"]["by_silent_channels"]
        counts = sorted(by_count, key=int)
        overconfidence["households"][home] = {
            "silent_channels": [int(k) for k in counts],
            "gap": [
                by_count[k]["confidence"] - by_count[k]["accuracy"] for k in counts
            ],
        }
    pooled: dict[int, list[float]] = {}
    for series in overconfidence["households"].values():
        for k, gap in zip(series["silent_channels"], series["gap"], strict=True):
            pooled.setdefault(k, []).append(gap)
    overconfidence["mean"] = {
        str(k): sum(v) / len(v) for k, v in sorted(pooled.items())
    }
    quiet_runs = {
        state: {
            home: {
                "predicted_first": entry["cases"][state]["steps"][0][
                    "predicted_confidence"
                ],
                "confidence": [s["confidence"] for s in entry["cases"][state]["steps"]],
            }
            for home, entry in households.items()
            if state in entry["cases"]
        }
        for state in results["quiet_states"]
    }
    return {
        "joint-silence": {"points": points},
        "inflation": inflation,
        "overconfidence": overconfidence,
        "quiet-runs": quiet_runs,
    }


def data_sha256(data: Mapping[str, Any]) -> str:
    """SHA-256 of one figure's data, as canonical JSON."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def draw_figures(payload: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    """Draw every figure as SVG into *out_dir*; returns each figure's path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = figure_data(payload)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = f"record {payload['experiment']}, recorded {payload['recorded_at']}"
    paths: dict[str, Path] = {}
    with plt.rc_context(
        {
            "svg.hashsalt": "phase3-correlated-silence",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        for name, draw in (
            ("joint-silence", _joint_silence),
            ("inflation", _inflation),
            ("overconfidence", _overconfidence),
            ("quiet-runs", _quiet_runs),
        ):
            figure = draw(plt, data[name])
            path = out_dir / f"phase3-silence-{name}.svg"
            figure.savefig(
                path,
                format="svg",
                metadata={
                    "Date": None,
                    "Creator": None,
                    "Title": f"Phase 3.3 correlated silence: {name}",
                    "Source": source,
                    "Description": f"data sha256 {data_sha256(data[name])}",
                },
            )
            plt.close(figure)
            paths[name] = path
    return paths


def _joint_silence(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axes = plt.subplots(figsize=(5.5, 4.5))
    points = data["points"]
    for state, colour in _COLOURS.items():
        chosen = [p for p in points if p["state"] == state]
        if chosen:
            axes.scatter(
                [p["independent"] for p in chosen],
                [p["observed"] for p in chosen],
                s=18,
                color=colour,
                edgecolor="black",
                linewidth=0.3,
                label=state,
            )
    values = [p["independent"] for p in points] + [p["observed"] for p in points]
    if values:
        low, high = min(values), max(0.0, max(values))
        axes.plot([low, high], [low, high], color="grey", linewidth=0.8, zorder=0)
    axes.set_xlabel("log joint silence if channels were independent")
    axes.set_ylabel("log observed joint silence")
    axes.set_title("Every channel silent: observed against independence")
    axes.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return figure


def _inflation(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axes = plt.subplots(figsize=(7.0, 3.8))
    homes = data["households"]
    positions = range(len(homes))
    for key, colour, marker, label in (
        ("dependence", "#0072B2", "o", "dependence alone (D1)"),
        ("declared", "#D55E00", "s", "declared rates (S1)"),
    ):
        pairs = [
            (x, math.exp(v)) for x, v in zip(positions, data[key]) if v is not None
        ]
        axes.scatter(
            [x for x, _ in pairs],
            [y for _, y in pairs],
            color=colour,
            marker=marker,
            s=22,
            label=label,
        )
    axes.axhline(1.0, color="grey", linewidth=0.8)
    axes.axhline(1.25, color="grey", linewidth=0.8, linestyle="--")
    axes.set_yscale("log")
    axes.set_xticks(list(positions))
    axes.set_xticklabels(homes, rotation=60, fontsize=7)
    axes.set_ylabel("silence-evidence inflation factor")
    axes.set_title("How much independence overstates joint-silence evidence")
    axes.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return figure


def _overconfidence(plt: Any, data: Mapping[str, Any]) -> Any:
    from matplotlib.ticker import MaxNLocator

    figure, axes = plt.subplots(figsize=(5.5, 4.0))
    for series in data["households"].values():
        axes.plot(
            series["silent_channels"],
            series["gap"],
            color="#56B4E9",
            linewidth=0.6,
            alpha=0.7,
        )
    mean = data["mean"]
    axes.plot(
        [int(k) for k in mean],
        list(mean.values()),
        color="#0072B2",
        linewidth=2.0,
        marker="o",
        label="mean across households",
    )
    axes.axhline(0.0, color="grey", linewidth=0.8)
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_xlabel("silent channels in the window")
    axes.set_ylabel("confidence minus accuracy")
    axes.set_title("Overconfidence by the number of silent channels")
    axes.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return figure


def _quiet_runs(plt: Any, data: Mapping[str, Any]) -> Any:
    states = list(data)
    figure, panels = plt.subplots(
        1, len(states), figsize=(3.0 * len(states), 3.4), sharey=True, squeeze=False
    )
    for axes, state in zip(panels[0], states, strict=True):
        for case in data[state].values():
            steps = list(range(len(case["confidence"]) + 1))
            axes.plot(
                steps,
                [case["predicted_first"], *case["confidence"]],
                color=_COLOURS.get(state, "black"),
                linewidth=0.7,
                alpha=0.8,
            )
        if not data[state]:
            axes.text(
                0.5, 0.5, "no qualifying run", ha="center", transform=axes.transAxes
            )
        axes.set_title(f"{state} ({len(data[state])} households)")
        axes.set_xlabel("fully silent windows")
        axes.set_ylim(0.0, 1.02)
    panels[0][0].set_ylabel("posterior confidence")
    figure.suptitle("Representative quiet periods: confidence window by window")
    figure.tight_layout()
    return figure
