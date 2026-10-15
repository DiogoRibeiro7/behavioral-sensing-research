"""The matched-sensor figures, drawn from the study's record.

Each figure reads summaries from the record and nothing else, so it can be
drawn again from the committed record alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .matched_sensors_planning import MOMENTS
from .matched_sensors_protocol import (
    C2,
    FEATURES,
    MATCHED,
    PROFILES,
    STANDARD,
    TRACKED_FEATURE,
)
from .uncertainty_figures import data_sha256

PREFIX = "matched-sensors"
FIGURES = ("detection", "tracking", "moments")

_COLOURS = {STANDARD: "#0072B2", MATCHED: "#D55E00"}
_LABELS = {STANDARD: "the simulator's sensors", MATCHED: "TIHM-matched sensors"}
_SHORT = {
    "kitchen_per_day": "kitchen a day",
    "bathroom_per_day": "bathroom a day",
    "bedroom_per_day": "bedroom a day",
    "living_per_day": "living room a day",
    "hall_per_day": "hallway a day",
    "switch_share": "after another room",
    "night_bedroom": "bedroom a night",
    "retrigger_gap_seconds": "retrigger gap",
}
_FEATURE_LABELS = {
    "bathroom_activity_hours": "bathroom hours",
    "kitchen_activity_hours": "kitchen hours",
    "sleeping_hours": "hours of sleep",
}


def _point(entry: Mapping[str, Any]) -> dict[str, float | None]:
    interval = entry.get("interval") or {}
    return {
        "estimate": entry.get("estimate"),
        "low": interval.get("low"),
        "high": interval.get("high"),
    }


def figure_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What each figure plots, from the record."""
    results = payload["results"]
    by_profile = results["E2"]["by_profile"]
    return {
        "detection": {
            profile: {
                name: _point(by_profile[profile][name])
                for name in ("detected", "false_detections", "excess_detection")
            }
            for profile in PROFILES
        },
        "tracking": {
            "margin": results["criteria"][C2]["margin"],
            "features": {
                feature: {
                    profile: _point(results["E4"][feature][profile])
                    for profile in PROFILES
                }
                for feature in FEATURES
            },
        },
        "moments": {
            name: {
                "tihm": results["E6"]["tihm"][name]["median"],
                **{
                    profile: results["E6"][profile][name]["median"]
                    for profile in PROFILES
                },
            }
            for name in MOMENTS
        },
    }


def _errors(points: list[dict[str, float | None]]) -> list[list[float]]:
    lows, highs = [], []
    for point in points:
        estimate = point["estimate"] or 0.0
        lows.append(estimate - (point["low"] if point["low"] is not None else estimate))
        highs.append(
            (point["high"] if point["high"] is not None else estimate) - estimate
        )
    return [lows, highs]


def _detection(plt: Any, data: Mapping[str, Any]) -> Any:
    names = ("detected", "false_detections", "excess_detection")
    titles = ("detected", "falsely detected", "excess detection")
    figure, axis = plt.subplots(figsize=(6.0, 3.0))
    width = 0.36
    for offset, profile in ((-width / 2, STANDARD), (width / 2, MATCHED)):
        points = [data[profile][name] for name in names]
        axis.bar(
            [k + offset for k in range(len(names))],
            [p["estimate"] or 0.0 for p in points],
            width=width,
            color=_COLOURS[profile],
            label=_LABELS[profile],
            yerr=_errors(points),
            capsize=3,
            error_kw={"lw": 0.8},
        )
    axis.axhline(0.0, color="#555555", lw=0.6)
    axis.set_xticks(range(len(names)), titles)
    axis.set_ylabel("share of homes; excess is a mean")
    axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return figure


def _tracking(plt: Any, data: Mapping[str, Any]) -> Any:
    features = list(data["features"])
    figure, axis = plt.subplots(figsize=(6.0, 2.6))
    for offset, profile in ((0.12, STANDARD), (-0.12, MATCHED)):
        points = [data["features"][f][profile] for f in features]
        axis.errorbar(
            [p["estimate"] or 0.0 for p in points],
            [k + offset for k in range(len(features))],
            xerr=_errors(points),
            fmt="o",
            color=_COLOURS[profile],
            label=_LABELS[profile],
            markersize=4,
            capsize=2,
            lw=0.8,
        )
    place = features.index(TRACKED_FEATURE)
    axis.plot(
        [data["margin"], data["margin"]],
        [place - 0.4, place + 0.4],
        color="#555555",
        lw=0.8,
        ls="--",
    )
    axis.text(
        data["margin"] - 0.01,
        place + 0.3,
        "margin",
        ha="right",
        va="center",
        fontsize=7,
        color="#555555",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_yticks(
        range(len(features)), [_FEATURE_LABELS.get(f, f) for f in features], fontsize=7
    )
    axis.set_xlabel("mean within-home Spearman correlation with the truth")
    axis.legend(frameon=False, fontsize=7, loc="upper left")
    figure.tight_layout()
    return figure


def _moments(plt: Any, data: Mapping[str, Any]) -> Any:
    names = list(data)
    figure, axis = plt.subplots(figsize=(6.0, 3.2))
    for offset, profile in ((0.12, STANDARD), (-0.12, MATCHED)):
        ratios = []
        for name in names:
            tihm, value = data[name]["tihm"], data[name][profile]
            if not tihm or value is None or value <= 0:
                ratios.append(math.nan)
            else:
                ratios.append(math.log2(value / tihm))
        axis.plot(
            ratios,
            [k + offset for k in range(len(names))],
            "o",
            color=_COLOURS[profile],
            label=_LABELS[profile],
            markersize=4,
        )
    for k, name in enumerate(names):
        if data[name][STANDARD] is None or data[name][STANDARD] <= 0:
            axis.text(
                -0.7,
                k + 0.12,
                "the simulator has no such sensor",
                ha="right",
                va="center",
                fontsize=7,
                color=_COLOURS[STANDARD],
            )
    axis.axvline(0.0, color="#555555", lw=0.8)
    axis.set_yticks(range(len(names)), [_SHORT[n] for n in names], fontsize=7)
    axis.set_xlabel("log2 of the simulated median over TIHM's")
    axis.legend(frameon=False, fontsize=7, loc="upper right")
    figure.tight_layout()
    return figure


def draw_figures(payload: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    """Draw every figure as SVG into *out_dir*; returns each figure's path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not payload["results"]["check"]["reproduced"]:
        return {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = figure_data(payload)
    drawers = {"detection": _detection, "tracking": _tracking, "moments": _moments}
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
                "simulated homes",
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
                    "Title": f"The simulator's homes with TIHM's sensors: {name}",
                    "Source": f"record {payload['experiment']}, "
                    f"recorded {payload['recorded_at']}",
                    "Description": f"data sha256 {data_sha256(data[name])}",
                },
            )
            plt.close(figure)
            paths[name] = path
    return paths
