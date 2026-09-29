"""Figures for the hurdle predictive checks, drawn from the record alone.

:func:`figure_data` extracts what each figure plots from the record as written,
and :func:`draw_figures` draws it. Each SVG carries the SHA-256 of the data it
plots in its metadata, so a committed figure can be checked against the
published record without comparing rendered pixels.

The data are copied from the record verbatim. Every transformation, such as
an exponential, a sum or a ratio, is applied only when drawing, so the digest
does not depend on the platform that computes it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .predictive_checks import OWN, RESULT_SCHEMA

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

FIGURES = ("dispersion", "count-distribution", "runs", "heterogeneity")


def _by_household(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"values": {}, "mean": None}
    return {
        "values": dict(summary["values"]),
        "mean": summary.get("mean"),
    }


def figure_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What each figure plots, copied verbatim from the record."""
    results = payload["results"]
    if results.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    configuration = payload["configuration"]
    states = results["states"]
    common = results["common_states"]
    by_state = results["groups"]["state"][OWN]
    households = results["households"]
    distribution: dict[str, dict[str, Any]] = {}
    for state in common:
        distribution[state] = {}
        for home, entry in households.items():
            cells = [
                {
                    "observed": list(cell["observed_bins"]),
                    "expected": list(cell["references"][OWN]["expected_bins"]),
                }
                for cell in entry["cells"]
                if cell["state"] == state
                and cell["references"][OWN].get("expected_bins") is not None
            ]
            if cells:
                distribution[state][home] = cells
    return {
        "dispersion": {
            "minimal_ratio": configuration["minimal_ratio"],
            "states": {
                state: _by_household(by_state["active_dispersion"].get(state))
                for state in states
                if state in by_state["active_dispersion"]
            },
        },
        "count-distribution": {
            "bin_edges": list(configuration["distribution"]["bin_edges"]),
            "states": distribution,
        },
        "runs": {
            "minimal_ratio": configuration["minimal_ratio"],
            **{
                statistic: {
                    state: _by_household(by_state[statistic].get(state))
                    for state in states
                    if state in by_state[statistic]
                }
                for statistic in ("quiet_runs", "bursts")
            },
        },
        "heterogeneity": {
            "households": sorted(households),
            "states": common,
            "household_state": {
                state: dict(
                    by_state["active_dispersion"].get(state, {}).get("values", {})
                )
                for state in common
            },
            "channel_state": {
                state: {
                    channel: dict(entry)
                    for channel, entry in results["state_channel"][OWN][
                        "active_dispersion"
                    ]
                    .get(state, {})
                    .items()
                }
                for state in common
            },
        },
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
            "svg.hashsalt": "phase3-hurdle-predictive-checks",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        for name, draw in (
            ("dispersion", _dispersion),
            ("count-distribution", _count_distribution),
            ("runs", _runs),
            ("heterogeneity", _heterogeneity),
        ):
            figure = draw(plt, data[name])
            path = out_dir / f"phase3-predictive-{name}.svg"
            figure.savefig(
                path,
                format="svg",
                metadata={
                    "Date": None,
                    "Creator": None,
                    "Title": f"Hurdle predictive checks: {name}",
                    "Source": source,
                    "Description": f"data sha256 {data_sha256(data[name])}",
                },
            )
            plt.close(figure)
            paths[name] = path
    return paths


#: Ratios that may label a logarithmic axis.
_RATIO_TICKS = (0.125, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def _ratio_axis(axes: Any) -> None:
    """A logarithmic ratio axis labelled with plain numbers."""
    axes.set_yscale("log")
    low, high = axes.get_ylim()
    ticks = [t for t in _RATIO_TICKS if low <= t <= high]
    axes.set_yticks(ticks)
    axes.set_yticklabels([f"{t:g}" for t in ticks])
    axes.minorticks_off()


def _households_by_state(
    axes: Any, series: Mapping[str, Any], minimal: float, label: str
) -> None:
    """Each household's ratio per state, with the household mean and its interval."""
    states = list(series)
    for x, state in enumerate(states):
        values = [math.exp(v) for v in series[state]["values"].values()]
        offsets = [x + 0.25 * ((i % 7) - 3) / 3 for i in range(len(values))]
        axes.scatter(
            offsets,
            values,
            s=14,
            color=_COLOURS.get(state, "grey"),
            edgecolor="black",
            linewidth=0.3,
            zorder=2,
        )
        mean = series[state]["mean"]
        if mean is not None and mean.get("interval"):
            axes.errorbar(
                [x + 0.4],
                [math.exp(mean["estimate"])],
                yerr=[
                    [math.exp(mean["estimate"]) - math.exp(mean["interval"]["low"])],
                    [math.exp(mean["interval"]["high"]) - math.exp(mean["estimate"])],
                ],
                fmt="D",
                color="black",
                markersize=4,
                capsize=3,
                zorder=3,
            )
    axes.axhline(1.0, color="grey", linewidth=0.8, zorder=1)
    for bound in (minimal, 1.0 / minimal):
        axes.axhline(bound, color="grey", linewidth=0.8, linestyle="--", zorder=1)
    _ratio_axis(axes)
    axes.set_xticks(range(len(states)))
    axes.set_xticklabels(states, rotation=30, ha="right")
    axes.set_ylabel(label)


def _dispersion(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axes = plt.subplots(figsize=(6.5, 4.2))
    _households_by_state(
        axes,
        data["states"],
        data["minimal_ratio"],
        "active variance, observed over zero-truncated Poisson",
    )
    axes.set_title("Is the active count under-dispersed? One point per household")
    figure.tight_layout()
    return figure


def _count_distribution(plt: Any, data: Mapping[str, Any]) -> Any:
    edges = data["bin_edges"]
    labels = [
        str(low) if high - low == 1 else f"{low}-{high - 1}"
        for low, high in zip(edges, edges[1:], strict=False)
    ] + [f"{edges[-1]}+"]
    states = list(data["states"])
    figure, panels = plt.subplots(
        1, len(states), figsize=(3.2 * len(states), 3.6), sharey=True, squeeze=False
    )
    for axes, state in zip(panels[0], states, strict=True):
        for cells in data["states"][state].values():
            observed = [sum(c["observed"][i] for c in cells) for i in range(len(edges))]
            expected = [sum(c["expected"][i] for c in cells) for i in range(len(edges))]
            points = [
                (i, (o + 0.5) / (e + 0.5))
                for i, (o, e) in enumerate(zip(observed, expected, strict=True))
                if i > 0 and (o > 0 or e >= 0.5)
            ]
            axes.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                color=_COLOURS.get(state, "grey"),
                linewidth=0.7,
                alpha=0.8,
            )
        axes.axhline(1.0, color="grey", linewidth=0.8)
        _ratio_axis(axes)
        axes.set_xticks(range(1, len(labels)))
        axes.set_xticklabels(labels[1:], rotation=60, fontsize=7)
        axes.set_title(f"{state} ({len(data['states'][state])} households)")
        axes.set_xlabel("activations in an active window")
    panels[0][0].set_ylabel("windows observed over expected, own fit")
    figure.suptitle(
        "Active-count distribution against the hurdle: one line per household"
    )
    figure.tight_layout()
    return figure


def _runs(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, panels = plt.subplots(1, 2, figsize=(10.0, 4.0), squeeze=False)
    for axes, (statistic, title) in zip(
        panels[0],
        (
            ("quiet_runs", "Windows in long quiet runs"),
            ("bursts", "Windows in bursts"),
        ),
        strict=True,
    ):
        _households_by_state(
            axes, data[statistic], data["minimal_ratio"], "observed over predicted"
        )
        axes.set_title(title)
    figure.suptitle("Runs in time against independence given the state")
    figure.tight_layout()
    return figure


def _heterogeneity(plt: Any, data: Mapping[str, Any]) -> Any:
    import numpy as np

    homes = data["households"]
    states = data["states"]
    channels = sorted({c for per in data["channel_state"].values() for c in per})
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(11.0, 5.6),
        gridspec_kw={"width_ratios": [1.0, 1.2]},
        layout="constrained",
    )
    grid = np.full((len(homes), len(states)), np.nan)
    for j, state in enumerate(states):
        for i, home in enumerate(homes):
            value = data["household_state"][state].get(home)
            if value is not None:
                grid[i, j] = value
    limit = max(0.5, float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 0.5)
    image = left.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    left.set_xticks(range(len(states)))
    left.set_xticklabels(states, rotation=30, ha="right")
    left.set_yticks(range(len(homes)))
    left.set_yticklabels(homes, fontsize=7)
    left.set_title("Each household")
    table = np.full((len(channels), len(states)), np.nan)
    for j, state in enumerate(states):
        for i, channel in enumerate(channels):
            entry = data["channel_state"][state].get(channel)
            if entry is not None:
                table[i, j] = entry["median"]
                right.text(j, i, str(entry["n"]), ha="center", va="center", fontsize=7)
    right.imshow(table, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    right.set_xticks(range(len(states)))
    right.set_xticklabels(states, rotation=30, ha="right")
    right.set_yticks(range(len(channels)))
    right.set_yticklabels(channels)
    right.set_title("Median by channel (households)")
    colour = figure.colorbar(image, ax=[left, right], shrink=0.8)
    colour.set_label("log active variance, observed over predicted")
    figure.suptitle("Where the active count is over-dispersed")
    return figure
