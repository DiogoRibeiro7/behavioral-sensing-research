"""Figures for the Phase 5 external results, drawn from the record alone.

:func:`figure_data` copies what each figure plots from the record verbatim, and
:func:`draw_figures` draws it. Each SVG carries the SHA-256 of the data it
plots in its metadata.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .external_experiment import CONDITIONS, ORACLE, RESULT_SCHEMA

FIGURES = ("metrics", "priors", "rates")

_COLOURS = {
    "declared": "#999999",
    "zero_shot": "#0072B2",
    "adapted": "#009E73",
    "oracle": "#E69F00",
    "truth": "#000000",
    "development": "#56B4E9",
    "predicted": "#D55E00",
}

_PRIOR_LABELS = {
    "truth": "scored labels",
    "development": "development panel",
    "predicted": "zero-shot predictions",
}


def figure_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What each figure plots, copied verbatim from the record."""
    results = payload["results"]
    if results.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    homes = {h: e for h, e in results["households"].items() if e.get("eligible")}
    return {
        "metrics": {
            h: {
                "conditions": {
                    c: e["conditions"][c]["balanced_accuracy"] for c in CONDITIONS
                },
                ORACLE: e["diagnostics"]["oracle"]["balanced_accuracy"],
                "chance": e["chance"],
            }
            for h, e in homes.items()
        },
        "priors": {
            h: {
                "truth": e["diagnostics"]["state_prior_shift"]["truth"],
                "development": e["diagnostics"]["state_prior_shift"].get("development"),
                "predicted": e["diagnostics"]["state_prior_shift"]["predicted"],
            }
            for h, e in homes.items()
        },
        "rates": {h: e["diagnostics"]["event_rates"] for h, e in homes.items()},
    }


def data_sha256(data: Any) -> str:
    """SHA-256 of a figure's data, canonically serialised."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _metrics(plt: Any, data: Mapping[str, Any]) -> Any:
    homes = list(data)
    figure, axes = plt.subplots(
        1, len(homes), figsize=(3.4 * len(homes), 3.2), squeeze=False
    )
    names = [*CONDITIONS, ORACLE]
    for axis, home in zip(axes[0], homes):
        entry = data[home]
        for k, name in enumerate(names):
            value = entry[ORACLE] if name == ORACLE else entry["conditions"][name]
            interval = value.get("interval")
            error = None
            if interval is not None:
                error = [
                    [value["estimate"] - interval["low"]],
                    [interval["high"] - value["estimate"]],
                ]
            axis.bar(k, value["estimate"], color=_COLOURS[name], yerr=error, capsize=3)
        axis.axhline(entry["chance"], color="#555555", ls=":", lw=1.0)
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(
            [n.replace("_", "-") for n in names], rotation=30, fontsize=7
        )
        axis.set_ylim(0.0, 1.0)
        axis.set_title(home)
    axes[0][0].set_ylabel("balanced accuracy")
    figure.tight_layout()
    return figure


def _priors(plt: Any, data: Mapping[str, Any]) -> Any:
    homes = list(data)
    figure, axes = plt.subplots(
        1, len(homes), figsize=(4.0 * len(homes), 3.2), squeeze=False
    )
    for axis, home in zip(axes[0], homes):
        entry = data[home]
        states = list(entry["truth"])
        width = 0.27
        for k, name in enumerate(("truth", "development", "predicted")):
            shares = entry[name]
            if shares is None:
                continue
            axis.bar(
                [i + (k - 1) * width for i in range(len(states))],
                [shares.get(s, 0.0) for s in states],
                width,
                color=_COLOURS[name],
                label=_PRIOR_LABELS[name],
            )
        axis.set_xticks(range(len(states)))
        axis.set_xticklabels(
            [s.replace("_", " ") for s in states], rotation=30, fontsize=7
        )
        axis.set_title(home)
    axes[0][0].set_ylabel("share of scored windows")
    axes[0][-1].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return figure


def _rates(plt: Any, data: Mapping[str, Any]) -> Any:
    homes = list(data)
    rows = max(sum(len(states) for states in data[h].values()) for h in homes)
    figure, axes = plt.subplots(
        1,
        len(homes),
        figsize=(4.4 * len(homes), max(3.4, 0.3 * rows + 1.0)),
        squeeze=False,
    )
    for axis, home in zip(axes[0], homes):
        labels, here, casas = [], [], []
        for channel, states in data[home].items():
            for state, v in states.items():
                labels.append(f"{channel}, {state}")
                here.append(1.0 - v["silence"])
                casas.append(1.0 - v["casas_silence"])
        positions = range(len(labels))
        axis.scatter(
            here, positions, color=_COLOURS["zero_shot"], label="here", zorder=3
        )
        axis.scatter(
            casas, positions, color=_COLOURS["declared"], label="CASAS", zorder=3
        )
        for y, a, b in zip(positions, here, casas):
            axis.plot([a, b], [y, y], color="#cccccc", zorder=1)
        axis.set_yticks(list(positions))
        axis.set_yticklabels(labels, fontsize=7)
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("share of windows with activity")
        axis.set_title(home)
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
    drawers = {"metrics": _metrics, "priors": _priors, "rates": _rates}
    paths: dict[str, Path] = {}
    with plt.rc_context(
        {
            "svg.hashsalt": "phase5-external-results",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        for name in FIGURES:
            figure = drawers[name](plt, data[name])
            path = out_dir / f"phase5-external-{name}.svg"
            figure.savefig(
                path,
                format="svg",
                metadata={
                    "Date": None,
                    "Creator": None,
                    "Title": f"Phase 5 external results: {name}",
                    "Source": source,
                    "Description": f"data sha256 {data_sha256(data[name])}",
                },
            )
            plt.close(figure)
            paths[name] = path
    return paths
