"""Figures for the threshold-calibration pages, drawn from the records alone.

:func:`null_figure_data` copies what each figure of the measurement on
synthetic days plots from its record, and :func:`draw_null_figures` draws
them. :func:`figure_data` and :func:`tihm_figure_data` do the same for the
record of the protocol's test and for the record of the description on TIHM,
and :func:`draw_figures` draws those. Each SVG carries the SHA-256 of the data
it plots in its metadata.
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


# ----------------------------------------------------------------------------
# The results
# ----------------------------------------------------------------------------
FIGURES = ("weeks", "curves", "tail")
TIHM_FIGURES = ("tihm",)
PREFIX = "threshold-calibration"
RESULT_SCHEMA = "threshold-calibration/1"
TIHM_SCHEMA = "threshold-calibration-tihm/1"

_CHANGES = {
    "change": "the step change",
    "small_change": "the smaller step",
    "gradual_change": "the gradual change",
}
_TESTED_STYLES = (
    ("#D55E00", "-"),
    ("#0072B2", ":"),
    ("#0072B2", "-"),
    ("#56B4E9", "-"),
)


def figure_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What each figure of the test plots, copied from its record."""
    results, configuration = payload["results"], payload["configuration"]
    if results.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    order = results["reported"]
    curves = {
        reference: dict(
            sorted(
                results["curves"][reference].items(), key=lambda item: float(item[0])
            )
        )
        for reference in REFERENCES
        if reference in results["curves"]
    }
    tail = {
        reference: results["calibration"]["tail"][reference]
        for reference in REFERENCES
        if reference in results["calibration"]["tail"]
    }
    arms = [arm for arm in configuration["simulator"]["arm_order"] if arm in _CHANGES]
    # The conditions the comparison is about, by reference: the default as it
    # ships, and the calibrated reference at the multiple nearest the match.
    tested = [results["shown"][role] for role in ("default", "same_false_alerts")]
    found = results["matched"]
    marked = {
        reference: sorted(
            {
                condition.partition("@")[2]
                for condition in tested
                if condition.partition("@")[0] == reference
            },
            key=float,
        )
        for reference in curves
    }
    thresholds = sorted(next(iter(tail.values())), key=float) if tail else []
    return {
        "weeks": {
            "homes": results["homes"],
            "conditions": order,
            "alerts": {
                c: results["stable"][c]["by_week_of_the_day_closed"] for c in order
            },
        },
        "curves": {
            "arms": arms,
            "marked": marked,
            "match": {
                "false_alerts": found["default"]["false_alerts_per_home"],
                "excess_detection": {
                    arm: found["excess_detection"][arm]["calibrated_at_the_match"]
                    for arm in arms
                },
            },
            "scales": {reference: list(points) for reference, points in curves.items()},
            "false_alerts": {
                reference: [
                    entry["false_alerts_per_home"]["estimate"]
                    for entry in points.values()
                ]
                for reference, points in curves.items()
            },
            "excess_detection": {
                arm: {
                    reference: [
                        entry[arm]["excess_detection"]["estimate"]
                        for entry in points.values()
                    ]
                    for reference, points in curves.items()
                }
                for arm in arms
            },
        },
        "tail": {
            "feature": results["calibration"]["feature"],
            "thresholds": [float(t) for t in thresholds],
            "stated": [tail[next(iter(tail))][t]["stated"] for t in thresholds],
            "share": {
                reference: [tail[reference][t]["estimate"] for t in thresholds]
                for reference in tail
            },
            "low": {
                reference: [tail[reference][t]["interval"]["low"] for t in thresholds]
                for reference in tail
            },
            "high": {
                reference: [tail[reference][t]["interval"]["high"] for t in thresholds]
                for reference in tail
            },
        },
    }


def tihm_figure_data(tihm: Mapping[str, Any]) -> dict[str, Any]:
    """What the figure of the description on TIHM plots, copied from its record."""
    results = tihm["results"]
    if results.get("result_schema") != TIHM_SCHEMA:
        raise ValueError(f"not a {TIHM_SCHEMA} record")
    feature = "sleeping_hours"
    rules = {
        rule: results["rules"][rule]
        for rule in tihm["configuration"]["tihm"]["silent_home_rule"]
    }
    first = next(iter(rules.values()))["tail"]
    thresholds = sorted(next(iter(first.values()))[feature], key=float)
    return {
        "tihm": {
            "feature": feature,
            "homes": results["homes"],
            "thresholds": [float(t) for t in thresholds],
            "stated": [
                next(iter(first.values()))[feature][t]["stated"] for t in thresholds
            ],
            "share": {
                rule: {
                    reference: [
                        entry["tail"][reference][feature][t]["share"]
                        for t in thresholds
                    ]
                    for reference in REFERENCES
                    if reference in entry["tail"]
                }
                for rule, entry in rules.items()
            },
        }
    }


def _weeks(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axis = plt.subplots(figsize=(6.6, 3.0))
    for condition, (colour, style) in zip(data["conditions"], _TESTED_STYLES):
        counts = data["alerts"][condition]
        axis.plot(
            range(1, len(counts) + 1),
            counts,
            color=colour,
            ls=style,
            marker="o",
            ms=3,
            lw=1.2,
            label=condition,
        )
    axis.set_xticks(range(1, len(data["alerts"][data["conditions"][0]]) + 1))
    axis.set_xlabel("week of the day whose close raised the alert")
    axis.set_ylabel(f"false alerts in\n{data['homes']} stable homes")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return figure


def _curves(plt: Any, data: Mapping[str, Any]) -> Any:
    arms = data["arms"]
    figure, axes = plt.subplots(
        1, len(arms), figsize=(3.3 * len(arms), 3.3), sharey=True, squeeze=False
    )
    for axis, arm in zip(axes[0], arms):
        for reference, false in data["false_alerts"].items():
            found = data["excess_detection"][arm][reference]
            colour = _COLOURS.get(reference, "#555555")
            axis.plot(
                false,
                [100.0 * value for value in found],
                marker="o",
                ms=3,
                lw=1.2,
                color=colour,
                label=f"{reference} reference",
            )
            for scale in data["marked"].get(reference, []):
                place = data["scales"][reference].index(scale)
                axis.plot(
                    [false[place]],
                    [100.0 * found[place]],
                    marker="o",
                    ms=7,
                    mfc="none",
                    mec=colour,
                    lw=0,
                )
                axis.annotate(
                    scale,
                    xy=(false[place], 100.0 * found[place]),
                    xytext=(5, -9),
                    textcoords="offset points",
                    fontsize=7,
                    color=colour,
                )
        matched = data["match"]["excess_detection"][arm]
        axis.axvline(data["match"]["false_alerts"], color="#888888", lw=0.6, ls=":")
        if matched is not None:
            axis.plot(
                [data["match"]["false_alerts"]],
                [100.0 * matched],
                marker="x",
                ms=6,
                lw=0,
                color=_COLOURS.get("calibrated", "#555555"),
                label="calibrated, at the default's false alerts",
            )
        axis.set_title(_CHANGES[arm], fontsize=9)
        axis.set_xlabel("false alerts per home")
        # The comparison is about where the default ships; the most sensitive
        # multiples raise many times as many false alerts, and the tables
        # hold them.
        axis.set_xlim(-0.05, max(4.0 * data["match"]["false_alerts"], 1.0))
        axis.axhline(0.0, color="#BBBBBB", lw=0.6)
    axes[0][0].set_ylabel("excess detection (% of homes)")
    axes[0][0].legend(frameon=False, fontsize=7, loc="lower right")
    figure.tight_layout()
    return figure


def _tail_axis(axis: Any, thresholds: Any, stated: Any) -> None:
    axis.plot(
        thresholds,
        [100.0 * value for value in stated],
        color=_COLOURS["stated"],
        ls=":",
        lw=1.0,
        label="what the threshold states",
    )
    axis.set_yscale("log")
    axis.set_yticks([0.1, 0.3, 1, 3, 10, 30])
    axis.set_yticklabels(["0.1%", "0.3%", "1%", "3%", "10%", "30%"])
    axis.set_ylim(0.05, 60)
    axis.set_xticks(thresholds)
    axis.set_xlabel("deviation threshold")


def _positive(values: Any) -> list[float]:
    """Shares in percent, with nothing where a share is zero or missing."""
    return [
        100.0 * value if value is not None and value > 0 else float("nan")
        for value in values
    ]


def _tail(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axis = plt.subplots(figsize=(5.6, 3.2))
    thresholds = data["thresholds"]
    _tail_axis(axis, thresholds, data["stated"])
    for reference, share in data["share"].items():
        colour = _COLOURS.get(reference, "#555555")
        axis.plot(
            thresholds,
            _positive(share),
            marker="o",
            ms=3,
            lw=1.2,
            color=colour,
            label=f"{reference} reference",
        )
        axis.fill_between(
            thresholds,
            _positive(data["low"][reference]),
            _positive(data["high"][reference]),
            color=colour,
            alpha=0.15,
            lw=0,
        )
    axis.set_ylabel(f"days of {data['feature']}\nat or past the threshold")
    axis.legend(frameon=False, fontsize=7, loc="lower left")
    figure.tight_layout()
    return figure


def _tihm_tail(plt: Any, data: Mapping[str, Any]) -> Any:
    rules = list(data["share"])
    figure, axes = plt.subplots(
        1, len(rules), figsize=(3.6 * len(rules), 3.2), sharey=True, squeeze=False
    )
    thresholds = data["thresholds"]
    for axis, rule in zip(axes[0], rules):
        _tail_axis(axis, thresholds, data["stated"])
        for reference, share in data["share"][rule].items():
            axis.plot(
                thresholds,
                _positive(share),
                marker="o",
                ms=3,
                lw=1.2,
                color=_COLOURS.get(reference, "#555555"),
                label=f"{reference} reference",
            )
        axis.set_title(
            "silent-home rule off" if rule == "off" else f"silent-home rule {rule}",
            fontsize=9,
        )
    axes[0][0].set_ylabel(f"days of {data['feature']}\nat or past the threshold")
    axes[0][0].legend(frameon=False, fontsize=7, loc="lower left")
    figure.tight_layout()
    return figure


def draw_figures(
    payload: Mapping[str, Any],
    out_dir: Path,
    tihm: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Draw every figure of the results as SVG into *out_dir*; returns their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = figure_data(payload)
    drawers = {"weeks": _weeks, "curves": _curves, "tail": _tail}
    jobs: list[tuple[str, Any, Any, Mapping[str, Any]]] = [
        (name, drawers[name], data[name], payload) for name in FIGURES
    ]
    if tihm is not None:
        jobs.append(("tihm", _tihm_tail, tihm_figure_data(tihm)["tihm"], tihm))
    paths: dict[str, Path] = {}
    with plt.rc_context(
        {
            "svg.hashsalt": PREFIX,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        for name, drawer, plotted, record in jobs:
            figure = drawer(plt, plotted)
            described = name in TIHM_FIGURES
            figure.text(
                0.995,
                0.005,
                (
                    f"{plotted['homes']} TIHM homes; a description, not a test"
                    if described
                    else f"{payload['results']['homes']} paired simulated homes"
                ),
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
                    "Title": "Threshold calibration"
                    + (", described on TIHM" if described else "")
                    + f": {name}",
                    "Source": f"record {record['experiment']}, "
                    f"recorded {record['recorded_at']}",
                    "Description": f"data sha256 {data_sha256(plotted)}",
                },
            )
            plt.close(figure)
            paths[name] = path
    return paths
