"""Figures for the TIHM alert-burden results, drawn from the records alone.

:func:`figure_data` copies what each figure plots from the record of the
protocol's run, and :func:`post_hoc_figure_data` from the record of the
descriptions made after it. :func:`draw_figures` draws them. Each SVG carries
the SHA-256 of the data it plots in its metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .external_figures import data_sha256
from .tihm_experiment import NO_LABEL
from .tihm_post_hoc import POST_HOC_SCHEMA
from .tihm_protocol import RESULT_SCHEMA

FIGURES = ("burden", "references", "profile")
POST_HOC_FIGURES = ("calendar", "replays", "reference-size")
PREFIX = "tihm-alert-burden"

_COLOURS = {
    "pipeline": "#0072B2",
    "simulator": "#E69F00",
    "reference": "#009E73",
    "chance": "#999999",
    "recorded": "#0072B2",
    "skipped": "#56B4E9",
    "shuffled": "#CC79A7",
    "gaussian": "#999999",
    "other": "#D55E00",
}
_SLOT_COLOURS = {
    "08:00": "#56B4E9",
    "12:00": "#D55E00",
    "18:00": "#009E73",
    "other": "#CC79A7",
    NO_LABEL: "#999999",
}
_REPLAYS = ("stationary_gaussian", "shuffled", "silent_days_skipped", "recorded")
_REPLAY_LABELS = {
    "stationary_gaussian": "stationary Gaussian values",
    "shuffled": "recorded values, shuffled",
    "silent_days_skipped": "recorded values, silent days left out",
    "recorded": "recorded values (the run)",
}
_REPLAY_COLOURS = {
    "stationary_gaussian": _COLOURS["gaussian"],
    "shuffled": _COLOURS["shuffled"],
    "silent_days_skipped": _COLOURS["skipped"],
    "recorded": _COLOURS["recorded"],
}
#: The features whose references are not held at the scale floor.
_PLOTTED_FEATURES = ("sleeping_hours", "away_hours")
_FEATURE_COLOURS = {"sleeping_hours": "#0072B2", "away_hours": "#D55E00"}


def figure_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What each figure of the protocol's run plots, copied from its record."""
    results = payload["results"]
    if results.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    burden, references = results["burden"], results["references"]
    return {
        "burden": {
            "B1": burden["B1_per_monitored_day"],
            "B2": burden["B2_simulator_feature_per_monitored_day"],
            "B3": burden["B3_per_evaluable_day"],
            "simulator": burden["simulator_reference"],
        },
        "references": {
            "label_days": references["label_days"],
            "flags": references["flags"],
            "random_expected_caught": references["random_expected_caught"],
            "pipeline": references["pipeline_deviating_days"]["share_of_label_days"],
            "event_count": references["event_count"]["share_of_label_days"],
            "label_history": references["label_history"]["share_of_label_days"],
        },
        "profile": {
            group: {"days": entry["days"], "hourly_mean_z": entry["hourly_mean_z"]}
            for group, entry in results["labels"]["hourly_profile"].items()
        },
    }


def post_hoc_figure_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What each post hoc figure plots, copied from the post hoc record."""
    results = payload["results"]
    if results.get("result_schema") != POST_HOC_SCHEMA:
        raise ValueError(f"not a {POST_HOC_SCHEMA} record")
    replays, size = results["replays"], results["reference_size"]
    return {
        "calendar": {
            "daily": results["calendar"]["daily"],
            "most_silent_date": results["calendar"]["most_silent_date"],
        },
        "replays": {
            name: {
                key: replays[name][key]
                for key in ("deviating_share", "change_verdicts")
            }
            for name in _REPLAYS
        },
        "reference-size": {
            "threshold": size["threshold"],
            "nominal": size["nominal"],
            "gaussian_null": size["gaussian_null"],
            "mean_and_sd_estimated": size["mean_and_sd_estimated"],
            "features": {
                feature: {
                    count: entry["share_at_threshold"]
                    for count, entry in size["features"][feature][
                        "weekday_aware_by_size"
                    ].items()
                }
                for feature in _PLOTTED_FEATURES
                if feature in size["features"]
            },
        },
    }


def _errors(entry: Mapping[str, Any]) -> list[list[float]] | None:
    interval = entry.get("interval")
    if interval is None:
        return None
    return [
        [entry["estimate"] - interval["low"]],
        [interval["high"] - entry["estimate"]],
    ]


def _burden(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axis = plt.subplots(figsize=(6.4, 2.6))
    feature = data["simulator"]["feature"]
    rows = [
        ("B3", "all features,\nper evaluable day"),
        ("B1", "all features,\nper monitored day"),
        ("B2", f"{feature},\nper monitored day"),
    ]
    for y, (key, _) in enumerate(rows):
        entry = data[key]
        axis.errorbar(
            entry["estimate"],
            y,
            xerr=_errors(entry),
            fmt="o",
            color=_COLOURS["pipeline"],
            capsize=3,
            label="TIHM homes" if y == 0 else None,
        )
    for arm, marker in (("stable_arm", "D"), ("changed_arm", "s")):
        axis.plot(
            data["simulator"][arm],
            len(rows) - 1,
            marker,
            color=_COLOURS["simulator"],
            label=f"simulator, {arm.replace('_', ' ')}",
        )
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([label for _, label in rows])
    axis.set_ylim(-0.6, len(rows) - 0.4)
    axis.set_xlim(left=0.0)
    axis.set_xlabel("behavioural alerts per person-day")
    axis.legend(frameon=False, fontsize=7, loc="upper right")
    figure.tight_layout()
    return figure


def _references(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axis = plt.subplots(figsize=(6.4, 2.6))
    chance = data["random_expected_caught"] / data["label_days"]
    rows = [
        ("pipeline", "pipeline:\ndeviating days", _COLOURS["pipeline"]),
        ("event_count", "event count\nagainst earlier days", _COLOURS["reference"]),
        ("label_history", "label history,\nno sensor", _COLOURS["reference"]),
    ]
    for y, (key, _, colour) in enumerate(rows):
        entry = data[key]
        axis.barh(y, entry["estimate"], color=colour, xerr=_errors(entry), capsize=3)
    axis.axvline(chance, color=_COLOURS["chance"], ls=":", lw=1.2)
    axis.text(
        chance,
        len(rows) - 0.45,
        " flagged at random",
        color="#555555",
        fontsize=7,
        va="center",
    )
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([label for _, label, _ in rows])
    axis.set_ylim(-0.6, len(rows) - 0.2)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("share of label days caught")
    figure.tight_layout()
    return figure


def _profile(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axis = plt.subplots(figsize=(6.4, 3.0))
    for group, entry in data.items():
        values = [
            float("nan") if value is None else value for value in entry["hourly_mean_z"]
        ]
        label = (
            f"no label ({entry['days']:,} days)"
            if group == NO_LABEL
            else f"first label at {group} ({entry['days']:,} days)"
        )
        axis.plot(
            range(24),
            values,
            color=_SLOT_COLOURS.get(group, "#000000"),
            lw=1.0 if group == NO_LABEL else 1.6,
            label=label,
        )
        if group[:2].isdigit():
            axis.axvline(
                int(group[:2]), color=_SLOT_COLOURS[group], ls=":", lw=0.9, zorder=0
            )
    axis.set_xticks(range(0, 24, 3))
    axis.set_xlim(0, 23)
    axis.set_xlabel("hour of the day")
    axis.set_ylabel("sensor events, mean z\nagainst the household's hour")
    axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return figure


_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def _calendar(plt: Any, data: Mapping[str, Any]) -> Any:
    daily = data["daily"]
    positions = range(len(daily))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 3.6),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 2)},
    )
    axes[0].plot(
        positions,
        [row["events"] for row in daily],
        color=_COLOURS["pipeline"],
        lw=1.2,
    )
    axes[0].set_ylabel("sensor events,\nall households")
    axes[0].set_ylim(bottom=0)
    axes[1].bar(
        positions,
        [row["behavioural_alerts"] for row in daily],
        color=_COLOURS["other"],
        width=0.9,
    )
    axes[1].set_ylabel("behavioural\nalerts")
    peak = data["most_silent_date"]
    if peak is not None and peak["silent_households"]:
        at = next(k for k, row in enumerate(daily) if row["date"] == peak["date"])
        for axis in axes:
            axis.axvline(at, color="#555555", ls=":", lw=1.0, zorder=0)
        axes[0].annotate(
            f"{peak['silent_households']} of {peak['monitored_households']} "
            "households silent",
            xy=(at, 0),
            xytext=(-8, 26),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color="#555555",
        )
    ticks = [
        k
        for k, row in enumerate(daily)
        if row["date"].endswith("-01") or row["date"].endswith("-15")
    ]
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(
        [
            f"{int(daily[k]['date'][8:10])} {_MONTHS[int(daily[k]['date'][5:7]) - 1]}"
            for k in ticks
        ]
    )
    axes[1].set_xlim(-0.5, len(daily) - 0.5)
    figure.tight_layout()
    return figure


def _point(entry: Any) -> tuple[float, list[list[float]] | None]:
    if isinstance(entry, Mapping):
        return entry["mean"], [
            [entry["mean"] - entry["low"]],
            [entry["high"] - entry["mean"]],
        ]
    return float(entry), None


def _replays(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), sharey=True)
    for y, name in enumerate(_REPLAYS):
        value, error = _point(data[name]["deviating_share"])
        axes[0].barh(y, value, color=_REPLAY_COLOURS[name], xerr=error, capsize=3)
        total, error = _point(data[name]["change_verdicts"])
        axes[1].barh(y, total, color=_REPLAY_COLOURS[name], xerr=error, capsize=3)
    axes[0].set_yticks(range(len(_REPLAYS)))
    axes[0].set_yticklabels([_REPLAY_LABELS[n].replace(", ", ",\n") for n in _REPLAYS])
    axes[0].set_xlabel("deviating share of evaluable days")
    axes[1].set_xlabel("change verdicts")
    figure.tight_layout()
    return figure


def _reference_size(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axis = plt.subplots(figsize=(6.4, 3.0))
    sizes = sorted(int(size) for size in data["gaussian_null"])
    axis.plot(
        sizes,
        [data["gaussian_null"][str(size)] for size in sizes],
        "-o",
        ms=3,
        color=_COLOURS["gaussian"],
        label="Gaussian value past a median and MAD",
    )
    estimated = sorted(int(size) for size in data["mean_and_sd_estimated"])
    axis.plot(
        estimated,
        [data["mean_and_sd_estimated"][str(size)] for size in estimated],
        "--",
        lw=1.0,
        color=_COLOURS["gaussian"],
        label="Gaussian value past a mean and SD",
    )
    for feature, shares in data["features"].items():
        counts = sorted(int(count) for count in shares)
        axis.plot(
            counts,
            [shares[str(count)] for count in counts],
            "o",
            ms=4,
            color=_FEATURE_COLOURS[feature],
            label=f"{feature}, same-weekday reference",
        )
    axis.axhline(data["nominal"], color="#555555", ls=":", lw=1.0)
    axis.text(
        sizes[0],
        data["nominal"],
        "known mean and SD",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    axis.set_ylim(0.0, 0.32)
    axis.set_yticks([0.0, 0.1, 0.2, 0.3])
    axis.set_yticklabels(["0", "10%", "20%", "30%"])
    axis.set_xlabel("days the reference was estimated from")
    axis.set_ylabel(f"share of days at {data['threshold']:g} robust SD")
    axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return figure


def draw_figures(
    payload: Mapping[str, Any],
    out_dir: Path,
    post_hoc: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Draw every figure as SVG into *out_dir*; returns each figure's path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Any, Any, Mapping[str, Any]]] = []
    data = figure_data(payload)
    drawers = {"burden": _burden, "references": _references, "profile": _profile}
    jobs += [(name, drawers[name], data[name], payload) for name in FIGURES]
    if post_hoc is not None:
        later = post_hoc_figure_data(post_hoc)
        extra = {
            "calendar": _calendar,
            "replays": _replays,
            "reference-size": _reference_size,
        }
        jobs += [
            (name, extra[name], later[name], post_hoc) for name in POST_HOC_FIGURES
        ]
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
            later = name in POST_HOC_FIGURES
            if later:
                figure.text(
                    0.995,
                    0.005,
                    "post hoc",
                    ha="right",
                    va="bottom",
                    fontsize=6,
                    color="#777777",
                )
            path = out_dir / f"{PREFIX}-{name}.svg"
            figure.savefig(
                path,
                format="svg",
                metadata={
                    "Date": None,
                    "Creator": None,
                    "Title": "TIHM alert burden"
                    + (", post hoc" if later else "")
                    + f": {name}",
                    "Source": f"record {record['experiment']}, "
                    f"recorded {record['recorded_at']}",
                    "Description": f"data sha256 {data_sha256(plotted)}",
                },
            )
            plt.close(figure)
            paths[name] = path
    return paths
