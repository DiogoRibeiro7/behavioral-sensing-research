"""Figures for the silent-home results, drawn from the records alone.

:func:`figure_data` copies what each figure plots from the record of the
protocol's test, and :func:`tihm_figure_data` from the record of the
description on TIHM. :func:`draw_figures` draws them. Each SVG carries the
SHA-256 of the data it plots in its metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .external_figures import data_sha256
from .silent_home_protocol import (
    COMMON,
    FLEET_ARMS,
    OFF,
    OUTAGE,
    RESULT_SCHEMA,
    STABLE,
    TIHM_SCHEMA,
)

FIGURES = ("timeline", "start-hour", "fleet")
TIHM_FIGURES = ("tihm",)
PREFIX = "silent-home"

_COLOURS = {
    "off": "#D55E00",
    "primary": "#0072B2",
    "other": "#56B4E9",
    "stable": "#999999",
    "threshold": "#555555",
}
_ARM_COLOURS = {STABLE: "#999999", OUTAGE: "#E69F00", COMMON: "#0072B2"}
_ARM_LABELS = {
    STABLE: "nothing injected",
    OUTAGE: "each home's own outage",
    COMMON: "one outage in every home",
}
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct")


def _order(configuration: Mapping[str, Any]) -> list[str]:
    """The rule off, the primary horizon, then the others."""
    rule = configuration["the_rule"]
    others = sorted(c for c in rule["conditions"] if c not in (OFF, rule["primary"]))
    return [OFF, rule["primary"], *others]


def figure_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What each figure of the test plots, copied from its record."""
    results, configuration = payload["results"], payload["configuration"]
    if results.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    order = _order(configuration)
    primary, homes = configuration["the_rule"]["primary"], results["homes"]
    timeline = results["timeline"]

    hours: dict[int, dict[str, Any]] = {}
    for row in results["per_home"].values():
        entry = hours.setdefault(
            int(row["outage_hour"]), {"homes": 0, **{c: 0 for c in (OFF, primary)}}
        )
        entry["homes"] += 1
        for condition in (OFF, primary):
            entry[condition] += (
                row[f"{OUTAGE}/{condition}/alerts_in_the_outage_window"]
                - row[f"{STABLE}/{condition}/alerts_in_the_outage_window"]
            )
    fleet = results["fleet"]
    return {
        "timeline": {
            "homes": homes,
            "days": timeline["days_since_the_outage_began"],
            "conditions": order,
            "outage": {c: timeline["alerts"][c][OUTAGE] for c in order},
            "stable": timeline["alerts"][OFF][STABLE],
        },
        "start-hour": {
            "homes": homes,
            "primary": primary,
            "primary_hours": configuration["fleet"]["horizon_hours"],
            "hours": {str(hour): hours[hour] for hour in sorted(hours)},
        },
        "fleet": {
            "homes": homes,
            "fraction": fleet["config"]["fraction"],
            "min_homes": fleet["config"]["min_homes"],
            "outage": fleet[COMMON]["outage"],
            "arms": {
                arm: {
                    key: fleet[arm][key]
                    for key in (
                        "first_assessment",
                        "hours_between_assessments",
                        "share_silent",
                    )
                }
                for arm in FLEET_ARMS
            },
        },
    }


def tihm_figure_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What the figure of the description plots, copied from its record."""
    results, configuration = payload["results"], payload["configuration"]
    if results.get("result_schema") != TIHM_SCHEMA:
        raise ValueError(f"not a {TIHM_SCHEMA} record")
    fleet = results["fleet"]
    order = _order(configuration)[:2]
    return {
        "tihm": {
            "fraction": fleet["config"]["fraction"],
            "min_homes": fleet["config"]["min_homes"],
            "horizon_hours": fleet["config"]["horizon_hours"],
            "first_assessment": fleet["first_assessment"],
            "hours_between_assessments": fleet["hours_between_assessments"],
            "share_silent": fleet["share_silent"],
            "stretches": [
                {
                    key: stretch[key]
                    for key in ("detected", "until", "homes", "monitored")
                }
                for stretch in fleet["stretches"]
            ],
            "conditions": order,
            "alerts_by_day": {
                condition: results["conditions"][condition]["behavioural_alerts"][
                    "by_day_summarised"
                ]
                for condition in order
            },
            # What the lower panel does not draw: the record has no silence
            # alert or refused day by date, so their totals are written on it.
            "totals": {
                condition: {
                    "behavioural_alerts": results["conditions"][condition][
                        "behavioural_alerts"
                    ]["all"],
                    "silence_alerts": results["conditions"][condition][
                        "silence_alerts"
                    ]["all"],
                    "days_refused": results["conditions"][condition][
                        "days_refused_because_of_the_rule"
                    ]["all"],
                    "monitored_days": results["conditions"][condition][
                        "monitored_days"
                    ],
                }
                for condition in order
            },
        }
    }


def _timeline(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axis = plt.subplots(figsize=(6.8, 2.9))
    homes, days = data["homes"], data["days"]
    styles = (("off", "-", 1.6), ("primary", "-", 1.6), ("other", "--", 1.2))
    for condition, (colour, style, width) in zip(data["conditions"], styles):
        axis.plot(
            days,
            [count / homes for count in data["outage"][condition]],
            color=_COLOURS[colour],
            ls=style,
            lw=width,
            marker="o",
            ms=2.5,
            label=f"outage, rule {'off' if condition == OFF else 'on, ' + condition}",
        )
    axis.plot(
        days,
        [count / homes for count in data["stable"]],
        color=_COLOURS["stable"],
        lw=1.0,
        label="nothing injected, rule off",
    )
    axis.set_xlabel(
        "whole days since the home's outage began; the window's last day is half a day"
    )
    axis.set_ylabel("behavioural alerts\nper home, per day")
    axis.set_ylim(bottom=0)
    axis.set_xlim(-0.5, len(days) - 0.5)
    axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return figure


def _start_hour(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axis = plt.subplots(figsize=(6.8, 2.9))
    hours = {int(hour): entry for hour, entry in data["hours"].items()}
    primary, width = data["primary"], 0.4
    for offset, condition, colour in ((-0.2, OFF, "off"), (0.2, primary, "primary")):
        axis.bar(
            [hour + offset for hour in hours],
            [entry[condition] / entry["homes"] for entry in hours.values()],
            width=width,
            color=_COLOURS[colour],
            label=f"rule {'off' if condition == OFF else 'on, ' + condition}",
        )
    for hour in range(24):
        axis.annotate(
            str(hours[hour]["homes"]) if hour in hours else "0",
            xy=(hour, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=6,
            color="#777777",
            annotation_clip=False,
        )
    # From this hour on an outage is declared late: the horizon has not passed
    # when the day it begins on is closed.
    edge = 24.0 - float(data["primary_hours"])
    axis.axvline(edge - 0.5, color=_COLOURS["threshold"], ls=":", lw=1.0)
    axis.annotate(
        "declared late from here",
        xy=(edge - 0.5, 0.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=6,
        color=_COLOURS["threshold"],
    )
    axis.set_xticks(range(0, 24, 2))
    axis.set_xlim(-0.7, 23.7)
    axis.set_xlabel("local hour the outage begins at; above, the homes at each hour")
    axis.set_ylabel("mean alerts per home in the window,\nminus the stable arm's")
    axis.axhline(0.0, color="#333333", lw=0.6)
    # Room above the bars for the legend, clear of every one of them.
    low, high = axis.get_ylim()
    axis.set_ylim(low, high + 0.22 * (high - low))
    axis.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    figure.tight_layout()
    return figure


def _days(first: str, hours: float, count: int) -> list[float]:
    """The assessments' positions, in days since the first of them."""
    del first
    return [hours * k / 24.0 for k in range(count)]


def _day_ticks(axis: Any, first: str, days: float, every: int) -> None:
    """Label the axis with dates, one every *every* days from a round day."""
    begin = datetime.fromisoformat(first)
    midnight = datetime(begin.year, begin.month, begin.day, tzinfo=begin.tzinfo)
    ticks, labels = [], []
    day = midnight + timedelta(days=1)
    while (day - begin) / timedelta(days=1) <= days:
        if (day.date() - date(day.year, 1, 1)).days % every == 0:
            ticks.append((day - begin) / timedelta(days=1))
            labels.append(f"{day.day} {_MONTHS[day.month - 1]}")
        day += timedelta(days=1)
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)


def _position(moment: str, first: str) -> float:
    return (datetime.fromisoformat(moment) - datetime.fromisoformat(first)) / timedelta(
        days=1
    )


def _fleet(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axis = plt.subplots(figsize=(6.8, 2.9))
    first, span = data["arms"][STABLE]["first_assessment"], 0.0
    for arm in FLEET_ARMS:
        entry = data["arms"][arm]
        offset = _position(entry["first_assessment"], first)
        days = [
            offset + day
            for day in _days(
                entry["first_assessment"],
                entry["hours_between_assessments"],
                len(entry["share_silent"]),
            )
        ]
        axis.plot(
            days,
            entry["share_silent"],
            color=_ARM_COLOURS[arm],
            lw=1.3,
            label=_ARM_LABELS[arm],
        )
        span = max(span, days[-1] if days else 0.0)
    axis.axvspan(
        _position(data["outage"]["begin"], first),
        _position(data["outage"]["end"], first),
        color="#DDDDDD",
        lw=0,
        zorder=0,
    )
    axis.axhline(data["fraction"], color=_COLOURS["threshold"], ls=":", lw=1.0)
    axis.annotate(
        "a common cause at or above this share,\n"
        f"of at least {data['min_homes']} homes",
        xy=(0.01, data["fraction"]),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="left",
        fontsize=7,
        color=_COLOURS["threshold"],
    )
    axis.annotate(
        "the outage\nevery home shares",
        xy=(_position(data["outage"]["end"], first), 0.9),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=7,
        color="#555555",
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("share of monitored homes\nsilent for the horizon")
    axis.set_xlim(0, span)
    _day_ticks(axis, first, span, 14)
    axis.legend(frameon=False, fontsize=7, loc="upper left")
    figure.tight_layout()
    return figure


def _tihm(plt: Any, data: Mapping[str, Any]) -> Any:
    figure, axes = plt.subplots(
        2, 1, figsize=(7.0, 3.8), sharex=True, gridspec_kw={"height_ratios": (2, 3)}
    )
    first = data["first_assessment"]
    days = _days(first, data["hours_between_assessments"], len(data["share_silent"]))
    axes[0].plot(days, data["share_silent"], color=_ARM_COLOURS[COMMON], lw=1.0)
    axes[0].axhline(data["fraction"], color=_COLOURS["threshold"], ls=":", lw=1.0)
    axes[0].annotate(
        f"a common cause at or above this share, of at least {data['min_homes']} "
        "homes",
        xy=(0.5, data["fraction"]),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        fontsize=6,
        color=_COLOURS["threshold"],
    )
    for stretch in data["stretches"]:
        begin, end = (
            _position(stretch["detected"], first),
            _position(stretch["until"], first),
        )
        for axis in axes:
            axis.axvspan(begin, end, color="#DDDDDD", lw=0, zorder=0)
        # How many homes a band is: a full band of three is not one of 47.
        late = begin > 0.5 * (days[-1] if days else 0.0)
        axes[0].annotate(
            f"{stretch['homes']} of {stretch['monitored']} homes",
            xy=(begin if late else end, 0.82),
            xytext=(-3 if late else 3, 0),
            textcoords="offset points",
            ha="right" if late else "left",
            fontsize=6,
            color="#555555",
        )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel(
        f"share of monitored\nhomes silent {data['horizon_hours']:g} h", fontsize=8
    )
    begin = datetime.fromisoformat(first)
    width = 0.4
    for offset, condition, colour in zip(
        (-0.2, 0.2), data["conditions"], ("off", "primary")
    ):
        counted = data["alerts_by_day"][condition]
        axes[1].bar(
            [
                # The middle of the local day the summary is of.
                (
                    datetime.fromisoformat(day).replace(tzinfo=begin.tzinfo)
                    + timedelta(hours=12)
                    - begin
                )
                / timedelta(days=1)
                + offset
                for day in counted
            ],
            list(counted.values()),
            width=width,
            color=_COLOURS[colour],
            label=f"rule {'off' if condition == OFF else 'on, ' + condition}",
        )
    axes[1].set_ylabel("behavioural alerts,\nby day summarised")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.1, 1))
    off, on = (data["totals"][condition] for condition in data["conditions"])
    axes[1].annotate(
        "not drawn, since the record has no date for them:\n"
        f"alerts about silence, {off['silence_alerts']:,} with the rule off and "
        f"{on['silence_alerts']:,} with it on\n"
        f"days refused, {off['days_refused']:,} and {on['days_refused']:,} of "
        f"{on['monitored_days']:,}\n"
        f"behavioural alerts in all, {off['behavioural_alerts']:,} and "
        f"{on['behavioural_alerts']:,}",
        xy=(0.1, 0.62),
        xycoords="axes fraction",
        va="top",
        fontsize=6,
        color="#555555",
    )
    span = days[-1] if days else 0.0
    axes[1].set_xlim(0, span)
    _day_ticks(axes[1], first, span, 14)
    figure.tight_layout()
    return figure


def draw_figures(
    payload: Mapping[str, Any],
    out_dir: Path,
    tihm: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Draw every figure as SVG into *out_dir*; returns each figure's path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = figure_data(payload)
    drawers = {"timeline": _timeline, "start-hour": _start_hour, "fleet": _fleet}
    jobs: list[tuple[str, Any, Any, Mapping[str, Any]]] = [
        (name, drawers[name], data[name], payload) for name in FIGURES
    ]
    if tihm is not None:
        jobs.append(("tihm", _tihm, tihm_figure_data(tihm)["tihm"], tihm))
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
                    "a description, not a test"
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
                    "Title": "The silent-home rule"
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
