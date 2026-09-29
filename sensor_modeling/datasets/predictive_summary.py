"""A concise Markdown summary of a hurdle predictive-check record, from it alone.

:func:`render_summary` reads nothing but the record as written, so the summary
can always be regenerated from the artifact and checked against it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .gap_summary import _number, _table
from .predictive_checks import (
    CONCLUSION_STATISTICS,
    CRITERIA,
    OWN,
    POPULATION,
    RESULT_SCHEMA,
    STATISTICS,
)

_SHAPE = ("active_dispersion", "variance", "q90", "q99", "tail_excess")
_RUNS = ("quiet_runs", "bursts")
_POPULATION = ("silence", "mean", "active_dispersion", "quiet_runs")
_VERDICTS = ("consistent", "small", "above", "below")


def _ratio(summary: Mapping[str, Any]) -> str:
    """A household mean log ratio as a ratio, with its interval, homes and verdict."""
    if summary.get("verdict") == "insufficient" or "mean" not in summary:
        return f"n/a ({summary.get('n', 0)} homes)"
    mean = summary["mean"]
    interval = mean["interval"]
    bounds = (
        f" [{_number(math.exp(interval['low']))}, {_number(math.exp(interval['high']))}]"
        if interval
        else ""
    )
    return (
        f"{_number(math.exp(mean['estimate']))}{bounds}, "
        f"{summary['above']} / {summary['below']}, {summary['verdict']}"
    )


def _group_table(
    groups: Mapping[str, Any],
    group: str,
    reference: str,
    statistics: Sequence[str],
    keys: Sequence[str],
) -> list[str]:
    rows = []
    for key in keys:
        cells = [
            (
                _ratio(groups[group][reference][statistic][key])
                if key in groups[group][reference][statistic]
                else "n/a"
            )
            for statistic in statistics
        ]
        if any(cell != "n/a" for cell in cells):
            rows.append([key, *cells])
    return _table([group.capitalize(), *statistics], rows)


def _keys(groups: Mapping[str, Any], group: str) -> list[str]:
    found: set[str] = set()
    for reference in groups[group].values():
        for per_key in reference.values():
            found.update(per_key)
    return sorted(found)


def render_summary(payload: Mapping[str, Any], *, level: int = 1) -> str:
    """A concise Markdown summary of a hurdle predictive-check record.

    Parameters
    ----------
    payload
        The record as written, for example from
        :func:`~sensor_modeling.evaluation.load_record`.
    level
        Heading level of the title. Sections are one level deeper.
    """
    results = payload["results"]
    if results.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    configuration = payload["configuration"]
    environment = payload["environment"]
    bootstrap = configuration["bootstrap"]
    replicates = configuration["replicates"]
    runs = configuration["runs"]
    confidence = round(100 * bootstrap["confidence"])
    title, section = "#" * level, "#" * (level + 1)
    dirty = " (uncommitted changes)" if environment.get("git_dirty") == "true" else ""
    households = results["households"]
    groups = results["groups"]
    states = results["states"]
    common = results["common_states"]
    conclusions = results["conclusions"]
    routing = results["routing"]
    ratio = configuration["minimal_ratio"]
    step = configuration["data"]["counts"]

    lines = [
        f"{title} Posterior predictive checks of the fitted hurdle model: summary",
        "",
        f"Generated from the record `{payload['experiment']}`: protocol "
        f"`{configuration['protocol_sha256'][:12]}`, commit "
        f"`{str(environment.get('git_commit', 'unknown'))[:12]}`{dirty}. "
        f"Status: {results['status']}.",
        "",
        "- Diagnostic only: no model is changed and no state is estimated.",
        f"- {len(households)} households, each examined once. A cell is one "
        "household's labelled windows of one state on one channel, with at least "
        f"{configuration['sufficiency']['min_windows']} windows; the counts are "
        f"{step}.",
        "- Two references: `own`, the hurdle fitted to the cell alone, which tests "
        "the family; and `population`, the fold's population hurdle fitted on the "
        "training households, which inference uses.",
        f"- Each cell is compared with {replicates['count']} replicates of its own "
        f"windows drawn from the reference, with a {round(100 * replicates['band'])}% "
        "band; `own` replicates are refitted.",
        f"- Ratios are observed over predicted. A ratio above one means more than "
        f"the model predicts. The minimal ratio is {ratio:g}. Long quiet runs are "
        f"{runs['quiet_run_windows']} or more silent windows, and bursts "
        f"{runs['burst_windows']} or more active windows, counted as windows in "
        "such runs.",
        "- Households are the unit: each household's value for a group is the mean "
        "of its cells' log ratios, and every interval is a household bootstrap, "
        f"{confidence}%, from {bootstrap['resamples']:,} resamples.",
        "",
        f"{section} Pre-specified conclusions",
        "",
        *_table(
            ["Question", *common, "Conclusion"],
            [
                [
                    {
                        "active_dispersion": "Is the active count under-dispersed?",
                        "quiet_runs": "Are long quiet runs in excess?",
                        "bursts": "Are bursts in excess?",
                    }[key],
                    *(entry["verdicts"][state] for state in common),
                    entry["conclusion"],
                ]
                for key, entry in ((k, conclusions[k]) for k in CONCLUSION_STATISTICS)
            ],
        ),
        f"Systematic: {CRITERIA['systematic']}. Partial: {CRITERIA['partial']}. "
        f"Absent: {CRITERIA['absent']}.",
        "",
        f"Routing: {CRITERIA['routing']}.",
        "",
        (
            f"**Indicated next family: {routing['description']}.**"
            if routing["family"] is not None
            else "**No next model family is indicated by the declared rule.**"
        ),
        "",
        f"{section} The shape of the active count, own reference",
        "",
        "The geometric mean over households of observed over predicted, with "
        "its interval, households above and below one, and the verdict:",
        "",
        *_group_table(groups, "state", OWN, _SHAPE, states),
        f"{section} Runs in time, own reference",
        "",
        "Windows in long quiet runs and in bursts, observed over predicted:",
        "",
        *_group_table(groups, "state", OWN, _RUNS, states),
        f"{section} Against the population model",
        "",
        "The same against the fold's population hurdle, which also carries the "
        "differences between households:",
        "",
        *_group_table(groups, "state", POPULATION, _POPULATION, states),
        f"{section} By channel, room and channel type",
        "",
        "Own reference, active dispersion and runs; population reference, silence:",
        "",
    ]
    for group in ("channel", "room", "modality"):
        lines += _table(
            [
                group.capitalize(),
                "active_dispersion",
                "quiet_runs",
                "bursts",
                "silence (population)",
            ],
            [
                [
                    key,
                    *(
                        (
                            _ratio(groups[group][OWN][s][key])
                            if key in groups[group][OWN][s]
                            else "n/a"
                        )
                        for s in ("active_dispersion", *_RUNS)
                    ),
                    (
                        _ratio(groups[group][POPULATION]["silence"][key])
                        if key in groups[group][POPULATION]["silence"]
                        else "n/a"
                    ),
                ]
                for key in _keys(groups, group)
            ],
        )
    slopes = results["variance_mean"]
    lines += [
        f"{section} Variance and mean",
        "",
        "Each household's slope of log active variance on log active mean across "
        "its cells: about 1 for Poisson-like counts, nearer 2 when the excess "
        "variance grows with the square of the mean. Household means with their "
        "intervals:",
        "",
        *_table(
            ["Slope", "Households", "Mean", "Median"],
            [
                [
                    key,
                    str(summary["n"]),
                    _level_value(summary, "mean"),
                    _level_value(summary, "median"),
                ]
                for key, summary in (
                    (k, slopes[k]) for k in ("observed", "predicted", "difference")
                )
            ],
        ),
        f"{section} Cells by verdict",
        "",
        "How many cells fall in each verdict, over every household and state:",
        "",
        *_table(
            ["Reference", "Statistic", *_VERDICTS],
            [
                [
                    reference,
                    statistic,
                    *(
                        str(
                            sum(
                                entry["verdicts"][reference][statistic].get(v, 0)
                                for entry in households.values()
                            )
                        )
                        for v in _VERDICTS
                    ),
                ]
                for reference in (OWN, POPULATION)
                for statistic in STATISTICS
                if any(
                    entry["verdicts"][reference][statistic].get(v, 0)
                    for entry in households.values()
                    for v in _VERDICTS
                )
            ],
        ),
        f"{section} Households",
        "",
        "Each household's cells, its own-reference active dispersion in each "
        "common state as a ratio, and its cells flagged above for active "
        "dispersion, quiet runs and bursts:",
        "",
        *_table(
            ["Household", "Fold", "Cells", *common, "Flagged"],
            [
                [
                    home,
                    entry["fold"],
                    str(len(entry["cells"])),
                    *(_household_ratio(groups, home, state) for state in common),
                    " / ".join(
                        str(entry["verdicts"][OWN][s].get("above", 0))
                        for s in ("active_dispersion", *_RUNS)
                    ),
                ]
                for home, entry in households.items()
            ],
        ),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def _level_value(summary: Mapping[str, Any], statistic: str) -> str:
    if statistic not in summary:
        return "n/a"
    estimate = summary[statistic]
    interval = estimate["interval"]
    bounds = (
        f" [{_number(interval['low'])}, {_number(interval['high'])}]"
        if interval
        else ""
    )
    return f"{_number(estimate['estimate'])}{bounds}"


def _household_ratio(groups: Mapping[str, Any], home: str, state: str) -> str:
    summary = groups["state"][OWN]["active_dispersion"].get(state)
    if summary is None or home not in summary["values"]:
        return "n/a"
    return _number(math.exp(summary["values"][home]))
