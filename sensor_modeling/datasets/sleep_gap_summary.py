"""The sleep-gap pages, generated from the frozen plan and the record.

:func:`render_plan` writes the plan page from the frozen plan file, and
:func:`render_page` the results page from the description's record. Every
number on either page comes from those files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .sleep_gap_experiment import CORRELATED, REFERENCED
from .sleep_gap_plan import (
    AN_HOUR_LATER,
    AS_RECORDED,
    ASLEEP,
    AWAKE_ON_THE_MAT,
    CLOCKS,
    COMPOSITION,
    DAY,
    DISTANCE_BINS,
    EVENTS,
    MAT_CLASSES,
    MAT_IN_BED,
    MAT_SLEEP,
    NIGHT,
    NOTHING_YET,
    OFF,
    OFF_THE_MAT,
    ON_THE_MAT,
    PIPELINE,
    QUIET,
    SINCE_BINS,
)
from .threshold_calibration_summary import _fixed

PLAN_FILE = "artifacts/sleep_gap/sleep_gap_plan.json"
RECORD_FILE = "artifacts/sleep_gap/sleep-gap.json"
PLAN_PAGE = "SLEEP_GAP_PLAN.md"
RESULTS_PAGE = "SLEEP_GAP_RESULTS.md"
FIGURE_DIR = "figures"
FIGURE_PREFIX = "sleep-gap"

CLOCK_TITLES = {AS_RECORDED: "mat clock as recorded", AN_HOUR_LATER: "an hour later"}
CLASS_TITLES = {
    ASLEEP: "the mat says asleep",
    AWAKE_ON_THE_MAT: "the mat says awake in bed",
    OFF_THE_MAT: "no mat record",
}
_QUANTITY_TITLES = {
    PIPELINE: "the pipeline's hours of sleep",
    ON_THE_MAT: "its hours of sleep while the mat has a record",
    OFF: "its hours of sleep with no mat record",
    MAT_SLEEP: "the mat's hours asleep",
    MAT_IN_BED: "the mat's hours in bed",
    QUIET: "the day's quiet hours",
    EVENTS: "the day's activations",
}
_COMPOSITION_TITLES = {
    PIPELINE: "The pipeline's hours of sleep",
    "pipeline_asleep": "of which while the mat says asleep",
    "pipeline_awake_on_the_mat": "of which while the mat says awake in bed",
    OFF: "of which with no mat record",
    MAT_SLEEP: "The mat's hours asleep",
    "mat_sleep_not_counted": "of which the pipeline did not count as sleep",
    "mat_asleep_unwatched": "of which no step of the pipeline covered",
    MAT_IN_BED: "The mat's hours in bed",
    "mat_in_bed_not_counted": "of which the pipeline did not count as sleep",
    "difference": "Pipeline minus the mat's sleep",
    "difference_in_bed": "Pipeline minus the mat's time in bed",
}
_SINCE_TITLES = {name: name.replace("_", " ") for name, _, _ in SINCE_BINS}
_SINCE_TITLES[NOTHING_YET] = "nothing has reported yet"
_DISTANCE_TITLES = {name: name.replace("_", " ") for name, _, _ in DISTANCE_BINS}
_PERIOD_TITLES = {DAY: "day, 07:00 to 22:00", NIGHT: "night, 22:00 to 07:00"}


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|")


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _sentence(text: str) -> str:
    text = text.strip()
    if not text.startswith("`"):
        text = text[0].upper() + text[1:]
    return text if text.endswith(".") else text + "."


def _title(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _n(value: Any, digits: int = 2) -> str:
    """A number as the page shows it; a missing one as a dash."""
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    return _fixed(value, digits, group=True)


def _interval(
    entry: Mapping[str, Any] | None, digits: int = 2, signed: bool = False
) -> str:
    """An estimate with its interval, as ``0.12 [0.05, 0.20]``."""
    if entry is None or entry.get("estimate") is None:
        return "–"

    def show(value: float) -> str:
        return _fixed(value, digits, sign=signed, group=True)

    text = show(float(entry["estimate"]))
    interval = entry.get("interval")
    if interval:
        text += f" [{show(float(interval['low']))}, {show(float(interval['high']))}]"
    return text


def _flatten(value: Any) -> str:
    if isinstance(value, Mapping):
        return "; ".join(
            f"{_title(k).lower()}: {_flatten(v)}" for k, v in value.items()
        )
    if isinstance(value, list | tuple):
        return ", ".join(_flatten(v) for v in value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _section(title: str, entries: Mapping[str, Any]) -> list[str]:
    lines = [f"## {title}", ""]
    for name, value in entries.items():
        lines.append(f"- **{_title(name)}.** {_sentence(_flatten(value))}")
    return [*lines, ""]


def _figure(name: str, alt: str) -> str:
    return f"![{alt}]({FIGURE_DIR}/{FIGURE_PREFIX}-{name}.svg)"


# ----------------------------------------------------------------------------
# The plan page
# ----------------------------------------------------------------------------
def render_plan(payload: Mapping[str, Any]) -> str:
    """The plan page, from the frozen plan file's content."""
    lines = [
        "# Where the pipeline's sleep parts from the sleep mat: the plan",
        "",
        "The sleep-mat comparison found that, within a TIHM home, the "
        "pipeline's daily hours of sleep barely follow a sleep mat's. This page "
        "is the plan for setting the pipeline's belief beside the mat minute "
        "by minute, written before any of it was computed, and generated "
        f"entirely from the frozen file `{PLAN_FILE}` by "
        "`sensor_modeling.datasets.sleep_gap_summary.render_plan`.",
        "",
        f"**Status: {payload['status']}.**",
        "",
        f"- **Plan digest.** `{payload['plan_sha256']}`.",
        f"- **The question.** {_sentence(payload['question'])}",
        "- **This page reports no result.**",
    ]
    at_freeze = payload.get("at_freeze")
    if at_freeze:
        versions = ", ".join(
            f"{name} {version}" for name, version in at_freeze["distributions"].items()
        )
        lines.append(
            "- **The code it was frozen against.** The digests of "
            f"{len(at_freeze['sources']):,} source files, every default "
            f"setting, and {versions}."
        )
    minutes = payload["the_minutes"]
    lines += [
        "",
        "## What had been seen",
        "",
        *[f"- {_sentence(item)}" for item in payload["inspected_before"]],
        "",
    ]
    lines += _section("The homes and days", payload["homes_and_days"])
    lines += [
        "## The minutes",
        "",
        f"- **Grid.** {_sentence(minutes['grid'])}",
        f"- **The pipeline.** {_sentence(minutes['the_pipeline'])}",
        "",
        *_table(
            ["What the mat says of a minute", "When"],
            [
                [CLASS_TITLES[name], _sentence(text)]
                for name, text in minutes["the_mat"].items()
            ],
        ),
        "",
        *[
            f"- **{_title(CLOCK_TITLES[name])}.** {_sentence(text)}"
            for name, text in minutes["clocks"].items()
        ],
        f"- **The context's moment.** {_sentence(minutes['context']['moment'])}",
        f"- **The sensor that last reported.** "
        f"{_sentence(minutes['context']['sensor'])}",
        "- **The time since it.** "
        + _sentence(
            "; ".join(
                f"{_SINCE_TITLES.get(name, name)}: {text}"
                for name, text in minutes["context"]["since"].items()
            )
        ),
        f"- **Period.** {_sentence(minutes['context']['period'])}",
        "- **The time to the nearest mat record.** "
        + _sentence(
            "; ".join(
                f"{_DISTANCE_TITLES[name]}: {text}"
                for name, text in minutes["context"]["distance_to_the_mat"].items()
            )
        ),
        f"- **Quiet hours.** {_sentence(minutes['context']['quiet_hours'])}",
        "",
        "## Estimands",
        "",
        *_table(
            ["", "Estimand"],
            [[k, _sentence(v)] for k, v in payload["estimands"].items()],
        ),
        "",
    ]
    lines += _section("Definitions", payload["definitions"])
    lines += _section("Intervals", payload["intervals"])
    check = payload["the_check"]
    lines += [
        "## The check",
        "",
        *[f"- {_sentence(item)}" for item in check["what"]],
        f"- **Otherwise.** {_sentence(check['otherwise'])}",
        "",
        "## What each pattern would mean",
        "",
        "Written down before anything was computed.",
        "",
        *[f"- {_sentence(item)}" for item in payload["what_the_patterns_would_mean"]],
        "",
        "## Reporting",
        "",
        _sentence(payload["reporting"]),
        "",
        "## Pinned records",
        "",
        *_table(
            ["File", "SHA-256"],
            [
                [f"`{name}`", f"`{digest}`"]
                for name, digest in payload["pinned_records"].items()
            ],
        ),
        "",
        "## What this cannot show",
        "",
        *[f"- {_sentence(item)}" for item in payload["what_this_cannot_show"]],
        "",
        "## Data",
        "",
        _sentence(payload["acknowledgement"]),
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# The results page
# ----------------------------------------------------------------------------
def _changed(configuration: Mapping[str, Any]) -> str:
    changed = configuration.get("code_changed_since_the_freeze") or {}
    names = [name for group in changed.values() for name in group]
    return "None." if not names else ", ".join(f"`{n}`" for n in names) + "."


def _median(entry: Mapping[str, Any], digits: int = 2) -> str:
    return _n(entry.get("median"), digits)


def _composition_part(clocks: Mapping[str, Any]) -> list[str]:
    header = [""]
    for clock in CLOCKS:
        header += [f"{_title(CLOCK_TITLES[clock])}: mean", "median"]
    rows = []
    for name in COMPOSITION:
        row = [_COMPOSITION_TITLES[name]]
        for clock in CLOCKS:
            entry = clocks[clock]["D1"][name]
            row += [_interval(entry), _median(entry)]
        rows.append(row)
    first = clocks[AS_RECORDED]["D1"]

    def value(name: str) -> str:
        return _n(first[name]["estimate"])

    return [
        "## The hours of sleep, by what the mat says (D1)",
        "",
        "Hours a day: the mean over homes of each home's mean over its matched "
        "days, with a Student t interval over homes, and the median over homes.",
        "",
        *_table(header, rows),
        "",
        f"With the mat's clock as recorded, the pipeline's {value('difference')} "
        f"hours a day more sleep than the mat's are {value(OFF)} hours counted "
        f"with no mat record, plus {value('pipeline_awake_on_the_mat')} counted "
        f"while the mat says awake in bed, less {value('mat_sleep_not_counted')} "
        "hours of the mat's sleep the pipeline did not count as sleep. Against "
        f"the mat's time in bed, {value('difference_in_bed')} hours: "
        f"{value(OFF)} counted with no record, less "
        f"{value('mat_in_bed_not_counted')} hours in bed not counted as sleep.",
        "",
        _figure("composition", "The pipeline's hours of sleep beside the mat's"),
        "",
    ]


def _per_home_part(clocks: Mapping[str, Any]) -> list[str]:
    per_home = clocks[AS_RECORDED]["per_home"]
    names = (PIPELINE, OFF, MAT_SLEEP, MAT_IN_BED, "difference")
    rows = [
        [
            f"`{home}`",
            row["matched_days"],
            *[_n(row[name]) for name in names],
            _n(row[f"r[{MAT_SLEEP}~{PIPELINE}]"]),
        ]
        for home, row in per_home.items()
    ]
    return [
        "With the mat's clock as recorded, home by home, in hours a day:",
        "",
        *_table(
            [
                "Home",
                "Days",
                "Pipeline's sleep",
                "of which no mat record",
                "Mat asleep",
                "Mat in bed",
                "Pipeline minus mat",
                "Correlation with the mat's sleep",
            ],
            rows,
        ),
        "",
    ]


def _hourly_part(clocks: Mapping[str, Any]) -> list[str]:
    hourly = clocks[AS_RECORDED]["D2"]
    rows = []
    for hour in range(24):
        on = (
            hourly[f"pipeline_{ASLEEP}"][hour]
            + hourly[f"pipeline_{AWAKE_ON_THE_MAT}"][hour]
        )
        rows.append(
            [
                f"{hour:02d}:00",
                _n(hourly[MAT_IN_BED][hour]),
                _n(hourly[MAT_SLEEP][hour]),
                _n(on),
                _n(hourly[f"pipeline_{OFF_THE_MAT}"][hour]),
            ]
        )
    return [
        "## Hour by hour (D2)",
        "",
        "The mean over homes of each home's mean hours a day in each local hour, "
        "with the mat's clock as recorded; the figure shows both clocks. The "
        "pipeline's hours at 00 and 23 are lower by construction: the interval "
        "across midnight is counted by neither day.",
        "",
        _figure("hourly", "The pipeline's sleep and the mat's, hour by hour"),
        "",
        *_table(
            [
                "Hour",
                "Mat in bed",
                "Mat asleep",
                "Pipeline's sleep, mat record",
                "Pipeline's sleep, no mat record",
            ],
            rows,
        ),
        "",
    ]


def _beliefs_part(clocks: Mapping[str, Any], states: Sequence[str]) -> list[str]:
    lines = [
        "## What the pipeline believed, by what the mat says (D3)",
        "",
        "The mean over homes of each home's mean belief in each state over its "
        "watched minutes of each class.",
        "",
    ]
    for clock in CLOCKS:
        rows = []
        for name in MAT_CLASSES:
            entry = clocks[clock]["D3"][name]
            mean = entry["mean"] or [None] * len(states)
            rows.append([CLASS_TITLES[name], entry["homes"], *[_n(v) for v in mean]])
        lines += [
            f"**{_title(CLOCK_TITLES[clock])}.**",
            "",
            *_table(["When", "Homes", *[f"`{s}`" for s in states]], rows),
            "",
        ]
    return lines


_CONTEXT_HEADER = [
    "",
    "Pipeline's sleep, hours a day",
    "No mat record, hours a day",
    "Belief in sleeping",
]


def _context_rows(
    cells: Mapping[str, Any], title: Any, order: Sequence[str] | None = None
) -> list[list[str]]:
    keys = (
        [k for k in order if k in cells]
        if order is not None
        else sorted(cells, key=lambda k: -cells[k]["pipeline_sleep_hours"])
    )
    return [
        [
            title(key),
            _n(cells[key]["pipeline_sleep_hours"]),
            _n(cells[key]["hours_with_no_record"]),
            _interval(cells[key]["sleeping_belief"]),
        ]
        for key in keys
    ]


def _sensor_title(key: str) -> str:
    return _SINCE_TITLES[NOTHING_YET] if key == NOTHING_YET else key


def _joint_title(key: str) -> str:
    sensor, since, period = key.split("|")
    if sensor == NOTHING_YET:
        return f"{_SINCE_TITLES[NOTHING_YET]}, by {period}"
    return f"{sensor}, {_SINCE_TITLES[since]} before, by {period}"


def _distance_period_title(key: str) -> str:
    distance, period = key.split("|")
    return f"{_DISTANCE_TITLES[distance]}, by {period}"


_EXIT_TITLES = {
    "exit": "an exit door, front or back",
    "other": "another sensor",
    NOTHING_YET: _SINCE_TITLES[NOTHING_YET],
}

#: Rows of the joint table under this many hours a day of the pipeline's sleep
#: are left to the record.
JOINT_FLOOR_HOURS = 0.05


def _context_part(clocks: Mapping[str, Any]) -> list[str]:
    since_order = [name for name, _, _ in SINCE_BINS] + [NOTHING_YET]
    lines = [
        "## With no mat record: what the sensors had last said (D4)",
        "",
        "Over the minutes with no mat record: the pipeline's hours of sleep and "
        "the hours of such minutes, each a mean over homes of a home's mean a "
        "day, and the mean over homes of the pipeline's belief in sleeping over "
        "them. The sensor and the time since it are as of the step whose belief "
        "covers the minute.",
        "",
    ]
    for clock in CLOCKS:
        context = clocks[clock]["D4"]
        name = _title(CLOCK_TITLES[clock])
        lines += [
            f"**{name}, by the sensor that last reported.**",
            "",
            *_table(_CONTEXT_HEADER, _context_rows(context["sensor"], _sensor_title)),
            "",
            f"**{name}, by the time since it.**",
            "",
            *_table(
                _CONTEXT_HEADER,
                _context_rows(
                    context["since"], lambda k: _SINCE_TITLES[k], since_order
                ),
            ),
            "",
            f"**{name}, by whether the sensor was an exit.**",
            "",
            *_table(
                _CONTEXT_HEADER,
                _context_rows(
                    context["exit"],
                    lambda k: _EXIT_TITLES[k],
                    ["exit", "other", NOTHING_YET],
                ),
            ),
            "",
            f"**{name}, by period of the day.**",
            "",
            *_table(
                _CONTEXT_HEADER,
                _context_rows(
                    context["period"], lambda k: _PERIOD_TITLES[k], [DAY, NIGHT]
                ),
            ),
            "",
        ]
    joint = {
        key: cell
        for key, cell in clocks[AS_RECORDED]["D4"]["joint"].items()
        if cell["pipeline_sleep_hours"] >= JOINT_FLOOR_HOURS
    }
    lines += [
        "**Mat clock as recorded, by the sensor, the time since it and the "
        "period together.** Rows with under "
        f"{_n(JOINT_FLOOR_HOURS)} hours a day of the pipeline's sleep are in "
        "the record.",
        "",
        *_table(_CONTEXT_HEADER, _context_rows(joint, _joint_title)),
        "",
        _figure("context", "The pipeline's sleep with no mat record"),
        "",
    ]
    return lines


def _ordered(entries: Mapping[str, Any], order: Sequence[tuple[str, str]]) -> list[str]:
    """The record's names in the order the description declares them."""
    names = [f"{left}~{right}" for left, right in order]
    return [name for name in names if name in entries] + sorted(
        name for name in entries if name not in names
    )


def _pair_title(name: str) -> str:
    left, right = name.split("~")
    return f"{_QUANTITY_TITLES[left]}, with {_QUANTITY_TITLES[right]}"


def _tracking_part(clocks: Mapping[str, Any]) -> list[str]:
    first = clocks[AS_RECORDED]["D5"]
    clocks_header = ["", *[_title(CLOCK_TITLES[c]) for c in CLOCKS]]
    header = ["Correlation"]
    for clock in CLOCKS:
        header += [_title(CLOCK_TITLES[clock]), "Swapped-mask reference"]
    header.append("Constant")
    rows = []
    for name in _ordered(first["correlations"], CORRELATED):
        row = [_sentence(_pair_title(name))[:-1]]
        for clock in CLOCKS:
            entry = clocks[clock]["D5"]
            row.append(_interval(entry["correlations"][name]))
            reference = entry["references"].get(name)
            row.append(_interval(reference) if reference else "")
        row.append(
            ", ".join(
                str(clocks[c]["D5"]["correlations"][name]["constant_homes"])
                for c in CLOCKS
            )
        )
        rows.append(row)
    excess = [
        [
            _sentence(_pair_title(name))[:-1],
            *[
                f"{_interval(clocks[c]['D5']['excess'][name], signed=True)}; "
                f"{clocks[c]['D5']['excess'][name]['homes_above_zero']} of "
                f"{clocks[c]['D5']['excess'][name]['homes']} homes above zero"
                for c in CLOCKS
            ],
        ]
        for name in _ordered(first["excess"], REFERENCED)
    ]
    shares = [
        [
            _sentence(_QUANTITY_TITLES[part])[:-1],
            *[_interval(clocks[c]["D5"]["shares"][part]) for c in CLOCKS],
        ]
        for part in first["shares"]
    ]
    spread = [
        [
            _sentence(_QUANTITY_TITLES[name])[:-1],
            *[_interval(clocks[c]["D5"]["spread"][name]) for c in CLOCKS],
        ]
        for name in first["spread"]
    ]
    return [
        "## Day to day (D5)",
        "",
        "The mean over homes of the within-home Spearman correlation over "
        "matched days. The swapped-mask reference sets every day's beliefs "
        "beside every other day's mat, the mean over all cyclic shifts of the "
        "days: what the mat's own hours give a part of the pipeline's sleep "
        "with no day-specific tracking. A home whose values on either side are constant "
        "counts as 0; the last column gives how many did, under each clock.",
        "",
        *_table(header, rows),
        "",
        "Each home's correlation minus its swapped-mask reference, as a mean "
        "over homes, with the homes above zero:",
        "",
        *_table(clocks_header, excess),
        "",
        "The share of the variance of the pipeline's daily hours of sleep that "
        "moves with each part, its covariance with the part over its variance, "
        "as a mean over homes; the two shares of a home sum to one:",
        "",
        *_table(clocks_header, shares),
        "",
        "The mean over homes of the within-home standard deviation, in hours:",
        "",
        *_table(clocks_header, spread),
        "",
    ]


def _without_part(clocks: Mapping[str, Any]) -> list[str]:
    first = clocks[AS_RECORDED]["D6"]
    clocks_header = ["", *[_title(CLOCK_TITLES[c]) for c in CLOCKS]]
    rows = [
        [
            _COMPOSITION_TITLES[name],
            *[_interval(clocks[c]["D6"]["D1"][name]) for c in CLOCKS],
        ]
        for name in COMPOSITION
    ]
    rows += [
        [
            _sentence(_pair_title(name))[:-1],
            *[_interval(clocks[c]["D6"]["D5"]["correlations"][name]) for c in CLOCKS],
        ]
        for name in _ordered(first["D5"]["correlations"], CORRELATED)
    ]
    rows += [
        [
            f"Variance share: {_QUANTITY_TITLES[part]}",
            *[_interval(clocks[c]["D6"]["D5"]["shares"][part]) for c in CLOCKS],
        ]
        for part in first["D5"]["shares"]
    ]
    left_out = ", ".join(f"`{h}`" for h in first["left_out"]) or "none"
    return [
        "## Without the homes the mat stages mostly awake (D6)",
        "",
        f"Left out: {left_out}. D2 to D4 and D7 without them are in the record.",
        "",
        *_table(clocks_header, rows),
        "",
    ]


def _distance_part(clocks: Mapping[str, Any], short_hours: float) -> list[str]:
    order = [name for name, _, _ in DISTANCE_BINS]
    lines = [
        "## With no mat record: how far from the mat's records (D7)",
        "",
        "As D4, by the time from the minute to the nearest mat record, earlier "
        "or later, in the home's whole file.",
        "",
    ]
    for clock in CLOCKS:
        lines += [
            f"**{_title(CLOCK_TITLES[clock])}.**",
            "",
            *_table(
                _CONTEXT_HEADER,
                _context_rows(
                    clocks[clock]["D7"]["distance"],
                    lambda k: _DISTANCE_TITLES[k],
                    order,
                ),
            ),
            "",
        ]
    by_period = clocks[AS_RECORDED]["D7"]["distance_period"]
    lines += [
        "**Mat clock as recorded, by the time to the nearest record and the "
        "period together.**",
        "",
        *_table(
            _CONTEXT_HEADER,
            _context_rows(
                by_period,
                _distance_period_title,
                [f"{d}|{p}" for d in order for p in (DAY, NIGHT)],
            ),
        ),
        "",
    ]
    short = clocks[AS_RECORDED]["D7"]["short_nights"]
    lines += [
        f"With the mat's clock as recorded, {short['days']:,} matched days in "
        f"{short['homes']} homes have fewer than {_n(short_hours, 0)} hours of "
        "mat records; on "
        "them falls a mean over homes of "
        f"{_interval(short['share_of_sleep_with_no_record'])} of a home's sleep "
        "counted with no record.",
        "",
    ]
    return lines


def render_page(payload: Mapping[str, Any]) -> str:
    """The results page, from the record."""
    results = payload["results"]
    configuration = payload["configuration"]
    environment = payload.get("environment", {})
    commit = str(environment.get("git_commit", ""))[:7]
    check = results["check"]
    clocks = results["clocks"]
    homes = clocks[AS_RECORDED]["homes"]
    lines = [
        "# Where the pipeline's sleep parts from the sleep mat: results",
        "",
        "The frozen [plan](SLEEP_GAP_PLAN.md), run as declared. This page is "
        f"generated entirely from `{RECORD_FILE}` by "
        "`sensor_modeling.datasets.sleep_gap_summary.render_page`, and a test "
        "checks that the committed page is exactly that rendering.",
        "",
        f"**Status: exploratory description of {len(homes)} homes of people "
        "living with dementia, planned after the sleep-mat comparison had been "
        "read and frozen before anything of it was computed. No criterion, no "
        "margin. Nothing in the pipeline was changed or fitted. The mat's stages "
        "are the device's own and are not validated, and a minute with no mat "
        "record may be an empty bed, a person asleep elsewhere or a mat that "
        "stopped.**",
        "",
        f"- **Plan digest.** `{configuration['plan_sha256']}`.",
        f"- **Run.** Commit `{commit}`, recorded {payload['recorded_at']}.",
        f"- **Homes.** {len(homes)}, on {check['pairs']['matched_days']:,} "
        "matched days.",
        f"- **Code changed since the freeze.** {_changed(configuration)}",
        "",
        "## The check",
        "",
        "With the rule off the run gave the published alert-burden counts for "
        f"all {check['off']['homes']} mat homes, and with the rule at 12 hours "
        "the silent-home record's. Every home's number of matched days and "
        "per-home statistics equal the sleep-mat record's. The repeat's daily "
        "summaries equal the run's; on every matched day the minutes sum, for "
        "every state and for the time watched, to the day's hours, to within "
        f"{check['beliefs']['largest_difference_hours']:.1e} hours, and the "
        "mat's minutes to its hours under each clock. No matched day has a "
        "clock change, and D1's difference from the mat's sleep is the "
        f"sleep-mat record's {_n(check['difference']['hours'])} hours.",
        "",
    ]
    lines += _composition_part(clocks)
    lines += _per_home_part(clocks)
    lines += _hourly_part(clocks)
    lines += _beliefs_part(clocks, results["states"])
    lines += _context_part(clocks)
    lines += _tracking_part(clocks)
    lines += _without_part(clocks)
    lines += _distance_part(clocks, float(configuration["short_night_hours"]))
    lines += [
        "## Notes",
        "",
        *[f"- {_sentence(note)}" for note in payload.get("notes", [])],
        "",
    ]
    return "\n".join(lines)
