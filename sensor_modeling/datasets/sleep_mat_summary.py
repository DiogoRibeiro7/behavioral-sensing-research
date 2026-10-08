"""The sleep-mat pages, generated from the frozen protocol and the records.

:func:`render_protocol` writes the protocol page from the frozen protocol file,
and :func:`render_page` the results page from the record of the comparison on
TIHM and the record of the simulated reference. Every number on either page
comes from those files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .sleep_mat_protocol import (
    ANALYSES,
    IN_BED,
    PRIMARY,
    REFERENCES,
    RULE,
    SLEEP,
    WITH_SILENT_DAYS,
)
from .threshold_calibration_summary import _fixed, _percent_of

PROTOCOL_FILE = "artifacts/sleep_mat/sleep_mat_protocol.json"
RECORD_FILE = "artifacts/sleep_mat/sleep-mat.json"
SIMULATED_FILE = "artifacts/sleep_mat/sleep-mat-simulated.json"
PROTOCOL_PAGE = "SLEEP_MAT_PROTOCOL.md"
RESULTS_PAGE = "SLEEP_MAT_RESULTS.md"
FIGURE_DIR = "figures"
FIGURE_PREFIX = "sleep-mat"

_REFERENCE_TITLES = {SLEEP: "the mat's sleep", IN_BED: "the mat's hours in bed"}


def _analysis_title(name: str, rule_hours: float) -> str:
    return {
        PRIMARY: "rule off, silenced days left out (primary)",
        WITH_SILENT_DAYS: "rule off, every usable day",
        RULE: f"rule on at {rule_hours:g} hours, silenced days left out",
    }[name]


_READINGS = {
    "follows": "follows",
    "does_not_follow": "does not follow",
    "agrees": "agrees",
    "does_not_agree": "does not agree",
    "inconclusive": "inconclusive",
    "not_estimable": "not estimable",
}
_KINDS = {
    "abrupt_change": "an abrupt change",
    "persistent_change": "a persistent change",
    "gradual_drift": "a gradual drift",
}


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


def _criterion(name: str) -> tuple[str, str]:
    number, _, claim = name.partition("_")
    return number, _title(claim)


def _code(names: Sequence[Any]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _n(value: Any, digits: int = 2) -> str:
    """A number as the page shows it; a missing one as a dash."""
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    return _fixed(value, digits, group=True)


def _signed(value: Any, digits: int = 2) -> str:
    return "–" if value is None else _fixed(value, digits, sign=True)


def _interval(
    entry: Mapping[str, Any] | None, digits: int = 2, signed: bool = False
) -> str:
    """An estimate with its interval, as ``0.12 [0.05, 0.20]``."""
    if entry is None or entry.get("estimate") is None:
        return "–"
    show = _signed if signed else _n
    text = show(float(entry["estimate"]), digits)
    interval = entry.get("interval")
    if interval:
        text += (
            f" [{show(float(interval['low']), digits)}, "
            f"{show(float(interval['high']), digits)}]"
        )
    return text


def _limits(entry: Mapping[str, Any]) -> str:
    if entry.get("bias") is None:
        return "–"
    return f"{_signed(entry['low'], 1)} to {_signed(entry['high'], 1)}"


def _figure(name: str, alt: str) -> str:
    return f"![{alt}]({FIGURE_DIR}/{FIGURE_PREFIX}-{name}.svg)"


def _count(number: int, noun: str, plural: str | None = None) -> str:
    return f"{number:,} {noun}" if number == 1 else f"{number:,} {plural or noun + 's'}"


# ----------------------------------------------------------------------------
# The protocol page
# ----------------------------------------------------------------------------
def render_protocol(payload: Mapping[str, Any]) -> str:
    """The protocol page, from the frozen protocol file's content."""
    mat, pipeline, sim = (
        payload["the_mat"],
        payload["pipeline"],
        payload["simulated_reference"],
    )
    intervals = payload["intervals"]
    check = pipeline["check"]
    lines = [
        "# The pipeline's hours of sleep beside a sleep mat: the protocol",
        "",
        "Every alerting study in this repository counts its detections on "
        "`sleeping_hours`, the hours of a day the pipeline's state filter gives "
        "to sleeping. In 17 of the 56 homes of the [TIHM dataset]"
        "(TIHM_ALERT_BURDEN_RESULTS.md) a mat under the mattress recorded each "
        "minute a person was in bed. This page is the protocol for setting the "
        "two side by side, generated entirely from the frozen file "
        f"`{PROTOCOL_FILE}` by "
        "`sensor_modeling.datasets.sleep_mat_summary.render_protocol`.",
        "",
        f"**Status: {payload['status']}.**",
        "",
        f"- **Protocol digest.** `{payload['protocol_sha256']}`.",
        f"- **The question.** {_sentence(payload['question'])}",
        "- **This page reports no result.** The run checks the frozen file, "
        "the pinned records and the dataset's files before it runs.",
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
    lines += [
        "",
        "## What had been seen",
        "",
        *[f"- {_sentence(item)}" for item in payload["inspected_before"]],
        "",
        "## The data",
        "",
        f"{payload['data']['dataset']}. {payload['data']['citation']} "
        f"Licence: {payload['data']['licence']}.",
        "",
        *_table(
            ["File", "SHA-256", "Read"],
            [
                [
                    name,
                    f"`{digest}`",
                    "yes" if name in payload["data"]["read"] else "pinned, not read",
                ]
                for name, digest in payload["data"]["files"].items()
            ],
        ),
        "",
        f"Redistributed: {payload['data']['redistributed']}.",
        "",
        "## The mat",
        "",
        f"{_sentence(mat['what_it_is'])} Its file is `{mat['file']}`, with the "
        f"columns {_code(mat['columns'])} and timestamps as "
        f"`{mat['time_format']}`.",
        "",
        f"- **Stages.** {_code(mat['states'])}; asleep: {_code(mat['asleep'])}.",
        f"- **Clock.** {_sentence(mat['clock'])}",
        f"- **Standing.** {_sentence(mat['standing'])}",
        "",
        "## The pipeline",
        "",
        f"{_sentence(pipeline['what'])} Its step is {pipeline['step_minutes']} "
        f"minutes; everything else is {pipeline['everything_else']}.",
        "",
        *_table(
            ["Run", "What it is"],
            [[f"`{name}`", _sentence(text)] for name, text in pipeline["runs"].items()],
        ),
        "",
        *_table(
            ["Setting", "Value"],
            [
                ["day coverage a usable day needs", pipeline["min_day_coverage"]],
                ["day observed a usable day needs", pipeline["min_day_observed"]],
                *[
                    [f"baseline: {name.replace('_', ' ')}", value]
                    for name, value in pipeline["baseline"].items()
                ],
                *[
                    [f"alert policy: {name.replace('_', ' ')}", value]
                    for name, value in pipeline["alert_policy"].items()
                ],
            ],
        ),
        "",
        "**The check.** Nothing is reported unless both runs reproduce the "
        "published records:",
        "",
        *[
            f"- **`{name}`.** {_sentence(check[name])}"
            for name in pipeline["runs"]
            if name in check
        ],
        *[f"- `{name}`: `{digest}`." for name, digest in check["records"].items()],
        "",
        "## Definitions",
        "",
        *[
            f"- **{_title(name)}.** {_sentence(text)}"
            for name, text in payload["definitions"].items()
        ],
        "",
        "## The analyses",
        "",
        *_table(
            ["Analysis", "What it is"],
            [
                [f"`{name}`", _sentence(text)]
                for name, text in payload["analyses"].items()
            ],
        ),
        "",
        "## Estimands",
        "",
        *_table(
            ["Estimand", "Definition"],
            [[name, _sentence(text)] for name, text in payload["estimands"].items()],
        ),
        "",
        "## Criteria",
        "",
        *_table(
            ["Criterion", "Claim", "Rule"],
            [
                [*_criterion(name), _sentence(text)]
                for name, text in payload["criteria"].items()
                if name.startswith("C")
            ],
        ),
        "",
        f"- **Estimable.** {_sentence(payload['criteria']['estimable'])}",
        f"- **Multiplicity.** {_sentence(payload['criteria']['multiplicity'])}",
        "",
        "### The margins",
        "",
        *[
            f"- **{_title(name)}.** {_sentence(text)}"
            for name, text in payload["margins"].items()
        ],
        "",
        "### What the planning says to expect",
        "",
        *[
            f"- **{name}.** {_sentence(text)}"
            for name, text in payload["expected_readings"].items()
        ],
        "",
        "## Intervals",
        "",
        f"{_sentence(intervals['method'])}",
        "",
        f"- **Unit.** {intervals['unit'].capitalize()}; the interval chosen is "
        f"`{intervals['chosen']}`.",
        f"- **Why this interval.** {_sentence(intervals['why'])}",
        f"- **The planning's days.** {_sentence(intervals['planning_days'])}",
        f"- **Counts.** {_sentence(intervals['counts'])}",
        f"- **Planning record.** `{intervals['planning_record']}`, "
        f"`{intervals['planning_record_sha256']}`.",
        "",
        "## The simulated reference",
        "",
        f"**{_sentence(sim['standing'])}** {_sentence(sim['evidence'])}",
        "",
        *_table(
            ["Field", "Value"],
            [
                ["homes", sim["homes"]],
                ["seed root", f"`{sim['seed_root']}`"],
                ["seeds", sim["seeds"]],
                ["days in each record", sim["days"]],
                ["first day", sim["start"]],
                ["timezone", f"`{sim['timezone']}`"],
                ["household", sim["household"]],
                ["sensors", _code(sim["sensors"])],
                ["delivery", sim["delivery"]],
                ["step, minutes", sim["step_minutes"]],
            ],
        ),
        "",
        f"- **Why these sensors.** {_sentence(sim['why_these_sensors'])}",
        f"- **Truth.** {_sentence(sim['truth'])}",
        f"- **Matched day.** {_sentence(sim['matched_day'])}",
        "",
        "## Reporting",
        "",
        f"{_sentence(payload['reporting'])}",
        "",
        "What this cannot show:",
        "",
        *[f"- {_sentence(item)}" for item in payload["what_this_cannot_show"]],
        "",
        f"{_sentence(payload['acknowledgement'])}",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# The results page
# ----------------------------------------------------------------------------
def _check_part(results: Mapping[str, Any], rule_hours: float) -> list[str]:
    off, rule = results["check"]["off"], results["check"]["rule_12h"]
    return [
        "## The runs",
        "",
        f"With the silent-home rule off, the {off['homes']} mat homes give every "
        "count of their rows in the published alert-burden record: "
        f"{off['monitored_days']:,} monitored days and "
        f"{_count(off['behavioural_alerts'], 'behavioural alert')}. With it on "
        f"at {rule_hours:g} hours they give the silent-home description's counts: "
        f"{_count(rule['days_refused_because_of_the_rule'], 'day')} refused "
        "because of the rule and "
        f"{_count(rule['silence_alerts'], 'silence alert')}.",
    ]


def _criteria_part(results: Mapping[str, Any]) -> list[str]:
    criteria = results["criteria"]
    c1 = criteria["C1_the_pipeline_follows_the_mat"]
    c2 = criteria["C2_the_pipeline_agrees_in_level"]
    e1, e2 = c1["estimate"], c2["estimate"]
    rows = [
        [
            "C1",
            "The pipeline follows the mat",
            f"**{_READINGS[c1['reading']]}**",
            f"E1, the mean within-home Spearman correlation over {e1['homes']} "
            f"homes, is {_interval(e1)}; the margin is {_n(c1['margin'], 1)}.",
        ],
        [
            "C2",
            "The pipeline agrees in level",
            f"**{_READINGS[c2['reading']]}**",
            f"E2, the mean difference over {e2['homes']} homes, is "
            f"{_interval(e2, signed=True)} hours; the margin is "
            f"±{_n(c2['margin'], 0)} hour.",
        ],
    ]
    reading = {
        "follows": "On the days the sensors reported, the pipeline's hours of "
        "sleep rise and fall with the mat's, within a home, by at least the "
        "declared margin.",
        "does_not_follow": "On the days the sensors reported, the pipeline's "
        "hours of sleep do not rise and fall with the mat's, within a home, by "
        "as much as the declared margin: a day the pipeline calls unusual is "
        "not reliably a day of unusual sleep.",
        "inconclusive": "Whether the pipeline's hours of sleep follow the "
        "mat's by the declared margin is not settled by these homes.",
        "not_estimable": "Too few homes were included to judge C1.",
    }[c1["reading"]]
    level = {
        "agrees": "The two agree in level within an hour.",
        "does_not_agree": "The two do not agree in level: the mean difference "
        "lies wholly beyond an hour.",
        "inconclusive": "Whether the two agree in level within an hour is not "
        "settled.",
        "not_estimable": "Too few homes were included to judge C2.",
    }[c2["reading"]]
    return [
        "## The criteria",
        "",
        "Both were fixed, with their margins, before any value of the pipeline "
        "was set beside any record of the mat. They are read in the primary "
        "analysis: the rule off, silenced days left out, which are the days "
        "with no activity record and those the run with the rule on flags as "
        "touched by a silence of the whole home. C1 is the only primary "
        "criterion, and the two are not adjusted for each other.",
        "",
        *_table(["Criterion", "Claim", "Reading", "What decided it"], rows),
        "",
        f"- **C1.** {reading}",
        f"- **C2.** {level}",
        "- **The mat is not a gold standard.** Its stages are the device's own. "
        "A disagreement may be the mat's as well as the pipeline's.",
    ]


def _homes_part(results: Mapping[str, Any], minimum: int) -> list[str]:
    days_rows, agreement_rows = [], []
    for home, row in results["homes"].items():
        primary = row.get(PRIMARY) or {}
        sleep = primary.get(SLEEP) or {}
        bed = primary.get(IN_BED) or {}
        days_rows.append(
            [
                f"`{home}`",
                f"{row['mat_records']:,}",
                row["mat_days"],
                row["mat_observed_days"],
                row["monitored_days"],
                row["usable_days"]["off"],
                row["silent_days"],
                row["silent_matched_days"],
                row["partly_silent_matched_days"],
                primary.get("matched_days", 0),
                "yes" if primary.get("included") else "no",
                _n(row["staged_asleep_share"]),
            ]
        )
        if primary.get("included"):
            agreement_rows.append(
                [
                    f"`{home}`",
                    _n(sleep.get("spearman")),
                    _n(bed.get("spearman")),
                    _signed(sleep.get("mean_difference"), 1),
                    _signed(bed.get("mean_difference"), 1),
                    _n(sleep.get("median_pipeline"), 1),
                    _n(sleep.get("median_mat"), 1),
                    _n(bed.get("median_mat"), 1),
                    (
                        "–"
                        if row["most_negative_lag"] is None
                        else row["most_negative_lag"]
                    ),
                ]
            )
    return [
        "## The homes",
        "",
        "Every mat home's days. A silent day has no activity record; a "
        "partly silent day has records and lost time to a silence of the whole "
        "home, as the run with the rule on flags it. Both are counted among "
        "the days that would otherwise be matched, and both are left out of "
        f"the primary analysis. A home enters the estimands with at least "
        f"{minimum} matched days.",
        "",
        *_table(
            [
                "Home",
                "Mat records",
                "Mat days",
                "Mat-observed",
                "Monitored",
                "Usable",
                "Silent",
                "Silent, matched",
                "Partly silent, matched",
                "Matched",
                "Included",
                "Staged asleep",
            ],
            days_rows,
        ),
        "",
        "The included homes beside the mat, in the primary analysis. The last "
        "column is the lag of E11 at which the home's hourly activity is most "
        "opposed to its minutes in bed.",
        "",
        *_table(
            [
                "Home",
                "Spearman, sleep",
                "Spearman, in bed",
                "Difference, sleep (h)",
                "Difference, in bed (h)",
                "Median pipeline (h)",
                "Median mat sleep (h)",
                "Median mat in bed (h)",
                "Most opposed at lag",
            ],
            agreement_rows,
        ),
    ]


def _agreement_part(results: Mapping[str, Any], rule_hours: float) -> list[str]:
    rows = []
    for name in ANALYSES:
        entry = results["analyses"][name]
        for reference in REFERENCES:
            values = entry[reference]
            rows.append(
                [
                    _analysis_title(name, rule_hours),
                    _REFERENCE_TITLES[reference],
                    len(entry["homes"]),
                    _interval(values["spearman"]),
                    _interval(values["pearson"]),
                    _interval(values["mean_difference"], signed=True),
                    _limits(values["limits_of_agreement"]),
                ]
            )
    constant = sorted(
        {
            home
            for name in ANALYSES
            for reference in REFERENCES
            for home in results["analyses"][name][reference][
                "homes_with_constant_values"
            ]
        }
    )
    repeats = results["analyses"][RULE] == results["analyses"][PRIMARY] and all(
        row.get(RULE) == row.get(PRIMARY) for row in results["homes"].values()
    )
    return [
        "## Agreement (E1 to E5, E9)",
        "",
        "Means over included homes, with Student t intervals over homes. The "
        "difference is the pipeline's value minus the mat's, in hours; the "
        "limits of agreement add the spread of homes' mean differences to the "
        "spread of days within a home.",
        "",
        *_table(
            [
                "Analysis",
                "Against",
                "Homes",
                "Spearman",
                "Pearson",
                "Difference (h)",
                "Limits of agreement (h)",
            ],
            rows,
        ),
        "",
        (
            "Homes whose values were constant on either source, counted as 0: "
            + _code(constant)
            + "."
            if constant
            else "No included home had constant values on either source."
        ),
        *(
            [
                "",
                "With the rule on, every home has the same matched days and the "
                "same values on them as in the primary analysis, so its rows "
                "repeat the primary's: the rule refuses the silenced days, which "
                "the primary leaves out already.",
            ]
            if repeats
            else []
        ),
        "",
        _figure("homes", "Within-home correlation of each home with the mat"),
        "",
        _figure("differences", "Mean difference of each home from the mat"),
    ]


def _deviations_part(results: Mapping[str, Any], minimum: int) -> list[str]:
    e6 = results["E6_deviations"]
    flagged = e6["pipeline_past_threshold"]
    return [
        "## The deviations (E6)",
        "",
        "The mat's sleep was passed through a personal baseline with the "
        "default configuration, as the pipeline passes its own days. The two "
        "baselines saw different histories: the pipeline's was given its "
        "silent days too.",
        "",
        f"- **Correlation of the two deviations.** {_interval(e6['spearman'])}, "
        f"over {e6['spearman']['homes']} homes with at least {minimum} days on "
        "which both baselines gave a verdict.",
        f"- **Days the pipeline's deviation reached {_n(e6['threshold'], 0)}.** "
        f"{flagged:,}. On {e6['same_sign']:,} of them the mat's deviation had "
        f"the same sign, and on {e6['also_past_threshold']:,} it also reached "
        f"{_n(e6['threshold'], 0)}. Days of one home are not independent, so no "
        "interval is given.",
    ]


def _alerts_part(results: Mapping[str, Any]) -> list[str]:
    e7 = results["E7_alerts"]
    kinds = ", ".join(
        f"{count:,} for {_KINDS.get(kind, kind)}"
        for kind, count in sorted(e7["by_kind"].items(), key=lambda item: -item[1])
    )
    median, baseline = e7["median"], e7["mat_baseline"]
    return [
        "## The alerts about sleep (E7)",
        "",
        f"With the rule off the pipeline raised "
        f"{_count(e7['alerts'], 'behavioural alert')} about the hours of sleep "
        f"in the mat homes: {kinds or 'none'}. An alert for a drift is not "
        "judged, since its direction is its slope's, which the run does not "
        "keep. "
        + (
            "None of the alerts for an abrupt or a persistent change came on a "
            "matched day of the primary analysis, so none was judged."
            if not e7["judged"]
            else "Of the alerts for an abrupt or a persistent change, "
            f"{e7['judged']:,} came on a matched day of the primary analysis."
        ),
        *(
            []
            if not e7["judged"]
            else [
                "",
                *_table(
                    ["Judged against", "Same direction", "Opposite", "Neither"],
                    [
                        [
                            "the home's median mat sleep",
                            median["same_side"],
                            median["opposite"],
                            f"{median['at_the_median']} at the median",
                        ],
                        [
                            "the mat baseline's deviation",
                            baseline["same_sign"],
                            baseline["opposite"],
                            f"{baseline['no_verdict']} without a verdict, "
                            f"{baseline['zero']} at zero",
                        ],
                    ],
                ),
            ]
        ),
    ]


def _silent_part(results: Mapping[str, Any]) -> list[str]:
    e8 = results["E8_silent_days"]
    with_one = sum(1 for row in e8["homes"].values() if row["silent_days"])
    return [
        "## Silent days (E8)",
        "",
        f"The mat homes have {_count(e8['silent_days'], 'silent day')} with the "
        f"rule off, {e8['usable']:,} of them usable, in "
        f"{_count(with_one, 'home')}. On the {e8['mat_observed']:,} the "
        "mat observed, the pipeline's median is "
        f"{_n(e8['median_pipeline_hours'], 1)} hours of sleep, and the mat's "
        f"{_n(e8['median_mat_sleep_hours'], 1)} hours asleep and "
        f"{_n(e8['median_mat_in_bed_hours'], 1)} in bed.",
    ]


def _clocks_part(results: Mapping[str, Any], shifts: Sequence[float]) -> list[str]:
    e10, e11 = results["E10_alignment"], results["E11_clocks"]
    shift_text = " and ".join(f"{h:+g}" for h in shifts)
    rows = [
        [
            f"{shift} h",
            len(entry["homes"]),
            _interval(entry[SLEEP]["spearman"]),
            _interval(entry[SLEEP]["mean_difference"], signed=True),
        ]
        for shift, entry in sorted(e10.items(), key=lambda item: float(item[0]))
    ]
    lags = [
        [f"{lag} h", _interval(entry["spearman"])]
        for lag, entry in sorted(e11["lags"].items(), key=lambda item: int(item[0]))
    ]
    return [
        "## The clocks (E10, E11)",
        "",
        "The dataset does not say which clock either file is in. E10 shifts the "
        f"mat's clock by {shift_text} hours and recomputes E1 and E2; +1 stands "
        "for a mat clock in UTC while the activity clock is local. An hour moves "
        "little of a night across midnight, so E10 shows how much the result "
        "depends on the clock and cannot find the right one.",
        "",
        *_table(["Shift", "Homes", "E1", "E2 (h)"], rows),
        "",
        "E11 reads only the inputs: the activity records in each local hour of "
        "the matched days, against the mat's minutes in bed in the same hour "
        "with its clock shifted by the lag. A person in bed moves little, so "
        "the correlation should be most negative where the clocks agree. "
        + (
            f"It is most negative at a lag of {e11['most_negative']} h. "
            if e11["most_negative"] is not None
            else "It could not be computed. "
        )
        + "Whatever it shows, every other estimand keeps the mat's clock as read.",
        "",
        *_table(["Lag", "Mean within-home Spearman"], lags),
        "",
        _figure("clocks", "Correlation of hourly activity with time in bed by lag"),
    ]


def _staged_part(results: Mapping[str, Any]) -> list[str]:
    e12 = results["E12_staged_asleep"]
    others = [
        row["staged_asleep_share"]
        for home, row in results["homes"].items()
        if home not in e12["left_out"] and row["staged_asleep_share"] is not None
    ]
    floor = int(100 * min(others)) if others else None
    return [
        "## Without the homes the mat stages as mostly awake (E12)",
        "",
        f"In {_code(e12['left_out'])} the mat stages less than "
        f"{_percent_of(e12['share'], 0)} of the minutes in bed as asleep"
        + (f", where every other mat home stages at least {floor}%" if floor else "")
        + ". The cut was fixed from the mat's file alone, before the comparison. "
        "Without those homes:",
        "",
        *_table(
            ["Against", "Homes", "Spearman", "Pearson", "Difference (h)"],
            [
                [
                    _REFERENCE_TITLES[reference],
                    len(e12["homes"]),
                    _interval(e12[reference]["spearman"]),
                    _interval(e12[reference]["pearson"]),
                    _interval(e12[reference]["mean_difference"], signed=True),
                ]
                for reference in REFERENCES
            ],
        ),
    ]


def _simulated_part(simulated: Mapping[str, Any], minimum: int) -> list[str]:
    results = simulated["results"]
    sleep = results[SLEEP]
    homes = len(results["homes"])
    return [
        "## The simulated reference (S1)",
        "",
        f"**The evidence is simulated.** {homes} simulated homes reduced to "
        "their event sensors, set against the simulator's true hours of sleep: "
        "what the estimands give when the observation model is the one that "
        f"made the data. {results['included']} homes had at least {minimum} "
        "matched days.",
        "",
        *_table(
            ["Estimand", "Value"],
            [
                ["Spearman, mean within home", _interval(sleep["spearman"])],
                ["Pearson, mean within home", _interval(sleep["pearson"])],
                ["difference, hours", _interval(sleep["mean_difference"], signed=True)],
                ["limits of agreement, hours", _limits(sleep["limits_of_agreement"])],
                [
                    "median within-home SD of the true hours",
                    _n(sleep["median_within_home_sd_of_truth"]),
                ],
            ],
        ),
    ]


def render_page(payload: Mapping[str, Any], simulated: Mapping[str, Any]) -> str:
    """The results page, from the TIHM record and the simulated record."""
    configuration, results = payload["configuration"], payload["results"]
    changed = configuration.get("code_changed_since_the_freeze") or {}
    changed_names = [name for names in changed.values() for name in names]
    between = configuration.get("between_the_freeze_and_the_run") or []
    dirty = payload.get("environment", {}).get("git_dirty") != "false"
    minimum = int(configuration["min_matched_days"])
    rule_hours = float(configuration["rule_hours"])
    lines = [
        "# The pipeline's hours of sleep beside a sleep mat: results",
        "",
        "Generated by `sensor_modeling.datasets.sleep_mat_summary.render_page` "
        f"from `{RECORD_FILE}` and `{SIMULATED_FILE}`, under "
        "[the protocol](SLEEP_MAT_PROTOCOL.md), "
        f"`{configuration['protocol_sha256']}`.",
        "",
        "**A measurement on a real cohort.** The pipeline ran on the TIHM homes "
        "that have a sleep mat with the alert-burden protocol's frozen settings; "
        "nothing in it was changed or fitted. The records hold per-home "
        "summaries and no day's values.",
        "",
        f"- **Commit.** `{payload['environment'].get('git_commit', '')[:7]}`"
        + (", from a modified tree." if dirty else ", from a clean tree."),
        (
            "- **The code.** The sources, libraries and defaults recorded at the "
            "freeze are those the run used."
            if not changed_names
            else "- **The code.** These had changed by the run: "
            + _code(changed_names)
            + "."
        ),
    ]
    if between:
        lines += [
            "",
            "## Between the freeze and the run",
            "",
            *[f"- {_sentence(item)}" for item in between],
        ]
    lines += [
        "",
        *_check_part(results, rule_hours),
        "",
        *_criteria_part(results),
        "",
        *_agreement_part(results, rule_hours),
        "",
        *_homes_part(results, minimum),
        "",
        *_deviations_part(results, minimum),
        "",
        *_alerts_part(results),
        "",
        *_silent_part(results),
        "",
        *_clocks_part(results, configuration["shifts_hours"]),
        "",
        *_staged_part(results),
        "",
        *_simulated_part(simulated, minimum),
        "",
        "## Acknowledgement",
        "",
        "TIHM is by Palermo et al., *Scientific Data* 10, 606 (2023), under CC "
        "BY 4.0. Surrey and Borders Partnership NHS Foundation Trust and Howz "
        "are acknowledged, as the dataset asks. The dataset is not "
        "redistributed here.",
        "",
    ]
    return "\n".join(lines)
