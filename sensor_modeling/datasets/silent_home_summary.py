"""The silent-home pages, generated from the frozen protocol and the records.

:func:`render_protocol` writes ``docs/SILENT_HOME_PROTOCOL.md`` from the frozen
protocol file alone. It reports no result.

:func:`render_page` writes ``docs/SILENT_HOME_RESULTS.md`` from the record of
the protocol's test on simulated homes and, in a part of its own, from the
record of the description on TIHM. Every home is shown.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from .silent_home_protocol import (
    CHANGE,
    CHANGE_AFTER_OUTAGE,
    FLEET_ARMS,
    IN_TIME,
    LATE,
    OFF,
    OUTAGE,
    RESULT_SCHEMA,
    SHORT_OUTAGE,
    STABLE,
    TIHM_SCHEMA,
    SilentHomeProtocol,
    declared_protocol,
)

PROTOCOL_FILE = "artifacts/silent_home/silent_home_protocol.json"
RECORD_FILE = "artifacts/silent_home/silent-home.json"
TIHM_RECORD_FILE = "artifacts/silent_home/silent-home-tihm.json"
PROTOCOL_PAGE = "SILENT_HOME_PROTOCOL.md"
RESULTS_PAGE = "SILENT_HOME_RESULTS.md"
FIGURE_DIR = "figures"
FIGURE_PREFIX = "silent-home"


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


def _bullets(entries: Mapping[str, Any]) -> list[str]:
    return [
        f"- **{_title(name)}.** {_sentence(str(text))}"
        for name, text in entries.items()
    ]


def _code(names: Sequence[Any]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _criterion(name: str) -> tuple[str, str]:
    """Split ``C1_the_outage_raises_alerts`` into its number and its claim."""
    number, _, claim = name.partition("_")
    return number, _title(claim)


def render_protocol(payload: Mapping[str, Any]) -> str:
    """The protocol page, from the frozen protocol file's content."""
    rule, simulator, fleet = payload["the_rule"], payload["simulator"], payload["fleet"]
    pipeline, tihm, groups = (
        simulator["pipeline"],
        payload["tihm"],
        simulator["outage_groups"],
    )
    conditions = list(rule["conditions"])
    run = {(arm, condition) for arm, condition in simulator["runs"]}
    through_the_pipeline = [
        arm for arm in simulator["arms"] if any(a == arm for a, _ in run)
    ]
    lines = [
        "# The silent-home rule: the protocol",
        "",
        "A home of event sensors that stops reporting is read as a home asleep. "
        "The [TIHM run](TIHM_ALERT_BURDEN_RESULTS.md) showed it, and an opt-in "
        f"rule, `{rule['setting']}`, was built from what it showed. This page "
        "is the protocol for finding out whether the rule does what it was "
        "built for, generated entirely from the frozen file "
        f"`{PROTOCOL_FILE}` by "
        "`sensor_modeling.datasets.silent_home_summary.render_protocol`.",
        "",
        f"**Status: {payload['status']}.**",
        "",
        f"- **Protocol digest.** `{payload['protocol_sha256']}`.",
        f"- **The question.** {_sentence(payload['question'])}",
        "- **This page reports no result.** The scoring run checks the frozen "
        "file before it runs.",
        "",
        "## What had been seen",
        "",
        "The rule was designed from TIHM, so the homes that showed the problem "
        "cannot also be the evidence that the rule solves it. The test is made "
        "on simulated homes that had not been run with the rule on.",
        "",
        *[f"- {_sentence(item)}" for item in payload["inspected_before"]],
        "",
        "## The rule",
        "",
        f"`{rule['setting']}` is {rule['default']} by default. When it is set "
        "and no sensor of a home has reported for that long, the home is "
        "treated as not observed from its last observation on, the days that "
        "lost any time that way are refused by the baseline, and one "
        "data-quality alert is raised. See [the inference contract]"
        "(inference.md).",
        "",
        *_table(
            ["Condition", "What it is"],
            [
                [
                    f"`{name}`" + (", primary" if name == rule["primary"] else ""),
                    _sentence(text),
                ]
                for name, text in rule["conditions"].items()
            ],
        ),
        "",
        f"The horizons are {rule['horizons']}.",
        "",
        "## The simulated homes",
        "",
        f"**{_sentence(simulator['evidence'])}**",
        "",
        *_table(
            ["Field", "Value"],
            [
                ["homes", simulator["homes"]],
                ["seed root", f"`{simulator['seed_root']}`"],
                ["seeds", simulator["seeds"]],
                ["days in each record", simulator["days"]],
                ["first day", f"{simulator['start']}, day 0"],
                ["timezone", f"`{simulator['timezone']}`"],
                ["household", simulator["household"]],
                ["sensors", _code(simulator["sensors"])],
                ["sensors left out", _code(simulator["sensors_left_out"])],
                ["replications", simulator["replications"]],
            ],
        ),
        "",
        f"- **Why sensors are left out.** {_sentence(simulator['why_left_out'])}",
        f"- **Delivery.** {_sentence(simulator['delivery'])}",
        f"- **Pairing.** {_sentence(simulator['pairing'])}",
        "",
        "### The pipeline",
        "",
        f"{_sentence(pipeline['what'])}",
        "",
        *_table(
            ["Setting", "Value"],
            [
                ["step", f"{pipeline['step_minutes']} minutes"],
                ["baseline features", _code(pipeline["features"])],
                ["minimum day coverage", pipeline["min_day_coverage"]],
                ["minimum day observed", pipeline["min_day_observed"]],
                *[
                    [f"baseline {k.replace('_', ' ')}", v]
                    for k, v in pipeline["baseline"].items()
                ],
                *[
                    [f"alert policy {k.replace('_', ' ')}", v]
                    for k, v in pipeline["alert_policy"].items()
                ],
            ],
        ),
        "",
        "### Arms",
        "",
        *_table(
            ["Arm", "What is injected"],
            [[f"`{arm}`", _sentence(text)] for arm, text in simulator["arms"].items()],
        ),
        "",
        "### Runs",
        "",
        "Every home is run through the pipeline once for each mark.",
        "",
        *_table(
            ["Arm", *[f"`{condition}`" for condition in conditions]],
            [
                [
                    f"`{arm}`",
                    *[
                        "run" if (arm, condition) in run else ""
                        for condition in conditions
                    ],
                ]
                for arm in through_the_pipeline
            ],
        ),
        "",
        f"That is {len(simulator['runs'])} runs of the pipeline for each home.",
        "",
        "### When an outage begins",
        "",
        f"{_sentence(groups['why'])}",
        "",
        *_table(
            ["Group", "The outage", "Homes"],
            [
                [f"`{name}`", _sentence(groups[name]), groups["homes"][name]]
                for name in groups["homes"]
            ],
        ),
        "",
        f"The outages drawn begin between day {simulator['outage_days']['first']} "
        f"and day {simulator['outage_days']['last']}.",
        "",
        "## Definitions",
        "",
        *_bullets(payload["definitions"]),
        "",
        "## Estimands",
        "",
        *_table(
            ["Estimand", "What it is"],
            [
                [name.replace("_", " "), _sentence(text)]
                for name, text in payload["estimands"].items()
            ],
        ),
        "",
        "## Criteria",
        "",
        "Fixed before any simulated home was run with the rule on.",
        "",
        *_table(
            ["Criterion", "Claim", "How it is decided"],
            [
                [*_criterion(name), _sentence(text)]
                for name, text in payload["criteria"].items()
            ],
        ),
        "",
        "## The fleet check",
        "",
        f"{_sentence(fleet['what'])}",
        "",
        *_table(
            ["Setting", "Value"],
            [
                ["horizon in hours", f"{fleet['horizon_hours']:g}"],
                ["fraction of monitored homes", fleet["fraction"]],
                ["fewest homes", fleet["min_homes"]],
                ["hours between assessments", f"{fleet['every_hours']:g}"],
            ],
        ),
        "",
        f"- **What it does not test.** {_sentence(fleet['what_it_does_not_test'])}",
        "",
        "## Uncertainty",
        "",
        *_bullets(payload["bootstrap"]),
        "",
        "## What the simulator cannot show",
        "",
        *[f"- {_sentence(item)}" for item in payload["what_the_simulator_cannot_show"]],
        "",
        "## TIHM",
        "",
        f"**{_sentence(tihm['standing'])}**",
        "",
        *_table(
            ["Field", "Value"],
            [
                ["dataset", tihm["dataset"]],
                ["citation", tihm["citation"]],
                ["licence", tihm["licence"]],
                ["archive SHA-256", f"`{tihm['archive_sha256']}`"],
                ["step", f"{tihm['step_minutes']} minutes"],
                ["conditions", _code(tihm["conditions"])],
                ["everything else", tihm["everything_else"]],
                ["fleet check", tihm["fleet"]],
            ],
        ),
        "",
        f"- **Check.** {_sentence(tihm['check'])}",
        "- **Reported.** "
        + _sentence("; ".join(str(item) for item in tihm["reported"])),
        f"- **Not reported.** {_sentence(tihm['not_reported'])}",
        "- **Redistribution.** None: the run reads an extracted download and "
        "verifies every file's digest.",
        "",
        "## Reporting",
        "",
        f"{_sentence(payload['reporting'])}",
        "",
        "## The homes",
        "",
        "Each home's seed, and the day and local hour its own outage begins " "at.",
        "",
        *_table(
            ["Seed", "Day", "Hour", "Group"],
            [
                [
                    f"`{home['seed']}`",
                    home["day"],
                    f"{home['hour']:02d}:00",
                    f"`{home['group']}`",
                ]
                for home in simulator["outages"]
            ],
        ),
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# The results page
# ----------------------------------------------------------------------------
def _n(value: Any, digits: int = 2) -> str:
    """A number as the page shows it; a missing one as a dash."""
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def _signed(value: Any, digits: int = 2) -> str:
    return "–" if value is None else f"{value:+.{digits}f}"


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


def _share(entry: Mapping[str, Any]) -> str:
    """A share of homes with its count and Wilson interval."""
    if entry.get("estimate") is None:
        return "–"
    text = f"{entry['count']} of {entry['homes']} ({entry['estimate']:.0%})"
    interval = entry.get("interval")
    if interval:
        text += f" [{interval['low']:.0%}, {interval['high']:.0%}]"
    return text


def _spread_text(entry: Mapping[str, Any], digits: int = 1) -> str:
    if not entry.get("homes"):
        return "–"
    return (
        f"{_n(float(entry['median']), digits)} "
        f"({_n(float(entry['min']), digits)} to {_n(float(entry['max']), digits)})"
    )


def _count(number: int, noun: str) -> str:
    """A count with its noun, in the singular when there is one."""
    return f"{number:,} {noun}" if number == 1 else f"{number:,} {noun}s"


def _moment(text: str | None) -> str:
    """An ISO moment to the minute, in UTC as recorded."""
    return "–" if not text else text[:16].replace("T", " ")


def _figure(name: str, alt: str) -> str:
    return f"![{alt}]({FIGURE_DIR}/{FIGURE_PREFIX}-{name}.svg)"


def _conditions(configuration: Mapping[str, Any]) -> list[str]:
    """The conditions in the protocol's order: the rule off, then each horizon."""
    rule = configuration["the_rule"]
    hours = {
        name: float(text.rsplit(" ", 2)[-2])
        for name, text in rule["conditions"].items()
        if name != OFF
    }
    others = sorted((n for n in hours if n != rule["primary"]), key=hours.__getitem__)
    return [OFF, rule["primary"], *others]


def _ordered(entries: Mapping[str, Any], order: Sequence[str]) -> list[tuple[str, Any]]:
    """The entries named in *order*, in that order."""
    return [(name, entries[name]) for name in order if name in entries]


def _verdicts_part(payload: Mapping[str, Any], lines: list[str]) -> None:
    results, configuration = payload["results"], payload["configuration"]
    judged = results["criteria"]
    primary = configuration["the_rule"]["primary"]
    outage, detection = results["outage"], results["detection"]
    stable, short = results["stable"][primary], results["short_outage"][primary]
    reported = results["reporting"][primary]["reported"]
    fleet = results["fleet"]
    basis = {
        "C1": f"E1 is {_interval(outage['off']['excess'])} alerts per home",
        "C2": f"E3 is {_interval(outage[primary]['removed'])} and E2 is "
        f"{_interval(outage[primary]['excess'])}, against a margin of "
        f"{judged['C2']['margin']:g}",
        "C3": f"the rule changes an alert in "
        f"{stable['homes_in_which_the_rule_changes_an_alert']} homes and raises "
        f"{stable['silence_alerts']} silence alerts",
        "C4": f"{_share(detection['change']['off']['detected'])} detected with "
        f"the rule off; on minus off is "
        f"{_interval(detection['change']['on_minus_off'], signed=True)}, against "
        f"a margin of {judged['C4']['margin']:g}",
        "C5": f"{_share(detection['change_after_outage']['off']['detected'])} "
        "detected with the rule off; on minus off is "
        f"{_interval(detection['change_after_outage']['on_minus_off'], signed=True)}"
        f", against a margin of {judged['C5']['margin']:g}",
        "C6": f"{_share(reported)} reported, against {judged['C6']['needed']:.0%}",
        "C7": "stretches found: "
        + ", ".join(
            f"{count} in `{arm}`"
            for arm, count in _ordered(judged["C7"]["stretches"], FLEET_ARMS)
        )
        + f"; {fleet['outage_common']['stretches_overlapping_the_outage']} "
        "overlapping the outage",
        "C8": f"the rule changes an alert in "
        f"{short['homes_in_which_the_rule_changes_an_alert']} homes and raises a "
        f"silence alert in {short['homes_with_a_silence_alert']}",
    }
    claims = {
        name.partition("_")[0]: _title(name.partition("_")[2])
        for name in configuration["criteria"]
    }
    lines.extend(
        [
            "## The criteria",
            "",
            "Each was fixed, with its margin, before any simulated home had been "
            f"run with the rule on. They are decided at the primary horizon, "
            f"`{primary}`: C1, C2, C4 and C5 on intervals that resample homes, "
            "the others on counts.",
            "",
            *_table(
                ["Criterion", "Claim", "Verdict", "What decided it"],
                [
                    [
                        name,
                        claims[name],
                        f"**{entry['verdict']}**",
                        _sentence(basis[name]),
                    ]
                    for name, entry in sorted(judged.items())
                ],
            ),
            "",
            *_reading(payload),
        ]
    )


def _reading(payload: Mapping[str, Any]) -> list[str]:
    """What each verdict's word does and does not say, from the numbers."""
    results, configuration = payload["results"], payload["configuration"]
    judged = results["criteria"]
    primary = configuration["the_rule"]["primary"]
    lines = [
        "- **C1 and C8 are not successes or failures of the rule.** C1 says "
        "whether the simulator shows the problem at all, and C8 states a limit "
        "of the rule, whichever way it comes out.",
    ]
    excess = results["outage"][primary]["excess"]
    if excess["interval"] and excess["interval"]["high"] < 0.0:
        lines.append(
            "- **E2 lies below zero.** With the rule on the outage arm raises "
            "fewer alerts in its window than the same homes with nothing "
            "injected. C2 asks only that E2 lie below its margin, so it counts "
            "this as success. Whether it is a loss of sensitivity after an "
            "outage is not settled; what the record holds on it is the two "
            "detection arms set side by side, and the description after the "
            "run."
        )
    for name, arm in (("C4", CHANGE), ("C5", CHANGE_AFTER_OUTAGE)):
        if judged[name]["verdict"] != "inconclusive":
            continue
        difference = results["detection"][arm]["on_minus_off"]
        interval = difference["interval"]
        fewer = interval["high"] < 0.0
        lines.append(
            f"- **{name} is inconclusive, which is not no difference.** The "
            f"interval of the difference, [{_signed(interval['low'])}, "
            f"{_signed(interval['high'])}], lies on both sides of the margin of "
            f"{judged[name]['margin']:g}"
            + (
                " and excludes zero. With the rule on, fewer homes meet the "
                "detection definition; whether the loss is within the margin "
                "is not decided."
                if fewer
                else ". Whether the rule costs detection is not decided."
            )
        )
    return [*lines, ""]


def _against_stable(
    outage: Mapping[str, Any], order: Sequence[str], timeline: Mapping[str, Any]
) -> list[str]:
    """In words: each condition's alerts beside the stable arm's, and when."""
    pairs = [
        f"{entry['alerts_in_the_window']:,} against "
        f"{entry['stable_alerts_in_the_window']:,} under `{condition}`"
        for condition, entry in _ordered(outage, order)
    ]
    days = timeline["days_since_the_outage_began"]
    off = timeline["alerts"][OFF][OUTAGE]
    still = timeline["alerts"][OFF][STABLE]

    def during(counts: Sequence[int], first: int, last: int) -> int:
        return sum(c for day, c in zip(days, counts) if first <= day < last)

    end = len(days)
    return [
        "The outage arm's alerts in the window, against the same homes' with "
        "nothing injected: " + "; ".join(pairs) + ".",
        "",
        f"With the rule off, {during(off, 0, 7):,} of the {sum(off):,} alerts in "
        "the window are raised in the first seven days after the outage began "
        f"and {during(off, 21, end):,} from the twenty-second day on. The same "
        f"homes with nothing injected raise {during(still, 0, 7):,} and "
        f"{during(still, 21, end):,} in those days.",
    ]


def _homes_below(rows: Mapping[str, Mapping[str, Any]], condition: str) -> int:
    """Homes whose outage arm raises fewer alerts in the window than their stable arm."""
    return sum(
        1
        for row in rows.values()
        if row[f"{OUTAGE}/{condition}/alerts_in_the_outage_window"]
        < row[f"{STABLE}/{condition}/alerts_in_the_outage_window"]
    )


def _first_day_refused(
    rows: Mapping[str, Mapping[str, Any]], condition: str, group: str | None = None
) -> int:
    """Homes in which the day the outage began on was refused by the baseline."""
    return sum(
        1
        for row in rows.values()
        if row.get(f"{OUTAGE}/{condition}/first_outage_day_refused")
        and (group is None or row["group"] == group)
    )


def _outage_part(payload: Mapping[str, Any], lines: list[str]) -> None:
    results, configuration = payload["results"], payload["configuration"]
    rule = configuration["the_rule"]
    outage, mcse = results["outage"], payload.get("mcse") or {}
    groups = configuration["simulator"]["outage_groups"]
    primary = rule["primary"]
    order = _conditions(configuration)
    lines.extend(
        [
            "## What an outage raises, and what the rule leaves (E1, E2, E3)",
            "",
            "A home's behavioural alerts in its outage window, minus the same "
            "home's alerts in the same window when nothing was injected.",
            "",
            *_table(
                [
                    "Condition",
                    "Excess alerts per home",
                    "Alerts in the window",
                    "In the stable arm",
                    "Homes with more alerts than their stable arm",
                    "Homes with fewer",
                    "Removed by the rule, per home",
                ],
                [
                    [
                        f"`{condition}`",
                        _interval(entry["excess"]),
                        _n(entry["alerts_in_the_window"]),
                        _n(entry["stable_alerts_in_the_window"]),
                        _n(entry["homes_with_an_excess"]),
                        _n(_homes_below(results["per_home"], condition)),
                        _interval(entry.get("removed")),
                    ]
                    for condition, entry in _ordered(outage, order)
                ],
            ),
            "",
            f"E1 is the first row, E2 the row of `{primary}` and E3 its last "
            "column; the other row is the other horizon (S1). The homes with "
            "more and with fewer alerts than their stable arm are counted from "
            "each home's own values. Monte Carlo standard errors: "
            + ", ".join(
                f"{name.replace('_', ' ')} {value:.3f}"
                for name, value in mcse.items()
                if name.startswith(("E1", "E2", "E3", "S1"))
            )
            + ".",
            "",
            *_against_stable(outage, order, results["timeline"]),
            "",
            _figure(
                "timeline",
                "Behavioural alerts per home on each day since the outage began, "
                "with the rule off and on",
            ),
            "",
            "### By when the outage begins (E2 by group)",
            "",
            f"{_sentence(groups['why'])} No criterion is stated on the groups.",
            "",
            *_table(
                [
                    "Group",
                    "Homes",
                    "Condition",
                    "Excess per home",
                    "Its Monte Carlo standard error",
                    "Removed per home",
                    "Its Monte Carlo standard error",
                ],
                [
                    [
                        f"`{group}`",
                        entry["by_group"][group]["homes"],
                        f"`{condition}`",
                        _interval(entry["by_group"][group]),
                        _n(entry["by_group"][group].get("mcse"), 3),
                        _interval((entry.get("removed_by_group") or {}).get(group)),
                        _n(
                            (
                                (entry.get("removed_by_group") or {}).get(group) or {}
                            ).get("mcse"),
                            3,
                        ),
                    ]
                    for group in (IN_TIME, LATE)
                    for condition, entry in _ordered(outage, order)
                ],
            ),
            "",
        ]
    )
    rows = payload["results"]["per_home"]
    lines.extend(
        [
            "The groups are declared by the hour an outage begins at, and not by "
            f"what the pipeline did. Under `{primary}` the day on which the "
            "outage began was refused by the baseline in "
            + " and in ".join(
                f"{_first_day_refused(rows, primary, group)} of the "
                f"{groups['homes'][group]} homes of `{group}`"
                for group in (IN_TIME, LATE)
            )
            + ". A silence is dated from the home's last observation, which "
            "precedes the outage by minutes to hours, and a day is closed by "
            "the first step after midnight, so an outage declared late can "
            "still be seen before its day closes.",
            "",
            *[
                f"Under `{condition}` the day on which the outage began was "
                f"refused in {_first_day_refused(rows, condition)} of the "
                f"{len(rows)} homes. The protocol's S1 says that at a horizon of "
                "a whole day no outage is seen before the day it begins on has "
                "closed; the record does not bear that out, for the same reason."
                for condition in order[2:]
                if _first_day_refused(rows, condition)
            ],
            "",
            _figure(
                "start-hour",
                "Mean excess alerts per home by the hour the outage begins at, "
                "with the rule off and on",
            ),
            "",
        ]
    )


def _stable_part(payload: Mapping[str, Any], lines: list[str]) -> None:
    results = payload["results"]
    stable = results["stable"]
    order = _conditions(payload["configuration"])
    lines.extend(
        [
            "## A home in which nothing is wrong (E4)",
            "",
            f"{_n(stable['person_days'])} person-days.",
            "",
            *_table(
                [
                    "Condition",
                    "Behavioural alerts",
                    "Per person-day",
                    "Homes in which the rule changes an alert",
                    "Silence alerts",
                    "Days refused",
                ],
                [
                    [
                        f"`{condition}`",
                        _n(entry["behavioural_alerts"]),
                        _interval(entry["per_person_day"], digits=4),
                        _n(entry.get("homes_in_which_the_rule_changes_an_alert")),
                        _n(entry.get("silence_alerts")),
                        _n(entry.get("days_refused")),
                    ]
                    for condition, entry in _ordered(stable, order)
                ],
            ),
            "",
        ]
    )


def _never_acted(results: Mapping[str, Any], arm: str, primary: str) -> list[str]:
    """Say so when the rule did nothing in an arm, so the comparison is of one run."""
    totals = results["runs"][f"{arm}/{primary}"]
    if totals["silence_alerts"] or totals["days_refused_for_silence"]:
        return []
    return [
        "In this arm the rule never acted: no home was silent for the horizon, "
        "so the two conditions gave the same runs and the difference could not "
        "have been anything but zero. It says the rule does nothing where no "
        "simulated home is silent, which the stable arm says too.",
        "",
    ]


def _detection_part(payload: Mapping[str, Any], lines: list[str]) -> None:
    results, configuration = payload["results"], payload["configuration"]
    mcse = payload.get("mcse") or {}
    lines.extend(
        [
            "## Detecting a real change (E5, E6)",
            "",
            f"{_sentence(configuration['definitions']['detected'])}",
            "",
        ]
    )
    for name, arm in (("E5", "change"), ("E6", "change_after_outage")):
        entry = results["detection"][arm]
        difference = entry["on_minus_off"]
        conditions = [c for c in _conditions(configuration) if c in entry]
        lines.extend(
            [
                f"### `{arm}` ({name})",
                "",
                *_table(
                    [
                        "Condition",
                        "Homes detected",
                        "Median delay in days",
                        "Detections",
                        f"False detections, in `{entry[conditions[0]]['false_detections_in']}`",
                    ],
                    [
                        [
                            f"`{condition}`",
                            _share(entry[condition]["detected"]),
                            _interval(entry[condition]["delay_days"], digits=1),
                            _n(entry[condition]["delay_days"]["detections"]),
                            _share(entry[condition]["false_detections"]),
                        ]
                        for condition in conditions
                    ],
                ),
                "",
                f"On minus off, paired by home: {_interval(difference, signed=True)}"
                f", Monte Carlo standard error "
                f"{_n(mcse.get(f'{name}_on_minus_off'), 3)}. "
                f"{difference['detected_only_with_the_rule_on']} homes were "
                "detected only with the rule on and "
                f"{difference['detected_only_with_the_rule_off']} only with it "
                "off.",
                "",
                *_never_acted(results, arm, configuration["the_rule"]["primary"]),
            ]
        )

    primary = configuration["the_rule"]["primary"]
    plain, after = (
        results["detection"][CHANGE],
        results["detection"][CHANGE_AFTER_OUTAGE],
    )
    lines.extend(
        [
            "### The two arms side by side",
            "",
            "**Put together after the run.** Each number below is part of E5 or "
            "E6; setting them side by side is not an estimand, and no criterion "
            "compares the two arms. That the detections after an outage with "
            "the rule off may be alerts the outage itself raises was suggested "
            "before the run by a home outside the protocol: see "
            "[Between the freeze and the run](#between-the-freeze-and-the-run).",
            "",
            *[
                f"The record's note on the pairing: {note} So the last column, "
                "which is counted in the stable home's record with the outage, "
                "is from another realisation of the home than the first two."
                for note in payload["notes"]
                if "share a seed" in note
            ],
            "",
            *_table(
                [
                    "Condition",
                    "Detected with no outage",
                    "Detected after an outage",
                    "An outage and no change meets the definition",
                ],
                [
                    [
                        f"`{condition}`",
                        _share(plain[condition]["detected"]),
                        _share(after[condition]["detected"]),
                        _share(after[condition]["false_detections"]),
                    ]
                    for condition in (OFF, primary)
                ],
            ),
            "",
        ]
    )


def _reporting_part(payload: Mapping[str, Any], lines: list[str]) -> None:
    results = payload["results"]
    reporting = dict(
        _ordered(results["reporting"], _conditions(payload["configuration"]))
    )
    lines.extend(
        [
            "## Is the silence reported (E7)",
            "",
            *_table(
                [
                    "Condition",
                    "Homes reported",
                    "Hours to the first silence alert, median (range)",
                    "Silence alerts per home inside the outage, median (range)",
                    "Silence alerts outside it, all homes",
                    "Days refused per home, median (range)",
                ],
                [
                    [
                        f"`{condition}`",
                        _share(entry["reported"]),
                        _spread_text(entry["hours_to_the_first_silence_alert"]),
                        _spread_text(entry["silence_alerts_per_home"], 0),
                        _n(entry["silence_alerts_outside_the_outage"]),
                        _spread_text(entry["days_refused_per_home"], 0),
                    ]
                    for condition, entry in reporting.items()
                ],
            ),
            "",
            "A silence is dated from the home's last observation, which precedes "
            "the outage by minutes to hours. That is why the first alert can "
            "come sooner than a horizon after the outage began.",
            "",
            "Homes by the number of their days the baseline refused: "
            + "; ".join(
                f"`{condition}`: "
                + ", ".join(
                    f"{count} with {days}"
                    for days, count in sorted(
                        entry["days_refused_distribution"].items(),
                        key=lambda item: int(item[0]),
                    )
                )
                for condition, entry in reporting.items()
            )
            + ".",
            "",
        ]
    )


def _fleet_part(payload: Mapping[str, Any], lines: list[str]) -> None:
    results, configuration = payload["results"], payload["configuration"]
    fleet = results["fleet"]
    common = fleet["outage_common"]
    lines.extend(
        [
            "## A silence that homes share (E8)",
            "",
            f"The fleet check calls a common cause when at least "
            f"{fleet['config']['fraction']:.0%} of the monitored homes, and at "
            f"least {fleet['config']['min_homes']}, have been silent for "
            f"{fleet['config']['horizon_hours']:g} hours.",
            "",
            *_table(
                [
                    "Arm",
                    "Stretches of common silence",
                    "Largest share of homes silent at one assessment",
                    "Assessments",
                ],
                [
                    [
                        f"`{arm}`",
                        len(entry["stretches"]),
                        f"{entry['largest_share_silent']:.0%}",
                        _n(entry["assessments"]),
                    ]
                    for arm, entry in _ordered(fleet, FLEET_ARMS)
                ],
            ),
            "",
        ]
    )
    for stretch in common["stretches"]:
        lines.append(
            f"- In `outage_common`: silent since {_moment(stretch['since'])}, "
            f"called common at {_moment(stretch['detected'])}, last called so "
            f"at {_moment(stretch['until'])}; {stretch['homes']} of "
            f"{stretch['monitored']} homes at most. The outage ran from "
            f"{_moment(common['outage']['begin'])} to "
            f"{_moment(common['outage']['end'])}, in UTC."
        )
    hours = common["hours_to_the_first_assessment_that_calls_it_common"]
    if hours is not None:
        lines.append(
            f"- The first assessment to call it common came {_n(float(hours), 1)} "
            "hours after the outage began."
        )
    lines.extend(
        [
            f"- {_sentence(configuration['fleet']['what_it_does_not_test'])}",
            "",
            _figure(
                "fleet",
                "Share of homes silent at each assessment, in the three arms the "
                "fleet check reads",
            ),
            "",
        ]
    )


def _short_part(
    payload: Mapping[str, Any], protocol: SilentHomeProtocol, lines: list[str]
) -> None:
    results, configuration = payload["results"], payload["configuration"]
    short = results["short_outage"]
    primary = configuration["the_rule"]["primary"]
    on, mcse = short[primary], payload.get("mcse") or {}
    seen = []
    for seed in on["homes_with_a_silence_alert_listed"]:
        begin, end = protocol.short_outage(int(seed))
        alerts = results["raw"]["homes"][str(seed)][f"{SHORT_OUTAGE}/{primary}"][
            "silence_alerts"
        ]
        first = datetime.fromisoformat(alerts[0])
        gap = (first - end) / timedelta(hours=1)
        seen.append(
            f"In home `{seed}` the alert came "
            f"{(first - begin) / timedelta(hours=1):.1f} hours after the outage "
            f"began, {abs(gap):.1f} hours {'after' if gap >= 0 else 'before'} "
            "it ended: the home's last observation before the outage and its "
            "first after it were further apart than the horizon."
        )
    lines.extend(
        [
            "## An outage shorter than the horizon (E9)",
            "",
            *_table(
                ["Condition", "Excess alerts per home"],
                [
                    [f"`{condition}`", _interval(entry["excess"])]
                    for condition, entry in _ordered(short, _conditions(configuration))
                ],
            ),
            "",
            f"With the rule on it changes a behavioural alert in "
            f"{_count(on['homes_in_which_the_rule_changes_an_alert'], 'home')}, "
            f"raises a silence alert in "
            f"{_count(on['homes_with_a_silence_alert'], 'home')} and has "
            f"{_count(on['days_refused'], 'day')} refused. Monte Carlo standard "
            f"errors of the excess: {_n(mcse.get('E9_off'), 3)} with the rule "
            f"off and {_n(mcse.get('E9_on'), 3)} with it on.",
            "",
            *seen,
            *([""] if seen else []),
        ]
    )


def _kinds(alerts: Sequence[Sequence[Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for alert in alerts:
        name = f"{alert[1]}, {str(alert[2]).replace('_', ' ') or 'not kept'}"
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _within(
    alerts: Sequence[Sequence[Any]], begin: datetime, end: datetime
) -> list[Any]:
    return [
        alert for alert in alerts if begin <= datetime.fromisoformat(alert[0]) < end
    ]


def _both(rows: Mapping[str, Mapping[str, Any]], condition: str) -> list[int]:
    """Homes by whether the change is detected with no outage and after one."""
    key = "meets_the_detection_definition"
    counts = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for row in rows.values():
        counts[
            (
                bool(row[f"{CHANGE}/{condition}/{key}"]),
                bool(row[f"{CHANGE_AFTER_OUTAGE}/{condition}/{key}"]),
            )
        ] += 1
    return list(counts.values())


def _later(rows: Mapping[str, Mapping[str, Any]], primary: str) -> list[str]:
    """Among homes detected after an outage either way, which condition is sooner."""
    later = earlier = same = 0
    for row in rows.values():
        off = row[f"{CHANGE_AFTER_OUTAGE}/{OFF}/delay_days"]
        on = row[f"{CHANGE_AFTER_OUTAGE}/{primary}/delay_days"]
        if off is None or on is None:
            continue
        later += on > off
        earlier += on < off
        same += on == off
    return [
        f"Of the {later + earlier + same} homes detected after an outage under "
        f"both conditions, the first alert with the rule on comes later than "
        f"with it off in {later}, earlier in {earlier} and at the same moment "
        f"in {same}."
    ]


def _described_part(
    payload: Mapping[str, Any], protocol: SilentHomeProtocol, lines: list[str]
) -> None:
    """What kind of alerts stand behind the counts: described after the run."""
    results, configuration = payload["results"], payload["configuration"]
    primary = configuration["the_rule"]["primary"]
    raw = results["raw"]["homes"]
    detection = protocol.detection_window()
    rows: list[list[Any]] = []
    for label, arm, which in (
        ("its outage window", OUTAGE, "outage"),
        ("its outage window", STABLE, "outage"),
        ("the detection window", CHANGE, "detection"),
        ("the detection window", STABLE, "detection"),
        ("the detection window", CHANGE_AFTER_OUTAGE, "detection"),
        ("the detection window", OUTAGE, "detection"),
    ):
        for condition in (OFF, primary):
            found: list[Any] = []
            for seed, runs in raw.items():
                begin, end = (
                    protocol.window(int(seed), OUTAGE)
                    if which == "outage"
                    else detection
                )
                found += _within(
                    runs[f"{arm}/{condition}"]["behavioural_alerts"], begin, end
                )
            kinds = _kinds(found)
            rows.append(
                [
                    f"`{arm}`",
                    f"`{condition}`",
                    label,
                    len(found),
                    "; ".join(f"{count} {name}" for name, count in kinds.items())
                    or "none",
                ]
            )
    parts: dict[tuple[str, str], list[int]] = {}
    for condition in (OFF, primary):
        for arm in (OUTAGE, STABLE):
            first = rest = 0
            for seed, runs in raw.items():
                begin, end = protocol.window(int(seed), OUTAGE)
                split = protocol.outage(int(seed))[1] + timedelta(days=1)
                alerts = runs[f"{arm}/{condition}"]["behavioural_alerts"]
                first += len(_within(alerts, begin, split))
                rest += len(_within(alerts, split, end))
            parts[(condition, arm)] = [first, rest]
    kept = len(raw) - _first_day_refused(results["per_home"], primary)
    lines.extend(
        [
            "## What the alerts were (described after the run)",
            "",
            "**Not an estimand of the protocol.** The run kept, beside each "
            "behavioural alert, the verdict of the baseline that raised it. No "
            "criterion reads it. It is shown because a count of alerts does not "
            "say what was alerted about: an alert about sleep in the detection "
            "window counts as a detection whatever raised it. Each row counts "
            "every home's alerts of every feature in the window named.",
            "",
            *_table(
                ["Arm", "Condition", "Window", "Alerts", "By feature and verdict"],
                rows,
            ),
            "",
            "The detection definition, home by home, in the change arm and in "
            "the same home's change after an outage:",
            "",
            *_table(
                [
                    "Condition",
                    "Detected in both",
                    "Only with no outage",
                    "Only after an outage",
                    "In neither",
                ],
                [
                    [f"`{condition}`", *_both(results["per_home"], condition)]
                    for condition in (OFF, primary)
                ],
            ),
            "",
            *_later(results["per_home"], primary),
            "",
            "The outage window in two parts. With the rule on a refused day "
            "raises nothing, and the days an outage touches are refused "
            f"except, in {kept} of {len(raw)} homes, the day it began on. So the "
            "first part runs from the outage's beginning to a day after its "
            "end, and the second is the rest of the window.",
            "",
            *_table(
                [
                    "Condition",
                    "Arm",
                    "Alerts to a day after the outage's end",
                    "Alerts in the rest of the window",
                ],
                [
                    [f"`{condition}`", f"`{arm}`", *parts[(condition, arm)]]
                    for condition in (OFF, primary)
                    for arm in (OUTAGE, STABLE)
                ],
            ),
            "",
            f"Under `{primary}` the outage arm raises "
            f"{parts[(primary, OUTAGE)][1]:,} alerts in the rest of the window "
            "and the same homes with nothing injected "
            f"{parts[(primary, STABLE)][1]:,}. Whether a difference there is a "
            "loss of sensitivity after an outage is not settled. The nearest "
            "evidence is the two detection arms set side by side above, whose "
            "change begins in that part of the window. Why the alerts are "
            "missing, no arm can say.",
            "",
        ]
    )


def _between_part(payload: Mapping[str, Any], lines: list[str]) -> None:
    configuration = payload["configuration"]
    lines.extend(
        [
            "## Between the freeze and the run",
            "",
            "The protocol was frozen and pushed before any simulated home had "
            "been run with the rule on. What was run and read after that, and "
            "before the run, on homes that are not the protocol's:",
            "",
            *[
                f"- {_sentence(text)}"
                for text in configuration["between_the_freeze_and_the_run"]
            ],
            "- One more simulated home is not in the record's list, and should "
            "have been. A unit test committed before the run, "
            "`test_online_pipeline.py::TestSilentHome::"
            "test_where_there_are_canaries_they_speak_first`, runs the test "
            "suite's home of seed 2024, twelve days long with every sensor and "
            "an outage of 40 hours, with the rule on. It was read for when the "
            "silence alert comes and which days are refused. Twelve days are "
            "fewer than the baseline needs before it raises a behavioural "
            "alert, so it showed nothing about one.",
            "",
            "## What this does not show",
            "",
            *[
                f"- {_sentence(text)}"
                for text in configuration["what_the_simulator_cannot_show"]
            ],
            "",
        ]
    )


def _homes_part(payload: Mapping[str, Any], lines: list[str]) -> None:
    results, configuration = payload["results"], payload["configuration"]
    primary = configuration["the_rule"]["primary"]
    others = [c for c in _conditions(configuration)[2:] if c in results["outage"]]

    def excess(row: Mapping[str, Any], condition: str) -> int:
        return int(
            row[f"outage/{condition}/alerts_in_the_outage_window"]
            - row[f"stable/{condition}/alerts_in_the_outage_window"]
        )

    def flag(row: Mapping[str, Any], key: str) -> str:
        return "yes" if row[key] else "no"

    lines.extend(
        [
            "## Every simulated home",
            "",
            "Excess alerts are the home's alerts in its outage window minus its "
            f"stable arm's. The three columns after them are under `{primary}`. "
            "Each cell of the last two holds two answers: whether the change "
            "was detected with no outage, and whether it was detected after "
            "the outage.",
            "",
            *_table(
                [
                    "Seed",
                    "Outage begins",
                    "Excess, `off`",
                    f"Excess, `{primary}`",
                    *[f"Excess, `{c}`" for c in others],
                    "Silence alerts",
                    "Days refused",
                    "First outage day refused",
                    "Detected, `off`",
                    f"Detected, `{primary}`",
                ],
                [
                    [
                        f"`{seed}`",
                        f"day {row['outage_day']}, {row['outage_hour']:02d}:00",
                        excess(row, "off"),
                        excess(row, primary),
                        *[excess(row, c) for c in others],
                        row[f"outage/{primary}/silence_alerts"],
                        row[f"outage/{primary}/days_refused"],
                        flag(row, f"outage/{primary}/first_outage_day_refused"),
                        *[
                            flag(row, f"change/{c}/meets_the_detection_definition")
                            + " / "
                            + flag(
                                row,
                                f"change_after_outage/{c}/meets_the_detection_definition",
                            )
                            for c in ("off", primary)
                        ],
                    ]
                    for seed, row in sorted(
                        results["per_home"].items(), key=lambda item: int(item[0])
                    )
                ],
            ),
            "",
        ]
    )


def _tihm_part(payload: Mapping[str, Any], lines: list[str]) -> None:
    results, configuration = payload["results"], payload["configuration"]
    environment = payload["environment"]
    declared = configuration["silent_home_protocol"]
    fleet, check = results["fleet"], results["check"]
    conditions = dict(_ordered(results["conditions"], declared["conditions"]))
    lines.extend(
        [
            "## TIHM: what the rule does where the problem was found",
            "",
            f"**{_sentence(declared['standing'])}** This part is generated from a "
            f"second record, `{TIHM_RECORD_FILE}`.",
            "",
            f"- **Run.** Commit `{environment['git_commit'][:7]}`, "
            + (
                "with no uncommitted change"
                if environment["git_dirty"] == "false"
                else "with uncommitted changes"
            )
            + f"; recorded {payload['recorded_at']}.",
            f"- **Check.** With the rule off the run gives "
            f"{_n(check['monitored_days'])} monitored days and "
            f"{_n(check['behavioural_alerts'])} behavioural alerts, the published "
            "record's"
            + (
                f", and the same two counts in each of "
                f"{check['homes_compared_one_by_one']} homes"
                if check["homes_compared_one_by_one"]
                else ""
            )
            + ".",
            f"- **Not reported.** {_sentence(declared['not_reported'])}",
            f"- **Dataset.** {declared['citation']} Licence: {declared['licence']}. "
            "The dataset is not redistributed here.",
            "",
            *_table(
                [
                    "Condition",
                    "Monitored days",
                    "Usable days",
                    "Evaluable days",
                    "Days refused because of the rule (homes)",
                    "Behavioural alerts (homes)",
                    "Silence alerts (homes)",
                    "Alerts of both kinds",
                ],
                [
                    [
                        f"`{name}`",
                        _n(entry["monitored_days"]),
                        _n(entry["usable_days"]),
                        _n(entry["evaluable_days"]),
                        f"{_n(entry['days_refused_because_of_the_rule']['all'])} "
                        f"({entry['days_refused_because_of_the_rule']['homes_with_one']})",
                        f"{_n(entry['behavioural_alerts']['all'])} "
                        f"({entry['behavioural_alerts']['homes_with_one']})",
                        f"{_n(entry['silence_alerts']['all'])} "
                        f"({entry['silence_alerts']['homes_with_one']})",
                        _n(
                            entry["behavioural_alerts"]["all"]
                            + entry["silence_alerts"]["all"]
                        ),
                    ]
                    for name, entry in conditions.items()
                ],
            ),
            "",
            *[
                f"- Under `{name}` the rule refuses "
                f"{entry['days_refused_because_of_the_rule']['all'] / entry['monitored_days']:.1%}"
                " of the monitored days, and usable days fall by "
                f"{conditions[OFF]['usable_days'] - entry['usable_days']:,}. "
                "The pipeline raises "
                f"{entry['behavioural_alerts']['all'] / entry['monitored_days']:.3f} "
                "behavioural alerts and "
                f"{entry['silence_alerts']['per_monitored_day']:.3f} silence "
                "alerts per monitored day, where with the rule off it raises "
                f"{conditions[OFF]['behavioural_alerts']['all'] / conditions[OFF]['monitored_days']:.3f}"
                " behavioural alerts."
                for name, entry in conditions.items()
                if name != OFF
            ],
            "- A silence alert is repeated once per cooldown while a silence "
            "lasts, so there are fewer silences than silence alerts. The record "
            "does not count silences.",
            "",
            "An alert is the same alert under two conditions when it is in the "
            "same home, about the same feature, raised by the same day.",
            "",
            *_table(
                [
                    "Condition",
                    "Alerts also raised with the rule off",
                    "Raised only with the rule off",
                    "Raised only under this condition",
                    "By feature",
                ],
                [
                    [
                        f"`{name}`",
                        _n(
                            entry["behavioural_alerts"]["also_raised_with_the_rule_off"]
                        ),
                        _n(
                            entry["behavioural_alerts"]["raised_only_with_the_rule_off"]
                        ),
                        _n(
                            entry["behavioural_alerts"][
                                "raised_only_under_this_condition"
                            ]
                        ),
                        ", ".join(
                            f"{count} `{feature}`"
                            for feature, count in entry["behavioural_alerts"][
                                "by_feature"
                            ].items()
                        )
                        or "none",
                    ]
                    for name, entry in conditions.items()
                ],
            ),
            "",
            "### Silence shared across the homes",
            "",
            f"The fleet check, with the settings of the test, over each home's "
            f"activity records: {len(fleet['stretches'])} stretches of common "
            f"silence in {_n(fleet['assessments'])} assessments. The largest "
            f"share of monitored homes silent at one assessment was "
            f"{fleet['largest_share_silent']:.0%}, at "
            f"{_moment(fleet['largest_share_at'])}. Times are local to the "
            "homes.",
            "",
        ]
    )
    if fleet["stretches"]:
        lines.extend(
            _table(
                [
                    "Silent since",
                    "Called common at",
                    "Last called so at",
                    "Homes",
                    "Monitored",
                ],
                [
                    [
                        _moment(s["since"]),
                        _moment(s["detected"]),
                        _moment(s["until"]),
                        s["homes"],
                        s["monitored"],
                    ]
                    for s in fleet["stretches"]
                ],
            )
        )
        lines.append("")
    lines.extend(
        [
            "Silence alerts dated inside a stretch of common silence: "
            + "; ".join(
                f"`{name}`: "
                f"{entry['silence_alerts']['inside_a_stretch_of_common_silence']} "
                f"of {entry['silence_alerts']['all']}"
                for name, entry in conditions.items()
                if name != "off"
            )
            + ".",
            "",
            _figure(
                "tihm",
                "TIHM: the share of homes silent at each assessment, and the "
                "behavioural alerts by day with the rule off and on",
            ),
            "",
            "### Alerts by day",
            "",
            "The days whose summary raised the most alerts with the rule off, "
            "which are the alert-burden results' alert days, and what the same "
            "days raised with the rule on.",
            "",
        ]
    )
    off = conditions[OFF]["behavioural_alerts"]["by_day_summarised"]
    # The eighth busiest day's count, and every day that reaches it.
    cut = sorted(off.values(), reverse=True)[:8][-1] if off else 0
    busiest = sorted(day for day, count in off.items() if count >= cut)
    names = list(conditions)
    lines.extend(
        _table(
            ["Day summarised", *[f"`{name}`" for name in names]],
            [
                [
                    day,
                    *[
                        conditions[name]["behavioural_alerts"]["by_day_summarised"].get(
                            day, 0
                        )
                        for name in names
                    ],
                ]
                for day in busiest
            ],
        )
    )
    on = [name for name in names if name != OFF]
    homes = sorted(conditions[OFF]["silence_alerts"]["per_home"])
    lines.extend(
        [
            "",
            f"These are the {len(busiest)} days with at least {cut} alerts with "
            "the rule off.",
            "",
            "### Every TIHM home",
            "",
            "Each home's days refused because of the rule and its silence "
            "alerts, by the dataset's identifier.",
            "",
            *_table(
                [
                    "Home",
                    *[f"Days refused, `{name}`" for name in on],
                    *[f"Silence alerts, `{name}`" for name in on],
                ],
                [
                    [
                        f"`{home}`",
                        *[
                            conditions[name]["days_refused_because_of_the_rule"][
                                "per_home"
                            ][home]
                            for name in on
                        ],
                        *[
                            conditions[name]["silence_alerts"]["per_home"][home]
                            for name in on
                        ],
                    ]
                    for home in homes
                ],
            ),
            "",
        ]
    )


def render_page(
    payload: Mapping[str, Any],
    tihm: Mapping[str, Any] | None = None,
    protocol: SilentHomeProtocol | None = None,
) -> str:
    """The results page, from the record of the test and of the description.

    The record holds every alert with its moment, and the protocol says which
    window a moment falls in. *protocol* is the one the record was made under,
    the declared one unless given, and a record made under another is refused.
    """
    results, configuration = payload["results"], payload["configuration"]
    if results.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    if tihm is not None and tihm["results"].get("result_schema") != TIHM_SCHEMA:
        raise ValueError(f"not a {TIHM_SCHEMA} record")
    protocol = protocol if protocol is not None else declared_protocol()
    if protocol.sha256() != configuration["protocol_sha256"]:
        raise ValueError("the record was not made under this protocol")
    environment = payload["environment"]
    simulator = configuration["simulator"]
    lines = [
        "# The silent-home rule: results",
        "",
        f"The frozen [silent-home protocol]({PROTOCOL_PAGE}) run as declared. "
        f"This page is generated from the record, `{RECORD_FILE}`, "
        + (
            f"and its part on TIHM from a second record, `{TIHM_RECORD_FILE}`, "
            if tihm is not None
            else ""
        )
        + "by `sensor_modeling.datasets.silent_home_summary.render_page`, and a "
        "test checks that the committed page is exactly that rendering. The "
        "record holds every alert with its moment; the windows a moment falls "
        "in are the frozen protocol's.",
        "",
        f"**The evidence is simulated.** {results['homes']} paired simulated "
        f"homes of {results['days']} days, each run in the "
        f"{len(simulator['runs'])} pairs of an arm and a condition the protocol "
        "declares. Every result is a statement about this repository's "
        "simulator.",
        "",
        f"- **Protocol.** SHA-256 `{configuration['protocol_sha256']}`, checked "
        "against the frozen file before the run.",
        f"- **Run.** Commit `{environment['git_commit'][:7]}`, "
        + (
            "with no uncommitted change"
            if environment["git_dirty"] == "false"
            else "with uncommitted changes"
        )
        + f"; recorded {payload['recorded_at']}.",
        f"- **The rule.** `{configuration['the_rule']['setting']}`, "
        f"{configuration['the_rule']['default']} by default. Conditions: "
        + "; ".join(
            f"`{name}`, {text}"
            for name, text in _ordered(
                configuration["the_rule"]["conditions"], _conditions(configuration)
            )
        )
        + ".",
        f"- **Homes.** Seeds from root `{simulator['seed_root']}`; sensors "
        f"{_code(simulator['sensors'])}.",
        f"- {payload['notes'][0]}",
        "- **No pilot.** No pilot informed the protocol: no simulated home had "
        "been run with the rule on when it was frozen. The count of homes is "
        "the floor `docs/SIMULATION_PROTOCOLS.md` sets, and the Monte Carlo "
        "standard error achieved is given beside every paired difference.",
        "",
    ]
    _verdicts_part(payload, lines)
    _outage_part(payload, lines)
    _stable_part(payload, lines)
    _detection_part(payload, lines)
    _reporting_part(payload, lines)
    _fleet_part(payload, lines)
    _short_part(payload, protocol, lines)
    _described_part(payload, protocol, lines)
    _between_part(payload, lines)
    if tihm is not None:
        _tihm_part(tihm, lines)
    _homes_part(payload, lines)
    return "\n".join(lines)
