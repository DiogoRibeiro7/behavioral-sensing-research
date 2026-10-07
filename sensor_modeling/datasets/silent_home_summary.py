"""The silent-home pages, generated from the frozen protocol and the records.

:func:`render_protocol` writes ``docs/SILENT_HOME_PROTOCOL.md`` from the frozen
protocol file alone. It reports no result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PROTOCOL_FILE = "artifacts/silent_home/silent_home_protocol.json"
PROTOCOL_PAGE = "SILENT_HOME_PROTOCOL.md"


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
