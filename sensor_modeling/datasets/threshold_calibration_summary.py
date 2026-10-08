"""The threshold-calibration pages, generated from the frozen protocol and the records.

:func:`render_protocol` writes ``docs/THRESHOLD_CALIBRATION_PROTOCOL.md`` from
the frozen protocol file alone. It reports no result.

:func:`render_page` writes ``docs/THRESHOLD_CALIBRATION_RESULTS.md`` from the
record of the protocol's test on simulated homes and, in a part of its own,
from the record of the description on TIHM. Every home is shown.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PROTOCOL_FILE = "artifacts/threshold_calibration/threshold_calibration_protocol.json"
PROTOCOL_PAGE = "THRESHOLD_CALIBRATION_PROTOCOL.md"
FIGURE_DIR = "figures"


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
    """Split ``C1_more_is_detected`` into its number and its claim."""
    number, _, claim = name.partition("_")
    return number, _title(claim)


def _settings(pipeline: Mapping[str, Any]) -> list[list[Any]]:
    return [
        ["step", f"{pipeline['step_minutes']} minutes"],
        ["baseline features", _code(pipeline["features"])],
        ["minimum day coverage", pipeline["min_day_coverage"]],
        ["minimum day observed", pipeline["min_day_observed"]],
        *[
            [f"baseline {key.replace('_', ' ')}", value]
            for key, value in pipeline["baseline"].items()
        ],
        *[
            [f"alert policy {key.replace('_', ' ')}", value]
            for key, value in pipeline["alert_policy"].items()
        ],
    ]


def _homes(simulator: Mapping[str, Any], planned_on: Mapping[str, Any]) -> list[str]:
    pipeline, replay = simulator["pipeline"], simulator["replay"]
    planned = (
        [
            f"- **The trials it was planned on.** `{planned_on['record']}`, "
            f"SHA-256 `{planned_on['sha256']}`: {planned_on['what']}.",
        ]
        if planned_on["sha256"]
        else []
    )
    return [
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
        f"- **How many homes.** {_sentence(simulator['planning'])}",
        *planned,
        f"- **Why sensors are left out.** {_sentence(simulator['why_left_out'])}",
        f"- **Delivery.** {_sentence(simulator['delivery'])}",
        f"- **Pairing.** {_sentence(simulator['pairing'])}",
        f"- **Calendar.** {_sentence(simulator['calendar'])}",
        "",
        "### The pipeline",
        "",
        f"{_sentence(pipeline['what'])}",
        "",
        *_table(["Setting", "Value"], _settings(pipeline)),
        "",
        f"Every other setting: {pipeline['every_other_setting']}.",
        "",
        "### One run, many thresholds",
        "",
        f"- **What.** {_sentence(replay['what'])}",
        f"- **Why.** {_sentence(replay['why'])}",
        f"- **Check.** {_sentence(replay['check'])}",
        f"- **Homes checked against the pipeline.** {_code(replay['checked_seeds'])}.",
        "",
        "### Arms",
        "",
        *_table(
            ["Arm", "What is injected"],
            [
                [f"`{arm}`", _sentence(simulator["arms"][arm])]
                for arm in simulator["arm_order"]
            ],
        ),
        "",
    ]


_SETTING_TITLES = {
    "plain": "Days with no weekly rhythm",
    "weekly_rhythm": "Days with a weekly rhythm",
}


def _expected(expected: Mapping[str, Any]) -> list[str]:
    settings = expected["settings"]

    def row(
        label: str, values: Mapping[str, float], prefix: str, multiple: str
    ) -> list[str]:
        return [
            label,
            multiple,
            f"{100.0 * values[prefix + '_false']:.1f}%",
            f"{100.0 * values[prefix + '_found']:.1f}%",
        ]

    lines = [
        "## What synthetic days lead one to expect",
        "",
        f"{_sentence(expected['what'])}",
    ]
    for name in ("plain", "weekly_rhythm"):
        if name not in settings:
            continue
        values = settings[name]
        lines += [
            "",
            f"**{_SETTING_TITLES[name]}.**",
            "",
            *_table(
                [
                    "Reference",
                    "Multiple",
                    "A change verdict with no step added",
                    "With a step of two standard deviations",
                ],
                [
                    row("default, as declared", values, "default", "1"),
                    row(
                        "calibrated, at the same false alerts",
                        values,
                        "at_the_same_false_alerts",
                        f"{values['same_false_alerts']:g}",
                    ),
                ],
            ),
        ]
    return [*lines, ""]


def _tihm(tihm: Mapping[str, Any]) -> list[str]:
    return [
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
                ["conditions", f"all {len(tihm['conditions'])} described conditions"],
                ["silent-home rule", _code(tihm["silent_home_rule"])],
                [
                    "homes checked with the calibrated pipeline",
                    f"the first {tihm['checked_homes']} by identifier",
                ],
                ["everything else", tihm["everything_else"]],
                *[
                    [f"published record `{name}`", f"SHA-256 `{digest}`"]
                    for name, digest in tihm["published_records"].items()
                ],
            ],
        ),
        "",
        f"- **Replay.** {_sentence(tihm['replay'])}",
        f"- **Check.** {_sentence(tihm['check'])}",
        "- **Reported.** "
        + _sentence("; ".join(str(item) for item in tihm["reported"])),
        f"- **Not reported.** {_sentence(tihm['not_reported'])}",
        f"- **What it cannot say.** {_sentence(tihm['what_it_cannot_say'])}",
        "- **Redistribution.** None: the run reads an extracted download and "
        "verifies every file's digest.",
        "- **Acknowledgement.** Surrey and Borders Partnership NHS Foundation "
        "Trust and Howz, as the dataset asks.",
        "",
    ]


def _criteria(criteria: Mapping[str, Any]) -> list[str]:
    return [
        "## Criteria",
        "",
        "Fixed before any simulated home was run with the calibrated reference. "
        "Both are decided on all the homes.",
        "",
        *_table(
            ["Criterion", "Claim", "How it is decided"],
            [
                [*_criterion(name), _sentence(criteria[name])]
                for name in criteria["order"]
            ],
        ),
        "",
        "The margins, and where each comes from:",
        "",
        *_bullets(
            {name: criteria["margins"][name] for name in criteria["margin_order"]}
        ),
        "",
        "How the criteria are read:",
        "",
        *[f"- {_sentence(item)}" for item in criteria["readings"]],
        "",
    ]


def _frozen_code(payload: Mapping[str, Any]) -> list[str]:
    frozen = payload.get("at_freeze")
    if frozen is None:
        return []
    sources = frozen["sources"]
    scripts = sorted(name for name in sources if name.startswith("scripts/"))
    return [
        "## The code the protocol was frozen against",
        "",
        f"The frozen file records the content of {len(sources)} source files: "
        f"every module of the package, {len(sources) - len(scripts)} of them, "
        "since a home's days pass through most of it before a threshold is "
        f"asked anything, and the scripts that freeze and run the protocol, "
        f"{_code(scripts)}. It also records the versions of the numerical "
        "libraries and every default setting of the pipeline, the baseline, "
        "the alert policy, the health monitor, the context estimator and the "
        "simulated household. None of this is part of the protocol's digest. "
        "The run compares all of it with what it uses, and refuses to go on "
        "when anything differs unless it is told why; the record then says "
        "what differed and why.",
        "",
        *_table(
            ["Library", "Version at the freeze"],
            [
                [f"`{name}`", version]
                for name, version in sorted(frozen["distributions"].items())
            ],
        ),
        "",
    ]


def _matching(matching: Mapping[str, Any]) -> list[str]:
    titles = {
        "curve": "The operating curve",
        "same_false_alerts": "The same false alerts",
        "bracketed": "When there is no match",
        "what_it_uses": "What the rule uses",
        "why": "Why the match is placed on the curve",
        "why_not_the_same_detection": "Why not the same detection",
        "nearest_multiple": "The condition described in full",
    }
    return [
        "## Where the two references are compared",
        "",
        *[
            f"- **{titles[name]}.** {_sentence(matching[name])}"
            for name in matching["order"]
        ],
        "",
    ]


def render_protocol(payload: Mapping[str, Any]) -> str:
    """The protocol page, from the frozen protocol file's content."""
    option, conditions = payload["the_option"], payload["conditions"]
    simulator, definitions = payload["simulator"], payload["definitions"]
    names = {
        "default": f"`{conditions['default']}`",
        "declared": f"`{conditions['declared']}`",
        "same_false_alerts": "`calibrated@` the multiple nearest the match",
    }
    lines = [
        "# Threshold calibration: the protocol",
        "",
        "The personal baseline's deviation threshold is passed far more often "
        "than it states. The [TIHM run](TIHM_ALERT_BURDEN_RESULTS.md) showed "
        f"it, and an opt-in reference, `{option['setting']}`, was built from "
        "what it showed and designed on [synthetic days]"
        "(THRESHOLD_CALIBRATION_NULL.md). This page is the protocol for finding "
        "out whether that reference is better in a simulated home, generated "
        f"entirely from the frozen file `{PROTOCOL_FILE}` by "
        "`sensor_modeling.datasets.threshold_calibration_summary.render_protocol`.",
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
        "The reference was built from TIHM and designed on synthetic days, so "
        "neither can be the evidence that it helps. The test is made on "
        "simulated homes that had not been run with it.",
        "",
        *[f"- {_sentence(item)}" for item in payload["inspected_before"]],
        "",
        "## The option",
        "",
        f"`{option['setting']}` is {option['default']} by default. With it on, "
        f"{option['what_it_changes']}. See [the inference contract](inference.md).",
        "",
        f"- **Designed on.** {_sentence(option['designed_on'])}",
        f"- **The measurement it rests on.** `{option['null_record']}`, SHA-256 "
        f"`{option['null_record_sha256']}`.",
        "",
        "## Conditions",
        "",
        f"{_sentence(conditions['naming'])}",
        "",
        *_table(
            ["Role", "Condition", "What it is"],
            [
                [
                    role.replace("_", " "),
                    names[role],
                    _sentence(conditions["roles"][role]),
                ]
                for role in conditions["role_order"]
            ],
        ),
        "",
        "- **Why not at the declared thresholds.** "
        + _sentence(conditions["why_not_as_declared"]),
        "- **Described as well.** Both references at "
        + ", ".join(f"{scale:g}" for scale in conditions["curve_scales"])
        + f" times the declared thresholds: {len(conditions['described'])} "
        "conditions, those above among them.",
        "",
        *_matching(payload["matching"]),
        "## What the outcomes turn on",
        "",
        f"{_sentence(payload['what_the_outcomes_turn_on'])}",
        "",
        *_homes(simulator, payload["planned_on"]),
        "## Definitions",
        "",
        *_bullets({name: definitions[name] for name in definitions["order"]}),
        "",
        "## Estimands",
        "",
        *_table(
            ["Estimand", "What it is"],
            [
                [name, _sentence(text)]
                for name, text in sorted(
                    payload["estimands"].items(), key=lambda item: int(item[0][1:])
                )
            ],
        ),
        "",
        *_criteria(payload["criteria"]),
        *_expected(payload["expected_from_synthetic_days"]),
        "## Uncertainty",
        "",
        *_bullets(
            {name: payload["bootstrap"][name] for name in payload["bootstrap"]["order"]}
        ),
        "",
        "## What the simulator cannot show",
        "",
        *[f"- {_sentence(item)}" for item in payload["what_the_simulator_cannot_show"]],
        "",
        *_tihm(payload["tihm"]),
        "## Reporting",
        "",
        f"{_sentence(payload['reporting'])}",
        "",
        *_frozen_code(payload),
        "## The homes",
        "",
        "The seeds of the homes, in order.",
        "",
        _code(simulator["study_seeds"]) + ".",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# The results page
# ----------------------------------------------------------------------------
RECORD_FILE = "artifacts/threshold_calibration/threshold-calibration.json"
TIHM_RECORD_FILE = "artifacts/threshold_calibration/threshold-calibration-tihm.json"
RESULTS_PAGE = "THRESHOLD_CALIBRATION_RESULTS.md"
FIGURE_PREFIX = "threshold-calibration"
RESULT_SCHEMA = "threshold-calibration/1"
TIHM_SCHEMA = "threshold-calibration-tihm/1"

_ARM_TITLES = {
    "change": "the step change",
    "small_change": "the smaller step",
    "gradual_change": "the gradual change",
}
_ROLE_TITLES = {
    "default": "the default, as it ships",
    "declared": "calibrated, at the declared thresholds",
    "same_false_alerts": "calibrated, at the multiple nearest the same false alerts",
}
_STATUS_TITLES = {
    "bracketed": "bracketed",
    "not_reached": "not reached",
    "at_the_end_of_the_grid": "at the end of the grid",
}
_REFERENCES = ("default", "calibrated")


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

    def bound(value: Any) -> str:
        return "none" if value is None else show(float(value), digits)

    text = show(float(entry["estimate"]), digits)
    interval = entry.get("interval")
    if interval:
        text += f" [{bound(interval['low'])}, {bound(interval['high'])}]"
    return text


def _difference(entry: Mapping[str, Any] | None, digits: int = 2) -> str:
    """A paired difference with its interval and its Monte Carlo error."""
    if entry is None or entry.get("estimate") is None:
        return "–"
    text = _interval(entry, digits, signed=True)
    if entry.get("mcse") is not None:
        text += f", SE {entry['mcse']:.{digits}f}"
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


def _percent(entry: Mapping[str, Any] | None, digits: int = 2) -> str:
    """A share of days with its interval, in percent."""
    if entry is None or entry.get("estimate") is None:
        return "–"
    text = f"{100.0 * entry['estimate']:.{digits}f}%"
    interval = entry.get("interval")
    if interval:
        text += (
            f" [{100.0 * interval['low']:.{digits}f}, "
            f"{100.0 * interval['high']:.{digits}f}]"
        )
    return text


def _threshold(entry: Mapping[str, Any]) -> str:
    """The threshold a share is equivalent to, with its interval."""
    if entry.get("estimate") is None:
        return "–"

    def show(number: Any) -> str:
        return "none passed" if number is None else f"{number:.2f}"

    text = show(entry.get("equivalent_threshold"))
    interval = entry.get("equivalent_threshold_interval")
    if interval:
        text += f" [{show(interval['low'])}, {show(interval['high'])}]"
    return text


def _counts(counts: Mapping[str, int]) -> str:
    """Counts by name, largest first."""
    if not counts:
        return "–"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{name.replace('_', ' ')} {count:,}" for name, count in ordered)


def _figure_of(name: str, alt: str) -> str:
    return f"![{alt}]({FIGURE_DIR}/{FIGURE_PREFIX}-{name}.svg)"


def _by_reference(entries: Mapping[str, Any]) -> dict[str, Any]:
    """Entries by reference, the default first."""
    return {name: entries[name] for name in _REFERENCES if name in entries}


def _by_scale(entries: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Entries by multiple, the smallest first."""
    return sorted(entries.items(), key=lambda item: float(item[0]))


def _roles(results: Mapping[str, Any]) -> dict[str, list[str]]:
    """What each reported condition is shown as; one may hold two roles."""
    roles: dict[str, list[str]] = {condition: [] for condition in results["reported"]}
    for role in _ROLE_TITLES:
        roles[results["shown"][role]].append(role)
    return roles


def _named(results: Mapping[str, Any], condition: str) -> str:
    """A condition with what it is shown as."""
    roles = _roles(results).get(condition, [])
    return f"`{condition}`" + (
        ", " + " and ".join(_ROLE_TITLES[role] for role in roles) if roles else ""
    )


def _thresholds_of(payload: Mapping[str, Any], condition: str) -> str:
    """A condition's two thresholds, from the models the record lists."""
    for model in payload["models"]:
        if model["name"] == f"replay_{condition}":
            settings = model["configuration"]
            return (
                f"{settings['deviation_threshold']:.3g} and "
                f"{settings['trend_threshold']:.3g}"
            )
    return "–"


def _arms(configuration: Mapping[str, Any]) -> list[str]:
    return [
        arm for arm in configuration["simulator"]["arm_order"] if arm in _ARM_TITLES
    ]


def _code_line(configuration: Mapping[str, Any]) -> str:
    """Whether the code a run used is the code recorded at the freeze."""
    changed = configuration["code_changed_since_the_freeze"]
    names = [
        name
        for group in ("sources", "distributions", "defaults")
        for name in changed.get(group, [])
    ]
    if not names:
        return (
            "The source files, the libraries' versions and the default settings "
            "recorded at the freeze are those the run used."
        )
    return (
        "Code recorded at the freeze had changed by the run: "
        + _code(names)
        + f". {configuration['why_the_code_changed']}"
    )


def _results_head(payload: Mapping[str, Any]) -> list[str]:
    configuration, environment = payload["configuration"], payload["environment"]
    results, simulator = payload["results"], configuration["simulator"]
    check = results["check"]
    dirty = environment.get("git_dirty") != "false"
    code = _code_line(configuration)
    return [
        "# Threshold calibration: results",
        "",
        "The frozen [threshold-calibration protocol]"
        "(THRESHOLD_CALIBRATION_PROTOCOL.md) run as declared. This page is "
        f"generated from the record, `{RECORD_FILE}`, and its part on TIHM "
        f"from a second record, `{TIHM_RECORD_FILE}`, by "
        "`sensor_modeling.datasets.threshold_calibration_summary.render_page`, "
        "and a test checks that the committed page is exactly that rendering.",
        "",
        f"**The evidence is simulated.** {results['homes']} paired simulated "
        f"homes of {results['days']} days, each run in "
        f"{len(simulator['arm_order'])} arms and replayed under "
        f"{len(configuration['conditions']['described'])} conditions. "
        "Every result is a statement about this repository's simulator.",
        "",
        f"- **Protocol.** SHA-256 `{configuration['protocol_sha256']}`, checked "
        "against the frozen file before the run.",
        f"- **Run.** Commit `{environment.get('git_commit', 'unknown')[:7]}`, "
        + ("with uncommitted changes" if dirty else "with no uncommitted change")
        + f"; recorded {payload['recorded_at']}.",
        f"- **Code.** {code}",
        f"- **The option.** `{configuration['the_option']['setting']}`, off by "
        "default.",
        f"- **Homes.** Seeds from root `{simulator['seed_root']}`; sensors "
        + _code(simulator["sensors"])
        + ".",
        "- **Pre-specified.** The protocol was frozen before any simulated "
        "home was run with the calibrated reference.",
        "- **The replay was checked.** Under the default reference it returned "
        "the pipeline's own verdicts and alerts in all "
        f"{check['default_replays']:,} runs. On {len(check['homes_checked'])} "
        "of the homes, the ones the protocol names, the pipeline was also run "
        "with the calibrated reference itself, and the replay returned all "
        f"{check['calibrated_runs_checked']} of those runs.",
    ]


def _resamples(entry: Mapping[str, Any]) -> str:
    """How many resamples had a match, and what became of the others."""
    counts = entry["resamples"]
    return (
        f"{counts['bracketed']:,} of {counts['all']:,} resamples bracketed, "
        f"{counts['not_reached']:,} not reached and "
        f"{counts['at_the_end_of_the_grid']:,} at the end of the grid"
    )


def _match_part(payload: Mapping[str, Any]) -> list[str]:
    results, configuration = payload["results"], payload["configuration"]
    found, rule = results["matched"], configuration["matching"]
    match, default = found["match"], found["default"]
    arms = _arms(configuration)
    if match["status"] == "bracketed":
        low, high = match["between"]
        where = (
            f"at {match['multiple']:.3f} times the declared thresholds, between "
            f"the multiples {low:g} and {high:g} of the grid"
        )
    elif match["status"] == "not_reached":
        where = (
            "not reached: at every multiple of the grid the calibrated "
            "reference raises more false alerts than the default"
        )
    else:
        where = (
            "at the end of the grid: at its smallest multiple the calibrated "
            "reference already raises no more false alerts than the default"
        )
    return [
        "",
        "## At the same false alerts (E1, E4)",
        "",
        f"{_sentence(rule['same_false_alerts'])} {_sentence(rule['what_it_uses'])}",
        "",
        f"- **The default.** `{default['condition']}` raises "
        f"{_n(default['false_alerts_per_home'])} false alerts per home.",
        f"- **The match.** On the {results['homes']} homes it is {where}.",
        "- **The condition described in full.** "
        f"`{results['shown']['same_false_alerts']}`, at the multiple of the grid "
        "the protocol's rule gives for the match. It is chosen from the data, "
        "and no criterion reads it.",
        "",
        *_table(
            [
                "Change",
                "Default",
                "Calibrated, at the match",
                "Calibrated minus default",
                "Resamples",
            ],
            [
                [
                    _ARM_TITLES[arm],
                    _n(found["excess_detection"][arm]["default"]),
                    _n(found["excess_detection"][arm]["calibrated_at_the_match"]),
                    _difference(found["excess_detection"][arm]),
                    _resamples(found["excess_detection"][arm]),
                ]
                for arm in arms
            ],
        ),
        "",
        "The numbers are mean excess detection: the share of homes detected "
        "minus the share falsely detected. The first row is E1, on which C1 is "
        "decided, and the others are E4, on which nothing is. A bound shown as "
        "none does not exist: too many resamples had no match.",
    ]


_READING_SENTENCES = {
    "better": "At the false alerts of the default as it ships, the calibrated "
    "reference finds more of the step change than the default.",
    "worse": "At the false alerts of the default as it ships, the calibrated "
    "reference finds less of the step change than the default.",
    "not shown": "At the false alerts of the default as it ships, no difference "
    "in what the two references find is shown either way.",
    "uninformative": "The default finds too little of the step change beyond "
    "chance for the comparison to mean anything, and nothing is claimed.",
}


def _criteria_part(payload: Mapping[str, Any]) -> list[str]:
    results, configuration = payload["results"], payload["configuration"]
    judged = results["criteria"]
    first, second = configuration["criteria"]["order"]
    one, two = judged[first], judged[second]
    number, claim = _criterion(first)
    rows = [
        [
            number,
            claim,
            f"**{one['verdict']}**",
            "The default's excess detection is "
            f"{_interval(one['default_excess_detection'])}, where below "
            f"{one['informative_at']:g} nothing is claimed, and the match is "
            f"{_STATUS_TITLES[one['match']['status']]}. E1 is "
            f"{_interval(one['E1'], signed=True)}.",
        ]
    ]
    number, claim = _criterion(second)
    for threshold, entry in sorted(two["E3"].items(), key=lambda item: -float(item[0])):
        rows.append(
            [
                f"{number}, at {threshold}",
                claim,
                f"**{two['verdicts'][threshold]}**",
                f"{_percent(entry)} of days are at or past {threshold}, which "
                "on Gaussian days is a threshold of "
                f"{_threshold(entry)}; the band is "
                f"{entry['threshold'] - two['tolerance']:g} to "
                f"{entry['threshold'] + two['tolerance']:g}.",
            ]
        )
    return [
        "",
        "## The criteria",
        "",
        "Each was fixed, with its margin, before any simulated home had been "
        "run with the calibrated reference. They are decided on the "
        f"{results['homes']} homes. The intervals resample those homes and are "
        "not adjusted for there being two criteria.",
        "",
        *_table(["Criterion", "Claim", "Verdict", "What decided it"], rows),
        "",
        "Read as the protocol fixed it:",
        "",
        f"- **{_READING_SENTENCES[judged['reading']]}**",
        "- **Whether the calibrated reference's threshold means what it says "
        "on these homes, at all three thresholds together:** "
        f"{two['together']}.",
        "- **Not shown is not no difference.** An inconclusive verdict means "
        "the interval reached the value it was judged against, or that one of "
        "its bounds does not exist because too many resamples had no match.",
    ]


def _stable_part(payload: Mapping[str, Any]) -> list[str]:
    results = payload["results"]
    order = results["reported"]
    stable, against = results["stable"], results["against_the_default"]
    rows = [
        [
            _named(results, condition),
            _thresholds_of(payload, condition),
            _n(stable[condition]["behavioural_alerts"]),
            _interval(stable[condition]["per_home"]),
            _n(stable[condition]["per_person_day"]["estimate"], 4),
            _n(stable[condition]["homes_with_one"]),
            (
                _difference(against[condition]["false_alerts"])
                if condition in against
                else "–"
            ),
        ]
        for condition in order
    ]
    weeks = len(stable[order[0]]["by_week_of_the_day_closed"])
    return [
        "",
        "## False alerts (E2, E5)",
        "",
        "Nothing is injected in the stable arm, so every behavioural alert "
        f"there is a false one. {results['homes']} homes, "
        f"{results['person_days']:,} person-days.",
        "",
        _figure_of("weeks", "False alerts by week of the record, by condition"),
        "",
        *_table(
            [
                "Condition",
                "Thresholds",
                "False alerts",
                "Per home",
                "Per person-day",
                "Homes with one",
                "Minus the default, per home",
            ],
            rows,
        ),
        "",
        "What raised them:",
        "",
        *_table(
            ["Condition", "By verdict", "By feature", "By direction"],
            [
                [
                    f"`{condition}`",
                    _counts(stable[condition]["by_verdict"]),
                    _counts(stable[condition]["by_feature"]),
                    _counts(stable[condition]["by_direction"]),
                ]
                for condition in order
            ],
        ),
        "",
        "By the week of the record whose day, on closing, raised them:",
        "",
        *_table(
            ["Condition", *(str(week) for week in range(1, weeks + 1))],
            [
                [
                    f"`{condition}`",
                    *(
                        _n(count)
                        for count in stable[condition]["by_week_of_the_day_closed"]
                    ),
                ]
                for condition in order
            ],
        ),
    ]


def _detection_table(
    payload: Mapping[str, Any], arm: str, expected: bool = False
) -> list[str]:
    results = payload["results"]
    detection, against = results["detection"][arm], results["against_the_default"]
    rows = []
    for condition in results["reported"]:
        entry = detection[condition]
        shown = entry["with_the_expected_direction"] if expected else entry
        row = [
            f"`{condition}`",
            _share(shown["detected"]),
            _share(shown["false_detections"]),
            _interval(shown["excess_detection"]),
        ]
        if not expected:
            row += [
                (
                    _difference(against[condition][arm]["excess_detection"])
                    if condition in against
                    else "–"
                ),
                (
                    _difference(against[condition][arm]["detected"])
                    if condition in against
                    else "–"
                ),
                (
                    _difference(against[condition][arm]["false_detections"])
                    if condition in against
                    else "–"
                ),
                _interval(entry["delay_days"], 1),
            ]
        rows.append(row)
    header = ["Condition", "Detected", "False detections", "Excess detection"]
    if not expected:
        header += [
            "Excess, minus the default",
            "Detected, minus the default",
            "False detections, minus the default",
            "Median delay, days",
        ]
    return _table(header, rows)


def _detection_part(payload: Mapping[str, Any]) -> list[str]:
    results, configuration = payload["results"], payload["configuration"]
    order = results["reported"]
    lines = [
        "",
        "## Detection (E2, E5)",
        "",
        "A home is detected when a behavioural alert about its hours of sleep "
        "is raised in the arm's window. A false detection is the same in the "
        "same home's stable record, and the excess is the one minus the other: "
        "what a condition finds beyond what it would have raised anyway. These "
        "tables are of conditions that were run; C1 is decided at the match, "
        "above, and not here. The calibrated reference at the multiple nearest "
        "the match was chosen from these homes: its differences from the "
        "default describe it, and their intervals do not carry the choice.",
    ]
    for arm in _arms(configuration):
        detection = results["detection"][arm]
        lines += [
            "",
            f"### {_ARM_TITLES[arm].capitalize()}, `{arm}`",
            "",
            f"{_sentence(configuration['simulator']['arms'][arm])}",
            "",
            *_detection_table(payload, arm),
            "",
            "The first alert that counts, by the verdict behind it and by its "
            "direction:",
            "",
            *_table(
                ["Condition", "Verdict", "Direction"],
                [
                    [
                        f"`{condition}`",
                        _counts(detection[condition]["first_alert"]["verdict"]),
                        _counts(detection[condition]["first_alert"]["direction"]),
                    ]
                    for condition in order
                ],
            ),
            "",
            "When the alert must also be of a decrease, which is the direction "
            "of the injected change:",
            "",
            *_detection_table(payload, arm, expected=True),
        ]
    gradual = results["detection"].get("gradual_change")
    if gradual:
        weeks = gradual[order[0]]["by_week"]["week_of_the_window"]
        lines += [
            "",
            "Homes first detected in the gradual change, and homes with a "
            "first false detection in the stable record, by week of the window:",
            "",
            *_table(
                ["Condition", "Record", *(str(week) for week in weeks)],
                [
                    [f"`{condition}`", label, *(_n(count) for count in counts)]
                    for condition in order
                    for label, counts in (
                        (
                            "gradual change",
                            gradual[condition]["by_week"]["first_detected"],
                        ),
                        (
                            "stable",
                            gradual[condition]["by_week"]["first_false_detection"],
                        ),
                    )
                ],
            ),
        ]
    return lines


def _calibration_part(payload: Mapping[str, Any]) -> list[str]:
    results = payload["results"]
    calibration = results["calibration"]
    verdicts = results["criteria"]["C2_the_threshold_means_what_it_says"]["verdicts"]
    thresholds = sorted(calibration["thresholds"], key=float, reverse=True)
    rows, phases, others = [], [], []
    features: list[str] = []
    for threshold in thresholds:
        for reference, entry in _by_reference(
            calibration["thresholds"][threshold]
        ).items():
            rows.append(
                [
                    reference,
                    threshold,
                    f"{100.0 * entry['stated']:.2f}%",
                    _percent(entry),
                    _threshold(entry),
                    verdicts[threshold] if reference == "calibrated" else "described",
                ]
            )
            phases.append(
                [
                    reference,
                    threshold,
                    _percent(entry["by_phase"]["pooled"]),
                    _percent(entry["by_phase"]["weekday_aware"]),
                ]
            )
            features = sorted(entry["other_features"])
            others.append(
                [
                    reference,
                    threshold,
                    *(_percent(entry["other_features"][name]) for name in features),
                ]
            )
    tail = _by_reference(calibration["tail"])
    points = sorted(next(iter(tail.values())), key=float) if tail else []
    return [
        "",
        "## Does the threshold mean what it says (E3, C2)",
        "",
        f"The share of evaluable days of `{calibration['feature']}` in the "
        "stable records whose deviation is at or past a "
        "threshold, beside what the threshold states for Gaussian days, and "
        "the threshold that share is equivalent to. A day's deviation does not "
        "depend on the threshold it is read against, so the rows of one "
        "reference are points on one curve.",
        "",
        _figure_of("tail", "Share of days past a threshold, for both references"),
        "",
        *_table(
            [
                "Reference",
                "Threshold",
                "Stated",
                "Deviating",
                "Equivalent threshold",
                "C2",
            ],
            rows,
        ),
        "",
        "The whole tail, for each reference:",
        "",
        *_table(
            ["Reference", *(f"At {point}" for point in points)],
            [
                [reference, *(_percent(tail[reference][point]) for point in points)]
                for reference in tail
            ],
        ),
        *(
            [
                "",
                "What the thresholds state: "
                + ", ".join(
                    f"{100.0 * tail[next(iter(tail))][point]['stated']:.2f}% at {point}"
                    for point in points
                )
                + ".",
            ]
            if tail
            else []
        ),
        "",
        "Before and after the reference becomes weekday-aware:",
        "",
        *_table(
            [
                "Reference",
                "Threshold",
                "All the earlier days behind it",
                "Its own weekday behind it",
            ],
            phases,
        ),
        "",
        "The other features, on which no criterion is stated:",
        "",
        *_table(
            ["Reference", "Threshold", *(f"`{name}`" for name in features)], others
        ),
    ]


def _verdicts_part(payload: Mapping[str, Any]) -> list[str]:
    results, configuration = payload["results"], payload["configuration"]
    rows = []
    for arm in configuration["simulator"]["arm_order"]:
        for condition in results["reported"]:
            entry = results["verdicts"][arm][condition]
            rows.append(
                [
                    f"`{arm}`",
                    f"`{condition}`",
                    _n(entry["change_verdicts"]),
                    _counts(entry["by_kind"]),
                    _n(sum(entry["raised_an_alert"].values())),
                    _n(entry["raised_none"].get("not_attributable_enough", 0)),
                    _n(entry["raised_none"].get("graded_too_low_or_withheld", 0)),
                    _n(entry["notices_of_a_burst"]),
                ]
            )
    return [
        "",
        "## From a verdict to an alert (E5)",
        "",
        "The alert policy stands between a change verdict and an alert. It "
        "asks that the day be attributable enough, grades the verdict against "
        "its threshold, and withholds a repeat. A verdict on a day that was "
        "attributable enough and raised no alert was graded below the minimum "
        "score, repeated one inside the cooldown, or fell in a burst; the "
        "record does not tell those apart.",
        "",
        *_table(
            [
                "Arm",
                "Condition",
                "Change verdicts",
                "By kind",
                "Raised an alert",
                "Not attributable enough",
                "Graded too low, or withheld",
                "Notices of a burst",
            ],
            rows,
        ),
    ]


def _curves_part(payload: Mapping[str, Any]) -> list[str]:
    results, configuration = payload["results"], payload["configuration"]
    curves, reading = _by_reference(results["curves"]), results["curves_reading"]
    arms = _arms(configuration)
    lines = [
        "",
        "## The operating curves (E6)",
        "",
        "Both references at every multiple of the declared thresholds: mean "
        "false alerts per home, and in each change arm the excess detection "
        "and, after it, the share of homes detected. No criterion is stated "
        "on these.",
        "",
        _figure_of(
            "curves",
            "Excess detection against false alerts per home, for both references",
        ),
    ]
    for reference, points in curves.items():
        lines += [
            "",
            f"**The {reference} reference.**",
            "",
            *_table(
                [
                    "Multiple",
                    "False alerts per home",
                    *(f"{_ARM_TITLES[arm].capitalize()}" for arm in arms),
                ],
                [
                    [
                        scale,
                        _interval(entry["false_alerts_per_home"]),
                        *(
                            f"{_interval(entry[arm]['excess_detection'])}; "
                            f"{entry[arm]['detected']['estimate']:.0%} detected"
                            for arm in arms
                        ),
                    ]
                    for scale, entry in _by_scale(points)
                ],
            ),
        ]
    if reading:
        scales = len(next(iter(curves.values())))
        lines += [
            "",
            "The reading the protocol fixed, which compares the estimates on "
            "the same homes, takes the best of the multiples and carries no "
            f"uncertainty. For how many of the {scales} multiples of one "
            "reference does some multiple of the other have no more false "
            "alerts and no less excess detection:",
            "",
            *_table(
                [
                    "Change",
                    "Default multiples matched by a calibrated one",
                    "Calibrated multiples matched by a default one",
                ],
                [
                    [
                        _ARM_TITLES[arm],
                        _n(
                            len(
                                reading[arm][
                                    "default_multiples_a_calibrated_one_is_no_worse_than"
                                ]
                            )
                        ),
                        _n(
                            len(
                                reading[arm][
                                    "calibrated_multiples_a_default_one_is_no_worse_than"
                                ]
                            )
                        ),
                    ]
                    for arm in arms
                ],
            ),
        ]
    return lines


def _features_part(payload: Mapping[str, Any]) -> list[str]:
    features = payload["results"]["features"]
    return [
        "",
        "## What the days look like",
        "",
        "Not an estimand. The calibrated score is exact for independent "
        "Gaussian days, and this says how far the days of the stable records "
        "are from that: each feature's spread, how much a "
        "weekend day differs from a weekday, and how often the feature is all "
        "but absent from a day.",
        "",
        *_table(
            [
                "Feature",
                "Median SD, hours",
                "Weekend minus weekday, hours",
                "Days under a hundredth of an hour",
            ],
            [
                [
                    f"`{name}`",
                    _n(entry["median_standard_deviation_hours"]),
                    _signed(entry["median_weekend_minus_weekday_hours"]),
                    (
                        "–"
                        if entry["share_of_days_under_a_hundredth_of_an_hour"] is None
                        else f"{entry['share_of_days_under_a_hundredth_of_an_hour']:.1%}"
                    ),
                ]
                for name, entry in sorted(features.items())
            ],
        ),
    ]


def _between_part(payload: Mapping[str, Any]) -> list[str]:
    listed = payload["configuration"]["between_the_freeze_and_the_run"]
    if not listed:
        return []
    return [
        "",
        "## Between the freeze and the run",
        "",
        "What was run and read after the protocol was frozen and before it was "
        "run. None of it touched a home of the protocol.",
        "",
        *[f"- {_sentence(item)}" for item in listed],
    ]


def _tihm_cell(entry: Mapping[str, Any]) -> str:
    if entry["share"] is None:
        return "–"
    equivalent = entry["equivalent_threshold"]
    shown = "none passed" if equivalent is None else f"{equivalent:.2f}"
    return f"{100.0 * entry['share']:.1f}% ({shown})"


def _tihm_part(tihm: Mapping[str, Any], payload: Mapping[str, Any]) -> list[str]:
    results, configuration = tihm["results"], tihm["configuration"]
    if results.get("result_schema") != TIHM_SCHEMA:
        raise ValueError(f"not a {TIHM_SCHEMA} record")
    if configuration["protocol_sha256"] != payload["configuration"]["protocol_sha256"]:
        raise ValueError("the two records were not made under the same protocol")
    declared = configuration["tihm"]
    environment = tihm["environment"]
    roles = _roles(payload["results"])
    lines = [
        "",
        "## TIHM: what the references do on the homes that showed the problem",
        "",
        f"**{_sentence(declared['standing'])}** "
        f"{_sentence(declared['what_it_cannot_say'])}",
        "",
        _figure_of(
            "tihm",
            "Share of TIHM days past a threshold, for both references, with the "
            "silent-home rule off and on",
        ),
        "",
        f"- **Run.** Commit `{environment.get('git_commit', 'unknown')[:7]}`, "
        + (
            "with uncommitted changes"
            if environment.get("git_dirty") != "false"
            else "with no uncommitted change"
        )
        + f"; recorded {tihm['recorded_at']}. {results['homes']} homes.",
        f"- **Code.** {_code_line(configuration)}",
        "- **Checked.** The replay under the default reference returned each "
        "run's verdicts and alerts. In "
        f"{len(results['check']['calibrated_runs']['homes'])} homes, the first "
        "by identifier, the pipeline was also run with the calibrated "
        "reference itself, and the replay under "
        f"`{results['check']['calibrated_runs']['condition']}` returned what "
        "those runs concluded. The homes are the published record's, and the "
        "runs gave the published records' counts: "
        + "; ".join(
            f"{results['check'][rule]['monitored_days']:,} monitored days and "
            f"{results['check'][rule]['behavioural_alerts']} behavioural alerts "
            f"with the rule `{rule}`"
            for rule in declared["silent_home_rule"]
        )
        + ".",
        f"- **Not reported.** {_sentence(declared['not_reported'])}",
        f"- **The dataset.** {declared['citation']} {declared['licence']}. The "
        "dataset is not redistributed here.",
    ]
    for rule in declared["silent_home_rule"]:
        entry = results["rules"][rule]
        conditions = entry["conditions"]
        tail = _by_reference(entry["tail"])
        features = sorted(next(iter(tail.values()))) if tail else []
        scales = sorted({name.partition("@")[2] for name in conditions}, key=float)
        lines += [
            "",
            f"### The silent-home rule `{rule}`",
            "",
            f"{entry['monitored_days']:,} monitored days, "
            f"{entry['usable_days']:,} usable and {entry['evaluable_days']:,} "
            "evaluable.",
            "",
            "Behavioural alerts, and in brackets the homes with one, at each "
            "multiple of the declared thresholds. The last column says what a "
            "multiple was shown as on the simulated homes.",
            "",
            *_table(
                ["Multiple", "Default", "Calibrated", "On the simulated homes"],
                [
                    [
                        scale,
                        *(
                            f"{conditions[f'{reference}@{scale}']['behavioural_alerts']['all']:,} "
                            f"({conditions[f'{reference}@{scale}']['behavioural_alerts']['homes_with_one']})"
                            for reference in _REFERENCES
                        ),
                        "; ".join(
                            _ROLE_TITLES[role]
                            for reference in _REFERENCES
                            for role in roles.get(f"{reference}@{scale}", [])
                        ),
                    ]
                    for scale in scales
                ],
            ),
            "",
            "What raised them, under the conditions described in full on the "
            "simulated homes:",
            "",
            *_table(
                ["Condition", "Behavioural alerts", "By verdict", "Change verdicts"],
                [
                    [
                        f"`{condition}`",
                        _n(conditions[condition]["behavioural_alerts"]["all"]),
                        _counts(
                            conditions[condition]["behavioural_alerts"]["by_verdict"]
                        ),
                        _n(conditions[condition]["change_verdicts"]["all"]),
                    ]
                    for condition in payload["results"]["reported"]
                ],
            ),
        ]
        for feature in features:
            points = sorted(tail[next(iter(tail))][feature], key=float)
            lines += [
                "",
                f"The share of evaluable days of `{feature}` at or past a "
                "threshold, and in brackets the threshold that share is "
                "equivalent to on Gaussian days:",
                "",
                *_table(
                    ["Reference", *(f"At {point}" for point in points)],
                    [
                        [
                            reference,
                            *(
                                _tihm_cell(tail[reference][feature][point])
                                for point in points
                            ),
                        ]
                        for reference in tail
                    ],
                ),
            ]
    return lines


def _limits_part(payload: Mapping[str, Any]) -> list[str]:
    cannot = payload["configuration"]["what_the_simulator_cannot_show"]
    return [
        "",
        "## What this does not show",
        "",
        *[f"- {_sentence(item)}" for item in cannot],
    ]


def _homes_part(payload: Mapping[str, Any]) -> list[str]:
    configuration, results = payload["configuration"], payload["results"]
    arms = _arms(configuration)
    homes = payload["household_metrics"]["simulated_homes"]
    return [
        "",
        "## Every simulated home",
        "",
        "False alerts under "
        + ", ".join(f"`{condition}`" for condition in results["reported"])
        + ", in that order. Each string has one character for each of those "
        "conditions, in the same order: 1 where the home is detected, and "
        "after the stroke 1 where its stable record is falsely detected.",
        "",
        *_table(
            [
                "Seed",
                "Usable days",
                "False alerts",
                *(_ARM_TITLES[arm].capitalize() for arm in arms),
            ],
            [
                [
                    f"`{seed}`",
                    row["usable_days"],
                    " ".join(str(count) for count in row["false_alerts"]),
                    *(
                        f"{row['detected'][arm]} / {row['false_detection'][arm]}"
                        for arm in arms
                    ),
                ]
                for seed, row in sorted(homes.items(), key=lambda item: int(item[0]))
            ],
        ),
    ]


def render_page(
    payload: Mapping[str, Any], tihm: Mapping[str, Any] | None = None
) -> str:
    """Render the results page from the records."""
    if payload["results"].get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    lines = [
        *_results_head(payload),
        *_criteria_part(payload),
        *_match_part(payload),
        *_stable_part(payload),
        *_detection_part(payload),
        *_calibration_part(payload),
        *_verdicts_part(payload),
        *_curves_part(payload),
        *_features_part(payload),
        *_between_part(payload),
        *(_tihm_part(tihm, payload) if tihm is not None else []),
        *_limits_part(payload),
        *_homes_part(payload),
    ]
    return "\n".join(lines) + "\n"
