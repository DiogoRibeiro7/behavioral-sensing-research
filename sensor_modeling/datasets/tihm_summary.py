"""The TIHM pages, generated entirely from the frozen protocol and the records.

:func:`render_protocol` writes ``docs/TIHM_ALERT_BURDEN_PROTOCOL.md`` from the
frozen protocol file alone. It reports no result.

:func:`render_page` writes ``docs/TIHM_ALERT_BURDEN_RESULTS.md`` from the
record of the protocol's run, and, apart from it and marked as such, from the
record of the descriptions made after that run had been read. Every household
is shown, whatever the pipeline raised in it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .tihm_post_hoc import POST_HOC_SCHEMA
from .tihm_protocol import RESULT_SCHEMA

PROTOCOL_FILE = "artifacts/tihm/alert_burden_protocol.json"
MAPPING_FILE = "artifacts/tihm/tihm_mapping.json"
RECORD_FILE = "artifacts/tihm/tihm-alert-burden.json"
POST_HOC_FILE = "artifacts/tihm/tihm-alert-burden-post-hoc.json"
PROTOCOL_PAGE = "TIHM_ALERT_BURDEN_PROTOCOL.md"
RESULTS_PAGE = "TIHM_ALERT_BURDEN_RESULTS.md"
FIGURE_DIR = "figures"
FIGURE_PREFIX = "tihm-alert-burden"


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
    text = text[0].upper() + text[1:]
    return text if text.endswith(".") else text + "."


def _bullets(entries: Mapping[str, Any]) -> list[str]:
    return [
        f"- **{name.replace('_', ' ').capitalize()}.** {_sentence(str(text))}"
        for name, text in entries.items()
    ]


def render_protocol(payload: Mapping[str, Any]) -> str:
    """The protocol page, from the frozen protocol file's content."""
    dataset, mapping, pipeline = (
        payload["dataset"],
        payload["mapping"],
        payload["pipeline"],
    )
    published, labels = payload["published_baseline"], payload["label_description"]
    lines = [
        "# TIHM: the alert-burden protocol",
        "",
        "The TIHM dataset holds 56 homes of people living with dementia, with "
        "alerts a clinical monitoring team verified. This page is the protocol "
        "for running the online pipeline on it, generated entirely from the "
        f"frozen file `{PROTOCOL_FILE}` by "
        "`sensor_modeling.datasets.tihm_summary.render_protocol`.",
        "",
        f"**Status: {payload['status']}.**",
        "",
        f"- **Protocol digest.** `{payload['protocol_sha256']}`.",
        f"- **The question.** {_sentence(payload['question'])}",
        "- **This page reports no result.** The scoring run checks the frozen "
        "file, the mapping and every dataset file's digest before it runs.",
        "",
        "## What had been seen",
        "",
        "The protocol was written after the labels had been analysed. It fixes "
        "every definition before any pipeline output is compared with a label; "
        "it does not make the result a held-out claim.",
        "",
        *[f"- {_sentence(item)}" for item in payload["inspected_before"]],
        "",
        "## The dataset",
        "",
        *_table(
            ["Field", "Value"],
            [
                ["name", dataset["name"]],
                ["version", dataset["version"]],
                ["source", dataset["source"]],
                ["licence", dataset["licence"]],
                ["citation", dataset["citation"]],
                ["archive SHA-256", f"`{dataset['archive_sha256']}`"],
                ["timezone", f"`{dataset['timezone']}`, declared"],
            ],
        ),
        "",
        "Every file's SHA-256:",
        "",
        *_table(
            ["File", "SHA-256", "Use"],
            [
                [
                    name,
                    f"`{digest}`",
                    (
                        "read through the contract"
                        if name in dataset["read_through_the_contract"]
                        else (
                            "read outside the contract"
                            if name in dataset["read_outside_the_contract"]
                            else "pinned, not read"
                        )
                    ),
                ]
                for name, digest in dataset["files"].items()
            ],
        ),
        "",
        *_bullets(dataset["read_outside_the_contract"]),
        "- **Redistribution.** None: the run reads an extracted download and "
        "verifies every file's digest.",
        f"- **Households.** {_sentence(payload['households'])}",
        "",
        "## Ontology mapping",
        "",
        f"The mapping is `{MAPPING_FILE}`, SHA-256 `{mapping['sha256']}`.",
        "",
        *_table(
            ["Label", "Outcome"],
            [[f"`{label}`", outcome] for label, outcome in mapping["labels"].items()],
        ),
        "",
        *_table(
            ["Location name", "Native sensor type"],
            [[f"`{name}`", f"`{kind}`"] for name, kind in mapping["sensors"].items()],
        ),
        "",
        f"- **Consequence.** {_sentence(mapping['consequence'])}",
        "",
        "## The pipeline",
        "",
        f"{_sentence(pipeline['what'])} Fitted on TIHM: "
        f"{pipeline['fitted_on_tihm']}.",
        "",
        *_table(
            ["Setting", "Value"],
            [
                ["step", f"{pipeline['step_minutes']} minutes"],
                ["baseline features", ", ".join(pipeline["features"])],
                ["minimum day coverage", pipeline["min_day_coverage"]],
                ["minimum day observed", pipeline["min_day_observed"]],
                ["attribute activity to the resident", pipeline["attribute_activity"]],
                *[[f"baseline {k}", v] for k, v in pipeline["baseline"].items()],
                *[
                    [f"alert policy {k}", v]
                    for k, v in pipeline["alert_policy"].items()
                ],
            ],
        ),
        "",
        "## Definitions",
        "",
        *_bullets(payload["definitions"]),
        "",
        "## Estimands",
        "",
        *_table(
            ["Estimand", "What it is"],
            [[name, _sentence(text)] for name, text in payload["estimands"].items()],
        ),
        "",
        "## References",
        "",
        "Two rules that use no model, and chance. They say what a flag has to " "beat.",
        "",
        *_bullets(payload["references"]),
        "",
        "## The published baseline",
        "",
        f"The dataset is published with a notebook, `{published['notebook']}`, "
        f"SHA-256 `{published['notebook_sha256']}`, that trains a "
        f"{published['model']} to recognise label days and prints its confusion "
        "matrices. They are quoted here; the model is not retrained.",
        "",
        *_table(
            ["Test week, newest first", "TN", "FP", "FN", "TP"],
            [
                [k + 1, m[0][0], m[0][1], m[1][0], m[1][1]]
                for k, m in enumerate(published["confusion_matrices_newest_first"])
            ],
        ),
        "",
        *_bullets(
            {
                k: published[k]
                for k in ("protocol", "check", "label_history_here", "not_done")
            }
        ),
        "",
        "## Describing the labels",
        "",
        f"- **Slots.** A label is on a slot when it is stamped less than "
        f"{labels['slot_tolerance_seconds']} seconds after "
        f"{', '.join(f'{h:02d}:00' for h in labels['slots'])}.",
        f"- **Hourly profile.** {_sentence(labels['hourly_profile'])}",
        f"- **Blocks.** {', '.join(labels['blocks'])}.",
        f"- **Full day.** {_sentence(labels['full_day'])}",
        "",
        "The limits the dataset paper states for the daily measurements:",
        "",
        *_table(
            ["Label", "Device", "Below", "Above"],
            [
                [label, device, below, above]
                for label, limits in labels["stated_limits"].items()
                for device, below, above in limits
            ],
        ),
        "",
        "## Uncertainty",
        "",
        *_bullets(payload["bootstrap"]),
        "",
        "## The simulator's figures",
        "",
        "Beside B2, as a reference and not a test.",
        "",
        *_table(
            ["Field", "Value"],
            [
                [name.replace("_", " "), f"`{value}`"]
                for name, value in payload["simulator_reference"].items()
            ],
        ),
        "",
        "## Reporting",
        "",
        f"{_sentence(payload['reporting'])}",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# The results page
# ----------------------------------------------------------------------------
def _n(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if float(value).is_integer() and abs(value) >= 10:
        return f"{int(value):,}"
    return f"{value:.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{100.0 * value:.{digits}f}%"


def _count(value: Any) -> str:
    """A count that ties at a cut may have left fractional."""
    if value is None:
        return "—"
    return f"{int(value):,}" if float(value).is_integer() else f"{value:.1f}"


def _rate(entry: Mapping[str, Any], digits: int = 4) -> str:
    """An estimate and its interval, as a rate."""
    if entry.get("estimate") is None:
        return "—"
    text = f"{entry['estimate']:.{digits}f}"
    interval = entry.get("interval")
    if interval is None:
        return text
    return f"{text} [{interval['low']:.{digits}f}, {interval['high']:.{digits}f}]"


def _share(entry: Mapping[str, Any]) -> str:
    """An estimate and its interval, as a percentage."""
    if entry.get("estimate") is None:
        return "—"
    text = _pct(entry["estimate"])
    interval = entry.get("interval")
    if interval is None:
        return text
    return f"{text} [{_pct(interval['low'])}, {_pct(interval['high'])}]"


def _points(entry: Mapping[str, Any]) -> str:
    """A difference of shares and its interval, in percentage points."""
    if entry.get("estimate") is None:
        return "—"
    text = f"{100.0 * entry['estimate']:+.1f}"
    interval = entry.get("interval")
    if interval is None:
        return text
    return (
        f"{text} [{100.0 * interval['low']:+.1f}, " f"{100.0 * interval['high']:+.1f}]"
    )


def _against(entry: Mapping[str, Any], value: float, name: str) -> str:
    """Where an interval lies against *value*, in words."""
    interval = entry.get("interval")
    if interval is None:
        return "no interval"
    if interval["low"] > value:
        return f"above {name}"
    if interval["high"] < value:
        return f"below {name}"
    return f"includes {name}"


def _range(entry: Any, style: str = "count") -> str:
    """A replay's value, or its mean and range over replicates."""
    fmt = _pct if style == "share" else (lambda v: f"{v:,.1f}")
    if isinstance(entry, Mapping):
        return f"{fmt(entry['mean'])} ({fmt(entry['low'])} to {fmt(entry['high'])})"
    return _pct(entry) if style == "share" else _count(entry)


def _name(text: str) -> str:
    return text.replace("_", " ")


def _alerts_per_day(count: int, entry: Mapping[str, Any]) -> str:
    """A count and its rate; a count of zero is given no interval."""
    if not count:
        return "none"
    return f"{_n(count)}, {_rate(entry)} per monitored day"


def _signed(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    text = f"{value:+.{digits}f}"
    return text[1:] if float(text) == 0.0 else text


_DAY_GROUPS = ("alert_days", "deviating_days", "label_days", "all_evaluable_days")


def _ordered(groups: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The day groups in a fixed order, whatever order the record holds them in."""
    known = [(name, groups[name]) for name in _DAY_GROUPS if name in groups]
    return known + [(n, g) for n, g in groups.items() if n not in _DAY_GROUPS]


_REPLAY_TITLES = {
    "recorded": "Recorded",
    "silent_days_skipped": "Silent days left out",
    "shuffled": "Shuffled",
    "stationary_gaussian": "Stationary Gaussian",
}


def _flag_rows(entries: Mapping[str, Mapping[str, Any]]) -> list[list[Any]]:
    return [
        [
            name,
            f"{_count(e['label_days_flagged'])} of {_count(e['label_days'])}",
            _share(e["share_of_label_days_flagged"]),
            _share(e["share_of_other_days_flagged"]),
            _points(e["difference"]),
            _against(e["difference"], 0.0, "zero"),
            _share(e["share_of_flagged_days_labelled"]),
        ]
        for name, e in entries.items()
    ]


_FLAG_HEADER = [
    "Flag",
    "Label days flagged",
    "Share of label days",
    "Share of other days",
    "Difference, points",
    "The interval of the difference",
    "Share of flagged days that are label days",
]


def _figure(name: str, alt: str) -> str:
    return f"![{alt}]({FIGURE_DIR}/{FIGURE_PREFIX}-{name}.svg)"


def _protocol_part(payload: Mapping[str, Any], lines: list[str]) -> None:
    results = payload["results"]
    configuration, environment = payload["configuration"], payload["environment"]
    contract, monitoring = results["contract"], results["monitoring"]
    burden, association = results["burden"], results["association"]
    references, published = results["references"], results["published_baseline"]
    labels, dataset = results["labels"], configuration["dataset"]
    primary = labels["primary"]
    add = lines.append

    add(f"**Status: {configuration['status']}.**")
    add("")
    add(
        f"- **Protocol.** SHA-256 `{configuration['protocol_sha256']}`, checked "
        "against the frozen file, with the mapping and every dataset file's "
        "digest, before the run."
    )
    add(
        f"- **Run.** Commit `{environment['git_commit'][:7]}`, "
        + (
            "with no uncommitted change"
            if environment["git_dirty"] == "false"
            else "with uncommitted changes"
        )
        + f"; recorded {payload['recorded_at']}."
    )
    lines.extend(f"- {note}" for note in payload["notes"])
    add(
        f"- **Dataset.** {dataset['citation']} Licence: {dataset['licence']}. "
        "The dataset is not redistributed here."
    )
    add("")

    # ------------------------------------------------------------------
    add("## What the contract converts")
    add("")
    refused = contract["refused"]
    lines.extend(
        _table(
            ["", "Count"],
            [
                ["households", _n(contract["households"])],
                ["converted and run", _n(contract["converted"])],
                [
                    "refused by the contract",
                    ", ".join(sorted(refused)) if refused else "none",
                ],
                *[
                    [f"sensor events, {_name(outcome)}", _n(count)]
                    for outcome, count in contract["events"].items()
                ],
                ["labels, as point annotations", _n(contract["point_annotations"])],
                [
                    "annotated seconds that map to a behavioural state",
                    _n(contract["scorable_annotated_seconds"], 0),
                ],
                *[
                    [f"validation notice `{code}`", _n(count)]
                    for code, count in contract["issues"].items()
                ],
            ],
        )
    )
    add("")
    if not contract["scorable_annotated_seconds"]:
        add(
            "No annotated time maps to a behavioural state, so state inference "
            "is not scored on this dataset. Everything below is about what the "
            "pipeline raises, not about whether its states are right."
        )
        add("")

    # ------------------------------------------------------------------
    add("## How much of the record the pipeline uses (H1)")
    add("")
    lines.extend(
        _table(
            ["", "Value"],
            [
                ["monitored days", _n(monitoring["monitored_days"])],
                [
                    "usable days",
                    f"{_n(monitoring['usable_days'])}, "
                    f"{_share(monitoring['usable_share'])}",
                ],
                [
                    "evaluable days",
                    f"{_n(monitoring['evaluable_days'])}, "
                    f"{_share(monitoring['evaluable_share'])}",
                ],
                [
                    "households with an evaluable day",
                    f"{_n(monitoring['households_with_an_evaluable_day'])} of "
                    f"{_n(contract['converted'])}",
                ],
                ["mean sensor coverage", _n(monitoring["mean_coverage"])],
                ["mean observed fraction of the day", _n(monitoring["mean_observed"])],
                [
                    "mean abstention on usable days",
                    _pct(monitoring["mean_abstention"], 2),
                ],
                [
                    "system-health alerts",
                    _alerts_per_day(
                        monitoring["system_health_alerts"],
                        monitoring["system_health_per_monitored_day"],
                    ),
                ],
                [
                    "data-quality alerts",
                    _alerts_per_day(
                        monitoring["data_quality_alerts"],
                        monitoring["data_quality_per_monitored_day"],
                    ),
                ],
            ],
        )
    )
    add("")
    add("Feature-days by the baseline's verdict:")
    add("")
    lines.extend(
        _table(
            ["Verdict", "Feature-days"],
            [
                [_name(kind), _n(count)]
                for kind, count in monitoring["verdicts"].items()
            ],
        )
    )
    add("")

    # ------------------------------------------------------------------
    simulator = burden["simulator_reference"]
    add("## The alert burden (B1 to B3)")
    add("")
    add(_figure("burden", "Behavioural alerts per person-day"))
    add("")
    lines.extend(
        _table(
            ["Estimand", "Behavioural alerts per person-day"],
            [
                [
                    "B1: all features, per monitored day",
                    _rate(burden["B1_per_monitored_day"]),
                ],
                [
                    f"B2: `{simulator['feature']}` alone, per monitored day",
                    _rate(burden["B2_simulator_feature_per_monitored_day"]),
                ],
                [
                    f"simulator, stable arm, `{simulator['feature']}`: a "
                    "reference, not a test",
                    _n(simulator["stable_arm"]),
                ],
                [
                    f"simulator, changed arm, `{simulator['feature']}`: a "
                    "reference, not a test",
                    _n(simulator["changed_arm"]),
                ],
                [
                    "B3: all features, per evaluable day",
                    _rate(burden["B3_per_evaluable_day"]),
                ],
            ],
        )
    )
    add("")
    add(f"The simulator's figures are from {simulator['source']}.")
    add("")
    lines.extend(
        _table(
            ["", "Value"],
            [
                ["behavioural alerts", _n(burden["behavioural_alerts"])],
                *[
                    [f"about `{feature}`", _n(count)]
                    for feature, count in burden["by_feature"].items()
                ],
                *[
                    [f"of severity {severity}", _n(count)]
                    for severity, count in burden["by_severity"].items()
                ],
                [
                    "alert days",
                    f"{_n(burden['alert_days'])}, "
                    f"{_share(burden['alert_day_share_of_evaluable_days'])} of "
                    "evaluable days",
                ],
                [
                    "deviating days",
                    f"{_n(burden['deviating_days'])}, "
                    f"{_share(burden['deviating_day_share_of_evaluable_days'])} "
                    "of evaluable days",
                ],
                [
                    "households with an alert",
                    f"{_n(burden['households_with_an_alert'])} of "
                    f"{_n(contract['converted'])}",
                ],
                ["most alerts in one household", _n(burden["largest_household_count"])],
            ],
        )
    )
    add("")

    # ------------------------------------------------------------------
    a1, a2, a3 = (
        association["A1_alert_days"],
        association["A2_deviating_days"],
        association["A3_deviation_score"],
    )
    add("## Whether the flags relate to the verified labels (A1 to A3)")
    add("")
    add(
        f"Of the {_n(a1['days'])} evaluable days, {_n(a1['label_days'])} are "
        f"`{primary['label']}` label days, in "
        f"{_n(a1['households_with_a_label_day'])} households: "
        f"{_share(a1['label_day_share'])} of evaluable days."
    )
    add("")
    lines.extend(
        _table(_FLAG_HEADER, _flag_rows({"A1: alert day": a1, "A2: deviating day": a2}))
    )
    add("")
    add(
        "- **A3.** A label day has a higher deviation score than another "
        f"evaluable day of the same household with probability {_rate(a3, 3)}, "
        f"over {_count(a3['pairs'])} pairs in {_n(a3['households_with_pairs'])} "
        "households. A score unrelated to the labels gives 0.5; the interval "
        f"{_against(a3, 0.5, '0.5')}."
    )
    add("")
    other = association["any_label"]
    add("The same against a label of any type:")
    add("")
    lines.extend(
        _table(
            _FLAG_HEADER,
            _flag_rows(
                {
                    "alert day": other["alert_days"],
                    "deviating day": other["deviating_days"],
                }
            ),
        )
    )
    add("")
    add(
        "- **Deviation score.** Concordance "
        f"{_rate(other['deviation_score'], 3)}, over "
        f"{_count(other['deviation_score']['pairs'])} pairs in "
        f"{_n(other['deviation_score']['households_with_pairs'])} households; "
        f"the interval {_against(other['deviation_score'], 0.5, '0.5')}."
    )
    add("")

    # ------------------------------------------------------------------
    add("## What a flag has to beat (R1)")
    add("")
    add(_figure("references", "Label days caught at an equal number of flags"))
    add("")
    add(
        f"Each rule flags {_n(references['flags'])} of the "
        f"{_n(references['days'])} evaluable days, as many as the pipeline has "
        f"deviating days. There are {_n(references['label_days'])} label days."
    )
    add("")
    rows = [
        [
            "pipeline: deviating days",
            _count(references["pipeline_deviating_days"]["label_days_caught"]),
            _share(references["pipeline_deviating_days"]["share_of_label_days"]),
            "—",
        ],
        [
            "flagged at random, expected",
            _count(references["random_expected_caught"]),
            _pct(
                references["random_expected_caught"] / references["label_days"]
                if references["label_days"]
                else None
            ),
            "—",
        ],
    ]
    for name, title in (
        ("event_count", "event count against the household's earlier days"),
        ("label_history", "label history, no sensor"),
    ):
        entry = references[name]
        rows.append(
            [
                title,
                _count(entry["label_days_caught"]),
                _share(entry["share_of_label_days"]),
                _points(entry["minus_pipeline"]),
            ]
        )
    lines.extend(
        _table(
            [
                "Rule",
                "Label days caught",
                "Share of label days",
                "Minus the pipeline, points",
            ],
            rows,
        )
    )
    add("")
    concordance = references["event_count"]["concordance"]
    add(
        "- **Event count, without a cut.** A label day has a higher event-count "
        "score than another evaluable day of the same household with "
        f"probability {_rate(concordance, 3)}; the interval "
        f"{_against(concordance, 0.5, '0.5')}."
    )
    add("")

    # ------------------------------------------------------------------
    quoted = published["published"]
    add("## On the published baseline's protocol (P1)")
    add("")
    if not published["reproduced_test_periods"]:
        add(
            "**Not reported.** The test periods rebuilt here do not have the "
            "published numbers of person-days and label days:"
        )
        add("")
    else:
        add(
            "Every test period rebuilt here has the published number of "
            "person-days and label days:"
        )
        add("")
    lines.extend(
        _table(
            ["Test week, newest first", "Person-days", "Label days"],
            [
                [k + 1, _n(period["person_days"]), _n(period["label_days"])]
                for k, period in enumerate(published["test_periods"])
            ],
        )
    )
    add("")
    if published["reproduced_test_periods"]:
        add(
            f"Each rule raises the published model's {_n(quoted['alerts'])} "
            f"alerts, week by week, over {_n(quoted['test_person_days'])} "
            f"person-days with {_n(quoted['label_days'])} label days."
        )
        add("")
        lines.extend(
            _table(
                ["Rule", "Label days caught", "Share of label days"],
                [
                    [
                        "published logistic regression, quoted",
                        _count(quoted["label_days_caught"]),
                        _pct(quoted["share_of_label_days_caught"]),
                    ],
                    [
                        "label history, no sensor",
                        _count(published["label_history"]["label_days_caught"]),
                        _share(published["label_history"]["share_of_label_days"]),
                    ],
                    [
                        "event count against the household's earlier days",
                        _count(published["event_count"]["label_days_caught"]),
                        _share(published["event_count"]["share_of_label_days"]),
                    ],
                    [
                        "flagged at random, expected",
                        _count(quoted["random_expected_caught"]),
                        _pct(quoted["random_expected_caught"] / quoted["label_days"]),
                    ],
                ],
            )
        )
        add("")
        add(
            "- **The published model, quoted.** Mean weekly sensitivity "
            f"{_n(quoted['mean_weekly_sensitivity'])} and specificity "
            f"{_n(quoted['mean_weekly_specificity'])}; "
            f"{_n(quoted['alerts_on_other_days'])} of its alerts are on other "
            f"days, so {_pct(quoted['share_of_alerts_on_label_days'])} of its "
            "alerts are on label days; "
            f"{_n(quoted['alerts_per_person_month'], 2)} alerts per "
            "person-month, "
            f"{_n(quoted['alerts_on_other_days_per_person_month'], 2)} of them "
            "on other days."
        )
        add(
            f"- **Concentration.** {_n(published['households_in_test'])} "
            "households have a test day, "
            f"{_n(published['households_with_a_label_day'])} of them a label "
            "day, and the five with most hold "
            f"{_pct(published['top_five_households_share_of_label_days'])} of "
            "the label days."
        )
        add("")

    # ------------------------------------------------------------------
    add("## What the labels are")
    add("")
    lines.extend(
        _table(
            ["Label", "Rows", "Label days", "Households"],
            [
                [
                    f"`{kind}`",
                    _n(labels["rows_by_type"][kind]),
                    _n(labels["label_days_by_type"][kind]),
                    _n(labels["households_by_type"][kind]),
                ]
                for kind in labels["rows_by_type"]
            ],
        )
    )
    add("")
    add(
        f"{_n(labels['households_without_any_label'])} households have no label "
        f"of any type. Of the {_n(primary['rows'])} `{primary['label']}` rows, on "
        f"{_n(primary['label_days'])} days:"
    )
    add("")
    slots = ", ".join(
        f"{_n(count)} at {slot}" if slot[:2].isdigit() else f"{_n(count)} elsewhere"
        for slot, count in primary["rows_by_slot"].items()
    )
    add(
        f"- **When.** {slots}; the latest on a slot is "
        f"{_n(primary['latest_seconds_past_a_slot'])} seconds past it."
    )
    add(
        f"- **Who.** {_n(primary['households'])} households have one and "
        f"{_n(primary['households_never_labelled'])} never do; the household "
        f"with most holds {_pct(primary['largest_household_share'])} of the "
        "label days and the five with most "
        f"{_pct(primary['five_largest_households_share'])}."
    )
    add(
        "- **After a label day.** "
        f"{_pct(primary['share_labelled_after_a_label_day'])} of the days "
        "after a label day are label days, against "
        f"{_pct(primary['share_labelled_after_another_day'])} of the days after "
        "another monitored day."
    )
    add("")
    add(_figure("profile", "Hourly sensor events on label days"))
    add("")
    profile = labels["hourly_profile"]
    blocks = [name for group in profile.values() for name in group["block_mean_z"]]
    blocks = list(dict.fromkeys(blocks))
    add(
        "Sensor events in each block of the day, as a mean z-score against the "
        "same household's block over its full monitored days:"
    )
    add("")
    lines.extend(
        _table(
            ["Slot of the day's first label", "Days", *blocks],
            [
                [
                    "no label" if group == "none" else group,
                    _n(entry["days"]),
                    *[_signed(entry["block_mean_z"].get(block)) for block in blocks],
                ]
                for group, entry in profile.items()
            ],
        )
    )
    add("")
    add("Label days against the limits the dataset paper states:")
    add("")
    lines.extend(
        _table(
            [
                "Label",
                "Label days",
                "Days with a reading",
                "Days meeting the limits",
                "Label days meeting them",
                "Label days without a reading",
                "Days meeting them without a label",
                "Readings, lowest to highest",
            ],
            [
                [
                    f"`{label}`",
                    _n(entry["label_days"]),
                    _n(entry["days_with_a_reading"]),
                    _n(entry["days_meeting_the_limits"]),
                    _n(entry["label_days_meeting_the_limits"]),
                    _n(entry["label_days_without_a_reading"]),
                    _n(entry["days_meeting_the_limits_without_a_label"]),
                    "; ".join(
                        f"{device} {'—' if span is None else f'{span[0]:g} to {span[1]:g}'}"
                        for device, span in entry["reading_range"].items()
                    ),
                ]
                for label, entry in labels["stated_limits"].items()
            ],
        )
    )
    add("")

    # ------------------------------------------------------------------
    add("## Every household")
    add("")
    lines.extend(
        _table(
            [
                "Household",
                "Monitored days",
                "Usable",
                "Evaluable",
                "Behavioural alerts",
                "Alert days",
                "Deviating days",
                f"{primary['label']} label days",
                "Days with any label",
                "Sensor events",
            ],
            [
                [
                    f"`{name}`",
                    _n(row["monitored_days"]),
                    _n(row["usable_days"]),
                    _n(row["evaluable_days"]),
                    _n(row["behavioural_alerts"]),
                    _n(row["alert_days"]),
                    _n(row["deviating_days"]),
                    _n(row["label_days"]),
                    _n(row["any_label_days"]),
                    _n(row["events"]),
                ]
                for name, row in results["households"].items()
            ],
        )
    )
    add("")


def _post_hoc_part(payload: Mapping[str, Any], lines: list[str]) -> None:
    results = payload["results"]
    add = lines.append
    described = results["describes"]
    hours, silent = results["state_hours"], results["silent_days"]
    replays, size = results["replays"], results["reference_size"]
    direction = results["direction"]

    add("## After the run: post hoc descriptions")
    add("")
    add(f"**Status: {results['status']}.**")
    add("")
    add(
        f"This part is generated from a second record, `{POST_HOC_FILE}`. It "
        f"describes the run recorded {described['recorded_at']} at commit "
        f"`{described['git_commit'][:7]}`: the households were run again, and "
        "the descriptions were refused unless scoring them gave the results "
        "above."
    )
    add("")
    add(
        "The protocol reports with no retuning and no added reference after "
        "scoring, and that holds for everything above. What follows adds "
        "comparisons and one changed replay to explain those results. It "
        "replaces none of them, it carries no interval over households, and "
        "every heading and figure below is post hoc."
    )
    add("")
    lines.extend(f"- {note}" for note in payload["notes"])
    add("")

    # ------------------------------------------------------------------
    add("### Post hoc: what the filter infers")
    add("")
    add(
        f"Expected hours per state over the {_n(hours['usable_days'])} usable "
        "days. No annotation says what the residents were doing, so these are "
        "not scored; they are what the baseline's features are made of."
    )
    add("")
    lines.extend(
        _table(
            ["State", "Median hours", "Tenth percentile", "Ninetieth percentile"],
            [
                [
                    _name(state),
                    _n(entry["median"], 2),
                    _n(entry["tenth"], 2),
                    _n(entry["ninetieth"], 2),
                ]
                for state, entry in hours["hours"].items()
            ],
        )
    )
    add("")
    household = hours["household_median_sleeping_hours"]
    add(
        "A household's median day has between "
        f"{_n(household['smallest'], 1)} and {_n(household['largest'], 1)} "
        f"hours inferred as sleeping, with a median of "
        f"{_n(household['median'], 1)} over {_n(household['households'])} "
        "households."
    )
    add("")

    # ------------------------------------------------------------------
    add("### Post hoc: silent days")
    add("")
    add(f"A silent day is {silent['definition']}.")
    add("")
    medians = silent["median_sleeping_hours"]
    lines.extend(
        _table(
            ["", "Value"],
            [
                [
                    "silent days",
                    f"{_n(silent['silent_days'])} of "
                    f"{_n(silent['monitored_days'])} monitored days, "
                    f"{_pct(silent['silent_days'] / silent['monitored_days'])}",
                ],
                ["households with one", _n(silent["households"])],
                ["longest run of silent days", _n(silent["longest_run"])],
                [
                    "runs by length in days",
                    ", ".join(
                        f"{length}: {count}"
                        for length, count in sorted(
                            silent["runs_by_length"].items(),
                            key=lambda item: int(item[0]),
                        )
                    )
                    or "none",
                ],
                ["silent days the pipeline counts as usable", _n(silent["usable"])],
                ["silent days that are evaluable", _n(silent["evaluable"])],
                [
                    "median hours inferred as sleeping, silent days",
                    _n(medians["silent_days"], 1),
                ],
                [
                    "median hours inferred as sleeping, other usable days",
                    _n(medians["other_days"], 1),
                ],
                [
                    "change verdicts on a silent day",
                    f"{_n(silent['change_verdicts_on_silent_days'])} of "
                    f"{_n(silent['change_verdicts'])}",
                ],
                [
                    "behavioural alerts on a silent day",
                    f"{_n(silent['alerts_on_silent_days'])} of "
                    f"{_n(silent['behavioural_alerts'])}",
                ],
                *[
                    [f"of them, `{key}`", _n(count)]
                    for key, count in silent[
                        "alerts_on_silent_days_by_feature_and_verdict"
                    ].items()
                ],
            ],
        )
    )
    add("")
    lines.extend(
        _table(
            ["Evaluable days", "Days", "Silent", "Share silent"],
            [
                [
                    _name(name),
                    _n(entry["days"]),
                    _n(entry["silent"]),
                    _pct(entry["share"]),
                ]
                for name, entry in _ordered(silent["share_silent"])
            ],
        )
    )
    add("")

    # ------------------------------------------------------------------
    calendar = results["calendar"]
    peak, after = (
        calendar["most_silent_date"],
        calendar["day_after_the_most_silent_date"],
    )
    add("### Post hoc: the calendar")
    add("")
    add(_figure("calendar", "Post hoc: sensor events and behavioural alerts by date"))
    add("")
    if peak is not None and peak["silent_households"]:
        add(
            f"The date on which most households are silent is {peak['date']}: "
            f"{_n(peak['silent_households'])} of the "
            f"{_n(peak['monitored_households'])} monitored that day. The days "
            "around it:"
        )
        add("")
        dates = [row["date"] for row in calendar["daily"]]
        at = dates.index(peak["date"])
        lines.extend(
            _table(
                [
                    "Date",
                    "Households monitored",
                    "Silent",
                    "Sensor events, all households",
                    "Behavioural alerts",
                ],
                [
                    [
                        row["date"],
                        _n(row["monitored_households"]),
                        _n(row["silent_households"]),
                        _n(row["events"]),
                        _n(row["behavioural_alerts"]),
                    ]
                    for row in calendar["daily"][max(0, at - 3) : at + 4]
                ],
            )
        )
        add("")
        if after is not None:
            add(
                f"The day after it, {after['date']}, holds "
                f"{_n(after['behavioural_alerts'])} of the "
                f"{_n(silent['behavioural_alerts'])} behavioural alerts, "
                f"{_pct(calendar['share_of_alerts_on_the_day_after'])}."
            )
            add("")
    else:
        add("No household has a silent day.")
        add("")

    # ------------------------------------------------------------------
    add("### Post hoc: replays of the baseline")
    add("")
    add(_figure("replays", "Post hoc: the baseline replayed over four sets of values"))
    add("")
    lines.extend(
        f"- **{title}.** {_sentence(replays['what_each_is'][name])}"
        for name, title in _REPLAY_TITLES.items()
    )
    add("")
    kinds = list(replays["stationary_gaussian"]["verdicts"])
    evaluable = replays["recorded"]["evaluable_days"]
    lines.extend(
        _table(
            [
                "Values",
                "Evaluable days",
                "Deviating share of evaluable days",
                "Change verdicts",
                *[_name(kind).capitalize() for kind in kinds],
            ],
            [
                [
                    _REPLAY_TITLES[name],
                    _n(replays[name].get("evaluable_days", evaluable)),
                    _range(replays[name]["deviating_share"], "share"),
                    _range(replays[name]["change_verdicts"]),
                    *[_range(replays[name]["verdicts"].get(kind, 0)) for kind in kinds],
                ]
                for name in _REPLAY_TITLES
            ],
        )
    )
    add("")
    replicates = replays["stationary_gaussian"]["replicates"]
    add(
        f"The shuffled and stationary rows are the mean over {_n(replicates)} "
        "replicates, with the range that holds the central "
        f"{_pct(payload['configuration']['confidence'], 0)} of them. A change "
        "verdict is one of abrupt change, persistent change or gradual drift."
    )
    add("")
    recorded, skipped = replays["recorded"], replays["silent_days_skipped"]
    add(
        "- **The recorded replay gives the run's own verdicts:** "
        f"{_n(recorded['reproduces_the_run'])}, with "
        f"{_n(recorded['feature_days_differing'])} of "
        f"{_n(recorded['feature_days_compared'])} feature-days differing in "
        "verdict or deviation."
    )
    add(
        "- **What leaving silent days out removes.** "
        f"{_n(silent['change_verdicts_on_silent_days'])} of the run's "
        f"{_n(silent['change_verdicts'])} change verdicts are on silent days "
        "themselves, which the changed replay does not judge; the rest of the "
        "difference is on later days whose history no longer holds them. The "
        "two rows count different numbers of evaluable days."
    )
    add(
        "- **Label days, as a description and not the protocol's A2.** With "
        "silent days left out, "
        f"{_pct(skipped['share_of_label_days_deviating'])} of "
        f"{_n(skipped['label_days'])} label days deviate, against "
        f"{_pct(skipped['share_of_other_days_deviating'])} of other evaluable "
        "days. The rule was written after the alerts had been read, and the "
        "figures carry no interval."
    )
    add("")

    # ------------------------------------------------------------------
    add("### Post hoc: reference size")
    add("")
    add(_figure("reference-size", "Post hoc: days at the threshold, by reference size"))
    add("")
    add(
        f"The baseline calls a day deviating at {size['threshold']:g} robust "
        "standard deviations from a median, with the spread taken from the "
        "median absolute deviation. Against a known mean and standard "
        f"deviation a Gaussian value is that far out {_pct(size['nominal'], 2)} "
        f"of the time. The reference is pooled until {size['weekday_min_samples']} "
        "days of the same weekday exist, and is then estimated from those days "
        f"alone; the first verdict needs {size['min_samples']} days. The table "
        "gives how often a Gaussian value reaches the threshold against a "
        "mean and standard deviation of that many others, which is exact, and "
        "against a median and MAD of that many others, from "
        f"{_n(size['draws'])} draws per size, beside the share of feature-days "
        "that reach it here. Those shares are raw: a cell of a few dozen "
        "feature-days is noisy, and none has an interval."
    )
    add("")
    plotted = [
        f for f in size["features"] if size["features"][f]["weekday_aware"]["days"]
    ]
    counts = sorted(
        {int(c) for f in plotted for c in size["features"][f]["weekday_aware_by_size"]}
    )
    lines.extend(
        _table(
            [
                "Same-weekday days in the reference",
                "Gaussian value past a mean and SD",
                "Gaussian value past a median and MAD",
                *[f"`{feature}` past its reference" for feature in plotted],
            ],
            [
                [
                    count,
                    _pct(size["mean_and_sd_estimated"].get(str(count))),
                    _pct(size["gaussian_null"].get(str(count))),
                    *[
                        (
                            "—"
                            if str(count)
                            not in size["features"][f]["weekday_aware_by_size"]
                            else "{} of {}".format(
                                _pct(
                                    size["features"][f]["weekday_aware_by_size"][
                                        str(count)
                                    ]["share_at_threshold"]
                                ),
                                _n(
                                    size["features"][f]["weekday_aware_by_size"][
                                        str(count)
                                    ]["days"]
                                ),
                            )
                        )
                        for f in plotted
                    ],
                ]
                for count in counts
            ],
        )
    )
    add("")
    lines.extend(
        _table(
            [
                "Feature",
                "Feature-days with a verdict",
                "At the threshold",
                "Above",
                "Below",
                "Gaussian value past a median and MAD, same reference sizes",
                "At the threshold, pooled reference",
                "At the threshold, same-weekday reference",
                "References at the scale floor",
            ],
            [
                [
                    f"`{feature}`",
                    _n(entry["days"]),
                    _pct(entry["share_at_threshold"]),
                    _pct(entry["above"]),
                    _pct(entry["below"]),
                    _pct(entry["gaussian_expectation"]),
                    f"{_pct(entry['pooled']['share_at_threshold'])} of "
                    f"{_n(entry['pooled']['days'])}",
                    f"{_pct(entry['weekday_aware']['share_at_threshold'])} of "
                    f"{_n(entry['weekday_aware']['days'])}",
                    _pct(entry["references_at_the_scale_floor"]),
                ]
                for feature, entry in size["features"].items()
            ],
        )
    )
    add("")
    largest = max(size["gaussian_null_mcse"].values(), default=0.0)
    add(
        f"The scale floor is {size['scale_floor']:g} hours; {size['note']}. The "
        "Monte Carlo standard error of a simulated rate is at most "
        f"{100.0 * largest:.2f} points."
    )
    add("")

    # ------------------------------------------------------------------
    add("### Post hoc: direction")
    add("")
    drivers = direction["deviating_day_drivers"]
    keys = sorted({k for group in drivers.values() for k in group if k != "days"})
    add(
        "The feature with the largest deviation on each deviating day, and "
        "which side of its reference the day fell:"
    )
    add("")
    lines.extend(
        _table(
            ["Driver", "Label days", "Other days"],
            [
                [
                    f"`{key.split(': ')[0]}`, {key.split(': ')[1]}",
                    _n(drivers["label_days"].get(key, 0)),
                    _n(drivers["other_days"].get(key, 0)),
                ]
                for key in keys
            ]
            + [
                [
                    "all deviating days",
                    _n(drivers["label_days"]["days"]),
                    _n(drivers["other_days"]["days"]),
                ]
            ],
        )
    )
    add("")
    lines.extend(
        _table(
            [
                "Feature",
                "Median deviation, label days",
                "Median deviation, other days",
                "At the threshold, label days",
                "At the threshold, other days",
            ],
            [
                [
                    f"`{feature}`",
                    _signed(entry["label_days"]["median"]),
                    _signed(entry["other_days"]["median"]),
                    _pct(entry["label_days"]["share_at_threshold"]),
                    _pct(entry["other_days"]["share_at_threshold"]),
                ]
                for feature, entry in direction["feature_deviation"].items()
            ],
        )
    )
    add("")
    add(f"The event-count score is {direction['event_count_z_is']}:")
    add("")
    lines.extend(
        _table(
            ["Evaluable days", "Days", "Mean event-count score", "Median"],
            [
                [
                    _name(name),
                    _n(entry["days"]),
                    _signed(entry["mean"]),
                    _signed(entry["median"]),
                ]
                for name, entry in _ordered(direction["event_count_z"])
            ],
        )
    )
    add("")
    add("Behavioural alerts by feature and the verdict behind them:")
    add("")
    lines.extend(
        _table(
            ["Feature and verdict", "Alerts"],
            [
                [f"`{key.split(': ')[0]}`, {_name(key.split(': ')[1])}", _n(count)]
                for key, count in results["alerts_by_feature_and_verdict"].items()
            ],
        )
    )
    add("")


def render_page(
    payload: Mapping[str, Any], post_hoc: Mapping[str, Any] | None = None
) -> str:
    """The results page, from the run's record and the post hoc record alone."""
    results = payload["results"]
    if results.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    if post_hoc is not None and (
        post_hoc["results"].get("result_schema") != POST_HOC_SCHEMA
    ):
        raise ValueError(f"not a {POST_HOC_SCHEMA} record")
    lines = [
        "# TIHM: alert-burden results",
        "",
        f"The frozen [alert-burden protocol]({PROTOCOL_PAGE}) run as declared on "
        f"the {results['contract']['households']} homes of the TIHM dataset. "
        f"This page is generated entirely from the record, `{RECORD_FILE}`, "
        + (
            f"and its last part from a second record, `{POST_HOC_FILE}`, "
            if post_hoc is not None
            else ""
        )
        + "by `sensor_modeling.datasets.tihm_summary.render_page`, and a test "
        "checks that the committed page is exactly that rendering.",
        "",
    ]
    _protocol_part(payload, lines)
    if post_hoc is not None:
        _post_hoc_part(post_hoc, lines)
    return "\n".join(lines)
