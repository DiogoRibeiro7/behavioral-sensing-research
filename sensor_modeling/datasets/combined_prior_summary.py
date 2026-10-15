"""The combined-prior pages, generated from the frozen protocol and the record.

:func:`render_protocol` writes the protocol page from the frozen protocol file,
and :func:`render_page` the results page from the record. Every number on
either page comes from those files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .sleep_gap_summary import _flatten, _interval, _n, _section, _sentence, _table

PROTOCOL_FILE = "artifacts/phase3/combined_prior_protocol.json"
RECORD_FILE = "artifacts/phase3/phase3-combined-prior.json"
PROTOCOL_PAGE = "PHASE3_COMBINED_PRIOR_PROTOCOL.md"
RESULTS_PAGE = "PHASE3_COMBINED_PRIOR_RESULTS.md"
METRIC_TITLES = {
    "balanced_accuracy": "Balanced accuracy",
    "log_loss": "Log loss",
    "brier": "Brier score",
    "calibration_error": "Calibration error",
}


def render_protocol(payload: Mapping[str, Any]) -> str:
    """The protocol page, from the frozen protocol file's content."""
    lines = [
        "# Phase 3: the time prior with the fitted channels, in the recursion: "
        "the protocol",
        "",
        "Generated entirely from the frozen file "
        f"`{PROTOCOL_FILE}` by "
        "`sensor_modeling.datasets.combined_prior_summary.render_protocol`.",
        "",
        f"**Status: {payload['status']}.**",
        "",
        f"- **Protocol digest.** `{payload['protocol_sha256']}`.",
        f"- **The question.** {_sentence(payload['question'])}",
        "- **This page reports no result.**",
        "",
        "## What had been seen",
        "",
        *[f"- {_sentence(item)}" for item in payload["inspected_before"]],
        "",
    ]
    base = payload["base_protocol"]
    lines += [
        "## The base protocol",
        "",
        f"- **File.** `{base['file']}`, SHA-256 `{base['sha256']}`, protocol "
        f"digest `{base['protocol_sha256']}`.",
        f"- **Used.** {_sentence(base['used'])}",
        "",
    ]
    lines += _section("The recursion", payload["recursion"])
    lines += [
        "## Models",
        "",
        *_table(
            ["Model", "Channels", "Time prior"],
            [
                [f"`{name}`", spec["channels"], _flatten(spec["time_prior"])]
                for name, spec in payload["models"].items()
            ],
        ),
        "",
        "## Estimands",
        "",
        *_table(
            ["", "Role", "Question", "Model", "Reference", "Rule"],
            [
                [
                    e["key"],
                    e["role"],
                    _sentence(e["question"]),
                    f"`{e['model']}`",
                    f"`{e['reference']}`",
                    e["rule"],
                ]
                for e in payload["estimands"]
            ],
        ),
        "",
    ]
    lines += _section("Criteria", payload["criteria"])
    lines += _section("Per-state recall", payload["per_state_recall"])
    joint = payload["interaction"]
    differences = payload["minimal_differences"]
    lines += [
        "## Interaction",
        "",
        f"- **What.** {_sentence(joint['what'])}",
        f"- **Metrics.** {_titles(joint['metrics']).capitalize()}.",
        f"- **Reading.** {_sentence(joint['reading'])}",
        "",
        "## Metrics and minimal differences",
        "",
        *_table(
            ["Metric", "Minimal difference"],
            [[METRIC_TITLES[m], f"{differences[m]:g}"] for m in payload["metrics"]]
            + [["Recall of one state", f"{differences['recall']:g}"]],
        ),
        "",
    ]
    lines += _section("Bootstrap", payload["bootstrap"])
    check = payload["the_check"]
    lines += [
        "## The check",
        "",
        *[f"- {_sentence(item)}" for item in check["what"]],
        f"- **Tolerance.** {check['tolerance']:g}.",
        f"- **Otherwise.** {_sentence(check['otherwise'])}",
        "",
        "## Pinned records",
        "",
        *_table(
            ["File", "SHA-256"],
            [[f"`{k}`", f"`{v}`"] for k, v in payload["pinned_records"].items()],
        ),
        "",
        "## What this cannot show",
        "",
        *[f"- {_sentence(item)}" for item in payload["what_this_cannot_show"]],
        "",
    ]
    return "\n".join(lines)


def _titles(metrics: Sequence[str]) -> str:
    named = [METRIC_TITLES[m].lower() for m in metrics]
    return ", ".join(named[:-1]) + " and " + named[-1]


def _level(entry: Mapping[str, Any]) -> str:
    return _interval(
        {"estimate": entry["mean"], "interval": entry.get("mean_interval")}, 3
    )


def _difference(comparison: Mapping[str, Any]) -> str:
    return _interval(comparison["mean"], 3, signed=True)


def render_page(payload: Mapping[str, Any]) -> str:
    """The results page, from the record."""
    results = payload["results"]
    configuration = payload["configuration"]
    environment = payload.get("environment", {})
    metrics = list(configuration["metrics"])
    check = results["check"]
    lines = [
        "# Phase 3: the time prior with the fitted channels, in the recursion: "
        "results",
        "",
        "The frozen [protocol](PHASE3_COMBINED_PRIOR_PROTOCOL.md), run as declared. "
        f"Generated entirely from `{RECORD_FILE}` by "
        "`sensor_modeling.datasets.combined_prior_summary.render_page`.",
        "",
        f"**Status: {results['status']}.**",
        "",
        f"- **Protocol digest.** `{configuration['protocol_sha256']}`.",
        f"- **Run.** Commit `{str(environment.get('git_commit', ''))[:7]}`, recorded "
        f"{payload['recorded_at']}, on {environment.get('platform', 'an unrecorded platform')}.",
        f"- **Conclusion of K1.** {results['conclusion']}.",
        "",
        "## The check",
        "",
        f"The fold fits of {', '.join(check['fits']['folds'])} equal the "
        "fitted-rates record's, and the recursions without the time prior give "
        f"back its scores for all {check['scores']['households']} households, to "
        f"within {configuration['the_check']['tolerance']:g}. "
        + (
            "They are bit-identical."
            if check["fits"]["identical_digests"] and check["scores"]["identical"]
            else "They are not bit-identical, which the tolerance allows for: the "
            "record was made on another platform."
        ),
        "",
        "## Each model in the recursion",
        "",
        "The mean over households, with its interval.",
        "",
        *_table(
            ["Model", *[METRIC_TITLES[m] for m in metrics]],
            [
                [f"`{cell['model']}`", *[_level(cell["metrics"][m]) for m in metrics]]
                for cell in results["cells"]
            ],
        ),
        "",
        "## The estimands",
        "",
        "Mean paired differences, oriented so that a positive value favours the "
        "model, with their intervals and verdicts.",
        "",
        *_table(
            [
                "",
                "Model against reference",
                *[METRIC_TITLES[m] for m in metrics],
                "Conclusion",
            ],
            [
                [
                    e["key"],
                    f"`{e['model']}` against `{e['reference']}`",
                    *[
                        f"{_difference(e['comparisons'][m])}; {e['verdicts'][m]}"
                        for m in metrics
                    ],
                    e["conclusion"],
                ]
                for e in results["estimands"]
            ],
        ),
        "",
    ]
    for e in results["estimands"]:
        recall = e.get("per_state_recall")
        if not recall:
            continue
        lines += [
            f"**{e['key']}, recall by state.**",
            "",
            *_table(
                ["State", "Difference", "Verdict", "Households"],
                [
                    [
                        f"`{state}`",
                        _difference(entry["comparison"]),
                        entry["verdict"],
                        _n(entry["comparison"]["n"]),
                    ]
                    for state, entry in recall.items()
                ],
            ),
            "",
        ]
    lines += [
        "## Interaction",
        "",
        "Per household, the prior's gain with the fitted channels minus its gain "
        "with the declared channels; positive means the two combine better than "
        "additively. Described, not judged.",
        "",
        *_table(
            ["Metric", "K1 minus K2"],
            [
                [METRIC_TITLES[m], _difference(results["interaction"][m])]
                for m in configuration["interaction"]["metrics"]
            ],
        ),
        "",
        "## Notes",
        "",
        *[f"- {_sentence(note)}" for note in payload.get("notes", [])],
        "",
    ]
    return "\n".join(lines)


__all__ = ["render_page", "render_protocol"]
