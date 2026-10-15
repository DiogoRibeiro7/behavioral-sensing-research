"""The ContextAct transfer pages, generated from the frozen protocol and the record.

:func:`render_protocol` writes the protocol page from the frozen protocol file,
and :func:`render_page` the results page from the record. Every number on
either page comes from those files.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .combined_prior_experiment import INTERACTION_METRICS
from .combined_prior_summary import METRIC_TITLES, _difference, _level
from .sleep_gap_summary import _flatten, _n, _section, _sentence, _table
from .transfer_experiment import PROTOCOL_FILE, RECORD_FILE

PROTOCOL_PAGE = "CONTEXTACT_TRANSFER_PROTOCOL.md"
RESULTS_PAGE = "CONTEXTACT_TRANSFER_RESULTS.md"
TITLE = "The Phase 3 formulation on ContextAct@A4H"


def _bullets(items: list[str]) -> list[str]:
    return [f"- {_sentence(item)}" for item in items]


def _mean(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _instrumentation(declared: Mapping[str, Any]) -> list[str]:
    """The inputs report the protocol declares, and what it found."""
    if not declared:
        return []
    report = declared["report"]
    lines = [
        "## What the inputs report found",
        "",
        f"From `{report['file']}`, SHA-256 `{report['sha256']}`.",
        "",
    ]
    if "eligible_periods" in declared:
        lines += [
            "- **Eligible periods.** "
            f"{', '.join(declared['eligible_periods']) or 'none'}.",
            *[
                f"- **Channels in {period}.** "
                + (", ".join(f"`{c}`" for c in channels) or "none")
                + "."
                for period, channels in declared["instrumented"].items()
            ],
            "- **Days with enough labelled windows.** "
            f"{_n(declared['scored_days'])}.",
            "- **Periods in which each state occurs.** "
            + "; ".join(
                f"`{state}` {count}"
                for state, count in declared["periods_with_each_state"].items()
            )
            + ".",
            "",
        ]
    return lines


def render_protocol(payload: Mapping[str, Any]) -> str:
    """The protocol page, from the frozen protocol file's content."""
    training = payload["training"]
    reproduction = training["reproduction"]
    dataset = payload["dataset"]
    models = payload["models"]
    baselines = models["baselines"]
    differences = payload["minimal_differences"]
    names = ", ".join("`" + model["name"] + "`" for model in baselines["models"])
    lines = [
        f"# {TITLE}: the protocol",
        "",
        f"Generated entirely from the frozen file `{PROTOCOL_FILE}` by "
        "`sensor_modeling.datasets.transfer_summary.render_protocol`.",
        "",
        f"**Status: {payload['status']}.**",
        "",
        f"- **Protocol digest.** `{payload['protocol_sha256']}`.",
        f"- **The question.** {_sentence(payload['question'])}",
        "- **This page reports no result.**",
        "",
        "## What had been seen",
        "",
        *_bullets(payload["inspected_before"]),
        "",
        "## The base protocol",
        "",
        f"- **File.** `{payload['base_protocol']['file']}`, protocol digest "
        f"`{payload['base_protocol']['protocol_sha256']}`.",
        f"- **Used.** {_sentence(payload['base_protocol']['used'])}",
        "",
        "## Training, and its reproduction",
        "",
        f"- **Homes.** {len(training['homes'])} development homes: "
        f"{', '.join(training['homes'])}. {_sentence(training['fitted'])}",
        f"- **Channels.** {_sentence(reproduction['channels'])} Digest "
        f"`{reproduction['channels_sha256']}`.",
        f"- **Time prior.** {_sentence(reproduction['time_prior'])}",
        f"- **The opt-in.** {_sentence(reproduction['pipeline'])}",
        "",
        "## The dataset",
        "",
        f"- **Periods.** {', '.join(dataset['periods'])}.",
        f"- **Mapping.** `{dataset['mapping']['file']}`, mapping digest "
        f"`{dataset['mapping']['mapping_sha256']}`.",
        f"- **Reading.** {_sentence(dataset['reading'])}",
        f"- **Adaptation.** {_sentence(dataset['adaptation'])}",
        "",
        *_table(
            ["File", "SHA-256"],
            [[f"`{k}`", f"`{v}`"] for k, v in dataset["files"].items()],
        ),
        "",
        *_instrumentation(payload["instrumentation"]),
        "## Models",
        "",
        "In the recursion, as in the combined-prior experiment:",
        "",
        *_table(
            ["Model", "Channels", "Time prior"],
            [
                [f"`{name}`", spec["channels"], _flatten(spec["time_prior"])]
                for name, spec in models["recursion"].items()
            ],
        ),
        "",
        f"Baselines at `{baselines['information_set']}`, fitted by the matched "
        f"runner: {names}. {_sentence(baselines['why'])}",
        "",
        f"**Missing channels.** {_sentence(models['missing_channels'])}",
        "",
    ]
    lines += _section("Units", payload["units"])
    lines += [
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
    lines += [
        "## Descriptions",
        "",
        f"- **Recall by state.** For "
        f"{', '.join(payload['per_state_recall']['estimands'])}: "
        f"{payload['per_state_recall']['verdict']}.",
        f"- **Interaction.** {_sentence(payload['interaction'])}",
        f"- **By period.** {_sentence(payload['by_period'])}",
        "",
        "## Metrics and minimal differences",
        "",
        *_table(
            ["Metric", "Minimal difference"],
            [[METRIC_TITLES[m], f"{differences[m]:g}"] for m in payload["metrics"]]
            + [
                ["Recall of one state", f"{differences['recall']:g}"],
                ["Balanced accuracy against chance", f"{differences['chance']:g}"],
            ],
        ),
        "",
    ]
    lines += _section("Bootstrap", payload["bootstrap"])
    lines += [
        "## One shot",
        "",
        _sentence(payload["one_shot"]),
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
        *_bullets(payload["what_this_cannot_show"]),
        "",
    ]
    return "\n".join(lines)


def render_page(payload: Mapping[str, Any]) -> str:
    """The results page, from the record."""
    results = payload["results"]
    configuration = payload["configuration"]
    environment = payload.get("environment", {})
    metrics = list(configuration["metrics"])
    estimands = results["estimands"]
    check = results["check"]
    lines = [
        f"# {TITLE}: results",
        "",
        "The frozen [protocol](CONTEXTACT_TRANSFER_PROTOCOL.md), run once as "
        f"declared. Generated entirely from `{RECORD_FILE}` by "
        "`sensor_modeling.datasets.transfer_summary.render_page`.",
        "",
        f"**Status: {results['status']}.**",
        "",
        f"- **Protocol digest.** `{configuration['protocol_sha256']}`.",
        f"- **Run.** Commit `{str(environment.get('git_commit', ''))[:7]}`, recorded "
        f"{payload['recorded_at']}, on "
        f"{environment.get('platform', 'an unrecorded platform')}.",
        f"- **Conclusion of K1.** {results['conclusion']}.",
        f"- **Days.** {_n(check['days'])} scored: "
        + ", ".join(f"{_n(n)} in {p}" for p, n in check["periods"].items())
        + f". Unscored, with too few labelled windows: "
        f"{_n(len(check['unscored_days']))}. Excluded periods: "
        f"{', '.join(check['excluded_periods']) or 'none'}. On every scored day "
        "every model scored the same windows in each true state.",
        f"- **Reproduction.** The refitted channels have the held-out record's "
        f"digest, `{results['training']['channels_sha256']}`; the refitted time "
        f"prior was {results['training']['prior_max_difference']:.1e} from the "
        "shipped one, which was used. The online pipeline's opt-in was at most "
        + ", ".join(
            f"{value:.1e} from the recursion in {period}"
            for period, value in check["pipeline_max_difference"].items()
        )
        + ".",
        "",
        "## Each model on ContextAct",
        "",
        "The mean over days, with its interval.",
        "",
        *_table(
            ["Model", "Information", *[METRIC_TITLES[m] for m in metrics]],
            [
                [
                    f"`{cell['model']}`",
                    cell["information_set"],
                    *[_level(cell["metrics"][m]) for m in metrics],
                ]
                for cell in results["cells"]
            ],
        ),
        "",
        "## The estimands",
        "",
        "Mean paired differences over days, oriented so that a positive value "
        "favours the model, with their intervals and verdicts.",
        "",
        *_table(
            [
                "",
                "Model against reference",
                "Metric",
                "Difference; verdict",
                "Conclusion",
            ],
            [
                [
                    e["key"],
                    f"`{e['model']}` against `{e['reference']}`",
                    METRIC_TITLES[metric],
                    f"{_difference(e['comparisons'][metric])}; {e['verdicts'][metric]}",
                    e["conclusion"] if metric == "balanced_accuracy" else "",
                ]
                for e in estimands
                for metric in e["comparisons"]
            ]
            + [
                [
                    e["key"],
                    f"`{e['model']}` against `{e['reference']}`",
                    "—",
                    "—",
                    e["conclusion"],
                ]
                for e in estimands
                if not e["comparisons"]
            ],
        ),
        "",
        *[
            f"- **{e['key']}.** {_sentence(e['note'])}"
            for e in estimands
            if e.get("note")
        ],
        *[
            f"- **{e['key']}.** Days left out with fewer than two labelled states: "
            f"{_n(e['days_left_out'])}."
            for e in estimands
            if "days_left_out" in e
        ],
        "",
    ]
    for e in estimands:
        recall = e.get("per_state_recall")
        if not recall:
            continue
        lines += [
            f"**{e['key']}, recall by state.**",
            "",
            *_table(
                ["State", "Difference", "Verdict", "Days"],
                [
                    [
                        f"`{state}`",
                        (
                            _difference(entry["comparison"])
                            if entry["comparison"]
                            else "—"
                        ),
                        entry["verdict"],
                        _n(entry["comparison"]["n"]) if entry["comparison"] else "0",
                    ]
                    for state, entry in recall.items()
                ],
            ),
            "",
        ]
    periods = results["by_period"]
    recursions = list(next(iter(periods.values()))["balanced_accuracy"])
    lines += [
        "## Interaction",
        "",
        "Per day, the prior's gain with the fitted channels minus its gain with "
        "the declared channels. Described, not judged.",
        "",
        *_table(
            ["Metric", "K1 minus K2"],
            [
                [METRIC_TITLES[m], _difference(results["interaction"][m])]
                for m in INTERACTION_METRICS
            ],
        ),
        "",
        "## By period",
        "",
        "Mean daily balanced accuracy of each recursion within each period, and "
        "the mean of K1. Described, not judged.",
        "",
        *_table(
            ["Period", "Days", *[f"`{m}`" for m in recursions], "K1"],
            [
                [
                    name,
                    _n(entry["days"]),
                    *[_mean(entry["balanced_accuracy"][m]) for m in recursions],
                    _mean(entry["k1"], signed=True),
                ]
                for name, entry in periods.items()
            ],
        ),
        "",
        "## Notes",
        "",
        *_bullets(payload.get("notes", [])),
        "",
    ]
    return "\n".join(lines)


__all__ = ["PROTOCOL_PAGE", "RESULTS_PAGE", "render_page", "render_protocol"]
