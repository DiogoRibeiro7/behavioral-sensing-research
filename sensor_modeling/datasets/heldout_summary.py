"""The held-out confirmation pages, generated from the frozen protocol and the record.

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

PROTOCOL_FILE = "artifacts/heldout/heldout_protocol.json"
RECORD_FILE = "artifacts/heldout/casas-heldout-confirmation.json"
PROTOCOL_PAGE = "HELDOUT_CONFIRMATION_PROTOCOL.md"
RESULTS_PAGE = "HELDOUT_CONFIRMATION_RESULTS.md"
TITLE = "The Phase 3 formulation on held-out CASAS homes"


def _bullets(items: list[str]) -> list[str]:
    return [f"- {_sentence(item)}" for item in items]


def _instrumentation(declared: Mapping[str, Any]) -> list[str]:
    """The inputs report the protocol declares, as tables."""
    if not declared:
        return []
    lacking = declared["held_out_homes_lacking_one"]
    every = declared["channels_every_training_home_instruments"]
    return [
        "## What each held-out home instruments",
        "",
        f"From the inputs report `{declared['report']['file']}`, SHA-256 "
        f"`{declared['report']['sha256']}`. Every training home instruments "
        f"{', '.join(f'`{c}`' for c in every)}.",
        "",
        *_table(
            ["Home", "Channels", "Lacks", "Windows", "Labelled", "States"],
            [
                [
                    home,
                    _n(len(entry["instrumented"])),
                    ", ".join(f"`{c}`" for c in lacking.get(home, [])) or "—",
                    _n(entry["windows"]),
                    _n(entry["labelled"]),
                    _n(len(entry["states"])),
                ]
                for home, entry in declared["test"].items()
            ],
        ),
        "",
        "Held-out homes in which each state occurs: "
        + "; ".join(
            f"`{state}` {count}"
            for state, count in declared["held_out_homes_with_each_state"].items()
        )
        + ".",
        "",
    ]


def render_protocol(payload: Mapping[str, Any]) -> str:
    """The protocol page, from the frozen protocol file's content."""
    households = payload["households"]
    test = households["test"]
    models = payload["models"]
    baselines = models["baselines"]
    differences = payload["minimal_differences"]
    names = ", ".join("`" + model["name"] + "`" for model in baselines["models"])
    lines = [
        f"# {TITLE}: the protocol",
        "",
        f"Generated entirely from the frozen file `{PROTOCOL_FILE}` by "
        "`sensor_modeling.datasets.heldout_summary.render_protocol`.",
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
        "## Households",
        "",
        f"- **Training.** {len(households['training']['homes'])} development homes: "
        f"{', '.join(households['training']['homes'])}. "
        f"{_sentence(households['training']['fitted'])}",
        f"- **Held out.** {len(test['homes'])} homes of the frozen cohort, "
        f"`{test['manifest']}`. {_sentence(test['reading'])}",
        "",
        *_table(
            ["Family", "Homes", "Identifiers"],
            [
                [
                    f"`{name}`",
                    _n(count),
                    ", ".join(
                        h["id"] for h in test["homes"] if h["id"].startswith(name)
                    ),
                ]
                for name, count in sorted(test["families"].items())
            ],
        ),
        "",
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
        f"**Windows.** {_sentence(payload['windows'])}",
        "",
        f"**Unscored homes.** {_sentence(payload['unscored'])}",
        "",
        *_instrumentation(payload["instrumentation"]),
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
        f"- **By family.** {_sentence(payload['by_family'])}",
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
    lines = [
        f"# {TITLE}: results",
        "",
        "The frozen [protocol](HELDOUT_CONFIRMATION_PROTOCOL.md), run once as "
        f"declared. Generated entirely from `{RECORD_FILE}` by "
        "`sensor_modeling.datasets.heldout_summary.render_page`.",
        "",
        f"**Status: {results['status']}.**",
        "",
        f"- **Protocol digest.** `{configuration['protocol_sha256']}`.",
        f"- **Run.** Commit `{str(environment.get('git_commit', ''))[:7]}`, recorded "
        f"{payload['recorded_at']}, on "
        f"{environment.get('platform', 'an unrecorded platform')}.",
        f"- **Conclusion of K1.** {results['conclusion']}.",
        f"- **Households.** {len(results['training']['homes'])} training, "
        f"{results['check']['households']} held-out homes scored, unscored: "
        f"{', '.join(results['check']['unscored']) or 'none'}. In every scored home "
        "every model scored the same windows in each true state.",
        "",
        "## Each model on the held-out homes",
        "",
        "The mean over households, with its interval.",
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
                for e in estimands
            ],
        ),
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
    families = results["by_family"]
    recursions = list(next(iter(families.values()))["balanced_accuracy"])
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
                for m in INTERACTION_METRICS
            ],
        ),
        "",
        "## By family",
        "",
        "Mean household balanced accuracy of each recursion within each CASAS "
        "family, and the mean of K1. Described, not judged: some families have "
        "two or three homes.",
        "",
        *_table(
            ["Family", "Homes", *[f"`{m}`" for m in recursions], "K1"],
            [
                [
                    f"`{name}`",
                    _n(entry["households"]),
                    *[_mean(entry["balanced_accuracy"][m]) for m in recursions],
                    _mean(entry["k1"], signed=True),
                ]
                for name, entry in families.items()
            ],
        ),
        "",
        "## Notes",
        "",
        *_bullets(payload.get("notes", [])),
        "",
    ]
    return "\n".join(lines)


def _mean(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


__all__ = ["render_page", "render_protocol"]
