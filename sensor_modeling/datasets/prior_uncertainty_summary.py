"""The pages of Phase 4 for the time prior, from the frozen protocol and the record.

:func:`render_protocol` writes the protocol page from the frozen protocol file.
:func:`render_page` writes the results page from the record with the Phase 4
page's renderer, which this record's layout shares, and its own title, checks
and limits. Every number on either page comes from those files.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .prior_uncertainty_experiment import PROTOCOL_FILE, RECORD_FILE
from .sleep_gap_summary import _flatten, _section, _sentence, _table
from .uncertainty_experiment import SIGNALS
from .uncertainty_summary import PageText
from .uncertainty_summary import render_page as _render

PROTOCOL_PAGE = "PHASE4_TIME_PRIOR_PROTOCOL.md"
RESULTS_PAGE = "PHASE4_TIME_PRIOR_RESULTS.md"
FIGURE_PREFIX = "phase4-time-prior"
TITLE = "Phase 4 for the time prior: uncertainty diagnostics against confidence"


def render_protocol(payload: Mapping[str, Any]) -> str:
    """The protocol page, from the frozen protocol file's content."""
    lines = [
        f"# {TITLE}: the protocol",
        "",
        f"Generated entirely from the frozen file `{PROTOCOL_FILE}` by "
        "`sensor_modeling.datasets.prior_uncertainty_summary.render_protocol`.",
        "",
        f"**Status: {payload['status']}.**",
        "",
        f"- **Protocol digest.** `{payload['protocol_sha256']}`.",
        f"- **The question.** {_sentence(payload['question'])}?".replace(".?", "?"),
        "- **This page reports no result.**",
        "",
        "## What had been seen",
        "",
        *[f"- {_sentence(item)}" for item in payload["inspected_before"]],
        "",
        "## The base protocols",
        "",
    ]
    for title, key in (("Phase 4", "base_protocol"), ("Time prior", "time_prior")):
        base = payload[key]
        lines.append(
            f"- **{title}.** `{base['file']}`, SHA-256 `{base['sha256']}`, protocol "
            f"digest `{base['protocol_sha256']}`. Used: {base['used']}."
        )
    predictions = payload["predictions"]
    ensemble = payload["ensemble"]
    lines += [
        "",
        "## What changes",
        "",
        f"- **The predictions.** `{predictions['model']}`: "
        f"{_sentence(predictions['description'])}",
        "- **The ensemble.** "
        + ", ".join(f"`{s['name']}`" for s in ensemble["specifications"])
        + f": {_sentence(ensemble['reason'])}",
        f"- **The transition.** {_sentence(ensemble['transition'])}",
        f"- **The evidence groups.** {_sentence(payload['evidence_groups']['groups'])}",
        "",
        "## Signals",
        "",
        *_table(
            ["Signal", "Direction", "Missing values", "Definition"],
            [
                [
                    f"`{name}`",
                    f"`{payload['signals'][name]['direction']}`",
                    f"`{payload['signals'][name]['missing']}`",
                    payload["signals"][name]["definition"],
                ]
                for name in SIGNALS
            ],
        ),
        "",
        f"Every signal is compared with `{payload['comparator']}`; the structural "
        "diagnostics are "
        + ", ".join(f"`{s}`" for s in payload["structural_signals"])
        + ".",
        "",
    ]
    lines += _section("Evaluation", payload["evaluation"])
    lines += _section("Estimands", payload["estimands"])
    differences = payload["minimal_differences"]
    lines += [
        *_table(
            ["Quantity", "Minimal difference"],
            [[f"`{k}`", f"{differences[k]:g}"] for k in sorted(differences)],
        ),
        "",
        f"- **Verdicts.** {_sentence(payload['verdicts'])}",
        f"- **Difficult minority states.** "
        f"{_sentence(payload['difficult_minority_states']['rule'])}",
        f"- **Guard.** {_sentence(payload['rejection_bias']['guard'])}",
        "",
    ]
    lines += _section("Decision", payload["decision"])
    lines += _section("Bootstrap", payload["bootstrap"])
    check = payload["the_check"]
    rule = payload["rule_signal"]
    lines += [
        "## The check",
        "",
        *[f"- {_sentence(item)}" for item in check["what"]],
        f"- **Tolerance.** {check['tolerance']:g}.",
        f"- **Otherwise.** {_sentence(check['otherwise'])}",
        "",
        "## The abstention rule's signal",
        "",
        f"- **The rule.** {_sentence(rule['rule'])}",
        f"- **Use.** {_sentence(rule['use'])}",
        f"- **Thresholds here.** {_sentence(payload['thresholds'])}",
        "",
    ]
    lines += _section("One run", payload["one_run"])
    lines += [
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
        f"**Tuning.** {_sentence(_flatten(payload['tuning']))}",
        "",
    ]
    return "\n".join(lines)


def _checks(payload: Mapping[str, Any]) -> list[str]:
    check = payload["results"]["check"]
    tolerance = payload["configuration"]["the_check"]["tolerance"]
    scores, fits = check["scores"], check["fits"]

    def exactly(identical: bool) -> str:
        return (
            "bit for bit"
            if identical
            else "though not bit for bit: the record was made on another platform"
        )

    return [
        f"- **Reproduction.** The fold fits of {', '.join(fits['folds'])} equal "
        f"the fitted-rates record's to within {tolerance:g}, "
        f"{exactly(fits['identical_digests'])}, and the ensemble's population is "
        f"each fold's fitted channels exactly. `{scores['model']}` gives back the "
        f"combined-prior record's scores for all {scores['households']} "
        f"households to within {tolerance:g}, {exactly(scores['identical'])}.",
    ]


def _rule_signal(payload: Mapping[str, Any]) -> list[str]:
    chosen = payload["results"]["rule_signal"]
    rule = payload["configuration"]["rule_signal"]
    return [
        "### The abstention rule's signal",
        "",
        f"`{chosen['signal']}`: {chosen['reason']}. The rule, declared before "
        f"scoring: {rule['rule']}.",
        "",
    ]


def _not_shown(payload: Mapping[str, Any]) -> list[str]:
    return [
        f"- {_sentence(item)}"
        for item in payload["configuration"]["what_this_cannot_show"]
    ]


TEXT = PageText(
    title=TITLE,
    intro="Phase 4 compared structural uncertainty diagnostics with posterior "
    "confidence on the hurdle recursion without the time prior. The reference "
    "has since become the fitted channels with the time prior. This "
    "pre-specified experiment repeats the comparison on its predictions, with the "
    "same signals, ensemble, estimands and decision rule, and picks, by a rule "
    "declared before scoring, the signal an abstention rule will use.",
    record_file=RECORD_FILE,
    protocol_file=PROTOCOL_FILE,
    module="prior_uncertainty_summary",
    figure_prefix=FIGURE_PREFIX,
    held_out="The development homes have been inspected in earlier work, by "
    "Phase 4 among others. The 43 held-out CASAS homes are not scored.",
    checks=_checks,
    after_answer=_rule_signal,
    not_shown=_not_shown,
)


def render_page(payload: Mapping[str, Any]) -> str:
    """The results page, from the record."""
    return _render(payload, TEXT)


__all__ = [
    "FIGURE_PREFIX",
    "PROTOCOL_FILE",
    "PROTOCOL_PAGE",
    "RECORD_FILE",
    "RESULTS_PAGE",
    "TEXT",
    "render_page",
    "render_protocol",
]
