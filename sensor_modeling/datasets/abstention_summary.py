"""The abstention-rule pages, from the frozen protocol and the record."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .abstention_experiment import PROTOCOL_FILE, RECORD_FILE
from .sleep_gap_summary import _flatten, _section, _sentence, _table
from .threshold_calibration_summary import _fixed

PROTOCOL_PAGE = "PHASE4_ABSTENTION_PROTOCOL.md"
RESULTS_PAGE = "PHASE4_ABSTENTION_RESULTS.md"
TITLE = "Phase 4: an abstention rule for the fitted channels with the time prior"


def render_protocol(payload: Mapping[str, Any]) -> str:
    """The protocol page, from the frozen protocol file's content."""
    lines = [
        f"# {TITLE}: the protocol",
        "",
        f"Generated from `{PROTOCOL_FILE}` by "
        "`sensor_modeling.datasets.abstention_summary.render_protocol`. "
        "This page reports no result.",
        "",
        f"**Status: {payload['status']}.** Digest `{payload['protocol_sha256']}`.",
        "",
        f"**The question.** {_sentence(payload['question'])}",
        "",
        "## What had been seen",
        "",
        *[f"- {_sentence(item)}" for item in payload["inspected_before"]],
        "",
    ]
    lines += _section("Step 1", payload["step1"])
    lines += _section("Selection", payload["selection"])
    lines += _section(
        "Fitting and scoring",
        {
            "cross fitting": payload["cross_fitting"],
            "final rule": payload["final_rule"],
        },
    )
    lines += _section("Estimands", payload["estimands"])
    lines += _section(
        "Reading", {"verdicts": payload["verdicts"], "decision": payload["decision"]}
    )
    lines += _section(
        "Edge cases and adoption",
        {"edge cases": payload["edge_cases"], "adoption": payload["adoption"]},
    )
    lines += _section("Parameters", payload["parameters"])
    lines += _section("Bootstrap", payload["bootstrap"])
    check = payload["the_check"]
    lines += [
        "## The check",
        "",
        *[f"- {_sentence(item)}" for item in check["what"]],
        f"- **Tolerance.** {check['tolerance']:g}; otherwise " f"{check['otherwise']}.",
        "",
        *_section("One run", payload["one_run"]),
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


def _n(value: Any, digits: int = 3) -> str:
    return "–" if value is None else _fixed(value, digits)


def _difference(entry: Mapping[str, Any]) -> str:
    mean = entry["comparison"]["mean"]
    low, high = mean["interval"]["low"], mean["interval"]["high"]
    return (
        f"{_fixed(mean['estimate'], 3, sign=True)} [{_fixed(low, 3, sign=True)}, "
        f"{_fixed(high, 3, sign=True)}]; {entry['verdict']}"
    )


def _selection_rows(name: str, selection: Mapping[str, Any]) -> list[list[str]]:
    return [
        [
            name,
            _fixed(step["target"], 2),
            _n(step["coverage"]),
            *[_n(r, 2) for r in step["ratios"].values()],
            "holds" if step["guard_holds"] else "fails",
        ]
        for step in selection["tried"]
    ]


def render_page(payload: Mapping[str, Any]) -> str:
    """The results page, from the record."""
    results = payload["results"]
    config = payload["configuration"]
    environment = payload.get("environment", {})
    rule = results["rule"]
    guard = results["guard_held_out"]
    guarded = list(guard["ratios"])
    rows = [
        row
        for fold, entry in results["folds"].items()
        for row in _selection_rows(fold, entry["selection"])
    ] + _selection_rows("all 20 homes", rule)
    frozen = (
        f"abstain when `{rule['signal']}` exceeds {_n(rule['threshold'], 4)}, which "
        f"keeps {_n(rule['tried'][-1]['coverage'])} of the windows it was chosen on"
        if rule["abstains"]
        else "never abstain: the guard holds at no coverage below 1"
    )
    estimands = results["estimands"]
    lines = [
        f"# {TITLE}: results",
        "",
        "The frozen [protocol](PHASE4_ABSTENTION_PROTOCOL.md), run once. Generated "
        f"from `{RECORD_FILE}` by "
        "`sensor_modeling.datasets.abstention_summary.render_page`.",
        "",
        f"**Status: {results['status']}.** Protocol `{config['protocol_sha256']}`; "
        f"commit `{str(environment.get('git_commit', ''))[:7]}`.",
        "",
        f"- **Decision.** {_sentence(results['decision'])}",
        f"- **The rule frozen.** {_sentence(frozen)}",
        "",
        "## Selection",
        "",
        "Each coverage tried from the target up, with each guarded state's kept "
        "share over the coverage kept, pooled over the homes the threshold was "
        "chosen on:",
        "",
        *_table(
            [
                "Chosen on",
                "Target",
                "Coverage kept",
                *[f"`{s}`" for s in guarded],
                "Guard",
            ],
            rows,
        ),
        "",
        "## On the held-out homes",
        "",
        "Each fold's threshold applied to its held-out homes. Differences are per "
        "home, with household intervals; positive favours the rule.",
        "",
        *_table(
            ["Estimand", "Difference"],
            [
                ["A1, error: full coverage minus kept", _difference(estimands["A1"])],
                [
                    "A2, balanced accuracy: kept minus full coverage",
                    _difference(estimands["A2"]),
                ],
            ],
        ),
        "",
        f"Pooled over the held-out homes the rule keeps {_n(guard['coverage'])} of "
        "the windows, and the guarded states keep "
        + ", ".join(f"`{s}` {_n(r, 2)}" for s, r in guard["ratios"].items())
        + " of their proportional share: the guard "
        + ("holds." if guard["holds"] else "fails."),
        "",
        *_table(
            ["Home", "Coverage", "Error kept", "Error full", "BA kept", "BA full"],
            [
                [
                    home,
                    _n(e["coverage"]),
                    _n(e["kept"]["error"]),
                    _n(e["full"]["error"]),
                    _n(e["kept"]["balanced_accuracy"]),
                    _n(e["full"]["balanced_accuracy"]),
                ]
                for home, e in results["households"].items()
            ],
        ),
        "",
        "## What this cannot show",
        "",
        *[f"- {_sentence(item)}" for item in config["what_this_cannot_show"]],
        "",
    ]
    return "\n".join(lines)


__all__ = ["PROTOCOL_PAGE", "RESULTS_PAGE", "render_page", "render_protocol"]
