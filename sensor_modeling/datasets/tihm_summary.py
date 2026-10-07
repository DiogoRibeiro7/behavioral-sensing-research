"""The TIHM pages, generated entirely from the frozen protocol and the record.

:func:`render_protocol` writes ``docs/TIHM_ALERT_BURDEN_PROTOCOL.md`` from the
frozen protocol file alone. It reports no result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PROTOCOL_FILE = "artifacts/tihm/alert_burden_protocol.json"
MAPPING_FILE = "artifacts/tihm/tihm_mapping.json"
PROTOCOL_PAGE = "TIHM_ALERT_BURDEN_PROTOCOL.md"


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
        *_bullets(payload["simulator_reference"]),
        "",
        "## Reporting",
        "",
        f"{_sentence(payload['reporting'])}",
        "",
    ]
    return "\n".join(lines)
