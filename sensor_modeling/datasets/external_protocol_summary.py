"""The Phase 5 protocol page, generated entirely from the frozen protocol file.

:func:`render_protocol` writes the documentation page from
``artifacts/phase5/external_protocol.json`` alone, so the page cannot disagree
with what was frozen.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .external_protocol import PROTOCOL_SCHEMA

PROTOCOL_FILE = "artifacts/phase5/external_protocol.json"


#: A leading identifier, such as ``pool_channels`` or ``sensor_modeling.x``.
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*[_.(]")

#: The order the page lists each section's entries in.
_ORDER = {
    "preprocessing": (
        "adapter",
        "conversion",
        "windows",
        "channels",
        "labels",
        "scored_windows",
    ),
    "conditions": ("zero_shot", "adapted", "same_windows"),
    "adaptation_rules": ("permitted", "forbidden", "adapted", "not_adapted"),
    "estimands": ("T", "A", "C", "S"),
}


def _sentence(text: str) -> str:
    """*text* with its first letter raised, unless it starts with an identifier."""
    if _IDENTIFIER.match(text):
        return text
    return text[:1].upper() + text[1:]


def _ordered(section: str, entries: Mapping[str, Any]) -> list[tuple[str, Any]]:
    order = _ORDER[section]
    return [(k, entries[k]) for k in order] + [
        (k, v) for k, v in entries.items() if k not in order
    ]


def _cell(text: Any) -> str:
    return "—" if text is None else str(text).replace("|", "\\|")


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def render_protocol(payload: Mapping[str, Any]) -> str:
    """The whole protocol page, from the frozen protocol alone."""
    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"not a {PROTOCOL_SCHEMA} declaration")
    lines: list[str] = []
    add = lines.append
    dataset = payload["dataset"]
    provenance = dataset["provenance"]
    households = payload["households"]
    periods = payload["periods"]
    mapping = payload["mapping"]
    models = payload["models"]

    add("# Phase 5: the external-generalisation protocol")
    add("")
    add(
        "ROADMAP Phase 5 asks whether results survive outside the CASAS "
        "ecosystem. This page is the protocol for the first external evaluation, "
        f"generated entirely from the frozen file `{PROTOCOL_FILE}` by "
        "`sensor_modeling.datasets.external_protocol_summary.render_protocol`. A "
        "test checks that the committed page is exactly that rendering."
    )
    add("")
    add(f"**Status: {payload['status']}.**")
    add("")
    add(f"- **Protocol digest.** `{payload['protocol_sha256']}`.")
    add(f"- **The question.** {_sentence(payload['question'])}?")
    add(
        "- **This page reports no performance.** The scoring run must check the "
        "frozen file, and reproduce its digests, before it scores anything."
    )
    add("")

    add("## The dataset")
    add("")
    lines += _table(
        ["Field", "Value"],
        [
            ["name", provenance["name"]],
            ["version", provenance["version"]],
            ["source", provenance["source"]],
            ["licence", provenance["licence"]],
            ["citation", provenance["citation"]],
            ["retrieved", provenance["retrieved"]],
            ["archive SHA-256", f"`{dataset['archive']['sha256']}`"],
        ],
    )
    add("")
    add("Every file's SHA-256:")
    add("")
    lines += _table(
        ["File", "SHA-256"],
        [[name, f"`{digest}`"] for name, digest in provenance["files"].items()],
    )
    add("")
    for note in provenance["notes"]:
        add(f"- {_sentence(note)}.")
    add(f"- **Why this dataset.** {_sentence(dataset['choice'])}.")
    add(f"- **Redistribution.** {_sentence(dataset['redistribution'])}.")
    add("")

    add("## Households and periods")
    add("")
    add(f"- **Inclusion.** {_sentence(households['inclusion'])}.")
    add(f"- **Exclusion.** {_sentence(households['exclusion'])}.")
    add(f"- **Eligibility.** {_sentence(households['eligibility'])}.")
    add(f"- **Timezone.** `{periods['timezone']}`.")
    add(f"- **Adaptation period.** {_sentence(periods['adaptation'])}.")
    add(f"- **Scored period.** {_sentence(periods['scored'])}.")
    add(f"- **Use.** {_sentence(periods['use'])}.")
    add("")
    lines += _table(
        ["Home", "Labelled days", "Scored days"],
        [
            [home, days, periods["scored_days"][home]]
            for home, days in households["labelled_days"].items()
        ],
    )
    add("")

    add("## Ontology mapping")
    add("")
    add(
        f"The mapping is `{mapping['file']}`, SHA-256 `{mapping['sha256']}`. "
        f"{_sentence(mapping['written_from'])}"
    )
    add("")
    lines += _table(
        ["Label", "Outcome", "Targets"],
        [
            [f"`{label}`", entry["outcome"], ", ".join(entry["targets"]) or None]
            for label, entry in mapping["labels"].items()
        ],
    )
    add("")
    add(f"- **New values.** {_sentence(mapping['new_values'])}.")
    add(
        "- **Scored states.** "
        + ", ".join(f"`{s}`" for s in payload["scored_states"])
        + "."
    )
    add("")
    add("### Unsupported states")
    add("")
    lines += _table(
        ["State", "Why"],
        [[f"`{s}`", why] for s, why in payload["unsupported_states"].items()],
    )
    add("")

    add("## Sensor mapping")
    add("")
    for home, rows in payload["sensor_mapping"].items():
        add(f"### {home}")
        add("")
        lines += _table(
            ["Sensor", "Native type", "Type", "Modality", "Room", "Model channel"],
            [
                [
                    f"`{r['sensor']}`",
                    r["native_type"],
                    r["type_status"],
                    r["modality"] and f"{r['modality']} ({r['kind']})",
                    r["room"],
                    r["model_channel"] or f"unsupported: {r['unsupported_because']}",
                ]
                for r in rows
            ],
        )
        add("")
        missing = payload["uninstrumented_channels"][home]
        add(
            "Model channels with no sensor here: "
            + (", ".join(f"`{c}`" for c in missing) if missing else "none")
            + "."
        )
        add("")

    add("## Preprocessing")
    add("")
    for key, text in _ordered("preprocessing", payload["preprocessing"]):
        if isinstance(text, list):
            text = ", ".join(f"`{t}`" for t in text)
        add(f"- **{_sentence(key.replace('_', ' '))}.** {_sentence(text)}.")
    add("")

    add("## Models and conditions")
    add("")
    declared = models["declared"]
    zero = models["zero_shot"]
    adapted = models["adapted"]
    ensemble = models["structural_ensemble"]
    add(
        f"- **Declared.** `{declared['specification']}`: {declared['description']}; "
        f"{declared['use']}."
    )
    add(
        f"- **Zero-shot.** `{zero['specification']}`: {zero['population']}, "
        f"{zero['pseudo_windows']:g} pseudo-windows, population SHA-256 "
        f"`{zero['population_sha256']}`. {_sentence(zero['inference'])}."
    )
    add(
        f"- **Adapted.** {_sentence(adapted['adaptation'])}, at strength "
        f"{adapted['pooling']['strength']:g} ({adapted['strength_source']})."
    )
    add(
        "- **Structural ensemble.** "
        + ", ".join(f"`{s}`" for s in ensemble["specifications"])
        + f": {ensemble['source']}."
    )
    add(f"- **Code.** {_sentence(models['code'])}.")
    add("")
    lines += _table(
        ["Population", "SHA-256"],
        [[name, f"`{d}`"] for name, d in ensemble["population_sha256"].items()],
    )
    add("")
    add(
        "The development homes: "
        + ", ".join(zero["development_homes"])
        + f". Splits SHA-256 `{zero['splits_sha256']}`."
    )
    add("")
    for key, text in _ordered("conditions", payload["conditions"]):
        add(f"- **{_sentence(key.replace('_', ' '))}.** {_sentence(text)}.")
    add("")
    add("### Adaptation rules")
    add("")
    for key, text in _ordered("adaptation_rules", payload["adaptation_rules"]):
        add(f"- **{_sentence(key.replace('_', ' '))}.** {_sentence(text)}.")
    add("")

    add("## Evaluation")
    add("")
    metrics = payload["metrics"]
    add(
        "- **Primary metrics.** "
        + ", ".join(f"`{m}`" for m in metrics["primary"])
        + "."
    )
    add(
        "- **Secondary metrics.** "
        + ", ".join(f"`{m}`" for m in metrics["secondary"])
        + "."
    )
    for key in ("selective", "chance", "scoring"):
        add(f"- **{_sentence(key)}.** {_sentence(metrics[key])}.")
    add("")
    boot = payload["bootstrap"]
    add(
        f"- **Bootstrap.** {_sentence(boot['unit'])}: {boot['resamples']:,} "
        f"resamples, {int(boot['confidence'] * 100)}% {boot['interval']} "
        f"intervals, seed {boot['seed']}. {_sentence(boot['paired'])}. "
        f"{_sentence(boot['reason'])}."
    )
    add("")
    lines += _table(
        ["Quantity", "Minimal difference"],
        [[f"`{k}`", f"{v:g}"] for k, v in payload["minimal_differences"].items()],
    )
    add("")
    add("### Estimands and criteria")
    add("")
    lines += _table(
        ["Estimand", "Definition", "Criterion"],
        [
            [key, _sentence(text), _sentence(payload["criteria"][key])]
            for key, text in _ordered("estimands", payload["estimands"])
        ],
    )
    add("")
    add(f"- **Verdicts.** {_sentence(payload['verdicts'])}.")
    add(f"- **Claims.** {_sentence(payload['criteria']['claims'])}.")
    add(f"- **Reporting.** {_sentence(payload['reporting'])}.")
    add("")
    return "\n".join(lines)
