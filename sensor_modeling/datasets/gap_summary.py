"""A concise Markdown summary of a recoverable-gap record, generated from it alone.

:func:`render_summary` reads nothing but the record as written, so the summary
can always be regenerated from the artifact and checked against it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .recoverable_gap import (
    CANDIDATES,
    FILTER,
    FORMULATION_PAIRS,
    LABELS,
    NESTED_PAIRS,
    REFERENCE,
    RESULT_SCHEMA,
)

#: What each nested pair of sets adds.
ADDED = {
    ("I0", "I1"): "time of day",
    ("I0", "I2"): "history",
    ("I1", "I3"): "history",
    ("I2", "I3"): "time of day",
    ("I0", "I3"): "both",
}

_MINUS = "−"


def _number(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    text = f"{value:+.3f}" if signed else f"{value:.3f}"
    return text.replace("-", _MINUS)


def _difference(comparison: Mapping[str, Any]) -> str:
    """Mean paired difference, its interval, and households favouring the model."""
    mean = comparison["mean"]
    interval = mean["interval"]
    bounds = (
        f" [{_number(interval['low'], True)}, {_number(interval['high'], True)}]"
        if interval
        else ""
    )
    return (
        f"{_number(mean['estimate'], True)}{bounds}, "
        f"{comparison['favours_model']}/{comparison['n']}"
    )


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    def row(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    align = ["---", *("---:" for _ in header[1:])]
    return [row(header), row(align), *(row(r) for r in rows), ""]


def _merged(entries: Sequence[Mapping[str, str]]) -> list[tuple[str, str, str]]:
    """Unsupported entries sharing a model and a reason, with their sets joined."""
    merged: dict[tuple[str, str], list[str]] = {}
    for entry in entries:
        merged.setdefault((entry["model"], entry["reason"]), []).append(
            entry["information_set"]
        )
    return [
        (model, ", ".join(labels), reason) for (model, reason), labels in merged.items()
    ]


class _Record:
    """Lookups into one record's results."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.results = payload["results"]
        if self.results.get("result_schema") != RESULT_SCHEMA:
            raise ValueError(f"not a {RESULT_SCHEMA} record")
        self.cells = {
            (cell["model"], cell["information_set"]): cell
            for cell in self.results["cells"]
        }

    def supported(self, model: str, label: str) -> bool:
        return bool(self.cells[(model, label)]["status"] != "unsupported")

    def level(self, model: str, label: str, metric: str) -> str:
        """Median, then mean with the interval of the mean."""
        if not self.supported(model, label):
            return "unsupported"
        summary = self.cells[(model, label)]["metrics"][metric]
        mean = _number(summary["mean"])
        interval = summary["mean_interval"]
        if interval:
            mean += f" [{_number(interval['low'])}, {_number(interval['high'])}]"
        return f"{_number(summary['median'])} ({mean})"

    def median(self, model: str, label: str, metric: str) -> str:
        if not self.supported(model, label):
            return "unsupported"
        return _number(self.cells[(model, label)]["metrics"][metric]["median"])

    def compared(self, group: str, metric: str, **where: str) -> str | None:
        for entry in self.results[group]:
            if entry["metric"] == metric and all(
                entry.get(key) == value for key, value in where.items()
            ):
                return _difference(entry["comparison"])
        return None


def render_summary(payload: Mapping[str, Any], *, level: int = 1) -> str:
    """A concise Markdown summary of a recoverable-gap record.

    Parameters
    ----------
    payload
        The record as written, for example from
        :func:`~sensor_modeling.evaluation.load_record`.
    level
        Heading level of the title. Sections are one level deeper.
    """
    record = _Record(payload)
    results = record.results
    configuration = payload["configuration"]
    environment = payload["environment"]
    bootstrap = configuration["bootstrap"]
    confidence = round(100 * bootstrap["confidence"])
    title, section = "#" * level, "#" * (level + 1)
    ba = "balanced_accuracy"
    candidates = tuple(results.get("candidates", CANDIDATES))
    pairs = tuple(
        (a, b) for a, b in results.get("formulation_pairs", FORMULATION_PAIRS)
    )
    models = (*candidates, REFERENCE)
    dirty = " (uncommitted changes)" if environment.get("git_dirty") == "true" else ""

    lines = [
        f"{title} Recoverable-information gap: summary",
        "",
        f"Generated from the record `{payload['experiment']}`: protocol "
        f"`{configuration['protocol_sha256'][:12]}`, commit "
        f"`{str(environment.get('git_commit', 'unknown'))[:12]}`{dirty}. "
        f"Status: {results['status']}.",
        "",
        f"- {len(results['households'])} households in "
        f"{len(configuration['folds'])} cross-fitted folds. Each is scored once, "
        "by models never fitted on it.",
        f"- Intervals are {confidence}% household bootstrap intervals from "
        f"{bootstrap['resamples']:,} resamples.",
        "- Differences are paired by household. A positive value favours the "
        "first model, or the larger set.",
        "- No setting was selected on any data.",
        "",
        f"{section} What each model receives",
        "",
        *_table(
            ["Model", *LABELS],
            [
                [
                    model,
                    *(
                        "matched" if record.supported(model, label) else "unsupported"
                        for label in LABELS
                    ),
                ]
                for model in candidates
            ],
        ),
        "The production `filter` is matched to no set. It conditions on every "
        "earlier window, so its information strictly contains `I0` and `I2`, "
        "and neither contains nor is contained in `I1` or `I3`.",
        "",
        f"{section} Balanced accuracy",
        "",
        f"Median over households, then the mean with its {confidence}% interval:",
        "",
        *_table(
            ["Model", *LABELS],
            [
                [model, *(record.level(model, label, ba) for label in LABELS)]
                for model in models
            ],
        ),
        f"Production filter, unrestricted: {record.level(FILTER, 'unbounded', ba)}.",
        "",
    ]

    probabilistic = [
        metric
        for metric in ("log_loss", "brier", "calibration_error")
        if metric in configuration["metrics"]
    ]
    if probabilistic:
        rows = [
            [model, label, *(record.median(model, label, m) for m in probabilistic)]
            for model in models
            for label in LABELS
            if record.supported(model, label)
        ]
        rows.append(
            [
                FILTER,
                "unbounded",
                *(record.median(FILTER, "unbounded", m) for m in probabilistic),
            ]
        )
        lines += [
            f"{section} Probability quality",
            "",
            "Median over households. Lower is better:",
            "",
            *_table(["Model", "Set", *probabilistic], rows),
        ]

    lines += [
        f"{section} What added information is worth",
        "",
        "The model is held fixed. Each cell gives the mean paired difference in "
        "balanced accuracy, its interval, and how many households improved:",
        "",
        *_table(
            ["Added", "Sets", *candidates],
            [
                [
                    ADDED[(small, large)],
                    f"{small} to {large}",
                    *(
                        record.compared(
                            "information_gains",
                            ba,
                            model=model,
                            **{"from": small, "to": large},
                        )
                        or "unsupported"
                        for model in candidates
                    ),
                ]
                for small, large in NESTED_PAIRS
            ],
        ),
        f"{section} What the formulation is worth",
        "",
        "The information is held fixed. Each cell gives the first model minus "
        "the second, in balanced accuracy:",
        "",
        *_table(
            ["Set", *(f"{a} {_MINUS} {b}" for a, b in pairs)],
            [
                [
                    label,
                    *(
                        record.compared(
                            "formulation_gaps",
                            ba,
                            model=a,
                            reference=b,
                            information_set=label,
                        )
                        or "unsupported"
                        for a, b in pairs
                    ),
                ]
                for label in LABELS
            ],
        ),
        f"{section} Interactions",
        "",
        "How much more the added information is worth to the first model than "
        "to the second, in balanced accuracy. A value away from zero means that "
        "information gains and formulation gaps do not add up:",
        "",
        *_table(
            ["Sets", *(f"{a} vs {b}" for a, b in pairs)],
            [
                [
                    f"{small} to {large}",
                    *(
                        record.compared(
                            "interactions",
                            ba,
                            model=a,
                            reference=b,
                            **{"from": small, "to": large},
                        )
                        or "not estimable"
                        for a, b in pairs
                    ),
                ]
                for small, large in NESTED_PAIRS
            ],
        ),
        f"{section} Against the production filter",
        "",
        "These comparisons are not matched. Where the filter's information "
        "contains the set, a model that scores higher cannot owe it to seeing "
        "more:",
        "",
        *_table(
            ["Model", "Set", "Relation", "Balanced accuracy difference"],
            [
                [
                    entry["model"],
                    entry["information_set"],
                    entry["relation"],
                    _difference(entry["comparison"]),
                ]
                for entry in results["against_filter"]
                if entry["metric"] == ba
            ],
        ),
    ]

    recall = [(model, "I2") for model in candidates if record.supported(model, "I2")]
    needs_hour = [
        (model, "I3")
        for model in candidates
        if not record.supported(model, "I2") and record.supported(model, "I3")
    ]
    recall += [*needs_hour, (FILTER, "unbounded")]
    note = (
        " A model that needs the hour is shown at `I3`, the richest set it can consume."
        if needs_hour
        else ""
    )
    lines += [
        f"{section} Per-state recall",
        "",
        "Median over the households where the state occurs, at `I2` (the richest "
        "set every model can consume) and for the filter. The record holds every "
        f"cell:{note}",
        "",
        *_table(
            [
                "State",
                *(model if at != "I3" else f"{model} (I3)" for model, at in recall),
            ],
            [
                [
                    state,
                    *(
                        _number(record.cells[cell]["per_state_recall"][state]["median"])
                        for cell in recall
                    ),
                ]
                for state in results["states"]
            ],
        ),
        f"{section} Not compared",
        "",
        *(
            f"- `{model}` in {labels}: {reason}."
            for model, labels, reason in _merged(results["unsupported"])
        ),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"
