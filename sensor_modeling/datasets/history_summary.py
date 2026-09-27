"""A concise Markdown summary of an explicit-history experiment record, from it alone.

:func:`render_summary` reads nothing but the record as written, so the summary
can always be regenerated from the artifact and checked against it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .gap_summary import _number, _table, inference_line
from .history_experiment import CRITERIA, RESULT_SCHEMA

_PROBABILITY = ("log_loss", "brier", "calibration_error")


def _estimate(estimate: Mapping[str, Any]) -> str:
    interval = estimate["interval"]
    bounds = (
        f" [{_number(interval['low'], True)}, {_number(interval['high'], True)}]"
        if interval
        else ""
    )
    return f"{_number(estimate['estimate'], True)}{bounds}"


def _homes(comparison: Mapping[str, Any]) -> str:
    return f"{comparison['favours_model']} / {comparison['favours_reference']}"


def _level(summary: Mapping[str, Any]) -> str:
    interval = summary.get("mean_interval")
    mean = _number(summary["mean"])
    if interval:
        mean += f" [{_number(interval['low'])}, {_number(interval['high'])}]"
    return f"{_number(summary['median'])} ({mean})"


def render_summary(payload: Mapping[str, Any], *, level: int = 1) -> str:
    """A concise Markdown summary of an explicit-history experiment record.

    Parameters
    ----------
    payload
        The record as written, for example from
        :func:`~sensor_modeling.evaluation.load_record`.
    level
        Heading level of the title. Sections are one level deeper.
    """
    results = payload["results"]
    if results.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    configuration = payload["configuration"]
    environment = payload["environment"]
    bootstrap = configuration["bootstrap"]
    minimal = configuration["minimal_differences"]
    confidence = round(100 * bootstrap["confidence"])
    title, section = "#" * level, "#" * (level + 1)
    dirty = " (uncommitted changes)" if environment.get("git_dirty") == "true" else ""
    estimands = {e["key"]: e for e in results["estimands"]}
    accuracy = "balanced_accuracy"

    def gain(key: str) -> str:
        return _estimate(estimands[key]["comparisons"][accuracy]["mean"])

    lines = [
        f"{title} Explicit history state: summary",
        "",
        f"Generated from the record `{payload['experiment']}`: protocol "
        f"`{configuration['protocol_sha256'][:12]}`, commit "
        f"`{str(environment.get('git_commit', 'unknown'))[:12]}`{dirty}. "
        f"Status: {results['status']}.",
        "",
        inference_line(payload),
        f"- {len(results['households'])} households in "
        f"{len(configuration['households']['folds'])} cross-fitted folds, each "
        "scored once by models never fitted on it.",
        "- Differences are paired by household, with the mean and the median and "
        f"their {confidence}% household bootstrap intervals from "
        f"{bootstrap['resamples']:,} resamples. A positive value favours the "
        "first model or the larger set.",
        "- Formulation estimands compare two models on one set. Information "
        "estimands compare one model family across nested sets.",
        "- Minimal important differences: "
        + ", ".join(f"{metric} {value:g}" for metric, value in sorted(minimal.items()))
        + ".",
        "",
        f"{section} Pre-specified conclusions",
        "",
        *_table(
            ["Question", "Conclusion", "Rule"],
            [
                [
                    "Does the explicit history state add to the original model "
                    "on identical information? (H1)",
                    results["conclusions"]["explicit_history"],
                    CRITERIA["success"],
                ],
                [
                    "Does it add when the hour is known? (S1)",
                    results["conclusions"]["with_the_hour"],
                    CRITERIA["with_the_hour"],
                ],
            ],
        ),
        f"Recent history is worth {gain('H2')} to the explicit-history model (H2) "
        f"and {gain('R1')} to the original model (R1); H1, their difference on "
        f"the same homes, is {gain('H1')}. {CRITERIA['materially_more']}.",
        "",
        f"Verdicts: {CRITERIA['verdicts']}.",
        "",
        f"{section} Every cell",
        "",
        "Balanced accuracy as the median, then the mean with its interval. The "
        "other metrics are medians, and lower is better:",
        "",
        *_table(
            [
                "Model",
                "Set",
                "Balanced accuracy",
                "Log loss",
                "Brier",
                "Calibration error",
            ],
            [
                [
                    cell["model"],
                    cell["information_set"],
                    _level(cell["metrics"][accuracy]),
                    *(_number(cell["metrics"][m]["median"]) for m in _PROBABILITY),
                ]
                for cell in results["cells"]
            ],
        ),
        f"{section} Estimands: balanced accuracy",
        "",
        *_table(
            [
                "Key",
                "Kind",
                "Comparison",
                "Mean difference",
                "Median difference",
                "Homes improved / worsened",
                "Verdict",
            ],
            [
                [
                    e["key"],
                    e["kind"],
                    f"{e['model']} vs {e['reference']}",
                    _estimate(e["comparisons"][accuracy]["mean"]),
                    _estimate(e["comparisons"][accuracy]["median"]),
                    _homes(e["comparisons"][accuracy]),
                    e["verdicts"][accuracy],
                ]
                for e in results["estimands"]
            ],
        ),
        f"{section} Estimands: probability quality",
        "",
        "Mean differences. Positive values favour the first model, that is, its "
        "loss or calibration error is lower:",
        "",
        *_table(
            ["Key", "Log loss", "Brier", "Calibration error"],
            [
                [
                    e["key"],
                    *(
                        f"{_estimate(e['comparisons'][m]['mean'])}, {e['verdicts'][m]}"
                        for m in _PROBABILITY
                    ),
                ]
                for e in results["estimands"]
            ],
        ),
    ]

    recall = [e for e in results["estimands"] if "per_state_recall" in e]
    lines += [
        f"{section} Per-state recall",
        "",
        "Mean differences, homes improved and worsened, and verdicts. A state is "
        "left out of a household where it never occurs:",
        "",
        *_table(
            ["State", "Rare", *(f"{e['key']} ({e['kind']})" for e in recall)],
            [
                [
                    state,
                    "yes" if state in results["rare_states"] else "no",
                    *(
                        f"{_estimate(e['per_state_recall'][state]['comparison']['mean'])}, "
                        f"{_homes(e['per_state_recall'][state]['comparison'])}, "
                        f"{e['per_state_recall'][state]['verdict']}"
                        for e in recall
                    ),
                ]
                for state in results["states"]
            ],
        ),
        f"{section} Time and history interaction",
        "",
        f"{CRITERIA['interaction'][:1].upper()}{CRITERIA['interaction'][1:]}:",
        "",
        *_table(
            [
                "Family",
                "Balanced accuracy",
                "Homes improved / worsened",
                "Log loss",
                "Verdict",
            ],
            [
                [
                    i["family"],
                    _estimate(i["comparisons"][accuracy]["mean"]),
                    _homes(i["comparisons"][accuracy]),
                    _estimate(i["comparisons"]["log_loss"]["mean"]),
                    i["verdicts"][accuracy],
                ]
                for i in results["interactions"]
            ],
        ),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"
