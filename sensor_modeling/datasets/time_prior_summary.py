"""A concise Markdown summary of a time-prior experiment record, from the record alone.

:func:`render_summary` reads nothing but the record as written, so the summary
can always be regenerated from the artifact and checked against it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .gap_summary import _number, _table
from .time_prior_experiment import CRITERIA, RESULT_SCHEMA

_PROBABILITY = ("log_loss", "brier", "calibration_error")


def _interval(comparison: Mapping[str, Any]) -> str:
    mean = comparison["mean"]
    interval = mean["interval"]
    bounds = (
        f" [{_number(interval['low'], True)}, {_number(interval['high'], True)}]"
        if interval
        else ""
    )
    return f"{_number(mean['estimate'], True)}{bounds}"


def _homes(comparison: Mapping[str, Any]) -> str:
    return f"{comparison['favours_model']}/{comparison['n']}"


def render_summary(payload: Mapping[str, Any], *, level: int = 1) -> str:
    """A concise Markdown summary of a time-prior experiment record.

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
    conclusions = results["conclusions"]
    estimands = results["estimands"]

    lines = [
        f"{title} Hierarchical time-of-day prior: summary",
        "",
        f"Generated from the record `{payload['experiment']}`: protocol "
        f"`{configuration['protocol_sha256'][:12]}`, commit "
        f"`{str(environment.get('git_commit', 'unknown'))[:12]}`{dirty}. "
        f"Status: {results['status']}.",
        "",
        f"- {len(results['households'])} households in "
        f"{len(configuration['households']['folds'])} cross-fitted folds, each "
        "scored once by models never fitted on it.",
        f"- Differences are means of paired household differences, with "
        f"{confidence}% household bootstrap intervals from "
        f"{bootstrap['resamples']:,} resamples. A positive value favours the "
        "first model.",
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
                    "Can the hierarchical model use the hour? (P1)",
                    conclusions["time_of_day"],
                    CRITERIA["success"],
                ],
                [
                    "Does it improve on the original model? (P2)",
                    conclusions["against_original"],
                    CRITERIA["practical"],
                ],
                [
                    "Does household adaptation help? (A-I1)",
                    conclusions["household_adaptation"],
                    CRITERIA["hierarchy"],
                ],
            ],
        ),
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
                    _level(cell["metrics"]["balanced_accuracy"]),
                    *(
                        _number(cell["metrics"][metric]["median"])
                        for metric in _PROBABILITY
                    ),
                ]
                for cell in results["cells"]
            ],
        ),
        f"{section} Estimands: balanced accuracy",
        "",
        *_table(
            [
                "Key",
                "Comparison",
                "Mean difference",
                "Homes favouring the first",
                "Verdict",
            ],
            [
                [
                    e["key"],
                    f"{e['model']} vs {e['reference']}",
                    _interval(e["comparisons"]["balanced_accuracy"]),
                    _homes(e["comparisons"]["balanced_accuracy"]),
                    e["verdicts"]["balanced_accuracy"],
                ]
                for e in estimands
            ],
        ),
        f"{section} Estimands: probability quality",
        "",
        "Positive values favour the first model, that is, its loss or calibration "
        "error is lower:",
        "",
        *_table(
            ["Key", "Log loss", "Brier", "Calibration error"],
            [
                [
                    e["key"],
                    *(
                        f"{_interval(e['comparisons'][metric])}, {e['verdicts'][metric]}"
                        for metric in _PROBABILITY
                    ),
                ]
                for e in estimands
            ],
        ),
    ]

    recall = [e for e in estimands if "per_state_recall" in e]
    lines += [
        f"{section} Per-state recall",
        "",
        "Rare states have under the declared share of labelled time in either "
        "fold's training homes. A state is left out of a household where it "
        "never occurs:",
        "",
        *_table(
            [
                "State",
                "Rare",
                *(f"{e['key']} ({e['model']} vs {e['reference']})" for e in recall),
            ],
            [
                [
                    state,
                    "yes" if state in results["rare_states"] else "no",
                    *(
                        f"{_interval(e['per_state_recall'][state]['comparison'])}, "
                        f"{_homes(e['per_state_recall'][state]['comparison'])}, "
                        f"{e['per_state_recall'][state]['verdict']}"
                        for e in recall
                    ),
                ]
                for state in results["states"]
            ],
        ),
        f"{section} Household adaptation",
        "",
        "Each held-out home's deviation is fitted from its own labels in the "
        f"first {configuration['pooling']['adaptation_arm']['window_days']:g} "
        "days, and it is scored only after them. The adapted prior is compared "
        "with the population prior on those moments. The declared pooling "
        "strength is marked; the others are sensitivity checks:",
        "",
        *_table(
            [
                "Key",
                "Set",
                "Declared",
                "Balanced accuracy",
                "Log loss",
                "Calibration error",
            ],
            [
                [
                    a["key"],
                    a["information_set"],
                    "yes" if a["declared"] else "no",
                    f"{_interval(a['comparisons']['balanced_accuracy'])}, "
                    f"{_homes(a['comparisons']['balanced_accuracy'])}, "
                    f"{a['verdicts']['balanced_accuracy']}",
                    f"{_interval(a['comparisons']['log_loss'])}, {a['verdicts']['log_loss']}",
                    f"{_interval(a['comparisons']['calibration_error'])}, "
                    f"{a['verdicts']['calibration_error']}",
                ]
                for a in results["adaptation"]
            ],
        ),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def _level(summary: Mapping[str, Any]) -> str:
    interval = summary.get("mean_interval")
    mean = _number(summary["mean"])
    if interval:
        mean += f" [{_number(interval['low'])}, {_number(interval['high'])}]"
    return f"{_number(summary['median'])} ({mean})"
