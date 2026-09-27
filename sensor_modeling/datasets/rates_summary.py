"""A concise Markdown summary of a fitted-rates experiment record, from it alone.

:func:`render_summary` reads nothing but the record as written, so the summary
can always be regenerated from the artifact and checked against it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import mean
from typing import Any

from .gap_summary import _number, _table
from .history_summary import _estimate, _homes, _level
from .rates_experiment import CRITERIA, RESULT_SCHEMA

_PROBABILITY = ("log_loss", "brier", "calibration_error")
_FAMILIES = ("declared", "hurdle", "poisson")


def _ratio(estimate: Mapping[str, Any] | None) -> str:
    if not estimate or estimate["estimate"] is None:
        return ""
    interval = estimate["interval"]
    bounds = (
        f" [{math.exp(interval['low']):.2f}, {math.exp(interval['high']):.2f}]"
        if interval
        else ""
    )
    return f"{math.exp(estimate['estimate']):.2f}{bounds}"


def _count(summary: Mapping[str, Any]) -> str:
    if not summary.get("n"):
        return "0"
    return f"{summary['above']} / {summary['below']} of {summary['n']}"


def _fitted_silence(fitted: Mapping[str, Any], states: list[str]) -> list[list[str]]:
    """Per state: declared, hurdle and fitted-Poisson silence, and dispersion."""
    rows = []
    for state in states:
        declared, hurdle, poisson, dispersion = [], [], [], []
        for fold in fitted.values():
            for channel in fold["channels"]["channels"].values():
                entry = channel[state]
                if not entry["windows"]:
                    continue
                declared.append(math.exp(-entry["declared_mean"]))
                hurdle.append(entry["silence"])
                poisson.append(math.exp(-entry["mean"]))
                if entry["active_variance"] is not None and entry["active_mean"]:
                    dispersion.append(entry["active_variance"] / entry["active_mean"])
        rows.append(
            [
                state,
                _number(mean(sorted(declared))) if declared else "n/a",
                _number(mean(sorted(hurdle))) if hurdle else "n/a",
                _number(mean(sorted(poisson))) if poisson else "n/a",
                (
                    _number(sorted(dispersion)[len(dispersion) // 2])
                    if dispersion
                    else "n/a"
                ),
            ]
        )
    return rows


def render_summary(payload: Mapping[str, Any], *, level: int = 1) -> str:
    """A concise Markdown summary of a fitted-rates experiment record.

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

    lines = [
        f"{title} Fitted silence and activity rates: summary",
        "",
        f"Generated from the record `{payload['experiment']}`: protocol "
        f"`{configuration['protocol_sha256'][:12]}`, commit "
        f"`{str(environment.get('git_commit', 'unknown'))[:12]}`{dirty}. "
        f"Status: {results['status']}.",
        "",
        f"- {len(results['households'])} households in "
        f"{len(configuration['households']['folds'])} cross-fitted folds, each "
        "scored once by channel models fitted without it.",
        "- Every comparison is between two channel observation models on "
        "identical information: one set, or the recursion over every window "
        "(`R`).",
        "- Differences are paired by household, with the mean and the median and "
        f"their {confidence}% household bootstrap intervals from "
        f"{bootstrap['resamples']:,} resamples. A positive value favours the "
        "fitted model.",
        "- Minimal important differences: "
        + ", ".join(f"{metric} {value:g}" for metric, value in sorted(minimal.items()))
        + ".",
        "",
        f"{section} Pre-specified conclusions",
        "",
        *_table(
            ["Question", "Conclusion"],
            [
                [
                    "Do fitted hurdle channels improve on the declared rates, "
                    "current windows? (F0)",
                    results["conclusions"]["current_windows"],
                ],
                [
                    "And in the filter's recursion over every window? (FR)",
                    results["conclusions"]["recursion"],
                ],
            ],
        ),
        f"Success: {CRITERIA['success']}. Trade-off: {CRITERIA['trade-off']}. "
        f"Failure: {CRITERIA['failure']}.",
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
        f"{section} Estimands",
        "",
        "Mean differences, with households improved and worsened. Positive values "
        "favour the fitted model, that is, higher accuracy or lower loss and "
        "calibration error:",
        "",
        *_table(
            [
                "Key",
                "Comparison",
                "Balanced accuracy",
                "Log loss",
                "Brier",
                "Calibration error",
                "Conclusion",
            ],
            [
                [
                    e["key"],
                    f"{e['model']} vs {e['reference']}",
                    *(
                        f"{_estimate(e['comparisons'][m]['mean'])}, "
                        f"{_homes(e['comparisons'][m])}, {e['verdicts'][m]}"
                        for m in (accuracy, *_PROBABILITY)
                    ),
                    e["conclusion"],
                ]
                for e in results["estimands"]
            ],
        ),
    ]

    recall = [estimands[key] for key in ("F0", "FR")]
    quiet = configuration["silence_settings"]["quiet_states"]
    lines += [
        f"{section} Per-state recall",
        "",
        "Mean differences, homes improved and worsened, and verdicts:",
        "",
        *_table(
            ["State", "Rare", *(f"{e['key']}" for e in recall)],
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
        f"{section} Mechanisms",
        "",
        "One value per household, against zero. Inflation reductions are log "
        "ratios, also shown as ratios; accumulation reductions are slopes of "
        "overconfidence per quiet hour:",
        "",
        *_table(
            [
                "Key",
                "Quantity",
                "Mean",
                "As a ratio",
                "Households above / below",
                "Verdict",
            ],
            [
                [
                    m["key"],
                    m["quantity"],
                    _estimate(m["summary"].get("mean")),
                    (
                        _ratio(m["summary"].get("mean"))
                        if m["scale"] == "log_ratio"
                        else ""
                    ),
                    _count(m["summary"]),
                    m["verdict"],
                ]
                for m in results["mechanisms"]
            ],
        ),
        f"{section} Quiet runs",
        "",
        "In each household's Phase 3.3 representative quiet runs, how many end "
        "in `sleeping` after the recorded windows, and the median confidence then:",
        "",
        *_table(
            ["Channels", *quiet],
            [
                [
                    family,
                    *(
                        f"{results['drift'][family][state]['ending_sleeping']} of "
                        f"{results['drift'][family][state]['runs']}, "
                        f"{_number(results['drift'][family][state]['median_confidence'])}"
                        for state in quiet
                    ),
                ]
                for family in _FAMILIES
            ],
        ),
        f"{section} Fitted silence",
        "",
        "Mean over channels and folds of each family's silence probability in a "
        "window, and the median over channels and folds of an active window's "
        "count variance over its mean, from the training households:",
        "",
        *_table(
            ["State", "Declared", "Hurdle", "Fitted Poisson", "Active dispersion"],
            _fitted_silence(results["fitted"], results["states"]),
        ),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"
