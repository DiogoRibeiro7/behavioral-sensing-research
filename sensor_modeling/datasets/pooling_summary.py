"""A concise Markdown summary of a partial-pooling experiment record, from it alone.

:func:`render_summary` reads nothing but the record as written, so the summary
can always be regenerated from the artifact and checked against it.
"""

from __future__ import annotations

from collections.abc import Mapping
from statistics import median
from typing import Any

from .gap_summary import _number, _table, inference_line
from .history_summary import _estimate, _homes, _level
from .pooling_experiment import CRITERIA, RESULT_SCHEMA

_PROBABILITY = ("brier", "calibration_error")


def _days(value: float) -> str:
    return f"{value:g} day" if value == 1 else f"{value:g} days"


def _median_or_na(values: list[float]) -> str:
    return _number(median(values)) if values else "n/a"


def _shrinkage_by_state(
    households: Mapping[str, Any], states: list[str], arm: str
) -> list[list[str]]:
    """Per state: the pooled silence parameter across households and channels."""
    rows = []
    for index, state in enumerate(states):
        counts, shrinkage, raw_gap, pooled_gap = [], [], [], []
        for entry in households.values():
            arm_entry = entry["arms"][arm]
            if not arm_entry["eligible"]:
                continue
            for channel in arm_entry["pooled"]["channels"]:
                silence = channel["silence"]
                count = silence["count"][index]
                counts.append(float(count))
                shrinkage.append(float(silence["shrinkage"][index]))
                if silence["raw"][index] is not None:
                    raw_gap.append(
                        abs(silence["raw"][index] - silence["population"][index])
                    )
                    pooled_gap.append(
                        abs(silence["pooled"][index] - silence["population"][index])
                    )
        rows.append(
            [
                state,
                _median_or_na(counts),
                _median_or_na(shrinkage),
                _median_or_na(raw_gap),
                _median_or_na(pooled_gap),
            ]
        )
    return rows


def render_summary(payload: Mapping[str, Any], *, level: int = 1) -> str:
    """A concise Markdown summary of a partial-pooling experiment record.

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
    days = configuration["arm_days"]
    households = results["households"]
    estimands = {e["key"]: e for e in results["estimands"]}

    lines = [
        f"{title} Partial pooling of household parameters: summary",
        "",
        f"Generated from the record `{payload['experiment']}`: protocol "
        f"`{configuration['protocol_sha256'][:12]}`, commit "
        f"`{str(environment.get('git_commit', 'unknown'))[:12]}`{dirty}. "
        f"Status: {results['status']}.",
        "",
        inference_line(payload),
        f"- {len(households)} held-out households in "
        f"{len(configuration['households']['folds'])} cross-fitted folds. Each is "
        "scored once, by a population and a selected strength fitted without it.",
        f"- Two adaptation arms: {_days(days['week'])} (`week`) and "
        f"{_days(days['day'])} (`day`). Within an arm, every model is scored on the "
        "same labelled windows, after the household's adaptation window. Nothing "
        "is compared across arms.",
        f"- The declared pooling strength is {configuration['pooling']['strength']:g} "
        "windows; the unconstrained model uses "
        f"{configuration['models']['unconstrained']['strength']:g}.",
        "- Differences are paired by household, with the mean and the median and "
        f"their {confidence}% household bootstrap intervals from "
        f"{bootstrap['resamples']:,} resamples. A positive value favours the "
        "first model.",
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
                    "Does partial pooling improve on the population, current "
                    "windows? (P1)",
                    results["conclusions"]["pooling_current_windows"],
                ],
                [
                    "And in the recursion? (P2)",
                    results["conclusions"]["pooling_recursion"],
                ],
                [
                    "Does unconstrained per-home fitting overfit small homes? (O1)",
                    results["conclusions"]["overfitting_small_homes"],
                ],
            ],
        ),
        f"Pooling success: {CRITERIA['pooling_success']}. Trade-off: "
        f"{CRITERIA['pooling_trade-off']}. Failure: {CRITERIA['pooling_failure']}.",
        "",
        f"Overfits: {CRITERIA['overfits']}. Does not overfit: "
        f"{CRITERIA['does not overfit']}.",
        "",
        f"Verdicts: {CRITERIA['verdicts']}.",
        "",
        f"{section} Strength selection",
        "",
        "Mean log loss of the left-out training households at each strength, and "
        "the strength selected. Held-out households are never used:",
        "",
    ]
    grid = [f"{s:g}" for s in configuration["selection"]["grid"]]
    lines += _table(
        ["Fold", "Training households", *grid, "Selected"],
        [
            [
                fold,
                str(len(selection["households"])),
                *(_number(selection["mean_log_loss"][s]) for s in grid),
                f"{selection['selected']:g}",
            ]
            for fold, selection in results["selection"].items()
        ],
    )
    lines += [
        f"{section} Every cell",
        "",
        "Balanced accuracy as the median, then the mean with its interval. The "
        "other metrics are medians, and lower is better:",
        "",
        *_table(
            [
                "Model",
                "Setting",
                "Arm",
                "Households",
                "Balanced accuracy",
                "Log loss",
                "Brier",
                "Calibration error",
            ],
            [
                [
                    cell["model"],
                    cell["information_set"],
                    cell["arm"],
                    str(cell["households"]),
                    _level(cell["metrics"]["balanced_accuracy"]),
                    *(
                        _number(cell["metrics"][m]["median"])
                        for m in ("log_loss", "brier", "calibration_error")
                    ),
                ]
                for cell in results["cells"]
            ],
        ),
        f"{section} Estimands: log loss",
        "",
        "The primary metric. Positive values favour the first model, whose log "
        "loss is lower:",
        "",
        *_table(
            [
                "Key",
                "Comparison",
                "Mean difference",
                "Median difference",
                "Homes improved / worsened",
                "Verdict",
                "Conclusion",
            ],
            [
                [
                    e["key"],
                    f"{e['model']} vs {e['reference']}",
                    _estimate(e["comparisons"]["log_loss"]["mean"]),
                    _estimate(e["comparisons"]["log_loss"]["median"]),
                    _homes(e["comparisons"]["log_loss"]),
                    e["verdicts"]["log_loss"],
                    e["conclusion"],
                ]
                for e in results["estimands"]
            ],
        ),
        f"{section} Estimands: the guards",
        "",
        "Mean differences, homes improved and worsened, and verdicts:",
        "",
        *_table(
            ["Key", "Balanced accuracy", "Brier", "Calibration error"],
            [
                [
                    e["key"],
                    *(
                        f"{_estimate(e['comparisons'][m]['mean'])}, "
                        f"{_homes(e['comparisons'][m])}, {e['verdicts'][m]}"
                        for m in ("balanced_accuracy", *_PROBABILITY)
                    ),
                ]
                for e in results["estimands"]
            ],
        ),
        f"{section} By amount of adaptation data",
        "",
        "Within each arm, the eligible households are split at the median number "
        "of labelled adaptation windows. Log loss mean difference, homes "
        "improved and worsened, and conclusion:",
        "",
        *_table(
            ["Key", "Stratum", "Households", "Log loss", "Homes", "Conclusion"],
            [
                [
                    e["key"],
                    name,
                    str(len(stratum["households"])),
                    _estimate(stratum["comparisons"]["log_loss"]["mean"]),
                    _homes(stratum["comparisons"]["log_loss"]),
                    stratum["conclusion"],
                ]
                for e in results["estimands"]
                if e["role"] == "primary"
                for name, stratum in e["strata"].items()
            ],
        ),
        f"{section} Per-state recall",
        "",
        "Mean differences, homes improved and worsened, and verdicts:",
        "",
        *_table(
            ["State", "Rare", "P1", "O1"],
            [
                [
                    state,
                    "yes" if state in results["rare_states"] else "no",
                    *(
                        f"{_estimate(estimands[k]['per_state_recall'][state]['comparison']['mean'])}, "
                        f"{_homes(estimands[k]['per_state_recall'][state]['comparison'])}, "
                        f"{estimands[k]['per_state_recall'][state]['verdict']}"
                        for k in ("P1", "O1")
                    ),
                ]
                for state in results["states"]
            ],
        ),
        f"{section} Pooled silence by state",
        "",
        "For the silence probability pooled with the declared strength, across "
        "households and channels: the median number of household windows, the "
        "median effective shrinkage, and the median distance of the raw and the "
        "pooled estimate from the population's. Week arm, then day arm:",
        "",
    ]
    for arm in ("week", "day"):
        lines += _table(
            [
                f"State ({arm})",
                "Windows",
                "Shrinkage",
                "Raw from population",
                "Pooled from population",
            ],
            _shrinkage_by_state(households, results["states"], arm),
        )
    lines += [
        f"{section} Households",
        "",
        "Each household's labelled adaptation windows per arm (an asterisk marks "
        "a household left out of that arm) and its log loss change in the "
        "primary estimands:",
        "",
        *_table(
            ["Household", "Fold", "Week windows", "Day windows", "P1", "P2", "O1"],
            [
                [
                    home,
                    entry["fold"],
                    f"{entry['arms']['week']['adaptation_windows']}"
                    + ("" if entry["arms"]["week"]["eligible"] else "*"),
                    f"{entry['arms']['day']['adaptation_windows']}"
                    + ("" if entry["arms"]["day"]["eligible"] else "*"),
                    *(
                        _number(
                            estimands[k]["comparisons"]["log_loss"]["differences"].get(
                                home
                            ),
                            True,
                        )
                        for k in ("P1", "P2", "O1")
                    ),
                ]
                for home, entry in households.items()
            ],
        ),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"
