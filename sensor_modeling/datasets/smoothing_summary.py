"""A concise Markdown summary of a fixed-lag smoothing experiment record, from it alone.

:func:`render_summary` reads nothing but the record as written, so the summary
can always be regenerated from the artifact and checked against it. Every
difference it shows is labelled a smoothing gain with its delay, never an
online improvement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .gap_summary import _number, _table
from .history_summary import _estimate, _homes, _level
from .smoothing_experiment import (
    ALL_TRANSITIONS,
    CRITERIA,
    ONLINE_KEY,
    RESULT_SCHEMA,
    regime_key,
)

_METRICS = ("balanced_accuracy", "log_loss", "brier", "calibration_error")
_SHARES = ("changed", "corrected", "broken", "corrections_among_changes")


def _minutes(value: float) -> str:
    return f"{value:g} min"


def _judged(entry: Mapping[str, Any]) -> str:
    """A compared quantity: its mean difference, homes improved and worsened, verdict."""
    comparison = entry["comparison"]
    if comparison is None:
        return "n/a"
    return f"{_estimate(comparison['mean'])}, {_homes(comparison)}, {entry['verdict']}"


def _across(summary: Mapping[str, Any]) -> str:
    """A per-household value across households: its mean and interval."""
    if not summary.get("n"):
        return "n/a"
    mean = summary["mean"]
    interval = mean["interval"]
    bounds = (
        f" [{_number(interval['low'])}, {_number(interval['high'])}]"
        if interval
        else ""
    )
    return f"{_number(mean['estimate'])}{bounds}"


def _regime_names(configuration: Mapping[str, Any]) -> list[str]:
    """The regimes in declared order: the filter, then each smoother by lag.

    The written record sorts its keys, so its own order is not the declared one.
    """
    return [ONLINE_KEY, *(regime_key(lag) for lag in configuration["lags"]["windows"])]


def _delay(results: Mapping[str, Any], key: str) -> str:
    return _minutes(results["regimes"][key]["delay_seconds"] / 60.0)


def _gain_rows(
    estimands: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> list[list[str]]:
    return [
        [
            e["key"],
            e["model"],
            *(
                f"{_estimate(e['comparisons'][m]['mean'])}, "
                f"{_homes(e['comparisons'][m])}, {e['verdicts'][m]}"
                for m in metrics
            ),
            e["conclusion"],
        ]
        for e in estimands
    ]


def render_summary(payload: Mapping[str, Any], *, level: int = 1) -> str:
    """A concise Markdown summary of a fixed-lag smoothing experiment record.

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
    households = results["households"]
    estimands = results["estimands"]
    regimes = results["regimes"]
    names = _regime_names(configuration)
    focus = results["focus_states"]
    groups = [ALL_TRANSITIONS, *focus]
    smoothers = [key for key in names if key != ONLINE_KEY]

    lines = [
        f"{title} Fixed-lag smoothing against online filtering: summary",
        "",
        f"Generated from the record `{payload['experiment']}`: protocol "
        f"`{configuration['protocol_sha256'][:12]}`, commit "
        f"`{str(environment.get('git_commit', 'unknown'))[:12]}`{dirty}. "
        f"Status: {results['status']}.",
        "",
        "- Inference regimes compared: "
        + "; ".join(regimes[key]["label"] for key in names)
        + ". The record states the longest, "
        f"{payload['inference']['label']}, which bounds every estimate in it. "
        "Every cell and comparison carries its own regime.",
        "- **Every difference below is a smoothing gain.** The smoother reads up "
        "to its lag after each window and can report only that long after it. "
        "No difference is an improvement of the online filter.",
        f"- {len(households)} held-out households in "
        f"{len(configuration['households']['folds'])} cross-fitted folds, each "
        "scored once by channel models fitted without it.",
        f"- Formulation: {configuration['formulation']['model']}.",
        f"- Every regime is scored on the same labelled windows. The last "
        f"{configuration['lags']['windows'][-1]} windows of each recording are "
        "not scored, so every smoothed estimate reads its full lag.",
        "- Differences are paired by household, with the mean and the median and "
        f"their {confidence}% household bootstrap intervals from "
        f"{bootstrap['resamples']:,} resamples. A positive value favours the "
        "smoother.",
        "- Minimal important differences: "
        + ", ".join(f"{metric} {value:g}" for metric, value in sorted(minimal.items()))
        + ".",
        "",
        f"{section} Pre-specified conclusions",
        "",
        *_table(
            ["Question", "Reporting delay", "Conclusion"],
            [
                [
                    f"How much does the {regimes[e['model']]['label']} recover over "
                    f"the online filter? ({e['key']})",
                    _delay(results, e["model"]),
                    e["conclusion"],
                ]
                for e in estimands
            ],
        ),
        f"Gain: {CRITERIA['gain']}. Trade-off: {CRITERIA['trade-off']}. "
        f"Probability gain: {CRITERIA['probability gain']}. No gain: "
        f"{CRITERIA['no gain']}.",
        "",
        f"Verdicts: {CRITERIA['verdicts']}.",
        "",
        f"{section} Every regime",
        "",
        "Balanced accuracy as the median, then the mean with its interval. The "
        "other metrics are medians, and lower is better:",
        "",
        *_table(
            [
                "Regime",
                "Reporting delay",
                "Balanced accuracy",
                "Log loss",
                "Brier",
                "Calibration error",
            ],
            [
                [
                    cell["regime"],
                    _delay(results, cell["regime"]),
                    _level(cell["metrics"]["balanced_accuracy"]),
                    *(
                        _number(cell["metrics"][m]["median"])
                        for m in ("log_loss", "brier", "calibration_error")
                    ),
                ]
                for cell in results["cells"]
            ],
        ),
        f"{section} Smoothing gains",
        "",
        "Each smoother against the online filter on the same windows: mean "
        "difference, homes improved and worsened, and verdict. Positive values "
        "favour the smoother, that is, higher accuracy or lower loss and "
        "calibration error. Each is available only after the smoother's delay:",
        "",
        *_table(
            [
                "Key",
                "Smoother",
                "Balanced accuracy",
                "Log loss",
                "Brier",
                "Calibration error",
                "Conclusion",
            ],
            _gain_rows(estimands, _METRICS),
        ),
        f"{section} Per-state recall",
        "",
        "Smoothing gain in each state's recall: mean difference, homes improved "
        "and worsened, and verdict:",
        "",
        *_table(
            ["State", "Rare", "Focus", *(e["key"] for e in estimands)],
            [
                [
                    state,
                    "yes" if state in results["rare_states"] else "no",
                    "yes" if state in focus else "no",
                    *(_judged(e["per_state_recall"][state]) for e in estimands),
                ]
                for state in results["states"]
            ],
        ),
        f"{section} What smoothing changes",
        "",
        "Against the online filter's most probable state on the same windows: the "
        "share of windows changed, the share of the filter's errors corrected, "
        "the share of its correct windows made wrong, and the share of changes "
        "that are corrections. Household means with their intervals:",
        "",
        *_table(
            [
                "Key",
                "Reporting delay",
                "Changed",
                "Corrected",
                "Made wrong",
                "Corrections among changes",
            ],
            [
                [
                    e["key"],
                    _minutes(e["operational"]["reporting_delay_minutes"]),
                    *(_across(e["operational"][name]) for name in _SHARES),
                ]
                for e in estimands
            ],
        ),
        "Within the windows whose labelled state is each focus state:",
        "",
        *_table(
            ["State", "Key", "Changed", "Corrected", "Made wrong"],
            [
                [
                    state,
                    e["key"],
                    *(
                        _across(e["operational"]["by_state"][state][name])
                        for name in ("changed", "corrected", "broken")
                    ),
                ]
                for state in focus
                for e in estimands
            ],
        ),
        f"{section} Transitions",
        "",
        f"Accuracy within {configuration['transitions']['boundary_windows']} "
        "windows of a true transition, before or after it, and further from "
        "every transition. Household means with their intervals:",
        "",
        *_table(
            ["Regime", "Near transitions", "Elsewhere"],
            [
                [
                    key,
                    _across(results["transition_levels"][key]["boundary_accuracy"]),
                    _across(results["transition_levels"][key]["interior_accuracy"]),
                ]
                for key in names
            ],
        ),
        "Detection rate, and decision delay in minutes, including the regime's "
        "reporting delay. Household means of each household's rate and median "
        "delay:",
        "",
        *_table(
            ["Transitions into", "Regime", "Detection rate", "Decision delay"],
            [
                [
                    group,
                    key,
                    *(
                        _across(
                            results["transition_levels"][key]["transitions"][group][q]
                        )
                        for q in ("detection_rate", "decision_delay_minutes")
                    ),
                ]
                for group in groups
                for key in names
            ],
        ),
        "Smoothing gains near transitions: mean difference, homes improved and "
        "worsened, and verdict. For decision delay, a positive value means the "
        "smoother reports the new state sooner:",
        "",
        *_table(
            ["Key", "Accuracy near transitions"]
            + [f"{q} ({g})" for g in groups for q in ("detection", "delay")],
            [
                [
                    e["key"],
                    _judged(e["transitions"]["boundary_accuracy"]),
                    *(
                        _judged(e["transitions"]["groups"][g][q])
                        for g in groups
                        for q in ("detection_rate", "decision_delay_minutes")
                    ),
                ]
                for e in estimands
            ],
        ),
        f"{section} Evidence read",
        "",
        "Each regime's scored estimates: how many, the first and last prediction "
        "timestamps, the latest evidence any of them read, and the most future "
        "information any of them used:",
        "",
        *_table(
            [
                "Regime",
                "Causal",
                "Predictions",
                "First prediction",
                "Last prediction",
                "Latest evidence",
                "Largest lead",
            ],
            [
                [
                    key,
                    "yes" if regime["causal"] else "no",
                    f"{regime['evidence']['predictions']:,}",
                    regime["evidence"]["first_prediction"],
                    regime["evidence"]["last_prediction"],
                    regime["evidence"]["latest_evidence"],
                    _minutes(regime["evidence"]["max_lead_seconds"] / 60.0),
                ]
                for key, regime in ((k, regimes[k]) for k in names)
            ],
        ),
        f"{section} Households",
        "",
        "Each household's scored windows and transitions, and its balanced "
        "accuracy smoothing gain per smoother:",
        "",
        *_table(
            [
                "Household",
                "Fold",
                "Scored windows",
                "Transitions",
                *(e["key"] for e in estimands),
            ],
            [
                [
                    home,
                    entry["fold"],
                    str(entry["scored"]),
                    str(
                        entry["regimes"][ONLINE_KEY]["transitions"][ALL_TRANSITIONS][
                            "transitions"
                        ]
                    ),
                    *(
                        _number(
                            e["comparisons"]["balanced_accuracy"]["differences"].get(
                                home
                            ),
                            True,
                        )
                        for e in estimands
                    ),
                ]
                for home, entry in households.items()
            ],
        ),
    ]
    if not smoothers:  # pragma: no cover - the protocol declares at least one lag
        raise ValueError("the record compares no smoother")
    return "\n".join(lines).rstrip("\n") + "\n"
