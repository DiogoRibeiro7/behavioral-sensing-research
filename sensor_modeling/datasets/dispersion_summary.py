"""A concise Markdown summary of a hurdle negative-binomial evaluation, from its record alone.

:func:`render_summary` reads nothing but the record as written, so the summary
can always be regenerated from the artifact and checked against it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .dispersion_experiment import (
    CRITERIA,
    ESTIMANDS,
    MODELS,
    RECURSION,
    RESULT_SCHEMA,
    SETTINGS,
)
from .gap_summary import _number, _table
from .history_summary import _estimate, _homes, _level

_METRICS = ("balanced_accuracy", "log_loss", "brier", "calibration_error")
_QUESTIONS = (
    ("home_active_recall", "1. Recovers home_active recall?"),
    ("calibration", "2. Preserves the calibration improvement?"),
    ("log_loss", "3. Improves log loss?"),
    ("balanced_accuracy", "4. Avoids degrading balanced accuracy?"),
    ("decision", "Decision"),
)
_CLASSES = (
    "poisson",
    "moderate",
    "extreme, low data",
    "extreme, flat",
    "extreme, supported",
)


def _judged(comparison: Mapping[str, Any], verdict: str) -> str:
    return f"{_estimate(comparison['mean'])}, {_homes(comparison)}, {verdict}"


def _alpha(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3g}"


def _dispersion_table(
    estimates: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[str]:
    """Each channel's dispersion per state, fold by fold."""
    folds = list(estimates)
    states: list[str] = []
    channels: list[str] = []
    lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for fold, rows in estimates.items():
        for row in rows:
            if row["state"] not in states:
                states.append(row["state"])
            if row["channel"] not in channels:
                channels.append(row["channel"])
            lookup[(fold, row["channel"], row["state"])] = row
    body = []
    for channel in sorted(channels):
        cells = []
        for state in states:
            parts = [
                (
                    _alpha(lookup[(f, channel, state)]["dispersion"])
                    if (f, channel, state) in lookup
                    else "n/a"
                )
                for f in folds
            ]
            cells.append(" / ".join(parts))
        body.append([channel, *cells])
    return _table(["Channel", *states], body)


def render_summary(payload: Mapping[str, Any], *, level: int = 1) -> str:
    """A concise Markdown summary of a hurdle negative-binomial evaluation record.

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
    by_key = {e["key"]: e for e in results["estimands"]}
    keys = [e.key for e in ESTIMANDS]
    questions = results["questions"]
    dispersion = results["dispersion"]
    extreme = dispersion["extreme"]
    thresholds = configuration["dispersion"]
    folds = list(dispersion["estimates"])

    lines = [
        f"{title} The hurdle negative binomial against the hurdle-Poisson: summary",
        "",
        f"Generated from the record `{payload['experiment']}`: protocol "
        f"`{configuration['protocol_sha256'][:12]}`, commit "
        f"`{str(environment.get('git_commit', 'unknown'))[:12]}`{dirty}. "
        f"Status: {results['status']}.",
        "",
        "- Inference regime: online filter; with current windows (`I0`) and the "
        "filter's recursion over every window (`R`).",
        f"- {len(households)} held-out households in "
        f"{len(configuration['households']['folds'])} cross-fitted folds, each "
        "scored once by channel models fitted without it.",
        "- Three channel observation models, `declared`, `hurdle` and "
        "`hurdle_nb`, with the same prior, transition, channels, windows and "
        "inference.",
        "- Differences are paired by household, with the mean and the median and "
        f"their {confidence}% household bootstrap intervals from "
        f"{bootstrap['resamples']:,} resamples. A positive value favours the first "
        "model.",
        "- Minimal important differences: "
        + ", ".join(f"{metric} {value:g}" for metric, value in sorted(minimal.items()))
        + ".",
        "",
        f"{section} Pre-specified questions",
        "",
        *_table(
            ["Question", *SETTINGS],
            [
                [label, *(questions[s][key] for s in SETTINGS)]
                for key, label in _QUESTIONS
            ],
        ),
        f"**Overall: {results['conclusion']}.**",
        "",
        f"Adopt: {CRITERIA['adopt']}. Trade-off: {CRITERIA['trade-off']}. Reject: "
        f"{CRITERIA['reject']}. Overall: {CRITERIA['overall']}.",
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
                "Setting",
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
                        _number(cell["metrics"][m]["median"])
                        for m in ("log_loss", "brier", "calibration_error")
                    ),
                ]
                for cell in results["cells"]
            ],
        ),
        f"{section} Estimands",
        "",
        "Mean differences, households improved and worsened, and verdicts. "
        "Positive values favour the first model, that is, higher accuracy or "
        "lower loss and calibration error:",
        "",
        *_table(
            ["Key", "Comparison", *_METRICS],
            [
                [
                    key,
                    f"{by_key[key]['model']} vs {by_key[key]['reference']}",
                    *(
                        _judged(
                            by_key[key]["comparisons"][m], by_key[key]["verdicts"][m]
                        )
                        for m in _METRICS
                    ),
                ]
                for key in keys
            ],
        ),
        f"{section} Per-state recall",
        "",
        "Mean differences, households improved and worsened, and verdicts:",
        "",
        *_table(
            ["State", "Rare", *keys],
            [
                [
                    state,
                    "yes" if state in results["rare_states"] else "no",
                    *(
                        _judged(
                            by_key[key]["per_state_recall"][state]["comparison"],
                            by_key[key]["per_state_recall"][state]["verdict"],
                        )
                        for key in keys
                    ),
                ]
                for state in results["states"]
            ],
        ),
        f"{section} Overconfidence along quiet runs",
        "",
        "In the recursion, the slope per hour of confidence minus correctness "
        "along runs of fully silent windows, within the predicted state. Lower "
        "is better, and a calibrated model has zero. Each model's household "
        "median, then mean with its interval:",
        "",
        *_table(
            ["Model", "Slope per hour"],
            [[model, _level(results["quiet_runs"][model])] for model in MODELS],
        ),
        "Paired differences in the recursion; a positive value means the first "
        "model's slope is lower:",
        "",
        *_table(
            ["Key", "Comparison", "Slope"],
            [
                [
                    key,
                    f"{by_key[key]['model']} vs {by_key[key]['reference']}",
                    _judged(
                        by_key[key]["quiet_runs"]["comparison"],
                        by_key[key]["quiet_runs"]["verdict"],
                    ),
                ]
                for key in keys
                if by_key[key]["setting"] == RECURSION
            ],
        ),
        f"{section} Estimated dispersion",
        "",
        "The negative binomial's dispersion `alpha` per channel and state, fitted "
        f"on each fold's training households ({' / '.join(folds)}). Zero is the "
        "Poisson; the bound is "
        f"{configuration['fitting']['max_dispersion']:g}:",
        "",
        *_dispersion_table(dispersion["estimates"]),
        f"Each estimate's class. Extreme is {thresholds['extreme']}; low data is "
        f"{thresholds['low_data']}; flat means {thresholds['flat']}. Channel and "
        "state estimates over both folds:",
        "",
        *_table(
            ["State", *_CLASSES],
            [
                [
                    state,
                    *(str(dispersion["by_state"][state].get(c, 0)) for c in _CLASSES),
                ]
                for state in results["states"]
                if state in dispersion["by_state"]
            ],
        ),
        f"Extreme estimates: {extreme['extreme']}, of which {extreme['low_data']} "
        f"low data, {extreme['flat']} flat and {extreme['supported']} supported. "
        f"**Do extreme estimates indicate insufficient data? {extreme['answer']}.**",
        "",
        *_table(
            [
                "Fold",
                "Channel",
                "State",
                "Active windows",
                "alpha",
                "Gain over Poisson",
                "Flatness",
                "Class",
            ],
            [
                [
                    fold,
                    row["channel"],
                    row["state"],
                    str(row["active_windows"]),
                    _alpha(row["dispersion"]),
                    _number(row["gain_over_poisson"]),
                    _number(row["flatness"]),
                    row["class"],
                ]
                for fold, rows in dispersion["estimates"].items()
                for row in rows
                if str(row["class"]).startswith("extreme")
            ],
        ),
        f"{section} Households",
        "",
        "Each household's change with the negative binomial against the "
        "hurdle-Poisson: balanced accuracy and log loss in each setting, and the "
        "quiet-run slope in the recursion. Positive favours the negative binomial:",
        "",
        *_table(
            [
                "Household",
                "Fold",
                "N0 balanced accuracy",
                "N0 log loss",
                "NR balanced accuracy",
                "NR log loss",
                "NR slope",
            ],
            [
                [
                    home,
                    entry["fold"],
                    *(
                        _number(
                            by_key[key]["comparisons"][metric]["differences"].get(home),
                            True,
                        )
                        for key in ("N0", "NR")
                        for metric in ("balanced_accuracy", "log_loss")
                    ),
                    _number(
                        by_key["NR"]["quiet_runs"]["comparison"]["differences"].get(
                            home
                        ),
                        True,
                    ),
                ]
                for home, entry in households.items()
            ],
        ),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"
