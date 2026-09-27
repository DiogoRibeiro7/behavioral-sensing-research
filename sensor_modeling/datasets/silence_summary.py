"""A concise Markdown summary of a correlated-silence diagnostic record, from it alone.

:func:`render_summary` reads nothing but the record as written, so the summary
can always be regenerated from the artifact and checked against it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import median
from typing import Any

from .gap_summary import _number, _table
from .silence_dependence import CRITERIA, RESULT_SCHEMA


def _estimate(estimate: Mapping[str, Any] | None) -> str:
    if not estimate:
        return "n/a"
    interval = estimate["interval"]
    bounds = (
        f" [{_number(interval['low'], True)}, {_number(interval['high'], True)}]"
        if interval
        else ""
    )
    return f"{_number(estimate['estimate'], True)}{bounds}"


def _factor(estimate: Mapping[str, Any] | None) -> str:
    """A mean log ratio as a ratio, with its interval."""
    if not estimate or estimate["estimate"] is None:
        return "n/a"
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


def _signed(value: float | None) -> str:
    return _number(value, True)


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{math.exp(value):.2f}"


def render_summary(payload: Mapping[str, Any], *, level: int = 1) -> str:
    """A concise Markdown summary of a correlated-silence diagnostic record.

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
    minimal = configuration["minimal_effects"]
    confidence = round(100 * bootstrap["confidence"])
    title, section = "#" * level, "#" * (level + 1)
    dirty = " (uncommitted changes)" if environment.get("git_dirty") == "true" else ""
    households = results["households"]
    estimands = {e["key"]: e for e in results["estimands"]}
    step_minutes = round(
        configuration["streams"]["information_set"]["step_seconds"] / 60
    )

    lines = [
        f"{title} Correlated silence: summary",
        "",
        f"Generated from the record `{payload['experiment']}`: protocol "
        f"`{configuration['protocol_sha256'][:12]}`, commit "
        f"`{str(environment.get('git_commit', 'unknown'))[:12]}`{dirty}. "
        f"Status: {results['status']}.",
        "",
        f"- {len(households)} households, each counted once. Streams are the "
        f"instrumented evidence channels in {step_minutes}-minute windows.",
        "- Every value is computed within a household. Across households, the "
        f"mean and median carry {confidence}% household bootstrap intervals from "
        f"{bootstrap['resamples']:,} resamples. Zero is independence, or a "
        "calibrated model for slopes.",
        f"- Quiet states: {', '.join(results['quiet_states'])}.",
        "- Minimal effects: "
        + ", ".join(f"{scale} {value:.3g}" for scale, value in sorted(minimal.items()))
        + ". A log ratio of log 1.25 is a ratio of 1.25.",
        "",
        f"{section} Pre-specified conclusion",
        "",
        *_table(
            ["Question", "Verdict"],
            [
                [
                    "Does independence materially inflate silence evidence? (D1)",
                    estimands["D1"]["verdict"],
                ],
                [
                    "Does overconfidence grow with the number of silent channels? (C1)",
                    estimands["C1"]["verdict"],
                ],
                ["The correlated-silence hypothesis", results["conclusion"]],
            ],
        ),
        f"Supported: {CRITERIA['supported']}. Weakened: {CRITERIA['weakened']}.",
        "",
        f"Verdicts: {CRITERIA['verdicts']}.",
        "",
        f"{section} Estimands",
        "",
        "Log ratios are also shown as ratios, where 1 is independence:",
        "",
        *_table(
            [
                "Key",
                "Role",
                "Quantity",
                "Mean",
                "Median",
                "As a ratio",
                "Households above / below zero",
                "Verdict",
            ],
            [
                [
                    e["key"],
                    e["role"],
                    e["quantity"],
                    _estimate(e["summary"].get("mean")),
                    _estimate(e["summary"].get("median")),
                    (
                        _factor(e["summary"].get("mean"))
                        if e["scale"] == "log_ratio"
                        else ""
                    ),
                    _count(e["summary"]),
                    e["verdict"],
                ]
                for e in results["estimands"]
            ],
        ),
        f"{section} Joint silence by state",
        "",
        "Mean log ratio of the observed probability that every channel is silent "
        "to each independence baseline, with the households where it is "
        "estimable. The difference is in probability:",
        "",
        *_table(
            [
                "State",
                "Households",
                "Against observed marginals",
                "Against Poisson streams",
                "Against the declared rates",
                "Difference",
            ],
            [
                [
                    state,
                    str(entry["log_joint_silence"]["summary"]["n"]),
                    f"{_estimate(entry['log_joint_silence']['summary'].get('mean'))}, "
                    f"{entry['log_joint_silence']['verdict']}",
                    _estimate(
                        entry["log_joint_silence_poisson"]["summary"].get("mean")
                    ),
                    _estimate(
                        entry["log_joint_silence_declared"]["summary"].get("mean")
                    ),
                    _estimate(entry["joint_silence_difference"]["summary"].get("mean")),
                ]
                for state, entry in (
                    (s, results["per_state"][s]) for s in results["states"]
                )
            ],
        ),
        f"{section} Pairwise dependence by state",
        "",
        "Mean across households of each household's median over channel pairs "
        "(log odds ratio of silence, log pair silence ratio), mean pairwise count "
        "correlation, and log dispersion of the number of silent channels:",
        "",
        *_table(
            [
                "State",
                "Log odds ratio of silence",
                "Log pair silence ratio",
                "Count correlation",
                "Log dispersion",
            ],
            [
                [
                    state,
                    *(
                        f"{_estimate(entry[q]['summary'].get('mean'))} "
                        f"(n = {entry[q]['summary']['n']})"
                        for q in (
                            "log_odds_silence",
                            "log_pair_silence",
                            "count_correlation",
                            "log_dispersion",
                        )
                    ),
                ]
                for state, entry in (
                    (s, results["per_state"][s]) for s in results["states"]
                )
            ],
        ),
        f"{section} Channel pairs in the quiet states",
        "",
        "Mean across households, each household averaged over its quiet states "
        "where the statistic is estimable:",
        "",
        *_table(
            [
                "Pair",
                "Log odds ratio of silence",
                "Households above / below zero",
                "Log pair silence ratio",
                "Households above / below zero",
            ],
            [
                [
                    pair.replace("|", " and "),
                    _estimate(s["log_odds_silence"].get("mean")),
                    _count(s["log_odds_silence"]),
                    _estimate(s["log_pair_silence"].get("mean")),
                    _count(s["log_pair_silence"]),
                ]
                for pair, s in results["per_pair"].items()
            ],
        ),
        f"{section} Households",
        "",
        "Each household's values. D1 and S1 are shown as inflation factors, D2 as "
        "a ratio; C1 and S8 are slopes of overconfidence:",
        "",
        *_table(
            [
                "Household",
                "Channels",
                "D1 inflation",
                "S1 declared inflation",
                "D2 joint silence",
                "C1 per silent channel",
                "S8 per quiet hour",
            ],
            [
                [
                    home,
                    str(len(entry["channels"])),
                    _ratio(entry["values"]["log_inflation"]),
                    _ratio(entry["values"]["log_inflation_declared"]),
                    _ratio(entry["values"]["log_joint_silence_quiet"]),
                    _signed(entry["values"]["gap_slope"]),
                    _signed(entry["values"]["accumulation_slope"]),
                ]
                for home, entry in households.items()
            ],
        ),
    ]

    association = results["association"]
    interval = association.get("interval")
    lines += [
        f"Across households, the Spearman correlation of D1 and C1 is "
        f"{_signed(association.get('spearman'))}"
        + (
            f" [{_signed(interval['low'])}, {_signed(interval['high'])}]"
            if interval
            else ""
        )
        + f", n = {association['n']}.",
        "",
        f"{section} Representative quiet periods",
        "",
        "One pre-specified quiet run per household and quiet state. Medians "
        "across households of the confidence predicted by the transition before "
        "the first window, and of the posterior confidence after windows 1, 6 "
        "and the last recorded:",
        "",
    ]
    rows = []
    for state in results["quiet_states"]:
        cases = [
            entry["cases"][state]["steps"]
            for entry in households.values()
            if state in entry["cases"]
        ]
        if not cases:
            rows.append([state, "0", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        last = len(cases[0])

        def middle(values: list[float]) -> str:
            return _number(median(values))

        rows.append(
            [
                state,
                str(len(cases)),
                middle([c[0]["predicted_confidence"] for c in cases]),
                middle([c[0]["confidence"] for c in cases]),
                middle([c[min(5, last - 1)]["confidence"] for c in cases]),
                middle([c[last - 1]["confidence"] for c in cases]),
                f"{median(c[0]['silent_channels'] for c in cases):g}",
            ]
        )
    lines += _table(
        [
            "State",
            "Households",
            "Predicted, window 1",
            "After window 1",
            "After window 6",
            "After the last",
            "Silent channels",
        ],
        rows,
    )
    return "\n".join(lines).rstrip("\n") + "\n"
