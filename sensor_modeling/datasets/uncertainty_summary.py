"""The Phase 4 uncertainty-diagnostics page, generated entirely from its record.

:func:`render_page` writes the whole documentation page from the published
experiment record: the protocol as the record declares it, every curve summary,
every paired comparison, the rejection bias by state, and the decisions. Nothing
on the page is typed by hand, so the page cannot disagree with the record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .uncertainty_experiment import COMPARATOR, DECISIONS, RESULT_SCHEMA, SIGNALS

PROTOCOL_FILE = "artifacts/phase4/uncertainty_protocol.json"
RECORD_FILE = "artifacts/phase4/phase4-uncertainty-diagnostics.json"
FIGURE_DIR = "figures"

#: Each estimand's label, in the order the page lists them.
ESTIMAND_LABELS = {
    "aurc": "AURC, the primary estimand",
    "balanced_accuracy_area": "Balanced accuracy over the grid",
    "at_inspection": "At each inspection level",
    "state_coverage": "Retained share of each state",
    "pairing": "Pairing",
}


def _sentence(text: str) -> str:
    """*text* with its first letter raised, and nothing else changed."""
    return text[:1].upper() + text[1:]


def _n(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _signed(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}"


def _interval(low: Any, high: Any, digits: int = 3) -> str:
    if low is None or high is None:
        return "—"
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def _signed_interval(low: Any, high: Any, digits: int = 3) -> str:
    if low is None or high is None:
        return "—"
    return f"[{low:+.{digits}f}, {high:+.{digits}f}]"


def _cell(text: str) -> str:
    """Table cell text, with the pipe that would end the cell escaped."""
    return text.replace("|", "\\|")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(_cell(h) for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _comparison(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Mean difference, its interval, households, and the verdict."""
    comparison = item["comparison"]
    if comparison is None:
        return "—", "—", "0", item["verdict"]
    mean = comparison["mean"]
    interval = mean["interval"]
    return (
        _signed(mean["estimate"]),
        (
            _signed_interval(interval["low"], interval["high"])
            if interval is not None
            else "—"
        ),
        str(comparison["n"]),
        item["verdict"],
    )


def _index(grid: Sequence[float], level: float) -> int:
    return min(range(len(grid)), key=lambda k: abs(grid[k] - level))


def render_page(payload: Mapping[str, Any]) -> str:
    """The whole documentation page, from the record alone."""
    results = payload["results"]
    if results.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    config = payload["configuration"]
    section = payload["selective_prediction"]
    grid = section["coverages"]
    inspection = config["evaluation"]["inspection"]
    levels = [f"{c:g}" for c in inspection]
    compared = config["compared"]
    structural = config["structural_signals"]
    difficult = results["difficult_minority_states"]
    key = results["key_question"]
    minimal = config["minimal_differences"]
    lines: list[str] = []
    add = lines.append

    add("# Phase 4: richer uncertainty diagnostics against confidence")
    add("")
    add(
        "Paper 1 found that confidence, entropy, margin, evidence strength and "
        "information gain fail as standalone uncertainty signals. Phase 4 built "
        "three structural diagnostics: model-structure disagreement, "
        "evidence-channel disagreement and predictive mismatch. This "
        "pre-specified experiment compares them with posterior confidence and "
        "entropy as signals for rejecting the same predictions, on the "
        "development households."
    )
    add("")
    add(
        "This page is generated entirely from the experiment record, "
        f"`{RECORD_FILE}`, by "
        "`sensor_modeling.datasets.uncertainty_summary.render_page`, and the "
        "figures by `sensor_modeling.datasets.uncertainty_figures.draw_figures`. "
        "A test checks that the committed page is exactly that rendering."
    )
    add("")
    add("**Status: pre-specified, development panel.**")
    add("")
    add(
        "- **Protocol.** Every signal, estimand, minimal difference and decision "
        "rule was frozen before any household was scored."
    )
    add(
        "- **Not a held-out claim.** The development homes have been inspected in "
        "earlier work. The frozen external cohort is not touched."
    )
    add(f"- **Thresholds.** {_sentence(config['thresholds'])}.")
    add("")

    # ------------------------------------------------------------------
    add("## The answer")
    add("")
    add(f"> {_sentence(key['question'])}?")
    add("")
    answer = f"**{_sentence(key['answer'])}.** " + (
        "Materially better: "
        + ", ".join(f"`{s}`" for s in key["materially_better"])
        + "."
        if key["materially_better"]
        else "No structural diagnostic is materially better than confidence "
        "by the declared rule."
    )
    add(answer)
    add("")
    rows = []
    for name in compared:
        entry = results["comparisons"][name]
        mean, interval, n, verdict = _comparison(entry["aurc"])
        guard = entry["guard"]
        rows.append(
            [
                f"`{name}`" + (" (structural)" if name in structural else ""),
                mean,
                interval,
                verdict,
                "holds" if guard["holds"] else f"fails ({len(guard['failures'])})",
                f"**{entry['decision']}**",
            ]
        )
    lines += _table(
        [
            "Signal",
            "AURC: confidence − signal",
            "95% interval",
            "Verdict",
            "Minority-state guard",
            "Decision",
        ],
        rows,
    )
    add("")
    add(
        f"The AURC difference is paired by household, and each household counts "
        f"once. A positive difference favours the signal. The minimal difference "
        f"is {_n(minimal['aurc'], 2)}."
    )
    add("")

    # ------------------------------------------------------------------
    add("## Protocol")
    add("")
    households = config["households"]
    add(f"The protocol is `{PROTOCOL_FILE}`, SHA-256 `{config['protocol_sha256']}`.")
    add("")
    add(
        f"- **Households.** {len(results['households'])} development homes in the "
        f"frozen folds of `{households['splits_file']}`. Each home is held out "
        "once and scored by models fitted on its fold's training homes."
    )
    predictions = config["predictions"]
    add(f"- **Predictions.** `{predictions['model']}`: {predictions['description']}.")
    add(
        f"- **Coverage grid.** {_n(grid[0], 2)} to {_n(grid[-1], 2)} in "
        f"{len(grid)} levels; inspection levels {', '.join(levels)}."
    )
    boot = config["bootstrap"]
    curve = config["evaluation"]["curve_bootstrap"]
    add(
        f"- **Uncertainty.** Households, never timestamps, are resampled: "
        f"{curve['resamples']:,} resamples for the pooled curves, "
        f"{boot['resamples']:,} for the paired comparisons, "
        f"{int(boot['confidence'] * 100)}% percentile intervals, seed "
        f"{boot['seed']}."
    )
    add("")
    add("### Signals")
    add("")
    lines += _table(
        ["Signal", "Direction", "Missing values", "Definition"],
        [
            [
                f"`{name}`",
                f"`{config['signals'][name]['direction']}`",
                f"`{config['signals'][name]['missing']}`",
                config["signals"][name]["definition"],
            ]
            for name in SIGNALS
        ],
    )
    add("")
    add(
        "The structural ensemble is "
        + ", ".join(f"`{s['name']}`" for s in config["ensemble"]["specifications"])
        + f": {config['ensemble']['reason']}."
    )
    add("")
    add("### Estimands and decision")
    add("")
    for name, label in ESTIMAND_LABELS.items():
        add(f"- **{label}.** {_sentence(config['estimands'][name])}.")
    add("")
    lines += _table(
        ["Quantity", "Minimal difference"],
        [[f"`{k}`", _n(minimal[k], 2)] for k in sorted(minimal)],
    )
    add("")
    add(f"- **Verdicts.** {_sentence(config['verdicts'])}.")
    rule = config["difficult_minority_states"]["rule"]
    add(f"- **Difficult minority states.** {_sentence(rule)}.")
    add(f"- **Guard.** {_sentence(config['rejection_bias']['guard'])}.")
    add(f"- **Absolute bias.** {_sentence(config['rejection_bias']['absolute'])}.")
    for decision in DECISIONS:
        add(
            f"- **{_sentence(decision)}.** "
            f"{_sentence(config['decision'][decision])}."
        )
    add(f"- **The key question.** {_sentence(config['decision']['key_question'])}.")
    add("")

    # ------------------------------------------------------------------
    add("## Selective-risk curves")
    add("")
    add(
        f"![Selective error against coverage]({FIGURE_DIR}/phase4-uncertainty-error.svg)"
    )
    add("")
    add(
        f"![Selective balanced accuracy against coverage]"
        f"({FIGURE_DIR}/phase4-uncertainty-balanced-accuracy.svg)"
    )
    add("")
    add(
        f"![Calibration error of retained predictions against coverage]"
        f"({FIGURE_DIR}/phase4-uncertainty-calibration.svg)"
    )
    add("")
    random = section["reference"]["random"]
    oracle = section["reference"]["oracle"]
    add(
        f"The panel pools {sum(h['windows'] for h in section['households'].values()):,} "
        f"scored windows. At full coverage the reference's error is "
        f"{_n(random['error'])}, its balanced accuracy "
        f"{_n(random['balanced_accuracy'])} and its calibration error "
        f"{_n(random['calibration_error'])}. Random rejection keeps these at every "
        f"coverage. The oracle's AURC is {_n(oracle['aurc'])}."
    )
    add("")
    add("### Curve summaries, pooled over the panel")
    add("")
    rows = []
    for name in SIGNALS:
        summary = section["signals"][name]["summary"]
        rows.append(
            [
                f"`{name}`",
                _n(summary["aurc"]["estimate"]),
                _interval(summary["aurc"]["low"], summary["aurc"]["high"]),
                _n(summary["excess_aurc"]["estimate"]),
                _n(summary["gain"]["estimate"]),
                _interval(summary["gain"]["low"], summary["gain"]["high"]),
            ]
        )
    lines += _table(
        [
            "Signal",
            "AURC",
            "95% interval",
            "Excess over oracle",
            "Gain",
            "95% interval",
        ],
        rows,
    )
    add("")
    add(
        f"Random rejection's AURC is {_n(random['risk'])}. The gain is 1 for the "
        "oracle and 0 for random rejection."
    )
    add("")
    for metric, title in (
        ("error", "Selective error"),
        ("balanced_accuracy", "Selective balanced accuracy"),
        ("calibration_error", "Calibration error of retained predictions"),
    ):
        add(f"### {title}, pooled, at the inspection levels")
        add("")
        rows = []
        for name in SIGNALS:
            pooled = section["signals"][name]["pooled"]
            cells = [f"`{name}`"]
            for level in inspection:
                k = _index(grid, level)
                value = _n(pooled[metric][k])
                intervals = pooled["intervals"].get(metric)
                if intervals is not None and intervals["low"] is not None:
                    value += " " + _interval(intervals["low"][k], intervals["high"][k])
                cells.append(value)
            rows.append(cells)
        lines += _table(["Signal", *[f"coverage {c}" for c in levels]], rows)
        add("")

    # ------------------------------------------------------------------
    add("## Household-level paired differences")
    add("")
    add(
        "Each household's own curve retains the given fraction of its own "
        "windows. Differences are oriented so that a positive value favours the "
        "signal over confidence."
    )
    add("")
    rows = []
    for name in compared:
        entry = results["comparisons"][name]
        for label, item in (
            ("AURC", entry["aurc"]),
            ("balanced accuracy over the grid", entry["balanced_accuracy_area"]),
        ):
            mean, interval, n, verdict = _comparison(item)
            rows.append([f"`{name}`", label, mean, interval, n, verdict])
        for level in levels:
            for metric, item in entry["at"][level].items():
                mean, interval, n, verdict = _comparison(item)
                rows.append(
                    [
                        f"`{name}`",
                        f"{metric.replace('_', ' ')} at {level}",
                        mean,
                        interval,
                        n,
                        verdict,
                    ]
                )
    lines += _table(
        [
            "Signal",
            "Quantity",
            "Mean difference",
            "95% interval",
            "Households",
            "Verdict",
        ],
        rows,
    )
    add("")

    # ------------------------------------------------------------------
    add("## Rejection bias by behavioural state")
    add("")
    add(
        f"![Retained share of each difficult minority state over coverage]"
        f"({FIGURE_DIR}/phase4-uncertainty-retention.svg)"
    )
    add("")
    add("### The states")
    add("")
    lines += _table(
        ["State", "Share of scored windows", "Reference recall", "Difficult minority"],
        [
            [f"`{state}`", _n(e["share"]), _n(e["recall"]), _n(e["difficult_minority"])]
            for state, e in results["states"].items()
        ],
    )
    add("")
    add(
        "Difficult minority states: "
        + (", ".join(f"`{s}`" for s in difficult) if difficult else "none")
        + "."
    )
    add("")
    bias = results["rejection_bias"]
    states = list(results["states"])
    for level in levels:
        add(f"### Retained share over coverage, pooled, at {level}")
        add("")
        add(
            "1 is proportional rejection; below 1, the state is rejected more than "
            "the panel."
        )
        add("")
        rows = [
            [f"`{name}`", *[_n(bias[name][level][s]["ratio"], 2) for s in states]]
            for name in SIGNALS
        ]
        lines += _table(["Signal", *[f"`{s}`" for s in states]], rows)
        add("")
    add("### Retained share against confidence, paired by household")
    add("")
    add(
        "Signal minus confidence: negative when the signal rejects the state more. "
        f"The minimal difference is {_n(minimal['state_coverage'], 2)}."
    )
    add("")
    rows = []
    for name in compared:
        for level in levels:
            for state in states:
                item = results["comparisons"][name]["state_coverage"][level][state]
                mean, interval, n, verdict = _comparison(item)
                marker = " (difficult minority)" if state in difficult else ""
                rows.append(
                    [
                        f"`{name}`",
                        level,
                        f"`{state}`{marker}",
                        mean,
                        interval,
                        n,
                        verdict,
                    ]
                )
    lines += _table(
        [
            "Signal",
            "Coverage",
            "State",
            "Mean difference",
            "95% interval",
            "Households",
            "Verdict",
        ],
        rows,
    )
    add("")

    # ------------------------------------------------------------------
    add("## Checks")
    add("")
    reproduction = results["reproduction"]["published"]
    if reproduction is not None:
        add(
            f"- **Reproduction.** At full coverage, every household's accuracy, "
            f"balanced accuracy and calibration error differ from the "
            f"`{reproduction['cell']}` cell of `{reproduction['experiment']}` "
            f"(commit `{reproduction['git_commit'][:7]}`) by at most "
            f"{reproduction['max_abs_difference']:.2e}, over "
            f"{reproduction['households']} households."
        )
    for name in SIGNALS:
        undefined = results["signals"][name]["undefined_windows"]
        if undefined:
            add(
                f"- **Undefined values.** `{name}` is undefined in {undefined:,} "
                f"windows, ranked by its missing policy, "
                f"`{results['signals'][name]['missing']}`."
            )
    add(
        f"- **Comparator.** Every signal is compared with `{COMPARATOR}` on the "
        "same predictions and the same household resamples."
    )
    add("")

    add("## What this does not show")
    add("")
    add(
        "- **No threshold.** No coverage level or signal value is chosen for "
        "deployment."
    )
    add(
        "- **Development panel only.** A held-out claim needs the frozen external "
        "cohort."
    )
    add(
        "- **One reference model.** Every signal ranks the predictions of the "
        f"`{predictions['model']}` recursion."
    )
    add(
        "- **Signals as declared.** No signal was added, tuned or combined after "
        "the results were seen."
    )
    add("")
    return "\n".join(lines)
