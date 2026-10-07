"""The page for the measurement of the baseline's thresholds on synthetic days.

:func:`render_page` builds ``docs/THRESHOLD_CALIBRATION_NULL.md`` from the
record alone. Every number and every sentence that states a finding is
computed from the record, so the page can be regenerated from it and a test
checks that the committed page is exactly that rendering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .threshold_null import (
    CALIBRATED,
    DEFAULT,
    NONE,
    RAMP2,
    REFERENCES,
    RESULT_SCHEMA,
    SHAPES,
    STEP2,
    STEP3,
    WEEKEND,
    WEEKEND_STEP2,
)

NULL_PAGE = "THRESHOLD_CALIBRATION_NULL.md"
FIGURE_DIR = "figures"
RECORD_PATH = "artifacts/threshold_calibration/threshold-null.json"

_SHAPE_TITLES = {
    NONE: "Nothing added",
    WEEKEND: "A weekly rhythm",
    STEP2: "A step of 2 SD",
    STEP3: "A step of 3 SD",
    RAMP2: "A ramp of 2 SD",
    WEEKEND_STEP2: "A rhythm and a step of 2 SD",
}
_CHANGES = (STEP2, STEP3, RAMP2, WEEKEND_STEP2)
_SETTING_TITLES = {
    "plain": "Days with no weekly rhythm",
    "weekly_rhythm": "Days with a weekly rhythm",
}
_VERDICT_TITLES = {
    "ordinary": "Ordinary",
    "temporary_disturbance": "Temporary disturbance",
    "persistent_change": "Persistent change",
    "abrupt_change": "Abrupt change",
    "gradual_drift": "Gradual drift",
}


def _pct(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{100.0 * value:.{digits}f}%"


def _rate(entry: Mapping[str, Any], digits: int = 2) -> str:
    """A share of days with its standard error in brackets."""
    if entry.get("rate") is None:
        return "–"
    return f"{_pct(entry['rate'], digits)} ({100.0 * entry['se']:.{digits}f})"


def _share(entry: Mapping[str, Any]) -> str:
    """A share of series with its standard error in brackets."""
    return f"{_pct(entry['share'], 1)} ({100.0 * entry['se']:.1f})"


def _points(entry: Mapping[str, Any]) -> str:
    """A paired difference of two shares, in points, with its standard error."""
    return f"{100.0 * entry['estimate']:+.1f} ({100.0 * entry['se']:.1f})"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _figure(name: str, caption: str) -> str:
    return f"![{caption}]({FIGURE_DIR}/threshold-null-{name}.svg)"


def _scales(configuration: Mapping[str, Any]) -> list[str]:
    return [f"{scale:g}" for scale in configuration["scales"]]


def _thresholds(configuration: Mapping[str, Any], scale: str) -> str:
    baseline = configuration["baseline"]
    return (
        f"{baseline['deviation_threshold'] * float(scale):.3g} and "
        f"{baseline['trend_threshold'] * float(scale):.3g}"
    )


def dominated(operating: Mapping[str, Any], shape: str) -> list[str]:
    """The default's scales that some calibrated scale is at least as good as.

    A calibrated scale is at least as good when it reports a change no more
    often with no change added and finds the shape's change no less often.
    For the shape with a weekly rhythm, no change added is the rhythm alone.
    The comparison is of the estimates and takes no account of their errors.
    """
    quiet = operating[WEEKEND if shape == WEEKEND_STEP2 else NONE]
    moved = operating[shape]

    def point(reference: str, scale: str) -> tuple[float, float]:
        return (
            quiet[reference][scale]["found"]["share"],
            moved[reference][scale]["found"]["share"],
        )

    found = []
    for scale in quiet[DEFAULT]:
        false, hit = point(DEFAULT, scale)
        if any(
            point(CALIBRATED, other)[0] <= false and point(CALIBRATED, other)[1] >= hit
            for other in quiet[CALIBRATED]
        ):
            found.append(scale)
    return found


def _head(payload: Mapping[str, Any]) -> list[str]:
    configuration, environment = payload["configuration"], payload["environment"]
    dirty = environment.get("git_dirty") != "false"
    return [
        "# The baseline's thresholds on days with nothing in them",
        "",
        "The personal baseline calls a day deviating when it lies three robust "
        "standard deviations from its reference. This page measures how often "
        "that happens when nothing has changed, for the default reference and "
        "for the opt-in calibrated one, `BaselineConfig.calibrated`, and what "
        "each finds when something has. It is generated from the record, "
        f"`{RECORD_PATH}`, by "
        "`sensor_modeling.datasets.threshold_null_summary.render_page`, and a "
        "test checks that the committed page is exactly that rendering.",
        "",
        "**The values are synthetic.** Each series is "
        f"{configuration['days']} independent draws from one Gaussian, one a "
        "day, given to the baseline directly. No home is simulated and no "
        "state is inferred. A real day is not an independent Gaussian draw, so "
        "this says what each reference does when its own assumptions hold, and "
        "nothing about a home.",
        "",
        f"- **Run.** Commit `{environment.get('git_commit', 'unknown')[:7]}`, "
        + ("with uncommitted changes" if dirty else "with no uncommitted change")
        + f"; recorded {payload['recorded_at']}.",
        f"- **Series.** {configuration['rate_series']:,} behind the shares of "
        f"days and {configuration['series']:,} behind the shares of series, all "
        f"derived from root `{configuration['seed_root']}`. Every reference, "
        "shape and threshold is run on the same series.",
        "- **Standing.** A measurement, not a test. "
        + " ".join(
            text[0].upper() + text[1:] + "."
            for text in configuration["what_was_seen_before"]
        ),
        "- **Standard errors** are in brackets, in the unit of the figure "
        "beside them. A share of days treats the series as the unit, since the "
        "days of one series share a history.",
    ]


def _sizes(payload: Mapping[str, Any]) -> list[str]:
    results, configuration = payload["results"], payload["configuration"]
    baseline = configuration["baseline"]
    threshold = f"{baseline['deviation_threshold']:g}"
    cells = {ref: results["rates"][NONE][ref] for ref in REFERENCES}
    stated = results["nominal"][threshold]
    rows = [
        [
            f"All the earlier days, {baseline['min_samples']} to "
            f"{baseline['weekday_min_samples'] * 7 - 1}",
            *(_rate(cells[ref]["phase"]["pooled"][threshold]) for ref in REFERENCES),
        ]
    ]
    sizes = sorted(cells[DEFAULT]["by_weekday_size"], key=int)
    rows += [
        [
            f"{size} of the same weekday",
            *(
                _rate(cells[ref]["by_weekday_size"][size][threshold])
                for ref in REFERENCES
            ),
        ]
        for size in sizes
    ]
    rows.append(
        [
            "**Every evaluable day**",
            *(
                f"**{_rate(cells[ref]['phase']['all'][threshold])}**"
                for ref in REFERENCES
            ),
        ]
    )
    overall = {ref: cells[ref]["phase"]["all"][threshold]["rate"] for ref in REFERENCES}
    first, last = sizes[0], sizes[-1]
    small = cells[DEFAULT]["by_weekday_size"][first][threshold]["rate"]
    large = cells[DEFAULT]["by_weekday_size"][last][threshold]["rate"]
    return [
        "",
        f"## What a threshold of {threshold} means",
        "",
        f"A Gaussian value lies {threshold} standard deviations or more from "
        f"its mean {_pct(stated)} of the time. That is what the threshold "
        "states. The table gives how often a day reaches it, by the days its "
        "reference rests on.",
        "",
        _figure(
            "sizes",
            "Share of days past the declared threshold, by the days behind the "
            "reference",
        ),
        "",
        *_table(["Days behind the centre", "Default", "Calibrated"], rows),
        "",
        f"- **The default reference passes its threshold on {_pct(overall[DEFAULT])} "
        f"of days**, {overall[DEFAULT] / stated:.0f} times what it states. With "
        f"{first} days of the weekday behind it the share is {_pct(small, 1)}, "
        f"and with {last} it is still {_pct(large, 1)}.",
        f"- **The calibrated reference passes it on {_pct(overall[CALIBRATED])}.** "
        "Its centre rests on the same few days. Its scale rests on all of them.",
        "- **The scale is where the default goes wrong.** The spread of four "
        "values is often far too small, and the day is then divided by it.",
    ]


def _other_thresholds(payload: Mapping[str, Any]) -> list[str]:
    results, configuration = payload["results"], payload["configuration"]
    cells = {ref: results["rates"][NONE][ref] for ref in REFERENCES}
    thresholds = [f"{t:g}" for t in configuration["rate_thresholds"]]
    rows = [
        [
            threshold,
            _pct(results["nominal"][threshold]),
            _rate(cells[DEFAULT]["phase"]["all"][threshold]),
            _rate(cells[CALIBRATED]["phase"]["all"][threshold]),
            _rate(cells[CALIBRATED]["phase"]["pooled"][threshold]),
            _rate(cells[CALIBRATED]["phase"]["weekday_aware"][threshold]),
        ]
        for threshold in thresholds
    ]
    weeks = sorted(cells[CALIBRATED]["by_week"], key=int)
    by_week = [
        [
            week,
            *(_rate(cells[CALIBRATED]["by_week"][week][t]) for t in thresholds),
        ]
        for week in weeks
    ]
    ratios = [
        cells[CALIBRATED]["by_week"][week][t]["rate"] / results["nominal"][t]
        for week in weeks
        for t in thresholds
    ]
    return [
        "",
        "## The calibrated reference at other thresholds",
        "",
        "A threshold that means what it says should do so wherever it is set. "
        "The share of evaluable days at or past each threshold:",
        "",
        *_table(
            [
                "Threshold",
                "Stated",
                "Default",
                "Calibrated",
                "Calibrated, before the reference is weekday-aware",
                "Calibrated, once it is",
            ],
            rows,
        ),
        "",
        "The calibrated reference by week of the series, since the days behind "
        "its scale grow from week to week:",
        "",
        *_table(["Week", *(f"At {t}" for t in thresholds)], by_week),
        "",
        f"Over the {len(ratios)} cells of that table the share is between "
        f"{min(ratios):.2f} and {max(ratios):.2f} times what the threshold "
        "states.",
    ]


def _verdicts(payload: Mapping[str, Any]) -> list[str]:
    results = payload["results"]
    rows = []
    for kind, title in _VERDICT_TITLES.items():
        rows.append(
            [
                title,
                *(
                    _rate(results["rates"][shape][ref]["verdicts"][kind], 3)
                    for shape in (NONE, WEEKEND)
                    for ref in REFERENCES
                ),
            ]
        )
    rows.append(
        [
            "**A change of any kind**",
            *(
                f"**{_rate(results['rates'][shape][ref]['change_verdicts'], 3)}**"
                for shape in (NONE, WEEKEND)
                for ref in REFERENCES
            ),
        ]
    )
    plain = results["rates"][NONE][DEFAULT]
    drift = plain["verdicts"]["gradual_drift"]["rate"]
    change = plain["change_verdicts"]["rate"]
    rhythm = {
        ref: results["rates"][WEEKEND][ref]["verdicts"]["gradual_drift"]["rate"]
        for ref in REFERENCES
    }
    flat = {
        ref: results["rates"][NONE][ref]["verdicts"]["gradual_drift"]["rate"]
        for ref in REFERENCES
    }
    return [
        "",
        "## What is reported when nothing changed",
        "",
        "The baseline's verdicts on the same days, at the declared thresholds. "
        "A persistent change, an abrupt change and a gradual drift are the "
        "three the pipeline can raise a behavioural alert about. The second "
        "pair of columns adds a weekly rhythm to the days, which is not a "
        "change.",
        "",
        *_table(
            [
                "Verdict",
                "Default",
                "Calibrated",
                "Default, weekly rhythm",
                "Calibrated, weekly rhythm",
            ],
            rows,
        ),
        "",
        f"- **The default reports a change on {_pct(change, 3)} of days with "
        f"nothing in them**, and {drift / change:.0%} of those reports are of a "
        "gradual drift. The drift's movement is divided by the same scale as "
        "the deviation, so a scale that is too small makes a trend out of "
        "noise.",
        "- **With a weekly rhythm added**, gradual drifts go from "
        f"{_pct(flat[DEFAULT], 3)} to {_pct(rhythm[DEFAULT], 3)} of days for the "
        f"default and from {_pct(flat[CALIBRATED], 3)} to "
        f"{_pct(rhythm[CALIBRATED], 3)} for the calibrated one. The default "
        "fits its trend to the days as they are. The calibrated reference fits "
        "it to each day's distance from the centre of its own weekday, so once "
        "every weekday has a centre of its own a rhythm is no part of its "
        "trend.",
    ]


def _needed(payload: Mapping[str, Any]) -> list[str]:
    needed = payload["results"]["needed_threshold"]
    sizes = sorted(needed["by_size"], key=int)
    rows = [
        [
            size,
            _rate(needed["by_size"][size]["passes_the_declared_threshold"], 2),
            f"{needed['by_size'][size]['needed_threshold']['value']:.1f} "
            f"[{needed['by_size'][size]['needed_threshold']['low']:.1f}, "
            f"{needed['by_size'][size]['needed_threshold']['high']:.1f}]",
        ]
        for size in sizes
    ]
    first, last = needed["by_size"][sizes[0]], needed["by_size"][sizes[-1]]
    return [
        "",
        "## The threshold a reference of a few days would need",
        "",
        "The other way to make the threshold honest is to leave the reference "
        "as it is and raise the threshold with the sample size. The table "
        "gives, for a median and MAD of that many other days, how often a "
        f"Gaussian day passes {needed['declared_threshold']:g}, and the value "
        f"it passes {_pct(needed['nominal'])} of the time, from "
        f"{needed['draws']:,} draws for each size. The interval is at 95%.",
        "",
        *_table(
            [
                "Days behind the reference",
                f"Passes {needed['declared_threshold']:g}",
                "Threshold needed",
            ],
            rows,
        ),
        "",
        f"With {sizes[0]} days the threshold would have to be "
        f"{first['needed_threshold']['value']:.0f}, and with {sizes[-1]} it is "
        f"still {last['needed_threshold']['value']:.1f}. A threshold that high "
        "is passed only by a day far outside anything the others showed, so a "
        "weekday reference would say nothing for its first months. This is why "
        "the calibrated reference changes the scale and not the threshold.",
    ]


def _operating(payload: Mapping[str, Any]) -> list[str]:
    results, configuration = payload["results"], payload["configuration"]
    operating = results["operating"]
    scales = _scales(configuration)
    lines = [
        "",
        "## Finding a change at the same rate of false reports",
        "",
        "Two rules that do not fire equally often cannot be compared at one "
        "threshold. Each reference is run here at multiples of its declared "
        "thresholds, both multiplied together, on the same series: with "
        "nothing added, with a step, and with a slow ramp. A series counts "
        f"when a change verdict falls on a day from {configuration['step_day']} "
        f"to {configuration['step_day'] + configuration['window_days'] - 1}, the "
        "three weeks from the step.",
        "",
        _figure(
            "operating",
            "Share of series in which a change is found against the share with "
            "a false report, for both references",
        ),
    ]
    for reference in REFERENCES:
        rows = [
            [
                scale,
                _thresholds(configuration, scale),
                *(
                    _share(operating[shape][reference][scale]["found"])
                    for shape in SHAPES
                ),
            ]
            for scale in scales
        ]
        lines += [
            "",
            f"**The {reference} reference.** Share of series with a change "
            "verdict in the window:",
            "",
            *_table(
                ["Multiple", "Thresholds", *(_SHAPE_TITLES[s] for s in SHAPES)], rows
            ),
        ]
    counts = {shape: dominated(operating, shape) for shape in _CHANGES}
    lines += [
        "",
        "Read the two tables against each other. For each multiple the default "
        "was run at, is there a calibrated multiple that reports a change no "
        "more often when no change was added, and finds the change no less "
        "often? Comparing the estimates, without their errors:",
        "",
        *(
            f"- **{_SHAPE_TITLES[shape]}:** for {len(found)} of the "
            f"{len(scales)} multiples."
            for shape, found in counts.items()
        ),
    ]
    return lines


def _matching(payload: Mapping[str, Any]) -> list[str]:
    results, configuration = payload["results"], payload["configuration"]
    operating, matching = results["operating"], results["matching"]
    rule = configuration["matching"]
    lines = [
        "",
        "## Where the calibrated reference matches the default",
        "",
        "At the declared thresholds the two references are far apart, so the "
        "calibrated one is also read where it matches the default as "
        "declared, by a rule fixed before the measurement:",
        "",
        f"- **The same detection:** {rule['same_detection']}.",
        f"- **The same false reports:** {rule['same_false_reports']}.",
        "",
        f"{_sentence(rule['why'])}",
    ]
    for name, entry in matching.items():
        quiet, moved = entry["nothing_changed"], entry["a_step"]
        scales = list(entry["calibrated_minus_default"])
        rows = [
            [
                "Default, as declared",
                _thresholds(configuration, "1"),
                *(_share(operating[shape][DEFAULT]["1"]["found"]) for shape in SHAPES),
            ]
        ]
        for scale in scales:
            rows.append(
                [
                    f"Calibrated at {scale}",
                    _thresholds(configuration, scale),
                    *(
                        _share(operating[shape][CALIBRATED][scale]["found"])
                        for shape in SHAPES
                    ),
                ]
            )
            rows.append(
                [
                    "minus the default, in points",
                    "",
                    *(
                        _points(entry["calibrated_minus_default"][scale][shape])
                        for shape in SHAPES
                    ),
                ]
            )
        kept, same = f"{entry['same_detection']:g}", f"{entry['same_false_reports']:g}"
        default = {
            shape: operating[shape][DEFAULT]["1"]["found"]["share"] for shape in SHAPES
        }

        def mine(scale: str, shape: str) -> float:
            share: float = operating[shape][CALIBRATED][scale]["found"]["share"]
            return share

        lines += [
            "",
            f"### {_SETTING_TITLES[name]}",
            "",
            f"No change added is `{quiet}`, and the step is `{moved}`.",
            "",
            *_table(
                ["Reference", "Thresholds", *(_SHAPE_TITLES[s] for s in SHAPES)], rows
            ),
            "",
            "- **The same detection is at "
            + (
                kept
                if entry["same_detection_was_reached"]
                else f"none; {kept} is the smallest"
            )
            + ".** There the calibrated reference finds the step in "
            f"{_pct(mine(kept, moved), 1)} of series against the default's "
            f"{_pct(default[moved], 1)}, and reports falsely in "
            f"{_pct(mine(kept, quiet), 1)} against {_pct(default[quiet], 1)}.",
            "- **The same false reports are at "
            + (
                same
                if entry["same_false_reports_was_reached"]
                else f"none; {same} is the largest"
            )
            + ".** There it reports falsely in "
            f"{_pct(mine(same, quiet), 1)} of series against "
            f"{_pct(default[quiet], 1)}, and finds the step in "
            f"{_pct(mine(same, moved), 1)} against {_pct(default[moved], 1)}.",
            "- **At the declared thresholds** it finds the step in "
            f"{_pct(mine('1', moved), 1)} of series and reports falsely in "
            f"{_pct(mine('1', quiet), 1)}.",
        ]
    return lines


def _sentence(text: str) -> str:
    text = text.strip()
    text = text[0].upper() + text[1:]
    return text if text.endswith(".") else text + "."


def _limits(payload: Mapping[str, Any]) -> list[str]:
    baseline = payload["configuration"]["baseline"]
    return [
        "",
        "## What this does not show",
        "",
        "- **Nothing about a home.** A day's hours of sleep are not independent "
        "Gaussian draws: they have long tails, they depend on the day before, "
        "and the hours of a state that is seldom entered pile up at zero. The "
        "Student t behind the calibrated score is exact for none of that.",
        "- **Nothing about alerts.** Verdicts are counted. The alert policy, "
        "which grades a verdict and withholds a repeat, is not applied.",
        f"- **The floor on the scale is {baseline['min_scale']:g} of a standard "
        "deviation here.** In a home it is a quarter of an hour, whatever the "
        "spread of the feature, so it binds more or less often than it does on "
        "these series.",
        "- **Where the references match here is not where they match in a "
        "home.** The threshold-calibration protocol finds that on simulated "
        "homes of its own, and tests the result on others.",
    ]


def render_page(payload: Mapping[str, Any]) -> str:
    """Render the page from the record of the measurement."""
    if payload["results"].get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    lines = [
        *_head(payload),
        *_sizes(payload),
        *_other_thresholds(payload),
        *_verdicts(payload),
        *_needed(payload),
        *_operating(payload),
        *_matching(payload),
        *_limits(payload),
    ]
    return "\n".join(lines) + "\n"
