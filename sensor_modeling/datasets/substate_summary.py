"""The activity sub-state pages, generated from the frozen protocol and the record.

:func:`render_protocol` writes the protocol page from the frozen protocol file,
and :func:`render_page` the results page from the record. Every number on
either page comes from those files.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .combined_prior_summary import METRIC_TITLES, _difference, _level
from .predictive_summary import _ratio
from .sleep_gap_summary import _flatten, _n, _section, _sentence, _table
from .threshold_calibration_summary import _fixed

PROTOCOL_FILE = "artifacts/phase3/substates_protocol.json"
RECORD_FILE = "artifacts/phase3/phase3-activity-substates.json"
PROTOCOL_PAGE = "PHASE3_ACTIVITY_SUBSTATES_PROTOCOL.md"
RESULTS_PAGE = "PHASE3_ACTIVITY_SUBSTATES_RESULTS.md"
TITLE = "Phase 3.3 follow-up: activity sub-states in the recursion"


def render_protocol(payload: Mapping[str, Any]) -> str:
    """The protocol page, from the frozen protocol file's content."""
    lines = [
        f"# {TITLE}: the protocol",
        "",
        f"Generated entirely from the frozen file `{PROTOCOL_FILE}` by "
        "`sensor_modeling.datasets.substate_summary.render_protocol`.",
        "",
        f"**Status: {payload['status']}.**",
        "",
        f"- **Protocol digest.** `{payload['protocol_sha256']}`.",
        f"- **The question.** {_sentence(payload['question'])}",
        "- **This page reports no result.**",
        "",
        "## What had been seen",
        "",
        *[f"- {_sentence(item)}" for item in payload["inspected_before"]],
        "",
        "## The base protocols",
        "",
    ]
    for title, key in (
        ("Combined prior", "base_protocol"),
        ("Predictive checks", "predictive_protocol"),
    ):
        base = payload[key]
        lines += [
            f"- **{title}.** `{base['file']}`, SHA-256 `{base['sha256']}`, "
            f"protocol digest `{base['protocol_sha256']}`. "
            f"Used: {base['used']}.",
        ]
    family = payload["family"]
    lines += [
        "",
        "## The model family",
        "",
        f"- **Chosen.** {_sentence(family['chosen'])}",
        *[f"- {_sentence(reason)}" for reason in family["why_one_family"]],
        f"- **Split states.** {_sentence(family['split_states'])}",
        f"- **Not split.** {_sentence(family['not_split'])}",
        f"- **Output.** {_sentence(family['output'])}",
        f"- **Exactness.** {_sentence(family['exactness'])}",
        f"- **Not tested.** {_sentence('; '.join(family['not_tested']))}",
        "",
    ]
    lines += _section("Sub-states", payload["substates"])
    lines += _section("Fitting", payload["fitting"])
    lines += _section("The recursion", payload["recursion"])
    lines += [
        "## Runtime",
        "",
        _sentence(payload["runtime"]),
        "",
        "## Models",
        "",
        *_table(
            ["Model", "Channels", "Transition"],
            [
                [f"`{name}`", spec["channels"], spec["transition"]]
                for name, spec in payload["models"].items()
            ],
        ),
        "",
        "## Estimands",
        "",
        *_table(
            ["", "Role", "Question", "Model", "Reference", "Rule"],
            [
                [
                    e["key"],
                    e["role"],
                    _sentence(e["question"]),
                    f"`{e['model']}`",
                    f"`{e['reference']}`",
                    e["rule"],
                ]
                for e in payload["estimands"]
            ],
        ),
        "",
    ]
    lines += _section("Criteria", payload["criteria"])
    lines += _section("Per-state recall", payload["per_state_recall"])
    lines += _section("The mechanism", payload["mechanism"])
    lines += _section("One run", payload["one_run"])
    differences = payload["minimal_differences"]
    lines += [
        "## Metrics and minimal differences",
        "",
        *_table(
            ["Metric", "Minimal difference"],
            [[METRIC_TITLES[m], f"{differences[m]:g}"] for m in payload["metrics"]]
            + [["Recall of one state", f"{differences['recall']:g}"]],
        ),
        "",
    ]
    lines += _section("Bootstrap", payload["bootstrap"])
    check = payload["the_check"]
    lines += [
        "## The check",
        "",
        *[f"- {_sentence(item)}" for item in check["what"]],
        f"- **Tolerance.** {check['tolerance']:g}.",
        f"- **Otherwise.** {_sentence(check['otherwise'])}",
        "",
        "## Pinned records",
        "",
        *_table(
            ["File", "SHA-256"],
            [[f"`{k}`", f"`{v}`"] for k, v in payload["pinned_records"].items()],
        ),
        "",
        "## What this cannot show",
        "",
        *[f"- {_sentence(item)}" for item in payload["what_this_cannot_show"]],
        "",
        f"**Tuning.** {_sentence(_flatten(payload['tuning']))}",
        "",
    ]
    return "\n".join(lines)


def _share(value: float | None) -> str:
    return "–" if value is None else _fixed(value, 3)


def _check_sentence(check: Mapping[str, Any], tolerance: float) -> str:
    identical = (
        check["fits"]["identical_digests"]
        and check["scores"]["identical"]
        and check["runs"]["identical"]
    )
    return (
        f"The fold fits of {', '.join(check['fits']['folds'])} equal the "
        f"fitted-rates record's; `{check['scores']['model']}` gives back the "
        "combined-prior record's scores for all "
        f"{check['scores']['households']} households; and the quiet-run "
        f"statistic against the population hurdle gives back the predictive-check "
        f"record's in all {_n(check['runs']['cells'])} cells; each to within "
        f"{tolerance:g}. Both models scored the same labelled windows in all "
        f"{check['windows']['households']} households. "
        + (
            "They are bit-identical."
            if identical
            else "They are not all bit-identical, which the tolerance allows for: "
            "the published records were made on other platforms."
        )
    )


def render_page(payload: Mapping[str, Any]) -> str:
    """The results page, from the record."""
    results = payload["results"]
    configuration = payload["configuration"]
    environment = payload.get("environment", {})
    metrics = list(configuration["metrics"])
    mechanism = results["mechanism"]
    lines = [
        f"# {TITLE}: results",
        "",
        "The frozen [protocol](PHASE3_ACTIVITY_SUBSTATES_PROTOCOL.md), run as "
        f"declared. Generated entirely from `{RECORD_FILE}` by "
        "`sensor_modeling.datasets.substate_summary.render_page`.",
        "",
        f"**Status: {results['status']}.**",
        "",
        f"- **Protocol digest.** `{configuration['protocol_sha256']}`.",
        f"- **Run.** Commit `{str(environment.get('git_commit', ''))[:7]}`, recorded "
        f"{payload['recorded_at']}, on "
        f"{environment.get('platform', 'an unrecorded platform')}.",
        f"- **Conclusion of W1.** {_sentence(results['conclusion'])}",
        "- **M1, whether the excess of long quiet runs falls.** "
        f"{_sentence(mechanism['conclusion'])}",
        "",
        "## The check",
        "",
        _check_sentence(results["check"], configuration["the_check"]["tolerance"]),
        "",
        "## Each model in the recursion",
        "",
        "The mean over households, with its interval.",
        "",
        *_table(
            ["Model", *[METRIC_TITLES[m] for m in metrics]],
            [
                [f"`{cell['model']}`", *[_level(cell["metrics"][m]) for m in metrics]]
                for cell in results["cells"]
            ],
        ),
        "",
        "Recall of each state, the mean over households with its interval:",
        "",
        *_table(
            ["State", *[f"`{cell['model']}`" for cell in results["cells"]]],
            [
                [
                    f"`{state}`",
                    *[
                        _level(cell["per_state_recall"][state])
                        for cell in results["cells"]
                    ],
                ]
                for state in results["states"]
            ],
        ),
        "",
        "## The primary estimand",
        "",
        "Mean paired differences, oriented so that a positive value favours the "
        "model, with their intervals and verdicts.",
        "",
    ]
    for e in results["estimands"]:
        focus = e["focus"]
        lines += [
            *_table(
                [
                    "",
                    "Model against reference",
                    *[METRIC_TITLES[m] for m in metrics],
                    f"`{focus['state']}` recall",
                    "Conclusion",
                ],
                [
                    [
                        e["key"],
                        f"`{e['model']}` against `{e['reference']}`",
                        *[
                            f"{_difference(e['comparisons'][m])}; {e['verdicts'][m]}"
                            for m in metrics
                        ],
                        f"{_difference(e['per_state_recall'][focus['state']]['comparison'])}"
                        f"; {focus['verdict']}",
                        e["conclusion"],
                    ]
                ],
            ),
            "",
            "The other common states' recall, which the rule also guards: "
            + "; ".join(
                f"`{state}` {verdict}" for state, verdict in e["guarded_states"].items()
            )
            + ".",
            "",
            f"**{e['key']}, recall by state.**",
            "",
            *_table(
                ["State", "Difference", "Verdict", "Households"],
                [
                    [
                        f"`{state}`",
                        _difference(entry["comparison"]),
                        entry["verdict"],
                        _n(entry["comparison"]["n"]),
                    ]
                    for state, entry in e["per_state_recall"].items()
                ],
            ),
            "",
        ]
    lines += [
        "## The mechanism",
        "",
        "Windows in quiet runs of at least an hour, against the predictive "
        "distribution. Each household's value is the mean over its cells of the "
        "log ratio, shown as a ratio: the geometric mean over households, its "
        "interval, households above and below one, and the verdict.",
        "",
        "**M1.** How much nearer the observed long quiet runs the model's "
        "prediction is than the reference's, whichever side of them each lies: "
        "the exponential of the mean fall in the absolute log ratio. Above one "
        "means the excess falls.",
        "",
        *_table(
            ["State", "Fall in misfit", "Reading"],
            [
                [f"`{state}`", _ratio(entry), entry["reading"]]
                for state, entry in mechanism["states"].items()
            ],
        ),
        "",
        f"Conclusion of M1: {mechanism['conclusion']}.",
        "",
        "The ratio that remains in each split state, observed over predicted, "
        "under each model, described; below one means the model predicts more "
        "long quiet runs than occur:",
        "",
        *_table(
            ["State", "Reference", "Model"],
            [
                [f"`{state}`", _ratio(entry["reference"]), _ratio(entry["model"])]
                for state, entry in mechanism["remaining"].items()
            ],
        ),
        "",
        "## The fitted sub-states",
        "",
        "Per fold and split state: the training windows and stretches EM used; "
        "the quiet sub-state's share of the windows, counted over channels; each "
        "sub-state's silence probability, averaged over the channels; the "
        "probability of entering the state quiet, of staying quiet and of "
        "staying busy over one step; and the training log-likelihood at the "
        "last E-step, with its change over the last iteration. The quiet "
        "sub-state is the one whose silence probability is higher.",
        "",
    ]
    rows = []
    for fold, fitted in results["fitted"].items():
        for state, entry in fitted["states"].items():
            trace = entry["trace"]
            rows.append(
                [
                    fold,
                    f"`{state}`",
                    _n(entry["windows"]),
                    _n(entry["stretches"]),
                    _share(entry["share"][0]),
                    " / ".join(_share(x) for x in entry["mean_silence"]),
                    _fixed(entry["entry"][0], 3),
                    _fixed(entry["within"][0][0], 3),
                    _fixed(entry["within"][1][1], 3),
                    _n(trace[-1], 1) if trace else "–",
                    (
                        _fixed(entry["last_change"], 6, sign=True)
                        if entry["last_change"] is not None
                        else "–"
                    ),
                ]
            )
    lines += [
        *_table(
            [
                "Fold",
                "State",
                "Windows",
                "Stretches",
                "Quiet share",
                "Silence, quiet / busy",
                "Enter quiet",
                "Stay quiet",
                "Stay busy",
                "Log-likelihood",
                "Last change",
            ],
            rows,
        ),
        "",
        "## Notes",
        "",
        *[f"- {_sentence(note)}" for note in payload.get("notes", [])],
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "PROTOCOL_FILE",
    "PROTOCOL_PAGE",
    "RECORD_FILE",
    "RESULTS_PAGE",
    "render_page",
    "render_protocol",
]
