"""The Phase 5 external-results page, generated entirely from its record.

:func:`render_page` writes the documentation page from the published record
alone: the declared conclusions, every home's metrics and paired differences,
and, apart from them, the descriptive diagnostics and the separation of
losses. Every home is shown, however it performed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .external_experiment import CONDITIONS, RESULT_SCHEMA

RECORD_FILE = "artifacts/phase5/phase5-external-ordonez-results.json"
PROTOCOL_PAGE = "PHASE5_EXTERNAL_PROTOCOL.md"
FIGURE_DIR = "figures"

_ESTIMANDS = {
    "T": "Transfer: zero-shot against the declared rates",
    "A": "Adaptation: adapted against zero-shot",
    "C": "Above chance: zero-shot balanced accuracy against chance",
    "S": "Phase 4 direction: structural disagreement against confidence",
}
_METRICS = (
    "balanced_accuracy",
    "log_loss",
    "brier",
    "calibration_error",
    "macro_f1",
    "accuracy",
)


def _n(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _signed(value: Any) -> str:
    return "—" if value is None else f"{value:+.3f}"


def _band(interval: Mapping[str, float] | None, signed: bool = False) -> str:
    if interval is None:
        return "—"
    if signed:
        return f"[{interval['low']:+.3f}, {interval['high']:+.3f}]"
    return f"[{interval['low']:.3f}, {interval['high']:.3f}]"


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|")


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def render_page(payload: Mapping[str, Any]) -> str:
    """The whole results page, from the record alone."""
    results = payload["results"]
    if results.get("result_schema") != RESULT_SCHEMA:
        raise ValueError(f"not a {RESULT_SCHEMA} record")
    homes = results["households"]
    eligible = [h for h, e in homes.items() if e.get("eligible")]
    conclusions = results["conclusions"]
    protocol = results["protocol"]
    lines: list[str] = []
    add = lines.append

    add("# Phase 5: external generalisation results")
    add("")
    add(
        "The frozen [external-generalisation protocol](" + PROTOCOL_PAGE + ") run "
        "exactly as declared, on the UCI ADL Binary dataset of Ordóñez et al.: "
        "two single-resident homes in Spain, collected independently of CASAS, "
        "with a different sensing layout. This page is generated entirely from "
        f"the record, `{RECORD_FILE}`, by "
        "`sensor_modeling.datasets.external_summary.render_page`, and a test "
        "checks that the committed page is exactly that rendering."
    )
    add("")
    add("**Status: pre-specified, external, two homes.**")
    add("")
    add(
        f"- **Protocol.** Frozen in `{protocol['frozen_in']}` before any external "
        f"scoring, SHA-256 `{protocol['sha256']}`. The run checked the frozen "
        "file, and reproduced every CASAS population digest, before it scored."
    )
    add(
        "- **Two homes.** Every conclusion is about these two homes, not a "
        "population of homes. Intervals resample local days within a home."
    )
    add(
        "- **Descriptive sections.** The failure causes and the separation of "
        "losses are descriptive. They were declared in the scoring code before "
        "the run, and change no conclusion."
    )
    add("")

    # ------------------------------------------------------------------
    add("## The declared conclusions")
    add("")
    rows = []
    for key, title in _ESTIMANDS.items():
        entry = conclusions[key]
        metric = "aurc" if key == "S" else "balanced_accuracy"
        verdicts = "; ".join(f"{h}: {v}" for h, v in entry[metric].items())
        rows.append([key, title, verdicts, f"**{entry['conclusion']}**"])
    lines += _table(["", "Estimand", "Verdict per home", "Conclusion"], rows)
    add("")
    add("Homes improved and worsened, by the point difference:")
    add("")
    rows = []
    for key in ("T", "A"):
        for metric, counts in conclusions["improved"][key].items():
            rows.append(
                [key, metric.replace("_", " "), counts["improved"], counts["worsened"]]
            )
    lines += _table(["Estimand", "Metric", "Improved", "Worsened"], rows)
    add("")
    add(
        "Log loss, the Brier score and the calibration error count as improved "
        "when they fall."
    )
    add("")

    # ------------------------------------------------------------------
    add("## Every home")
    add("")
    add(f"![Balanced accuracy by condition]({FIGURE_DIR}/phase5-external-metrics.svg)")
    add("")
    for home, entry in homes.items():
        add(f"### {home}")
        add("")
        if not entry.get("eligible"):
            add(
                "**Not eligible.** Blocking validation errors: "
                + (", ".join(entry["validation"]["blocking"]) or "none")
                + f"; eligibility facts: {entry.get('eligibility')}."
            )
            add("")
            continue
        add(
            f"{entry['scored_windows']:,} scored windows over "
            f"{len(entry['scored_days'])} days, after the adaptation period ending "
            f"{entry['adaptation_end']}. Chance is {_n(entry['chance'])}."
        )
        add("")
        rows = []
        for metric in _METRICS:
            row = [metric.replace("_", " ")]
            for condition in CONDITIONS:
                value = entry["conditions"][condition][metric]
                row.append(f"{_n(value['estimate'])} {_band(value['interval'])}")
            rows.append(row)
        oracle = entry["diagnostics"]["oracle"]
        for row, metric in zip(rows, _METRICS):
            row.append(_n(oracle[metric]["estimate"]))
        lines += _table(
            ["Metric", "declared", "zero-shot", "adapted", "oracle (descriptive)"], rows
        )
        add("")
        add("Per-state recall:")
        add("")
        states = [
            s
            for s, v in entry["conditions"]["zero_shot"]["per_class_recall"].items()
            if v is not None
        ]
        lines += _table(
            ["State", *[c.replace("_", "-") for c in CONDITIONS]],
            [
                [
                    f"`{s}`",
                    *[
                        _n(entry["conditions"][c]["per_class_recall"][s])
                        for c in CONDITIONS
                    ],
                ]
                for s in states
            ],
        )
        add("")

    # ------------------------------------------------------------------
    add("## Paired differences")
    add("")
    add(
        "Oriented so that a positive difference favours the first model named in "
        "the estimand. Intervals resample the scored period's days."
    )
    add("")
    rows = []
    for home in eligible:
        estimands = homes[home]["estimands"]
        for key in _ESTIMANDS:
            for metric, item in estimands[key].items():
                rows.append(
                    [
                        home,
                        key,
                        metric.replace("_", " "),
                        _signed(item["estimate"]),
                        _band(item["interval"], signed=True),
                        _n(item["minimal"], 2),
                        item["verdict"],
                    ]
                )
    lines += _table(
        [
            "Home",
            "Estimand",
            "Metric",
            "Difference",
            "95% interval",
            "Minimal",
            "Verdict",
        ],
        rows,
    )
    add("")
    rows = [
        [
            home,
            _n(homes[home]["selective"]["confidence_aurc"]),
            _n(homes[home]["selective"]["structural_aurc"]),
        ]
        for home in eligible
    ]
    add("Error AURC, rejecting by each signal within the home:")
    add("")
    lines += _table(["Home", "confidence", "structural disagreement"], rows)
    add("")

    # ------------------------------------------------------------------
    add("## Failure causes (descriptive)")
    add("")
    add(
        f"![Scored states against the development panel]({FIGURE_DIR}/phase5-external-priors.svg)"
    )
    add("")
    add(
        f"![Event rates against the CASAS population]({FIGURE_DIR}/phase5-external-rates.svg)"
    )
    add("")
    for home in eligible:
        d = homes[home]["diagnostics"]
        add(f"### {home}")
        add("")
        obs = d["unsupported_observations"]
        add(
            f"- **Missing sensor semantics.** {_pct(obs['fraction'])} of the scored "
            f"period's {obs['activations']:,} sensor activations come from sensors "
            "that feed no model channel: "
            + "; ".join(
                f"{reason}: {count:,}"
                for reason, count in obs["by_reason"].items()
                if reason != "routed"
            )
            + "."
        )
        coverage = d["mapping_coverage"]
        add(
            f"- **Ontology mismatch.** {_pct(coverage['scorable_fraction'])} of the "
            "scored period's annotated time is scorable. By disposition, in hours: "
            + ", ".join(f"{k} {v / 3600:.1f}" for k, v in coverage["seconds"].items())
            + ". Whole recording: "
            f"{coverage['dispositions']['impossible_intervals']} impossible "
            f"intervals, {coverage['dispositions']['point_annotations']} point "
            "annotations."
        )
        rooms = d["room_structure"]
        add(
            "- **Room structure.** Instrumented model channels: "
            + ", ".join(f"`{c}`" for c in rooms["routed_channels"])
            + "; without a sensor: "
            + (", ".join(f"`{c}`" for c in rooms["uninstrumented_channels"]) or "none")
            + ". Scored states' rooms: "
            + "; ".join(
                f"`{s}` {v['room'] or 'no room'}"
                + (
                    f" ({v['channel']})"
                    if v["channel"]
                    else (" (no channel)" if v["room"] else "")
                )
                for s, v in rooms["scored_state_rooms"].items()
            )
            + ". Fixture-level sensor types: "
            + ", ".join(rooms["approximate_sensor_types"])
            + "."
        )
        calibration = d["calibration_shift"]
        development = calibration.get("development")
        add(
            f"- **Calibration shift.** Zero-shot mean confidence "
            f"{_n(calibration['mean_confidence'])} against accuracy "
            f"{_n(calibration['accuracy'])}: a gap of {_signed(calibration['gap'])}"
            + (
                f", against {_signed(development['gap'])} for the same model on the "
                "development panel"
                if development
                else ""
            )
            + "."
        )
        shift = d["state_prior_shift"]
        add(
            "- **State-prior shift.** Scored states here: "
            + ", ".join(f"{k} {_pct(v)}" for k, v in shift["truth"].items())
            + (
                "; on the development panel: "
                + ", ".join(f"{k} {_pct(v)}" for k, v in shift["development"].items())
                + f" (Jensen-Shannon {_n(shift['divergence_truth_development'])} bits)"
                if "development" in shift
                else ""
            )
            + f". Zero-shot predictions of unsupported states: "
            f"{_pct(shift['predicted_unsupported'])}."
        )
        add("")
        add(
            "**Different event rates.** Scored windows here against the CASAS population:"
        )
        add("")
        rows = []
        for channel, states in d["event_rates"].items():
            for state, v in states.items():
                rows.append(
                    [
                        f"`{channel}`",
                        f"`{state}`",
                        v["windows"],
                        _n(v["silence"]),
                        _n(v["casas_silence"]),
                        _n(v["active_mean"], 2),
                        _n(v["casas_active_mean"], 2),
                    ]
                )
        lines += _table(
            [
                "Channel",
                "State",
                "Windows",
                "Silent here",
                "Silent in CASAS",
                "Active mean here",
                "Active mean in CASAS",
            ],
            rows,
        )
        add("")

    # ------------------------------------------------------------------
    add("## Separating the losses (descriptive)")
    add("")
    add(
        "The oracle runs the same recursion with each home's own channel "
        "parameters, fitted on all its labelled windows, including the scored "
        "ones. It is a ceiling for these channels and this model family, not a "
        "condition."
    )
    add("")
    add(
        "- **Dataset incompatibility.** What cannot be compared: unscorable "
        "annotated time and unsupported observations."
    )
    add(
        "- **Sensing-information limitation.** What even the oracle cannot "
        "recover: its shortfall from perfect balanced accuracy."
    )
    add(
        "- **Model failure.** What the transferred parameters lose: the oracle's "
        "balanced accuracy minus the zero-shot's."
    )
    add("")
    rows = []
    for home in eligible:
        a = homes[home]["attribution"]
        rows.append(
            [
                home,
                _pct(a["dataset_incompatibility"]["unscorable_labelled_fraction"]),
                _pct(a["dataset_incompatibility"]["unsupported_observation_fraction"]),
                _n(a["sensing_limitation"]["oracle_balanced_accuracy"]),
                _n(a["sensing_limitation"]["oracle_shortfall"]),
                _signed(a["model_failure"]["zero_shot_below_oracle"]),
                _signed(a["model_failure"]["calibration_gap"]),
            ]
        )
    lines += _table(
        [
            "Home",
            "Unscorable annotated time",
            "Unsupported observations",
            "Oracle balanced accuracy",
            "Oracle shortfall",
            "Zero-shot below oracle",
            "Zero-shot calibration gap",
        ],
        rows,
    )
    add("")

    # ------------------------------------------------------------------
    add("## Checks")
    add("")
    populations = results["populations"]
    add(
        "- **Populations.** "
        + "; ".join(
            f"`{name}` {'reproduced' if p['reproduced'] else 'NOT reproduced'}"
            for name, p in populations.items()
        )
        + "."
    )
    for home, entry in homes.items():
        issues = entry["validation"]["issues"]
        add(
            f"- **{home} validation.** "
            + ", ".join(f"{i['code']} ({i['severity']}, {i['count']})" for i in issues)
            + f". Eligible: {_n(entry.get('eligible'))}."
        )
    add("")
    add("## What this does not show")
    add("")
    add("- **A population claim.** Two homes are two case studies.")
    add(
        "- **The unsupported states.** `home_active`, `kitchen_activity` and "
        "`bed_awake` cannot be scored: the meal labels are ambiguous."
    )
    add(
        "- **Causal proof of the failure causes.** The diagnostics describe the "
        "data; they were not part of the declared evaluation."
    )
    add("")
    return "\n".join(lines)
