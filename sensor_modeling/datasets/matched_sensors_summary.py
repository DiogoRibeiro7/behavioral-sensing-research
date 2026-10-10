"""The matched-sensor pages, generated from the frozen protocol and the record.

:func:`render_protocol` writes the protocol page from the frozen protocol file,
and :func:`render_page` the results page from the study's record. Every number
on either page comes from those files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .matched_sensors_planning import MOMENTS
from .matched_sensors_protocol import (
    C1,
    C2,
    C3,
    FEATURES,
    MATCHED,
    PROFILES,
    SENSITIVITY,
    STANDARD,
    STATE_HOURS,
)
from .threshold_calibration_summary import _fixed, _percent_of

PROTOCOL_FILE = "artifacts/matched_sensors/matched_sensors_protocol.json"
RECORD_FILE = "artifacts/matched_sensors/matched-sensors.json"
PLANNING_FILE = "artifacts/matched_sensors/matched-sensors-planning.json"
PROTOCOL_PAGE = "MATCHED_SENSORS_PROTOCOL.md"
RESULTS_PAGE = "MATCHED_SENSORS_RESULTS.md"
FIGURE_DIR = "figures"
FIGURE_PREFIX = "matched-sensors"

_READINGS = {
    "survives": "survives",
    "does_not_survive": "does not survive",
    "follows": "follows",
    "does_not_follow": "does not follow",
    "reproduced": "reproduced",
    "not_reproduced": "not reproduced",
    "inconclusive": "inconclusive",
}
_PROFILE_TITLES = {
    STANDARD: "standard",
    MATCHED: "matched",
    SENSITIVITY: "sensitivity",
}
_CRITERIA_TITLES = {
    C1: "C1 The detection survives the matched profile",
    C2: "C2 The hours of sleep still follow the truth",
    C3: "C3 The kitchen and bathroom collapse is reproduced",
    "multiplicity": "Multiplicity",
}
_MOMENT_TITLES = {
    "kitchen_per_day": "kitchen motion activations a day",
    "bathroom_per_day": "bathroom motion activations a day",
    "bedroom_per_day": "bedroom motion activations a day",
    "living_per_day": "living-room motion activations a day",
    "hall_per_day": "hallway motion activations a day",
    "switch_share": "share of motion activations after another room's",
    "night_bedroom": "bedroom motion activations a night, 00:00 to 06:00",
    "retrigger_gap_seconds": "median gap between one sensor's activations "
    "under ten minutes apart, seconds",
}


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|")


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _sentence(text: str) -> str:
    text = text.strip()
    if not text.startswith("`"):
        text = text[0].upper() + text[1:]
    return text if text.endswith(".") else text + "."


def _title(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _n(value: Any, digits: int = 2) -> str:
    """A number as the page shows it; a missing one as a dash."""
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    return _fixed(value, digits, group=True)


def _signed(value: Any, digits: int = 2) -> str:
    return "–" if value is None else _fixed(value, digits, sign=True)


def _interval(
    entry: Mapping[str, Any] | None, digits: int = 2, signed: bool = False
) -> str:
    """An estimate with its interval, as ``0.12 [0.05, 0.20]``."""
    if entry is None or entry.get("estimate") is None:
        return "–"
    show = _signed if signed else _n
    text = show(float(entry["estimate"]), digits)
    interval = entry.get("interval")
    if interval:
        text += (
            f" [{show(float(interval['low']), digits)}, "
            f"{show(float(interval['high']), digits)}]"
        )
    return text


def _share(entry: Mapping[str, Any]) -> str:
    """A share of homes as ``293 (73.3% [68.7, 77.4])``."""
    if entry.get("estimate") is None:
        return "–"
    text = f"{entry['count']:,} ({_percent_of(entry['estimate'], 1)}"
    interval = entry.get("interval")
    if interval:
        text += (
            f" [{_fixed(interval['low'], 1, scale=2)}, "
            f"{_fixed(interval['high'], 1, scale=2)}]"
        )
    return text + ")"


def _figure(name: str, alt: str) -> str:
    return f"![{alt}]({FIGURE_DIR}/{FIGURE_PREFIX}-{name}.svg)"


def _flatten(value: Any) -> str:
    if isinstance(value, Mapping):
        return "; ".join(
            f"{_title(k).lower()}: {_flatten(v)}" for k, v in value.items()
        )
    if isinstance(value, list | tuple):
        return ", ".join(_flatten(v) for v in value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _section(title: str, entries: Mapping[str, Any]) -> list[str]:
    lines = [f"## {title}", ""]
    for name, value in entries.items():
        lines.append(f"- **{_title(name)}.** {_sentence(_flatten(value))}")
    return [*lines, ""]


# ----------------------------------------------------------------------------
# The protocol page
# ----------------------------------------------------------------------------
def _profiles_part(profiles: Mapping[str, Any]) -> list[str]:
    settings = profiles["settings"]
    labels = (STANDARD, MATCHED, SENSITIVITY)
    return [
        "## The profiles",
        "",
        *[
            f"- **{_title(_PROFILE_TITLES[label])}.** {_sentence(profiles[label])}"
            for label in labels
        ],
        f"- **Chosen by.** {_sentence(profiles['chosen_by'])}",
        f"- **Pairing.** {_sentence(profiles['pairing'])}",
        f"- **Delivery.** {_sentence(profiles['delivery'])}",
        "",
        *_table(
            ["Setting", "Meaning", *[_title(_PROFILE_TITLES[x]) for x in labels]],
            [
                [
                    _title(name),
                    _sentence(row["meaning"]),
                    *[_flatten(row[label]) for label in labels],
                ]
                for name, row in settings.items()
            ],
        ),
        "",
    ]


def render_protocol(payload: Mapping[str, Any]) -> str:
    """The protocol page, from the frozen protocol file's content."""
    lines = [
        "# The simulator's homes with TIHM's sensors: the protocol",
        "",
        "Every detection this repository reports in simulation comes from "
        "homes whose event sensors fire at rates written into the simulator. "
        "This page is the protocol for drawing those homes' event sensors "
        "again under a profile matched to TIHM's sensor records, generated "
        f"entirely from the frozen file `{PROTOCOL_FILE}` by "
        "`sensor_modeling.datasets.matched_sensors_summary.render_protocol`.",
        "",
        f"**Status: {payload['status']}.**",
        "",
        f"- **Protocol digest.** `{payload['protocol_sha256']}`.",
        f"- **The question.** {_sentence(payload['question'])}",
        "- **This page reports no result.** The run checks the frozen file and "
        "the pinned records before it runs.",
    ]
    at_freeze = payload.get("at_freeze")
    if at_freeze:
        versions = ", ".join(
            f"{name} {version}" for name, version in at_freeze["distributions"].items()
        )
        lines.append(
            "- **The code it was frozen against.** The digests of "
            f"{len(at_freeze['sources']):,} source files, every default "
            f"setting, and {versions}."
        )
    lines += [
        "",
        "## What had been seen",
        "",
        *[f"- {_sentence(item)}" for item in payload["inspected_before"]],
        "",
    ]
    lines += _section("The homes", payload["homes"])
    lines += _profiles_part(payload["profiles"])
    lines += _section("The pipeline", payload["pipeline"])
    lines += [
        "## What the outcomes turn on",
        "",
        "Written down before any home's pipeline output under spill-over was read.",
        "",
        *[f"- {_sentence(item)}" for item in payload["what_the_outcomes_turn_on"]],
        "",
    ]
    lines += _section("The check", payload["check"])
    lines += _section("Definitions", payload["definitions"])
    lines += [
        "## Estimands",
        "",
        *_table(
            ["", "Estimand"],
            [[k, _sentence(v)] for k, v in payload["estimands"].items()],
        ),
        "",
        "## Criteria",
        "",
        *[
            f"- **{_CRITERIA_TITLES.get(name, _title(name))}.** {_sentence(text)}"
            for name, text in payload["criteria"].items()
        ],
        "",
    ]
    lines += _section("Margins", payload["margins"])
    lines += [
        "## Pinned records",
        "",
        *_table(
            ["File", "SHA-256"],
            [
                [f"`{name}`", f"`{digest}`"]
                for name, digest in payload["pinned_records"].items()
            ],
        ),
        "",
    ]
    lines += _section("Intervals", payload["intervals"])
    lines += [
        "## Reporting",
        "",
        _sentence(payload["reporting"]),
        "",
        "## What this cannot show",
        "",
        *[f"- {_sentence(item)}" for item in payload["what_this_cannot_show"]],
        "",
        "## Data",
        "",
        f"{payload['data']['dataset']}: {payload['data']['read']}. "
        f"{_sentence(payload['acknowledgement'])}",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# The results page
# ----------------------------------------------------------------------------
def _criteria_part(results: Mapping[str, Any]) -> list[str]:
    criteria = results["criteria"]
    c1, c2, c3 = criteria[C1], criteria[C2], criteria[C3]
    level = results["E5"]
    kitchen = level["kitchen_activity"][MATCHED]["pooled_day_median_interval"]
    bathroom = level["bathroom_activity"][MATCHED]["pooled_day_median_interval"]
    rows = [
        [
            _CRITERIA_TITLES[C1],
            f"**{_READINGS[c1['reading']]}**",
            f"E1 = {_interval(c1['estimate'], 2, True)}; margin "
            f"{_signed(-c1['margin'])}",
        ],
        [
            _CRITERIA_TITLES[C2],
            f"**{_READINGS[c2['reading']]}**",
            f"matched E4 = {_interval(c2['estimate'])} over "
            f"{c2['estimate']['homes']:,} homes; margin {_n(c2['margin'])}",
        ],
        [
            _CRITERIA_TITLES[C3],
            f"**{_READINGS[c3['reading']]}**",
            f"pooled-day medians {_interval(kitchen, 2)} h in the kitchen and "
            f"{_interval(bathroom, 3)} h in the bathroom; ceilings "
            f"{_n(c3['ceilings'][0])} and {_n(c3['ceilings'][1])} h",
        ],
    ]
    return [
        "## The criteria",
        "",
        *_table(["Criterion", "Reading", "What decided it"], rows),
        "",
    ]


def _detection_rows(
    by_profile: Mapping[str, Any], profiles: Sequence[str], key: str | None = None
) -> list[list[str]]:
    rows = []
    for profile in profiles:
        entry = by_profile[profile] if key is None else by_profile[profile][key]
        rows.append(
            [
                _PROFILE_TITLES[profile],
                _share(entry["detected"]),
                _share(entry["false_detections"]),
                _interval(entry["excess_detection"]),
            ]
        )
    return rows


def _detection_part(results: Mapping[str, Any]) -> list[str]:
    e2 = results["E2"]
    header = ["Profile", "Detected", "Falsely detected", "Excess detection"]
    diff = e2["matched_minus_standard"]
    delays = ", ".join(
        f"{_PROFILE_TITLES[p]} {_interval(e2['by_profile'][p]['delay_days'], 1)}"
        for p in PROFILES
    )
    return [
        "## Detection of the step change (E1, E2)",
        "",
        *_table(header, _detection_rows(e2["by_profile"], PROFILES)),
        "",
        f"Matched minus standard, a mean over {results['homes']:,} homes of a "
        f"paired difference: excess detection {_interval(results['E1'], 2, True)}, "
        f"detected {_interval(diff['detected'], 2, True)}, falsely detected "
        f"{_interval(diff['false_detections'], 2, True)}. Median delay of a "
        f"detection, in days: {delays}.",
        "",
        "Counting only alerts of a decrease:",
        "",
        *_table(header, _detection_rows(e2["by_profile"], PROFILES, "decrease_only")),
        "",
        _figure("detection", "Detection of the step change under each profile"),
        "",
    ]


def _alerts_part(results: Mapping[str, Any]) -> list[str]:
    e3 = results["E3"]
    features = sorted({f for p in PROFILES for f in e3[p]["per_person_day"]})
    rows = [
        [
            _PROFILE_TITLES[profile],
            _interval(e3[profile]["false_alerts"]),
            *[
                (
                    _interval(e3[profile]["per_person_day"][f], 4)
                    if f in e3[profile]["per_person_day"]
                    else "0"
                )
                for f in features
            ],
        ]
        for profile in PROFILES
    ]
    return [
        "## False alerts (E3)",
        "",
        "In the stable arm every behavioural alert is a false one. Alerts about "
        f"each feature are per person-day, over {e3[STANDARD]['person_days']:,} "
        "person-days.",
        "",
        *_table(
            ["Profile", "False alerts per home", *[f"`{f}`" for f in features]],
            rows,
        ),
        "",
    ]


def _tracking_part(results: Mapping[str, Any]) -> list[str]:
    e4 = results["E4"]
    rows = [
        [
            f"`{feature}`",
            _interval(e4[feature][STANDARD]),
            _interval(e4[feature][MATCHED]),
            _interval(e4[feature]["matched_minus_standard"], 2, True),
        ]
        for feature in FEATURES
    ]
    return [
        "## How each feature follows the truth (E4)",
        "",
        "The mean over homes of the within-home Spearman correlation between the "
        "pipeline's value and the simulator's truth on each profile's matched "
        "days of the stable arm.",
        "",
        *_table(["Feature", "Standard", "Matched", "Matched minus standard"], rows),
        "",
        _figure(
            "tracking",
            "Within-home correlation with the truth for each feature and profile",
        ),
        "",
    ]


def _level_part(results: Mapping[str, Any]) -> list[str]:
    e5 = results["E5"]
    rows = []
    for name in STATE_HOURS:
        cells = [f"`{name}`", _n(e5[name][STANDARD]["pooled_day_median_truth"])]
        for profile in PROFILES:
            cells.append(_n(e5[name][profile]["pooled_day_median_pipeline"]))
        rows.append(cells)
    differences = [
        [
            f"`{feature}`",
            *[
                _interval(
                    e5[feature.removesuffix("_hours")][p]["mean_difference"], 2, True
                )
                for p in PROFILES
            ],
        ]
        for feature in FEATURES
    ]
    return [
        "## Hours a day (E5)",
        "",
        "Pooled-day medians over every matched day of every home. The truth is "
        "on the standard profile's matched days.",
        "",
        *_table(["State", "Truth", "Standard", "Matched"], rows),
        "",
        "The mean over homes of each home's mean difference, pipeline minus "
        "truth, in hours:",
        "",
        *_table(["Feature", "Standard", "Matched"], differences),
        "",
    ]


def _moments_part(results: Mapping[str, Any]) -> list[str]:
    e6 = results["E6"]
    rows = [
        [
            _MOMENT_TITLES[name],
            _n(e6["tihm"][name]["median"], 2),
            _n(e6[STANDARD][name]["median"], 2),
            _n(e6[MATCHED][name]["median"], 2),
        ]
        for name in MOMENTS
    ]
    return [
        "## The sensor records beside TIHM's (E6)",
        "",
        "Medians over homes. TIHM's are from the planning record; the two "
        "profiles' are from the stable records of the study's homes.",
        "",
        *_table(["Moment", "TIHM", "Standard", "Matched"], rows),
        "",
        _figure(
            "moments",
            "The sensor records' moments under each profile, beside TIHM's",
        ),
        "",
    ]


def _withheld_part(results: Mapping[str, Any]) -> list[str]:
    e7 = results["E7"]
    rows = [
        [
            _PROFILE_TITLES[p],
            _n(e7[p]["change_verdicts"]),
            _n(e7[p]["raised_an_alert"]),
            _n(e7[p]["raised_none"]["not_attributable_enough"]),
            _n(e7[p]["raised_none"]["graded_too_low_or_withheld"]),
            _n(e7[p]["notices_of_a_burst"]),
            _interval(e7[p]["attribution_at_close"], 3),
        ]
        for p in PROFILES
    ]
    return [
        "## What became of the verdicts (E7)",
        "",
        "In the stable arm. A verdict held back at the gate is one whose day's "
        "coverage times attribution was under the confidence gate.",
        "",
        *_table(
            [
                "Profile",
                "Change verdicts",
                "Raised an alert",
                "Held back at the gate",
                "Held back otherwise",
                "Burst notices",
                "Attribution at a day's close",
            ],
            rows,
        ),
        "",
    ]


def _beliefs_part(results: Mapping[str, Any]) -> list[str]:
    e8 = results["E8"]
    lines = [
        "## The belief after a room's activations alone (E8)",
        "",
        "In the stable arm, the mean belief at the end of the steps whose window "
        "held motion activations of one room's sensor and of no other motion "
        "sensor.",
        "",
    ]
    for room in ("kitchen", "bathroom"):
        states = list(e8[STANDARD][room]["mean_belief"])
        lines += [
            *_table(
                [f"After {room} alone", "Steps", *[f"`{s}`" for s in states]],
                [
                    [
                        _PROFILE_TITLES[p],
                        _n(e8[p][room]["steps"]),
                        *[_n(e8[p][room]["mean_belief"].get(s)) for s in states],
                    ]
                    for p in PROFILES
                ],
            ),
            "",
        ]
    return lines


def _sensitivity_part(results: Mapping[str, Any]) -> list[str]:
    e9 = results["E9"]
    if not e9.get("homes"):
        return []
    profile = e9["profile"]
    e4 = e9["E4"]
    return [
        "## The sensitivity profile (E9)",
        "",
        "The planning record's second-nearest grid point, presence scale "
        f"{_n(profile['presence_scale'])} and spill-over "
        f"{_n(profile['spill_rate'])} an hour, against the standard profile on "
        f"the study's first {e9['homes']:,} homes.",
        "",
        *_table(
            ["Profile", "Detected", "Falsely detected", "Excess detection"],
            _detection_rows(e9["E2"]["by_profile"], (STANDARD, SENSITIVITY)),
        ),
        "",
        "Excess detection, sensitivity minus standard: "
        f"{_interval(e9['E1'], 2, True)}. Within-home correlation with the "
        "truth, standard and sensitivity: "
        + "; ".join(
            f"`{feature}` {_interval(e4[feature][STANDARD])} and "
            f"{_interval(e4[feature][SENSITIVITY])}"
            for feature in FEATURES
        )
        + ".",
        "",
    ]


def _changed(configuration: Mapping[str, Any]) -> str:
    changed = configuration.get("code_changed_since_the_freeze") or {}
    found = [
        f"{group}: {', '.join(names)}" for group, names in changed.items() if names
    ]
    return "; ".join(found) + "." if found else "None."


def render_page(payload: Mapping[str, Any]) -> str:
    """The results page, from the study's record."""
    configuration = payload["configuration"]
    results = payload["results"]
    environment = payload.get("environment", {})
    check = results["check"]
    lines = [
        "# The simulator's homes with TIHM's sensors: results",
        "",
        "The frozen [matched-sensor protocol](MATCHED_SENSORS_PROTOCOL.md), run "
        f"as declared. This page is generated entirely from `{RECORD_FILE}` by "
        "`sensor_modeling.datasets.matched_sensors_summary.render_page`, and a "
        "test checks that the committed page is exactly that rendering.",
        "",
        f"**Status: {configuration['status']}.**",
        "",
        f"- **Protocol digest.** `{configuration['protocol_sha256']}`.",
        f"- **Run.** Commit `{str(environment.get('git_commit', ''))[:7]}`, "
        f"recorded {payload.get('recorded_at', '')}.",
        f"- **Homes.** {check['homes']:,}, each in both arms under the standard "
        "and the matched profiles.",
        f"- **Code changed since the freeze.** {_changed(configuration)}",
        "",
        "## The check",
        "",
    ]
    if not check["reproduced"]:
        lines += [
            "The standard profile's runs did not reproduce the published record "
            f"in {len(check['runs_that_differ']):,} of {check['runs_compared']:,} "
            "runs, so nothing is reported.",
            "",
        ]
        return "\n".join(lines)
    lines += [
        f"All {check['runs_compared']:,} runs of the standard profile raised, home "
        "by home and arm by arm, the alerts the published threshold-calibration "
        "record gives under the default reference.",
        "",
    ]
    for part in (
        _criteria_part,
        _detection_part,
        _alerts_part,
        _tracking_part,
        _level_part,
        _withheld_part,
        _beliefs_part,
        _moments_part,
        _sensitivity_part,
    ):
        lines += part(results)
    lines += [
        "## Notes",
        "",
        *[f"- {_sentence(note)}" for note in payload.get("notes", [])],
        "",
    ]
    return "\n".join(lines)
