"""Figures and numbers for the article on decomposing inferred sleep against a sleep mat.

Everything is read from the records committed under ``artifacts/``; nothing is
recomputed from the TIHM dataset, which is not redistributed. Run from the
repository's root::

    python articles/sleep-mat-decomposition/make_figures.py

It writes four PNG figures to ``articles/sleep-mat-decomposition/figures`` and,
beside this script, ``numbers.json``: every number the article quotes from the
records, unrounded, under the record it comes from. The article rounds half
away from zero. Numbers from the cited literature are not in it.

TIHM is by Palermo et al., Scientific Data 10, 606 (2023), under CC BY 4.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SIMULATED = "#2a78d6"
SIMULATED_LIGHT = "#9cc1ee"
REAL = "#eb6834"
NEUTRAL = "#8f8e8a"
TIHM_CREDIT = (
    "Author's figure from the repository's records. TIHM data: Palermo et al., "
    "Scientific Data 10, 606 (2023), CC BY 4.0."
)
MIXED_CREDIT = (
    "Author's figure from the repository's records. Simulated homes, and TIHM data: "
    "Palermo et al., Scientific Data 10, 606 (2023), CC BY 4.0."
)

MATCHED = "artifacts/matched_sensors/matched-sensors.json"
GAP = "artifacts/sleep_gap/sleep-gap.json"
SLEEP_MAT = "artifacts/sleep_mat/sleep-mat.json"
SLEEP_MAT_SIMULATED = "artifacts/sleep_mat/sleep-mat-simulated.json"
PLANNING = "artifacts/matched_sensors/matched-sensors-planning.json"
POST_HOC = "artifacts/tihm/tihm-alert-burden-post-hoc.json"
GAP_PLAN = "artifacts/sleep_gap/sleep_gap_plan.json"

FEATURES = (
    ("sleeping_hours", "hours of sleep"),
    ("kitchen_activity_hours", "hours in the kitchen"),
    ("bathroom_activity_hours", "hours in the bathroom"),
)
SINCE = (
    ("under_10_minutes", "under\n10 min"),
    ("10_to_60_minutes", "10 to\n60 min"),
    ("1_to_3_hours", "1 to\n3 h"),
    ("3_hours_or_more", "3 h or\nmore"),
)


def load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def interval(entry: dict[str, Any]) -> list[float]:
    """An estimate and its interval, as [estimate, low, high]."""
    return [entry["estimate"], entry["interval"]["low"], entry["interval"]["high"]]


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
        }
    )


def finish(figure: Any, name: str, credit: str) -> None:
    figure.text(0.01, 0.01, credit, fontsize=7, color=INK_2, ha="left", va="bottom")
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.png", dpi=200, metadata={"Software": None})
    plt.close(figure)


# ----------------------------------------------------------------------------
# The matched-sensor study and the sleep-mat comparison
# ----------------------------------------------------------------------------
def matched(numbers: dict[str, Any]) -> None:
    """The simulator's homes under its own sensors and under TIHM-matched ones."""
    record = load(MATCHED)
    results = record["results"]
    configuration = record["configuration"]
    mat = load(SLEEP_MAT)["results"]["analyses"]["off"]["sleep"]
    simulated = load(SLEEP_MAT_SIMULATED)["results"]["sleep"]
    by_profile = results["E2"]["by_profile"]
    numbers["sleep_mat"] = {
        "record": SLEEP_MAT,
        "homes": mat["spearman"]["homes"],
        "spearman": interval(mat["spearman"]),
        "mean_difference_hours": interval(mat["mean_difference"]),
        "simulated_record": SLEEP_MAT_SIMULATED,
        "simulated_spearman": interval(simulated["spearman"]),
    }
    planning = load(PLANNING)["results"]["tihm"]
    post_hoc = load(POST_HOC)["results"]["state_hours"]
    numbers["tihm_sensors"] = {
        "record": PLANNING,
        "homes": planning["homes"],
        "hold_off_seconds": planning["sensor_facts"]["hold_off_seconds"],
        "contacts": planning["sensor_facts"]["contacts"],
        "homes_with_a_hallway": planning["moments"]["hall_per_day"]["homes"],
    }
    numbers["tihm_state_hours"] = {
        "record": POST_HOC,
        "usable_days": post_hoc["usable_days"],
        "pooled_day_median_hours": {
            state: post_hoc["hours"][state]["median"]
            for state in ("kitchen_activity", "bathroom_activity", "sleeping")
        },
    }
    numbers["matched_sensors"] = {
        "record": MATCHED,
        "homes": results["homes"],
        "profiles": {
            name: configuration["profiles"]["settings"][setting]
            for name, setting in (
                ("hold_off_seconds", "hold_off_seconds"),
                ("spill_rate", "spill_rate"),
                ("presence_scale", "presence_scale"),
            )
        },
        "check_reproduced": results["check"]["reproduced"],
        "excess_detection": {
            profile: interval(by_profile[profile]["excess_detection"])
            for profile in ("standard", "matched")
        },
        "E1_matched_minus_standard": interval(results["E1"]),
        "false_alerts_per_home": {
            profile: interval(results["E3"][profile]["false_alerts"])
            for profile in ("standard", "matched")
        },
        "within_home_correlation": {
            feature: {
                profile: interval(results["E4"][feature][profile])
                for profile in ("standard", "matched")
            }
            for feature, _ in FEATURES
        },
        "pooled_day_median_hours": {
            state: {
                "truth": results["E5"][state]["standard"]["pooled_day_median_truth"],
                "standard": results["E5"][state]["standard"][
                    "pooled_day_median_pipeline"
                ],
                "matched": results["E5"][state]["matched"][
                    "pooled_day_median_pipeline"
                ],
            }
            for state in ("kitchen_activity", "bathroom_activity", "sleeping")
        },
        "moments": {
            name: {
                "tihm": results["E6"]["tihm"][name]["median"],
                "standard": results["E6"]["standard"][name]["median"],
                "matched": results["E6"]["matched"][name]["median"],
            }
            for name in results["E6"]["tihm"]
        },
        "criteria": {
            name: entry["reading"] for name, entry in results["criteria"].items()
        },
        "E8_kitchen_belief_after_kitchen_alone": {
            profile: results["E8"][profile]["kitchen"]["mean_belief"][
                "kitchen_activity"
            ]
            for profile in ("standard", "matched")
        },
        "E9_sensitivity_minus_standard": interval(results["E9"]["E1"]),
    }
    plan = load(GAP_PLAN)
    numbers["sleep_gap_plan"] = {
        "record": GAP_PLAN,
        "review_of_the_mat_file": next(
            item for item in plan["inspected_before"] if "75 stretches" in item
        ),
    }

    figure, axis = plt.subplots(figsize=(7.2, 3.9))
    rows = [("tihm", "TIHM homes: hours of sleep\nagainst the sleep mat")]
    rows += [
        (feature, f"Simulated homes: {label}\nagainst the truth")
        for feature, label in FEATURES
    ]
    for k, (name, _) in enumerate(rows):
        y = len(rows) - 1 - k
        if name == "tihm":
            estimate, low, high = interval(mat["spearman"])
            axis.errorbar(
                estimate,
                y,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                color=REAL,
                markersize=7,
                capsize=3,
                lw=1.2,
            )
            continue
        for offset, profile, colour, label in (
            (0.14, "standard", SIMULATED, "the simulator's own sensors"),
            (-0.14, "matched", NEUTRAL, "sensors matched to TIHM's records"),
        ):
            estimate, low, high = interval(results["E4"][name][profile])
            axis.errorbar(
                estimate,
                y + offset,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                color=colour,
                markersize=7,
                capsize=3,
                lw=1.2,
                label=label if name == "sleeping_hours" else None,
            )
    axis.set_yticks(
        range(len(rows)), [label for _, label in reversed(rows)], fontsize=8.5
    )
    axis.set_xlim(0.0, 1.0)
    axis.axvline(0.5, color=INK_2, lw=0.8, ls="--")
    axis.text(0.51, -0.45, "margin of 0.5", fontsize=7.5, color=INK_2, va="center")
    axis.set_ylim(-0.6, len(rows) - 0.5)
    axis.set_xlabel("mean within-home Spearman correlation, over homes")
    axis.set_title("TIHM-like sensors left simulated sleep on track")
    axis.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.35, -0.2),
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    finish(figure, "matched", MIXED_CREDIT)


# ----------------------------------------------------------------------------
# The minute-by-minute description
# ----------------------------------------------------------------------------
def gap(numbers: dict[str, Any]) -> None:
    record = load(GAP)
    results = record["results"]
    clocks = results["clocks"]
    first = clocks["as_recorded"]
    per_home = first["per_home"]
    numbers["sleep_gap"] = {
        "record": GAP,
        "plan_commit": record["environment"]["git_commit"][:7],
        "homes": len(first["homes"]),
        "matched_days": results["check"]["pairs"]["matched_days"],
        "largest_difference_hours": results["check"]["beliefs"][
            "largest_difference_hours"
        ],
        "D1": {
            clock: {
                name: interval(entry) + [entry["median"]]
                for name, entry in clocks[clock]["D1"].items()
            }
            for clock in clocks
        },
        "D3": {
            clock: {
                name: dict(zip(results["states"], entry["mean"]))
                for name, entry in clocks[clock]["D3"].items()
            }
            for clock in clocks
        },
        "D4_since": {
            key: {
                "pipeline_sleep_hours": cell["pipeline_sleep_hours"],
                "hours_with_no_record": cell["hours_with_no_record"],
                "sleeping_belief": interval(cell["sleeping_belief"]),
            }
            for key, cell in first["D4"]["since"].items()
        },
        "D4_sensor": {
            key: {
                "pipeline_sleep_hours": cell["pipeline_sleep_hours"],
                "sleeping_belief": interval(cell["sleeping_belief"]),
            }
            for key, cell in first["D4"]["sensor"].items()
        },
        "D4_exit": {
            key: cell["pipeline_sleep_hours"]
            for key, cell in first["D4"]["exit"].items()
        },
        "D4_exit_an_hour_or_more_after": sum(
            cell["pipeline_sleep_hours"]
            for key, cell in first["D4"]["joint"].items()
            if key.split("|")[0] in ("Front Door", "Back Door")
            and key.split("|")[1] in ("1_to_3_hours", "3_hours_or_more")
        ),
        "D4_period": {
            key: {
                "pipeline_sleep_hours": cell["pipeline_sleep_hours"],
                "hours_with_no_record": cell["hours_with_no_record"],
            }
            for key, cell in first["D4"]["period"].items()
        },
        "D5": {
            clock: {
                "correlations": {
                    name: interval(entry)
                    for name, entry in clocks[clock]["D5"]["correlations"].items()
                },
                "references": {
                    name: interval(entry)
                    for name, entry in clocks[clock]["D5"]["references"].items()
                },
                "excess": {
                    name: interval(entry) + [entry["homes_above_zero"]]
                    for name, entry in clocks[clock]["D5"]["excess"].items()
                },
                "shares": {
                    name: interval(entry)
                    for name, entry in clocks[clock]["D5"]["shares"].items()
                },
            }
            for clock in clocks
        },
        "D6": {
            "left_out": first["D6"]["left_out"],
            "difference": interval(first["D6"]["D1"]["difference"]),
            "pipeline_off_the_mat": interval(first["D6"]["D1"]["pipeline_off_the_mat"]),
            "share_off": interval(first["D6"]["D5"]["shares"]["pipeline_off_the_mat"]),
        },
        "D7_under_30_minutes": {
            clock: clocks[clock]["D7"]["distance"]["under_30_minutes"][
                "pipeline_sleep_hours"
            ]
            for clock in clocks
        },
        "largest_home": max(
            (
                (home, row["difference"], row["pipeline_off_the_mat"])
                for home, row in per_home.items()
            ),
            key=lambda item: item[1],
        ),
    }

    # The day, hour by hour, with the mat's clock as recorded.
    hourly = first["D2"]
    hours = np.arange(24)
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    bottom = np.zeros(24)
    for name, colour, label in (
        ("pipeline_asleep", SIMULATED, "pipeline's sleep, mat says asleep"),
        (
            "pipeline_awake_on_the_mat",
            SIMULATED_LIGHT,
            "pipeline's sleep, mat says awake in bed",
        ),
        ("pipeline_off_the_mat", REAL, "pipeline's sleep, no mat record"),
    ):
        values = np.asarray(hourly[name])
        axis.bar(
            hours + 0.5, values, width=0.92, bottom=bottom, color=colour, label=label
        )
        bottom += values
    in_bed = np.asarray(hourly["mat_in_bed"])
    axis.step(
        np.append(hours, 24),
        np.append(in_bed, in_bed[-1]),
        where="post",
        color=INK,
        lw=1.4,
        label="mat: in bed",
    )
    axis.set_xlim(0, 24)
    axis.set_ylim(0, 1.0)
    axis.set_xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    axis.set_ylabel("hours of each hour, a day")
    axis.set_title("By day, the pipeline's sleep has no mat under it")
    axis.legend(
        frameon=False,
        fontsize=7.5,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.14),
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    finish(figure, "hourly", TIHM_CREDIT)

    # The belief in sleep with no mat record, by how long the home had been quiet.
    since = first["D4"]["since"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    x = np.arange(len(SINCE))
    beliefs = [interval(since[key]["sleeping_belief"]) for key, _ in SINCE]
    left.bar(
        x,
        [b[0] for b in beliefs],
        width=0.6,
        color=REAL,
        yerr=[[b[0] - b[1] for b in beliefs], [b[2] - b[0] for b in beliefs]],
        capsize=3,
        error_kw={"lw": 1.0, "ecolor": INK_2},
    )
    left.set_xticks(x, [label for _, label in SINCE], fontsize=8)
    left.set_ylim(0, 1.05)
    left.set_ylabel("belief in sleeping")
    left.set_title("How sure it is", fontsize=10.5)
    right.bar(
        x,
        [since[key]["pipeline_sleep_hours"] for key, _ in SINCE],
        width=0.6,
        color=REAL,
    )
    right.set_xticks(x, [label for _, label in SINCE], fontsize=8)
    right.set_ylabel("hours a day")
    right.set_title("How much sleep it counts", fontsize=10.5)
    figure.supxlabel(
        "time since the home's last activation, minutes with no mat record",
        fontsize=9,
        color=INK_2,
        y=0.08,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    finish(figure, "quiet", TIHM_CREDIT)

    # Each home's correlation beside what the mat's own hours give.
    figure, axis = plt.subplots(figsize=(5.6, 5.0))
    for part, colour, label in (
        ("pipeline_on_the_mat", SIMULATED, "sleep counted while the mat has a record"),
        ("pipeline_off_the_mat", REAL, "sleep counted with no mat record"),
    ):
        name = f"mat_sleep~{part}"
        xs = [row[f"reference[{name}]"] for row in per_home.values()]
        ys = [row[f"r[{name}]"] for row in per_home.values()]
        axis.scatter(
            xs,
            ys,
            s=36,
            color=colour,
            label=label,
            zorder=3,
            edgecolor=SURFACE,
            linewidth=0.8,
        )
    axis.plot([-1, 1], [-1, 1], color=INK_2, lw=0.9, ls="--", zorder=2)
    axis.text(
        0.95,
        -0.9,
        "on the dashed line: no tracking\nbeyond the mat's own hours",
        fontsize=7.5,
        color=INK_2,
        ha="right",
        va="bottom",
    )
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.set_aspect("equal")
    axis.set_xlabel("swapped-mask reference: other days' beliefs, same mat")
    axis.set_ylabel("correlation with the mat's hours asleep")
    axis.set_title(
        "Each home twice: the mat's own hours\ngive most of each correlation",
        fontsize=10.5,
    )
    axis.legend(frameon=False, fontsize=7.5, loc="upper left")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    finish(figure, "masked", TIHM_CREDIT)


def main() -> None:
    style()
    numbers: dict[str, Any] = {}
    matched(numbers)
    gap(numbers)
    (HERE / "numbers.json").write_text(
        json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
