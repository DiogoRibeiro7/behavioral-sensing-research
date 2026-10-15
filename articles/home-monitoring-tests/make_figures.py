"""Figures and numbers for the article on testing a home monitoring pipeline.

Everything is read from the records committed under ``artifacts/``; nothing is
recomputed from the TIHM dataset, which is not redistributed. Run from the
repository's root::

    python articles/home-monitoring-tests/make_figures.py

It writes four PNG figures to ``articles/home-monitoring-tests/figures`` and,
beside this script, ``numbers.json``: every number the article quotes from the
records, unrounded, under the record it comes from. The article rounds half
away from zero. Numbers from the cited literature are not in it.

TIHM is by Palermo et al., Scientific Data 10, 606 (2023), under CC BY 4.0.
"""

from __future__ import annotations

import json
import statistics
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
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
REAL = "#eb6834"
NEUTRAL = "#8f8e8a"
TIHM_CREDIT = (
    "Author's figure from the repository's records. TIHM data: Palermo et al., "
    "Scientific Data 10, 606 (2023), CC BY 4.0."
)
SYNTHETIC_CREDIT = (
    "Author's figure from the repository's records. Synthetic days, no TIHM data."
)

AT_3 = "3"


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


def simulator(numbers: dict[str, Any]) -> None:
    """The 400 simulated homes of the threshold-calibration record."""
    record = load("artifacts/threshold_calibration/threshold-calibration.json")
    results = record["results"]
    person_days = results["person_days"]
    default = results["detection"]["change"]["default@1"]
    stable = results["stable"]["default@1"]
    changed_sleep = results["verdicts"]["change"]["default@1"]["by_feature"]
    matched = results["matched"]
    c2 = results["criteria"]["C2_the_threshold_means_what_it_says"]["E3"][AT_3]
    curves = results["curves"]
    numbers["simulator"] = {
        "record": "artifacts/threshold_calibration/threshold-calibration.json",
        "step_change": record["configuration"]["simulator"]["arms"]["change"],
        "window": record["configuration"]["definitions"]["window"],
        "homes": results["homes"],
        "days": results["days"],
        "person_days": person_days,
        "detected": {
            "count": default["detected"]["count"],
            "share": interval(default["detected"]),
            "median_delay_days": statistics.median(default["delay_days"]["delays"]),
        },
        "false_detections": {
            "count": default["false_detections"]["count"],
            "share": interval(default["false_detections"]),
        },
        "stable_arm": {
            "behavioural_alerts": stable["behavioural_alerts"],
            "sleeping_hours_alerts": stable["by_feature"]["sleeping_hours"],
            "per_person_day": stable["behavioural_alerts"] / person_days,
            "sleep_per_person_day": stable["by_feature"]["sleeping_hours"]
            / person_days,
            "false_alerts_per_home": stable["per_home"]["estimate"],
        },
        "changed_arm": {
            "sleeping_hours_alerts": changed_sleep["sleeping_hours"],
            "sleep_per_person_day": changed_sleep["sleeping_hours"] / person_days,
        },
        "calibrated_at_the_declared_thresholds": {
            "false_alerts": results["stable"]["calibrated@1"]["behavioural_alerts"],
            "excess_detection": curves["calibrated"]["1"]["change"]["excess_detection"][
                "estimate"
            ],
            "default_excess_detection": curves["default"]["1"]["change"][
                "excess_detection"
            ]["estimate"],
            "share_past_3_on_simulated_sleep": interval(c2),
        },
        "at_the_same_false_alerts": {
            "false_alerts_per_home": matched["default"]["false_alerts_per_home"],
            "calibrated": matched["excess_detection"]["change"][
                "calibrated_at_the_match"
            ],
            "default": matched["excess_detection"]["change"]["default"],
            "difference": interval(matched["excess_detection"]["change"]),
            "small_change_difference": interval(
                matched["excess_detection"]["small_change"]
            ),
            "gradual_change_difference": interval(
                matched["excess_detection"]["gradual_change"]
            ),
        },
    }


def alert_burden(numbers: dict[str, Any]) -> None:
    """The protocol's run on TIHM and the descriptions made after it."""
    results = load("artifacts/tihm/tihm-alert-burden.json")["results"]
    post_hoc = load("artifacts/tihm/tihm-alert-burden-post-hoc.json")["results"]
    burden = results["burden"]
    a1 = results["association"]["A1_alert_days"]
    labels = results["labels"]["primary"]
    silent = post_hoc["silent_days"]
    direction = post_hoc["direction"]["event_count_z"]
    calendar = post_hoc["calendar"]
    simulated = numbers["simulator"]
    sleep_rate = burden["B2_simulator_feature_per_monitored_day"]["estimate"]
    numbers["alert_burden"] = {
        "records": [
            "artifacts/tihm/tihm-alert-burden.json",
            "artifacts/tihm/tihm-alert-burden-post-hoc.json",
        ],
        "homes": len(results["households"]),
        "monitored_days": results["monitoring"]["monitored_days"],
        "behavioural_alerts": burden["behavioural_alerts"],
        "sleeping_hours_alerts": burden["by_feature"]["sleeping_hours"],
        "per_monitored_day": interval(burden["B1_per_monitored_day"]),
        "sleep_per_monitored_day": interval(
            burden["B2_simulator_feature_per_monitored_day"]
        ),
        "sleep_rate_over_the_simulators_stable_arm": sleep_rate
        / simulated["stable_arm"]["sleep_per_person_day"],
        "sleep_rate_over_the_simulators_changed_arm": sleep_rate
        / simulated["changed_arm"]["sleep_per_person_day"],
        "protocol_simulator_reference": {
            **burden["simulator_reference"],
            "ratio_to_stable_arm": sleep_rate
            / burden["simulator_reference"]["stable_arm"],
            "ratio_to_changed_arm": sleep_rate
            / burden["simulator_reference"]["changed_arm"],
        },
        "system_health_alerts": results["monitoring"]["system_health_alerts"],
        "data_quality_alerts": results["monitoring"]["data_quality_alerts"],
        "A1": {
            "label_days": a1["label_days"],
            "label_days_with_an_alert": a1["label_days_flagged"],
            "share_of_label_days": a1["share_of_label_days_flagged"]["estimate"],
            "share_of_other_days": a1["share_of_other_days_flagged"]["estimate"],
            "difference": interval(a1["difference"]),
        },
        "A3_concordance": interval(results["association"]["A3_deviation_score"]),
        "agitation_labels": {
            "label_days_in_the_dataset": labels["label_days"],
            "share_labelled_after_a_label_day": labels[
                "share_labelled_after_a_label_day"
            ],
            "share_labelled_after_another_day": labels[
                "share_labelled_after_another_day"
            ],
            "five_largest_homes_share": labels["five_largest_households_share"],
        },
        "post_hoc": {
            "status": post_hoc["status"],
            "silent_days": silent["silent_days"],
            "silent_days_counted_usable": silent["usable"],
            "median_sleeping_hours_on_silent_days": silent["median_sleeping_hours"][
                "silent_days"
            ],
            "median_home_sleeping_hours": post_hoc["state_hours"][
                "household_median_sleeping_hours"
            ]["median"],
            "most_silent_date": calendar["most_silent_date"],
            "day_after": calendar["day_after_the_most_silent_date"],
            "deviating_days_mean_event_count_z": direction["deviating_days"]["mean"],
            "label_days_mean_event_count_z": direction["label_days"]["mean"],
            "silent_share_of_deviating_days": silent["share_silent"]["deviating_days"],
        },
    }


def silent_home(numbers: dict[str, Any]) -> None:
    """The simulated test of the silent-home rule and its description on TIHM."""
    record = load("artifacts/silent_home/silent-home.json")
    results = record["results"]
    criteria = results["criteria"]
    detection = results["detection"]
    tihm = load("artifacts/silent_home/silent-home-tihm.json")["results"]
    h12 = tihm["conditions"]["h12"]
    numbers["silent_home"] = {
        "records": [
            "artifacts/silent_home/silent-home.json",
            "artifacts/silent_home/silent-home-tihm.json",
        ],
        "outage": record["configuration"]["simulator"]["arms"]["outage"],
        "homes": results["homes"],
        "E1_extra_alerts_with_the_rule_off": [
            criteria["C1"]["E1"],
            criteria["C1"]["interval"]["low"],
            criteria["C1"]["interval"]["high"],
        ],
        "E3_alerts_removed_from_the_window": [
            criteria["C2"]["E3"],
            criteria["C2"]["E3_interval"]["low"],
            criteria["C2"]["E3_interval"]["high"],
        ],
        "E2_extra_alerts_with_the_rule_on": [
            criteria["C2"]["E2"],
            criteria["C2"]["E2_interval"]["low"],
            criteria["C2"]["E2_interval"]["high"],
        ],
        "rule_horizon_hours": record["configuration"]["fleet"]["horizon_hours"],
        "rule_conditions": record["configuration"]["the_rule"]["conditions"],
        "stable_person_days": results["stable"]["person_days"],
        "stable_silence_alerts_at_12_hours": results["stable"]["h12"]["silence_alerts"],
        "stable_homes_in_which_the_rule_changes_an_alert": criteria["C3"][
            "homes_in_which_the_rule_changes_an_alert"
        ],
        "silence_alerts_per_outage_home": results["reporting"]["h12"][
            "silence_alerts_per_home"
        ]["median"],
        "cooldown_hours": record["configuration"]["simulator"]["pipeline"][
            "alert_policy"
        ]["cooldown_hours"],
        "detected_after_an_outage": {
            "rule_off": detection["change_after_outage"]["off"]["detected"]["count"],
            "rule_on": detection["change_after_outage"]["h12"]["detected"]["count"],
            "C5_verdict": criteria["C5"]["verdict"],
        },
        "detected_without_an_outage": detection["change"]["off"]["detected"]["count"],
        "outage_alone_meets_the_detection_definition": {
            "rule_off": detection["change_after_outage"]["off"]["false_detections"][
                "count"
            ],
            "rule_on": detection["change_after_outage"]["h12"]["false_detections"][
                "count"
            ],
        },
        "tihm_rule_at_12_hours": {
            "removed": h12["behavioural_alerts"]["raised_only_with_the_rule_off"],
            "added": h12["behavioural_alerts"]["raised_only_under_this_condition"],
            "left": h12["behavioural_alerts"]["all"],
            "days_refused": h12["days_refused_because_of_the_rule"]["all"],
            "homes_with_a_refused_day": h12["days_refused_because_of_the_rule"][
                "homes_with_one"
            ],
            "silence_alerts": h12["silence_alerts"]["all"],
            "silence_alerts_inside_a_common_silence": h12["silence_alerts"][
                "inside_a_stretch_of_common_silence"
            ],
            "silence_alerts_per_monitored_day": h12["silence_alerts"]["all"]
            / h12["monitored_days"],
            "common_silences": [
                {
                    "homes": s["homes"],
                    "monitored": s["monitored"],
                    "since": s["since"],
                    "until": s["until"],
                }
                for s in tihm["fleet"]["stretches"]
            ],
        },
    }


def threshold_tihm(numbers: dict[str, Any]) -> None:
    """The TIHM description of the calibrated reference."""
    rules = load("artifacts/threshold_calibration/threshold-calibration-tihm.json")[
        "results"
    ]["rules"]
    stated = numbers["thresholds"]["stated_at_3"]
    shares = {
        rule: {
            reference: rules[rule]["tail"][reference]["sleeping_hours"][AT_3]["share"]
            for reference in ("default", "calibrated")
        }
        for rule in ("off", "h12")
    }
    numbers["thresholds"]["tihm_sleep_share_past_3"] = {
        "record": "artifacts/threshold_calibration/threshold-calibration-tihm.json",
        "rule_off": shares["off"],
        "rule_on_at_12_hours": shares["h12"],
        "calibrated_over_stated_rule_off": shares["off"]["calibrated"] / stated,
        "calibrated_over_stated_rule_on": shares["h12"]["calibrated"] / stated,
    }


def sleep_mat(numbers: dict[str, Any]) -> None:
    tihm = load("artifacts/sleep_mat/sleep-mat.json")
    simulated = load("artifacts/sleep_mat/sleep-mat-simulated.json")
    real_rows = [r for r in tihm["household_metrics"]["off"].values() if r["included"]]
    real = np.array([r["sleep_spearman"] for r in real_rows])
    sim_rows = [r for r in simulated["results"]["homes"].values() if r["included"]]
    sim = np.array([r["sleep"]["spearman"] for r in sim_rows])
    criteria = tihm["results"]["criteria"]
    e1 = criteria["C1_the_pipeline_follows_the_mat"]["estimate"]
    s1 = simulated["results"]["sleep"]["spearman"]
    rng = np.random.default_rng(7)
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.grid(axis="y", visible=False)
    for row, values, colour, summary in (
        (1.0, sim, SIMULATED, s1),
        (0.0, real, REAL, e1),
    ):
        jitter = rng.uniform(-0.16, 0.16, values.size)
        axis.scatter(
            values,
            row + jitter,
            s=26,
            color=colour,
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
        )
        low, high = summary["interval"]["low"], summary["interval"]["high"]
        axis.plot([low, high], [row - 0.27, row - 0.27], color=INK, lw=2, zorder=4)
        axis.plot([summary["estimate"]], [row - 0.27], "o", color=INK, ms=6, zorder=5)
        axis.text(
            high + 0.02,
            row - 0.27,
            f"mean {summary['estimate']:.2f} [{low:.2f}, {high:.2f}]",
            va="center",
            fontsize=8.5,
            color=INK,
        )
    axis.axvline(0.5, color=INK_2, lw=1)
    axis.text(0.505, 1.42, "declared margin, 0.5", fontsize=8.5, color=INK_2)
    axis.axvline(0.0, color=GRID, lw=1)
    axis.set_yticks(
        [1.0, 0.0],
        [
            f"{sim.size} simulated homes\nagainst true sleep",
            f"{real.size} TIHM homes\nagainst the sleep mat",
        ],
    )
    axis.set_ylim(-0.55, 1.55)
    axis.set_xlim(-0.35, 1.0)
    axis.set_xlabel(
        "within-home Spearman correlation of the pipeline's daily hours of sleep"
    )
    axis.set_title("The same estimand in the simulator and in real homes")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    finish(figure, "sleep-mat-correlation", TIHM_CREDIT)

    results = tihm["results"]
    primary = results["analyses"]["off"]
    extreme = results["homes"]["f220c"]["off"]
    e6 = results["E6_deviations"]
    e8 = results["E8_silent_days"]
    e12 = results["E12_staged_asleep"]
    lags = results["E11_clocks"]["lags"]
    most_negative = min(lags, key=lambda lag: lags[lag]["spearman"]["estimate"])
    staged = {
        home: row["staged_asleep_share"] for home, row in results["homes"].items()
    }
    margin = criteria["C1_the_pipeline_follows_the_mat"]["margin"]
    protocol = load("artifacts/sleep_mat/sleep_mat_protocol.json")
    numbers["sleep_mat"] = {
        "records": [
            "artifacts/sleep_mat/sleep-mat.json",
            "artifacts/sleep_mat/sleep-mat-simulated.json",
        ],
        "mat_homes": len(results["homes"]),
        "included_home": protocol["definitions"]["included_home"],
        "reviews_before_the_freeze": [
            entry for entry in protocol["inspected_before"] if "review" in entry
        ],
        "real_homes": int(real.size),
        "simulated_homes": int(sim.size),
        "margin": margin,
        "three_deviations_at_the_margin": 3 * margin,
        "three_deviations_at_e1": 3 * e1["estimate"],
        "e1": interval(e1),
        "s1": interval(s1),
        "real_min_max": [float(real.min()), float(real.max())],
        "simulated_min_max": [float(sim.min()), float(sim.max())],
        "e2_sleep_mean_difference": interval(
            criteria["C2_the_pipeline_agrees_in_level"]["estimate"]
        ),
        "e3_in_bed_mean_difference": interval(primary["in_bed"]["mean_difference"]),
        "median_home_mean_difference": statistics.median(
            r["sleep_mean_difference"] for r in real_rows
        ),
        "extreme_home": {
            "home": "f220c",
            "staged_asleep_share": staged["f220c"],
            "mean_difference": extreme["sleep"]["mean_difference"],
            "pipeline_median": extreme["sleep"]["median_pipeline"],
            "mat_sleep_median": extreme["sleep"]["median_mat"],
            "mat_in_bed_median": extreme["in_bed"]["median_mat"],
        },
        "e6_deviations": {
            "spearman": interval(e6["spearman"]),
            "homes": e6["spearman"]["homes"],
            "pipeline_past_threshold": e6["pipeline_past_threshold"],
            "same_sign": e6["same_sign"],
            "also_past_threshold": e6["also_past_threshold"],
        },
        "e8_silent_days": {
            "mat_observed": e8["mat_observed"],
            "pipeline_median": e8["median_pipeline_hours"],
            "mat_sleep_median": e8["median_mat_sleep_hours"],
        },
        "e10_mat_clock": {
            lag: {
                "spearman": entry["sleep"]["spearman"]["estimate"],
                "homes": len(entry["homes"]),
            }
            for lag, entry in results["E10_alignment"].items()
        },
        "e11_hourly": {
            "at_zero": interval(lags["+0"]["spearman"]),
            "most_negative_lag": most_negative,
            "most_negative": lags[most_negative]["spearman"]["estimate"],
        },
        "e12_without_the_staged_awake_homes": {
            "share": e12["share"],
            "left_out": {home: staged[home] for home in e12["left_out"]},
            "spearman": interval(e12["sleep"]["spearman"]),
            "mean_difference": interval(e12["sleep"]["mean_difference"]),
        },
    }


def calendar(numbers: dict[str, Any]) -> None:
    record = load("artifacts/silent_home/silent-home-tihm.json")["results"]
    off = record["conditions"]["off"]["behavioural_alerts"]["by_day_summarised"]
    on = record["conditions"]["h12"]["behavioural_alerts"]["by_day_summarised"]
    first, last = date(2019, 4, 1), date(2019, 6, 30)
    days = [first + timedelta(days=k) for k in range((last - first).days + 1)]
    figure, axis = plt.subplots(figsize=(7.2, 3.4))
    for series, colour, label in (
        (off, REAL, "silent-home rule off (the default)"),
        (on, SIMULATED, "rule on at 12 hours"),
    ):
        values = [series.get(day.isoformat(), 0) for day in days]
        axis.plot(
            days, values, color=colour, lw=2, solid_joinstyle="round", label=label
        )
    peak_day = max(off, key=off.__getitem__)
    peak = off[peak_day]
    axis.annotate(
        f"{peak} alerts about {int(peak_day[8:])} June, the day after\n"
        "every monitored home went silent",
        xy=(date.fromisoformat(peak_day), peak),
        xytext=(date(2019, 4, 3), peak - 3),
        fontsize=8.5,
        color=INK,
        va="center",
        arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 1},
    )
    axis.set_ylabel("behavioural alerts per day")
    axis.set_title("Alerts on 56 TIHM homes, by the day they were about")
    axis.legend(frameon=False, fontsize=8.5, loc="center left")
    axis.set_ylim(0, peak + 4)
    axis.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=(1, 15)))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    finish(figure, "alerts-by-day", TIHM_CREDIT)
    numbers["calendar"] = {
        "record": "artifacts/silent_home/silent-home-tihm.json",
        "off_total": sum(off.values()),
        "on_total": sum(on.values()),
        "peak_day": peak_day,
        "peak": peak,
        "on_max": max(on.values()),
    }


def thresholds(numbers: dict[str, Any]) -> None:
    record = load("artifacts/threshold_calibration/threshold-null.json")
    results = record["results"]
    levels = ["1.5", "1.8", "2", "2.5", AT_3]
    x = [float(t) for t in levels]
    rates = results["rates"]["none"]
    stated = [100 * results["nominal"][t] for t in levels]
    default = [100 * rates["default"]["phase"]["all"][t]["rate"] for t in levels]
    calibrated = [100 * rates["calibrated"]["phase"]["all"][t]["rate"] for t in levels]
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.plot(x, stated, color=INK_2, lw=1.5, label="what the threshold states")
    axis.plot(
        x,
        default,
        "o-",
        color=REAL,
        lw=2,
        ms=6,
        mec=SURFACE,
        mew=1.5,
        label="default reference",
    )
    axis.plot(
        x,
        calibrated,
        "o-",
        color=SIMULATED,
        lw=2,
        ms=6,
        mec=SURFACE,
        mew=1.5,
        label="calibrated reference",
    )
    axis.set_yscale("log")
    axis.set_yticks(
        [0.1, 0.3, 1, 3, 10, 30], ["0.1%", "0.3%", "1%", "3%", "10%", "30%"]
    )
    axis.minorticks_off()
    axis.set_xticks(x, [f"{t:g}" for t in x])
    axis.set_xlabel("deviation threshold, in standard deviations")
    axis.set_ylabel("days at or past it")
    axis.text(3.02, default[-1], f"{default[-1]:.2f}%", va="center", fontsize=8.5)
    axis.text(
        3.02,
        calibrated[-1] * 1.12,
        f"{calibrated[-1]:.2f}%",
        va="bottom",
        fontsize=8.5,
    )
    axis.text(
        3.02,
        stated[-1] * 0.88,
        f"{stated[-1]:.2f}%",
        va="top",
        fontsize=8.5,
        color=INK_2,
    )
    axis.set_xlim(1.4, 3.3)
    axis.set_title("Synthetic days with nothing in them that pass the threshold")
    axis.legend(frameon=False, fontsize=8.5, loc="lower left")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    finish(figure, "threshold-null", SYNTHETIC_CREDIT)
    default_rate = rates["default"]["phase"]["all"][AT_3]["rate"]
    numbers["thresholds"] = {
        "record": "artifacts/threshold_calibration/threshold-null.json",
        "recorded_at": record["recorded_at"],
        "weekday_min_samples": record["resolved_defaults"]["baseline"][
            "weekday_min_samples"
        ],
        "stated_at_3": results["nominal"][AT_3],
        "default_at_3": default_rate,
        "default_at_3_with_four_days_of_the_weekday": rates["default"][
            "by_weekday_size"
        ]["4"][AT_3]["rate"],
        "calibrated_at_3": rates["calibrated"]["phase"]["all"][AT_3]["rate"],
        "ratio_default_to_stated": default_rate / results["nominal"][AT_3],
        "gaussian_threshold_passed_at_the_default_rate": statistics.NormalDist().inv_cdf(
            1 - default_rate / 2
        ),
    }


def rules(numbers: dict[str, Any]) -> None:
    record = load("artifacts/tihm/tihm-alert-burden.json")["results"]["references"]
    label_days = record["label_days"]
    bars = [
        (
            "the pipeline's deviating days",
            record["pipeline_deviating_days"]["label_days_caught"],
            REAL,
        ),
        (
            "days busier than the home's\nown earlier days",
            record["event_count"]["label_days_caught"],
            NEUTRAL,
        ),
        (
            "days in the homes with the most\nearlier labels (no sensor used)",
            record["label_history"]["label_days_caught"],
            NEUTRAL,
        ),
    ]
    figure, axis = plt.subplots(figsize=(7.2, 3.1))
    axis.grid(axis="y", visible=False)
    for place, (label, value, colour) in enumerate(bars):
        axis.barh(place, value, height=0.42, color=colour)
        axis.text(value + 1, place, f"{value:.0f}", va="center", fontsize=9)
    chance = record["random_expected_caught"]
    axis.axvline(chance, color=INK, lw=1)
    axis.text(chance + 0.8, 2.48, f"chance, {chance:.0f}", fontsize=8.5, color=INK)
    axis.set_yticks(range(len(bars)), [b[0] for b in bars])
    axis.set_xlim(0, label_days)
    axis.set_ylim(-0.5, 2.8)
    axis.set_xlabel(
        f"verified agitation label days caught, of {label_days}, "
        f"each flagging {record['flags']} days"
    )
    figure.suptitle(
        "The pipeline against two rules with no model",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    finish(figure, "rules", TIHM_CREDIT)
    numbers["rules"] = {
        "record": "artifacts/tihm/tihm-alert-burden.json",
        "label_days": label_days,
        "flags": record["flags"],
        "pipeline": record["pipeline_deviating_days"]["label_days_caught"],
        "event_count": record["event_count"]["label_days_caught"],
        "label_history": record["label_history"]["label_days_caught"],
        "chance": chance,
    }


def main() -> None:
    style()
    numbers: dict[str, Any] = {}
    simulator(numbers)
    alert_burden(numbers)
    rules(numbers)
    silent_home(numbers)
    calendar(numbers)
    thresholds(numbers)
    threshold_tihm(numbers)
    sleep_mat(numbers)
    (HERE / "numbers.json").write_text(
        json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(numbers, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
