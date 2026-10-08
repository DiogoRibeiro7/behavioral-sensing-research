"""Planning trials for the sleep-mat protocol, on synthetic data.

The protocol judges two means over homes against margins, with 14 or so homes.
Before it was frozen, the criteria were tried here on synthetic studies with
the mat homes' numbers of days, to choose the interval and to say how often
each reading comes out when the truth is known. Nothing here reads TIHM.

**Tracking (C1).** Each synthetic home's days are jointly Gaussian with a
correlation drawn around a centre, with a spread between homes on Fisher's
scale. Optionally a share of days has its pipeline value replaced by one far
above every other, as a day with no sensor event is read as a whole day
asleep. The estimand is the mean over homes of the within-home Spearman
correlation, and its true value for a scenario is computed on many more homes.

**Level (C2).** Each home has a bias drawn around a centre, and its days
differ from the bias by Gaussian noise. The estimand is the mean over homes
of the home's mean difference, whose true value is the centre.

**Outlying homes.** In three of the mat homes the mat stages less than half
of the time in bed as asleep, where every other included home is above 0.79.
If the mat's stages are what differs in those homes, they would follow the
pipeline less and differ from it more. Every scenario is also run with those
three homes given no correlation and four more hours of bias.

Two intervals over homes are compared: a percentile bootstrap and a Student
t interval. The record is ``artifacts/sleep_mat/sleep-mat-planning.json``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ..evaluation import ExperimentRecord
from ..fusion import ONLINE, NotEnumerated

EXPERIMENT = "sleep-mat-planning"
RESULT_SCHEMA = "sleep-mat-planning/1"

#: The two intervals compared.
PERCENTILE = "percentile_bootstrap"
STUDENT = "student_t"
METHODS = (PERCENTILE, STUDENT)


@dataclass(frozen=True)
class Planning:
    """The planning trials.

    Attributes
    ----------
    seed_root
        Every draw comes from ``numpy.random.default_rng([seed_root, ...])``.
    replications, resamples
        Synthetic studies per scenario, and bootstrap resamples per study.
    truth_homes
        Synthetic homes per number of days used to compute a true value.
    days
        Each synthetic home's number of days: the mat homes with at least 14
        days whose neighbouring days also have a record, from the mat's file
        alone.
    centres, spreads, contamination
        The tracking scenarios: the correlation of a typical home, the spread
        of homes about it on Fisher's scale, and the share of days replaced.
    biases, bias_spreads, within_sd
        The level scenarios: the typical bias in hours, the spread of homes'
        biases, and the spread of a home's days about its bias.
    outliers, outlier_positions, outlier_centre, outlier_extra_bias
        Whether the outlying homes are tried; which homes they are, by place
        in ``days``; their correlation; and the bias added to theirs.
    tracking_margin, level_margin, confidence
        The protocol's margins and level.
    """

    seed_root: int = 20261009
    replications: int = 1000
    resamples: int = 2000
    truth_homes: int = 4000
    days: tuple[int, ...] = (55, 52, 64, 75, 41, 53, 59, 79, 66, 52, 17, 27, 69, 47)
    centres: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    spreads: tuple[float, ...] = (0.0, 0.3, 0.6)
    contamination: tuple[float, ...] = (0.0, 0.03)
    biases: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 3.0)
    bias_spreads: tuple[float, ...] = (0.5, 1.0, 2.0)
    within_sd: float = 1.5
    outliers: tuple[bool, ...] = (False, True)
    outlier_positions: tuple[int, ...] = (0, 10, 13)
    outlier_centre: float = 0.0
    outlier_extra_bias: float = 4.0
    tracking_margin: float = 0.5
    level_margin: float = 1.0
    confidence: float = 0.95

    def __post_init__(self) -> None:
        """Validate the trials."""
        if len(self.days) < 3 or min(self.days) < 3:
            raise ValueError("at least three homes of at least three days")
        if self.replications < 100:
            raise ValueError("a planning trial needs at least 100 replications")
        if any(not 0 <= k < len(self.days) for k in self.outlier_positions):
            raise ValueError("an outlying home must be one of the homes")


def _ranks(values: np.ndarray) -> np.ndarray:
    """Ranks along the last axis, ties averaged."""
    return np.asarray(stats.rankdata(values, axis=-1), dtype=float)


def spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman's correlation of each row of *x* with the same row of *y*."""
    rx, ry = _ranks(x), _ranks(y)
    rx = rx - rx.mean(axis=-1, keepdims=True)
    ry = ry - ry.mean(axis=-1, keepdims=True)
    return np.asarray(
        (rx * ry).sum(axis=-1)
        / np.sqrt((rx * rx).sum(axis=-1) * (ry * ry).sum(axis=-1)),
        dtype=float,
    )


def _pairs(
    rng: np.random.Generator, rho: np.ndarray, days: int, contamination: float
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic days of homes with correlations *rho*, one home a row."""
    z = rng.standard_normal((rho.size, days, 2))
    x = z[..., 0]
    y = rho[:, None] * z[..., 0] + np.sqrt(1.0 - rho[:, None] ** 2) * z[..., 1]
    if contamination > 0.0:
        replaced = rng.random(x.shape) < contamination
        x = np.where(replaced, 10.0, x)
    return x, y


def _home_correlations(
    rng: np.random.Generator,
    planning: Planning,
    centre: float,
    spread: float,
    contamination: float,
    homes_per_size: int = 1,
    outliers: bool = False,
) -> np.ndarray:
    """Within-home Spearman correlations, ``homes_per_size`` rows per size."""
    out = np.empty((homes_per_size, len(planning.days)))
    for column, days in enumerate(planning.days):
        typical = (
            planning.outlier_centre
            if outliers and column in planning.outlier_positions
            else centre
        )
        rho = np.tanh(
            np.arctanh(typical) + spread * rng.standard_normal(homes_per_size)
        )
        x, y = _pairs(rng, rho, days, contamination)
        out[:, column] = spearman_rows(x, y)
    return out


def intervals(
    values: np.ndarray, rng: np.random.Generator, planning: Planning
) -> dict[str, tuple[float, float]]:
    """Both intervals for the mean of *values*, one value per home."""
    homes = values.size
    tail = (1.0 - planning.confidence) / 2.0
    means = values[rng.integers(0, homes, (planning.resamples, homes))].mean(axis=1)
    outside = int(np.floor(round(tail * planning.resamples, 9)))
    ordered = np.sort(means)
    percentile = (
        float(ordered[outside]),
        float(ordered[planning.resamples - 1 - outside]),
    )
    centre = float(values.mean())
    half = float(
        stats.t.ppf(1.0 - tail, homes - 1) * values.std(ddof=1) / np.sqrt(homes)
    )
    return {PERCENTILE: percentile, STUDENT: (centre - half, centre + half)}


def tracking_verdict(low: float, high: float, margin: float) -> str:
    """C1's reading of an interval."""
    if low > margin:
        return "follows"
    if high < margin:
        return "does_not_follow"
    return "inconclusive"


def level_verdict(low: float, high: float, margin: float) -> str:
    """C2's reading of an interval."""
    if -margin < low and high < margin:
        return "agrees"
    if high < -margin or low > margin:
        return "does_not_agree"
    return "inconclusive"


def _summary(
    found: dict[str, list[tuple[float, float]]],
    truth: float,
    reading: Any,
    margin: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method, bounds in found.items():
        verdicts = [reading(low, high, margin) for low, high in bounds]
        out[method] = {
            "coverage": float(np.mean([low <= truth <= high for low, high in bounds])),
            "mean_width": float(np.mean([high - low for low, high in bounds])),
            "verdicts": {
                name: verdicts.count(name) / len(verdicts)
                for name in sorted(set(verdicts))
            },
        }
    return out


def tracking_cell(
    planning: Planning,
    centre: float,
    spread: float,
    contamination: float,
    key: int,
    outliers: bool = False,
) -> dict[str, Any]:
    """One tracking scenario: its true value and how each interval reads it."""
    truth_rng = np.random.default_rng([planning.seed_root, key, 0])
    truth = float(
        _home_correlations(
            truth_rng,
            planning,
            centre,
            spread,
            contamination,
            planning.truth_homes,
            outliers,
        ).mean()
    )
    rng = np.random.default_rng([planning.seed_root, key, 1])
    studies = _home_correlations(
        rng, planning, centre, spread, contamination, planning.replications, outliers
    )
    found: dict[str, list[tuple[float, float]]] = {m: [] for m in METHODS}
    for row in studies:
        for method, bounds in intervals(row, rng, planning).items():
            found[method].append(bounds)
    return {
        "centre": centre,
        "spread": spread,
        "contamination": contamination,
        "outliers": outliers,
        "truth": truth,
        **_summary(found, truth, tracking_verdict, planning.tracking_margin),
    }


def level_cell(
    planning: Planning, bias: float, spread: float, key: int, outliers: bool = False
) -> dict[str, Any]:
    """One level scenario: how each interval reads a known bias.

    With the outlying homes, the true value is the mean over homes of their
    expected biases, which the extra bias of three homes raises.
    """
    rng = np.random.default_rng([planning.seed_root, key, 2])
    extra = np.zeros(len(planning.days))
    if outliers:
        extra[list(planning.outlier_positions)] = planning.outlier_extra_bias
    truth = float(bias + extra.mean())
    found: dict[str, list[tuple[float, float]]] = {m: [] for m in METHODS}
    for _ in range(planning.replications):
        home_bias = bias + extra + spread * rng.standard_normal(len(planning.days))
        means = np.array(
            [
                (b + planning.within_sd * rng.standard_normal(days)).mean()
                for b, days in zip(home_bias, planning.days)
            ]
        )
        for method, bounds in intervals(means, rng, planning).items():
            found[method].append(bounds)
    return {
        "bias": bias,
        "spread": spread,
        "outliers": outliers,
        "truth": truth,
        **_summary(found, truth, level_verdict, planning.level_margin),
    }


def _range(cells: Sequence[dict[str, Any]], method: str) -> dict[str, float]:
    values = [cell[method]["coverage"] for cell in cells]
    return {"lowest": min(values), "highest": max(values)}


def plan(planning: Planning) -> dict[str, Any]:
    """Every scenario of both criteria, and the coverage each interval reached."""
    tracking: list[dict[str, Any]] = []
    key = 0
    for outliers in planning.outliers:
        for contamination in planning.contamination:
            for spread in planning.spreads:
                for centre in planning.centres:
                    key += 1
                    tracking.append(
                        tracking_cell(
                            planning, centre, spread, contamination, key, outliers
                        )
                    )
    level: list[dict[str, Any]] = []
    for outliers in planning.outliers:
        for spread in planning.bias_spreads:
            for bias in planning.biases:
                key += 1
                level.append(level_cell(planning, bias, spread, key, outliers))
    every = tracking + level
    coverage = {method: _range(every, method) for method in METHODS}
    chosen = max(METHODS, key=lambda method: coverage[method]["lowest"])
    return {
        "tracking": tracking,
        "level": level,
        "coverage": coverage,
        "coverage_tracking": {method: _range(tracking, method) for method in METHODS},
        "coverage_level": {method: _range(level, method) for method in METHODS},
        "chosen": chosen,
        "rule": "the interval whose lowest coverage over every scenario is "
        "the higher is chosen",
    }


def run_planning(
    planning: Planning | None = None, *, output_dir: Path | None = None
) -> tuple[ExperimentRecord, Path | None]:
    """Run the trials and write their record."""
    planning = planning or Planning()
    results = plan(planning)
    record = ExperimentRecord(
        experiment=EXPERIMENT,
        configuration={**asdict(planning), "result_schema": RESULT_SCHEMA},
        inference=ONLINE,
        evidence=NotEnumerated(
            "nothing is inferred and nothing is run: each home is a row of "
            "synthetic days drawn from distributions written down here"
        ),
        seeds=[planning.seed_root],
        results=results,
        data_source="synthetic",
        notes=[
            "Synthetic data only. Nothing here reads TIHM or runs the "
            "pipeline, so nothing here is evidence about a home.",
            "Made before the sleep-mat protocol was frozen, to choose its "
            "interval and to say how often each criterion's readings come out "
            "when the truth is known.",
            f"{planning.replications:,} synthetic studies in each scenario, "
            "each with a generator derived from one recorded root. "
            "docs/SIMULATION_PROTOCOLS.md sets a floor of 100.",
            f"Each study resamples its homes {planning.resamples:,} times for "
            "the bootstrap, fewer than a protocol would.",
        ],
    )
    path = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = record.write(Path(output_dir) / f"{EXPERIMENT}.json")
    return record, path
