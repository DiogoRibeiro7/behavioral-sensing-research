"""What the threshold-calibration protocol can show, tried on curves written down by hand.

The protocol compares two references where the calibrated one's operating
curve has the false alerts of the default as it ships. Before it was frozen,
its criterion was tried on outcomes drawn from operating curves that are
written down in this module: how many false alerts a home has, and whether a
change is found in it, at each multiple of each reference's thresholds.

**No home is simulated here and no reference is run.** The curves are
assumptions. They take their sizes from what the silent-home record gives the
default reference, and say nothing about the calibrated one: its curve is the
default's, moved along the multiples, with its false alerts multiplied by a
ratio that is the truth of each trial. A ratio of one is a reference no
better and no worse than the default.

Each trial draws a cohort of homes, places the match with the functions the
scoring uses, and decides the criterion as the protocol does. The same trial
also reads the other match, at the same detection, which the protocol states
no estimand on; this is where the reason for that is measured.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import special

from ..evaluation import ExperimentRecord
from ..fusion import ONLINE, NotEnumerated
from .threshold_calibration_experiment import crossing, matched_estimate, read_at
from .threshold_calibration_protocol import BRACKETED, ThresholdCalibrationProtocol

EXPERIMENT = "threshold-calibration-planning"
RESULT_SCHEMA = "threshold-calibration-planning/1"

#: The two matches a trial reads.
SAME_FALSE_ALERTS = "same_false_alerts"
SAME_DETECTION = "same_detection"

#: What a trial concludes at a match.
BETTER, WORSE, NOT_SHOWN, NOT_BRACKETED = (
    "better",
    "worse",
    "not_shown",
    "not_bracketed",
)
OUTCOMES = (BETTER, WORSE, NOT_SHOWN, NOT_BRACKETED)

#: The multiples at which the default reference's curve is written down.
KNOTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.5)

#: Its mean false alerts per home at each of them: one a home at its declared
#: thresholds, as in the silent-home record. Between two knots the logarithm
#: is taken to be a straight line.
FALSE_ALERTS = (
    36.0,
    24.0,
    16.0,
    10.0,
    6.5,
    4.3,
    2.6,
    1.7,
    1.0,
    0.30,
    0.08,
    0.01,
    0.0005,
)

#: The share of stable homes it alerts on in the step's window: 12 in 100 at
#: its declared thresholds, as in the silent-home record.
FALSE_DETECTION = (
    0.90,
    0.80,
    0.70,
    0.52,
    0.40,
    0.30,
    0.22,
    0.16,
    0.12,
    0.04,
    0.01,
    0.002,
    0.0,
)

#: The share of homes in which it finds the step, for three defaults: one
#: that finds most steps at its declared thresholds, as in the silent-home
#: record's 75 in 100; one that finds nearly all of them; and one that finds
#: few.
DETECTION = {
    "most": (1.0, 1.0, 1.0, 0.99, 0.97, 0.94, 0.90, 0.83, 0.75, 0.55, 0.38, 0.17, 0.01),
    "nearly_all": (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.99,
        0.98,
        0.97,
        0.95,
        0.85,
        0.65,
        0.30,
        0.02,
    ),
    "few": (
        0.97,
        0.95,
        0.90,
        0.82,
        0.74,
        0.66,
        0.58,
        0.50,
        0.42,
        0.26,
        0.15,
        0.05,
        0.0,
    ),
}

#: The functions of the scoring a trial decides its criterion with. Their
#: source is recorded with the trials, so that a protocol that quotes the
#: trials can tell whether they were made with the code it will be scored by.
SCORING_FUNCTIONS = (crossing, read_at, matched_estimate)


def scoring_sources() -> dict[str, str]:
    """The SHA-256 of the source of each function a trial decides with."""
    return {
        function.__name__: hashlib.sha256(
            inspect.getsource(function).encode("utf-8")
        ).hexdigest()
        for function in SCORING_FUNCTIONS
    }


SHAPES = tuple(DETECTION)


@dataclass(frozen=True)
class Cell:
    """One truth the criterion is tried under.

    Attributes
    ----------
    shape
        Which default: how much of the step it finds.
    ratio
        The calibrated reference's false alerts, and false detections, as a
        multiple of the default's at the same detection. One is the same
        curve.
    offset
        The calibrated reference at a multiple s is the default at s over
        this, so that the match falls between two multiples of the grid at a
        place that depends on it.
    correlation
        How far a home that is found under one reference is found under the
        other: the correlation of the two latent values.
    dispersion
        The shape of the gamma-distributed rate that makes some homes raise
        more false alerts than others; smaller is more uneven.
    alerts
        A multiple of every false-alert curve, for both references: below one
        the default raises fewer than one a home, and many homes raise none.
    """

    shape: str
    ratio: float
    offset: float
    correlation: float = 0.6
    dispersion: float = 1.9
    alerts: float = 1.0

    @property
    def usual(self) -> bool:
        """Whether the homes are as in the main cells."""
        return (self.correlation, self.dispersion, self.alerts) == (0.6, 1.9, 1.0)

    @property
    def name(self) -> str:
        """A name that orders nothing and identifies the cell."""
        return (
            f"{self.shape}/ratio={self.ratio:g}/offset={self.offset:g}"
            f"/correlation={self.correlation:g}/dispersion={self.dispersion:g}"
            f"/alerts={self.alerts:g}"
        )


@dataclass(frozen=True)
class Planning:
    """Every setting of the trials.

    Attributes
    ----------
    seed_root
        The root every cell's generator is derived from.
    replications
        Trials in each cell.
    homes, scales, confidence
        The cohort of each trial, the grid of multiples and the interval's
        level: the protocol's.
    resamples
        Resamples of homes in each trial. Fewer than the protocol's, since
        each cell holds many trials.
    ratios, offsets
        The truths tried for each default. With a ratio of one, the match
        falls near the offset, so the offsets are where on the grid the match
        is tried.
    """

    seed_root: int = 20261011
    replications: int = 1000
    homes: int = ThresholdCalibrationProtocol().homes
    scales: tuple[float, ...] = ThresholdCalibrationProtocol().curve_scales
    confidence: float = ThresholdCalibrationProtocol().confidence
    resamples: int = 1000
    ratios: tuple[float, ...] = (1.0, 0.5, 0.75, 1.5)
    offsets: tuple[float, ...] = (0.45, 0.6, 0.75, 0.9, 1.1)

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.replications < 100:
            raise ValueError("docs/SIMULATION_PROTOCOLS.md sets a floor of 100 trials")
        if 1.0 not in self.scales or 1.0 not in self.ratios:
            raise ValueError("the declared thresholds and the same curve must be tried")
        if min(self.homes, self.resamples) < 2:
            raise ValueError("a trial needs homes and resamples")

    def cells(self) -> tuple[Cell, ...]:
        """Every truth tried, in the order their generators are derived in.

        Each default at each ratio and offset; then, for the default that
        finds most steps and the same curve: homes less alike under the two
        references, homes more uneven in their false alerts, and defaults
        that raise a quarter as many false alerts and four times as many.
        """
        main = [
            Cell(shape, ratio, offset)
            for shape in SHAPES
            for ratio in self.ratios
            for offset in self.offsets
        ]
        return (
            *main,
            Cell("most", 1.0, 0.6, correlation=0.2),
            Cell("most", 1.0, 0.6, dispersion=0.6),
            Cell("most", 1.0, 0.6, alerts=0.25),
            Cell("most", 1.0, 0.6, alerts=4.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the settings and the curves, in a stable serialisable form."""
        return {
            "schema": "threshold-calibration-planning-settings/1",
            "name": EXPERIMENT,
            "evidence": "drawn from operating curves written down by hand. No "
            "home is simulated and no reference is run, so nothing here is "
            "evidence about a home or about the calibrated reference",
            "seed_root": self.seed_root,
            "seeds": "cell k of `cells`, in order, has the generator of "
            "`numpy.random.SeedSequence(seed_root).spawn(len(cells))[k]`",
            "replications": self.replications,
            "homes": self.homes,
            "scales": list(self.scales),
            "confidence": self.confidence,
            "resamples": self.resamples,
            "ratios": list(self.ratios),
            "offsets": list(self.offsets),
            "cells": [cell.name for cell in self.cells()],
            "scoring_functions": scoring_sources(),
            "curves": {
                "multiples": list(KNOTS),
                "false_alerts_per_home": list(FALSE_ALERTS),
                "false_detection": list(FALSE_DETECTION),
                "detection": {shape: list(DETECTION[shape]) for shape in SHAPES},
                "between_two_multiples": "a straight line, for false alerts "
                "in their logarithm",
                "where_the_sizes_come_from": "the silent-home record gives the "
                "default reference, at its declared thresholds, one false "
                "alert a home with a standard deviation of 1.2, the step "
                "found in 75 homes of 100 and 12 of 100 stable homes alerted "
                "on in its window. The rest of each curve is an assumption",
            },
            "a_trial": {
                "the_calibrated_reference": "at a multiple s it is the default "
                "at s over the offset, with its false alerts and its false "
                "detections multiplied by the ratio",
                "false_alerts": "a home's false alerts at the smallest "
                "multiple are Poisson with the curve's mean times a rate that "
                "is gamma-distributed over homes with mean one. The same rate "
                "serves both references, so a home that raises many under one "
                "raises many under the other. At each larger multiple a home "
                "keeps each of its false alerts with the probability that "
                "takes one mean to the next, so its counts fall as the "
                "thresholds rise",
                "detection": "a home is found at a multiple when a latent "
                "uniform value is below the curve there, and falsely detected "
                "likewise with another. The latent values of the two "
                "references are correlated. A home's false detection is drawn "
                "apart from its false alerts, although in a home it is one of "
                "them",
                "excess_detection": "found, minus falsely detected",
                "the_criterion": "the scoring's own `crossing`, `read_at` and "
                "`matched_estimate`, on the cohort and on every resample of "
                "it. Better when the interval at the same false alerts lies "
                "above zero, worse when it lies below, not bracketed when the "
                "match is not on the cohort as it is, and not shown otherwise",
                "the_other_match": "the largest multiple that finds at least "
                "as much as the default, and the false alerts read there, "
                "minus the default's. Better when that interval lies below "
                "zero and worse when it lies above. The protocol states no "
                "estimand on it",
            },
        }


def curves(shape: str, multiples: np.ndarray) -> tuple[np.ndarray, ...]:
    """The default's false alerts, detection and false detection at multiples."""
    knots = np.array(KNOTS)
    return (
        np.exp(np.interp(multiples, knots, np.log(FALSE_ALERTS))),
        np.interp(multiples, knots, DETECTION[shape]),
        np.interp(multiples, knots, FALSE_DETECTION),
    )


def draw(
    rng: np.random.Generator, cell: Cell, config: Planning
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One cohort: each home's false alerts and excess detection at every multiple.

    Returns, for the default and for the calibrated reference, two arrays of
    whole numbers with a row for each home and a column for each multiple.
    """
    scales = np.array(config.scales)
    rate = rng.gamma(cell.dispersion, 1.0 / cell.dispersion, size=config.homes)
    related = [[1.0, cell.correlation], [cell.correlation, 1.0]]
    latent = rng.multivariate_normal([0.0, 0.0], related, size=(2, config.homes))
    uniform = special.ndtr(latent)
    cohort = {}
    for place, reference in enumerate(("default", "calibrated")):
        if reference == "default":
            mean, found, false = curves(cell.shape, scales)
        else:
            mean, found, false = curves(cell.shape, scales / cell.offset)
            mean, false = cell.ratio * mean, np.minimum(cell.ratio * false, 1.0)
        mean = cell.alerts * mean
        counts = np.empty((config.homes, scales.size), dtype=np.int64)
        counts[:, 0] = rng.poisson(mean[0] * rate)
        for column in range(1, scales.size):
            kept = min(mean[column] / mean[column - 1], 1.0)
            counts[:, column] = rng.binomial(counts[:, column - 1], kept)
        detected = uniform[0, :, place][:, None] < found[None, :]
        falsely = uniform[1, :, place][:, None] < false[None, :]
        cohort[reference] = (counts, detected.astype(np.int64) - falsely)
    return cohort


def _verdict(estimate: Mapping[str, Any], more_is_better: bool) -> str:
    """Better, worse, not shown or not bracketed, from an estimate's interval."""
    if estimate["status"] != BRACKETED:
        return NOT_BRACKETED
    low, high = estimate["interval"]["low"], estimate["interval"]["high"]
    above, below = low is not None and low > 0.0, high is not None and high < 0.0
    if above:
        return BETTER if more_is_better else WORSE
    if below:
        return WORSE if more_is_better else BETTER
    return NOT_SHOWN


def trial(rng: np.random.Generator, cell: Cell, config: Planning) -> dict[str, Any]:
    """Draw one cohort and read both matches on it, as the scoring would."""
    cohort = draw(rng, cell, config)
    false, net = cohort["calibrated"]
    declared = config.scales.index(1.0)
    default_false = cohort["default"][0][:, declared]
    default_net = cohort["default"][1][:, declared]
    homes = config.homes
    resampled = rng.multinomial(
        homes, np.full(homes, 1.0 / homes), size=config.resamples
    )
    samples = np.vstack([np.ones((1, homes), dtype=np.int64), resampled])
    false, net = samples @ false, samples @ net
    default_false, default_net = samples @ default_false, samples @ default_net

    place, share, status = crossing(-false[:, ::-1], -default_false)
    quiet = matched_estimate(
        read_at(net[:, ::-1] / homes, place, share) - default_net / homes,
        status,
        config,  # type: ignore[arg-type]
        homes,
    )
    place, share, status = crossing(net, default_net)
    kept = matched_estimate(
        read_at(false / homes, place, share) - default_false / homes,
        status,
        config,  # type: ignore[arg-type]
        homes,
    )
    return {
        SAME_FALSE_ALERTS: {**quiet, "verdict": _verdict(quiet, more_is_better=True)},
        SAME_DETECTION: {**kept, "verdict": _verdict(kept, more_is_better=False)},
    }


def _share(count: int, trials: int) -> dict[str, float]:
    share = count / trials
    return {"share": share, "se": math.sqrt(share * (1.0 - share) / trials)}


def _summarise(found: Sequence[Mapping[str, Any]], resamples: int) -> dict[str, Any]:
    """What a cell's trials concluded at one match."""
    estimates = np.array(
        [trial["estimate"] for trial in found if trial["estimate"] is not None]
    )
    errors = [trial["mcse"] for trial in found if trial["mcse"] is not None]
    return {
        **{
            outcome: _share(
                sum(1 for trial in found if trial["verdict"] == outcome), len(found)
            )
            for outcome in OUTCOMES
        },
        "trials_with_an_estimate": int(estimates.size),
        "mean_estimate": float(estimates.mean()) if estimates.size else None,
        "sd_of_estimates": (
            float(estimates.std(ddof=1)) if estimates.size > 1 else None
        ),
        "mean_bootstrap_error": float(np.mean(errors)) if errors else None,
        "resamples_bracketed": float(
            np.mean([trial["resamples"][BRACKETED] for trial in found]) / resamples
        ),
    }


def _kept(config: Planning, cell: Cell, directory: Path) -> Path:
    """Where a finished cell is kept: named by the settings and the cell."""
    settings = json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(settings + cell.name.encode("utf-8")).hexdigest()
    return directory / f"cell-{digest[:24]}.json"


def run_cell(
    arguments: tuple[Cell, Planning, np.random.SeedSequence, Path | None],
) -> dict[str, Any]:
    """Every trial of one cell, or the cell as kept if it was finished before."""
    cell, config, seed, directory = arguments
    if directory is not None:
        path = _kept(config, cell, directory)
        if path.exists():
            kept: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return kept
    found = _trials(cell, config, seed)
    if directory is not None:
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(found), encoding="utf-8")
        partial.replace(path)
    return found


def _trials(
    cell: Cell, config: Planning, seed: np.random.SeedSequence
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    trials = [trial(rng, cell, config) for _ in range(config.replications)]
    return {
        "cell": cell.name,
        "shape": cell.shape,
        "ratio": cell.ratio,
        "offset": cell.offset,
        "correlation": cell.correlation,
        "dispersion": cell.dispersion,
        "alerts": cell.alerts,
        "replications": config.replications,
        **{
            match: _summarise([found[match] for found in trials], config.resamples)
            for match in (SAME_FALSE_ALERTS, SAME_DETECTION)
        },
    }


def _range(values: Sequence[float | None]) -> dict[str, float] | None:
    """The smallest and the largest of the values that exist; none if none do."""
    kept = [float(value) for value in values if value is not None]
    return {"low": min(kept), "high": max(kept)} if kept else None


def _ranges(chosen: Sequence[Mapping[str, Any]], match: str) -> dict[str, Any]:
    return {
        "cells": len(chosen),
        **{
            outcome: _range([cell[match][outcome]["share"] for cell in chosen])
            for outcome in OUTCOMES
        },
        "mean_estimate": _range([cell[match]["mean_estimate"] for cell in chosen]),
        "sd_of_estimates": _range([cell[match]["sd_of_estimates"] for cell in chosen]),
    }


def summary(cells: Sequence[Mapping[str, Any]], config: Planning) -> dict[str, Any]:
    """The ranges the protocol quotes, over the cells of each kind.

    At the same false alerts: for each ratio, over every cell with it,
    whatever the default, the place of the match and the homes; and for each
    default and ratio, over the main cells. At the same detection, for each
    default and ratio over the main cells, since it is the default that
    decides how that match behaves.
    """
    usual = [
        cell
        for cell in cells
        if (cell["correlation"], cell["dispersion"], cell["alerts"]) == (0.6, 1.9, 1.0)
    ]
    found: dict[str, Any] = {SAME_FALSE_ALERTS: {}, SAME_DETECTION: {}}
    for ratio in config.ratios:
        name = f"{ratio:g}"
        found[SAME_FALSE_ALERTS][name] = _ranges(
            [cell for cell in cells if cell["ratio"] == ratio], SAME_FALSE_ALERTS
        )
        for match in (SAME_FALSE_ALERTS, SAME_DETECTION):
            for shape in SHAPES:
                chosen = [
                    cell
                    for cell in usual
                    if (cell["shape"], cell["ratio"]) == (shape, ratio)
                ]
                if chosen:
                    found[match].setdefault("by_shape", {}).setdefault(shape, {})[
                        name
                    ] = _ranges(chosen, match)
    return found


def plan(
    config: Planning | None = None,
    *,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run every cell. The result does not depend on *jobs*.

    With *checkpoint*, each cell is kept in that directory as soon as it is
    finished and read back from it if the run is started again; a kept cell
    is named by the settings it was run under.
    """
    config = config or Planning()
    cells = config.cells()
    seeds = np.random.SeedSequence(config.seed_root).spawn(len(cells))
    if checkpoint is not None:
        Path(checkpoint).mkdir(parents=True, exist_ok=True)
    tasks = [(cell, config, seed, checkpoint) for cell, seed in zip(cells, seeds)]
    done: list[dict[str, Any]] = []
    if jobs <= 1:
        for task in tasks:
            done.append(run_cell(task))
            if progress is not None:
                progress(len(done), len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for found in pool.map(run_cell, tasks):
                done.append(found)
                if progress is not None:
                    progress(len(done), len(tasks))
    return {
        "result_schema": RESULT_SCHEMA,
        "cells": done,
        "summary": summary(done, config),
    }


@dataclass(frozen=True)
class PlanningResult:
    """A run of the trials: the record, and where it was written."""

    record: ExperimentRecord
    path: Path | None


def run_planning(
    config: Planning | None = None,
    *,
    output_dir: Path | None = None,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
    checkpoint: Path | None = None,
) -> PlanningResult:
    """Run the trials and write their record."""
    config = config or Planning()
    results = plan(config, jobs=jobs, progress=progress, checkpoint=checkpoint)
    record = ExperimentRecord(
        experiment=EXPERIMENT,
        configuration={**config.to_dict(), "result_schema": RESULT_SCHEMA},
        inference=ONLINE,
        evidence=NotEnumerated(
            "nothing is inferred and nothing is run: each home is a row of "
            "counts and flags drawn from curves written down by hand"
        ),
        seeds=[config.seed_root],
        results=results,
        data_source="synthetic",
        metric_definitions=dict(config.to_dict()["a_trial"]),
        notes=[
            "Drawn from operating curves written down by hand. No home is "
            "simulated and no reference is run, so nothing here is evidence "
            "about a home or about the calibrated reference.",
            "Made to plan the threshold-calibration protocol before it was "
            "frozen: to see how often its criterion is wrong when the two "
            "references have one curve, and what it can show when they have "
            "not.",
            f"{config.replications:,} trials in each cell, each with its own "
            "generator derived from one recorded root. "
            "docs/SIMULATION_PROTOCOLS.md sets a floor of 100, and every share "
            "is given with its standard error.",
            f"Each trial resamples its homes {config.resamples:,} times, fewer "
            "than the protocol does.",
        ],
        mcse={
            f"{cell['cell']}/{match}/{outcome}": cell[match][outcome]["se"]
            for cell in results["cells"]
            for match in (SAME_FALSE_ALERTS, SAME_DETECTION)
            for outcome in (BETTER, WORSE)
        },
    )
    path = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = record.write(Path(output_dir) / f"{EXPERIMENT}.json")
    return PlanningResult(record, path)
