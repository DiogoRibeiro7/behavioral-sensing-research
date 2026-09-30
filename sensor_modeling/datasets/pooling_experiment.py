"""Phase 3.4: the pre-specified evaluation of partial pooling of household parameters.

The fitted hurdle channel models (Phase 3.3 follow-up) give every household
its fold's population parameters. The partial-pooling framework
(:mod:`~sensor_modeling.datasets.partial_pooling`) lets a household move
toward its own values as it accumulates labelled data. This experiment
measures whether that helps on households none of the parameters were fitted
on, and whether fitting each household on its own overfits small homes.

Every setting, comparison and criterion is declared in
:class:`PoolingProtocol` and frozen in a committed file before any household
is scored.

Models
------
All five share the prior, transition and channels, and differ only in the
hurdle channel parameters:

| Model | Channel parameters |
| --- | --- |
| ``declared`` | the declared rates, for context |
| ``population`` | fitted per fold on the training households |
| ``pooled`` | the household pooled toward the population with the declared strength |
| ``pooled_selected`` | the same with a strength selected on training households only |
| ``unconstrained`` | the household's own estimates, the strength only keeping probabilities off 0 and 1 |

Each is scored in ``I0``, current windows, and in ``R``, the filter's
recursion over every window.

Adaptation arms
---------------
Each held-out household adapts on its labelled windows in the first days of
its recording, and is scored on its labelled windows after them. Every model
in an arm is scored on exactly those windows. Two arms are declared: a week,
the Phase 3.1 window, and a day, which leaves most cells with little data.
Nothing is compared across arms.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.households import household_values, recall_of
from ..evaluation.metrics import PredictionMetrics
from ..evaluation.provenance import (
    ExperimentRecord,
    InputArtifact,
    ModelRecord,
    ReportedInterval,
)
from ..fusion.regime import ONLINE
from ..states.ontology import BehaviouralState, StateOntology
from .casas import CasasRecording
from .channel_models import (
    DECLARED as DECLARED_FAMILY,
)
from .channel_models import (
    HURDLE,
    ChannelStatistics,
    FittedChannels,
    HouseholdChannels,
    combine_statistics,
    filter_recursion,
    home_statistics,
    household_channel_counts,
    household_statistics,
    pool_channels,
    total_loglik,
)
from .information_sets import (
    EvidenceChannel,
    EvidenceResolution,
    InformationComponent,
    InformationSet,
    build_feature_table,
)
from .matched_evaluation import (
    MATCHED_METRIC_DEFINITIONS,
    HouseholdSplit,
    _finite,
    _Household,
    _regular_moments,
    online_evidence,
)
from .partial_pooling import PoolingConfig
from .recoverable_gap import (
    DEFAULT_METRICS,
    GENERATIVE_CONFIGURATION,
    FrozenSplits,
    GapProtocol,
    _cell,
    _paired,
    _score,
)
from .restricted_filter import restricted_posteriors
from .time_prior_experiment import MINIMAL_DIFFERENCES, verdict

PROTOCOL_SCHEMA = "pooling-protocol/1"
RESULT_SCHEMA = "pooling-results/1"

DECLARED = "declared"
POPULATION = "population"
POOLED = "pooled"
SELECTED = "pooled_selected"
UNCONSTRAINED = "unconstrained"
MODELS = (DECLARED, POPULATION, POOLED, SELECTED, UNCONSTRAINED)

CURRENT = "I0"
RECURSION = "R"
SETTINGS = (CURRENT, RECURSION)

WEEK = "week"
DAY = "day"

#: Which models each arm scores. The strength is selected on the week arm only.
ARM_MODELS: dict[str, tuple[str, ...]] = {
    WEEK: (DECLARED, POPULATION, POOLED, SELECTED, UNCONSTRAINED),
    DAY: (DECLARED, POPULATION, POOLED, UNCONSTRAINED),
}


def cell_key(model: str, setting: str, arm: str) -> str:
    """The name of one scored cell: model, setting and adaptation arm."""
    return f"{model}@{setting}/{arm}"


@dataclass(frozen=True)
class Comparison:
    """One pre-specified paired comparison within one setting and arm."""

    key: str
    role: str
    rule: str
    question: str
    model: str
    reference: str
    setting: str
    arm: str

    def to_dict(self) -> dict[str, str]:
        """Return a serialisable form."""
        return {
            "key": self.key,
            "role": self.role,
            "rule": self.rule,
            "question": self.question,
            "model": cell_key(self.model, self.setting, self.arm),
            "reference": cell_key(self.reference, self.setting, self.arm),
        }


ESTIMANDS: tuple[Comparison, ...] = (
    Comparison(
        "P1",
        "primary",
        "pooling",
        "does partial pooling improve on the population, current windows",
        POOLED,
        POPULATION,
        CURRENT,
        WEEK,
    ),
    Comparison(
        "P2",
        "primary",
        "pooling",
        "does partial pooling improve on the population, the recursion",
        POOLED,
        POPULATION,
        RECURSION,
        WEEK,
    ),
    Comparison(
        "O1",
        "primary",
        "overfitting",
        "does unconstrained per-home fitting overfit small homes: pooled against "
        "unconstrained after one day",
        POOLED,
        UNCONSTRAINED,
        CURRENT,
        DAY,
    ),
    Comparison(
        "O2",
        "secondary",
        "overfitting",
        "the same after one week",
        POOLED,
        UNCONSTRAINED,
        CURRENT,
        WEEK,
    ),
    Comparison(
        "O3",
        "secondary",
        "overfitting",
        "the same after one day, the recursion",
        POOLED,
        UNCONSTRAINED,
        RECURSION,
        DAY,
    ),
    Comparison(
        "S1",
        "secondary",
        "pooling",
        "does a strength selected on training households improve on the "
        "population, current windows",
        SELECTED,
        POPULATION,
        CURRENT,
        WEEK,
    ),
    Comparison(
        "S2",
        "secondary",
        "pooling",
        "the same, the recursion",
        SELECTED,
        POPULATION,
        RECURSION,
        WEEK,
    ),
    Comparison(
        "S3",
        "secondary",
        "pooling",
        "does partial pooling improve on the population after one day",
        POOLED,
        POPULATION,
        CURRENT,
        DAY,
    ),
    Comparison(
        "S4",
        "secondary",
        "pooling",
        "does unconstrained per-home fitting improve on the population after one week",
        UNCONSTRAINED,
        POPULATION,
        CURRENT,
        WEEK,
    ),
    Comparison(
        "S5",
        "secondary",
        "pooling",
        "the same after one day",
        UNCONSTRAINED,
        POPULATION,
        CURRENT,
        DAY,
    ),
    Comparison(
        "C1",
        "context",
        "pooling",
        "the population fit against the declared rates, current windows",
        POPULATION,
        DECLARED,
        CURRENT,
        WEEK,
    ),
    Comparison(
        "C2",
        "context",
        "pooling",
        "the same, the recursion",
        POPULATION,
        DECLARED,
        RECURSION,
        WEEK,
    ),
)

#: Estimands whose per-state recall is reported.
RECALL_ESTIMANDS = ("P1", "O1")

CRITERIA: dict[str, str] = {
    "verdicts": "favours model: mean >= delta and lower bound > 0; favours "
    "reference: mean <= -delta and upper bound < 0; negligible: the whole "
    "interval within (-delta, delta); uncertain: anything else",
    "pooling_success": "log loss favours the adapted model, and neither balanced "
    "accuracy nor calibration error favours the reference",
    "pooling_trade-off": "log loss favours the adapted model and balanced "
    "accuracy favours the reference",
    "pooling_failure": "log loss is negligible or favours the reference",
    "pooling_inconclusive": "anything else",
    "overfits": "log loss favours the pooled model over the unconstrained one",
    "does not overfit": "log loss is negligible or favours the unconstrained model",
    "overfitting_inconclusive": "anything else",
    "primary": "P1 and P2 answer whether pooling helps; O1 whether unconstrained "
    "fitting overfits small homes; the others are read by the same rules",
    "why_log_loss": "adaptation changes each household's probability model, so a "
    "proper scoring rule is the target; balanced accuracy and calibration error "
    "are guards",
}


def pooling_conclusion(verdicts: Mapping[str, str]) -> str:
    """Success, trade-off, failure or inconclusive, for an adapted model."""
    if verdicts["log_loss"] in ("negligible", "favours reference"):
        return "failure"
    if verdicts["log_loss"] == "favours model":
        if verdicts["balanced_accuracy"] == "favours reference":
            return "trade-off"
        if verdicts["calibration_error"] != "favours reference":
            return "success"
    return "inconclusive"


def overfitting_conclusion(verdicts: Mapping[str, str]) -> str:
    """Whether unconstrained fitting overfits, from pooled against unconstrained."""
    if verdicts["log_loss"] == "favours model":
        return "overfits"
    if verdicts["log_loss"] in ("negligible", "favours reference"):
        return "does not overfit"
    return "inconclusive"


_RULES = {"pooling": pooling_conclusion, "overfitting": overfitting_conclusion}


@dataclass(frozen=True)
class PoolingProtocol:
    """Everything the experiment fixes before any household is scored."""

    folds: tuple[HouseholdSplit, ...]
    splits_sha256: str
    resolution: EvidenceResolution = field(default_factory=EvidenceResolution)
    population_pseudo_windows: float = 12.0
    strength: float = 288.0
    unconstrained_strength: float = 0.5
    strength_grid: tuple[float, ...] = (24.0, 72.0, 288.0, 1152.0, 4608.0)
    arm_days: Mapping[str, float] = field(default_factory=lambda: {WEEK: 7.0, DAY: 1.0})
    min_adaptation_windows: int = 12
    min_scored_windows: int = 288
    seed: int = 0
    metrics: tuple[str, ...] = DEFAULT_METRICS
    resamples: int = 10_000
    confidence: float = 0.95
    minimal_differences: Mapping[str, float] = field(
        default_factory=lambda: dict(MINIMAL_DIFFERENCES)
    )
    rare_share: float = 0.05
    name: str = "phase3-partial-pooling"

    def __post_init__(self) -> None:
        """Validate through the recoverable-gap protocol, then the rest."""
        base = self.as_gap_protocol()
        if not isinstance(self.splits_sha256, str) or len(self.splits_sha256) != 64:
            raise ValueError("splits_sha256 must be a SHA-256 hex digest")
        strengths = [
            self.population_pseudo_windows,
            self.strength,
            self.unconstrained_strength,
            *self.strength_grid,
        ]
        if any(not math.isfinite(s) or s <= 0.0 for s in strengths):
            raise ValueError("every pooling strength must be positive")
        grid = tuple(float(s) for s in self.strength_grid)
        if len(grid) < 2 or list(grid) != sorted(set(grid)):
            raise ValueError("the strength grid needs distinct increasing values")
        if self.unconstrained_strength >= min(self.strength, grid[0]):
            raise ValueError("the unconstrained strength must be the smallest")
        if set(self.arm_days) != {WEEK, DAY} or any(
            not math.isfinite(d) or d <= 0.0 for d in self.arm_days.values()
        ):
            raise ValueError("arm_days must give a positive length for week and day")
        for name in ("min_adaptation_windows", "min_scored_windows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        missing = sorted({*base.metrics, "recall"} - set(self.minimal_differences))
        if missing or any(
            not math.isfinite(v) or v <= 0.0 for v in self.minimal_differences.values()
        ):
            raise ValueError(
                f"every metric needs a positive minimal difference; missing {missing}"
            )
        if not 0.0 < self.rare_share < 1.0:
            raise ValueError("rare_share must lie in (0, 1)")
        object.__setattr__(self, "folds", base.folds)
        object.__setattr__(self, "metrics", base.metrics)
        object.__setattr__(self, "strength_grid", grid)
        object.__setattr__(self, "arm_days", dict(self.arm_days))
        object.__setattr__(self, "minimal_differences", dict(self.minimal_differences))

    def as_gap_protocol(self) -> GapProtocol:
        """The equivalent recoverable-gap protocol, for validation and helpers."""
        return GapProtocol(
            tuple(self.folds),
            seed=self.seed,
            metrics=tuple(self.metrics),
            resamples=self.resamples,
            confidence=self.confidence,
        )

    @property
    def information_set(self) -> InformationSet:
        """``I0``: each channel's current window."""
        return InformationSet(
            "current",
            frozenset({InformationComponent.CURRENT_EVIDENCE}),
            self.resolution,
        )

    @property
    def homes(self) -> tuple[str, ...]:
        """Every held-out household, sorted."""
        return tuple(sorted(home for fold in self.folds for home in fold.test))

    def strength_of(self, model: str, selected: float) -> float | None:
        """The pooling strength *model* uses, or ``None`` if it does not pool."""
        return {
            POOLED: self.strength,
            SELECTED: selected,
            UNCONSTRAINED: self.unconstrained_strength,
        }.get(model)

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        info = self.information_set
        return {
            "schema": PROTOCOL_SCHEMA,
            "name": self.name,
            "households": {
                "splits_file": "artifacts/phase1/household_splits.json",
                "splits_sha256": self.splits_sha256,
                "folds": [
                    {**fold.to_dict(), "sha256": fold.sha256()} for fold in self.folds
                ],
                "use": "each household is held out once; its fold's training "
                "households fit the population and select the pooling strength",
            },
            "settings": {
                CURRENT: {**info.to_dict(), "sha256": info.sha256()},
                RECURSION: "the filter's recursion fed every window of the "
                "recording from the stationary distribution",
            },
            "models": {
                DECLARED: {**GENERATIVE_CONFIGURATION, "channels": "declared"},
                POPULATION: {
                    **GENERATIVE_CONFIGURATION,
                    "channels": "hurdle, fitted per fold on the training "
                    "households as in the Phase 3.3 follow-up",
                    "pseudo_windows": self.population_pseudo_windows,
                },
                POOLED: {
                    "channels": "hurdle, the household pooled toward the population",
                    "strength": self.strength,
                },
                SELECTED: {
                    "channels": "hurdle, the household pooled toward the "
                    "population with the strength selected on the fold's "
                    "training households",
                    "grid": list(self.strength_grid),
                },
                UNCONSTRAINED: {
                    "channels": "hurdle, the household's own estimates",
                    "strength": self.unconstrained_strength,
                    "why": "half a window keeps a probability off 0 and 1; with "
                    "a few windows the household's own data dominate, and a "
                    "cell without any takes the population value",
                },
            },
            "arms": {
                arm: {
                    "adaptation": f"the household's labelled windows closing within "
                    f"{days:g} days of its first observation",
                    "scored": "its labelled windows closing after that",
                    "models": list(ARM_MODELS[arm]),
                }
                for arm, days in self.arm_days.items()
            },
            "arm_days": dict(self.arm_days),
            "pooling": {
                "estimator": "theta_pop + w (theta_raw - theta_pop), w = n / (n + "
                "kappa), for the silence probability and the active-window mean "
                "of every channel and state",
                "strength": self.strength,
                "why": "288 windows is 24 labelled hours, the framework's default "
                "and the Phase 3.1 household pooling strength",
            },
            "selection": {
                "grid": list(self.strength_grid),
                "procedure": "for each fold, leave one training household out at a "
                "time: fit the population on the other training households, pool "
                "the left-out household with its week-arm window, score its "
                "labelled windows after the window in I0, and take the mean log "
                "loss over the eligible training households",
                "choice": "the smallest mean log loss; a tie goes to the larger "
                "strength",
                "held_out": "the fold's held-out households are never used",
            },
            "eligibility": {
                "min_adaptation_windows": self.min_adaptation_windows,
                "min_scored_windows": self.min_scored_windows,
                "rule": "a household enters an arm's comparisons with at least "
                "min_adaptation_windows labelled windows in its adaptation "
                "window and min_scored_windows labelled windows after it; "
                "otherwise it is reported and left out of every model in that "
                "arm",
                "cells": "a channel and state with no household windows takes "
                "the population value in every adapted model",
            },
            "stratification": "within each arm, eligible households are sorted by "
            "labelled adaptation windows (ties by name); the first half is "
            "'less data', the rest 'more data'; every estimand is also reported "
            "within each stratum",
            "metrics": list(self.metrics),
            "per_state_recall": {
                "estimands": list(RECALL_ESTIMANDS),
                "rare_rule": f"share of labelled time below {self.rare_share} in "
                "either fold's training households",
            },
            "calibration": "expected calibration error over 10 confidence bins",
            "estimands": [estimand.to_dict() for estimand in ESTIMANDS],
            "minimal_differences": dict(self.minimal_differences),
            "criteria": dict(CRITERIA),
            "bootstrap": {
                "unit": "household",
                "statistics": "mean and median paired differences",
                "resamples": self.resamples,
                "confidence": self.confidence,
                "interval": "percentile",
            },
            "seeds": {"bootstrap": self.seed},
            "inference": "online filter: I0 and R read no window after the scored one",
            "tuning": "none on held-out households: the strength is declared, or "
            "selected on training households only",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_protocol(splits: FrozenSplits) -> PoolingProtocol:
    """The protocol as declared for the development panel's frozen folds."""
    return PoolingProtocol(splits.folds, splits.sha256)


def check_frozen_protocol(protocol: PoolingProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after scoring begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


# ----------------------------------------------------------------------------
# One household
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class _Prepared:
    """One household's table, counts, labels and arm cut-offs, computed once."""

    table: Any
    rows: np.ndarray
    labels: tuple[BehaviouralState, ...]
    counts: Mapping[EvidenceChannel, np.ndarray]
    moments: Sequence[datetime]
    cutoffs: Mapping[str, datetime]
    own: Mapping[str, Mapping[EvidenceChannel, ChannelStatistics]]


def _prepare(
    recording: CasasRecording,
    protocol: PoolingProtocol,
    ontology: StateOntology,
    *,
    household: str,
) -> _Prepared:
    resolution = protocol.resolution
    moments = _regular_moments(recording, resolution.step)
    table = build_feature_table(
        recording, protocol.information_set, moments, household=household
    )
    built = _Household.build(household, "test", table, recording)
    counts, _, _, _ = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    start = min(o.timestamp for o in recording.observations)
    cutoffs = {
        arm: start + timedelta(days=days) for arm, days in protocol.arm_days.items()
    }
    own = {
        arm: household_statistics(
            recording, resolution, ontology, household=household, until=cutoff
        )
        for arm, cutoff in cutoffs.items()
    }
    return _Prepared(
        table, built.labelled, built.labels, counts, list(table.moments), cutoffs, own
    )


def _eligibility(
    prepared: _Prepared, arm: str, protocol: PoolingProtocol
) -> dict[str, Any]:
    cutoff = prepared.cutoffs[arm]
    closes = [prepared.moments[int(row)] for row in prepared.rows]
    adaptation = sum(1 for moment in closes if moment <= cutoff)
    after = [i for i, moment in enumerate(closes) if moment > cutoff]
    return {
        "cutoff": cutoff.isoformat(),
        "adaptation_windows": adaptation,
        "scored_windows": len(after),
        "eligible": adaptation >= protocol.min_adaptation_windows
        and len(after) >= protocol.min_scored_windows,
        "after": after,
    }


def _beliefs(
    prepared: _Prepared,
    models: Mapping[EvidenceChannel, Any],
    ontology: StateOntology,
    resolution: EvidenceResolution,
) -> dict[str, np.ndarray]:
    current = restricted_posteriors(prepared.table, models, ontology)[prepared.rows]
    _, posterior = filter_recursion(
        total_loglik(models, prepared.counts),
        ontology.transition(resolution.step),
        ontology.stationary(),
    )
    return {CURRENT: current, RECURSION: posterior[prepared.rows]}


def _adapted(
    population: FittedChannels,
    recording: CasasRecording,
    prepared: _Prepared,
    arm: str,
    strength: float,
    protocol: PoolingProtocol,
    ontology: StateOntology,
    household: str,
) -> HouseholdChannels:
    return pool_channels(
        population,
        recording.registry,
        prepared.own[arm],
        household=household,
        resolution=protocol.resolution,
        config=PoolingConfig(strength),
        until=prepared.cutoffs[arm],
        ontology=ontology,
    )


def select_strength(
    recordings: Mapping[str, CasasRecording],
    fold: HouseholdSplit,
    statistics: Mapping[str, Mapping[EvidenceChannel, ChannelStatistics]],
    protocol: PoolingProtocol,
    ontology: StateOntology,
) -> dict[str, Any]:
    """The fold's pooling strength, chosen on its training households alone."""
    states = tuple(ontology.states)
    per_home: dict[str, dict[str, float]] = {}
    for home in sorted(fold.train):
        prepared = _prepare(recordings[home], protocol, ontology, household=home)
        eligibility = _eligibility(prepared, WEEK, protocol)
        if not eligibility["eligible"]:
            continue
        others = {h: statistics[h] for h in fold.train if h != home}
        population = combine_statistics(
            others, states=states, pseudo_windows=protocol.population_pseudo_windows
        )
        after = eligibility["after"]
        labels = [prepared.labels[i] for i in after]
        per_home[home] = {}
        for strength in protocol.strength_grid:
            adapted = _adapted(
                population,
                recordings[home],
                prepared,
                WEEK,
                strength,
                protocol,
                ontology,
                home,
            )
            beliefs = restricted_posteriors(prepared.table, adapted.models(), ontology)[
                prepared.rows
            ][after]
            loss = _score(labels, beliefs, states).log_loss
            if loss is None:  # pragma: no cover - every posterior has probabilities
                raise ValueError(f"log loss is undefined for {home!r}")
            per_home[home][f"{strength:g}"] = loss
    if not per_home:
        raise ValueError(f"no training household of {fold.name} is eligible")
    means = {
        f"{s:g}": float(np.mean([scores[f"{s:g}"] for scores in per_home.values()]))
        for s in protocol.strength_grid
    }
    best = min(means.values())
    selected = max(s for s in protocol.strength_grid if means[f"{s:g}"] == best)
    return {
        "per_home": per_home,
        "mean_log_loss": means,
        "selected": selected,
        "households": sorted(per_home),
    }


def household_scores(
    recording: CasasRecording,
    protocol: PoolingProtocol,
    population: FittedChannels,
    selected: float,
    *,
    household: str,
    ontology: StateOntology,
) -> dict[str, Any]:
    """Every cell's scores and the pooling diagnostics for one held-out household."""
    states = tuple(ontology.states)
    resolution = protocol.resolution
    prepared = _prepare(recording, protocol, ontology, household=household)
    registry = recording.registry
    common = {
        DECLARED: population.models(registry, resolution, DECLARED_FAMILY, ontology),
        POPULATION: population.models(registry, resolution, HURDLE, ontology),
    }
    beliefs = {
        model: _beliefs(prepared, m, ontology, resolution)
        for model, m in common.items()
    }
    scores: dict[str, PredictionMetrics] = {}
    arms: dict[str, Any] = {}
    for arm, models in ARM_MODELS.items():
        eligibility = _eligibility(prepared, arm, protocol)
        after = eligibility.pop("after")
        adapted: dict[str, HouseholdChannels] = {}
        for model in models:
            strength = protocol.strength_of(model, selected)
            if strength is None:
                continue
            adapted[model] = _adapted(
                population,
                recording,
                prepared,
                arm,
                strength,
                protocol,
                ontology,
                household,
            )
        arms[arm] = {
            **eligibility,
            "pooled": adapted[POOLED].to_dict(),
        }
        if not eligibility["eligible"]:
            continue
        labels = [prepared.labels[i] for i in after]
        for model in models:
            posterior = (
                beliefs[model]
                if model in beliefs
                else _beliefs(prepared, adapted[model].models(), ontology, resolution)
            )
            for setting in SETTINGS:
                scores[cell_key(model, setting, arm)] = _score(
                    labels, posterior[setting][after], states
                )
    return {"scores": scores, "arms": arms, "labelled": int(prepared.rows.size)}


# ----------------------------------------------------------------------------
# Across households
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PoolingResult:
    """A completed experiment: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def strata(windows: Mapping[str, int]) -> dict[str, list[str]]:
    """The pre-specified split by adaptation data: less, then more."""
    ordered = sorted(windows, key=lambda home: (windows[home], home))
    half = len(ordered) // 2
    return {"less data": sorted(ordered[:half]), "more data": sorted(ordered[half:])}


def _shares(
    fold: HouseholdSplit,
    states: tuple[BehaviouralState, ...],
    statistics: Mapping[str, Mapping[EvidenceChannel, ChannelStatistics]],
) -> dict[str, float]:
    """Each state's share of labelled windows in the fold's training households."""
    totals = np.zeros(len(states))
    for home in fold.train:
        parts = statistics[home]
        if parts:
            totals += next(iter(parts.values())).windows
    return {state.value: float(v) for state, v in zip(states, totals / totals.sum())}


def run_pooling(
    recordings: Mapping[str, CasasRecording],
    protocol: PoolingProtocol,
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> PoolingResult:
    """Run the pre-specified experiment and build its record."""
    homes = sorted({h for fold in protocol.folds for h in (*fold.train, *fold.test)})
    missing = sorted(set(homes) - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    ontology = StateOntology()
    space = tuple(ontology.states)
    gap = protocol.as_gap_protocol()
    statistics = {
        home: home_statistics(
            recordings[home], protocol.resolution, ontology, household=home
        )
        for home in homes
    }

    fits: dict[str, dict[str, Any]] = {}
    for fold in protocol.folds:
        population = combine_statistics(
            {h: statistics[h] for h in fold.train},
            states=space,
            pseudo_windows=protocol.population_pseudo_windows,
        )
        fits[fold.name] = {
            "population": population,
            "selection": select_strength(
                recordings, fold, statistics, protocol, ontology
            ),
            "shares": _shares(fold, space, statistics),
        }
    rare = sorted(
        s.value
        for s in space
        if any(f["shares"][s.value] < protocol.rare_share for f in fits.values())
    )

    scores: dict[str, dict[str, PredictionMetrics]] = {}
    households: dict[str, dict[str, Any]] = {}
    for fold in protocol.folds:
        fit = fits[fold.name]
        for home in sorted(fold.test):
            scored = household_scores(
                recordings[home],
                protocol,
                fit["population"],
                fit["selection"]["selected"],
                household=home,
                ontology=ontology,
            )
            for key, metrics in scored.pop("scores").items():
                scores.setdefault(key, {})[home] = metrics
            households[home] = {"fold": fold.name, **scored}

    def values(key: str, metric: Any) -> dict[str, float | None]:
        return household_values(scores.get(key, {}), metric)

    cells = []
    for arm, models in ARM_MODELS.items():
        for model in models:
            for setting in SETTINGS:
                key = cell_key(model, setting, arm)
                if scores.get(key):
                    cells.append(
                        {
                            **_cell(model, setting, scores[key], space, gap),
                            "arm": arm,
                            "key": key,
                        }
                    )

    eligible = {
        arm: {
            home: entry["arms"][arm]["adaptation_windows"]
            for home, entry in households.items()
            if entry["arms"][arm]["eligible"]
        }
        for arm in ARM_MODELS
    }
    for arm, members in eligible.items():
        if len(members) < 2:
            raise ValueError(
                f"the {arm} arm has {len(members)} eligible households; paired "
                "household comparisons need at least two"
            )
    arm_strata = {arm: strata(members) for arm, members in eligible.items()}

    def compare(estimand: Comparison, homes_in: Sequence[str] | None) -> dict[str, Any]:
        model_key = cell_key(estimand.model, estimand.setting, estimand.arm)
        reference_key = cell_key(estimand.reference, estimand.setting, estimand.arm)

        def restrict(values_: Mapping[str, float | None]) -> dict[str, float | None]:
            if homes_in is None:
                return dict(values_)
            return {h: v for h, v in values_.items() if h in homes_in}

        comparisons = {
            metric: _paired(
                restrict(values(model_key, metric)),
                restrict(values(reference_key, metric)),
                metric,
                gap,
            )
            for metric in protocol.metrics
        }
        verdicts = {
            metric: verdict(comparison, protocol.minimal_differences[metric])
            for metric, comparison in comparisons.items()
        }
        return {
            "comparisons": comparisons,
            "verdicts": verdicts,
            "conclusion": _RULES[estimand.rule](verdicts),
        }

    estimands: list[dict[str, Any]] = []
    for estimand in ESTIMANDS:
        entry: dict[str, Any] = {**estimand.to_dict(), **compare(estimand, None)}
        entry["strata"] = {
            name: {"households": members, **compare(estimand, members)}
            for name, members in arm_strata[estimand.arm].items()
            if len(members) >= 2
        }
        if estimand.key in RECALL_ESTIMANDS:
            model_key = cell_key(estimand.model, estimand.setting, estimand.arm)
            reference_key = cell_key(estimand.reference, estimand.setting, estimand.arm)
            entry["per_state_recall"] = {}
            for state in space:
                comparison = _paired(
                    values(model_key, recall_of(state)),
                    values(reference_key, recall_of(state)),
                    "balanced_accuracy",
                    gap,
                )
                entry["per_state_recall"][state.value] = {
                    "rare": state.value in rare,
                    "comparison": comparison,
                    "verdict": verdict(
                        comparison, protocol.minimal_differences["recall"]
                    ),
                }
        estimands.append(entry)

    by_key = {e["key"]: e for e in estimands}
    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified; development panel, which earlier work has inspected",
        "states": [state.value for state in space],
        "rare_states": rare,
        "households": dict(sorted(households.items())),
        "strata": arm_strata,
        "cells": cells,
        "estimands": estimands,
        "conclusions": {
            "pooling_current_windows": by_key["P1"]["conclusion"],
            "pooling_recursion": by_key["P2"]["conclusion"],
            "overfitting_small_homes": by_key["O1"]["conclusion"],
        },
        "selection": {fold: fit["selection"] for fold, fit in fits.items()},
        "fitted": {
            fold: {
                "population": {
                    **fit["population"].to_dict(),
                    "sha256": fit["population"].sha256(),
                },
                "training_shares": fit["shares"],
            }
            for fold, fit in fits.items()
        },
    }
    # The arms' scored windows are nested, so a household's predictions are
    # the windows after its earliest eligible cut-off.
    first = {
        home: min(
            datetime.fromisoformat(entry["cutoff"])
            for entry in households[home]["arms"].values()
            if entry["eligible"]
        )
        for home in households
        if any(entry["eligible"] for entry in households[home]["arms"].values())
    }
    record = ExperimentRecord(
        experiment=protocol.name,
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        inference=ONLINE,
        evidence=online_evidence(
            recordings, sorted(first), protocol.resolution.step, after=first
        ),
        seeds=[protocol.seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions=MATCHED_METRIC_DEFINITIONS,
        notes=[
            "Every setting, estimand and criterion was declared in the protocol "
            "before any household was scored.",
            "Within an arm, every model is scored on the same labelled windows, "
            "after the household's adaptation window; nothing is compared across "
            "arms.",
            "The population and the selected strength come from each fold's "
            "training households only.",
            "Verdicts compare effect sizes and household bootstrap intervals with "
            "declared minimal differences; they are not significance tests.",
        ],
        inputs=list(inputs),
        split={
            "folds": [{**f.to_dict(), "sha256": f.sha256()} for f in protocol.folds]
        },
        information_set={
            **protocol.information_set.to_dict(),
            "sha256": protocol.information_set.sha256(),
        },
        preprocessing={
            "moments": "one per step from each recording's first observation to "
            "its last",
            "labels": "truth_series of each recording's annotations; unlabelled "
            "windows are not scored",
            "step_seconds": protocol.resolution.step.total_seconds(),
        },
        models=[
            ModelRecord(
                model, {"channels": model}, "restricted_posteriors and filter_recursion"
            )
            for model in MODELS
        ],
        household_metrics={
            key: {h: m.to_dict() for h, m in per_home.items()}
            for key, per_home in sorted(scores.items())
        },
        intervals=_intervals(estimands),
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return PoolingResult(record=record, path=path)


def _intervals(estimands: Sequence[Mapping[str, Any]]) -> list[ReportedInterval]:
    reported: list[ReportedInterval] = []
    for entry in estimands:
        for metric, comparison in entry["comparisons"].items():
            for statistic in ("mean", "median"):
                estimate = comparison[statistic]
                if estimate["interval"] is None:
                    continue
                reported.append(
                    ReportedInterval(
                        label=f"{entry['key']}: {statistic} {metric} difference",
                        estimate=estimate["estimate"],
                        low=estimate["interval"]["low"],
                        high=estimate["interval"]["high"],
                        confidence=estimate["interval"]["confidence"],
                        method=estimate["interval"]["method"],
                        unit="household",
                        n=comparison["n"],
                    )
                )
    return reported
