"""Phase 3: the time prior and the fitted channels together, in the recursion.

Phase 3 left two successes that have never been evaluated together where the
online filter works, in its recursion over every window. The fitted
hurdle-Poisson channels improved the recursion (``FR``, a success); the
hierarchical periodic time prior improved the generative model on declared
information sets, at the declared hour (Phase 3.1, a success), and has never
entered the recursion. This experiment runs the recursion with the prior's
hour-dependent transition, with the declared and with the fitted channels, on
the development panel's frozen folds, and asks whether the prior adds to the
reference formulation.

Everything is declared in :class:`CombinedPriorProtocol` and frozen in a
committed file before any household is scored. The fold fits are those of the
fitted-rates protocol, and the two recursions without the prior must give back
that published record's scores before anything is reported.

Models, all scored in the recursion ``R`` on identical windows:

| Model | Channels | Transition |
| --- | --- | --- |
| ``filter_declared`` | declared Poisson | the ontology's, every window |
| ``filter_hurdle`` | fitted hurdle | the ontology's, every window |
| ``filter_periodic`` | declared Poisson | the time prior's at each window's hour |
| ``filter_periodic_hurdle`` | fitted hurdle | the time prior's at each window's hour |
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.households import household_values, recall_of
from ..evaluation.metrics import PredictionMetrics
from ..evaluation.provenance import ExperimentRecord, InputArtifact, ModelRecord
from ..fusion.regime import ONLINE
from ..states.ontology import StateOntology
from . import rates_experiment
from .casas import CasasRecording
from .channel_models import (
    DECLARED,
    HURDLE,
    filter_recursion,
    household_channel_counts,
    total_loglik,
)
from .history_experiment import Estimand
from .matched_evaluation import MATCHED_METRIC_DEFINITIONS, _finite, online_evidence
from .periodic_prior import PeriodicStatePrior
from .rates_experiment import FittedRatesProtocol, FoldFit, fold_fits
from .rates_experiment import conclusion as channel_conclusion
from .recoverable_gap import FrozenSplits, _cell, _paired, _score
from .time_prior_experiment import conclusion as prior_conclusion
from .time_prior_experiment import verdict

PROTOCOL_SCHEMA = "combined-prior-protocol/1"
RESULT_SCHEMA = "combined-prior-results/1"
EXPERIMENT = "phase3-combined-prior"
RECURSION = "R"

FILTER_DECLARED = "filter_declared"
FILTER_HURDLE = "filter_hurdle"
FILTER_PERIODIC = "filter_periodic"
FILTER_PERIODIC_HURDLE = "filter_periodic_hurdle"
#: Each model's channel family and whether its transition is the time prior's.
MODELS: dict[str, tuple[str, bool]] = {
    FILTER_DECLARED: (DECLARED, False),
    FILTER_HURDLE: (HURDLE, False),
    FILTER_PERIODIC: (DECLARED, True),
    FILTER_PERIODIC_HURDLE: (HURDLE, True),
}
#: The models whose published scores the run must give back.
REPRODUCED = (FILTER_DECLARED, FILTER_HURDLE)

#: The records the run reads and must reproduce, and their digests.
FITTED_RATES_PROTOCOL = "artifacts/phase3/fitted_rates_protocol.json"
FITTED_RATES_PROTOCOL_SHA256 = (
    "f695f972fcfcd11266a8a1a720932442e8b7d455bdec7dd6304ecab41843075d"
)
FITTED_RATES_RECORD = "artifacts/phase3/phase3-fitted-rates.json"
FITTED_RATES_RECORD_SHA256 = (
    "807d60fee5adff828c5a727eedbc8858493e816bdf60a555a004ea8a035d590c"
)
PINNED_RECORDS = {
    FITTED_RATES_PROTOCOL: FITTED_RATES_PROTOCOL_SHA256,
    FITTED_RATES_RECORD: FITTED_RATES_RECORD_SHA256,
}


def _estimand(
    key: str, role: str, question: str, model: str, reference: str
) -> Estimand:
    return Estimand(
        key, role, "formulation", question, model, RECURSION, reference, RECURSION
    )


ESTIMANDS: tuple[Estimand, ...] = (
    _estimand(
        "K1",
        "primary",
        "what the time prior adds to the reference formulation, the fitted "
        "hurdle channels, in the recursion",
        FILTER_PERIODIC_HURDLE,
        FILTER_HURDLE,
    ),
    _estimand(
        "K2",
        "secondary",
        "what the time prior adds to the declared channels in the recursion",
        FILTER_PERIODIC,
        FILTER_DECLARED,
    ),
    _estimand(
        "K3",
        "secondary",
        "what the fitted channels add with the time prior in the recursion",
        FILTER_PERIODIC_HURDLE,
        FILTER_PERIODIC,
    ),
)
#: The rule each estimand is read by, and those whose per-state recall is kept.
RULES = {"K1": "time prior", "K2": "time prior", "K3": "fitted channels"}
RECALL_ESTIMANDS = ("K1", "K2")
#: The interaction: the prior's gain with the fitted channels against its gain
#: with the declared ones, per household.
INTERACTION_METRICS = ("balanced_accuracy", "log_loss", "calibration_error")

INSPECTED_BEFORE = (
    "The fitted-rates record: in the recursion the fitted hurdle channels "
    "score a mean household balanced accuracy of 0.456 against 0.417 for the "
    "declared channels, FR +0.039 [+0.015, +0.062], a success; with the time "
    "prior on the declared information sets I1 and I3 the fitted channels are "
    "a success as well (F1, F3); and the fitted Poisson channels score 0.477 "
    "in the recursion.",
    "The time-prior record: on I1 the prior is worth +0.131 [+0.120, +0.142] "
    "balanced accuracy to the generative model, almost all of it in the "
    "recall of away; recent history adds a negligible +0.002 to it (S6, a "
    "failure).",
    "The predictive checks: active counts over-dispersed and silence in long "
    "runs within away, home_active and home_inactive; and the fitted channels' "
    "loss of 0.23 of home_active recall.",
    "An unregistered result with the production filter and the earlier "
    "versions of both parts, fitted sensor rates and the v0.3 circadian term: "
    "together they scored 0.434 balanced accuracy, against 0.460 for the "
    "circadian term alone and 0.449 for neither, in docs/real_data.md.",
    "No recursion with the time prior's transition had been run on any CASAS "
    "household before this protocol was frozen; the code was exercised on "
    "simulated homes only.",
)


def periodic_transitions(
    prior: PeriodicStatePrior, ontology: StateOntology, step: Any
) -> np.ndarray:
    """``(24, states, states)``: the prior's transition over one step at each hour.

    A held-out household has no deviation of its own, so its prior is the
    population's.
    """
    timed = prior.ontology(ontology)
    return np.stack([timed.transition_at_hour(step, hour) for hour in range(24)])


def periodic_recursion(
    loglik: np.ndarray, transitions: np.ndarray, hours: Sequence[int], prior: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The filter's recursion with a transition that depends on the local hour.

    As :func:`~sensor_modeling.datasets.channel_models.filter_recursion`, but
    the transition into window ``t`` is ``transitions[hours[t]]``, the hour at
    the end of the window, as the production filter reads it.
    """
    predicted = np.empty_like(loglik, dtype=float)
    posterior = np.empty_like(predicted)
    belief = np.asarray(prior, dtype=float)
    for row in range(loglik.shape[0]):
        belief = belief @ transitions[int(hours[row])]
        predicted[row] = belief
        weights = belief * np.exp(loglik[row] - loglik[row].max())
        belief = weights / weights.sum()
        posterior[row] = belief
    return predicted, posterior


@dataclass(frozen=True)
class CombinedPriorProtocol:
    """Everything the experiment fixes before any household is scored.

    Attributes
    ----------
    base
        The fitted-rates protocol, whose folds, fits, metrics, minimal
        differences and bootstrap are used unchanged.
    tolerance
        How far a reproduced score or fitted parameter may lie from the
        published one before nothing is reported.
    """

    base: FittedRatesProtocol
    tolerance: float = 1e-6
    inspected_before: tuple[str, ...] = INSPECTED_BEFORE
    name: str = EXPERIMENT

    def __post_init__(self) -> None:
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        base = self.base
        return {
            "schema": PROTOCOL_SCHEMA,
            "name": self.name,
            "status": "pre-specified, on the development panel, which earlier work "
            "has inspected; not a held-out claim",
            "question": "whether the Phase 3.1 time prior, entering the filter's "
            "recursion as an hour-dependent transition, adds to the reference "
            "formulation with fitted hurdle channels",
            "inspected_before": list(self.inspected_before),
            "base_protocol": {
                "file": FITTED_RATES_PROTOCOL,
                "sha256": FITTED_RATES_PROTOCOL_SHA256,
                "protocol_sha256": base.sha256(),
                "used": "its folds, its fold fits of the channels and the time "
                "prior on training households only, its metrics, minimal "
                "differences and household bootstrap",
            },
            "households": {
                "folds": [
                    {**fold.to_dict(), "sha256": fold.sha256()} for fold in base.folds
                ],
                "splits_sha256": base.splits_sha256,
            },
            "recursion": {
                "windows": "every window of the recording at the base protocol's "
                f"step of {base.information_sets[0].resolution.step.total_seconds():g}"
                " seconds; both sides of every comparison receive the same windows, "
                "and only labelled windows are scored",
                "without_the_prior": "the ontology's transition over one step for "
                "every window, from its stationary distribution one step before the "
                "first window, as the fitted-rates record ran it",
                "with_the_prior": "the transition into each window is the time "
                "prior's at the local hour at the window's end, the population prior "
                "for every held-out household; the belief one step before the first "
                "window is the prior's distribution at that window's hour",
                "channels": "declared Poisson or fitted hurdle, as the base protocol "
                "fits them",
                "regime": "online: no estimate reads a window after its own",
            },
            "models": {
                name: {"channels": family, "time_prior": timed}
                for name, (family, timed) in MODELS.items()
            },
            "estimands": [
                {**estimand.to_dict(), "rule": RULES[estimand.key]}
                for estimand in ESTIMANDS
            ],
            "criteria": {
                "verdicts": "favours model: mean >= delta and lower bound > 0; "
                "favours reference: mean <= -delta and upper bound < 0; negligible: "
                "the whole interval within (-delta, delta); uncertain: anything else",
                "time prior": "Phase 3.1's rule: success when balanced accuracy "
                "favours the model and neither log loss nor calibration error "
                "favours the reference; failure when balanced accuracy is "
                "negligible or favours the reference; inconclusive otherwise",
                "fitted channels": "the fitted-rates rule: success when calibration "
                "error favours the model and neither log loss nor balanced accuracy "
                "favours the reference; trade-off when calibration error favours "
                "the model and balanced accuracy the reference; failure when "
                "calibration error is negligible or favours the reference",
                "primary": "K1 alone decides whether the combination is adopted as "
                "the formulation the held-out confirmation will freeze",
                "multiplicity": "K2 and K3 are secondary and read by their rules; "
                "nothing is adjusted",
            },
            "per_state_recall": {
                "estimands": list(RECALL_ESTIMANDS),
                "verdict": "the same verdicts with the recall minimal difference",
            },
            "interaction": {
                "what": "per household, the prior's gain with the fitted channels "
                "minus its gain with the declared channels, K1 minus K2",
                "metrics": list(INTERACTION_METRICS),
                "reading": "described with its interval, not judged: a negative "
                "value means the two parts combine less than additively",
            },
            "metrics": list(base.metrics),
            "minimal_differences": dict(base.minimal_differences),
            "bootstrap": {
                "unit": "household",
                "statistics": "mean and median paired differences",
                "resamples": base.resamples,
                "confidence": base.confidence,
                "seed": base.seed,
                "interval": "percentile",
            },
            "the_check": {
                "what": [
                    "every fold's fitted channels and time prior equal the "
                    "fitted-rates record's, parameter by parameter",
                    "every household's scores of filter_declared and filter_hurdle "
                    "in the recursion equal the fitted-rates record's",
                ],
                "tolerance": self.tolerance,
                "otherwise": "nothing is reported, not even per household",
            },
            "pinned_records": dict(PINNED_RECORDS),
            "what_this_cannot_show": [
                "A held-out result: the 20 homes have been inspected in earlier "
                "work. The confirmation on CASAS homes outside the panel needs its "
                "own protocol, frozen after this result, with the Phase 2 "
                "baselines under matched information.",
                "Anything about TIHM or the sleep feature: the CASAS labels are "
                "activities, not the pipeline's daily features.",
            ],
            "tuning": "none: every setting is the base protocol's or declared here",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def check_pinned_records(root: Path = Path(".")) -> None:
    """Refuse unless every pinned record under *root* is the file pinned."""
    for name, digest in PINNED_RECORDS.items():
        raw = (Path(root) / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"{name} is not the file the protocol pins")


def declared_protocol(
    splits: FrozenSplits, root: Path = Path(".")
) -> CombinedPriorProtocol:
    """The protocol as declared, on the frozen fitted-rates protocol.

    Refuses when the base protocol is not the one frozen in the repository at
    *root*, or a pinned record is not the file pinned.
    """
    base = rates_experiment.declared_protocol(splits)
    rates_experiment.check_frozen_protocol(base, Path(root) / FITTED_RATES_PROTOCOL)
    check_pinned_records(root)
    return CombinedPriorProtocol(base)


def write_protocol(protocol: CombinedPriorProtocol, path: Path) -> str:
    """Write the declaration with its digest and the moment it was frozen."""
    payload = {
        **protocol.to_dict(),
        "protocol_sha256": protocol.sha256(),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return protocol.sha256()


def check_frozen_protocol(protocol: CombinedPriorProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    frozen.pop("frozen_at", None)
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after scoring begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


# ----------------------------------------------------------------------------
# Scoring one household
# ----------------------------------------------------------------------------
def household_scores(
    recording: CasasRecording,
    fit: FoldFit,
    *,
    household: str,
    ontology: StateOntology,
    protocol: CombinedPriorProtocol,
) -> dict[str, PredictionMetrics]:
    """Each model's scores in the recursion for one held-out household."""
    states = tuple(ontology.states)
    resolution = protocol.base.information_sets[0].resolution
    counts, rows, labels, moments = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    hours = [moment.hour for moment in moments]
    homogeneous = ontology.transition(resolution.step)
    timed = periodic_transitions(fit.periodic, ontology, resolution.step)
    start = fit.periodic.probabilities([hours[0]])[0]
    truth = [states[int(i)] for i in labels]
    scores: dict[str, PredictionMetrics] = {}
    for name, (family, with_prior) in MODELS.items():
        models = fit.channels.models(recording.registry, resolution, family, ontology)
        loglik = total_loglik(models, counts)
        if with_prior:
            _, posterior = periodic_recursion(loglik, timed, hours, start)
        else:
            _, posterior = filter_recursion(loglik, homogeneous, ontology.stationary())
        scores[name] = _score(truth, posterior[rows], states)
    return scores


# ----------------------------------------------------------------------------
# The checks
# ----------------------------------------------------------------------------
def _close(a: Any, b: Any, tolerance: float) -> bool:
    """Whether two JSON trees agree, numbers to within *tolerance*."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_close(a[k], b[k], tolerance) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_close(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a == b)
    if a is None or b is None:  # strict JSON writes a missing value as null
        other = b if a is None else a
        return other is None or (isinstance(other, float) and math.isnan(other))
    if isinstance(a, int | float) and isinstance(b, int | float):
        if math.isnan(float(a)) or math.isnan(float(b)):
            return math.isnan(float(a)) and math.isnan(float(b))
        return abs(float(a) - float(b)) <= tolerance
    return bool(a == b)


def check_fits(
    fits: Mapping[str, FoldFit], published: Mapping[str, Any], tolerance: float
) -> dict[str, Any]:
    """Refuse fold fits other than the fitted-rates record's.

    The record stores each fit with its digest. A digest changes with the last
    bit of any parameter, so the parameters are compared to within *tolerance*
    and whether the digests also agree is reported beside.
    """
    if set(fits) != set(published):
        raise ValueError("the folds differ from the fitted-rates record's")
    pairs = [
        (fit.channels, published[fold]["channels"]) for fold, fit in fits.items()
    ] + [
        (fit.periodic, published[fold]["periodic_prior"]) for fold, fit in fits.items()
    ]
    differing = [
        theirs.get("sha256", "")
        for mine, theirs in pairs
        if not _close(
            mine.to_dict(),
            {k: v for k, v in theirs.items() if k != "sha256"},
            tolerance,
        )
    ]
    if differing:
        raise ValueError(
            f"{len(differing)} fold fits differ from the fitted-rates record's; "
            "nothing is reported"
        )
    return {
        "folds": sorted(fits),
        "reproduced": True,
        "identical_digests": all(
            mine.sha256() == theirs.get("sha256") for mine, theirs in pairs
        ),
    }


def check_scores(
    scores: Mapping[str, Mapping[str, PredictionMetrics]],
    published: Mapping[str, Mapping[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    """Refuse recursions without the prior whose scores are not the record's."""
    differing = []
    identical = True
    for model in REPRODUCED:
        there = published[f"{model}@{RECURSION}"]
        for home, metrics in scores[model].items():
            mine = json.loads(json.dumps(metrics.to_dict()))
            if home not in there or not _close(mine, there[home], tolerance):
                differing.append(f"{model} {home}")
            else:
                identical = identical and mine == there[home]
    if differing:
        raise ValueError(
            f"{len(differing)} reproduced scores differ from the fitted-rates "
            "record's; nothing is reported"
        )
    return {
        "models": list(REPRODUCED),
        "households": len(scores[REPRODUCED[0]]),
        "reproduced": True,
        "identical": identical,
    }


# ----------------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------------
def _estimand_entry(
    estimand: Estimand,
    scores: Mapping[str, Mapping[str, PredictionMetrics]],
    protocol: CombinedPriorProtocol,
    states: tuple[Any, ...],
) -> dict[str, Any]:
    base = protocol.base
    gap = base.as_gap_protocol()
    comparisons = {
        metric: _paired(
            household_values(scores[estimand.model], metric),
            household_values(scores[estimand.reference], metric),
            metric,
            gap,
        )
        for metric in base.metrics
    }
    verdicts = {
        metric: verdict(comparison, base.minimal_differences[metric])
        for metric, comparison in comparisons.items()
    }
    rule = RULES[estimand.key]
    entry: dict[str, Any] = {
        **estimand.to_dict(),
        "rule": rule,
        "comparisons": comparisons,
        "verdicts": verdicts,
        "conclusion": (
            prior_conclusion(verdicts)
            if rule == "time prior"
            else channel_conclusion(verdicts)
        ),
    }
    if estimand.key in RECALL_ESTIMANDS:
        entry["per_state_recall"] = {}
        for state in states:
            comparison = _paired(
                household_values(scores[estimand.model], recall_of(state)),
                household_values(scores[estimand.reference], recall_of(state)),
                "balanced_accuracy",
                gap,
            )
            entry["per_state_recall"][state.value] = {
                "comparison": comparison,
                "verdict": verdict(comparison, base.minimal_differences["recall"]),
            }
    return entry


def interaction(
    scores: Mapping[str, Mapping[str, PredictionMetrics]],
    protocol: CombinedPriorProtocol,
) -> dict[str, Any]:
    """K1 minus K2 per household, for each declared metric, with its interval."""
    gap = protocol.base.as_gap_protocol()
    out: dict[str, Any] = {}
    for metric in INTERACTION_METRICS:
        values = {model: household_values(scores[model], metric) for model in MODELS}

        def gain(model: str, reference: str) -> dict[str, float | None]:
            return {
                home: (
                    None
                    if values[model][home] is None or values[reference][home] is None
                    else float(values[model][home]) - float(values[reference][home])  # type: ignore[arg-type]
                )
                for home in values[model]
            }

        out[metric] = _paired(
            gain(FILTER_PERIODIC_HURDLE, FILTER_HURDLE),
            gain(FILTER_PERIODIC, FILTER_DECLARED),
            metric,
            gap,
        )
    return out


@dataclass(frozen=True)
class CombinedPriorResult:
    """A completed experiment: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def run_combined_prior(
    recordings: Mapping[str, CasasRecording],
    protocol: CombinedPriorProtocol,
    published: Mapping[str, Any],
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> CombinedPriorResult:
    """Run the pre-specified experiment; report nothing unless the checks hold.

    *published* is the fitted-rates record, as loaded.
    """
    base = protocol.base
    missing = sorted(set(base.homes) - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    ontology = StateOntology()
    states = tuple(ontology.states)
    fits = fold_fits(recordings, base, ontology)
    check: dict[str, Any] = {
        "fits": check_fits(fits, published["results"]["fitted"], protocol.tolerance)
    }
    scores: dict[str, dict[str, PredictionMetrics]] = {model: {} for model in MODELS}
    folds: dict[str, str] = {}
    for fold in base.folds:
        for home in sorted(fold.test):
            folds[home] = fold.name
            for model, metrics in household_scores(
                recordings[home],
                fits[fold.name],
                household=home,
                ontology=ontology,
                protocol=protocol,
            ).items():
                scores[model][home] = metrics
    check["scores"] = check_scores(
        scores, published["household_metrics"], protocol.tolerance
    )
    gap = base.as_gap_protocol()
    estimands = [_estimand_entry(e, scores, protocol, states) for e in ESTIMANDS]
    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified; development panel, which earlier work has inspected",
        "check": check,
        "states": [state.value for state in states],
        "folds": dict(sorted(folds.items())),
        "cells": [
            _cell(model, RECURSION, scores[model], states, gap) for model in MODELS
        ],
        "estimands": estimands,
        "interaction": interaction(scores, protocol),
        "conclusion": next(e["conclusion"] for e in estimands if e["key"] == "K1"),
    }
    record = ExperimentRecord(
        experiment=protocol.name,
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        inference=ONLINE,
        evidence=online_evidence(
            recordings, base.homes, base.information_sets[0].resolution.step
        ),
        seeds=[base.seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions=MATCHED_METRIC_DEFINITIONS,
        notes=[
            "Every setting, estimand and criterion was declared in the protocol "
            "before any household was scored.",
            "The fold fits are the fitted-rates protocol's, on training households "
            "only, and the recursions without the time prior reproduce that "
            "record's scores.",
            "Verdicts compare effect sizes and household bootstrap intervals with "
            "declared minimal differences; they are not significance tests.",
        ],
        inputs=list(inputs),
        split={"folds": [{**f.to_dict(), "sha256": f.sha256()} for f in base.folds]},
        models=[
            ModelRecord(
                name,
                {"channels": family, "time_prior": timed},
                "periodic_recursion" if timed else "filter_recursion",
            )
            for name, (family, timed) in MODELS.items()
        ],
        household_metrics={
            f"{model}@{RECURSION}": {h: m.to_dict() for h, m in per_home.items()}
            for model, per_home in scores.items()
        },
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return CombinedPriorResult(record=record, path=path)


__all__ = [
    "PINNED_RECORDS",
    "CombinedPriorProtocol",
    "check_frozen_protocol",
    "check_pinned_records",
    "declared_protocol",
    "household_scores",
    "interaction",
    "periodic_recursion",
    "periodic_transitions",
    "run_combined_prior",
    "write_protocol",
]
