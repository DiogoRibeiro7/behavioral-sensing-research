"""Phase 5: the external-generalisation evaluation, run exactly as frozen.

The protocol, ``artifacts/phase5/external_protocol.json``, was frozen in commit
``a863bff`` before any external scoring. This module executes it:

1. **Refuse any other protocol**, and reproduce every CASAS population digest
   the protocol pins before anything is scored.
2. **Apply the eligibility rule.** Validate each home under the frozen mapping,
   and keep its tolerated errors' items unscored.
3. **Score the conditions.** The declared rates, the zero-shot recursion and
   the adapted recursion are scored on the same windows: the scored period's
   labelled windows with a scored state.
4. **Estimate the differences.** Each estimand, T, A, C and S, is a paired
   difference per home, with a percentile interval from resampling the scored
   period's local days, and the declared verdict. The declared criteria give
   the conclusions.

Descriptive diagnostics, not in the protocol
--------------------------------------------
The protocol's reporting rule adds no model or metric to the evaluation. The
diagnostics below were declared in this module, before the run, to explain
the results. They enter no estimand, verdict or conclusion.

- **Unsupported observations.** The share of the scored period's sensor
  activations from sensors that feed no model channel, by reason.
- **Mapping coverage.** The scored period's annotated time by disposition:
  exact, approximate, ambiguous, unmappable, conflicting or undeclared.
- **Room structure.** The model channels each home instruments, and whether
  each scored state's room has one.
- **Event rates.** For each instrumented channel and scored state: the
  scored windows' silence and active mean, against the CASAS population's.
- **Calibration shift.** The zero-shot mean confidence, accuracy and their
  gap, against the same model's on the CASAS development panel (Phase 4
  record).
- **State-prior shift.** The scored states' distribution against the
  development panel's and the model's stationary distribution, and the
  distribution of the zero-shot predictions.
- **The in-sample oracle.** The same recursion with each home's own channel
  parameters, fitted on all its labelled windows, including the scored ones,
  at a pooling strength of 0.5. It is a ceiling for these channels and this
  model family, not a condition.

These give a descriptive separation of three losses:

- **Dataset incompatibility:** what cannot be compared, such as unscorable
  annotated time, unsupported states and unsupported observations.
- **Sensing-information limitation:** how far the oracle, knowing the home's
  own channel behaviour, stays from perfect.
- **Model failure:** how far the transferred parameters fall below the oracle.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from ..evaluation.disagreement import generalised_jensen_shannon, jensen_shannon
from ..evaluation.metrics import EPSILON
from ..evaluation.provenance import ExperimentRecord, InputArtifact, ModelRecord
from ..evaluation.resampling import percentile_interval, resample_indices
from ..evaluation.selective import DEFAULT_COVERAGES, _area, _Ranking
from ..external import (
    CanonicalHousehold,
    HouseholdData,
    Outcome,
    to_canonical,
    validate_household,
)
from ..external.contract import DatasetAdapter
from ..external.ordonez import ON, ORDONEZ_MAPPING, TIMEZONE
from ..fusion.regime import ONLINE, NotEnumerated
from ..states.ontology import BehaviouralState, StateOntology
from .casas import CasasRecording
from .channel_models import (
    DECLARED,
    HURDLE,
    HURDLE_NB,
    FittedChannels,
    filter_recursion,
    home_statistics,
    household_channel_counts,
    household_statistics,
    pool_channels,
    total_loglik,
)
from .external_protocol import (
    ADAPTED,
    POPULATION_SHA256,
    ZERO_SHOT,
    ExternalProtocol,
    adaptation_end,
    eligibility,
    model_channel,
    scored_states,
    sensor_table,
)
from .information_sets import EvidenceChannel
from .partial_pooling import PoolingConfig
from .restricted_filter import channel_likelihoods
from .structural_models import fit_samples
from .time_prior_experiment import verdict

RESULT_SCHEMA = "external-results/1"

#: The commit that froze the protocol.
PROTOCOL_COMMIT = "a863bff"

#: Every scored condition, in report order.
CONDITIONS = (DECLARED, ZERO_SHOT, ADAPTED)

#: The in-sample oracle: descriptive only.
ORACLE = "oracle"
ORACLE_STRENGTH = 0.5

#: Metrics with a paired verdict, and whether higher is better.
VERDICT_METRICS = {
    "balanced_accuracy": True,
    "log_loss": False,
    "calibration_error": False,
    "brier": False,
}

#: Calibration bins, as the package's calibration error uses.
BINS = 10

#: Resamples bootstrapped at once, which bounds memory.
_CHUNK = 500


# ----------------------------------------------------------------------------
# Additive per-window statistics, so days can be resampled exactly
# ----------------------------------------------------------------------------
class _Columns:
    def __init__(self, states: int) -> None:
        self.states = states
        self.count = 0
        self.truth = slice(1, 1 + states)
        self.hits = slice(1 + states, 1 + 2 * states)
        self.predicted = slice(1 + 2 * states, 1 + 3 * states)
        self.log_loss = 1 + 3 * states
        self.brier = self.log_loss + 1
        start = self.brier + 1
        self.bin_count = slice(start, start + BINS)
        self.bin_correct = slice(start + BINS, start + 2 * BINS)
        self.bin_confidence = slice(start + 2 * BINS, start + 3 * BINS)
        self.size = start + 3 * BINS


def window_statistics(truth: np.ndarray, posterior: np.ndarray) -> np.ndarray:
    """Each window's additive statistics under the package's metric definitions."""
    n, size = posterior.shape
    cols = _Columns(size)
    x = np.zeros((n, cols.size))
    rows = np.arange(n)
    predicted = np.argmax(posterior, axis=1)
    confidence = posterior.max(axis=1)
    correct = predicted == truth
    x[:, cols.count] = 1.0
    x[rows, cols.truth.start + truth] = 1.0
    x[rows, cols.hits.start + truth] = correct
    x[rows, cols.predicted.start + predicted] = 1.0
    x[:, cols.log_loss] = -np.log(np.maximum(posterior[rows, truth], EPSILON))
    one_hot = np.zeros_like(posterior)
    one_hot[rows, truth] = 1.0
    x[:, cols.brier] = ((posterior - one_hot) ** 2).sum(axis=1)
    bins = np.clip(np.ceil(confidence * BINS).astype(int) - 1, 0, BINS - 1)
    x[rows, cols.bin_count.start + bins] = confidence > 0.0
    x[rows, cols.bin_correct.start + bins] = correct & (confidence > 0.0)
    x[rows, cols.bin_confidence.start + bins] = np.where(
        confidence > 0.0, confidence, 0.0
    )
    return x


def metrics_from_sums(sums: np.ndarray, states: int) -> dict[str, np.ndarray]:
    """The package's metrics from summed statistics, over the leading axes."""
    cols = _Columns(states)
    n = sums[..., cols.count]
    truth, hits, predicted = (
        sums[..., cols.truth],
        sums[..., cols.hits],
        sums[..., cols.predicted],
    )
    present = truth > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(present, hits / truth, np.nan)
        precision = np.where(predicted > 0, hits / predicted, 0.0)
        f1 = np.where(
            present & (precision + np.nan_to_num(recall) > 0),
            2 * precision * np.nan_to_num(recall) / (precision + np.nan_to_num(recall)),
            0.0,
        )
        balanced = np.nansum(np.where(present, recall, 0.0), axis=-1) / present.sum(-1)
        macro_f1 = np.where(present, f1, 0.0).sum(axis=-1) / present.sum(-1)
        count = sums[..., cols.bin_count]
        weight = count / n[..., None]
        gap = np.abs(
            np.where(count > 0, sums[..., cols.bin_correct] / count, 0.0)
            - np.where(count > 0, sums[..., cols.bin_confidence] / count, 0.0)
        )
        ece = np.where(count > 0, weight * gap, 0.0).sum(axis=-1)
        return {
            "n": n,
            "accuracy": hits.sum(axis=-1) / n,
            "balanced_accuracy": balanced,
            "macro_f1": macro_f1,
            "log_loss": sums[..., cols.log_loss] / n,
            "brier": sums[..., cols.brier] / n,
            "calibration_error": ece,
            "chance": 1.0 / present.sum(axis=-1),
            "recall": recall,
            "mean_confidence": sums[..., cols.bin_confidence].sum(axis=-1) / n,
        }


def _interval(values: np.ndarray, confidence: float) -> dict[str, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    found = percentile_interval(finite, confidence)
    return {"low": found.low, "high": found.high, "confidence": confidence}


def _comparison(
    estimate: float,
    replicates: np.ndarray,
    confidence: float,
    minimal: float,
    first: str,
) -> dict[str, Any]:
    """A paired difference, positive favouring *first*, with the declared verdict."""
    interval = _interval(replicates, confidence)
    compared = {"mean": {"estimate": estimate, "interval": interval}}
    reading = verdict(compared, minimal)
    labels = {
        "favours model": f"favours {first}",
        "favours reference": "favours the other",
    }
    return {
        "estimate": estimate,
        "interval": interval,
        "minimal": minimal,
        "verdict": labels.get(reading, reading),
    }


# ----------------------------------------------------------------------------
# One home
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class _Scored:
    truth: np.ndarray
    rows: np.ndarray
    days: np.ndarray
    day_names: tuple[str, ...]
    posteriors: Mapping[str, np.ndarray]
    ensemble: np.ndarray
    counts: Mapping[EvidenceChannel, np.ndarray]
    moments: Sequence[datetime]
    cut: datetime


def _recursions(
    recording: CasasRecording,
    household: str,
    samples: Mapping[str, FittedChannels],
    protocol: ExternalProtocol,
    ontology: StateOntology,
    cut: datetime,
) -> tuple[dict[str, np.ndarray], list[np.ndarray], Any]:
    resolution = protocol.resolution
    counts, rows, labels, moments = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    transition = ontology.transition(resolution.step)
    prior = ontology.stationary()
    population = samples["all"]

    def run(models: Mapping[EvidenceChannel, Any]) -> np.ndarray:
        _, posterior = filter_recursion(total_loglik(models, counts), transition, prior)
        return posterior

    registry = recording.registry
    posteriors = {
        DECLARED: run(population.models(registry, resolution, DECLARED, ontology)),
        ZERO_SHOT: run(population.models(registry, resolution, HURDLE, ontology)),
    }
    own = household_statistics(
        recording, resolution, ontology, household=household, until=cut
    )
    posteriors[ADAPTED] = run(
        pool_channels(
            population,
            registry,
            own,
            household=household,
            resolution=resolution,
            config=protocol.pooling,
            until=cut,
            ontology=ontology,
        ).models()
    )
    last = moments[-1]
    everything = household_statistics(
        recording, resolution, ontology, household=household, until=last
    )
    posteriors[ORACLE] = run(
        pool_channels(
            population,
            registry,
            everything,
            household=household,
            resolution=resolution,
            config=PoolingConfig(ORACLE_STRENGTH),
            until=last,
            ontology=ontology,
        ).models()
    )
    ensemble = [
        posteriors[ZERO_SHOT],
        run(population.models(registry, resolution, HURDLE_NB, ontology)),
        run(samples["half_a"].models(registry, resolution, HURDLE, ontology)),
        run(samples["half_b"].models(registry, resolution, HURDLE, ontology)),
    ]
    return posteriors, ensemble, (counts, rows, labels, moments)


def _error_aurc(
    risk: np.ndarray,
    errors: np.ndarray,
    days: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """The error AURC of ranking by *risk*, under each row of day weights."""
    x = np.column_stack([np.ones(errors.size), errors.astype(float)])
    ranking = _Ranking(risk, days, x, int(days.max()) + 1)
    out = []
    for begin in range(0, weights.shape[0], _CHUNK):
        kept = ranking.contributions(weights[begin : begin + _CHUNK], grid).sum(axis=2)
        out.append(_area(kept[..., 1] / kept[..., 0], grid))
    values: np.ndarray = np.concatenate(out)
    return values


def score_home(
    data: HouseholdData,
    samples: Mapping[str, FittedChannels],
    protocol: ExternalProtocol,
    *,
    development: Mapping[str, Any] | None = None,
    source: str = "",
) -> dict[str, Any]:
    """Every declared result and descriptive diagnostic for one home."""
    ontology = StateOntology()
    space = tuple(ontology.states)
    names = [s.value for s in space]
    report = validate_household(data, protocol.mapping)
    eligible, blocking = eligibility(report)
    zone = ZoneInfo(TIMEZONE)
    cut = adaptation_end(data).replace(tzinfo=zone)
    entry: dict[str, Any] = {
        "validation": {
            "issues": [i.to_dict() for i in report.issues],
            "blocking": blocking,
        },
        "adaptation_end": cut.isoformat(),
    }
    if not eligible:
        entry["eligible"] = False
        return entry
    converted = to_canonical(data, protocol.mapping, strict=False, source=source)
    recording = converted.recording
    posteriors, ensemble, (counts, rows, labels, moments) = _recursions(
        recording, data.household, samples, protocol, ontology, cut
    )
    scored_index = {names.index(s) for s in scored_states(protocol.mapping)}
    keep = np.array(
        [
            moments[int(r)] > cut and int(t) in scored_index
            for r, t in zip(rows, labels)
        ],
        dtype=bool,
    )
    rows, truth = rows[keep], labels[keep]
    step = protocol.resolution.step
    dates = [(moments[int(r)] - step).date().isoformat() for r in rows]
    day_names = tuple(sorted(set(dates)))
    days = np.array([day_names.index(d) for d in dates])
    routed = sorted(c.name for c in counts)
    entry["eligibility"] = {
        "scored_days": len(day_names),
        "routed_channels": routed,
        "timezone": data.timezone,
    }
    eligible = (
        len(day_names) >= protocol.adaptation_days
        and data.timezone is not None
        and len(routed) >= 2
    )
    entry["eligible"] = bool(eligible)
    if not eligible:
        return entry

    # The conditions, and the oracle, on the scored windows.
    conditions = (*CONDITIONS, ORACLE)
    stats = {c: window_statistics(truth, posteriors[c][rows]) for c in conditions}
    by_day = {}
    for c, x in stats.items():
        sums = np.zeros((len(day_names), x.shape[1]))
        np.add.at(sums, days, x)
        by_day[c] = sums
    draws = resample_indices(len(day_names), protocol.resamples, protocol.seed)
    weights = np.stack(
        [np.bincount(d, minlength=len(day_names)) for d in draws]
    ).astype(float)
    point = {
        c: metrics_from_sums(by_day[c].sum(axis=0), len(space)) for c in conditions
    }
    replicate = {
        c: metrics_from_sums(weights @ by_day[c], len(space)) for c in conditions
    }
    confidence = protocol.confidence
    minimal = protocol.minimal_differences

    def level(c: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for metric in (
            "balanced_accuracy",
            "log_loss",
            "brier",
            "calibration_error",
            "macro_f1",
            "accuracy",
            "mean_confidence",
        ):
            out[metric] = {
                "estimate": float(point[c][metric]),
                "interval": _interval(replicate[c][metric], confidence),
            }
        out["per_class_recall"] = {
            names[k]: (None if math.isnan(v) else float(v))
            for k, v in enumerate(point[c]["recall"])
        }
        out["n"] = int(point[c]["n"])
        return out

    def paired(first: str, second: str) -> dict[str, Any]:
        out = {}
        for metric, higher in VERDICT_METRICS.items():
            sign = 1.0 if higher else -1.0
            estimate = sign * float(point[first][metric] - point[second][metric])
            replicates = sign * (replicate[first][metric] - replicate[second][metric])
            out[metric] = _comparison(
                estimate, replicates, confidence, minimal[metric], first
            )
        return out

    # Selective: confidence against the structural ensemble's discordance.
    stacked = np.stack([p[rows] for p in ensemble], axis=1)
    discordance = generalised_jensen_shannon(stacked) / math.log2(len(ensemble))
    zero = posteriors[ZERO_SHOT][rows]
    errors = np.argmax(zero, axis=1) != truth
    grid = np.array(DEFAULT_COVERAGES)
    ones = np.ones((1, len(day_names)))
    aurc = {
        "confidence": _error_aurc(-zero.max(axis=1), errors, days, ones, grid)[0],
        "structural": _error_aurc(discordance, errors, days, ones, grid)[0],
    }
    aurc_replicates = {
        "confidence": _error_aurc(-zero.max(axis=1), errors, days, weights, grid),
        "structural": _error_aurc(discordance, errors, days, weights, grid),
    }
    chance_estimate = float(
        point[ZERO_SHOT]["balanced_accuracy"] - point[ZERO_SHOT]["chance"]
    )
    chance_replicates = (
        replicate[ZERO_SHOT]["balanced_accuracy"] - replicate[ZERO_SHOT]["chance"]
    )
    entry.update(
        {
            "scored_windows": int(rows.size),
            "scored_days": list(day_names),
            "conditions": {c: level(c) for c in CONDITIONS},
            "chance": float(point[ZERO_SHOT]["chance"]),
            "selective": {
                "confidence_aurc": float(aurc["confidence"]),
                "structural_aurc": float(aurc["structural"]),
            },
            "estimands": {
                "T": paired(ZERO_SHOT, DECLARED),
                "A": paired(ADAPTED, ZERO_SHOT),
                "C": {
                    "balanced_accuracy": _comparison(
                        chance_estimate,
                        chance_replicates,
                        confidence,
                        minimal["chance"],
                        ZERO_SHOT,
                    )
                },
                "S": {
                    "aurc": _comparison(
                        float(aurc["confidence"] - aurc["structural"]),
                        aurc_replicates["confidence"] - aurc_replicates["structural"],
                        confidence,
                        minimal["aurc"],
                        "structural",
                    )
                },
            },
            "diagnostics": _diagnostics(
                data,
                converted,
                samples["all"],
                protocol,
                counts=counts,
                rows=rows,
                truth=truth,
                cut=cut,
                zero=zero,
                oracle=level(ORACLE),
                development=development,
                ontology=ontology,
            ),
        }
    )
    return entry


# ----------------------------------------------------------------------------
# Descriptive diagnostics
# ----------------------------------------------------------------------------
def _segment_reason(label: str, state: BehaviouralState | None) -> str:
    resolutions = [ORDONEZ_MAPPING.label(part) for part in label.split("+")]
    if state is not None:
        approximate = any(
            r.status == Outcome.APPROXIMATE.value for r in resolutions if r.usable
        )
        return Outcome.APPROXIMATE.value if approximate else Outcome.EXACT.value
    if len({r.target for r in resolutions if r.usable}) > 1:
        return "conflict"
    statuses = sorted({r.status for r in resolutions if not r.usable})
    return statuses[0] if len(statuses) == 1 else "unusable"


def _distribution(values: Mapping[str, float]) -> dict[str, float]:
    total = sum(values.values())
    return {k: v / total for k, v in values.items()} if total > 0 else dict(values)


def _divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    a = np.array([[p.get(k, 0.0) for k in keys]])
    b = np.array([[q.get(k, 0.0) for k in keys]])
    return float(jensen_shannon(a, b)[0])


def _diagnostics(
    data: HouseholdData,
    converted: CanonicalHousehold,
    population: FittedChannels,
    protocol: ExternalProtocol,
    *,
    counts: Mapping[EvidenceChannel, np.ndarray],
    rows: np.ndarray,
    truth: np.ndarray,
    cut: datetime,
    zero: np.ndarray,
    oracle: Mapping[str, Any],
    development: Mapping[str, Any] | None,
    ontology: StateOntology,
) -> dict[str, Any]:
    names = [s.value for s in ontology.states]
    scored = scored_states(protocol.mapping)
    naive_cut = cut.replace(tzinfo=None)
    home = data.household

    # Unsupported observations: the scored period's activations by model channel.
    described = {s.sensor_id: s for s in data.sensors}
    reasons: dict[str, int] = {}
    total = 0
    for event in data.events:
        if event.value != ON or event.timestamp <= naive_cut:
            continue
        total += 1
        sensor = described.get(event.sensor_id)
        if sensor is None:
            reason = "unknown sensor"
        else:
            location, kind, place = sensor.sensor_id.split(".")
            channel, why = model_channel((location, kind, place))
            reason = "routed" if channel else why
        reasons[reason] = reasons.get(reason, 0) + 1
    unsupported = total - reasons.get("routed", 0)

    # Mapping coverage over the scored period.
    coverage: dict[str, float] = {}
    for segment in converted.recording.activities:
        start, end = max(segment.start, cut), segment.end
        if end <= start:
            continue
        key = _segment_reason(segment.label, segment.state)
        coverage[key] = coverage.get(key, 0.0) + (end - start).total_seconds()
    labelled = sum(coverage.values())
    scorable = coverage.get("exact", 0.0) + coverage.get("approximate", 0.0)

    # Room structure.
    table = sensor_table(home)
    routed = sorted({r["model_channel"] for r in table if r["model_channel"]})
    rooms = {s: ontology.rooms.get(BehaviouralState(s)) for s in scored}

    # Event rates on the scored windows, against the population's.
    declared = {
        c: t.expected for c, t in _declared_terms(converted, protocol, ontology).items()
    }
    rates: dict[str, Any] = {}
    for evidence, column in sorted(counts.items()):
        values = np.asarray(column, dtype=float)[rows]
        fitted = population.parameters(evidence, declared[evidence])
        per_state: dict[str, Any] = {}
        for s in scored:
            k = names.index(s)
            mask = truth == k
            if not mask.any():
                continue
            here = values[mask]
            active = here[here > 0]
            per_state[s] = {
                "windows": int(mask.sum()),
                "silence": float((here == 0).mean()),
                "casas_silence": float(fitted["silence"][k]),
                "active_mean": float(active.mean()) if active.size else None,
                "casas_active_mean": float(fitted["active_mean"][k]),
            }
        rates[evidence.name] = per_state

    # Calibration and state-prior shift.
    predicted = np.argmax(zero, axis=1)
    accuracy = float((predicted == truth).mean())
    confidence = float(zero.max(axis=1).mean())
    truth_share = _distribution(
        {s: float((truth == names.index(s)).sum()) for s in scored}
    )
    predicted_share = {
        names[k]: float((predicted == k).mean()) for k in range(len(names))
    }
    stationary = ontology.stationary()
    stationary_share = _distribution(
        {s: float(stationary[names.index(s)]) for s in scored}
    )
    shift: dict[str, Any] = {
        "truth": truth_share,
        "stationary": stationary_share,
        "predicted": predicted_share,
        "predicted_unsupported": float(
            sum(v for k, v in predicted_share.items() if k not in scored)
        ),
        "divergence_truth_stationary": _divergence(truth_share, stationary_share),
    }
    calibration: dict[str, Any] = {
        "mean_confidence": confidence,
        "accuracy": accuracy,
        "gap": confidence - accuracy,
    }
    if development is not None:
        shift["development"] = _distribution(
            {s: development["states"][s]["share"] for s in scored}
        )
        shift["divergence_truth_development"] = _divergence(
            truth_share, shift["development"]
        )
        calibration["development"] = dict(development["calibration"])
        calibration["development"]["gap"] = (
            calibration["development"]["mean_confidence"]
            - calibration["development"]["accuracy"]
        )

    return {
        "unsupported_observations": {
            "activations": total,
            "unsupported": unsupported,
            "fraction": unsupported / total if total else None,
            "by_reason": dict(sorted(reasons.items())),
        },
        "mapping_coverage": {
            "seconds": dict(sorted(coverage.items())),
            "scorable_fraction": scorable / labelled if labelled else None,
            "dispositions": {
                "point_annotations": converted.dispositions["point_annotations"],
                "impossible_intervals": converted.dispositions["impossible_intervals"],
                "events": converted.dispositions["events"],
            },
        },
        "room_structure": {
            "routed_channels": routed,
            "uninstrumented_channels": protocol.to_dict()["uninstrumented_channels"][
                home
            ],
            "scored_state_rooms": {
                s: {
                    "room": room,
                    "channel": (
                        f"{room}_motion"
                        if room and f"{room}_motion" in routed
                        else None
                    ),
                }
                for s, room in rooms.items()
            },
            "approximate_sensor_types": sorted(
                {r["native_type"] for r in table if r["type_status"] == "approximate"}
            ),
        },
        "event_rates": rates,
        "calibration_shift": calibration,
        "state_prior_shift": shift,
        "oracle": oracle,
    }


def _declared_terms(
    converted: CanonicalHousehold, protocol: ExternalProtocol, ontology: StateOntology
) -> Mapping[EvidenceChannel, Any]:
    terms, _ = channel_likelihoods(
        converted.recording.registry, protocol.resolution, ontology
    )
    return terms


def attribution(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The descriptive separation of one home's losses."""
    diagnostics = entry["diagnostics"]
    zero = entry["conditions"][ZERO_SHOT]["balanced_accuracy"]["estimate"]
    oracle = diagnostics["oracle"]["balanced_accuracy"]["estimate"]
    return {
        "dataset_incompatibility": {
            "unscorable_labelled_fraction": (
                1.0 - diagnostics["mapping_coverage"]["scorable_fraction"]
                if diagnostics["mapping_coverage"]["scorable_fraction"] is not None
                else None
            ),
            "unsupported_observation_fraction": diagnostics["unsupported_observations"][
                "fraction"
            ],
        },
        "sensing_limitation": {
            "oracle_balanced_accuracy": oracle,
            "oracle_shortfall": 1.0 - oracle,
            "oracle_above_chance": oracle - entry["chance"],
        },
        "model_failure": {
            "zero_shot_below_oracle": oracle - zero,
            "calibration_gap": diagnostics["calibration_shift"]["gap"],
        },
    }


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------
def reproduce_populations(
    recordings: Mapping[str, CasasRecording], protocol: ExternalProtocol
) -> dict[str, FittedChannels]:
    """The protocol's CASAS populations, refused unless every digest matches."""
    ontology = StateOntology()
    homes = protocol.development_homes
    statistics = {
        h: home_statistics(recordings[h], protocol.resolution, ontology, household=h)
        for h in homes
    }
    samples = fit_samples(
        statistics,
        homes,
        states=tuple(ontology.states),
        pseudo_windows=protocol.pseudo_windows,
    )
    for name, fit in samples.items():
        if fit.sha256() != POPULATION_SHA256[name]:
            raise ValueError(
                f"the {name} population does not reproduce its frozen digest; "
                "nothing is scored"
            )
    return samples


def _conclusion(verdicts: Sequence[str], estimand: str) -> str:
    first = {"T": ZERO_SHOT, "A": ADAPTED, "C": ZERO_SHOT, "S": "structural"}[estimand]
    good = f"favours {first}"
    if all(v == good for v in verdicts):
        return {"T": "transfers", "A": "helps", "C": "above chance", "S": "survives"}[
            estimand
        ]
    if all(v in ("negligible", "favours the other") for v in verdicts):
        if estimand == "S":
            return (
                "reverses"
                if all(v == "favours the other" for v in verdicts)
                else ("inconclusive")
            )
        return {
            "T": "does not transfer",
            "A": "does not help",
            "C": "not above chance",
        }[estimand]
    return "inconclusive"


def conclusions(households: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The declared criteria, over the eligible homes."""
    homes = [h for h, e in households.items() if e.get("eligible")]
    out: dict[str, Any] = {"homes": homes}
    for estimand in ("T", "A"):
        accuracy = [
            households[h]["estimands"][estimand]["balanced_accuracy"]["verdict"]
            for h in homes
        ]
        loss = [
            households[h]["estimands"][estimand]["log_loss"]["verdict"] for h in homes
        ]
        reading = _conclusion(accuracy, estimand)
        if reading in ("transfers", "helps") and any(
            v == "favours the other" for v in loss
        ):
            reading = "inconclusive"
        out[estimand] = {
            "conclusion": reading,
            "balanced_accuracy": dict(zip(homes, accuracy)),
            "log_loss": dict(zip(homes, loss)),
        }
    for estimand, metric in (("C", "balanced_accuracy"), ("S", "aurc")):
        verdicts = [
            households[h]["estimands"][estimand][metric]["verdict"] for h in homes
        ]
        out[estimand] = {
            "conclusion": _conclusion(verdicts, estimand),
            metric: dict(zip(homes, verdicts)),
        }
    out["improved"] = {
        estimand: {
            metric: {
                "improved": sum(
                    1
                    for h in homes
                    if households[h]["estimands"][estimand][metric]["estimate"] > 0
                ),
                "worsened": sum(
                    1
                    for h in homes
                    if households[h]["estimands"][estimand][metric]["estimate"] < 0
                ),
            }
            for metric in VERDICT_METRICS
        }
        for estimand in ("T", "A")
    }
    return out


def development_reference(record: Mapping[str, Any]) -> dict[str, Any]:
    """The same model's calibration and state shares on the development panel."""
    random = record["selective_prediction"]["reference"]["random"]
    return {
        "record": record["experiment"],
        "calibration": {
            "mean_confidence": random["mean_confidence"],
            "accuracy": 1.0 - random["error"],
            "calibration_error": random["calibration_error"],
            "balanced_accuracy": random["balanced_accuracy"],
        },
        "states": {
            s: {"share": e["share"]} for s, e in record["results"]["states"].items()
        },
    }


@dataclass(frozen=True)
class ExternalResult:
    """A completed run: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def run_external(
    adapter: DatasetAdapter,
    samples: Mapping[str, FittedChannels],
    protocol: ExternalProtocol,
    *,
    protocol_sha256: str,
    development: Mapping[str, Any] | None = None,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> ExternalResult:
    """Score every candidate home as the frozen protocol declares."""
    households: dict[str, dict[str, Any]] = {}
    for home in adapter.households():
        entry = score_home(
            adapter.load(home),
            samples,
            protocol,
            development=development,
            source=adapter.provenance.name,
        )
        if entry.get("eligible"):
            entry["attribution"] = attribution(entry)
        households[home] = entry
    results = {
        "result_schema": RESULT_SCHEMA,
        "protocol": {
            "file": "artifacts/phase5/external_protocol.json",
            "sha256": protocol.sha256(),
            "file_sha256": protocol_sha256,
            "frozen_in": PROTOCOL_COMMIT,
        },
        "populations": {
            name: {
                "sha256": fit.sha256(),
                "reproduced": fit.sha256() == POPULATION_SHA256[name],
            }
            for name, fit in samples.items()
        },
        "scored_states": scored_states(protocol.mapping),
        "households": households,
        "conclusions": conclusions(households),
        "development": development,
        "descriptive_note": "diagnostics and attribution are descriptive, declared "
        "in the scoring code before the run, and enter no estimand or conclusion",
    }
    eligible = [h for h, e in households.items() if e.get("eligible")]
    record = ExperimentRecord(
        experiment="phase5-external-ordonez-results",
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        inference=ONLINE,
        evidence=NotEnumerated(
            "per-home 5-minute online windows of the external recordings; the "
            "recursion reads no window after the scored one"
        ),
        seeds=[protocol.seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions={
            "balanced_accuracy": "Mean recall over the scored states present in "
            "the home's scored windows.",
            "log_loss": "Mean negative log posterior of the true state, floored "
            "at 1e-12.",
            "brier": "Mean squared error of the posterior against the one-hot "
            "truth, summed over the states.",
            "calibration_error": "Expected calibration error over 10 confidence "
            "bins.",
            "chance": "One over the number of scored states present.",
            "aurc": "Error averaged over the default coverage grid when rejecting "
            "by the signal, within the home.",
        },
        notes=[
            f"The protocol was frozen in {PROTOCOL_COMMIT} before any external "
            "scoring; this run checks the frozen file and reproduces every "
            "population digest before it scores.",
            "Intervals resample the scored period's local days within each home; "
            "two homes support no between-household inference.",
            "Diagnostics, the oracle and the attribution are descriptive and "
            "enter no conclusion.",
        ],
        inputs=list(inputs),
        models=[
            ModelRecord(DECLARED, {"specification": "declared"}, "filter_recursion"),
            ModelRecord(
                ZERO_SHOT,
                {"specification": "hurdle/population/all"},
                "filter_recursion",
            ),
            ModelRecord(
                ADAPTED,
                {"pooling": protocol.pooling.to_dict(), "until": "adaptation end"},
                "pool_channels and filter_recursion",
            ),
            ModelRecord(
                ORACLE,
                {"pooling": {"strength": ORACLE_STRENGTH}, "until": "last window"},
                "descriptive in-sample ceiling",
            ),
        ],
        household_metrics={
            condition: {
                h: {
                    k: v["estimate"] if isinstance(v, dict) and "estimate" in v else v
                    for k, v in households[h]["conditions"][condition].items()
                }
                for h in eligible
            }
            for condition in CONDITIONS
        },
    )
    path = (
        record.write(Path(output_dir) / "phase5-external-ordonez-results.json")
        if output_dir is not None
        else None
    )
    return ExternalResult(record=record, path=path)


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return _finite(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def protocol_file_sha256(path: Path) -> str:
    """SHA-256 of a text file with its line endings normalised to LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
