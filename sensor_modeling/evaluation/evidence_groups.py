"""Disagreement between evidence groups: which kinds of evidence support which state.

The filter combines every sensor as conditionally independent given the state.
When motion sensors point to one behaviour and the bed to another, the
posterior settles between them, and its confidence alone does not say that the
evidence conflicted. This module splits each filter update's evidence into
groups of sensors of one kind, records what each group supports, and measures
how far the groups disagree. It is diagnostic only: it changes no filter output
and sets no abstention rule.

The groups
----------
Sensor groups come from each sensor's modality in the registry:

============  ====================================================
Group         Modalities
============  ====================================================
``motion``    motion, radar: ambient movement
``contact``   door, contact, vibration: doors and objects handled
``bed``       bed pressure
``wearable``  wearable motion and physiology, proximity beacons:
              evidence bound to the person
``other``     any other modality with an emission model
============  ====================================================

``context`` is the filter's own prediction before the window's evidence: what
the state dynamics, and a circadian ontology's time of day, expect.

What each group supports
------------------------
A sensor group's posterior is its evidence in the window alone: the summed
log-likelihoods of its available sensors, tempered by reliability and
attribution exactly as the filter tempers them, and normalised over the states
with no prior. The context group's posterior is the prediction. The full
posterior is the filter's.

A group *favours* the states within :data:`TIE` of its largest log posterior.
Several states can tie: a bed sensor cannot tell sleep from lying awake, so it
favours both, and neither is its choice over the other.

Missing is not disagreement
---------------------------
A sensor's status keeps the health semantics of
:class:`~sensor_modeling.fusion.StateEstimate`:

``unavailable``
    Reliability below the filter's evidence floor: the estimate's *missing*.
``reporting``
    Available, and supplied records in the window.
``silent``
    Available, no records, and a likelihood that is not flat: a trusted event
    sensor's silence counts against the states that would have triggered it.
``no data``
    Available, no records, and a flat likelihood: a state or sample sensor
    says nothing between reports.

The estimate's *silent* is ``silent`` and ``no data`` together. A group is
``absent`` without sensors, ``unavailable`` when every sensor is,
``uninformative`` when its available sensors' likelihood is flat, and
otherwise ``reporting`` or ``silent`` by whether any available sensor
reported. Only ``reporting`` and ``silent`` groups, and the context, are
*identifiable* and enter the comparison. Unavailable sensors' residual
evidence stays out of the group's posterior but is kept in the decomposition
below, which accounts for everything the filter combined.

Disagreement
------------
Over the identifiable groups:

* two groups *disagree on the state* when the states they favour are disjoint;
* the *vote disagreement* is the share of groups not voting for the plurality
  state, each group splitting its vote across the states it favours;
* every pair has a Jensen-Shannon divergence in bits, in ``[0, 1]``, and all
  of them together a generalised divergence, normalised to ``[0, 1]``;
* two disagreeing groups *conflict* when each gives its own favoured states at
  least :data:`CONFLICT_RATIO` times the likelihood of every state the other
  favours. The ratio describes the evidence; nothing decides by it.

Dominance
---------
With ``s*`` the full posterior's most probable state and ``s2`` the next, the
log odds between them split exactly into one term per group::

    log P(s*) / P(s2) = log p(s*) / p(s2) + Σ_g [ℓ_g(s*) − ℓ_g(s2)]

where ``p`` is the prediction and ``ℓ_g`` the group's summed log-likelihood.
A group's share is the absolute value of its term over the sum of all absolute
terms. The group with the largest share *dominates* when its share exceeds
:data:`DOMINANCE_SHARE`: it moves the decision more than every other group
together. Two groups that cancel exactly, each with half, dominate nothing.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from ..fusion.estimate import StateEstimate
from ..fusion.filter import MultimodalBayesFilter, WindowTerms
from ..observations.types import Modality
from .disagreement import generalised_jensen_shannon, jensen_shannon

#: The layout of a recorder's results.
RESULT_LAYOUT = "evidence-group-disagreement/1"

#: The format of a written per-window trace.
TRACE_FORMAT = "evidence-group-trace/1"

#: Log-probability differences below this are ties; a log-likelihood whose
#: range is below it is flat and favours no state.
TIE = 1e-9

#: Two disagreeing groups *conflict* when each gives its own favoured states at
#: least this likelihood ratio over every state the other favours.
CONFLICT_RATIO = 2.0

#: Share of the decision's log odds above which one group dominates.
DOMINANCE_SHARE = 0.5


class EvidenceGroup(str, Enum):
    """A kind of evidence: sensors of related modalities, or the prediction."""

    MOTION = "motion"
    CONTACT = "contact"
    BED = "bed"
    WEARABLE = "wearable"
    OTHER = "other"
    CONTEXT = "context"


#: Which group each modality's sensors belong to; any other modality is ``other``.
MODALITY_GROUPS: dict[Modality, EvidenceGroup] = {
    Modality.MOTION: EvidenceGroup.MOTION,
    Modality.RADAR: EvidenceGroup.MOTION,
    Modality.DOOR: EvidenceGroup.CONTACT,
    Modality.CONTACT: EvidenceGroup.CONTACT,
    Modality.VIBRATION: EvidenceGroup.CONTACT,
    Modality.BED_PRESSURE: EvidenceGroup.BED,
    Modality.WEARABLE_MOTION: EvidenceGroup.WEARABLE,
    Modality.WEARABLE_PHYSIOLOGY: EvidenceGroup.WEARABLE,
    Modality.PROXIMITY: EvidenceGroup.WEARABLE,
}

#: The sensor groups, in report order; the context follows them.
SENSOR_GROUPS = (
    EvidenceGroup.MOTION,
    EvidenceGroup.CONTACT,
    EvidenceGroup.BED,
    EvidenceGroup.WEARABLE,
    EvidenceGroup.OTHER,
)

#: Every group, in report order.
GROUPS = (*SENSOR_GROUPS, EvidenceGroup.CONTEXT)

# Sensor and group statuses.
ABSENT = "absent"
UNAVAILABLE = "unavailable"
UNINFORMATIVE = "uninformative"
NO_DATA = "no data"
SILENT = "silent"
REPORTING = "reporting"
PREDICTED = "predicted"

#: Group statuses that enter the comparison.
IDENTIFIABLE = frozenset({REPORTING, SILENT, PREDICTED})


def group_of(modality: Modality) -> EvidenceGroup:
    """The evidence group of a sensor of *modality*."""
    return MODALITY_GROUPS.get(Modality(modality), EvidenceGroup.OTHER)


def _flat(values: np.ndarray) -> bool:
    return bool(np.ptp(values) <= TIE)


def _normalised(log_values: np.ndarray) -> np.ndarray:
    values: np.ndarray = np.exp(log_values - logsumexp(log_values))
    return values


def _log(probabilities: np.ndarray) -> np.ndarray:
    values: np.ndarray = np.log(np.maximum(probabilities, 1e-300))
    return values


@dataclass(frozen=True)
class GroupEvidence:
    """What one evidence group contributed to one window.

    Attributes
    ----------
    group
        The group.
    status
        ``absent``, ``unavailable``, ``uninformative``, ``silent`` or
        ``reporting``; ``predicted`` for the context.
    sensors
        Each of the group's sensors and its own status.
    loglik
        Every term the filter combined for the group: its sensors' summed
        log-likelihoods, unavailable ones included, or the log prediction.
    posterior
        What the group supports: its available sensors' normalised likelihood,
        or the prediction. ``None`` unless the group is identifiable.
    """

    group: EvidenceGroup
    status: str
    sensors: Mapping[str, str]
    loglik: np.ndarray
    posterior: np.ndarray | None

    @property
    def identifiable(self) -> bool:
        """Whether the group enters the comparison."""
        return self.posterior is not None

    @property
    def favoured(self) -> frozenset[int]:
        """The states the group's posterior ties at its largest; empty if not identifiable."""
        if self.posterior is None:
            return frozenset()
        logs = _log(self.posterior)
        return frozenset(int(i) for i in np.flatnonzero(logs >= logs.max() - TIE))


@dataclass(frozen=True)
class GroupWindow:
    """One filter update's evidence, split by group, and how the groups disagree."""

    at: str
    states: tuple[str, ...]
    full: np.ndarray
    groups: tuple[GroupEvidence, ...]
    conflict_ratio: float = CONFLICT_RATIO

    @property
    def identifiable(self) -> tuple[GroupEvidence, ...]:
        """The groups that enter the comparison, in report order."""
        return tuple(g for g in self.groups if g.identifiable)

    @property
    def ranked(self) -> tuple[int, int]:
        """The full posterior's most probable state and the next; ties go to the earlier."""
        ordered = np.argsort(-self.full, kind="stable")
        return int(ordered[0]), int(ordered[1])

    @property
    def decision(self) -> int:
        """The full posterior's most probable state: the filter's own."""
        return self.ranked[0]

    def _pairs(self) -> list[tuple[GroupEvidence, GroupEvidence]]:
        return list(combinations(self.identifiable, 2))

    @staticmethod
    def _key(a: GroupEvidence, b: GroupEvidence) -> str:
        return f"{a.group.value}|{b.group.value}"

    def pairwise(self) -> dict[str, float]:
        """The Jensen-Shannon divergence, in bits, of every identifiable pair."""
        out: dict[str, float] = {}
        for a, b in self._pairs():
            assert a.posterior is not None and b.posterior is not None
            divergence = jensen_shannon(a.posterior[None, :], b.posterior[None, :])
            out[self._key(a, b)] = float(divergence[0])
        return out

    def disagreements(self) -> list[str]:
        """Identifiable pairs whose favoured states are disjoint."""
        return [
            self._key(a, b) for a, b in self._pairs() if not a.favoured & b.favoured
        ]

    def conflicts(self) -> list[str]:
        """Disagreeing pairs where each side is decisive against the other's states."""
        threshold = math.log(self.conflict_ratio)
        found = []
        for a, b in self._pairs():
            if a.favoured & b.favoured:
                continue
            assert a.posterior is not None and b.posterior is not None
            log_a, log_b = _log(a.posterior), _log(b.posterior)
            a_margin = log_a.max() - max(log_a[j] for j in b.favoured)
            b_margin = log_b.max() - max(log_b[i] for i in a.favoured)
            if a_margin >= threshold and b_margin >= threshold:
                found.append(self._key(a, b))
        return found

    def vote_disagreement(self) -> float | None:
        """Share of identifiable groups not voting for the plurality state."""
        groups = self.identifiable
        if len(groups) < 2:
            return None
        votes = np.zeros(len(self.states))
        for g in groups:
            for i in g.favoured:
                votes[i] += 1.0 / len(g.favoured)
        return float(1.0 - votes.max() / len(groups))

    def discordance(self) -> float | None:
        """The generalised Jensen-Shannon divergence of the identifiable groups, in bits."""
        stacked = [g.posterior for g in self.identifiable if g.posterior is not None]
        if len(stacked) < 2:
            return None
        return float(generalised_jensen_shannon(np.stack(stacked)[None, :, :])[0])

    def contributions(self) -> dict[str, float]:
        """Each group's term in the log odds of the decision over the runner-up.

        They sum to the full posterior's log odds between the two states.
        """
        first, second = self.ranked
        return {
            g.group.value: float(g.loglik[first] - g.loglik[second])
            for g in self.groups
        }

    def dominance(self) -> dict[str, Any]:
        """The group with the largest share of the decision's log odds."""
        terms = self.contributions()
        total = sum(abs(v) for v in terms.values())
        if total == 0.0:
            return {"group": None, "share": None, "dominates": False}
        group = max(terms, key=lambda k: abs(terms[k]))
        share = abs(terms[group]) / total
        return {"group": group, "share": share, "dominates": share > DOMINANCE_SHARE}

    def to_dict(self) -> dict[str, Any]:
        """The window's record: every group, the comparison and the dominance."""
        count = len(self.identifiable)
        compared = count >= 2
        discordance = self.discordance()
        first, second = self.ranked
        labels = self.states

        def described(g: GroupEvidence) -> dict[str, Any]:
            posterior = g.posterior
            return {
                "status": g.status,
                "identifiable": g.identifiable,
                "sensors": dict(g.sensors),
                "posterior": (
                    dict(zip(labels, posterior.tolist()))
                    if posterior is not None
                    else None
                ),
                "favoured": [labels[i] for i in sorted(g.favoured)],
                "supports_decision": (
                    self.decision in g.favoured if posterior is not None else None
                ),
                "divergence_from_full": (
                    float(jensen_shannon(posterior[None, :], self.full[None, :])[0])
                    if posterior is not None
                    else None
                ),
            }

        return {
            "at": self.at,
            "full": {
                "posterior": dict(zip(labels, self.full.tolist())),
                "state": labels[first],
                "runner_up": labels[second],
            },
            "groups": {g.group.value: described(g) for g in self.groups},
            "identifiable": [g.group.value for g in self.identifiable],
            "compared": compared,
            "state_disagreement": bool(self.disagreements()) if compared else None,
            "disagreements": self.disagreements(),
            "vote_disagreement": self.vote_disagreement(),
            "pairwise": self.pairwise(),
            "discordance": discordance,
            "normalised_discordance": (
                discordance / math.log2(count) if discordance is not None else None
            ),
            "conflicts": self.conflicts(),
            "contributions": self.contributions(),
            "dominance": self.dominance(),
        }


def _sensor_group(
    group: EvidenceGroup,
    members: Sequence[str],
    terms: WindowTerms,
    evidence_floor: float,
    size: int,
) -> GroupEvidence:
    sensors: dict[str, str] = {}
    loglik = np.zeros(size)
    usable = np.zeros(size)
    for sensor_id in members:
        values = np.asarray(terms.likelihoods[sensor_id], dtype=float)
        loglik = loglik + values
        if terms.reliability[sensor_id] < evidence_floor:
            sensors[sensor_id] = UNAVAILABLE
            continue
        usable = usable + values
        if terms.observations[sensor_id] > 0:
            sensors[sensor_id] = REPORTING
        elif _flat(values):
            sensors[sensor_id] = NO_DATA
        else:
            sensors[sensor_id] = SILENT
    statuses = set(sensors.values())
    if not sensors:
        status = ABSENT
    elif statuses == {UNAVAILABLE}:
        status = UNAVAILABLE
    elif _flat(usable):
        status = UNINFORMATIVE
    elif REPORTING in statuses:
        status = REPORTING
    else:
        status = SILENT
    posterior = _normalised(usable) if status in IDENTIFIABLE else None
    return GroupEvidence(group, status, sensors, loglik, posterior)


def group_window(
    terms: WindowTerms,
    modalities: Mapping[str, Modality],
    states: Sequence[str],
    *,
    evidence_floor: float,
    conflict_ratio: float = CONFLICT_RATIO,
) -> GroupWindow:
    """Split one update's evidence by group, and compare the groups.

    Parameters
    ----------
    terms
        The update's terms, from :meth:`MultimodalBayesFilter.evidence_terms`.
    modalities
        Each sensor's modality, which chooses its group; a sensor not listed
        falls in ``other``.
    states
        The ontology's state labels, in order.
    evidence_floor
        The filter's floor: a sensor whose reliability is below it is missing.
    conflict_ratio
        The likelihood ratio each side of a conflict must reach.
    """
    if not math.isfinite(conflict_ratio) or conflict_ratio <= 1.0:
        raise ValueError("conflict_ratio must exceed 1")
    size = len(states)
    predicted = np.asarray(terms.predicted, dtype=float)
    if predicted.shape != (size,):
        raise ValueError("the terms must have one entry per state")
    members: dict[EvidenceGroup, list[str]] = {g: [] for g in SENSOR_GROUPS}
    for sensor_id in terms.likelihoods:
        members[group_of(modalities.get(sensor_id, Modality.OTHER))].append(sensor_id)
    groups = [
        _sensor_group(group, members[group], terms, evidence_floor, size)
        for group in SENSOR_GROUPS
    ]
    context = predicted / predicted.sum()
    groups.append(
        GroupEvidence(EvidenceGroup.CONTEXT, PREDICTED, {}, _log(predicted), context)
    )
    return GroupWindow(
        at=terms.at.isoformat(),
        states=tuple(states),
        full=terms.posterior,
        groups=tuple(groups),
        conflict_ratio=conflict_ratio,
    )


def _declared_modality(bayes: MultimodalBayesFilter, sensor_id: str) -> Modality:
    """The modality the filter's registry declares for *sensor_id*, or ``OTHER``."""
    spec = bayes.registry.get(sensor_id) if bayes.registry is not None else None
    return spec.modality if spec is not None else Modality.OTHER


def _share(items: Sequence[Any], condition: Callable[[Any], bool]) -> float | None:
    return sum(1 for item in items if condition(item)) / len(items) if items else None


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


class GroupDisagreementRecorder:
    """Records the evidence-group comparison of every update of one filter.

    Attach it to a filter's :attr:`~sensor_modeling.fusion.MultimodalBayesFilter.observers`,
    directly or through :meth:`attach`. It reads each update's terms and changes
    nothing the filter does. Groups come from the filter's registry: a sensor it
    does not declare falls in ``other``.

    Only the plain recursion is supported. A filter that overrides ``update``,
    such as the history-aware one, adds terms the recorder would not see, so it
    is refused.
    """

    def __init__(
        self, bayes: MultimodalBayesFilter, *, conflict_ratio: float = CONFLICT_RATIO
    ) -> None:
        if type(bayes).update is not MultimodalBayesFilter.update:
            raise TypeError(
                f"{type(bayes).__name__} overrides update; its evidence terms "
                "are not the recorded ones"
            )
        if not math.isfinite(conflict_ratio) or conflict_ratio <= 1.0:
            raise ValueError("conflict_ratio must exceed 1")
        self.states = tuple(bayes.ontology.labels())
        self.modalities = {
            sensor_id: _declared_modality(bayes, sensor_id)
            for sensor_id in bayes.emissions
        }
        self.evidence_floor = bayes.config.evidence_floor
        self.conflict_ratio = conflict_ratio
        self.windows: list[dict[str, Any]] = []

    @classmethod
    def attach(
        cls, bayes: MultimodalBayesFilter, *, conflict_ratio: float = CONFLICT_RATIO
    ) -> GroupDisagreementRecorder:
        """A recorder observing *bayes* from its next update on."""
        recorder = cls(bayes, conflict_ratio=conflict_ratio)
        bayes.observers.append(recorder)
        return recorder

    def __call__(self, terms: WindowTerms, estimate: StateEstimate) -> None:
        """Record one update."""
        window = group_window(
            terms,
            self.modalities,
            self.states,
            evidence_floor=self.evidence_floor,
            conflict_ratio=self.conflict_ratio,
        )
        # The estimate renormalises its belief, which can move its last bits.
        if not np.allclose(window.full, estimate.belief, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "the recorded terms do not reproduce the filter's belief"
            )
        self.windows.append(window.to_dict())

    def _group_summary(self, group: str) -> dict[str, Any]:
        windows = self.windows
        statuses: dict[str, int] = {}
        for w in windows:
            status = w["groups"][group]["status"]
            statuses[status] = statuses.get(status, 0) + 1
        present = [w for w in windows if w["groups"][group]["identifiable"]]
        return {
            "statuses": dict(sorted(statuses.items())),
            "identifiable": _share(
                windows, lambda w: w["groups"][group]["identifiable"]
            ),
            "supports_decision": _share(
                present, lambda w: w["groups"][group]["supports_decision"]
            ),
            "mean_divergence_from_full": _mean(
                [w["groups"][group]["divergence_from_full"] for w in present]
            ),
            "dominant": _share(
                windows,
                lambda w: w["dominance"]["group"] == group
                and w["dominance"]["dominates"],
            ),
        }

    def summary(self) -> dict[str, Any]:
        """The recorded windows, summarised: availability, agreement and dominance."""
        windows = self.windows
        compared = [w for w in windows if w["compared"]]
        pairs: dict[str, Any] = {}
        for a, b in combinations([g.value for g in GROUPS], 2):
            key = f"{a}|{b}"
            both = [w for w in windows if key in w["pairwise"]]
            if both:
                pairs[key] = {
                    "windows": len(both),
                    "mean_divergence": _mean([w["pairwise"][key] for w in both]),
                    "disagree": _share(both, lambda w: key in w["disagreements"]),
                    "conflict": _share(both, lambda w: key in w["conflicts"]),
                }
        return {
            "windows": len(windows),
            "compared": len(compared),
            "state_disagreement": _share(compared, lambda w: w["state_disagreement"]),
            "conflict": _share(compared, lambda w: bool(w["conflicts"])),
            "mean_vote_disagreement": _mean([w["vote_disagreement"] for w in compared]),
            "mean_normalised_discordance": _mean(
                [w["normalised_discordance"] for w in compared]
            ),
            "dominated": _share(windows, lambda w: w["dominance"]["dominates"]),
            "groups": {g.value: self._group_summary(g.value) for g in GROUPS},
            "pairs": pairs,
        }

    def results(self, trace: Mapping[str, str] | None = None) -> dict[str, Any]:
        """A machine-readable block for an experiment record's results.

        *trace* references the per-window file, as ``{"path": ..., "sha256": ...}``
        from :meth:`write`.
        """
        return {
            "layout": RESULT_LAYOUT,
            "definitions": {
                "group_posterior": "the group's available sensors' summed, tempered "
                "log-likelihood in the window, normalised with no prior; for the "
                "context, the prediction before the window's evidence",
                "identifiable": "reporting or silent groups whose likelihood is not "
                "flat, and the context; missing evidence is never compared",
                "state_disagreement": "two identifiable groups whose favoured "
                f"states, those within {TIE:g} of the largest log posterior, are "
                "disjoint",
                "divergence": "Jensen-Shannon divergence, base-2 logarithms, in [0, 1]",
                "conflict": "two disagreeing groups each giving its own favoured "
                f"states at least {self.conflict_ratio:g} times the likelihood of "
                "every state the other favours",
                "dominance": "the group with the largest share of the absolute terms "
                "of the decision's log odds over the runner-up; it dominates at a "
                f"share above {DOMINANCE_SHARE:g}",
            },
            "conflict_ratio": self.conflict_ratio,
            "evidence_floor": self.evidence_floor,
            "states": list(self.states),
            "sensors": {
                sensor_id: {
                    "modality": modality.value,
                    "group": group_of(modality).value,
                }
                for sensor_id, modality in sorted(self.modalities.items())
            },
            "summary": self.summary(),
            "trace": dict(trace) if trace is not None else None,
        }

    def write(self, path: Path) -> str:
        """Write every window's record as canonical gzip JSON; return its SHA-256.

        The same windows always give the same bytes.
        """
        text = json.dumps(
            {
                "format": TRACE_FORMAT,
                "states": list(self.states),
                "windows": self.windows,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
            handle.write(text.encode("utf-8"))
        data = buffer.getvalue()
        Path(path).write_bytes(data)
        return hashlib.sha256(data).hexdigest()


def read_trace(path: Path, sha256: str | None = None) -> dict[str, Any]:
    """Read a trace written by :meth:`GroupDisagreementRecorder.write`.

    With *sha256*, a file whose digest differs is refused.
    """
    data = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"the trace at {path} does not match its recorded digest")
    payload: dict[str, Any] = json.loads(gzip.decompress(data).decode("utf-8"))
    if payload.get("format") != TRACE_FORMAT:
        raise ValueError(f"not a {TRACE_FORMAT} trace")
    return payload
