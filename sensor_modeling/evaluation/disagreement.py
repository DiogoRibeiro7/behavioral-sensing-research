"""How much scientifically plausible model specifications disagree, window by window.

Paper 1 found that no scalar read off one posterior (confidence, entropy,
margin, evidence strength or information gain) is a credible standalone
uncertainty signal. ROADMAP Phase 4 asks instead for structural uncertainty:
how much the prediction depends on assumptions the evidence does not settle.
This module measures that from the posteriors of several fitted model
specifications over the same windows. It is diagnostic only. It changes no
decision rule and sets no threshold.

For each window it records:

- each specification's posterior and most probable state;
- the Jensen-Shannon divergence between every pair of posteriors, in bits;
- the vote disagreement: the share of specifications that do not vote for the
  plurality state;
- each state's probability spread: its largest minus its smallest posterior;
- a consensus summary: the mean posterior, its most probable state, and the
  generalised Jensen-Shannon divergence of the specifications, the entropy of
  the mean posterior minus the mean entropy.

Bounds
------
With base-2 logarithms every pairwise divergence lies in ``[0, 1]``. The
generalised divergence of ``M`` equally weighted specifications lies in
``[0, log2 M]``, and is also reported divided by ``log2 M``. Vote disagreement
lies in ``[0, 1 − 1/M]``, and a spread in ``[0, 1]``. Each is zero exactly when
the specifications agree, and each is symmetric in the specifications.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import xlogy

from .households import summarise_households

#: The distance between two posteriors.
DISTANCE = "Jensen-Shannon divergence, base-2 logarithms, in [0, 1]"

#: The format of a written trace.
TRACE_FORMAT = "disagreement-trace/1"

#: How far a posterior's probabilities may sum from one before it is refused.
_TOLERANCE = 1e-6

#: Divergences below this many bits are rounding, and are reported as zero, so
#: identical specifications disagree by exactly nothing.
ROUNDING_FLOOR = 1e-12


def _entropy_bits(probabilities: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits along the last axis; ``0 log 0`` is 0."""
    values: np.ndarray = -xlogy(probabilities, probabilities).sum(axis=-1) / math.log(
        2.0
    )
    return values


def jensen_shannon(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits between matching rows of two posteriors.

    ``H((p + q) / 2) − (H(p) + H(q)) / 2``: symmetric, zero only for identical
    rows, and at most 1, reached by posteriors with disjoint support.
    """
    p = np.asarray(first, dtype=float)
    q = np.asarray(second, dtype=float)
    if p.shape != q.shape:
        raise ValueError("the two posteriors must have the same shape")
    divergence = (
        _entropy_bits((p + q) / 2.0) - (_entropy_bits(p) + _entropy_bits(q)) / 2.0
    )
    clipped: np.ndarray = np.clip(divergence, 0.0, 1.0)
    clipped[clipped < ROUNDING_FLOOR] = 0.0
    return clipped


def generalised_jensen_shannon(posteriors: np.ndarray) -> np.ndarray:
    """``H(mean posterior) − mean H(posterior)`` in bits, per window.

    *posteriors* is ``(windows, models, states)``. The result lies in
    ``[0, log2 models]``.
    """
    stacked = np.asarray(posteriors, dtype=float)
    models = stacked.shape[1]
    values = _entropy_bits(stacked.mean(axis=1)) - _entropy_bits(stacked).mean(axis=1)
    clipped: np.ndarray = np.clip(values, 0.0, math.log2(models) if models > 1 else 0.0)
    clipped[clipped < ROUNDING_FLOOR] = 0.0
    return clipped


def _checked(
    posterior: np.ndarray, name: str, shape: tuple[int, int] | None
) -> np.ndarray:
    values = np.array(posterior, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"the posterior of {name!r} must be (windows, states)")
    if shape is not None and values.shape != shape:
        raise ValueError(f"the posterior of {name!r} does not cover the same windows")
    if not np.all(np.isfinite(values)) or (values < 0.0).any():
        raise ValueError(f"the posterior of {name!r} must be finite and non-negative")
    if np.any(np.abs(values.sum(axis=1) - 1.0) > _TOLERANCE):
        raise ValueError(f"every row of the posterior of {name!r} must sum to one")
    # Kept exactly as given, so the reference posterior, and with it the
    # deployed decision, is the model's own to the last bit.
    return values


def _frozen(values: np.ndarray) -> np.ndarray:
    values = np.array(values)
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class DisagreementTrace:
    """Every window's comparison of several model specifications, for one household.

    Build it with :func:`compare_posteriors`. Every per-window quantity is
    derived from ``posteriors``, so a trace read back from a file is checked by
    recomputing it.

    Attributes
    ----------
    household
        Whose windows these are.
    timestamps
        Each window's moment, ISO 8601.
    states
        The state order of every posterior.
    models
        The specifications' names, in the order of ``posteriors``' second axis.
    reference
        The deployed specification: its most probable state is the decision,
        which this trace does not change.
    posteriors
        ``(windows, models, states)``.
    truth
        Each window's labelled state index, or ``-1``; recorded for later
        analyses, and used by nothing here.
    """

    household: str
    timestamps: tuple[str, ...]
    states: tuple[str, ...]
    models: tuple[str, ...]
    reference: str
    posteriors: np.ndarray
    truth: np.ndarray

    def __post_init__(self) -> None:
        """Validate the layout and freeze the arrays."""
        posteriors = np.asarray(self.posteriors, dtype=float)
        windows = len(self.timestamps)
        if posteriors.shape != (windows, len(self.models), len(self.states)):
            raise ValueError("posteriors must be (windows, models, states)")
        if len(self.models) < 2 or len(set(self.models)) != len(self.models):
            raise ValueError("at least two distinct specifications are required")
        if self.reference not in self.models:
            raise ValueError("the reference must be one of the specifications")
        if len(set(self.states)) != len(self.states) or not self.states:
            raise ValueError("states must be distinct and non-empty")
        truth = np.asarray(self.truth, dtype=int)
        if (
            truth.shape != (windows,)
            or truth.min(initial=0) < -1
            or truth.max(initial=-1) >= len(self.states)
        ):
            raise ValueError("truth needs one state index or -1 per window")
        object.__setattr__(self, "timestamps", tuple(self.timestamps))
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "models", tuple(self.models))
        object.__setattr__(self, "posteriors", _frozen(posteriors))
        object.__setattr__(self, "truth", _frozen(truth))

    # ------------------------------------------------------------------
    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        """Every unordered pair of specifications, in model order."""
        return tuple(combinations(self.models, 2))

    @property
    def argmax(self) -> np.ndarray:
        """``(windows, models)``: each specification's most probable state index."""
        values: np.ndarray = np.argmax(self.posteriors, axis=2)
        return values

    @property
    def decision(self) -> np.ndarray:
        """The deployed decision: the reference specification's most probable state."""
        values: np.ndarray = self.argmax[:, self.models.index(self.reference)]
        return values

    @property
    def pairwise(self) -> np.ndarray:
        """``(windows, pairs)``: the Jensen-Shannon divergence of each pair, in bits."""
        index = {m: i for i, m in enumerate(self.models)}
        columns = [
            jensen_shannon(self.posteriors[:, index[a]], self.posteriors[:, index[b]])
            for a, b in self.pairs
        ]
        values: np.ndarray = np.column_stack(columns)
        return values

    @property
    def votes(self) -> np.ndarray:
        """``(windows, states)``: how many specifications vote for each state."""
        values: np.ndarray = (
            self.argmax[:, :, None] == np.arange(len(self.states))[None, None, :]
        ).sum(axis=1)
        return values

    @property
    def plurality(self) -> np.ndarray:
        """The state with the most votes; a tie goes to the earliest state."""
        values: np.ndarray = np.argmax(self.votes, axis=1)
        return values

    @property
    def vote_disagreement(self) -> np.ndarray:
        """The share of specifications not voting for the plurality state."""
        values: np.ndarray = 1.0 - self.votes.max(axis=1) / len(self.models)
        return values

    @property
    def spread(self) -> np.ndarray:
        """``(windows, states)``: largest minus smallest posterior of each state."""
        values: np.ndarray = self.posteriors.max(axis=1) - self.posteriors.min(axis=1)
        return values

    @property
    def consensus(self) -> np.ndarray:
        """``(windows, states)``: the mean posterior of the specifications."""
        values: np.ndarray = self.posteriors.mean(axis=1)
        return values

    @property
    def discordance(self) -> np.ndarray:
        """The generalised Jensen-Shannon divergence of the specifications, in bits."""
        return generalised_jensen_shannon(self.posteriors)

    @property
    def normalised_discordance(self) -> np.ndarray:
        """The discordance over its maximum, ``log2`` of the specifications: in [0, 1]."""
        values: np.ndarray = self.discordance / math.log2(len(self.models))
        return values

    # ------------------------------------------------------------------
    def windows(self) -> list[dict[str, Any]]:
        """Every window's record, as the task lists it."""
        out: list[dict[str, Any]] = []
        pairwise, argmax, spread = self.pairwise, self.argmax, self.spread
        consensus, discordance = self.consensus, self.discordance
        votes, plurality, disagreement = (
            self.votes,
            self.plurality,
            self.vote_disagreement,
        )
        for t, moment in enumerate(self.timestamps):
            out.append(
                {
                    "timestamp": moment,
                    "posteriors": {
                        m: self.posteriors[t, i].tolist()
                        for i, m in enumerate(self.models)
                    },
                    "argmax": {
                        m: self.states[int(argmax[t, i])]
                        for i, m in enumerate(self.models)
                    },
                    "decision": self.states[
                        int(argmax[t, self.models.index(self.reference)])
                    ],
                    "pairwise": {
                        f"{a}|{b}": float(pairwise[t, j])
                        for j, (a, b) in enumerate(self.pairs)
                    },
                    "votes": {
                        s: int(votes[t, k])
                        for k, s in enumerate(self.states)
                        if votes[t, k]
                    },
                    "plurality": self.states[int(plurality[t])],
                    "vote_disagreement": float(disagreement[t]),
                    "spread": dict(zip(self.states, spread[t].tolist())),
                    "consensus": dict(zip(self.states, consensus[t].tolist())),
                    "consensus_state": self.states[int(np.argmax(consensus[t]))],
                    "discordance": float(discordance[t]),
                    "unanimous": bool(votes[t].max() == len(self.models)),
                }
            )
        return out

    def summary(self) -> dict[str, Any]:
        """The household's disagreement over its windows: means and shares."""
        if not self.timestamps:
            return {"windows": 0}
        pairwise, argmax = self.pairwise, self.argmax
        decision = self.decision
        discordance = self.discordance
        return {
            "windows": len(self.timestamps),
            "discordance": _distribution(discordance),
            "normalised_discordance": _distribution(self.normalised_discordance),
            "mean_pairwise": _distribution(pairwise.mean(axis=1)),
            "vote_disagreement": _distribution(self.vote_disagreement),
            "unanimous": float(np.mean(self.votes.max(axis=1) == len(self.models))),
            "decision_is_plurality": float(np.mean(decision == self.plurality)),
            "spread": {
                s: float(v) for s, v in zip(self.states, self.spread.mean(axis=0))
            },
            "pairs": {
                f"{a}|{b}": {
                    "mean_divergence": float(pairwise[:, j].mean()),
                    "argmax_differs": float(
                        np.mean(
                            argmax[:, self.models.index(a)]
                            != argmax[:, self.models.index(b)]
                        )
                    ),
                }
                for j, (a, b) in enumerate(self.pairs)
            },
            "agrees_with_decision": {
                m: float(np.mean(argmax[:, i] == decision))
                for i, m in enumerate(self.models)
            },
        }

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """The trace's inputs, from which every other quantity is recomputed."""
        return {
            "format": TRACE_FORMAT,
            "household": self.household,
            "timestamps": list(self.timestamps),
            "states": list(self.states),
            "models": list(self.models),
            "reference": self.reference,
            "posteriors": self.posteriors.tolist(),
            "truth": self.truth.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DisagreementTrace:
        """Rebuild a trace written by :meth:`to_dict`."""
        if payload.get("format") != TRACE_FORMAT:
            raise ValueError(f"not a {TRACE_FORMAT} trace")
        return cls(
            household=str(payload["household"]),
            timestamps=tuple(payload["timestamps"]),
            states=tuple(payload["states"]),
            models=tuple(payload["models"]),
            reference=str(payload["reference"]),
            posteriors=np.array(payload["posteriors"], dtype=float).reshape(
                len(payload["timestamps"]),
                len(payload["models"]),
                len(payload["states"]),
            ),
            truth=np.array(payload["truth"], dtype=int),
        )

    def write(self, path: Path) -> str:
        """Write the trace as canonical gzip-compressed JSON; return its SHA-256.

        The gzip header carries no time or name, so writing the same trace twice
        gives the same bytes.
        """
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
            handle.write(text.encode("utf-8"))
        data = buffer.getvalue()
        Path(path).write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def read(cls, path: Path, sha256: str | None = None) -> DisagreementTrace:
        """Read a trace written by :meth:`write`, checking its digest if given."""
        data = Path(path).read_bytes()
        if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError(f"the trace at {path} does not match its recorded digest")
        return cls.from_dict(json.loads(gzip.decompress(data).decode("utf-8")))


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def compare_posteriors(
    posteriors: Mapping[str, np.ndarray],
    *,
    household: str,
    timestamps: Sequence[str],
    states: Sequence[str],
    reference: str,
    truth: np.ndarray | None = None,
) -> DisagreementTrace:
    """Compare several specifications' posteriors over the same windows.

    Parameters
    ----------
    posteriors
        One ``(windows, states)`` posterior per specification, by name. The
        specifications are ordered by name, so the result does not depend on the
        mapping's order.
    household, timestamps, states
        Whose windows, their moments, and the state order.
    reference
        The deployed specification, whose most probable state stays the
        decision.
    truth
        Each window's labelled state index, or ``-1``. Optional.
    """
    names = sorted(posteriors)
    shape = (len(timestamps), len(states))
    stacked = np.stack([_checked(posteriors[n], n, shape) for n in names], axis=1)
    return DisagreementTrace(
        household=household,
        timestamps=tuple(timestamps),
        states=tuple(states),
        models=tuple(names),
        reference=reference,
        posteriors=stacked,
        truth=(
            np.full(len(timestamps), -1, dtype=int)
            if truth is None
            else np.asarray(truth, dtype=int)
        ),
    )


# ----------------------------------------------------------------------------
# In an experiment record
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DisagreementReport:
    """What an experiment record keeps of a structural-disagreement analysis.

    Attributes
    ----------
    specifications
        Each specification's name, the assumptions it makes, and the published
        evidence that supports each assumption.
    reference
        The deployed specification.
    states
        The state order.
    households
        Per household: its windows, the summary of its trace, and where the
        trace is written, with its SHA-256, or ``None`` if it is not.
    """

    specifications: tuple[Mapping[str, Any], ...]
    reference: str
    states: tuple[str, ...]
    households: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        """Validate the report through its serialised form."""
        problems = validate_disagreement(self.to_dict())
        if problems:
            raise ValueError("; ".join(problems))

    @classmethod
    def from_traces(
        cls,
        traces: Sequence[DisagreementTrace],
        specifications: Sequence[Mapping[str, Any]],
        files: Mapping[str, Mapping[str, str]] | None = None,
    ) -> DisagreementReport:
        """A report of *traces*, one per household, with each trace's file if written."""
        if not traces:
            raise ValueError("a report needs at least one household's trace")
        first = traces[0]
        for trace in traces:
            if (trace.models, trace.states, trace.reference) != (
                first.models,
                first.states,
                first.reference,
            ):
                raise ValueError("every trace must compare the same specifications")
        households = {
            trace.household: {
                "summary": trace.summary(),
                "trace": (
                    dict(files[trace.household])
                    if files and trace.household in files
                    else None
                ),
            }
            for trace in traces
        }
        return cls(tuple(specifications), first.reference, first.states, households)

    def across_households(self) -> dict[str, Any]:
        """Each household's mean disagreement, described across households."""

        def values(key: str) -> dict[str, float | None]:
            return {
                h: (e["summary"][key]["mean"] if e["summary"].get("windows") else None)
                for h, e in self.households.items()
            }

        return {
            key: summarise_households(values(key)).to_dict()
            for key in (
                "discordance",
                "normalised_discordance",
                "mean_pairwise",
                "vote_disagreement",
            )
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the serialisable form an experiment record carries."""
        return {
            "distance": DISTANCE,
            "trace_format": TRACE_FORMAT,
            "reference": self.reference,
            "states": list(self.states),
            "specifications": [dict(s) for s in self.specifications],
            "households": {h: dict(e) for h, e in sorted(self.households.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DisagreementReport:
        """Rebuild a report written by :meth:`to_dict`."""
        return cls(
            specifications=tuple(payload["specifications"]),
            reference=str(payload["reference"]),
            states=tuple(payload["states"]),
            households=dict(payload["households"]),
        )


_SUMMARY_BOUNDS = {
    "discordance": None,  # [0, log2 M], checked with M
    "normalised_discordance": 1.0,
    "mean_pairwise": 1.0,
    "vote_disagreement": 1.0,
}


def validate_disagreement(payload: Mapping[str, Any]) -> list[str]:
    """Every problem with a record's structural-disagreement section."""
    problems: list[str] = []
    expected = {
        "distance",
        "trace_format",
        "reference",
        "states",
        "specifications",
        "households",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        return [f"structural_disagreement must have exactly {sorted(expected)}"]
    if payload["distance"] != DISTANCE:
        problems.append(f"structural_disagreement.distance must be {DISTANCE!r}")
    if payload["trace_format"] != TRACE_FORMAT:
        problems.append(
            f"structural_disagreement.trace_format must be {TRACE_FORMAT!r}"
        )
    specifications = payload["specifications"]
    names = [s.get("name") for s in specifications if isinstance(s, Mapping)]
    if (
        len(names) != len(specifications)
        or len(names) < 2
        or len(set(names)) != len(names)
    ):
        problems.append(
            "structural_disagreement needs two or more distinct named specifications"
        )
    for spec in specifications:
        if not isinstance(spec, Mapping):
            continue
        assumptions = spec.get("assumptions")
        if not isinstance(assumptions, list) or not assumptions:
            problems.append(
                f"structural_disagreement.specifications: {spec.get('name')!r} "
                "lists no assumptions"
            )
            continue
        for assumption in assumptions:
            if not isinstance(assumption, Mapping) or not assumption.get("support"):
                problems.append(
                    f"structural_disagreement.specifications: {spec.get('name')!r} "
                    "has an assumption without published support"
                )
    if payload["reference"] not in names:
        problems.append(
            "structural_disagreement.reference is not one of the specifications"
        )
    states = payload["states"]
    if not isinstance(states, list) or not states or len(set(states)) != len(states):
        problems.append("structural_disagreement.states must be distinct and non-empty")
    models = len(names)
    pairs = {f"{a}|{b}" for a, b in combinations(sorted(n for n in names if n), 2)}
    for home, entry in payload["households"].items():
        where = f"structural_disagreement.households.{home}"
        if not isinstance(entry, Mapping) or set(entry) != {"summary", "trace"}:
            problems.append(f"{where} must have exactly summary and trace")
            continue
        trace = entry["trace"]
        if trace is not None and (
            not isinstance(trace, Mapping)
            or set(trace) != {"file", "sha256"}
            or not isinstance(trace["sha256"], str)
            or len(trace["sha256"]) != 64
        ):
            problems.append(f"{where}.trace must be null or a file with its SHA-256")
        summary = entry["summary"]
        if not isinstance(summary, Mapping) or not summary.get("windows"):
            continue
        for key, bound in _SUMMARY_BOUNDS.items():
            limit = (
                bound
                if bound is not None
                else (math.log2(models) if models > 1 else 0.0)
            )
            for statistic, value in summary[key].items():
                if not (0.0 <= value <= limit + 1e-9):
                    problems.append(
                        f"{where}.summary.{key}.{statistic} is outside [0, {limit:g}]"
                    )
        if set(summary["pairs"]) != pairs:
            problems.append(
                f"{where}.summary.pairs must cover every pair of specifications"
            )
        for name, value in summary["spread"].items():
            if name not in states or not 0.0 <= value <= 1.0 + 1e-9:
                problems.append(
                    f"{where}.summary.spread.{name} is not a state spread in [0, 1]"
                )
    return problems
