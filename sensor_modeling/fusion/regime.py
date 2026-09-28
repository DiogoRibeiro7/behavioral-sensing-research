"""Online filtering and fixed-lag smoothing are different inference regimes.

The filter's estimate for a moment uses only the evidence at or before it. It
is what a live system can report at that moment. A fixed-lag smoother revises
that estimate with up to ``lag`` later windows. It can therefore only be
reported ``lag`` windows later, and it reads evidence a live system does not
yet have. The roadmap's rule (3.5) is that a gain obtained with future
evidence must never be reported as an online gain.

This module makes the regime explicit so the two cannot be blurred by
accident:

- :class:`InferenceRegime` names the mode and, for a smoother, its lag.
  Every experiment record carries one
  (:class:`~sensor_modeling.evaluation.ExperimentRecord`).
- :func:`regime_beliefs` is the one way to turn filtered beliefs into the
  estimates a regime reports. Online, it returns them unchanged. A smoother
  applies :func:`~sensor_modeling.fusion.smooth_beliefs` with the declared
  lag. Neither inference algorithm is changed.
- :func:`require_online` and :func:`require_same_regime` fail loudly when a
  smoothed result is presented as online, or combined with online results
  into one.
- :func:`check_evidence_access` and :func:`assert_respects_horizon` check that
  an estimate reads no evidence beyond its regime's horizon: its own moment
  online, and ``lag`` windows later for a smoother.

A smoother with lag zero reads nothing after its moment and reports exactly the
filtered estimate. It is still labelled a smoother, so a record says what was
run rather than what it happened to equal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from .smoothing import smooth_beliefs


class InferenceMode(str, Enum):
    """How far after a moment its estimate may read evidence."""

    ONLINE_FILTER = "online_filter"
    """Evidence at or before the moment only: what a live system reports."""

    FIXED_LAG_SMOOTHER = "fixed_lag_smoother"
    """Evidence up to a fixed number of windows after the moment."""


class EvidenceLeakageError(ValueError):
    """An estimate reads evidence beyond its regime, or is reported as another regime."""


def _minutes(delta: timedelta) -> str:
    minutes = delta.total_seconds() / 60.0
    return f"{minutes:g} min"


@dataclass(frozen=True)
class InferenceRegime:
    """The inference regime an estimate, a result or a record belongs to.

    Attributes
    ----------
    mode
        Online filter or fixed-lag smoother.
    lag_steps
        Windows after its moment that a smoother's estimate may read. Always
        zero online.
    step
        Width of one window. A smoother needs it, so that its lag, which is
        also its reporting delay, is stated in time.
    """

    mode: InferenceMode
    lag_steps: int = 0
    step: timedelta | None = None

    def __post_init__(self) -> None:
        """Validate the regime."""
        mode = InferenceMode(self.mode)
        object.__setattr__(self, "mode", mode)
        lag = self.lag_steps
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
            raise ValueError("lag_steps must be a non-negative integer")
        if mode is InferenceMode.ONLINE_FILTER:
            if lag != 0:
                raise EvidenceLeakageError(
                    "an online filter reads no evidence after its moment, so its "
                    "lag is zero; a positive lag is a fixed-lag smoother"
                )
            if self.step is not None:
                raise ValueError("an online filter has no lag, so it takes no step")
        elif not isinstance(self.step, timedelta) or self.step <= timedelta(0):
            raise ValueError("a fixed-lag smoother needs a positive window step")

    @classmethod
    def online(cls) -> InferenceRegime:
        """The online filter."""
        return cls(InferenceMode.ONLINE_FILTER)

    @classmethod
    def smoother(cls, lag_steps: int, step: timedelta) -> InferenceRegime:
        """A fixed-lag smoother reading *lag_steps* windows of *step* ahead."""
        return cls(InferenceMode.FIXED_LAG_SMOOTHER, lag_steps, step)

    @property
    def is_online(self) -> bool:
        """Whether estimates read only evidence at or before their moment."""
        return self.mode is InferenceMode.ONLINE_FILTER

    @property
    def lag(self) -> timedelta:
        """How far after its moment an estimate reads, and so its reporting delay."""
        if self.step is None:
            return timedelta(0)
        return self.step * self.lag_steps

    @property
    def is_causal(self) -> bool:
        """Whether estimates read no evidence after their moment.

        The online filter is causal. A smoother is causal only with lag zero,
        where it reports exactly the filtered estimate. Any positive lag uses
        future information.
        """
        return self.lag_steps == 0

    @property
    def delay(self) -> timedelta:
        """The effective delay: how long after its moment an estimate can be reported."""
        return self.lag

    @property
    def label(self) -> str:
        """A human-readable name that no report may leave out."""
        if self.is_online:
            return "online filter"
        windows = "window" if self.lag_steps == 1 else "windows"
        return (
            f"fixed-lag smoother, lag {self.lag_steps} {windows} ({_minutes(self.lag)})"
        )

    def horizon(self, moment: datetime) -> datetime:
        """The latest moment whose evidence an estimate for *moment* may read."""
        return moment + self.lag

    def horizon_row(self, row: int, rows: int) -> int:
        """The last window an estimate for window *row* may read, of *rows*."""
        return min(row + self.lag_steps, rows - 1)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form, with its label, causality and delay."""
        return {
            "mode": self.mode.value,
            "lag_steps": self.lag_steps,
            "step_seconds": (
                self.step.total_seconds() if self.step is not None else None
            ),
            "label": self.label,
            "causal": self.is_causal,
            "delay_seconds": self.delay.total_seconds(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InferenceRegime:
        """Rebuild a regime written by :meth:`to_dict`.

        A label, causality or delay that does not describe the regime is
        refused. Causality and delay are derived, and may be absent.
        """
        seconds = payload["step_seconds"]
        regime = cls(
            InferenceMode(payload["mode"]),
            payload["lag_steps"],
            timedelta(seconds=seconds) if seconds is not None else None,
        )
        if payload["label"] != regime.label:
            raise EvidenceLeakageError(
                f"the label {payload['label']!r} does not describe the regime "
                f"{regime.label!r}"
            )
        if "causal" in payload and payload["causal"] is not regime.is_causal:
            raise EvidenceLeakageError(
                f"the {regime.label} is {'' if regime.is_causal else 'not '}causal, "
                f"but the payload says causal={payload['causal']!r}"
            )
        if (
            "delay_seconds" in payload
            and payload["delay_seconds"] != regime.delay.total_seconds()
        ):
            raise EvidenceLeakageError(
                f"the {regime.label} has a delay of {regime.delay.total_seconds():g} "
                f"s, not {payload['delay_seconds']!r}"
            )
        return regime


#: The online filter, the regime of every estimate a live system can report.
ONLINE = InferenceRegime.online()


def require_online(regime: InferenceRegime, what: str) -> None:
    """Refuse *what* unless it is online.

    Raises
    ------
    EvidenceLeakageError
        If *regime* reads evidence after its moment: a smoothed result must be
        reported as smoothed, never as an online one.
    """
    if not regime.is_online:
        raise EvidenceLeakageError(
            f"{what} must be online, but it is a {regime.label}, which reads "
            f"evidence up to {_minutes(regime.lag)} after each moment; report it "
            "as a smoothed result, separately from online ones"
        )


def require_same_regime(
    regimes: Iterable[InferenceRegime], what: str
) -> InferenceRegime:
    """The one regime shared by everything *what* combines.

    Raises
    ------
    EvidenceLeakageError
        If the regimes differ: online and smoothed results, or smoothers with
        different lags, may not be combined into one result.
    """
    distinct = list(dict.fromkeys(regimes))
    if not distinct:
        raise ValueError(f"{what} combines nothing")
    if len(distinct) > 1:
        raise EvidenceLeakageError(
            f"{what} would combine results from different inference regimes: "
            f"{', '.join(r.label for r in distinct)}; report each separately"
        )
    return distinct[0]


def check_evidence_access(
    regime: InferenceRegime, moment: datetime, evidence: datetime
) -> None:
    """Refuse evidence observed at *evidence* for an estimate of *moment*.

    Raises
    ------
    EvidenceLeakageError
        If *evidence* lies after the regime's horizon for *moment*.
    """
    if evidence > regime.horizon(moment):
        raise EvidenceLeakageError(
            f"an estimate for {moment.isoformat()} under the {regime.label} may "
            f"read evidence up to {regime.horizon(moment).isoformat()}, not "
            f"{evidence.isoformat()}"
        )


def regime_beliefs(
    filtered: np.ndarray, transition: np.ndarray, regime: InferenceRegime
) -> np.ndarray:
    """The estimates *regime* reports, from the online filter's beliefs.

    Parameters
    ----------
    filtered
        ``(windows, states)`` beliefs of an online filter over consecutive
        windows: row ``t`` reads evidence up to window ``t`` only.
    transition
        The filter's one-window transition matrix.
    regime
        Online returns *filtered* unchanged. A fixed-lag smoother revises row
        ``t`` with rows up to ``t + lag``, and nothing later.
    """
    array = np.asarray(filtered, dtype=float)
    if regime.is_online:
        online: np.ndarray = array.copy()
        return online
    return smooth_beliefs(array, transition, lag=regime.lag_steps)


@dataclass(frozen=True)
class ReportedEstimate:
    """One estimate as a regime reports it: for when, on what, and when it exists.

    Attributes
    ----------
    prediction_at
        The moment the estimate is about.
    evidence_until
        The latest evidence it read. Online, the moment itself. A smoother reads
        up to its lag after the moment, or to the end of the recording.
    available_at
        When it can first be reported. Online, the moment itself. A smoother
        must wait for its last evidence, so this is ``evidence_until``.
    belief
        The posterior, read-only: a reported estimate is never revised.
    regime
        The regime that produced it.
    """

    prediction_at: datetime
    evidence_until: datetime
    available_at: datetime
    belief: np.ndarray
    regime: InferenceRegime

    def __post_init__(self) -> None:
        """Refuse an estimate that reads beyond its regime, and freeze its belief."""
        if self.evidence_until < self.prediction_at:
            raise ValueError("an estimate reads at least the evidence of its moment")
        check_evidence_access(self.regime, self.prediction_at, self.evidence_until)
        expected = self.prediction_at if self.regime.is_online else self.evidence_until
        if self.available_at != expected:
            raise EvidenceLeakageError(
                f"an estimate of the {self.regime.label} is available at "
                f"{expected.isoformat()}, not {self.available_at.isoformat()}"
            )
        belief = np.array(self.belief, dtype=float)
        belief.setflags(write=False)
        object.__setattr__(self, "belief", belief)

    @property
    def lead(self) -> timedelta:
        """How far past its moment the estimate read: the future information used."""
        return self.evidence_until - self.prediction_at

    @property
    def uses_future_evidence(self) -> bool:
        """Whether it read any evidence after its moment."""
        return self.lead > timedelta(0)


def regime_estimates(
    moments: Sequence[datetime],
    filtered: np.ndarray,
    transition: np.ndarray,
    regime: InferenceRegime,
) -> tuple[ReportedEstimate, ...]:
    """The estimates *regime* reports, each with its evidence and availability.

    Parameters
    ----------
    moments
        The windows' moments, strictly increasing. A smoother's windows must be
        spaced by its step, so its lag in windows is its lag in time.
    filtered, transition
        As for :func:`regime_beliefs`.
    regime
        The regime to report under.
    """
    stamps = list(moments)
    beliefs = regime_beliefs(filtered, transition, regime)
    if len(stamps) != beliefs.shape[0]:
        raise ValueError("one moment is needed per window")
    if any(later <= earlier for earlier, later in zip(stamps, stamps[1:])):
        raise ValueError("moments must be strictly increasing")
    if not regime.is_online and any(
        later - earlier != regime.step for earlier, later in zip(stamps, stamps[1:])
    ):
        raise ValueError("a smoother's windows must be spaced by its step")
    count = len(stamps)
    reported = []
    for row, moment in enumerate(stamps):
        until = stamps[regime.horizon_row(row, count)]
        reported.append(
            ReportedEstimate(
                prediction_at=moment,
                evidence_until=until,
                available_at=moment if regime.is_online else until,
                belief=beliefs[row],
                regime=regime,
            )
        )
    return tuple(reported)


@dataclass(frozen=True)
class EvidenceSummary:
    """The prediction and evidence timestamps behind a set of reported estimates.

    Attributes
    ----------
    predictions
        How many estimates.
    first_prediction, last_prediction
        The earliest and latest moments they are about.
    latest_evidence
        The latest evidence any of them read.
    max_lead
        The largest lead of an estimate's evidence over its moment: the most
        future information any of them used.
    """

    predictions: int
    first_prediction: datetime
    last_prediction: datetime
    latest_evidence: datetime
    max_lead: timedelta

    def __post_init__(self) -> None:
        """Check the summary is internally consistent."""
        if self.predictions < 1:
            raise ValueError("an evidence summary needs at least one prediction")
        if not self.first_prediction <= self.last_prediction <= self.latest_evidence:
            raise ValueError(
                "first prediction, last prediction and latest evidence are out of order"
            )
        if self.max_lead < timedelta(0):
            raise ValueError("a lead cannot be negative")
        if self.latest_evidence - self.last_prediction > self.max_lead:
            raise ValueError("the latest evidence lies beyond the largest lead")

    @classmethod
    def of(cls, pairs: Iterable[tuple[datetime, datetime]]) -> EvidenceSummary:
        """Summarise ``(prediction_at, evidence_until)`` pairs."""
        items = list(pairs)
        if not items:
            raise ValueError("an evidence summary needs at least one prediction")
        if any(until < at for at, until in items):
            raise ValueError("an estimate reads at least the evidence of its moment")
        return cls(
            predictions=len(items),
            first_prediction=min(at for at, _ in items),
            last_prediction=max(at for at, _ in items),
            latest_evidence=max(until for _, until in items),
            max_lead=max(until - at for at, until in items),
        )

    @classmethod
    def online(cls, moments: Iterable[datetime]) -> EvidenceSummary:
        """Online estimates: each reads evidence up to its own moment."""
        return cls.of((moment, moment) for moment in moments)

    @classmethod
    def of_estimates(cls, estimates: Iterable[ReportedEstimate]) -> EvidenceSummary:
        """Summarise reported estimates."""
        return cls.of((e.prediction_at, e.evidence_until) for e in estimates)

    @classmethod
    def combine(cls, summaries: Iterable[EvidenceSummary]) -> EvidenceSummary:
        """One summary for several sets of estimates."""
        parts = list(summaries)
        if not parts:
            raise ValueError("nothing to combine")
        return cls(
            predictions=sum(p.predictions for p in parts),
            first_prediction=min(p.first_prediction for p in parts),
            last_prediction=max(p.last_prediction for p in parts),
            latest_evidence=max(p.latest_evidence for p in parts),
            max_lead=max(p.max_lead for p in parts),
        )

    def check(self, regime: InferenceRegime) -> None:
        """Refuse a summary whose estimates read beyond *regime*'s horizon."""
        if self.max_lead > regime.delay:
            raise EvidenceLeakageError(
                f"estimates read up to {self.max_lead.total_seconds():g} s past their "
                f"moments, but the {regime.label} may read "
                f"{regime.delay.total_seconds():g} s"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form."""
        return {
            "enumerated": True,
            "predictions": self.predictions,
            "first_prediction": self.first_prediction.isoformat(),
            "last_prediction": self.last_prediction.isoformat(),
            "latest_evidence": self.latest_evidence.isoformat(),
            "max_lead_seconds": self.max_lead.total_seconds(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvidenceSummary:
        """Rebuild a summary written by :meth:`to_dict`."""
        return cls(
            predictions=int(payload["predictions"]),
            first_prediction=datetime.fromisoformat(payload["first_prediction"]),
            last_prediction=datetime.fromisoformat(payload["last_prediction"]),
            latest_evidence=datetime.fromisoformat(payload["latest_evidence"]),
            max_lead=timedelta(seconds=float(payload["max_lead_seconds"])),
        )


@dataclass(frozen=True)
class NotEnumerated:
    """Why a result's predictions and evidence timestamps were not listed.

    Only a causal result may say this: a smoother must always record the
    future information it used.
    """

    reason: str

    def __post_init__(self) -> None:
        """Require a reason."""
        if not str(self.reason).strip():
            raise ValueError("say why the predictions were not enumerated")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form."""
        return {"enumerated": False, "reason": self.reason}


def evidence_from_dict(payload: Mapping[str, Any]) -> EvidenceSummary | NotEnumerated:
    """Rebuild an evidence record written by either ``to_dict``."""
    if payload.get("enumerated") is True:
        return EvidenceSummary.from_dict(payload)
    if payload.get("enumerated") is False and set(payload) == {"enumerated", "reason"}:
        return NotEnumerated(str(payload["reason"]))
    raise ValueError("evidence must be an enumerated summary or a reason")


def check_evidence(
    evidence: EvidenceSummary | NotEnumerated, regime: InferenceRegime
) -> None:
    """Refuse evidence a regime could not have produced, or left unlisted.

    Raises
    ------
    EvidenceLeakageError
        If the estimates read beyond the regime's horizon, or if a non-causal
        result does not list the future information it used.
    """
    if isinstance(evidence, NotEnumerated):
        if not regime.is_causal:
            raise EvidenceLeakageError(
                f"a {regime.label} uses future information, so its predictions and "
                "evidence timestamps must be recorded"
            )
        return
    evidence.check(regime)


def assert_respects_horizon(
    estimate: Callable[[np.ndarray], np.ndarray],
    evidence: np.ndarray,
    regime: InferenceRegime,
    *,
    rows: Sequence[int] | None = None,
    perturb: Callable[[np.ndarray], np.ndarray] | None = None,
    atol: float = 1e-12,
) -> None:
    """Check that each estimate reads no evidence beyond its regime's horizon.

    For each row ``t`` checked, every window after the horizon is replaced by
    ``perturb`` of itself, and the estimate for ``t`` must not change.

    Parameters
    ----------
    estimate
        Maps ``(windows, ...)`` evidence to ``(windows, states)`` estimates.
    evidence
        The evidence, one row per window.
    regime
        The regime *estimate* claims.
    rows
        Rows to check. Defaults to every row that has a later window.
    perturb
        How later evidence is changed. Defaults to reversing its order and
        adding one, which changes any evidence that is not constant.

    Raises
    ------
    EvidenceLeakageError
        If an estimate changes when only evidence beyond its horizon changes.
    """
    values = np.asarray(evidence)
    count = values.shape[0]
    change = perturb or (lambda later: later[::-1] + 1)
    baseline = np.asarray(estimate(values))
    checked = rows if rows is not None else range(count - 1)
    for row in checked:
        last = regime.horizon_row(row, count)
        if last >= count - 1:
            continue
        altered = values.copy()
        altered[last + 1 :] = change(values[last + 1 :])
        revised = np.asarray(estimate(altered))
        if not np.allclose(revised[row], baseline[row], rtol=0.0, atol=atol):
            raise EvidenceLeakageError(
                f"the estimate for window {row} changed when only windows after "
                f"{last} changed, which the {regime.label} may not read"
            )
