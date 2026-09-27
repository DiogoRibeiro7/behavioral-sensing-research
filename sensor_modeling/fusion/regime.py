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
        """Return a serialisable form, with its label."""
        return {
            "mode": self.mode.value,
            "lag_steps": self.lag_steps,
            "step_seconds": (
                self.step.total_seconds() if self.step is not None else None
            ),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InferenceRegime:
        """Rebuild a regime written by :meth:`to_dict`, refusing a wrong label."""
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
