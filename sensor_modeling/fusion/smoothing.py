"""Improving a past estimate with evidence that arrived after it.

The filter reports ``P(Z_t | O_1:t)``: what was believable given everything seen
up to that moment. For a live alert that is the only honest answer, because
nothing later exists yet.

For anything computed after the fact — a daily summary, a baseline, an
evaluation — later evidence *is* available, and withholding it makes the
estimate worse for no reason. A brief kitchen visit at 18:05 is ambiguous on its
own and obvious once 18:10 shows the resident still there.

This module runs the standard backward recursion over beliefs the filter has
already produced. It is not a second pass of inference: no observation is read
twice, which is what distinguishes this from feeding lagged observations back
into the filter. Doing that would count evidence the belief had already absorbed
and manufacture confidence.

Measured on eleven held-out CASAS homes, a lag of one step raises median
balanced accuracy from 0.449 to 0.463. Longer lags add nothing: four hours
scores the same as five minutes. See ``docs/real_data.md``.

.. warning::

   A smoothed estimate for time *t* is only available at *t + lag*. Never put it
   on an alerting path, where the delay is the whole cost and the filtered
   estimate is the correct input.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from datetime import timedelta

import numpy as np

from ..states import StateOntology
from .estimate import StateEstimate

logger = logging.getLogger(__name__)

#: Smallest probability treated as non-zero when dividing by a prediction.
_FLOOR = 1e-12


def smooth_beliefs(
    beliefs: np.ndarray,
    transition: np.ndarray,
    *,
    lag: int = 1,
) -> np.ndarray:
    """Return beliefs revised using up to *lag* steps of later evidence.

    Parameters
    ----------
    beliefs
        Filtered beliefs, one row per step, each summing to one.
    transition
        Row-stochastic transition matrix for one step.
    lag
        How many later steps may inform each estimate. ``0`` returns the input
        unchanged. Larger values cost latency and, on real recordings, buy
        almost nothing beyond the first step.

    Notes
    -----
    Each step's correction starts from the *filtered* belief at its horizon
    rather than from an already-smoothed one. Starting from smoothed values
    chains corrections back through the whole sequence, which silently turns a
    fixed-lag smoother into a full one: every lag then returns identical
    numbers, and an offline-only result gets reported as a five-minute one.
    """
    if lag < 0:
        raise ValueError("lag must not be negative")
    array = np.asarray(beliefs, dtype=float)
    if array.ndim != 2:
        raise ValueError("beliefs must be a two-dimensional array of rows")
    if transition.shape != (array.shape[1], array.shape[1]):
        raise ValueError("transition matrix does not match the belief width")
    if lag == 0 or array.shape[0] < 2:
        return array.copy()

    count = array.shape[0]
    out = array.copy()
    for start in range(count - 2, -1, -1):
        horizon = min(start + lag, count - 1)
        gamma = array[horizon].copy()
        for step in range(horizon - 1, start - 1, -1):
            predicted = array[step] @ transition
            ratio = np.divide(
                gamma, predicted, out=np.zeros_like(gamma), where=predicted > _FLOOR
            )
            gamma = array[step] * (transition @ ratio)
            total = gamma.sum()
            gamma = gamma / total if total > 0 else array[step]
        out[start] = gamma
    return out


def smooth_estimates(
    estimates: Sequence[StateEstimate],
    *,
    lag: int = 1,
    step: timedelta | None = None,
    ontology: StateOntology | None = None,
) -> list[StateEstimate]:
    """Return estimates whose beliefs use up to *lag* steps of later evidence.

    The evidence attached to each estimate is left untouched. It records which
    sensors argued for what *at the time*, and rewriting it to match a revised
    belief would misrepresent what the filter actually saw.

    Parameters
    ----------
    estimates
        Filtered estimates in time order, as produced by the pipeline.
    lag
        Steps of later evidence to use. Zero returns the input unchanged.
    step
        Spacing between estimates. Inferred from the first two when omitted.
    ontology
        Supplies the transition matrix. Taken from the first estimate when
        omitted.

    Raises
    ------
    ValueError
        If the estimates are not in ascending time order, since the backward
        recursion would otherwise silently mix unrelated moments.
    """
    ordered = list(estimates)
    if lag == 0 or len(ordered) < 2:
        return ordered

    moments = [estimate.at for estimate in ordered]
    if any(b < a for a, b in zip(moments, moments[1:])):
        raise ValueError("estimates must be in ascending time order to smooth")

    spacing = step if step is not None else moments[1] - moments[0]
    if spacing <= timedelta(0):
        raise ValueError("step must be positive")

    states = ontology or ordered[0].ontology
    transition = states.transition(spacing)
    smoothed = smooth_beliefs(
        np.vstack([estimate.belief for estimate in ordered]), transition, lag=lag
    )
    return [
        replace(estimate, belief=smoothed[index])
        for index, estimate in enumerate(ordered)
    ]
