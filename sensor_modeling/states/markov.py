"""Continuous-time Markov chain mechanics shared by the latent-state models.

Both the behavioural state ontology and the occupancy context model are
continuous-time chains over a small discrete state set. The maths is the same
in each case -- build a generator from dwell times and permitted jumps,
exponentiate it over an arbitrary interval, solve for the stationary
distribution -- so it lives here once rather than being written twice.

Working in continuous time is what lets both models accept observations that
arrive whenever they happen to arrive, without resampling anything onto a
common grid.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import expm


def build_generator(rates: np.ndarray, jumps: np.ndarray) -> np.ndarray:
    """Build a transition rate matrix from exit rates and permitted jumps.

    Parameters
    ----------
    rates
        Total exit rate of each state, in transitions per second. The inverse
        of the state's mean dwell time.
    jumps
        Non-negative ``(n, n)`` weights describing where each state may go.
        Rows are normalised to distribute that state's exit rate; the
        diagonal is ignored. A row of zeros falls back to a uniform jump to
        every other state, so a state can never become a trap by accident.

    Returns
    -------
    numpy.ndarray
        The generator ``Q``, whose rows sum to zero.
    """
    exit_rates = np.asarray(rates, dtype=float)
    weights = np.array(jumps, dtype=float, copy=True)
    size = exit_rates.size
    if weights.shape != (size, size):
        raise ValueError("jumps must be a square matrix matching the number of states")
    if exit_rates.min() <= 0.0 or not np.all(np.isfinite(exit_rates)):
        raise ValueError("exit rates must be positive and finite")
    if weights.min() < 0.0 or not np.all(np.isfinite(weights)):
        raise ValueError("jump weights must be non-negative and finite")

    np.fill_diagonal(weights, 0.0)
    row_totals = weights.sum(axis=1)
    uniform = (np.ones((size, size)) - np.eye(size)) / max(size - 1, 1)
    weights = np.where(row_totals[:, None] > 0.0, weights, uniform)
    row_totals = weights.sum(axis=1)

    generator: np.ndarray = weights * (exit_rates / row_totals)[:, None]
    np.fill_diagonal(generator, -exit_rates)
    return generator


def transition_matrix(generator: np.ndarray, seconds: float) -> np.ndarray:
    """Return ``expm(Q * seconds)`` as a row-stochastic matrix.

    A non-positive interval yields the identity, so repeated updates at the
    same instant leave a belief untouched.
    """
    matrix = np.asarray(generator, dtype=float)
    if seconds <= 0.0:
        return np.eye(matrix.shape[0])
    return _cached_transition(matrix.tobytes(), matrix.shape[0], round(seconds, 3))


def stationary_distribution(generator: np.ndarray) -> np.ndarray:
    """Return the long-run distribution implied by *generator*.

    This is the most defensible prior for a filter that has seen no evidence
    yet: it is what the declared dynamics say about the system on average.
    """
    matrix = np.asarray(generator, dtype=float)
    size = matrix.shape[0]
    system = np.vstack([matrix.T, np.ones(size)])
    target = np.zeros(size + 1)
    target[-1] = 1.0
    solution, *_ = np.linalg.lstsq(system, target, rcond=None)
    distribution = np.clip(solution, 0.0, None)
    total = distribution.sum()
    if total <= 0.0:
        return np.full(size, 1.0 / size)
    normalised: np.ndarray = distribution / total
    return normalised


@lru_cache(maxsize=512)
def _cached_transition(generator_bytes: bytes, size: int, seconds: float) -> np.ndarray:
    """Exponentiate a generator, caching on the rounded interval.

    Ambient streams produce the same handful of intervals over and over, so
    caching keeps the matrix exponential off the hot path of online updates.
    """
    generator = np.frombuffer(generator_bytes, dtype=float).reshape(size, size)
    matrix = np.clip(np.asarray(expm(generator * seconds), dtype=float), 0.0, None)
    row_sums = matrix.sum(axis=1, keepdims=True)
    stochastic: np.ndarray = matrix / np.where(row_sums > 0.0, row_sums, 1.0)
    return stochastic
