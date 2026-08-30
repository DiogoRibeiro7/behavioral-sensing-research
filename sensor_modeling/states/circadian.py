"""Time-of-day transition weighting for behavioural state dynamics.

The v0.2 ontology uses one homogeneous continuous-time generator.  This module
adds a deliberately small extension for v0.3: keep each state's declared exit
rate, but allow the *destination* probabilities of permitted transitions to
change by local-time band.  The dwell-time semantics therefore remain intact;
only where the chain prefers to go next changes with time of day.

The module is parameter-free by default.  A schedule has to be supplied
explicitly, which keeps development-data tuning separate from the mathematical
mechanism and from the untouched external-validation cohort.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .markov import build_generator, transition_matrix
from .ontology import BehaviouralState, StateOntology


@dataclass(frozen=True)
class CircadianBand:
    """Destination-state multipliers active from ``start_hour`` local time.

    Bands wrap over midnight.  ``start_hour`` is an integer in ``[0, 23]``.
    A multiplier of one leaves a destination unchanged, values above one make
    it more likely conditional on leaving the current state, and zero removes
    that destination for the band when alternatives remain.
    """

    start_hour: int
    destination_weights: Mapping[BehaviouralState, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.start_hour, int) or not 0 <= self.start_hour <= 23:
            raise ValueError("start_hour must be an integer in [0, 23]")
        for state, weight in self.destination_weights.items():
            if not isinstance(state, BehaviouralState):
                raise TypeError("circadian destination keys must be BehaviouralState values")
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("circadian destination weights must be finite and non-negative")


@dataclass(frozen=True)
class CircadianSchedule:
    """Piecewise-constant local-time transition preferences."""

    bands: Sequence[CircadianBand]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.bands, key=lambda band: band.start_hour))
        if not ordered:
            raise ValueError("a circadian schedule needs at least one band")
        starts = [band.start_hour for band in ordered]
        if len(starts) != len(set(starts)):
            raise ValueError("circadian band start hours must be unique")
        object.__setattr__(self, "bands", ordered)

    def band_at(self, moment: datetime) -> CircadianBand:
        """Return the band active at the timezone-aware *moment*."""
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("moment must be timezone-aware")
        hour = moment.hour
        candidates = [band for band in self.bands if band.start_hour <= hour]
        return candidates[-1] if candidates else self.bands[-1]

    def generator(self, ontology: StateOntology, moment: datetime) -> np.ndarray:
        """Build the generator active at *moment* while preserving exit rates."""
        band = self.band_at(moment)
        size = ontology.size
        rates = np.array(
            [1.0 / ontology.dwell[state].total_seconds() for state in ontology.states],
            dtype=float,
        )
        jumps = np.zeros((size, size), dtype=float)
        for row, state in enumerate(ontology.states):
            for target in ontology.jumps.get(state, ontology.states):
                if target not in ontology.states or target is state:
                    continue
                weight = float(band.destination_weights.get(target, 1.0))
                jumps[row, ontology.index(target)] = weight
        return build_generator(rates, jumps)

    def transition(
        self, ontology: StateOntology, moment: datetime, elapsed: timedelta
    ) -> np.ndarray:
        """Return a row-stochastic transition matrix within the active band.

        Callers that cross a band boundary should split the interval and
        multiply the resulting matrices in chronological order.  The online
        pipeline naturally advances on short fixed steps, so this primitive is
        sufficient without embedding calendar logic into the CTMC algebra.
        """
        return transition_matrix(
            self.generator(ontology, moment), elapsed.total_seconds()
        )
