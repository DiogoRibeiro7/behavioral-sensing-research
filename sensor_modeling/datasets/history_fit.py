"""Fit the history state's coefficients from labelled training households.

For each group ``(s, r)`` of :class:`~sensor_modeling.fusion.history.HistoryModel`,
the rows are the labelled windows in state ``s`` and the instrumented channels
whose room relates to ``s`` by ``r``, with the current window and every one of
its ``k`` history windows observed. Each row gives:

- ``y``: the channel's activations in the current window;
- ``μ``: its memoryless expected activations in state ``s``;
- ``z = log(1 + h) − E_s[log(1 + H)]``: its history feature.

The coefficient maximises the penalised Poisson log-likelihood

.. math::

    \\sum_{\\text{rows}} \\big(y\\, \\beta z - \\mu\\, e^{\\beta z}\\big)
    - \\tfrac{\\tau}{2} \\beta^2,

which is strictly concave, so its maximum is unique. Newton's method from
``β = 0`` finds it deterministically. The declared rates are not refitted, so
``β`` measures only what the history adds to them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from ..fusion.history import (
    HistoryConfig,
    HistoryModel,
    expected_log1p_poisson,
    relation,
)
from ..states.ontology import StateOntology
from .casas import CasasRecording, truth_series
from .information_sets import (
    EvidenceResolution,
    InformationComponent,
    InformationSet,
    build_feature_table,
    evidence_column,
)
from .matched_evaluation import _regular_moments
from .restricted_filter import channel_likelihoods

_Rows = dict[str, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]]


def history_rows(
    recording: CasasRecording,
    *,
    config: HistoryConfig,
    ontology: StateOntology,
    resolution: EvidenceResolution,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One household's fitting rows, ``(z, y, μ)`` by group key."""
    window = config.window_steps
    declared = replace(resolution, history_steps=window)
    information_set = InformationSet(
        "history-fit",
        frozenset(
            {InformationComponent.CURRENT_EVIDENCE, InformationComponent.RECENT_HISTORY}
        ),
        declared,
    )
    table = build_feature_table(
        recording,
        information_set,
        _regular_moments(recording, declared.step),
        household="fit",
    )
    truth = truth_series(recording.activities, list(table.moments))
    index = {state: position for position, state in enumerate(ontology.states)}
    codes = np.array([index.get(label, -1) if label else -1 for label in truth])
    terms, _ = channel_likelihoods(recording.registry, declared, ontology)

    rows: _Rows = {}
    for channel, term in terms.items():
        now = table.column(evidence_column(channel.name, 0))
        past = np.column_stack(
            [
                table.column(evidence_column(channel.name, lag))
                for lag in range(1, window + 1)
            ]
        )
        usable = (codes >= 0) & ~np.isnan(now) & ~np.isnan(past).any(axis=1)
        reference = expected_log1p_poisson(window * term.expected)
        feature = np.log1p(np.where(usable, past.sum(axis=1), 0.0))
        for position, state in enumerate(ontology.states):
            mask = usable & (codes == position)
            if not mask.any():
                continue
            key = f"{state.value}|{relation(ontology.room_of(state), channel.room)}"
            z, y, mu = rows.setdefault(key, ([], [], []))
            z.append(feature[mask] - reference[position])
            y.append(now[mask])
            mu.append(np.full(int(mask.sum()), term.expected[position]))
    return {
        key: (np.concatenate(z), np.concatenate(y), np.concatenate(mu))
        for key, (z, y, mu) in rows.items()
    }


def _objective(
    beta: float, z: np.ndarray, y: np.ndarray, mu: np.ndarray, tau: float
) -> float:
    return float(np.sum(y * beta * z - mu * np.exp(beta * z)) - 0.5 * tau * beta**2)


def fit_coefficient(
    z: np.ndarray, y: np.ndarray, mu: np.ndarray, precision: float
) -> float:
    """The penalised Poisson maximum for one group; zero with no rows."""
    if z.size == 0:
        return 0.0
    beta = 0.0
    for _ in range(200):
        rate = mu * np.exp(beta * z)
        gradient = float(np.sum(y * z - rate * z)) - precision * beta
        curvature = -float(np.sum(rate * z * z)) - precision
        step = gradient / curvature
        current = _objective(beta, z, y, mu, precision)
        length = 1.0
        while (
            _objective(beta - length * step, z, y, mu, precision) < current
            and length > 1e-12
        ):
            length *= 0.5
        beta -= length * step
        if abs(length * step) <= 1e-12 * (1.0 + abs(beta)):
            return beta
    raise RuntimeError("the history coefficient fit did not converge")


def fit_history_model(
    recordings: Mapping[str, CasasRecording],
    *,
    config: HistoryConfig | None = None,
    ontology: StateOntology | None = None,
    resolution: EvidenceResolution | None = None,
) -> HistoryModel:
    """Fit every group's coefficient from the given households' labels.

    Parameters
    ----------
    recordings
        Only households whose labels the evaluation permits for fitting, such
        as a fold's training households. No other household is read.
    config, ontology, resolution
        Declared settings, the ontology and the evidence resolution; the
        defaults are the filter's and the information sets'.
    """
    settings = config or HistoryConfig()
    ontology = ontology or StateOntology()
    resolution = resolution or EvidenceResolution()
    if not recordings:
        raise ValueError("at least one household is required")
    pooled: _Rows = {}
    for household in sorted(recordings):
        for key, (z, y, mu) in history_rows(
            recordings[household],
            config=settings,
            ontology=ontology,
            resolution=resolution,
        ).items():
            parts = pooled.setdefault(key, ([], [], []))
            parts[0].append(z)
            parts[1].append(y)
            parts[2].append(mu)
    null = HistoryModel.null(ontology, settings)
    coefficients: dict[str, float] = {}
    support: dict[str, int] = {}
    for key in null.coefficients:
        z_parts, y_parts, mu_parts = pooled.get(key, ([], [], []))
        z = np.concatenate(z_parts) if z_parts else np.empty(0)
        y = np.concatenate(y_parts) if y_parts else np.empty(0)
        mu = np.concatenate(mu_parts) if mu_parts else np.empty(0)
        coefficients[key] = fit_coefficient(z, y, mu, settings.precision)
        support[key] = int(z.size)
    return HistoryModel(
        states=null.states,
        rooms=null.rooms,
        config=settings,
        coefficients=coefficients,
        fitted_on=tuple(recordings),
        support=support,
    )
