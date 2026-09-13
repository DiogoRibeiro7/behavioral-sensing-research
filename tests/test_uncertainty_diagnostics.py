"""Tests for real-data uncertainty diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from sensor_modeling.datasets.evaluate import (
    UncertaintyDiagnostics,
    uncertainty_diagnostics,
    uncertainty_panel_summary,
)
from sensor_modeling.fusion.estimate import EvidenceContribution, StateEstimate
from sensor_modeling.observations.types import Modality
from sensor_modeling.online.pipeline import PipelineStep
from sensor_modeling.states import BehaviouralState, StateOntology

S = BehaviouralState
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _step(
    belief: list[float],
    *,
    support: float,
    state: BehaviouralState,
    information_gain: float | None = None,
) -> PipelineStep:
    """Build the minimal pipeline-step shape needed by the diagnostic helper."""
    ontology = StateOntology()
    vector = np.zeros(ontology.size, dtype=float)
    vector[ontology.states.index(state)] = belief[0]
    remainder = (1.0 - belief[0]) / (ontology.size - 1)
    vector[vector == 0.0] = remainder

    estimate = StateEstimate(
        at=T0,
        ontology=ontology,
        belief=vector,
        evidence=(
            EvidenceContribution(
                sensor_id="motion",
                modality=Modality.MOTION,
                support=support,
                reliability=1.0,
                attribution=1.0,
                observations=1,
            ),
        ),
        completeness=1.0,
        min_confidence=0.35,
        min_completeness=0.25,
        information_gain=information_gain,
    )
    return cast(PipelineStep, SimpleNamespace(state=estimate))


def _diagnostics(
    *,
    confidence_auc: float | None,
    margin_auc: float | None,
    entropy_auc: float | None,
    evidence_auc: float | None,
    information_gain_auc: float | None,
) -> UncertaintyDiagnostics:
    """Build a household diagnostic with only panel-level fields varying."""
    return UncertaintyDiagnostics(
        scored=100,
        correct=50,
        median_confidence_correct=None,
        median_confidence_incorrect=None,
        median_margin_correct=None,
        median_margin_incorrect=None,
        median_normalised_entropy_correct=None,
        median_normalised_entropy_incorrect=None,
        median_evidence_strength_correct=None,
        median_evidence_strength_incorrect=None,
        median_information_gain_correct=None,
        median_information_gain_incorrect=None,
        confidence_correctness_auc=confidence_auc,
        margin_correctness_auc=margin_auc,
        normalised_entropy_correctness_auc=entropy_auc,
        evidence_strength_correctness_auc=evidence_auc,
        information_gain_correctness_auc=information_gain_auc,
    )


def test_uncertainty_diagnostics_split_correct_from_incorrect() -> None:
    """Diagnostics should preserve the contrast the abstention study needs."""
    steps = [
        _step(
            [0.80],
            support=2.0,
            state=S.HOME_ACTIVE,
            information_gain=0.12,
        ),
        _step(
            [0.90],
            support=0.2,
            state=S.KITCHEN_ACTIVITY,
            information_gain=0.03,
        ),
    ]
    truth = [S.HOME_ACTIVE, S.BATHROOM_ACTIVITY]

    result = uncertainty_diagnostics(truth, steps)

    assert result.scored == 2
    assert result.correct == 1
    assert result.median_confidence_correct == pytest.approx(0.80)
    assert result.median_confidence_incorrect == pytest.approx(0.90)
    assert result.median_evidence_strength_correct == pytest.approx(2.0)
    assert result.median_evidence_strength_incorrect == pytest.approx(0.2)
    assert result.median_information_gain_correct == pytest.approx(0.12)
    assert result.median_information_gain_incorrect == pytest.approx(0.03)
    assert result.confidence_correctness_auc == pytest.approx(0.0)
    assert result.margin_correctness_auc == pytest.approx(0.0)
    assert result.normalised_entropy_correctness_auc == pytest.approx(0.0)
    assert result.evidence_strength_correctness_auc == pytest.approx(1.0)
    assert result.information_gain_correctness_auc == pytest.approx(1.0)


def test_uncertainty_diagnostics_ignore_unlabelled_steps() -> None:
    """Unlabelled positions must not enter correct/incorrect summaries."""
    steps = [
        _step(
            [0.75],
            support=1.0,
            state=S.HOME_ACTIVE,
            information_gain=0.07,
        ),
        _step(
            [0.99],
            support=9.0,
            state=S.KITCHEN_ACTIVITY,
            information_gain=0.90,
        ),
    ]

    result = uncertainty_diagnostics([S.HOME_ACTIVE, None], steps)

    assert result.scored == 1
    assert result.correct == 1
    assert result.median_confidence_correct == pytest.approx(0.75)
    assert result.median_information_gain_correct == pytest.approx(0.07)
    assert result.median_confidence_incorrect is None
    assert result.median_evidence_strength_incorrect is None
    assert result.median_information_gain_incorrect is None
    assert result.confidence_correctness_auc is None
    assert result.information_gain_correctness_auc is None


def test_uncertainty_diagnostics_preserve_missing_information_gain() -> None:
    """Missing diagnostics must not be reinterpreted as zero information gain."""
    steps = [
        _step([0.80], support=2.0, state=S.HOME_ACTIVE, information_gain=None),
        _step(
            [0.90],
            support=0.2,
            state=S.KITCHEN_ACTIVITY,
            information_gain=0.0,
        ),
    ]
    truth = [S.HOME_ACTIVE, S.BATHROOM_ACTIVITY]

    result = uncertainty_diagnostics(truth, steps)

    assert result.median_information_gain_correct is None
    assert result.median_information_gain_incorrect == pytest.approx(0.0)
    assert result.information_gain_correctness_auc is None


def test_uncertainty_diagnostics_auc_gives_half_credit_for_ties() -> None:
    """Tied diagnostic values should contribute half a favourable comparison."""
    steps = [
        _step(
            [0.80],
            support=1.0,
            state=S.HOME_ACTIVE,
            information_gain=0.2,
        ),
        _step(
            [0.80],
            support=1.0,
            state=S.KITCHEN_ACTIVITY,
            information_gain=0.2,
        ),
    ]
    truth = [S.HOME_ACTIVE, S.BATHROOM_ACTIVITY]

    result = uncertainty_diagnostics(truth, steps)

    assert result.confidence_correctness_auc == pytest.approx(0.5)
    assert result.margin_correctness_auc == pytest.approx(0.5)
    assert result.normalised_entropy_correctness_auc == pytest.approx(0.5)
    assert result.evidence_strength_correctness_auc == pytest.approx(0.5)
    assert result.information_gain_correctness_auc == pytest.approx(0.5)


def test_uncertainty_panel_summary_weights_homes_equally_and_reports_coverage() -> None:
    """Panel summaries use one value per home and keep missing coverage visible."""
    homes = [
        _diagnostics(
            confidence_auc=0.40,
            margin_auc=0.45,
            entropy_auc=0.55,
            evidence_auc=0.60,
            information_gain_auc=0.70,
        ),
        _diagnostics(
            confidence_auc=0.60,
            margin_auc=0.65,
            entropy_auc=0.75,
            evidence_auc=0.80,
            information_gain_auc=None,
        ),
        _diagnostics(
            confidence_auc=0.90,
            margin_auc=0.85,
            entropy_auc=0.95,
            evidence_auc=1.00,
            information_gain_auc=0.50,
        ),
    ]

    result = uncertainty_panel_summary(homes)

    assert result["homes"] == 3
    assert result["confidence_homes"] == 3
    assert result["median_confidence_correctness_auc"] == pytest.approx(0.60)
    assert result["median_margin_correctness_auc"] == pytest.approx(0.65)
    assert result["median_normalised_entropy_correctness_auc"] == pytest.approx(0.75)
    assert result["median_evidence_strength_correctness_auc"] == pytest.approx(0.80)
    assert result["information_gain_homes"] == 2
    assert result["median_information_gain_correctness_auc"] == pytest.approx(0.60)
