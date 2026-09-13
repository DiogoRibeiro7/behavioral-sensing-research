"""Running the standard pipeline against a real recording.

Everything else in this package is scored on a simulator this project wrote.
This module runs the same inference, unchanged, over data the project did not
generate, which is the only way to find out whether the architecture survives
contact with reality.

It deliberately does not tune anything. No emission rate, dwell time or
threshold is refitted to the dataset. A result produced here is therefore a
lower bound on what the approach could do with fitted parameters, and an honest
measure of how far the declared defaults transfer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from statistics import median
from typing import Any

from ..evaluation.metrics import StateMetrics, state_metrics
from ..online import BehaviouralSensingPipeline, PipelineConfig
from ..online.pipeline import PipelineStep, scoring_steps
from ..states.ontology import BehaviouralState
from .casas import CasasRecording, truth_series

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UncertaintyDiagnostics:
    """Uncertainty summaries split by whether a scored estimate was correct."""

    scored: int
    correct: int
    median_confidence_correct: float | None
    median_confidence_incorrect: float | None
    median_margin_correct: float | None
    median_margin_incorrect: float | None
    median_normalised_entropy_correct: float | None
    median_normalised_entropy_incorrect: float | None
    median_evidence_strength_correct: float | None
    median_evidence_strength_incorrect: float | None
    median_information_gain_correct: float | None
    median_information_gain_incorrect: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        """Return a serialisable representation."""
        return {
            "scored": self.scored,
            "correct": self.correct,
            "median_confidence_correct": self.median_confidence_correct,
            "median_confidence_incorrect": self.median_confidence_incorrect,
            "median_margin_correct": self.median_margin_correct,
            "median_margin_incorrect": self.median_margin_incorrect,
            "median_normalised_entropy_correct": self.median_normalised_entropy_correct,
            "median_normalised_entropy_incorrect": self.median_normalised_entropy_incorrect,
            "median_evidence_strength_correct": self.median_evidence_strength_correct,
            "median_evidence_strength_incorrect": self.median_evidence_strength_incorrect,
            "median_information_gain_correct": self.median_information_gain_correct,
            "median_information_gain_incorrect": self.median_information_gain_incorrect,
        }


def _median(values: list[float]) -> float | None:
    """Return the median, or ``None`` when no values are available."""
    return float(median(values)) if values else None


def _evidence_strength(step: PipelineStep) -> float:
    """Mean absolute log-likelihood margin from informative sensors.

    This separates posterior certainty from the amount of interval-level sensor
    evidence that produced it. A highly concentrated posterior with little
    evidence is exactly the pattern expected if the transition prior is carrying
    more confidence than the observations justify.
    """
    margins = [
        abs(contribution.support)
        for contribution in step.state.evidence
        if contribution.informative
    ]
    return float(sum(margins) / len(margins)) if margins else 0.0


def uncertainty_diagnostics(
    truth: list[BehaviouralState | None], steps: list[PipelineStep]
) -> UncertaintyDiagnostics:
    """Summarise uncertainty diagnostics on scored positions."""
    if len(truth) != len(steps):
        raise ValueError("truth and steps must have the same length")

    correct_confidence: list[float] = []
    incorrect_confidence: list[float] = []
    correct_margin: list[float] = []
    incorrect_margin: list[float] = []
    correct_entropy: list[float] = []
    incorrect_entropy: list[float] = []
    correct_evidence: list[float] = []
    incorrect_evidence: list[float] = []
    correct_information_gain: list[float] = []
    incorrect_information_gain: list[float] = []

    scored = 0
    correct = 0
    for label, step in zip(truth, steps, strict=True):
        if label is None:
            continue

        scored += 1
        is_correct = step.state.state == label
        if is_correct:
            correct += 1

        confidence = step.state.confidence
        margin = step.state.margin
        entropy = step.state.normalised_entropy
        evidence = _evidence_strength(step)
        information_gain = step.state.information_gain

        if is_correct:
            correct_confidence.append(confidence)
            correct_margin.append(margin)
            correct_entropy.append(entropy)
            correct_evidence.append(evidence)
            if information_gain is not None:
                correct_information_gain.append(information_gain)
        else:
            incorrect_confidence.append(confidence)
            incorrect_margin.append(margin)
            incorrect_entropy.append(entropy)
            incorrect_evidence.append(evidence)
            if information_gain is not None:
                incorrect_information_gain.append(information_gain)

    return UncertaintyDiagnostics(
        scored=scored,
        correct=correct,
        median_confidence_correct=_median(correct_confidence),
        median_confidence_incorrect=_median(incorrect_confidence),
        median_margin_correct=_median(correct_margin),
        median_margin_incorrect=_median(incorrect_margin),
        median_normalised_entropy_correct=_median(correct_entropy),
        median_normalised_entropy_incorrect=_median(incorrect_entropy),
        median_evidence_strength_correct=_median(correct_evidence),
        median_evidence_strength_incorrect=_median(incorrect_evidence),
        median_information_gain_correct=_median(correct_information_gain),
        median_information_gain_incorrect=_median(incorrect_information_gain),
    )


@dataclass(frozen=True)
class DatasetEvaluation:
    """What the pipeline achieved on a real recording, and on how much of it.

    Attributes
    ----------
    metrics
        State-inference quality over the positions that carried a label.
    uncertainty
        Confidence, posterior-shape and interval-level evidence summaries split
        by correct and incorrect scored estimates.
    steps
        Pipeline steps produced.
    scored
        Positions that had a mapped annotation and were therefore scored.
    labelled_fraction
        Fraction of the recording's span covered by mapped annotation.
    recording
        What the adapter discarded on the way in.
    """

    metrics: StateMetrics
    uncertainty: UncertaintyDiagnostics
    steps: int
    scored: int
    labelled_fraction: float
    recording: dict[str, Any]

    @property
    def scored_fraction(self) -> float:
        """Share of pipeline steps that could be scored at all."""
        return self.scored / self.steps if self.steps else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form, coverage alongside the scores."""
        return {
            "metrics": self.metrics.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "steps": self.steps,
            "scored": self.scored,
            "scored_fraction": self.scored_fraction,
            "labelled_fraction": self.labelled_fraction,
            "recording": self.recording,
        }


def evaluate_recording(
    recording: CasasRecording,
    *,
    step: timedelta = timedelta(minutes=10),
    config: PipelineConfig | None = None,
) -> DatasetEvaluation:
    """Run the standard pipeline over a recording and score it where labelled.

    Parameters
    ----------
    recording
        A parsed recording, from :func:`~sensor_modeling.datasets.read_casas`.
    step
        Inference step. Real recordings are far denser than the simulator's, so
        a short step produces a great many mostly-unlabelled positions.
    config
        Pipeline configuration. The timezone is taken from the observations if
        not supplied, since the recording already fixed it.

    Raises
    ------
    ValueError
        If the recording produced no observations, or if nothing in it carried
        a mapped label. Returning a metric computed over nothing would look
        like a result.
    """
    if not recording.observations:
        raise ValueError("recording contains no usable observations")

    tz = recording.observations[0].timestamp.tzinfo
    settings = config or PipelineConfig(tz=tz, step=step)

    pipeline = BehaviouralSensingPipeline(recording.registry, config=settings)
    steps = pipeline.run(recording.observations)
    steps.extend(pipeline.close(recording.observations[-1].timestamp))
    steps = scoring_steps(steps)
    if not steps:
        raise ValueError("pipeline produced no steps for this recording")

    truth = truth_series(recording.activities, [s.at for s in steps])
    scored = sum(1 for label in truth if label is not None)
    if scored == 0:
        raise ValueError(
            "no pipeline step fell inside a mapped annotation, so nothing can "
            "be scored; check the activity mapping and the step size"
        )

    return DatasetEvaluation(
        metrics=state_metrics(truth, [s.state for s in steps]),
        uncertainty=uncertainty_diagnostics(truth, steps),
        steps=len(steps),
        scored=scored,
        labelled_fraction=recording.labelled_fraction,
        recording=recording.summary(),
    )
