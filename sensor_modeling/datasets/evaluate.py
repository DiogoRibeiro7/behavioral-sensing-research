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
from typing import Any

from ..evaluation.metrics import StateMetrics, state_metrics
from ..online import BehaviouralSensingPipeline, PipelineConfig
from .casas import CasasRecording, truth_series

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetEvaluation:
    """What the pipeline achieved on a real recording, and on how much of it.

    Attributes
    ----------
    metrics
        State-inference quality over the positions that carried a label.
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
    steps: int
    scored: int
    labelled_fraction: float
    recording: dict[str, Any]

    @property
    def scored_fraction(self) -> float:
        """Share of pipeline steps that could be scored at all."""
        return self.scored / self.steps if self.steps else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form, coverage alongside the scores.

        Coverage travels with the metrics deliberately. A balanced accuracy
        computed over a tenth of a recording is a different claim from one
        computed over most of it, and separating the two invites the reader to
        forget the difference.
        """
        return {
            "metrics": self.metrics.to_dict(),
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
        steps=len(steps),
        scored=scored,
        labelled_fraction=recording.labelled_fraction,
        recording=recording.summary(),
    )
