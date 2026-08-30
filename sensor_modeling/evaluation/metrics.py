"""Evaluation metrics appropriate to each inference problem.

Accuracy alone is close to useless here. The states are heavily imbalanced --
a resident is asleep or quietly at home for most of the day -- so a model that
predicts ``home_inactive`` forever scores well and knows nothing. And because
every stage of this platform reports a probability, an evaluation that only
looks at the argmax throws away exactly the part that matters: whether a
confidence of 0.9 means anything.

So the metrics here cover four separate questions:

.. code-block:: text

    state inference     balanced accuracy, macro F1, log loss, Brier,
                        calibration error, transition timing
    attribution         precision, recall, F1, calibration
    change detection    detection delay, false positives per person-day
    comparison          paired differences with bootstrap intervals and
                        effect sizes, not p-values alone

The abstention convention is stated once and applied consistently: a reported
``UNKNOWN`` is never correct, because ``UNKNOWN`` is never a true label. It
therefore costs recall. Selective accuracy and abstention rate are reported
alongside so that a model which declines usefully can be distinguished from
one that is simply wrong.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from ..fusion.estimate import StateEstimate
from ..states.ontology import BehaviouralState, StateOntology

EPSILON = 1e-12


@dataclass(frozen=True)
class StateMetrics:
    """Quality of a run of behavioural state estimates.

    Attributes
    ----------
    n
        Number of scored estimates.
    accuracy
        Fraction correct, counting abstentions as errors.
    selective_accuracy
        Fraction correct among the estimates that did commit to a state.
    abstention_rate
        Fraction of estimates that declined to name a state.
    balanced_accuracy
        Mean per-class recall, which is what stops a model that always
        predicts the majority state from looking good.
    macro_f1
        Unweighted mean F1 across the classes present in the truth.
    log_loss
        Mean negative log probability assigned to the true state.
    brier
        Multiclass Brier score of the full posterior, in ``[0, 2]``.
    calibration_error
        Expected calibration error: how far stated confidence is from
        observed accuracy.
    per_class_recall
        Recall for each state present in the truth.
    """

    n: int
    accuracy: float
    selective_accuracy: float
    abstention_rate: float
    balanced_accuracy: float
    macro_f1: float
    log_loss: float
    brier: float
    calibration_error: float
    per_class_recall: dict[BehaviouralState, float]

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the metrics."""
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "selective_accuracy": self.selective_accuracy,
            "abstention_rate": self.abstention_rate,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "log_loss": self.log_loss,
            "brier": self.brier,
            "calibration_error": self.calibration_error,
            "per_class_recall": {
                state.value: value for state, value in self.per_class_recall.items()
            },
        }


def _expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, bins: int = 10
) -> float:
    """Return the bin-weighted gap between confidence and accuracy."""
    if confidences.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        total += weight * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(total)


def state_metrics(
    truth: Sequence[BehaviouralState | None],
    estimates: Sequence[StateEstimate],
    *,
    calibration_bins: int = 10,
) -> StateMetrics:
    """Score a run of state estimates against known labels.

    Parameters
    ----------
    truth
        True state at each estimate, or ``None`` where no truth exists.
        Unlabelled positions are skipped rather than guessed at.
    estimates
        The corresponding estimates, in the same order.
    calibration_bins
        Number of confidence bins used for the calibration error.

    Raises
    ------
    ValueError
        If the two sequences differ in length, or nothing is labelled.
    """
    if len(truth) != len(estimates):
        raise ValueError("truth and estimates must be the same length")
    pairs = [
        (actual, estimate)
        for actual, estimate in zip(truth, estimates)
        if actual is not None
    ]
    if not pairs:
        raise ValueError("no labelled estimates to score")

    ontology: StateOntology = pairs[0][1].ontology
    states = ontology.states
    index = {state: position for position, state in enumerate(states)}

    reported = [estimate.state for _, estimate in pairs]
    actuals = [actual for actual, _ in pairs]
    beliefs = np.vstack([estimate.belief for _, estimate in pairs])
    confidences = np.array([estimate.confidence for _, estimate in pairs])
    argmax_correct = np.array(
        [actual is estimate.most_likely for actual, estimate in pairs],
        dtype=float,
    )

    correct = sum(1 for actual, said in zip(actuals, reported) if actual is said)
    abstained = sum(1 for said in reported if said is BehaviouralState.UNKNOWN)
    decided = len(pairs) - abstained

    truth_positions = np.array(
        [index[actual] for actual in actuals if actual in index], dtype=int
    )
    scored = np.array([actual in index for actual in actuals], dtype=bool)
    true_probability = np.full(len(pairs), EPSILON)
    true_probability[scored] = beliefs[scored, truth_positions]
    log_loss = float(-np.log(np.maximum(true_probability, EPSILON)).mean())

    one_hot = np.zeros_like(beliefs)
    one_hot[np.arange(len(pairs))[scored], truth_positions] = 1.0
    brier = float(((beliefs - one_hot) ** 2).sum(axis=1).mean())

    recalls: dict[BehaviouralState, float] = {}
    f1_scores: list[float] = []
    for state in sorted({a for a in actuals}, key=lambda s: s.value):
        actual_count = sum(1 for a in actuals if a is state)
        hits = sum(
            1 for a, said in zip(actuals, reported) if a is state and said is state
        )
        predicted_count = sum(1 for said in reported if said is state)
        recall = hits / actual_count if actual_count else 0.0
        precision = hits / predicted_count if predicted_count else 0.0
        recalls[state] = recall
        f1_scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

    return StateMetrics(
        n=len(pairs),
        accuracy=correct / len(pairs),
        selective_accuracy=(
            sum(
                1
                for actual, said in zip(actuals, reported)
                if said is not BehaviouralState.UNKNOWN and actual is said
            )
            / decided
            if decided
            else 0.0
        ),
        abstention_rate=abstained / len(pairs),
        balanced_accuracy=(float(np.mean(list(recalls.values()))) if recalls else 0.0),
        macro_f1=float(np.mean(f1_scores)) if f1_scores else 0.0,
        log_loss=log_loss,
        brier=brier,
        calibration_error=_expected_calibration_error(
            confidences, argmax_correct, calibration_bins
        ),
        per_class_recall=recalls,
    )


@dataclass(frozen=True)
class TimingMetrics:
    """How closely inferred state changes line up with real ones."""

    true_transitions: int
    matched: int
    median_error: float
    mean_absolute_error: float

    @property
    def matched_fraction(self) -> float:
        """Fraction of true transitions that were matched at all."""
        if self.true_transitions == 0:
            return 0.0
        return self.matched / self.true_transitions

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the metrics."""
        return {
            "true_transitions": self.true_transitions,
            "matched": self.matched,
            "matched_fraction": self.matched_fraction,
            "median_error_seconds": self.median_error,
            "mean_absolute_error_seconds": self.mean_absolute_error,
        }


def transition_timing(
    truth_times: Sequence[datetime],
    inferred_times: Sequence[datetime],
    *,
    tolerance: timedelta = timedelta(minutes=30),
) -> TimingMetrics:
    """Compare when state changes truly happened to when they were inferred.

    Each true transition is matched to the nearest inferred one within
    *tolerance*, and each inferred transition may be used only once, so a
    model that flickers rapidly cannot match every truth by accident.
    """
    remaining = sorted(inferred_times)
    errors: list[float] = []
    for moment in sorted(truth_times):
        if not remaining:
            break
        deltas = [abs((candidate - moment).total_seconds()) for candidate in remaining]
        best = int(np.argmin(deltas))
        if deltas[best] <= tolerance.total_seconds():
            errors.append((remaining[best] - moment).total_seconds())
            remaining.pop(best)
    return TimingMetrics(
        true_transitions=len(truth_times),
        matched=len(errors),
        median_error=float(np.median(errors)) if errors else float("nan"),
        mean_absolute_error=float(np.mean(np.abs(errors))) if errors else float("nan"),
    )


@dataclass(frozen=True)
class BinaryMetrics:
    """Quality of a probabilistic binary judgement, such as attribution."""

    n: int
    precision: float
    recall: float
    f1: float
    calibration_error: float
    positive_rate: float

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the metrics."""
        return {
            "n": self.n,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "calibration_error": self.calibration_error,
            "positive_rate": self.positive_rate,
        }


def binary_metrics(
    truth: Sequence[bool],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
    calibration_bins: int = 10,
) -> BinaryMetrics:
    """Score probabilistic binary judgements such as visitor presence."""
    if len(truth) != len(probabilities):
        raise ValueError("truth and probabilities must be the same length")
    if not truth:
        raise ValueError("no observations to score")

    actual = np.array(truth, dtype=bool)
    scores = np.asarray(probabilities, dtype=float)
    predicted = scores >= threshold

    true_positive = int((predicted & actual).sum())
    predicted_positive = int(predicted.sum())
    actual_positive = int(actual.sum())
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0

    # Calibration is scored on the stated probability of the predicted class,
    # so that a confident "no visitor" is judged as strictly as a confident
    # "visitor".
    confidence = np.where(predicted, scores, 1.0 - scores)
    hits = (predicted == actual).astype(float)

    return BinaryMetrics(
        n=len(truth),
        precision=precision,
        recall=recall,
        f1=(
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        ),
        calibration_error=_expected_calibration_error(
            confidence, hits, calibration_bins
        ),
        positive_rate=float(actual.mean()),
    )


@dataclass(frozen=True)
class DetectionMetrics:
    """Quality of behavioural change detection over a monitoring period."""

    true_changes: int
    detected: int
    false_positives: int
    person_days: float
    delays_days: tuple[float, ...]

    @property
    def recall(self) -> float:
        """Fraction of real changes that were detected in time."""
        return self.detected / self.true_changes if self.true_changes else 0.0

    @property
    def precision(self) -> float:
        """Fraction of raised alerts that corresponded to a real change."""
        total = self.detected + self.false_positives
        return self.detected / total if total else 0.0

    @property
    def false_positives_per_person_day(self) -> float:
        """Alert burden, the number that decides whether a system is usable."""
        return self.false_positives / self.person_days if self.person_days else 0.0

    @property
    def median_delay_days(self) -> float:
        """Median days between a change occurring and being reported."""
        return float(np.median(self.delays_days)) if self.delays_days else float("nan")

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the metrics."""
        return {
            "true_changes": self.true_changes,
            "detected": self.detected,
            "false_positives": self.false_positives,
            "person_days": self.person_days,
            "recall": self.recall,
            "precision": self.precision,
            "false_positives_per_person_day": self.false_positives_per_person_day,
            "median_delay_days": self.median_delay_days,
        }


def detection_metrics(
    detected_days: Sequence[object],
    true_change_days: Sequence[object],
    *,
    person_days: float,
    max_delay_days: float = 14.0,
) -> DetectionMetrics:
    """Score change detections against known change points.

    A detection counts only if it falls at or after the true change and
    within *max_delay_days*. Anything else is a false positive, including a
    detection that precedes the change: a system that alarms before anything
    happened has not detected it early, it has alarmed at noise.
    """
    if person_days <= 0:
        raise ValueError("person_days must be positive")

    unclaimed = sorted(detected_days)  # type: ignore[type-var]
    delays: list[float] = []
    matched: set[int] = set()

    for change in sorted(true_change_days):  # type: ignore[type-var]
        for position, moment in enumerate(unclaimed):
            if position in matched:
                continue
            delay = (moment - change).days  # type: ignore[operator]
            if 0 <= delay <= max_delay_days:
                delays.append(float(delay))
                matched.add(position)
                break

    return DetectionMetrics(
        true_changes=len(true_change_days),
        detected=len(delays),
        false_positives=len(unclaimed) - len(matched),
        person_days=float(person_days),
        delays_days=tuple(delays),
    )


@dataclass(frozen=True)
class PairedDifference:
    """A paired comparison between two methods on the same trajectories."""

    n: int
    mean_difference: float
    ci_low: float
    ci_high: float
    effect_size: float
    wins: int
    losses: int
    mcse: float = 0.0
    """Monte Carlo standard error of the mean difference.

    The precision the replication count bought. An interval without one cannot
    be told apart from an interval that is narrow only because the sample was
    small; see docs/SIMULATION_PROTOCOLS.md for choosing n from a target MCSE.
    """

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval excludes no difference at all."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable form of the comparison."""
        return {
            "n": self.n,
            "mean_difference": self.mean_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "effect_size": self.effect_size,
            "wins": self.wins,
            "losses": self.losses,
            "mcse": self.mcse,
            "excludes_zero": self.excludes_zero,
        }


def paired_difference(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> PairedDifference:
    """Compare two methods evaluated on the same simulated trajectories.

    Pairing matters more than it might seem. Simulated households differ from
    each other far more than two sensor configurations differ on one
    household, so an unpaired comparison buries a real effect under
    between-household variance. Every ablation in this package therefore
    evaluates all configurations on identical trajectories, and this function
    is what preserves that structure in the analysis.

    Reports a bootstrap interval and a standardised effect size rather than a
    p-value: with simulations, any effect can be made "significant" simply by
    running more seeds, so the size of the difference and its uncertainty are
    the informative quantities.
    """
    if len(treatment) != len(control):
        raise ValueError("paired comparison requires equal-length sequences")
    if len(treatment) < 2:
        raise ValueError("paired comparison requires at least two pairs")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")

    differences = np.asarray(treatment, dtype=float) - np.asarray(control, dtype=float)
    if not np.all(np.isfinite(differences)):
        raise ValueError("paired values must be finite")

    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(resamples, differences.size), replace=True)
    means = draws.mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    spread = float(differences.std(ddof=1))

    return PairedDifference(
        n=differences.size,
        mean_difference=float(differences.mean()),
        ci_low=float(np.quantile(means, tail)),
        ci_high=float(np.quantile(means, 1.0 - tail)),
        effect_size=(
            float(differences.mean() / spread)
            if spread > 0
            else (0.0 if math.isclose(float(differences.mean()), 0.0) else math.inf)
        ),
        wins=int((differences > 0).sum()),
        losses=int((differences < 0).sum()),
        mcse=float(spread / math.sqrt(differences.size)) if spread > 0 else 0.0,
    )


def summarise(values: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    """Return mean, standard deviation, and range for each named series."""
    return {
        name: {
            "mean": float(np.mean(series)),
            "sd": float(np.std(series, ddof=1)) if len(series) > 1 else 0.0,
            "min": float(np.min(series)),
            "max": float(np.max(series)),
            "n": float(len(series)),
        }
        for name, series in values.items()
        if len(series) > 0
    }
