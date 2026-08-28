"""Tests for evaluation metrics and the paired ablation framework."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from sensor_modeling.evaluation import (
    SensorConfiguration,
    binary_metrics,
    detection_metrics,
    leave_one_out,
    named_subsets,
    paired_difference,
    run_ablation,
    state_metrics,
    summarise,
    transition_timing,
)
from sensor_modeling.evaluation.ablation import AblationReport, AblationRun
from sensor_modeling.fusion.estimate import StateEstimate
from sensor_modeling.simulation import HouseholdConfig, build_registry
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

T0 = datetime(2024, 5, 1, tzinfo=timezone.utc)
ONTOLOGY = StateOntology()


def estimate(
    state: S,
    *,
    confidence: float = 0.9,
    completeness: float = 1.0,
    min_confidence: float = 0.35,
    at: datetime = T0,
) -> StateEstimate:
    """Build an estimate concentrated on *state*."""
    belief = np.full(ONTOLOGY.size, (1.0 - confidence) / (ONTOLOGY.size - 1))
    belief[ONTOLOGY.index(state)] = confidence
    return StateEstimate(
        at=at,
        ontology=ONTOLOGY,
        belief=belief,
        evidence=(),
        completeness=completeness,
        min_confidence=min_confidence,
        min_completeness=0.25,
    )


class TestStateMetrics:
    def test_perfect_predictions_score_perfectly(self) -> None:
        truth = [S.SLEEPING, S.KITCHEN_ACTIVITY, S.AWAY]
        estimates = [estimate(state, confidence=0.999) for state in truth]
        metrics = state_metrics(truth, estimates)
        assert metrics.accuracy == 1.0
        assert metrics.balanced_accuracy == pytest.approx(1.0)
        assert metrics.macro_f1 == pytest.approx(1.0)
        assert metrics.log_loss < 0.01
        assert metrics.brier < 0.01

    def test_balanced_accuracy_punishes_predicting_only_the_majority(self) -> None:
        """This is why accuracy alone is not reported anywhere."""
        truth = [S.HOME_INACTIVE] * 90 + [S.KITCHEN_ACTIVITY] * 10
        estimates = [estimate(S.HOME_INACTIVE) for _ in truth]
        metrics = state_metrics(truth, estimates)
        assert metrics.accuracy == pytest.approx(0.9)
        assert metrics.balanced_accuracy == pytest.approx(0.5)

    def test_abstentions_cost_accuracy_and_are_reported_separately(self) -> None:
        truth = [S.SLEEPING] * 10
        estimates = [
            estimate(S.SLEEPING, confidence=0.2, min_confidence=0.5) for _ in truth
        ]
        metrics = state_metrics(truth, estimates)
        assert metrics.abstention_rate == 1.0
        assert metrics.accuracy == 0.0
        assert metrics.selective_accuracy == 0.0

    def test_selective_accuracy_credits_useful_declining(self) -> None:
        truth = [S.SLEEPING] * 5 + [S.KITCHEN_ACTIVITY] * 5
        estimates = [estimate(S.SLEEPING) for _ in range(5)]
        estimates += [
            estimate(S.AWAY, confidence=0.2, min_confidence=0.5) for _ in range(5)
        ]
        metrics = state_metrics(truth, estimates)
        assert metrics.abstention_rate == 0.5
        assert metrics.selective_accuracy == 1.0
        assert metrics.accuracy == 0.5

    def test_overconfidence_shows_up_as_calibration_error(self) -> None:
        truth = [S.SLEEPING] * 5 + [S.AWAY] * 5
        confident_but_wrong = [estimate(S.SLEEPING, confidence=0.99) for _ in truth]
        metrics = state_metrics(truth, confident_but_wrong)
        assert metrics.calibration_error > 0.4

    def test_honest_uncertainty_calibrates_better_than_false_confidence(self) -> None:
        truth = [S.SLEEPING] * 5 + [S.AWAY] * 5
        honest = [estimate(S.SLEEPING, confidence=0.5) for _ in truth]
        overconfident = [estimate(S.SLEEPING, confidence=0.99) for _ in truth]
        assert (
            state_metrics(truth, honest).calibration_error
            < state_metrics(truth, overconfident).calibration_error
        )

    def test_log_loss_penalises_confident_errors_more_than_hedged_ones(self) -> None:
        truth = [S.AWAY] * 4
        hedged = [estimate(S.SLEEPING, confidence=0.4) for _ in truth]
        certain = [estimate(S.SLEEPING, confidence=0.99) for _ in truth]
        assert (
            state_metrics(truth, certain).log_loss
            > state_metrics(truth, hedged).log_loss
        )

    def test_unlabelled_positions_are_skipped(self) -> None:
        truth = [S.SLEEPING, None, S.SLEEPING]
        estimates = [estimate(S.SLEEPING) for _ in truth]
        assert state_metrics(truth, estimates).n == 2

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            state_metrics([S.SLEEPING], [])

    def test_no_labels_is_an_error_not_a_perfect_score(self) -> None:
        with pytest.raises(ValueError, match="no labelled estimates"):
            state_metrics([None], [estimate(S.SLEEPING)])

    def test_metrics_serialise(self) -> None:
        payload = state_metrics([S.SLEEPING], [estimate(S.SLEEPING)]).to_dict()
        assert payload["n"] == 1
        assert "per_class_recall" in payload


class TestTransitionTiming:
    def test_exact_transitions_have_zero_error(self) -> None:
        moments = [T0, T0 + timedelta(hours=2)]
        metrics = transition_timing(moments, moments)
        assert metrics.matched == 2
        assert metrics.median_error == 0.0

    def test_late_detection_is_reported_as_a_positive_offset(self) -> None:
        metrics = transition_timing([T0], [T0 + timedelta(minutes=10)])
        assert metrics.median_error == pytest.approx(600.0)

    def test_transitions_beyond_tolerance_go_unmatched(self) -> None:
        metrics = transition_timing(
            [T0], [T0 + timedelta(hours=4)], tolerance=timedelta(minutes=30)
        )
        assert metrics.matched == 0
        assert metrics.matched_fraction == 0.0
        assert math.isnan(metrics.median_error)

    def test_one_inferred_transition_cannot_match_two_true_ones(self) -> None:
        """A flickering model must not claim credit twice."""
        metrics = transition_timing(
            [T0, T0 + timedelta(minutes=5)], [T0 + timedelta(minutes=1)]
        )
        assert metrics.matched == 1
        assert metrics.to_dict()["true_transitions"] == 2


class TestBinaryMetrics:
    def test_perfect_attribution_scores_perfectly(self) -> None:
        metrics = binary_metrics([True, False, True], [0.99, 0.01, 0.98])
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0

    def test_missed_visitors_reduce_recall(self) -> None:
        metrics = binary_metrics([True, True, False], [0.9, 0.1, 0.1])
        assert metrics.recall == pytest.approx(0.5)
        assert metrics.precision == 1.0

    def test_confident_mistakes_show_up_as_calibration_error(self) -> None:
        metrics = binary_metrics([False] * 10, [0.99] * 10)
        assert metrics.calibration_error > 0.9

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            binary_metrics([True], [0.5, 0.5])

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no observations"):
            binary_metrics([], [])


class TestDetectionMetrics:
    def test_a_timely_detection_records_its_delay(self) -> None:
        metrics = detection_metrics(
            [date(2024, 5, 5)], [date(2024, 5, 1)], person_days=30.0
        )
        assert metrics.detected == 1
        assert metrics.median_delay_days == 4.0
        assert metrics.recall == 1.0
        assert metrics.false_positives == 0

    def test_an_alert_before_the_change_is_a_false_positive(self) -> None:
        """Alarming before anything happened is noise, not early detection."""
        metrics = detection_metrics(
            [date(2024, 4, 20)], [date(2024, 5, 1)], person_days=30.0
        )
        assert metrics.detected == 0
        assert metrics.false_positives == 1

    def test_a_detection_beyond_the_window_does_not_count(self) -> None:
        metrics = detection_metrics(
            [date(2024, 6, 30)],
            [date(2024, 5, 1)],
            person_days=60.0,
            max_delay_days=14.0,
        )
        assert metrics.detected == 0
        assert metrics.false_positives == 1

    def test_alert_burden_is_expressed_per_person_day(self) -> None:
        metrics = detection_metrics(
            [date(2024, 5, 2), date(2024, 5, 3), date(2024, 5, 4)],
            [],
            person_days=30.0,
        )
        assert metrics.false_positives == 3
        assert metrics.false_positives_per_person_day == pytest.approx(0.1)
        assert metrics.precision == 0.0

    def test_a_quiet_system_with_no_changes_scores_cleanly(self) -> None:
        metrics = detection_metrics([], [], person_days=30.0)
        assert metrics.false_positives_per_person_day == 0.0
        assert math.isnan(metrics.median_delay_days)

    def test_person_days_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="person_days"):
            detection_metrics([], [], person_days=0.0)


class TestPairedComparison:
    def test_a_consistent_improvement_excludes_zero(self) -> None:
        control = [0.60, 0.62, 0.58, 0.61, 0.59, 0.63]
        treatment = [value + 0.10 for value in control]
        result = paired_difference(treatment, control)
        assert result.mean_difference == pytest.approx(0.10)
        assert result.excludes_zero
        assert result.wins == len(control)

    def test_noise_does_not_exclude_zero(self) -> None:
        rng = np.random.default_rng(0)
        control = list(rng.normal(0.6, 0.05, size=12))
        treatment = list(rng.normal(0.6, 0.05, size=12))
        assert not paired_difference(treatment, control).excludes_zero

    def test_pairing_finds_an_effect_that_pooling_would_hide(self) -> None:
        """Between-household variance dwarfs the effect; pairing removes it.

        This is the whole justification for the paired ablation design: the
        same data yields a decisive answer when the households are matched
        and an inconclusive one when they are pooled.
        """
        rng = np.random.default_rng(3)
        household = rng.normal(0.6, 0.15, size=20)
        control = list(household)
        treatment = list(household + 0.03 + rng.normal(0.0, 0.005, size=20))

        paired = paired_difference(treatment, control)
        assert paired.excludes_zero
        assert paired.mean_difference == pytest.approx(0.03, abs=0.01)

        # Pooling the two groups instead of matching them leaves an interval
        # dominated by how much households differ from each other.
        pooled_error = float(
            np.sqrt(
                np.var(treatment, ddof=1) / len(treatment)
                + np.var(control, ddof=1) / len(control)
            )
        )
        assert abs(paired.mean_difference) < 1.96 * pooled_error

    def test_effect_size_is_reported_alongside_the_interval(self) -> None:
        control = [0.5, 0.5, 0.5, 0.5]
        treatment = [0.6, 0.6, 0.6, 0.6]
        result = paired_difference(treatment, control)
        assert math.isinf(result.effect_size)
        assert result.to_dict()["wins"] == 4

    def test_unequal_or_tiny_samples_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            paired_difference([1.0, 2.0], [1.0])
        with pytest.raises(ValueError, match="at least two pairs"):
            paired_difference([1.0], [1.0])

    def test_non_finite_values_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            paired_difference([1.0, float("nan")], [1.0, 1.0])

    def test_summarise_reports_spread(self) -> None:
        stats = summarise({"a": [1.0, 2.0, 3.0]})
        assert stats["a"]["mean"] == 2.0
        assert stats["a"]["n"] == 3.0


class TestAblation:
    def test_configurations_restrict_the_registry(self) -> None:
        registry = build_registry()
        configuration = SensorConfiguration("pair", ("front_door", "bed_pressure"))
        assert configuration.restrict(registry).sensor_ids() == [
            "front_door",
            "bed_pressure",
        ]

    def test_an_empty_configuration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no sensors"):
            SensorConfiguration("empty", ())

    def test_leave_one_out_builds_the_full_set_plus_each_removal(self) -> None:
        registry = build_registry()
        configurations = leave_one_out(registry, ["front_door", "bed_pressure"])
        names = [configuration.name for configuration in configurations]
        assert names == ["all", "without_front_door", "without_bed_pressure"]
        assert len(configurations[1].sensors) == len(registry) - 1

    def test_a_paired_sweep_evaluates_every_configuration_on_every_seed(self) -> None:
        report = run_ablation(
            named_subsets(
                {
                    "all": tuple(build_registry().sensor_ids()),
                    "no_wearable": (
                        "front_door",
                        "bedroom_motion",
                        "kitchen_motion",
                        "bed_pressure",
                    ),
                }
            ),
            seeds=[1, 2],
            household=HouseholdConfig(days=3),
            step=timedelta(minutes=15),
        )
        assert report.seeds == [1, 2]
        assert len(report.runs) == 4
        assert len(report.series("all")) == 2

    def test_richer_sensing_beats_a_minimal_subset(self) -> None:
        report = run_ablation(
            named_subsets(
                {
                    "all": tuple(build_registry().sensor_ids()),
                    "minimal": ("front_door", "bed_pressure"),
                }
            ),
            seeds=[5, 6, 7],
            household=HouseholdConfig(days=3),
            step=timedelta(minutes=15),
        )
        difference = report.compare("all", "minimal")
        assert difference.mean_difference > 0
        assert difference.wins == 3

    def test_an_incomplete_sweep_is_refused_rather_than_silently_unpaired(self) -> None:
        """Comparing configurations across different households is exactly the
        mistake the paired design exists to prevent."""
        metrics = state_metrics([S.SLEEPING], [estimate(S.SLEEPING)])
        report = AblationReport(
            runs=[
                AblationRun("all", 1, metrics, ("a",)),
                AblationRun("all", 2, metrics, ("a",)),
                AblationRun("partial", 1, metrics, ("a",)),
            ]
        )
        with pytest.raises(ValueError, match="missing seeds"):
            report.series("partial")

    def test_duplicate_configuration_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            run_ablation(
                [
                    SensorConfiguration("same", ("front_door",)),
                    SensorConfiguration("same", ("bed_pressure",)),
                ],
                seeds=[1],
                household=HouseholdConfig(days=2),
            )

    def test_an_empty_sweep_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one configuration"):
            run_ablation([], seeds=[1])

    def test_interaction_measures_departure_from_additivity(self) -> None:
        metrics = state_metrics([S.SLEEPING], [estimate(S.SLEEPING)])

        def run(name: str, score: float) -> list[AblationRun]:
            adjusted = state_metrics(
                [S.SLEEPING], [estimate(S.SLEEPING, confidence=0.9)]
            )
            object.__setattr__(adjusted, "balanced_accuracy", score)
            return [AblationRun(name, seed, adjusted, ("a",)) for seed in (1, 2)]

        report = AblationReport(
            runs=[
                *run("all", 0.90),
                *run("without_a", 0.85),
                *run("without_b", 0.85),
                *run("without_both", 0.50),
            ]
        )
        assert metrics.n == 1
        # Each sensor alone is worth 0.05; together they are worth 0.40, so
        # the pair is strongly complementary.
        assert report.interaction("all", "without_a", "without_b", "without_both") > 0.2
