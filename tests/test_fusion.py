"""Tests for the state ontology and multimodal Bayesian fusion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sensor_modeling.fusion import (
    BernoulliEmission,
    BetaEmission,
    FusionConfig,
    GaussianEmission,
    MultimodalBayesFilter,
    NonMonotonicUpdateError,
    PoissonEventEmission,
    belief_from_mapping,
    belief_matrix,
)
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
)
from sensor_modeling.states import BehaviouralState, StateOntology

T0 = datetime(2024, 5, 1, 3, 0, tzinfo=timezone.utc)
S = BehaviouralState


def ontology() -> StateOntology:
    """The default seven-state ontology."""
    return StateOntology()


def deployment() -> SensorRegistry:
    """A small multimodal deployment: motion, bed, wearable, radar."""
    return SensorRegistry.from_specs(
        [
            SensorSpec("kitchen_motion", Modality.MOTION, room="kitchen"),
            SensorSpec(
                "bed",
                Modality.BED_PRESSURE,
                kind=ObservationKind.STATE,
                room="bedroom",
                expected_interval=timedelta(minutes=5),
            ),
            SensorSpec(
                "wearable",
                Modality.WEARABLE_MOTION,
                kind=ObservationKind.SAMPLE,
                expected_interval=timedelta(minutes=1),
                attributable=True,
            ),
        ]
    )


def emissions() -> list:
    """Observation models matching the deployment."""
    onto = ontology()
    return [
        PoissonEventEmission(
            "kitchen_motion",
            rates={S.KITCHEN_ACTIVITY: 40.0, S.HOME_ACTIVE: 3.0},
            default_rate=0.05,
        ),
        BernoulliEmission(
            "bed",
            probabilities={S.SLEEPING: 0.98, S.BED_AWAKE: 0.95},
            default_probability=0.02,
        ),
        GaussianEmission(
            "wearable",
            means={
                S.SLEEPING: 0.02,
                S.BED_AWAKE: 0.2,
                S.HOME_INACTIVE: 0.3,
                S.HOME_ACTIVE: 1.5,
                S.KITCHEN_ACTIVITY: 1.2,
                S.BATHROOM_ACTIVITY: 1.0,
                S.AWAY: 1.8,
            },
            sigmas={state: 0.35 for state in onto.states},
        ),
    ]


def bed(at: datetime, value: float) -> Observation:
    """A bed-occupancy state report."""
    return Observation(at, "bed", Modality.BED_PRESSURE, ObservationKind.STATE, value)


def wearable(at: datetime, value: float) -> Observation:
    """A wearable activity sample."""
    return Observation(
        at, "wearable", Modality.WEARABLE_MOTION, ObservationKind.SAMPLE, value
    )


def kitchen(at: datetime) -> Observation:
    """A kitchen motion activation."""
    return Observation(
        at, "kitchen_motion", Modality.MOTION, ObservationKind.EVENT, 1.0
    )


def make_filter(**kwargs: object) -> MultimodalBayesFilter:
    """Build a filter over the standard deployment."""
    return MultimodalBayesFilter(
        ontology(), emissions(), registry=deployment(), **kwargs  # type: ignore[arg-type]
    )


class TestStateOntology:
    def test_generator_rows_sum_to_zero(self) -> None:
        assert np.abs(ontology().generator.sum(axis=1)).max() < 1e-12

    def test_transition_matrices_are_row_stochastic(self) -> None:
        matrix = ontology().transition(timedelta(minutes=15))
        assert np.allclose(matrix.sum(axis=1), 1.0)
        assert matrix.min() >= 0.0

    def test_zero_elapsed_time_is_the_identity(self) -> None:
        assert np.allclose(ontology().transition(timedelta(0)), np.eye(7))

    def test_short_intervals_preserve_state_more_than_long_ones(self) -> None:
        onto = ontology()
        short = np.diag(onto.transition(timedelta(minutes=1)))
        long = np.diag(onto.transition(timedelta(hours=2)))
        assert np.all(short > long)

    def test_long_intervals_converge_to_the_stationary_distribution(self) -> None:
        onto = ontology()
        matrix = onto.transition(timedelta(days=7))
        assert np.allclose(
            matrix, np.tile(onto.stationary(), (onto.size, 1)), atol=1e-3
        )

    def test_stationary_distribution_is_a_distribution(self) -> None:
        stationary = ontology().stationary()
        assert stationary.min() >= 0.0
        assert stationary.sum() == pytest.approx(1.0)

    def test_unknown_is_not_an_occupiable_state(self) -> None:
        with pytest.raises(ValueError, match="abstention"):
            StateOntology(
                states=(S.AWAY, S.UNKNOWN), dwell={S.AWAY: timedelta(hours=1)}
            )

    def test_every_state_needs_a_dwell_time(self) -> None:
        with pytest.raises(ValueError, match="no mean dwell time"):
            StateOntology(
                states=(S.AWAY, S.HOME_ACTIVE), dwell={S.AWAY: timedelta(hours=1)}
            )

    def test_duplicate_states_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            StateOntology(
                states=(S.AWAY, S.AWAY),
                dwell={S.AWAY: timedelta(hours=1)},
            )

    def test_room_lookups(self) -> None:
        onto = ontology()
        assert onto.room_of(S.KITCHEN_ACTIVITY) == "kitchen"
        assert onto.room_of(S.HOME_ACTIVE) is None
        assert set(onto.states_in_room("bedroom")) == {S.SLEEPING, S.BED_AWAKE}


class TestEmissionModels:
    def test_zero_reliability_contributes_nothing(self) -> None:
        emission = PoissonEventEmission(
            "kitchen_motion", rates={S.KITCHEN_ACTIVITY: 40.0}
        )
        result = emission.log_likelihood(
            ontology(), [kitchen(T0)], timedelta(minutes=5), reliability=0.0
        )
        assert np.allclose(result, 0.0)

    def test_zero_attribution_contributes_nothing(self) -> None:
        emission = PoissonEventEmission(
            "kitchen_motion", rates={S.KITCHEN_ACTIVITY: 40.0}
        )
        result = emission.log_likelihood(
            ontology(), [kitchen(T0)], timedelta(minutes=5), attribution=0.0
        )
        assert np.allclose(result, 0.0)

    def test_partial_reliability_scales_the_evidence(self) -> None:
        emission = PoissonEventEmission(
            "kitchen_motion", rates={S.KITCHEN_ACTIVITY: 40.0}
        )
        full = emission.log_likelihood(ontology(), [kitchen(T0)], timedelta(minutes=5))
        half = emission.log_likelihood(
            ontology(), [kitchen(T0)], timedelta(minutes=5), reliability=0.5
        )
        assert np.allclose(half, 0.5 * full)

    def test_poisson_silence_penalises_high_rate_states(self) -> None:
        """A quiet hour is evidence against being in the kitchen."""
        onto = ontology()
        emission = PoissonEventEmission(
            "kitchen_motion", rates={S.KITCHEN_ACTIVITY: 40.0}, default_rate=0.05
        )
        result = emission.log_likelihood(onto, [], timedelta(hours=1))
        assert result[onto.index(S.KITCHEN_ACTIVITY)] < result[onto.index(S.SLEEPING)]

    def test_poisson_activations_favour_high_rate_states(self) -> None:
        onto = ontology()
        emission = PoissonEventEmission(
            "kitchen_motion", rates={S.KITCHEN_ACTIVITY: 40.0}, default_rate=0.05
        )
        events = [kitchen(T0 + timedelta(seconds=20 * i)) for i in range(5)]
        result = emission.log_likelihood(onto, events, timedelta(minutes=5))
        assert onto.states[int(np.argmax(result))] is S.KITCHEN_ACTIVITY

    def test_emission_output_is_centred_on_its_best_state(self) -> None:
        onto = ontology()
        emission = PoissonEventEmission(
            "kitchen_motion", rates={S.KITCHEN_ACTIVITY: 40.0}
        )
        result = emission.log_likelihood(onto, [kitchen(T0)], timedelta(minutes=5))
        assert result.max() == pytest.approx(0.0)

    def test_bernoulli_uses_only_the_latest_state_report(self) -> None:
        onto = ontology()
        emission = BernoulliEmission("bed", probabilities={S.SLEEPING: 0.98})
        repeated = emission.log_likelihood(
            onto, [bed(T0, 1.0), bed(T0, 1.0), bed(T0, 1.0)], timedelta(minutes=5)
        )
        single = emission.log_likelihood(onto, [bed(T0, 1.0)], timedelta(minutes=5))
        assert np.allclose(repeated, single)

    def test_beta_emission_prefers_states_matching_the_reported_probability(
        self,
    ) -> None:
        onto = ontology()
        emission = BetaEmission(
            "radar",
            means={S.SLEEPING: 0.9, S.BED_AWAKE: 0.9, S.AWAY: 0.05},
            default_mean=0.4,
            concentration=20.0,
        )
        observation = Observation(
            T0, "radar", Modality.RADAR, ObservationKind.SAMPLE, 0.95
        )
        result = emission.log_likelihood(onto, [observation], timedelta(minutes=1))
        assert result[onto.index(S.AWAY)] < result[onto.index(S.SLEEPING)]

    def test_no_observations_means_no_preference_for_value_models(self) -> None:
        onto = ontology()
        gaussian = GaussianEmission("wearable", means={S.SLEEPING: 0.0})
        assert np.allclose(gaussian.log_likelihood(onto, [], timedelta(minutes=5)), 0.0)

    @pytest.mark.parametrize(
        ("factory", "message"),
        [
            (lambda: PoissonEventEmission("s", rates={S.AWAY: -1.0}), "non-negative"),
            (lambda: GaussianEmission("s", sigmas={S.AWAY: 0.0}), "positive"),
            (lambda: BernoulliEmission("s", probabilities={S.AWAY: 1.5}), r"\[0, 1\]"),
            (lambda: BetaEmission("s", concentration=0.0), "positive"),
            (lambda: PoissonEventEmission("", rates={}), "sensor_id"),
            (lambda: PoissonEventEmission("s", weight=-1.0), "weight"),
        ],
    )
    def test_invalid_emission_configuration_is_rejected(
        self, factory: object, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            factory()  # type: ignore[operator]


class TestFusion:
    @staticmethod
    def run_night(bayes: MultimodalBayesFilter, steps: int = 12) -> object:
        """Drive the filter through a stretch of in-bed, low-motion evidence."""
        estimate = None
        for step in range(steps):
            now = T0 + timedelta(minutes=5 * step)
            estimate = bayes.update(now, [bed(now, 1.0), wearable(now, 0.02)])
        return estimate

    def test_bed_and_low_wearable_motion_infer_sleep(self) -> None:
        estimate = self.run_night(make_filter())
        assert estimate.state is S.SLEEPING  # type: ignore[union-attr]
        assert estimate.confidence > 0.8  # type: ignore[union-attr]

    def test_kitchen_bursts_infer_kitchen_activity(self) -> None:
        bayes = make_filter()
        now = T0
        for step in range(12):
            now = T0 + timedelta(minutes=5 * step)
            records = [bed(now, 0.0), wearable(now, 1.3)]
            records += [kitchen(now - timedelta(seconds=20 * i)) for i in range(4)]
            estimate = bayes.update(now, records)
        assert estimate.state is S.KITCHEN_ACTIVITY

    def test_a_dead_sensor_does_not_become_evidence_of_inactivity(self) -> None:
        """The whole point: reliability zero removes a sensor, it does not
        turn its silence into an observation."""
        bayes = make_filter()
        now = T0
        for step in range(12):
            now = T0 + timedelta(minutes=5 * step)
            records = [bed(now, 0.0)]
            records += [kitchen(now - timedelta(seconds=20 * i)) for i in range(4)]
            estimate = bayes.update(now, records, reliabilities={"wearable": 0.0})
        assert estimate.state is S.KITCHEN_ACTIVITY
        assert "wearable" in estimate.missing
        assert estimate.completeness == pytest.approx(2 / 3)

    def test_total_sensor_failure_produces_an_abstention(self) -> None:
        bayes = make_filter()
        now = T0
        dead = {"wearable": 0.0, "bed": 0.0, "kitchen_motion": 0.0}
        for step in range(24):
            now = T0 + timedelta(minutes=15 * step)
            estimate = bayes.update(now, [], reliabilities=dead)
        assert estimate.state is S.UNKNOWN
        assert estimate.abstained
        assert estimate.completeness == 0.0
        assert set(estimate.missing) == set(dead)

    def test_prolonged_absence_of_evidence_erodes_confidence(self) -> None:
        bayes = make_filter()
        confident = self.run_night(bayes)
        now = T0 + timedelta(hours=1)
        for step in range(20):
            now = now + timedelta(hours=1)
            faded = bayes.update(now, [], reliabilities={"wearable": 0.0, "bed": 0.0})
        assert faded.confidence < confident.confidence  # type: ignore[union-attr]

    def test_contradictory_sensors_are_recorded_not_averaged_away(self) -> None:
        bayes = make_filter()
        now = T0 + timedelta(minutes=30)
        # The bed says the resident is lying down while the kitchen sensor
        # fires repeatedly. Both cannot be about the same person.
        records = [bed(now, 1.0)]
        records += [kitchen(now - timedelta(seconds=10 * i)) for i in range(8)]
        estimate = bayes.update(now, records)
        assert estimate.contradicting
        assert {c.sensor_id for c in estimate.contradicting} <= {
            "bed",
            "kitchen_motion",
            "wearable",
        }

    def test_a_silent_but_trusted_sensor_is_not_reported_as_missing(self) -> None:
        estimate = self.run_night(make_filter())
        assert "kitchen_motion" in estimate.silent  # type: ignore[union-attr]
        assert "kitchen_motion" not in estimate.missing  # type: ignore[union-attr]

    def test_attribution_discounts_ambient_evidence(self) -> None:
        """Activity that may have been a visitor should move the resident's
        state less far than activity known to be theirs.

        The invariant is directional rather than about any single state: the
        more the evidence is discounted, the closer the posterior stays to
        what the dynamics alone predicted.
        """
        records_at = T0 + timedelta(minutes=10)
        events = [kitchen(records_at - timedelta(seconds=20 * i)) for i in range(8)]

        def belief_after(share: float | None) -> np.ndarray:
            bayes = make_filter()
            bayes.update(T0, [])
            attribution = None if share is None else {"kitchen_motion": share}
            return bayes.update(records_at, events, attribution=attribution).belief

        def distance(left: np.ndarray, right: np.ndarray) -> float:
            return float(0.5 * np.abs(left - right).sum())

        baseline = belief_after(0.0)
        partial = belief_after(0.2)
        full = belief_after(None)

        assert distance(partial, baseline) < distance(full, baseline)

    def test_updates_must_not_move_backwards_in_time(self) -> None:
        bayes = make_filter()
        bayes.update(T0 + timedelta(hours=1), [])
        with pytest.raises(NonMonotonicUpdateError):
            bayes.update(T0, [])

    def test_observations_from_unmodelled_sensors_are_ignored(self) -> None:
        bayes = make_filter()
        stray = Observation(
            T0, "unmodelled", Modality.VIBRATION, ObservationKind.EVENT, 1.0
        )
        estimate = bayes.update(T0, [stray])
        assert {c.sensor_id for c in estimate.evidence} == set(bayes.emissions)

    def test_snapshot_and_restore_reproduce_the_belief(self) -> None:
        bayes = make_filter()
        self.run_night(bayes)
        state = bayes.snapshot()

        restarted = make_filter()
        restarted.restore(state)
        assert np.allclose(restarted.belief, bayes.belief)
        assert restarted.at == bayes.at

    def test_restore_rejects_a_snapshot_from_a_different_ontology(self) -> None:
        bayes = make_filter()
        bayes.update(T0, [])
        state = bayes.snapshot()
        state["states"] = ["away", "home_active"]
        with pytest.raises(ValueError, match="different state ontology"):
            bayes.restore(state)

    def test_a_filter_needs_at_least_one_emission_model(self) -> None:
        with pytest.raises(ValueError, match="at least one emission model"):
            MultimodalBayesFilter(ontology(), [])

    def test_duplicate_emission_models_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate emission model"):
            MultimodalBayesFilter(
                ontology(),
                [PoissonEventEmission("a"), PoissonEventEmission("a")],
            )

    def test_invalid_prior_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="prior"):
            MultimodalBayesFilter(ontology(), emissions(), prior=np.zeros(7))

    def test_invalid_fusion_config_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            FusionConfig(min_confidence=1.5)


class TestEstimateHelpers:
    def test_explanation_names_the_supporting_sensors(self) -> None:
        bayes = make_filter()
        estimate = TestFusion.run_night(bayes)
        text = estimate.explain()  # type: ignore[union-attr]
        assert "sleeping" in text
        assert "supported by" in text

    def test_explanation_of_an_abstention_states_the_reason(self) -> None:
        bayes = make_filter(config=FusionConfig(min_completeness=0.9))
        estimate = bayes.update(T0, [], reliabilities=0.1)
        assert "unknown" in estimate.explain()
        assert "coverage" in estimate.explain()

    def test_estimate_serialises(self) -> None:
        bayes = make_filter()
        payload = TestFusion.run_night(bayes).to_dict()  # type: ignore[union-attr]
        assert payload["state"] == "sleeping"
        assert set(payload["probabilities"]) == set(ontology().labels())
        assert isinstance(payload["evidence"], list)

    def test_entropy_is_maximal_for_a_flat_belief(self) -> None:
        bayes = make_filter()
        estimate = bayes.update(T0, [], reliabilities=0.0)
        flat = make_filter(prior=np.full(7, 1 / 7))
        undecided = flat.update(T0, [], reliabilities=0.0)
        assert undecided.normalised_entropy == pytest.approx(1.0)
        assert estimate.normalised_entropy <= 1.0

    def test_belief_helpers(self) -> None:
        onto = ontology()
        belief = belief_from_mapping(onto, {S.SLEEPING: 1.0})
        assert belief[onto.index(S.SLEEPING)] == 1.0
        assert belief.sum() == 1.0
        assert belief_matrix([]).shape == (0, 0)

        bayes = make_filter()
        estimates = [bayes.update(T0 + timedelta(minutes=5 * i), []) for i in range(3)]
        assert belief_matrix(estimates).shape == (3, onto.size)
