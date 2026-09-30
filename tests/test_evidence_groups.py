"""Tests for evidence-group disagreement diagnostics.

The filter's update is split into its terms without changing what it computes,
and each window's evidence is compared group against group. The four synthetic
situations the diagnostics must tell apart are groups that agree, groups that
conflict, a group that is missing, and a group whose sensors are unreliable;
missing and unreliable evidence must never count as disagreement.
"""

from __future__ import annotations

import gzip
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.evaluation import ExperimentRecord, load_record
from sensor_modeling.evaluation.evidence_groups import (
    CONFLICT_RATIO,
    GROUPS,
    MODALITY_GROUPS,
    RESULT_LAYOUT,
    TIE,
    EvidenceGroup,
    GroupDisagreementRecorder,
    GroupWindow,
    group_of,
    group_window,
    read_trace,
)
from sensor_modeling.fusion import (
    ONLINE,
    BernoulliEmission,
    GaussianEmission,
    MultimodalBayesFilter,
    NonMonotonicUpdateError,
    NotEnumerated,
    PoissonEventEmission,
    StateEstimate,
    WindowTerms,
)
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
)
from sensor_modeling.online import BehaviouralSensingPipeline, PipelineConfig
from sensor_modeling.simulation import HouseholdConfig, SimulationResult, simulate
from sensor_modeling.states import BehaviouralState, StateOntology

S = BehaviouralState
T0 = datetime(2024, 5, 1, 3, 0, tzinfo=timezone.utc)

# ----------------------------------------------------------------------
# Synthetic windows: three states and one sensor per group.

STATES = ("away", "home_active", "sleeping")
MODALITIES = {
    "hall_motion": Modality.MOTION,
    "front_door": Modality.DOOR,
    "back_door": Modality.DOOR,
    "bed": Modality.BED_PRESSURE,
    "wearable": Modality.WEARABLE_MOTION,
}
FLOOR = 0.05


def favouring(state: str, ratio: float) -> np.ndarray:
    """A log-likelihood giving *state* *ratio* times every other state's."""
    values = np.zeros(len(STATES))
    values[STATES.index(state)] = math.log(ratio)
    centred: np.ndarray = values - values.mean()
    return centred


def terms(
    sensors: dict[str, tuple[np.ndarray, float, int]],
    predicted: Any = None,
) -> WindowTerms:
    """Window terms from ``sensor: (log-likelihood, reliability, records)``."""
    prior = np.full(len(STATES), 1.0 / len(STATES)) if predicted is None else predicted
    return WindowTerms(
        at=T0,
        elapsed=timedelta(minutes=5),
        predicted=np.asarray(prior, dtype=float),
        likelihoods={s: np.asarray(v[0], dtype=float) for s, v in sensors.items()},
        reliability={s: v[1] for s, v in sensors.items()},
        attribution={s: 1.0 for s in sensors},
        observations={s: v[2] for s, v in sensors.items()},
    )


def window(
    sensors: dict[str, tuple[np.ndarray, float, int]], **kwargs: Any
) -> GroupWindow:
    """Group one synthetic window."""
    predicted = kwargs.pop("predicted", None)
    return group_window(
        terms(sensors, predicted), MODALITIES, STATES, evidence_floor=FLOOR, **kwargs
    )


def group(w: GroupWindow, name: str) -> Any:
    """One group of a window."""
    return next(g for g in w.groups if g.group.value == name)


def log_odds(w: GroupWindow) -> float:
    """The full posterior's log odds of its first state over its second."""
    first, second = w.ranked
    return float(math.log(w.full[first]) - math.log(w.full[second]))


# ----------------------------------------------------------------------
# A small real deployment, for the filter and the emissions' semantics.


def deployment() -> SensorRegistry:
    """A door, a motion sensor, a bed sensor and a wearable."""
    return SensorRegistry.from_specs(
        [
            SensorSpec("front_door", Modality.DOOR, room="hall"),
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


def make_filter() -> MultimodalBayesFilter:
    """A filter over the small deployment."""
    ontology = StateOntology()
    return MultimodalBayesFilter(
        ontology,
        [
            PoissonEventEmission("front_door", rates={S.AWAY: 0.2}, default_rate=3.0),
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
                means={S.SLEEPING: 0.02, S.BED_AWAKE: 0.2, S.HOME_ACTIVE: 1.5},
                sigmas={state: 0.35 for state in ontology.states},
                default_mean=0.5,
                default_sigma=0.35,
            ),
        ],
        registry=deployment(),
    )


def bed(at: datetime, value: float) -> Observation:
    """A bed-occupancy report."""
    return Observation(at, "bed", Modality.BED_PRESSURE, ObservationKind.STATE, value)


def kitchen(at: datetime) -> Observation:
    """A kitchen motion activation."""
    return Observation(
        at, "kitchen_motion", Modality.MOTION, ObservationKind.EVENT, 1.0
    )


def statuses(record: dict[str, Any], wanted: set[str]) -> set[str]:
    """Every sensor of a window record whose status is one of *wanted*."""
    return {
        sensor
        for g in record["groups"].values()
        for sensor, status in g["sensors"].items()
        if status in wanted
    }


# ----------------------------------------------------------------------
class TestFilterTerms:
    """The update split into terms computes exactly what it did."""

    def test_terms_do_not_change_the_filter(self) -> None:
        bayes = make_filter()
        bayes.update(T0, [bed(T0, 1.0)])
        before, at = bayes.belief, bayes.at
        bayes.evidence_terms(
            T0 + timedelta(minutes=5), [kitchen(T0 + timedelta(minutes=2))]
        )
        assert np.array_equal(bayes.belief, before)
        assert bayes.at == at

    def test_terms_give_the_update_posterior_bit_for_bit(self) -> None:
        bayes = make_filter()
        bayes.update(T0, [bed(T0, 1.0)])
        later = T0 + timedelta(minutes=5)
        batch = [kitchen(T0 + timedelta(minutes=2)), bed(later, 0.0)]
        weights = {"front_door": 0.3, "bed": 0.8}
        terms_ = bayes.evidence_terms(
            later, batch, reliabilities=weights, attribution=0.7
        )
        bayes.update(later, batch, reliabilities=weights, attribution=0.7)
        assert np.array_equal(terms_.posterior, bayes.belief)
        assert terms_.observations == {
            "front_door": 0,
            "kitchen_motion": 1,
            "bed": 1,
            "wearable": 0,
        }
        assert terms_.reliability["front_door"] == 0.3
        assert terms_.attribution["bed"] == 0.7

    def test_observers_see_every_update_once(self) -> None:
        bayes = make_filter()
        seen: list[tuple[WindowTerms, StateEstimate]] = []
        bayes.observers.append(lambda t, e: seen.append((t, e)))
        first = bayes.update(T0, [bed(T0, 1.0)])
        second = bayes.update(T0 + timedelta(minutes=5), [])
        assert [e for _, e in seen] == [first, second]
        assert np.allclose(seen[1][0].posterior, second.belief, rtol=0.0, atol=1e-15)

    def test_terms_refuse_to_go_back_in_time(self) -> None:
        bayes = make_filter()
        bayes.update(T0)
        with pytest.raises(NonMonotonicUpdateError):
            bayes.evidence_terms(T0 - timedelta(seconds=1))


# ----------------------------------------------------------------------
class TestSyntheticChannels:
    """The four situations: agreement, conflict, missing and unreliable evidence."""

    def test_channels_that_agree(self) -> None:
        w = window(
            {
                "hall_motion": (favouring("sleeping", 3.0), 1.0, 0),
                "front_door": (favouring("sleeping", 2.0), 1.0, 0),
                "bed": (favouring("sleeping", 20.0), 1.0, 1),
                "wearable": (favouring("sleeping", 8.0), 1.0, 4),
            },
            predicted=[0.1, 0.2, 0.7],
        )
        record = w.to_dict()
        assert record["full"]["state"] == "sleeping"
        assert record["identifiable"] == [
            "motion",
            "contact",
            "bed",
            "wearable",
            "context",
        ]
        assert record["state_disagreement"] is False
        assert record["disagreements"] == [] and record["conflicts"] == []
        assert record["vote_disagreement"] == 0.0
        assert all(
            record["groups"][g]["supports_decision"] for g in record["identifiable"]
        )
        assert all(0.0 <= v < 0.3 for v in record["pairwise"].values())

    def test_identical_channels_do_not_diverge(self) -> None:
        w = window(
            {
                "hall_motion": (favouring("away", 5.0), 1.0, 1),
                "bed": (favouring("away", 5.0), 1.0, 1),
            }
        )
        assert w.pairwise()["motion|bed"] == 0.0

    def test_channels_that_conflict(self) -> None:
        w = window(
            {
                "hall_motion": (favouring("home_active", 50.0), 1.0, 6),
                "bed": (favouring("sleeping", 50.0), 1.0, 1),
                "wearable": (favouring("sleeping", 4.0), 1.0, 3),
            }
        )
        record = w.to_dict()
        assert record["state_disagreement"] is True
        assert record["disagreements"] == ["motion|bed", "motion|wearable"]
        assert record["conflicts"] == ["motion|bed", "motion|wearable"]
        assert record["pairwise"]["motion|bed"] > 0.5
        assert record["pairwise"]["bed|wearable"] < record["pairwise"]["motion|bed"]
        # Two votes for sleeping, one for home_active, a uniform context split.
        assert record["vote_disagreement"] == pytest.approx(1 - (2 + 1 / 3) / 4)
        assert record["groups"]["motion"]["supports_decision"] is False

    def test_weak_disagreement_is_not_conflict(self) -> None:
        w = window(
            {
                "hall_motion": (favouring("home_active", 1.5), 1.0, 1),
                "bed": (favouring("sleeping", 1.5), 1.0, 1),
            }
        )
        assert w.disagreements() == ["motion|bed"]
        assert w.conflicts() == []
        assert window(
            {
                "hall_motion": (favouring("home_active", 1.5), 1.0, 1),
                "bed": (favouring("sleeping", 1.5), 1.0, 1),
            },
            conflict_ratio=1.4,
        ).conflicts() == ["motion|bed"]

    def test_one_channel_missing(self) -> None:
        agreeing = {
            "hall_motion": (favouring("home_active", 6.0), 1.0, 2),
            "front_door": (favouring("home_active", 2.0), 1.0, 0),
            # A sample sensor with no samples says nothing: a flat likelihood.
            "wearable": (np.zeros(3), 1.0, 0),
        }
        record = window(agreeing).to_dict()
        assert record["groups"]["bed"]["status"] == "absent"
        assert record["groups"]["wearable"]["status"] == "uninformative"
        assert record["groups"]["wearable"]["sensors"] == {"wearable": "no data"}
        for missing in ("bed", "wearable", "other"):
            assert record["groups"][missing]["identifiable"] is False
            assert record["groups"][missing]["posterior"] is None
            assert record["groups"][missing]["divergence_from_full"] is None
            assert all(missing not in pair for pair in record["pairwise"])
        assert record["identifiable"] == ["motion", "contact", "context"]
        assert record["state_disagreement"] is False
        assert record["conflicts"] == []

    def test_one_channel_unreliable(self) -> None:
        conflicting = favouring("away", 1e40)
        trusted = window(
            {
                "hall_motion": (favouring("home_active", 20.0), 1.0, 3),
                "bed": (conflicting * 0.9, 0.9, 1),
            }
        )
        assert trusted.conflicts() == ["motion|bed"]
        # The same bed sensor, discounted below the evidence floor by its health:
        # its residual evidence still enters the filter, but is not compared.
        w = window(
            {
                "hall_motion": (favouring("home_active", 20.0), 1.0, 3),
                "bed": (conflicting * 0.02, 0.02, 1),
            }
        )
        record = w.to_dict()
        assert record["groups"]["bed"]["status"] == "unavailable"
        assert record["groups"]["bed"]["sensors"] == {"bed": "unavailable"}
        assert record["groups"]["bed"]["identifiable"] is False
        assert record["disagreements"] == [] and record["conflicts"] == []
        assert record["identifiable"] == ["motion", "context"]
        assert record["contributions"]["bed"] != 0.0
        assert sum(record["contributions"].values()) == pytest.approx(log_odds(w))

    def test_unreliable_sensor_leaves_its_group_to_the_others(self) -> None:
        w = window(
            {
                "front_door": (favouring("away", 1e40) * 0.01, 0.01, 1),
                "back_door": (favouring("home_active", 3.0), 1.0, 0),
            }
        )
        contact = group(w, "contact")
        assert contact.status == "silent"
        assert contact.sensors == {"front_door": "unavailable", "back_door": "silent"}
        assert contact.posterior is not None
        assert np.allclose(contact.posterior, [0.2, 0.6, 0.2])
        assert contact.favoured == frozenset({STATES.index("home_active")})


# ----------------------------------------------------------------------
class TestSensorHealthSemantics:
    """Statuses from real emissions keep the estimate's missing and silent."""

    def test_silence_unavailability_and_no_data(self) -> None:
        bayes = make_filter()
        bayes.update(T0)
        record = group_window(
            bayes.evidence_terms(
                T0 + timedelta(minutes=10),
                [],
                reliabilities={"kitchen_motion": 0.0},
            ),
            {s: bayes.registry.get(s).modality for s in bayes.emissions},  # type: ignore[union-attr]
            bayes.ontology.labels(),
            evidence_floor=bayes.config.evidence_floor,
        ).to_dict()
        # A trusted door that stayed shut is evidence; a broken motion sensor
        # is not; a bed and a wearable with no reports say nothing.
        assert record["groups"]["contact"]["status"] == "silent"
        assert record["groups"]["motion"]["status"] == "unavailable"
        assert record["groups"]["bed"]["status"] == "uninformative"
        assert record["groups"]["wearable"]["status"] == "uninformative"
        assert record["identifiable"] == ["contact", "context"]

    def test_statuses_match_the_estimate(self) -> None:
        bayes = make_filter()
        recorder = GroupDisagreementRecorder.attach(bayes)
        estimates = [
            bayes.update(T0, [bed(T0, 1.0)]),
            bayes.update(
                T0 + timedelta(minutes=5),
                [kitchen(T0 + timedelta(minutes=3))],
                reliabilities={"bed": 0.01, "front_door": 0.5},
            ),
            bayes.update(T0 + timedelta(minutes=10), [], reliabilities=0.0),
        ]
        for estimate, record in zip(estimates, recorder.windows, strict=True):
            assert statuses(record, {"unavailable"}) == set(estimate.missing)
            assert statuses(record, {"silent", "no data"}) == set(estimate.silent)
            assert statuses(record, {"reporting"}) == {
                c.sensor_id for c in estimate.evidence if c.reported and c.informative
            }
        # With every sensor missing, only the context is left, and nothing is compared.
        assert recorder.windows[-1]["identifiable"] == ["context"]
        assert recorder.windows[-1]["compared"] is False
        assert recorder.windows[-1]["state_disagreement"] is None


# ----------------------------------------------------------------------
class TestMeasures:
    """Ties, votes, the log-odds decomposition, bounds and determinism."""

    def test_tied_states_are_favoured_together(self) -> None:
        # A bed sensor that cannot tell home_active from sleeping favours both,
        # and so agrees with a wearable that favours sleeping.
        w = window(
            {
                "bed": (np.array([-2.0, 0.0, 0.0]), 1.0, 1),
                "wearable": (favouring("sleeping", 5.0), 1.0, 2),
            }
        )
        assert group(w, "bed").favoured == frozenset({1, 2})
        assert w.disagreements() == []
        # Its vote splits between the two: 1/2 + 1 + 1/3 for sleeping of 3.
        assert w.vote_disagreement() == pytest.approx(1 - (1 / 2 + 1 + 1 / 3) / 3)

    def test_ties_follow_the_declared_tolerance(self) -> None:
        near = np.array([-2.0, 0.0, -TIE / 2])
        assert group(window({"bed": (near, 1.0, 1)}), "bed").favoured == {1, 2}
        apart = np.array([-2.0, 0.0, -10 * TIE])
        assert group(window({"bed": (apart, 1.0, 1)}), "bed").favoured == {1}

    def test_contributions_decompose_the_decision_exactly(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(50):
            sensors = {
                s: (
                    rng.normal(0, 3, 3),
                    float(rng.choice([1.0, 0.5, 0.01])),
                    int(rng.integers(0, 3)),
                )
                for s in MODALITIES
            }
            w = window(sensors, predicted=rng.dirichlet(np.ones(3)))
            assert sum(w.contributions().values()) == pytest.approx(
                log_odds(w), abs=1e-9
            )

    def test_dominance(self) -> None:
        strong = window(
            {
                "bed": (favouring("sleeping", 1e6), 1.0, 1),
                "hall_motion": (favouring("sleeping", 2.0), 1.0, 0),
                "wearable": (favouring("home_active", 2.0), 1.0, 2),
            }
        )
        assert strong.dominance()["group"] == "bed"
        assert strong.dominance()["dominates"] is True
        balanced = window(
            {
                "bed": (favouring("sleeping", 4.0), 1.0, 1),
                "hall_motion": (favouring("sleeping", 4.0), 1.0, 0),
                "wearable": (favouring("sleeping", 4.0), 1.0, 2),
            }
        )
        assert balanced.dominance()["share"] == pytest.approx(1 / 3)
        assert balanced.dominance()["dominates"] is False
        nothing = window({"wearable": (np.zeros(3), 1.0, 0)})
        assert nothing.dominance() == {"group": None, "share": None, "dominates": False}

    def test_a_tied_decision_is_the_earlier_state(self) -> None:
        w = window(
            {
                "hall_motion": (favouring("home_active", 5.0), 1.0, 1),
                "bed": (favouring("sleeping", 5.0), 1.0, 1),
            }
        )
        record = w.to_dict()
        assert w.ranked == (1, 2) and w.decision == int(np.argmax(w.full))
        assert (record["full"]["state"], record["full"]["runner_up"]) == (
            "home_active",
            "sleeping",
        )
        assert record["groups"]["motion"]["supports_decision"] is True
        # Motion and bed cancel exactly: half each, and neither dominates.
        assert record["dominance"]["share"] == pytest.approx(0.5)
        assert record["dominance"]["dominates"] is False

    def test_context_can_dominate(self) -> None:
        w = window(
            {"hall_motion": (favouring("home_active", 2.0), 1.0, 1)},
            predicted=[0.001, 0.009, 0.99],
        )
        assert w.to_dict()["full"]["state"] == "sleeping"
        assert w.dominance()["group"] == "context"
        assert w.dominance()["dominates"] is True

    def test_bounds(self) -> None:
        rng = np.random.default_rng(11)
        for _ in range(200):
            sensors = {
                s: (
                    rng.normal(0, float(rng.choice([0.1, 3.0, 40.0])), 3)
                    * float(rng.integers(0, 2)),
                    float(rng.choice([1.0, 0.3, 0.0])),
                    int(rng.integers(0, 2)),
                )
                for s in MODALITIES
                if rng.random() < 0.8
            }
            record = window(
                sensors, predicted=rng.dirichlet(np.ones(3) * 0.3)
            ).to_dict()
            count = len(record["identifiable"])
            assert all(0.0 <= v <= 1.0 for v in record["pairwise"].values())
            for g in record["groups"].values():
                if g["divergence_from_full"] is not None:
                    assert 0.0 <= g["divergence_from_full"] <= 1.0
            if record["compared"]:
                assert 0.0 <= record["normalised_discordance"] <= 1.0 + 1e-12
                assert 0.0 <= record["vote_disagreement"] <= 1 - 1 / count + 1e-12
            share = record["dominance"]["share"]
            assert share is None or 0.0 <= share <= 1.0

    def test_symmetry(self) -> None:
        a, b = favouring("home_active", 7.0), favouring("sleeping", 3.0)
        forward = window({"hall_motion": (a, 1.0, 1), "bed": (b, 1.0, 1)})
        swapped = window({"hall_motion": (b, 1.0, 1), "bed": (a, 1.0, 1)})
        assert forward.pairwise()["motion|bed"] == swapped.pairwise()["motion|bed"]
        assert forward.conflicts() == swapped.conflicts()

    def test_determinism(self) -> None:
        sensors = {
            "hall_motion": (favouring("home_active", 7.0), 1.0, 1),
            "bed": (favouring("sleeping", 3.0), 0.6, 1),
        }
        first = json.dumps(window(sensors).to_dict(), sort_keys=True)
        assert json.dumps(window(sensors).to_dict(), sort_keys=True) == first

    def test_invalid_settings_are_refused(self) -> None:
        with pytest.raises(ValueError, match="conflict_ratio"):
            window({}, conflict_ratio=1.0)
        with pytest.raises(ValueError, match="one entry per state"):
            window({}, predicted=[0.5, 0.5])


# ----------------------------------------------------------------------
class TestGroups:
    """Groups come from the sensor ontology's modalities."""

    def test_every_modality_has_a_group(self) -> None:
        for modality in Modality:
            assert group_of(modality) in GROUPS
        assert group_of(Modality.RADAR) is EvidenceGroup.MOTION
        assert group_of(Modality.PROXIMITY) is EvidenceGroup.WEARABLE
        assert group_of(Modality.ENVIRONMENTAL) is EvidenceGroup.OTHER
        assert EvidenceGroup.CONTEXT not in MODALITY_GROUPS.values()

    def test_undeclared_sensors_fall_in_other(self) -> None:
        w = group_window(
            terms({"mystery": (favouring("away", 3.0), 1.0, 1)}),
            {},
            STATES,
            evidence_floor=FLOOR,
        )
        assert group(w, "other").sensors == {"mystery": "reporting"}


# ----------------------------------------------------------------------
class TestRecorder:
    """The recorder observes a filter without changing it."""

    def test_filters_that_override_update_are_refused(self) -> None:
        class Custom(MultimodalBayesFilter):
            def update(self, *args: Any, **kwargs: Any) -> StateEstimate:
                return super().update(*args, **kwargs)

        bayes = make_filter()
        custom = Custom(bayes.ontology, bayes.emissions.values(), registry=deployment())
        with pytest.raises(TypeError, match="overrides update"):
            GroupDisagreementRecorder(custom)
        with pytest.raises(ValueError, match="conflict_ratio"):
            GroupDisagreementRecorder(bayes, conflict_ratio=0.5)

    def test_trace_is_canonical_and_checked(self, tmp_path: Path) -> None:
        bayes = make_filter()
        recorder = GroupDisagreementRecorder.attach(bayes)
        bayes.update(T0, [bed(T0, 1.0)])
        bayes.update(T0 + timedelta(minutes=5), [kitchen(T0 + timedelta(minutes=1))])
        first = recorder.write(tmp_path / "a.json.gz")
        second = recorder.write(tmp_path / "b.json.gz")
        assert first == second
        assert (tmp_path / "a.json.gz").read_bytes() == (
            tmp_path / "b.json.gz"
        ).read_bytes()
        payload = read_trace(tmp_path / "a.json.gz", first)
        assert payload["windows"] == json.loads(json.dumps(recorder.windows))
        with pytest.raises(ValueError, match="digest"):
            read_trace(tmp_path / "a.json.gz", "0" * 64)
        (tmp_path / "c.json").write_bytes(
            gzip.compress(b'{"format": "something else"}')
        )
        with pytest.raises(ValueError, match="not a"):
            read_trace(tmp_path / "c.json")


@pytest.fixture(scope="module")
def household() -> SimulationResult:
    """One simulated day with every group of sensors."""
    return simulate(HouseholdConfig(days=1, seed=3, carer_weekday_visits=True))


class TestPipeline:
    """The recorder on the online pipeline over a simulated household."""

    def test_recording_changes_nothing(self, household: SimulationResult) -> None:
        config = PipelineConfig(tz=household.config.tz)
        plain = BehaviouralSensingPipeline(household.registry, config=config)
        observed = BehaviouralSensingPipeline(household.registry, config=config)
        recorder = GroupDisagreementRecorder.attach(observed.filter)
        expected = plain.run(household.observations)
        steps = observed.run(household.observations)
        assert len(recorder.windows) == len(steps) == len(expected)
        for a, b in zip(expected, steps, strict=True):
            assert np.array_equal(a.state.belief, b.state.belief)
            assert a.state.evidence == b.state.evidence
        for step, record in zip(steps, recorder.windows, strict=True):
            assert statuses(record, {"unavailable"}) == set(step.state.missing)
            assert statuses(record, {"silent", "no data"}) == set(step.state.silent)
            labels = step.state.ontology.labels()
            assert record["full"]["state"] == labels[int(np.argmax(step.state.belief))]

    def test_results_enter_an_experiment_record(
        self, household: SimulationResult, tmp_path: Path
    ) -> None:
        pipeline = BehaviouralSensingPipeline(
            household.registry, config=PipelineConfig(tz=household.config.tz)
        )
        recorder = GroupDisagreementRecorder.attach(pipeline.filter)
        pipeline.run(household.observations)
        digest = recorder.write(tmp_path / "trace.json.gz")
        results = recorder.results({"path": "trace.json.gz", "sha256": digest})
        assert results["layout"] == RESULT_LAYOUT
        assert results["conflict_ratio"] == CONFLICT_RATIO
        assert {v["group"] for v in results["sensors"].values()} == {
            "motion",
            "contact",
            "bed",
            "wearable",
        }
        summary = results["summary"]
        assert summary["windows"] == len(recorder.windows)
        assert set(summary["groups"]) == {g.value for g in GROUPS}
        assert summary["groups"]["other"]["statuses"] == {"absent": summary["windows"]}
        for value in (
            summary["state_disagreement"],
            summary["conflict"],
            summary["mean_normalised_discordance"],
            summary["dominated"],
        ):
            assert 0.0 <= value <= 1.0
        record = ExperimentRecord(
            experiment="evidence-groups",
            configuration={},
            inference=ONLINE,
            evidence=NotEnumerated("a unit-test record"),
            data_source="synthetic-test",
            results={"evidence_groups": results},
        )
        path = record.write(tmp_path / "record.json")
        assert load_record(path)["results"]["evidence_groups"] == json.loads(
            json.dumps(results)
        )
