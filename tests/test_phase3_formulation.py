"""Tests for the phase3 formulation in the online pipeline.

They guard what the opt-in rests on:
- the pipeline's beliefs with it are the research recursion's, step by step;
- a channel emission reads every sensor it pools, and is the research hurdle;
- a state the training homes never showed takes the home's declared count;
- the parameters are the ones the held-out record digests, and anything else,
  or anything the formulation was not fitted for, is refused;
- the pipeline's defaults are unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    fit_periodic_prior,
    hour_state_counts,
)
from sensor_modeling.datasets import phase3_formulation as p3
from sensor_modeling.datasets.channel_models import (
    HURDLE,
    FittedChannels,
    HurdleChannel,
    fit_channel_models,
    household_channel_counts,
    total_loglik,
)
from sensor_modeling.datasets.combined_prior_experiment import (
    periodic_recursion,
    periodic_transitions,
)
from sensor_modeling.datasets.information_sets import (
    EvidenceChannel,
    EvidenceResolution,
)
from sensor_modeling.datasets.matched_evaluation import _regular_moments
from sensor_modeling.datasets.periodic_prior import PeriodicStatePrior
from sensor_modeling.fusion import HurdleChannelEmission, MultimodalBayesFilter
from sensor_modeling.fusion.defaults import default_emissions
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
)
from sensor_modeling.online import BehaviouralSensingPipeline, PipelineConfig
from sensor_modeling.online.pipeline import scoring_steps
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState, StateOntology

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "sensor_modeling" / "data" / p3.PARAMETERS_FILE
ONTOLOGY = StateOntology()
STATES = tuple(ONTOLOGY.states)
STEP = timedelta(minutes=5)
RESOLUTION = EvidenceResolution()
UTC = timezone.utc


def simulated(seed: int, without: BehaviouralState | None = None) -> CasasRecording:
    """A simulated home, with the labels of *without* removed."""
    sim = simulate(HouseholdConfig(days=3, seed=seed))
    return CasasRecording(
        sim.registry,
        sim.observations,
        tuple(
            ActivityInterval(e.state.value, e.start, e.end, e.state)
            for e in sim.truth.episodes
            if e.state is not without
        ),
    )


def fit(
    homes: dict[str, CasasRecording],
) -> tuple[FittedChannels, PeriodicStatePrior]:
    channels = fit_channel_models(
        homes, resolution=RESOLUTION, pseudo_windows=12.0, ontology=ONTOLOGY
    )
    counts = {
        home: hour_state_counts(
            recording, _regular_moments(recording, STEP), states=STATES, step=STEP
        )
        for home, recording in homes.items()
    }
    return channels, fit_periodic_prior(counts, states=STATES)


def written(
    directory: Path, channels: FittedChannels, prior: PeriodicStatePrior
) -> p3.Phase3Formulation:
    path = p3.write_parameters(
        p3.parameters_payload(channels, prior, {"fitted": "simulated homes"}),
        directory / p3.PARAMETERS_FILE,
    )
    return p3.Phase3Formulation.load(
        path,
        channels_sha256=channels.sha256(),
        periodic_prior_sha256=prior.sha256(),
    )


@pytest.fixture(scope="module")
def target() -> CasasRecording:
    return simulated(1)


@pytest.fixture(scope="module")
def fitted() -> tuple[FittedChannels, PeriodicStatePrior]:
    return fit({f"sim{seed}": simulated(seed) for seed in (2, 3, 4)})


@pytest.fixture(scope="module")
def partial() -> tuple[FittedChannels, PeriodicStatePrior]:
    """Fitted on homes whose `bed_awake` was never labelled."""
    return fit(
        {
            f"sim{seed}": simulated(seed, without=BehaviouralState.BED_AWAKE)
            for seed in (2, 3)
        }
    )


@pytest.fixture
def formulation(
    fitted: tuple[FittedChannels, PeriodicStatePrior],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> p3.Phase3Formulation:
    """The pipeline loads parameters fitted on simulated homes."""
    loaded = written(tmp_path, *fitted)
    monkeypatch.setattr(
        p3.Phase3Formulation, "load", classmethod(lambda cls, *a, **k: loaded)
    )
    return loaded


def research_posterior(
    recording: CasasRecording, channels: FittedChannels, prior: PeriodicStatePrior
) -> tuple[np.ndarray, list[datetime]]:
    """The combined-prior recursion of one home, as the held-out run scores it."""
    counts, _, _, moments = household_channel_counts(
        recording, RESOLUTION, ONTOLOGY, household="target"
    )
    hours = [moment.hour for moment in moments]
    models = channels.models(recording.registry, RESOLUTION, HURDLE, ONTOLOGY)
    _, posterior = periodic_recursion(
        total_loglik(models, counts),
        periodic_transitions(prior, ONTOLOGY, STEP),
        hours,
        prior.probabilities([hours[0]])[0],
    )
    return posterior, moments


def pipeline_beliefs(recording: CasasRecording) -> tuple[np.ndarray, list[datetime]]:
    pipeline = BehaviouralSensingPipeline(
        recording.registry, config=PipelineConfig(formulation="phase3")
    )
    steps = scoring_steps(pipeline.run(recording.observations))
    return np.array([step.state.belief for step in steps]), [s.at for s in steps]


# ----------------------------------------------------------------------------
# The pipeline is the research recursion
# ----------------------------------------------------------------------------
class TestExactness:
    def test_the_pipeline_is_the_research_recursion(
        self,
        target: CasasRecording,
        fitted: tuple[FittedChannels, PeriodicStatePrior],
        formulation: p3.Phase3Formulation,
    ) -> None:
        posterior, moments = research_posterior(target, *fitted)
        beliefs, at = pipeline_beliefs(target)
        rows = min(len(moments), len(at))
        assert rows >= 600
        assert at[:rows] == moments[:rows]
        assert len({moment.hour for moment in moments[:rows]}) == 24
        np.testing.assert_allclose(beliefs[:rows], posterior[:rows], atol=1e-10)

    def test_a_state_the_training_homes_never_showed(
        self,
        target: CasasRecording,
        partial: tuple[FittedChannels, PeriodicStatePrior],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        channels, prior = partial
        unseen = {
            (name, state)
            for name, entries in channels.to_dict()["channels"].items()
            for state, entry in entries.items()
            if entry["silence"] is None
        }
        assert {state for _, state in unseen} == {BehaviouralState.BED_AWAKE.value}
        loaded = written(tmp_path, channels, prior)
        monkeypatch.setattr(
            p3.Phase3Formulation, "load", classmethod(lambda cls, *a, **k: loaded)
        )
        posterior, moments = research_posterior(target, channels, prior)
        beliefs, at = pipeline_beliefs(target)
        rows = min(len(moments), len(at))
        assert at[:rows] == moments[:rows]
        np.testing.assert_allclose(beliefs[:rows], posterior[:rows], atol=1e-10)

    def test_unseen_parameters_come_from_the_declared_count(
        self,
        target: CasasRecording,
        partial: tuple[FittedChannels, PeriodicStatePrior],
        tmp_path: Path,
    ) -> None:
        channels, prior = partial
        research = channels.models(target.registry, RESOLUTION, HURDLE, ONTOLOGY)
        emissions = written(tmp_path, channels, prior).emissions(target.registry)
        assert len(emissions) == len(research)
        for emission in emissions:
            model = research[
                next(c for c in research if f"channel:{c.name}" == emission.sensor_id)
            ]
            assert isinstance(model, HurdleChannel)
            for index, state in enumerate(STATES):
                assert emission.silence[state] == pytest.approx(
                    model.silence[index], rel=1e-15
                )
                assert emission.rate[state] == pytest.approx(
                    model.rate[index], rel=1e-15
                )

    def test_a_home_routes_only_its_instrumented_channels(
        self, target: CasasRecording, formulation: p3.Phase3Formulation
    ) -> None:
        keys = {e.sensor_id for e in formulation.emissions(target.registry)}
        assert keys == {
            "channel:bathroom_motion",
            "channel:bedroom_motion",
            "channel:hall_door",
            "channel:kitchen_motion",
            "channel:living_motion",
        }

    def test_it_starts_from_the_prior_one_window_before_the_first(
        self, target: CasasRecording, formulation: p3.Phase3Formulation
    ) -> None:
        pipeline = BehaviouralSensingPipeline(
            target.registry, config=PipelineConfig(formulation="phase3")
        )
        observations = sorted(target.observations, key=lambda o: o.timestamp)
        first = observations[0].timestamp
        pipeline.run(observations[:1], until=first)
        expected = MultimodalBayesFilter(
            formulation.ontology(),
            formulation.emissions(target.registry),
            target.registry,
        )
        expected.reset(formulation.start(first), at=first - STEP)
        estimate = expected.update(first, observations[:1])
        assert pipeline.filter.at == first
        np.testing.assert_allclose(pipeline.filter.belief, estimate.belief, atol=1e-15)


# ----------------------------------------------------------------------------
# The channel emission and the filter's routing
# ----------------------------------------------------------------------------
def _channel(**overrides: Any) -> HurdleChannelEmission:
    values: dict[str, Any] = {
        "sensors": ("m1", "m2"),
        "silence": {s: 0.2 + 0.1 * i for i, s in enumerate(STATES)},
        "rate": {s: 0.5 + 0.3 * i for i, s in enumerate(STATES)},
        "modality": Modality.MOTION,
    }
    values.update(overrides)
    return HurdleChannelEmission("channel:kitchen_motion", **values)


def _event(sensor: str, at: datetime, value: float = 1.0) -> Observation:
    return Observation(at, sensor, Modality.MOTION, ObservationKind.EVENT, value)


T0 = datetime(2024, 3, 4, 12, 0, tzinfo=UTC)


class TestChannelEmission:
    def test_it_is_the_research_hurdle(self) -> None:
        emission = _channel()
        research = HurdleChannel(
            EvidenceChannel("kitchen", Modality.MOTION),
            np.array([emission.silence[s] for s in STATES]),
            np.array([emission.rate[s] for s in STATES]),
        )
        for count in range(6):
            records = [_event("m1", T0)] * count
            got = emission._raw_log_likelihood(ONTOLOGY, records, STEP)
            np.testing.assert_allclose(
                got, research.loglik(np.array([float(count)]))[0], atol=1e-15
            )

    def test_only_records_with_a_value_are_activations(self) -> None:
        emission = _channel()
        quiet = [_event("m1", T0, 0.0), _event("m2", T0, 0.0)]
        np.testing.assert_array_equal(
            emission._raw_log_likelihood(ONTOLOGY, quiet, STEP),
            emission._raw_log_likelihood(ONTOLOGY, [], STEP),
        )

    def test_it_scores_only_whole_windows(self) -> None:
        emission = _channel()
        np.testing.assert_array_equal(
            emission.log_likelihood(ONTOLOGY, [], timedelta(0)), np.zeros(len(STATES))
        )
        with pytest.raises(ValueError, match="per window of 300 s"):
            emission.log_likelihood(ONTOLOGY, [_event("m1", T0)], timedelta(0))
        with pytest.raises(ValueError, match="advanced 600 s"):
            emission.log_likelihood(ONTOLOGY, [], timedelta(minutes=10))

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"sensors": ()}, "distinct sensors"),
            ({"sensors": ("m1", "m1")}, "distinct sensors"),
            ({"sensors": ("channel:kitchen_motion",)}, "must not be one of its"),
            ({"silence": {STATES[0]: 0.5}}, "same states"),
            ({"silence": {s: 1.0 for s in STATES}}, "strictly in"),
            ({"rate": {s: 0.0 for s in STATES}}, "positive and finite"),
            ({"window": timedelta(0)}, "window must be positive"),
        ],
    )
    def test_it_refuses_invalid_parameters(
        self, overrides: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _channel(**overrides)

    def test_it_refuses_an_ontology_with_a_state_it_lacks(self) -> None:
        emission = _channel(
            silence={s: 0.5 for s in STATES[:-1]}, rate={s: 1.0 for s in STATES[:-1]}
        )
        with pytest.raises(ValueError, match="no parameter for"):
            emission.log_likelihood(ONTOLOGY, [], STEP)


class TestRouting:
    def test_a_channel_reads_every_sensor_it_pools(self) -> None:
        emission = _channel()
        filt = MultimodalBayesFilter(ONTOLOGY, [emission])
        filt.reset(at=T0)
        records = [
            _event("m1", T0 + timedelta(minutes=1)),
            _event("m2", T0 + timedelta(minutes=2)),
            _event("m2", T0 + timedelta(minutes=3)),
            _event("other", T0 + timedelta(minutes=4)),
        ]
        terms = filt.evidence_terms(T0 + STEP, records)
        assert terms.observations == {"channel:kitchen_motion": 3}
        raw = emission._raw_log_likelihood(ONTOLOGY, records[:3], STEP)
        np.testing.assert_allclose(
            terms.likelihoods["channel:kitchen_motion"], raw - raw.max(), atol=1e-15
        )

    def test_a_sensor_read_by_two_models_is_refused(self) -> None:
        other = _channel(sensors=("m2", "m3"))
        other.sensor_id = "channel:other"
        with pytest.raises(ValueError, match="'m2' is read by more than one"):
            MultimodalBayesFilter(ONTOLOGY, [_channel(), other])

    def test_a_channel_reports_its_modality(self) -> None:
        filt = MultimodalBayesFilter(ONTOLOGY, [_channel()], SensorRegistry())
        filt.reset(at=T0)
        estimate = filt.update(T0 + STEP, [_event("m1", T0 + STEP)])
        assert [c.modality for c in estimate.evidence] == [Modality.MOTION]

    def test_reset_sets_the_clock(self) -> None:
        started = MultimodalBayesFilter(ONTOLOGY, [_channel()])
        started.reset(at=T0)
        assert started.evidence_terms(T0 + STEP).elapsed == STEP
        cleared = MultimodalBayesFilter(ONTOLOGY, [_channel()])
        cleared.reset(at=T0)
        cleared.reset()
        assert started.at == T0 and cleared.at is None
        with pytest.raises(ValueError):
            cleared.reset(at=datetime(2024, 3, 4, 12, 0))


# ----------------------------------------------------------------------------
# What is refused
# ----------------------------------------------------------------------------
class TestRefusals:
    def test_a_step_other_than_the_fitted_window(self) -> None:
        with pytest.raises(ValueError, match="step must be that window"):
            PipelineConfig(formulation="phase3", step=timedelta(minutes=10))

    def test_an_unknown_formulation(self) -> None:
        with pytest.raises(ValueError, match="formulation must be one of"):
            PipelineConfig(formulation="better")

    def test_emissions_or_an_ontology_of_its_own(
        self, target: CasasRecording, formulation: p3.Phase3Formulation
    ) -> None:
        config = PipelineConfig(formulation="phase3")
        with pytest.raises(ValueError, match="brings its own emission models"):
            BehaviouralSensingPipeline(
                target.registry,
                emissions=default_emissions(target.registry, ONTOLOGY),
                config=config,
            )
        circadian = formulation.ontology()
        with pytest.raises(ValueError, match="default ontology"):
            BehaviouralSensingPipeline(target.registry, circadian, config=config)

    def test_a_home_with_no_routable_channel(
        self, formulation: p3.Phase3Formulation
    ) -> None:
        registry = SensorRegistry.from_specs(
            [
                SensorSpec(
                    "lounge_pir", Modality.MOTION, ObservationKind.EVENT, room="lounge"
                ),
                SensorSpec(
                    "bed", Modality.BED_PRESSURE, ObservationKind.STATE, room="bedroom"
                ),
            ]
        )
        with pytest.raises(ValueError, match="routes to a channel"):
            BehaviouralSensingPipeline(
                registry, config=PipelineConfig(formulation="phase3")
            )

    def test_a_missing_parameter_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(FileNotFoundError):
            p3.Phase3Formulation.load(tmp_path / "absent.json")
        monkeypatch.setattr(p3, "PARAMETERS_FILE", "absent.json")
        with pytest.raises(FileNotFoundError, match="is not installed"):
            p3.Phase3Formulation.load()

    def test_parameters_that_are_not_the_confirmed_ones(
        self, fitted: tuple[FittedChannels, PeriodicStatePrior], tmp_path: Path
    ) -> None:
        channels, prior = fitted
        payload = p3.parameters_payload(channels, prior, {})
        path = p3.write_parameters(payload, tmp_path / "p.json")
        with pytest.raises(ValueError, match="channel parameters are not"):
            p3.Phase3Formulation.load(path)
        digests = {
            "channels_sha256": channels.sha256(),
            "periodic_prior_sha256": prior.sha256(),
        }
        p3.Phase3Formulation.load(path, **digests)

        changed = json.loads(path.read_text(encoding="utf-8"))
        entry = changed["channels"]["channels"]["kitchen_motion"]["sleeping"]
        entry["silence"] = entry["silence"] * 0.999
        p3.write_parameters(changed, path)
        with pytest.raises(ValueError, match="channel parameters are not"):
            p3.Phase3Formulation.load(path, **digests)

        changed = dict(payload)
        changed["periodic_prior"] = dict(payload["periodic_prior"])
        changed["periodic_prior"]["population"] = (
            np.array(payload["periodic_prior"]["population"]) + 1e-6
        ).tolist()
        p3.write_parameters(changed, path)
        with pytest.raises(ValueError, match="time prior is not"):
            p3.Phase3Formulation.load(path, **digests)

        for key, value, message in (
            ("schema", "other/1", "not a phase3-parameters"),
            ("step_seconds", 600, "five-minute window"),
        ):
            p3.write_parameters({**payload, key: value}, path)
            with pytest.raises(ValueError, match=message):
                p3.Phase3Formulation.load(path, **digests)


# ----------------------------------------------------------------------------
# Defaults, and the parameters the package ships
# ----------------------------------------------------------------------------
class TestDefaults:
    def test_the_default_is_the_declared_formulation(
        self, target: CasasRecording
    ) -> None:
        assert PipelineConfig().formulation == "declared"
        default = BehaviouralSensingPipeline(target.registry)
        explicit = BehaviouralSensingPipeline(
            target.registry, config=PipelineConfig(formulation="declared")
        )
        observations = target.observations[:2000]
        a = scoring_steps(default.run(observations))
        b = scoring_steps(explicit.run(observations))
        assert len(a) == len(b) > 0
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x.state.belief, y.state.belief)
        assert all(not k.startswith("channel:") for k in default.filter.emissions)


class TestShippedParameters:
    def test_the_expected_digests_are_the_heldout_records(self) -> None:
        record = json.loads((ROOT / p3.HELDOUT_RECORD).read_text(encoding="utf-8"))
        training = record["results"]["training"]
        assert training["channels_sha256"] == p3.CHANNELS_SHA256
        assert training["periodic_prior_sha256"] == p3.PERIODIC_PRIOR_SHA256

    @pytest.mark.skipif(not SHIPPED.exists(), reason="parameters not yet shipped")
    def test_the_shipped_parameters_are_the_heldout_fit(self) -> None:
        record = json.loads((ROOT / p3.HELDOUT_RECORD).read_text(encoding="utf-8"))
        training = record["results"]["training"]
        payload = json.loads(SHIPPED.read_text(encoding="utf-8"))
        assert p3.canonical_sha256(payload["channels"]) == training["channels_sha256"]
        prior = PeriodicStatePrior.from_dict(payload["periodic_prior"])
        assert prior.sha256() == training["periodic_prior_sha256"]
        assert payload["channels"]["fitted_on"] == training["homes"]
        formulation = p3.Phase3Formulation.load()
        assert formulation.prior.fitted_on == tuple(training["homes"])
