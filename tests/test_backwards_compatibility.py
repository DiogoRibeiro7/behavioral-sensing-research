"""The original toolkit must keep working unchanged.

The multimodal layer is additive. Every one of these tests exercises an API
that existed before it, in the way it was used before it, and would fail if
the newer work had quietly changed behaviour underneath.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

import sensor_modeling
from sensor_modeling.analysis import AnalysisPipeline, calculate_behavioral_metrics
from sensor_modeling.change_point import EmbeddingCPD
from sensor_modeling.hmm import HierarchicalHMM
from sensor_modeling.models.bernoulli_ar.base_model import BernoulliAutoregressiveModel
from sensor_modeling.models.change_point_detection.pelt import PELTChangePointDetector
from sensor_modeling.models.nhpp_pelt.model import NHPPPELT, NHPPConfig
from sensor_modeling.observations import (
    Modality,
    ObservationKind,
    ObservationStream,
    SensorRegistry,
    SensorSpec,
    observations_from_dataset,
)
from sensor_modeling.utils import SensorDataset, simulate_sensor_data


@pytest.fixture(scope="module")
def dataset() -> SensorDataset:
    """The legacy simulated dataset, generated exactly as before."""
    return simulate_sensor_data(n_days=3, n_sensors=3, seed=42)


class TestOriginalModels:
    def test_bernoulli_autoregressive_still_fits(self, dataset: SensorDataset) -> None:
        model = BernoulliAutoregressiveModel(
            list(dataset.data.columns), dataset.data.columns[0]
        )
        result = model.fit(dataset)
        assert isinstance(result, dict)

    def test_nhpp_pelt_still_fits(self, dataset: SensorDataset) -> None:
        model = NHPPPELT(NHPPConfig(n_basis=4, min_seg_len=1))
        model.fit(dataset, sensor=dataset.data.columns[0])
        assert hasattr(model, "changepoints_")

    def test_the_nhpp_arm_of_the_pipeline_actually_runs(
        self, dataset: SensorDataset
    ) -> None:
        """Regression: the default config violated the model's own constraint.

        `n_basis=3` is below `degree+1` for the cubic default, so every fit
        raised and the pipeline recorded an error rather than a result. The
        arm appeared present in the output and had never once worked.
        """
        results = AnalysisPipeline().run(dataset)
        assert "error" not in results["nhpp"]
        assert "changepoints" in results["nhpp"]

    def test_the_hmm_variants_still_fit_and_predict(
        self, dataset: SensorDataset
    ) -> None:
        model = HierarchicalHMM(n_states=3, random_state=0)
        values = dataset.data.to_numpy(dtype=float)
        model.fit(values)
        states = model.predict(values)
        assert states.shape[0] == values.shape[0]

    def test_pelt_still_segments(self) -> None:
        signal = np.concatenate([np.zeros(40), np.ones(40) * 5.0])
        found = PELTChangePointDetector(penalty=1.0).detect(signal)
        assert found and 30 <= found[0] <= 50

    def test_the_lightweight_change_point_detectors_still_run(self) -> None:
        detector = EmbeddingCPD(window=5)
        detector.fit(np.concatenate([np.zeros(30), np.ones(30)]))
        assert detector.predict() is not None

    def test_the_analysis_pipeline_still_runs(self, dataset: SensorDataset) -> None:
        results = AnalysisPipeline().run(dataset)
        assert set(results) == {"ar", "hmm", "cpd", "nhpp"}

    def test_behavioural_metrics_still_compute(self, dataset: SensorDataset) -> None:
        metrics = calculate_behavioral_metrics(dataset.data)
        assert isinstance(metrics, dict)


class TestOriginalDataLayer:
    def test_sensor_dataset_keeps_its_shape(self, dataset: SensorDataset) -> None:
        assert dataset.data.shape == (3 * 96, 3)
        assert dataset.to_dataframe() is dataset.data

    def test_simulation_is_still_reproducible(self) -> None:
        first = simulate_sensor_data(n_days=1, n_sensors=2, seed=7)
        second = simulate_sensor_data(n_days=1, n_sensors=2, seed=7)
        assert first.data.equals(second.data)

    def test_event_sequences_still_extract(self, dataset: SensorDataset) -> None:
        events = dataset.to_event_sequences(dataset.data.columns[0])
        assert isinstance(events, list)


class TestPublicSurface:
    def test_the_original_subpackages_are_still_advertised(self) -> None:
        for name in (
            "models",
            "analysis",
            "utils",
            "change_point",
            "hmm",
            "data",
            "visualization",
        ):
            assert name in sensor_modeling.__all__

    def test_the_version_is_still_exposed(self) -> None:
        assert sensor_modeling.__version__


class TestBridgeBetweenLayers:
    def test_a_legacy_dataset_reaches_the_new_pipeline(
        self, dataset: SensorDataset
    ) -> None:
        """The additive claim, demonstrated end to end."""
        registry = SensorRegistry.from_specs(
            [
                SensorSpec(name, Modality.MOTION, room=f"room_{index}")
                for index, name in enumerate(dataset.data.columns)
            ]
        )
        observations = list(
            observations_from_dataset(dataset, registry, tz=None)
            if dataset.data.index.tz is not None
            else observations_from_dataset(
                dataset, registry, tz=__import__("datetime").timezone.utc
            )
        )
        assert observations
        assert all(obs.modality is Modality.MOTION for obs in observations)

    def test_new_observations_frame_back_into_the_legacy_shape(
        self, dataset: SensorDataset
    ) -> None:
        from datetime import timezone

        registry = SensorRegistry.from_specs(
            [SensorSpec(name, Modality.MOTION) for name in dataset.data.columns]
        )
        stream = ObservationStream.from_observations(
            observations_from_dataset(dataset, registry, tz=timezone.utc)
        )
        frame = stream.event_counts("15min", sensor_ids=list(dataset.data.columns))
        assert list(frame.columns) == list(dataset.data.columns)
        assert frame.index.tz is not None

    def test_the_two_layers_disagree_about_nothing_they_share(
        self, dataset: SensorDataset
    ) -> None:
        """Total activations must survive the round trip."""
        from datetime import timezone

        registry = SensorRegistry.from_specs(
            [SensorSpec(name, Modality.MOTION) for name in dataset.data.columns]
        )
        stream = ObservationStream.from_observations(
            observations_from_dataset(dataset, registry, tz=timezone.utc)
        )
        counts = stream.event_counts("15min", sensor_ids=list(dataset.data.columns))
        for column in dataset.data.columns:
            assert counts[column].sum() == dataset.data[column].sum()

    def test_a_state_sensor_declaration_changes_only_the_framing(self) -> None:
        """The same records framed as states rather than events."""
        from datetime import timezone

        data = simulate_sensor_data(n_days=1, n_sensors=1, seed=3)
        column = data.data.columns[0]
        registry = SensorRegistry.from_specs(
            [SensorSpec(column, Modality.BED_PRESSURE, kind=ObservationKind.STATE)]
        )
        stream = ObservationStream.from_observations(
            observations_from_dataset(data, registry, tz=timezone.utc)
        )
        states = stream.state_frame("15min", max_hold=timedelta(hours=1))
        assert column in states.columns
