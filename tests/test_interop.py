"""Tests for the FHIR-style export.

The property under test throughout is that a downstream reader can always
tell a measurement from an inference. An export that blurs the two is how a
research prototype ends up quoted as a clinical fact.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sensor_modeling.alerts import Alert, AlertKind, AlertSeverity
from sensor_modeling.fusion.estimate import StateEstimate
from sensor_modeling.interop import (
    CODE_SYSTEM,
    PROVENANCE_DERIVED_FEATURE,
    PROVENANCE_INFERRED,
    PROVENANCE_MEASURED,
    alert_resource,
    bundle,
    measured_only,
    observation_resource,
    state_resource,
    summarise_provenance,
)
from sensor_modeling.observations import (
    Modality,
    Observation,
    ObservationFlag,
    ObservationKind,
    SensorRegistry,
    SensorSpec,
    Unit,
)
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

T0 = datetime(2024, 5, 1, 8, 0, tzinfo=timezone.utc)
ONTOLOGY = StateOntology()


def registry() -> SensorRegistry:
    """A tiny deployment with a measured and a derived sensor."""
    return SensorRegistry.from_specs(
        [
            SensorSpec(
                "kitchen_motion",
                Modality.MOTION,
                room="kitchen",
                description="PIR covering the kitchen.",
            ),
            SensorSpec(
                "living_radar",
                Modality.RADAR,
                kind=ObservationKind.SAMPLE,
                unit=Unit.COUNT,
                room="living",
                expected_interval=timedelta(minutes=1),
                description="Derived mmWave track count.",
            ),
        ]
    )


def motion() -> Observation:
    """A directly measured activation."""
    return Observation(
        T0, "kitchen_motion", Modality.MOTION, ObservationKind.EVENT, 1.0
    )


def radar() -> Observation:
    """A derived feature with its own confidence."""
    return Observation(
        T0,
        "living_radar",
        Modality.RADAR,
        ObservationKind.SAMPLE,
        2.0,
        unit=Unit.COUNT,
        confidence=0.75,
    )


def estimate(*, confidence: float = 0.9, completeness: float = 1.0) -> StateEstimate:
    """An inferred state."""
    belief = np.full(ONTOLOGY.size, (1.0 - confidence) / (ONTOLOGY.size - 1))
    belief[ONTOLOGY.index(S.KITCHEN_ACTIVITY)] = confidence
    return StateEstimate(
        at=T0,
        ontology=ONTOLOGY,
        belief=belief,
        evidence=(),
        completeness=completeness,
        min_confidence=0.35,
        min_completeness=0.25,
    )


def alert(kind: AlertKind = AlertKind.BEHAVIOURAL_CHANGE) -> Alert:
    """An algorithmic alert."""
    return Alert(
        at=T0,
        kind=kind,
        severity=AlertSeverity.ATTENTION,
        subject="sleeping_hours",
        summary="Sustained decrease in sleeping_hours",
        score=0.5,
        confidence=0.8,
        evidence={"coverage": 0.9},
        caveats=("sensor coverage was 90% over the period",),
    )


def provenance_of(resource: dict) -> str:
    """Return the provenance kind recorded on a resource."""
    for extension in resource.get("extension", []):
        if extension.get("url") == f"{CODE_SYSTEM}/provenance":
            for inner in extension["extension"]:
                if inner.get("url") == "kind":
                    return str(inner["valueCode"])
    raise AssertionError("every exported resource must record its provenance")


class TestMeasurementsAndFeatures:
    def test_a_direct_reading_is_exported_as_measured(self) -> None:
        resource = observation_resource(motion(), registry())
        assert resource["resourceType"] == "Observation"
        assert resource["status"] == "final"
        assert provenance_of(resource) == PROVENANCE_MEASURED

    def test_a_derived_feature_is_not_exported_as_a_measurement(self) -> None:
        """A radar track count is an estimate, and must say so."""
        resource = observation_resource(radar(), registry())
        assert provenance_of(resource) == PROVENANCE_DERIVED_FEATURE

    def test_quality_and_confidence_are_carried_through(self) -> None:
        resource = observation_resource(radar(), registry())
        quality = next(
            ext
            for ext in resource["extension"]
            if ext["url"] == f"{CODE_SYSTEM}/quality"
        )
        values = {inner["url"]: inner for inner in quality["extension"]}
        assert values["confidence"]["valueDecimal"] == pytest.approx(0.75)
        assert values["kind"]["valueCode"] == "sample"

    def test_the_declared_semantics_reach_the_export(self) -> None:
        resource = observation_resource(motion(), registry())
        assert "PIR covering the kitchen" in resource["code"]["text"]
        assert resource["bodySite"]["text"] == "kitchen"

    def test_event_semantics_are_stated_explicitly(self) -> None:
        """A reader must not fill an event stream's gaps with zeros."""
        resource = observation_resource(motion(), registry())
        note = next(
            ext
            for ext in resource["extension"]
            if ext["url"] == f"{CODE_SYSTEM}/event-semantics"
        )
        assert "absence of evidence" in note["valueString"]

    def test_ingestion_repairs_are_disclosed(self) -> None:
        repaired = motion().with_flags(ObservationFlag.CLOCK_ADJUSTED)
        resource = observation_resource(repaired, registry())
        flags = next(
            ext
            for ext in resource["extension"]
            if ext["url"] == f"{CODE_SYSTEM}/ingestion-flags"
        )
        assert "clock_adjusted" in flags["valueString"]

    def test_an_unregistered_sensor_still_exports(self) -> None:
        assert observation_resource(motion())["resourceType"] == "Observation"


class TestInferredStates:
    def test_an_inference_is_never_marked_final(self) -> None:
        resource = state_resource(estimate())
        assert resource["status"] == "preliminary"
        assert provenance_of(resource) == PROVENANCE_INFERRED

    def test_an_inference_states_its_method_and_carries_a_disclaimer(self) -> None:
        resource = state_resource(estimate())
        assert "Bayesian" in resource["method"]["text"]
        assert "not a diagnosis" in resource["note"][0]["text"]

    def test_the_whole_posterior_is_exported_not_only_the_winner(self) -> None:
        resource = state_resource(estimate())
        components = {
            component["code"]["text"]: component["valueQuantity"]["value"]
            for component in resource["component"]
        }
        assert set(components) == set(ONTOLOGY.labels())
        assert sum(components.values()) == pytest.approx(1.0)

    def test_an_abstention_exports_a_data_absent_reason_not_a_value(self) -> None:
        """`unknown` must not be exportable as a behavioural finding."""
        resource = state_resource(estimate(confidence=0.2, completeness=0.1))
        assert "valueCodeableConcept" not in resource
        assert resource["dataAbsentReason"]["coding"][0]["code"] == (
            "insufficient-evidence"
        )

    def test_inference_quality_is_attached(self) -> None:
        resource = state_resource(estimate(completeness=0.6))
        quality = next(
            ext
            for ext in resource["extension"]
            if ext["url"] == f"{CODE_SYSTEM}/inference-quality"
        )
        values = {inner["url"]: inner for inner in quality["extension"]}
        assert values["completeness"]["valueDecimal"] == pytest.approx(0.6)
        assert values["abstained"]["valueBoolean"] is False

    def test_attribution_is_exported_when_supplied(self) -> None:
        resource = state_resource(estimate(), attribution=0.4)
        attribution = next(
            ext
            for ext in resource["extension"]
            if ext["url"] == f"{CODE_SYSTEM}/attribution"
        )
        values = {inner["url"]: inner for inner in attribution["extension"]}
        assert values["residentProbability"]["valueDecimal"] == pytest.approx(0.4)


class TestAlerts:
    def test_an_alert_is_not_an_observation(self) -> None:
        """An alert is a judgement, not a record of anything observed."""
        assert alert_resource(alert())["resourceType"] == "DetectedIssue"

    def test_alert_severity_maps_to_the_fhir_scale(self) -> None:
        assert alert_resource(alert())["severity"] == "moderate"

    def test_caveats_survive_the_export(self) -> None:
        resource = alert_resource(alert())
        assert any(
            "coverage" in item["action"]["text"] for item in resource["mitigation"]
        )

    def test_a_system_health_alert_is_not_attached_to_a_patient(self) -> None:
        """A failing sensor is equipment maintenance, not a clinical finding."""
        resource = alert_resource(alert(AlertKind.SYSTEM_HEALTH))
        assert "patient" not in resource
        assert resource["code"]["coding"][0]["code"] == "alert.system-health"

    def test_a_behavioural_alert_is_attached_to_the_subject(self) -> None:
        resource = alert_resource(alert(), subject="Patient/abc")
        assert resource["patient"]["reference"] == "Patient/abc"


class TestBundle:
    @staticmethod
    def full() -> dict:
        return bundle(
            observations=[motion(), radar()],
            estimates=[estimate()],
            alerts=[alert()],
            registry=registry(),
        )

    def test_a_bundle_collects_every_kind(self) -> None:
        payload = self.full()
        kinds = [entry["resource"]["resourceType"] for entry in payload["entry"]]
        assert kinds.count("Observation") == 3
        assert kinds.count("DetectedIssue") == 1

    def test_provenance_is_countable_in_one_call(self) -> None:
        counts = summarise_provenance(self.full())
        assert counts[PROVENANCE_MEASURED] == 1
        assert counts[PROVENANCE_DERIVED_FEATURE] == 1
        assert counts[PROVENANCE_INFERRED] == 2

    def test_measured_only_excludes_every_inference(self) -> None:
        """The guarantee a consumer most needs: give me only measurements."""
        measured = measured_only(self.full())
        assert len(measured) == 1
        assert measured[0]["device"]["display"] == "kitchen_motion"

    def test_the_bundle_is_tagged_as_a_research_prototype(self) -> None:
        tag = self.full()["meta"]["tag"][0]
        assert tag["code"] == "research-prototype"
        assert "not clinically validated" in tag["display"]

    def test_the_bundle_is_json_serialisable(self) -> None:
        payload = json.loads(json.dumps(self.full()))
        assert payload["resourceType"] == "Bundle"

    def test_an_empty_bundle_is_valid(self) -> None:
        payload = bundle()
        assert payload["entry"] == []
        assert summarise_provenance(payload) == {}
