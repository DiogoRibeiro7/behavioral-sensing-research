"""FHIR-style export that keeps measurement and inference distinguishable.

The hazard this module exists to avoid is specific. Once a behavioural
conclusion is written into a clinical record it looks like every other entry
there, and a downstream reader has no way to tell that ``sleeping`` was
inferred by a Markov filter from a bed sensor and a wearable rather than
measured. Exporting inferred states as though they were observations is how a
research prototype ends up quoted as a clinical fact.

So the four kinds this platform keeps separate stay separate on the way out:

.. code-block:: text

    measured observation  ->  Observation, status "final",
                              derived from a physical Device
    derived feature       ->  Observation, status "final", but carrying the
                              upstream device's own confidence
    inferred state        ->  Observation, status "preliminary", with an
                              explicit method, the full posterior as
                              components, and derivedFrom provenance
    algorithmic alert     ->  DetectedIssue, never an Observation

Every exported resource carries provenance: what produced it, from what, when,
and with what quality. Inferred resources additionally carry the coverage and
attribution behind them, and a note recording that they are algorithmic
output from a research toolkit.

.. warning::

   This is a **FHIR-style** export for interoperability prototyping. It is not
   a validated FHIR profile, it has not been conformance-tested against a FHIR
   server, and none of the codes below come from a recognised terminology. Do
   not present its output as clinically validated data.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from ..alerts.alert import Alert, AlertKind, AlertSeverity
from ..fusion.estimate import StateEstimate
from ..observations.observation import Observation
from ..observations.registry import SensorRegistry
from ..observations.types import ObservationKind

logger = logging.getLogger(__name__)

#: Local code system for this toolkit's own vocabulary. Deliberately a
#: project URI rather than a borrowed clinical one: these codes have no
#: standardised meaning and must not be mistaken for LOINC or SNOMED.
CODE_SYSTEM = "https://github.com/DiogoRibeiro7/behavioral-sensing-research/codes"

#: Fixed disclaimer attached to every inferred resource.
RESEARCH_NOTE = (
    "Algorithmically inferred from ambient sensor data by the sensor-modeling "
    "research toolkit. Not a measured clinical observation, not a diagnosis, "
    "and not the output of a medical device."
)

#: How an exported resource was produced.
PROVENANCE_MEASURED = "measured"
PROVENANCE_DERIVED_FEATURE = "derived-feature"
PROVENANCE_INFERRED = "inferred"

_SEVERITY_TO_FHIR = {
    AlertSeverity.INFORMATION: "low",
    AlertSeverity.ATTENTION: "moderate",
    AlertSeverity.URGENT: "high",
}


def _instant(moment: datetime) -> str:
    """Render a timezone-aware instant in the FHIR ``instant`` format."""
    return moment.isoformat()


def _coding(code: str, display: str) -> dict[str, Any]:
    """Build a CodeableConcept in this toolkit's own code system."""
    return {
        "coding": [{"system": CODE_SYSTEM, "code": code, "display": display}],
        "text": display,
    }


def _provenance_extension(kind: str, detail: str) -> dict[str, Any]:
    """Build the extension that records how a resource was produced.

    This is the load-bearing piece of the export. A reader that understands
    nothing else about these resources can still read this and know whether
    they are looking at a measurement or at a conclusion.
    """
    return {
        "url": f"{CODE_SYSTEM}/provenance",
        "extension": [
            {"url": "kind", "valueCode": kind},
            {"url": "detail", "valueString": detail},
        ],
    }


def observation_resource(
    observation: Observation, registry: SensorRegistry | None = None
) -> dict[str, Any]:
    """Export one sensor observation as a FHIR-style ``Observation``.

    A record whose ``confidence`` is below one is exported as a *derived
    feature* rather than a measurement, because something upstream computed
    it and attached its own uncertainty.
    """
    spec = registry.get(observation.sensor_id) if registry is not None else None
    derived = observation.confidence < 1.0
    kind = PROVENANCE_DERIVED_FEATURE if derived else PROVENANCE_MEASURED
    description = (
        spec.description
        if spec is not None and spec.description
        else f"{observation.modality.value} sensor reading"
    )

    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "id": f"obs-{observation.sensor_id}-{int(observation.timestamp.timestamp())}",
        "status": "final",
        "category": [_coding("device-measurement", "Device measurement")],
        "code": _coding(
            f"sensor.{observation.modality.value}",
            description,
        ),
        "effectiveDateTime": _instant(observation.timestamp),
        "valueQuantity": {
            "value": observation.value,
            "unit": observation.unit.value,
            "system": f"{CODE_SYSTEM}/units",
            "code": observation.unit.value,
        },
        "device": {"display": observation.sensor_id},
        "extension": [
            _provenance_extension(
                kind,
                (
                    "Feature computed by the reporting device; the value is an "
                    "estimate, not a direct measurement."
                    if derived
                    else "Direct sensor measurement."
                ),
            ),
            {
                "url": f"{CODE_SYSTEM}/quality",
                "extension": [
                    {"url": "quality", "valueDecimal": observation.quality},
                    {"url": "confidence", "valueDecimal": observation.confidence},
                    {"url": "kind", "valueCode": observation.kind.value},
                    {"url": "source", "valueString": observation.source},
                ],
            },
        ],
    }
    if spec is not None and spec.room:
        resource["bodySite"] = _coding(f"room.{spec.room}", spec.room)
    if observation.flags:
        resource["extension"].append(
            {
                "url": f"{CODE_SYSTEM}/ingestion-flags",
                "valueString": ",".join(
                    sorted(flag.value for flag in observation.flags)
                ),
            }
        )
    # An event stream's silence is not a zero, so the export says so rather
    # than leaving a reader to infer a value for the gaps.
    if observation.kind is ObservationKind.EVENT:
        resource["extension"].append(
            {
                "url": f"{CODE_SYSTEM}/event-semantics",
                "valueString": (
                    "Event stream: absence of a record is absence of evidence, "
                    "not an observation of zero."
                ),
            }
        )
    return resource


def state_resource(
    estimate: StateEstimate,
    *,
    subject: str = "Patient/example",
    attribution: float | None = None,
) -> dict[str, Any]:
    """Export an inferred behavioural state as a FHIR-style ``Observation``.

    Marked ``preliminary`` rather than ``final``, given an explicit ``method``
    describing the inference, and accompanied by the whole posterior as
    components. A reader is never shown only the winning label.

    An abstaining estimate exports a ``dataAbsentReason`` instead of a value,
    which is the correct FHIR idiom for "we do not know" and stops
    ``unknown`` from being read as a behavioural finding.
    """
    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "id": f"state-{int(estimate.at.timestamp())}",
        "status": "preliminary",
        "category": [_coding("activity", "Activity")],
        "code": _coding("behavioural-state", "Inferred behavioural state"),
        "subject": {"reference": subject},
        "effectiveDateTime": _instant(estimate.at),
        "method": _coding(
            "multimodal-bayes-filter",
            "Recursive Bayesian fusion of multimodal ambient sensor evidence",
        ),
        "note": [{"text": RESEARCH_NOTE}],
        "extension": [
            _provenance_extension(
                PROVENANCE_INFERRED,
                "Posterior over latent behavioural states; not a measurement.",
            ),
            {
                "url": f"{CODE_SYSTEM}/inference-quality",
                "extension": [
                    {"url": "confidence", "valueDecimal": estimate.confidence},
                    {"url": "completeness", "valueDecimal": estimate.completeness},
                    {
                        "url": "normalisedEntropy",
                        "valueDecimal": estimate.normalised_entropy,
                    },
                    {"url": "abstained", "valueBoolean": estimate.abstained},
                ],
            },
        ],
        "component": [
            {
                "code": _coding(f"state.{state.value}", state.value),
                "valueQuantity": {"value": probability, "unit": "probability"},
            }
            for state, probability in estimate.probabilities.items()
        ],
    }

    if estimate.abstained:
        resource["dataAbsentReason"] = _coding(
            "insufficient-evidence",
            "Insufficient or unreliable sensor evidence to infer a state",
        )
    else:
        resource["valueCodeableConcept"] = _coding(
            f"state.{estimate.state.value}", estimate.state.value
        )

    if attribution is not None:
        resource["extension"].append(
            {
                "url": f"{CODE_SYSTEM}/attribution",
                "extension": [
                    {"url": "residentProbability", "valueDecimal": attribution},
                    {
                        "url": "note",
                        "valueString": (
                            "Probability that the monitored resident, rather "
                            "than another person present, generated the "
                            "ambient evidence behind this estimate."
                        ),
                    },
                ],
            }
        )

    contributing = [c.sensor_id for c in estimate.evidence if c.informative]
    if contributing:
        resource["derivedFrom"] = [
            {"display": sensor_id} for sensor_id in sorted(contributing)
        ]
    if estimate.missing:
        resource["extension"].append(
            {
                "url": f"{CODE_SYSTEM}/absent-evidence",
                "valueString": ",".join(sorted(estimate.missing)),
            }
        )
    return resource


def alert_resource(alert: Alert, *, subject: str = "Patient/example") -> dict[str, Any]:
    """Export an alert as a FHIR-style ``DetectedIssue``.

    Deliberately **not** an ``Observation``. An alert is an algorithmic
    judgement that something warrants attention, not a record of anything
    being observed, and giving it the same resource type as a measurement is
    precisely the conflation this module exists to prevent.
    """
    resource: dict[str, Any] = {
        "resourceType": "DetectedIssue",
        "id": f"alert-{alert.identifier}",
        "status": "preliminary",
        "severity": _SEVERITY_TO_FHIR[alert.severity],
        "code": _coding(f"alert.{alert.kind.value}", alert.kind.value),
        "identifiedDateTime": _instant(alert.at),
        "detail": alert.summary,
        "author": {"display": "sensor-modeling research toolkit"},
        "extension": [
            _provenance_extension(
                PROVENANCE_INFERRED,
                "Algorithmic alert derived from inferred behaviour.",
            ),
            {
                "url": f"{CODE_SYSTEM}/alert-grading",
                "extension": [
                    {"url": "score", "valueDecimal": alert.score},
                    {"url": "confidence", "valueDecimal": alert.confidence},
                    {"url": "subjectOfAlert", "valueString": alert.subject},
                ],
            },
        ],
        "note": [{"text": RESEARCH_NOTE}],
    }

    if alert.kind is AlertKind.SYSTEM_HEALTH:
        # A failing sensor is an equipment issue. Attaching it to a patient
        # would make a maintenance problem look like a clinical finding.
        resource["code"] = _coding("alert.system-health", "Sensing system health")
    else:
        resource["patient"] = {"reference": subject}

    if alert.caveats:
        resource["mitigation"] = [
            {"action": _coding("caveat", caveat)} for caveat in alert.caveats
        ]
    return resource


def bundle(
    *,
    observations: Iterable[Observation] = (),
    estimates: Iterable[StateEstimate] = (),
    alerts: Iterable[Alert] = (),
    registry: SensorRegistry | None = None,
    subject: str = "Patient/example",
    attribution: Mapping[datetime, float] | None = None,
) -> dict[str, Any]:
    """Assemble a FHIR-style ``Bundle`` of the requested resources.

    Parameters
    ----------
    observations, estimates, alerts
        The measurements, inferences and alerts to export. Each is rendered
        with the provenance appropriate to its kind.
    registry
        Sensor declarations, used to label observations with their room and
        documented semantics.
    subject
        Reference for the monitored person.
    attribution
        Optional per-estimate attribution probability, keyed by estimate time.
    """
    entries: list[dict[str, Any]] = []
    for observation in observations:
        entries.append({"resource": observation_resource(observation, registry)})
    for estimate in estimates:
        share = attribution.get(estimate.at) if attribution is not None else None
        entries.append(
            {"resource": state_resource(estimate, subject=subject, attribution=share)}
        )
    for alert in alerts:
        entries.append({"resource": alert_resource(alert, subject=subject)})

    logger.info("Assembled FHIR-style bundle with %d entries", len(entries))
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
        "meta": {
            "tag": [
                {
                    "system": CODE_SYSTEM,
                    "code": "research-prototype",
                    "display": (
                        "FHIR-style export from a research toolkit. Not a "
                        "validated profile and not clinically validated data."
                    ),
                }
            ]
        },
    }


def summarise_provenance(payload: Mapping[str, Any]) -> dict[str, int]:
    """Count how many resources in a bundle are measured versus inferred.

    Provided so a consumer can assert, in one line, that it has not been
    handed inferences dressed as measurements.
    """
    counts: dict[str, int] = {}
    for entry in payload.get("entry", []):
        resource = entry.get("resource", {})
        for extension in resource.get("extension", []):
            if extension.get("url") != f"{CODE_SYSTEM}/provenance":
                continue
            for inner in extension.get("extension", []):
                if inner.get("url") == "kind":
                    kind = str(inner.get("valueCode"))
                    counts[kind] = counts.get(kind, 0) + 1
    return counts


def measured_only(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """Return only the genuinely measured resources from a bundle."""
    return [
        entry["resource"]
        for entry in payload.get("entry", [])
        if any(
            inner.get("valueCode") == PROVENANCE_MEASURED
            for extension in entry.get("resource", {}).get("extension", [])
            if extension.get("url") == f"{CODE_SYSTEM}/provenance"
            for inner in extension.get("extension", [])
        )
    ]
