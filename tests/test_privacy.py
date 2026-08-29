"""Tests for pseudonymisation and export redaction."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sensor_modeling.interop import (
    Pseudonymiser,
    RedactionPolicy,
    SaltError,
    bundle,
    identifiers_in,
    redact,
    redact_bundle,
    research_identifier,
)
from sensor_modeling.observations import Modality, Observation, ObservationKind

SALT = "a-long-enough-study-salt"
T0 = datetime(2024, 5, 1, 8, 0, tzinfo=timezone.utc)


def observation(**overrides: object) -> Observation:
    """A measured activation with optional overrides."""
    payload: dict[str, object] = {
        "timestamp": T0,
        "sensor_id": "kitchen_motion",
        "modality": Modality.MOTION,
        "kind": ObservationKind.EVENT,
        "value": 1.0,
    }
    payload.update(overrides)
    return Observation(**payload)  # type: ignore[arg-type]


class TestPseudonyms:
    def test_the_same_identifier_always_maps_to_the_same_pseudonym(self) -> None:
        """Stability across runs is what lets a study link records."""
        first = Pseudonymiser(salt=SALT).pseudonym("patient_7")
        second = Pseudonymiser(salt=SALT).pseudonym("patient_7")
        assert first == second

    def test_different_identifiers_map_differently(self) -> None:
        maker = Pseudonymiser(salt=SALT)
        assert maker.pseudonym("patient_7") != maker.pseudonym("patient_8")

    def test_a_different_salt_gives_a_different_pseudonym(self) -> None:
        """Without the salt the mapping cannot be reconstructed."""
        assert Pseudonymiser(salt=SALT).pseudonym("patient_7") != Pseudonymiser(
            salt="an-entirely-different-salt"
        ).pseudonym("patient_7")

    def test_a_pseudonym_does_not_contain_the_identifier(self) -> None:
        assert "patient_7" not in Pseudonymiser(salt=SALT).pseudonym("patient_7")

    def test_a_bare_hash_of_the_identifier_does_not_match(self) -> None:
        """Guards the keying: an unsalted digest would be trivially reversed."""
        import hashlib

        bare = hashlib.sha256(b"patient_7").hexdigest()
        assert bare[:16] not in Pseudonymiser(salt=SALT).pseudonym("patient_7")

    def test_a_short_salt_is_refused(self) -> None:
        with pytest.raises(SaltError, match="at least 16 characters"):
            Pseudonymiser(salt="short")

    def test_an_empty_identifier_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Pseudonymiser(salt=SALT).pseudonym("  ")

    def test_an_implausible_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="length"):
            Pseudonymiser(salt=SALT, length=4)

    def test_a_mapping_covers_every_identifier(self) -> None:
        mapping = Pseudonymiser(salt=SALT).mapping(["a", "b", "c"])
        assert set(mapping) == {"a", "b", "c"}
        assert len(set(mapping.values())) == 3

    def test_study_scoping_prevents_joining_across_studies(self) -> None:
        """The same person must not be linkable between two studies."""
        first = research_identifier("sleepstudy", "patient_7", salt=SALT)
        second = research_identifier("mobilitystudy", "patient_7", salt=SALT)
        assert first != second
        assert first.startswith("sleepstudy-")


class TestRedaction:
    def test_free_form_metadata_is_removed(self) -> None:
        payload = {"value": 1.0, "context": {"installer": "Ana"}, "note": "private"}
        assert redact(payload) == {"value": 1.0}

    def test_contact_details_are_scrubbed_from_remaining_text(self) -> None:
        payload = {"summary": "contact ana@example.com or +351 912 345 678"}
        cleaned = redact(payload)
        assert "ana@example.com" not in cleaned["summary"]
        assert "912" not in cleaned["summary"]

    def test_scrubbing_can_be_disabled(self) -> None:
        policy = RedactionPolicy(scrub_patterns=False, drop_keys=frozenset())
        payload = {"summary": "ana@example.com"}
        assert redact(payload, policy)["summary"] == "ana@example.com"

    def test_a_key_can_be_kept_deliberately(self) -> None:
        policy = RedactionPolicy(keep_keys=frozenset({"description"}))
        payload = {"description": "PIR in the kitchen", "note": "gone"}
        cleaned = redact(payload, policy)
        assert "description" in cleaned
        assert "note" not in cleaned

    def test_nested_structures_are_walked(self) -> None:
        payload = {"entry": [{"resource": {"note": "x", "value": 2}}]}
        assert redact(payload) == {"entry": [{"resource": {"value": 2}}]}

    def test_the_input_is_not_modified(self) -> None:
        payload = {"note": "private", "value": 1}
        redact(payload)
        assert "note" in payload

    def test_pseudonyms_replace_exact_matches(self) -> None:
        cleaned = redact(
            {"reference": "Patient/abc"},
            pseudonyms={"Patient/abc": "subj-1234"},
        )
        assert cleaned["reference"] == "subj-1234"

    def test_timestamps_are_not_mistaken_for_telephone_numbers(self) -> None:
        payload = {"effectiveDateTime": "2024-05-01T08:00:00+00:00"}
        assert redact(payload)["effectiveDateTime"] == "2024-05-01T08:00:00+00:00"

    def test_numeric_values_are_untouched(self) -> None:
        payload = {"value": 0.9123456789, "count": 42}
        assert redact(payload) == payload


class TestBundleRedaction:
    @staticmethod
    def exported() -> dict:
        return bundle(
            observations=[
                observation(context={"installer": "call Ana on +351 912 345 678"})
            ],
            subject="Patient/mrs-silva",
        )

    def test_subject_and_sensor_are_pseudonymised(self) -> None:
        safe = redact_bundle(
            self.exported(),
            salt=SALT,
            subjects=["Patient/mrs-silva"],
            sensors=["kitchen_motion"],
        )
        found = identifiers_in(safe)
        assert "kitchen_motion" not in found
        assert "Patient/mrs-silva" not in found
        assert any(name.startswith("sens-") for name in found)

    def test_a_pseudonymised_field_is_kept_rather_than_dropped(self) -> None:
        """Dropping it would destroy the ability to link records at all."""
        safe = redact_bundle(self.exported(), salt=SALT, sensors=["kitchen_motion"])
        resource = safe["entry"][0]["resource"]
        assert "device" in resource
        assert resource["device"]["display"].startswith("sens-")

    def test_incidental_metadata_is_removed(self) -> None:
        safe = redact_bundle(self.exported(), salt=SALT)
        assert "Ana" not in str(safe)
        assert "912" not in str(safe)

    def test_the_measurement_itself_survives(self) -> None:
        """Redaction must not destroy the data the export exists to carry."""
        safe = redact_bundle(self.exported(), salt=SALT, sensors=["kitchen_motion"])
        resource = safe["entry"][0]["resource"]
        assert resource["effectiveDateTime"] == T0.isoformat()
        assert resource["valueQuantity"]["value"] == 1.0
        assert resource["status"] == "final"

    def test_provenance_survives_redaction(self) -> None:
        """A reader must still be able to tell measurement from inference."""
        safe = redact_bundle(self.exported(), salt=SALT)
        resource = safe["entry"][0]["resource"]
        assert any("provenance" in ext["url"] for ext in resource["extension"])

    def test_the_bundle_is_tagged_as_pseudonymised_not_anonymised(self) -> None:
        safe = redact_bundle(self.exported(), salt=SALT)
        tag = safe["meta"]["tag"][-1]
        assert tag["code"] == "pseudonymised"
        assert "not anonymised" in tag["display"]

    def test_redaction_is_reproducible(self) -> None:
        first = redact_bundle(self.exported(), salt=SALT, sensors=["kitchen_motion"])
        second = redact_bundle(self.exported(), salt=SALT, sensors=["kitchen_motion"])
        assert first == second

    def test_a_weak_salt_is_refused_at_the_bundle_level(self) -> None:
        with pytest.raises(SaltError):
            redact_bundle(self.exported(), salt="tiny", subjects=["a"])

    def test_identifiers_can_be_inspected_before_export(self) -> None:
        """So an unredacted identifier is found before it leaves, not after."""
        assert "kitchen_motion" in identifiers_in(self.exported())
