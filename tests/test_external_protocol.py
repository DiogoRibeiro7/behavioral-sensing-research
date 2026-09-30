"""Tests for the frozen Phase 5 external-generalisation protocol.

The protocol, its mapping and its page must be exactly what the code declares;
every item the protocol must record is recorded; the dataset adapter must read
the dataset's format; and the frozen pipeline must run end to end, on synthetic
data in that format, without any external result being computed.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import ActivityInterval, CasasRecording
from sensor_modeling.datasets.channel_models import (
    DECLARED,
    HURDLE,
    filter_recursion,
    home_statistics,
    household_channel_counts,
    household_statistics,
    pool_channels,
    total_loglik,
)
from sensor_modeling.datasets.external_protocol import (
    ADAPTED,
    POPULATION_SHA256,
    TOLERATED,
    ZERO_SHOT,
    adaptation_end,
    check_frozen_protocol,
    declared_protocol,
    eligibility,
    model_channel,
    scored_states,
    sensor_table,
    unsupported_states,
)
from sensor_modeling.datasets.external_protocol_summary import render_protocol
from sensor_modeling.datasets.information_sets import EvidenceResolution
from sensor_modeling.datasets.partial_pooling import PoolingConfig
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.structural_models import fit_samples
from sensor_modeling.external import (
    ContractError,
    OntologyMapping,
    to_canonical,
    validate_household,
)
from sensor_modeling.external.ordonez import (
    FILES,
    HOUSEHOLDS,
    LABELS,
    ORDONEZ_MAPPING,
    SENSORS,
    TIMEZONE,
    OrdonezAdapter,
    native_type,
    read_household,
    sensor_id,
)
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts" / "phase5" / "external_protocol.json"
MAPPING = ROOT / "artifacts" / "phase5" / "ordonez_mapping.json"
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
DOC = ROOT / "docs" / "PHASE5_EXTERNAL_PROTOCOL.md"


# ----------------------------------------------------------------------------
class TestFrozenProtocol:
    """The frozen files are exactly what the code declares."""

    def test_the_protocol_file_is_the_declared_protocol(self) -> None:
        declared = declared_protocol(load_frozen_splits(SPLITS))
        check_frozen_protocol(declared, PROTOCOL)
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert frozen["protocol_sha256"] == declared.sha256()

    def test_a_changed_protocol_is_refused(self, tmp_path: Path) -> None:
        declared = declared_protocol(load_frozen_splits(SPLITS))
        payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        payload["periods"]["scored_days"]["OrdonezA"] = 8
        changed = tmp_path / "protocol.json"
        changed.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="cannot change"):
            check_frozen_protocol(declared, changed)

    def test_the_mapping_file_is_the_declared_mapping(self) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        mapping = OntologyMapping.read(MAPPING, frozen["mapping"]["sha256"])
        assert mapping.to_dict() == ORDONEZ_MAPPING.to_dict()

    def test_the_page_is_exactly_the_rendering_of_the_protocol(self) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        committed = DOC.read_text(encoding="utf-8").replace("\r\n", "\n")
        assert committed == render_protocol(frozen)

    def test_every_required_item_is_recorded(self) -> None:
        frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        provenance = frozen["dataset"]["provenance"]
        assert provenance["name"] and provenance["version"] and provenance["source"]
        assert "10.24432/C5J02M" in provenance["source"]
        assert provenance["files"] == FILES
        assert len(frozen["dataset"]["archive"]["sha256"]) == 64
        households = frozen["households"]
        for key in ("inclusion", "exclusion", "eligibility"):
            assert households[key]
        assert set(frozen["sensor_mapping"]) == set(HOUSEHOLDS)
        assert frozen["preprocessing"]["conversion"]
        assert frozen["unsupported_states"] and frozen["unsupported_sensors"]
        assert frozen["metrics"]["primary"]
        models = frozen["models"]
        assert models[ZERO_SHOT]["population_sha256"] == POPULATION_SHA256["all"]
        assert models["structural_ensemble"]["population_sha256"] == POPULATION_SHA256
        assert set(frozen["conditions"]) >= {ZERO_SHOT, ADAPTED}
        assert frozen["adaptation_rules"]["forbidden"]
        assert frozen["bootstrap"]["unit"].startswith("local calendar days")
        assert set(frozen["criteria"]) >= set(frozen["estimands"])

    def test_it_holds_no_result(self) -> None:
        text = PROTOCOL.read_text(encoding="utf-8")
        frozen = json.loads(text)
        assert "results" not in frozen
        assert "no model performance" in frozen["status"]


# ----------------------------------------------------------------------------
class TestMapping:
    """The mapping covers the documentation, and nothing is guessed."""

    def test_every_documented_value_is_declared(self) -> None:
        for label in LABELS:
            assert ORDONEZ_MAPPING.label(label).status != "undeclared", label
        for home in HOUSEHOLDS:
            for location, kind, place in SENSORS[home]:
                assert ORDONEZ_MAPPING.sensor_type(
                    native_type(location, kind)
                ).status != ("undeclared")
                assert ORDONEZ_MAPPING.location(place).status != "undeclared"

    def test_meals_are_ambiguous_and_leaving_is_away(self) -> None:
        for meal in ("Breakfast", "Lunch", "Dinner", "Snack"):
            assert ORDONEZ_MAPPING.label(meal).status == "ambiguous"
        leaving = ORDONEZ_MAPPING.label("Leaving")
        assert leaving.status == "approximate"
        assert leaving.target.value == "away"

    def test_scored_and_unsupported_states(self) -> None:
        assert scored_states() == [
            "away",
            "home_inactive",
            "sleeping",
            "bathroom_activity",
        ]
        unsupported = unsupported_states()
        assert set(unsupported) == {"home_active", "bed_awake", "kitchen_activity"}
        assert "ambiguous" in unsupported["kitchen_activity"]
        assert unsupported["bed_awake"] == "no label names it"

    def test_the_model_channels_each_sensor_feeds(self) -> None:
        assert model_channel(("Maindoor", "Magnetic", "Entrance")) == (
            "hall_door",
            "routed",
        )
        assert model_channel(("Shower", "PIR", "Bathroom"))[0] == "bathroom_motion"
        assert model_channel(("Door", "PIR", "Living"))[0] == "living_motion"
        channel, reason = model_channel(("Fridge", "Magnetic", "Kitchen"))
        assert channel is None and "kitchen_contact" in reason
        channel, reason = model_channel(("Bed", "Pressure", "Bedroom"))
        assert channel is None and "state sensor" in reason
        channel, reason = model_channel(("Toaster", "Electric", "Kitchen"))
        assert channel is None and "unmappable" in reason
        routed = {r["model_channel"] for r in sensor_table("OrdonezA")} - {None}
        assert routed == {"bathroom_motion", "kitchen_motion", "hall_door"}


# ----------------------------------------------------------------------------
# Synthetic files in the dataset's exact format
# ----------------------------------------------------------------------------
HEADER_SENSORS = (
    "Start time          \tEnd time            \tLocation\tType\t\tPlace\n"
    "--------------------\t--------------------\t--------\t--------\t-----\n"
)
HEADER_ADLS = (
    "Start time          \tEnd time            \tActivity\t\n"
    "--------------------\t--------------------\t--------\n"
)
DAY0 = datetime(2011, 11, 28)


def _t(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def write_home(directory: Path, home: str = "OrdonezA", days: int = 9) -> None:
    """A synthetic home in the dataset's format: a plausible daily routine."""
    sensors: list[str] = []
    adls: list[str] = []

    def activate(start: datetime, minutes: float, where: tuple[str, str, str]) -> None:
        end = start + timedelta(minutes=minutes)
        sensors.append(
            f"{_t(start)}\t\t{_t(end)}\t\t{where[0]}\t\t{where[1]}\t{where[2]}\n"
        )

    def annotate(start: datetime, minutes: float, label: str) -> None:
        end = start + timedelta(minutes=minutes)
        adls.append(f"{_t(start)}\t\t{_t(end)}\t\t{label}\t\n")

    for day in range(days):
        base = DAY0 + timedelta(days=day)
        annotate(base + timedelta(hours=0, minutes=30), 7 * 60, "Sleeping")
        activate(base + timedelta(minutes=30), 7 * 60, ("Bed", "Pressure", "Bedroom"))
        annotate(base + timedelta(hours=7, minutes=40), 10, "Toileting")
        activate(base + timedelta(hours=7, minutes=41), 2, ("Basin", "PIR", "Bathroom"))
        activate(
            base + timedelta(hours=7, minutes=45), 0.2, ("Toilet", "Flush", "Bathroom")
        )
        annotate(base + timedelta(hours=8), 20, "Breakfast")
        activate(base + timedelta(hours=8, minutes=1), 5, ("Cooktop", "PIR", "Kitchen"))
        activate(
            base + timedelta(hours=8, minutes=3), 1, ("Fridge", "Magnetic", "Kitchen")
        )
        annotate(base + timedelta(hours=9), 180, "Leaving")
        activate(base + timedelta(hours=9), 0.1, ("Maindoor", "Magnetic", "Entrance"))
        activate(base + timedelta(hours=12), 0.1, ("Maindoor", "Magnetic", "Entrance"))
        annotate(base + timedelta(hours=13), 240, "Spare_Time/TV")
        activate(base + timedelta(hours=13), 240, ("Seat", "Pressure", "Living"))
        annotate(base + timedelta(hours=17, minutes=30), 15, "Showering")
        activate(
            base + timedelta(hours=17, minutes=31), 10, ("Shower", "PIR", "Bathroom")
        )
    (directory / f"{home}_Sensors.txt").write_text(
        HEADER_SENSORS + "".join(sensors), encoding="latin-1"
    )
    (directory / f"{home}_ADLs.txt").write_text(
        HEADER_ADLS + "".join(adls), encoding="latin-1"
    )


class TestAdapter:
    """The dataset's own format, read into the contract."""

    def test_it_reads_the_format(self, tmp_path: Path) -> None:
        write_home(tmp_path, days=2)
        data = read_household(tmp_path, "OrdonezA")
        assert data.timezone == TIMEZONE and data.residents == 1
        assert len(data.sensors) == len(SENSORS["OrdonezA"])
        # Nine activations a day, each an ON and an OFF.
        assert len(data.events) == 2 * 9 * 2
        first = data.events[0]
        assert (first.sensor_id, first.value) == (
            sensor_id("Bed", "Pressure", "Bedroom"),
            "ON",
        )
        assert data.annotations[0].label == "Sleeping"
        assert data.annotations[0].semantics.value == "interval"
        report = validate_household(data, ORDONEZ_MAPPING)
        assert not any(i.code.startswith("undeclared") for i in report.issues)

    def test_an_undocumented_sensor_is_kept_and_reported(self, tmp_path: Path) -> None:
        write_home(tmp_path, days=1)
        path = tmp_path / "OrdonezA_Sensors.txt"
        extra = "2011-11-28 10:00:00\t\t2011-11-28 10:01:00\t\tDoor\t\tPIR\tAttic\n"
        path.write_text(path.read_text(encoding="latin-1") + extra, encoding="latin-1")
        data = read_household(tmp_path, "OrdonezA")
        assert sensor_id("Door", "PIR", "Attic") in {s.sensor_id for s in data.sensors}
        report = validate_household(data, ORDONEZ_MAPPING)
        assert "undeclared_sensor" in report.codes()
        assert eligibility(report) == (True, [])

    def test_malformed_files_are_refused(self, tmp_path: Path) -> None:
        write_home(tmp_path, days=1)
        (tmp_path / "OrdonezA_ADLs.txt").write_text(
            HEADER_ADLS + "2011-11-28 10:00:00\t\tSleeping\n", encoding="latin-1"
        )
        with pytest.raises(ContractError, match="fields"):
            read_household(tmp_path, "OrdonezA")
        (tmp_path / "OrdonezA_ADLs.txt").write_text("no header\n", encoding="latin-1")
        with pytest.raises(ContractError, match="header"):
            read_household(tmp_path, "OrdonezA")
        with pytest.raises(KeyError):
            read_household(tmp_path, "OrdonezC")

    def test_the_periods(self, tmp_path: Path) -> None:
        write_home(tmp_path, days=2)
        data = OrdonezAdapter(tmp_path).load("OrdonezA")
        assert adaptation_end(data) == DAY0 + timedelta(days=7)


class TestEligibility:
    """Tolerated errors leave items unscored; any other excludes the home."""

    def test_tolerated_and_blocking_errors(self, tmp_path: Path) -> None:
        write_home(tmp_path, days=1)
        data = read_household(tmp_path, "OrdonezA")
        first = data.annotations[0]
        assert first.end is not None
        backwards = replace(first, start=first.end, end=first.start)
        tolerated = replace(data, annotations=(*data.annotations, backwards))
        report = validate_household(tolerated, ORDONEZ_MAPPING)
        assert "impossible_interval" in report.codes()
        assert eligibility(report) == (True, [])
        assert "impossible_interval" in TOLERATED
        blocked = validate_household(replace(data, timezone=None), ORDONEZ_MAPPING)
        assert eligibility(blocked) == (False, ["missing_timezone"])


# ----------------------------------------------------------------------------
def simulated(seed: int) -> CasasRecording:
    sim = simulate(HouseholdConfig(days=2, seed=seed))
    return CasasRecording(
        sim.registry,
        sim.observations,
        tuple(
            ActivityInterval(e.state.value, e.start, e.end, e.state)
            for e in sim.truth.episodes
        ),
    )


class TestFrozenPipelineRuns:
    """The declared pipeline executes on data in the dataset's format; nothing is scored."""

    def test_zero_shot_and_adapted_recursions(self, tmp_path: Path) -> None:
        resolution, ontology = EvidenceResolution(), StateOntology()
        homes = {f"sim{s}": simulated(s) for s in range(1, 5)}
        statistics = {
            h: home_statistics(r, resolution, ontology, household=h)
            for h, r in homes.items()
        }
        population = fit_samples(
            statistics,
            sorted(homes),
            states=tuple(ontology.states),
            pseudo_windows=12.0,
        )["all"]

        write_home(tmp_path, days=9)
        data = read_household(tmp_path, "OrdonezA")
        report = validate_household(data, ORDONEZ_MAPPING)
        assert eligibility(report)[0]
        converted = to_canonical(data, ORDONEZ_MAPPING, strict=False, source="test")
        recording = converted.recording
        counts, rows, labels, moments = household_channel_counts(
            recording, resolution, ontology, household="OrdonezA"
        )
        assert {c.name for c in counts} == {
            "bathroom_motion",
            "kitchen_motion",
            "hall_door",
        }

        transition = ontology.transition(resolution.step)
        posteriors = {}
        for family in (DECLARED, HURDLE):
            models = population.models(recording.registry, resolution, family, ontology)
            _, posteriors[family] = filter_recursion(
                total_loglik(models, counts), transition, ontology.stationary()
            )
        cut = adaptation_end(data).replace(tzinfo=moments[0].tzinfo)
        own = household_statistics(
            recording, resolution, ontology, household="OrdonezA", until=cut
        )
        adapted = pool_channels(
            population,
            recording.registry,
            own,
            household="OrdonezA",
            resolution=resolution,
            config=PoolingConfig(288.0),
            until=cut,
            ontology=ontology,
        ).models()
        _, posteriors[ADAPTED] = filter_recursion(
            total_loglik(adapted, counts), transition, ontology.stationary()
        )
        for posterior in posteriors.values():
            assert posterior.shape == (len(moments), len(ontology.states))
            assert np.allclose(posterior.sum(axis=1), 1.0)
        scored = [r for r in rows if moments[int(r)] >= cut]
        assert scored and rows.size > len(scored)
        assert set(np.unique(labels)) <= {
            ontology.states.index(s)
            for s in ontology.states
            if s.value in scored_states()
        }

    def test_the_declared_pipeline_names_what_it_uses(self) -> None:
        frozen: dict[str, Any] = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert "to_canonical" in frozen["preprocessing"]["conversion"]
        assert "pool_channels" in frozen["models"][ADAPTED]["adaptation"]
        assert frozen["models"][ADAPTED]["pooling"] == {"strength": 288.0}
