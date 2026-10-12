"""Tests for the ContextAct@A4H adapter, on synthetic files in the dataset's format.

No file of the dataset is used. They guard what the adapter rests on:
- the description file is read by cell position, so empty cells shift nothing;
- the logs are read exactly as recorded, with the annotation variables kept out
  of the sensors and the position never read;
- activity records pair into intervals, and every record that cannot is kept
  and reported rather than dropped;
- every label, sensor type and location is declared, and each sensor feeds the
  CASAS-trained model's channel its room and modality name, or none with a
  reason.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from sensor_modeling.external import (
    ContractError,
    OntologyMapping,
    Severity,
    to_canonical,
    validate_household,
)
from sensor_modeling.external.contextact import (
    CONTEXTACT_MAPPING,
    DESCRIPTION,
    EXTRA_LABELS,
    FILES,
    HOUSEHOLDS,
    LABELS,
    LOGS,
    MAPPING_FILE,
    PROVENANCE,
    TIMEZONE,
    UNDOCUMENTED,
    UNSPECIFIED,
    ContextActAdapter,
    eligibility,
    model_channel,
    occupancy,
    pair_annotations,
    read_description,
    read_household,
    read_log,
    sensor_table,
)
from sensor_modeling.external.mapping import UNDECLARED
from sensor_modeling.states import BehaviouralState

S = BehaviouralState
ROOT = Path(__file__).resolve().parents[1]
JULY = HOUSEHOLDS[0]

HEADER: list[str | None] = [
    "Sensor Id in Dataset",
    "Sensor Type",
    "Measured Property",
    "Variable Type [range]",
    "Measure Unit",
    "Sensor Location",
]
VARIABLES: list[list[str | None]] = [
    ["Presence_Salon", "Infrared Movement Sensor", "Presence Living Room", None, None, "Living Room"],
    ["Presence_Table", "Infrared Movement Sensor", "Presence Dining Room", None, None, "Dining Room"],
    ["Presence_Bureau", "Infrared Movement Sensor", "Presence Study Room", None, None, "Study Room"],
    ["Presence_Lit", "Infrared Movement Sensor", "Presence Bedroom", None, None, "Bedroom"],
    ["Presence_SDB", "Infrared Movement Sensor", "Presence Bathroom", None, None, "Bathroom"],
    ["Presence_Cuisine", "Infrared Movement Sensor", "Presence Kitchen", None, None, "Kitchen"],
    ["C6", "Contact Sensor", "Opening Main Entrance Door", "Binary", "OPEN-CLOSED", "Main Entrance"],
    ["C2", "Contact Sensor", "Garden Door Opening", "Binary", "OPEN-CLOSED", "First Floor"],
    ["C3", "Contact Sensor", "Fridge Door Opening", "Binary", "OPEN-CLOSED", "Kitchen"],
    ["bed_pressure", "Pressure Sensor", "Bed Presence", "Number", "[0-1024]", "Bedroom"],
    ["R5_Presence", "Presence Sensor", "R5", None, None, "Hallway"],
    ["Temperature_Salon", "Temperature Sensor", "Temperature Living Room", None, None, "Living Room"],
    ["Gaz_Logement", None, "Gaz", None, None, None],
    ["current_activity", "ACTIVITY", "Self-annotated Activity", "String", None, None],
    ["current_people", "PEOPLE", "Self-annotated Number of People", "Number", None, None],
    ["current_position", "POSITION", "Self-annotated Position", "String", None, None],
]  # fmt: skip


def write_xlsx(path: Path, rows: list[list[str | None]]) -> None:
    """A minimal workbook: shared strings and one sheet with cell references."""
    shared: list[str] = []
    index: dict[str, int] = {}
    xml_rows = []
    for number, row in enumerate(rows, start=1):
        cells = []
        for column, value in enumerate(row):
            if value is None:
                continue
            if value not in index:
                index[value] = len(shared)
                shared.append(value)
            reference = f"{chr(ord('A') + column)}{number}"
            cells.append(f'<c r="{reference}" t="s"><v>{index[value]}</v></c>')
        xml_rows.append(f'<row r="{number}">{"".join(cells)}</row>')
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    strings = "".join(f"<si><t>{escape(s)}</t></si>" for s in shared)
    with zipfile.ZipFile(path, "w") as workbook:
        workbook.writestr("xl/sharedStrings.xml", f"<sst {ns}>{strings}</sst>")
        workbook.writestr(
            "xl/worksheets/sheet1.xml",
            f"<worksheet {ns}><sheetData>{''.join(xml_rows)}</sheetData></worksheet>",
        )


def line(stamp: str, sensor: str, value: str) -> str:
    return f'"2016-07-25 {stamp}";"{sensor}";"{value}"'


LOG = [
    line("08:00:00,100", "current_people", "1"),
    line("08:00:01,000", "Presence_Salon", "ON"),
    line("08:00:30,5", "Presence_Salon", "OFF"),
    line("08:01:00,000", "current_activity", "Manger_Start"),
    line("08:01:10,000", "current_position", "Salon"),
    line("08:02:00,000", "bed_pressure", "163"),
    line("08:03:00,000", "Temperature_Salon", "21.5"),
    line("08:04:00,000", "tv_salon_volume", "12"),
    line("08:05:00,000", "current_activity", "Douche_End"),
    line("08:06:00,000", "current_activity", "Sortir_Start"),
    line("08:06:05,000", "C6", "OPEN"),
    line("08:06:09,000", "C6", "CLOSED"),
    line("08:07:00,000", "current_people", "0"),
    line("09:00:00,000", "current_people", "2"),
    line("09:30:00,000", "current_activity", "Sortir_Stop"),
    line("09:31:00,000", "current_people", "1"),
    line("09:40:00,000", "current_activity", "TV_Start"),
    line("09:45:00,000", "current_activity", "TV_Start"),
    line("09:50:00,000", "current_activity", "TV_End"),
    line("10:00:00,000", "Presence_Lit", "ON"),
]


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    write_xlsx(tmp_path / DESCRIPTION, [HEADER, *VARIABLES])
    for household, name in LOGS.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        records = LOG if household == JULY else LOG[:3]
        path.write_text("\n".join(records) + "\n", encoding="latin-1")
    return tmp_path


# ----------------------------------------------------------------------------
# The description file
# ----------------------------------------------------------------------------
class TestDescription:
    def test_it_reads_each_variable_by_cell_position(self, dataset: Path) -> None:
        described = read_description(dataset / DESCRIPTION)
        assert described["Presence_Table"].sensor_type == "Infrared Movement Sensor"
        assert described["Presence_Table"].location == "Dining Room"
        assert described["bed_pressure"].location == "Bedroom"
        assert described["Gaz_Logement"].sensor_type == UNSPECIFIED
        assert described["Gaz_Logement"].location is None

    def test_a_contact_sensors_type_names_what_it_measures(self, dataset: Path) -> None:
        described = read_description(dataset / DESCRIPTION)
        assert described["C6"].sensor_type == (
            "Contact Sensor/Opening Main Entrance Door"
        )
        assert described["C3"].sensor_type == "Contact Sensor/Fridge Door Opening"

    def test_the_annotation_variables_are_not_sensors(self, dataset: Path) -> None:
        described = read_description(dataset / DESCRIPTION)
        assert not {"current_activity", "current_people", "current_position"} & set(
            described
        )
        assert len(described) == len(VARIABLES) - 3

    def test_it_refuses_a_variable_listed_twice(self, tmp_path: Path) -> None:
        write_xlsx(tmp_path / "d.xlsx", [HEADER, VARIABLES[0], VARIABLES[0]])
        with pytest.raises(ContractError, match="twice"):
            read_description(tmp_path / "d.xlsx")

    def test_it_refuses_a_file_without_its_columns(self, tmp_path: Path) -> None:
        write_xlsx(tmp_path / "d.xlsx", [HEADER[:2], VARIABLES[0][:2]])
        with pytest.raises(ContractError, match="no column"):
            read_description(tmp_path / "d.xlsx")


# ----------------------------------------------------------------------------
# The logs, the annotations and the occupancy
# ----------------------------------------------------------------------------
class TestLog:
    def test_it_reads_each_record_as_recorded(self, dataset: Path) -> None:
        records = list(read_log(dataset / LOGS[JULY]))
        assert len(records) == len(LOG)
        assert records[1] == (datetime(2016, 7, 25, 8, 0, 1), "Presence_Salon", "ON")
        assert records[2][0] == datetime(2016, 7, 25, 8, 0, 30, 500000)

    def test_it_reads_bytes_that_are_not_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "log"
        path.write_bytes(
            line("08:00:00,000", "music_salonTitle", "Caf\xe9").encode("latin-1")
        )
        assert list(read_log(path))[0][2] == "Caf\xe9"

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("2016-07-25 08:00:00,000;Presence_Salon;ON", "is not a record"),
            ('"25/07/2016 08:00";"Presence_Salon";"ON"', "no valid timestamp"),
        ],
    )
    def test_it_refuses_a_line_that_is_not_a_record(
        self, tmp_path: Path, text: str, message: str
    ) -> None:
        path = tmp_path / "log"
        path.write_text(text + "\n", encoding="latin-1")
        with pytest.raises(ContractError, match=message):
            list(read_log(path))


T = datetime(2016, 7, 25, 8, 0)


def at(minute: int) -> datetime:
    return T.replace(minute=minute)


class TestAnnotations:
    def test_a_start_pairs_with_the_next_end_or_stop(self) -> None:
        got = pair_annotations(
            [
                (at(0), "Dormir_Start"),
                (at(5), "Dormir_End"),
                (at(6), "Sortir_Start"),
                (at(9), "Sortir_Stop"),
            ]
        )
        assert [(a.label, a.start, a.end) for a in got] == [
            ("Dormir", at(0), at(5)),
            ("Sortir", at(6), at(9)),
        ]

    def test_unpaired_records_are_kept_and_cannot_be_scored(self) -> None:
        got = pair_annotations(
            [
                (at(0), "Manger_Start"),
                (at(1), "Douche_End"),
                (at(2), "TV_Start"),
                (at(3), "TV_Start"),
                (at(4), "TV_End"),
            ]
        )
        assert [(a.label, a.start, a.end) for a in got] == [
            ("Manger", at(0), None),
            ("Douche", at(1), at(1)),
            ("TV", at(2), None),
            ("TV", at(3), at(4)),
        ]

    def test_it_refuses_an_unknown_marker(self) -> None:
        with pytest.raises(ContractError, match="neither"):
            pair_annotations([(at(0), "Dormir_Pause")])
        with pytest.raises(ContractError, match="neither"):
            pair_annotations([(at(0), "Dormir")])


class TestOccupancy:
    def test_each_count_holds_until_the_next(self) -> None:
        got = occupancy([(at(0), "1"), (at(10), "0"), (at(20), "2")], at(30))
        assert [(p.start, p.end, p.residents) for p in got] == [
            (at(0), at(10), 1),
            (at(20), at(30), 2),
        ]

    def test_it_refuses_a_count_that_is_not_a_number(self) -> None:
        with pytest.raises(ContractError, match="not a number"):
            occupancy([(at(0), "deux")], at(1))


class TestHousehold:
    def test_it_reads_a_period_in_the_contracts_terms(self, dataset: Path) -> None:
        data = read_household(dataset, JULY)
        assert data.household == JULY
        assert data.timezone == TIMEZONE
        assert data.residents == 1
        values = {(e.sensor_id, e.value) for e in data.events}
        assert ("Presence_Salon", "ON") in values
        assert ("bed_pressure", 163.0) in values
        assert ("Temperature_Salon", 21.5) in values
        assert not {e.sensor_id for e in data.events} & {
            "current_activity",
            "current_people",
            "current_position",
        }
        sensors = {s.sensor_id: s for s in data.sensors}
        assert sensors["tv_salon_volume"].sensor_type == UNDOCUMENTED
        assert sensors["tv_salon_volume"].location is None
        assert [a.label for a in data.annotations] == [
            "Manger",
            "Douche",
            "Sortir",
            "TV",
            "TV",
        ]
        assert [p.residents for p in data.occupancy] == [1, 2, 1]

    def test_the_adapter_serves_both_periods(self, dataset: Path) -> None:
        adapter = ContextActAdapter(dataset)
        assert adapter.households() == HOUSEHOLDS
        assert adapter.provenance is PROVENANCE
        assert adapter.load(HOUSEHOLDS[1]).household == HOUSEHOLDS[1]
        with pytest.raises(KeyError):
            adapter.load("A4H-2017-01")

    def test_it_refuses_a_log_with_no_sensor_record(self, dataset: Path) -> None:
        (dataset / LOGS[JULY]).write_text(LOG[0] + "\n", encoding="latin-1")
        with pytest.raises(ContractError, match="no sensor record"):
            read_household(dataset, JULY)


# ----------------------------------------------------------------------------
# The mapping, the channels and validation
# ----------------------------------------------------------------------------
class TestMapping:
    def test_every_label_is_declared(self) -> None:
        for label in (*LABELS, *EXTRA_LABELS):
            assert CONTEXTACT_MAPPING.label(label).status != UNDECLARED, label

    def test_the_states_it_can_score(self) -> None:
        scored = {
            CONTEXTACT_MAPPING.label(label).target
            for label in (*LABELS, *EXTRA_LABELS)
            if CONTEXTACT_MAPPING.label(label).usable
        }
        assert scored == {
            S.SLEEPING,
            S.AWAY,
            S.BATHROOM_ACTIVITY,
            S.KITCHEN_ACTIVITY,
            S.HOME_ACTIVE,
            S.HOME_INACTIVE,
        }
        assert CONTEXTACT_MAPPING.label("Musique").status == "ambiguous"

    def test_it_round_trips_through_its_frozen_form(self, tmp_path: Path) -> None:
        digest = CONTEXTACT_MAPPING.write(tmp_path / "mapping.json")
        read = OntologyMapping.read(tmp_path / "mapping.json", digest)
        assert read.sha256() == CONTEXTACT_MAPPING.sha256() == digest

    @pytest.mark.skipif(
        not (ROOT / MAPPING_FILE).exists(), reason="mapping not yet frozen"
    )
    def test_the_frozen_file_is_the_declared_mapping(self) -> None:
        frozen = OntologyMapping.read(ROOT / MAPPING_FILE)
        assert frozen.sha256() == CONTEXTACT_MAPPING.sha256()
        recorded = json.loads((ROOT / MAPPING_FILE).read_text(encoding="utf-8"))
        assert recorded["mapping_sha256"] == CONTEXTACT_MAPPING.sha256()


class TestChannels:
    def test_each_sensor_feeds_its_channel_or_none_with_a_reason(
        self, dataset: Path
    ) -> None:
        rows = {r["sensor"]: r for r in sensor_table(read_household(dataset, JULY))}
        assert {name: r["model_channel"] for name, r in rows.items() if r["model_channel"]} == {
            "Presence_Salon": "living_motion",
            "Presence_Table": "living_motion",
            "Presence_Bureau": "living_motion",
            "Presence_Lit": "bedroom_motion",
            "Presence_SDB": "bathroom_motion",
            "Presence_Cuisine": "kitchen_motion",
            "C6": "hall_door",
        }  # fmt: skip
        assert "ambiguous" in rows["C2"]["unsupported_because"]
        assert "no kitchen_contact channel" in rows["C3"]["unsupported_because"]
        assert "unmappable" in rows["R5_Presence"]["unsupported_because"]
        assert "sample sensor" in rows["bed_pressure"]["unsupported_because"]
        assert all(
            r["unsupported_because"] for r in rows.values() if not r["model_channel"]
        )

    def test_it_reads_a_sensor_as_its_mapping_says(self, dataset: Path) -> None:
        sensors = {s.sensor_id: s for s in read_household(dataset, JULY).sensors}
        assert model_channel(sensors["C6"]) == ("hall_door", "routed")


class TestValidation:
    def test_a_period_is_eligible_with_its_unpaired_records_unscored(
        self, dataset: Path
    ) -> None:
        report = validate_household(read_household(dataset, JULY), CONTEXTACT_MAPPING)
        errors = {i.code for i in report.issues if i.severity is Severity.ERROR}
        assert errors == {"impossible_interval"}
        assert "multi_resident" in {i.code for i in report.issues}
        assert eligibility(report) == (True, [])

    def test_any_other_error_excludes_the_period(self, dataset: Path) -> None:
        log = dataset / LOGS[JULY]
        log.write_text(
            log.read_text(encoding="latin-1")
            + line("11:00:00,000", "current_activity", "Lire_Start")
            + "\n"
            + line("11:01:00,000", "current_activity", "Lire_End")
            + "\n",
            encoding="latin-1",
        )
        data = read_household(dataset, JULY)
        report = validate_household(data, CONTEXTACT_MAPPING)
        assert "undeclared_label" in {i.code for i in report.issues}
        assert eligibility(report) == (True, [])  # left unscored, and reported
        undeclared = OntologyMapping(
            dataset="contextact-a4h", version="0", labels=(), sensor_types=()
        )
        eligible, blocking = eligibility(validate_household(data, undeclared))
        assert not eligible
        assert "undeclared_sensor" in blocking

    def test_it_converts_to_the_canonical_form(self, dataset: Path) -> None:
        converted = to_canonical(
            read_household(dataset, JULY), CONTEXTACT_MAPPING, strict=False
        )
        rooms = {spec.sensor_id: spec.room for spec in converted.recording.registry}
        assert rooms["Presence_Table"] == "living"
        assert rooms["C6"] == "hall"
        states = {a.state for a in converted.recording.activities}
        assert S.AWAY in states


class TestProvenance:
    def test_it_pins_every_file_the_adapter_reads(self) -> None:
        assert DESCRIPTION in FILES
        assert set(LOGS.values()) <= set(FILES)
        assert all(len(digest) == 64 for digest in FILES.values())
        assert PROVENANCE.files == dict(sorted(FILES.items()))
