"""Regression tests for the pre-test v0.3 profile-freeze contract."""

from pathlib import Path

import pytest

from scripts.freeze_v03_circadian_profile import _home_id, _select_fitting_paths
from scripts.validate_v03_profile_freeze import (
    EXPECTED_EXCLUSIONS,
    EXPECTED_PROFILE_STATES,
    _canonical_fit_sha256,
    validate,
)

PANEL_IDS = [
    "hh101",
    "hh102",
    "hh103",
    "hh105",
    "hh106",
    "hh107",
    "hh108",
    "hh110",
    "hh111",
    "hh114",
    "hh118",
    "hh119",
    "hh120",
    "hh121",
    "hh122",
    "hh123",
    "hh124",
    "hh125",
    "hh126",
    "hh127",
    "hh129",
    "hh130",
]


def test_final_fit_excludes_the_two_multi_resident_development_homes() -> None:
    panel = [Path(f"{home_id}.csv") for home_id in PANEL_IDS]
    fitting = _select_fitting_paths(panel)
    fitting_ids = [_home_id(path) for path in fitting]

    assert len(fitting_ids) == 20
    assert "hh107" not in fitting_ids
    assert "hh121" not in fitting_ids
    assert set(fitting_ids) == set(PANEL_IDS) - {"hh107", "hh121"}


def test_final_fit_refuses_a_panel_missing_a_metadata_exclusion() -> None:
    panel_ids = ["hh128" if home_id == "hh121" else home_id for home_id in PANEL_IDS]
    panel = [Path(f"{home_id}.csv") for home_id in panel_ids]

    with pytest.raises(SystemExit, match="metadata exclusions"):
        _select_fitting_paths(panel)


def _valid_payload() -> dict[str, object]:
    panel = [
        {
            "id": home_id,
            "filename": f"{home_id}.csv",
            "bytes": 1_000_000 + index,
            "sha256": f"{index + 1:064x}",
        }
        for index, home_id in enumerate(PANEL_IDS)
    ]
    fitting_ids = [
        home_id for home_id in PANEL_IDS if home_id not in EXPECTED_EXCLUSIONS
    ]
    fit = {
        "profile": {state: [1.0] * 24 for state in EXPECTED_PROFILE_STATES},
        "recordings": 20,
        "labelled_seconds": 1.0,
        "minimum_multiplier": 0.25,
        "maximum_multiplier": 4.0,
    }
    return {
        "schema_version": 2,
        "status": "frozen-development-fit",
        "candidate": "v0.2 + circadian prior only",
        "source": {
            "provider": "CASAS / Zenodo",
            "record": "15708568",
            "size_convention": "equivalent_decimal_MB_and_binary_MiB",
            "byte_limit_exclusive": 12_000_000,
            "size_rule_evaluation": {
                "decimal_MB": {
                    "byte_limit_exclusive": 12_000_000,
                    "development_panel_count": 22,
                },
                "binary_MiB": {
                    "byte_limit_exclusive": 12 * 1024 * 1024,
                    "development_panel_count": 22,
                },
            },
            "cohort_invariant_across_size_conventions": True,
        },
        "development_panel_count": 22,
        "development_panel_homes": panel,
        "fitting_home_count": 20,
        "fitting_home_ids": fitting_ids,
        "fitting_exclusions": [
            {"id": home_id, "reason": reason}
            for home_id, reason in sorted(EXPECTED_EXCLUSIONS.items())
        ],
        "fitting_revision": "a" * 40,
        "fit": fit,
        "profile_sha256": _canonical_fit_sha256(fit),
        "test_outcomes_inspected": False,
    }


def test_validator_accepts_only_the_20_home_fitting_partition() -> None:
    payload = _valid_payload()
    validate(payload, expected_revision="a" * 40)

    payload["fitting_home_ids"] = list(payload["fitting_home_ids"]) + ["hh107"]
    payload["fitting_home_count"] = 21
    with pytest.raises(AssertionError):
        validate(payload, expected_revision="a" * 40)
