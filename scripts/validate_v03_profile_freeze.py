"""Validate a frozen v0.3 circadian-profile artifact without scoring homes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_CANDIDATE = "v0.2 + circadian prior only"
EXPECTED_PANEL_COUNT = 22
EXPECTED_FITTING_COUNT = 20
EXPECTED_EXCLUSIONS = {
    "hh107": "two-resident CASAS metadata",
    "hh121": "two-resident CASAS metadata",
}
DECIMAL_LIMIT = 12_000_000
BINARY_LIMIT = 12 * 1024 * 1024
EXPECTED_PROFILE_STATES = {
    "away",
    "bathroom_activity",
    "bed_awake",
    "home_active",
    "home_inactive",
    "kitchen_activity",
    "sleeping",
}


def _canonical_fit_sha256(fit: dict[str, object]) -> str:
    payload = json.dumps(
        fit, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_size_rule(
    source: dict[str, object], homes: list[dict[str, object]]
) -> None:
    convention = source["size_convention"]
    limit = source["byte_limit_exclusive"]
    evaluation = source.get("size_rule_evaluation")
    invariant = source.get("cohort_invariant_across_size_conventions", False)

    if convention in {"decimal_MB", "binary_MiB"}:
        expected_limit = DECIMAL_LIMIT if convention == "decimal_MB" else BINARY_LIMIT
        assert limit == expected_limit
        assert invariant is False
    else:
        assert convention == "equivalent_decimal_MB_and_binary_MiB"
        assert limit == DECIMAL_LIMIT
        assert invariant is True
        assert isinstance(evaluation, dict)
        assert evaluation == {
            "decimal_MB": {
                "byte_limit_exclusive": DECIMAL_LIMIT,
                "development_panel_count": EXPECTED_PANEL_COUNT,
            },
            "binary_MiB": {
                "byte_limit_exclusive": BINARY_LIMIT,
                "development_panel_count": EXPECTED_PANEL_COUNT,
            },
        }

    assert isinstance(limit, int)
    assert all(int(row["bytes"]) < limit for row in homes)


def validate(
    payload: dict[str, object], *, expected_revision: str | None = None
) -> None:
    assert payload["schema_version"] == 2
    assert payload["status"] == "frozen-development-fit"
    assert payload["candidate"] == EXPECTED_CANDIDATE
    assert payload["test_outcomes_inspected"] is False
    assert payload["development_panel_count"] == EXPECTED_PANEL_COUNT
    assert payload["fitting_home_count"] == EXPECTED_FITTING_COUNT

    panel = payload["development_panel_homes"]
    assert isinstance(panel, list)
    assert len(panel) == EXPECTED_PANEL_COUNT
    panel_ids = [row["id"] for row in panel]
    names = [row["filename"] for row in panel]
    digests = [row["sha256"] for row in panel]
    assert len(set(panel_ids)) == EXPECTED_PANEL_COUNT
    assert len(set(names)) == EXPECTED_PANEL_COUNT
    assert len(set(digests)) == EXPECTED_PANEL_COUNT
    for row in panel:
        assert isinstance(row["bytes"], int) and row["bytes"] > 0
        assert isinstance(row["sha256"], str) and len(row["sha256"]) == 64

    fitting_ids = payload["fitting_home_ids"]
    assert isinstance(fitting_ids, list)
    assert len(fitting_ids) == EXPECTED_FITTING_COUNT
    assert len(set(fitting_ids)) == EXPECTED_FITTING_COUNT

    exclusions = payload["fitting_exclusions"]
    assert isinstance(exclusions, list)
    exclusion_map = {row["id"]: row["reason"] for row in exclusions}
    assert exclusion_map == EXPECTED_EXCLUSIONS
    assert len(exclusions) == len(EXPECTED_EXCLUSIONS)

    panel_set = set(panel_ids)
    fitting_set = set(fitting_ids)
    exclusion_set = set(EXPECTED_EXCLUSIONS)
    assert exclusion_set <= panel_set
    assert fitting_set <= panel_set
    assert fitting_set.isdisjoint(exclusion_set)
    assert fitting_set | exclusion_set == panel_set

    source = payload["source"]
    assert isinstance(source, dict)
    assert source["provider"] == "CASAS / Zenodo"
    assert source["record"] == "15708568"
    _validate_size_rule(source, panel)

    fit = payload["fit"]
    assert isinstance(fit, dict)
    profile = fit["profile"]
    assert isinstance(profile, dict)
    assert set(profile) == EXPECTED_PROFILE_STATES
    assert fit["recordings"] == EXPECTED_FITTING_COUNT
    assert fit["labelled_seconds"] > 0
    minimum = float(fit["minimum_multiplier"])
    maximum = float(fit["maximum_multiplier"])
    assert minimum > 0
    assert maximum >= minimum
    for values in profile.values():
        assert len(values) == 24
        assert all(minimum <= float(value) <= maximum for value in values)

    expected_hash = _canonical_fit_sha256(fit)
    assert payload["profile_sha256"] == expected_hash
    assert len(expected_hash) == 64

    if expected_revision is not None:
        assert payload["fitting_revision"] == expected_revision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--expected-revision")
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    validate(payload, expected_revision=args.expected_revision)
    print(
        json.dumps(
            {
                "valid_v03_profile_freeze": True,
                "profile_sha256": payload["profile_sha256"],
                "fitting_revision": payload["fitting_revision"],
                "development_panel_count": payload["development_panel_count"],
                "fitting_home_count": payload["fitting_home_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
