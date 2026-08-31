"""Validate a frozen v0.3 circadian-profile artifact without scoring homes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_CANDIDATE = "v0.2 + circadian prior only"
EXPECTED_HOME_COUNT = 22
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


def validate(
    payload: dict[str, object], *, expected_revision: str | None = None
) -> None:
    assert payload["schema_version"] == 1
    assert payload["status"] == "frozen-development-fit"
    assert payload["candidate"] == EXPECTED_CANDIDATE
    assert payload["test_outcomes_inspected"] is False
    assert payload["development_home_count"] == EXPECTED_HOME_COUNT

    homes = payload["development_homes"]
    assert isinstance(homes, list)
    assert len(homes) == EXPECTED_HOME_COUNT
    ids = [row["id"] for row in homes]
    names = [row["filename"] for row in homes]
    digests = [row["sha256"] for row in homes]
    assert len(set(ids)) == EXPECTED_HOME_COUNT
    assert len(set(names)) == EXPECTED_HOME_COUNT
    assert len(set(digests)) == EXPECTED_HOME_COUNT
    for row in homes:
        assert isinstance(row["bytes"], int) and row["bytes"] > 0
        assert isinstance(row["sha256"], str) and len(row["sha256"]) == 64

    source = payload["source"]
    assert source["provider"] == "CASAS / Zenodo"
    assert source["record"] == "15708568"
    assert source["size_convention"] in {"decimal_MB", "binary_MiB"}
    assert source["byte_limit_exclusive"] in {12_000_000, 12 * 1024 * 1024}
    # Where more than one 12 MB convention selects the documented cohort, the
    # recorded one must be among them: the prose ambiguity is then resolved by
    # the archive itself rather than by a choice made during fitting.
    equivalent = source.get("equivalent_size_conventions")
    if equivalent is not None:
        assert isinstance(equivalent, list) and equivalent
        assert set(equivalent) <= {"decimal_MB", "binary_MiB"}
        assert source["size_convention"] in equivalent
    # Where more than one 12 MB convention selects the documented cohort, the
    # recorded one must be among them: the prose ambiguity is then resolved by
    # the files themselves rather than by a choice made here.
    equivalent = source.get("equivalent_size_conventions")
    if equivalent is not None:
        assert isinstance(equivalent, list) and equivalent
        assert set(equivalent) <= {"decimal_MB", "binary_MiB"}
        assert source["size_convention"] in equivalent
    assert all(row["bytes"] < source["byte_limit_exclusive"] for row in homes)

    fit = payload["fit"]
    profile = fit["profile"]
    assert set(profile) == EXPECTED_PROFILE_STATES
    assert fit["recordings"] == EXPECTED_HOME_COUNT
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
                "development_home_count": payload["development_home_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
