from __future__ import annotations

import json
from pathlib import Path

from scripts import inventory_v03_external_archive as inventory


def _registry() -> dict[str, object]:
    return {
        "source": {"record": "15708568", "archive": "labeled_data.zip"},
        "single_resident_home_ids": ["hh101", "hh104", "aruba"],
        "two_resident_home_ids": ["hh107"],
        "more_than_two_resident_home_ids": ["cairo"],
        "unknown_resident_home_ids": ["mva001"],
        "development_only_home_ids": ["hh101"],
    }


def test_inventory_excludes_development_and_hashes_candidates(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    (root / "hh104").mkdir(parents=True)
    (root / "hh101").mkdir(parents=True)
    (root / "hh104" / "hh104.csv").write_text(
        '2011-01-01,00:00:00.000001,Kitchen,ON,Cook="begin"\n',
        encoding="utf-8",
    )
    (root / "hh101" / "hh101.csv").write_text(
        '2011-01-01,00:00:00.000001,Kitchen,ON,Cook="begin"\n',
        encoding="utf-8",
    )

    payload = inventory.build_inventory(root, _registry())
    homes = {row["home_id"]: row for row in payload["homes"]}

    assert set(homes) == {"aruba", "hh104"}
    assert homes["hh104"]["archive_member_found"] is True
    assert homes["hh104"]["files"][0]["sha256"]
    assert homes["hh104"]["files"][0]["raw_text_audit"][
        "annotation_marker_labels"
    ] == {"Cook": 1}
    assert homes["aruba"]["archive_member_found"] is False
    assert payload["model_inference_performed"] is False
    assert payload["performance_metrics_computed"] is False


def test_home_matching_requires_explicit_path_component() -> None:
    path = Path("root/not-hh104-ish/data.csv")
    assert inventory._home_matches(path, {"hh104"}) == []
    assert inventory._home_matches(Path("root/hh104/data.csv"), {"hh104"}) == [
        "hh104"
    ]


def test_output_is_json_serialisable(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    payload = inventory.build_inventory(root, _registry())
    json.dumps(payload, allow_nan=False)
