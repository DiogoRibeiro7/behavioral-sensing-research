"""Tests for the Paper 1 uncertainty-panel source contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyse_uncertainty_panel.py"
MANIFEST = ROOT / "artifacts" / "v03" / "external_cohort_manifest.json"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyse_uncertainty_panel", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


panel = _load_script()


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_development_sources_match_frozen_22_home_panel() -> None:
    sources = panel.development_sources(_manifest())

    assert list(sources) == _manifest()["development_panel"]
    assert len(sources) == 22
    assert all(
        entry["reason"] == "development panel; outcomes inspected"
        for entry in sources.values()
    )


def test_development_sources_refuse_missing_screened_identity() -> None:
    manifest = _manifest()
    missing_home = manifest["development_panel"][0]
    manifest["screened"] = [
        row for row in manifest["screened"] if row["id"] != missing_home
    ]

    with pytest.raises(SystemExit, match="missing screened source identities"):
        panel.development_sources(manifest)


def test_verify_recording_refuses_wrong_size(tmp_path: Path) -> None:
    path = tmp_path / "hh101.csv"
    path.write_bytes(b"abc")
    entry = {"id": "hh101", "bytes": 4, "sha256": "irrelevant"}

    with pytest.raises(SystemExit, match="source size"):
        panel.verify_recording(path, entry)


def test_verify_recording_refuses_wrong_sha256(tmp_path: Path) -> None:
    path = tmp_path / "hh101.csv"
    path.write_bytes(b"abc")
    entry = {"id": "hh101", "bytes": 3, "sha256": "0" * 64}

    with pytest.raises(SystemExit, match="source SHA-256"):
        panel.verify_recording(path, entry)
