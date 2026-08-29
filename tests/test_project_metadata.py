"""Tests for release and citation metadata consistency."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONCEPT_DOI = "10.5281/zenodo.17070041"


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _regex_value(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_release_versions_are_consistent_across_metadata_files():
    pyproject = _read("pyproject.toml")
    package_init = _read("sensor_modeling/__init__.py")
    citation = yaml.safe_load(_read("CITATION.cff"))
    zenodo = json.loads(_read(".zenodo.json"))
    readme = _read("README.md")

    versions = {
        "pyproject": _regex_value(r'^version = "([^"]+)"', pyproject),
        "package": _regex_value(r'__version__ = "([^"]+)"', package_init),
        "citation": citation["version"],
        "zenodo": zenodo["version"],
        "readme": _regex_value(r"version=\{([^}]+)\}", readme),
    }

    assert len(set(versions.values())) == 1, versions


def test_zenodo_concept_doi_is_referenced_consistently():
    pyproject = _read("pyproject.toml")
    citation = yaml.safe_load(_read("CITATION.cff"))
    readme = _read("README.md")

    citation_dois = {
        identifier["value"]
        for identifier in citation["identifiers"]
        if identifier["type"] == "doi"
    }

    assert CONCEPT_DOI in pyproject
    assert CONCEPT_DOI in readme
    assert CONCEPT_DOI in citation_dois
    assert citation["preferred-citation"]["doi"] == CONCEPT_DOI


def test_zenodo_notes_disclaim_medical_device_status():
    """The archive must carry the disclaimer, not only the repository.

    A Zenodo record is often the only thing a downstream reader sees, so the
    scope limit has to travel with it rather than living solely in the README.
    """
    zenodo = json.loads(_read(".zenodo.json"))

    notes = zenodo.get("notes", "")
    assert "not a medical device" in notes.lower()
    assert "simulator" in notes.lower()


def test_zenodo_and_citation_describe_the_same_software():
    """A description that drifts from the citation misleads whoever cites it."""
    zenodo = json.loads(_read(".zenodo.json"))
    citation = yaml.safe_load(_read("CITATION.cff"))

    for text in (zenodo["description"], citation["abstract"]):
        assert "ambient" in text.lower()
        assert "not a medical device" in text.lower()


def test_zenodo_metadata_has_required_repository_linkage():
    zenodo = json.loads(_read(".zenodo.json"))

    related_identifiers = zenodo["related_identifiers"]

    assert zenodo["upload_type"] == "software"
    assert zenodo["access_right"] == "open"
    assert zenodo["license"] == "mit"
    assert any(
        item["identifier"]
        == "https://github.com/DiogoRibeiro7/behavioral-sensing-research"
        and item["relation"] == "isIdenticalTo"
        and item["resource_type"] == "software"
        for item in related_identifiers
    )
