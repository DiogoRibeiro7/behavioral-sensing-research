"""Tests for finding and verifying recordings under an extracted archive."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sensor_modeling.datasets.archive import locate, sha256


def _write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


class TestSha256:
    def test_it_is_the_digest_of_the_bytes_across_chunks(self, tmp_path: Path) -> None:
        content = bytes(range(256)) * 9000  # more than one 1 MiB chunk
        digest = _write(tmp_path / "big.bin", content)
        assert sha256(tmp_path / "big.bin") == digest

    def test_it_does_not_normalise_line_endings(self, tmp_path: Path) -> None:
        crlf = _write(tmp_path / "crlf.txt", b"a\r\nb\r\n")
        lf = _write(tmp_path / "lf.txt", b"a\nb\n")
        assert sha256(tmp_path / "crlf.txt") == crlf != sha256(tmp_path / "lf.txt")
        assert lf == sha256(tmp_path / "lf.txt")

    def test_an_empty_file(self, tmp_path: Path) -> None:
        assert _write(tmp_path / "empty", b"") == sha256(tmp_path / "empty")


class TestLocate:
    def test_it_finds_every_recording_at_any_depth(self, tmp_path: Path) -> None:
        b = _write(tmp_path / "labeled" / "deep" / "hh102.ann.txt", b"two")
        a = _write(tmp_path / "labeled" / "hh101.ann.txt", b"one")
        homes = {
            "hh102": {"filename": "hh102.ann.txt", "sha256": b},
            "hh101": {"filename": "hh101.ann.txt", "sha256": a},
        }
        paths = locate(tmp_path, homes)
        assert list(paths) == ["hh101", "hh102"]
        assert paths["hh101"] == tmp_path / "labeled" / "hh101.ann.txt"
        assert paths["hh102"] == tmp_path / "labeled" / "deep" / "hh102.ann.txt"

    def test_a_directory_of_the_same_name_is_not_a_recording(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "other" / "hh101.ann.txt").mkdir(parents=True)
        digest = _write(tmp_path / "hh101.ann.txt", b"one")
        homes = {"hh101": {"filename": "hh101.ann.txt", "sha256": digest}}
        assert locate(tmp_path, homes) == {"hh101": tmp_path / "hh101.ann.txt"}

    def test_it_refuses_a_missing_recording(self, tmp_path: Path) -> None:
        homes = {"hh101": {"filename": "hh101.ann.txt", "sha256": "0" * 64}}
        with pytest.raises(ValueError, match="expected one hh101.ann.txt, found 0"):
            locate(tmp_path, homes)

    def test_it_refuses_two_recordings_of_one_name(self, tmp_path: Path) -> None:
        digest = _write(tmp_path / "a" / "hh101.ann.txt", b"one")
        _write(tmp_path / "b" / "hh101.ann.txt", b"one")
        homes = {"hh101": {"filename": "hh101.ann.txt", "sha256": digest}}
        with pytest.raises(ValueError, match="found 2"):
            locate(tmp_path, homes)

    def test_it_refuses_a_recording_that_differs_from_its_digest(
        self, tmp_path: Path
    ) -> None:
        digest = _write(tmp_path / "hh101.ann.txt", b"one")
        (tmp_path / "hh101.ann.txt").write_bytes(b"one, edited")
        homes = {"hh101": {"filename": "hh101.ann.txt", "sha256": digest}}
        with pytest.raises(ValueError, match="hh101: recording does not match"):
            locate(tmp_path, homes)
