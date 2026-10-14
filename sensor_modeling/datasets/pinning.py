"""What a one-shot freeze records about the code, and what its run requires.

A one-shot experiment freezes its protocol together with the digest of every
source the run depends on: every module of the package and a few named files,
with the libraries' versions beside them. The run refuses when a source
differs, unless an amendment declared before the run gives the source's new
digest and the reason for it. These helpers do that for any experiment, with
its own list of named files.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = [
    "code_changes",
    "code_now",
    "file_sha256",
    "library_changes",
    "read_amendments",
]

PACKAGE = "sensor_modeling"


def file_sha256(path: Path) -> str:
    """SHA-256 of a text file, whatever its line endings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def code_now(
    root: Path,
    files: Sequence[str],
    distributions: Sequence[str] = (),
    package: str = PACKAGE,
) -> dict[str, Any]:
    """The digest of every module of *package* and of *files* under *root*, now.

    The libraries named in *distributions* are recorded with their versions.
    """
    root = Path(root)
    modules = sorted(
        path.relative_to(root).as_posix()
        for path in (root / package).rglob("*.py")
        if "__pycache__" not in path.parts
    )
    return {
        "sources": {name: file_sha256(root / name) for name in (*modules, *files)},
        "distributions": {
            "python": platform.python_version(),
            **{name: metadata.version(name) for name in distributions},
        },
    }


def library_changes(path: Path, distributions: Sequence[str]) -> list[str]:
    """The libraries of *distributions* whose version differs now from the freeze."""
    frozen = json.loads(Path(path).read_text(encoding="utf-8")).get("at_freeze")
    if frozen is None:
        raise ValueError(f"{path} does not record the code it was frozen against")
    then = frozen["distributions"]
    return sorted(
        name for name in distributions if then.get(name) != metadata.version(name)
    )


def read_amendments(path: Path) -> list[dict[str, Any]]:
    """The declared amendments, or none when the file does not exist.

    Each amendment names its ``reason`` and, in ``sources``, the digest of
    every pinned source it changes, ``None`` for one it removes.
    """
    if not Path(path).exists():
        return []
    amendments = json.loads(Path(path).read_text(encoding="utf-8"))["amendments"]
    for amendment in amendments:
        if not str(amendment.get("reason", "")).strip():
            raise ValueError("every amendment needs its reason")
        if not isinstance(amendment.get("sources"), dict) or not amendment["sources"]:
            raise ValueError("every amendment names the sources it changes")
    return list(amendments)


def code_changes(
    path: Path,
    root: Path,
    files: Sequence[str],
    amendments: Sequence[Mapping[str, Any]] = (),
    package: str = PACKAGE,
) -> list[str]:
    """The pinned sources that differ now from the freeze at *path* and its amendments.

    A source an amendment changes is expected at the digest the amendment
    gives, the last amendment's when several change it.
    """
    frozen = json.loads(Path(path).read_text(encoding="utf-8")).get("at_freeze")
    if frozen is None:
        raise ValueError(f"{path} does not record the code it was frozen against")
    now = code_now(root, files, package=package)["sources"]
    then = dict(frozen["sources"])
    for amendment in amendments:
        for name, digest in amendment["sources"].items():
            if digest is None:
                then.pop(name, None)
            else:
                then[name] = digest
    return sorted(
        name for name in set(then) | set(now) if then.get(name) != now.get(name)
    )
