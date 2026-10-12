"""Find recordings under an extracted archive and verify them by digest.

A run on real data reads its recordings from a downloaded archive, and refuses
any recording whose bytes differ from the digest its splits file, cohort
manifest or protocol records. These two helpers do that once, for new run
scripts to import.

The run scripts behind published records keep their own copies of this code.
They are the provenance of those records, and several are pinned by frozen
protocols, so they are not edited to use this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

__all__ = ["locate", "sha256"]

_CHUNK = 1024 * 1024


def sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks, with line endings untouched."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate(root: Path, homes: Mapping[str, Mapping[str, str]]) -> dict[str, Path]:
    """Find every home's recording under *root* and verify it against its digest.

    Parameters
    ----------
    root
        The directory the archive was extracted to. Recordings are searched
        for at any depth.
    homes
        For each home, the recording's ``"filename"`` and its ``"sha256"``,
        as a splits file or cohort manifest records them.

    Returns
    -------
    dict
        Each home's recording, keyed by home in sorted order.

    Raises
    ------
    ValueError
        If a file name matches no file or more than one under *root*, or if a
        recording's bytes differ from its digest.
    """
    paths: dict[str, Path] = {}
    for home, entry in sorted(homes.items()):
        matches = sorted(p for p in Path(root).rglob(entry["filename"]) if p.is_file())
        if len(matches) != 1:
            raise ValueError(f"expected one {entry['filename']}, found {len(matches)}")
        if sha256(matches[0]) != entry["sha256"]:
            raise ValueError(f"{home}: recording does not match its frozen digest")
        paths[home] = matches[0]
    return paths
