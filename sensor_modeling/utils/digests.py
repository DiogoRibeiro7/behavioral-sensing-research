"""Digests for provenance records, stable across platforms.

A frozen artifact carries the digest of the file it was derived from, and a
later run recomputes that digest to decide whether the input still is what was
frozen. That only works if the digest describes the content. Git rewrites line
endings on checkout for platforms that ask for CRLF, so hashing the raw bytes of
a tracked text file records where the repository was cloned rather than what the
file says: the same content verifies on Linux and is rejected on Windows.

Use :func:`text_file_sha256` for files git may rewrite. Hash raw bytes directly
for data that never passes through git, such as a downloaded archive, where
every byte is meant to be exactly as distributed and a normalising digest would
hide real corruption.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["text_file_sha256"]


def text_file_sha256(path: Path) -> str:
    """Return the SHA-256 of *path* with CRLF newlines normalised to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
