"""The digest rule that gates the frozen v0.3 external scoring.

``text_file_sha256`` is the single source of truth shared by the script that
records artifact digests and the script that verifies them. A regression that
dropped the newline normalisation would still return a well-formed 64-character
digest, so nothing would look wrong until a Windows checkout refused a frozen
cohort that had not changed. That happened once, and it stopped the one-shot
scoring run. These tests exist so it cannot happen silently again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sensor_modeling.utils import text_file_sha256

LF = "\n"
CRLF = "\r\n"


def test_crlf_and_lf_forms_of_the_same_text_digest_identically(
    tmp_path: Path,
) -> None:
    """The digest must describe the content, not the checkout that produced it."""
    content = f"alpha{LF}beta{LF}gamma{LF}"
    lf_file = tmp_path / "lf.json"
    crlf_file = tmp_path / "crlf.json"
    lf_file.write_bytes(content.encode("utf-8"))
    crlf_file.write_bytes(content.replace(LF, CRLF).encode("utf-8"))

    # The two files genuinely differ on disk, or the test proves nothing.
    assert lf_file.read_bytes() != crlf_file.read_bytes()
    assert text_file_sha256(lf_file) == text_file_sha256(crlf_file)


def test_the_digest_is_the_lf_form_rather_than_the_bytes_on_disk(
    tmp_path: Path,
) -> None:
    """Normalisation must actually happen, and must land on the LF form.

    Asserting only that CRLF and LF agree would still pass if the function
    normalised in the other direction, which would disagree with every digest
    already recorded in the frozen artifacts.
    """
    content = f"alpha{LF}beta{LF}"
    crlf_file = tmp_path / "crlf.json"
    crlf_file.write_bytes(content.replace(LF, CRLF).encode("utf-8"))

    lf_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    raw_digest = hashlib.sha256(crlf_file.read_bytes()).hexdigest()

    assert text_file_sha256(crlf_file) == lf_digest
    assert text_file_sha256(crlf_file) != raw_digest


def test_a_content_change_still_changes_the_digest(tmp_path: Path) -> None:
    """Normalising newlines must not blunt the check it exists to support."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(f"alpha{CRLF}beta{CRLF}".encode())
    second.write_bytes(f"alpha{CRLF}gamma{CRLF}".encode())

    assert text_file_sha256(first) != text_file_sha256(second)


def test_the_frozen_cohort_manifest_verifies_against_its_declaration() -> None:
    """The exact check that gates the one-shot scoring, on the real artifacts.

    This is the producer/verifier agreement the shared helper exists to keep:
    the declaration records a digest of the manifest, and scoring refuses to
    run unless recomputing it agrees. It must hold on every platform's
    checkout, which is what the unit tests above cannot assert on their own.
    """
    manifest = Path("artifacts/v03/external_cohort_manifest.json")
    declaration = json.loads(
        Path("artifacts/v03/external_cohort_freeze_declaration.json").read_text(
            encoding="utf-8"
        )
    )

    assert text_file_sha256(manifest) == declaration["manifest_sha256"]
