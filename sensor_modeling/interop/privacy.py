"""Pseudonymisation and redaction for exported research data.

Two claims this module does **not** make, stated first because getting them
wrong is how re-identification happens:

*Pseudonymisation is not anonymisation.* Replacing an identifier with a
pseudonym removes the name, not the person. A behavioural record is a detailed
account of when somebody sleeps, eats and leaves the house; anyone with a
little side information can often re-identify it. Pseudonymised exports remain
personal data and must be handled as such.

*A hash is not a pseudonym.* Hashing a short identifier such as ``patient_7``
protects nothing: an attacker hashes every plausible identifier and matches.
Pseudonyms here are therefore keyed with a secret salt, so the mapping cannot
be reconstructed without it. The salt must be supplied, never defaulted, and
must be kept separately from the data it protects.

What the module does provide: deterministic pseudonyms that are stable across
runs and machines, so a longitudinal study can link a subject's records
without holding their identity; and a redaction pass that strips the
free-form metadata an export does not need.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Fields whose values are free-form and most likely to carry incidental
#: identifiers -- a room named after a person, an installer's note, a device
#: label containing a street address.
DEFAULT_REDACTED_KEYS: frozenset[str] = frozenset(
    {"context", "note", "notes", "detail", "description", "display", "text"}
)

#: Patterns that look like contact details or locations wherever they appear.
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){9,14}\d(?!\d)")
_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE)

REDACTED = "[redacted]"


class SaltError(ValueError):
    """Raised when a pseudonymisation salt is missing or too weak."""


@dataclass
class Pseudonymiser:
    """Deterministic, keyed pseudonyms for identifiers.

    Parameters
    ----------
    salt
        Secret key. Must be at least 16 characters. Keep it separately from
        the exported data: together, they reverse the pseudonymisation for
        anyone who can enumerate candidate identifiers.
    prefix
        Prepended to each pseudonym so its kind stays legible. The prefix is
        cosmetic and provides no separation on its own.
    domain
        Optional separation label. Two pseudonymisers sharing a salt but
        differing in *domain* produce unrelated digests for the same
        identifier, because the domain is mixed into the key rather than into
        the visible text. Leave empty to keep a single global namespace.
    length
        Hexadecimal characters retained. The default gives a collision
        probability far below one in a billion for study-sized cohorts while
        staying short enough to read.

    Notes
    -----
    The same identifier and salt always produce the same pseudonym, on any
    machine and in any process, which is what lets a longitudinal study link
    records without holding identities.
    """

    salt: str
    prefix: str = "subj"
    length: int = 16
    domain: str = ""

    def __post_init__(self) -> None:
        """Reject a salt too weak to be worth having."""
        if not isinstance(self.salt, str) or len(self.salt) < 16:
            raise SaltError(
                "salt must be at least 16 characters; a short or guessable "
                "salt lets an attacker reconstruct the mapping by enumerating "
                "candidate identifiers"
            )
        if not 8 <= self.length <= 64:
            raise ValueError("length must lie between 8 and 64 characters")

    def _key(self) -> bytes:
        """Return the HMAC key, separated by domain when one is set.

        Deriving a subkey is what makes the domain meaningful. Mixing it into
        the pseudonym text instead would leave the digest unchanged, so records
        for one subject would stay joinable across domains by comparing the
        part after the prefix.
        """
        key = self.salt.encode("utf-8")
        if self.domain:
            key = hmac.new(
                key, b"domain:" + self.domain.encode("utf-8"), hashlib.sha256
            ).digest()
        return key

    def pseudonym(self, identifier: str) -> str:
        """Return the stable pseudonym for *identifier*."""
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("identifier must be a non-empty string")
        digest = hmac.new(
            self._key(), identifier.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return f"{self.prefix}-{digest[: self.length]}"

    def mapping(self, identifiers: Iterable[str]) -> dict[str, str]:
        """Return the pseudonym for each identifier.

        The result is the re-identification key. Store it separately from the
        exported data, or not at all.
        """
        return {identifier: self.pseudonym(identifier) for identifier in identifiers}


def research_identifier(study: str, subject: str, *, salt: str) -> str:
    """Return a reproducible identifier for a subject within a study.

    Scoping by study means the same person carries different identifiers in
    different studies, so records cannot be joined across them by identifier
    alone. The study is used to derive a study-specific key, not merely as a
    label: sharing a salt across studies would otherwise leave every digest
    identical and the records trivially linkable.
    """
    if not isinstance(study, str) or not study.strip():
        raise ValueError("study must be a non-empty string")
    return Pseudonymiser(salt=salt, prefix=study, domain=study).pseudonym(subject)


@dataclass
class RedactionPolicy:
    """What an export should strip before leaving the research environment.

    Parameters
    ----------
    drop_keys
        Keys removed entirely wherever they appear.
    scrub_patterns
        Whether to replace anything resembling an email address, telephone
        number or postcode in remaining free text.
    keep_keys
        Keys preserved even if they appear in *drop_keys*. Use sparingly and
        deliberately.
    """

    drop_keys: frozenset[str] = field(default=DEFAULT_REDACTED_KEYS)
    scrub_patterns: bool = True
    keep_keys: frozenset[str] = field(default_factory=frozenset)

    def should_drop(self, key: str) -> bool:
        """Whether a key is removed under this policy."""
        return key in self.drop_keys and key not in self.keep_keys


def _scrub(text: str) -> str:
    """Replace contact details and locations in free text."""
    scrubbed = _EMAIL.sub(REDACTED, text)
    scrubbed = _PHONE.sub(REDACTED, scrubbed)
    return _POSTCODE.sub(REDACTED, scrubbed)


def redact(
    payload: Any,
    policy: RedactionPolicy | None = None,
    *,
    pseudonyms: Mapping[str, str] | None = None,
) -> Any:
    """Return a copy of *payload* with identifying material removed.

    Walks any nested structure of mappings and sequences, so it applies
    equally to a single resource, a bundle, or a whole experiment record.

    Parameters
    ----------
    payload
        The structure to redact. Not modified.
    policy
        What to strip. Defaults to removing free-form metadata and scrubbing
        contact details from remaining text.
    pseudonyms
        Replacements applied to any string that exactly matches a key. Use
        :meth:`Pseudonymiser.mapping` to build it.
    """
    rules = policy or RedactionPolicy()
    replacements = pseudonyms or {}

    if isinstance(payload, Mapping):
        result: MutableMapping[str, Any] = {}
        for key, value in payload.items():
            name = str(key)
            if rules.should_drop(name):
                continue
            result[name] = redact(value, rules, pseudonyms=replacements)
        return result

    if isinstance(payload, (list, tuple)):
        return [redact(item, rules, pseudonyms=replacements) for item in payload]

    if isinstance(payload, str):
        if payload in replacements:
            return replacements[payload]
        return _scrub(payload) if rules.scrub_patterns else payload

    return payload


def redact_bundle(
    bundle: Mapping[str, Any],
    *,
    salt: str,
    subjects: Iterable[str] = (),
    sensors: Iterable[str] = (),
    policy: RedactionPolicy | None = None,
) -> dict[str, Any]:
    """Pseudonymise and redact an exported bundle in one pass.

    Subjects and sensors are pseudonymised under separate prefixes, so a
    reader can still tell which kind of identifier they are looking at
    without being able to recover either.

    Where a pseudonym is supplied, the field carrying the identifier is
    *kept* and its value replaced, rather than dropped. Dropping it would be
    stronger, but it would also destroy the ability to link a record to its
    sensor or subject, which is the whole point of pseudonymising rather
    than deleting.
    """
    subject_names = list(subjects)
    sensor_names = list(sensors)
    replacements: dict[str, str] = {}
    if subject_names:
        replacements.update(
            Pseudonymiser(salt=salt, prefix="subj").mapping(subject_names)
        )
    if sensor_names:
        replacements.update(
            Pseudonymiser(salt=salt, prefix="sens").mapping(sensor_names)
        )

    # Keep the fields the pseudonyms are meant to land in. Without this the
    # default policy would drop `display` outright and the sensor pseudonyms
    # would have nowhere to go.
    rules = policy or RedactionPolicy()
    if replacements:
        rules = RedactionPolicy(
            drop_keys=rules.drop_keys,
            scrub_patterns=rules.scrub_patterns,
            keep_keys=rules.keep_keys | frozenset({"display", "reference"}),
        )

    redacted = redact(bundle, rules, pseudonyms=replacements)
    if not isinstance(redacted, dict):  # pragma: no cover - bundles are mappings
        raise TypeError("a bundle must be a mapping")

    meta = redacted.setdefault("meta", {})
    tags = meta.setdefault("tag", [])
    tags.append(
        {
            "code": "pseudonymised",
            "display": (
                "Identifiers replaced with keyed pseudonyms and free-form "
                "metadata removed. Pseudonymised behavioural data remains "
                "personal data: it is not anonymised and can often be "
                "re-identified from side information."
            ),
        }
    )
    logger.info("Redacted bundle: %d identifiers pseudonymised", len(replacements))
    return redacted


def identifiers_in(bundle: Mapping[str, Any]) -> set[str]:
    """Collect the sensor and subject identifiers a bundle carries.

    Provided so a caller can see what would be pseudonymised before doing it,
    rather than discovering an unredacted identifier after export.
    """
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in {"display", "reference"} and isinstance(value, str):
                    found.add(value)
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(bundle)
    return found
