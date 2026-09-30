"""The explicit mapping from a dataset's native vocabulary to this repository's ontology.

An external dataset's labels, sensor types and locations rarely correspond one
to one with the ontology. Every correspondence is therefore declared, entry by
entry, with one of four outcomes:

``exact``
    The native value means the ontology's target. One target.
``approximate``
    It is best represented by one target, but not exactly: the rationale says
    what is lost or added.
``unmappable``
    No target represents it. The rationale says why. Nothing is scored for it.
``ambiguous``
    Several targets fit and the native value does not say which. The
    candidates are listed, and nothing is scored for it: choosing one would
    invent an annotation the dataset did not make.

A native value the mapping does not declare at all is ``undeclared``. It is
reported, never dropped and never guessed. A missing location is ``missing``.

A mapping is data. It is written to a file with its SHA-256, so it can be
frozen before any external result is seen, and a result can cite exactly the
mapping it used.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..observations import Modality, ObservationKind, Unit
from ..states import BehaviouralState

MAPPING_SCHEMA = "external-ontology-mapping/1"


class Outcome(str, Enum):
    """How a native value corresponds to the ontology."""

    EXACT = "exact"
    APPROXIMATE = "approximate"
    UNMAPPABLE = "unmappable"
    AMBIGUOUS = "ambiguous"


#: A native value the mapping does not declare.
UNDECLARED = "undeclared"

#: A sensor the dataset gives no location for.
MISSING = "missing"

#: Every status a resolution can have.
STATUSES = (*(o.value for o in Outcome), UNDECLARED, MISSING)

#: The repository's canonical rooms: those its states are located in, and those
#: its evidence channels resolve motion and doors to.
CANONICAL_ROOMS = frozenset({"bathroom", "bedroom", "hall", "kitchen", "living"})

#: States a label may map to: every latent state, never the abstention value.
MAPPABLE_STATES = frozenset(
    s for s in BehaviouralState if s is not BehaviouralState.UNKNOWN
)


class MappingError(ValueError):
    """A mapping that is not a valid declaration."""


@dataclass(frozen=True)
class SensorSemantics:
    """What a native sensor type means in the canonical observation model.

    Attributes
    ----------
    modality, kind
        The canonical modality and temporal semantics.
    activation
        Raw tokens that mean an activation (an event) or the ``on`` level (a
        state).
    deactivation
        Raw tokens that end an activation or mean ``off``.
    numeric
        Whether raw values are numbers, as a sample's are.
    unit
        The unit of numeric values.
    """

    modality: Modality
    kind: ObservationKind
    activation: frozenset[str] = frozenset()
    deactivation: frozenset[str] = frozenset()
    numeric: bool = False
    unit: Unit = Unit.NONE

    def __post_init__(self) -> None:
        """Require values that can be interpreted under the declared kind."""
        object.__setattr__(self, "modality", Modality(self.modality))
        object.__setattr__(self, "kind", ObservationKind(self.kind))
        object.__setattr__(self, "unit", Unit(self.unit))
        object.__setattr__(self, "activation", frozenset(self.activation))
        object.__setattr__(self, "deactivation", frozenset(self.deactivation))
        if self.activation & self.deactivation:
            raise MappingError("a token cannot both activate and deactivate")
        if self.kind is ObservationKind.SAMPLE and not self.numeric:
            raise MappingError("a sample's values must be numeric")
        if self.kind is not ObservationKind.SAMPLE and not (
            self.numeric or self.activation
        ):
            raise MappingError(
                "an event or state needs its activation tokens, or numeric values"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form."""
        return {
            "modality": self.modality.value,
            "kind": self.kind.value,
            "activation": sorted(self.activation),
            "deactivation": sorted(self.deactivation),
            "numeric": self.numeric,
            "unit": self.unit.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SensorSemantics:
        """Rebuild semantics written by :meth:`to_dict`."""
        return cls(
            Modality(payload["modality"]),
            ObservationKind(payload["kind"]),
            frozenset(payload["activation"]),
            frozenset(payload["deactivation"]),
            bool(payload["numeric"]),
            Unit(payload["unit"]),
        )


@dataclass(frozen=True)
class MappingEntry:
    """One native value and how it corresponds to the ontology.

    Attributes
    ----------
    native
        The dataset's value.
    outcome
        One of :class:`Outcome`.
    targets
        One target for exact and approximate, two or more for ambiguous, none
        for unmappable.
    rationale
        Why: required for every outcome but exact.
    """

    native: str
    outcome: Outcome
    targets: tuple[Any, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        """Check the targets and rationale agree with the outcome."""
        outcome = Outcome(self.outcome)
        targets = tuple(self.targets)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "targets", targets)
        if not str(self.native).strip():
            raise MappingError("a mapping entry needs a native value")
        expected = {
            Outcome.EXACT: len(targets) == 1,
            Outcome.APPROXIMATE: len(targets) == 1,
            Outcome.UNMAPPABLE: not targets,
            Outcome.AMBIGUOUS: len(set(targets)) >= 2
            and len(set(targets)) == len(targets),
        }[outcome]
        if not expected:
            raise MappingError(
                f"{self.native!r}: an {outcome.value} entry has "
                "one target if exact or approximate, none if unmappable, and two "
                "or more distinct ones if ambiguous"
            )
        if outcome is not Outcome.EXACT and not self.rationale.strip():
            raise MappingError(
                f"{self.native!r}: an {outcome.value} entry needs a rationale"
            )


def exact(native: str, target: Any) -> MappingEntry:
    """``native`` means ``target``."""
    return MappingEntry(native, Outcome.EXACT, (target,))


def approximate(native: str, target: Any, rationale: str) -> MappingEntry:
    """``native`` is best represented by ``target``, for the stated reason."""
    return MappingEntry(native, Outcome.APPROXIMATE, (target,), rationale)


def unmappable(native: str, rationale: str) -> MappingEntry:
    """``native`` has no counterpart, for the stated reason."""
    return MappingEntry(native, Outcome.UNMAPPABLE, (), rationale)


def ambiguous(native: str, targets: Sequence[Any], rationale: str) -> MappingEntry:
    """``native`` fits several targets and does not say which."""
    return MappingEntry(native, Outcome.AMBIGUOUS, tuple(targets), rationale)


@dataclass(frozen=True)
class Resolution:
    """What a native value resolves to.

    Attributes
    ----------
    native
        The value looked up, or ``None`` for a missing location.
    status
        An outcome's value, :data:`UNDECLARED` or :data:`MISSING`.
    targets, rationale
        As declared; empty when undeclared or missing.
    """

    native: str | None
    status: str
    targets: tuple[Any, ...] = ()
    rationale: str = ""

    @property
    def usable(self) -> bool:
        """Whether it has exactly one target to use: exact or approximate."""
        return self.status in (Outcome.EXACT.value, Outcome.APPROXIMATE.value)

    @property
    def target(self) -> Any:
        """The one target of a usable resolution."""
        if not self.usable:
            raise MappingError(
                f"{self.native!r} is {self.status}: it has no single target"
            )
        return self.targets[0]


def _table(entries: Iterable[MappingEntry], kind: str) -> dict[str, MappingEntry]:
    table: dict[str, MappingEntry] = {}
    for entry in entries:
        if entry.native in table:
            raise MappingError(f"{kind} {entry.native!r} is declared twice")
        table[entry.native] = entry
    return table


def _resolve(table: Mapping[str, MappingEntry], native: str) -> Resolution:
    entry = table.get(native)
    if entry is None:
        return Resolution(native, UNDECLARED)
    return Resolution(native, entry.outcome.value, entry.targets, entry.rationale)


@dataclass(frozen=True)
class OntologyMapping:
    """A dataset's declared correspondence to the ontology.

    Attributes
    ----------
    dataset, version
        The dataset and release the mapping was written for.
    labels
        Native annotation labels to :class:`~sensor_modeling.states.BehaviouralState`.
    sensor_types
        Native sensor types to :class:`SensorSemantics`.
    locations
        Native locations to a room in :data:`CANONICAL_ROOMS`.
    description
        How the mapping was made, and by whom.
    """

    dataset: str
    version: str
    labels: tuple[MappingEntry, ...]
    sensor_types: tuple[MappingEntry, ...]
    locations: tuple[MappingEntry, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        """Check every target and index the tables."""
        if not str(self.dataset).strip() or not str(self.version).strip():
            raise MappingError("a mapping names its dataset and version")
        for name in ("labels", "sensor_types", "locations"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for entry in self.labels:
            if any(t not in MAPPABLE_STATES for t in entry.targets):
                raise MappingError(
                    f"label {entry.native!r} must map to latent states, not "
                    f"{list(entry.targets)}"
                )
        for entry in self.sensor_types:
            if any(not isinstance(t, SensorSemantics) for t in entry.targets):
                raise MappingError(
                    f"sensor type {entry.native!r} needs SensorSemantics"
                )
        for entry in self.locations:
            if any(t not in CANONICAL_ROOMS for t in entry.targets):
                raise MappingError(
                    f"location {entry.native!r} must map to one of "
                    f"{sorted(CANONICAL_ROOMS)}"
                )
        object.__setattr__(self, "_labels", _table(self.labels, "label"))
        object.__setattr__(self, "_types", _table(self.sensor_types, "sensor type"))
        object.__setattr__(self, "_locations", _table(self.locations, "location"))

    def label(self, native: str) -> Resolution:
        """What a native annotation label resolves to."""
        return _resolve(self._labels, native)  # type: ignore[attr-defined]

    def sensor_type(self, native: str) -> Resolution:
        """What a native sensor type resolves to."""
        return _resolve(self._types, native)  # type: ignore[attr-defined]

    def location(self, native: str | None) -> Resolution:
        """What a native location resolves to; :data:`MISSING` for none."""
        if native is None:
            return Resolution(None, MISSING)
        return _resolve(self._locations, native)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return the declaration, in a stable serialisable form."""

        def entries(table: Sequence[MappingEntry], encode: Any) -> list[dict[str, Any]]:
            return [
                {
                    "native": e.native,
                    "outcome": e.outcome.value,
                    "targets": [encode(t) for t in e.targets],
                    "rationale": e.rationale,
                }
                for e in sorted(table, key=lambda e: e.native)
            ]

        return {
            "schema": MAPPING_SCHEMA,
            "dataset": self.dataset,
            "version": self.version,
            "description": self.description,
            "labels": entries(self.labels, lambda t: t.value),
            "sensor_types": entries(self.sensor_types, lambda t: t.to_dict()),
            "locations": entries(self.locations, str),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OntologyMapping:
        """Rebuild a mapping written by :meth:`to_dict`."""
        if payload.get("schema") != MAPPING_SCHEMA:
            raise MappingError(f"not a {MAPPING_SCHEMA} declaration")

        def entries(
            rows: Sequence[Mapping[str, Any]], decode: Any
        ) -> tuple[MappingEntry, ...]:
            return tuple(
                MappingEntry(
                    row["native"],
                    Outcome(row["outcome"]),
                    tuple(decode(t) for t in row["targets"]),
                    row["rationale"],
                )
                for row in rows
            )

        return cls(
            dataset=payload["dataset"],
            version=payload["version"],
            description=payload.get("description", ""),
            labels=entries(payload["labels"], BehaviouralState),
            sensor_types=entries(payload["sensor_types"], SensorSemantics.from_dict),
            locations=entries(payload["locations"], str),
        )

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def write(self, path: Path) -> str:
        """Write the declaration as JSON; return its SHA-256."""
        text = json.dumps(
            {**self.to_dict(), "mapping_sha256": self.sha256()},
            indent=2,
            sort_keys=True,
        )
        Path(path).write_text(text + "\n", encoding="utf-8", newline="\n")
        return self.sha256()

    @classmethod
    def read(cls, path: Path, sha256: str | None = None) -> OntologyMapping:
        """Read a declaration, refusing one whose content does not match its digest."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        recorded = payload.pop("mapping_sha256", None)
        mapping = cls.from_dict(payload)
        if recorded is not None and recorded != mapping.sha256():
            raise MappingError(f"{path} does not match its recorded digest")
        if sha256 is not None and sha256 != mapping.sha256():
            raise MappingError(f"{path} is not the mapping with digest {sha256}")
        return mapping
