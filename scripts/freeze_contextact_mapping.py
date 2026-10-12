"""Freeze the ContextAct@A4H ontology mapping, before any model is run on the data.

Writes :data:`~sensor_modeling.external.contextact.CONTEXTACT_MAPPING` to
:data:`~sensor_modeling.external.contextact.MAPPING_FILE` and prints its digest.
It refuses to replace a frozen file that differs: a changed mapping is a new
mapping, with a new file.

Usage::

    python scripts/freeze_contextact_mapping.py
"""

from __future__ import annotations

from pathlib import Path

from sensor_modeling.external import OntologyMapping
from sensor_modeling.external.contextact import CONTEXTACT_MAPPING, MAPPING_FILE


def main() -> None:
    path = Path(MAPPING_FILE)
    if path.exists():
        frozen = OntologyMapping.read(path)
        if frozen.sha256() != CONTEXTACT_MAPPING.sha256():
            raise SystemExit(f"{path} is frozen and differs from the declared mapping")
        print(f"{path} is already frozen, sha256 {frozen.sha256()}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = CONTEXTACT_MAPPING.write(path)
    print(f"written {path}, sha256 {digest}")


if __name__ == "__main__":
    main()
