"""Guards against defects that only appear on some Python versions.

The observation layer once used ``MappingProxyType({})`` as a dataclass field
default. That is accepted on 3.10 (which only rejects ``list``/``dict``/``set``)
and on 3.12+ (where ``mappingproxy`` became hashable), but rejected on 3.11,
whose mutable-default check asks whether the default's type is hashable. The
failure happened at class-definition time, so every module that imported
``Observation`` failed to import and 21 test files could not even be collected.

Running the suite on a single interpreter cannot catch that class of bug, so
these tests assert the underlying invariant instead of the symptom.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
from collections.abc import Mapping, Set
from datetime import date, datetime, timedelta
from enum import Enum

import pytest

import sensor_modeling

# Defaults that are safe to share between instances on every supported
# interpreter: immutable, hashable, and hashable *consistently* across versions.
IMMUTABLE = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    tuple,
    frozenset,
    Enum,
    timedelta,
    datetime,
    date,
)


def _modules() -> list[str]:
    names = [sensor_modeling.__name__]
    for info in pkgutil.walk_packages(
        sensor_modeling.__path__, prefix=f"{sensor_modeling.__name__}."
    ):
        names.append(info.name)
    return sorted(names)


MODULES = _modules()


@pytest.mark.parametrize("name", MODULES)
def test_every_module_imports(name: str) -> None:
    """A module that cannot be imported cannot be tested."""
    importlib.import_module(name)


def test_the_package_exposes_a_version() -> None:
    assert isinstance(sensor_modeling.__version__, str)
    assert sensor_modeling.__version__


def _dataclass_fields():
    for name in MODULES:
        module = importlib.import_module(name)
        for attribute in vars(module).values():
            if not isinstance(attribute, type):
                continue
            if not dataclasses.is_dataclass(attribute):
                continue
            if attribute.__module__ != name:
                continue  # only report it where it is defined
            for field in dataclasses.fields(attribute):
                yield attribute, field


def test_no_dataclass_shares_a_mutable_default() -> None:
    """Mutable defaults are shared between instances and are version-fragile.

    ``default_factory`` is the portable spelling. This catches the 3.11
    ``mappingproxy`` regression on any interpreter, including ones where the
    dataclasses machinery itself would accept it.
    """
    offenders = []
    for owner, field in _dataclass_fields():
        default = field.default
        if default is dataclasses.MISSING:
            continue
        if isinstance(default, IMMUTABLE):
            continue
        if isinstance(default, (Mapping, Set, list, bytearray)):
            offenders.append(
                f"{owner.__module__}.{owner.__qualname__}.{field.name} "
                f"= {type(default).__name__}"
            )

    assert not offenders, (
        "these fields use a shared container as a default; use default_factory: "
        + ", ".join(offenders)
    )
