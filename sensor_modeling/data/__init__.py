"""Data loading, preprocessing, validation, and simulation utilities."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import loaders, preprocessing, synthetic, validation

__all__ = ["loaders", "preprocessing", "validation", "synthetic"]


def __getattr__(name: str) -> ModuleType:
    """Lazily import public data submodules to avoid ingestion import cycles."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
