"""Public SmartGridStorageArbitrage environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "SmartGridStorageEnvironment",
    "SmartGridStorageTools",
    "SmartGridStoragePrompts",
]

_LAZY_ATTRS = {
    "SmartGridStorageEnvironment": ".smart_grid_storage_env:SmartGridStorageEnvironment",
    "SmartGridStorageTools": ".smart_grid_storage_tools:SmartGridStorageTools",
    "SmartGridStoragePrompts": ".smart_grid_storage_prompts:SmartGridStoragePrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.smart_grid_storage")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
