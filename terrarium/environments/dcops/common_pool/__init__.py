"""Public CommonPool environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "CommonPoolEnvironment",
    "CommonPoolTools",
    "CommonPoolPrompts",
]

_LAZY_ATTRS = {
    "CommonPoolEnvironment": ".common_pool_env:CommonPoolEnvironment",
    "CommonPoolTools": ".common_pool_tools:CommonPoolTools",
    "CommonPoolPrompts": ".common_pool_prompts:CommonPoolPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.common_pool")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
