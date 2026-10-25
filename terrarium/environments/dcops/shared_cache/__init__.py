"""Public SharedCache environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "SharedCacheEnvironment",
    "SharedCacheTools",
    "SharedCachePrompts",
]

_LAZY_ATTRS = {
    "SharedCacheEnvironment": ".shared_cache_env:SharedCacheEnvironment",
    "SharedCacheTools": ".shared_cache_tools:SharedCacheTools",
    "SharedCachePrompts": ".shared_cache_prompts:SharedCachePrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.shared_cache")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
