"""Public DisasterRelief environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "DisasterReliefEnvironment",
    "DisasterReliefTools",
    "DisasterReliefPrompts",
]

_LAZY_ATTRS = {
    "DisasterReliefEnvironment": ".disaster_relief_env:DisasterReliefEnvironment",
    "DisasterReliefTools": ".disaster_relief_tools:DisasterReliefTools",
    "DisasterReliefPrompts": ".disaster_relief_prompts:DisasterReliefPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.disaster_relief")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
