"""Public SybilCoalition environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "SybilCoalitionEnvironment",
    "SybilCoalitionTools",
    "SybilCoalitionPrompts",
]

_LAZY_ATTRS = {
    "SybilCoalitionEnvironment": ".sybil_coalition_env:SybilCoalitionEnvironment",
    "SybilCoalitionTools": ".sybil_coalition_tools:SybilCoalitionTools",
    "SybilCoalitionPrompts": ".sybil_coalition_prompts:SybilCoalitionPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.sybil_coalition")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
