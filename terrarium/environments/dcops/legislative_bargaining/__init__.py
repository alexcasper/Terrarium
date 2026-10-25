"""Public LegislativeBargaining environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "LegislativeBargainingEnvironment",
    "LegislativeBargainingTools",
    "LegislativeBargainingPrompts",
]

_LAZY_ATTRS = {
    "LegislativeBargainingEnvironment": ".legislative_bargaining_env:LegislativeBargainingEnvironment",
    "LegislativeBargainingTools": ".legislative_bargaining_tools:LegislativeBargainingTools",
    "LegislativeBargainingPrompts": ".legislative_bargaining_prompts:LegislativeBargainingPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(
        module_path, "terrarium.environments.dcops.legislative_bargaining"
    )
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
