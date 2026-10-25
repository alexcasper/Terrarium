"""Public CoalitionFormation environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "CoalitionFormationEnvironment",
    "CoalitionFormationTools",
    "CoalitionFormationPrompts",
]

_LAZY_ATTRS = {
    "CoalitionFormationEnvironment": ".coalition_formation_env:CoalitionFormationEnvironment",
    "CoalitionFormationTools": ".coalition_formation_tools:CoalitionFormationTools",
    "CoalitionFormationPrompts": ".coalition_formation_prompts:CoalitionFormationPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(
        module_path, "terrarium.environments.dcops.coalition_formation"
    )
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
