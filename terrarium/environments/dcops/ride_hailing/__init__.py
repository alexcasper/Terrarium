"""Public RideHailingDispatch environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "RideHailingDispatchEnvironment",
    "RideHailingDispatchTools",
    "RideHailingPrompts",
]

_LAZY_ATTRS = {
    "RideHailingDispatchEnvironment": ".ride_hailing_env:RideHailingDispatchEnvironment",
    "RideHailingDispatchTools": ".ride_hailing_tools:RideHailingDispatchTools",
    "RideHailingPrompts": ".ride_hailing_prompts:RideHailingPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.ride_hailing")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
