"""Public NetworkRouting environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "NetworkRoutingEnvironment",
    "NetworkRoutingTools",
    "NetworkRoutingPrompts",
]

_LAZY_ATTRS = {
    "NetworkRoutingEnvironment": ".network_routing_env:NetworkRoutingEnvironment",
    "NetworkRoutingTools": ".network_routing_tools:NetworkRoutingTools",
    "NetworkRoutingPrompts": ".network_routing_prompts:NetworkRoutingPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.network_routing")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
