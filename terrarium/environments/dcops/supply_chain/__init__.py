"""Public SupplyChain environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "SupplyChainEnvironment",
    "SupplyChainTools",
    "SupplyChainPrompts",
]

_LAZY_ATTRS = {
    "SupplyChainEnvironment": ".supply_chain_env:SupplyChainEnvironment",
    "SupplyChainTools": ".supply_chain_tools:SupplyChainTools",
    "SupplyChainPrompts": ".supply_chain_prompts:SupplyChainPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.supply_chain")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
