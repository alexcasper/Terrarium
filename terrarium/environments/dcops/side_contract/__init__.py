"""Public SideContract environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "SideContractEnvironment",
    "SideContractTools",
    "SideContractPrompts",
]

_LAZY_ATTRS = {
    "SideContractEnvironment": ".side_contract_env:SideContractEnvironment",
    "SideContractTools": ".side_contract_tools:SideContractTools",
    "SideContractPrompts": ".side_contract_prompts:SideContractPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(
        module_path, "terrarium.environments.dcops.side_contract"
    )
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
