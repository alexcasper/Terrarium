"""Public FederatedUpdate environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "FederatedUpdateEnvironment",
    "FederatedUpdateTools",
    "FederatedUpdatePrompts",
]

_LAZY_ATTRS = {
    "FederatedUpdateEnvironment": ".federated_update_env:FederatedUpdateEnvironment",
    "FederatedUpdateTools": ".federated_update_tools:FederatedUpdateTools",
    "FederatedUpdatePrompts": ".federated_update_prompts:FederatedUpdatePrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.federated_update")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
