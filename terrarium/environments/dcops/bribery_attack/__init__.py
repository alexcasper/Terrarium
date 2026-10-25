"""Public BriberyAttack environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BriberyAttackEnvironment",
    "BriberyAttackTools",
    "BriberyAttackPrompts",
]

_LAZY_ATTRS = {
    "BriberyAttackEnvironment": ".bribery_attack_env:BriberyAttackEnvironment",
    "BriberyAttackTools": ".bribery_attack_tools:BriberyAttackTools",
    "BriberyAttackPrompts": ".bribery_attack_prompts:BriberyAttackPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(
        module_path, "terrarium.environments.dcops.bribery_attack"
    )
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
