"""Public PublicGoods environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "PublicGoodsEnvironment",
    "PublicGoodsTools",
    "PublicGoodsPrompts",
]

_LAZY_ATTRS = {
    "PublicGoodsEnvironment": ".public_goods_env:PublicGoodsEnvironment",
    "PublicGoodsTools": ".public_goods_tools:PublicGoodsTools",
    "PublicGoodsPrompts": ".public_goods_prompts:PublicGoodsPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.public_goods")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
