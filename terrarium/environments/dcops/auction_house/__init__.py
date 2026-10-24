"""Public AuctionHouse environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AuctionHouseEnvironment",
    "AuctionHouseTools",
    "AuctionHousePrompts",
]

_LAZY_ATTRS = {
    "AuctionHouseEnvironment": ".auction_house_env:AuctionHouseEnvironment",
    "AuctionHouseTools": ".auction_house_tools:AuctionHouseTools",
    "AuctionHousePrompts": ".auction_house_prompts:AuctionHousePrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.auction_house")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
