"""Public AdAuctionGSP environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AdAuctionGSPEnvironment",
    "AdAuctionGSPTools",
    "AdAuctionGSPPrompts",
]

_LAZY_ATTRS = {
    "AdAuctionGSPEnvironment": ".ad_auction_gsp_env:AdAuctionGSPEnvironment",
    "AdAuctionGSPTools": ".ad_auction_gsp_tools:AdAuctionGSPTools",
    "AdAuctionGSPPrompts": ".ad_auction_gsp_prompts:AdAuctionGSPPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.ad_auction_gsp")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
