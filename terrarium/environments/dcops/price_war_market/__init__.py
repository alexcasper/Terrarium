"""Public PriceWarMarket environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "PriceWarMarketEnvironment",
    "PriceWarMarketTools",
    "PriceWarMarketPrompts",
]

_LAZY_ATTRS = {
    "PriceWarMarketEnvironment": ".price_war_market_env:PriceWarMarketEnvironment",
    "PriceWarMarketTools": ".price_war_market_tools:PriceWarMarketTools",
    "PriceWarMarketPrompts": ".price_war_market_prompts:PriceWarMarketPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.price_war_market")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
