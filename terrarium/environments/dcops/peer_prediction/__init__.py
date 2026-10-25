"""Public PeerPredictionReporting environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "PeerPredictionReportingEnvironment",
    "PeerPredictionReportingTools",
    "PeerPredictionReportingPrompts",
]

_LAZY_ATTRS = {
    "PeerPredictionReportingEnvironment": ".peer_prediction_env:PeerPredictionReportingEnvironment",
    "PeerPredictionReportingTools": ".peer_prediction_tools:PeerPredictionReportingTools",
    "PeerPredictionReportingPrompts": ".peer_prediction_prompts:PeerPredictionReportingPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.peer_prediction")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
