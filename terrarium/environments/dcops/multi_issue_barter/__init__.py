"""Public MultiIssueBarter environment namespace for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "MultiIssueBarterEnvironment",
    "MultiIssueBarterTools",
    "MultiIssueBarterPrompts",
]

_LAZY_ATTRS = {
    "MultiIssueBarterEnvironment": ".multi_issue_barter_env:MultiIssueBarterEnvironment",
    "MultiIssueBarterTools": ".multi_issue_barter_tools:MultiIssueBarterTools",
    "MultiIssueBarterPrompts": ".multi_issue_barter_prompts:MultiIssueBarterPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, "terrarium.environments.dcops.multi_issue_barter")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
