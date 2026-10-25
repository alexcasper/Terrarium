"""Public InstanceDifficultyAutotuner environment and calibration utilities for Terrarium."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "InstanceDifficultyAutotunerEnvironment",
    "InstanceDifficultyAutotunerTools",
    "InstanceDifficultyAutotunerPrompts",
    "DifficultyAutotuner",
    "DifficultyKnobSpec",
]

_LAZY_ATTRS = {
    "InstanceDifficultyAutotunerEnvironment": ".difficulty_autotuner_env:InstanceDifficultyAutotunerEnvironment",
    "DifficultyAutotuner": ".difficulty_autotuner_env:DifficultyAutotuner",
    "DifficultyKnobSpec": ".difficulty_autotuner_env:DifficultyKnobSpec",
    "InstanceDifficultyAutotunerTools": ".difficulty_autotuner_tools:InstanceDifficultyAutotunerTools",
    "InstanceDifficultyAutotunerPrompts": ".difficulty_autotuner_prompts:InstanceDifficultyAutotunerPrompts",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(
        module_path, "terrarium.environments.dcops.difficulty_autotuner"
    )
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
