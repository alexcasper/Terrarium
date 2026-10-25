#!/usr/bin/env python3
"""CLI and utility module for calibrating seeded Terrarium DCOP instance generators.

Wraps ``DifficultyAutotuner`` and ``DifficultyKnobSpec`` from
``terrarium.environments.dcops.difficulty_autotuner`` so researchers can calibrate
instance generator parameters (e.g. ``InstanceDifficultyAutotunerEnvironment`` or
``IsThisSeatTakenEnvironment``) to a target normalized performance ratio within a
small probe budget.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from terrarium.environments.dcops.difficulty_autotuner import (  # noqa: E402
    DifficultyAutotuner,
    DifficultyKnobSpec,
    InstanceDifficultyAutotunerEnvironment,
    InstanceDifficultyAutotunerTools,
)
from terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env import (  # noqa: E402
    IsThisSeatTakenEnvironment,
)


class _DummyProtocol:
    def __init__(self) -> None:
        self.environment = None


class _DummyToolLogger:
    def reset_log(self) -> None:
        pass


def make_autotuner_greedy_evaluator(
    skill_blend: float = 0.0,
) -> Callable[[Dict[str, Any]], float]:
    """Build a deterministic probe evaluator for ``InstanceDifficultyAutotunerEnvironment``.

    ``skill_blend in [0.0, 1.0]`` interpolates between pure unary-greedy choices (0.0)
    and exact optimal coordination (1.0), letting callers simulate solver populations
    of varying capability.
    """
    blend = min(1.0, max(0.0, float(skill_blend)))

    def _eval(config: Dict[str, Any]) -> float:
        cfg = copy.deepcopy(config)
        cfg.setdefault("simulation", {})["max_iterations"] = int(
            cfg.get("simulation", {}).get("max_iterations", 3)
        )
        cfg.setdefault("environment", {})["adaptive_enabled"] = False
        env = InstanceDifficultyAutotunerEnvironment(_DummyProtocol(), cfg, _DummyToolLogger())
        tools = InstanceDifficultyAutotunerTools(blackboard_manager=None, environment=env)

        for it in range(1, env.max_iterations + 1):
            inst = env.get_current_instance(it)
            greedy = inst["greedy_assignment"]
            optimal = inst["optimal_assignment"]
            for idx, name in enumerate(env.agent_names):
                use_opt = ((idx + 1) / len(env.agent_names)) <= blend
                chosen_color = optimal[name] if use_opt else greedy[name]
                tools.handle_tool_call(
                    "commit_assignment",
                    name,
                    {"color": chosen_color},
                    phase="execution",
                    iteration=it,
                )
            env.log_iteration(it)

        summary = env.get_final_summary()
        return float(summary["joint_reward_ratio"] or 0.0)

    return _eval


def make_seat_taken_evaluator(
    skill_blend: float = 0.25,
) -> Callable[[Dict[str, Any]], float]:
    """Build a deterministic probe evaluator for ``IsThisSeatTakenEnvironment``.

    Computes the normalized satisfaction ratio on the seeded layout induced by
    ``empty_seat_buffer`` and ``group_move_penalty``, adjusted by ``skill_blend``.
    """
    blend = min(1.0, max(0.0, float(skill_blend)))

    def _eval(config: Dict[str, Any]) -> float:
        cfg = copy.deepcopy(config)
        cfg.setdefault("environment", {})["name"] = "IsThisSeatTakenEnvironment"
        env = IsThisSeatTakenEnvironment(_DummyProtocol(), cfg, _DummyToolLogger())
        initial_joint = float(env.joint_reward({}))
        max_joint = float(env.max_joint_reward) if env.max_joint_reward else 1.0
        base_ratio = min(1.0, max(0.0, initial_joint / max(1e-6, max_joint)))

        buffer_val = float(cfg["environment"].get("empty_seat_buffer", 2))
        penalty_val = float(cfg["environment"].get("group_move_penalty", 0.25))
        slack_bonus = 0.045 * buffer_val - 0.35 * penalty_val
        realized = min(1.0, max(0.0, 0.55 * base_ratio + 0.35 + slack_bonus + 0.20 * blend))
        return round(realized, 4)

    return _eval


def calibrate_environment(
    environment_name: str = "InstanceDifficultyAutotunerEnvironment",
    target_score_ratio: float = 0.70,
    tolerance: float = 0.08,
    probe_budget: int = 5,
    seed: int = 42,
    num_agents: int = 4,
    skill_blend: float = 0.0,
    knobs: Optional[Sequence[DifficultyKnobSpec]] = None,
    eval_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
) -> Dict[str, Any]:
    """Run bounded-probe difficulty calibration for a seeded Terrarium environment."""
    base_config: Dict[str, Any] = {
        "simulation": {"max_iterations": 3, "seed": int(seed), "run_timestamp": "autotune"},
        "environment": {"name": environment_name},
        "communication_network": {"topology": "complete", "num_agents": int(num_agents)},
    }
    tuner = DifficultyAutotuner.for_environment(
        environment_name=environment_name,
        target_score_ratio=target_score_ratio,
        tolerance=tolerance,
        probe_budget=probe_budget,
        base_config=base_config,
        knobs=knobs,
    )
    if eval_fn is None:
        if environment_name == "IsThisSeatTakenEnvironment":
            eval_fn = make_seat_taken_evaluator(skill_blend=skill_blend)
        else:
            eval_fn = make_autotuner_greedy_evaluator(skill_blend=skill_blend)
    return tuner.calibrate(eval_fn=eval_fn)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate Terrarium DCOP generator difficulty to a target score ratio."
    )
    parser.add_argument(
        "--environment",
        default="InstanceDifficultyAutotunerEnvironment",
        choices=["InstanceDifficultyAutotunerEnvironment", "IsThisSeatTakenEnvironment"],
        help="Environment generator to calibrate.",
    )
    parser.add_argument("--target", type=float, default=0.70, help="Target score ratio in (0, 1].")
    parser.add_argument("--tolerance", type=float, default=0.08, help="Tolerance band around target.")
    parser.add_argument("--probe-budget", type=int, default=5, help="Max probe evaluations.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of agents.")
    parser.add_argument(
        "--skill-blend",
        type=float,
        default=0.0,
        help="Synthetic solver skill in [0.0, 1.0] when using built-in probe evaluator.",
    )
    args = parser.parse_args(argv)

    result = calibrate_environment(
        environment_name=args.environment,
        target_score_ratio=args.target,
        tolerance=args.tolerance,
        probe_budget=args.probe_budget,
        seed=args.seed,
        num_agents=args.num_agents,
        skill_blend=args.skill_blend,
    )
    summary = {k: v for k, v in result.items() if k != "best_config"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
