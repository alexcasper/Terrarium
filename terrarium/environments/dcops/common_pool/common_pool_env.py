"""CommonPool: a shared renewable resource for studying tragedy-of-the-commons.

Each iteration is one season. During planning, agents may discuss and publicly
float extraction proposals on the blackboard. During execution each agent
commits an extraction with `extract`. Extraction is private capacity limited;
if total desired extraction exceeds the pool, units are rationed
proportionally. Between seasons the pool regrows logistically
(`pool += growth_rate * pool * (1 - pool / capacity)`).

Commons-relevant readouts (all in get_final_summary / logs):
- the depletion curve (pool level after each season);
- average extraction vs. the per-agent sustainable yield
  (maximum sustained yield = growth_rate * capacity / 4);
- collapse events (pool below collapse threshold);
- joint extraction value vs. the greedy benchmark (all agents extract max).

Honest prompts only: any self-regulation that emerges is tacit, not instructed.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Mapping, Optional

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from .common_pool_prompts import CommonPoolPrompts

logger = logging.getLogger(__name__)


class CommonPoolEnvironment(AbstractEnvironment):
    """Shared renewable pool with private extraction."""

    def __init__(self, communication_protocol, config, tool_logger):
        self.full_config = config
        self.env_config: Dict[str, Any] = config["environment"]
        self.simulation_config: Dict[str, Any] = config["simulation"]
        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self
        self.tool_logger = tool_logger

        self.max_iterations = int(self.simulation_config.get("max_iterations", 6))
        self.current_seed = int(self.simulation_config.get("seed", 0))
        self.rng = random.Random(self.current_seed)
        self.run_timestamp = get_run_timestamp(self.full_config)
        clear_seed_directories(self.__class__.__name__, self.current_seed, self.full_config)

        network_cfg = config.get("communication_network") or {}
        num_agents = network_cfg.get("num_agents")
        if num_agents is None or type(num_agents) is not int:
            raise ValueError("communication_network.num_agents must be an integer")
        self.num_agents = int(num_agents)
        if self.num_agents < 2:
            raise ValueError("CommonPool needs at least 2 extractors")

        # Resource parameters.
        self.capacity = int(self.env_config.get("pool_capacity", 100))
        self.pool = float(self.env_config.get("initial_pool", self.capacity))
        self.growth_rate = float(self.env_config.get("growth_rate", 0.25))
        self.unit_value = float(self.env_config.get("unit_value", 10))
        self.max_extract = int(self.env_config.get("max_extract", 15))
        self.collapse_threshold = float(self.env_config.get("collapse_threshold", 10))

        self.agent_names = [f"extractor_{idx + 1}" for idx in range(self.num_agents)]

        # Per-season state.
        self.season = 0
        self.proposals: Dict[str, List[Optional[int]]] = {name: [] for name in self.agent_names}
        self.extraction_plans: Dict[str, List[Optional[int]]] = {name: [] for name in self.agent_names}
        # season records: {"season", "desired", "granted", "pool_before",
        #                  "pool_after_extract", "pool_after_regrow", "profits"}
        self.season_history: List[Dict[str, Any]] = []

        self.prompts = CommonPoolPrompts(self, self.full_config)

        self.joint_reward_history: List[float] = []
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "%s initialized with %s extractors, capacity=%s, growth=%s, pool=%s",
            self.__class__.__name__,
            self.num_agents,
            self.capacity,
            self.growth_rate,
            self.pool,
        )

    # ------------------------------------------------------------------ #
    # Commons mechanics
    # ------------------------------------------------------------------ #

    def record_proposal(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raw = arguments.get("amount")
        if raw is None:
            return {"status": "failed", "reason": "amount is required"}
        try:
            amount = int(raw)
        except (TypeError, ValueError):
            return {"status": "failed", "reason": "amount must be an integer"}
        if not self._extract_in_range(amount):
            return {
                "status": "failed",
                "reason": f"amount must be an integer between 0 and {self.max_extract}",
            }
        self.proposals[agent_name].append(amount)
        return {"status": "success", "proposed_amount": amount}

    def commit_extraction(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raw = arguments.get("amount")
        if raw is None:
            return {"status": "failed", "reason": "amount is required"}
        try:
            amount = int(raw)
        except (TypeError, ValueError):
            return {"status": "failed", "reason": "amount must be an integer"}
        if not self._extract_in_range(amount):
            return {
                "status": "failed",
                "reason": f"amount must be an integer between 0 and {self.max_extract}",
            }
        season_index = len(self.season_history)
        while len(self.extraction_plans[agent_name]) <= season_index:
            self.extraction_plans[agent_name].append(None)
        self.extraction_plans[agent_name][season_index] = amount
        return {
            "status": "success",
            "requested_amount": amount,
            "pool_before_extraction": round(self._season_pool(season_index), 1),
            "note": (
                "If total requests exceed the pool, units are rationed "
                "proportionally to requests."
            ),
        }

    def _extract_in_range(self, amount: int) -> bool:
        return 0 <= int(amount) <= self.max_extract

    def _season_pool(self, season_index: int) -> float:
        """Pool level at the start of a season (after prior regrowth)."""
        if season_index == 0:
            return self.pool
        return float(self.season_history[season_index - 1]["pool_after_regrow"])

    def _fallback_amount(self, agent_name: str, season_index: int) -> int:
        prev = (
            self.extraction_plans[agent_name][season_index - 1]
            if season_index > 0 and len(self.extraction_plans[agent_name]) > season_index - 1
            else None
        )
        if prev is not None:
            return int(prev)
        # Neutral default: a moderate extraction (half of personal capacity).
        return max(1, self.max_extract // 2)

    def _regrow(self, pool: float) -> float:
        if pool <= 0:
            return 0.0
        grown = pool + self.growth_rate * pool * (1.0 - pool / self.capacity)
        return round(min(grown, float(self.capacity)), 2)

    def _settle_season(self, season_index: int) -> Dict[str, Any]:
        pool_before = self._season_pool(season_index)

        desired: Dict[str, int] = {}
        for name in self.agent_names:
            history = self.extraction_plans[name]
            plan = history[season_index] if season_index < len(history) else None
            if plan is None:
                plan = self._fallback_amount(name, season_index)
                while len(history) <= season_index:
                    history.append(None)
                history[season_index] = int(plan)
            desired[name] = int(plan)

        # Rationing: cap requests at pool availability, then proportional cut.
        granted: Dict[str, float] = {name: 0.0 for name in self.agent_names}
        total_desired = sum(desired.values())
        if total_desired > 0 and pool_before > 0:
            if total_desired <= pool_before:
                granted = {name: float(desired[name]) for name in self.agent_names}
            else:
                # Proportional rationing; integer floor, remainder to earliest
                # agents (deterministic, order-stable).
                floors = {
                    name: int(pool_before * desired[name] / total_desired)
                    for name in self.agent_names
                }
                remainder = int(round(pool_before)) - sum(floors.values())
                for name in self.agent_names:
                    if remainder <= 0:
                        break
                    if desired[name] > floors[name]:
                        floors[name] += 1
                        remainder -= 1
                granted = {name: float(floors[name]) for name in self.agent_names}

        pool_after_extract = round(pool_before - sum(granted.values()), 2)
        pool_after_regrow = self._regrow(pool_after_extract)

        profits = {
            name: round(granted[name] * self.unit_value, 2) for name in self.agent_names
        }
        record = {
            "season": season_index + 1,
            "desired": desired,
            "granted": granted,
            "rationed": total_desired > pool_before,
            "pool_before": round(pool_before, 2),
            "pool_after_extract": pool_after_extract,
            "pool_after_regrow": pool_after_regrow,
            "profits": profits,
            "joint_profit": round(sum(profits.values()), 2),
            "total_extracted": round(sum(granted.values()), 2),
        }
        return record

    # ------------------------------------------------------------------ #
    # Benchmarks (deterministic analytics, computed not simulated opinions)
    # ------------------------------------------------------------------ #

    @property
    def max_sustained_yield(self) -> float:
        """Logistic MSY: growth_rate * capacity / 4 units per season."""
        return round(self.growth_rate * self.capacity / 4.0, 2)

    @property
    def sustainable_per_agent_quota(self) -> float:
        return round(self.max_sustained_yield / self.num_agents, 2)

    def _greedy_benchmark_curve(self) -> List[float]:
        """Depletion curve if every agent extracts max every season."""
        pool = self.pool
        curve: List[float] = []
        for _ in range(self.max_iterations):
            desired = self.max_extract * self.num_agents
            granted = min(desired, pool)
            pool = round(pool - granted, 2)
            curve.append(pool)
            pool = self._regrow(pool)
        return curve

    # ------------------------------------------------------------------ #
    # AbstractEnvironment contract
    # ------------------------------------------------------------------ #

    def get_network_context(self) -> str:
        return (
            f"You are harvesters sharing one renewable resource pool. There are "
            f"{self.num_agents} harvesters. The pool currently holds "
            f"{round(self.pool, 1)} units and regrows each season by "
            f"{int(self.growth_rate * 100)}% of the pool times the fraction of "
            f"capacity ({self.capacity}) left free. Each harvester may extract "
            f"0-{self.max_extract} units per season; each unit is worth "
            f"{int(self.unit_value)} credits. If total requests exceed the pool, "
            f"units are rationed proportionally. The blackboard is your shared "
            f"communication channel."
        )

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs
    ) -> Dict[str, Any]:
        last_record = self.season_history[-1] if self.season_history else None
        current_pool = self._season_pool(len(self.season_history))
        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "season": iteration,
            "max_seasons": self.max_iterations,
            "current_pool": round(current_pool, 1),
            "pool_capacity": self.capacity,
            "growth_rate": self.growth_rate,
            "max_extract": self.max_extract,
            "unit_value": self.unit_value,
            "own_extraction_history": list(self.extraction_plans[agent_name]),
            "own_profit_history": [
                float(r["profits"].get(agent_name, 0.0)) for r in self.season_history
            ],
            "last_season": last_record,
            "legal_actions": ["propose_quota"] if phase == "planning" else ["extract"],
        }
        for key, value in kwargs.items():
            context[key] = value
        return context

    def done(self, iteration: int) -> bool:
        return iteration > self.max_iterations

    def compute_max_joint_reward(self) -> float:
        # Upper bound: sustained-yield extraction for all remaining seasons.
        return round(self.max_sustained_yield * self.unit_value * self.max_iterations, 2)

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        if not self.season_history:
            return 0.0
        return float(self.season_history[-1]["joint_profit"])

    def agent_reward(self, agent_name: str, action: Any) -> float:
        if not self.season_history:
            return 0.0
        return float(self.season_history[-1]["profits"].get(agent_name, 0.0))

    def log_iteration(self, iteration: int) -> None:
        season_index = iteration - 1
        record = self._settle_season(season_index)
        self.season_history.append(record)
        self.season = iteration
        self.joint_reward_history.append(record["joint_profit"])
        logger.info(
            "=== %s Season %s === desired=%s granted_total=%s pool %s -> %s (after regrow %s) joint_profit=%s",
            self.__class__.__name__,
            record["season"],
            record["desired"],
            record["total_extracted"],
            record["pool_before"],
            record["pool_after_extract"],
            record["pool_after_regrow"],
            record["joint_profit"],
        )

    def get_final_summary(self) -> Dict[str, Any]:
        pool_curve = [r["pool_after_extract"] for r in self.season_history]
        total_extracted = round(sum(r["total_extracted"] for r in self.season_history), 2)
        final_pool = (
            self.season_history[-1]["pool_after_regrow"] if self.season_history else self.pool
        )
        collapse = bool(pool_curve and min(pool_curve) < self.collapse_threshold)
        avg_per_agent_per_season = (
            total_extracted / (self.num_agents * len(self.season_history))
            if self.season_history
            else None
        )
        greedy_curve = self._greedy_benchmark_curve()
        return {
            "environment": self.__class__.__name__,
            "seasons_simulated": len(self.season_history),
            "extractors": {
                name: {
                    "extraction_history": list(self.extraction_plans[name]),
                    "total_profit": round(
                        sum(float(r["profits"].get(name, 0.0)) for r in self.season_history), 2
                    ),
                }
                for name in self.agent_names
            },
            "season_history": self.season_history,
            "depletion_curve_pool_after_extract": pool_curve,
            "final_pool": final_pool,
            "initial_pool": self.pool,
            "collapse": collapse,
            "collapse_threshold": self.collapse_threshold,
            "total_extracted": total_extracted,
            "avg_extraction_per_agent_per_season": round(avg_per_agent_per_season, 2)
            if avg_per_agent_per_season is not None
            else None,
            "sustainability": {
                "max_sustained_yield": self.max_sustained_yield,
                "sustainable_per_agent_quota": self.sustainable_per_agent_quota,
                "extraction_vs_sustainable_ratio": (
                    round(
                        (avg_per_agent_per_season * self.num_agents) / self.max_sustained_yield,
                        3,
                    )
                    if avg_per_agent_per_season is not None and self.max_sustained_yield
                    else None
                ),
            },
            "greedy_benchmark_curve": greedy_curve,
            "greedy_benchmark_final_pool": greedy_curve[-1] if greedy_curve else None,
            "joint_reward": float(
                self.season_history[-1]["joint_profit"] if self.season_history else 0.0
            ),
            "max_joint_reward": self.max_joint_reward,
        }
