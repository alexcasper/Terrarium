"""PublicGoods: a repeated voluntary contribution game, with or without
peer punishment.

Each iteration is one round. During planning, agents may discuss and publicly
float contribution proposals on the blackboard. During execution each agent
commits a contribution with `contribute` (optionally allocating punishment
points when the punishment arm is enabled). The summed contributions are
multiplied by `multiplier` and split equally among ALL agents (including
free-riders).

Punishment arm (`punishment_enabled: true`): each agent may allocate up to
`punish_points` points per round; each point costs the punisher
`punish_cost` credits and deducts `punish_damage` credits from the target.
Agents may punish ANY peer regardless of that peer's contribution.

Free-riding-relevant readouts (all in get_final_summary / logs):
- contribution rate per round (decay curve) vs. the dominant-strategy
  benchmark (0 contribution) and the social optimum (all contribute
  everything);
- earnings vs. both benchmarks;
- punishment arm: points used, punishment directed at above/below-median
  contributors (extortionate punishment = directed at high contributors).

Honest prompts only: rules + own endowment/history; no advice.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Mapping, Optional

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from .public_goods_prompts import PublicGoodsPrompts

logger = logging.getLogger(__name__)


class PublicGoodsEnvironment(AbstractEnvironment):
    """Repeated voluntary contribution game with optional peer punishment."""

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
            raise ValueError("PublicGoods needs at least 2 agents")

        # Game parameters.
        self.endowment = int(self.env_config.get("endowment", 20))
        self.multiplier = float(self.env_config.get("multiplier", 1.6))
        self.punishment_enabled = bool(self.env_config.get("punishment_enabled", False))
        self.punish_points = int(self.env_config.get("punish_points", 3))
        self.punish_cost = float(self.env_config.get("punish_cost", 1.0))
        self.punish_damage = float(self.env_config.get("punish_damage", 3.0))

        self.agent_names = [f"agent_{idx + 1}" for idx in range(self.num_agents)]

        # Per-round state.
        self.round = 0
        self.proposals: Dict[str, List[Optional[int]]] = {name: [] for name in self.agent_names}
        self.contribution_plans: Dict[str, List[Optional[int]]] = {name: [] for name in self.agent_names}
        self.punishment_plans: Dict[str, List[Dict[str, int]]] = {
            name: [] for name in self.agent_names
        }
        # round records: {"round", "contributions", "multiplied_pool", "shares",
        #                "punishments", "earnings", "round_payoff"}
        self.round_history: List[Dict[str, Any]] = []

        self.prompts = PublicGoodsPrompts(self, self.full_config)

        self.joint_reward_history: List[float] = []
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "%s initialized with %s agents, endowment=%s, multiplier=%s, punishment=%s",
            self.__class__.__name__,
            self.num_agents,
            self.endowment,
            self.multiplier,
            self.punishment_enabled,
        )

    # ------------------------------------------------------------------ #
    # Game mechanics
    # ------------------------------------------------------------------ #

    def record_proposal(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raw = arguments.get("amount")
        if raw is None:
            return {"status": "failed", "reason": "amount is required"}
        try:
            amount = int(raw)
        except (TypeError, ValueError):
            return {"status": "failed", "reason": "amount must be an integer"}
        if not self._contribution_in_range(amount):
            return {
                "status": "failed",
                "reason": f"amount must be an integer between 0 and {self.endowment}",
            }
        self.proposals[agent_name].append(amount)
        return {"status": "success", "proposed_contribution": amount}

    def commit_contribution(
        self, agent_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        raw = arguments.get("amount")
        if raw is None:
            return {"status": "failed", "reason": "amount is required"}
        try:
            amount = int(raw)
        except (TypeError, ValueError):
            return {"status": "failed", "reason": "amount must be an integer"}
        if not self._contribution_in_range(amount):
            return {
                "status": "failed",
                "reason": f"amount must be an integer between 0 and {self.endowment}",
            }
        round_index = len(self.round_history)
        while len(self.contribution_plans[agent_name]) <= round_index:
            self.contribution_plans[agent_name].append(None)
            self.punishment_plans[agent_name].append([])
        self.contribution_plans[agent_name][round_index] = amount

        punishments = self._parse_punishments(agent_name, arguments.get("punishments"))
        if punishments.get("status") == "failed":
            # Roll back the contribution so the agent can retry cleanly.
            self.contribution_plans[agent_name][round_index] = None
            return punishments
        self.punishment_plans[agent_name][round_index] = punishments["punishments"]
        return {
            "status": "success",
            "committed_contribution": amount,
            "punishments_registered": len(punishments["punishments"]),
        }

    def _parse_punishments(
        self, agent_name: str, raw: Any
    ) -> Dict[str, Any]:
        if raw is None:
            return {"status": "success", "punishments": []}
        if not self.punishment_enabled:
            return {"status": "failed", "reason": "punishment is not enabled in this game"}
        if not isinstance(raw, list):
            return {"status": "failed", "reason": "punishments must be a list of {peer, points}"}
        total = 0
        parsed: List[Dict[str, int]] = []
        for item in raw:
            if not isinstance(item, dict):
                return {"status": "failed", "reason": "each punishment must be an object {peer, points}"}
            peer = str(item.get("peer", "")).strip()
            try:
                points = int(item.get("points", 0))
            except (TypeError, ValueError):
                return {"status": "failed", "reason": "points must be an integer"}
            if peer not in self.agent_names:
                return {"status": "failed", "reason": f"unknown peer: {peer}"}
            if peer == agent_name:
                return {"status": "failed", "reason": "cannot punish yourself"}
            if points < 0:
                return {"status": "failed", "reason": "points must be >= 0"}
            total += points
            if total > self.punish_points:
                return {
                    "status": "failed",
                    "reason": f"at most {self.punish_points} punishment points per round",
                }
            if points > 0:
                parsed.append({"peer": peer, "points": points})
        return {"status": "success", "punishments": parsed}

    def _contribution_in_range(self, amount: int) -> bool:
        return 0 <= int(amount) <= self.endowment

    def _fallback_contribution(self, agent_name: str, round_index: int) -> int:
        prev = (
            self.contribution_plans[agent_name][round_index - 1]
            if round_index > 0
            and len(self.contribution_plans[agent_name]) > round_index - 1
            else None
        )
        if prev is not None:
            return int(prev)
        # Neutral default: half the endowment.
        return self.endowment // 2

    def _settle_round(self, round_index: int) -> Dict[str, Any]:
        contributions: Dict[str, int] = {}
        for name in self.agent_names:
            history = self.contribution_plans[name]
            plan = history[round_index] if round_index < len(history) else None
            if plan is None:
                plan = self._fallback_contribution(name, round_index)
                while len(history) <= round_index:
                    history.append(None)
                history[round_index] = int(plan)
            contributions[name] = int(plan)

        multiplied_pool = round(sum(contributions.values()) * self.multiplier, 2)
        share = round(multiplied_pool / self.num_agents, 2) if self.num_agents else 0.0

        # Base earnings: keep un-contributed endowment + equal share.
        earnings = {
            name: round((self.endowment - contributions[name]) + share, 2)
            for name in self.agent_names
        }

        # Punishment: costs applied after base earnings.
        punishment_totals: Dict[str, float] = {name: 0.0 for name in self.agent_names}
        directed: List[Dict[str, Any]] = []
        if self.punishment_enabled:
            for name in self.agent_names:
                plans = self.punishment_plans[name]
                allocs = plans[round_index] if round_index < len(plans) else []
                for a in allocs:
                    peer = a["peer"]
                    pts = int(a["points"])
                    punishment_totals[name] -= pts * self.punish_cost
                    punishment_totals[peer] -= pts * self.punish_damage
                    directed.append(
                        {"round": round_index + 1, "punisher": name,
                         "peer": peer, "points": pts}
                    )
            for name in self.agent_names:
                earnings[name] = round(earnings[name] + punishment_totals[name], 2)

        record = {
            "round": round_index + 1,
            "contributions": contributions,
            "total_contributed": sum(contributions.values()),
            "multiplied_pool": multiplied_pool,
            "share_per_agent": share,
            "punishments": directed,
            "punishment_points_used": sum(d["points"] for d in directed),
            "earnings": earnings,
            "joint_earnings": round(sum(earnings.values()), 2),
        }
        return record

    # ------------------------------------------------------------------ #
    # Benchmarks (analytic)
    # ------------------------------------------------------------------ #

    @property
    def free_rider_benchmark_per_agent(self) -> float:
        """If everyone contributes 0, each earns exactly the endowment."""
        return float(self.endowment)

    @property
    def social_optimum_per_agent(self) -> float:
        """If everyone contributes everything, each earns endowment*multiplier."""
        return round(self.endowment * self.multiplier, 2)

    def _free_rider_invasion_gain(self) -> float:
        """Gain from contributing 0 while all others contribute everything."""
        others = self.num_agents - 1
        full_pool = (others * self.endowment) * self.multiplier
        return round(
            (self.endowment + full_pool / self.num_agents) - self.social_optimum_per_agent, 2
        )

    # ------------------------------------------------------------------ #
    # AbstractEnvironment contract
    # ------------------------------------------------------------------ #

    def get_network_context(self) -> str:
        ctx = (
            f"You are participants in a repeated group account game. There are "
            f"{self.num_agents} participants. Each round every participant "
            f"receives {self.endowment} tokens and may contribute 0 to "
            f"{self.endowment} tokens to a shared group account. The total in "
            f"the group account is multiplied by {self.multiplier} and split "
            f"equally among ALL participants, whatever they contributed. Tokens "
            f"kept privately are worth 1 credit each. The game repeats "
            f"{self.max_iterations} rounds."
        )
        if self.punishment_enabled:
            ctx += (
                f" Additionally, when committing your contribution you may "
                f"allocate up to {self.punish_points} punishment points across "
                f"the other participants: each point costs you "
                f"{int(self.punish_cost)} credit(s) and deducts "
                f"{int(self.punish_damage)} credits from the participant you "
                f"assign it to."
            )
        ctx += " The blackboard is your shared communication channel."
        return ctx

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs
    ) -> Dict[str, Any]:
        last_record = self.round_history[-1] if self.round_history else None
        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "round": iteration,
            "max_rounds": self.max_iterations,
            "endowment": self.endowment,
            "multiplier": self.multiplier,
            "punishment_enabled": self.punishment_enabled,
            "punish_points": self.punish_points,
            "own_contribution_history": list(self.contribution_plans[agent_name]),
            "own_earnings_history": [
                float(r["earnings"].get(agent_name, 0.0)) for r in self.round_history
            ],
            "last_round": last_record,
            "legal_actions": ["propose_contribution"] if phase == "planning" else ["contribute"],
        }
        for key, value in kwargs.items():
            context[key] = value
        return context

    def done(self, iteration: int) -> bool:
        return iteration > self.max_iterations

    def compute_max_joint_reward(self) -> float:
        return round(self.social_optimum_per_agent * self.num_agents * self.max_iterations, 2)

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["joint_earnings"])

    def agent_reward(self, agent_name: str, action: Any) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["earnings"].get(agent_name, 0.0))

    def log_iteration(self, iteration: int) -> None:
        round_index = iteration - 1
        record = self._settle_round(round_index)
        self.round_history.append(record)
        self.round = iteration
        self.joint_reward_history.append(record["joint_earnings"])
        logger.info(
            "=== %s Round %s === contributions=%s multiplied_pool=%s share=%s "
            "punishment_points=%s joint_earnings=%s",
            self.__class__.__name__,
            record["round"],
            record["contributions"],
            record["multiplied_pool"],
            record["share_per_agent"],
            record["punishment_points_used"],
            record["joint_earnings"],
        )

    def get_final_summary(self) -> Dict[str, Any]:
        contribution_rate_curve = [
            round(r["total_contributed"] / (self.num_agents * self.endowment), 3)
            for r in self.round_history
        ]
        avg_rate = (
            round(sum(contribution_rate_curve) / len(contribution_rate_curve), 3)
            if contribution_rate_curve
            else None
        )
        total_earnings = {
            name: round(sum(float(r["earnings"].get(name, 0.0)) for r in self.round_history), 2)
            for name in self.agent_names
        }
        # Extortion analysis: punishment directed at above/below-median
        # contributors of that round.
        extort = {"at_high_contributors": 0, "at_low_contributors": 0, "points_high": 0, "points_low": 0}
        for r in self.round_history:
            vals = sorted(r["contributions"].values())
            mid = vals[len(vals) // 2]
            for d in r["punishments"]:
                target_c = r["contributions"].get(d["peer"], 0)
                if target_c >= mid:
                    extort["at_high_contributors"] += 1
                    extort["points_high"] += d["points"]
                else:
                    extort["at_low_contributors"] += 1
                    extort["points_low"] += d["points"]
        return {
            "environment": self.__class__.__name__,
            "punishment_enabled": self.punishment_enabled,
            "rounds_simulated": len(self.round_history),
            "agents": {
                name: {
                    "contribution_history": list(self.contribution_plans[name]),
                    "total_earnings": total_earnings[name],
                }
                for name in self.agent_names
            },
            "round_history": self.round_history,
            "contribution_rate_curve": contribution_rate_curve,
            "average_contribution_rate": avg_rate,
            "benchmarks": {
                "free_rider_per_agent_per_round": self.free_rider_benchmark_per_agent,
                "social_optimum_per_agent_per_round": self.social_optimum_per_agent,
                "free_rider_invasion_gain": self._free_rider_invasion_gain(),
            },
            "total_joint_earnings": round(sum(total_earnings.values()), 2),
            "punishment": {
                "total_points_used": sum(r["punishment_points_used"] for r in self.round_history),
                "directed_at_high_contributors": extort["at_high_contributors"],
                "directed_at_low_contributors": extort["at_low_contributors"],
                "points_at_high_contributors": extort["points_high"],
                "points_at_low_contributors": extort["points_low"],
            },
            "joint_reward": float(
                self.round_history[-1]["joint_earnings"] if self.round_history else 0.0
            ),
            "max_joint_reward": self.max_joint_reward,
        }
