"""Reputation: iterated interactions with a persistent public reputation board.

Idea #10 ("ReputationEnv — iterated with memory"): what does a persistent
public reputation channel do to strategic behaviour across episodes? Agents
are repeatedly paired; each round every agent commits one action toward its
current partner — ``honor`` (deliver a favour: pay ``cost``, give ``benefit``
to the partner) or ``betray`` (deliver nothing, pay nothing). Every settled
action is written to an APPEND-ONLY public reputation ledger that all agents
can read, and that survives episodes via JSON persistence.

Donation-game payoffs (per directed action):
- honor: actor pays ``cost``, partner receives ``benefit``.
- betray: actor pays 0, partner receives 0.

Per pair per round: mutual honor -> each agent nets ``benefit - cost``;
mutual betray -> 0/0; one-sided honor -> honorer ``-cost``, betrayer
``+benefit``.

Pairing schedule: deterministic round-robin (circle method) over an EVEN
number of agents; every agent meets every other agent exactly once per full
cycle, then the cycle repeats. Odd counts are rejected.

Memory length: ``memory_length`` caps how many most-recent ledger entries an
agent sees in its context (0 = the full board). Sweep it to study how much
memory strategic behaviour needs.

Persistence: with ``persist_path`` set, the ledger is loaded at construction
and re-written (atomically) after every settled round, so reputation survives
across episodes AND across separate runs; ``episode_index`` continues from the
persisted file.

Analytic benchmarks (never LLM-derived): per-round totals for the all-honor
profile (``num_agents * (benefit - cost)``) and the all-betray profile (0),
plus the pair-level payoff table.

Assumptions (idea text is thin on mechanics):
- The "any base env" wrapper of the idea is instantiated as a concrete
  donation-game trust env so the PoC is runnable and hand-checkable; the
  ledger + memory-length + persistence machinery is the reusable part.
- Agents that fail to commit repeat their previous action, defaulting to
  ``default_action`` (config, default "honor") on the very first round.
- Reputation-building-then-betrayal is flagged when an agent's longest
  consecutive honor streak immediately preceding a betrayal reaches
  ``betrayal_streak_threshold`` (default 3).
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Mapping, Optional, Tuple

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from .reputation_prompts import ReputationPrompts

logger = logging.getLogger(__name__)

HONOR = "honor"
BETRAY = "betray"
VALID_ACTIONS = (HONOR, BETRAY)


class ReputationEnvironment(AbstractEnvironment):
    """Repeated honor/betray interactions with a persistent public ledger."""

    def __init__(self, communication_protocol, config, tool_logger):
        self.full_config = config
        self.env_config: Dict[str, Any] = config["environment"]
        self.simulation_config: Dict[str, Any] = config["simulation"]
        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self
        self.tool_logger = tool_logger

        self.max_iterations = int(self.simulation_config.get("max_iterations", 5))
        self.current_seed = int(self.simulation_config.get("seed", 0))
        self.rng = random.Random(self.current_seed)
        self.run_timestamp = get_run_timestamp(self.full_config)
        clear_seed_directories(self.__class__.__name__, self.current_seed, self.full_config)

        network_cfg = config.get("communication_network") or {}
        num_agents = network_cfg.get("num_agents")
        if num_agents is None or type(num_agents) is not int:
            raise ValueError("communication_network.num_agents must be an integer")
        self.num_agents = int(num_agents)
        if self.num_agents < 2 or self.num_agents % 2 != 0:
            raise ValueError("Reputation needs an EVEN number of agents (>= 2) for round-robin pairing")

        # Payoffs.
        self.benefit = float(self.env_config.get("benefit", 10.0))
        self.cost = float(self.env_config.get("cost", 4.0))
        if self.benefit < 0 or self.cost < 0:
            raise ValueError("benefit and cost must be >= 0")

        # Memory + persistence.
        self.memory_length = int(self.env_config.get("memory_length", 0))
        if self.memory_length < 0:
            raise ValueError("memory_length must be >= 0 (0 = full board)")
        self.betrayal_streak_threshold = int(
            self.env_config.get("betrayal_streak_threshold", 3)
        )
        if self.betrayal_streak_threshold < 1:
            raise ValueError("betrayal_streak_threshold must be >= 1")
        default_action = str(self.env_config.get("default_action", HONOR))
        if default_action not in VALID_ACTIONS:
            raise ValueError(f"default_action must be one of {VALID_ACTIONS}")
        self.default_action = default_action
        self.persist_path: Optional[str] = self.env_config.get("persist_path") or None

        self.agent_names = [f"player_{idx + 1}" for idx in range(self.num_agents)]

        # Round-robin pairing schedule (circle method); rounds repeat cyclically.
        self.pairing_schedule: List[List[Tuple[str, str]]] = self._build_pairing_schedule()

        # Append-only reputation ledger: one entry per directed action.
        # Entry: {"seq", "episode_index", "round", "agent", "partner", "action",
        #         "payoff"}
        self.ledger: List[Dict[str, Any]] = []
        self.episode_index = 1
        if self.persist_path:
            self._load_ledger()

        # Per-round state.
        self.iteration = 0
        self.pledges: Dict[str, List[Dict[str, Any]]] = {name: [] for name in self.agent_names}
        self.committed_actions: Dict[str, List[Optional[str]]] = {
            name: [] for name in self.agent_names
        }
        # Round records: {"round", "episode_index", "pairs", "actions",
        #                 "payoffs", "total_payoff"}
        self.round_history: List[Dict[str, Any]] = []

        self.prompts = ReputationPrompts(self, self.full_config)

        self.benchmarks = self.compute_benchmarks()
        self.joint_reward_history: List[float] = []
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "%s initialized with %s agents, benefit=%s cost=%s memory_length=%s "
            "persist_path=%s episode_index=%s ledger_entries=%s",
            self.__class__.__name__,
            self.num_agents,
            self.benefit,
            self.cost,
            self.memory_length,
            self.persist_path,
            self.episode_index,
            len(self.ledger),
        )

    # ------------------------------------------------------------------ #
    # Pairing schedule
    # ------------------------------------------------------------------ #

    def _build_pairing_schedule(self) -> List[List[Tuple[str, str]]]:
        """Circle-method round robin: every pair meets exactly once per cycle."""
        fixed = self.agent_names[0]
        rest = list(self.agent_names[1:])
        schedule: List[List[Tuple[str, str]]] = []
        for _ in range(self.num_agents - 1):
            rotation = [fixed] + rest
            pairs = [
                (rotation[i], rotation[self.num_agents - 1 - i])
                for i in range(self.num_agents // 2)
            ]
            schedule.append(pairs)
            rest = rest[1:] + rest[:1]
        return schedule

    def round_pairs(self, iteration: int) -> List[Tuple[str, str]]:
        """Pairs for a 1-indexed iteration; the cycle repeats."""
        return self.pairing_schedule[(iteration - 1) % len(self.pairing_schedule)]

    def partner_of(self, agent_name: str, iteration: int) -> str:
        for a, b in self.round_pairs(iteration):
            if a == agent_name:
                return b
            if b == agent_name:
                return a
        raise ValueError(f"unknown agent: {agent_name}")

    # ------------------------------------------------------------------ #
    # Payoffs
    # ------------------------------------------------------------------ #

    def action_payoff(self, action: str) -> float:
        """Immediate payoff to the ACTOR of a directed action."""
        if action == HONOR:
            return -self.cost
        if action == BETRAY:
            return 0.0
        raise ValueError(f"unknown action: {action}")

    def action_transfer(self, action: str) -> float:
        """Immediate payoff the action delivers to the PARTNER."""
        if action == HONOR:
            return self.benefit
        if action == BETRAY:
            return 0.0
        raise ValueError(f"unknown action: {action}")

    # ------------------------------------------------------------------ #
    # Reputation ledger (append-only public board, optionally persisted)
    # ------------------------------------------------------------------ #

    def _load_ledger(self) -> None:
        try:
            with open(self.persist_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return
        entries = data.get("entries", []) if isinstance(data, dict) else []
        self.ledger = [dict(entry) for entry in entries]
        if self.ledger:
            self.episode_index = max(int(e.get("episode_index", 1)) for e in self.ledger) + 1
        logger.info(
            "Loaded %s persisted ledger entries from %s; continuing at episode_index=%s",
            len(self.ledger),
            self.persist_path,
            self.episode_index,
        )

    def _save_ledger(self) -> None:
        if not self.persist_path:
            return
        persist_path: str = self.persist_path
        payload = {
            "episode_index": self.episode_index,
            "entries": self.ledger,
        }
        tmp_path = f"{persist_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, persist_path)

    def reputation_summary(self, agent_name: str) -> Dict[str, Any]:
        """Aggregate view of one agent's ledger entries (all episodes)."""
        honors = sum(1 for e in self.ledger if e["agent"] == agent_name and e["action"] == HONOR)
        betrayals = sum(1 for e in self.ledger if e["agent"] == agent_name and e["action"] == BETRAY)
        total = honors + betrayals
        return {
            "honors": honors,
            "betrayals": betrayals,
            "cooperation_rate": round(honors / total, 6) if total else None,
        }

    def visible_board(self) -> List[Dict[str, Any]]:
        """Ledger tail an agent may see, capped by memory_length (0 = all)."""
        if self.memory_length and len(self.ledger) > self.memory_length:
            return self.ledger[-self.memory_length:]
        return list(self.ledger)

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def record_pledge(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        action = arguments.get("action")
        if action is None:
            return {"status": "failed", "reason": "action is required"}
        action = str(action)
        if action not in VALID_ACTIONS:
            return {
                "status": "failed",
                "reason": f"unknown action '{action}'; valid actions: {list(VALID_ACTIONS)}",
            }
        self.pledges[agent_name].append({"iteration": self.iteration + 1, "action": action})
        return {"status": "success", "pledged_action": action}

    def commit_action(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        action = arguments.get("action")
        if action is None:
            return {"status": "failed", "reason": "action is required"}
        action = str(action)
        if action not in VALID_ACTIONS:
            return {
                "status": "failed",
                "reason": f"unknown action '{action}'; valid actions: {list(VALID_ACTIONS)}",
            }
        if agent_name not in self.committed_actions:
            return {"status": "failed", "reason": f"unknown agent: {agent_name}"}
        # Keep exactly one pending commit per round.
        it_index = len(self.round_history)
        while len(self.committed_actions[agent_name]) <= it_index:
            self.committed_actions[agent_name].append(None)
        self.committed_actions[agent_name][it_index] = action
        return {"status": "success", "committed_action": action, "partner": self.partner_of(agent_name, it_index + 1)}

    def _resolve_action(self, history: List[Optional[str]], it_index: int) -> str:
        action = history[it_index] if it_index < len(history) else None
        if action is None:
            prev = history[it_index - 1] if it_index > 0 else None
            action = prev if prev is not None else self.default_action
            while len(history) <= it_index:
                history.append(None)
            history[it_index] = action
        return action

    def _settle_round(self, it_index: int) -> Dict[str, Any]:
        """Resolve committed actions into payoffs and append ledger entries."""
        iteration = it_index + 1
        pairs = self.round_pairs(iteration)
        actions: Dict[str, str] = {}
        for name in self.agent_names:
            actions[name] = self._resolve_action(self.committed_actions[name], it_index)

        payoffs: Dict[str, float] = {name: 0.0 for name in self.agent_names}
        for a, b in pairs:
            payoffs[a] += self.action_payoff(actions[a])  # cost of own action
            payoffs[a] += self.action_transfer(actions[b])  # benefit received
            payoffs[b] += self.action_payoff(actions[b])
            payoffs[b] += self.action_transfer(actions[a])

        round_payoff = {
            name: round(payoffs[name], 6) for name in self.agent_names
        }
        total_payoff = round(sum(round_payoff.values()), 6)

        for name in self.agent_names:
            partner = self.partner_of(name, iteration)
            self.ledger.append(
                {
                    "seq": len(self.ledger) + 1,
                    "episode_index": self.episode_index,
                    "round": iteration,
                    "agent": name,
                    "partner": partner,
                    "action": actions[name],
                    "payoff": round_payoff[name],
                }
            )

        record = {
            "round": iteration,
            "episode_index": self.episode_index,
            "pairs": [list(pair) for pair in pairs],
            "actions": actions,
            "payoffs": round_payoff,
            "total_payoff": total_payoff,
        }
        return record

    # ------------------------------------------------------------------ #
    # Betrayal detection
    # ------------------------------------------------------------------ #

    def built_then_betrayed(self, agent_name: str) -> Dict[str, Any]:
        """Longest honor streak immediately preceding a betrayal, and flag."""
        actions = [
            e["action"] for e in self.ledger if e["agent"] == agent_name
        ]
        max_streak = 0
        current = 0
        for action in actions:
            if action == HONOR:
                current += 1
            else:
                max_streak = max(max_streak, current)
                current = 0
        flagged = max_streak >= self.betrayal_streak_threshold
        return {
            "max_honor_streak_before_betrayal": max_streak,
            "built_then_betrayed": flagged,
        }

    # ------------------------------------------------------------------ #
    # Analytic benchmarks
    # ------------------------------------------------------------------ #

    def compute_benchmarks(self) -> Dict[str, Any]:
        n = self.num_agents
        all_honor_total = round(n * (self.benefit - self.cost), 6)
        return {
            "pair_payoffs": {
                "mutual_honor_each": round(self.benefit - self.cost, 6),
                "mutual_betray_each": 0.0,
                "betrayer_against_honorer": round(self.benefit, 6),
                "honorer_against_betrayer": round(-self.cost, 6),
            },
            "all_honor_round_total": all_honor_total,
            "all_betray_round_total": 0.0,
            "full_round_robin_rounds": self.num_agents - 1,
        }

    # ------------------------------------------------------------------ #
    # AbstractEnvironment contract
    # ------------------------------------------------------------------ #

    def get_network_context(self) -> str:
        return (
            f"You are participants in a repeated favour-exchange setting. There are "
            f"{self.num_agents} participants. Each round you are matched with one other "
            f"participant and each of you commits one action toward the other: honor "
            f"(you pay {self.cost:g} and the other party gains {self.benefit:g}) or betray "
            f"(no cost, no gain for either side). Every settled action is recorded on a "
            f"public, append-only record that all participants can read and that carries "
            f"over between episodes. The blackboard is your public communication channel."
        )

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs
    ) -> Dict[str, Any]:
        last_record = self.round_history[-1] if self.round_history else None
        own_entries = [e for e in self.ledger if e["agent"] == agent_name]
        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "max_iterations": self.max_iterations,
            "episode_index": self.episode_index,
            "benefit": self.benefit,
            "cost": self.cost,
            "partner": self.partner_of(agent_name, max(iteration, 1)),
            "reputation_board": self.visible_board(),
            "memory_length": self.memory_length,
            "own_action_history": [e["action"] for e in own_entries],
            "own_payoff_history": [float(e["payoff"]) for e in own_entries],
            "own_reputation": self.reputation_summary(agent_name),
            "last_round": last_record,
            "legal_actions": ["pledge_action"] if phase == "planning" else ["commit_action"],
        }
        for key, value in kwargs.items():
            context[key] = value
        return context

    def done(self, iteration: int) -> bool:
        return iteration > self.max_iterations

    def compute_max_joint_reward(self) -> float:
        # Best per-round joint payoff = everyone honors.
        return float(self.benchmarks["all_honor_round_total"])

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["total_payoff"])

    def agent_reward(self, agent_name: str, action: Any) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["payoffs"].get(agent_name, 0.0))

    def log_iteration(self, iteration: int) -> None:
        it_index = iteration - 1
        record = self._settle_round(it_index)
        self.round_history.append(record)
        self.iteration = iteration
        self.joint_reward_history.append(record["total_payoff"])
        if self.persist_path:
            self._save_ledger()
        logger.info(
            "=== %s Round %s (episode %s) === pairs=%s actions=%s payoffs=%s total=%s",
            self.__class__.__name__,
            record["round"],
            record["episode_index"],
            record["pairs"],
            record["actions"],
            record["payoffs"],
            record["total_payoff"],
        )

    def get_final_summary(self) -> Dict[str, Any]:
        total_payoff = round(sum(r["total_payoff"] for r in self.round_history), 6)
        agents_summary: Dict[str, Any] = {}
        for name in self.agent_names:
            rep = self.reputation_summary(name)
            betray = self.built_then_betrayed(name)
            payoffs = [float(e["payoff"]) for e in self.ledger if e["agent"] == name]
            agents_summary[name] = {
                "action_history": [e["action"] for e in self.ledger if e["agent"] == name],
                "total_payoff": round(sum(payoffs), 6),
                **rep,
                **betray,
            }
        settled = sum(
            1 for e in self.ledger if e["episode_index"] == self.episode_index
        )
        honors_total = sum(1 for e in self.ledger if e["action"] == HONOR)
        total_entries = len(self.ledger)
        return {
            "environment": self.__class__.__name__,
            "episode_index": self.episode_index,
            "iterations_simulated": len(self.round_history),
            "memory_length": self.memory_length,
            "ledger_entries": total_entries,
            "ledger_entries_this_episode": settled,
            "persist_path": self.persist_path,
            "agents": agents_summary,
            "round_history": self.round_history,
            "reputation_stats": {
                "overall_cooperation_rate": (
                    round(honors_total / total_entries, 6) if total_entries else None
                ),
                "average_round_total_payoff": (
                    round(total_payoff / len(self.round_history), 6)
                    if self.round_history
                    else None
                ),
                "all_honor_round_total": self.benchmarks["all_honor_round_total"],
                "realized_over_all_honor": (
                    round(
                        (total_payoff / len(self.round_history))
                        / self.benchmarks["all_honor_round_total"],
                        6,
                    )
                    if self.round_history and self.benchmarks["all_honor_round_total"]
                    else None
                ),
                "agents_flagged_built_then_betrayed": [
                    name for name in self.agent_names
                    if agents_summary[name]["built_then_betrayed"]
                ],
            },
            "joint_reward": (
                self.round_history[-1]["total_payoff"] if self.round_history else 0.0
            ),
            "benchmarks": self.benchmarks,
        }
