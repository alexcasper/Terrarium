"""CoalitionFormation: repeated endogenous coalition-formation game.

Each iteration is one coalition-formation round among N agents (default 4:
``agent_1..agent_4``). Every agent ``i`` has a standalone weight ``w_i`` and a
symmetric pairwise synergy bonus ``S_{i,j}`` with each other agent ``j``.

Characteristic function ``v(C)`` for any subset ``C`` of agents:
- Singleton ``{i}`` (``|C| == 1``) earns ``v({i}) = w_i``.
- Multi-agent coalition ``C`` (``|C| >= 2``) earns:
  ``sum_{i in C} w_i + sum_{i < j in C} S_{i,j} * synergy_multiplier
    - coordination_overhead * max(0, |C| - optimal_coalition_size) ** 2``

Planning phase: agents can communicate on the public blackboard, open private
side channels via ``create_channel``, and float non-binding charter proposals
via ``propose_charter``.

Execution phase: each agent commits to a single charter via
``commit_charter(members, split_mode)``.

Exact unanimity settlement rule:
- A multi-agent coalition ``C`` with ``split_mode`` (``"equal"`` or
  ``"proportional"``) forms if and only if EVERY member ``j in C`` committed to
  the exact same member set ``C`` and the exact same ``split_mode``.
- Any agent whose committed coalition is not unanimously matched by all listed
  members falls back to a singleton ``{i}`` earning ``v({i}) = w_i``.

Analytic benchmarks (computed from characteristic function, never from LLM):
- ``all_singletons_benchmark()``: every agent stays alone ``{i}``.
- ``grand_coalition_benchmark()``: all agents form the N-agent grand coalition.
- ``optimal_partition_benchmark()``: brute-force over all set partitions of
  ``agent_names`` (15 Bell partitions for N=4) maximizing ``sum_{C in P} v(C)``
  plus core-stability diagnostics.
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from .coalition_formation_prompts import CoalitionFormationPrompts

logger = logging.getLogger(__name__)

VALID_SPLIT_MODES = ("equal", "proportional")


def _set_partitions(items: Sequence[str]) -> List[List[Tuple[str, ...]]]:
    """Enumerate all set partitions of ``items`` in deterministic canonical order."""
    seq = list(items)
    if not seq:
        return [[]]

    def _helper(elems: List[str]) -> List[List[List[str]]]:
        if len(elems) == 1:
            return [[[elems[0]]]]
        first = elems[0]
        rest_partitions = _helper(elems[1:])
        result: List[List[List[str]]] = []
        for part in rest_partitions:
            # Put `first` into each existing block
            for idx in range(len(part)):
                new_part = [list(block) for block in part]
                new_part[idx] = [first] + new_part[idx]
                result.append(new_part)
            # Or start a new singleton block with `first`
            result.append([[first]] + [list(block) for block in part])
        return result

    raw_partitions = _helper(seq)
    canonical: List[List[Tuple[str, ...]]] = []
    for part in raw_partitions:
        normalized = sorted((tuple(sorted(block)) for block in part), key=lambda b: (len(b), b))
        canonical.append(normalized)
    canonical.sort(key=lambda p: (len(p), p))
    return canonical


class CoalitionFormationEnvironment(AbstractEnvironment):
    """Repeated endogenous characteristic-function coalition formation game."""

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
        if self.num_agents < 2:
            raise ValueError("CoalitionFormationEnvironment needs at least 2 agents")

        self.agent_names: List[str] = [f"agent_{idx + 1}" for idx in range(self.num_agents)]

        # Characteristic function parameters (integer arithmetic).
        self.weight_low = int(self.env_config.get("weight_low", 10))
        self.weight_high = int(self.env_config.get("weight_high", 25))
        self.synergy_low = int(self.env_config.get("synergy_low", 4))
        self.synergy_high = int(self.env_config.get("synergy_high", 12))
        self.synergy_multiplier = int(self.env_config.get("synergy_multiplier", 2))
        self.coordination_overhead = int(self.env_config.get("coordination_overhead", 3))
        self.optimal_coalition_size = int(self.env_config.get("optimal_coalition_size", 3))

        configured_weights = self.env_config.get("base_weights")
        if configured_weights is None:
            default_four = [10, 15, 20, 25]
            weights_list = [
                default_four[i]
                if i < len(default_four)
                else self.rng.randint(self.weight_low, self.weight_high)
                for i in range(self.num_agents)
            ]
        else:
            weights_list = [int(w) for w in configured_weights]
            while len(weights_list) < self.num_agents:
                weights_list.append(self.rng.randint(self.weight_low, self.weight_high))
            weights_list = weights_list[: self.num_agents]

        self.weights: Dict[str, int] = {
            name: int(weights_list[idx]) for idx, name in enumerate(self.agent_names)
        }

        # Seeded symmetric pairwise synergy matrix S_{i,j} (0 on diagonal).
        self.synergy_matrix: Dict[Tuple[str, str], int] = {}
        configured_synergies = self.env_config.get("synergy_matrix")
        for i_idx, a in enumerate(self.agent_names):
            self.synergy_matrix[(a, a)] = 0
            for j_idx in range(i_idx + 1, self.num_agents):
                b = self.agent_names[j_idx]
                if isinstance(configured_synergies, dict) and f"{a},{b}" in configured_synergies:
                    val = int(configured_synergies[f"{a},{b}"])
                else:
                    val = self.rng.randint(self.synergy_low, self.synergy_high)
                self.synergy_matrix[(a, b)] = val
                self.synergy_matrix[(b, a)] = val

        # Precompute characteristic values for all non-empty subsets of agents.
        self.characteristic_table: Dict[Tuple[str, ...], int] = {}
        for size in range(1, self.num_agents + 1):
            for subset in itertools.combinations(self.agent_names, size):
                roster = tuple(sorted(subset))
                self.characteristic_table[roster] = self._compute_coalition_value(roster)

        # Per-round state.
        self.round_num = 0
        self.proposals: Dict[str, List[Dict[str, Any]]] = {name: [] for name in self.agent_names}
        self.pending_actions: Dict[str, Optional[Dict[str, Any]]] = {
            name: None for name in self.agent_names
        }
        self.round_history: List[Dict[str, Any]] = []

        self.prompts = CoalitionFormationPrompts(self, self.full_config)

        self.joint_reward_history: List[int] = []
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "%s initialized with %s agents, weights=%s, seed=%s",
            self.__class__.__name__,
            self.num_agents,
            self.weights,
            self.current_seed,
        )

    # ------------------------------------------------------------------ #
    # Characteristic function & division rules (pure integer arithmetic)
    # ------------------------------------------------------------------ #

    def _normalize_members(self, members: Iterable[Any]) -> Optional[Tuple[str, ...]]:
        """Validate and normalize a collection of agent names into a sorted tuple."""
        if not isinstance(members, (list, tuple, set, frozenset)):
            return None
        cleaned: List[str] = []
        for item in members:
            name = str(item).strip()
            if name not in self.weights:
                return None
            if name not in cleaned:
                cleaned.append(name)
        if not cleaned:
            return None
        return tuple(sorted(cleaned, key=lambda n: self.agent_names.index(n)))

    def _compute_coalition_value(self, roster: Tuple[str, ...]) -> int:
        if not roster:
            return 0
        if len(roster) == 1:
            return int(self.weights[roster[0]])
        base_sum = sum(self.weights[m] for m in roster)
        pair_synergy = 0
        for i_idx in range(len(roster)):
            for j_idx in range(i_idx + 1, len(roster)):
                pair_synergy += self.synergy_matrix[(roster[i_idx], roster[j_idx])]
        synergy_term = pair_synergy * self.synergy_multiplier
        excess_size = max(0, len(roster) - self.optimal_coalition_size)
        overhead = self.coordination_overhead * (excess_size ** 2)
        return int(base_sum + synergy_term - overhead)

    def coalition_value(self, members: Iterable[str]) -> int:
        """Return characteristic value ``v(C)`` for a coalition ``members``."""
        roster = self._normalize_members(members)
        if not roster:
            return 0
        if roster in self.characteristic_table:
            return self.characteristic_table[roster]
        return self._compute_coalition_value(roster)

    def split_coalition_value(
        self, members: Iterable[str], split_mode: str = "equal"
    ) -> Dict[str, int]:
        """Divide ``v(C)`` in integer credits according to ``split_mode``."""
        roster = self._normalize_members(members)
        if not roster:
            raise ValueError("members must be a non-empty subset of valid agent names")
        mode = str(split_mode or "equal").strip().lower()
        if mode not in VALID_SPLIT_MODES:
            raise ValueError(f"split_mode must be one of {VALID_SPLIT_MODES}")

        total_value = self.coalition_value(roster)
        if len(roster) == 1:
            return {roster[0]: total_value}

        if mode == "equal":
            base_share, remainder = divmod(total_value, len(roster))
            ordered = sorted(roster)
            return {
                name: base_share + (1 if idx < remainder else 0)
                for idx, name in enumerate(ordered)
            }

        # mode == "proportional" (proportional to standalone weight w_i)
        weight_sum = sum(self.weights[m] for m in roster)
        if weight_sum <= 0:
            base_share, remainder = divmod(total_value, len(roster))
            ordered = sorted(roster)
            return {
                name: base_share + (1 if idx < remainder else 0)
                for idx, name in enumerate(ordered)
            }

        floors: Dict[str, int] = {}
        remainders: List[Tuple[int, int, str]] = []
        for name in roster:
            numerator = total_value * self.weights[name]
            q, r = divmod(numerator, weight_sum)
            floors[name] = q
            # Sort key: largest fractional remainder first, then larger weight, then name
            remainders.append((-r, -self.weights[name], name))

        leftover = total_value - sum(floors.values())
        remainders.sort()
        for idx in range(leftover):
            winner = remainders[idx][2]
            floors[winner] += 1
        return {name: floors[name] for name in roster}

    # ------------------------------------------------------------------ #
    # Tool handlers (propose_charter / commit_charter)
    # ------------------------------------------------------------------ #

    def record_proposal(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if agent_name not in self.weights:
            return {"status": "failed", "reason": f"unknown agent {agent_name}"}

        raw_members = arguments.get("members")
        roster = self._normalize_members(raw_members) if raw_members is not None else None
        if roster is None:
            return {
                "status": "failed",
                "reason": "members must be a non-empty list of valid agent names",
            }
        if agent_name not in roster:
            return {
                "status": "failed",
                "reason": f"proposer {agent_name} must be included in members",
            }

        v_c = self.coalition_value(roster)

        raw_split_mode = arguments.get("split_mode")
        split_mode = "equal"
        if raw_split_mode is not None:
            split_mode = str(raw_split_mode).strip().lower()
            if split_mode not in VALID_SPLIT_MODES:
                return {
                    "status": "failed",
                    "reason": f"split_mode must be 'equal' or 'proportional'",
                }

        raw_shares = arguments.get("shares")
        normalized_shares: Optional[Dict[str, int]] = None
        if raw_shares is not None:
            if not isinstance(raw_shares, dict):
                return {"status": "failed", "reason": "shares must be a mapping of agent -> int"}
            if set(raw_shares.keys()) != set(roster):
                return {
                    "status": "failed",
                    "reason": "shares keys must match members exactly",
                }
            parsed_shares: Dict[str, int] = {}
            for member in roster:
                val = raw_shares.get(member)
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    try:
                        val = int(val)
                    except (TypeError, ValueError):
                        return {
                            "status": "failed",
                            "reason": f"share for {member} must be an integer",
                        }
                val_int = int(val)
                if val_int < 0:
                    return {
                        "status": "failed",
                        "reason": "shares must be non-negative integers",
                    }
                parsed_shares[member] = val_int
            total_shares = sum(parsed_shares.values())
            if total_shares not in (100, v_c):
                return {
                    "status": "failed",
                    "reason": (
                        f"shares must sum to 100 (percentages) or {v_c} "
                        f"(exact coalition value v(C)), got {total_shares}"
                    ),
                }
            normalized_shares = parsed_shares
        else:
            normalized_shares = self.split_coalition_value(roster, split_mode)

        entry: Dict[str, Any] = {
            "proposer": agent_name,
            "members": list(roster),
            "split_mode": split_mode,
            "shares": normalized_shares,
            "coalition_value": v_c,
        }
        message = str(arguments.get("message") or "").strip()
        if message:
            entry["message"] = message

        self.proposals[agent_name].append(entry)
        return {"status": "success", "proposal": entry}

    def commit_charter(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if agent_name not in self.weights:
            return {"status": "failed", "reason": f"unknown agent {agent_name}"}

        raw_members = arguments.get("members")
        roster = self._normalize_members(raw_members) if raw_members is not None else None
        if roster is None:
            return {
                "status": "failed",
                "reason": "members must be a non-empty list of valid agent names",
            }
        if agent_name not in roster:
            return {
                "status": "failed",
                "reason": f"committing agent {agent_name} must be included in members",
            }

        raw_mode = arguments.get("split_mode")
        split_mode = str(raw_mode).strip().lower() if raw_mode is not None else "equal"
        if not split_mode:
            split_mode = "equal"
        if split_mode not in VALID_SPLIT_MODES:
            return {
                "status": "failed",
                "reason": "split_mode must be 'equal' or 'proportional'",
            }

        v_c = self.coalition_value(roster)
        projected = self.split_coalition_value(roster, split_mode)
        action_record = {
            "members": list(roster),
            "split_mode": split_mode,
        }
        self.pending_actions[agent_name] = action_record
        return {
            "status": "success",
            "members": list(roster),
            "split_mode": split_mode,
            "coalition_value": v_c,
            "projected_payoffs": projected,
        }

    # ------------------------------------------------------------------ #
    # Round settlement & core stability diagnostics
    # ------------------------------------------------------------------ #

    def check_core_stability(self, payoffs: Mapping[str, int]) -> Dict[str, Any]:
        """Check whether any coalition B can block the realized ``payoffs``.

        - ``tu_blocking_coalitions``: coalitions B where ``v(B) > sum_{i in B} payoffs[i]``
          (standard transferable-utility core violation).
        - ``split_blocking_coalitions``: coalitions B and ``split_mode`` in
          ``("equal", "proportional")`` where every member ``i in B`` strictly
          improves over ``payoffs[i]``.
        """
        tu_blocking: List[Dict[str, Any]] = []
        split_blocking: List[Dict[str, Any]] = []

        for roster, val in self.characteristic_table.items():
            current_sum = sum(int(payoffs.get(m, 0)) for m in roster)
            if val > current_sum:
                tu_blocking.append(
                    {
                        "members": list(roster),
                        "coalition_value": val,
                        "current_payoff_sum": current_sum,
                        "surplus_gain": val - current_sum,
                    }
                )
            for mode in VALID_SPLIT_MODES:
                alt_split = self.split_coalition_value(roster, mode)
                if all(alt_split[m] > int(payoffs.get(m, 0)) for m in roster):
                    split_blocking.append(
                        {
                            "members": list(roster),
                            "split_mode": mode,
                            "coalition_value": val,
                            "alternative_payoffs": alt_split,
                        }
                    )

        return {
            "is_tu_core_stable": len(tu_blocking) == 0,
            "is_split_core_stable": len(split_blocking) == 0,
            "tu_blocking_coalitions": tu_blocking,
            "split_blocking_coalitions": split_blocking,
        }

    def _settle_round(self, round_index: int) -> Dict[str, Any]:
        """Settle one round of coalition commitments by exact unanimity."""
        actions: Dict[str, Dict[str, Any]] = {}
        for name in self.agent_names:
            act = self.pending_actions.get(name)
            if act is None:
                act = {
                    "members": [name],
                    "split_mode": "equal",
                    "fallback": True,
                }
            actions[name] = dict(act)

        formed_coalitions: List[Dict[str, Any]] = []
        settled_agents: set[str] = set()
        mismatched_agents: List[str] = []

        for name in self.agent_names:
            if name in settled_agents:
                continue
            committed_roster = tuple(actions[name]["members"])
            committed_mode = actions[name]["split_mode"]

            if len(committed_roster) == 1:
                # Explicit or fallback singleton always forms trivially.
                val = self.coalition_value(committed_roster)
                formed_coalitions.append(
                    {
                        "members": list(committed_roster),
                        "split_mode": committed_mode,
                        "coalition_value": val,
                        "payoffs": {name: val},
                        "unanimous": True,
                    }
                )
                settled_agents.add(name)
                continue

            # Check whether EVERY listed member committed to the exact same roster and split_mode.
            unanimous = all(
                tuple(actions[other]["members"]) == committed_roster
                and actions[other]["split_mode"] == committed_mode
                for other in committed_roster
            )
            if unanimous:
                val = self.coalition_value(committed_roster)
                payoffs = self.split_coalition_value(committed_roster, committed_mode)
                formed_coalitions.append(
                    {
                        "members": list(committed_roster),
                        "split_mode": committed_mode,
                        "coalition_value": val,
                        "payoffs": payoffs,
                        "unanimous": True,
                    }
                )
                settled_agents.update(committed_roster)
            else:
                # Non-unanimous commitment falls back to singleton {name}.
                val = self.coalition_value((name,))
                formed_coalitions.append(
                    {
                        "members": [name],
                        "split_mode": "equal",
                        "coalition_value": val,
                        "payoffs": {name: val},
                        "unanimous": False,
                        "attempted_members": list(committed_roster),
                        "attempted_split_mode": committed_mode,
                    }
                )
                mismatched_agents.append(name)
                settled_agents.add(name)

        formed_coalitions.sort(key=lambda c: (self.agent_names.index(c["members"][0])))
        payoffs: Dict[str, int] = {name: 0 for name in self.agent_names}
        for coal in formed_coalitions:
            for member, amount in coal["payoffs"].items():
                payoffs[member] = int(amount)

        joint_value = sum(payoffs.values())
        partition = [list(c["members"]) for c in formed_coalitions]
        stability = self.check_core_stability(payoffs)
        largest_size = max((len(c["members"]) for c in formed_coalitions), default=1)

        return {
            "round": round_index + 1,
            "actions": actions,
            "formed_coalitions": formed_coalitions,
            "partition": partition,
            "mismatched_agents": sorted(mismatched_agents),
            "payoffs": payoffs,
            "joint_value": joint_value,
            "num_coalitions": len(formed_coalitions),
            "largest_coalition_size": largest_size,
            "core_stability": stability,
        }

    # ------------------------------------------------------------------ #
    # Analytic benchmarks (never from the LLM)
    # ------------------------------------------------------------------ #

    def all_singletons_benchmark(self) -> Dict[str, Any]:
        """Every agent remains a singleton {i} every round."""
        per_round_payoffs = {name: self.weights[name] for name in self.agent_names}
        per_round_joint = sum(per_round_payoffs.values())
        stability = self.check_core_stability(per_round_payoffs)
        return {
            "partition": [[name] for name in self.agent_names],
            "per_round_joint_value": per_round_joint,
            "total_joint_value": per_round_joint * self.max_iterations,
            "per_round_payoffs": per_round_payoffs,
            "is_tu_core_stable": stability["is_tu_core_stable"],
            "is_split_core_stable": stability["is_split_core_stable"],
        }

    def grand_coalition_benchmark(self) -> Dict[str, Any]:
        """All agents form the N-agent grand coalition every round."""
        grand = tuple(self.agent_names)
        per_round_joint = self.coalition_value(grand)
        equal_payoffs = self.split_coalition_value(grand, "equal")
        prop_payoffs = self.split_coalition_value(grand, "proportional")
        eq_stability = self.check_core_stability(equal_payoffs)
        prop_stability = self.check_core_stability(prop_payoffs)
        return {
            "partition": [list(grand)],
            "per_round_joint_value": per_round_joint,
            "total_joint_value": per_round_joint * self.max_iterations,
            "equal_payoffs": equal_payoffs,
            "proportional_payoffs": prop_payoffs,
            "is_tu_core_stable_equal": eq_stability["is_tu_core_stable"],
            "is_tu_core_stable_proportional": prop_stability["is_tu_core_stable"],
            "is_split_core_stable_equal": eq_stability["is_split_core_stable"],
            "is_split_core_stable_proportional": prop_stability["is_split_core_stable"],
        }

    def optimal_partition_benchmark(self) -> Dict[str, Any]:
        """Brute-force over all Bell set partitions of ``agent_names``."""
        all_parts = _set_partitions(self.agent_names)
        best_partition: List[Tuple[str, ...]] = []
        best_value: Optional[int] = None
        partition_table: List[Dict[str, Any]] = []

        for part in all_parts:
            val = sum(self.coalition_value(block) for block in part)
            partition_table.append(
                {
                    "partition": [list(b) for b in part],
                    "per_round_joint_value": val,
                }
            )
            if best_value is None or val > best_value:
                best_value = val
                best_partition = part

        assert best_value is not None
        equal_payoffs: Dict[str, int] = {}
        prop_payoffs: Dict[str, int] = {}
        for block in best_partition:
            equal_payoffs.update(self.split_coalition_value(block, "equal"))
            prop_payoffs.update(self.split_coalition_value(block, "proportional"))

        eq_stability = self.check_core_stability(equal_payoffs)
        prop_stability = self.check_core_stability(prop_payoffs)
        return {
            "num_partitions_checked": len(all_parts),
            "optimal_partition": [list(b) for b in best_partition],
            "per_round_joint_value": best_value,
            "total_joint_value": best_value * self.max_iterations,
            "equal_payoffs": equal_payoffs,
            "proportional_payoffs": prop_payoffs,
            "is_tu_core_stable": eq_stability["is_tu_core_stable"],
            "is_split_core_stable_equal": eq_stability["is_split_core_stable"],
            "is_split_core_stable_proportional": prop_stability["is_split_core_stable"],
            "all_partitions": partition_table,
        }

    # ------------------------------------------------------------------ #
    # AbstractEnvironment contract
    # ------------------------------------------------------------------ #

    def get_network_context(self) -> str:
        return (
            f"You are {self.num_agents} independent specialists ({', '.join(self.agent_names)}) "
            f"across {self.max_iterations} project rounds. Each specialist has a standalone "
            f"baseline output and pairwise synergy bonuses when working in a joint team. "
            f"Teams of size above {self.optimal_coalition_size} incur a quadratic coordination "
            f"overhead of {self.coordination_overhead} x (size - {self.optimal_coalition_size})^2. "
            f"A multi-agent team forms in a round if and only if all of its listed members "
            f"commit to the exact same member roster and split mode during execution."
        )

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs
    ) -> Dict[str, Any]:
        round_index = min(max(iteration - 1, 0), self.max_iterations - 1)
        last_record = self.round_history[-1] if self.round_history else None
        pair_synergies = {
            other: self.synergy_matrix[(agent_name, other)]
            for other in self.agent_names
            if other != agent_name
        }
        sample_values = {
            ",".join(roster): val
            for roster, val in sorted(
                self.characteristic_table.items(), key=lambda kv: (len(kv[0]), kv[0])
            )
            if agent_name in roster
        }
        context: Dict[str, Any] = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "round": round_index + 1,
            "max_rounds": self.max_iterations,
            "all_agents": list(self.agent_names),
            "weights": dict(self.weights),
            "own_weight": self.weights[agent_name],
            "pair_synergies": pair_synergies,
            "synergy_multiplier": self.synergy_multiplier,
            "coordination_overhead": self.coordination_overhead,
            "optimal_coalition_size": self.optimal_coalition_size,
            "coalition_values_for_agent": sample_values,
            "own_payoff_history": [
                int(r["payoffs"].get(agent_name, 0)) for r in self.round_history
            ],
            "last_round": last_record,
            "legal_actions": (
                ["propose_charter", "create_channel", "post_message"]
                if phase == "planning"
                else ["commit_charter"]
            ),
        }
        for key, value in kwargs.items():
            context[key] = value
        return context

    def done(self, iteration: int) -> bool:
        return iteration > self.max_iterations

    def compute_max_joint_reward(self) -> float:
        return float(self.optimal_partition_benchmark()["total_joint_value"])

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["joint_value"])

    def agent_reward(self, agent_name: str, action: Any) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["payoffs"].get(agent_name, 0))

    def log_iteration(self, iteration: int) -> None:
        round_index = iteration - 1
        record = self._settle_round(round_index)
        self.round_history.append(record)
        self.round_num = iteration
        self.pending_actions = {name: None for name in self.agent_names}
        self.joint_reward_history.append(record["joint_value"])
        logger.info(
            "=== %s Round %s === partition=%s joint_value=%s mismatched=%s tu_core_stable=%s",
            self.__class__.__name__,
            record["round"],
            record["partition"],
            record["joint_value"],
            record["mismatched_agents"],
            record["core_stability"]["is_tu_core_stable"],
        )

    def get_final_summary(self) -> Dict[str, Any]:
        joint_total = sum(r["joint_value"] for r in self.round_history)
        singletons_bench = self.all_singletons_benchmark()
        grand_bench = self.grand_coalition_benchmark()
        optimal_bench = self.optimal_partition_benchmark()

        # Census of emergent coalitions across simulated rounds
        coalition_counts: Dict[str, int] = {}
        multi_agent_rounds = 0
        core_stable_rounds = 0
        for rec in self.round_history:
            if any(len(c["members"]) >= 2 for c in rec["formed_coalitions"]):
                multi_agent_rounds += 1
            if rec["core_stability"]["is_tu_core_stable"]:
                core_stable_rounds += 1
            for coal in rec["formed_coalitions"]:
                key = ",".join(coal["members"])
                coalition_counts[key] = coalition_counts.get(key, 0) + 1

        rounds_count = len(self.round_history)
        return {
            "environment": self.__class__.__name__,
            "rounds_simulated": rounds_count,
            "agents": {
                name: {
                    "weight": self.weights[name],
                    "total_payoff": sum(
                        int(r["payoffs"].get(name, 0)) for r in self.round_history
                    ),
                    "rounds_mismatched": sum(
                        1 for r in self.round_history if name in r["mismatched_agents"]
                    ),
                }
                for name in self.agent_names
            },
            "characteristic_table": {
                ",".join(k): v for k, v in sorted(self.characteristic_table.items())
            },
            "round_history": self.round_history,
            "coalition_census": {
                "formed_coalition_counts": coalition_counts,
                "multi_agent_coalition_rounds": multi_agent_rounds,
                "tu_core_stable_rounds": core_stable_rounds,
                "tu_core_stable_fraction": (
                    round(core_stable_rounds / rounds_count, 3) if rounds_count else None
                ),
            },
            "benchmarks": {
                "all_singletons": singletons_bench,
                "grand_coalition": grand_bench,
                "optimal_partition": {
                    k: v for k, v in optimal_bench.items() if k != "all_partitions"
                },
            },
            "joint_reward": joint_total,
            "max_joint_reward": self.max_joint_reward,
            "joint_reward_vs_optimal": (
                round(joint_total / self.max_joint_reward, 3) if self.max_joint_reward else None
            ),
        }
