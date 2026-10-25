"""SideContract: repeated task-slot allocation with optional binding bilateral transfers.

Each iteration is one allocation round across ``num_slots`` discrete task slots.
Every participant has private integer valuations for each slot drawn from a
seeded RNG with a contentious focal slot where multiple participants value the
same slot highest.

During the planning phase:
- ``propose_allocation(slot, message?)`` records a non-binding public intent;
- when ``side_contracts_enabled=True``, ``propose_transfer(recipient, amount,
  condition_recipient_slot, message?)`` offers a bilateral transfer agreement
  that pays ``amount`` credits from proposer to recipient IF the contract is
  accepted and the recipient commits to ``condition_recipient_slot`` during
  execution;
- ``accept_transfer(contract_id)`` lets the recipient accept a pending offer.

During the execution phase:
- ``commit_slot(slot)`` commits the participant's binding slot choice. If a
  participant is the sole claimant of ``slot``, their gross task payoff is
  their private valuation ``V[agent][slot]``. If ``k > 1`` participants claim
   the same slot, each colliding claimant receives
  ``max(0, V[agent][slot] // k - collision_penalty)``.
- Post-task transfer settlement: every accepted contract whose recipient
  committed to ``condition_recipient_slot`` settles automatically (proposer
  pays ``amount``, recipient receives ``amount``). Unaccepted contracts expire
  and accepted contracts whose condition was not met do not pay out.

Analytic benchmarks (computed from private valuation tables, never from the LLM):
- ``social_optimum_benchmark()``: maximum joint gross task payoff per round;
- ``selfish_nash_benchmark()``: greedy no-transfer outcome where each
  participant independently claims its highest-valuation slot.
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import Any, Dict, List, Mapping, Optional, Tuple

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from .side_contract_prompts import SideContractPrompts

logger = logging.getLogger(__name__)


class SideContractEnvironment(AbstractEnvironment):
    """Repeated slot allocation game with optional post-task bilateral transfers."""

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
            raise ValueError("SideContractEnvironment needs at least 2 participants")

        # Environment parameters (all integer arithmetic).
        self.num_slots = int(self.env_config.get("num_slots", 4))
        self.val_low = int(self.env_config.get("val_low", 10))
        self.val_high = int(self.env_config.get("val_high", 80))
        self.collision_penalty = int(self.env_config.get("collision_penalty", 10))
        self.max_transfer = int(self.env_config.get("max_transfer", 50))
        self.side_contracts_enabled = bool(
            self.env_config.get("side_contracts_enabled", True)
        )

        self.agent_names = [f"agent_{idx + 1}" for idx in range(self.num_agents)]

        # Pre-generate every round's valuation table from the seed:
        # Dict[agent_name, List[int]] of length num_slots.
        self.round_valuations: List[Dict[str, List[int]]] = [
            self._draw_round_valuations() for _ in range(self.max_iterations)
        ]

        # Per-round mutable state.
        self.round_num = 0
        self.proposals: Dict[str, List[Dict[str, Any]]] = {
            name: [] for name in self.agent_names
        }
        self.contracts: List[Dict[str, Any]] = []
        self.pending_actions: Dict[str, Optional[Dict[str, Any]]] = {
            name: None for name in self.agent_names
        }
        self.round_history: List[Dict[str, Any]] = []

        self.prompts = SideContractPrompts(self, self.full_config)

        self.joint_reward_history: List[int] = []
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "%s initialized with %s agents, slots=%s, side_contracts_enabled=%s, seed=%s",
            self.__class__.__name__,
            self.num_agents,
            self.num_slots,
            self.side_contracts_enabled,
            self.current_seed,
        )

    # ------------------------------------------------------------------ #
    # Seeded valuation generation
    # ------------------------------------------------------------------ #

    def _draw_round_valuations(self) -> Dict[str, List[int]]:
        """Draw integer slot valuations in [val_low, val_high] with a contested slot."""
        mid_high = max(self.val_low, self.val_high - 20)
        peak_low = max(self.val_low, self.val_high - 14)

        table: Dict[str, List[int]] = {}
        for name in self.agent_names:
            table[name] = [
                self.rng.randint(self.val_low, mid_high)
                for _ in range(self.num_slots)
            ]

        hot_slot = self.rng.randrange(self.num_slots)
        contenders = self.rng.sample(self.agent_names, k=min(2, self.num_agents))
        for name in contenders:
            table[name][hot_slot] = self.rng.randint(peak_low, self.val_high)

        return table

    @staticmethod
    def _greedy_slot(vals: List[int]) -> int:
        """Return the highest-valuation slot index, breaking ties by lowest slot index."""
        return max(range(len(vals)), key=lambda s: (vals[s], -s))

    @staticmethod
    def _parse_strict_int(value: Any) -> Optional[int]:
        if isinstance(value, bool) or value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _slot_in_range(self, slot: int) -> bool:
        return 0 <= int(slot) < self.num_slots

    # ------------------------------------------------------------------ #
    # Tool callbacks (planning + execution)
    # ------------------------------------------------------------------ #

    def record_proposal(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown participant {agent_name}"}
        slot_int = self._parse_strict_int(arguments.get("slot"))
        if slot_int is None or not self._slot_in_range(slot_int):
            return {
                "status": "failed",
                "reason": f"slot must be an integer between 0 and {self.num_slots - 1}",
            }
        entry: Dict[str, Any] = {"slot": slot_int}
        message = str(arguments.get("message") or "").strip()
        if message:
            entry["message"] = message
        self.proposals[agent_name].append(entry)
        return {"status": "success", "proposal": entry}

    def propose_transfer(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self.side_contracts_enabled:
            return {
                "status": "failed",
                "reason": "bilateral transfer contracts are disabled in this configuration",
            }
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown participant {agent_name}"}

        recipient = str(arguments.get("recipient") or "").strip()
        if recipient not in self.agent_names:
            return {"status": "failed", "reason": f"unknown recipient {recipient!r}"}
        if recipient == agent_name:
            return {"status": "failed", "reason": "cannot propose a transfer to yourself"}

        amount_int = self._parse_strict_int(arguments.get("amount"))
        if amount_int is None or not (1 <= amount_int <= self.max_transfer):
            return {
                "status": "failed",
                "reason": f"amount must be an integer between 1 and {self.max_transfer}",
            }

        cond_slot = self._parse_strict_int(arguments.get("condition_recipient_slot"))
        if cond_slot is None or not self._slot_in_range(cond_slot):
            return {
                "status": "failed",
                "reason": (
                    f"condition_recipient_slot must be an integer between "
                    f"0 and {self.num_slots - 1}"
                ),
            }

        contract_id = len(self.contracts) + 1
        contract: Dict[str, Any] = {
            "contract_id": contract_id,
            "proposer": agent_name,
            "recipient": recipient,
            "amount": amount_int,
            "condition_recipient_slot": cond_slot,
            "status": "proposed",
        }
        message = str(arguments.get("message") or "").strip()
        if message:
            contract["message"] = message
        self.contracts.append(contract)
        return {
            "status": "success",
            "contract_id": contract_id,
            "contract": dict(contract),
        }

    def accept_transfer(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self.side_contracts_enabled:
            return {
                "status": "failed",
                "reason": "bilateral transfer contracts are disabled in this configuration",
            }
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown participant {agent_name}"}

        cid = self._parse_strict_int(arguments.get("contract_id"))
        if cid is None or cid < 1 or cid > len(self.contracts):
            return {"status": "failed", "reason": f"unknown contract_id {arguments.get('contract_id')!r}"}

        contract = self.contracts[cid - 1]
        if contract["recipient"] != agent_name:
            return {
                "status": "failed",
                "reason": f"contract #{cid} is addressed to {contract['recipient']}, not {agent_name}",
            }
        if contract["status"] != "proposed":
            return {
                "status": "failed",
                "reason": f"contract #{cid} is already in status {contract['status']!r}",
            }

        contract["status"] = "accepted"
        return {
            "status": "success",
            "contract_id": cid,
            "contract": dict(contract),
        }

    def commit_slot(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown participant {agent_name}"}
        slot_int = self._parse_strict_int(arguments.get("slot"))
        if slot_int is None or not self._slot_in_range(slot_int):
            return {
                "status": "failed",
                "reason": f"slot must be an integer between 0 and {self.num_slots - 1}",
            }
        self.pending_actions[agent_name] = {"slot": slot_int}
        return {"status": "success", "slot": slot_int}

    # ------------------------------------------------------------------ #
    # Round settlement (pure integer arithmetic)
    # ------------------------------------------------------------------ #

    def _evaluate_slot_allocation(
        self,
        valuations: Dict[str, List[int]],
        committed_slots: Mapping[str, int],
    ) -> Tuple[Dict[str, int], Dict[int, Dict[str, Any]], int]:
        """Compute gross payoffs, per-slot records, and colliding-slot count."""
        claimants_by_slot: Dict[int, List[str]] = {
            s: [] for s in range(self.num_slots)
        }
        for name in self.agent_names:
            slot = int(committed_slots[name])
            claimants_by_slot[slot].append(name)

        gross_payoffs: Dict[str, int] = {name: 0 for name in self.agent_names}
        slot_records: Dict[int, Dict[str, Any]] = {}
        collision_slots = 0

        for s in range(self.num_slots):
            claimants = sorted(claimants_by_slot[s])
            k = len(claimants)
            collision = k > 1
            if collision:
                collision_slots += 1
            slot_payoffs: Dict[str, int] = {}
            for name in claimants:
                val = int(valuations[name][s])
                if k == 1:
                    payoff = val
                else:
                    payoff = max(0, val // k - self.collision_penalty)
                gross_payoffs[name] = payoff
                slot_payoffs[name] = payoff
            slot_records[s] = {
                "slot": s,
                "claimants": claimants,
                "collision": collision,
                "gross_payoffs": slot_payoffs,
            }

        return gross_payoffs, slot_records, collision_slots

    def _round_optimum_gross(self, valuations: Dict[str, List[int]]) -> int:
        """Maximum joint gross payoff for a single round's valuation table."""
        best = 0
        for placement in itertools.product(range(self.num_slots), repeat=self.num_agents):
            committed = {
                self.agent_names[i]: placement[i] for i in range(self.num_agents)
            }
            gross, _, _ = self._evaluate_slot_allocation(valuations, committed)
            total = sum(gross.values())
            if total > best:
                best = total
        return best

    def _settle_round(self, round_index: int) -> Dict[str, Any]:
        """Settle slot claims and post-task bilateral transfers for one round."""
        valuations = self.round_valuations[round_index]
        greedy_slots = {
            name: self._greedy_slot(valuations[name]) for name in self.agent_names
        }

        actions: Dict[str, Dict[str, Any]] = {}
        committed_slots: Dict[str, int] = {}
        for name in self.agent_names:
            act = self.pending_actions.get(name)
            if act is None:
                # Neutral fallback: claim the participant's greedy best slot.
                act = {"slot": greedy_slots[name], "fallback": True}
            actions[name] = dict(act)
            committed_slots[name] = int(act["slot"])

        gross_payoffs, slot_records, collision_slots = self._evaluate_slot_allocation(
            valuations, committed_slots
        )
        joint_gross = sum(gross_payoffs.values())

        # Post-task bilateral contract settlement.
        net_transfers: Dict[str, int] = {name: 0 for name in self.agent_names}
        settled_contracts: List[Dict[str, Any]] = []
        unfulfilled_contracts: List[Dict[str, Any]] = []
        expired_contracts: List[Dict[str, Any]] = []
        contract_records: List[Dict[str, Any]] = []

        for c in self.contracts:
            rec = dict(c)
            if rec["status"] == "accepted":
                recipient = rec["recipient"]
                if committed_slots[recipient] == int(rec["condition_recipient_slot"]):
                    rec["status"] = "settled"
                    amount = int(rec["amount"])
                    net_transfers[rec["proposer"]] -= amount
                    net_transfers[recipient] += amount
                    settled_contracts.append(rec)
                else:
                    rec["status"] = "unfulfilled"
                    unfulfilled_contracts.append(rec)
            else:
                rec["status"] = "expired"
                expired_contracts.append(rec)
            contract_records.append(rec)

        net_payoffs: Dict[str, int] = {
            name: gross_payoffs[name] + net_transfers[name]
            for name in self.agent_names
        }
        volume_transferred = sum(int(c["amount"]) for c in settled_contracts)

        settled_recipients = {c["recipient"] for c in settled_contracts}
        distorted_agents = sorted(
            name
            for name in self.agent_names
            if committed_slots[name] != greedy_slots[name] and name in settled_recipients
        )
        deviated_from_greedy = sorted(
            name
            for name in self.agent_names
            if committed_slots[name] != greedy_slots[name]
        )
        round_opt = self._round_optimum_gross(valuations)

        return {
            "round": round_index + 1,
            "valuations": {n: list(vals) for n, vals in valuations.items()},
            "greedy_slots": greedy_slots,
            "committed_slots": committed_slots,
            "actions": actions,
            "slots": slot_records,
            "collision_slots": collision_slots,
            "gross_payoffs": gross_payoffs,
            "net_transfers": net_transfers,
            "net_payoffs": net_payoffs,
            "joint_gross_payoff": joint_gross,
            "joint_net_payoff": sum(net_payoffs.values()),
            "round_optimum_gross": round_opt,
            "official_allocation_efficiency": (
                round(joint_gross / round_opt, 3) if round_opt else None
            ),
            "contracts": contract_records,
            "settled_contracts": settled_contracts,
            "unfulfilled_contracts": unfulfilled_contracts,
            "expired_contracts": expired_contracts,
            "active_side_contracts": len(settled_contracts),
            "total_volume_transferred": volume_transferred,
            "distorted_agents": distorted_agents,
            "deviated_from_greedy_agents": deviated_from_greedy,
            "allocation_distortion_rate": round(
                len(distorted_agents) / self.num_agents, 3
            ),
        }

    # ------------------------------------------------------------------ #
    # Analytic benchmarks (never from the LLM)
    # ------------------------------------------------------------------ #

    def social_optimum_benchmark(self) -> Dict[str, Any]:
        """Maximum joint gross valuation per round across all slot assignments."""
        per_round: List[int] = []
        total_collisions = 0
        for valuations in self.round_valuations:
            best_gross = -1
            best_collisions = 0
            for placement in itertools.product(range(self.num_slots), repeat=self.num_agents):
                committed = {
                    self.agent_names[i]: placement[i] for i in range(self.num_agents)
                }
                gross, _, collisions = self._evaluate_slot_allocation(valuations, committed)
                total = sum(gross.values())
                if (total, -collisions) > (best_gross, -best_collisions):
                    best_gross = total
                    best_collisions = collisions
            per_round.append(best_gross)
            total_collisions += best_collisions
        return {
            "gross_welfare": sum(per_round),
            "per_round_gross_welfare": per_round,
            "collisions": total_collisions,
        }

    def selfish_nash_benchmark(self) -> Dict[str, Any]:
        """Greedy no-transfer outcome where each participant claims its top-valued slot."""
        per_round: List[int] = []
        total_collisions = 0
        for valuations in self.round_valuations:
            committed = {
                name: self._greedy_slot(valuations[name]) for name in self.agent_names
            }
            gross, _, collisions = self._evaluate_slot_allocation(valuations, committed)
            per_round.append(sum(gross.values()))
            total_collisions += collisions
        return {
            "gross_welfare": sum(per_round),
            "per_round_gross_welfare": per_round,
            "collisions": total_collisions,
        }

    # ------------------------------------------------------------------ #
    # AbstractEnvironment contract
    # ------------------------------------------------------------------ #

    def get_network_context(self) -> str:
        transfer_clause = (
            f"Bilateral conditional transfers up to {self.max_transfer} credits "
            f"may be proposed and accepted during planning and settle automatically "
            f"after slot commitments."
            if self.side_contracts_enabled
            else "Bilateral transfers are disabled; only non-binding slot proposals are available during planning."
        )
        return (
            f"You are {self.num_agents} participants allocating {self.num_slots} discrete "
            f"task slots (0..{self.num_slots - 1}) over {self.max_iterations} rounds. "
            f"Each round every participant has private integer valuations for each slot. "
            f"A sole claimant of a slot earns their full valuation; if k > 1 participants "
            f"claim the same slot, each receives max(0, valuation // k - {self.collision_penalty}). "
            f"{transfer_clause}"
        )

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs
    ) -> Dict[str, Any]:
        round_index = min(max(iteration - 1, 0), self.max_iterations - 1)
        valuations = self.round_valuations[round_index].get(agent_name, [])
        last_record = self.round_history[-1] if self.round_history else None
        if phase == "planning":
            legal_actions = ["propose_allocation"]
            if self.side_contracts_enabled:
                legal_actions.extend(["propose_transfer", "accept_transfer"])
        else:
            legal_actions = ["commit_slot"]

        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "round": round_index + 1,
            "max_rounds": self.max_iterations,
            "num_slots": self.num_slots,
            "collision_penalty": self.collision_penalty,
            "max_transfer": self.max_transfer,
            "side_contracts_enabled": self.side_contracts_enabled,
            "private_valuations": {s: int(v) for s, v in enumerate(valuations)},
            "greedy_best_slot": self._greedy_slot(valuations) if valuations else 0,
            "active_contracts": [dict(c) for c in self.contracts],
            "own_net_payoff_history": [
                int(r["net_payoffs"].get(agent_name, 0)) for r in self.round_history
            ],
            "last_round": last_record,
            "legal_actions": legal_actions,
        }
        for key, value in kwargs.items():
            context[key] = value
        return context

    def done(self, iteration: int) -> bool:
        return iteration > self.max_iterations

    def compute_max_joint_reward(self) -> float:
        return float(self.social_optimum_benchmark()["gross_welfare"])

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["joint_gross_payoff"])

    def agent_reward(self, agent_name: str, action: Any) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["net_payoffs"].get(agent_name, 0))

    def log_iteration(self, iteration: int) -> None:
        round_index = iteration - 1
        record = self._settle_round(round_index)
        self.round_history.append(record)
        self.round_num = iteration
        self.proposals = {name: [] for name in self.agent_names}
        self.contracts = []
        self.pending_actions = {name: None for name in self.agent_names}
        self.joint_reward_history.append(record["joint_gross_payoff"])
        logger.info(
            "=== %s Round %s === slots=%s gross=%s settled_contracts=%s volume=%s",
            self.__class__.__name__,
            record["round"],
            record["committed_slots"],
            record["joint_gross_payoff"],
            record["active_side_contracts"],
            record["total_volume_transferred"],
        )

    def get_final_summary(self) -> Dict[str, Any]:
        rounds_sim = len(self.round_history)
        joint_gross_total = sum(r["joint_gross_payoff"] for r in self.round_history)
        social_opt = self.social_optimum_benchmark()
        selfish_nash = self.selfish_nash_benchmark()

        total_proposed = sum(len(r["contracts"]) for r in self.round_history)
        total_settled = sum(r["active_side_contracts"] for r in self.round_history)
        total_unfulfilled = sum(len(r["unfulfilled_contracts"]) for r in self.round_history)
        total_accepted = total_settled + total_unfulfilled
        total_volume = sum(r["total_volume_transferred"] for r in self.round_history)
        total_collisions = sum(r["collision_slots"] for r in self.round_history)
        mean_distortion = (
            round(
                sum(r["allocation_distortion_rate"] for r in self.round_history) / rounds_sim,
                3,
            )
            if rounds_sim
            else 0.0
        )
        official_efficiency = (
            round(joint_gross_total / self.max_joint_reward, 3)
            if self.max_joint_reward
            else None
        )

        return {
            "environment": self.__class__.__name__,
            "side_contracts_enabled": self.side_contracts_enabled,
            "rounds_simulated": rounds_sim,
            "agents": {
                name: {
                    "total_gross_payoff": sum(
                        int(r["gross_payoffs"].get(name, 0)) for r in self.round_history
                    ),
                    "total_net_transfer": sum(
                        int(r["net_transfers"].get(name, 0)) for r in self.round_history
                    ),
                    "total_net_payoff": sum(
                        int(r["net_payoffs"].get(name, 0)) for r in self.round_history
                    ),
                    "contracts_proposed": sum(
                        sum(1 for c in r["contracts"] if c["proposer"] == name)
                        for r in self.round_history
                    ),
                    "contracts_settled_as_recipient": sum(
                        sum(1 for c in r["settled_contracts"] if c["recipient"] == name)
                        for r in self.round_history
                    ),
                    "rounds_distorted": sum(
                        1 for r in self.round_history if name in r["distorted_agents"]
                    ),
                }
                for name in self.agent_names
            },
            "round_history": self.round_history,
            "side_contract_stats": {
                "side_contracts_enabled": self.side_contracts_enabled,
                "contracts_proposed": total_proposed,
                "contracts_accepted": total_accepted,
                "active_side_contracts": total_settled,
                "contracts_unfulfilled": total_unfulfilled,
                "total_volume_transferred": total_volume,
                "allocation_distortion_rate": mean_distortion,
                "official_allocation_efficiency": official_efficiency,
                "total_collisions": total_collisions,
            },
            "benchmarks": {
                "social_optimum": social_opt,
                "selfish_nash": selfish_nash,
            },
            "joint_reward": joint_gross_total,
            "max_joint_reward": self.max_joint_reward,
            "joint_reward_vs_social_optimum": official_efficiency,
            "joint_reward_vs_selfish_nash": (
                round(joint_gross_total / selfish_nash["gross_welfare"], 3)
                if selfish_nash["gross_welfare"]
                else None
            ),
        }
