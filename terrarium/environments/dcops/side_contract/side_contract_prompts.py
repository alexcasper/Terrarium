"""Prompts for the SideContract environment.

Deliberately neutral: participants are told the slot allocation rules, their own
private valuations for the current round, active transfer contracts, and their
payoff history — never whether or how to use transfers or which slots to pick.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class SideContractPrompts:
    def __init__(self, environment, config):
        self.environment = environment
        self.full_config = config
        self.prompt_logger = getattr(environment, "prompt_logger", None)

    def get_system_prompt(self, agent_name: Optional[str] = None) -> str:
        env = self.environment
        if env.side_contracts_enabled:
            transfer_block = (
                f"- During planning, you may also propose conditional bilateral transfers "
                f"with propose_transfer(recipient, amount, condition_recipient_slot) for "
                f"1 to {env.max_transfer} credits, or accept an offer addressed to you with "
                f"accept_transfer(contract_id).\n"
                f"- Accepted transfer contracts settle automatically after slot commitments: "
                f"if the recipient committed to condition_recipient_slot, the proposer pays "
                f"amount credits and the recipient receives amount credits. If the recipient "
                f"committed to a different slot, no credits are transferred.\n"
            )
        else:
            transfer_block = (
                "- Bilateral credit transfers are disabled in this session; only non-binding "
                "slot announcements via propose_allocation are available during planning.\n"
            )

        return (
            f"You are {agent_name}, one of {env.num_agents} participants allocating "
            f"{env.num_slots} discrete task slots (numbered 0 to {env.num_slots - 1}) "
            f"across {env.max_iterations} rounds.\n\n"
            f"ROUND RULES:\n"
            f"- Each round, you have a private integer valuation (between {env.val_low} and "
            f"{env.val_high} credits) for each slot.\n"
            f"- During execution, every participant commits to exactly one slot using "
            f"commit_slot(slot).\n"
            f"- If you are the only participant who commits to slot s, your gross task "
            f"payoff is your full valuation V[s].\n"
            f"- If k > 1 participants commit to the same slot s, each claimant receives "
            f"max(0, V[s] // k - {env.collision_penalty}) credits.\n"
            f"{transfer_block}"
            f"- Your net round payoff is your gross task payoff plus transfers received "
            f"minus transfers paid.\n\n"
            f"Before each round's execution phase you may communicate with the other "
            f"participants on the shared blackboard.\n\n"
            f"Your goal: maximise your own total net payoff across all rounds."
        )

    def get_user_prompt(
        self,
        agent_name: str,
        agent_context: Dict[str, Any],
        blackboard_context: Dict[str, Any],
    ) -> str:
        system_prompt = self.get_system_prompt(agent_name)
        user_prompt = self._get_user_prompt_impl(
            agent_name, agent_context, blackboard_context
        )

        if self.prompt_logger:
            self.prompt_logger.log_prompts(
                agent_name=agent_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                phase=agent_context.get("phase", "unknown"),
                iteration=agent_context.get("iteration"),
                round_num=agent_context.get("planning_round"),
            )
        return user_prompt

    def _get_user_prompt_impl(
        self,
        agent_name: str,
        agent_context: Dict[str, Any],
        blackboard_context: Dict[str, Any],
    ) -> str:
        phase = str(agent_context.get("phase", "unknown"))
        round_num = agent_context.get("round")
        lines = [
            f"ROUND {round_num} of {agent_context.get('max_rounds')} — {phase.upper()} PHASE."
        ]

        vals = agent_context.get("private_valuations") or {}
        vals_str = ", ".join(
            f"slot {s}: {v} credits"
            for s, v in sorted(vals.items(), key=lambda kv: int(kv[0]))
        )
        lines.append(
            f"Your private slot valuations this round: {vals_str} "
            f"(highest-value slot: slot {agent_context.get('greedy_best_slot')})."
        )

        contracts = agent_context.get("active_contracts") or []
        if contracts:
            contract_strs = [
                f"#{c['contract_id']} ({c['proposer']} -> {c['recipient']}: "
                f"{c['amount']} credits if slot {c['condition_recipient_slot']}, "
                f"status={c['status']})"
                for c in contracts
            ]
            lines.append("Current round transfer contracts: " + "; ".join(contract_strs) + ".")
        elif agent_context.get("side_contracts_enabled", True):
            lines.append("Current round transfer contracts: none recorded yet.")

        last = agent_context.get("last_round")
        if last:
            lines.append(
                f"Previous round result: committed slots {last['committed_slots']}; "
                f"your gross payoff {last['gross_payoffs'].get(agent_name, 0)} credits; "
                f"your net transfer {last['net_transfers'].get(agent_name, 0)} credits; "
                f"your net payoff {last['net_payoffs'].get(agent_name, 0)} credits."
            )
        else:
            lines.append("This is the first round; there is no history yet.")

        net_history = agent_context.get("own_net_payoff_history") or []
        if net_history:
            lines.append(f"Your net payoffs so far by round: {net_history} credits.")

        if blackboard_context:
            bb_blocks = [
                f"=== BLACKBOARD {bb_id} ===\n{str(content).strip()}"
                for bb_id, content in blackboard_context.items()
                if content and str(content).strip()
            ]
            if bb_blocks:
                lines.append("\n\n".join(bb_blocks))

        if phase == "planning":
            if agent_context.get("side_contracts_enabled", True):
                lines.append(
                    "The blackboard above shows the conversation so far. You may send "
                    "messages with post_message, announce a non-binding intended slot with "
                    "propose_allocation(slot=...), offer a conditional transfer with "
                    "propose_transfer(recipient=..., amount=..., condition_recipient_slot=...), "
                    "or accept an offer addressed to you with accept_transfer(contract_id=...). "
                    "End your turn after your tool calls."
                )
            else:
                lines.append(
                    "The blackboard above shows the conversation so far. You may send "
                    "messages with post_message and/or announce a non-binding intended "
                    "slot with propose_allocation(slot=...). End your turn after your tool calls."
                )
        else:
            lines.append(
                "Commit your binding slot choice for this round using commit_slot(slot=...). "
                "Your commit ends your turn."
            )
        return "\n\n".join(lines)
