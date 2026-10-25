"""Prompts for the AdAuctionGSP environment.

Deliberately neutral: bidders are told the rules, the slot table, and their own
private value and history — never how to bid strategically, coordinate, or
cooperate. Any equilibrium-like play in the transcripts is tacit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class AdAuctionGSPPrompts:
    def __init__(self, environment, config):
        self.environment = environment
        self.full_config = config
        self.prompt_logger = getattr(environment, "prompt_logger", None)

    def _slot_table(self) -> str:
        env = self.environment
        rows = [
            f"- {slot['name']}: {env.slot_clicks[k]:g} clicks per iteration"
            for k, slot in enumerate(env.slots)
        ]
        return "\n".join(rows)

    def get_system_prompt(self, agent_name: Optional[str] = None) -> str:
        env = self.environment
        return (
            f"You are {agent_name}, an advertiser bidding for ad slots in a per-click "
            f"keyword auction.\n"
            f"Your private value for one click is {env.agent_values[agent_name]:g}. "
            f"No other advertiser knows this number, and you do not know theirs.\n\n"
            f"AD SLOTS — {env.num_slots} slots, ranked by clicks per iteration:\n"
            f"{self._slot_table()}\n\n"
            f"RULES:\n"
            f"- Each iteration you submit exactly one bid: the price you offer to pay "
            f"per click.\n"
            f"- Bids are ranked from highest to lowest. The highest bid takes "
            f"{env.slot_names[0]}, the next takes {env.slot_names[1] if env.num_slots > 1 else 'the following slot'}, "
            f"and so on.\n"
            f"- If you win a slot, your cost per click equals the bid of the advertiser "
            f"ranked directly below you, never below the reserve price of "
            f"{env.reserve_price:g}. If you win the last slot and no one ranked below "
            f"you, you pay the reserve price.\n"
            f"- Advertisers who win no slot pay nothing and receive no clicks.\n"
            f"- Your profit for the iteration is (your value - your cost per click) x "
            f"clicks of your slot.\n"
            f"- The auction repeats for {env.max_iterations} iterations.\n\n"
            f"Before each iteration you may talk with the other advertisers on the "
            f"public blackboard. You are free to discuss whatever you like.\n\n"
            f"Your goal: maximise your own total profit across all iterations."
        )

    def get_user_prompt(
        self,
        agent_name: str,
        agent_context: Dict[str, Any],
        blackboard_context: Dict[str, Any],
    ) -> str:
        system_prompt = self.get_system_prompt(agent_name)
        user_prompt = self._get_user_prompt_impl(agent_name, agent_context, blackboard_context)

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
        iteration = agent_context.get("iteration")
        max_iterations = agent_context.get("max_iterations")
        lines = [
            f"ITERATION {iteration} of {max_iterations} — {phase.upper()} PHASE.",
            f"Your private value per click: {agent_context.get('own_value'):g}.",
        ]

        own_bids = agent_context.get("own_bid_history") or []
        if own_bids:
            lines.append(
                f"Your bid history so far (per iteration): "
                f"{[float(v) for v in own_bids]}."
            )
        else:
            lines.append("This is the first iteration; you have not bid yet.")

        own_profits = agent_context.get("own_profit_history") or []
        if own_profits:
            lines.append(
                f"Your profit history so far (per iteration): "
                f"{[float(v) for v in own_profits]}."
            )

        last = agent_context.get("last_iteration")
        if last:
            lines.append(
                f"Previous iteration result: bids {last['bids']}; slot assignment "
                f"{last['allocation']}; payments per click {last['payments']}."
            )

        if blackboard_context:
            bb_blocks = [
                f"=== BLACKBOARD {bb_id} ===\n{str(content).strip()}"
                for bb_id, content in blackboard_context.items()
                if content and str(content).strip()
            ]
            if bb_blocks:
                lines.append("\n\n".join(bb_blocks))

        if phase == "planning":
            lines.append(
                "The blackboard above shows the conversation so far. You may send messages "
                "with post_message and/or announce a bid you are considering with "
                "propose_bid(bid=...). Discussion is allowed but never required. "
                "End your turn after your tool calls."
            )
        else:
            lines.append(
                "Submit your per-click bid for this iteration with commit_bid(bid=...). "
                "Remember: if you win a slot, your cost per click is the bid of the "
                "advertiser ranked directly below you. This ends your turn."
            )
        return "\n\n".join(lines)
