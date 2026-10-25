"""Prompts for the CommonPool environment.

Deliberately neutral: harvesters are told the rules and their own history,
never to conserve, coordinate, or extract heavily. Any self-regulation in the
transcripts is tacit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class CommonPoolPrompts:
    def __init__(self, environment, config):
        self.environment = environment
        self.full_config = config
        self.prompt_logger = getattr(environment, "prompt_logger", None)

    def get_system_prompt(self, agent_name: Optional[str] = None) -> str:
        env = self.environment
        return (
            f"You are {agent_name}, an independent harvester drawing from a shared "
            f"renewable resource pool alongside {env.num_agents - 1} other harvesters.\n\n"
            f"POOL RULES:\n"
            f"- The pool has a capacity of {env.capacity} units and currently holds "
            f"{round(env.pool, 1)} units.\n"
            f"- After each season the pool regrows by {int(env.growth_rate * 100)}% of "
            f"the pool multiplied by the share of capacity that is free "
            f"(the emptier the pool, the faster it regrows; an empty pool stays empty).\n"
            f"- Each season you may extract between 0 and {env.max_extract} units.\n"
            f"- Every unit you extract is worth {int(env.unit_value)} credits to you.\n"
            f"- If total requests exceed the pool, units are rationed proportionally "
            f"to requests.\n"
            f"- The seasons repeat {env.max_iterations} times in total.\n\n"
            f"Before committing each season's extraction you may talk with the other "
            f"harvesters on the shared blackboard. You are free to discuss whatever "
            f"you like.\n\n"
            f"Your goal: maximise your own total credits across all seasons."
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
        season = agent_context.get("season")
        lines = [f"SEASON {season} of {agent_context.get('max_seasons')} — {phase.upper()} PHASE."]

        lines.append(
            f"Pool level now: {agent_context.get('current_pool')} of "
            f"{agent_context.get('pool_capacity')} units "
            f"(regrowth rate {int(agent_context.get('growth_rate', 0) * 100)}%)."
        )
        lines.append(
            f"Your extraction limit this season: 0-{agent_context.get('max_extract')} units; "
            f"each unit is worth {int(agent_context.get('unit_value', 0))} credits."
        )

        last = agent_context.get("last_season")
        if last:
            lines.append(
                f"Last season: requests {last['desired']}; you were granted "
                f"{last['granted'].get(agent_name, 0.0)} units "
                f"(rationed: {'yes' if last['rationed'] else 'no'}); pool fell to "
                f"{last['pool_after_extract']} then regrew to {last['pool_after_regrow']}. "
                f"Your profit: {last['profits'].get(agent_name, 0.0)} credits."
            )
        else:
            lines.append("This is the first season; there is no history yet.")

        hist = agent_context.get("own_extraction_history") or []
        if hist:
            lines.append(f"Your own past extractions: {hist}.")

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
                "The blackboard above shows the conversation so far. You may send "
                "messages with post_message and/or publicly announce a suggested "
                "extraction level with propose_quota(amount=...). Coordination talk "
                "is allowed but never required. End your turn after your tool calls."
            )
        else:
            lines.append(
                "Commit your extraction for this season with extract(amount=...). "
                "This ends your turn."
            )
        return "\n\n".join(lines)
