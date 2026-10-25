"""Prompts for the PublicGoods environment.

Deliberately neutral: agents are told the rules and their own history, never
to contribute, free-ride, punish, or refrain from punishing. Any contribution
norms or punishment patterns in the transcripts are tacit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class PublicGoodsPrompts:
    def __init__(self, environment, config):
        self.environment = environment
        self.full_config = config
        self.prompt_logger = getattr(environment, "prompt_logger", None)

    def get_system_prompt(self, agent_name: Optional[str] = None) -> str:
        env = self.environment
        s = (
            f"You are {agent_name}, an independent participant in a repeated "
            f"group account game alongside {env.num_agents - 1} other "
            f"participants.\n\n"
            f"GAME RULES:\n"
            f"- Each round you receive {env.endowment} tokens.\n"
            f"- You may contribute between 0 and {env.endowment} tokens to the "
            f"group account; tokens you keep stay yours.\n"
            f"- The group account total is multiplied by {env.multiplier} and "
            f"split equally among ALL {env.num_agents} participants, whatever "
            f"they contributed.\n"
            f"- The game runs for {env.max_iterations} rounds.\n"
        )
        if env.punishment_enabled:
            s += (
                f"- When committing your contribution you may also assign up to "
                f"{env.punish_points} punishment points across the other "
                f"participants: each point costs you {int(env.punish_cost)} "
                f"credit and deducts {int(env.punish_damage)} credits from the "
                f"participant it is assigned to.\n"
            )
        s += (
            f"\nBefore committing each round's contribution you may talk with "
            f"the other participants on the shared blackboard. You are free to "
            f"discuss whatever you like.\n\n"
            f"Your goal: maximise your own total credits across all rounds."
        )
        return s

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
        rnd = agent_context.get("round")
        lines = [f"ROUND {rnd} of {agent_context.get('max_rounds')} — {phase.upper()} PHASE."]

        lines.append(
            f"Your endowment this round: {agent_context.get('endowment')} tokens; "
            f"group account multiplier {agent_context.get('multiplier')}, split "
            f"equally among all participants."
        )

        last = agent_context.get("last_round")
        if last:
            contribs = last["contributions"]
            anon = ", ".join(
                f"{name}: {contribs[name]}" for name in sorted(contribs)
            )
            lines.append(
                f"Last round contributions: {anon}; group account paid "
                f"{last['share_per_agent']} credits to each participant. "
                f"Your earnings: {last['earnings'].get(agent_name, 0.0)} credits."
            )
            if last.get("punishments"):
                pl = "; ".join(
                    f"{d['punisher']} -> {d['peer']} ({d['points']} pts)"
                    for d in last["punishments"]
                )
                lines.append(f"Punishment assigned last round: {pl}.")
            else:
                lines.append("No punishment was assigned last round.")
        else:
            lines.append("This is the first round; there is no history yet.")

        hist = agent_context.get("own_contribution_history") or []
        if hist:
            lines.append(f"Your own past contributions: {hist}.")

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
                "The blackboard above shows the conversation so far. You may "
                "send messages with post_message and/or publicly announce a "
                "suggested contribution with propose_contribution(amount=...). "
                "Discussion is allowed but never required. End your turn after "
                "your tool calls."
            )
        else:
            if agent_context.get("punishment_enabled"):
                lines.append(
                    "Commit your contribution with contribute(amount=...). You "
                    "may optionally include punishments=[{{\"peer\": \"...\", "
                    "\"points\": N}}, ...]. This ends your turn."
                )
            else:
                lines.append(
                    "Commit your contribution with contribute(amount=...). "
                    "This ends your turn."
                )
        return "\n\n".join(lines)
