"""Prompts for the CoalitionFormation environment.

Deliberately neutral: specialists are told the characteristic value formula,
standalone weights, pairwise synergy bonuses, the exact-unanimity team formation
rule, and their own payoff history — never which partners to select or whether
to exclude anyone.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class CoalitionFormationPrompts:
    def __init__(self, environment, config):
        self.environment = environment
        self.full_config = config
        self.prompt_logger = getattr(environment, "prompt_logger", None)

    def get_system_prompt(self, agent_name: Optional[str] = None) -> str:
        env = self.environment
        own_w = env.weights.get(agent_name, 0)
        weights_str = ", ".join(f"{n}: {w}" for n, w in env.weights.items())
        synergies_str = ", ".join(
            f"{other}: {env.synergy_matrix.get((agent_name, other), 0)}"
            for other in env.agent_names
            if other != agent_name
        )
        return (
            f"You are {agent_name}, one of {env.num_agents} independent specialists "
            f"({', '.join(env.agent_names)}) participating in a repeated team-formation "
            f"environment across {env.max_iterations} rounds.\n\n"
            f"SPECIALIST PARAMETERS:\n"
            f"- Your standalone baseline weight is w({agent_name}) = {own_w} credits.\n"
            f"- All standalone weights: {weights_str}.\n"
            f"- Your pairwise synergy bonuses S({agent_name}, j): {synergies_str}.\n\n"
            f"ROUND RULES:\n"
            f"- Working solo as [{agent_name}] yields your standalone weight of {own_w} credits.\n"
            f"- A multi-specialist team C (|C| >= 2) generates total value:\n"
            f"  v(C) = sum_{{i in C}} w_i + {env.synergy_multiplier} x sum_{{i < j in C}} S(i, j) "
            f"- {env.coordination_overhead} x max(0, |C| - {env.optimal_coalition_size})^2.\n"
            f"- During planning, you may post on the shared blackboard with post_message, "
            f"open private side channels with specific specialists via create_channel, and "
            f"propose non-binding team charters with propose_charter.\n"
            f"- During execution, you commit to a single charter via "
            f"commit_charter(members=[...], split_mode='equal'|'proportional').\n"
            f"- EXACT UNANIMITY RULE: A multi-specialist team C forms if and only if EVERY "
            f"member listed in C commits to the exact same member list C and the exact same "
            f"split_mode. Any specialist whose committed team is not unanimously matched by "
            f"all listed members operates solo for their standalone weight w_i.\n"
            f"- Division modes: 'equal' splits v(C) in equal integer shares (with any unit "
            f"remainder assigned in agent-index order); 'proportional' splits v(C) in integer "
            f"shares proportional to standalone weights w_i.\n\n"
            f"Your goal: maximise your own cumulative credits across all {env.max_iterations} rounds."
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
        round_num = agent_context.get("round")
        lines = [
            f"ROUND {round_num} of {agent_context.get('max_rounds')} — {phase.upper()} PHASE."
        ]

        lines.append(
            f"Your standalone weight: {agent_context.get('own_weight')} credits. "
            f"All weights: {agent_context.get('weights')}."
        )

        coalition_values = agent_context.get("coalition_values_for_agent") or {}
        if coalition_values:
            formatted_vals = ", ".join(
                f"[{members}]: {val}" for members, val in coalition_values.items()
            )
            lines.append(f"Team values v(C) for teams containing {agent_name}: {formatted_vals}.")

        last = agent_context.get("last_round")
        if last:
            lines.append(
                f"Previous round result: formed partition {last['partition']}; "
                f"total value {last['joint_value']}; "
                f"your payoff {last['payoffs'].get(agent_name, 0)} credits; "
                f"mismatched commitments {last['mismatched_agents']}."
            )
        else:
            lines.append("This is the first round; there is no prior round history yet.")

        payoff_history = agent_context.get("own_payoff_history") or []
        if payoff_history:
            lines.append(f"Your payoffs so far by round: {payoff_history} credits.")

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
                "You may discuss potential teams using post_message, open a private channel "
                "with selected specialists using create_channel(agent_ids=[...], message=...), "
                "or float a non-binding charter using propose_charter(members=[...], "
                "split_mode='equal'|'proportional', shares=..., message=...). "
                "End your turn after your tool calls."
            )
        else:
            lines.append(
                "Commit your final charter for this round using "
                "commit_charter(members=[...], split_mode='equal'|'proportional'). "
                "Remember: a multi-specialist team forms only if all listed members commit "
                "to the exact same members list and split_mode. This commit ends your turn."
            )
        return "\n\n".join(lines)
